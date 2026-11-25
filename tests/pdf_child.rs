//! Testes do leitor de PDF isolado (D55): rodam o BINÁRIO de verdade (`CARGO_BIN_EXE_buriti`), porque o `lopdf`
//! pode abortar e só um processo real prova que o servidor sobrevive. Num teste unitário, `current_exe()` seria
//! o próprio executável de teste, então a reexecução só faz sentido aqui.
//!
//! TODO processo lançado tem vigia: um fio mata o PID (só o nosso) depois do prazo, e o `Drop` também mata.

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

const EXE: &str = env!("CARGO_BIN_EXE_buriti");

// ------------------------------------------------------------------ PDFs de teste

/// PDF mínimo escrito à mão, com tabela xref correta. `content` é o fluxo de conteúdo da única página.
fn minimal_pdf(content: &[u8], filter: Option<&str>) -> Vec<u8> {
    let mut stream = format!("<< /Length {} {} >>\nstream\n", content.len(), filter.map(|f| format!("/Filter /{f}")).unwrap_or_default()).into_bytes();
    stream.extend_from_slice(content);
    stream.extend_from_slice(b"\nendstream");
    let objs: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_vec(),
        stream,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_vec(),
    ];
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offs = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offs.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(o);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for o in &offs {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).as_bytes());
    out
}

fn hello_pdf() -> Vec<u8> {
    minimal_pdf(b"BT /F1 12 Tf 72 700 Td (Hello World) Tj ET", None)
}

/// Bomba de descompressão: um fluxo FlateDecode pequeno que expande para `mb` MB de zeros.
fn bomb_pdf(mb: usize) -> Vec<u8> {
    use flate2::{write::ZlibEncoder, Compression};
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
    let chunk = vec![0u8; 1 << 20];
    for _ in 0..mb {
        enc.write_all(&chunk).unwrap();
    }
    let z = enc.finish().unwrap();
    assert!(z.len() < 4 * 1024 * 1024, "a bomba tem de ser pequena ({} bytes)", z.len());
    minimal_pdf(&z, Some("FlateDecode"))
}

fn pseudo_random(n: usize, mut seed: u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        })
        .collect()
}

#[allow(deprecated)]
fn encrypted_pdf() -> Vec<u8> {
    use lopdf::{Document, EncryptionState, EncryptionVersion, Permissions};
    let mut doc = Document::load_mem(&hello_pdf()).unwrap();
    let id = lopdf::Object::string_literal(b"0123456789abcdef".to_vec());
        doc.trailer.set("ID", lopdf::Object::Array(vec![id.clone(), id]));
        let reference = doc.clone();
    let v = EncryptionVersion::V2 { document: &reference, owner_password: "dono", user_password: "segredo", key_length: 128, permissions: Permissions::all() };
    doc.encrypt(&EncryptionState::try_from(v).unwrap()).unwrap();
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

// -------------------------------------------------------------- execução com vigia

/// Roda `buriti --pdf-extract` com a entrada dada; mata o filho (só o PID dele) se passar de `limit`.
/// Devolve (código de saída ou None se foi morto pelo vigia, stdout, tempo).
fn run_child(input: &[u8], env: &[(&str, &str)], limit: Duration) -> (Option<i32>, Vec<u8>, Duration) {
    let mut cmd = Command::new(EXE);
    cmd.arg("--pdf-extract").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let start = Instant::now();
    let mut child = cmd.spawn().expect("spawn");
    let mut stdin = child.stdin.take().unwrap();
    let data = input.to_vec();
    let w = thread::spawn(move || {
        let _ = stdin.write_all(&data);
    });
    let mut so = child.stdout.take().unwrap();
    let r = thread::spawn(move || {
        let mut v = Vec::new();
        let _ = so.by_ref().take(16 << 20).read_to_end(&mut v);
        v
    });
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break Some(s);
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let _ = w.join();
    let out = r.join().unwrap();
    (status.and_then(|s| s.code()), out, start.elapsed())
}

/// Servidor Buriti real numa porta livre e pasta nova, com vigia e `Drop` que matam o processo.
struct Server {
    child: Arc<Mutex<Child>>,
    port: u16,
    token: String,
    home: PathBuf,
}

impl Server {
    fn start(env: &[(&str, &str)]) -> Server {
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let home = std::env::temp_dir().join(format!("buriti-pdf-test-{}-{}", std::process::id(), port));
        std::fs::create_dir_all(&home).unwrap();
        let mut cmd = Command::new(EXE);
        cmd.env("BURITI_HOME", &home).env("BURITI_PORT", port.to_string()).env("BURITI_NO_BROWSER", "1").env("BURITI_KEY_FILE", home.join("key.bin"));
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = cmd.spawn().expect("spawn server");
        let stdout = child.stdout.take().unwrap();
        let child = Arc::new(Mutex::new(child));
        // vigia: no máximo 180 s de vida, aconteça o que acontecer
        let watch = child.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(180));
            let _ = watch.lock().unwrap().kill();
        });
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut sent = false;
            let mut buf = String::new();
            let mut stdout = stdout;
            let mut chunk = [0u8; 1024];
            while let Ok(n) = stdout.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if !sent {
                    buf.push_str(&String::from_utf8_lossy(&chunk[..n]));
                    if let Some(i) = buf.find("#token=") {
                        let tok: String = buf[i + 7..].chars().take_while(|c| c.is_ascii_hexdigit()).collect();
                        if tok.len() >= 8 {
                            let _ = tx.send(tok);
                            sent = true;
                        }
                    }
                }
            }
        });
        let token = rx.recv_timeout(Duration::from_secs(30)).expect("o servidor não imprimiu o token a tempo");
        Server { child, port, token, home }
    }

    /// POST cru com o token; devolve (status, corpo).
    fn post_pdf(&self, body: &[u8], with_token: bool) -> (u16, String) {
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        let mut head = format!("POST /api/attach/pdf HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/pdf\r\nContent-Length: {}\r\nConnection: close\r\n", self.port, body.len());
        if with_token {
            head.push_str(&format!("X-Buriti-Token: {}\r\n", self.token));
        }
        head.push_str("\r\n");
        s.write_all(head.as_bytes()).unwrap();
        s.write_all(body).unwrap();
        let mut resp = Vec::new();
        let _ = s.read_to_end(&mut resp);
        let text = String::from_utf8_lossy(&resp).into_owned();
        let status = text.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
        (status, body)
    }

    fn alive(&self) -> bool {
        self.child.lock().unwrap().try_wait().unwrap().is_none()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let mut c = self.child.lock().unwrap();
        let _ = c.kill();
        let _ = c.wait();
        let _ = std::fs::remove_dir_all(&self.home);
    }
}

/// O processo com este PID existe? (só PID, nunca nome)
fn pid_alive(pid: u32) -> bool {
    #[cfg(windows)]
    {
        let out = Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH"]).output().unwrap();
        String::from_utf8_lossy(&out.stdout).contains(&format!(" {pid} "))
    }
    #[cfg(not(windows))]
    {
        Command::new("kill").args(["-0", &pid.to_string()]).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
    }
}

/// Mata só este PID (limpeza de um filho que o teste criou).
fn kill_pid(pid: u32) {
    #[cfg(windows)]
    let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/F"]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    #[cfg(not(windows))]
    let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
}

fn wait_gone(pid: u32, secs: u64) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        if !pid_alive(pid) {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

fn read_pid(file: &PathBuf, secs: u64) -> Option<u32> {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(secs) {
        if let Some(p) = std::fs::read_to_string(file).ok().and_then(|t| t.trim().parse().ok()) {
            return Some(p);
        }
        thread::sleep(Duration::from_millis(100));
    }
    None
}

fn contains(hay: &[u8], needle: &str) -> bool {
    String::from_utf8_lossy(hay).contains(needle)
}

// ------------------------------------------------------------------------ testes do filho

#[test]
fn filho_extrai_pdf_valido() {
    let (code, out, _) = run_child(&hello_pdf(), &[], Duration::from_secs(30));
    assert_eq!(code, Some(0));
    assert!(contains(&out, "Hello World"), "{}", String::from_utf8_lossy(&out));
}

#[test]
fn filho_termina_em_entradas_malformadas() {
    let valid = hello_pdf();
    let mut huge_count = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 4000000000 >>\nendobj\n".to_vec();
    huge_count.extend_from_slice(b"xref\n0 4000000000\ntrailer\n<< /Size 4000000000 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n");
    let mut many_objs = b"%PDF-1.4\n".to_vec();
    for i in 1..=60000u32 {
        many_objs.extend_from_slice(format!("{i} 0 obj\n<< /K [{} 0 R {} 0 R] >>\nendobj\n", i + 1, i + 2).as_bytes());
    }
    let mut after_header = b"%PDF-1.7\n".to_vec();
    after_header.extend_from_slice(&pseudo_random(200_000, 7));
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("truncado ao meio", valid[..valid.len() / 2].to_vec()),
        ("truncado após o cabeçalho", b"%PDF-1.4\n1 0 obj\n<< /Type".to_vec()),
        ("bytes aleatórios após %PDF-", after_header),
        ("contagens enormes", huge_count),
        ("60 mil objetos encadeados", many_objs),
    ];
    for (name, data) in cases {
        let (code, out, t) = run_child(&data, &[], Duration::from_secs(25));
        // qualquer desfecho serve (erro, vazio, até aborto) MENOS travar: o vigia devolveria None
        assert!(code.is_some(), "{name}: o filho não terminou em 25 s (vigia o matou)");
        assert!(out.len() < 4 << 20, "{name}: saída grande demais");
        eprintln!("{name}: código {code:?} em {t:?}");
    }
}

#[test]
fn filho_resiste_a_bomba_de_descompressao() {
    let bomb = bomb_pdf(600); // ~600 MB depois de descomprimir; o teto é 64 MB
    assert!(bomb.len() < 5 * 1024 * 1024);
    let (code, out, t) = run_child(&bomb, &[], Duration::from_secs(25));
    eprintln!("bomba: código {code:?} em {t:?}");
    assert!(code.is_some(), "a bomba travou o filho");
    assert_ne!(code, Some(0), "a bomba não pode virar texto com sucesso");
    assert!(out.is_empty() || !contains(&out, "\0\0\0\0"), "saída com zeros da bomba");
}

#[test]
fn filho_reconhece_pdf_com_senha() {
    let (code, out, _) = run_child(&encrypted_pdf(), &[], Duration::from_secs(30));
    assert_eq!(code, Some(20), "saída: {}", String::from_utf8_lossy(&out));
}

// ------------------------------------------------------------ testes do servidor (rota real)

#[test]
fn rota_funciona_e_exige_token() {
    let srv = Server::start(&[]);
    let (st, body) = srv.post_pdf(&hello_pdf(), true);
    assert_eq!(st, 200, "{body}");
    assert!(body.contains("Hello World") && body.contains("\"truncated\":false"), "{body}");
    assert_eq!(srv.post_pdf(&hello_pdf(), false).0, 401);
    let (st, body) = srv.post_pdf(&encrypted_pdf(), true);
    assert_eq!(st, 400);
    assert!(body.contains("err_attach_encrypted"), "{body}");
    // PDF sem texto selecionável (página vazia)
    let (st, body) = srv.post_pdf(&minimal_pdf(b"", None), true);
    assert_eq!(st, 400);
    assert!(body.contains("err_attach_nopdftext"), "{body}");
}

#[test]
fn servidor_sobrevive_a_pdfs_ruins() {
    let srv = Server::start(&[]);
    let mut garbage = b"%PDF-1.7\n".to_vec();
    garbage.extend_from_slice(&pseudo_random(100_000, 99));
    let t0 = Instant::now();
    for (name, data) in [("aleatório", garbage), ("truncado", hello_pdf()[..60].to_vec()), ("bomba", bomb_pdf(600))] {
        let (st, body) = srv.post_pdf(&data, true);
        assert_eq!(st, 400, "{name}: {body}");
        assert!(!body.is_empty(), "{name}: sem mensagem");
        assert!(srv.alive(), "{name}: o servidor MORREU");
        eprintln!("{name}: {st} {body}");
    }
    assert!(t0.elapsed() < Duration::from_secs(60));
    // depois de tudo isso o servidor ainda atende um PDF bom
    assert_eq!(srv.post_pdf(&hello_pdf(), true).0, 200);
}

#[test]
fn filho_que_aborta_nao_derruba_o_servidor() {
    for modo in ["abort", "panic"] {
        let srv = Server::start(&[("BURITI_TEST_PDF_CHILD", modo)]);
        // várias vezes: a corrida entre o fim da saída (EOF) e a leitura do código de saída só aparecia de vez em quando
        for i in 0..20 {
            let (st, body) = srv.post_pdf(&hello_pdf(), true);
            assert_eq!(st, 400, "{modo} #{i}: {body}");
            assert!(body.contains("err_attach_crash"), "{modo} #{i}: {body}");
        }
        assert!(srv.alive(), "{modo}: o servidor MORREU");
        assert_eq!(srv.post_pdf(&hello_pdf(), false).0, 401, "{modo}: servidor não responde mais");
    }
}

#[test]
fn filho_que_trava_e_morto_no_tempo_limite() {
    let pidfile = std::env::temp_dir().join(format!("buriti-pdf-pid-hang-{}", std::process::id()));
    let _ = std::fs::remove_file(&pidfile);
    let srv = Server::start(&[("BURITI_TEST_PDF_CHILD", "hang"), ("BURITI_TEST_PDF_TIMEOUT_MS", "1500"), ("BURITI_TEST_PDF_PIDFILE", pidfile.to_str().unwrap())]);
    let t0 = Instant::now();
    let (st, body) = srv.post_pdf(&hello_pdf(), true);
    let took = t0.elapsed();
    assert_eq!(st, 400, "{body}");
    assert!(body.contains("err_attach_timeout") && body.contains("over 2 s"), "{body}");
    assert!(took >= Duration::from_millis(1400) && took < Duration::from_secs(10), "tempo {took:?}");
    assert!(srv.alive());
    // o pai matou E esperou o filho: ele não pode sobrar
    let pid = read_pid(&pidfile, 5).expect("o filho não gravou o PID");
    let gone = wait_gone(pid, 5);
    if !gone {
        kill_pid(pid);
    }
    assert!(gone, "o filho {pid} sobrou depois do tempo limite");
    let _ = std::fs::remove_file(&pidfile);
}

#[test]
fn saida_gigante_do_filho_e_cortada() {
    // o filho despeja 'a' sem parar: o pai corta no teto, mata o filho e devolve texto cortado (2M caracteres)
    let srv = Server::start(&[("BURITI_TEST_PDF_CHILD", "spam")]);
    let (st, body) = srv.post_pdf(&hello_pdf(), true);
    assert_eq!(st, 200, "{}", &body[..body.len().min(200)]);
    assert!(body.contains("\"truncated\":true") && body.contains("\"chars\":2000000"), "{}", &body[..body.len().min(200)]);
    assert!(srv.alive());
}

// ------------------------------------------------- filho nunca fica órfão (vigia interno)

#[test]
fn filho_com_entrada_aberta_sai_sozinho_no_prazo() {
    // entrada que nunca recebe dados nem EOF: sem o vigia interno o filho ficaria bloqueado para sempre
    let mut cmd = Command::new(EXE);
    cmd.arg("--pdf-extract").env("BURITI_TEST_PDF_CHILD_DEADLINE_MS", "1500").stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = cmd.spawn().unwrap();
    let pid = child.id();
    let t0 = Instant::now();
    let code = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st.code();
        }
        if t0.elapsed() > Duration::from_secs(10) {
            let _ = child.kill(); // só o nosso PID
            let _ = child.wait();
            panic!("o filho {pid} NÃO saiu sozinho em 10 s (sem vigia interno)");
        }
        thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(code, Some(24), "código do vigia");
    assert!(t0.elapsed() < Duration::from_secs(6), "demorou {:?}", t0.elapsed());
}

#[test]
fn filho_nao_sobra_quando_o_servidor_morre() {
    // o servidor (pai) morre no meio de uma extração travada: o filho órfão tem de sair sozinho (vigia de 6 s; folga para a máquina carregada)
    let pidfile = std::env::temp_dir().join(format!("buriti-pdf-pid-orphan-{}", std::process::id()));
    let _ = std::fs::remove_file(&pidfile);
    let srv = Server::start(&[("BURITI_TEST_PDF_CHILD", "hang"), ("BURITI_TEST_PDF_CHILD_DEADLINE_MS", "6000"), ("BURITI_TEST_PDF_PIDFILE", pidfile.to_str().unwrap())]);
    let port = srv.port;
    let token = srv.token.clone();
    // pedido em outra thread (fica preso até o servidor morrer)
    let req = thread::spawn(move || {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let body = hello_pdf();
        let head = format!("POST /api/attach/pdf HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Buriti-Token: {token}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
        let _ = s.write_all(head.as_bytes());
        let _ = s.write_all(&body);
        let mut v = Vec::new();
        let _ = s.read_to_end(&mut v);
    });
    let pid = read_pid(&pidfile, 15).expect("o filho não gravou o PID");
    assert!(pid_alive(pid), "o filho deveria estar vivo (travado)");
    drop(srv); // mata o servidor: o filho fica órfão
    let gone = wait_gone(pid, 20);
    if !gone {
        kill_pid(pid); // limpeza do nosso PID
    }
    let _ = req.join();
    let _ = std::fs::remove_file(&pidfile);
    assert!(gone, "o filho órfão {pid} continuou vivo depois da morte do servidor");
}
