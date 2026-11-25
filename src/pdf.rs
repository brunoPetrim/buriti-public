//! Extração de texto de PDF (D55), isolada em um PROCESSO FILHO do próprio Buriti.
//!
//! O `lopdf` tem centenas de `unwrap()`/`panic!` e o perfil release usa `panic = "abort"`: um PDF malformado
//! derrubaria o servidor inteiro. Por isso o servidor nunca roda o `lopdf` em si: ele reexecuta o próprio
//! binário (`std::env::current_exe()`) com o argumento interno `--pdf-extract`, entrega os bytes pela entrada
//! padrão e lê o texto na saída padrão. Pânico, aborto, laço infinito (tempo limite) ou saída gigante só
//! matam o filho; o pai devolve uma mensagem clara. O argumento é tratado em `main` ANTES de qualquer outra
//! coisa (sem servidor, portas, navegador nem configurações).
//!
//! Protocolo do filho: stdin = bytes do PDF; stdout = texto (UTF-8); código de saída 0 = ok, 20 = protegido por
//! senha, 21 = páginas demais, 22 = erro de leitura (mensagem curta em stderr), 23 = entrada inválida; qualquer
//! outro (aborto, pânico, sinal) = falha do leitor. Limites de memória do processo NÃO são impostos (não há
//! forma portátil sem dependência nova): o que limita é o teto de descompressão do `lopdf` (64 MB por fluxo),
//! o tempo e o corte da saída.

use std::{
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use crate::attach::MAX_FILE_BYTES;

/// Argumento interno que faz o binário virar apenas um extrator de PDF (stdin -> stdout).
pub const PDF_CHILD_ARG: &str = "--pdf-extract";
/// Tempo limite padrão de uma extração; `BURITI_TEST_PDF_TIMEOUT_MS` o reduz (só testes).
pub const PDF_TIMEOUT: Duration = Duration::from_secs(20);
/// Teto da saída lida do filho (bytes). 8 MiB > 4 x 2M caracteres: se o teto for atingido, o texto já passa do
/// corte de caracteres de `attach::finish_pdf_text` e a resposta sai marcada como cortada.
pub const PDF_OUT_CAP: usize = 8 * 1024 * 1024;
/// Teto de descompressão por fluxo (objetos, xref e conteúdo de página).
pub const MAX_DECOMPRESSED: usize = 64 * 1024 * 1024;
/// Mais páginas que isto: recusa (a extração seria lenta e o texto passaria do contexto de qualquer modelo).
pub const MAX_PDF_PAGES: usize = 500;

const EXIT_ENCRYPTED: i32 = 20;
const EXIT_TOO_MANY_PAGES: i32 = 21;
const EXIT_FAILED: i32 = 22;
const EXIT_BAD_INPUT: i32 = 23;
/// Código de saída do vigia interno do filho (prazo total estourado).
const EXIT_WATCHDOG: i32 = 24;
/// Prazo TOTAL de vida do filho, contado do começo. Vale mesmo que o pai tenha morrido sem matá-lo, que a entrada
/// nunca receba EOF ou que a extração trave: o filho se encerra sozinho (maior que o tempo limite do pai, 20 s).
/// `BURITI_TEST_PDF_CHILD_DEADLINE_MS` o reduz (só testes).
pub const CHILD_DEADLINE: Duration = Duration::from_secs(30);

// Mensagens com código (D83): a interface as traduz pelo código; o inglês vai junto para os outros clientes.
pub fn msg_encrypted() -> String {
    crate::errcode::coded("err_attach_encrypted", &[], "The PDF is password-protected or encrypted; it cannot be read.")
}
pub fn msg_crash() -> String {
    crate::errcode::coded("err_attach_crash", &[], "The PDF reader failed while processing this file (it may be corrupted or too complex).")
}
pub fn msg_timeout(t: Duration) -> String {
    let secs = t.as_millis().div_ceil(1000).to_string();
    crate::errcode::coded("err_attach_timeout", &[&secs], format!("Reading the PDF took too long (over {secs} s) and was cancelled."))
}
pub fn msg_pages() -> String {
    crate::errcode::coded("err_attach_pages", &[&MAX_PDF_PAGES.to_string()], format!("The PDF has too many pages (limit of {MAX_PDF_PAGES})."))
}
/// Mensagens já finais (não levam o prefixo "Could not read the PDF:").
pub fn is_final_message(e: &str) -> bool {
    matches!(crate::errcode::code_of(e).as_deref(), Some("err_attach_encrypted" | "err_attach_crash" | "err_attach_timeout" | "err_attach_pages"))
}

// ------------------------------------------------------------------ núcleo (roda no filho)

#[derive(Debug, PartialEq)]
pub enum CoreErr {
    Encrypted,
    TooManyPages(usize),
    Failed(String),
}

pub fn extract_in_process(bytes: &[u8]) -> Result<String, CoreErr> {
    extract_with_limits(bytes, MAX_PDF_PAGES)
}

pub fn extract_with_limits(bytes: &[u8], max_pages: usize) -> Result<String, CoreErr> {
    let opts = lopdf::LoadOptions { max_decompressed_size: Some(MAX_DECOMPRESSED), ..Default::default() };
    let doc = match lopdf::Document::load_mem_with_options(bytes, opts) {
        Ok(d) => d,
        Err(lopdf::Error::InvalidPassword) => return Err(CoreErr::Encrypted),
        Err(e) if e.to_string().contains("requires a password") => return Err(CoreErr::Encrypted),
        Err(e) => return Err(CoreErr::Failed(e.to_string())),
    };
    // com senha de abertura o `lopdf` carrega sem os objetos e o dicionário /Encrypt continua no trailer
    if doc.is_encrypted() {
        return Err(CoreErr::Encrypted);
    }
    let pages: Vec<u32> = doc.get_pages().keys().copied().collect();
    if pages.len() > max_pages {
        return Err(CoreErr::TooManyPages(pages.len()));
    }
    if pages.is_empty() {
        return Err(CoreErr::Failed("o PDF não tem páginas".into()));
    }
    doc.extract_text_with_limit(&pages, MAX_DECOMPRESSED).map_err(|e| CoreErr::Failed(e.to_string()))
}

/// Corpo do processo filho (`buriti --pdf-extract`): devolve o código de saída.
pub fn child_main() -> i32 {
    // VIGIA: um fio que encerra o processo no prazo, aconteça o que acontecer (leitura da entrada parada, laço
    // infinito do lopdf, pai morto). Sem ele, um filho órfão bloqueado ficaria vivo para sempre (visto em teste).
    let deadline = std::env::var("BURITI_TEST_PDF_CHILD_DEADLINE_MS").ok().and_then(|v| v.parse::<u64>().ok()).map(Duration::from_millis).unwrap_or(CHILD_DEADLINE);
    thread::spawn(move || {
        thread::sleep(deadline);
        std::process::exit(EXIT_WATCHDOG);
    });
    // gancho só de teste: grava o PID para o teste achar o filho (nunca por nome) e confirmar que ele não sobra
    if let Some(f) = std::env::var_os("BURITI_TEST_PDF_PIDFILE") {
        let _ = std::fs::write(f, std::process::id().to_string());
    }
    // ganchos SÓ de teste: simulam um filho que trava, aborta ou despeja saída sem fim
    match std::env::var("BURITI_TEST_PDF_CHILD").as_deref() {
        Ok("hang") => loop {
            thread::sleep(Duration::from_secs(3600));
        },
        Ok("abort") => std::process::abort(),
        Ok("panic") => panic!("pânico simulado"),
        Ok("spam") => {
            let chunk = vec![b'a'; 64 * 1024];
            let mut out = std::io::stdout().lock();
            while out.write_all(&chunk).is_ok() {}
            return 0;
        }
        _ => {}
    }
    let mut buf = Vec::new();
    if std::io::stdin().lock().take(MAX_FILE_BYTES as u64 + 1).read_to_end(&mut buf).is_err() || buf.is_empty() || buf.len() > MAX_FILE_BYTES {
        eprintln!("entrada inválida");
        return EXIT_BAD_INPUT;
    }
    match extract_in_process(&buf) {
        Ok(text) => {
            let mut out = std::io::stdout().lock();
            if out.write_all(text.as_bytes()).and_then(|_| out.flush()).is_err() {
                return EXIT_FAILED;
            }
            0
        }
        Err(CoreErr::Encrypted) => EXIT_ENCRYPTED,
        Err(CoreErr::TooManyPages(n)) => {
            eprintln!("{n} páginas");
            EXIT_TOO_MANY_PAGES
        }
        Err(CoreErr::Failed(m)) => {
            eprintln!("{m}");
            EXIT_FAILED
        }
    }
}

// --------------------------------------------------------------------- pai (roda no servidor)

pub fn timeout_from_env() -> Duration {
    std::env::var("BURITI_TEST_PDF_TIMEOUT_MS").ok().and_then(|v| v.parse::<u64>().ok()).map(Duration::from_millis).unwrap_or(PDF_TIMEOUT)
}

/// Extrai o texto reexecutando este binário; bloqueante (o servidor chama dentro de `spawn_blocking`).
/// Uma extração por vez: PDFs simultâneos esperam na fila em vez de lançar vários processos pesados.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn extract_via_child(bytes: &[u8]) -> Result<String, String> {
    let _fila = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let exe = std::env::current_exe().map_err(|e| format!("não foi possível localizar o executável: {e}"))?;
    run_child(&exe, bytes, timeout_from_env())
}

/// Garante `kill` + `wait` do filho em QUALQUER saída de `run_child` (retorno antecipado, pânico): nenhum zumbi nem órfão.
/// (Se a rota assíncrona for cancelada por desconexão do cliente, o `spawn_blocking` segue até o fim e passa por aqui;
/// se o próprio servidor morrer, o vigia do filho e o EOF da entrada o encerram.)
struct Reaper(std::process::Child);
impl std::ops::Deref for Reaper {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for Reaper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Drop for Reaper {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub fn run_child(exe: &Path, bytes: &[u8], timeout: Duration) -> Result<String, String> {
    let mut cmd = Command::new(exe);
    cmd.arg(PDF_CHILD_ARG).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = Reaper(cmd.spawn().map_err(|e| format!("não foi possível iniciar o leitor de PDF: {e}"))?);
    let (Some(mut stdin), Some(mut stdout), Some(mut stderr)) = (child.stdin.take(), child.stdout.take(), child.stderr.take()) else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("não foi possível abrir os canais do leitor de PDF".into());
    };
    let data = bytes.to_vec();
    // três threads: escrever a entrada, ler a saída (com teto) e ler o erro (curto); nenhuma bloqueia o laço do tempo limite
    let writer = thread::spawn(move || {
        let _ = stdin.write_all(&data); // o filho pode sair antes de ler tudo: erro ignorado; fechar a entrada = fim
    });
    // `capped`: a leitura parou porque a saída bateu no TETO. O fim da leitura por EOF (o filho fechou a saída, por
    // exemplo ao sair ou dar pânico) NÃO é teto: tratá-lo como teto mataria o filho antes de ler o código de saída e
    // devolveria "sucesso com texto vazio" (bug visto em D55: pânico virava "PDF sem texto").
    let capped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let capped_w = capped.clone();
    let reader = thread::spawn(move || {
        let mut v = Vec::new();
        let _ = (&mut stdout).take(PDF_OUT_CAP as u64).read_to_end(&mut v);
        if v.len() >= PDF_OUT_CAP {
            capped_w.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        v
    });
    let err_reader = thread::spawn(move || {
        let mut v = Vec::new();
        let _ = (&mut stderr).take(4096).read_to_end(&mut v);
        v
    });

    let start = Instant::now();
    let (status, killed_for_cap, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(s)) => break (Some(s), false, false),
            Ok(None) => {}
            Err(_) => break (None, false, false),
        }
        if capped.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = child.kill();
            let s = child.wait().ok();
            break (s, true, false);
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            break (None, false, true);
        }
        thread::sleep(Duration::from_millis(10));
    };
    let _ = writer.join();
    let out = reader.join().unwrap_or_default();
    let err = err_reader.join().unwrap_or_default();

    if timed_out {
        return Err(msg_timeout(timeout));
    }
    if killed_for_cap {
        // saída acima do teto: usa o que veio (o corte de caracteres e o aviso de "cortado" ficam em finish_pdf_text)
        return Ok(String::from_utf8_lossy(&out).into_owned());
    }
    let Some(status) = status else { return Err(msg_crash()) };
    match status.code() {
        Some(0) => Ok(String::from_utf8_lossy(&out).into_owned()),
        Some(EXIT_ENCRYPTED) => Err(msg_encrypted()),
        Some(EXIT_TOO_MANY_PAGES) => Err(msg_pages()),
        Some(EXIT_FAILED) | Some(EXIT_BAD_INPUT) => {
            let m = String::from_utf8_lossy(&err).trim().to_string();
            Err(if m.is_empty() { "unknown failure".to_string() } else { m })
        }
        _ => Err(msg_crash()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{
        content::{Content, Operation},
        dictionary, Document, EncryptionState, EncryptionVersion, Object, Permissions, Stream,
    };

    /// PDF de teste com uma página por item; o texto é escrito em WinAnsi (bytes), com a fonte Helvetica.
    pub fn build_pdf(pages: &[&[u8]]) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica", "Encoding" => "WinAnsiEncoding" });
        let resources_id = doc.add_object(dictionary! { "Font" => dictionary! { "F1" => font_id } });
        let mut kids: Vec<Object> = Vec::new();
        for text in pages {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 12.into()]),
                    Operation::new("Td", vec![50.into(), 700.into()]),
                    Operation::new("Tj", vec![Object::string_literal(text.to_vec())]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page", "Parent" => pages_id, "Contents" => content_id, "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            });
            kids.push(page_id.into());
        }
        doc.objects.insert(pages_id, Object::Dictionary(dictionary! { "Type" => "Pages", "Kids" => kids, "Count" => pages.len() as i64 }));
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    #[allow(deprecated)]
    fn encrypt_pdf(plain: &[u8]) -> Vec<u8> {
        let mut doc = Document::load_mem(plain).unwrap();
        let id = lopdf::Object::string_literal(b"0123456789abcdef".to_vec());
        doc.trailer.set("ID", lopdf::Object::Array(vec![id.clone(), id]));
        let reference = doc.clone();
        let version = EncryptionVersion::V2 { document: &reference, owner_password: "dono", user_password: "segredo", key_length: 128, permissions: Permissions::all() };
        let state = EncryptionState::try_from(version).unwrap();
        doc.encrypt(&state).unwrap();
        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    #[test]
    fn extrai_texto_simples() {
        let t = extract_in_process(&build_pdf(&[b"Hello World"])).unwrap();
        assert!(t.contains("Hello World"), "{t:?}");
    }

    #[test]
    fn extrai_varias_paginas_com_acentos() {
        // WinAnsi: \xE7 = ç, \xE3 = ã, \xE9 = é
        let t = extract_in_process(&build_pdf(&[b"Primeira A\xE7\xE3o", b"Segunda caf\xE9"])).unwrap();
        assert!(t.contains("Primeira Ação"), "{t:?}");
        assert!(t.contains("Segunda café"), "{t:?}");
        assert!(t.find("Primeira").unwrap() < t.find("Segunda").unwrap(), "ordem das páginas: {t:?}");
    }

    #[test]
    fn limite_de_paginas() {
        let pdf = build_pdf(&[b"um", b"dois", b"tres"]);
        assert_eq!(extract_with_limits(&pdf, 2), Err(CoreErr::TooManyPages(3)));
        assert!(extract_with_limits(&pdf, 3).is_ok()); // exatamente no limite passa
    }

    #[test]
    fn pdf_com_senha_e_reconhecido() {
        let enc = encrypt_pdf(&build_pdf(&[b"segredo"]));
        assert_eq!(extract_in_process(&enc), Err(CoreErr::Encrypted));
    }

    #[test]
    fn mensagens_finais_nao_levam_prefixo() {
        for m in [msg_encrypted(), msg_crash(), msg_pages(), msg_timeout(Duration::from_millis(1500))] {
            assert!(is_final_message(&m), "{m}");
        }
        assert!(!is_final_message("couldn't parse input"));
        assert_eq!(crate::errcode::english(&msg_timeout(Duration::from_millis(1500))), "Reading the PDF took too long (over 2 s) and was cancelled.");
        assert_eq!(crate::errcode::english(&msg_timeout(Duration::from_secs(20))), "Reading the PDF took too long (over 20 s) and was cancelled.");
        assert_eq!(crate::errcode::parse(&msg_timeout(Duration::from_secs(20))).unwrap().1, vec!["20"]);
    }
}
