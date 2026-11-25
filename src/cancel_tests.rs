//! D76: um cliente que desconecta (o editor derrubou um autocompletar) para a geração na hora e o pedido mais novo roda em seguida.
//! O teste usa o servidor HTTP de verdade (axum + hyper, em uma porta livre de loopback) e um motor falso com a mesma estrutura do
//! motor próprio (uma sessão única, uma thread bloqueante por pedido, `tx.is_closed()` conferido a cada passo). Tudo tem prazo.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use axum::body::Bytes;
use futures_util::{stream, StreamExt};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::{mpsc, RwLock},
};

use super::*;
use crate::engine::{ByteStream, CompletionRequest, EngineStatus};

const STEP: Duration = Duration::from_millis(20);

#[derive(Default)]
struct Log(Mutex<Vec<String>>);
impl Log {
    fn add(&self, s: String) {
        self.0.lock().unwrap().push(s);
    }
    fn has(&self, s: &str) -> bool {
        self.0.lock().unwrap().iter().any(|x| x == s)
    }
}

/// Motor falso. O prompt escolhe o comportamento: começa com "long" = 200 passos de geração (4 s); "silent" = 200 passos de
/// "prefill" sem emitir nada antes de gerar; qualquer outro = 3 passos.
struct Fake {
    session: Arc<Mutex<()>>,
    /// bug plantado: o motor "esquece" de conferir se o cliente saiu (o teste tem de reprovar)
    blind: bool,
    log: Arc<Log>,
}

impl Fake {
    fn start(&self, prompt: String) -> ByteStream {
        let (tx, mut rx) = mpsc::channel::<Bytes>(64);
        let (session, log, blind) = (self.session.clone(), self.log.clone(), self.blind);
        tokio::task::spawn_blocking(move || {
            let _guard = session.lock().unwrap_or_else(|e| e.into_inner()); // a "fila": uma sessão, um pedido por vez
            if tx.is_closed() && !blind {
                log.add(format!("dropped:{prompt}"));
                return;
            }
            log.add(format!("started:{prompt}"));
            let (prefill, gen) = if prompt.starts_with("silent") { (200, 200) } else if prompt.starts_with("long") { (0, 200) } else { (0, 3) };
            for _ in 0..prefill {
                if tx.is_closed() && !blind {
                    log.add(format!("cancelled:{prompt}"));
                    return;
                }
                std::thread::sleep(STEP);
            }
            for _ in 0..gen {
                if tx.is_closed() && !blind {
                    log.add(format!("cancelled:{prompt}"));
                    return;
                }
                let chunk = r#"data: {"object":"text_completion","choices":[{"index":0,"text":"x","finish_reason":null}]}"#;
                if tx.blocking_send(Bytes::from(format!("{chunk}\n\n"))).is_err() && !blind {
                    log.add(format!("cancelled:{prompt}"));
                    return;
                }
                std::thread::sleep(STEP);
            }
            log.add(format!("done:{prompt}"));
            let _ = tx.blocking_send(Bytes::from_static(b"data: [DONE]\n\n"));
        });
        Box::pin(stream::poll_fn(move |cx| rx.poll_recv(cx)).map(|b| Ok::<Bytes, Box<dyn std::error::Error + Send + Sync>>(b)))
    }
}

#[async_trait]
impl Engine for Fake {
    fn name(&self) -> &'static str {
        "fake"
    }
    async fn start(&self, _s: &Settings) -> Result<(), String> {
        Ok(())
    }
    async fn stop(&self) {}
    async fn status(&self) -> EngineStatus {
        EngineStatus { state: "ready", message: None }
    }
    async fn chat(&self, req: ChatRequest) -> Result<ByteStream, String> {
        Ok(self.start(req.messages.last().map(|m| m.content.clone()).unwrap_or_default()))
    }
    async fn complete(&self, req: CompletionRequest) -> Result<ByteStream, String> {
        Ok(self.start(req.prompt))
    }
}

struct Rig {
    port: u16,
    log: Arc<Log>,
    _dir: PathBuf,
}

async fn rig() -> Rig {
    rig_with(false).await
}

async fn rig_with(blind: bool) -> Rig {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!("buriti-cancel-test-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    let paths = Paths { home: dir.clone(), models: dir.join("models"), engine_dir: dir.join("engine"), history: dir.join("history"), settings_file: dir.join("settings.json") };
    let history = HistoryStore::open(dir.join("history"), &dir.join("key.bin")).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let log = Arc::new(Log::default());
    let fake: Arc<dyn Engine> = Arc::new(Fake { session: Arc::new(Mutex::new(())), log: log.clone(), blind });
    let settings = Settings { engine_kind: "native".into(), ..Settings::default() };
    let state = Arc::new(AppState {
        paths,
        settings: RwLock::new(settings),
        engines: Engines { llama: fake.clone(), native: fake },
        history,
        downloads: crate::download::Downloads::new(),
        token: "t".into(),
        api_key: RwLock::new("testkey".into()),
        port,
    });
    tokio::spawn(async move {
        let _ = axum::serve(listener, router(state)).await;
    });
    Rig { port, log, _dir: dir }
}

async fn post(port: u16, path: &str, body: &str) -> TcpStream {
    let mut s = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let req = format!("POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer testkey\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
    s.write_all(req.as_bytes()).await.unwrap();
    s
}

/// Lê até aparecer `needle` na resposta (ou o fim da conexão); devolve se apareceu.
async fn read_until(s: &mut TcpStream, needle: &str) -> bool {
    let mut acc = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match s.read(&mut buf).await {
            Ok(0) | Err(_) => return String::from_utf8_lossy(&acc).contains(needle),
            Ok(n) => {
                acc.extend_from_slice(&buf[..n]);
                if String::from_utf8_lossy(&acc).contains(needle) {
                    return true;
                }
            }
        }
    }
}

fn body(prompt: &str, stream: bool) -> String {
    json!({ "prompt": prompt, "max_tokens": 50, "stream": stream }).to_string()
}

/// A mais velha (`first`) está rodando ou no "prefill"; o cliente dela desconecta; a nova tem de começar logo, e a velha tem de
/// ter sido cancelada (não terminada). Devolve o tempo entre a desconexão e o 1º byte da nova.
async fn newer_runs_after_disconnect(first_prompt: &str, first_stream: bool, wait_for_chunk: bool, blind: bool) -> Duration {
    let r = rig_with(blind).await;
    let mut a = post(r.port, "/v1/completions", &body(first_prompt, first_stream)).await;
    if wait_for_chunk {
        assert!(read_until(&mut a, "\"text\":\"x\"").await, "o pedido velho não começou a gerar");
    } else {
        tokio::time::sleep(Duration::from_millis(200)).await; // deixa o motor entrar no prefill silencioso
        assert!(r.log.has(&format!("started:{first_prompt}")));
    }
    let mut b = post(r.port, "/v1/completions", &body("new", true)).await; // chega na fila com o velho ainda rodando
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!r.log.has("started:new"), "a fila deveria estar presa atrás do pedido velho");
    let t0 = Instant::now();
    drop(a); // o editor derrubou o autocompletar
    assert!(read_until(&mut b, "[DONE]").await, "o pedido novo não terminou");
    let took = t0.elapsed();
    assert!(r.log.has(&format!("cancelled:{first_prompt}")), "o velho deveria ter sido cancelado: {:?}", r.log.0.lock().unwrap());
    assert!(!r.log.has(&format!("done:{first_prompt}")));
    took
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn newer_request_runs_right_after_the_client_of_a_streaming_generation_leaves() {
    let t = tokio::time::timeout(Duration::from_secs(15), newer_runs_after_disconnect("long-stream", true, true, false)).await.expect("passou do prazo: o pedido velho não foi cancelado");
    eprintln!("stream gerando: o pedido novo terminou {} ms após a desconexão (o velho levaria 4 s)", t.as_millis());
    assert!(t < Duration::from_secs(1), "{t:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn newer_request_runs_right_after_the_client_leaves_during_a_silent_prefill() {
    let t = tokio::time::timeout(Duration::from_secs(15), newer_runs_after_disconnect("silent-stream", true, false, false)).await.expect("passou do prazo: o pedido velho não foi cancelado");
    eprintln!("stream no prefill: o pedido novo terminou {} ms após a desconexão (o velho levaria 8 s)", t.as_millis());
    assert!(t < Duration::from_secs(1), "{t:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn newer_request_runs_right_after_the_client_of_a_non_streaming_request_leaves() {
    let t = tokio::time::timeout(Duration::from_secs(15), newer_runs_after_disconnect("silent-plain", false, false, false)).await.expect("passou do prazo: o pedido velho não foi cancelado");
    eprintln!("sem stream, no prefill: o pedido novo terminou {} ms após a desconexão", t.as_millis());
    assert!(t < Duration::from_secs(1), "{t:?}");
}

/// Pedidos mortos NA FILA nunca rodam: com um pedido ativo e três esperando, todos os clientes saem; o pedido novo é o próximo.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dead_queued_requests_are_dropped_without_running() {
    let r = rig().await;
    let work = async {
        let mut a = post(r.port, "/v1/completions", &body("long-a", true)).await;
        assert!(read_until(&mut a, "\"text\":\"x\"").await);
        let q1 = post(r.port, "/v1/completions", &body("long-q1", true)).await;
        let q2 = post(r.port, "/v1/completions", &body("long-q2", false)).await;
        let q3 = post(r.port, "/v1/completions", &body("long-q3", true)).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let mut fresh = post(r.port, "/v1/completions", &body("new", true)).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        let t0 = Instant::now();
        drop((a, q1, q2, q3));
        assert!(read_until(&mut fresh, "[DONE]").await);
        t0.elapsed()
    };
    let took = tokio::time::timeout(Duration::from_secs(15), work).await.expect("passou do prazo: a fila guardou pedidos mortos");
    eprintln!("fila com 3 mortos: o pedido novo terminou {} ms após as desconexões (cada morto levaria 4 s)", took.as_millis());
    for q in ["long-q1", "long-q2", "long-q3"] {
        assert!(!r.log.has(&format!("started:{q}")), "o pedido morto {q} chegou a rodar: {:?}", r.log.0.lock().unwrap());
    }
    assert!(r.log.has("done:new"));
    assert!(took < Duration::from_secs(1), "{took:?}");
}

/// Teste com dentes: se o motor não conferir a desconexão, o pedido velho roda até o fim e o teste tem de reprovar.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[should_panic(expected = "deveria ter sido cancelado")]
async fn engine_that_ignores_the_disconnect_is_caught() {
    // o motor "cego" roda o pedido velho até o fim (alguns segundos na máquina de desenvolvimento, mais num runner lento do CI): o prazo
    // tem de ser folgado para o teste chegar à afirmação "deveria ter sido cancelado" e não falhar antes por tempo
    tokio::time::timeout(Duration::from_secs(120), newer_runs_after_disconnect("long-blind", true, true, true)).await.expect("passou do prazo");
}
