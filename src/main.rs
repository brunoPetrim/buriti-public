mod attach;
mod catalog;
mod config;
mod download;
mod embeddings;
mod engine;
mod errcode;
mod fim;
mod gpu;
mod grammar;
mod history;
mod image;
mod jsonschema;
mod native;
mod openai;
mod pdf;
mod server;
mod tools;

use std::sync::Arc;

use config::Paths;
use engine::{llama_server::LlamaServerEngine, Engine};
use history::HistoryStore;
use server::{AppState, Engines};
use tokio::sync::RwLock;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Processo filho do leitor de PDF (D55): tratado ANTES de tudo (sem servidor, portas, navegador nem configurações).
    if std::env::args_os().nth(1).is_some_and(|a| a == pdf::PDF_CHILD_ARG) {
        std::process::exit(pdf::child_main());
    }
    // `buriti --gpu-info [--json]` (G1, D62): lists GPU adapters and exits; also used as the isolated child of /api/system.
    if std::env::args_os().nth(1).is_some_and(|a| a == gpu::GPU_INFO_ARG) {
        std::process::exit(gpu::cli_main());
    }
    // `buriti --gpu-probe <selector>` (G10, D69): opens one adapter in this throw-away process; the server runs it before opening a GPU itself.
    if std::env::args_os().nth(1).is_some_and(|a| a == gpu::GPU_PROBE_ARG) {
        std::process::exit(gpu::probe_main());
    }
    // `buriti --gpu-bench [--json]` (G3, D63) and its per-adapter child: memory bandwidth of the GPUs and the CPU.
    if std::env::args_os().nth(1).is_some_and(|a| a == gpu::bench::GPU_BENCH_CHILD_ARG) {
        std::process::exit(gpu::bench::child_main());
    }
    if std::env::args_os().nth(1).is_some_and(|a| a == gpu::bench::GPU_BENCH_ARG) {
        std::process::exit(gpu::bench::cli_main());
    }
    // `buriti --gpu-matvec [--json]` (G4/G5, D64) and its per-adapter child: throughput of the matvec kernels.
    if std::env::args_os().nth(1).is_some_and(|a| a == gpu::mvbench::GPU_MATVEC_CHILD_ARG) {
        std::process::exit(gpu::mvbench::child_main());
    }
    if std::env::args_os().nth(1).is_some_and(|a| a == gpu::mvbench::GPU_MATVEC_ARG) {
        std::process::exit(gpu::mvbench::cli_main());
    }
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let paths = Paths::from_env();
    paths.ensure_dirs()?;
    let settings = paths.load_settings();
    let history = HistoryStore::open(paths.history.clone(), &paths.key_file())?;

    // Token de sessão aleatório: só quem abre a URL impressa aqui acessa a API.
    let mut raw = [0u8; 24];
    getrandom::fill(&mut raw).map_err(|e| format!("sem fonte de aleatoriedade: {e}"))?;
    let token: String = raw.iter().map(|b| format!("{b:02x}")).collect();

    let api_key = paths.load_or_create_api_key()?;

    // A porta é escolhida (e aberta) antes de tudo: se a preferida estiver ocupada, usa a próxima livre
    let (listener, port, port_note) = paths.bind_web().await?;
    let engines = Engines {
        llama: Arc::new(LlamaServerEngine::new(paths.clone(), settings.engine_port)),
        native: Arc::new(native::NativeEngine::new(paths.clone())),
    };

    // Vida real: se já há motor e modelo configurados, o motor sobe sozinho.
    let can_start = settings.engine_kind == "native" || paths.engine_binary().is_file();
    if can_start && settings.has_model() {
        let engine: Arc<dyn Engine> = engines.get(&settings.engine_kind);
        if let Err(e) = engine.start(&settings).await {
            eprintln!("Aviso: não foi possível iniciar o motor automaticamente: {e}");
        }
    }

    let state = Arc::new(AppState {
        paths: paths.clone(),
        settings: RwLock::new(settings),
        engines,
        history,
        downloads: download::Downloads::new(),
        token: token.clone(),
        api_key: RwLock::new(api_key),
        port,
    });

    let url = format!("http://127.0.0.1:{port}/#token={token}");
    if let Some(n) = &port_note {
        println!("Aviso: {n}\n");
    }
    println!("Buriti rodando. Dados em: {}\nAbra no navegador (abrindo automaticamente):\n\n  {url}\n", paths.home.display());
    println!("Buriti Code - API compativel com a OpenAI: http://127.0.0.1:{port}/v1  (chave em {})\n", paths.api_key_file().display());
    if std::env::var_os("BURITI_NO_BROWSER").is_none() {
        open_browser(&url);
    }

    axum::serve(listener, server::router(state.clone()))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;

    state.engines.llama.stop().await;
    state.engines.native.stop().await;
    Ok(())
}

/// Abre a URL no navegador padrão (a URL só tem hex no fragmento, sem caracteres especiais de shell).
fn open_browser(url: &str) {
    #[cfg(windows)]
    let r = {
        use std::os::windows::process::CommandExt;
        std::process::Command::new("cmd").args(["/C", "start", "", url]).creation_flags(0x0800_0000).spawn()
    };
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = r {
        eprintln!("Não foi possível abrir o navegador automaticamente: {e}");
    }
}
