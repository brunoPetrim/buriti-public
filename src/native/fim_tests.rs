//! Modelo dedicado de autocompletar (D78): o pedido FIM vai para a sessão própria e o KV cache do chat não muda.

use super::*;
use futures_util::StreamExt;

fn model_path() -> Option<std::path::PathBuf> {
    let path = std::env::var("BURITI_TEST_MODEL").unwrap_or_else(|_| format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR")));
    let p = std::path::PathBuf::from(path);
    if !p.exists() {
        eprintln!("modelo de teste ausente, pulando: {}", p.display());
        return None;
    }
    Some(p)
}

fn loaded(path: &std::path::Path) -> Loaded {
    let model = Model::load_with(path, false).expect("carregar modelo");
    let session = Mutex::new(model.new_session(512));
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    Loaded { model, vision: None, session, pool, gpu: None, note: String::new(), trie: Default::default() }
}

fn engine_with(main: Loaded, fim: FimState) -> (NativeEngine, Arc<Loaded>) {
    let e = NativeEngine::new(Paths::from_env());
    let main = Arc::new(main);
    *e.state.lock().unwrap() = State::Ready(main.clone());
    *e.fim.state.lock().unwrap() = fim;
    (e, main)
}

async fn drain(s: ByteStream) -> String {
    let mut s = s;
    let mut out = String::new();
    while let Some(Ok(b)) = s.next().await {
        out.push_str(&String::from_utf8_lossy(&b));
    }
    out
}

fn chat_req(q: &str) -> ChatRequest {
    ChatRequest { messages: vec![ChatMsg { role: "user".into(), content: q.into() }], temperature: Some(0.0), max_tokens: Some(4), stop: vec![], sampling: Default::default(), images: vec![], constraint: None }
}

fn fim_req(prefix: &str, suffix: &str) -> CompletionRequest {
    CompletionRequest { prompt: prefix.into(), suffix: Some(suffix.into()), temperature: Some(0.0), max_tokens: Some(4), stop: vec![], sampling: Default::default(), max_prompt_tokens: None }
}

fn plain_req(prefix: &str) -> CompletionRequest {
    CompletionRequest { suffix: None, ..fim_req(prefix, "") }
}

/// Estado observável de uma sessão: tokens do cache e os bytes do KV até o último token.
fn snap(l: &Loaded) -> (Vec<u32>, Vec<Vec<u16>>, Vec<Vec<u16>>) {
    let s = l.session.lock().unwrap();
    let (k, v) = s.kv();
    let n = s.tokens.len() * (k[0].len() / s.ctx());
    (s.tokens.clone(), k.iter().map(|x| x[..n].to_vec()).collect(), v.iter().map(|x| x[..n].to_vec()).collect())
}

const CODE: &str = "fn add(a: u32, b: u32) -> u32 {\n    ";
const REST: &str = "\n}\n";

/// O roteiro: chat, depois autocompletar. Devolve os estados (chat antes, chat depois, FIM depois).
async fn script(e: &NativeEngine, main: &Arc<Loaded>) -> ((Vec<u32>, Vec<Vec<u16>>, Vec<Vec<u16>>), (Vec<u32>, Vec<Vec<u16>>, Vec<Vec<u16>>)) {
    drain(e.chat(chat_req("Say hello in one word.")).await.unwrap()).await;
    let before = snap(main);
    assert!(!before.0.is_empty());
    drain(e.complete(fim_req(CODE, REST)).await.unwrap()).await;
    (before, snap(main))
}

fn fim_ready(e: &NativeEngine) -> Arc<Loaded> {
    match &*e.fim.state.lock().unwrap() {
        FimState::Ready(f) => f.clone(),
        _ => panic!("FIM não está pronto"),
    }
}

#[tokio::test]
async fn fim_request_goes_to_the_dedicated_session_and_leaves_the_chat_cache_alone() {
    let Some(p) = model_path() else { return };
    let (main, fim) = (loaded(&p), loaded(&p));
    if !main.model.tok.has_fim() {
        eprintln!("o modelo de teste não tem tokens FIM, pulando");
        return;
    }
    let (e, main) = engine_with(main, FimState::Ready(Arc::new(fim)));
    let fim = fim_ready(&e);
    assert!(fim.session.lock().unwrap().tokens.is_empty());

    let (before, after) = script(&e, &main).await;
    // o chat ficou exatamente como estava: tokens e bytes do KV (igualdade estrita)
    assert_eq!(before, after, "o autocompletar mexeu no cache do chat");
    // e o pedido FIM deixou o seu prompt na sessão dedicada
    let (ftoks, _, _) = snap(&fim);
    assert!(ftoks.len() > 8, "a sessão FIM não recebeu o pedido");
    let fim_prompt = main.model.tok.fim_prompt(CODE, REST).unwrap();
    assert_eq!(&ftoks[..fim_prompt.len()], &fim_prompt[..], "a sessão FIM não guarda o prompt FIM");

    // outro FIM com o prefixo estendido: reaproveita o cache da sessão dedicada (D76) e o chat segue intacto
    drain(e.complete(fim_req(&format!("{CODE}a + b"), REST)).await.unwrap()).await;
    assert_eq!(snap(&main), before, "o segundo autocompletar mexeu no cache do chat");
    // um turno de chat, por sua vez, não toca na sessão FIM
    let fim_before = snap(&fim);
    drain(e.chat(chat_req("And in two words?")).await.unwrap()).await;
    assert_eq!(snap(&fim), fim_before, "o chat mexeu no cache do autocompletar");
    assert_ne!(snap(&main).0, before.0, "o chat deveria ter avançado o seu cache");
    // completar sem `suffix` (texto cru) fica no modelo do chat
    let fim_before = snap(&fim);
    drain(e.complete(plain_req("hello world")).await.unwrap()).await;
    assert_eq!(snap(&fim), fim_before, "um completar sem suffix foi para o modelo FIM");
}

#[tokio::test]
async fn without_a_dedicated_model_autocomplete_evicts_the_chat_cache() {
    let Some(p) = model_path() else { return };
    let main = loaded(&p);
    if !main.model.tok.has_fim() {
        return;
    }
    let (e, main) = engine_with(main, FimState::Off);
    let (before, after) = script(&e, &main).await;
    assert_ne!(before.0, after.0, "no modo compartilhado o FIM deveria trocar o cache da sessão única (é o problema que o modelo dedicado resolve)");
}

/// Teste com dentes: se o pedido FIM fosse para a sessão do chat, o cache do chat mudaria e o teste reprova.
#[tokio::test]
#[should_panic(expected = "mexeu no cache do chat")]
async fn planted_bug_fim_routed_to_the_chat_session_is_caught() {
    let Some(p) = model_path() else { panic!("mexeu no cache do chat (sem modelo o teste não prova nada)") };
    let (main, fim) = (loaded(&p), loaded(&p));
    if !main.model.tok.has_fim() {
        panic!("mexeu no cache do chat (sem tokens FIM o teste não prova nada)");
    }
    let (e, main) = engine_with(main, FimState::Ready(Arc::new(fim)));
    PLANT_FIM_TO_CHAT.with(|c| c.set(true));
    let (before, after) = script(&e, &main).await;
    PLANT_FIM_TO_CHAT.with(|c| c.set(false));
    assert_eq!(before, after, "o autocompletar mexeu no cache do chat");
}

#[tokio::test]
async fn on_demand_loads_in_the_background_while_the_chat_model_serves_then_routes_to_the_dedicated_one() {
    let Some(p) = model_path() else { return };
    let main = loaded(&p);
    if !main.model.tok.has_fim() {
        return;
    }
    let job = FimJob { path: p.clone(), threads: 2, ctx: 512, plan: None };
    let (e, main) = engine_with(main, FimState::Idle(job));
    // o primeiro pedido FIM não espera a carga: o modelo do chat o atende (o cache dele muda) e a carga começa
    drain(e.complete(fim_req(CODE, REST)).await.unwrap()).await;
    assert!(!snap(&main).0.is_empty(), "o primeiro pedido deveria ter sido atendido pelo modelo do chat");
    let t0 = Instant::now();
    loop {
        if matches!(&*e.fim.state.lock().unwrap(), FimState::Ready(_)) {
            break;
        }
        assert!(t0.elapsed().as_secs() < 90, "a carga sob demanda não terminou em 90 s");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let before = snap(&main);
    drain(e.complete(fim_req(CODE, REST)).await.unwrap()).await;
    assert_eq!(snap(&main), before, "depois de carregado o FIM deveria ir para a sessão dedicada");
    assert!(!snap(&fim_ready(&e)).0.is_empty());
}

#[test]
fn failed_or_stale_loads_fall_back_to_shared() {
    let slot = Arc::new(FimSlot { state: Mutex::new(FimState::Loading), last_used: Mutex::new(Instant::now()) });
    let epoch = Arc::new(AtomicU64::new(5));
    // arquivo inexistente: volta para Off (modo compartilhado), sem pânico
    finish_fim_load(&slot, &epoch, 5, FimJob { path: "C:\\nao\\existe\\fim.gguf".into(), threads: 1, ctx: 512, plan: None }, false);
    assert!(matches!(&*slot.state.lock().unwrap(), FimState::Off));
    // carga antiga (o motor parou/reiniciou enquanto carregava): descartada, não publica nada
    *slot.state.lock().unwrap() = FimState::Loading;
    finish_fim_load(&slot, &epoch, 4, FimJob { path: "C:\\nao\\existe\\fim.gguf".into(), threads: 1, ctx: 512, plan: None }, false);
    assert!(matches!(&*slot.state.lock().unwrap(), FimState::Loading));
    let note = fim_fallback_note("x.");
    assert!(crate::errcode::english(&note).contains("the chat model serves both"), "{note}");
    assert_eq!(crate::errcode::code_of(&note).as_deref(), Some("err_fim_fallback"));
}

#[tokio::test]
async fn stop_unloads_the_dedicated_model() {
    let Some(p) = model_path() else { return };
    let (e, _main) = engine_with(loaded(&p), FimState::Ready(Arc::new(loaded(&p))));
    assert!(matches!(&*e.fim.state.lock().unwrap(), FimState::Ready(_)));
    e.stop().await;
    assert!(matches!(&*e.fim.state.lock().unwrap(), FimState::Off));
}

#[tokio::test]
async fn setup_with_the_setting_off_or_the_same_file_loads_nothing() {
    let Some(p) = model_path() else { return };
    let (e, main) = engine_with(loaded(&p), FimState::Off);
    let epoch = e.epoch.load(Ordering::SeqCst);
    // modo "off": nada é carregado
    setup_fim(&e.fim, &e.epoch, epoch, &main, &p, &Ok(Some(p.clone())), false, 512, 2, None);
    assert!(matches!(&*e.fim.state.lock().unwrap(), FimState::Off));
    // o mesmo arquivo do chat: não faz sentido duplicar
    setup_fim(&e.fim, &e.epoch, epoch, &main, &p, &Ok(Some(p.clone())), true, 512, 2, None);
    assert!(matches!(&*e.fim.state.lock().unwrap(), FimState::Off));
    // arquivo escolhido com defeito: o motor do chat segue e o estado fica Off
    setup_fim(&e.fim, &e.epoch, epoch, &main, &p, &Err("Modelo de autocompletar não encontrado: x".into()), true, 512, 2, None);
    assert!(matches!(&*e.fim.state.lock().unwrap(), FimState::Off));
}
