//! Motor próprio do Buriti (fase 2): inferência em Rust puro, sem llama.cpp.
//!
//! Hoje: arquitetura Qwen2 (Qwen 2.5), quantizações Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/F16/BF16/F32,
//! só CPU, com kernels AVX2 explícitos (reserva genérica em outras CPUs) e prefill em lote.
//! É experimental e mais lento que o llama.cpp.

pub(crate) mod f16;
pub(crate) mod gguf;
mod kernels;
pub(crate) mod model;
pub(crate) mod quant;
mod repack;
#[cfg(test)]
mod repack_exp;
#[cfg(test)]
mod reuse_tests;
#[cfg(test)]
mod fim_tests;
#[cfg(test)]
mod json_tests;
mod stopper;
mod tokenizer;
mod vision;
pub(crate) mod vmath;

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use async_trait::async_trait;
use axum::body::Bytes;
use futures_util::stream;
use serde_json::json;
use tokio::sync::mpsc;

use crate::{
    config::{Paths, Settings},
    engine::{self, ByteStream, ChatMsg, ChatRequest, CompletionRequest, Engine, EngineStatus, SamplingOpts},
};
use model::{Model, Session};
use tokenizer::Utf8Stream;

/// O backend de GPU (D69): o modelo inteiro na placa, um token por submissão e prefill em lote. Sem a feature `gpu` o tipo existe mas
/// nunca é construído.
#[cfg(feature = "gpu")]
type Gpu = crate::gpu::model::GpuModel;
#[cfg(not(feature = "gpu"))]
type Gpu = std::convert::Infallible;

struct Loaded {
    model: Model,
    /// torre de visão (`<modelo>.mmproj.gguf` ao lado do modelo; só no formato SmolVLM), se existir e combinar com o modelo
    vision: Option<vision::mmproj::Mmproj>,
    session: Mutex<Session>,
    pool: rayon::ThreadPool,
    /// se existe, o forward (prefill e geração) roda na GPU e o KV cache fica na memória de vídeo; `session` só guarda os tokens e os logits
    gpu: Option<Gpu>,
    /// aviso para a interface (em português; a interface o traduz por padrão de texto): onde está calculando ou por que a GPU não foi usada
    note: String,
    /// trie dos bytes do vocabulário, para a saída restrita por gramática (D82); montada na primeira vez que um pedido a usa
    trie: std::sync::OnceLock<Arc<crate::grammar::TokenTrie>>,
}

impl Loaded {
    fn trie(&self) -> Arc<crate::grammar::TokenTrie> {
        self.trie
            .get_or_init(|| {
                let tok = &self.model.tok;
                Arc::new(crate::grammar::TokenTrie::build(tok.vocab_size(), |id| tok.text_token_bytes(id)))
            })
            .clone()
    }
}

enum State {
    Stopped,
    Loading,
    Ready(Arc<Loaded>),
    Error(String),
}

/// O que fazer com o modelo FIM dedicado (D78), decidido por `fim::fim_plan` quando o modelo do chat fica pronto.
struct FimJob {
    path: std::path::PathBuf,
    threads: usize,
    ctx: usize,
    /// o plano (JSON da rota) que levou a esta carga
    plan: Option<serde_json::Value>,
}

enum FimState {
    /// modo compartilhado (ou desligado, ou sem modelo): nada carregado
    Off,
    /// modo `dedicated_on_demand` antes do primeiro pedido FIM (ou depois de descartado por inatividade)
    Idle(FimJob),
    Loading,
    Ready(Arc<Loaded>),
}

/// O segundo modelo (autocompletar) com a sua própria sessão e KV cache. Os pedidos com `suffix` vão para ele quando está pronto;
/// em qualquer outro estado o modelo do chat atende (modo compartilhado), então a carga nunca bloqueia nem quebra o chat.
struct FimSlot {
    state: Mutex<FimState>,
    last_used: Mutex<Instant>,
}

pub struct NativeEngine {
    paths: Paths,
    state: Arc<Mutex<State>>,
    /// cada start/stop incrementa; uma carga antiga que termina tarde é descartada
    epoch: Arc<AtomicU64>,
    fim: Arc<FimSlot>,
}

impl NativeEngine {
    pub fn new(paths: Paths) -> Self {
        Self { paths, state: Arc::new(Mutex::new(State::Stopped)), epoch: Arc::new(AtomicU64::new(0)), fim: Arc::new(FimSlot { state: Mutex::new(FimState::Off), last_used: Mutex::new(Instant::now()) }) }
    }

    /// Modelo que atende um pedido: o FIM dedicado para `suffix` (se pronto); o do chat em todo o resto. Com FIM em espera
    /// (`dedicated_on_demand`) o primeiro pedido dispara a carga em segundo plano e é atendido pelo modelo do chat.
    fn route(&self, main: &Arc<Loaded>, has_suffix: bool) -> Arc<Loaded> {
        #[cfg(test)]
        if PLANT_FIM_TO_CHAT.with(|c| c.get()) {
            return main.clone(); // bug plantado pelos testes: o pedido FIM vai para a sessão do chat
        }
        if !has_suffix {
            return main.clone();
        }
        let mut st = self.fim.state.lock().unwrap_or_else(|e| e.into_inner());
        if let FimState::Ready(f) = &*st {
            *self.fim.last_used.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
            return f.clone();
        }
        if matches!(&*st, FimState::Idle(_)) {
            if let FimState::Idle(job) = std::mem::replace(&mut *st, FimState::Loading) {
                drop(st);
                self.spawn_fim_load(job, self.epoch.load(Ordering::SeqCst), true);
            }
        }
        main.clone()
    }

    /// Carrega o modelo FIM numa thread à parte (o chat segue atendendo) e publica o resultado se o motor não mudou nesse meio tempo.
    fn spawn_fim_load(&self, job: FimJob, my_epoch: u64, watch_idle: bool) {
        let (slot, epoch) = (self.fim.clone(), self.epoch.clone());
        crate::fim::set_live(Some(crate::fim::Live { state: "loading", model: file_label(&job.path), plan: job.plan.clone(), ..Default::default() }));
        std::thread::spawn(move || finish_fim_load(&slot, &epoch, my_epoch, job, watch_idle));
    }
}

fn file_label(p: &std::path::Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// Corpo da carga do modelo FIM (CPU). Falha (arquivo, arquitetura, sem tokens FIM): volta para `Off` e avisa pela rota do plano.
fn finish_fim_load(slot: &Arc<FimSlot>, epoch: &Arc<AtomicU64>, my_epoch: u64, job: FimJob, watch_idle: bool) {
    let t0 = Instant::now();
    let res = load_fim(&job);
    if epoch.load(Ordering::SeqCst) != my_epoch {
        return; // o motor parou ou reiniciou enquanto carregava: descarta
    }
    let name = file_label(&job.path);
    match res {
        Ok(l) => {
            eprintln!("[native] modelo de autocompletar dedicado carregado em {:.1}s: {name}", t0.elapsed().as_secs_f32());
            *slot.last_used.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
            *slot.state.lock().unwrap_or_else(|e| e.into_inner()) = FimState::Ready(Arc::new(l));
            crate::fim::set_live(Some(crate::fim::Live { state: "ready", model: name, plan: job.plan.clone(), ..Default::default() }));
            if watch_idle {
                // sob demanda: descarta depois de `ON_DEMAND_IDLE_SECS` sem pedido FIM e volta a esperar
                let (slot, epoch) = (slot.clone(), epoch.clone());
                std::thread::spawn(move || loop {
                    std::thread::sleep(std::time::Duration::from_secs(30));
                    if epoch.load(Ordering::SeqCst) != my_epoch {
                        return;
                    }
                    let idle = slot.last_used.lock().unwrap_or_else(|e| e.into_inner()).elapsed().as_secs();
                    if idle >= crate::fim::ON_DEMAND_IDLE_SECS {
                        *slot.state.lock().unwrap_or_else(|e| e.into_inner()) = FimState::Idle(FimJob { path: job.path.clone(), threads: job.threads, ctx: job.ctx, plan: job.plan.clone() });
                        crate::fim::set_live(Some(crate::fim::Live { state: "idle", model: file_label(&job.path), plan: job.plan.clone(), ..Default::default() }));
                        eprintln!("[native] autocomplete model dropped after being idle");
                        return;
                    }
                });
            }
        }
        Err(e) => {
            eprintln!("[native] autocomplete model not loaded: {}", crate::errcode::english(&e));
            *slot.state.lock().unwrap_or_else(|e| e.into_inner()) = FimState::Off;
            crate::fim::set_live(Some(crate::fim::Live { state: "failed", note: fim_fallback_note(&e), model: name, plan: job.plan.clone() }));
        }
    }
}

/// Aviso, com código (D83), de que o autocompletar ficou no modelo do chat; `why` pode ser um erro com código e é traduzido por dentro.
fn fim_fallback_note(why: &str) -> String {
    let en = crate::errcode::english(why);
    crate::errcode::coded("err_fim_fallback", &[why], format!("Autocomplete model not used ({}); the chat model serves both.", en.trim_end_matches('.')))
}

/// Carrega o modelo FIM: sempre na CPU (com o reempacotamento), com metade das threads do chat (os dois podem rodar ao mesmo tempo).
fn load_fim(job: &FimJob) -> Result<Loaded, String> {
    let model = Model::load(&job.path)?;
    if !model.tok.has_fim() {
        return Err("este modelo não tem tokens de preenchimento (FIM); escolha um modelo de código como o Qwen2.5-Coder".into());
    }
    let ctx = job.ctx.min(model.cfg.n_ctx_train).max(256);
    let session = Mutex::new(model.new_session(ctx));
    let pool = rayon::ThreadPoolBuilder::new().num_threads(job.threads.max(1)).build().map_err(|e| format!("pool de threads: {e}"))?;
    Ok(Loaded { model, vision: None, session, pool, gpu: None, note: String::new(), trie: Default::default() })
}

/// Decide o modo do FIM pela memória medida (função pura `fim::fim_plan`) e prepara o slot. Roda depois que o modelo do chat está
/// pronto, na thread da carga (o chat já atende). `vram_mb`: da mesma fonte do plano de GPU.
#[allow(clippy::too_many_arguments)]
fn setup_fim(slot: &Arc<FimSlot>, epoch: &Arc<AtomicU64>, my_epoch: u64, chat: &Loaded, chat_path: &std::path::Path, fim_cfg: &Result<Option<std::path::PathBuf>, String>, enabled: bool, ctx: usize, threads: usize, vram_mb: Option<u64>) {
    use crate::fim::{fim_plan, FimInput, FimMode, Live};
    let publish_off = |state: &'static str, note: String, model: String, plan: Option<serde_json::Value>| {
        *slot.state.lock().unwrap_or_else(|e| e.into_inner()) = FimState::Off;
        crate::fim::set_live(Some(Live { state, note, model, plan }));
    };
    let path = match fim_cfg {
        Err(e) => return publish_off("failed", fim_fallback_note(e), String::new(), None),
        Ok(p) => p.clone(),
    };
    let same_file = path.as_ref().is_some_and(|p| p == chat_path || matches!((std::fs::canonicalize(p), std::fs::canonicalize(chat_path)), (Ok(a), Ok(b)) if a == b));
    let path = if same_file { None } else { path };
    let (ram_total_mb, ram_available_mb) = crate::embeddings::measure_ram();
    let input = FimInput {
        ram_total_mb,
        ram_available_mb,
        vram_mb,
        chat_on_gpu: chat.gpu.is_some(),
        chat_bytes: std::fs::metadata(chat_path).map(|m| m.len()).unwrap_or(0),
        fim_bytes: path.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()),
        ctx: ctx as u32,
        enabled,
    };
    let plan = fim_plan(&input);
    eprintln!("[native] plano do autocompletar: {} ({}), sobram {} MiB", plan.mode.name(), plan.reason_code, plan.free_mb);
    let pj = Some(plan.to_json());
    let model = path.as_ref().map(|p| file_label(p)).unwrap_or_default();
    let Some(path) = path.filter(|_| plan.mode != FimMode::Shared) else {
        return publish_off("off", String::new(), model, pj);
    };
    let job = FimJob { path, threads: threads.div_ceil(2), ctx, plan: pj.clone() };
    if plan.mode == FimMode::DedicatedOnDemand {
        *slot.state.lock().unwrap_or_else(|e| e.into_inner()) = FimState::Idle(job);
        crate::fim::set_live(Some(Live { state: "idle", model, plan: pj, ..Default::default() }));
        return;
    }
    *slot.state.lock().unwrap_or_else(|e| e.into_inner()) = FimState::Loading;
    crate::fim::set_live(Some(Live { state: "loading", model, plan: pj.clone(), ..Default::default() }));
    finish_fim_load(slot, epoch, my_epoch, job, false);
}

#[async_trait]
impl Engine for NativeEngine {
    fn name(&self) -> &'static str {
        "Buriti (motor próprio, experimental)"
    }

    fn supports_constraint(&self) -> bool {
        true
    }

    async fn start(&self, s: &Settings) -> Result<(), String> {
        self.stop().await;
        let path = s.model_file(&self.paths)?;
        let threads = if s.threads > 0 {
            s.threads as usize
        } else {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2)
        };
        let want_ctx = s.ctx_size as usize;
        // onde calcular (D69): só consulta as GPUs (num processo filho, para um driver que trava não derrubar o servidor) se não for "cpu"
        let (mode, device) = (s.processor.clone(), s.gpu_device.clone());
        // modelo FIM dedicado (D78): resolvido aqui; um arquivo ruim não impede o motor de subir (o chat segue, em modo compartilhado)
        let fim_cfg = s.fim_file(&self.paths);
        let fim_enabled = s.fim_mode != "off";
        let fim_slot = self.fim.clone();
        crate::fim::set_live(None);
        let plan = if mode == "cpu" { None } else { Some(gpu_plan().await) };
        let my_epoch = self.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        *self.state.lock().unwrap() = State::Loading;

        let (state, epoch) = (self.state.clone(), self.epoch.clone());
        tokio::task::spawn_blocking(move || {
            let t0 = Instant::now();
            let res = (|| -> Result<Loaded, String> {
                // com a GPU em jogo o reempacotamento da CPU não é necessário (e duplicaria os pesos na RAM): carrega sem ele e só refaz
                // com ele se a CPU for a escolhida
                let model = if mode == "cpu" { Model::load(&path)? } else { Model::load_with(&path, false)? };
                let ctx = want_ctx.min(model.cfg.n_ctx_train).max(256);
                let (gpu, note) = pick_backend(&model, &path, ctx, &mode, &device, plan.as_ref());
                let model = if gpu.is_none() && mode != "cpu" { Model::load(&path)? } else { model };
                // a torre de visão roda na CPU e injeta vetores no forward da CPU: com a GPU ativa as imagens não são oferecidas (tarefa G11)
                let vision = if gpu.is_some() { None } else { load_vision(&path, &model) };
                let session = Mutex::new(model.new_session(ctx));
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| format!("pool de threads: {e}"))?;
                Ok(Loaded { model, vision, session, pool, gpu, note, trie: Default::default() })
            })();
            if epoch.load(Ordering::SeqCst) == my_epoch {
                if res.is_ok() {
                    eprintln!("[native] modelo carregado em {:.1}s ({threads} threads)", t0.elapsed().as_secs_f32());
                }
                match res {
                    Ok(l) => {
                        let l = Arc::new(l);
                        *state.lock().unwrap() = State::Ready(l.clone());
                        // o chat já atende; o modelo FIM (se o plano de memória o permitir) carrega agora, nesta mesma thread
                        let vram = plan.as_ref().and_then(|p| {
                            let name = crate::fim::gpu_name_from_status(Some(&l.note))?;
                            p.infos.iter().find(|i| crate::gpu::same_gpu(&i.name, &name)).and_then(|i| i.vram_mb)
                        });
                        setup_fim(&fim_slot, &epoch, my_epoch, &l, &path, &fim_cfg, fim_enabled, want_ctx, threads, vram);
                    }
                    Err(e) => *state.lock().unwrap() = State::Error(e),
                }
            }
        });
        Ok(())
    }

    async fn stop(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        *self.state.lock().unwrap() = State::Stopped;
        *self.fim.state.lock().unwrap_or_else(|e| e.into_inner()) = FimState::Off; // descarrega o modelo FIM (quem ainda o usa segura o Arc até acabar)
        crate::fim::set_live(None);
    }

    async fn status(&self) -> EngineStatus {
        match &*self.state.lock().unwrap() {
            State::Stopped => EngineStatus { state: "stopped", message: None },
            State::Loading => EngineStatus { state: "starting", message: None },
            State::Ready(l) => EngineStatus { state: "ready", message: if l.note.is_empty() { None } else { Some(l.note.clone()) } },
            State::Error(e) => EngineStatus { state: "error", message: Some(e.clone()) },
        }
    }

    async fn chat(&self, req: ChatRequest) -> Result<ByteStream, String> {
        let loaded = self.ready()?;
        let (tx, rx) = mpsc::channel::<Bytes>(64);
        tokio::task::spawn_blocking(move || {
            // um único gerador por vez (o KV cache é compartilhado entre os turnos da conversa)
            with_session(&loaded.session, &tx, |sess| {
                loaded.pool.install(|| {
                    // saída restrita (D82): a máscara de tokens sai da gramática e da trie do vocabulário
                    let constraint = match &req.constraint {
                        None => None,
                        Some(g) => match crate::grammar::Constraint::new(g.clone(), loaded.trie(), loaded.model.tok.eos.clone()) {
                            Ok(c) => Some(c),
                            Err(e) => {
                                let _ = tx.blocking_send(sse_usage(Kind::Chat, &format!("[{e}]"), Some("stop"), Some((0, 0))));
                                return;
                            }
                        },
                    };
                    generate(&loaded.model, loaded.vision.as_ref(), loaded.gpu.as_ref(), sess, &req, &tx, constraint)
                })
            });
            let _ = tx.blocking_send(Bytes::from_static(b"data: [DONE]\n\n"));
        });
        Ok(receiver_stream(rx))
    }

    /// Embeddings com o modelo já carregado (D77). Sessão própria (não toca no KV cache do chat); roda no pool do motor.
    async fn embed(&self, job: crate::embeddings::EmbedJob) -> Result<crate::embeddings::EmbedOutput, String> {
        let loaded = self.ready()?;
        tokio::task::spawn_blocking(move || loaded.pool.install(|| crate::embeddings::embed_job(&loaded.model, &job))).await.map_err(|e| format!("embeddings: {e}"))?
    }

    fn tool_support(&self) -> crate::tools::ToolSupport {
        match &*self.state.lock().unwrap() {
            State::Ready(l) => l.model.tok.tool_support(),
            _ => crate::tools::ToolSupport::No("Nenhum modelo carregado no motor próprio.".into()),
        }
    }

    fn vision(&self) -> Option<crate::image::VisionCaps> {
        match &*self.state.lock().unwrap() {
            State::Ready(l) => l.vision.as_ref().map(|m| vision::prompt::caps(m, VISION_MAX_GRID)),
            _ => None,
        }
    }

    async fn complete(&self, req: CompletionRequest) -> Result<ByteStream, String> {
        let main = self.ready()?;
        // pedido FIM (com `suffix`): vai para o modelo dedicado, com sessão e KV cache próprios, se ele estiver pronto (D78)
        let loaded = self.route(&main, req.suffix.is_some());
        if req.suffix.is_some() && !loaded.model.tok.has_fim() {
            return Err(format!("{} este modelo não tem tokens de preenchimento (FIM); use um modelo de código como o Qwen2.5-Coder, ou envie o pedido sem `suffix`.", engine::BAD_REQUEST));
        }
        let (tx, rx) = mpsc::channel::<Bytes>(64);
        tokio::task::spawn_blocking(move || {
            with_session(&loaded.session, &tx, |sess| loaded.pool.install(|| complete(&loaded.model, loaded.gpu.as_ref(), sess, &req, &tx)));
            let _ = tx.blocking_send(Bytes::from_static(b"data: [DONE]\n\n"));
        });
        Ok(receiver_stream(rx))
    }
}

/// A "fila" do motor: cada pedido ocupa uma thread bloqueante que espera aqui a sessão (o KV cache é único), então os
/// pedidos rodam um por vez. Quando a vez chega, se o cliente já saiu (o editor cancela a cada tecla e fecha a conexão,
/// o que derruba o receptor do canal), o pedido é DESCARTADO sem tokenizar nem calcular o prefill: devolve `None` e
/// `f` nunca roda. (A cobertura durante o prefill e a geração fica em `run`, que confere o canal entre os lotes.)
fn with_session<S, R>(session: &Mutex<S>, tx: &mpsc::Sender<Bytes>, f: impl FnOnce(&mut S) -> R) -> Option<R> {
    let mut guard = session.lock().unwrap_or_else(|e| e.into_inner());
    if tx.is_closed() {
        return None;
    }
    let sess: &mut S = &mut guard; // o guard não é Send; a referência &mut é
    Some(f(sess))
}

/// As GPUs que a descoberta isolada achou e a memória de cada uma (D69). Vazio sem a feature `gpu`.
struct GpuPlan {
    adapters: Vec<crate::gpu::discovery::AdapterReport>,
    infos: Vec<crate::gpu::GpuInfo>,
}

async fn gpu_plan() -> GpuPlan {
    let d = crate::gpu::discover_isolated().await;
    let legacy = crate::gpu::detect().await;
    GpuPlan { adapters: d.adapters.clone(), infos: crate::gpu::merge(legacy, d) }
}

/// Decide e prepara o backend de GPU: `(Some(gpu), aviso)` ou `(None, aviso)`. Qualquer falha cai na CPU com o motivo no aviso.
/// Os avisos são erros com código (`errcode`, D83): a interface os traduz pelo código.
#[cfg(feature = "gpu")]
fn pick_backend(model: &Model, path: &std::path::Path, ctx: usize, mode: &str, device: &str, plan: Option<&GpuPlan>) -> (Option<Gpu>, String) {
    use crate::gpu::choose::{choose, need_bytes, Choice, CpuReason};
    let Some(plan) = plan else { return (None, String::new()) };
    let weights = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let need = need_bytes(weights, model.cfg.n_layer, model.cfg.n_kv * model.cfg.head_dim, ctx);
    let memory = |a: &crate::gpu::discovery::AdapterReport| plan.infos.iter().find(|i| crate::gpu::same_gpu(&i.name, &a.name)).and_then(|i| i.vram_mb);
    let off = |why: &str| crate::errcode::coded("err_gpu_off", &[why], format!("GPU not used ({}); computing on the CPU.", crate::errcode::english(why)));
    match choose(mode, device, &plan.adapters, &memory, need) {
        Choice::Cpu(CpuReason::DoesNotFit) => (None, crate::errcode::coded("err_gpu_nofit", &[], "The model does not fit in the GPU memory; computing on the CPU.")),
        // no modo automático a ausência de uma GPU adequada é o caso comum e não merece aviso
        Choice::Cpu(r) if mode == "auto" || r == CpuReason::Requested => {
            let _ = r;
            (None, String::new())
        }
        Choice::Cpu(CpuReason::SoftwareOnly) => (None, off(&crate::errcode::coded("err_gpu_sw_only", &[], "only a software renderer was found"))),
        Choice::Cpu(_) => (None, off(&crate::errcode::coded("err_gpu_none", &[], "no GPU was found"))),
        Choice::Gpu { index, name } => {
            // a verificação roda num processo filho: o driver da Intel HD 620 em Vulkan, por exemplo, derruba o processo ao abrir o dispositivo (D64)
            let probe = tokio::runtime::Handle::current().block_on(crate::gpu::probe_isolated(&index.to_string(), std::time::Duration::from_secs(45)));
            if let Err(e) = probe {
                eprintln!("[native] GPU {name}: a verificação falhou: {e}");
                return (None, off(&e));
            }
            let opened = crate::gpu::model::GpuModel::open(model, &crate::gpu::select::AdapterSelector::Index(index), ctx).and_then(|mut g| {
                g.enable_prefill(64.min(ctx))?;
                Ok(g)
            });
            match opened {
                Ok(g) => {
                    eprintln!("[native] calculando na GPU: {name} ({:.0} MiB de pesos)", g.weight_bytes as f64 / 1048576.0);
                    (Some(g), crate::errcode::coded("err_gpu_on", &[&name], format!("Computing on the GPU: {name}.")))
                }
                Err(e) => {
                    eprintln!("[native] GPU {name}: falhou ao carregar o modelo: {e}");
                    (None, off(&e))
                }
            }
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn pick_backend(_model: &Model, _path: &std::path::Path, _ctx: usize, _mode: &str, _device: &str, _plan: Option<&GpuPlan>) -> (Option<Gpu>, String) {
    (None, String::new())
}

impl NativeEngine {
    fn ready(&self) -> Result<Arc<Loaded>, String> {
        match &*self.state.lock().unwrap() {
            State::Ready(l) => Ok(l.clone()),
            _ => Err(crate::errcode::coded("err_engine_not_ready", &[], "The engine is not ready.")),
        }
    }
}

fn receiver_stream(rx: mpsc::Receiver<Bytes>) -> ByteStream {
    Box::pin(stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|b| (Ok::<Bytes, Box<dyn std::error::Error + Send + Sync>>(b), rx))
    }))
}

// ------------------------------------------------------------------ geração

/// Formato dos blocos SSE: chat (`delta.content`) ou texto cru (`text`).
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Chat,
    Text,
}

fn sse(kind: Kind, text: &str, finish: Option<&str>) -> Bytes {
    sse_usage(kind, text, finish, None)
}

/// Como `sse`, com a contagem de tokens `(do prompt, gerados)` no campo extra `usage` (só no bloco final):
/// `openai.rs` a lê e decide se a repassa (`stream_options.include_usage`) ou a soma na resposta única.
fn sse_usage(kind: Kind, text: &str, finish: Option<&str>, usage: Option<(usize, usize)>) -> Bytes {
    Bytes::from(format!("data: {}\n\n", chunk_json(kind, text, finish, usage)))
}

fn chunk_json(kind: Kind, text: &str, finish: Option<&str>, usage: Option<(usize, usize)>) -> serde_json::Value {
    let mut chunk = match kind {
        Kind::Chat => json!({ "object": "chat.completion.chunk",
            "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": finish }] }),
        Kind::Text => json!({ "object": "text_completion",
            "choices": [{ "index": 0, "text": text, "finish_reason": finish }] }),
    };
    if let Some((p, c)) = usage {
        chunk["usage"] = json!({ "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c });
    }
    chunk
}

/// A message the user must read that travels inside the chat stream (D83): the text delta is the English message in brackets, for
/// any client of the API, and the chunk also carries `buriti_error` ({code, args, en}), which the web UI translates into the language
/// the user chose (the UI shows the translation instead of the delta). Same as `sse_usage` otherwise.
fn sse_error(kind: Kind, code: &str, args: &[&str], en: &str, finish: Option<&str>, usage: Option<(usize, usize)>) -> Bytes {
    let mut chunk = chunk_json(kind, &format!("[{en}]"), finish, usage);
    chunk["buriti_error"] = serde_json::from_str(&crate::errcode::coded(code, args, en)).unwrap_or(serde_json::Value::Null);
    Bytes::from(format!("data: {chunk}\n\n"))
}

/// Tokens do prompt processados por passada no prefill (padrão 32; `BURITI_PREFILL_BATCH` só para experimentos de
/// desempenho: o resultado numérico não muda com o tamanho do lote, os kernels de 1, 4 e 8 ativações são idênticos).
fn prefill_batch() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("BURITI_PREFILL_BATCH").ok().and_then(|s| s.parse::<usize>().ok()).filter(|&n| (1..=1024).contains(&n)).unwrap_or(32))
}

/// Amostragem, na ordem padrão do llama.cpp: penalidades (repetição, frequência, presença) → top-k → top-p →
/// min-p → temperatura → sorteio. A temperatura vem por ÚLTIMO: ela redistribui as probabilidades só entre os
/// candidatos que já passaram nos cortes (aplicar antes alargaria ou estreitaria o top-p).
#[derive(Clone, Copy, Debug)]
struct Sampling {
    temperature: f32,
    /// 0 = sem corte por top-k
    top_k: usize,
    /// 1.0 = sem corte por top-p
    top_p: f32,
    /// 0.0 = sem corte por min-p
    min_p: f32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    seed: Option<u64>,
}

impl Sampling {
    /// Padrões do motor com os parâmetros que o pedido trouxe por cima.
    fn with(mut self, o: &SamplingOpts) -> Self {
        self.top_p = o.top_p.unwrap_or(self.top_p);
        self.top_k = o.top_k.map(|k| k as usize).unwrap_or(self.top_k);
        self.min_p = o.min_p.unwrap_or(self.min_p);
        self.repeat_penalty = o.repeat_penalty.unwrap_or(self.repeat_penalty);
        self.frequency_penalty = o.frequency_penalty.unwrap_or(self.frequency_penalty);
        self.presence_penalty = o.presence_penalty.unwrap_or(self.presence_penalty);
        self.seed = o.seed.or(self.seed);
        self
    }
}

/// Maior grade de fatias por lado aceita (3×3 + visão geral = 10 blocos de 64 vetores; o contexto decide o resto).
const VISION_MAX_GRID: usize = 3;

/// Procura `<nome do modelo sem .gguf>.mmproj.gguf` ao lado do modelo (é como o catálogo grava o par). Falha de leitura ou
/// projeção que não combina com o modelo: segue só com texto (aviso no console), nunca derruba o carregamento.
fn load_vision(model_path: &std::path::Path, model: &Model) -> Option<vision::mmproj::Mmproj> {
    model.tok.image_slot_id()?;
    let stem = model_path.file_stem()?.to_string_lossy().into_owned();
    let p = model_path.with_file_name(format!("{stem}.mmproj.gguf"));
    if !p.is_file() {
        return None;
    }
    match vision::mmproj::Mmproj::open(&p) {
        Ok(m) if m.hp.proj_dim == model.cfg.n_embd => Some(m),
        Ok(_) => {
            eprintln!("[native] mmproj ignorado: a projeção não combina com o modelo ({})", p.display());
            None
        }
        Err(e) => {
            eprintln!("[native] mmproj ignorado ({}): {e}", p.display());
            None
        }
    }
}

fn generate(model: &Model, mm: Option<&vision::mmproj::Mmproj>, gpu: Option<&Gpu>, sess: &mut Session, req: &ChatRequest, tx: &mpsc::Sender<Bytes>, constraint: Option<crate::grammar::Constraint>) {
    let ctx = sess.ctx();
    let max_new = (req.max_tokens.unwrap_or(1024) as usize).clamp(1, ctx / 2);
    let sampling = Sampling { temperature: req.temperature.unwrap_or(0.7).clamp(0.0, 2.0), top_k: 20, top_p: 0.8, min_p: 0.0, repeat_penalty: 1.05, frequency_penalty: 0.0, presence_penalty: 0.0, seed: None }
        .with(&req.sampling);

    // com imagem: sem descartar mensagens antigas (o índice da imagem e o cache dependem delas)
    if !req.images.is_empty() {
        let Some(mm) = mm else {
            let _ = tx.blocking_send(sse_error(Kind::Chat, "err_img_novision", &[], "The active model does not accept images.", Some("stop"), Some((0, 0))));
            return;
        };
        match vision::prompt::build_prompt(&model.tok, mm.hp.n_tokens(), &req.messages, &req.images) {
            Ok((prompt, ip)) if prompt.len() + max_new < ctx => {
                run(model, sess, prompt, max_new, sampling, &Stops { texts: &req.stop, ids: &[], suffix: None }, Kind::Chat, tx, Some((&ip, mm)), gpu, constraint);
            }
            Ok((prompt, _)) => {
                let _ = tx.blocking_send(sse_error(Kind::Chat, "err_img_ctx_over", &[], "The message with the image is too large for the configured context.", Some("length"), Some((prompt.len(), 0))));
            }
            Err(e) => {
                let _ = tx.blocking_send(sse_error(Kind::Chat, "err_img_process", &[&e], &format!("Failed to process the image: {e}"), Some("stop"), Some((0, 0))));
            }
        }
        return;
    }
    // prompt: se não couber no contexto, descarta as mensagens mais antigas (mantém a última)
    let mut msgs: Vec<ChatMsg> = req.messages.clone();
    let mut prompt = model.tok.chat_prompt(&msgs);
    while prompt.len() + max_new > ctx && msgs.len() > 1 {
        let drop_at = if msgs[0].role == "system" && msgs.len() > 2 { 1 } else { 0 };
        msgs.remove(drop_at);
        prompt = model.tok.chat_prompt(&msgs);
    }
    if prompt.len() + 1 >= ctx {
        let _ = tx.blocking_send(sse_error(Kind::Chat, "err_ctx_short", &[], "The message is too large for the configured context.", Some("length"), Some((prompt.len(), 0))));
        return;
    }
    run(model, sess, prompt, max_new, sampling, &Stops { texts: &req.stop, ids: &[], suffix: None }, Kind::Chat, tx, None, gpu, constraint);
}

/// O que encerra a geração além do fim de texto do modelo: textos do pedido, ids de tokens extras (fim de
/// arquivo/lacuna dos modelos de código) e o sufixo do FIM (o modelo voltou a copiar o que já existe).
struct Stops<'a> {
    texts: &'a [String],
    ids: &'a [u32],
    suffix: Option<&'a str>,
}

/// Completa texto cru (`/v1/completions`), com preenchimento no meio se houver `suffix`. Amostragem
/// própria para código: temperatura baixa e SEM penalidade de repetição (identificadores se repetem).
fn complete(model: &Model, gpu: Option<&Gpu>, sess: &mut Session, req: &CompletionRequest, tx: &mpsc::Sender<Bytes>) {
    let ctx = sess.ctx();
    let max_new = (req.max_tokens.unwrap_or(256) as usize).clamp(1, ctx / 2);
    let sampling = Sampling { temperature: req.temperature.unwrap_or(0.2).clamp(0.0, 2.0), top_k: 40, top_p: 0.95, min_p: 0.0, repeat_penalty: 1.0, frequency_penalty: 0.0, presence_penalty: 0.0, seed: None }
        .with(&req.sampling);
    // orçamento: o do pedido (ou o padrão do FIM, já resolvido em `openai.rs`), nunca acima do que cabe no contexto
    let ctx_budget = ctx.saturating_sub(max_new);
    let budget = req.max_prompt_tokens.map_or(ctx_budget, |n| n.min(ctx_budget));
    match completion_prompt(&model.tok, &req.prompt, req.suffix.as_deref(), budget) {
        Ok(prompt) => {
            let ids = model.tok.code_stop_ids();
            run(model, sess, prompt, max_new, sampling, &Stops { texts: &req.stop, ids: &ids, suffix: req.suffix.as_deref() }, Kind::Text, tx, None, gpu, None)
        }
        Err(()) => {
            let _ = tx.blocking_send(sse_usage(Kind::Text, "", Some("length"), Some((0, 0))));
        }
    }
}

/// Monta o prompt de completar texto; se passar do orçamento de tokens, corta o INÍCIO do texto antes do
/// cursor e o FIM do texto depois do cursor (o que está perto do cursor é o que mais importa), em linhas inteiras e
/// com cortes estáveis entre teclas (política e testes em `engine::trim_fim_prompt`, compartilhada com o llama.cpp).
fn completion_prompt(tok: &tokenizer::Tokenizer, prefix: &str, suffix: Option<&str>, budget: usize) -> Result<Vec<u32>, ()> {
    let build = |pre: &str, suf: &str| match suffix {
        Some(_) => tok.fim_prompt(pre, suf),
        None => Some(tok.raw_prompt(pre)),
    };
    let (pre, suf) = engine::trim_fim_prompt(prefix, suffix.unwrap_or(""), budget, |p, s| build(p, s).map(|ids| ids.len())).ok_or(())?;
    build(pre, suf).ok_or(())
}

/// Quantos tokens do início de `prompt` já têm o KV calculado na sessão (D76); a sessão fica truncada nesse ponto, e o prefill
/// recomeça nele. Vale para a CPU e para a GPU (o KV da GPU é posicional: reaproveitar = começar na posição `n`).
/// - O cache só vale para quem o escreveu: se o processador mudou (CPU/GPU), ele é descartado. (Modelo e contexto não precisam de
///   conferência: a sessão nasce com o modelo carregado e com o contexto dele, e uma nova carga cria uma sessão nova.)
/// - Com imagem (`image_limit` = chave da imagem e primeira posição de imagem) os ids dos vetores são todos iguais: o KV das
///   posições de imagem só vale se for a MESMA imagem.
/// - Ao menos o último token é recalculado, para obter os logits.
fn reusable_prefix(sess: &mut Session, prompt: &[u32], gpu: bool, image_limit: Option<(u64, usize)>) -> usize {
    let owner = if gpu { 2 } else { 1 };
    if sess.cache_owner != owner || sess.tokens.len() > sess.ctx() {
        sess.tokens.clear();
        sess.img_key = 0;
        sess.cache_owner = owner;
    }
    let mut common = sess.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    if let Some((key, first_slot)) = image_limit {
        if sess.img_key != key {
            common = common.min(first_slot);
        }
    }
    #[cfg(test)]
    {
        common += PLANT_EXTRA_REUSE.with(|c| c.get()); // bug plantado pelos testes: reaproveita tokens demais
    }
    let common = common.min(prompt.len().saturating_sub(1)); // ao menos o último token é recalculado
    sess.tokens.truncate(common);
    common
}

#[cfg(test)]
thread_local! {
    /// Só para o teste com dentes (D78): manda o pedido FIM para a sessão do chat.
    static PLANT_FIM_TO_CHAT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// Só para o teste com dentes: soma tokens ao prefixo reaproveitado (por thread, não contamina os outros testes).
    static PLANT_EXTRA_REUSE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[allow(clippy::too_many_arguments)]
fn run(model: &Model, sess: &mut Session, prompt: Vec<u32>, max_new: usize, sampling: Sampling, stops: &Stops, kind: Kind, tx: &mpsc::Sender<Bytes>, images: Option<(&vision::prompt::ImagePrompt, &vision::mmproj::Mmproj)>, gpu: Option<&Gpu>, mut constraint: Option<crate::grammar::Constraint>) {
    let _ = &gpu; // sem a feature `gpu` o parâmetro não é usado
    let ctx = sess.ctx();
    if prompt.is_empty() {
        let _ = tx.blocking_send(sse_usage(kind, "", Some("stop"), Some((0, 0))));
        return;
    }
    // o cliente já saiu enquanto o pedido esperava a vez (o editor cancela a cada tecla): não gasta o prefill
    if tx.is_closed() {
        return;
    }

    // reaproveita o KV cache do prefixo comum com o pedido anterior (D76): só os tokens depois dele são processados
    let image_limit = images.map(|(ip, _)| (ip.key, ip.slots.first().copied().unwrap_or(0)));
    let common = reusable_prefix(sess, &prompt, gpu.is_some(), image_limit);
    // vetores de imagem só são calculados (a torre custa segundos por bloco) se alguma posição de imagem for recalculada
    let mut embd: Vec<f32> = Vec::new();
    if let Some((ip, mm)) = images {
        if ip.slots.last().is_some_and(|&p| p >= common) {
            sess.img_key = 0; // o KV das posições de imagem vai ser reescrito
            match ip.embeddings(mm) {
                Ok(e) => embd = e,
                Err(e) => {
                    let _ = tx.blocking_send(sse_error(kind, "err_img_process", &[&e], &format!("Failed to process the image: {e}"), Some("stop"), Some((0, 0))));
                    return;
                }
            }
        }
    }

    let t0 = Instant::now();
    // prefill em lotes: cada linha de pesos é lida uma vez por lote, não uma vez por token
    let mut i = common;
    while i < prompt.len() {
        if tx.is_closed() {
            return; // cliente saiu no meio do prefill (pode levar dezenas de segundos a CPU): o cache fica consistente até aqui
        }
        // GPU (D69): o lote vai para a placa (as GEMMs processam o lote inteiro de uma vez); só o último lote calcula os logits
        #[cfg(feature = "gpu")]
        if let Some(g) = gpu {
            let end = (i + g.prefill_capacity()).min(prompt.len());
            match g.prefill_part(&prompt[i..end], i, end == prompt.len()) {
                Ok(l) => {
                    if let Some(l) = l {
                        sess.logits.copy_from_slice(&l);
                    }
                }
                Err(e) => {
                    sess.tokens.clear(); // o KV da GPU pode estar pela metade: nada dele é reaproveitado
                    let _ = tx.blocking_send(sse_error(kind, "err_gpu_fail", &[&e], &format!("GPU failure: {e}"), Some("stop"), Some((0, 0))));
                    return;
                }
            }
            sess.tokens.extend_from_slice(&prompt[i..end]);
            i = end;
            continue;
        }
        let end = (i + prefill_batch()).min(prompt.len());
        let ev: Vec<(usize, &[f32])> = match images {
            Some((ip, mm)) if !embd.is_empty() => {
                let d = mm.hp.proj_dim;
                ip.slots.iter().enumerate().filter(|(_, &p)| p >= i && p < end).map(|(k, &p)| (p - i, &embd[k * d..(k + 1) * d])).collect()
            }
            _ => Vec::new(),
        };
        model.forward_batch_embd(sess, &prompt[i..end], &ev, i, end == prompt.len());
        sess.tokens.extend_from_slice(&prompt[i..end]);
        i = end;
    }
    if let Some((ip, _)) = images {
        sess.img_key = ip.key;
    }
    let prefill = t0.elapsed().as_secs_f32();
    if std::env::var_os("BURITI_DIAG").is_some() {
        let mv: f64 = model::MV_NS.iter().map(|p| p.load(Ordering::Relaxed) as f64 / 1e9).sum();
        let ph: Vec<f64> = model::PH_NS.iter().map(|p| p.swap(0, Ordering::Relaxed) as f64 / 1e9).collect();
        eprintln!("[native] diag prefill {prefill:.2}s: multiplicações {mv:.2}s (dentro: quantizar ativação {:.2}s, transpor {:.2}s) | atenção {:.2}s | silu {:.2}s | resto {:.2}s", ph[4], ph[5], ph[1], ph[2], prefill as f64 - mv - ph[1] - ph[2]);
    }
    let n_prefill = prompt.len() - common;

    let mut rng = match sampling.seed {
        Some(s) => Rng::from_seed(s),
        None => Rng::new(),
    };
    let mut utf8 = Utf8Stream::default();
    let mut started = false; // já saiu o primeiro texto da resposta?
    let t1 = Instant::now();
    let mut generated = 0usize;
    let mut finish = "length";
    // textos de parada: segura o final do texto que ainda pode virar o começo de um deles
    let mut stopper = stopper::Stopper::new(stops.texts, stops.suffix);

    while generated < max_new && sess.tokens.len() < ctx {
        if tx.is_closed() {
            finish = "cancelled"; // o cliente (editor) fechou a conexão: não gasta mais CPU
            break;
        }
        let recent_from = sess.tokens.len().saturating_sub(64);
        // gramática (D82): os tokens que a quebrariam ficam com logit -infinito antes da amostragem
        if let Some(c) = constraint.as_mut() {
            if !c.mask(&mut sess.logits) {
                finish = "stop";
                break;
            }
        }
        let ts = Instant::now();
        let next = sample(&mut sess.logits, &sess.tokens[recent_from..], sampling, &mut rng);
        model::PH_NS[3].fetch_add(ts.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if model.tok.eos.contains(&next) || stops.ids.contains(&next) {
            finish = "stop";
            break;
        }
        generated += 1;
        if let Some(c) = constraint.as_mut() {
            c.accept(&model.tok.token_bytes(next));
        }
        let mut text = utf8.push(&model.tok.token_bytes(next));
        // SentencePiece com prefixo de espaço (Mistral...): o 1º token da resposta é "▁Palavra", que vira
        // " Palavra"; o espaço inicial é artefato do tokenizador e não deve aparecer no chat
        if !started && !text.is_empty() {
            started = true;
            if kind == Kind::Chat && model.tok.strips_first_space() {
                if let Some(rest) = text.strip_prefix(' ') {
                    text = rest.to_string();
                }
            }
        }
        // com textos de parada, só sai o que já não pode ser parte de um deles
        let (out, hit) = stopper.push(&text);
        if !out.is_empty() && tx.blocking_send(sse(kind, &out, None)).is_err() {
            // cliente desconectou: para de gerar (o estado do cache segue consistente)
            finish = "cancelled";
            break;
        }
        if hit {
            finish = "stop";
            break;
        }
        let pos = sess.tokens.len();
        #[cfg(feature = "gpu")]
        if let Some(g) = gpu {
            match g.forward(next, pos) {
                Ok(l) => sess.logits.copy_from_slice(&l),
                Err(e) => {
                    sess.tokens.clear();
                    let _ = tx.blocking_send(sse_error(kind, "err_gpu_fail", &[&e], &format!("GPU failure: {e}"), None, None));
                    finish = "stop";
                    break;
                }
            }
            sess.tokens.push(next);
            continue;
        }
        model.forward(sess, next, pos, true);
        sess.tokens.push(next);
    }
    let gen = t1.elapsed().as_secs_f32();
    // Diagnóstico por tipo de quantização (só com BURITI_DIAG=1); sempre zera os contadores.
    let diag = std::env::var_os("BURITI_DIAG").is_some();
    for (i, name) in gguf::Dtype::NAMES.iter().enumerate() {
        let ns = model::MV_NS[i].swap(0, Ordering::Relaxed);
        let by = model::MV_BYTES[i].swap(0, Ordering::Relaxed);
        if diag && ns > 0 {
            eprintln!("[native] diag: {name}: {:.2}s, {:.2} GB de pesos, {:.2} GB/s", ns as f64 / 1e9, by as f64 / 1e9, by as f64 / ns as f64);
        }
    }
    let ph: Vec<f64> = model::PH_NS.iter().map(|p| p.swap(0, Ordering::Relaxed) as f64 / 1e9).collect();
    if diag {
        eprintln!("[native] diag: fora das multiplicações: quantizar ativação {:.2}s, atenção {:.2}s, silu {:.2}s, amostragem {:.2}s", ph[0], ph[1], ph[2], ph[3]);
    }
    if finish != "cancelled" {
        // sobra do texto segurado: não era parada (se parou por um padrão, o `Stopper` já a descartou)
        let rest = stopper.flush();
        if !rest.is_empty() {
            let _ = tx.blocking_send(sse(kind, &rest, None));
        }
        // prompt_tokens conta o prompt inteiro, inclusive o prefixo reaproveitado do KV cache (como a OpenAI)
        let _ = tx.blocking_send(sse_usage(kind, "", Some(finish), Some((prompt.len(), generated))));
    }
    eprintln!(
        "[native] prompt: {n_prefill} tokens novos ({} reaproveitados) em {prefill:.1}s | geração: {generated} tokens em {gen:.1}s ({:.2} tok/s)",
        common,
        generated as f32 / gen.max(0.001)
    );
}

// ---------------------------------------------------------------- amostragem

struct Rng(u64);

impl Rng {
    fn new() -> Self {
        let mut b = [0u8; 8];
        let _ = getrandom::fill(&mut b);
        Rng(u64::from_le_bytes(b) | 1)
    }
    /// Semente fixa: a mesma semente, prompt e parâmetros dão o mesmo texto (splitmix64 evita estado zero ou fraco).
    fn from_seed(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng((z ^ (z >> 31)) | 1)
    }
    /// xorshift64*
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (r >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Penalidades sobre os últimos tokens, como o llama.cpp: a de repetição vale UMA vez por token distinto (não por
/// ocorrência); frequência e presença usam a contagem de ocorrências na janela.
fn apply_penalties(logits: &mut [f32], recent: &[u32], p: &Sampling) {
    if recent.is_empty() || (p.repeat_penalty == 1.0 && p.frequency_penalty == 0.0 && p.presence_penalty == 0.0) {
        return;
    }
    let mut counts: Vec<(u32, u32)> = Vec::with_capacity(recent.len().min(64));
    for &t in recent {
        match counts.iter_mut().find(|(x, _)| *x == t) {
            Some(c) => c.1 += 1,
            None => counts.push((t, 1)),
        }
    }
    for (t, c) in counts {
        let Some(l) = logits.get_mut(t as usize) else { continue };
        if p.repeat_penalty != 1.0 {
            *l = if *l <= 0.0 { *l * p.repeat_penalty } else { *l / p.repeat_penalty };
        }
        *l -= c as f32 * p.frequency_penalty + p.presence_penalty;
    }
}

fn sample(logits: &mut [f32], recent: &[u32], p: Sampling, rng: &mut Rng) -> u32 {
    apply_penalties(logits, recent, &p);
    if p.temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    }
    let n = logits.len();
    let k = if p.top_k == 0 { n } else { p.top_k.clamp(1, n) };
    let mut idx: Vec<u32> = (0..n as u32).collect();
    if k < n {
        idx.select_nth_unstable_by(k - 1, |a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
        idx.truncate(k);
    }
    idx.sort_unstable_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));

    // probabilidades com temperatura 1: é nelas que top-p e min-p cortam
    let max = logits[idx[0] as usize];
    let mut probs: Vec<f32> = idx.iter().map(|&i| (logits[i as usize] - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x /= sum);

    // top-p: mantém o menor conjunto cuja probabilidade acumulada alcança p
    let mut keep = probs.len();
    if p.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, &pr) in probs.iter().enumerate() {
            cum += pr;
            if cum >= p.top_p {
                keep = i + 1;
                break;
            }
        }
    }
    // min-p: descarta quem tem menos que min_p da probabilidade do melhor candidato (sempre sobra ao menos um)
    if p.min_p > 0.0 {
        let thr = p.min_p * probs[0];
        keep = keep.min(probs[..keep].iter().take_while(|&&x| x >= thr).count().max(1));
    }

    // temperatura por último, só entre os candidatos que sobraram
    let weights: Vec<f32> = idx[..keep].iter().map(|&i| ((logits[i as usize] - max) / p.temperature).exp()).collect();
    let total: f32 = weights.iter().sum();
    let mut r = rng.next_f32() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return idx[i];
        }
    }
    idx[keep - 1]
}

#[cfg(test)]
mod sampling_tests {
    use super::*;

    fn base() -> Sampling {
        Sampling { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, repeat_penalty: 1.0, frequency_penalty: 0.0, presence_penalty: 0.0, seed: None }
    }
    fn greedy() -> Sampling {
        Sampling { temperature: 0.0, ..base() }
    }
    fn draws(logits: &[f32], p: Sampling, n: usize, seed: u64) -> Vec<u32> {
        let mut rng = Rng::from_seed(seed);
        (0..n).map(|_| sample(&mut logits.to_vec(), &[], p, &mut rng)).collect()
    }

    #[test]
    fn repeat_penalty_applies_once_per_distinct_token() {
        // o token 0 aparece 3 vezes; por token distinto: 3.0 / 1.2 = 2.5 (> 2.0, ele continua o melhor).
        // Se a penalidade fosse por ocorrência: 3.0 / 1.2^3 = 1.74 e o token 1 (2.0) ganharia.
        let mut l = vec![3.0, 2.0, 0.0];
        let t = sample(&mut l, &[0, 0, 0], Sampling { repeat_penalty: 1.2, ..greedy() }, &mut Rng::from_seed(1));
        assert_eq!(t, 0);
        assert!((l[0] - 2.5).abs() < 1e-6, "{}", l[0]);
    }

    #[test]
    fn presence_and_frequency_penalties() {
        let run = |p: Sampling| sample(&mut vec![1.0, 0.9], &[0, 0], p, &mut Rng::from_seed(1));
        assert_eq!(run(greedy()), 0);
        assert_eq!(run(Sampling { presence_penalty: 0.5, ..greedy() }), 1); // 1.0 - 0.5 = 0.5 < 0.9
        assert_eq!(run(Sampling { frequency_penalty: 0.3, ..greedy() }), 1); // 1.0 - 2*0.3 = 0.4 < 0.9
        assert_eq!(run(Sampling { frequency_penalty: 0.04, ..greedy() }), 0); // 1.0 - 2*0.04 = 0.92 > 0.9
    }

    #[test]
    fn min_p_drops_unlikely_tokens() {
        let l = [5.0, 0.0, 0.0, 0.0];
        assert!(draws(&l, base(), 3000, 7).iter().any(|&t| t != 0), "sem min-p o controle tem de sortear outros tokens");
        assert!(draws(&l, Sampling { min_p: 0.5, ..base() }, 3000, 7).iter().all(|&t| t == 0));
    }

    #[test]
    fn temperature_comes_after_top_p() {
        // com T=1: p = 0.574 / 0.348 / 0.078 / 0.0005; top-p 0.6 mantém os DOIS primeiros.
        // Se a temperatura 0.3 viesse antes, o primeiro passaria de 0.6 sozinho e o token 1 nunca sairia.
        let l = [2.0, 1.5, 0.0, -5.0];
        let d = draws(&l, Sampling { top_p: 0.6, temperature: 0.3, ..base() }, 3000, 11);
        let n1 = d.iter().filter(|&&t| t == 1).count();
        assert!(n1 > 200 && n1 < 900, "token 1 saiu {n1} vezes (esperado ~16%)");
        assert!(d.iter().all(|&t| t <= 1), "tokens fora do top-p não podem sair");
    }

    #[test]
    fn top_k_limits_candidates() {
        let l = [1.0, 1.1, 0.9, 1.05];
        assert!(draws(&l, Sampling { top_k: 1, ..base() }, 500, 3).iter().all(|&t| t == 1));
        let d = draws(&l, Sampling { top_k: 2, ..base() }, 2000, 3);
        assert!(d.iter().all(|&t| t == 1 || t == 3) && d.contains(&1) && d.contains(&3));
    }

    #[test]
    fn seed_makes_sampling_reproducible() {
        let l: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let a = draws(&l, base(), 60, 42);
        assert_eq!(a, draws(&l, base(), 60, 42));
        assert_ne!(a, draws(&l, base(), 60, 43));
    }

    #[test]
    fn request_options_override_defaults() {
        let o = SamplingOpts { top_p: Some(0.5), top_k: Some(7), min_p: Some(0.1), repeat_penalty: Some(1.3), frequency_penalty: Some(0.2), presence_penalty: Some(0.1), seed: Some(9) };
        let s = base().with(&o);
        assert_eq!((s.top_p, s.top_k, s.min_p, s.repeat_penalty, s.seed), (0.5, 7, 0.1, 1.3, Some(9)));
        assert_eq!((s.frequency_penalty, s.presence_penalty), (0.2, 0.1));
        let d = base().with(&SamplingOpts::default());
        assert_eq!((d.top_p, d.top_k, d.seed), (1.0, 0, None));
    }

    #[test]
    fn request_whose_client_left_while_waiting_never_runs() {
        let session = Mutex::new(Vec::<u32>::new());
        // cliente ainda conectado: roda
        let (tx, _rx) = mpsc::channel::<Bytes>(4);
        assert_eq!(with_session(&session, &tx, |s| { s.push(1); s.len() }), Some(1));
        // o cliente saiu (receptor derrubado) enquanto esperava a vez: `f` não pode rodar nem tocar na sessão
        let (tx, rx) = mpsc::channel::<Bytes>(4);
        drop(rx);
        let mut ran = false;
        assert_eq!(with_session(&session, &tx, |s| { ran = true; s.push(2); }), None);
        assert!(!ran);
        assert_eq!(*session.lock().unwrap(), vec![1]);
    }

    #[test]
    fn waiting_request_is_dropped_only_after_the_lock_is_won() {
        // simula a fila: a thread A segura a sessão; B espera; o cliente de B sai enquanto espera; ao liberar, B não roda
        let session = Arc::new(Mutex::new(0u32));
        let guard = session.lock().unwrap();
        let (tx, rx) = mpsc::channel::<Bytes>(4);
        let s2 = session.clone();
        let t = std::thread::spawn(move || with_session(&*s2, &tx, |s| { *s += 1; }));
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(rx);
        drop(guard);
        assert_eq!(t.join().unwrap(), None);
        assert_eq!(*session.lock().unwrap(), 0);
    }

    #[test]
    fn final_chunk_carries_usage_only_when_given() {
        let parse = |b: Bytes| serde_json::from_str::<serde_json::Value>(std::str::from_utf8(&b).unwrap().strip_prefix("data: ").unwrap().trim()).unwrap();
        let v = parse(sse_usage(Kind::Chat, "", Some("stop"), Some((12, 5))));
        assert_eq!(v["usage"], json!({ "prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17 }));
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert!(parse(sse(Kind::Text, "x", None)).get("usage").is_none());
    }
}
