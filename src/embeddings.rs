//! `POST /v1/embeddings` do motor próprio (D77): vetores de texto a partir dos estados finais de um modelo decoder
//! (Qwen2/3, Llama 3.x, Gemma 3, Phi-3...), com pooling (média, primeiro ou último token, conforme `<arq>.pooling_type`
//! do GGUF) e normalização L2. NÃO há arquitetura de embedding de verdade (BERT, nomic-bert): o motor não a roda e
//! o carregador recusa o arquivo. Um modelo de chat usado como embedder dá vetores utilizáveis, mas de qualidade menor
//! que a de um modelo treinado para isso (a resposta avisa em `buriti.quality`).
//!
//! Política de memória (função pura `plan`): com RAM de sobra, um modelo de embeddings dedicado fica carregado ao
//! lado do modelo do chat (`dedicated`); com pouca, é carregado só durante o pedido e descartado (`on_demand`); sem
//! espaço para nenhum dos dois, usa o modelo do chat (`shared`, sinalizado). O pedido NUNCA troca de modelo em silêncio:
//! quem nomeia um modelo que não cabe recebe erro (misturar vetores de espaços diferentes num mesmo índice os inutiliza).

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use crate::{
    engine::BAD_REQUEST,
    native::model::{Model, Session},
    server::{active, Shared},
};

// ------------------------------------------------------------------ limites

/// Textos por pedido (a OpenAI aceita 2048; na CPU isso seria minutos de trabalho numa chamada só).
pub const MAX_INPUTS: usize = 256;
/// Tokens que cada lote do forward processa (o resultado numérico não depende deste valor: kernels idênticos).
const CHUNK: usize = 32;
/// Tempo parado até o modelo dedicado residente ser descartado.
const RESIDENT_IDLE: Duration = Duration::from_secs(600);

// ------------------------------------------------------------------ pooling

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// média dos estados de todos os tokens
    Mean,
    /// estado do primeiro token (CLS)
    Cls,
    /// estado do último token (modelos de embedding decoder, como o Qwen3-Embedding)
    Last,
}

impl Pooling {
    pub fn name(self) -> &'static str {
        match self {
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::Last => "last",
        }
    }

    /// `pooling_type` do GGUF (enum do llama.cpp: 0 nenhum, 1 média, 2 CLS, 3 último, 4 rank). Devolve o pooling e se ele
    /// veio do arquivo. Sem o campo (ou 0), um modelo de chat usa a MÉDIA (padrão do Buriti, `from_file = false`).
    pub fn from_gguf(v: Option<u64>) -> Result<(Pooling, bool), String> {
        match v {
            None | Some(0) => Ok((Pooling::Mean, false)),
            Some(1) => Ok((Pooling::Mean, true)),
            Some(2) => Ok((Pooling::Cls, true)),
            Some(3) => Ok((Pooling::Last, true)),
            Some(4) => Err(format!("{BAD_REQUEST} este modelo é de reclassificação (pooling `rank`), não de embeddings; o motor próprio não o suporta.")),
            Some(n) => Err(format!("{BAD_REQUEST} `pooling_type` {n} do modelo não é suportado para embeddings.")),
        }
    }
}

/// Acumula os estados finais (n × dim) em lotes, na ordem dos tokens. A soma da média é em f64 (a ordem é fixa, então o
/// resultado é determinístico e não depende de como os tokens foram divididos em lotes).
pub struct Pooler {
    kind: Pooling,
    dim: usize,
    n: usize,
    sum: Vec<f64>,
    first: Vec<f32>,
    last: Vec<f32>,
}

impl Pooler {
    pub fn new(kind: Pooling, dim: usize) -> Self {
        Pooler { kind, dim, n: 0, sum: if kind == Pooling::Mean { vec![0.0; dim] } else { Vec::new() }, first: Vec::new(), last: Vec::new() }
    }

    pub fn push(&mut self, rows: &[f32]) {
        assert!(self.dim > 0 && rows.len() % self.dim == 0, "linhas incompletas");
        for row in rows.chunks_exact(self.dim) {
            match self.kind {
                Pooling::Mean => self.sum.iter_mut().zip(row).for_each(|(s, x)| *s += *x as f64),
                Pooling::Cls => {
                    if self.n == 0 {
                        self.first = row.to_vec();
                    }
                }
                Pooling::Last => {
                    self.last.clear();
                    self.last.extend_from_slice(row);
                }
            }
            self.n += 1;
        }
    }

    /// `None` se nenhum token foi acumulado.
    pub fn finish(self) -> Option<Vec<f32>> {
        if self.n == 0 {
            return None;
        }
        Some(match self.kind {
            Pooling::Mean => self.sum.iter().map(|s| (*s / self.n as f64) as f32).collect(),
            Pooling::Cls => self.first,
            Pooling::Last => self.last,
        })
    }
}

/// Normaliza para norma L2 = 1 (conta em f64). Vetor nulo fica como está; valor não finito é erro.
pub fn l2_normalize(v: &mut [f32]) -> Result<(), String> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err("o modelo produziu valores não finitos (estouro numérico); tente outro modelo ou uma quantização maior.".into());
    }
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x = (*x as f64 / norm) as f32);
    }
    Ok(())
}

/// Base64 padrão (RFC 4648, com `=`) dos floats em little-endian: é o que o cliente Python da OpenAI pede por padrão.
pub fn base64_f32(v: &[f32]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

// ------------------------------------------------------------------ pedido

/// Trabalho entregue ao motor (já validado no formato; os limites de tokens são checados com o vocabulário do modelo).
#[derive(Clone, Debug)]
pub struct EmbedJob {
    pub inputs: Vec<String>,
    pub normalize: bool,
    /// `dimensions` do pedido: só é aceito se igual à dimensão do modelo (não há truncamento Matryoshka)
    pub dimensions: Option<usize>,
    /// máximo de tokens por texto (da política de memória; o motor ainda limita pelo contexto de treino e pelo KV)
    pub max_tokens: usize,
    /// bytes de KV cache que uma sessão de embedding pode ocupar (da política de memória)
    pub kv_budget_bytes: u64,
}

#[derive(Debug)]
pub struct EmbedOutput {
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: usize,
    pub pooling: Pooling,
    pub pooling_from_file: bool,
    pub normalized: bool,
}

/// Pedido da API (antes de escolher o modelo).
#[derive(Debug)]
pub struct EmbedRequest {
    pub inputs: Vec<String>,
    pub model: Option<String>,
    pub base64: bool,
    pub dimensions: Option<usize>,
    pub normalize: bool,
}

/// Lê e valida o corpo de `/v1/embeddings`. Recusa com mensagem clara o que não é suportado (ids de tokens, `dimensions`
/// inválido, formato desconhecido), em vez de ignorar.
pub fn parse_request(v: &Value) -> Result<EmbedRequest, String> {
    let inputs: Vec<String> = match v.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if a.is_empty() => return Err("`input` vazio: envie um texto ou uma lista de textos.".into()),
        Some(Value::Array(a)) => {
            let mut out = Vec::with_capacity(a.len());
            for (i, x) in a.iter().enumerate() {
                match x {
                    Value::String(s) => out.push(s.clone()),
                    Value::Number(_) | Value::Array(_) => {
                        return Err(format!("`input[{i}]`: listas de tokens (ids) não são suportadas, envie texto. (No LangChain use `check_embedding_ctx_length=False`; no cliente da OpenAI, strings.)"))
                    }
                    _ => return Err(format!("`input[{i}]` deve ser um texto.")),
                }
            }
            out
        }
        Some(_) => return Err("`input` deve ser um texto ou uma lista de textos.".into()),
        None => return Err("`input` ausente.".into()),
    };
    if inputs.len() > MAX_INPUTS {
        return Err(format!("`input` tem {} textos; o máximo por pedido é {MAX_INPUTS} (divida em pedidos menores).", inputs.len()));
    }
    if let Some(i) = inputs.iter().position(|s| s.is_empty()) {
        return Err(format!("`input[{i}]` está vazio: textos vazios não têm embedding."));
    }
    let base64 = match v.get("encoding_format") {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) if s == "float" => false,
        Some(Value::String(s)) if s == "base64" => true,
        Some(_) => return Err("`encoding_format` deve ser \"float\" ou \"base64\".".into()),
    };
    let dimensions = match v.get("dimensions") {
        None | Some(Value::Null) => None,
        Some(d) => match d.as_u64().filter(|d| *d > 0) {
            Some(d) => Some(d as usize),
            None => return Err("`dimensions` deve ser um inteiro positivo.".into()),
        },
    };
    let normalize = match v.get("normalize") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("`normalize` (extensão do Buriti) deve ser verdadeiro ou falso.".into()),
    };
    let model = v.get("model").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
    Ok(EmbedRequest { inputs, model, base64, dimensions, normalize })
}

/// Resposta no formato da OpenAI, mais o campo extra `buriti` (modo de memória, pooling, qualidade e recomendação).
pub fn build_response(out: &EmbedOutput, model: &str, base64: bool, meta: Value) -> Value {
    let data: Vec<Value> = out
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| json!({ "object": "embedding", "index": i, "embedding": if base64 { Value::String(base64_f32(v)) } else { json!(v) } }))
        .collect();
    json!({ "object": "list", "data": data, "model": model,
        "usage": { "prompt_tokens": out.prompt_tokens, "total_tokens": out.prompt_tokens },
        "buriti": meta })
}

// ------------------------------------------------------------------ cálculo

/// Posição de um token a partir de `tokenizer.ggml.eos_token_id` (modelos como o Qwen3-Embedding pedem o EOS no fim).
fn gguf_u64(m: &std::collections::HashMap<String, crate::native::gguf::Value>, key: &str) -> Option<u64> {
    m.get(key).and_then(|v| v.as_u64())
}

fn tokenize(model: &Model, text: &str, add_eos: Option<u32>) -> Vec<u32> {
    // texto do usuário nunca vira token especial: `encode_plain` não os interpreta (só o BOS do modelo é acrescentado)
    let mut ids = Vec::new();
    if let Some(b) = model.tok.bos() {
        ids.push(b);
    }
    ids.extend(model.tok.encode_plain(text));
    if let Some(e) = add_eos {
        ids.push(e);
    }
    ids
}

/// Tamanho do KV cache (bytes) por token de uma sessão do modelo: camadas × (K + V) × dimensão KV × 2 bytes (f16).
fn kv_bytes_per_token(model: &Model) -> u64 {
    (model.cfg.n_layer * model.cfg.n_kv * model.cfg.head_dim * 2 * 2) as u64
}

/// Máximo de tokens por texto para ESTE modelo: o da política, o contexto de treino e o que cabe no orçamento de KV.
pub fn effective_max_tokens(model: &Model, job: &EmbedJob) -> usize {
    let by_kv = (job.kv_budget_bytes / kv_bytes_per_token(model).max(1)) as usize;
    job.max_tokens.min(model.cfg.n_ctx_train).min(by_kv).max(16)
}

/// Calcula os embeddings de todos os textos (síncrono; rodar dentro do pool do motor). Cada texto usa a sua própria sessão
/// (KV isolado, sem tocar no cache do chat), então o resultado de um texto não depende dos outros do pedido.
pub fn embed_job(model: &Model, job: &EmbedJob) -> Result<EmbedOutput, String> {
    let meta = model.gguf_meta();
    let arch = meta.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("");
    let (pooling, from_file) = Pooling::from_gguf(gguf_u64(meta, &format!("{arch}.pooling_type")))?;
    let add_eos = meta.get("tokenizer.ggml.add_eos_token").and_then(|v| v.as_bool()).unwrap_or(false);
    let eos = if add_eos { gguf_u64(meta, "tokenizer.ggml.eos_token_id").map(|x| x as u32) } else { None };
    let dim = model.cfg.n_embd;
    if let Some(d) = job.dimensions {
        if d != dim {
            return Err(format!("{BAD_REQUEST} `dimensions` = {d}: este modelo produz vetores de {dim} dimensões e o Buriti não trunca vetores (omita `dimensions` ou use {dim})."));
        }
    }
    let cap = effective_max_tokens(model, job);

    // valida TUDO antes de calcular (um texto grande demais não gasta minutos de CPU nos anteriores)
    let mut all: Vec<Vec<u32>> = Vec::with_capacity(job.inputs.len());
    for (i, t) in job.inputs.iter().enumerate() {
        let ids = tokenize(model, t, eos);
        if ids.len() > cap {
            return Err(format!(
                "{BAD_REQUEST} `input[{i}]` tem {} tokens; o limite nesta máquina e com este modelo é {cap} (política de memória, contexto de treino {}). Divida o texto em partes menores.",
                ids.len(),
                model.cfg.n_ctx_train
            ));
        }
        all.push(ids);
    }

    let mut vectors = Vec::with_capacity(all.len());
    let mut prompt_tokens = 0usize;
    let mut sess: Option<(usize, Session)> = None;
    let mut hidden: Vec<f32> = Vec::new();
    for ids in &all {
        prompt_tokens += ids.len();
        // sessão do tamanho do texto (potência de 2, para reaproveitar entre textos parecidos); só depende deste texto
        let want = ids.len().next_power_of_two().max(256).min(cap).max(ids.len());
        if sess.as_ref().map(|(c, _)| *c) != Some(want) {
            sess = Some((want, model.new_session(want)));
        }
        let s = &mut sess.as_mut().unwrap().1;
        let mut pooler = Pooler::new(pooling, dim);
        let mut pos = 0;
        for chunk in ids.chunks(CHUNK) {
            hidden.clear();
            model.forward_hidden(s, chunk, pos, &mut hidden)?;
            pooler.push(&hidden);
            pos += chunk.len();
        }
        let mut v = pooler.finish().ok_or("texto sem tokens")?;
        if job.normalize {
            l2_normalize(&mut v)?;
        } else if v.iter().any(|x| !x.is_finite()) {
            return Err("o modelo produziu valores não finitos (estouro numérico).".into());
        }
        vectors.push(v);
    }
    Ok(EmbedOutput { vectors, prompt_tokens, pooling, pooling_from_file: from_file, normalized: job.normalize })
}

// ------------------------------------------------------- política de memória

const MIB: f64 = 1_048_576.0;
/// Tamanho de referência de um modelo de embeddings pequeno (Qwen3-Embedding 0,6B em Q8_0, ~0,64 GB [estimativa, o
/// arquivo não foi medido aqui]) para recomendar o que instalar quando ainda não há nenhum.
pub const REFERENCE_EMBED_BYTES: u64 = 680 * 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// o modelo do chat calcula os embeddings (qualidade menor, sem custo de memória extra)
    Shared,
    /// um modelo dedicado, carregado só durante o pedido e descartado depois
    OnDemand,
    /// um modelo dedicado, residente ao lado do modelo do chat (descartado após ocioso)
    Dedicated,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Shared => "shared",
            Mode::OnDemand => "on_demand",
            Mode::Dedicated => "dedicated",
        }
    }
}

/// Entradas medidas da máquina e dos arquivos.
#[derive(Clone, Copy, Debug)]
pub struct PlanInput {
    pub ram_total_mb: u64,
    /// RAM disponível agora (se conhecida)
    pub ram_available_mb: Option<u64>,
    /// memória de vídeo (se conhecida); só conta se o modelo do chat roda na GPU
    pub vram_mb: Option<u64>,
    /// o modelo do chat está na GPU (seus pesos quase não ocupam RAM)
    pub chat_on_gpu: bool,
    pub chat_bytes: u64,
    /// arquivo do modelo de embeddings dedicado, se há um candidato
    pub embed_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub mode: Mode,
    /// código estável da razão (a interface o traduz): `no_dedicated_model`, `fits_resident`, `fits_on_demand`, `does_not_fit`
    pub reason_code: &'static str,
    /// razão legível (pt-BR)
    pub reason: String,
    /// recomendação ao usuário (pt-BR) e o seu código: `use_dedicated`, `use_dedicated_on_demand`, `keep_chat_model`,
    /// `install_dedicated`, `install_dedicated_on_demand`
    pub recommendation_code: &'static str,
    pub recommendation: String,
    /// RAM que sobra para embeddings (MiB, pode ser negativa) depois da margem do sistema e do modelo do chat
    pub free_mb: i64,
    pub max_tokens: usize,
    pub kv_budget_bytes: u64,
}

/// Reserva para o sistema, o navegador e o próprio Buriti: o maior entre 2 GiB e 15% da RAM.
fn margin_mb(total_mb: f64) -> f64 {
    (total_mb * 0.15).max(2048.0)
}

/// RAM (MiB) que o modelo do chat ocupa: pesos mapeados + KV e buffers (~15% + 256 MiB). Na GPU, ~25% dos pesos + 300 MiB
/// (parte do arquivo permanece em cache) quando a memória de vídeo é conhecida.
fn chat_resident_mb(i: &PlanInput) -> f64 {
    let w = i.chat_bytes as f64 / MIB;
    if i.chat_on_gpu && i.vram_mb.is_some() {
        w * 0.25 + 300.0
    } else {
        w * 1.15 + 256.0
    }
}

/// Modo recomendado para a máquina (função pura, testada em perfis de 4, 8, 16 e 32 GB). Estimativas, não medições:
/// o consumo real de cada modelo depende do formato e do contexto; as margens são conservadoras de propósito.
pub fn plan(i: &PlanInput) -> Plan {
    let total = i.ram_total_mb as f64;
    let mut free = total - margin_mb(total) - chat_resident_mb(i);
    if let Some(a) = i.ram_available_mb {
        free = free.min(a as f64 - 512.0); // o que está livre de fato agora, com folga
    }
    let free_mb = free.round() as i64;

    // limites de entrada pelo que sobra (o motor ainda os reduz pelo KV real do modelo)
    let max_tokens = match free_mb {
        ..=-1 => 256,
        0..=511 => 512,
        512..=2047 => 2048,
        2048..=6143 => 4096,
        _ => 8192,
    };
    let kv_budget_bytes = (free.max(0.0) / 8.0).clamp(32.0, 1024.0) as u64 * 1_048_576;

    // custo estimado de um modelo de embeddings: residente (com folga para reempacotar e buffers) ou só durante o pedido
    let resident_need = |bytes: u64| bytes as f64 / MIB * 1.3 + 200.0 + 512.0;
    let on_demand_need = |bytes: u64| bytes as f64 / MIB * 1.15 + 150.0;
    let gb = |mb: f64| format!("{:.1} GB", mb / 1024.0);

    let (mode, reason_code, reason, recommendation_code, recommendation) = match i.embed_bytes {
        None => {
            // sem modelo dedicado: usa o do chat; recomenda instalar um se a máquina comportar
            let r = REFERENCE_EMBED_BYTES;
            let (code, rec) = if free >= resident_need(r) {
                ("install_dedicated", format!("Instale um modelo de embeddings pequeno (cerca de {}): cabe residente ao lado do modelo do chat e dá vetores de melhor qualidade.", gb(r as f64 / MIB)))
            } else if free >= on_demand_need(r) {
                ("install_dedicated_on_demand", format!("Um modelo de embeddings pequeno (cerca de {}) caberia, mas só carregado durante o pedido (mais lento no primeiro uso). Se a velocidade importar, mantenha o modelo do chat.", gb(r as f64 / MIB)))
            } else {
                ("keep_chat_model", "Nesta máquina, mantenha o modelo do chat para embeddings: não sobra memória para um segundo modelo.".to_string())
            };
            (Mode::Shared, "no_dedicated_model", "Nenhum modelo de embeddings dedicado foi escolhido: os vetores saem do modelo do chat (qualidade menor que a de um modelo treinado para isso).".to_string(), code, rec)
        }
        Some(b) if free >= resident_need(b) => (
            Mode::Dedicated,
            "fits_resident",
            format!("O modelo de embeddings ({}) cabe ao lado do modelo do chat com folga (sobram cerca de {} de RAM): fica residente.", gb(b as f64 / MIB), gb(free)),
            "use_dedicated",
            "Use o modelo de embeddings dedicado (residente).".to_string(),
        ),
        Some(b) if free >= on_demand_need(b) => (
            Mode::OnDemand,
            "fits_on_demand",
            format!("O modelo de embeddings ({}) cabe, mas sem folga para ficar residente (sobram cerca de {} de RAM): é carregado durante o pedido e descartado.", gb(b as f64 / MIB), gb(free)),
            "use_dedicated_on_demand",
            "Use o modelo dedicado sob demanda; o primeiro pedido de cada lote é mais lento (carga do modelo). Para vetores mais rápidos, feche outros programas ou use um modelo de chat menor.".to_string(),
        ),
        Some(b) => (
            Mode::Shared,
            "does_not_fit",
            format!("O modelo de embeddings ({}) não cabe junto do modelo do chat (sobram cerca de {} de RAM); os vetores saem do modelo do chat.", gb(b as f64 / MIB), gb(free.max(0.0))),
            "keep_chat_model",
            "Mantenha o modelo do chat para embeddings, ou troque por um modelo de chat menor para liberar memória.".to_string(),
        ),
    };
    Plan { mode, reason_code, reason, recommendation_code, recommendation, free_mb, max_tokens, kv_budget_bytes }
}

impl Plan {
    pub fn to_json(&self) -> Value {
        json!({ "mode": self.mode.name(), "reason_code": self.reason_code, "reason": self.reason,
            "recommendation_code": self.recommendation_code, "recommendation": self.recommendation,
            "free_mb": self.free_mb, "max_tokens_per_input": self.max_tokens })
    }

    /// Cópia com os limites de uma entrada (para o trabalho entregue ao motor).
    pub fn limits(&self) -> (usize, u64) {
        (self.max_tokens, self.kv_budget_bytes)
    }
}

/// RAM medida agora: (total, disponível) em MiB.
pub fn measure_ram() -> (u64, Option<u64>) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let avail = sys.available_memory();
    (sys.total_memory() / 1_048_576, if avail > 0 { Some(avail / 1_048_576) } else { None })
}

// --------------------------------------------------- modelo dedicado (CPU)

/// Modelo de embeddings dedicado, com o seu pool de threads.
pub struct Embedder {
    model: Model,
    pool: rayon::ThreadPool,
}

struct Resident {
    path: PathBuf,
    emb: Arc<Embedder>,
    last_used: Instant,
}

static RESIDENT: Mutex<Option<Resident>> = Mutex::new(None);
/// Um pedido de embeddings dedicado por vez (a CPU já é toda usada por um) e uma carga por vez.
static GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Arquiteturas que o motor próprio sabe rodar como embedder (decoders). BERT e afins não.
const RUNNABLE_ARCHS: [&str; 5] = ["qwen2", "qwen3", "llama", "gemma3", "phi3"];

/// Confere (só os metadados) se o arquivo é um decoder que o motor roda; devolve o motivo em pt-BR se não for.
pub fn check_embedder_file(path: &Path) -> Result<(), String> {
    let g = crate::native::gguf::Gguf::open(path)?;
    let arch = g.meta.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("?").to_string();
    if RUNNABLE_ARCHS.contains(&arch.as_str()) {
        return Ok(());
    }
    if matches!(arch.as_str(), "bert" | "nomic-bert" | "modern-bert" | "jina-bert-v2" | "xlm-roberta" | "roberta") {
        return Err(format!("{BAD_REQUEST} a arquitetura `{arch}` (modelo de embeddings de codificação, tipo BERT) ainda não é suportada pelo motor próprio. Use um modelo de embeddings baseado em decoder (por exemplo Qwen3-Embedding), ou omita `model` para usar o modelo do chat."));
    }
    Err(format!("{BAD_REQUEST} a arquitetura `{arch}` não é suportada pelo motor próprio para embeddings."))
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2)
}

fn load_embedder(path: &Path, repack: bool) -> Result<Embedder, String> {
    check_embedder_file(path)?;
    let model = if repack { Model::load(path)? } else { Model::load_with(path, false)? };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads()).build().map_err(|e| format!("pool de threads: {e}"))?;
    Ok(Embedder { model, pool })
}

/// Calcula com um modelo dedicado: `Dedicated` o deixa residente (descartado após `RESIDENT_IDLE` parado); `OnDemand`
/// o carrega para este pedido e o descarta ao fim. Qualquer outro modo é erro de uso.
pub async fn run_dedicated(path: PathBuf, mode: Mode, job: EmbedJob) -> Result<EmbedOutput, String> {
    assert!(mode != Mode::Shared, "run_dedicated só vale para modelo dedicado");
    let _gate = GATE.lock().await;
    let keep = mode == Mode::Dedicated;
    let cached = {
        let mut r = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
        match r.as_mut() {
            Some(x) if x.path == path => {
                x.last_used = Instant::now();
                Some(x.emb.clone())
            }
            _ => None,
        }
    };
    let loaded_now = cached.is_none();
    let p = path.clone();
    let emb = match cached {
        Some(e) => e,
        None => {
            // `Dedicated` reempacota os pesos (prefill mais rápido); `OnDemand` não, para gastar menos memória e tempo de carga
            let t0 = Instant::now();
            let e = Arc::new(tokio::task::spawn_blocking(move || load_embedder(&p, keep)).await.map_err(|e| format!("embeddings: {e}"))??);
            eprintln!("[embeddings] modelo dedicado carregado em {:.1}s ({}, modo {})", t0.elapsed().as_secs_f32(), path.display(), mode.name());
            e
        }
    };
    if keep && loaded_now {
        // troca o residente anterior (se havia outro) e agenda o descarte por ociosidade
        *RESIDENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Resident { path: path.clone(), emb: emb.clone(), last_used: Instant::now() });
        tokio::spawn(reap_when_idle(path.clone()));
    }
    let e2 = emb.clone();
    let out = tokio::task::spawn_blocking(move || e2.pool.install(|| embed_job(&e2.model, &job))).await.map_err(|e| format!("embeddings: {e}"))?;
    if !keep {
        eprintln!("[embeddings] modelo dedicado descartado (sob demanda)");
    }
    out
}

/// Descarta o modelo residente se ficar `RESIDENT_IDLE` sem uso (confere a cada espera e reagenda se houve uso).
async fn reap_when_idle(path: PathBuf) {
    loop {
        tokio::time::sleep(RESIDENT_IDLE).await;
        let mut r = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
        match r.as_ref() {
            Some(x) if x.path == path => {
                if x.last_used.elapsed() >= RESIDENT_IDLE {
                    *r = None;
                    eprintln!("[embeddings] modelo dedicado descartado (ocioso)");
                    return;
                }
            }
            _ => return, // foi trocado por outro (que tem o seu próprio vigia) ou já descartado
        }
    }
}

/// Descarta o residente agora (por exemplo, ao trocar de modelo ou parar o motor).
#[allow(dead_code)]
pub fn drop_resident() {
    *RESIDENT.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

// ------------------------------------------------------------ rotas da API

fn api_err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": { "message": msg.into(), "type": kind, "param": null, "code": null } }))).into_response()
}

fn failure(e: String) -> Response {
    match e.strip_prefix(BAD_REQUEST) {
        Some(m) => api_err(StatusCode::BAD_REQUEST, "invalid_request_error", m.trim().to_string()),
        None => api_err(StatusCode::BAD_GATEWAY, "engine_error", e),
    }
}

fn file_name_of(path: &str, fallback: &str) -> String {
    Path::new(path).file_name().map(|n| n.to_string_lossy().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| fallback.to_string())
}

/// Nome do arquivo do modelo de embeddings padrão (`BURITI_EMBED_MODEL`, um .gguf da pasta de modelos), se configurado.
fn default_embed_model() -> Option<String> {
    std::env::var("BURITI_EMBED_MODEL").ok().map(|s| s.trim().to_string()).filter(|s| crate::config::valid_model_name(s))
}

/// Resolve um nome da pasta de modelos para o arquivo (só nome simples, `.gguf` com o cabeçalho certo).
fn models_dir_file(models: &Path, name: &str) -> Option<PathBuf> {
    if !crate::config::valid_model_name(name) {
        return None;
    }
    let p = models.join(name);
    (p.is_file() && crate::config::is_gguf(&p)).then_some(p)
}

/// O que o pedido pede: o modelo do chat, ou um arquivo dedicado (`explicit` = o cliente o nomeou; senão veio do padrão).
enum Target {
    Chat,
    Dedicated { name: String, path: PathBuf, explicit: bool },
}

fn resolve_target(requested: Option<&str>, chat_id: &str, models: &Path) -> Result<Target, String> {
    match requested {
        Some(r) if r == chat_id => Ok(Target::Chat),
        Some(r) if r.to_ascii_lowercase().ends_with(".gguf") => match models_dir_file(models, r) {
            Some(path) => Ok(Target::Dedicated { name: r.to_string(), path, explicit: true }),
            None => Err(format!("Modelo de embeddings `{r}` não encontrado na pasta de modelos ({}). Coloque o .gguf lá, ou omita `model` para usar o modelo do chat.", models.display())),
        },
        // outros nomes (`text-embedding-3-small`, `default`...) são apelidos que os clientes mandam: valem como "o padrão do Buriti"
        _ => Ok(match default_embed_model().and_then(|n| models_dir_file(models, &n).map(|p| (n, p))) {
            Some((name, path)) if name != chat_id => Target::Dedicated { name, path, explicit: false },
            _ => Target::Chat,
        }),
    }
}

/// `POST /v1/embeddings`: mesmas regras de autenticação do resto de `/v1` (o middleware de `openai.rs` já passou por aqui).
pub async fn handler(State(st): State<Shared>, raw: Bytes) -> Response {
    let body = match serde_json::from_slice::<Value>(&raw) {
        Ok(v) if v.is_object() => v,
        Ok(_) => return api_err(StatusCode::BAD_REQUEST, "invalid_request_error", "O corpo deve ser um objeto JSON."),
        Err(e) => return api_err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("JSON inválido: {e}")),
    };
    let req = match parse_request(&body) {
        Ok(r) => r,
        Err(e) => return api_err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let (chat_id, chat_file) = {
        let s = st.settings.read().await;
        let id = if !s.model_path.is_empty() { file_name_of(&s.model_path, "") } else { s.model.clone() };
        (id, s.model_file(&st.paths).ok())
    };
    let target = match resolve_target(req.model.as_deref(), &chat_id, &st.paths.models) {
        Ok(t) => t,
        Err(e) => return api_err(StatusCode::NOT_FOUND, "model_not_found", e),
    };
    let eng = active(&st).await;
    let status = eng.status().await;
    let chat_on_gpu = status.message.as_deref().is_some_and(|m| m.starts_with("Calculando na GPU"));
    let (ram_total_mb, ram_available_mb) = measure_ram();
    let chat_bytes = chat_file.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
    let embed_bytes = match &target {
        Target::Dedicated { path, .. } => std::fs::metadata(path).ok().map(|m| m.len()),
        Target::Chat => None,
    };
    let vram_mb = crate::fim::read_vram_mb(crate::fim::gpu_name_from_status(status.message.as_deref()).as_deref()).await; // D78: a mesma fonte do plano de GPU
    let p = plan(&PlanInput { ram_total_mb, ram_available_mb, vram_mb, chat_on_gpu, chat_bytes, embed_bytes });

    // modo efetivo e arquivo usado
    let (mode, used_name, dedicated_path, fallback) = match &target {
        Target::Chat => (Mode::Shared, chat_id.clone(), None, false),
        Target::Dedicated { name, path, explicit } => match p.mode {
            Mode::Shared if *explicit => {
                return api_err(StatusCode::CONFLICT, "insufficient_memory", format!("O modelo de embeddings `{name}` não cabe na memória agora. {} {}", p.reason, p.recommendation));
            }
            // padrão do Buriti que não cabe: usa o modelo do chat e avisa (`fallback`); o nome do modelo na resposta diz qual foi usado
            Mode::Shared => (Mode::Shared, chat_id.clone(), None, true),
            m => (m, name.clone(), Some(path.clone()), false),
        },
    };
    let (max_tokens, kv_budget_bytes) = p.limits();
    let job = EmbedJob { inputs: req.inputs, normalize: req.normalize, dimensions: req.dimensions, max_tokens, kv_budget_bytes };
    let t0 = Instant::now();
    let result = match dedicated_path {
        Some(path) => run_dedicated(path, mode, job).await,
        None => {
            if status.state != "ready" {
                return api_err(StatusCode::CONFLICT, "model_not_ready", "Nenhum modelo carregado e pronto. Abra o Buriti, escolha o modelo e inicie o motor (ou nomeie um modelo de embeddings dedicado em `model`).");
            }
            eng.embed(job).await
        }
    };
    let out = match result {
        Ok(o) => o,
        Err(e) => return failure(e),
    };
    let dim = out.vectors.first().map(Vec::len).unwrap_or(0);
    eprintln!(
        "[embeddings] {} texto(s), {} tokens, {dim} dimensões, modo {} ({}) em {:.1}s{}",
        out.vectors.len(),
        out.prompt_tokens,
        mode.name(),
        if mode == Mode::Shared { "modelo do chat, qualidade menor" } else { "modelo dedicado" },
        t0.elapsed().as_secs_f32(),
        if fallback { " [o modelo dedicado padrão não coube: usado o do chat]" } else { "" }
    );
    let meta = json!({
        "mode": mode.name(),
        "quality": if mode == Mode::Shared { "chat_model" } else { "dedicated" },
        "fallback": fallback,
        "pooling": out.pooling.name(),
        "pooling_from_file": out.pooling_from_file,
        "normalized": out.normalized,
        "dimensions": dim,
        "plan": p.to_json(),
    });
    Json(build_response(&out, &used_name, req.base64, meta)).into_response()
}

/// `GET /v1/embeddings/plan`: o que a máquina comporta agora (RAM medida), o modo recomendado e os candidatos a modelo
/// dedicado da pasta de modelos (arquivos com "embed" no nome). Os mesmos campos servem à tela do Buriti.
pub async fn plan_handler(State(st): State<Shared>) -> Response {
    let (chat_id, chat_file) = {
        let s = st.settings.read().await;
        let id = if !s.model_path.is_empty() { file_name_of(&s.model_path, "") } else { s.model.clone() };
        (id, s.model_file(&st.paths).ok())
    };
    let status = active(&st).await.status().await;
    let chat_on_gpu = status.message.as_deref().is_some_and(|m| m.starts_with("Calculando na GPU"));
    let (ram_total_mb, ram_available_mb) = measure_ram();
    let chat_bytes = chat_file.as_ref().and_then(|p| std::fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
    let vram_mb = crate::fim::read_vram_mb(crate::fim::gpu_name_from_status(status.message.as_deref()).as_deref()).await; // D78
    let base = PlanInput { ram_total_mb, ram_available_mb, vram_mb, chat_on_gpu, chat_bytes, embed_bytes: None };
    let current = plan(&base);
    let models_dir = st.paths.models.clone();
    let chat_id2 = chat_id.clone();
    let candidates = tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&models_dir) else { return out };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".gguf") || !lower.contains("embed") || lower.contains(".mmproj") || name == chat_id2 {
                continue;
            }
            let size = e.metadata().map(|m| m.len()).unwrap_or(0);
            let check = check_embedder_file(&e.path());
            let pl = plan(&PlanInput { embed_bytes: Some(size), ..base });
            out.push(json!({ "name": name, "size_bytes": size, "runnable": check.is_ok(),
                "problem": check.err().map(|m| m.trim_start_matches(BAD_REQUEST).trim().to_string()),
                "plan": pl.to_json() }));
        }
        out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        out
    })
    .await
    .unwrap_or_default();
    Json(json!({
        "ram_total_mb": ram_total_mb, "ram_available_mb": ram_available_mb, "chat_model": chat_id, "chat_on_gpu": chat_on_gpu,
        "current": current.to_json(),
        "default_dedicated": default_embed_model(),
        "reference_embedder_mb": REFERENCE_EMBED_BYTES / 1_048_576,
        "candidates": candidates,
    }))
    .into_response()
}

// -------------------------------------------------------------------- testes

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- pooling e normalização contra uma referência em f64

    fn rows(n: usize, d: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n * d)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 6.0
            })
            .collect()
    }

    /// Referência ingênua em f64 (uma passada, sem lotes).
    fn reference(kind: Pooling, h: &[f32], d: usize, normalize: bool) -> Vec<f64> {
        let n = h.len() / d;
        let mut v: Vec<f64> = match kind {
            Pooling::Mean => (0..d).map(|j| (0..n).map(|i| h[i * d + j] as f64).sum::<f64>() / n as f64).collect(),
            Pooling::Cls => h[..d].iter().map(|x| *x as f64).collect(),
            Pooling::Last => h[(n - 1) * d..].iter().map(|x| *x as f64).collect(),
        };
        if normalize {
            let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter_mut().for_each(|x| *x /= nn);
        }
        v
    }

    /// O contrato do pooling: bate com a referência em f64 e não depende de como os tokens foram divididos em lotes.
    /// Recebe a implementação por parâmetro para o teste com bug plantado reprovar.
    fn check_pooling(pool: impl Fn(Pooling, &[f32], usize, &[usize], bool) -> Vec<f32>) {
        let d = 37;
        for kind in [Pooling::Mean, Pooling::Cls, Pooling::Last] {
            for n in [1usize, 2, 7, 33, 100] {
                let h = rows(n, d, n as u32);
                for normalize in [true, false] {
                    let want = reference(kind, &h, d, normalize);
                    for cuts in [vec![n], vec![1.min(n), n], vec![n / 2, n]] {
                        let got = pool(kind, &h, d, &cuts, normalize);
                        for (g, w) in got.iter().zip(&want) {
                            assert!((*g as f64 - w).abs() < 1e-6, "{kind:?} n={n} cortes={cuts:?}: {g} vs {w}");
                        }
                    }
                }
            }
        }
    }

    fn real_pool(kind: Pooling, h: &[f32], d: usize, cuts: &[usize], normalize: bool) -> Vec<f32> {
        let mut p = Pooler::new(kind, d);
        let mut lo = 0;
        for &hi in cuts {
            p.push(&h[lo * d..hi * d]);
            lo = hi;
        }
        let mut v = p.finish().unwrap();
        if normalize {
            l2_normalize(&mut v).unwrap();
        }
        v
    }

    #[test]
    fn pooling_matches_f64_reference_and_ignores_batching() {
        check_pooling(real_pool);
    }

    #[test]
    fn normalized_vectors_have_unit_norm() {
        let mut v: Vec<f32> = rows(1, 896, 3);
        l2_normalize(&mut v).unwrap();
        let n = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "{n}");
        let mut z = vec![0.0f32; 8];
        l2_normalize(&mut z).unwrap(); // vetor nulo: fica como está, sem NaN
        assert!(z.iter().all(|x| *x == 0.0));
        assert!(l2_normalize(&mut [1.0, f32::NAN]).is_err());
        assert!(l2_normalize(&mut [f32::INFINITY, 1.0]).is_err());
    }

    /// BUG PLANTADO 1: a média divide por n-1 (erro clássico). O contrato do pooling tem de reprovar.
    #[test]
    #[should_panic(expected = "vs")]
    fn planted_bug_mean_divides_by_n_minus_one_is_caught() {
        check_pooling(|kind, h, d, cuts, normalize| {
            let mut v = real_pool(kind, h, d, cuts, false);
            if kind == Pooling::Mean && h.len() / d > 1 {
                let n = (h.len() / d) as f32;
                v.iter_mut().for_each(|x| *x *= n / (n - 1.0));
            }
            if normalize {
                l2_normalize(&mut v).unwrap(); // a normalização esconderia o erro de escala: só o caso sem normalizar o pega
            }
            v
        });
    }

    /// BUG PLANTADO 2: o último token é perdido (o laço para um antes do fim).
    #[test]
    #[should_panic(expected = "vs")]
    fn planted_bug_dropped_last_token_is_caught() {
        check_pooling(|kind, h, d, _cuts, normalize| {
            let n = h.len() / d;
            let mut p = Pooler::new(kind, d);
            p.push(&h[..(n - 1) * d]);
            let mut v = p.finish().unwrap_or_else(|| vec![0.0; d]);
            if normalize && v.iter().any(|x| *x != 0.0) {
                l2_normalize(&mut v).unwrap();
            }
            v
        });
    }

    /// BUG PLANTADO 3: a normalização divide pela norma ao quadrado.
    #[test]
    #[should_panic(expected = "vs")]
    fn planted_bug_wrong_normalization_is_caught() {
        check_pooling(|kind, h, d, cuts, normalize| {
            let mut v = real_pool(kind, h, d, cuts, false);
            if normalize {
                let n2: f32 = v.iter().map(|x| x * x).sum();
                v.iter_mut().for_each(|x| *x /= n2);
            }
            v
        });
    }

    // ---------- base64

    #[test]
    fn base64_matches_rfc4648_vectors_and_roundtrips() {
        // vetores do RFC 4648 sobre bytes (f32 de bytes conhecidos: 4 bytes por valor)
        let enc = |b: &[u8]| {
            // reaproveita o codificador por meio de floats cujos bytes são `b` (b.len() múltiplo de 4)
            let f: Vec<f32> = b.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            base64_f32(&f)
        };
        assert_eq!(enc(b"foob"), "Zm9vYg==");
        assert_eq!(enc(b"foobarfo"), "Zm9vYmFyZm8=");
        assert_eq!(enc(b"abcdefghijkl"), "YWJjZGVmZ2hpamts");
        // 1.0f32 = 00 00 80 3F
        assert_eq!(base64_f32(&[1.0]), "AACAPw==");
        assert_eq!(base64_f32(&[]), "");
    }

    // ---------- pedido

    #[test]
    fn parses_input_forms_and_refuses_unsupported() {
        let ok = |v: Value| parse_request(&v).unwrap();
        assert_eq!(ok(json!({ "input": "a" })).inputs, vec!["a"]);
        let r = ok(json!({ "input": ["a", "b"], "model": "m.gguf", "encoding_format": "base64", "normalize": false }));
        assert_eq!((r.inputs.len(), r.base64, r.normalize, r.model.as_deref()), (2, true, false, Some("m.gguf")));
        assert!(!ok(json!({ "input": "a", "encoding_format": null })).base64);
        for bad in [
            json!({}),
            json!({ "input": [] }),
            json!({ "input": "" }),
            json!({ "input": ["a", ""] }),
            json!({ "input": [1, 2, 3] }),
            json!({ "input": [[1, 2]] }),
            json!({ "input": 5 }),
            json!({ "input": null }),
            json!({ "input": "a", "encoding_format": "hex" }),
            json!({ "input": "a", "dimensions": 0 }),
            json!({ "input": "a", "dimensions": "x" }),
            json!({ "input": "a", "normalize": "yes" }),
        ] {
            assert!(parse_request(&bad).is_err(), "deveria recusar {bad}");
        }
        assert_eq!(parse_request(&json!({ "input": "a", "dimensions": 256 })).unwrap().dimensions, Some(256));
        let many: Vec<String> = (0..MAX_INPUTS + 1).map(|i| format!("t{i}")).collect();
        assert!(parse_request(&json!({ "input": many })).unwrap_err().contains("máximo"));
        let msg = parse_request(&json!({ "input": [[1]] })).unwrap_err();
        assert!(msg.contains("tokens"), "{msg}");
    }

    #[test]
    fn response_has_openai_shape_and_base64_decodes_to_the_floats() {
        let out = EmbedOutput { vectors: vec![vec![0.5, -1.0], vec![2.0, 0.25]], prompt_tokens: 7, pooling: Pooling::Mean, pooling_from_file: false, normalized: true };
        let r = build_response(&out, "m.gguf", false, json!({ "mode": "shared" }));
        assert_eq!((r["object"].as_str(), r["model"].as_str(), r["data"][1]["index"].as_u64()), (Some("list"), Some("m.gguf"), Some(1)));
        assert_eq!(r["data"][0]["embedding"], json!([0.5, -1.0]));
        assert_eq!(r["usage"], json!({ "prompt_tokens": 7, "total_tokens": 7 }));
        assert_eq!(r["buriti"]["mode"], "shared");
        let r = build_response(&out, "m.gguf", true, json!({}));
        assert_eq!(r["data"][0]["embedding"], base64_f32(&[0.5, -1.0]));
    }

    #[test]
    fn pooling_from_gguf_metadata() {
        assert_eq!(Pooling::from_gguf(None).unwrap(), (Pooling::Mean, false));
        assert_eq!(Pooling::from_gguf(Some(0)).unwrap(), (Pooling::Mean, false));
        assert_eq!(Pooling::from_gguf(Some(1)).unwrap(), (Pooling::Mean, true));
        assert_eq!(Pooling::from_gguf(Some(2)).unwrap(), (Pooling::Cls, true));
        assert_eq!(Pooling::from_gguf(Some(3)).unwrap(), (Pooling::Last, true));
        assert!(Pooling::from_gguf(Some(4)).unwrap_err().contains("reclassificação"));
        assert!(Pooling::from_gguf(Some(9)).is_err());
    }

    // ---------- política de memória em perfis de máquina

    const GB: u64 = 1024 * 1024 * 1024;
    fn profile(total_gb: u64, chat_bytes: u64, embed_bytes: Option<u64>) -> Plan {
        plan(&PlanInput { ram_total_mb: total_gb * 1024, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes, embed_bytes })
    }
    const CHAT_SMALL: u64 = 400 * 1_048_576; // Qwen 0,5B Q4_K_M
    const CHAT_MID: u64 = 1100 * 1_048_576; // 1,5B Q4_K_M
    const CHAT_7B: u64 = 4500 * 1_048_576; // 7B Q4_K_M
    const EMBED: u64 = 680 * 1_048_576;

    #[test]
    fn memory_policy_on_4_8_16_and_32_gb_machines() {
        // 4 GB: modelo de chat pequeno -> o dedicado só cabe sob demanda; chat médio -> nem isso
        let p = profile(4, CHAT_SMALL, Some(EMBED));
        assert_eq!((p.mode, p.reason_code), (Mode::OnDemand, "fits_on_demand"), "{p:?}");
        assert_eq!(profile(4, CHAT_MID, Some(EMBED)).mode, Mode::Shared);
        assert_eq!(profile(4, CHAT_MID, Some(EMBED)).reason_code, "does_not_fit");
        // 8 GB: chat médio -> dedicado residente; chat de 7B -> volta ao modelo do chat
        assert_eq!(profile(8, CHAT_MID, Some(EMBED)).mode, Mode::Dedicated);
        assert_eq!(profile(8, CHAT_7B, Some(EMBED)).mode, Mode::Shared);
        // 16 e 32 GB: residente, mesmo com chat de 7B
        assert_eq!(profile(16, CHAT_7B, Some(EMBED)).mode, Mode::Dedicated);
        assert_eq!(profile(32, CHAT_7B, Some(EMBED)).mode, Mode::Dedicated);
        // um embedder grande (4 GB) na máquina de 8 GB com chat médio: não cabe residente
        assert_ne!(profile(8, CHAT_MID, Some(4 * GB)).mode, Mode::Dedicated);
        assert_eq!(profile(32, CHAT_7B, Some(4 * GB)).mode, Mode::Dedicated);
    }

    #[test]
    fn memory_policy_recommends_what_to_install_without_a_dedicated_model() {
        for (gb, chat, rec) in [(4u64, CHAT_SMALL, "install_dedicated_on_demand"), (4, CHAT_MID, "keep_chat_model"), (8, CHAT_MID, "install_dedicated"), (16, CHAT_7B, "install_dedicated"), (32, CHAT_7B, "install_dedicated")] {
            let p = profile(gb, chat, None);
            assert_eq!((p.mode, p.reason_code, p.recommendation_code), (Mode::Shared, "no_dedicated_model", rec), "{gb} GB: {p:?}");
            assert!(!p.reason.is_empty() && !p.recommendation.is_empty());
        }
    }

    #[test]
    fn memory_policy_limits_grow_with_free_ram_and_use_measured_availability() {
        let small = profile(4, CHAT_MID, None);
        let big = profile(32, CHAT_SMALL, None);
        assert!(small.max_tokens < big.max_tokens && small.kv_budget_bytes < big.kv_budget_bytes, "{small:?} {big:?}");
        assert_eq!(big.max_tokens, 8192);
        // a RAM disponível medida agora derruba o modo mesmo numa máquina grande (outros programas ocupando)
        let busy = plan(&PlanInput { ram_total_mb: 32 * 1024, ram_available_mb: Some(1200), vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_SMALL, embed_bytes: Some(EMBED) });
        assert_eq!(busy.mode, Mode::Shared, "{busy:?}");
        let free = plan(&PlanInput { ram_available_mb: Some(24 * 1024), ..PlanInput { ram_total_mb: 32 * 1024, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_SMALL, embed_bytes: Some(EMBED) } });
        assert_eq!(free.mode, Mode::Dedicated);
        // memória estourada: limites mínimos, nunca negativos
        let tiny = profile(2, CHAT_7B, None);
        assert!(tiny.free_mb < 0 && tiny.max_tokens == 256 && tiny.kv_budget_bytes >= 32 * 1_048_576);
    }

    #[test]
    fn memory_policy_counts_a_chat_model_on_the_gpu_as_mostly_off_ram() {
        let on_cpu = plan(&PlanInput { ram_total_mb: 8 * 1024, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_7B, embed_bytes: Some(EMBED) });
        let on_gpu = plan(&PlanInput { ram_total_mb: 8 * 1024, ram_available_mb: None, vram_mb: Some(8192), chat_on_gpu: true, chat_bytes: CHAT_7B, embed_bytes: Some(EMBED) });
        assert_eq!(on_cpu.mode, Mode::Shared);
        assert_eq!(on_gpu.mode, Mode::Dedicated, "{on_gpu:?}");
        // GPU sem memória de vídeo conhecida: não se presume nada, vale o caso da CPU
        let unknown = plan(&PlanInput { vram_mb: None, chat_on_gpu: true, ..PlanInput { ram_total_mb: 8 * 1024, ram_available_mb: None, vram_mb: None, chat_on_gpu: false, chat_bytes: CHAT_7B, embed_bytes: Some(EMBED) } });
        assert_eq!(unknown.mode, Mode::Shared);
    }

    #[test]
    fn plan_json_carries_the_ui_fields() {
        let j = profile(8, CHAT_MID, Some(EMBED)).to_json();
        for k in ["mode", "reason_code", "reason", "recommendation_code", "recommendation", "free_mb", "max_tokens_per_input"] {
            assert!(j.get(k).is_some(), "falta {k}");
        }
        assert_eq!(j["mode"], "dedicated");
    }

    // ---------- com modelo real (Qwen 2.5 0,5B, o mesmo dos outros testes; pula se ausente)

    fn test_model_path() -> Option<String> {
        let path = std::env::var("BURITI_TEST_MODEL").unwrap_or_else(|_| format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR")));
        if !Path::new(&path).exists() {
            eprintln!("modelo de teste ausente, pulando: {path}");
            return None;
        }
        Some(path)
    }

    fn job(inputs: &[&str]) -> EmbedJob {
        EmbedJob { inputs: inputs.iter().map(|s| s.to_string()).collect(), normalize: true, dimensions: None, max_tokens: 8192, kv_budget_bytes: 512 * 1_048_576 }
    }

    const TEXTS: [&str; 4] = ["The capital of France is Paris.", "fn main() { println!(\"hello\"); }", "Olá, mundo! Isto é um teste de embeddings.", "x"];

    #[test]
    fn real_model_is_deterministic_normalized_and_batch_equals_single() {
        let Some(path) = test_model_path() else { return };
        let m = Model::load(Path::new(&path)).expect("carregar modelo");
        let all = embed_job(&m, &job(&TEXTS)).unwrap();
        assert_eq!(all.vectors.len(), 4);
        assert_eq!((all.pooling, all.pooling_from_file), (Pooling::Mean, false)); // modelo de chat: sem pooling_type no arquivo
        let dim = m.cfg.n_embd;
        // determinismo: o mesmo pedido de novo dá os MESMOS bits
        let again = embed_job(&m, &job(&TEXTS)).unwrap();
        assert_eq!(all.vectors, again.vectors);
        // normalização L2
        for v in &all.vectors {
            assert_eq!(v.len(), dim);
            let n = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-5, "norma {n}");
        }
        // lote == cada texto sozinho (igualdade estrita) e a ordem da lista é a ordem dos índices
        let mut tokens = 0;
        for (i, t) in TEXTS.iter().enumerate() {
            let one = embed_job(&m, &job(&[t])).unwrap();
            assert_eq!(one.vectors[0], all.vectors[i], "texto {i}: lote difere do individual");
            tokens += one.prompt_tokens;
        }
        assert_eq!(tokens, all.prompt_tokens);
        // lista invertida: os mesmos vetores, na ordem nova
        let rev: Vec<&str> = TEXTS.iter().rev().copied().collect();
        let r = embed_job(&m, &job(&rev)).unwrap();
        for i in 0..4 {
            assert_eq!(r.vectors[i], all.vectors[3 - i]);
        }
        // textos diferentes dão vetores diferentes (o teste tem dentes: um embedder constante reprovaria)
        for i in 0..4 {
            for j in i + 1..4 {
                let cos: f64 = all.vectors[i].iter().zip(&all.vectors[j]).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
                assert!(cos < 0.9999, "textos {i} e {j} deram o mesmo vetor (cos {cos})");
            }
        }
        // sem normalizar: a norma não é 1 e a direção é a mesma
        let raw = embed_job(&m, &EmbedJob { normalize: false, ..job(&[TEXTS[0]]) }).unwrap();
        let n = raw.vectors[0].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() > 1e-3, "sem normalizar a norma deveria diferir de 1 ({n})");
        let cos: f64 = raw.vectors[0].iter().zip(&all.vectors[0]).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() / n;
        assert!((cos - 1.0).abs() < 1e-5, "{cos}");
    }

    #[test]
    fn real_model_enforces_token_and_dimension_limits() {
        let Some(path) = test_model_path() else { return };
        let m = Model::load(Path::new(&path)).expect("carregar modelo");
        let long = "palavra ".repeat(300);
        let n = tokenize(&m, &long, None).len();
        assert!(n > 200);
        // limite da política abaixo do tamanho do texto: erro de pedido (400), apontando o índice, e nada é calculado
        let e = embed_job(&m, &EmbedJob { max_tokens: n - 1, ..job(&["curto", &long]) }).unwrap_err();
        assert!(e.starts_with(BAD_REQUEST) && e.contains("input[1]") && e.contains("tokens"), "{e}");
        // no limite exato passa
        assert!(embed_job(&m, &EmbedJob { max_tokens: n, ..job(&[&long]) }).is_ok());
        // o orçamento de KV também limita (1 MiB de KV não comporta o texto longo)
        let e = embed_job(&m, &EmbedJob { kv_budget_bytes: 1 << 20, ..job(&[&long]) }).unwrap_err();
        assert!(e.contains("limite"), "{e}");
        // dimensions só vale se igual à do modelo
        assert!(embed_job(&m, &EmbedJob { dimensions: Some(m.cfg.n_embd), ..job(&["a"]) }).is_ok());
        let e = embed_job(&m, &EmbedJob { dimensions: Some(256), ..job(&["a"]) }).unwrap_err();
        assert!(e.starts_with(BAD_REQUEST) && e.contains("dimensões"), "{e}");
    }

    #[test]
    fn real_model_input_never_becomes_a_special_token() {
        let Some(path) = test_model_path() else { return };
        let m = Model::load(Path::new(&path)).expect("carregar modelo");
        let a = tokenize(&m, "<|im_end|>", None);
        assert!(a.len() > 1, "o texto `<|im_end|>` virou um token só: {a:?}");
        assert!(!m.tok.eos.iter().any(|e| a.contains(e)), "o texto do usuário injetou um token de fim");
    }

    /// Sessão de ctx diferente (potência de 2) não muda o resultado: o mesmo texto sozinho ou depois de um texto longo.
    #[test]
    fn real_model_result_does_not_depend_on_what_came_before() {
        let Some(path) = test_model_path() else { return };
        let m = Model::load(Path::new(&path)).expect("carregar modelo");
        let long = "palavra ".repeat(300);
        let alone = embed_job(&m, &job(&["curto"])).unwrap();
        let after = embed_job(&m, &job(&[&long, "curto"])).unwrap();
        assert_eq!(alone.vectors[0], after.vectors[1]);
    }

    /// Dedicado e sob demanda (mesmo arquivo, por falta de um embedder real aqui): igual ao cálculo direto com o modelo
    /// compartilhado, quando o carregamento é igual (`OnDemand` carrega sem reempacotar, `Dedicated` com).
    #[tokio::test]
    async fn dedicated_paths_load_compute_and_unload() {
        let Some(path) = test_model_path() else { return };
        let p = PathBuf::from(&path);
        let direct_plain = embed_job(&Model::load_with(&p, false).unwrap(), &job(&TEXTS[..2])).unwrap();
        let on_demand = tokio::time::timeout(Duration::from_secs(300), run_dedicated(p.clone(), Mode::OnDemand, job(&TEXTS[..2]))).await.expect("tempo esgotado").unwrap();
        assert_eq!(on_demand.vectors, direct_plain.vectors);
        assert!(RESIDENT.lock().unwrap().is_none(), "sob demanda não pode deixar o modelo residente");
        let direct_rep = embed_job(&Model::load(&p).unwrap(), &job(&TEXTS[..2])).unwrap();
        let resident = tokio::time::timeout(Duration::from_secs(300), run_dedicated(p.clone(), Mode::Dedicated, job(&TEXTS[..2]))).await.expect("tempo esgotado").unwrap();
        assert_eq!(resident.vectors, direct_rep.vectors);
        assert!(RESIDENT.lock().unwrap().as_ref().is_some_and(|r| r.path == p));
        // segundo pedido reaproveita o residente (mesmo resultado)
        let again = tokio::time::timeout(Duration::from_secs(300), run_dedicated(p.clone(), Mode::Dedicated, job(&TEXTS[..2]))).await.expect("tempo esgotado").unwrap();
        assert_eq!(again.vectors, resident.vectors);
        drop_resident();
        assert!(RESIDENT.lock().unwrap().is_none());
    }

    #[test]
    fn architecture_check_accepts_decoders_and_explains_bert() {
        let Some(path) = test_model_path() else { return };
        assert!(check_embedder_file(Path::new(&path)).is_ok());
        assert!(check_embedder_file(Path::new("nao-existe.gguf")).is_err());
    }
}
