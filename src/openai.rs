//! API compatível com a OpenAI em `/v1` (só em 127.0.0.1), para usar o modelo carregado como assistente de código ("Buriti Code")
//! em editores (Continue, Cline, JetBrains...): `GET /v1/models`, `POST /v1/chat/completions` e
//! `POST /v1/completions` (com `suffix` = preenchimento no meio, FIM). Autenticação: `Authorization: Bearer
//! <chave>` com a chave estável do usuário (`Paths::api_key_file`), além do `Host` de loopback.
//! `POST /v1/embeddings` e `GET /v1/embeddings/plan` ficam em `embeddings.rs` (D77). Chamada de funções (`tools`) só no chat e só no motor próprio (`tools.rs`, D79); não há múltiplas escolhas (`n` > 1).

use std::{
    collections::HashMap,
    convert::Infallible,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream, Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::Notify;

use crate::{
    engine::{self, sse_events, ChatMsg, ChatRequest, CompletionRequest, SamplingOpts},
    server::{active, Shared},
    tools,
};

pub fn router(state: Shared) -> Router<Shared> {
    Router::new()
        .route("/models", get(models))
        .route("/chat/completions", post(chat_completions))
        .route("/completions", post(completions))
        .route("/embeddings", post(crate::embeddings::handler)) // D77
        .route("/embeddings/plan", get(crate::embeddings::plan_handler))
        .route("/fim/plan", get(crate::fim::plan_handler)) // D78
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024)) // prompts de editor com muito código
        .layer(middleware::from_fn_with_state(state, guard))
}

/// Comparação em tempo constante (não vaza, pelo tempo, quantos caracteres da chave estão certos).
fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn oai_err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": { "message": msg.into(), "type": kind, "param": null, "code": null } }))).into_response()
}

/// Host de loopback (contra DNS rebinding) e chave da API (contra outros sites e programas locais).
async fn guard(State(st): State<Shared>, req: Request, next: Next) -> Response {
    let host_ok = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(|h| {
            let h = h.to_ascii_lowercase();
            h == format!("127.0.0.1:{}", st.port) || h == format!("localhost:{}", st.port)
        })
        .unwrap_or(false);
    if !host_ok {
        return oai_err(StatusCode::UNAUTHORIZED, "invalid_request_error", "Host inválido: use http://127.0.0.1:<porta>/v1");
    }
    let given = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer ").or_else(|| h.strip_prefix("bearer ")))
        .or_else(|| req.headers().get("x-api-key").and_then(|h| h.to_str().ok()))
        .map(str::trim)
        .unwrap_or("");
    let key = st.api_key.read().await.clone();
    if given.is_empty() || !ct_eq(given, &key) {
        return oai_err(StatusCode::UNAUTHORIZED, "invalid_request_error", "Chave de API ausente ou incorreta. Copie a chave na tela \"API\" do Buriti e envie `Authorization: Bearer <chave>`.");
    }
    next.run(req).await
}

fn model_id(s: &crate::config::Settings) -> String {
    if !s.model_path.is_empty() {
        std::path::Path::new(&s.model_path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
    } else {
        s.model.clone()
    }
}

async fn models(State(st): State<Shared>) -> Json<Value> {
    let id = model_id(&*st.settings.read().await);
    let mut data: Vec<Value> = if id.is_empty() { vec![] } else { vec![json!({ "id": id, "object": "model", "created": 0, "owned_by": "buriti" })] };
    // extensão do Buriti (D79): se o modelo carregado aceita `tools`, em que formato e com que ressalvas (ou o que usar no lugar)
    if let Some(m) = data.first_mut() {
        let e = active(&st).await;
        if e.status().await.state == "ready" {
            m["buriti_tools"] = e.tool_support().describe();
        }
    }
    Json(json!({ "object": "list", "data": data }))
}

// ---------------------------------------------------------------- entrada

fn parse_stop(v: &Value) -> Vec<String> {
    match v.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).take(8).collect(),
        _ => vec![],
    }
}

fn parse_max_tokens(v: &Value) -> Option<u32> {
    v.get("max_tokens").or_else(|| v.get("max_completion_tokens")).and_then(Value::as_u64).map(|n| n.min(u32::MAX as u64) as u32)
}

/// Parâmetros de amostragem do corpo do pedido. Valores fora da faixa são limitados à faixa válida (não viram erro:
/// os editores mandam valores estranhos de vez em quando). `null` equivale a ausente.
fn parse_sampling(v: &Value) -> SamplingOpts {
    let f = |k: &str| v.get(k).and_then(Value::as_f64).filter(|x| x.is_finite()).map(|x| x as f32);
    SamplingOpts {
        top_p: f("top_p").map(|x| x.clamp(0.0, 1.0)),
        top_k: v.get("top_k").and_then(Value::as_i64).map(|x| x.clamp(0, 100_000) as u32),
        min_p: f("min_p").map(|x| x.clamp(0.0, 1.0)),
        repeat_penalty: f("repeat_penalty").filter(|x| *x > 0.0).map(|x| x.min(4.0)),
        frequency_penalty: f("frequency_penalty").map(|x| x.clamp(-2.0, 2.0)),
        presence_penalty: f("presence_penalty").map(|x| x.clamp(-2.0, 2.0)),
        seed: v.get("seed").and_then(|s| s.as_u64().or_else(|| s.as_i64().map(|i| i as u64))),
    }
}

/// Orçamento de tokens do prompt de um completar. `max_prompt_tokens` (extensão do Buriti, opcional) vale para qualquer
/// completar e é limitado a `MIN_PROMPT_TOKENS..=MAX_PROMPT_TOKENS`; sem ele, um pedido com `suffix` (FIM) usa
/// `FIM_DEFAULT_PROMPT_TOKENS` e um sem `suffix` só tem o limite do contexto (`None`). Valor que não é inteiro não
/// negativo vale como ausente.
fn resolve_prompt_budget(v: &Value, has_suffix: bool) -> Option<usize> {
    match v.get("max_prompt_tokens").and_then(Value::as_u64) {
        Some(n) => Some((n.min(engine::MAX_PROMPT_TOKENS as u64) as usize).max(engine::MIN_PROMPT_TOKENS)),
        None => has_suffix.then_some(engine::FIM_DEFAULT_PROMPT_TOKENS),
    }
}

// ------------------------------------------------ "o último vence" (só FIM)

/// Regra "o último pedido vence" para `/v1/completions` COM `suffix` (nunca no chat). Motivo: o editor dispara um pedido
/// de autocompletar por tecla e cancela fechando a conexão, mas alguns demoram ou nem fecham, e o motor atende um pedido
/// por vez: sem isso, os pedidos velhos (já obsoletos) fazem fila na frente do atual e o prefill de cada um gasta dezenas
/// de segundos de CPU. RISCO: dois programas diferentes pedindo FIM ao mesmo tempo com o mesmo `User-Agent` (por
/// exemplo dois editores baseados no mesmo cliente HTTP) cancelariam um ao outro; o pedido cancelado recebe uma resposta
/// vazia e válida (texto vazio, `finish_reason: "stop"`), não um erro. Só vale para pedidos que ainda esperam ou ainda
/// não produziram nenhum texto: quem já começou a responder termina (ou é cancelado pelo cliente). Para desligar:
/// `BURITI_FIM_LATEST_WINS=0` ou `FIM_LATEST_WINS = false`.
pub const FIM_LATEST_WINS: bool = true;

fn latest_wins_enabled() -> bool {
    FIM_LATEST_WINS && std::env::var("BURITI_FIM_LATEST_WINS").map(|v| v.trim() != "0").unwrap_or(true)
}

/// Um completar FIM em andamento (esperando a vez, no prefill ou gerando).
struct Inflight {
    id: u64,
    /// acorda o pedido para ele encerrar (substituído por outro mais novo)
    cancel: Notify,
    /// já saiu texto para o cliente: não pode mais ser substituído
    produced: AtomicBool,
    superseded: AtomicBool,
}

/// Pedidos FIM em andamento por cliente (a chave é o `User-Agent`; sem ele, todos caem na chave vazia).
#[derive(Default)]
struct FimRegistry {
    map: Mutex<HashMap<String, Vec<Arc<Inflight>>>>,
    next_id: AtomicU64,
}

impl FimRegistry {
    /// Registra um pedido novo do cliente `key` e substitui os mais velhos do mesmo cliente que ainda não produziram texto.
    fn enter(&'static self, key: &str) -> FimTicket {
        let inflight = Arc::new(Inflight { id: self.next_id.fetch_add(1, Ordering::Relaxed), cancel: Notify::new(), produced: AtomicBool::new(false), superseded: AtomicBool::new(false) });
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        let list = map.entry(key.to_string()).or_default();
        for old in list.iter() {
            if !old.produced.load(Ordering::SeqCst) && !old.superseded.swap(true, Ordering::SeqCst) {
                old.cancel.notify_one();
            }
        }
        list.push(inflight.clone());
        FimTicket { inflight, reg: self, key: key.to_string() }
    }

    fn leave(&self, key: &str, id: u64) {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(list) = map.get_mut(key) {
            list.retain(|i| i.id != id);
            if list.is_empty() {
                map.remove(key);
            }
        }
    }
}

fn fim_registry() -> &'static FimRegistry {
    static R: OnceLock<FimRegistry> = OnceLock::new();
    R.get_or_init(FimRegistry::default)
}

/// Fica vivo enquanto o pedido FIM durar (preso ao fluxo ou ao handler); ao cair, sai do registro.
struct FimTicket {
    inflight: Arc<Inflight>,
    reg: &'static FimRegistry,
    key: String,
}

impl Drop for FimTicket {
    fn drop(&mut self) {
        self.reg.leave(&self.key, self.inflight.id);
    }
}

type EventStream = Pin<Box<dyn Stream<Item = Value> + Send>>;

/// Envolve o fluxo de eventos do motor: marca o pedido como "já produziu" no 1º texto e o encerra se for substituído.
/// Sem `ticket` (chat, ou completar sem `suffix`) devolve o fluxo como veio. Ao encerrar, o fluxo do motor é derrubado:
/// o receptor do canal cai e o motor próprio abandona o pedido (na fila, antes do prefill; no prefill, entre os lotes).
fn track(events: impl Stream<Item = Value> + Send + 'static, ticket: Option<FimTicket>) -> EventStream {
    let Some(ticket) = ticket else { return Box::pin(events) };
    let inflight = ticket.inflight.clone();
    let waiting = inflight.clone();
    let stop = async move { waiting.cancel.notified().await };
    Box::pin(events.take_until(stop).inspect(move |v| {
        let _keep_registered = &ticket;
        if !chunk_parts(v, false).0.is_empty() {
            inflight.produced.store(true, Ordering::SeqCst);
        }
    }))
}

/// Resposta de um pedido substituído antes de começar: completar vazio e válido (não é erro para o editor).
fn superseded_reply(stream: bool, mk: &dyn Fn(Vec<Value>, Option<Value>) -> Value) -> Response {
    let choice = json!({ "index": 0, "text": "", "logprobs": null, "finish_reason": "stop" });
    if stream {
        sse_response(stream::iter(vec![sse_line(&mk(vec![choice], None)), Bytes::from_static(b"data: [DONE]\n\n")]))
    } else {
        Json(mk(vec![choice], None)).into_response()
    }
}

/// Recusa o que a API não implementa em vez de ignorar em silêncio (o editor mostraria um resultado errado).
/// `tools` só existe no chat (D79); `functions` (o formato antigo) e `tools` em `/v1/completions` continuam recusados.
fn unsupported(v: &Value, chat: bool) -> Option<&'static str> {
    if v.get("n").and_then(Value::as_u64).is_some_and(|n| n != 1) {
        return Some("`n` diferente de 1 não é suportado.");
    }
    let nonempty = |k: &str| v.get(k).is_some_and(|x| !x.is_null() && x.as_array().map(|a| !a.is_empty()).unwrap_or(true));
    if nonempty("functions") {
        return Some("`functions` (formato antigo da OpenAI) não é suportado; envie `tools`.");
    }
    if !chat && nonempty("tools") {
        return Some("Chamada de funções (`tools`) só existe em /v1/chat/completions.");
    }
    None
}

/// `content` pode ser texto ou uma lista de partes (`[{"type":"text","text":"..."}]`).
pub(crate) fn content_text(c: &Value) -> Result<String, String> {
    match c {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") | None => out.push_str(p.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some(other) => return Err(format!("conteúdo do tipo `{other}` não é suportado (só texto).")),
                }
            }
            Ok(out)
        }
        _ => Err("`content` inválido.".into()),
    }
}

fn parse_messages(v: &Value) -> Result<Vec<ChatMsg>, String> {
    let arr = v.get("messages").and_then(Value::as_array).ok_or("`messages` ausente.")?;
    let mut out = Vec::new();
    for m in arr {
        let role = match m.get("role").and_then(Value::as_str).unwrap_or("") {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            other => return Err(format!("papel `{other}` não é suportado (use system, user ou assistant).")),
        };
        out.push(ChatMsg { role: role.to_string(), content: content_text(m.get("content").unwrap_or(&Value::Null))? });
    }
    if out.is_empty() {
        return Err("`messages` vazio.".into());
    }
    Ok(out)
}

// ----------------------------------------------------------------- saída

fn rand_id(prefix: &str) -> String {
    let mut b = [0u8; 12];
    let _ = getrandom::fill(&mut b);
    format!("{prefix}-{}", b.iter().map(|x| format!("{x:02x}")).collect::<String>())
}

/// Id de chamada de função no estilo da OpenAI (`call_<hex>`).
fn call_id() -> String {
    rand_id("call").replacen('-', "_", 1)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn sse_line(v: &Value) -> Bytes {
    Bytes::from(format!("data: {v}\n\n"))
}

async fn ready_engine(st: &Shared) -> Result<std::sync::Arc<dyn engine::Engine>, Response> {
    let e = active(st).await;
    if e.status().await.state != "ready" {
        return Err(oai_err(StatusCode::CONFLICT, "model_not_ready", "Nenhum modelo carregado e pronto. Abra o Buriti, escolha o modelo e inicie o motor."));
    }
    Ok(e)
}

fn engine_failure(e: String) -> Response {
    if let Some(msg) = e.strip_prefix(engine::BAD_REQUEST) {
        oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", msg.trim().to_string())
    } else {
        oai_err(StatusCode::BAD_GATEWAY, "engine_error", e)
    }
}

fn sse_response(body: impl futures_util::Stream<Item = Bytes> + Send + 'static) -> Response {
    let body = body.map(Ok::<_, Infallible>);
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Texto e motivo de parada de um bloco do motor (`delta.content` no chat, `text` no completion).
fn chunk_parts(v: &Value, chat: bool) -> (String, Option<String>) {
    let ch = &v["choices"][0];
    let text = if chat { ch["delta"]["content"].as_str() } else { ch["text"].as_str() };
    (text.unwrap_or("").to_string(), ch["finish_reason"].as_str().map(str::to_string))
}

/// Contagem de tokens de um bloco do motor (`usage`), normalizada para os três campos da OpenAI. Aceita só números;
/// o `total_tokens` que faltar é a soma dos outros dois. Qualquer outra coisa (por exemplo os `timings` do llama.cpp) é descartada.
fn normalize_usage(v: &Value) -> Option<Value> {
    let u = v.get("usage").filter(|u| u.is_object())?;
    let p = u.get("prompt_tokens").and_then(Value::as_u64)?;
    let c = u.get("completion_tokens").and_then(Value::as_u64)?;
    let t = u.get("total_tokens").and_then(Value::as_u64).unwrap_or(p + c);
    Some(json!({ "prompt_tokens": p, "completion_tokens": c, "total_tokens": t }))
}

/// `stream_options.include_usage` do pedido: só com `true` explícito o fluxo ganha o bloco final de `usage`.
fn wants_stream_usage(body: &Value) -> bool {
    body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false)
}

/// Blocos de saída (já no formato da OpenAI) para um evento do motor. `mk(choices, usage)` monta o bloco completo
/// (id, modelo...). O texto/parada sai como antes; com `include_usage`, a contagem do motor vira um bloco à parte, com
/// `choices: []` (como a OpenAI), logo depois do bloco do `finish_reason`. Sem `include_usage` o `usage` nunca sai.
fn stream_chunks(v: &Value, chat: bool, include_usage: bool, mk: &dyn Fn(Vec<Value>, Option<Value>) -> Value) -> Vec<Value> {
    let (text, finish) = chunk_parts(v, chat);
    let mut out = Vec::new();
    if !(text.is_empty() && finish.is_none()) {
        let choice = if chat {
            let delta = if text.is_empty() { json!({}) } else { json!({ "content": text }) };
            json!({ "index": 0, "delta": delta, "finish_reason": finish })
        } else {
            json!({ "index": 0, "text": text, "logprobs": null, "finish_reason": finish })
        };
        out.push(mk(vec![choice], None));
    }
    if include_usage {
        if let Some(u) = normalize_usage(v) {
            out.push(mk(vec![], Some(u)));
        }
    }
    out
}

/// Corpo JSON do pedido. Lê os bytes (sem exigir `Content-Type: application/json`, que alguns clientes omitem) e responde
/// erro de JSON no formato da OpenAI (o extrator padrão do axum responderia texto puro, que os editores mostram como "erro desconhecido").
fn parse_body(raw: &[u8]) -> Result<Value, Response> {
    match serde_json::from_slice::<Value>(raw) {
        Ok(v) if v.is_object() => Ok(v),
        Ok(_) => Err(oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", "O corpo deve ser um objeto JSON.")),
        Err(e) => Err(oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("JSON inválido: {e}"))),
    }
}

// -------------------------------------------------------------- endpoints

async fn chat_completions(State(st): State<Shared>, raw: Bytes) -> Response {
    let body = match parse_body(&raw) {
        Ok(b) => b,
        Err(r) => return r,
    };
    if let Some(m) = unsupported(&body, true) {
        return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", m);
    }
    // com ferramentas (D79) as mensagens são montadas por `tools::plan` (precisa do motor, que diz o formato do modelo)
    let uses_tools = tools::wants_tools(&body);
    let plain = if uses_tools {
        None
    } else {
        match parse_messages(&body) {
            Ok(m) => Some(m),
            Err(e) => return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        }
    };
    // saída restrita (D82): `response_format` com `json_object` ou `json_schema`
    let constraint = match crate::jsonschema::from_response_format(&body) {
        Ok(c) => c,
        Err(e) => return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    if constraint.is_some() && uses_tools {
        return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", "`response_format` cannot be combined with `tools` in this version.");
    }
    let engine = match ready_engine(&st).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    if constraint.is_some() && !engine.supports_constraint() {
        return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", "`response_format` is only available with Buriti's own engine (Settings, Engine); the llama.cpp engine does not apply it here.");
    }
    let (messages, tool) = match plain {
        Some(m) => (m, None),
        None => match tools::plan(&body, &raw, &engine.tool_support()) {
            Ok(p) => {
                let parses = p.parses();
                (p.messages, parses.then_some((p.fmt, p.offered, p.choice)))
            }
            Err(e) => return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        },
    };
    let req = ChatRequest {
        messages,
        temperature: body.get("temperature").and_then(Value::as_f64).map(|v| v as f32),
        max_tokens: parse_max_tokens(&body),
        stop: parse_stop(&body),
        sampling: parse_sampling(&body),
        images: vec![], // `image_url` ainda é recusado nesta API (precisaria de decodificadores de PNG/JPEG; D58)
        constraint,
    };
    let upstream = match engine.chat(req).await {
        Ok(s) => s,
        Err(e) => return engine_failure(e),
    };
    let (id, created, model) = (rand_id("chatcmpl"), unix_now(), model_id(&*st.settings.read().await));
    let (id_c, model_c) = (id.clone(), model.clone());
    let mk = move |choices: Vec<Value>, usage: Option<Value>| {
        let mut c = json!({ "id": id_c, "object": "chat.completion.chunk", "created": created, "model": model_c, "choices": choices });
        if let Some(u) = usage {
            c["usage"] = u;
        }
        c
    };

    if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        let include_usage = wants_stream_usage(&body);
        let first = sse_line(&mk(vec![json!({ "index": 0, "delta": { "role": "assistant", "content": "" }, "finish_reason": null })], None));
        let done = stream::once(async { Bytes::from_static(b"data: [DONE]\n\n") });
        if let Some((fmt, offered, choice)) = tool {
            // com ferramentas: o texto sai ao vivo até o início de uma chamada; a chamada sai inteira, num bloco, no fim
            let mut ts = tools::ToolStream::new(&fmt, &offered);
            let events = sse_events(upstream).flat_map(move |v| stream::iter(tool_stream_chunks(&v, &mut ts, &choice, include_usage, &mk).iter().map(sse_line).collect::<Vec<_>>()));
            return sse_response(stream::once(async move { first }).chain(events).chain(done));
        }
        let events = sse_events(upstream).flat_map(move |v| stream::iter(stream_chunks(&v, true, include_usage, &mk).iter().map(sse_line).collect::<Vec<_>>()));
        let all = stream::once(async move { first }).chain(events).chain(done);
        return sse_response(all);
    }

    let (mut text, mut finish, mut usage) = (String::new(), "stop".to_string(), None);
    let mut events = Box::pin(sse_events(upstream));
    while let Some(v) = events.next().await {
        let (t, f) = chunk_parts(&v, true);
        text.push_str(&t);
        if let Some(f) = f {
            finish = f;
        }
        usage = normalize_usage(&v).or(usage);
    }
    let mut extra = None;
    let message = match &tool {
        Some((fmt, offered, choice)) => {
            let o = tools::parse_output(fmt, &text, offered, &mut || call_id());
            extra = tool_error(&o, choice);
            if o.calls.is_empty() {
                json!({ "role": "assistant", "content": o.content.unwrap_or_default() })
            } else {
                finish = "tool_calls".to_string();
                json!({ "role": "assistant", "content": o.content, "tool_calls": tools::calls_value(&o.calls, false) })
            }
        }
        None => json!({ "role": "assistant", "content": text }),
    };
    let mut out = json!({ "id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{ "index": 0, "message": message, "finish_reason": finish }] });
    if let Some(e) = extra {
        out["buriti_tool_call_error"] = json!(e);
    }
    if let Some(u) = usage {
        out["usage"] = u;
    }
    Json(out).into_response()
}

/// O que há de errado com a resposta de um pedido com ferramentas, para o campo `buriti_tool_call_error` (extensão do
/// Buriti): o texto parecia uma chamada mas não passou na validação, ou `tool_choice` pedia uma chamada e não veio nenhuma.
/// Não há decodificação restrita: o JSON do modelo pode vir inválido e a escolha de `tool_choice` não é imposta.
fn tool_error(o: &tools::Outcome, choice: &tools::ToolChoice) -> Option<String> {
    if let Some(e) = &o.error {
        return Some(e.clone());
    }
    let forced = matches!(choice, tools::ToolChoice::Required | tools::ToolChoice::Named(_));
    (forced && o.calls.is_empty()).then(|| "`tool_choice` pedia uma chamada de função, mas o modelo respondeu só texto (sem decodificação restrita, o Buriti não consegue obrigar).".to_string())
}

/// Blocos de saída (formato da OpenAI) de um evento do motor num pedido com ferramentas: o texto comum vira `delta.content`
/// ao vivo; ao fim, o resto do texto, um bloco com TODAS as `tool_calls` (cada uma com o `arguments` completo) e o bloco
/// final com `finish_reason` `tool_calls` (ou o do motor, se não houve chamada válida, com `buriti_tool_call_error`).
fn tool_stream_chunks(v: &Value, ts: &mut tools::ToolStream, choice: &tools::ToolChoice, include_usage: bool, mk: &dyn Fn(Vec<Value>, Option<Value>) -> Value) -> Vec<Value> {
    let (text, finish) = chunk_parts(v, true);
    let content = |t: String| json!({ "index": 0, "delta": { "content": t }, "finish_reason": null });
    let mut out = Vec::new();
    if !text.is_empty() {
        let live = ts.feed(&text);
        if !live.is_empty() {
            out.push(mk(vec![content(live)], None));
        }
    }
    if let Some(f) = finish {
        let done = ts.finish(&mut || call_id());
        if !done.tail.is_empty() {
            out.push(mk(vec![content(done.tail.clone())], None));
        }
        let called = !done.outcome.calls.is_empty();
        if called {
            out.push(mk(vec![json!({ "index": 0, "delta": { "tool_calls": tools::calls_value(&done.outcome.calls, true) }, "finish_reason": null })], None));
        }
        let mut last = mk(vec![json!({ "index": 0, "delta": {}, "finish_reason": if called { "tool_calls".to_string() } else { f } })], None);
        if let Some(e) = tool_error(&done.outcome, choice) {
            last["buriti_tool_call_error"] = json!(e);
        }
        out.push(last);
    }
    if include_usage {
        if let Some(u) = normalize_usage(v) {
            out.push(mk(vec![], Some(u)));
        }
    }
    out
}

async fn completions(State(st): State<Shared>, headers: HeaderMap, raw: Bytes) -> Response {
    let body = match parse_body(&raw) {
        Ok(b) => b,
        Err(r) => return r,
    };
    if let Some(m) = unsupported(&body, false) {
        return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", m);
    }
    // `prompt`: texto, ou lista de textos (usa o primeiro); listas de tokens não são suportadas
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.first().is_some_and(Value::is_string) => a[0].as_str().unwrap_or("").to_string(),
        _ => return oai_err(StatusCode::BAD_REQUEST, "invalid_request_error", "`prompt` deve ser um texto."),
    };
    let suffix = body.get("suffix").and_then(Value::as_str).map(str::to_string);
    let has_suffix = suffix.is_some();
    let engine = match ready_engine(&st).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    let req = CompletionRequest {
        prompt,
        suffix,
        temperature: body.get("temperature").and_then(Value::as_f64).map(|v| v as f32),
        max_tokens: parse_max_tokens(&body),
        stop: parse_stop(&body),
        sampling: parse_sampling(&body),
        max_prompt_tokens: resolve_prompt_budget(&body, has_suffix),
    };
    let (id, created, model) = (rand_id("cmpl"), unix_now(), model_id(&*st.settings.read().await));
    let mk = move |choices: Vec<Value>, usage: Option<Value>| {
        let mut c = json!({ "id": id, "object": "text_completion", "created": created, "model": model, "choices": choices });
        if let Some(u) = usage {
            c["usage"] = u;
        }
        c
    };
    let stream_out = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    // "o último vence": registra antes de chamar o motor (para valer também enquanto o pedido ainda espera a vez)
    let ticket = (has_suffix && latest_wins_enabled()).then(|| {
        let key = headers.get(header::USER_AGENT).and_then(|h| h.to_str().ok()).unwrap_or("");
        fim_registry().enter(key)
    });
    let upstream = match &ticket {
        Some(t) => tokio::select! {
            r = engine.complete(req) => r,
            _ = t.inflight.cancel.notified() => return superseded_reply(stream_out, &mk),
        },
        None => engine.complete(req).await,
    };
    let upstream = match upstream {
        Ok(s) => s,
        Err(e) => return engine_failure(e),
    };
    let events = track(sse_events(upstream), ticket);

    if stream_out {
        let include_usage = wants_stream_usage(&body);
        let events = events.flat_map(move |v| stream::iter(stream_chunks(&v, false, include_usage, &mk).iter().map(sse_line).collect::<Vec<_>>()));
        return sse_response(events.chain(stream::once(async { Bytes::from_static(b"data: [DONE]\n\n") })));
    }

    let (mut text, mut finish, mut usage) = (String::new(), "stop".to_string(), None);
    let mut events = events;
    while let Some(v) = events.next().await {
        let (t, f) = chunk_parts(&v, false);
        text.push_str(&t);
        if let Some(f) = f {
            finish = f;
        }
        usage = normalize_usage(&v).or(usage);
    }
    Json(mk(vec![json!({ "index": 0, "text": text, "logprobs": null, "finish_reason": finish })], usage)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_compare() {
        assert!(ct_eq("abc", "abc"));
        assert!(!ct_eq("abc", "abd"));
        assert!(!ct_eq("abc", "abcd"));
        assert!(!ct_eq("", "a"));
    }

    #[test]
    fn parses_and_clamps_sampling_options() {
        let o = parse_sampling(&json!({ "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1, "frequency_penalty": 0.5, "presence_penalty": -0.5, "seed": 123 }));
        assert_eq!((o.top_p, o.top_k, o.min_p, o.repeat_penalty, o.seed), (Some(0.9), Some(40), Some(0.05), Some(1.1), Some(123)));
        assert_eq!((o.frequency_penalty, o.presence_penalty), (Some(0.5), Some(-0.5)));
        let o = parse_sampling(&json!({ "top_p": 7, "top_k": -3, "min_p": -1, "repeat_penalty": 0, "frequency_penalty": 9, "presence_penalty": -9, "seed": -1 }));
        assert_eq!((o.top_p, o.top_k, o.min_p, o.repeat_penalty), (Some(1.0), Some(0), Some(0.0), None));
        assert_eq!((o.frequency_penalty, o.presence_penalty), (Some(2.0), Some(-2.0)));
        assert_eq!(o.seed, Some(u64::MAX)); // semente negativa vale como número, não derruba o pedido
        let o = parse_sampling(&json!({ "top_p": null, "temperature": 0.5 }));
        assert!(o.top_p.is_none() && o.seed.is_none());
    }

    #[test]
    fn prompt_budget_default_override_and_clamp() {
        let none = json!({});
        assert_eq!(resolve_prompt_budget(&none, true), Some(engine::FIM_DEFAULT_PROMPT_TOKENS));
        assert_eq!(resolve_prompt_budget(&none, false), None); // sem suffix só vale o contexto
        assert_eq!(resolve_prompt_budget(&json!({ "max_prompt_tokens": 2000 }), true), Some(2000));
        assert_eq!(resolve_prompt_budget(&json!({ "max_prompt_tokens": 2000 }), false), Some(2000));
        assert_eq!(resolve_prompt_budget(&json!({ "max_prompt_tokens": 1 }), true), Some(engine::MIN_PROMPT_TOKENS));
        assert_eq!(resolve_prompt_budget(&json!({ "max_prompt_tokens": 0 }), true), Some(engine::MIN_PROMPT_TOKENS));
        assert_eq!(resolve_prompt_budget(&json!({ "max_prompt_tokens": u64::MAX }), true), Some(engine::MAX_PROMPT_TOKENS));
        // lixo vale como ausente (não derruba o pedido)
        for bad in [json!(null), json!(-5), json!("x"), json!(1.5)] {
            assert_eq!(resolve_prompt_budget(&json!({ "max_prompt_tokens": bad }), true), Some(engine::FIM_DEFAULT_PROMPT_TOKENS));
        }
    }

    fn leaked_registry() -> &'static FimRegistry {
        Box::leak(Box::new(FimRegistry::default()))
    }

    #[test]
    fn latest_wins_only_supersedes_requests_that_have_not_produced() {
        let r = leaked_registry();
        let a = r.enter("editor");
        let b = r.enter("editor");
        assert!(a.inflight.superseded.load(Ordering::SeqCst), "o pedido velho que ainda espera tem de ser substituído");
        assert!(!b.inflight.superseded.load(Ordering::SeqCst));
        // b já produziu texto: um pedido novo NÃO o cancela
        b.inflight.produced.store(true, Ordering::SeqCst);
        let c = r.enter("editor");
        assert!(!b.inflight.superseded.load(Ordering::SeqCst), "quem já respondeu termina");
        assert!(!c.inflight.superseded.load(Ordering::SeqCst));
        // outro cliente (outro User-Agent) nunca é afetado
        let d = r.enter("outro-programa");
        assert!(!c.inflight.superseded.load(Ordering::SeqCst) && !d.inflight.superseded.load(Ordering::SeqCst));
    }

    #[test]
    fn tickets_leave_the_registry_when_dropped() {
        let r = leaked_registry();
        let a = r.enter("x");
        let b = r.enter("x");
        drop(a);
        assert_eq!(r.map.lock().unwrap()["x"].len(), 1);
        drop(b);
        assert!(r.map.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn superseded_stream_ends_even_while_the_engine_is_silent() {
        // o motor ainda não mandou nada (fila ou prefill): o fluxo encerra assim que chega o pedido novo
        let r = leaked_registry();
        let old = track(stream::pending::<Value>(), Some(r.enter("x")));
        let _new = r.enter("x");
        let out = tokio::time::timeout(std::time::Duration::from_secs(2), old.collect::<Vec<_>>()).await.expect("o fluxo substituído deveria encerrar");
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn stream_that_already_produced_text_keeps_going() {
        let r = leaked_registry();
        let t = r.enter("x");
        let inflight = t.inflight.clone();
        let first = json!({ "choices": [{ "index": 0, "text": "ab", "finish_reason": null }] });
        let mut old = track(stream::iter(vec![first]).chain(stream::pending()), Some(t));
        assert!(old.next().await.is_some());
        let _new = r.enter("x");
        assert!(!inflight.superseded.load(Ordering::SeqCst));
        // continua vivo (esperando o motor), não foi encerrado
        assert!(tokio::time::timeout(std::time::Duration::from_millis(100), old.next()).await.is_err());
    }

    #[test]
    fn superseded_reply_is_an_empty_valid_completion() {
        let mk = |choices: Vec<Value>, _u: Option<Value>| json!({ "id": "x", "choices": choices });
        assert_eq!(superseded_reply(false, &mk).status(), StatusCode::OK);
        assert_eq!(superseded_reply(true, &mk).status(), StatusCode::OK);
    }

    #[test]
    fn body_errors_are_openai_shaped() {
        assert!(parse_body(b"{\"a\":1}").is_ok());
        for bad in [&b"not json"[..], b"[1,2]", b"", b"\"x\""] {
            let r = parse_body(bad).unwrap_err();
            assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        }
    }

    fn mk_test(choices: Vec<Value>, usage: Option<Value>) -> Value {
        let mut c = json!({ "id": "x", "choices": choices });
        if let Some(u) = usage {
            c["usage"] = u;
        }
        c
    }

    /// Fluxo do llama-server escrito à mão: texto, bloco com `finish_reason` e, por último, o bloco de `usage` com
    /// `choices: []` (e `timings`, que não deve vazar). Passa pelo mesmo `sse_events` + `stream_chunks` do servidor.
    async fn run_stream(sse: &str, chat: bool, include_usage: bool) -> Vec<Value> {
        let input: engine::ByteStream = Box::pin(stream::iter(vec![Ok::<Bytes, Box<dyn std::error::Error + Send + Sync>>(Bytes::from(sse.to_string()))]));
        sse_events(input).flat_map(|v| stream::iter(stream_chunks(&v, chat, include_usage, &mk_test))).collect().await
    }

    const LLAMA_CHAT: &str = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Ol\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11},\"timings\":{\"predicted_n\":2}}\n\n",
        "data: [DONE]\n\n"
    );

    #[tokio::test]
    async fn usage_chunk_only_with_include_usage() {
        // sem include_usage: nada de usage, mesmo que o motor mande
        let out = run_stream(LLAMA_CHAT, true, false).await;
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|c| c.get("usage").is_none()));
        // com include_usage: bloco extra, depois do finish_reason, com choices vazio e só os 3 campos
        let out = run_stream(LLAMA_CHAT, true, true).await;
        assert_eq!(out.len(), 3);
        assert_eq!(out[1]["choices"][0]["finish_reason"], "stop");
        assert_eq!(out[2]["choices"], json!([]));
        assert_eq!(out[2]["usage"], json!({ "prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11 }));
        assert!(out[2].get("timings").is_none());
    }

    #[tokio::test]
    async fn usage_rides_on_the_final_native_chunk() {
        // motor próprio: finish_reason e usage no MESMO bloco -> dois blocos de saída (finish, depois usage)
        let native = "data: {\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":6,\"total_tokens\":10}}\n\n";
        let out = run_stream(native, false, true).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["choices"][0]["finish_reason"], "length");
        assert!(out[0].get("usage").is_none());
        assert_eq!(out[1]["usage"]["total_tokens"], 10);
        assert_eq!(run_stream(native, false, false).await.len(), 1);
    }

    #[test]
    fn usage_helpers() {
        assert_eq!(normalize_usage(&json!({ "usage": { "prompt_tokens": 3, "completion_tokens": 4 } })), Some(json!({ "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7 })));
        assert!(normalize_usage(&json!({ "usage": null })).is_none());
        assert!(normalize_usage(&json!({ "usage": { "prompt_tokens": "a", "completion_tokens": 1 } })).is_none());
        assert!(normalize_usage(&json!({ "choices": [] })).is_none());
        assert!(wants_stream_usage(&json!({ "stream_options": { "include_usage": true } })));
        assert!(!wants_stream_usage(&json!({ "stream_options": { "include_usage": false } })));
        assert!(!wants_stream_usage(&json!({ "stream_options": null })));
        assert!(!wants_stream_usage(&json!({ "stream": true })));
    }

    #[test]
    fn parses_messages_and_rejects_unsupported() {
        let v = json!({ "messages": [
            { "role": "system", "content": "s" },
            { "role": "user", "content": [{ "type": "text", "text": "a" }, { "type": "text", "text": "b" }] },
            { "role": "assistant", "content": null } ] });
        let m = parse_messages(&v).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m[1].content, "ab");
        assert_eq!(m[2].content, "");
        assert!(parse_messages(&json!({ "messages": [{ "role": "tool", "content": "x" }] })).is_err());
        assert!(parse_messages(&json!({ "messages": [{ "role": "user", "content": [{ "type": "image_url" }] }] })).is_err());
        assert!(parse_messages(&json!({ "messages": [] })).is_err());
        // `tools` vale no chat (D79) e continua recusado em /v1/completions; `functions` (formato antigo) é recusado nos dois
        assert!(unsupported(&json!({ "tools": [{ "type": "function" }] }), true).is_none());
        assert!(unsupported(&json!({ "tools": [{ "type": "function" }] }), false).is_some());
        assert!(unsupported(&json!({ "functions": [{ "name": "f" }] }), true).is_some());
        assert!(unsupported(&json!({ "tools": [], "n": 1 }), false).is_none());
        assert!(unsupported(&json!({ "n": 2 }), true).is_some());
        assert_eq!(parse_stop(&json!({ "stop": "x" })), vec!["x".to_string()]);
        assert_eq!(parse_stop(&json!({ "stop": ["a", "b"] })).len(), 2);
    }

    // ---- ferramentas (D79): o fluxo como o motor próprio o manda, passando pelo mesmo `tool_stream_chunks` do servidor

    fn qwen_fmt() -> tools::ToolFormat {
        tools::ToolFormat::Hermes { qwen3: false, double_braces: true, default_system: None }
    }

    /// Eventos do motor: um por pedaço de texto e o último com `finish_reason` (e `usage`).
    fn engine_events(pieces: &[&str], finish: &str) -> Vec<Value> {
        let mut v: Vec<Value> = pieces.iter().map(|p| json!({ "choices": [{ "index": 0, "delta": { "content": p }, "finish_reason": null }] })).collect();
        v.push(json!({ "choices": [{ "index": 0, "delta": {}, "finish_reason": finish }], "usage": { "prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12 } }));
        v
    }

    fn run_tool_stream(pieces: &[&str], finish: &str, choice: tools::ToolChoice, usage: bool) -> Vec<Value> {
        let mut ts = tools::ToolStream::new(&qwen_fmt(), &["get_weather".to_string()]);
        engine_events(pieces, finish).iter().flat_map(|e| tool_stream_chunks(e, &mut ts, &choice, usage, &mk_test)).collect()
    }

    fn live_content(chunks: &[Value]) -> String {
        chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect()
    }

    #[test]
    fn tool_stream_emits_text_live_then_the_whole_call() {
        let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let text = format!("Vou ver.\n{call}");
        let pieces: Vec<String> = text.chars().map(String::from).collect();
        let pieces: Vec<&str> = pieces.iter().map(String::as_str).collect();
        let out = run_tool_stream(&pieces, "stop", tools::ToolChoice::Auto, true);
        assert_eq!(live_content(&out), "Vou ver.", "o texto antes da chamada sai ao vivo e a marca da chamada nunca vaza");
        // 1 bloco com TODAS as chamadas (arguments inteiro, texto JSON), depois o final, depois o usage
        let n = out.len();
        let calls = &out[n - 3]["choices"][0]["delta"]["tool_calls"];
        assert_eq!(calls.as_array().unwrap().len(), 1);
        assert_eq!((calls[0]["index"].clone(), calls[0]["type"].clone(), calls[0]["function"]["name"].clone()), (json!(0), json!("function"), json!("get_weather")));
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(out[n - 2]["choices"][0]["finish_reason"], "tool_calls");
        assert!(out[n - 2].get("buriti_tool_call_error").is_none());
        assert_eq!(out[n - 1]["choices"], json!([]));
        assert_eq!(out[n - 1]["usage"]["total_tokens"], 12);
        // ninguém mais leva finish_reason, e sem include_usage o bloco de usage não existe
        assert_eq!(out.iter().filter(|c| c["choices"][0]["finish_reason"].is_string()).count(), 1);
        assert_eq!(run_tool_stream(&pieces, "stop", tools::ToolChoice::Auto, false).len(), n - 1);
    }

    #[test]
    fn tool_stream_invalid_json_returns_raw_text_with_a_flag() {
        let out = run_tool_stream(&["Ok.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": "], "length", tools::ToolChoice::Auto, false);
        assert_eq!(live_content(&out), "Ok.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": ");
        let last = out.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "length", "sem chamada válida vale o motivo do motor");
        assert!(last["buriti_tool_call_error"].as_str().unwrap().contains("JSON"));
        assert!(out.iter().all(|c| c["choices"][0]["delta"].get("tool_calls").is_none()), "nunca inventa uma chamada");
        // função que não foi oferecida: também texto cru + aviso
        let out = run_tool_stream(&["<tool_call>{\"name\":\"apagar_tudo\",\"arguments\":{}}</tool_call>"], "stop", tools::ToolChoice::Auto, false);
        assert!(out.last().unwrap()["buriti_tool_call_error"].as_str().unwrap().contains("apagar_tudo"));
        assert!(out.iter().all(|c| c["choices"][0]["delta"].get("tool_calls").is_none()));
    }

    #[test]
    fn tool_choice_required_flags_a_text_only_answer() {
        let required = run_tool_stream(&["Paris faz 20 graus."], "stop", tools::ToolChoice::Required, false);
        assert_eq!(live_content(&required), "Paris faz 20 graus.");
        assert!(required.last().unwrap()["buriti_tool_call_error"].as_str().unwrap().contains("tool_choice"));
        // auto: texto puro é uma resposta normal, sem aviso
        let auto = run_tool_stream(&["Paris faz 20 graus."], "stop", tools::ToolChoice::Auto, false);
        assert!(auto.last().unwrap().get("buriti_tool_call_error").is_none());
        assert_eq!(auto.last().unwrap()["choices"][0]["finish_reason"], "stop");
        let named = tools::Outcome { content: Some("x".into()), calls: vec![], error: None };
        assert!(tool_error(&named, &tools::ToolChoice::Named("f".into())).is_some());
        assert!(tool_error(&named, &tools::ToolChoice::None).is_none());
    }
}
