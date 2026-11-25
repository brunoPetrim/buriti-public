use std::{path::PathBuf, sync::Arc};

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path, Query, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::{
    attach, catalog,
    config::{Paths, Settings},
    download::Downloads,
    engine::{ChatMsg, ChatRequest, Engine},
    gpu,
    history::{now, Chat, HistoryStore, Message},
};

/// Os dois motores disponíveis; só um fica ativo (conforme `Settings::engine_kind`).
pub struct Engines {
    pub llama: Arc<dyn Engine>,
    pub native: Arc<dyn Engine>,
}

impl Engines {
    pub fn get(&self, kind: &str) -> Arc<dyn Engine> {
        if kind == "native" { self.native.clone() } else { self.llama.clone() }
    }
}

pub struct AppState {
    pub paths: Paths,
    pub settings: RwLock<Settings>,
    pub engines: Engines,
    pub history: HistoryStore,
    pub downloads: Arc<Downloads>,
    pub token: String,
    /// Chave estável da API compatível com a OpenAI (`/v1`); ver `openai.rs`.
    pub api_key: RwLock<String>,
    pub port: u16,
}

pub type Shared = Arc<AppState>;
type ApiErr = (StatusCode, String);

/// Motor ativo no momento.
pub async fn active(st: &AppState) -> Arc<dyn Engine> {
    let kind = st.settings.read().await.engine_kind.clone();
    st.engines.get(&kind)
}

fn err500(e: impl ToString) -> ApiErr {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
fn err400(e: impl ToString) -> ApiErr {
    (StatusCode::BAD_REQUEST, e.to_string())
}

pub fn router(state: Shared) -> Router {
    let api = Router::new()
        .route("/settings", get(get_settings).put(put_settings))
        .route("/models", get(list_models))
        .route("/fs", get(fs_list))
        .route("/system", get(system))
        .route("/vision", get(vision_caps))
        .route("/catalog", get(catalog_list))
        .route("/catalog/download", post(catalog_download))
        .route("/engine/download", post(engine_download))
        .route("/downloads", get(downloads_list))
        .route("/downloads/cancel", post(downloads_cancel))
        .route("/engine/status", get(engine_status))
        .route("/engine/start", post(engine_start))
        .route("/engine/stop", post(engine_stop))
        // o limite padrão do axum (2 MB) barraria anexos de texto grandes; o servidor checa o contexto depois
        .route("/chat", post(chat).layer(DefaultBodyLimit::max(attach::CHAT_BODY_MAX)))
        // bytes de PDF vão só para este servidor local; o corpo é lido com teto dentro do handler
        .route("/attach/pdf", post(attach_pdf).layer(DefaultBodyLimit::disable()))
        .route("/apikey", get(apikey_get).post(apikey_regenerate))
        .route("/chats", get(chats_list).post(chat_create))
        .route("/chats/{id}", get(chat_get).put(chat_put).delete(chat_delete))
        .layer(middleware::from_fn_with_state(state.clone(), guard));

    Router::new()
        .route("/", get(index))
        .route("/i18n.js", get(i18n_js))
        .route("/i18n2.js", get(i18n2_js))
        .route("/i18n3.js", get(i18n3_js))
        .route("/i18n4.js", get(i18n4_js))
        .route("/i18n5.js", get(i18n5_js))
        .route("/favicon.ico", get(favicon))
        .route("/images/{name}", get(image))
        .nest("/api", api)
        .nest("/v1", crate::openai::router(state.clone()))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../web/index.html"))
}

fn js(body: &'static str) -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8")], body)
}
async fn i18n_js() -> impl IntoResponse {
    js(include_str!("../web/i18n.js"))
}
async fn i18n2_js() -> impl IntoResponse {
    js(include_str!("../web/i18n2.js"))
}
async fn i18n3_js() -> impl IntoResponse {
    js(include_str!("../web/i18n3.js"))
}

async fn i18n4_js() -> impl IntoResponse {
    js(include_str!("../web/i18n4.js"))
}

async fn i18n5_js() -> impl IntoResponse {
    js(include_str!("../web/i18n5.js"))
}

// Imagens embutidas no binário (um único executável). Lista fixa: nenhum caminho vem do cliente.
const LOGO: &[u8] = include_bytes!("../web/images/buriti_logo.png");
const LOGO_DARK: &[u8] = include_bytes!("../web/images/buriti_logo_dark_mode.png");
const ICON: &[u8] = include_bytes!("../web/images/buriti_icon.png");
const FAVICON: &[u8] = include_bytes!("../web/images/buriti_favicon.ico");

fn asset(content_type: &'static str, bytes: &'static [u8]) -> Response {
    ([(header::CONTENT_TYPE, content_type), (header::CACHE_CONTROL, "public, max-age=86400")], bytes).into_response()
}

async fn favicon() -> Response {
    asset("image/x-icon", FAVICON)
}

async fn image(Path(name): Path<String>) -> Response {
    match name.as_str() {
        "buriti_logo.png" => asset("image/png", LOGO),
        "buriti_logo_dark_mode.png" => asset("image/png", LOGO_DARK),
        "buriti_icon.png" => asset("image/png", ICON),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Protege a API: Host precisa ser localhost (contra DNS rebinding) e o token
/// de sessão precisa bater (contra outros sites do navegador chamando a API).
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
    let token_ok = req
        .headers()
        .get("x-buriti-token")
        .and_then(|h| h.to_str().ok())
        .map(|t| t == st.token)
        .unwrap_or(false);
    if !host_ok || !token_ok {
        return (StatusCode::UNAUTHORIZED, "não autorizado").into_response();
    }
    next.run(req).await
}

// ------------------------------------------------------------- configurações

async fn get_settings(State(st): State<Shared>) -> Json<Settings> {
    Json(st.settings.read().await.clone())
}

async fn put_settings(State(st): State<Shared>, Json(new): Json<Settings>) -> Result<Json<Value>, ApiErr> {
    let new = new.sanitize().map_err(err400)?;
    st.paths.save_settings(&new).map_err(err500)?;
    let old = std::mem::replace(&mut *st.settings.write().await, new.clone());

    // Se o motor estava ativo, reinicia com a nova configuração (trocando de motor, se for o caso).
    let old_engine = st.engines.get(&old.engine_kind);
    let new_engine = st.engines.get(&new.engine_kind);
    let was_active = old_engine.status().await.state != "stopped";
    if old.engine_kind != new.engine_kind {
        old_engine.stop().await;
    }
    let mut restarted = false;
    let mut message = None;
    if was_active {
        match new_engine.start(&new).await {
            Ok(()) => restarted = true,
            Err(e) => message = Some(e),
        }
    }
    Ok(Json(json!({ "settings": new, "restarted": restarted, "message": message })))
}

async fn list_models(State(st): State<Shared>) -> Result<Json<Value>, ApiErr> {
    let mut models = Vec::new();
    let mut rd = tokio::fs::read_dir(&st.paths.models).await.map_err(err500)?;
    while let Some(e) = rd.next_entry().await.map_err(err500)? {
        let name = e.file_name().to_string_lossy().to_string();
        if name.to_ascii_lowercase().ends_with(".gguf") {
            let size = e.metadata().await.map(|m| m.len()).unwrap_or(0);
            models.push(json!({ "name": name, "size": size }));
        }
    }
    models.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Ok(Json(json!({ "dir": st.paths.models.display().to_string(), "models": models })))
}

async fn system(State(st): State<Shared>) -> Json<Value> {
    let (legacy, adapters) = tokio::join!(gpu::detect(), gpu::discover_isolated());
    let gpus = gpu::merge(legacy, adapters);
    let bin = st.paths.engine_binary();
    let engine_name = active(&st).await.name();
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    Json(json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "windows": cfg!(windows),
        "cpu_threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "ram_mb": sys.total_memory() / 1_048_576,
        "gpus": gpus,
        "gpu_adapters": adapters.adapters,
        "gpu_note": adapters.reason,
        "engine_name": engine_name,
        "engine_found": bin.is_file(),
        "engine_path": bin.display().to_string(),
        "data_dir": st.paths.home.display().to_string(),
        "key_file": st.paths.key_file().display().to_string(),
    }))
}

// ------------------------------------------- navegador de arquivos (sem upload)

#[derive(Deserialize)]
struct FsQuery {
    path: Option<String>,
}

/// Lista pastas e arquivos .gguf do disco local para o usuário apontar um modelo.
/// Nada é lido nem enviado: só nomes. O arquivo escolhido é usado onde está.
async fn fs_list(Query(q): Query<FsQuery>) -> Result<Json<Value>, ApiErr> {
    let path = q.path.unwrap_or_default();
    tokio::task::spawn_blocking(move || fs_list_blocking(&path))
        .await
        .map_err(err500)?
        .map(Json)
        .map_err(err400)
}

fn fs_list_blocking(path: &str) -> Result<Value, String> {
    let mut entries: Vec<Value> = Vec::new();

    if path.is_empty() {
        // Visão inicial: atalhos + raízes (unidades no Windows, "/" nos demais).
        let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from);
        if let Some(h) = &home {
            for sub in ["", "Downloads", "Documents", "Desktop"] {
                let p = if sub.is_empty() { h.clone() } else { h.join(sub) };
                if p.is_dir() {
                    let name = if sub.is_empty() { "~".to_string() } else { sub.to_string() };
                    entries.push(json!({ "name": name, "path": p.display().to_string(), "dir": true }));
                }
            }
        }
        if cfg!(windows) {
            for l in b'A'..=b'Z' {
                let root = format!("{}:\\", l as char);
                if std::path::Path::new(&root).exists() {
                    entries.push(json!({ "name": root.clone(), "path": root, "dir": true }));
                }
            }
        } else {
            entries.push(json!({ "name": "/", "path": "/", "dir": true }));
        }
        return Ok(json!({ "path": "", "parent": null, "entries": entries }));
    }

    let p = std::path::Path::new(path);
    if !p.is_absolute() || !p.is_dir() {
        return Err(crate::errcode::coded("err_bad_folder", &[], "Invalid folder."));
    }
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for e in std::fs::read_dir(p).map_err(|e| e.to_string())?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let Ok(meta) = e.metadata() else { continue };
        let full = e.path().display().to_string();
        if meta.is_dir() {
            dirs.push(json!({ "name": name, "path": full, "dir": true }));
        } else if name.to_ascii_lowercase().ends_with(".gguf") {
            files.push(json!({ "name": name, "path": full, "dir": false, "size": meta.len() }));
        }
    }
    let key = |v: &Value| v["name"].as_str().unwrap_or("").to_lowercase();
    dirs.sort_by_key(key);
    files.sort_by_key(key);
    entries.extend(dirs);
    entries.extend(files);
    entries.truncate(3000);
    let parent = p.parent().map(|x| x.display().to_string()).unwrap_or_default(); // "" = voltar às raízes
    Ok(json!({ "path": path, "parent": parent, "entries": entries }))
}

// ------------------------------------------------------- catálogo e downloads

/// O que o motor ativo faz com imagens (`null` = nada); a interface só deixa preparar a imagem se houver visão.
async fn vision_caps(State(st): State<Shared>) -> Json<Value> {
    match active(&st).await.vision() {
        Some(c) => Json(json!({ "tile": c.tile, "per_tile": c.per_tile, "max_grid": c.max_grid })),
        None => Json(Value::Null),
    }
}

async fn catalog_list(State(st): State<Shared>) -> Json<Value> {
    let entries: Vec<Value> = catalog::CATALOG
        .iter()
        .map(|e| {
            json!({
                "id": e.id, "name": e.name, "desc": e.desc, "file": e.file,
                "size": e.size, "min_ram_gb": e.min_ram_gb, "native": e.native,
                "provider": e.provider, "released": e.released, "params": e.params, "ctx_k": e.ctx_k,
                "license": e.license, "non_commercial": e.non_commercial, "langs": e.langs, "english_best": e.english_best, "thinking": e.thinking, "coding": e.coding,
                "installed": st.paths.models.join(e.file).is_file() && catalog::mmproj_for(e.id).is_none_or(|_| st.paths.models.join(catalog::mmproj_file(e)).is_file()),
                "vision": catalog::mmproj_for(e.id).is_some(),
            })
        })
        .collect();
    Json(json!({ "entries": entries }))
}

#[derive(Deserialize)]
struct IdBody {
    id: String,
}
#[derive(Deserialize)]
struct BackendBody {
    backend: String,
}

async fn catalog_download(State(st): State<Shared>, Json(b): Json<IdBody>) -> Result<Json<Value>, ApiErr> {
    let entry = catalog::find(&b.id).ok_or((StatusCode::NOT_FOUND, "modelo desconhecido".to_string()))?;
    st.downloads.start_model(st.paths.clone(), entry).map_err(err400)?;
    Ok(Json(json!({ "ok": true })))
}

async fn engine_download(State(st): State<Shared>, Json(b): Json<BackendBody>) -> Result<Json<Value>, ApiErr> {
    st.downloads.start_engine(st.paths.clone(), b.backend).map_err(err400)?;
    Ok(Json(json!({ "ok": true })))
}

async fn downloads_list(State(st): State<Shared>) -> Json<Value> {
    Json(json!(st.downloads.list()))
}

async fn downloads_cancel(State(st): State<Shared>, Json(b): Json<IdBody>) -> Json<Value> {
    st.downloads.cancel(&b.id);
    Json(json!({ "ok": true }))
}

// --------------------------------------------------------------------- motor

async fn engine_status(State(st): State<Shared>) -> Json<Value> {
    Json(json!(active(&st).await.status().await))
}

async fn engine_start(State(st): State<Shared>) -> Result<Json<Value>, ApiErr> {
    let s = st.settings.read().await.clone();
    // libera os recursos do outro motor antes de subir este
    let other = if s.engine_kind == "native" { &st.engines.llama } else { &st.engines.native };
    other.stop().await;
    st.engines.get(&s.engine_kind).start(&s).await.map_err(err400)?;
    Ok(Json(json!({ "ok": true })))
}

async fn engine_stop(State(st): State<Shared>) -> Json<Value> {
    st.engines.llama.stop().await;
    st.engines.native.stop().await;
    Json(json!({ "ok": true }))
}

/// Encaminha o chat ao motor ativo com streaming (SSE), repassando só campos conhecidos.
async fn chat(State(st): State<Shared>, Json(body): Json<Value>) -> Result<Response, ApiErr> {
    let engine = active(&st).await;
    if engine.status().await.state != "ready" {
        return Err((StatusCode::CONFLICT, crate::errcode::coded("err_engine_not_ready", &[], "The engine is not ready.")));
    }
    let messages: Vec<ChatMsg> = serde_json::from_value(body.get("messages").cloned().unwrap_or(json!([])))
        .map_err(|e| err400(crate::errcode::coded("err_bad_messages", &[&e.to_string()], format!("invalid messages: {e}"))))?;
    if messages.is_empty() || messages.iter().any(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant")) {
        return Err(err400(crate::errcode::coded("err_empty_messages", &[], "empty messages or a message with an invalid role")));
    }
    // rede de segurança: anexos grandes não podem estourar o contexto configurado
    attach::check_context(&messages, st.settings.read().await.ctx_size).map_err(err400)?;
    let images = crate::image::parse_request_images(body.get("messages").unwrap_or(&Value::Null), engine.vision()).map_err(err400)?;
    let req = ChatRequest {
        messages,
        temperature: body.get("temperature").and_then(Value::as_f64).map(|v| v as f32),
        max_tokens: body.get("max_tokens").and_then(Value::as_u64).map(|v| v as u32),
        stop: vec![],
        sampling: Default::default(),
        images,
        constraint: None,
    };
    let stream = engine.chat(req).await.map_err(|e| (StatusCode::BAD_GATEWAY, e))?;
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .map_err(err500)
}

// ------------------------------------------------------- anexos (PDF → texto)

/// Recebe os bytes de um PDF (corpo cru, até 5 MB) e devolve o texto extraído. Nada é gravado em disco.
async fn attach_pdf(body: Body) -> Result<Json<Value>, ApiErr> {
    let bytes = axum::body::to_bytes(body, attach::MAX_FILE_BYTES + 1).await.map_err(|_| err400(attach::too_big()))?;
    // a extração é pesada em CPU: fora do executor assíncrono
    let r = tokio::task::spawn_blocking(move || attach::process_pdf(&bytes))
        .await
        .map_err(|_| err500("Não foi possível ler o PDF: falha interna na extração"))?;
    r.map(|t| Json(json!(t))).map_err(err400)
}

// ------------------------------------------------ chave da API (/v1)

/// Mostra a chave da API para a tela "API" (protegido pelo token de sessão, como o resto de /api).
async fn apikey_get(State(st): State<Shared>) -> Json<Value> {
    Json(json!({ "key": st.api_key.read().await.clone(), "port": st.port }))
}

/// Gera uma chave nova; a antiga deixa de valer na hora (os editores precisam da nova).
async fn apikey_regenerate(State(st): State<Shared>) -> Result<Json<Value>, ApiErr> {
    let key = st.paths.new_api_key().map_err(err500)?;
    *st.api_key.write().await = key.clone();
    Ok(Json(json!({ "key": key, "port": st.port })))
}

// ------------------------------------------------- histórico (criptografado)

async fn chats_list(State(st): State<Shared>) -> Json<Value> {
    Json(json!(st.history.list()))
}

async fn chat_create(State(st): State<Shared>) -> Result<Json<Value>, ApiErr> {
    let chat = Chat { id: HistoryStore::new_id(), title: String::new(), created: now(), updated: now(), messages: vec![] };
    st.history.save(&chat).map_err(err500)?;
    Ok(Json(json!(chat)))
}

async fn chat_get(State(st): State<Shared>, Path(id): Path<String>) -> Result<Json<Chat>, ApiErr> {
    st.history.load(&id).map(Json).map_err(|e| (StatusCode::NOT_FOUND, e))
}

#[derive(Deserialize)]
struct ChatBody {
    title: String,
    messages: Vec<Message>,
}

async fn chat_put(State(st): State<Shared>, Path(id): Path<String>, Json(b): Json<ChatBody>) -> Result<Json<Value>, ApiErr> {
    let created = st.history.load(&id).map(|c| c.created).unwrap_or_else(|_| now());
    let mut messages = b.messages;
    for m in &mut messages {
        m.sanitize(); // só metadados de anexo (nome, tamanho, tipo); nunca conteúdo
    }
    let chat = Chat { id, title: b.title.chars().take(80).collect(), created, updated: now(), messages };
    st.history.save(&chat).map_err(err400)?;
    Ok(Json(json!({ "ok": true })))
}

async fn chat_delete(State(st): State<Shared>, Path(id): Path<String>) -> Result<Json<Value>, ApiErr> {
    st.history.delete(&id).map_err(err400)?;
    Ok(Json(json!({ "ok": true })))
}

#[cfg(test)]
#[path = "cancel_tests.rs"]
mod cancel_tests;
