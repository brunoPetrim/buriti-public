use std::{
    process::Stdio,
    sync::atomic::{AtomicU16, Ordering},
    time::Duration,
};

use async_trait::async_trait;
use axum::body::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::{process::Child, process::Command, sync::Mutex};

use super::{sse_events, trim_fim_by_chars, ByteStream, ChatRequest, CompletionRequest, Engine, EngineStatus};
use crate::config::{Paths, Settings};

/// Motor da fase 1: gerencia um processo `llama-server` (llama.cpp) em 127.0.0.1.
pub struct LlamaServerEngine {
    paths: Paths,
    child: Mutex<Option<Child>>,
    last_error: Mutex<Option<String>>,
    port: AtomicU16,
    /// cliente com timeout curto (health check)
    http: reqwest::Client,
    /// cliente sem timeout total (streaming do chat)
    stream_http: reqwest::Client,
}

impl LlamaServerEngine {
    pub fn new(paths: Paths, port: u16) -> Self {
        Self {
            paths,
            child: Mutex::new(None),
            last_error: Mutex::new(None),
            port: AtomicU16::new(port),
            http: reqwest::Client::builder().timeout(Duration::from_millis(800)).build().expect("cliente http"),
            stream_http: reqwest::Client::builder().connect_timeout(Duration::from_secs(3)).build().expect("cliente http"),
        }
    }

    fn port(&self) -> u16 {
        self.port.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl Engine for LlamaServerEngine {
    fn name(&self) -> &'static str {
        "llama.cpp (llama-server)"
    }

    async fn start(&self, s: &Settings) -> Result<(), String> {
        self.stop().await;
        *self.last_error.lock().await = None;

        let bin = self.paths.engine_binary();
        if !bin.is_file() {
            return Err(crate::errcode::coded("err_engine_missing", &[&bin.display().to_string()], format!("Engine not found: {}", bin.display())));
        }
        let model = s.model_file(&self.paths)?;

        let log_path = self.paths.home.join("engine.log");
        let log = std::fs::File::create(&log_path).map_err(|e| format!("engine.log: {e}"))?;
        let log_err = log.try_clone().map_err(|e| format!("engine.log: {e}"))?;

        // porta interna: se a configurada estiver ocupada, usa outra livre (só este programa a usa)
        let engine_port = crate::config::free_local_port(s.engine_port);
        if engine_port != s.engine_port {
            eprintln!("[llama] porta {} ocupada; o llama-server usa a {engine_port}", s.engine_port);
        }
        let mut cmd = Command::new(&bin);
        cmd.arg("--host").arg("127.0.0.1")
            .arg("--port").arg(engine_port.to_string())
            .arg("-m").arg(&model)
            .arg("-c").arg(s.ctx_size.to_string())
            .arg("-ngl").arg(s.gpu_layers.to_string());
        if s.threads > 0 {
            cmd.arg("-t").arg(s.threads.to_string());
        }
        if let Some(dir) = bin.parent() {
            if !dir.as_os_str().is_empty() {
                cmd.current_dir(dir); // DLLs/libs ao lado do binário
            }
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err))
            .kill_on_drop(true);
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

        let child = cmd.spawn().map_err(|e| crate::errcode::coded("err_engine_start", &[&e.to_string()], format!("Failed to start the engine: {e}")))?;
        self.port.store(engine_port, Ordering::Relaxed);
        *self.child.lock().await = Some(child);
        Ok(())
    }

    async fn stop(&self) {
        let child = self.child.lock().await.take();
        if let Some(mut c) = child {
            let _ = c.kill().await;
        }
    }

    async fn status(&self) -> EngineStatus {
        let waited = {
            let mut guard = self.child.lock().await;
            match guard.as_mut() {
                None => None,
                Some(c) => {
                    let r = c.try_wait();
                    if matches!(r, Ok(Some(_))) {
                        *guard = None;
                    }
                    Some(r)
                }
            }
        };

        match waited {
            None => {
                let err = self.last_error.lock().await.clone();
                match err {
                    Some(m) => EngineStatus { state: "error", message: Some(m) },
                    None => EngineStatus { state: "stopped", message: None },
                }
            }
            Some(Ok(Some(code))) => {
                let m = crate::errcode::coded("err_engine_exit", &[&code.to_string()], format!("The engine exited ({code}). See engine.log"));
                *self.last_error.lock().await = Some(m.clone());
                EngineStatus { state: "error", message: Some(m) }
            }
            Some(Err(e)) => EngineStatus { state: "error", message: Some(e.to_string()) },
            Some(Ok(None)) => {
                let url = format!("http://127.0.0.1:{}/health", self.port());
                let ok = matches!(self.http.get(url).send().await, Ok(r) if r.status().is_success());
                EngineStatus { state: if ok { "ready" } else { "starting" }, message: None }
            }
        }
    }

    async fn chat(&self, req: ChatRequest) -> Result<ByteStream, String> {
        // `include_usage`: o llama-server manda a contagem de tokens num bloco final (`usage`), que o openai.rs repassa
        let mut payload = json!({ "messages": req.messages, "stream": true, "stream_options": { "include_usage": true } });
        if let Some(t) = req.temperature {
            payload["temperature"] = json!(t);
        }
        if let Some(m) = req.max_tokens {
            payload["max_tokens"] = json!(m);
        }
        if !req.stop.is_empty() {
            payload["stop"] = json!(req.stop);
        }
        req.sampling.add_to(&mut payload);
        self.post_stream("/v1/chat/completions", payload).await
    }

    async fn complete(&self, req: CompletionRequest) -> Result<ByteStream, String> {
        let temperature = req.temperature.unwrap_or(0.2);
        let max = req.max_tokens.unwrap_or(256);
        // Orçamento de prompt (mesma política do motor próprio, `trim_fim_prompt`), mas em CARACTERES: quem tokeniza é o
        // llama-server (aproximação de `CHARS_PER_TOKEN_ESTIMATE` caracteres por token; o prompt real fica um pouco abaixo).
        let (prompt, suffix) = match req.max_prompt_tokens {
            Some(budget) => {
                let (p, s) = trim_fim_by_chars(&req.prompt, req.suffix.as_deref().unwrap_or(""), budget);
                (p.to_string(), req.suffix.as_ref().map(|_| s.to_string()))
            }
            None => (req.prompt.clone(), req.suffix.clone()),
        };
        match suffix {
            // preenchimento no meio: o /infill do llama.cpp monta o prompt FIM do modelo (precisa dos tokens FIM no vocabulário)
            Some(suffix) => {
                let mut payload = json!({
                    "input_prefix": prompt, "input_suffix": suffix,
                    "n_predict": max, "temperature": temperature, "stream": true,
                });
                if !req.stop.is_empty() {
                    payload["stop"] = json!(req.stop);
                }
                req.sampling.add_to(&mut payload);
                let upstream = self.post_stream("/infill", payload).await?;
                // traduz `data: {"content": "...", "stop": bool}` para o formato `text_completion` da OpenAI
                let out = sse_events(upstream)
                    .map(|v| {
                        let chunk = infill_to_completion(&v);
                        Ok::<Bytes, Box<dyn std::error::Error + Send + Sync>>(Bytes::from(format!("data: {chunk}\n\n")))
                    })
                    .chain(futures_util::stream::once(async { Ok(Bytes::from_static(b"data: [DONE]\n\n")) }));
                Ok(Box::pin(out))
            }
            None => {
                let mut payload = json!({ "prompt": prompt, "stream": true, "max_tokens": max, "temperature": temperature, "stream_options": { "include_usage": true } });
                if !req.stop.is_empty() {
                    payload["stop"] = json!(req.stop);
                }
                req.sampling.add_to(&mut payload);
                self.post_stream("/v1/completions", payload).await
            }
        }
    }
}

impl LlamaServerEngine {
    /// POST ao llama-server local e devolve o corpo como fluxo de bytes.
    async fn post_stream(&self, path: &str, payload: Value) -> Result<ByteStream, String> {
        let url = format!("http://127.0.0.1:{}{}", self.port(), path);
        let resp = self.stream_http.post(url).json(&payload).send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(crate::errcode::coded("err_engine_http", &[&code.to_string(), &text.to_string()], format!("the engine answered {code}: {text}")));
        }
        Ok(Box::pin(resp.bytes_stream().map(|r| r.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>))))
    }
}

/// Traduz um bloco do `/infill` (`{"content", "stop", "stop_type", "tokens_evaluated", "tokens_predicted"}`) para o formato
/// `text_completion` da OpenAI. No bloco final, as contagens do llama.cpp viram `usage` (se vierem).
fn infill_to_completion(v: &Value) -> Value {
    let text = v.get("content").and_then(|c| c.as_str()).unwrap_or("");
    let stop = v.get("stop").and_then(|s| s.as_bool()).unwrap_or(false);
    let finish = if !stop {
        Value::Null
    } else if v.get("stop_type").and_then(|s| s.as_str()) == Some("limit") {
        json!("length")
    } else {
        json!("stop")
    };
    let mut chunk = json!({ "object": "text_completion", "choices": [{ "index": 0, "text": text, "finish_reason": finish }] });
    if stop {
        if let (Some(p), Some(c)) = (v.get("tokens_evaluated").and_then(Value::as_u64), v.get("tokens_predicted").and_then(Value::as_u64)) {
            chunk["usage"] = json!({ "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c });
        }
    }
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infill_chunks_become_text_completion_with_usage_at_the_end() {
        let mid = infill_to_completion(&json!({ "content": "ab", "stop": false, "tokens_evaluated": 7, "tokens_predicted": 1 }));
        assert_eq!(mid["choices"][0]["text"], "ab");
        assert!(mid["choices"][0]["finish_reason"].is_null());
        assert!(mid.get("usage").is_none()); // só o bloco final leva a contagem
        let end = infill_to_completion(&json!({ "content": "", "stop": true, "stop_type": "limit", "tokens_evaluated": 7, "tokens_predicted": 3 }));
        assert_eq!(end["choices"][0]["finish_reason"], "length");
        assert_eq!(end["usage"], json!({ "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10 }));
        let eos = infill_to_completion(&json!({ "content": "", "stop": true, "stop_type": "eos" }));
        assert_eq!(eos["choices"][0]["finish_reason"], "stop");
        assert!(eos.get("usage").is_none());
    }
}
