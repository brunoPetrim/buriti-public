use std::{error::Error, pin::Pin};

use async_trait::async_trait;
use axum::body::Bytes;
use futures_util::{stream, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Settings;

pub mod llama_server;

#[derive(Serialize, Clone)]
pub struct EngineStatus {
    /// "stopped" | "starting" | "ready" | "error"
    pub state: &'static str,
    pub message: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

/// Parâmetros de amostragem opcionais do pedido; `None` = o padrão do motor. `top_k`, `min_p` e `repeat_penalty`
/// são extensões do llama.cpp (a OpenAI só define `top_p`, `seed` e as penalidades de frequência e presença).
#[derive(Clone, Debug, Default)]
pub struct SamplingOpts {
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
}

pub struct ChatRequest {
    pub messages: Vec<ChatMsg>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Textos que encerram a geração (o texto de parada não é devolvido).
    pub stop: Vec<String>,
    pub sampling: SamplingOpts,
    /// Imagens das mensagens de usuário (já validadas em `image::parse_request_images`); só o motor próprio as usa.
    pub images: Vec<crate::image::ChatImage>,
    /// Gramática que restringe o texto gerado (`response_format`, D82): só o motor próprio a aplica.
    pub constraint: Option<std::sync::Arc<crate::grammar::Grammar>>,
}

/// Pedido de completar texto cru (`/v1/completions`), com ou sem sufixo (preenchimento no meio, FIM).
pub struct CompletionRequest {
    pub prompt: String,
    pub suffix: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub sampling: SamplingOpts,
    /// Orçamento de tokens do prompt (prefixo + sufixo + marcadores), já resolvido por `openai.rs`: com `suffix` e sem
    /// valor no pedido vale `FIM_DEFAULT_PROMPT_TOKENS`; sem `suffix` fica `None` (só o limite do contexto).
    pub max_prompt_tokens: Option<usize>,
}

// ------------------------------------------------- orçamento de prompt do FIM

/// Orçamento padrão (tokens) do prompt de um completar com `suffix`. Em autocompletar, o prefixo inteiro e o sufixo
/// inteiro do arquivo são recalculados a cada tecla (o KV cache só vale até o ponto da edição): na CPU o prefill é
/// ~25 a 37 tok/s (D44), então 1024 tokens ≈ 30 a 40 s a frio e bem menos com o prefixo reaproveitado. O que está perto
/// do cursor é o que mais importa; o resto do arquivo só custa tempo.
pub const FIM_DEFAULT_PROMPT_TOKENS: usize = 1024;
/// Faixa aceita para `max_prompt_tokens` no pedido (fora dela, é limitado).
pub const MIN_PROMPT_TOKENS: usize = 64;
pub const MAX_PROMPT_TOKENS: usize = 262_144;
/// O llama-server é quem tokeniza, então no caminho do llama.cpp o orçamento é uma ESTIMATIVA em caracteres: 3 caracteres
/// por token é conservador para código (o comum é 3 a 4), ou seja, o prompt real tende a ficar um pouco abaixo do orçamento.
pub const CHARS_PER_TOKEN_ESTIMATE: usize = 3;

/// Corta o prefixo (a partir do COMEÇO) e o sufixo (a partir do FIM), em linhas inteiras, até `size(prefixo, sufixo)`
/// (tokens do prompt montado) caber em `budget`. Devolve os trechos que ficam, ou `None` se `size` não consegue montar o
/// prompt (`None`) ou nem vazio cabe. Determinístico e estável:
/// - os cortes caem numa grade ABSOLUTA de `grade` bytes (a 1ª linha inteira a partir de `k * grade` no prefixo; o
///   último fim de linha até `m * grade` no sufixo), então enquanto o usuário digita o começo do prefixo NÃO muda a cada
///   tecla (um corte relativo ao comprimento mudaria sempre e invalidaria o KV cache inteiro); ele só avança quando o
///   prompt passa do orçamento, e então de uma grade (~1/8 do orçamento), dando folga até o próximo corte;
/// - a cada passo encolhe o lado MAIS LONGO, então o cursor mantém contexto dos dois lados;
/// - antes do laço, salta direto até ~8 bytes por token de orçamento por lado (arquivo de MB não custa milhares de
///   tokenizações). Um arquivo com mais de 8 bytes por token (indentação enorme) perde um pouco de contexto à toa.
pub fn trim_fim_prompt<'a>(prefix: &'a str, suffix: &'a str, budget: usize, mut size: impl FnMut(&str, &str) -> Option<usize>) -> Option<(&'a str, &'a str)> {
    let grade = (budget.saturating_mul(4) / 8).max(256);
    let cap = budget.saturating_mul(8);
    // número de passos do prefixo (k) e do sufixo (m: múltiplos de `grade` mantidos)
    let mut k = prefix.len().saturating_sub(cap) / grade;
    let mut m = if suffix.len() > cap { cap / grade } else { suffix.len().div_ceil(grade) };
    // Limite de passos (defesa): cada passo avança `k` (o corte do prefixo anda ao menos `grade` bytes) ou recua `m`, então
    // (P + S) / grade + folga passos esvaziam os dois lados; passar disso é defeito dos cortes e desiste em vez de girar.
    let mut passos_restantes = (prefix.len() + suffix.len()) / grade + 16;
    loop {
        let (pre, suf) = (&prefix[prefix_cut(prefix, k, grade)..], &suffix[..suffix_keep(suffix, m, grade)]);
        if size(pre, suf)? <= budget {
            return Some((pre, suf));
        }
        if (pre.is_empty() && suf.is_empty()) || passos_restantes == 0 {
            return None;
        }
        passos_restantes -= 1;
        if pre.len() >= suf.len() {
            k += 1;
        } else {
            m = m.saturating_sub(1);
        }
    }
}

/// Início do prefixo no passo `k`: a 1ª linha inteira a partir de `k * grade` (sem linha depois, corta nesse ponto).
fn prefix_cut(s: &str, k: usize, grade: usize) -> usize {
    if k == 0 {
        return 0;
    }
    let mut c = k.saturating_mul(grade);
    if c >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(c) {
        c += 1;
    }
    s[c..].find('\n').map(|p| c + p + 1).unwrap_or(c)
}

/// Fim do sufixo mantido com `m` unidades de `grade`: o último fim de linha (inclusive) até esse ponto; sem nenhum, 0.
fn suffix_keep(s: &str, m: usize, grade: usize) -> usize {
    let mut e = m.saturating_mul(grade);
    if e >= s.len() {
        return s.len();
    }
    while !s.is_char_boundary(e) {
        e -= 1;
    }
    s[..e].rfind('\n').map(|p| p + 1).unwrap_or(0)
}

/// Tokens estimados de um prompt de `prefixo` + `sufixo` (o llama-server tokeniza, não temos o vocabulário aqui).
pub fn approx_prompt_tokens(prefix: &str, suffix: &str) -> usize {
    (prefix.chars().count() + suffix.chars().count()).div_ceil(CHARS_PER_TOKEN_ESTIMATE)
}

/// `trim_fim_prompt` com a estimativa em caracteres (caminho do llama.cpp).
pub fn trim_fim_by_chars<'a>(prefix: &'a str, suffix: &'a str, budget: usize) -> (&'a str, &'a str) {
    trim_fim_prompt(prefix, suffix, budget, |p, s| Some(approx_prompt_tokens(p, s))).unwrap_or(("", ""))
}

impl SamplingOpts {
    /// Campos extras para o corpo JSON enviado ao llama-server (que aceita os mesmos nomes).
    pub fn add_to(&self, payload: &mut serde_json::Value) {
        let mut put = |k: &str, v: Option<serde_json::Value>| {
            if let Some(v) = v {
                payload[k] = v;
            }
        };
        put("top_p", self.top_p.map(|v| v.into()));
        put("top_k", self.top_k.map(|v| v.into()));
        put("min_p", self.min_p.map(|v| v.into()));
        put("repeat_penalty", self.repeat_penalty.map(|v| v.into()));
        put("frequency_penalty", self.frequency_penalty.map(|v| v.into()));
        put("presence_penalty", self.presence_penalty.map(|v| v.into()));
        put("seed", self.seed.map(|v| v.into()));
    }
}

/// Prefixo das mensagens de erro de pedido inválido do usuário (o servidor devolve 400, não 502).
pub const BAD_REQUEST: &str = "FIM:";

/// Fluxo de bytes em formato SSE (compatível com OpenAI): `data: {...}\n\n` ... `data: [DONE]\n\n`.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, Box<dyn Error + Send + Sync>>> + Send>>;

/// Contrato de um motor de inferência. A web e a API só conhecem este trait:
/// o motor próprio (`native`) e o llama.cpp são intercambiáveis.
#[async_trait]
pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    async fn start(&self, settings: &Settings) -> Result<(), String>;
    async fn stop(&self);
    async fn status(&self) -> EngineStatus;
    /// Gera a resposta em streaming (blocos `chat.completion.chunk`).
    async fn chat(&self, req: ChatRequest) -> Result<ByteStream, String>;
    /// Completa texto cru em streaming (blocos `text_completion`, texto em `choices[0].text`).
    async fn complete(&self, req: CompletionRequest) -> Result<ByteStream, String>;
    /// O que o motor sabe fazer com imagens (`None` = nada; é o caso do llama.cpp nesta versão).
    fn vision(&self) -> Option<crate::image::VisionCaps> {
        None
    }
    /// Chamada de funções (`tools`) no chat (D79): só o motor próprio, e só se o template do modelo carregado tiver ferramentas.
    fn tool_support(&self) -> crate::tools::ToolSupport {
        crate::tools::ToolSupport::llama_cpp()
    }
    /// Saída restrita por gramática (`response_format` com `json_object` ou `json_schema`, D82): só o motor próprio a aplica.
    fn supports_constraint(&self) -> bool {
        false
    }
    /// Embeddings (`/v1/embeddings`) com o modelo carregado. Só o motor próprio os calcula (D77); o llama.cpp recusa.
    async fn embed(&self, _job: crate::embeddings::EmbedJob) -> Result<crate::embeddings::EmbedOutput, String> {
        Err(format!("{} embeddings só estão disponíveis no motor próprio do Buriti (Configurações, Motor); com o llama.cpp esta rota não é usada.", BAD_REQUEST))
    }
}

/// Converte um fluxo SSE (`data: {json}\n\n`) em eventos JSON. Termina em `data: [DONE]` ou no fim do fluxo.
/// Os blocos podem chegar cortados em qualquer ponto: acumula até achar a linha em branco.
pub fn sse_events(input: ByteStream) -> impl Stream<Item = Value> + Send {
    struct St {
        input: ByteStream,
        buf: String,
        raw: Vec<u8>,
        eof: bool,
    }
    let st = St { input, buf: String::new(), raw: Vec::new(), eof: false };
    stream::unfold(st, |mut st| async move {
        loop {
            // um evento completo no buffer?
            let cut = st.buf.find("\n\n").map(|p| (p, 2)).or_else(|| st.buf.find("\r\n\r\n").map(|p| (p, 4)));
            let block = match cut {
                Some((p, n)) => {
                    let b = st.buf[..p].to_string();
                    st.buf.drain(..p + n);
                    Some(b)
                }
                None if st.eof && !st.buf.trim().is_empty() => Some(std::mem::take(&mut st.buf)),
                None => None,
            };
            if let Some(block) = block {
                for line in block.lines() {
                    let Some(data) = line.strip_prefix("data:") else { continue };
                    let data = data.trim();
                    if data == "[DONE]" {
                        return None;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        return Some((v, st));
                    }
                }
                continue;
            }
            if st.eof {
                return None;
            }
            match st.input.next().await {
                Some(Ok(b)) => {
                    st.raw.extend_from_slice(&b);
                    // só converte o que forma UTF-8 completo (um caractere pode vir partido entre blocos)
                    let valid = match std::str::from_utf8(&st.raw) {
                        Ok(s) => s.len(),
                        Err(e) => e.valid_up_to(),
                    };
                    let s = String::from_utf8_lossy(&st.raw[..valid]).into_owned();
                    st.buf.push_str(&s);
                    st.raw.drain(..valid);
                }
                _ => st.eof = true,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bs(parts: &[&str]) -> ByteStream {
        let v: Vec<Result<Bytes, Box<dyn Error + Send + Sync>>> = parts.iter().map(|p| Ok(Bytes::from(p.as_bytes().to_vec()))).collect();
        Box::pin(stream::iter(v))
    }

    /// "Tokenizador" de mentira: 4 bytes por token + 8 de marcadores.
    fn fake_size(p: &str, s: &str) -> Option<usize> {
        Some((p.len() + s.len()) / 4 + 8)
    }

    fn lines(tag: &str, n: usize) -> String {
        (0..n).map(|i| format!("{tag} {i:04}\n")).collect()
    }

    #[test]
    fn trim_leaves_a_small_prompt_alone() {
        let (p, s) = ("fn a() {\n", "}\n");
        assert_eq!(trim_fim_prompt(p, s, 1024, fake_size), Some((p, s)));
    }

    #[test]
    fn trim_cuts_the_far_ends_in_whole_lines_and_fits() {
        let pre = format!("{}cur", lines("antes", 2000)); // ~20 KB, o cursor termina numa linha incompleta
        let suf = lines("depois", 2000);
        let (p, s) = trim_fim_prompt(&pre, &suf, 1024, fake_size).unwrap();
        let used = fake_size(p, s).unwrap();
        assert!(used <= 1024, "{used}");
        assert!(used > 512, "cortou demais: {used}"); // não joga fora metade do orçamento
        assert!(pre.ends_with(p) && p.ends_with("cur"), "o que fica do prefixo é o trecho perto do cursor");
        assert!(p.starts_with("antes "), "o prefixo começa numa linha inteira: {:?}", &p[..12]);
        assert!(suf.starts_with(s) && s.starts_with("depois 0000\n"), "o sufixo mantém o começo");
        assert!(s.ends_with('\n'), "o sufixo termina numa linha inteira");
        assert!(p.len() > 1000 && s.len() > 1000, "os dois lados mantêm contexto");
    }

    #[test]
    fn trim_start_stays_put_while_typing() {
        // o começo do prefixo não pode mudar a cada tecla (invalidaria o KV cache inteiro)
        let base = lines("antes", 2000);
        let suf = lines("depois", 300);
        let mut starts = std::collections::BTreeSet::new();
        for extra in 0..100 {
            let pre = format!("{base}{}", "x".repeat(extra));
            let (p, _) = trim_fim_prompt(&pre, &suf, 1024, fake_size).unwrap();
            starts.insert(pre.len() - p.len());
        }
        assert!(starts.len() <= 2, "o corte andou {} vezes em 100 teclas: {starts:?}", starts.len());
    }

    #[test]
    fn trim_is_utf8_safe_and_deterministic() {
        let pre = "ção ã é 日本語\n".repeat(3000);
        let suf = "ção ã é 日本語\n".repeat(3000);
        for budget in [64, 100, 1024, 5000] {
            let a = trim_fim_prompt(&pre, &suf, budget, fake_size);
            assert_eq!(a, trim_fim_prompt(&pre, &suf, budget, fake_size));
            if let Some((p, s)) = a {
                assert!(fake_size(p, s).unwrap() <= budget);
            }
        }
    }

    #[test]
    fn trim_gives_up_when_nothing_fits_or_cannot_build() {
        assert_eq!(trim_fim_prompt("abc", "def", 4, |_, _| Some(100)), None);
        assert_eq!(trim_fim_prompt("abc", "def", 4, |_, _| None), None);
    }

    #[test]
    fn trim_always_terminates_even_on_huge_or_odd_inputs() {
        // guarda de tempo: o corte roda numa thread e o teste espera no máximo 30 s (nunca fica preso)
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let big = lines("x", 150_000); // ~1,5 MB
            let one_line = "a".repeat(1_000_000); // sem nenhuma quebra de linha
            let mut ok = true;
            for budget in [64, 1024, 32_768] {
                ok &= trim_fim_prompt(&big, &big, budget, fake_size).is_some_and(|(p, s)| fake_size(p, s).unwrap() <= budget);
                ok &= trim_fim_prompt(&one_line, &one_line, budget, fake_size).is_some_and(|(p, s)| fake_size(p, s).unwrap() <= budget);
                ok &= trim_fim_prompt("", &big, budget, fake_size).is_some();
                ok &= trim_fim_prompt(&big, "", budget, fake_size).is_some();
                ok &= trim_fim_prompt("", "", budget, fake_size).is_some();
                ok &= trim_fim_prompt(&big, &big, budget, |_, _| Some(usize::MAX)).is_none(); // nada cabe: desiste
            }
            let _ = tx.send(ok);
        });
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(30)), Ok(true), "o corte travou ou deu resultado errado");
    }

    #[test]
    fn trim_without_suffix_only_cuts_the_prefix() {
        let pre = lines("antes", 3000);
        let (p, s) = trim_fim_prompt(&pre, "", 1024, fake_size).unwrap();
        assert_eq!(s, "");
        assert!(pre.ends_with(p) && p.len() < pre.len() && fake_size(p, s).unwrap() <= 1024);
    }

    #[test]
    fn trim_by_chars_respects_the_estimate() {
        let (pre, suf) = (lines("antes", 2000), lines("depois", 2000));
        let (p, s) = trim_fim_by_chars(&pre, &suf, 1024);
        assert!(p.chars().count() + s.chars().count() <= 1024 * CHARS_PER_TOKEN_ESTIMATE);
        assert!(pre.ends_with(p) && suf.starts_with(s) && !p.is_empty() && !s.is_empty());
    }

    #[tokio::test]
    async fn sse_events_handles_split_blocks() {
        // blocos cortados no meio de um evento, de um "\n\n" e de um caractere UTF-8 (ã = 2 bytes)
        let raw = "data: {\"a\":\"ã\"}\n\ndata: {\"a\":2}\n\ndata: [DONE]\n\ndata: {\"a\":3}\n\n".as_bytes().to_vec();
        for cut in 1..raw.len() {
            let (x, y) = raw.split_at(cut);
            let v: Vec<Result<Bytes, Box<dyn Error + Send + Sync>>> = vec![Ok(Bytes::from(x.to_vec())), Ok(Bytes::from(y.to_vec()))];
            let evs: Vec<Value> = sse_events(Box::pin(stream::iter(v))).collect().await;
            assert_eq!(evs.len(), 2, "corte em {cut}");
            assert_eq!(evs[0]["a"], "ã");
            assert_eq!(evs[1]["a"], 2);
        }
        // sem [DONE] e sem a linha em branco final
        let evs: Vec<Value> = sse_events(bs(&["data: {\"a\":1}\n\ndata: {\"a\":2}"])).collect().await;
        assert_eq!(evs.len(), 2);
    }
}
