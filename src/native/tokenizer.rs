//! Tokenizador BPE byte-level (estilo GPT-2) lido do próprio GGUF, com o template de chat do modelo:
//! ChatML (Qwen2) ou o formato de cabeçalhos do Llama 3.x. O pré-tokenizador reproduz, sem usar
//! regex, o padrão:
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]*
//!   | \s*[\r\n]+ | \s+(?!\S) | \s+
//! No Llama 3 o número é `\p{N}{1,3}` (até 3 dígitos juntos) em vez de um dígito só.

use std::collections::HashMap;

use super::gguf::Gguf;
use crate::engine::ChatMsg;

pub struct Tokenizer {
    vocab: Vec<String>,
    to_id: HashMap<String, u32>,
    merge_rank: HashMap<String, u32>,
    /// tokens especiais (texto, id), do mais longo para o mais curto
    #[allow(dead_code)] // reservado: encode com tokens especiais
    specials: Vec<(String, u32)>,
    byte_enc: [char; 256],
    byte_dec: HashMap<char, u8>,
    pub eos: Vec<u32>,
    chat: Chat,
    /// quantos dígitos seguidos formam um pré-token (Qwen: 1; Llama 3: 3)
    digit_group: usize,
    /// tipo de cada token do GGUF (1 normal, 2 desconhecido, 3 controle, 4 definido pelo usuário, 5 não usado, 6 byte)
    types: Vec<u8>,
    /// tokenizador SentencePiece (Mistral, Gemma, Phi-3); `None` = BPE byte-level (Qwen, Llama 3)
    spm: Option<Spm>,
    /// Tokens "definidos pelo usuário" (tipo 4: "\n", sequências de espaços, `<think>`...), do mais
    /// longo para o mais curto. O llama.cpp os reconhece no texto ANTES do BPE/SentencePiece, mesmo
    /// sem interpretar tokens especiais. Ficam de fora os delimitadores de turno do próprio template
    /// (para o texto do usuário nunca poder injetá-los).
    user_tokens: Vec<(String, u32)>,
    /// o arquivo manda acrescentar o BOS no início do prompt (`tokenizer.ggml.add_bos_token`; o Phi-3 diz que não)
    add_bos: bool,
    /// pré-tokenizador "gpt-4o" (o200k) em vez do do Qwen/Llama 3
    gpt4o: bool,
    /// pré-tokenizador "starcoder" (StarCoder2): cada dígito sozinho + o padrão do GPT-2 (ver `next_pretoken_starcoder`)
    starcoder: bool,
    /// o que o template do arquivo permite com ferramentas (D79)
    tool_support: crate::tools::ToolSupport,
}

/// Trecho do texto depois de separar os tokens definidos pelo usuário.
enum Frag {
    Raw(String),
    Token(u32),
}

/// Dados do tokenizador SentencePiece: pontuação de cada token, tokens de byte e o prefixo de espaço.
struct Spm {
    scores: Vec<f32>,
    byte_ids: [Option<u32>; 256],
    /// coloca um "▁" antes do texto (o prefixo de espaço do SentencePiece)
    add_space_prefix: bool,
}

/// Formato de chat do modelo e os ids dos tokens especiais que ele usa.
enum Chat {
    /// `qwen3`: sem prompt de sistema padrão e sem o raciocínio (`<think>`) das respostas antigas
    ChatMl { im_start: u32, im_end: u32, qwen3: bool, sys: Option<&'static str> },
    Llama3 { bos: u32, start_header: u32, end_header: u32, eot: u32 },
    /// Mistral e Llama 2: `[INST] ... [/INST]`
    Inst { bos: u32, inst: u32, inst_end: u32, eos: u32 },
    /// Gemma: `<start_of_turn>user` ... `<end_of_turn>`
    Gemma { bos: u32, start_turn: u32, end_turn: u32 },
    /// Phi-3: `<|user|>` ... `<|end|>`
    Phi3 { bos: u32, system: u32, user: u32, assistant: u32, end: u32, nl: bool },
    /// StarCoder2: `instruct` = o GGUF traz o template `### Instruction` / `### Response` (15B-instruct);
    /// sem template (3B, modelo base) o "chat" é texto cru a ser continuado.
    StarCoder { bos: u32, eos: u32, instruct: bool },
    /// SmolVLM / Idefics3 (SmolLM2): `<|im_start|>User: ...<end_of_utterance>\nAssistant:`; NÃO é ChatML (D50/M3)
    SmolVlm { im_start: u32, end_utt: u32 },
}

/// Marcador de imagem no texto de uma mensagem (o `mtmd_default_marker()` do llama.cpp).
pub const IMAGE_MARKER: &str = "<__media__>";

/// Marcadores de código reconhecidos no texto dos pedidos de completar (Qwen2.5-Coder e a família StarCoder).
const CODE_MARKERS: [&str; 12] = [
    "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
    "<fim_pad>", "<file_sep>", "<reponame>", "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
];

/// Separa `text` nos marcadores dados (texto, id): o que está entre eles continua texto comum.
fn split_markers(text: &str, markers: &[(&str, u32)]) -> Vec<Seg> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let next = markers.iter().filter_map(|(m, id)| rest.find(m).map(|p| (p, m.len(), *id))).min_by_key(|(p, _, _)| *p);
        match next {
            Some((p, len, id)) => {
                if p > 0 {
                    out.push(Seg::Text(rest[..p].to_string()));
                }
                out.push(Seg::Special(id));
                rest = &rest[p + len..];
            }
            None => {
                if !rest.is_empty() || out.is_empty() {
                    out.push(Seg::Text(rest.to_string()));
                }
                return out;
            }
        }
    }
}

enum Seg {
    Special(u32),
    Text(String),
}

/// O formato de chat é escolhido pela ARQUITETURA do GGUF (não pelo texto do template). Para não montar em silêncio um
/// prompt errado, confere que o template embutido no arquivo, quando existe, usa os marcadores do formato escolhido
/// (ChatML `<|im_start|>`, Llama 3 `<|start_header_id|>`, Mistral `[INST]`, Gemma `<start_of_turn>`, Phi-3 `<|end|>`).
/// Um fine-tune com outro template (Alpaca, Zephyr, DeepSeek...) na mesma arquitetura é recusado. Sem template no arquivo
/// (modelo base), mantém o formato da arquitetura: é o que já funcionava e o llama.cpp também cai num padrão.
/// Chamada de funções (D79): vale só se o template EMBUTIDO no arquivo tiver ferramentas no formato que o Buriti sabe
/// escrever e ler (Hermes do Qwen2.5/Qwen3, Llama 3.1+). O Mistral v0.3 tem `[AVAILABLE_TOOLS]`, mas ainda não está implementado.
fn tool_support_for(chat: &Chat, arch: &str, tmpl: &str) -> crate::tools::ToolSupport {
    use crate::tools::{ToolFormat, ToolSupport};
    match chat {
        Chat::ChatMl { qwen3, sys, .. } if tmpl.contains("<tools>") && tmpl.contains("<tool_call>") => {
            // Os GGUFs oficiais do Qwen2.5 escrevem o exemplo do formato com chaves duplicadas (`{{"name": ...}}`, um escape que sobrou da
            // conversão). Medido (D79): o modelo copia as chaves duplicadas e erra o JSON em 5 de 5 pedidos; por isso o exemplo sai com
            // chaves simples, como no template original do Hugging Face (`double_braces: false`).
            ToolSupport::Yes(ToolFormat::Hermes { qwen3: *qwen3, double_braces: false, default_system: *sys })
        }
        Chat::Llama3 { .. } if tmpl.contains("Environment: ipython") && tmpl.contains("tool_calls") => ToolSupport::Yes(ToolFormat::Llama3),
        Chat::Inst { .. } if tmpl.contains("[AVAILABLE_TOOLS]") => ToolSupport::not_implemented("Mistral, [AVAILABLE_TOOLS]"),
        _ => ToolSupport::no_template(arch),
    }
}

/// Sistema do Llama 3 depois do cabeçalho: com ferramentas, `Environment: ipython` vem antes da data e o texto
/// (que já traz as ferramentas) não é aparado, como no template oficial.
fn llama3_system_text(system: &str, ipython: bool, today: &str) -> String {
    format!("\n\n{}Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n{system}", if ipython { "Environment: ipython\n" } else { "" })
}

fn check_template(chat: &Chat, arch: &str, tmpl: &str) -> Result<(), String> {
    let marker = match chat {
        Chat::ChatMl { .. } => "<|im_start|>",
        Chat::Llama3 { .. } => "<|start_header_id|>",
        Chat::Inst { .. } => "[INST]",
        Chat::Gemma { .. } => "<start_of_turn>",
        Chat::Phi3 { .. } => "<|end|>",
        Chat::StarCoder { .. } => "### Instruction",
        Chat::SmolVlm { .. } => "<end_of_utterance>",
    };
    if tmpl.trim().is_empty() || tmpl.contains(marker) {
        return Ok(());
    }
    Err(crate::errcode::coded("err_chat_template", &[arch], format!("the chat template of the model (architecture '{arch}') is not a supported format; use llama.cpp")))
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let model = g.meta.get("tokenizer.ggml.model").and_then(|v| v.as_str()).unwrap_or("");
        if model != "gpt2" && model != "llama" {
            return Err(crate::errcode::coded("err_tokenizer", &[&model], format!("tokenizer '{model}' is not supported yet (only gpt2/Qwen/Llama 3 BPE and SentencePiece)")));
        }
        let tokens = g.meta.get("tokenizer.ggml.tokens").and_then(|v| v.as_arr()).ok_or("GGUF sem tokenizer.ggml.tokens")?;
        let vocab: Vec<String> = tokens.iter().map(|t| t.as_str().unwrap_or("").to_string()).collect();
        let mut to_id = HashMap::with_capacity(vocab.len());
        for (i, t) in vocab.iter().enumerate() {
            to_id.entry(t.clone()).or_insert(i as u32);
        }

        let merge_rank = if model == "gpt2" {
            let merges = g.meta.get("tokenizer.ggml.merges").and_then(|v| v.as_arr()).ok_or("GGUF sem tokenizer.ggml.merges")?;
            merges
                .iter()
                .enumerate()
                .filter_map(|(i, m)| m.as_str().map(|s| (s.to_string(), i as u32)))
                .collect::<HashMap<_, _>>()
        } else {
            HashMap::new() // SentencePiece não usa merges: usa a pontuação dos tokens
        };

        // tipos: 3 = control, 4 = user-defined → tratados como especiais
        let types = g.meta.get("tokenizer.ggml.token_type").and_then(|v| v.as_arr());
        let type_ids: Vec<u8> = match types {
            Some(t) => t.iter().map(|x| x.as_u64().unwrap_or(1) as u8).collect(),
            None => vec![1; vocab.len()],
        };
        let mut specials: Vec<(String, u32)> = Vec::new();
        if let Some(types) = types {
            for (i, ty) in types.iter().enumerate() {
                if matches!(ty.as_u64(), Some(3) | Some(4)) && i < vocab.len() && !vocab[i].is_empty() {
                    specials.push((vocab[i].clone(), i as u32));
                }
            }
        }
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        // tabela byte <-> unicode do GPT-2
        let mut byte_enc = ['\0'; 256];
        let mut byte_dec = HashMap::new();
        let mut n = 0u32;
        for b in 0..=255u32 {
            let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let c = if printable {
                char::from_u32(b).unwrap()
            } else {
                let c = char::from_u32(256 + n).unwrap();
                n += 1;
                c
            };
            byte_enc[b as usize] = c;
            byte_dec.insert(c, b as u8);
        }

        let arch = g.meta.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("");
        let find = |s: &str| to_id.get(s).copied().ok_or_else(|| format!("token ausente: {s} (formato de chat do modelo não reconhecido)"));
        let tmpl = g.meta.get("tokenizer.chat_template").and_then(|v| v.as_str()).unwrap_or("");
        let meta_id = |k: &str| g.meta.get(k).and_then(|v| v.as_u64()).map(|x| x as u32);
        let (chat, digit_group, mut eos) = if model == "llama" {
            // SentencePiece: o formato de chat depende da arquitetura (e, no Llama/Mistral, do template do arquivo)
            let bos = meta_id("tokenizer.ggml.bos_token_id").or_else(|| to_id.get("<s>").copied()).or_else(|| to_id.get("<bos>").copied());
            let need_bos = || bos.ok_or_else(|| "GGUF sem token BOS".to_string());
            match arch {
                "gemma" | "gemma2" | "gemma3" => {
                    let end_turn = find("<end_of_turn>")?;
                    (Chat::Gemma { bos: need_bos()?, start_turn: find("<start_of_turn>")?, end_turn }, 1, vec![end_turn])
                }
                "phi3" => {
                    let end = find("<|end|>")?;
                    let mut eos = vec![end];
                    if let Some(&e) = to_id.get("<|endoftext|>") {
                        eos.push(e);
                    }
                    (Chat::Phi3 { bos: need_bos()?, system: find("<|system|>")?, user: find("<|user|>")?, assistant: find("<|assistant|>")?, end, nl: tmpl.contains("<|end|>
") }, 1, eos)
                }
                "llama" if tmpl.contains("[INST]") => {
                    // Mistral v0.3+: [INST] e [/INST] são tokens de controle próprios. Nos modelos mais antigos
                    // eles são texto comum (não suportado ainda).
                    let (inst, inst_end) = match (to_id.get("[INST]"), to_id.get("[/INST]")) {
                        (Some(&a), Some(&b)) if type_ids.get(a as usize).copied() == Some(3) => (a, b),
                        _ => return Err(crate::errcode::coded("err_inst_template", &[], "the [INST] template with text tokens (Mistral v0.1/v0.2, Llama 2) is not supported yet; use llama.cpp")),
                    };
                    let eos_id = meta_id("tokenizer.ggml.eos_token_id").or_else(|| to_id.get("</s>").copied()).ok_or("GGUF sem token EOS")?;
                    (Chat::Inst { bos: need_bos()?, inst, inst_end, eos: eos_id }, 1, vec![eos_id])
                }
                _ => return Err(crate::errcode::coded("err_sp_arch", &[&arch], format!("the SentencePiece model of architecture '{arch}' has no recognised chat format; use llama.cpp"))),
            }
        } else if arch == "phi3" {
            // Phi-4-mini: BPE (o200k) com as mesmas tags do Phi-3 (`<|user|>`...`<|end|>`), mas SEM quebras de linha
            let end = find("<|end|>")?;
            let mut eos = vec![end];
            if let Some(&e) = to_id.get("<|endoftext|>") {
                eos.push(e);
            }
            let bos = meta_id("tokenizer.ggml.bos_token_id").or_else(|| to_id.get("<|endoftext|>").copied()).unwrap_or(end);
            (Chat::Phi3 { bos, system: find("<|system|>")?, user: find("<|user|>")?, assistant: find("<|assistant|>")?, end, nl: tmpl.contains("<|end|>\n") }, 3, eos)
        } else if arch == "starcoder2" {
            // StarCoder2: BPE do GPT-2 com o pré-tokenizador "starcoder" (dígitos soltos). O 15B-instruct traz
            // o template `### Instruction`/`### Response` (conferido nos metadados do arquivo); o 3B é base, sem template.
            let pre = g.meta.get("tokenizer.ggml.pre").and_then(|v| v.as_str()).unwrap_or("");
            // os GGUFs do catálogo (second-state 3B, bartowski 15B) NÃO trazem `tokenizer.ggml.pre`: o llama.cpp b12606 cai então
            // no pré-tokenizador "default" (avisa "GENERATION QUALITY WILL BE DEGRADED"); aqui vale o certo, o do StarCoder2.
            if !pre.is_empty() && pre != "starcoder" {
                return Err(crate::errcode::coded("err_pretok_sc", &[&pre], format!("pre-tokenizer '{pre}' is not supported for StarCoder2 yet (only 'starcoder')")));
            }
            let end = find("<|endoftext|>")?;
            let bos = meta_id("tokenizer.ggml.bos_token_id").unwrap_or(end);
            (Chat::StarCoder { bos, eos: end, instruct: tmpl.contains("### Instruction") }, 1, vec![end])
        } else if arch == "llama" && g.meta.get("tokenizer.ggml.pre").and_then(|v| v.as_str()) == Some("smollm") {
            // SmolLM2 (SmolVLM): BPE com o pré-tokenizador "smollm" (igual ao "starcoder" no llama.cpp b12606: dígitos
            // soltos + padrão do GPT-2) e o template `User:`/`Assistant:` (o do arquivo contém <end_of_utterance>)
            if !tmpl.contains("<end_of_utterance>") {
                return Err("modelo SmolLM2 sem o template de chat do SmolVLM; use o llama.cpp".into());
            }
            let end_utt = find("<end_of_utterance>")?;
            let im_start = find("<|im_start|>")?;
            (Chat::SmolVlm { im_start, end_utt }, 1, vec![end_utt])
        } else if arch == "llama" {
            // Llama 3.x usa o pré-tokenizador "llama-bpe"; outros (Llama 2 etc.) não são suportados
            let pre = g.meta.get("tokenizer.ggml.pre").and_then(|v| v.as_str()).unwrap_or("");
            if pre != "llama-bpe" {
                return Err(crate::errcode::coded("err_pretok", &[&pre], format!("pre-tokenizer '{pre}' is not supported yet (only llama-bpe of Llama 3.x)")));
            }
            let eot = find("<|eot_id|>")?;
            let chat = Chat::Llama3 {
                bos: find("<|begin_of_text|>")?,
                start_header: find("<|start_header_id|>")?,
                end_header: find("<|end_header_id|>")?,
                eot,
            };
            let mut eos = vec![eot];
            for s in ["<|end_of_text|>", "<|eom_id|>"] {
                if let Some(&e) = to_id.get(s) {
                    eos.push(e);
                }
            }
            (chat, 3, eos)
        } else {
            let im_end = find("<|im_end|>")?;
            let mut eos = vec![im_end];
            if let Some(&e) = to_id.get("<|endoftext|>") {
                eos.push(e);
            }
            // sistema padrão (quando o chat não traz um): o Qwen2.5 usa o texto da Alibaba; o Qwen2.5-Coder
            // e outros ChatML usam "You are a helpful assistant."; o Qwen3 não injeta nada
            let sys = if arch == "qwen3" {
                None
            } else if tmpl.contains("You are Qwen, created by Alibaba Cloud") || !tmpl.contains("You are a helpful assistant") {
                Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
            } else {
                Some("You are a helpful assistant.")
            };
            (Chat::ChatMl { im_start: find("<|im_start|>")?, im_end, qwen3: arch == "qwen3", sys }, 1, eos)
        };
        check_template(&chat, arch, tmpl)?;
        let tool_support = tool_support_for(&chat, arch, tmpl);
        let is_smollm = matches!(chat, Chat::SmolVlm { .. });
        if let Some(e) = g.meta.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()) {
            if !eos.contains(&(e as u32)) {
                eos.push(e as u32);
            }
        }

        let spm = if model == "llama" {
            let scores: Vec<f32> = g
                .meta
                .get("tokenizer.ggml.scores")
                .and_then(|v| v.as_arr())
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                .ok_or("GGUF sem tokenizer.ggml.scores")?;
            if scores.len() != vocab.len() {
                return Err("tokenizer.ggml.scores com tamanho diferente do vocabulário".into());
            }
            // token de cada byte: "<0xXX>" ou, se faltar, o próprio caractere ASCII
            let mut byte_ids = [None; 256];
            for (b, slot) in byte_ids.iter_mut().enumerate() {
                *slot = to_id.get(&format!("<0x{b:02X}>")).copied().or_else(|| if b < 128 { to_id.get(&(b as u8 as char).to_string()).copied() } else { None });
            }
            let add_space_prefix = g.meta.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.as_bool()).unwrap_or(true);
            Some(Spm { scores, byte_ids, add_space_prefix })
        } else {
            None
        };

        // delimitadores de turno do template: nunca são reconhecidos no texto do usuário
        let mut protected: Vec<u32> = eos.clone();
        match &chat {
            Chat::ChatMl { im_start, im_end, .. } => protected.extend([*im_start, *im_end]),
            Chat::Llama3 { bos, start_header, end_header, eot } => protected.extend([*bos, *start_header, *end_header, *eot]),
            Chat::Inst { bos, inst, inst_end, eos } => protected.extend([*bos, *inst, *inst_end, *eos]),
            Chat::Gemma { bos, start_turn, end_turn } => protected.extend([*bos, *start_turn, *end_turn]),
            Chat::Phi3 { bos, system, user, assistant, end, .. } => protected.extend([*bos, *system, *user, *assistant, *end]),
            Chat::StarCoder { bos, eos, .. } => protected.extend([*bos, *eos]),
            Chat::SmolVlm { im_start, end_utt } => protected.extend([*im_start, *end_utt]),
        }
        let mut user_tokens: Vec<(String, u32)> = type_ids
            .iter()
            .enumerate()
            .filter(|(i, &t)| t == 4 && !protected.contains(&(*i as u32)) && i < &vocab.len() && !vocab[*i].is_empty())
            .map(|(i, _)| (vocab[i].clone(), i as u32))
            .collect();
        user_tokens.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        Ok(Tokenizer { vocab, to_id, merge_rank, specials, byte_enc, byte_dec, eos, chat, digit_group, types: type_ids, spm, user_tokens, add_bos: g.meta.get("tokenizer.ggml.add_bos_token").and_then(|v| v.as_bool()).unwrap_or(arch != "starcoder2" && !is_smollm), gpt4o: g.meta.get("tokenizer.ggml.pre").and_then(|v| v.as_str()) == Some("gpt-4o"), starcoder: arch == "starcoder2" || is_smollm, tool_support })
    }

    /// Token de início de texto (BOS), se o modelo exige (Llama 3); o prompt de chat já o inclui.
    #[cfg_attr(not(test), allow(dead_code))] // o prompt de chat já inclui o BOS; só os testes de logits o usam
    pub fn bos(&self) -> Option<u32> {
        match self.chat {
            Chat::Llama3 { bos, .. } | Chat::Inst { bos, .. } | Chat::Gemma { bos, .. } | Chat::Phi3 { bos, .. } | Chat::StarCoder { bos, .. } => if self.add_bos { Some(bos) } else { None },
            Chat::ChatMl { .. } | Chat::SmolVlm { .. } => None,
        }
    }

    /// Id do token cujo texto é exatamente `s` (BPE byte-level: `s` passa pela tabela byte -> unicode do GPT-2).
    pub fn lookup_text(&self, s: &str) -> Option<u32> {
        let enc: String = s.bytes().map(|b| self.byte_enc[b as usize]).collect();
        self.to_id.get(&enc).copied()
    }

    /// Id do `<image>` (o marcador de posição de cada vetor de imagem) no formato SmolVLM.
    pub fn image_slot_id(&self) -> Option<u32> {
        matches!(self.chat, Chat::SmolVlm { .. }).then(|| self.to_id.get("<image>").copied()).flatten()
    }

    /// Ids de UMA imagem no formato Idefics3 do llama.cpp b12606 (`mtmd.cpp`, `ov_img_first = false`): com grade
    /// `gx × gy` (> 0), cada fatia vira `<fake_token_around_image><row_Y_col_X>` + `per_tile` vetores, com `\n` entre as
    /// linhas (não depois da última); depois a visão geral: `\n\n<fake_token_around_image><global-img>` + `per_tile`
    /// vetores + `<fake_token_around_image>`. Cada vetor é o id de `<image>` (o motor troca o embedding dessa posição).
    /// Grade 0 × 0 = só a visão geral (HF: imagem única sem divisão, sem o `\n\n` inicial).
    pub fn idefics3_image_ids(&self, gx: usize, gy: usize, per_tile: usize) -> Option<Vec<u32>> {
        let slot = self.image_slot_id()?;
        let fake = *self.to_id.get("<fake_token_around_image>")?;
        let global = *self.to_id.get("<global-img>")?;
        let (nl, nl2) = (self.lookup_text("\n")?, self.lookup_text("\n\n")?);
        let mut out = Vec::new();
        for y in 0..gy {
            for x in 0..gx {
                out.push(fake);
                out.push(*self.to_id.get(&format!("<row_{}_col_{}>", y + 1, x + 1))?);
                out.extend(std::iter::repeat_n(slot, per_tile));
            }
            if y + 1 != gy {
                out.push(nl);
            }
        }
        if gx > 0 && gy > 0 {
            out.push(nl2);
        }
        out.push(fake);
        out.push(global);
        out.extend(std::iter::repeat_n(slot, per_tile));
        out.push(fake);
        Some(out)
    }

    /// Só para diagnóstico: o texto de uma sequência de ids COM os tokens de controle visíveis.
    #[cfg(test)]
    pub fn render_all(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.spm.is_some() && matches!(self.types.get(id as usize).copied(), Some(3) | Some(5)) {
                bytes.extend_from_slice(self.vocab[id as usize].as_bytes());
            } else {
                bytes.extend(self.token_bytes(id));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Só para diagnóstico: id de um texto no vocabulário e seu tipo e pontuação (SentencePiece).
    #[cfg(test)]
    pub fn debug_lookup(&self, s: &str) -> String {
        match self.to_id.get(s) {
            Some(&id) => format!("{s:?} -> id {id}, tipo {}, pontuacao {:?}", self.types.get(id as usize).copied().unwrap_or(0), self.spm.as_ref().map(|p| p.scores[id as usize])),
            None => format!("{s:?} -> ausente"),
        }
    }

    /// Ids dos tokens de preenchimento no meio (FIM): (prefixo, sufixo, meio). Qwen2.5-Coder usa
    /// `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`; a família StarCoder, `<fim_prefix>` etc.
    fn fim_tokens(&self) -> Option<(u32, u32, u32)> {
        for (p, s, m) in [("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"), ("<fim_prefix>", "<fim_suffix>", "<fim_middle>")] {
            if let (Some(&a), Some(&b), Some(&c)) = (self.to_id.get(p), self.to_id.get(s), self.to_id.get(m)) {
                return Some((a, b, c));
            }
        }
        None
    }

    /// Tokens que, ao serem gerados num completar de código, encerram a resposta: o fim da lacuna
    /// (`<|fim_pad|>`), o separador de arquivos e o nome do repositório (o modelo passou para "outro arquivo") e os
    /// próprios marcadores FIM (o modelo se perdeu). Só os que existem no vocabulário.
    pub fn code_stop_ids(&self) -> Vec<u32> {
        self.code_markers().into_iter().map(|(_, id)| id).collect()
    }

    /// Marcadores de código (FIM, separador de arquivos, nome do repositório) presentes no vocabulário: texto e id.
    fn code_markers(&self) -> Vec<(&'static str, u32)> {
        CODE_MARKERS.iter().filter_map(|n| self.to_id.get(*n).map(|&id| (*n, id))).collect()
    }

    /// Texto de um completar de código: os marcadores de código escritos no texto (como o Continue e outros plugins
    /// enviam, ex.: `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>` no próprio `prompt`, ou `<|file_sep|>arquivo` entre
    /// arquivos de contexto) viram os tokens especiais, como no llama.cpp. Nos pedidos de CHAT isso não acontece
    /// (o conteúdo do usuário nunca vira token especial). Os delimitadores do chat (`<|im_start|>` etc.) continuam texto.
    fn push_code_text(&self, segs: &mut Vec<Seg>, text: &str) {
        segs.extend(split_markers(text, &self.code_markers()));
    }

    pub fn has_fim(&self) -> bool {
        self.fim_tokens().is_some()
    }

    /// Prompt de completar texto cru (sem formato de chat): BOS (se o modelo pede) + texto.
    pub fn raw_prompt(&self, text: &str) -> Vec<u32> {
        let mut segs = Vec::new();
        if let Some(b) = self.bos() {
            segs.push(Seg::Special(b));
        }
        self.push_code_text(&mut segs, text);
        self.encode_segments(&segs)
    }

    /// Prompt de preenchimento no meio: `<prefixo>` texto antes `<sufixo>` texto depois `<meio>`.
    /// O modelo continua a partir de `<meio>` até o fim da lacuna. `None` se o modelo não tem tokens FIM.
    pub fn fim_prompt(&self, prefix: &str, suffix: &str) -> Option<Vec<u32>> {
        let (p, s, m) = self.fim_tokens()?;
        let mut segs = Vec::new();
        if let Some(b) = self.bos() {
            segs.push(Seg::Special(b));
        }
        segs.push(Seg::Special(p));
        self.push_code_text(&mut segs, prefix);
        segs.push(Seg::Special(s));
        self.push_code_text(&mut segs, suffix);
        segs.push(Seg::Special(m));
        Some(self.encode_segments(&segs))
    }

    /// O primeiro texto da resposta deve perder o espaço inicial? (SentencePiece com prefixo de espaço)
    pub fn strips_first_space(&self) -> bool {
        self.spm.as_ref().is_some_and(|s| s.add_space_prefix)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    // ------------------------------------------------------------ codificação

    /// Codifica texto comum (sem interpretar tokens especiais), como início de um texto.
    #[cfg_attr(not(test), allow(dead_code))] // o motor usa `chat_prompt`; os testes de tokenização usam esta
    pub fn encode_plain(&self, text: &str) -> Vec<u32> {
        self.encode_text(text, true)
    }

    /// `prev_special`: o que vem antes deste texto é o início da sequência ou um token especial. No
    /// SentencePiece, TODO trecho de texto nessa posição (início, depois do BOS, depois de `[INST]`,
    /// depois de um token definido pelo usuário...) recebe o prefixo de espaço ("▁"), como no llama.cpp.
    fn encode_text(&self, text: &str, prev_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut prev = prev_special;
        for f in self.partition(text) {
            match f {
                Frag::Token(id) => {
                    out.push(id);
                    prev = true;
                }
                Frag::Raw(s) => {
                    self.encode_raw(&s, prev, &mut out);
                    prev = false;
                }
            }
        }
        out
    }

    /// Separa o texto nos tokens definidos pelo usuário (os mais longos primeiro, como o llama.cpp).
    fn partition(&self, text: &str) -> Vec<Frag> {
        let mut frags = vec![Frag::Raw(text.to_string())];
        for (tok_text, id) in &self.user_tokens {
            if !frags.iter().any(|f| matches!(f, Frag::Raw(s) if s.contains(tok_text.as_str()))) {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len() + 2);
            for f in frags {
                match f {
                    Frag::Raw(s) if s.contains(tok_text.as_str()) => {
                        let mut rest = s.as_str();
                        while let Some(pos) = rest.find(tok_text.as_str()) {
                            if pos > 0 {
                                next.push(Frag::Raw(rest[..pos].to_string()));
                            }
                            next.push(Frag::Token(*id));
                            rest = &rest[pos + tok_text.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Frag::Raw(rest.to_string()));
                        }
                    }
                    other => next.push(other),
                }
            }
            frags = next;
        }
        frags
    }

    /// Codifica um trecho de texto já sem tokens definidos pelo usuário.
    fn encode_raw(&self, text: &str, first: bool, out: &mut Vec<u32>) {
        if let Some(spm) = &self.spm {
            self.spm_encode(spm, text, first && spm.add_space_prefix, out);
            return;
        }
        let chars: Vec<char> = text.chars().collect();
        if self.starcoder {
            // 1º passo do llama.cpp: cada dígito é um trecho; 2º: o padrão do GPT-2 em cada trecho sem dígitos
            let mut i = 0;
            while i < chars.len() {
                let end = if is_number(chars[i]) { i + 1 } else { chars[i..].iter().position(|&c| is_number(c)).map_or(chars.len(), |p| i + p) };
                let chunk = &chars[i..end];
                let mut j = 0;
                while j < chunk.len() {
                    let k = next_pretoken_starcoder(chunk, j);
                    let piece: String = chunk[j..k].iter().collect();
                    self.bpe(&piece, out);
                    j = k;
                }
                i = end;
            }
            return;
        }
        let mut i = 0;
        while i < chars.len() {
            let end = if self.gpt4o { next_pretoken_o200k(&chars, i) } else { next_pretoken(&chars, i, self.digit_group) };
            let piece: String = chars[i..end].iter().collect();
            self.bpe(&piece, out);
            i = end;
        }
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<String> = piece.bytes().map(|b| self.byte_enc[b as usize].to_string()).collect();
        while syms.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..syms.len() - 1 {
                let key = format!("{} {}", syms[i], syms[i + 1]);
                if let Some(&r) = self.merge_rank.get(&key) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }
        for s in syms {
            match self.to_id.get(&s) {
                Some(&id) => out.push(id),
                None => {
                    // não deveria ocorrer em BPE byte-level; recua para caracteres soltos
                    for c in s.chars() {
                        if let Some(&id) = self.to_id.get(&c.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// SentencePiece (como o `llm_tokenizer_spm` do llama.cpp): espaços viram "▁", cada caractere é um
    /// símbolo, e os pares adjacentes que formam um token do vocabulário são fundidos do de MAIOR
    /// pontuação para o menor (empate: o mais à esquerda). O que sobra fora do vocabulário vira bytes.
    fn spm_encode(&self, spm: &Spm, text: &str, add_prefix: bool, out: &mut Vec<u32>) {
        use std::{cmp::Ordering, collections::BinaryHeap};
        if text.is_empty() {
            return;
        }
        let mut s = String::with_capacity(text.len() + 3);
        if add_prefix {
            s.push('▁');
        }
        for c in text.chars() {
            s.push(if c == ' ' { '▁' } else { c });
        }
        // símbolos: um por caractere UTF-8; `n == 0` = fundido no vizinho da esquerda
        struct Sym {
            start: usize,
            n: usize,
            prev: isize,
            next: isize,
        }
        let mut syms: Vec<Sym> = Vec::new();
        for (i, c) in s.char_indices() {
            let k = syms.len() as isize;
            syms.push(Sym { start: i, n: c.len_utf8(), prev: k - 1, next: k + 1 });
        }
        if let Some(l) = syms.last_mut() {
            l.next = -1;
        }
        #[derive(PartialEq)]
        struct Bigram {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl Eq for Bigram {}
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Bigram {
            // heap máximo: maior pontuação primeiro; em empate, o par mais à esquerda
            fn cmp(&self, o: &Self) -> Ordering {
                self.score.total_cmp(&o.score).then_with(|| o.left.cmp(&self.left))
            }
        }
        let mut heap = BinaryHeap::new();
        let try_add = |syms: &Vec<Sym>, heap: &mut BinaryHeap<Bigram>, l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (a, b) = (&syms[l as usize], &syms[r as usize]);
            let piece = &s[a.start..a.start + a.n + b.n];
            if let Some(&id) = self.to_id.get(piece) {
                heap.push(Bigram { score: spm.scores[id as usize], left: l as usize, right: r as usize, size: piece.len() });
            }
        };
        for i in 1..syms.len() {
            try_add(&syms, &mut heap, i as isize - 1, i as isize);
        }
        while let Some(bg) = heap.pop() {
            let (ln, rn) = (syms[bg.left].n, syms[bg.right].n);
            if ln == 0 || rn == 0 || ln + rn != bg.size {
                continue; // par desatualizado: um dos lados já foi fundido
            }
            syms[bg.left].n += rn;
            syms[bg.right].n = 0;
            let next = syms[bg.right].next;
            syms[bg.left].next = next;
            if next >= 0 {
                syms[next as usize].prev = bg.left as isize;
            }
            let prev = syms[bg.left].prev;
            try_add(&syms, &mut heap, prev, bg.left as isize);
            try_add(&syms, &mut heap, bg.left as isize, next);
        }
        let mut i: isize = 0;
        while i != -1 && (i as usize) < syms.len() {
            let sym = &syms[i as usize];
            if sym.n > 0 {
                let piece = &s[sym.start..sym.start + sym.n];
                match self.to_id.get(piece) {
                    Some(&id) => out.push(id),
                    None => {
                        for &b in piece.as_bytes() {
                            if let Some(id) = spm.byte_ids[b as usize] {
                                out.push(id);
                            }
                        }
                    }
                }
            }
            i = sym.next;
        }
    }

    fn encode_segments(&self, segs: &[Seg]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut prev_special = true; // o início da sequência conta como "depois de um token especial"
        for s in segs {
            match s {
                Seg::Special(id) => {
                    out.push(*id);
                    prev_special = true;
                }
                Seg::Text(t) => {
                    out.extend(self.encode_text(t, prev_special));
                    prev_special = false;
                }
            }
        }
        out
    }

    /// Monta o prompt de chat no formato do modelo (ChatML do Qwen ou cabeçalhos do Llama 3). O
    /// conteúdo do usuário NUNCA é interpretado como token especial (evita injeção de "<|im_end|>",
    /// "<|eot_id|>" etc. no texto).
    pub fn chat_prompt(&self, msgs: &[ChatMsg]) -> Vec<u32> {
        self.chat_prompt_images(msgs, None)
    }

    /// Como `chat_prompt`, mas no formato SmolVLM cada ocorrência de `IMAGE_MARKER` no texto de uma mensagem de usuário
    /// vira os ids devolvidos por `expand(k)` (k = ordem da imagem no prompt): é assim que o llama.cpp marca a imagem
    /// (`<__media__>`). Sem `expand`, ou em outros formatos, o marcador é texto comum.
    pub fn chat_prompt_images(&self, msgs: &[ChatMsg], expand: Option<&dyn Fn(usize) -> Vec<u32>>) -> Vec<u32> {
        let mut segs: Vec<Seg> = Vec::new();
        let push_text = |segs: &mut Vec<Seg>, t: &str| match segs.last_mut() {
            Some(Seg::Text(s)) => s.push_str(t),
            _ => segs.push(Seg::Text(t.to_string())),
        };
        match self.chat {
            Chat::ChatMl { im_start, im_end, qwen3, sys } => {
                // Qwen2.5 injeta um sistema padrão quando não há (o texto depende do template); o Qwen3 não
                if let (Some(sys), true) = (sys, msgs.first().map(|m| m.role.as_str()) != Some("system")) {
                    segs.push(Seg::Special(im_start));
                    push_text(&mut segs, &format!("system\n{sys}"));
                    segs.push(Seg::Special(im_end));
                    push_text(&mut segs, "\n");
                }
                for m in msgs {
                    segs.push(Seg::Special(im_start));
                    // Qwen3: o raciocínio das respostas antigas não volta ao prompt (como no template oficial)
                    let content = if qwen3 && m.role == "assistant" { drop_thinking(&m.content) } else { m.content.as_str() };
                    push_text(&mut segs, &format!("{}\n{}", m.role, content));
                    segs.push(Seg::Special(im_end));
                    push_text(&mut segs, "\n");
                }
                segs.push(Seg::Special(im_start));
                push_text(&mut segs, "assistant\n");
            }
            Chat::Llama3 { bos, start_header, end_header, eot } => {
                // template oficial do Llama 3.1/3.2: o bloco de sistema sempre existe (com a data de
                // hoje); a 1ª mensagem de sistema, se houver, entra nele; o conteúdo é aparado (trim)
                // com ferramentas (D79) o papel interno `TOOLS_SYSTEM_ROLE` traz o texto já montado (não é aparado)
                let (system, rest, ipython) = match msgs.first() {
                    Some(m) if m.role == "system" => (m.content.trim(), &msgs[1..], false),
                    Some(m) if m.role == crate::tools::TOOLS_SYSTEM_ROLE => (m.content.as_str(), &msgs[1..], true),
                    _ => ("", msgs, false),
                };
                if self.add_bos {
                    segs.push(Seg::Special(bos));
                }
                segs.push(Seg::Special(start_header));
                push_text(&mut segs, "system");
                segs.push(Seg::Special(end_header));
                push_text(&mut segs, &llama3_system_text(system, ipython, &today()));
                segs.push(Seg::Special(eot));
                for m in rest {
                    segs.push(Seg::Special(start_header));
                    push_text(&mut segs, &m.role);
                    segs.push(Seg::Special(end_header));
                    push_text(&mut segs, &format!("\n\n{}", m.content.trim()));
                    segs.push(Seg::Special(eot));
                }
                segs.push(Seg::Special(start_header));
                push_text(&mut segs, "assistant");
                segs.push(Seg::Special(end_header));
                push_text(&mut segs, "\n\n");
            }
            Chat::Inst { bos, inst, inst_end, eos } => {
                // Template embutido no GGUF do Mistral v0.3: o usuário vira `'[INST] ' + conteúdo + ' [/INST]'`
                // (com espaço antes de [/INST]) e a resposta vem logo depois, seguida de `</s>`. O template não
                // prevê o papel `system`: ele vai para o início da 1ª mensagem do usuário, seguido de "\n",
                // como o llama.cpp faz.
                let (system, rest) = match msgs.first() {
                    Some(m) if m.role == "system" => (Some(m.content.as_str()), &msgs[1..]),
                    _ => (None, msgs),
                };
                if self.add_bos {
                    segs.push(Seg::Special(bos));
                }
                let mut first_user = true;
                for m in rest {
                    if m.role == "assistant" {
                        push_text(&mut segs, &m.content);
                        segs.push(Seg::Special(eos));
                    } else {
                        segs.push(Seg::Special(inst));
                        let sys = if first_user { system.map(|s| format!("{s}\n")).unwrap_or_default() } else { String::new() };
                        first_user = false;
                        push_text(&mut segs, &format!(" {sys}{} ", m.content));
                        segs.push(Seg::Special(inst_end));
                    }
                }
            }
            Chat::Gemma { bos, start_turn, end_turn } => {
                // o sistema (se houver) vira o prefixo da primeira mensagem do usuário
                let (prefix, rest) = match msgs.first() {
                    Some(m) if m.role == "system" => (format!("{}\n\n", m.content), &msgs[1..]),
                    _ => (String::new(), msgs),
                };
                if self.add_bos {
                    segs.push(Seg::Special(bos));
                }
                for (i, m) in rest.iter().enumerate() {
                    segs.push(Seg::Special(start_turn));
                    let role = if m.role == "assistant" { "model" } else { m.role.as_str() };
                    push_text(&mut segs, &format!("{role}\n{}{}", if i == 0 { prefix.as_str() } else { "" }, m.content.trim()));
                    segs.push(Seg::Special(end_turn));
                    push_text(&mut segs, "\n");
                }
                segs.push(Seg::Special(start_turn));
                push_text(&mut segs, "model\n");
            }
            Chat::Phi3 { bos, system, user, assistant, end, nl } => {
                // `nl`: o template do Phi-3 põe "\n" depois da tag e depois de <|end|>; o do Phi-4-mini não
                let nl = if nl { "\n" } else { "" };
                if self.add_bos {
                    segs.push(Seg::Special(bos));
                }
                for m in msgs {
                    let tag = match m.role.as_str() {
                        "system" => system,
                        "assistant" => assistant,
                        _ => user,
                    };
                    segs.push(Seg::Special(tag));
                    push_text(&mut segs, &format!("{nl}{}", m.content));
                    segs.push(Seg::Special(end));
                    if !nl.is_empty() {
                        push_text(&mut segs, nl);
                    }
                }
                segs.push(Seg::Special(assistant));
                if !nl.is_empty() {
                    push_text(&mut segs, nl);
                }
            }
            Chat::StarCoder { bos, eos, instruct: false } => {
                // Modelo base (sem template): o "chat" é texto cru, o modelo continua o que o usuário escreveu. As
                // mensagens são concatenadas sem separador (a resposta do modelo continua o texto e a mensagem
                // seguinte continua a resposta); mensagens de sistema são ignoradas (não há onde pô-las).
                let _ = eos;
                if self.add_bos {
                    segs.push(Seg::Special(bos));
                }
                for m in msgs.iter().filter(|m| m.role != "system") {
                    push_text(&mut segs, &m.content);
                }
            }
            Chat::StarCoder { bos, eos, instruct: true } => {
                // Template do GGUF do starcoder2-15b-instruct-v0.1 (sem `strip`, sem papel `system`: o template do
                // arquivo lança exceção com `system`; aqui o texto de sistema, se houver, vai no início da 1ª instrução):
                //   {{bos_token}}You are an exceptionally intelligent coding assistant ... instructions.\n\n
                //   (user)      ### Instruction\n{content}\n\n
                //   (outro)     ### Response\n{content}{eos_token}\n\n
                //   ... ### Response\n
                let (system, rest) = match msgs.first() {
                    Some(m) if m.role == "system" => (Some(m.content.as_str()), &msgs[1..]),
                    _ => (None, msgs),
                };
                segs.push(Seg::Special(bos));
                push_text(&mut segs, "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n");
                let mut first_user = true;
                for m in rest {
                    if m.role == "user" {
                        let sys = if first_user { system.map(|s| format!("{s}\n\n")).unwrap_or_default() } else { String::new() };
                        first_user = false;
                        push_text(&mut segs, &format!("### Instruction\n{sys}{}\n\n", m.content));
                    } else {
                        push_text(&mut segs, &format!("### Response\n{}", m.content));
                        segs.push(Seg::Special(eos));
                        push_text(&mut segs, "\n\n");
                    }
                }
                push_text(&mut segs, "### Response\n");
            }
            Chat::SmolVlm { im_start, end_utt } => {
                // Template do GGUF (o llama-server usa este, via jinja, no /apply-template; conferido): `<|im_start|>` e, por
                // mensagem, `Papel: texto<end_of_utterance>\n` com o papel capitalizado (inclusive `System`); termina em
                // `Assistant:`. O texto não é aparado. (O formato embutido do llama-chat.cpp difere no sistema; não é o usado.)
                segs.push(Seg::Special(im_start));
                let mut k = 0usize;
                for m in msgs {
                    let mut chars = m.role.chars();
                    let label = match chars.next() {
                        Some(c) => format!("{}{}: ", c.to_uppercase(), chars.as_str().to_lowercase()),
                        None => ": ".to_string(),
                    };
                    push_text(&mut segs, &label);
                    match (expand, m.role.as_str()) {
                        (Some(ex), "user") if m.content.contains(IMAGE_MARKER) => {
                            for (i, part) in m.content.split(IMAGE_MARKER).enumerate() {
                                if i > 0 {
                                    for id in ex(k) {
                                        segs.push(Seg::Special(id));
                                    }
                                    k += 1;
                                }
                                if !part.is_empty() {
                                    push_text(&mut segs, part);
                                }
                            }
                        }
                        _ => push_text(&mut segs, &m.content),
                    }
                    segs.push(Seg::Special(end_utt));
                    push_text(&mut segs, "\n");
                }
                push_text(&mut segs, "Assistant:");
            }
        }
        self.encode_segments(&segs)
    }

    // ------------------------------------------------------------ decodificação

    /// Bytes UTF-8 de um token.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(s) = self.vocab.get(id as usize) else { return Vec::new() };
        if self.spm.is_some() {
            return match self.types.get(id as usize).copied().unwrap_or(1) {
                // token de byte "<0xXX>"
                6 => s.strip_prefix("<0x").and_then(|h| h.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok()).map(|b| vec![b]).unwrap_or_default(),
                // controle / não usado: não aparece no texto
                3 | 5 => Vec::new(),
                _ => s.replace('▁', " ").into_bytes(),
            };
        }
        let mut out = Vec::with_capacity(s.len());
        for c in s.chars() {
            match self.byte_dec.get(&c) {
                Some(&b) => out.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        out
    }

    /// Bytes de um token que pode aparecer como texto, para a máscara da saída restrita (D82). `None` para tokens de controle,
    /// especiais, não usados e fim de texto: eles nunca podem ser casados como texto da gramática.
    pub fn text_token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        if self.eos.contains(&id) || self.is_special(id) {
            return None;
        }
        if !matches!(self.types.get(id as usize).copied().unwrap_or(1), 1 | 6) {
            return None;
        }
        let b = self.token_bytes(id);
        if b.is_empty() {
            None
        } else {
            Some(b)
        }
    }

    /// O que o template deste modelo permite com ferramentas (D79).
    pub fn tool_support(&self) -> crate::tools::ToolSupport {
        self.tool_support.clone()
    }

    #[allow(dead_code)]
    pub fn is_special(&self, id: u32) -> bool {
        self.specials.iter().any(|(_, i)| *i == id)
    }
}

/// Acumula bytes de tokens e devolve só texto UTF-8 completo (um caractere pode ocupar vários tokens).
#[derive(Default)]
pub struct Utf8Stream {
    buf: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    break;
                }
                Err(e) => {
                    let v = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.buf[..v]).unwrap());
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.buf.drain(..v + n);
                        }
                        None => {
                            self.buf.drain(..v); // sequência incompleta: espera mais bytes
                            break;
                        }
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------- pré-tokenizador

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_numeric()
}
fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Remove o raciocínio (`<think>...</think>`) de uma resposta anterior: fica o que vem depois do
/// último `</think>`, sem as quebras de linha iniciais (como o template oficial do Qwen3).
fn drop_thinking(content: &str) -> &str {
    match content.rfind("</think>") {
        Some(i) => content[i + "</think>".len()..].trim_start_matches('\n'),
        None => content,
    }
}

const MONTHS: [&str; 12] =["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

fn utc_today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    date_from_unix(secs)
}

/// Data de HOJE, no fuso local, no formato "%d %b %Y" que o template do Llama 3 usa ("21 Nov 2026"),
/// como o `strftime_now` do llama.cpp. (Só duas chamadas de sistema mínimas, sem dependências.)
#[cfg(windows)]
fn today() -> String {
    #[repr(C)]
    struct SystemTime {
        year: u16,
        month: u16,
        day_of_week: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        millis: u16,
    }
    extern "system" {
        fn GetLocalTime(t: *mut SystemTime);
    }
    let mut t = SystemTime { year: 0, month: 0, day_of_week: 0, day: 0, hour: 0, minute: 0, second: 0, millis: 0 };
    // SAFETY: GetLocalTime só escreve na estrutura apontada (layout SYSTEMTIME da API do Windows).
    unsafe { GetLocalTime(&mut t) };
    if (1..=12).contains(&t.month) && (1..=31).contains(&t.day) {
        format!("{:02} {} {}", t.day, MONTHS[(t.month - 1) as usize], t.year)
    } else {
        utc_today()
    }
}

#[cfg(unix)]
fn today() -> String {
    use std::ffi::{c_char, c_int, c_long};
    #[repr(C)]
    struct Tm {
        sec: c_int,
        min: c_int,
        hour: c_int,
        mday: c_int,
        mon: c_int,
        year: c_int,
        wday: c_int,
        yday: c_int,
        isdst: c_int,
        gmtoff: c_long,
        zone: *const c_char,
    }
    extern "C" {
        fn localtime_r(t: *const c_long, out: *mut Tm) -> *mut Tm;
    }
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as c_long).unwrap_or(0);
    let mut tm = Tm { sec: 0, min: 0, hour: 0, mday: 0, mon: 0, year: 0, wday: 0, yday: 0, isdst: 0, gmtoff: 0, zone: std::ptr::null() };
    // SAFETY: localtime_r é thread-safe e só escreve em `tm` (layout de `struct tm` do glibc e do macOS).
    let r = unsafe { localtime_r(&secs, &mut tm) };
    if !r.is_null() && (0..12).contains(&tm.mon) && (1..=31).contains(&tm.mday) {
        format!("{:02} {} {}", tm.mday, MONTHS[tm.mon as usize], tm.year as i64 + 1900)
    } else {
        utc_today()
    }
}

#[cfg(not(any(windows, unix)))]
fn today() -> String {
    utc_today()
}

fn date_from_unix(secs: i64) -> String {
    // dias desde 1970-01-01 -> data civil (algoritmo de Howard Hinnant)
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{d:02} {} {y}", MONTHS[(m - 1) as usize])
}

/// Pré-tokenizador "gpt-4o" (o200k: Phi-4-mini). Diferenças para o do Llama 3: as palavras separam
/// camelCase e MAIÚSCULAS, a contração (`'s`, `'ll`...) vai COLADA na palavra, e a pontuação leva `/` no fim:
///   `[^\r\n\p{L}\p{N}]?[^a-z]*[^A-Z]+ contr? | [^\r\n\p{L}\p{N}]?[^a-z]+[^A-Z]* contr? | \p{N}{1,3}
///    | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+` (nas classes de palavra só entram letras).
fn next_pretoken_o200k(c: &[char], i: usize) -> usize {
    let n = c.len();
    let ch = c[i];
    // "up": letra que não é a-z (maiúscula ASCII ou qualquer letra não ASCII); "lo": letra que não é A-Z
    let up = |x: char| is_letter(x) && !x.is_ascii_lowercase();
    let lo = |x: char| is_letter(x) && !x.is_ascii_uppercase();
    // contração opcional colada: 's 't 're 've 'm 'll 'd (sem diferenciar maiúsculas)
    let contraction = |e: usize| -> usize {
        if e + 1 < n && c[e] == '\'' {
            let a = c[e + 1].to_ascii_lowercase();
            if matches!(a, 's' | 't' | 'm' | 'd') {
                return e + 2;
            }
            if e + 2 < n && matches!((a, c[e + 2].to_ascii_lowercase()), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                return e + 3;
            }
        }
        e
    };

    // palavras: prefixo opcional (um caractere que não é quebra de linha, letra nem número) + letras
    let s = if !is_nl(ch) && !is_letter(ch) && !is_number(ch) { i + 1 } else { i };
    if s < n && is_letter(c[s]) {
        // A: up* lo+  (o `*` é guloso, mas recua se `lo+` não casar)
        let mut u = 0;
        while s + u < n && up(c[s + u]) {
            u += 1;
        }
        for k in (0..=u).rev() {
            let mut l = 0;
            while s + k + l < n && lo(c[s + k + l]) {
                l += 1;
            }
            if l >= 1 {
                return contraction(s + k + l);
            }
        }
        // B: up+ lo*  (só chega aqui numa sequência de maiúsculas ASCII seguida de não-letra)
        if u >= 1 {
            let mut l = 0;
            while s + u + l < n && lo(c[s + u + l]) {
                l += 1;
            }
            return contraction(s + u + l);
        }
    }

    // \p{N}{1,3}
    if is_number(ch) {
        let mut k = i + 1;
        while k < n && k - i < 3 && is_number(c[k]) {
            k += 1;
        }
        return k;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n/]*
    let j = if ch == ' ' { i + 1 } else { i };
    let mut k = j;
    while k < n && !c[k].is_whitespace() && !is_letter(c[k]) && !is_number(c[k]) {
        k += 1;
    }
    if k > j {
        while k < n && (is_nl(c[k]) || c[k] == '/') {
            k += 1;
        }
        return k;
    }

    // \s*[\r\n]+ | \s+(?!\S) | \s+
    let mut ws = i;
    while ws < n && c[ws].is_whitespace() {
        ws += 1;
    }
    if ws > i {
        if let Some(m) = (i..ws).rev().find(|&m| is_nl(c[m])) {
            return m + 1;
        }
        if ws == n {
            return ws;
        }
        if ws - 1 > i {
            return ws - 1;
        }
        return ws;
    }
    i + 1
}

/// Pré-tokenizador "starcoder" do llama.cpp (b12606, `LLAMA_VOCAB_PRE_TYPE_STARCODER`): os dígitos já foram
/// separados um a um (`\p{N}`); aqui o padrão do GPT-2 `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
/// sobre um trecho SEM dígitos `c` (o fim do trecho conta como fim do texto para o `(?!\S)`). Reproduz o
/// `unicode_regex_split_custom_gpt2`, inclusive o que ele faz com um espaço em branco isolado (vira pré-token, como se
/// o padrão tivesse `|\s+`). Fim (exclusivo) do pré-token que começa em `i`.
pub(crate) fn next_pretoken_starcoder(c: &[char], i: usize) -> usize {
    let n = c.len();
    let ch = c[i];
    if ch == '\'' && i + 1 < n {
        let a = c[i + 1];
        if matches!(a, 's' | 't' | 'm' | 'd') {
            return i + 2;
        }
        if i + 2 < n && matches!((a, c[i + 2]), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
            return i + 3;
        }
    }
    // ` ?` opcional: o caractere que decide é o seguinte ao espaço
    let f = if ch == ' ' { i + 1 } else { i };
    if f < n && is_letter(c[f]) {
        let mut k = f;
        while k < n && is_letter(c[k]) {
            k += 1;
        }
        return k;
    }
    if f < n && !c[f].is_whitespace() && !is_number(c[f]) {
        let mut k = f;
        while k < n && !c[k].is_whitespace() && !is_letter(c[k]) && !is_number(c[k]) {
            k += 1;
        }
        return k;
    }
    let mut ws = 0;
    while i + ws < n && c[i + ws].is_whitespace() {
        ws += 1;
    }
    if ws > 1 && i + ws < n {
        return i + ws - 1; // \s+(?!\S): o último espaço fica para o próximo pré-token
    }
    if ws > 0 {
        return i + ws;
    }
    i + 1
}

/// Fim (exclusivo) do próximo pré-token que começa em `i`. `digits` = quantos dígitos seguidos
/// formam um pré-token (1 no Qwen, 3 no Llama 3).
fn next_pretoken(c: &[char], i: usize, digits: usize) -> usize {
    let n = c.len();
    let ch = c[i];

    // 1) contrações: 's 't 're 've 'm 'll 'd (sem diferenciar maiúsculas)
    if ch == '\'' && i + 1 < n {
        let a = c[i + 1].to_ascii_lowercase();
        if matches!(a, 's' | 't' | 'm' | 'd') {
            return i + 2;
        }
        if i + 2 < n {
            let b = c[i + 2].to_ascii_lowercase();
            if matches!((a, b), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                return i + 3;
            }
        }
    }

    // 2) [^\r\n\p{L}\p{N}]?\p{L}+
    let j = if !is_nl(ch) && !is_letter(ch) && !is_number(ch) { i + 1 } else { i };
    if j < n && is_letter(c[j]) {
        let mut k = j;
        while k < n && is_letter(c[k]) {
            k += 1;
        }
        return k;
    }

    // 3) \p{N} (Qwen) ou \p{N}{1,3} (Llama 3): até `digits` dígitos seguidos
    if is_number(ch) {
        let mut k = i + 1;
        while k < n && k - i < digits && is_number(c[k]) {
            k += 1;
        }
        return k;
    }

    // 4)  ?[^\s\p{L}\p{N}]+[\r\n]*
    let j = if ch == ' ' { i + 1 } else { i };
    let mut k = j;
    while k < n && !c[k].is_whitespace() && !is_letter(c[k]) && !is_number(c[k]) {
        k += 1;
    }
    if k > j {
        while k < n && is_nl(c[k]) {
            k += 1;
        }
        return k;
    }

    // 5..7) espaços em branco
    let mut ws = i;
    while ws < n && c[ws].is_whitespace() {
        ws += 1;
    }
    if ws > i {
        // 5) \s*[\r\n]+  → termina após a última quebra de linha da sequência
        if let Some(m) = (i..ws).rev().find(|&m| is_nl(c[m])) {
            return m + 1;
        }
        // 6) \s+(?!\S)   → deixa o último espaço para o próximo token, se seguido de não-espaço
        if ws == n {
            return ws;
        }
        if ws - 1 > i {
            return ws - 1;
        }
        // 7) \s+
        return ws;
    }

    i + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_support_follows_the_embedded_template() {
        use crate::tools::{ToolFormat, ToolSupport};
        let chatml = Chat::ChatMl { im_start: 1, im_end: 2, qwen3: false, sys: Some("S") };
        let llama3 = Chat::Llama3 { bos: 1, start_header: 2, end_header: 3, eot: 4 };
        let inst = Chat::Inst { bos: 1, inst: 2, inst_end: 3, eos: 4 };
        let gemma = Chat::Gemma { bos: 1, start_turn: 2, end_turn: 3 };
        // trecho do template do Qwen2.5 (GGUF oficial): chaves duplicadas no exemplo (o Buriti escreve chaves simples)
        let qwen = "{%- if tools %}<tools>...{{\\\"name\\\": <function-name>}}\n<tool_call>"; // no arquivo: barra + aspas literais
        assert!(matches!(tool_support_for(&chatml, "qwen2", qwen), ToolSupport::Yes(ToolFormat::Hermes { double_braces: false, qwen3: false, default_system: Some("S") })), "chaves duplicadas do GGUF viram simples (D79)");
        let single = "{%- if tools %}<tools>...{\"name\": <function-name>}\n<tool_call>";
        assert!(matches!(tool_support_for(&chatml, "qwen2", single), ToolSupport::Yes(ToolFormat::Hermes { double_braces: false, .. })));
        // ChatML sem ferramentas (fine-tune, base): recusa com a recomendação
        assert!(matches!(tool_support_for(&chatml, "qwen2", "<|im_start|>user"), ToolSupport::No(m) if m.contains("Qwen2.5")));
        assert!(matches!(tool_support_for(&chatml, "qwen2", ""), ToolSupport::No(_)));
        assert!(matches!(tool_support_for(&llama3, "llama", "Environment: ipython ... tool_calls"), ToolSupport::Yes(ToolFormat::Llama3)));
        assert!(matches!(tool_support_for(&llama3, "llama", "<|start_header_id|>"), ToolSupport::No(_))); // Llama 3.0: sem ferramentas
        assert!(matches!(tool_support_for(&inst, "llama", "[AVAILABLE_TOOLS] [INST]"), ToolSupport::No(m) if m.contains("Mistral")));
        assert!(matches!(tool_support_for(&inst, "llama", "[INST]"), ToolSupport::No(_)));
        assert!(matches!(tool_support_for(&gemma, "gemma3", "<start_of_turn>"), ToolSupport::No(m) if m.contains("gemma3")));
    }

    #[test]
    fn llama3_system_block_with_and_without_tools() {
        assert_eq!(llama3_system_text("Seja breve.", false, "26 Jul 2024"), "\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nSeja breve.");
        assert_eq!(llama3_system_text("TOOLS\n\n", true, "26 Jul 2024"), "\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nTOOLS\n\n");
    }

    #[test]
    fn unknown_chat_template_is_refused() {
        let chatml = Chat::ChatMl { im_start: 1, im_end: 2, qwen3: false, sys: None };
        let llama3 = Chat::Llama3 { bos: 1, start_header: 2, end_header: 3, eot: 4 };
        let gemma = Chat::Gemma { bos: 1, start_turn: 2, end_turn: 3 };
        let phi = Chat::Phi3 { bos: 1, system: 2, user: 3, assistant: 4, end: 5, nl: true };
        let inst = Chat::Inst { bos: 1, inst: 2, inst_end: 3, eos: 4 };
        // templates reais (trechos) de cada família passam
        assert!(check_template(&chatml, "qwen2", "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}").is_ok());
        assert!(check_template(&llama3, "llama", "<|start_header_id|>user<|end_header_id|>").is_ok());
        assert!(check_template(&gemma, "gemma3", "<start_of_turn>user\n").is_ok());
        assert!(check_template(&phi, "phi3", "{{'<|user|>\n' + m + '<|end|>\n'}}").is_ok());
        assert!(check_template(&inst, "llama", "[INST] x [/INST]").is_ok());
        // sem template (modelo base): mantém o formato da arquitetura
        assert!(check_template(&chatml, "qwen2", "").is_ok());
        assert!(check_template(&chatml, "qwen2", "  \n").is_ok());
        // outro formato na mesma arquitetura (Alpaca, Zephyr, DeepSeek) é recusado, com a mensagem pedindo o llama.cpp
        let alpaca = "{% for m in messages %}### Instruction:\n{{ m.content }}\n\n### Response:\n{% endfor %}";
        assert_eq!(crate::errcode::english(&check_template(&chatml, "qwen2", alpaca).unwrap_err()), "the chat template of the model (architecture 'qwen2') is not a supported format; use llama.cpp");
        assert!(check_template(&chatml, "qwen2", "<|user|>\n{{ m }}</s>").is_err()); // Zephyr
        assert!(check_template(&llama3, "llama", "<|im_start|>user").is_err()); // marcador de OUTRA família não vale
        assert!(check_template(&gemma, "gemma3", alpaca).is_err());
        assert!(check_template(&phi, "phi3", alpaca).is_err());
        // StarCoder2: o template `### Instruction` do 15B-instruct passa; o do modelo base é vazio; ChatML não
        let sc_instruct = Chat::StarCoder { bos: 0, eos: 0, instruct: true };
        let sc_base = Chat::StarCoder { bos: 0, eos: 0, instruct: false };
        assert!(check_template(&sc_instruct, "starcoder2", "{{bos_token}}{{'### Instruction\n' + m}}{{'### Response\n'}}").is_ok());
        assert!(check_template(&sc_base, "starcoder2", "").is_ok());
        assert!(check_template(&sc_base, "starcoder2", "<|im_start|>user").is_err());
        assert!(check_template(&sc_instruct, "starcoder2", "<|im_start|>user").is_err());
    }

    #[test]
    fn drops_thinking() {
        assert_eq!(drop_thinking("<think>
raciocinio
</think>

Resposta"), "Resposta");
        assert_eq!(drop_thinking("sem raciocinio"), "sem raciocinio");
        assert_eq!(drop_thinking("<think>a</think>b<think>c</think>
d"), "d");
    }

    #[test]
    fn dates() {
        assert_eq!(date_from_unix(0), "01 Jan 1970");
        assert_eq!(date_from_unix(951_782_400), "29 Feb 2000"); // ano bissexto
        assert_eq!(date_from_unix(1_721_952_000), "26 Jul 2024"); // data usada pelo template do Llama
        assert_eq!(date_from_unix(1_721_952_000 + 86_399), "26 Jul 2024");
        assert_eq!(date_from_unix(1_721_952_000 + 86_400), "27 Jul 2024");
        assert_eq!(date_from_unix(4_102_444_800), "01 Jan 2100");
    }

    fn pieces(text: &str, digits: usize) -> Vec<String> {
        let c: Vec<char> = text.chars().collect();
        let (mut i, mut out) = (0, Vec::new());
        while i < c.len() {
            let e = next_pretoken(&c, i, digits);
            out.push(c[i..e].iter().collect());
            i = e;
        }
        out
    }

    fn pieces_o200k(text: &str) -> Vec<String> {
        let c: Vec<char> = text.chars().collect();
        let (mut i, mut out) = (0, Vec::new());
        while i < c.len() {
            let e = next_pretoken_o200k(&c, i);
            out.push(c[i..e].iter().collect());
            i = e;
        }
        out
    }

    #[test]
    fn o200k_words() {
        assert_eq!(pieces_o200k("Hello World"), ["Hello", " World"]);
        assert_eq!(pieces_o200k("helloWorld"), ["hello", "World"]); // camelCase separa
        assert_eq!(pieces_o200k("ALLCAPS word"), ["ALLCAPS", " word"]);
        assert_eq!(pieces_o200k("don't"), ["don't"]); // contração colada
        assert_eq!(pieces_o200k("IT'S"), ["IT'S"]);
        assert_eq!(pieces_o200k("a/b"), ["a", "/b"]); // a barra é o prefixo da palavra seguinte
        assert_eq!(pieces_o200k("x //
y"), ["x", " //
", "y"]); // pontuação leva / e quebra de linha
        assert_eq!(pieces_o200k("12345"), ["123", "45"]);
        assert_eq!(pieces_o200k("McDonald"), ["Mc", "Donald"]);
    }

    #[test]
    fn digit_grouping() {
        assert_eq!(pieces("12345", 1), ["1", "2", "3", "4", "5"]); // Qwen: um dígito
        assert_eq!(pieces("12345", 3), ["123", "45"]); // Llama 3: até 3
        assert_eq!(pieces("a 1234567 b", 3), ["a", " ", "123", "456", "7", " b"]);
        assert_eq!(pieces("x12", 3), ["x", "12"]);
    }

    #[test]
    fn splits_code_markers() {
        let m = [("<|a|>", 1u32), ("<|bc|>", 2)];
        let show = |v: Vec<Seg>| v.iter().map(|s| match s { Seg::Special(i) => format!("#{i}"), Seg::Text(t) => format!("\"{t}\"") }).collect::<Vec<_>>().join(" ");
        assert_eq!(show(split_markers("x<|a|>y<|bc|><|a|>", &m)), "\"x\" #1 \"y\" #2 #1");
        assert_eq!(show(split_markers("<|a|>", &m)), "#1");
        assert_eq!(show(split_markers("", &m)), "\"\"");
        assert_eq!(show(split_markers("<|im_start|>x", &m)), "\"<|im_start|>x\"");
        assert_eq!(show(split_markers("só ação <|bc|>é", &m)), "\"só ação \" #2 \"é\"");
    }

    /// Diagnóstico (modelo real): os marcadores FIM escritos como texto no prompt viram os mesmos tokens que
    /// `fim_prompt`? E quais ids encerram um completar de código? `BURITI_TEST_MODEL=... cargo test fim_tokens_diag -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn fim_tokens_diag() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("defina BURITI_TEST_MODEL");
        let g = Gguf::open(std::path::Path::new(&path)).expect("abrir GGUF");
        let tok = Tokenizer::from_gguf(&g).expect("tokenizador");
        let via_api = tok.fim_prompt("def f(x):\n    ", "\n    return y\n").expect("FIM");
        let via_text = tok.raw_prompt("<|fim_prefix|>def f(x):\n    <|fim_suffix|>\n    return y\n<|fim_middle|>");
        println!("FIM api : {via_api:?}");
        println!("FIM text: {via_text:?}");
        println!("stop ids: {:?}", tok.code_stop_ids());
        let sep = tok.raw_prompt("<|file_sep|>a.js\nx");
        println!("file_sep: {sep:?} (id {:?})", tok.to_id.get("<|file_sep|>"));
        assert_eq!(via_api, via_text, "o texto com marcadores FIM deve virar os mesmos tokens");
    }

    /// Diagnóstico (modelos reais): carrega o tokenizador de cada GGUF de `BURITI_TEST_MODELS` (caminhos separados por `;`)
    /// e imprime o resultado (ou o erro) mais os ids do prompt de chat de 3 conversas; serve para provar que uma mudança
    /// na escolha do formato de chat não altera o prompt de nenhum modelo que já funciona.
    /// `BURITI_TEST_MODELS="a.gguf;b.gguf" cargo test chat_ids_all_models -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn chat_ids_all_models() {
        let list = std::env::var("BURITI_TEST_MODELS").expect("defina BURITI_TEST_MODELS");
        let msg = |role: &str, content: &str| crate::engine::ChatMsg { role: role.into(), content: content.into() };
        let convs = [
            vec![msg("user", "Hello! What is 12345 + 678?")],
            vec![msg("system", "  You are a pirate.  "), msg("user", "Say hi"), msg("assistant", "Arr, hi!"), msg("user", "Again, please.")],
            vec![msg("user", "What is 2+2?"), msg("assistant", "<think>\nThe user asks a sum.\n</think>\n\n4."), msg("user", "And 3+3?")],
        ];
        for path in list.split(';').filter(|p| !p.is_empty()) {
            let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
            let g = Gguf::open(std::path::Path::new(path)).expect("abrir GGUF");
            let arch = g.meta.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("?").to_string();
            match Tokenizer::from_gguf(&g) {
                Err(e) => println!("MODEL {name} arch={arch} ERRO {e}"),
                Ok(tok) => {
                    println!("MODEL {name} arch={arch} OK");
                    for (i, c) in convs.iter().enumerate() {
                        let ids = tok.chat_prompt(c);
                        println!("IDS {name} {i} {}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
                    }
                }
            }
        }
    }
}
