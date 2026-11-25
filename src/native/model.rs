//! Transformer decoder em CPU: Qwen2 (RoPE NeoX, viés em Q/K/V) e Llama 3.x (RoPE de pares adjacentes, sem viés), ambos com GQA + RMSNorm + SwiGLU.
//! Prefill em lote e geração um token por vez; KV cache em f16. Pesos ficam mapeados em memória (mmap).

use std::path::Path;

use rayon::prelude::*;

// StarCoder2 (LayerNorm com viés, MLP sem porta): passagem direta e carregador num arquivo à parte, como submódulo
// para enxergar os itens privados deste (Tensor, Session, matvec/matmat...).
#[path = "starcoder2.rs"]
pub(crate) mod starcoder2;

use super::{
    gguf::{Dtype, Gguf, Value},
    kernels::{dot_row_q, dot_row_q_x4, dot_row_q_x8, Act, ActCache},
    repack::{Kind32, Repacked},
    f16::store_f16,
    vmath::{attend_group, gelu_mul, silu_mul},
    quant::{dequant_row, dot_f32, dot_row, f16_to_f32},
    tokenizer::Tokenizer,
};

pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub n_ctx_train: usize,
    pub n_vocab: usize,
    /// RoPE de pares adjacentes (Llama) em vez de pares (i, i + hd/2) (Qwen2/NeoX)
    pub rope_norm: bool,
    /// Gemma: multiplica o embedding por sqrt(n_embd)
    pub embed_scale: f32,
    /// GeGLU (Gemma) em vez de SwiGLU
    pub gelu: bool,
    /// janela deslizante das camadas locais (Gemma 3); 0 = atenção completa em todas
    pub window: usize,
    /// dimensões de cada cabeça que recebem RoPE (RoPE parcial no Phi-4-mini; = head_dim nos demais)
    pub n_rot: usize,
    /// fator que multiplica cos e sen do RoPE (`attn_factor` do longrope; 1 nos demais)
    pub rope_mscale: f32,
}

/// Ponteiro mutável que pode atravessar threads: o `matmat` o usa para que cada tarefa escreva só nas
/// suas colunas do destino (intervalos disjuntos). Ver o comentário de segurança no uso.
struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}
impl SyncPtr {
    #[inline(always)]
    fn ptr(&self) -> *mut f32 {
        self.0
    }
}

/// Matriz [rows x cols]: `rows` = saídas, `cols` = entradas (cada linha é contígua e quantizada por blocos).
pub struct Tensor {
    dtype: Dtype,
    rows: usize,
    cols: usize,
    off: usize,
    row_bytes: usize,
    /// Q4_K, Q5_K ou Q6_K reempacotado em grupos de 8 linhas (`repack.rs`): quando existe, TODAS as multiplicações
    /// deste tensor (um token ou lote) passam por ele, nunca pelo layout original do mmap (a ordem de
    /// soma em float difere entre os dois e o prefill tem de dar o mesmo resultado que a geração).
    /// Só tensores Q4_K/Q6_K das camadas com linhas múltiplas de 8 e colunas múltiplas de 256; o embedding e
    /// a saída (lidos linha a linha / possivelmente compartilhados) e os outros formatos ficam sem.
    rep: Option<Repacked>,
}

impl Tensor {
    /// Fatia de `count` linhas a partir da linha `start` (as linhas são contíguas): Q/K/V e gate/up
    /// fundidos (Phi-3) viram tensores independentes sem copiar nada.
    fn slice_rows(&self, start: usize, count: usize) -> Tensor {
        debug_assert!(self.rep.is_none(), "fatiar antes de reempacotar");
        Tensor { dtype: self.dtype, rows: count, cols: self.cols, off: self.off + start * self.row_bytes, row_bytes: self.row_bytes, rep: None }
    }

    /// Reempacota este tensor (se for Q4_K ou Q6_K elegível); devolve os bytes do mmap que ficaram sem uso.
    /// `BURITI_NO_REPACK_Q6=1` deixa os Q6_K no layout original (diagnóstico).
    fn try_repack(&mut self, mm: &[u8]) -> usize {
        if self.rep.is_some() {
            return 0;
        }
        let n = self.rows * self.row_bytes;
        self.rep = match self.dtype {
            Dtype::Q4K => Repacked::new(&mm[self.off..self.off + n], self.rows, self.cols),
            // Q5_K (D84): chave de diagnostico BURITI_NO_REPACK_Q5K
            Dtype::Q5K if super::repack::q5k_enabled() => Repacked::new_q5(&mm[self.off..self.off + n], self.rows, self.cols),
            Dtype::Q6K if super::repack::q6_enabled() => Repacked::new_q6(&mm[self.off..self.off + n], self.rows, self.cols),
            // formatos de bloco de 32 (D81); cada um tem a sua chave de diagnóstico BURITI_NO_REPACK_<formato>
            Dtype::Q8_0 if super::repack::enabled32(Kind32::Q8_0) => Repacked::new32(Kind32::Q8_0, &mm[self.off..self.off + n], self.rows, self.cols),
            Dtype::Q5_0 if super::repack::enabled32(Kind32::Q5_0) => Repacked::new32(Kind32::Q5_0, &mm[self.off..self.off + n], self.rows, self.cols),
            Dtype::Q4_0 if super::repack::enabled32(Kind32::Q4_0) => Repacked::new32(Kind32::Q4_0, &mm[self.off..self.off + n], self.rows, self.cols),
            _ => None,
        };
        if self.rep.is_some() { n } else { 0 }
    }

    #[inline]
    fn row<'a>(&self, mm: &'a [u8], r: usize) -> &'a [u8] {
        let s = self.off + r * self.row_bytes;
        &mm[s..s + self.row_bytes]
    }
}

struct Layer {
    /// normalização RMS por cabeça em Q e K (Qwen3, Gemma 3); vazio = sem
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// normas extras depois da atenção e da FFN, antes da soma residual (Gemma 3); vazio = sem
    post_attn_norm: Vec<f32>,
    post_ffn_norm: Vec<f32>,
    /// a atenção desta camada usa janela deslizante
    swa: bool,
    /// camada local do Gemma 3: RoPE com theta local
    local: bool,
    attn_norm: Vec<f32>,
    wq: Tensor,
    bq: Vec<f32>,
    wk: Tensor,
    bk: Vec<f32>,
    wv: Tensor,
    bv: Vec<f32>,
    wo: Tensor,
    ffn_norm: Vec<f32>,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
}

pub struct Model {
    pub cfg: Config,
    pub tok: Tokenizer,
    gguf: Gguf,
    layers: Vec<Layer>,
    embd: Tensor,
    output: Tensor,
    out_norm: Vec<f32>,
    inv_freq: Vec<f32>,
    /// frequências do RoPE das camadas locais (igual a `inv_freq` fora do Gemma 3)
    inv_freq_local: Vec<f32>,
    /// frequências do longrope para contexto acima de `n_ctx_orig` (vazio se o modelo não tem)
    inv_freq_long: Vec<f32>,
    n_ctx_orig: usize,
    /// pesos e extras do StarCoder2 (nesse caso `layers` fica vazio e `forward*` desviam para `starcoder2.rs`)
    sc2: Option<starcoder2::Sc2>,
}

/// Estado de uma conversa: KV cache + buffers de trabalho.
pub struct Session {
    pub tokens: Vec<u32>, // tokens cujo KV já está calculado (posições 0..len)
    /// chave (hash) das imagens cujos vetores estão no KV cache; 0 = nenhuma/desconhecida (D58: os ids de imagem são todos iguais,
    /// então o reaproveitamento de prefixo só vale além da 1ª posição de imagem se a chave coincidir)
    pub img_key: u64,
    /// quem escreveu o KV cache desta sessão (0 = ninguém ainda, 1 = CPU, 2 = GPU, D76): se o processador mudar, o cache não vale
    pub cache_owner: u8,
    ctx: usize,
    k: Vec<Vec<u16>>, // KV cache em f16 (metade da memória)
    v: Vec<Vec<u16>>,
    x: Vec<f32>,
    xn: Vec<f32>,
    tmp: Vec<f32>,
    q: Vec<f32>,
    kk: Vec<f32>,
    vv: Vec<f32>,
    att: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    pub logits: Vec<f32>,
}

impl Session {
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// O KV cache da CPU (K e V por camada), para os testes compararem byte a byte (D76).
    #[cfg(test)]
    pub(crate) fn kv(&self) -> (&[Vec<u16>], &[Vec<u16>]) {
        (&self.k, &self.v)
    }
}

impl Model {
    pub fn load(path: &Path) -> Result<Model, String> {
        Self::load_with(path, super::repack::enabled())
    }

    /// `repack`: reempacota os tensores Q4_K elegíveis das camadas (só faz efeito com AVX2/FMA/F16C).
    /// `load` decide por `repack::enabled()` (uma vez por processo); os testes escolhem à mão.
    pub fn load_with(path: &Path, repack: bool) -> Result<Model, String> {
        // com ativação em float (diagnóstico) não há kernel reempacotado
        let repack = repack && std::env::var_os("BURITI_FLOAT_ACT").is_none();
        let g = Gguf::open(path)?;
        let arch = g.meta.get("general.architecture").and_then(Value::as_str).unwrap_or("?").to_string();
        if arch == "starcoder2" {
            return starcoder2::load(g, repack);
        }
        // Qwen2: RoPE "NeoX" (pares i, i+hd/2), com viés em Q/K/V. Llama 3.x: RoPE de pares adjacentes
        // (os pesos Q/K do GGUF já vêm permutados para isso), sem viés, com fatores de frequência.
        // Qwen3: como o Qwen2 (RoPE NeoX), mas sem viés, com normalização RMS por cabeça em Q e K e
        // `head_dim` explícito no arquivo (pode diferir de n_embd / n_head).
        let rope_norm = match arch.as_str() {
            "qwen2" | "qwen3" | "gemma3" | "phi3" => false,
            "llama" => true,
            _ => {
                return Err(crate::errcode::coded(
                    "err_arch",
                    &[&arch],
                    format!("Architecture '{arch}' is not supported by the own engine yet (only qwen2/qwen3, llama / Llama 3.x, gemma3, phi3 and starcoder2). Use the llama.cpp engine."),
                ))
            }
        };
        let key = |k: &str| format!("{arch}.{k}");
        let n_layer = g.u64(&key("block_count"))? as usize;
        let n_embd = g.u64(&key("embedding_length"))? as usize;
        let n_ff = g.u64(&key("feed_forward_length"))? as usize;
        let n_head = g.u64(&key("attention.head_count"))? as usize;
        let n_kv = g.u64(&key("attention.head_count_kv"))? as usize;
        let n_ctx_train = g.u64(&key("context_length")).unwrap_or(4096) as usize;
        let eps = g.f64(&key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32;
        let theta = g.f64(&key("rope.freq_base")).unwrap_or(10000.0) as f32;
        if n_head == 0 || n_kv == 0 || n_head % n_kv != 0 {
            return Err("hiperparâmetros inconsistentes no modelo".into());
        }
        // head_dim: o valor explícito do arquivo (Qwen3, Gemma) ou n_embd / n_head
        let head_dim = match g.u64(&key("attention.key_length")) {
            Ok(k) => k as usize,
            Err(_) => {
                if n_embd % n_head != 0 {
                    return Err("hiperparâmetros inconsistentes no modelo (n_embd não divisível por n_head)".into());
                }
                n_embd / n_head
            }
        };
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err("head_dim ímpar não suportado".into());
        }
        if let Ok(x) = g.u64(&key("attention.value_length")) {
            if x as usize != head_dim {
                return Err(format!("{arch}.attention.value_length = {x} difere de head_dim = {head_dim}: não suportado pelo motor próprio"));
            }
        }
        // RoPE parcial (Phi-4-mini): só as primeiras `n_rot` dimensões de cada cabeça giram; o resto passa direto
        let n_rot = match g.u64(&key("rope.dimension_count")) {
            Ok(x) => x as usize,
            Err(_) => head_dim,
        };
        if n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim {
            return Err(format!("{arch}.rope.dimension_count = {n_rot} inválido para head_dim = {head_dim}"));
        }

        let tok = Tokenizer::from_gguf(&g)?;
        let n_vocab = tok.vocab_size();

        // Gemma 3: embedding x sqrt(n_embd), GeGLU, 2 normas extras por camada, janela deslizante nas camadas
        // locais (5 locais para 1 global) com theta de RoPE próprio.
        let gemma = arch == "gemma3";
        let window = if gemma { g.u64(&key("attention.sliding_window")).unwrap_or(0) as usize } else { 0 };
        let swa_pattern = if gemma { g.u64(&key("attention.sliding_window_pattern")).unwrap_or(6) as usize } else { 0 };
        let embed_scale = if gemma { (n_embd as f32).sqrt() } else { 1.0 };

        let t = |name: &str| load_tensor(&g, name);
        let v = |name: &str| load_vec(&g, name);
        // vieses de Q/K/V: o Qwen2 tem, o Llama não (vetor vazio = sem viés)
        let vopt = |name: &str| if g.tensors.contains_key(name) { load_vec(&g, name) } else { Ok(Vec::new()) };

        let embd = t("token_embd.weight")?;
        let output = match t("output.weight") {
            Ok(o) => o,
            Err(_) => t("token_embd.weight")?, // embeddings compartilhados
        };
        if embd.cols != n_embd || embd.rows < n_vocab || output.cols != n_embd {
            return Err("formato de token_embd/output inesperado".into());
        }

        // Phi-3: janela deslizante em TODAS as camadas (`sliding_window`, 2047 no mini-4k). O RoPE com
        // `longrope` (Phi-3.5, Phi-4-mini) ainda não é suportado: recusa em vez de rodar errado.
        let phi3 = arch == "phi3";
        let window = if phi3 { g.u64(&key("attention.sliding_window")).unwrap_or(0) as usize } else { window };

        let q_dim = n_head * head_dim;
        let kv_dim = n_kv * head_dim;
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            // Q, K e V podem vir fundidos num só tensor (Phi-3): linhas [Q | K | V]; gate e up também
            // (`ffn_up` com 2·n_ff linhas: [gate | up]). As linhas são contíguas, então são "fatias".
            let (wq, wk, wv) = match t(&p("attn_qkv.weight")) {
                Ok(qkv) if qkv.rows == q_dim + 2 * kv_dim => (qkv.slice_rows(0, q_dim), qkv.slice_rows(q_dim, kv_dim), qkv.slice_rows(q_dim + kv_dim, kv_dim)),
                Ok(_) => return Err(format!("attn_qkv com formato inesperado na camada {i}")),
                Err(_) => (t(&p("attn_q.weight"))?, t(&p("attn_k.weight"))?, t(&p("attn_v.weight"))?),
            };
            let (w_gate, w_up) = if g.tensors.contains_key(&p("ffn_gate.weight")) {
                (t(&p("ffn_gate.weight"))?, t(&p("ffn_up.weight"))?)
            } else {
                let gu = t(&p("ffn_up.weight"))?;
                if gu.rows != 2 * n_ff {
                    return Err(format!("ffn_up com formato inesperado na camada {i}"));
                }
                (gu.slice_rows(0, n_ff), gu.slice_rows(n_ff, n_ff))
            };
            let l = Layer {
                post_attn_norm: vopt(&p("post_attention_norm.weight"))?,
                post_ffn_norm: vopt(&p("post_ffw_norm.weight"))?,
                // janela deslizante: em todas as camadas (Phi-3) ou nas "locais" do Gemma 3; só o Gemma 3 também
                // usa outro theta de RoPE nessas camadas (`local`)
                swa: window > 0 && (phi3 || (swa_pattern > 0 && i % swa_pattern < swa_pattern - 1)),
                local: gemma && window > 0 && swa_pattern > 0 && i % swa_pattern < swa_pattern - 1,
                attn_norm: v(&p("attn_norm.weight"))?,
                wq,
                q_norm: vopt(&p("attn_q_norm.weight"))?,
                k_norm: vopt(&p("attn_k_norm.weight"))?,
                bq: vopt(&p("attn_q.bias"))?,
                wk,
                bk: vopt(&p("attn_k.bias"))?,
                wv,
                bv: vopt(&p("attn_v.bias"))?,
                wo: t(&p("attn_output.weight"))?,
                ffn_norm: v(&p("ffn_norm.weight"))?,
                w_gate,
                w_up,
                w_down: t(&p("ffn_down.weight"))?,
            };
            let ok = l.wq.cols == n_embd && l.wq.rows == q_dim
                && l.wk.rows == kv_dim && l.wv.rows == kv_dim
                && l.wo.cols == q_dim && l.wo.rows == n_embd
                && l.w_gate.cols == n_embd && l.w_gate.rows == n_ff
                && l.w_up.rows == n_ff && l.w_down.cols == n_ff && l.w_down.rows == n_embd
                && l.attn_norm.len() == n_embd && l.ffn_norm.len() == n_embd
                && (l.bq.is_empty() || l.bq.len() == q_dim)
                && (l.bk.is_empty() || l.bk.len() == kv_dim)
                && (l.bv.is_empty() || l.bv.len() == kv_dim)
                && (l.q_norm.is_empty() || l.q_norm.len() == head_dim)
                && (l.k_norm.is_empty() || l.k_norm.len() == head_dim)
                && (l.post_attn_norm.is_empty() || l.post_attn_norm.len() == n_embd)
                && (l.post_ffn_norm.is_empty() || l.post_ffn_norm.len() == n_embd);
            if !ok {
                return Err(format!("formato inesperado de tensores na camada {i}"));
            }
            layers.push(l);
        }
        let out_norm = v("output_norm.weight")?;
        // Reempacotamento Q4_K (E5/D48), uma vez, aqui: pesos das camadas, nunca o embedding/saída.
        let (mut n_rep, mut rep_src, mut rep_dst) = (0usize, 0usize, 0usize);
        let mut by_dtype = vec![0usize; Dtype::COUNT];
        let mut released: Vec<(usize, usize)> = Vec::new();
        if repack {
            let t0 = std::time::Instant::now();
            for l in layers.iter_mut() {
                for t in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.w_gate, &mut l.w_up, &mut l.w_down] {
                    let b = t.try_repack(&g.mmap);
                    if b > 0 {
                        n_rep += 1;
                        by_dtype[t.dtype as usize] += 1;
                        rep_src += b;
                        rep_dst += t.rep.as_ref().map_or(0, Repacked::size_bytes);
                        released.push((t.off, b));
                    }
                }
            }
            // As páginas originais desses tensores não serão mais lidas: tira do conjunto de trabalho (só Windows;
            // continuam mapeadas e legíveis, apenas saem da RAM do processo até alguém tocá-las de novo).
            let freed: usize = released.iter().map(|&(o, n)| release_pages(&g.mmap, o, n)).sum();
            if n_rep > 0 && std::env::var_os("BURITI_DIAG").is_some() {
                eprintln!("[native] páginas do mmap liberadas do conjunto de trabalho: {:.1} MB", freed as f64 / 1e6);
                let kinds: Vec<String> = by_dtype.iter().enumerate().filter(|(_, n)| **n > 0).map(|(i, n)| format!("{n} {}", Dtype::NAMES[i])).collect();
                eprintln!("[native] reempacotados {n_rep} tensores ({}): {:.1} MB -> {:.1} MB em {:.2} s", kinds.join(", "), rep_src as f64 / 1e6, rep_dst as f64 / 1e6, t0.elapsed().as_secs_f64());
            }
        }

        let half = n_rot / 2;
        let base: Vec<f32> = (0..half).map(|i| theta.powf(-(2.0 * i as f32) / n_rot as f32)).collect();
        // Llama 3.1/3.2: `rope_freqs.weight` guarda um fator por frequência (a escala para contextos
        // longos); a frequência efetiva é theta^(-2i/n) / fator. Sem o tensor, fator 1.
        let factors = if g.tensors.contains_key("rope_freqs.weight") { v("rope_freqs.weight")? } else { vec![1.0; half] };
        if factors.len() != half {
            return Err(format!("rope_freqs.weight com {} valores; esperado {half}", factors.len()));
        }
        // Phi-3.5 / Phi-4-mini ("longrope"): dois conjuntos de fatores, `short` (contexto configurado até o
        // original) e `long` (acima dele), escolhidos como no llama.cpp pelo contexto configurado.
        let rope_factors = |name: &str| -> Result<Option<Vec<f32>>, String> {
            if !g.tensors.contains_key(name) {
                return Ok(None);
            }
            let f = load_vec(&g, name)?;
            if f.len() != half {
                return Err(format!("{name} com {} valores; esperado {half}", f.len()));
            }
            Ok(Some(f))
        };
        let (short, long) = (rope_factors("rope_factors_short.weight")?, rope_factors("rope_factors_long.weight")?);
        let mut inv_freq: Vec<f32> = match &short {
            Some(s) => (0..half).map(|i| base[i] / s[i]).collect(),
            None => (0..half).map(|i| base[i] / factors[i]).collect(),
        };
        let inv_freq_long: Vec<f32> = match &long {
            Some(l) => (0..half).map(|i| base[i] / l[i]).collect(),
            None => Vec::new(),
        };
        let n_ctx_orig = g.u64(&key("rope.scaling.original_context_length")).unwrap_or(n_ctx_train as u64) as usize;
        // fator de escala da rotação (cos e sen multiplicados): `attn_factor` do arquivo, 1 se ausente
        let rope_mscale = g.f64(&key("rope.scaling.attn_factor")).unwrap_or(1.0) as f32;
        // escala linear do RoPE das camadas globais (Gemma 3 de 4B para cima)
        if g.meta.get(&key("rope.scaling.type")).and_then(Value::as_str) == Some("linear") {
            if let Some(f) = g.f64(&key("rope.scaling.factor")) {
                inv_freq.iter_mut().for_each(|x| *x /= f as f32);
            }
        }
        let inv_freq_local: Vec<f32> = if gemma {
            let theta_local = g.f64(&key("rope.freq_base_swa")).unwrap_or(10000.0) as f32;
            (0..half).map(|i| theta_local.powf(-(2.0 * i as f32) / n_rot as f32)).collect()
        } else {
            inv_freq.clone()
        };

        Ok(Model {
            cfg: Config { n_layer, n_embd, n_ff, n_head, n_kv, head_dim, eps, n_ctx_train, n_vocab, rope_norm, embed_scale, gelu: gemma, window, n_rot, rope_mscale },
            tok,
            gguf: g,
            layers,
            embd,
            output,
            out_norm,
            inv_freq,
            inv_freq_local,
            inv_freq_long,
            n_ctx_orig,
            sc2: None,
        })
    }

    /// Frequências do RoPE das camadas globais: com `longrope`, o conjunto `long` quando o contexto
    /// configurado passa do original de treino, senão o `short` (como o llama.cpp).
    fn inv_global(&self, ctx: usize) -> &[f32] {
        if !self.inv_freq_long.is_empty() && ctx > self.n_ctx_orig {
            &self.inv_freq_long
        } else {
            &self.inv_freq
        }
    }

    pub fn new_session(&self, ctx: usize) -> Session {
        let c = &self.cfg;
        let kv_dim = c.n_kv * c.head_dim;
        Session {
            tokens: Vec::new(),
            img_key: 0,
            cache_owner: 0,
            ctx,
            k: (0..c.n_layer).map(|_| vec![0u16; ctx * kv_dim]).collect(),
            v: (0..c.n_layer).map(|_| vec![0u16; ctx * kv_dim]).collect(),
            x: vec![0.0; c.n_embd],
            xn: vec![0.0; c.n_embd],
            tmp: vec![0.0; c.n_embd],
            q: vec![0.0; c.n_head * c.head_dim],
            kk: vec![0.0; kv_dim],
            vv: vec![0.0; kv_dim],
            att: vec![0.0; c.n_head * c.head_dim],
            gate: vec![0.0; c.n_ff],
            up: vec![0.0; c.n_ff],
            scores: vec![0.0; (c.n_head / c.n_kv) * ctx],
            logits: vec![0.0; c.n_vocab],
        }
    }

    /// out = W · x, paralelizado por linhas (rodar dentro de `pool.install`).
    /// `ac` guarda a entrada quantizada: multiplicações seguidas sobre o mesmo `x` a reaproveitam
    /// (o chamador dá `ac.clear()` quando `x` muda).
    fn matvec(&self, t: &Tensor, x: &[f32], out: &mut [f32], ac: &mut ActCache) {
        debug_assert_eq!(x.len(), t.cols);
        debug_assert!(out.len() <= t.rows);
        let mm = &self.gguf.mmap[..];
        let ta = std::time::Instant::now();
        let act = ac.get(t.dtype, x); // entrada quantizada em int8
        PH_NS[0].fetch_add(ta.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        if let Some(rp) = &t.rep {
            // Q4_K reempacotado: grupos de 8 linhas, uma ativação (mesmas contas do x4/x8 do matmat)
            let a = act.expect("tensor reempacotado exige ativação int8");
            assert_eq!(out.len(), t.rows);
            out.par_chunks_mut(16).enumerate().for_each(|(ci, chunk)| {
                for (gi, o8) in chunk.chunks_exact_mut(8).enumerate() {
                    let mut r = [[0f32; 8]; 1];
                    rp.gemm::<1>(ci * 2 + gi, [a], &mut r);
                    o8.copy_from_slice(&r[0]);
                }
            });
            let i = t.dtype as usize;
            MV_NS[i].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            MV_BYTES[i].fetch_add(rp.size_bytes() as u64, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        out.par_chunks_mut(16).enumerate().for_each(|(ci, chunk)| {
            for (j, o) in chunk.iter_mut().enumerate() {
                let row = t.row(mm, ci * 16 + j);
                *o = match act {
                    Some(a) => dot_row_q(t.dtype, row, a),
                    None => dot_row(t.dtype, row, x),
                };
            }
        });
        let i = t.dtype as usize;
        MV_NS[i].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        MV_BYTES[i].fetch_add((t.rows * t.row_bytes) as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// out[i] = W · xs[i] para `n` vetores de uma vez (`xs`: n×cols, `out`: n×rows_out, token a token).
    /// Cada linha de pesos é lida uma vez e usada para os `n` vetores: o prefill deixa de ser
    /// limitado pela banda da RAM.
    fn matmat(&self, t: &Tensor, xs: &[f32], n: usize, rows_out: usize, out: &mut [f32]) {
        let cols = t.cols;
        debug_assert_eq!(xs.len(), n * cols);
        debug_assert_eq!(out.len(), n * rows_out);
        debug_assert!(rows_out <= t.rows);
        let mm = &self.gguf.mmap[..];
        let t0 = std::time::Instant::now();
        let acts: Vec<Option<Act>> = (0..n).into_par_iter().map(|i| Act::new(t.dtype, &xs[i * cols..(i + 1) * cols])).collect();
        PH_NS[4].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let quantized = acts.iter().all(Option::is_some);
        if let Some(rp) = &t.rep {
            assert!(quantized && rows_out == t.rows, "tensor reempacotado exige ativação int8 e todas as linhas");
            // Tarefas por conjunto de grupos de 8 linhas (~8 KB de pesos por tarefa, como no caminho antigo),
            // lote de 8, 4 e 1 ativações: o mesmo resultado, bit a bit, para cada ativação em qualquer variante.
            let (mm_bytes, _) = matmat_blocking();
            let gpt = (mm_bytes / (8 * t.row_bytes.max(1))).max(1);
            let ngroups = rows_out / 8;
            let optr = SyncPtr(out.as_mut_ptr());
            (0..ngroups.div_ceil(gpt)).into_par_iter().for_each(|ci| {
                let a = |i: usize| acts[i].as_ref().unwrap();
                for g in ci * gpt..(ci * gpt + gpt).min(ngroups) {
                    // SAFETY: `out` tem n·rows_out elementos, exclusivo desta chamada; cada tarefa só escreve
                    // nas colunas 8g..8g+8 dos seus grupos (disjuntos entre tarefas).
                    let put = |tok: usize, r: &[f32; 8]| unsafe { std::ptr::copy_nonoverlapping(r.as_ptr(), optr.ptr().add(tok * rows_out + g * 8), 8) };
                    let mut i = 0;
                    while i + 8 <= n {
                        let mut r = [[0f32; 8]; 8];
                        rp.gemm::<8>(g, std::array::from_fn(|k| a(i + k)), &mut r);
                        for k in 0..8 {
                            put(i + k, &r[k]);
                        }
                        i += 8;
                    }
                    while i + 4 <= n {
                        let mut r = [[0f32; 8]; 4];
                        rp.gemm::<4>(g, std::array::from_fn(|k| a(i + k)), &mut r);
                        for k in 0..4 {
                            put(i + k, &r[k]);
                        }
                        i += 4;
                    }
                    while i < n {
                        let mut r = [[0f32; 8]; 1];
                        rp.gemm::<1>(g, [a(i)], &mut r);
                        put(i, &r[0]);
                        i += 1;
                    }
                }
            });
            let di = t.dtype as usize;
            MV_NS[di].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            MV_BYTES[di].fetch_add(rp.size_bytes() as u64, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        // Blocagem para o cache L1: um punhado de linhas de pesos (~8 KB) fica quente enquanto
        // percorremos os tokens de 8 em 8; as ativações do grupo (8 × cols bytes) também cabem.
        let (mm_bytes, mm_rows) = matmat_blocking();
        let rpc = (mm_bytes / t.row_bytes.max(1)).clamp(1, mm_rows);
        let ntasks = rows_out.div_ceil(rpc);
        // Cada tarefa é dona de um conjunto de LINHAS (de pesos) e escreve direto no destino na ordem
        // [token][linha]: sem buffer intermediário nem transposição.
        let optr = SyncPtr(out.as_mut_ptr());
        (0..ntasks).into_par_iter().for_each(|ci| {
            let r0 = ci * rpc;
            let nr = rpc.min(rows_out - r0);
            let row_of = |j: usize| t.row(mm, r0 + j);
            // SAFETY: `out` tem n·rows_out elementos e é exclusivo desta chamada; esta tarefa só
            // escreve nas colunas r0..r0+nr de cada token, e as tarefas têm intervalos de linhas disjuntos.
            let put = |tok: usize, j: usize, v: f32| unsafe { *optr.ptr().add(tok * rows_out + r0 + j) = v };
            if !quantized {
                for j in 0..nr {
                    for i in 0..n {
                        put(i, j, dot_row(t.dtype, row_of(j), &xs[i * cols..(i + 1) * cols]));
                    }
                }
                return;
            }
            let a = |i: usize| acts[i].as_ref().unwrap();
            let mut i = 0;
            while i + 8 <= n {
                // 8 tokens por passada (formatos K: os pesos são desempacotados uma vez para os 8)
                let g = [a(i), a(i + 1), a(i + 2), a(i + 3), a(i + 4), a(i + 5), a(i + 6), a(i + 7)];
                for j in 0..nr {
                    let r = dot_row_q_x8(t.dtype, row_of(j), g);
                    for k in 0..8 {
                        put(i + k, j, r[k]);
                    }
                }
                i += 8;
            }
            while i + 4 <= n {
                // 4 tokens por passada: os pesos são desempacotados uma vez só
                let g = [a(i), a(i + 1), a(i + 2), a(i + 3)];
                for j in 0..nr {
                    let r = dot_row_q_x4(t.dtype, row_of(j), g);
                    for k in 0..4 {
                        put(i + k, j, r[k]);
                    }
                }
                i += 4;
            }
            while i < n {
                for j in 0..nr {
                    put(i, j, dot_row_q(t.dtype, row_of(j), a(i)));
                }
                i += 1;
            }
        });
        let di = t.dtype as usize;
        MV_NS[di].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        MV_BYTES[di].fetch_add((t.rows * t.row_bytes) as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Processa `tokens` (posições `start..start+n`) de uma vez — usado no prefill. Escreve o KV
    /// dessas posições; com `want_logits`, deixa em `s.logits` as probabilidades do que vem depois
    /// do ÚLTIMO token. Equivale a chamar `forward` token a token (há teste comparando).
    pub fn forward_batch(&self, s: &mut Session, tokens: &[u32], start: usize, want_logits: bool) {
        self.forward_batch_embd(s, tokens, &[], start, want_logits)
    }

    /// Como `forward_batch`, com **embeddings prontos** no lugar do embedding de alguns tokens (visão, D50/M4):
    /// `embds` = (índice dentro de `tokens`, vetor de `n_embd` valores). O vetor entra DEPOIS da escala do embedding
    /// (o Gemma multiplica só o embedding de token por √n_embd; o llama.cpp também não escala a entrada por embedding
    /// [Suposto]) e o token naquele índice (o marcador `<image>`) só ocupa o lugar na sequência. Posição, RoPE,
    /// KV e atenção são os de qualquer token. Sem `embds`, é exatamente `forward_batch`.
    pub fn forward_batch_embd(&self, s: &mut Session, tokens: &[u32], embds: &[(usize, &[f32])], start: usize, want_logits: bool) {
        self.forward_batch_core(s, tokens, embds, start, want_logits, None)
    }

    /// Estados finais (depois da última camada e da norma de saída, o "result_norm" do llama.cpp) de TODOS os `tokens`
    /// do lote, anexados a `out` (n × n_embd). Para embeddings (`/v1/embeddings`): não calcula a matriz do vocabulário.
    /// A passagem é a mesma de `forward_batch` (mesmas contas, mesmo KV). StarCoder2 não é suportado.
    pub fn forward_hidden(&self, s: &mut Session, tokens: &[u32], start: usize, out: &mut Vec<f32>) -> Result<(), String> {
        if self.sc2.is_some() {
            return Err("embeddings não são suportados para o StarCoder2 no motor próprio".into());
        }
        self.forward_batch_core(s, tokens, &[], start, false, Some(out));
        Ok(())
    }

    /// Metadados do GGUF (só leitura): o módulo de embeddings lê `<arq>.pooling_type` e `tokenizer.ggml.add_eos_token`.
    pub fn gguf_meta(&self) -> &std::collections::HashMap<String, Value> {
        &self.gguf.meta
    }

    fn forward_batch_core(&self, s: &mut Session, tokens: &[u32], embds: &[(usize, &[f32])], start: usize, want_logits: bool, hidden: Option<&mut Vec<f32>>) {
        let n = tokens.len();
        if n == 0 {
            return;
        }
        if let Some(sc) = &self.sc2 {
            assert!(embds.is_empty(), "entrada por embedding não suportada no StarCoder2");
            return self.forward_batch_sc2(sc, s, tokens, start, want_logits);
        }
        assert!(embds.iter().all(|(i, v)| *i < n && v.len() == self.cfg.n_embd), "embedding de imagem com índice ou tamanho inválido");
        let c = &self.cfg;
        let (hd, n_head, n_kv, n_embd) = (c.head_dim, c.n_head, c.n_kv, c.n_embd);
        let (q_dim, kv_dim) = (n_head * hd, n_kv * hd);
        let group = n_head / n_kv;
        let mm = &self.gguf.mmap[..];
        assert!(start + n <= s.ctx && tokens.iter().all(|&t| (t as usize) < c.n_vocab));

        let mut x = vec![0.0f32; n * n_embd];
        for (i, &tok) in tokens.iter().enumerate() {
            dequant_row(self.embd.dtype, self.embd.row(mm, tok as usize), &mut x[i * n_embd..(i + 1) * n_embd]);
        }
        if c.embed_scale != 1.0 {
            x.iter_mut().for_each(|v| *v *= c.embed_scale); // Gemma: embedding · sqrt(n_embd)
        }
        for (i, v) in embds {
            x[i * n_embd..(i + 1) * n_embd].copy_from_slice(v);
        }
        let mut xn = vec![0.0f32; n * n_embd];
        let mut q = vec![0.0f32; n * q_dim];
        let mut kk = vec![0.0f32; n * kv_dim];
        let mut vv = vec![0.0f32; n * kv_dim];
        let mut att = vec![0.0f32; n * q_dim];
        let mut tmp = vec![0.0f32; n * n_embd];
        let mut gate = vec![0.0f32; n * c.n_ff];
        let mut up = vec![0.0f32; n * c.n_ff];
        let scale = 1.0 / (hd as f32).sqrt();
        let ctx = s.ctx;
        // seno e cosseno de cada posição do lote, calculados uma vez (iguais para todas as cabeças e camadas)
        let half = c.n_rot / 2; // RoPE parcial: só n_rot dimensões giram
        let m = c.rope_mscale;
        let inv_g = self.inv_global(ctx);
        let has_local = self.layers.iter().any(|l| l.local);
        let mut cs = vec![(0f32, 0f32); n * half];
        let mut cs_local = vec![(0f32, 0f32); if has_local { n * half } else { 0 }]; // camadas locais (Gemma 3)
        for i in 0..n {
            for j in 0..half {
                let (s1, c1) = ((start + i) as f32 * inv_g[j]).sin_cos();
                cs[i * half + j] = (s1 * m, c1 * m);
                if has_local {
                    let (s2, c2) = ((start + i) as f32 * self.inv_freq_local[j]).sin_cos();
                    cs_local[i * half + j] = (s2 * m, c2 * m);
                }
            }
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- atenção
            for i in 0..n {
                rmsnorm(&mut xn[i * n_embd..(i + 1) * n_embd], &x[i * n_embd..(i + 1) * n_embd], &l.attn_norm, c.eps);
            }
            self.matmat(&l.wq, &xn, n, q_dim, &mut q);
            self.matmat(&l.wk, &xn, n, kv_dim, &mut kk);
            self.matmat(&l.wv, &xn, n, kv_dim, &mut vv);
            for i in 0..n {
                let pos = start + i;
                let (qi, ki, vi) = (&mut q[i * q_dim..(i + 1) * q_dim], &mut kk[i * kv_dim..(i + 1) * kv_dim], &mut vv[i * kv_dim..(i + 1) * kv_dim]);
                for (a, b) in qi.iter_mut().zip(&l.bq) {
                    *a += b;
                }
                for (a, b) in ki.iter_mut().zip(&l.bk) {
                    *a += b;
                }
                for (a, b) in vi.iter_mut().zip(&l.bv) {
                    *a += b;
                }
                // Qwen3/Gemma 3: RMSNorm por cabeça em Q e K, antes do RoPE (mesma conta do caminho de um token)
                if !l.q_norm.is_empty() {
                    for h in 0..n_head {
                        rmsnorm_inplace(&mut qi[h * hd..(h + 1) * hd], &l.q_norm, c.eps);
                    }
                }
                if !l.k_norm.is_empty() {
                    for h in 0..n_kv {
                        rmsnorm_inplace(&mut ki[h * hd..(h + 1) * hd], &l.k_norm, c.eps);
                    }
                }
                let table = if l.local { &cs_local } else { &cs };
                let cs_i = &table[i * half..(i + 1) * half];
                for h in 0..n_head {
                    rope_cs(&mut qi[h * hd..(h + 1) * hd], cs_i, c.rope_norm);
                }
                for h in 0..n_kv {
                    rope_cs(&mut ki[h * hd..(h + 1) * hd], cs_i, c.rope_norm);
                }
                store_f16(ki, &mut s.k[li][pos * kv_dim..(pos + 1) * kv_dim]);
                store_f16(vi, &mut s.v[li][pos * kv_dim..(pos + 1) * kv_dim]);
            }
            // máscara causal: o token i só enxerga as posições 0..=start+i, então escrever o KV
            // do lote inteiro antes não muda nada
            let (kc, vc) = (&s.k[li], &s.v[li]);
            let t_att = std::time::Instant::now();
            // uma tarefa por (token, cabeça de KV): atende as `group` cabeças de consulta que a
            // compartilham, lendo cada K e V uma vez só
            att.par_chunks_mut(group * hd).enumerate().for_each_init(
                || vec![0.0f32; group * ctx],
                |sc, (idx, out)| {
                    let (i, g) = (idx / n_kv, idx % n_kv);
                    let qg = &q[i * q_dim + g * group * hd..i * q_dim + (g + 1) * group * hd];
                    // camada local (Gemma 3): só as últimas `window` posições; passa o cache a partir do
                    // início da janela e a posição relativa
                    let pos = start + i;
                    let t0 = if l.swa { (pos + 1).saturating_sub(c.window) } else { 0 };
                    attend_group(qg, &kc[t0 * kv_dim..], &vc[t0 * kv_dim..], pos - t0, g, group, hd, kv_dim, scale, sc, out);
                },
            );
            PH_NS[1].fetch_add(t_att.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            self.matmat(&l.wo, &att, n, n_embd, &mut tmp);
            if !l.post_attn_norm.is_empty() {
                // Gemma 3: norma depois da atenção, antes da soma residual
                for t in tmp.chunks_mut(n_embd) {
                    rmsnorm_inplace(t, &l.post_attn_norm, c.eps);
                }
            }
            for (a, d) in x.iter_mut().zip(&tmp) {
                *a += d;
            }

            // ---- feed-forward (SwiGLU, ou GeGLU no Gemma)
            for i in 0..n {
                rmsnorm(&mut xn[i * n_embd..(i + 1) * n_embd], &x[i * n_embd..(i + 1) * n_embd], &l.ffn_norm, c.eps);
            }
            self.matmat(&l.w_gate, &xn, n, c.n_ff, &mut gate);
            self.matmat(&l.w_up, &xn, n, c.n_ff, &mut up);
            let t_silu = std::time::Instant::now();
            if c.gelu {
                gate.par_chunks_mut(2048).zip(up.par_chunks(2048)).for_each(|(gc, uc)| gelu_mul(gc, uc));
            } else {
                gate.par_chunks_mut(2048).zip(up.par_chunks(2048)).for_each(|(gc, uc)| silu_mul(gc, uc));
            }
            PH_NS[2].fetch_add(t_silu.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            self.matmat(&l.w_down, &gate, n, n_embd, &mut tmp);
            if !l.post_ffn_norm.is_empty() {
                for t in tmp.chunks_mut(n_embd) {
                    rmsnorm_inplace(t, &l.post_ffn_norm, c.eps);
                }
            }
            for (a, d) in x.iter_mut().zip(&tmp) {
                *a += d;
            }
        }

        if let Some(h) = hidden {
            let base = h.len();
            h.resize(base + n * n_embd, 0.0);
            for i in 0..n {
                rmsnorm(&mut h[base + i * n_embd..base + (i + 1) * n_embd], &x[i * n_embd..(i + 1) * n_embd], &self.out_norm, c.eps);
            }
        }
        if want_logits {
            // só o último token precisa dos logits (a matriz do vocabulário é a maior de todas)
            rmsnorm(&mut s.xn, &x[(n - 1) * n_embd..], &self.out_norm, c.eps);
            self.matvec(&self.output, &s.xn, &mut s.logits, &mut ActCache::default());
        }
    }

    /// Processa UM token na posição `pos` (escreve o KV dessa posição). Com `want_logits`,
    /// deixa as probabilidades do próximo token em `s.logits`.
    pub fn forward(&self, s: &mut Session, token: u32, pos: usize, want_logits: bool) {
        if let Some(sc) = &self.sc2 {
            return self.forward_sc2(sc, s, token, pos, want_logits);
        }
        let c = &self.cfg;
        let (hd, n_head, n_kv) = (c.head_dim, c.n_head, c.n_kv);
        let kv_dim = n_kv * hd;
        let group = n_head / n_kv;
        let mm = &self.gguf.mmap[..];
        assert!(pos < s.ctx && (token as usize) < c.n_vocab);
        let ctx_cfg = s.ctx; // contexto configurado: escolhe os fatores do longrope
        let mut ac = ActCache::default();

        dequant_row(self.embd.dtype, self.embd.row(mm, token as usize), &mut s.x);
        if c.embed_scale != 1.0 {
            s.x.iter_mut().for_each(|v| *v *= c.embed_scale); // Gemma: embedding · sqrt(n_embd)
        }

        for (li, l) in self.layers.iter().enumerate() {
            // ---- atenção
            rmsnorm(&mut s.xn, &s.x, &l.attn_norm, c.eps);
            ac.clear();
            self.matvec(&l.wq, &s.xn, &mut s.q, &mut ac);
            self.matvec(&l.wk, &s.xn, &mut s.kk, &mut ac);
            self.matvec(&l.wv, &s.xn, &mut s.vv, &mut ac);
            for (a, b) in s.q.iter_mut().zip(&l.bq) {
                *a += b;
            }
            for (a, b) in s.kk.iter_mut().zip(&l.bk) {
                *a += b;
            }
            for (a, b) in s.vv.iter_mut().zip(&l.bv) {
                *a += b;
            }
            if !l.q_norm.is_empty() {
                for h in 0..n_head {
                    rmsnorm_inplace(&mut s.q[h * hd..(h + 1) * hd], &l.q_norm, c.eps);
                }
            }
            if !l.k_norm.is_empty() {
                for h in 0..n_kv {
                    rmsnorm_inplace(&mut s.kk[h * hd..(h + 1) * hd], &l.k_norm, c.eps);
                }
            }
            let inv = if l.local { &self.inv_freq_local[..] } else { self.inv_global(ctx_cfg) };
            for h in 0..n_head {
                rope(&mut s.q[h * hd..(h + 1) * hd], pos, inv, c.rope_norm, c.rope_mscale);
            }
            for h in 0..n_kv {
                rope(&mut s.kk[h * hd..(h + 1) * hd], pos, inv, c.rope_norm, c.rope_mscale);
            }
            store_f16(&s.kk, &mut s.k[li][pos * kv_dim..(pos + 1) * kv_dim]);
            store_f16(&s.vv, &mut s.v[li][pos * kv_dim..(pos + 1) * kv_dim]);

            let scale = 1.0 / (hd as f32).sqrt();
            let t_att = std::time::Instant::now();
            // camada local (Gemma 3): só as últimas `window` posições (cache a partir do início da janela)
            let t0 = if l.swa { (pos + 1).saturating_sub(c.window) } else { 0 };
            for g in 0..n_kv {
                // as mesmas contas do lote (`attend_group`): o resultado é idêntico
                let (a, b) = (g * group * hd, (g + 1) * group * hd);
                attend_group(&s.q[a..b], &s.k[li][t0 * kv_dim..], &s.v[li][t0 * kv_dim..], pos - t0, g, group, hd, kv_dim, scale, &mut s.scores, &mut s.att[a..b]);
            }
            PH_NS[1].fetch_add(t_att.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            ac.clear();
            self.matvec(&l.wo, &s.att, &mut s.tmp, &mut ac);
            if !l.post_attn_norm.is_empty() {
                rmsnorm_inplace(&mut s.tmp, &l.post_attn_norm, c.eps); // Gemma 3
            }
            for (x, d) in s.x.iter_mut().zip(&s.tmp) {
                *x += d;
            }

            // ---- feed-forward (SwiGLU, ou GeGLU no Gemma)
            rmsnorm(&mut s.xn, &s.x, &l.ffn_norm, c.eps);
            ac.clear();
            self.matvec(&l.w_gate, &s.xn, &mut s.gate, &mut ac);
            self.matvec(&l.w_up, &s.xn, &mut s.up, &mut ac);
            let t_silu = std::time::Instant::now();
            if c.gelu {
                gelu_mul(&mut s.gate, &s.up); // gelu(g) * u
            } else {
                silu_mul(&mut s.gate, &s.up); // silu(g) * u
            }
            PH_NS[2].fetch_add(t_silu.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            ac.clear();
            self.matvec(&l.w_down, &s.gate, &mut s.tmp, &mut ac);
            if !l.post_ffn_norm.is_empty() {
                rmsnorm_inplace(&mut s.tmp, &l.post_ffn_norm, c.eps);
            }
            for (x, d) in s.x.iter_mut().zip(&s.tmp) {
                *x += d;
            }
        }

        if want_logits {
            rmsnorm(&mut s.xn, &s.x, &self.out_norm, c.eps);
            // só as linhas do vocabulário (o tensor pode ter mais)
            ac.clear();
            self.matvec(&self.output, &s.xn, &mut s.logits, &mut ac);
        }
    }
}

/// Diagnóstico das fases fora das multiplicações (só na passada de um token): 0 = quantização da
/// ativação, 1 = atenção, 2 = SwiGLU (silu), 3 = amostragem (preenchido em mod.rs).
pub static PH_NS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// Diagnóstico por tipo de quantização (índice = `Dtype as usize`): ns e bytes de pesos lidos.
pub static MV_NS: [std::sync::atomic::AtomicU64; Dtype::COUNT] = [const { std::sync::atomic::AtomicU64::new(0) }; Dtype::COUNT];
pub static MV_BYTES: [std::sync::atomic::AtomicU64; Dtype::COUNT] = [const { std::sync::atomic::AtomicU64::new(0) }; Dtype::COUNT];

/// RMSNorm no próprio vetor (normas por cabeça de Q/K). A mesma conta de `rmsnorm`.
fn rmsnorm_inplace(v: &mut [f32], w: &[f32], eps: f32) {
    let ss = dot_f32(v, v) / v.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for (x, wv) in v.iter_mut().zip(w) {
        *x = *x * inv * wv;
    }
}

pub(crate) fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ss = dot_f32(x, x) / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((o, xv), wv) in out.iter_mut().zip(x).zip(w) {
        *o = xv * inv * wv;
    }
}

/// Rotaciona o par de índice `i`: com `norm` (Llama) os elementos adjacentes (2i, 2i+1); sem (Qwen2/NeoX),
/// os elementos (i, i + hd/2).
#[inline(always)]
fn rot_pair(v: &mut [f32], i: usize, half: usize, sin: f32, cos: f32, norm: bool) {
    let (ia, ib) = if norm { (2 * i, 2 * i + 1) } else { (i, i + half) };
    let (a, b) = (v[ia], v[ib]);
    v[ia] = a * cos - b * sin;
    v[ib] = a * sin + b * cos;
}

/// Mesmo que `rope`, com os pares (seno, cosseno) já calculados para a posição.
fn rope_cs(v: &mut [f32], cs: &[(f32, f32)], norm: bool) {
    let half = cs.len(); // n_rot / 2: só as primeiras n_rot dimensões giram (RoPE parcial)
    for i in 0..half {
        let (sin, cos) = cs[i];
        rot_pair(v, i, half, sin, cos, norm);
    }
}

/// RoPE: rotaciona cada par pelo ângulo `pos * inv_freq[i]`.
pub(crate) fn rope(v: &mut [f32], pos: usize, inv_freq: &[f32], norm: bool, mscale: f32) {
    let half = inv_freq.len();
    for i in 0..half {
        let (sin, cos) = (pos as f32 * inv_freq[i]).sin_cos();
        let (sin, cos) = (sin * mscale, cos * mscale);
        rot_pair(v, i, half, sin, cos, norm);
    }
}

fn load_tensor(g: &Gguf, name: &str) -> Result<Tensor, String> {
    let info = g.tensors.get(name).ok_or_else(|| format!("tensor ausente: {name}"))?;
    let dtype = Gguf::dtype_of(info).ok_or_else(|| crate::errcode::coded("err_quant", &[name, &info.ggml_type.to_string()], format!("tensor {name}: quantization type {} is not supported by the own engine yet", info.ggml_type)))?;
    let cols = info.dims[0];
    let rows: usize = info.dims[1..].iter().product();
    let (be, bb) = dtype.block();
    if cols % be != 0 {
        return Err(format!("tensor {name}: {cols} colunas não é múltiplo do bloco {be}"));
    }
    let row_bytes = cols / be * bb;
    let end = rows.checked_mul(row_bytes).and_then(|n| n.checked_add(info.offset));
    if end.is_none_or(|e| e > g.mmap.len()) {
        return Err(format!("tensor {name} ultrapassa o tamanho do arquivo"));
    }
    Ok(Tensor { dtype, rows, cols, off: info.offset, row_bytes, rep: None })
}

/// Dispensa do conjunto de trabalho do processo as páginas inteiras de `mm[off..off+n]` (Windows:
/// `VirtualUnlock` em páginas não travadas as remove do conjunto de trabalho; o erro NOT_LOCKED é o
/// esperado). É só um conselho: o conteúdo continua acessível (relido do arquivo se preciso). Devolve os
/// bytes dispensados (0 fora do Windows ou se falhar).
fn release_pages(mm: &[u8], off: usize, n: usize) -> usize {
    #[cfg(windows)]
    {
        extern "system" {
            fn VirtualUnlock(addr: *mut core::ffi::c_void, size: usize) -> i32;
        }
        const PAGE: usize = 4096;
        let base = mm.as_ptr() as usize;
        let start = (base + off).next_multiple_of(PAGE);
        let end = (base + off + n) / PAGE * PAGE;
        if end > start {
            // SAFETY: intervalo dentro do mapeamento; VirtualUnlock não altera conteúdo nem proteção.
            unsafe { VirtualUnlock(start as *mut core::ffi::c_void, end - start) };
            return end - start;
        }
    }
    let _ = (mm, off, n);
    0
}

/// Vetor f32 (normas e vieses) — F32 ou F16.
fn load_vec(g: &Gguf, name: &str) -> Result<Vec<f32>, String> {
    let t = load_tensor(g, name)?;
    let n = t.cols * t.rows;
    let bytes = &g.mmap[t.off..t.off + n * t.dtype.block().1];
    match t.dtype {
        Dtype::F32 => Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        Dtype::F16 => Ok(bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
        Dtype::BF16 => Ok(bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()),
        _ => Err(format!("tensor {name}: esperado F32/F16/BF16")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// O prefill em lote deve dar o mesmo resultado que processar token a token.
    /// Usa o modelo indicado em BURITI_TEST_MODEL (ou o Qwen 0,5B de buriti-data); pula se não existir.
    #[test]
    fn batch_equals_sequential() {
        let Some(path) = test_model_path() else { return };
        // nos dois caminhos: com os tensores Q4_K reempacotados (se a CPU permite) e no layout original
        for repack in [true, false] {
            let m = Model::load_with(Path::new(&path), repack).expect("carregar modelo");
            eprintln!("repack = {repack}: {} tensores reempacotados", n_repacked(&m));
            batch_vs_sequential(&m);
        }
    }

    /// Entrada por embedding (visão, D50/M4): trocar o embedding de alguns tokens pelo MESMO vetor que o token daria tem de
    /// ser idêntico (logits e bytes do KV), inclusive com o lote cortado em pontos diferentes; e com um vetor de OUTRO
    /// token o resultado tem de mudar (teste com dentes).
    #[test]
    fn embd_override_equals_token_path() {
        let Some(path) = test_model_path() else { return };
        let m = Model::load(Path::new(&path)).expect("carregar modelo");
        if m.sc2.is_some() {
            return;
        }
        let n_embd = m.cfg.n_embd;
        let row = |tok: u32| -> Vec<f32> {
            let mut v = vec![0f32; n_embd];
            dequant_row(m.embd.dtype, m.embd.row(&m.gguf.mmap[..], tok as usize), &mut v);
            v.iter_mut().for_each(|x| *x *= m.cfg.embed_scale);
            v
        };
        let toks: Vec<u32> = (0..45u32).map(|i| (i * 37 + 11) % 2000 + 5).collect();
        let pos: Vec<usize> = (5..21).chain([30, 44]).collect();
        let same: Vec<Vec<f32>> = pos.iter().map(|&i| row(toks[i])).collect();
        let other: Vec<Vec<f32>> = pos.iter().map(|&i| row(toks[i] + 1)).collect();
        let mk = |v: &Vec<Vec<f32>>, lo: usize, hi: usize| -> Vec<(usize, Vec<f32>)> { pos.iter().zip(v).filter(|(&i, _)| i >= lo && i < hi).map(|(&i, e)| (i - lo, e.clone())).collect() };
        let run = |v: &Vec<Vec<f32>>, cuts: &[usize]| -> Session {
            let mut s = m.new_session(128);
            let mut lo = 0;
            for &hi in cuts {
                let e = mk(v, lo, hi);
                let refs: Vec<(usize, &[f32])> = e.iter().map(|(i, x)| (*i, x.as_slice())).collect();
                m.forward_batch_embd(&mut s, &toks[lo..hi], &refs, lo, hi == toks.len());
                lo = hi;
            }
            s
        };
        let mut plain = m.new_session(128);
        m.forward_batch(&mut plain, &toks, 0, true);
        for cuts in [vec![45usize], vec![13, 45], vec![7, 20, 33, 45]] {
            let s = run(&same, &cuts);
            assert_eq!(s.logits, plain.logits, "logits diferem com cortes {cuts:?}");
            assert_eq!(s.k, plain.k, "KV (K) difere com cortes {cuts:?}");
            assert_eq!(s.v, plain.v, "KV (V) difere com cortes {cuts:?}");
        }
        let d = run(&other, &[45]);
        let maxd = d.logits.iter().zip(&plain.logits).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(maxd > 1e-3, "embeddings de outro token não mudaram os logits (diferença máx {maxd})");
    }

    /// `forward_hidden` (embeddings, D77): o estado da linha i é exatamente o `xn` que `forward_batch` deixa depois de
    /// processar os tokens 0..=i (igualdade estrita, com o lote cortado em pontos diferentes), e o último estado
    /// reproduz os logits contra uma referência em f64 (as linhas do vocabulário desquantizadas), dentro do ruído da ativação int8.
    #[test]
    fn hidden_states_equal_the_logit_path_and_reproduce_logits() {
        let Some(path) = test_model_path() else { return };
        let m = Model::load(Path::new(&path)).expect("carregar modelo");
        let n_embd = m.cfg.n_embd;
        let toks: Vec<u32> = (0..45u32).map(|i| (i * 37 + 11) % 2000 + 5).collect();
        let mut s = m.new_session(128);
        let mut h = Vec::new();
        for (lo, hi) in [(0usize, 13usize), (13, 40), (40, 45)] {
            m.forward_hidden(&mut s, &toks[lo..hi], lo, &mut h).unwrap();
        }
        assert_eq!(h.len(), toks.len() * n_embd);
        for i in [0usize, 1, 12, 13, 39, 44] {
            let mut r = m.new_session(128);
            m.forward_batch(&mut r, &toks[..=i], 0, true);
            assert_eq!(&h[i * n_embd..(i + 1) * n_embd], &r.xn[..], "estado do token {i} difere do caminho dos logits");
        }
        // referência em f64: logit(v) = saída[v] · estado_final
        let mut r = m.new_session(128);
        m.forward_batch(&mut r, &toks, 0, true);
        let last = &h[(toks.len() - 1) * n_embd..];
        let mut row = vec![0f32; n_embd];
        let (mut best, mut best_v) = (f64::MIN, 0usize);
        for v in 0..m.cfg.n_vocab {
            dequant_row(m.output.dtype, m.output.row(&m.gguf.mmap[..], v), &mut row);
            let l: f64 = row.iter().zip(last).map(|(a, b)| *a as f64 * *b as f64).sum();
            if l > best {
                (best, best_v) = (l, v);
            }
            if v % 5000 == 0 {
                assert!((l - r.logits[v] as f64).abs() < 0.5, "logit {v}: referência {l} vs motor {}", r.logits[v]);
            }
        }
        let top = r.logits.iter().cloned().fold(f32::MIN, f32::max);
        assert!((r.logits[best_v] - top).abs() < 0.5, "o melhor token da referência ({best_v}) não está entre os melhores do motor");
    }

    fn test_model_path() -> Option<String> {
        let path = std::env::var("BURITI_TEST_MODEL").unwrap_or_else(|_| {
            format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR"))
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("modelo de teste ausente, pulando: {path}");
            return None;
        }
        Some(path)
    }

    fn n_repacked(m: &Model) -> usize {
        m.layers.iter().map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].iter().filter(|t| t.rep.is_some()).count()).sum()
    }

    fn n_repacked_q6(m: &Model) -> usize {
        m.layers.iter().map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].iter().filter(|t| t.dtype == Dtype::Q6K && t.rep.is_some()).count()).sum()
    }

    /// Modelos para os testes de Q6_K: `BURITI_TEST_MODELS` (separados por `;`), senão `BURITI_TEST_MODEL`,
    /// senão os Qwen 0,5B e 1,5B de buriti-data. Só os que existem.
    fn test_model_paths() -> Vec<String> {
        let list: Vec<String> = match (std::env::var("BURITI_TEST_MODELS"), std::env::var("BURITI_TEST_MODEL")) {
            (Ok(v), _) => v.split(';').map(str::to_string).collect(),
            (_, Ok(v)) => vec![v],
            _ => ["qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf"].iter().map(|f| format!("{}/buriti-data/models/{f}", env!("CARGO_MANIFEST_DIR"))).collect(),
        };
        list.into_iter().filter(|p| std::path::Path::new(p).exists()).collect()
    }

    /// Com o Q6_K reempacotado (ligado por padrão), o prefill em lote continua igual ao token a token
    /// (KV cache idêntico byte a byte, logits e argmax coerentes), em modelos que TÊM tensores Q6_K de
    /// camada; e o caminho todo reempacotado fica perto do original. Com `BURITI_NO_REPACK_Q6` ligado o
    /// teste não prova nada sobre o Q6_K e avisa.
    #[test]
    fn batch_equals_sequential_with_q6_repacked() {
        let paths = test_model_paths();
        if paths.is_empty() {
            eprintln!("nenhum modelo de teste, pulando");
            return;
        }
        let can_repack = super::super::repack::Repacked::new(&[0u8; 144 * 8], 8, 256).is_some();
        let mut with_q6 = 0;
        for path in paths {
            let m = Model::load_with(Path::new(&path), true).expect("carregar modelo");
            let (n6, n) = (n_repacked_q6(&m), n_repacked(&m));
            eprintln!("{path}: {n} tensores reempacotados, {n6} deles Q6_K");
            if can_repack && super::super::repack::q6_enabled() {
                with_q6 += (n6 > 0) as usize;
            }
            batch_vs_sequential(&m);
            // contra o layout original (só a ordem de soma em float muda; mesma folga do teste do Q4_K)
            let off = Model::load_with(Path::new(&path), false).unwrap();
            let toks: Vec<u32> = (0..40u32).map(|i| 500 + (i * 53) % 7000).collect();
            let (mut a, mut b) = (m.new_session(128), off.new_session(128));
            m.forward_batch(&mut a, &toks, 0, true);
            off.forward_batch(&mut b, &toks, 0, true);
            let lsm = |v: &[f32]| { let mx = v.iter().cloned().fold(f32::MIN, f32::max); let z: f32 = v.iter().map(|x| (x - mx).exp()).sum(); v.iter().map(|x| x - mx - z.ln()).collect::<Vec<f32>>() };
            let (la, lb) = (lsm(&a.logits), lsm(&b.logits));
            let mut idx: Vec<usize> = (0..lb.len()).collect();
            idx.sort_by(|&i, &j| lb[j].total_cmp(&lb[i]));
            let top: f32 = idx[..10].iter().map(|&i| (la[i] - lb[i]).abs()).fold(0.0, f32::max);
            eprintln!("{path}: logprob dos 10 mais prováveis, repacked contra original: {top:.4}");
            assert_eq!(idx[0], a.logits.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0, "argmax difere");
            // limite 0,5: mesma razão do teste `repacked_close_to_plain` (D72)
            assert!(top < 0.5, "caminho reempacotado (com Q6_K) longe do original: {top}");
        }
        if can_repack && super::super::repack::q6_enabled() {
            assert!(with_q6 > 0, "nenhum dos modelos de teste tem tensor Q6_K de camada reempacotado: o teste não cobriu o Q6_K");
        }
    }

    /// Com Q8_0/Q5_0/Q4_0 reempacotados (D81, ligados por padrão), o prefill em lote continua igual ao token a token
    /// (KV cache idêntico byte a byte) e o caminho todo fica perto do original. Modelos: os de `test_model_paths`, o SmolVLM
    /// Q8_0 de buriti-data e os de `BURITI_TEST_MODELS32` (`;`). Informa quantos tensores de cada formato foram cobertos.
    #[test]
    fn batch_equals_sequential_with_b32_repacked() {
        let mut paths = test_model_paths();
        paths.push(format!("{}/buriti-data/models/smolvlm-500m-instruct-q8_0.gguf", env!("CARGO_MANIFEST_DIR")));
        if let Ok(extra) = std::env::var("BURITI_TEST_MODELS32") {
            paths.extend(extra.split(';').map(str::to_string));
        }
        paths.dedup();
        let can_repack = super::super::repack::Repacked::new(&[0u8; 144 * 8], 8, 256).is_some();
        let mut covered = [0usize; 3];
        for path in paths.into_iter().filter(|p| Path::new(p).exists()) {
            let m = Model::load_with(Path::new(&path), true).expect("carregar modelo");
            let count = |dt: Dtype| -> usize { m.layers.iter().map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].iter().filter(|t| t.dtype == dt && t.rep.is_some()).count()).sum() };
            let (n8, n5, n4) = (count(Dtype::Q8_0), count(Dtype::Q5_0), count(Dtype::Q4_0));
            eprintln!("{path}: reempacotados {n8} Q8_0, {n5} Q5_0, {n4} Q4_0");
            for (i, n) in [n8, n5, n4].iter().enumerate() {
                covered[i] += n;
            }
            batch_vs_sequential(&m);
            let off = Model::load_with(Path::new(&path), false).unwrap();
            let toks: Vec<u32> = (0..40u32).map(|i| 500 + (i * 53) % 7000).collect();
            let (mut a, mut b) = (m.new_session(128), off.new_session(128));
            m.forward_batch(&mut a, &toks, 0, true);
            off.forward_batch(&mut b, &toks, 0, true);
            let lsm = |v: &[f32]| { let mx = v.iter().cloned().fold(f32::MIN, f32::max); let z: f32 = v.iter().map(|x| (x - mx).exp()).sum(); v.iter().map(|x| x - mx - z.ln()).collect::<Vec<f32>>() };
            let (la, lb) = (lsm(&a.logits), lsm(&b.logits));
            let mut idx: Vec<usize> = (0..lb.len()).collect();
            idx.sort_by(|&i, &j| lb[j].total_cmp(&lb[i]));
            let top: f32 = idx[..10].iter().map(|&i| (la[i] - lb[i]).abs()).fold(0.0, f32::max);
            eprintln!("{path}: logprob dos 10 mais prováveis, repacked contra original: {top:.4}");
            // o argmax pode trocar num empate (texto sem sentido, vários candidatos a poucos centésimos): exige que o escolhido
            // esteja a menos de 0,15 de logprob do melhor do caminho original
            let am = a.logits.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
            eprintln!("{path}: argmax original {} (logprob {:.3}) contra repacked {am} (logprob {:.3} no original)", idx[0], lb[idx[0]], lb[am]);
            assert!(lb[idx[0]] - lb[am] < 0.15, "argmax difere sem ser empate");
            assert!(top < 0.5, "caminho reempacotado (com Q8_0/Q5_0/Q4_0) longe do original: {top}");
        }
        eprintln!("tensores cobertos: Q8_0 {}, Q5_0 {}, Q4_0 {}", covered[0], covered[1], covered[2]);
        if can_repack {
            use super::super::repack::{enabled32, Kind32};
            // sem o arquivo do modelo (o CI do GitHub não tem os modelos: buriti-data fica fora do repositório) não há o que cobrir
            let qwen05 = format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR"));
            let default_list = std::env::var("BURITI_TEST_MODELS").is_err() && std::env::var("BURITI_TEST_MODEL").is_err();
            if enabled32(Kind32::Q5_0) && default_list && Path::new(&qwen05).exists() {
                assert!(covered[1] > 0, "nenhum modelo de teste com Q5_0 de camada reempacotado");
            }
            if enabled32(Kind32::Q8_0) && Path::new(&format!("{}/buriti-data/models/smolvlm-500m-instruct-q8_0.gguf", env!("CARGO_MANIFEST_DIR"))).exists() {
                assert!(covered[0] > 0, "nenhum modelo de teste com Q8_0 de camada reempacotado");
            }
        }
    }

    /// Com o Q5_K reempacotado (D84, ligado por padrao), o prefill em lote continua igual ao token a token (KV cache
    /// identico byte a byte) e o caminho todo fica perto do original, em modelos que TEM tensores Q5_K de camada
    /// (Phi-3.5 mini Q4_K_M, incluindo as fatias do qkv e do gate_up fundidos). Modelos: os de `test_model_paths` e o Phi-3.5
    /// de `BURITI_TEST_PHI` (padrão: buriti-data/models, se existir). Com `BURITI_NO_REPACK_Q5K` ligado o teste nao prova nada sobre o Q5_K.
    #[test]
    fn batch_equals_sequential_with_q5k_repacked() {
        let mut paths = test_model_paths();
        let phi = std::env::var("BURITI_TEST_PHI").unwrap_or_else(|_| format!("{}/buriti-data/models/Phi-3.5-mini-instruct-Q4_K_M.gguf", env!("CARGO_MANIFEST_DIR")));
        if std::env::var("BURITI_TEST_MODELS").is_err() && Path::new(&phi).exists() {
            paths.push(phi);
        }
        let can_repack = super::super::repack::Repacked::new(&[0u8; 144 * 8], 8, 256).is_some();
        let mut covered = 0usize;
        for path in paths {
            let m = Model::load_with(Path::new(&path), true).expect("carregar modelo");
            let n5: usize = m.layers.iter().map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].iter().filter(|t| t.dtype == Dtype::Q5K && t.rep.is_some()).count()).sum();
            eprintln!("{path}: {} tensores reempacotados, {n5} deles Q5_K", n_repacked(&m));
            covered += n5;
            batch_vs_sequential(&m);
            let off = Model::load_with(Path::new(&path), false).unwrap();
            let toks: Vec<u32> = (0..40u32).map(|i| 500 + (i * 53) % 7000).collect();
            let (mut a, mut b) = (m.new_session(128), off.new_session(128));
            m.forward_batch(&mut a, &toks, 0, true);
            off.forward_batch(&mut b, &toks, 0, true);
            let lsm = |v: &[f32]| { let mx = v.iter().cloned().fold(f32::MIN, f32::max); let z: f32 = v.iter().map(|x| (x - mx).exp()).sum(); v.iter().map(|x| x - mx - z.ln()).collect::<Vec<f32>>() };
            let (la, lb) = (lsm(&a.logits), lsm(&b.logits));
            let mut idx: Vec<usize> = (0..lb.len()).collect();
            idx.sort_by(|&i, &j| lb[j].total_cmp(&lb[i]));
            let top: f32 = idx[..10].iter().map(|&i| (la[i] - lb[i]).abs()).fold(0.0, f32::max);
            let am = a.logits.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
            eprintln!("{path}: logprob dos 10 mais provaveis, repacked contra original: {top:.4}; argmax original {} (logprob {:.3}) contra repacked {am} (logprob {:.3} no original)", idx[0], lb[idx[0]], lb[am]);
            assert!(lb[idx[0]] - lb[am] < 0.15, "argmax difere sem ser empate");
            assert!(top < 0.5, "caminho reempacotado (com Q5_K) longe do original: {top}");
        }
        eprintln!("tensores Q5_K cobertos: {covered}");
        if can_repack && super::super::repack::q5k_enabled() && Path::new(&std::env::var("BURITI_TEST_PHI").unwrap_or_else(|_| format!("{}/buriti-data/models/Phi-3.5-mini-instruct-Q4_K_M.gguf", env!("CARGO_MANIFEST_DIR")))).exists() && std::env::var("BURITI_TEST_MODELS").is_err() {
            assert!(covered > 0, "nenhum modelo de teste com Q5_K de camada reempacotado: o teste nao cobriu o Q5_K");
        }
    }

    /// Diagnóstico (passo 1 do D52): bytes por tipo nos tensores das CAMADAS (`blk.*`), separando os que
    /// já são reempacotados (Q4_K elegível) dos elegíveis para um layout intercalado ainda não feito
    /// (linhas % 8 == 0 e colunas % tamanho do bloco == 0). Só lê o cabeçalho do GGUF:
    /// `BURITI_TEST_MODELS=a.gguf;b.gguf cargo test list_layer_bytes_by_dtype -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn list_layer_bytes_by_dtype() {
        for path in test_model_paths() {
            let g = Gguf::open(Path::new(&path)).expect("abrir GGUF");
            let mut per: Vec<(u64, u64, u64)> = vec![(0, 0, 0); Dtype::COUNT]; // (total, elegível, inelegível)
            for (name, info) in &g.tensors {
                if !name.starts_with("blk.") || info.dims.len() != 2 {
                    continue;
                }
                let Some(dt) = Gguf::dtype_of(info) else { continue };
                let (be, bb) = dt.block();
                let (cols, rows) = (info.dims[0], info.dims[1]);
                let bytes = (rows * cols / be * bb) as u64;
                per[dt as usize].0 += bytes;
                if rows % 8 == 0 && cols % (if be == 256 { 256 } else { 32 }) == 0 {
                    per[dt as usize].1 += bytes;
                } else {
                    per[dt as usize].2 += bytes;
                }
            }
            let total: u64 = per.iter().map(|p| p.0).sum();
            println!("{path}: tensores de camada, {:.1} MB no total", total as f64 / 1e6);
            for (i, p) in per.iter().enumerate().filter(|(_, p)| p.0 > 0) {
                println!("  {:5} total {:8.1} MB ({:4.1}%), elegível (linhas%8, colunas%256) {:8.1} MB, inelegível {:6.1} MB", Dtype::NAMES[i], p.0 as f64 / 1e6, 100.0 * p.0 as f64 / total as f64, p.1 as f64 / 1e6, p.2 as f64 / 1e6);
            }
        }
    }

    fn batch_vs_sequential(m: &Model) {
        let toks: Vec<u32> = (0..45u32).map(|i| 1000 + (i * 37) % 5000).collect();

        let mut a = m.new_session(256);
        for (i, &t) in toks.iter().enumerate() {
            m.forward(&mut a, t, i, i == toks.len() - 1);
        }

        let mut b = m.new_session(256);
        let mut i = 0;
        for chunk in [16usize, 16, 13] {
            m.forward_batch(&mut b, &toks[i..i + chunk], i, i + chunk == toks.len());
            i += chunk;
        }

        let (mut maxd, mut maxv) = (0f32, 0f32);
        for (x, y) in a.logits.iter().zip(&b.logits) {
            maxd = maxd.max((x - y).abs());
            maxv = maxv.max(x.abs());
        }
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
        eprintln!("logits: diferença máxima {maxd:.5} (maior valor {maxv:.2})");
        assert_eq!(argmax(&a.logits), argmax(&b.logits), "token mais provável difere");
        assert!(maxd < 0.05 * maxv.max(1.0), "logits divergem demais: {maxd}");

        // o KV cache também precisa coincidir (afeta as próximas respostas)
        let kv_dim = m.cfg.n_kv * m.cfg.head_dim;
        for li in 0..m.cfg.n_layer {
            for (x, y) in a.k[li][..toks.len() * kv_dim].iter().zip(&b.k[li]) {
                assert_eq!(x, y, "K difere na camada {li}");
            }
            for (x, y) in a.v[li][..toks.len() * kv_dim].iter().zip(&b.v[li]) {
                assert_eq!(x, y, "V difere na camada {li}");
            }
        }
    }

    /// O caminho reempacotado dá quase o mesmo que o original (só muda a ordem de soma em float) e
    /// de fato reempacota tensores (se a CPU tiver AVX2).
    #[test]
    fn repacked_close_to_plain() {
        let Some(path) = test_model_path() else { return };
        let (on, off) = (Model::load_with(Path::new(&path), true).unwrap(), Model::load_with(Path::new(&path), false).unwrap());
        assert_eq!(n_repacked(&off), 0);
        if super::super::repack::Repacked::new(&[0u8; 144 * 8], 8, 256).is_some() {
            assert!(n_repacked(&on) > 0, "nada foi reempacotado");
            // o embedding e a saída nunca são reempacotados
            assert!(on.embd.rep.is_none() && on.output.rep.is_none());
        }
        let toks: Vec<u32> = (0..40u32).map(|i| 500 + (i * 53) % 7000).collect();
        let (mut a, mut b) = (on.new_session(128), off.new_session(128));
        on.forward_batch(&mut a, &toks, 0, true);
        off.forward_batch(&mut b, &toks, 0, true);
        let maxv = b.logits.iter().fold(0f32, |m, v| m.max(v.abs()));
        let maxd = a.logits.iter().zip(&b.logits).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
        eprintln!("repack contra original: logits dif. máx. {maxd:.5} (maior valor {maxv:.2}), argmax {} / {}", argmax(&a.logits), argmax(&b.logits));
        assert_eq!(argmax(&a.logits), argmax(&b.logits));
        // A diferença de ordem de soma em float (~3e-7 por produto) muda o arredondamento de algumas ativações
        // int8 nas 24 camadas, e isso se acumula: compara-se a logprob dos 10 tokens mais prováveis (o que importa
        // na amostragem), não o logit bruto de todo o vocabulário. Referência medida (texto real, Qwen 0,5B): contra a
        // ativação em float, o caminho original erra até 0,16 e o reempacotado até 0,11 na logprob do top-8: mesmo ruído.
        let lsm = |v: &[f32]| { let mx = v.iter().cloned().fold(f32::MIN, f32::max); let z: f32 = v.iter().map(|x| (x - mx).exp()).sum(); v.iter().map(|x| x - mx - z.ln()).collect::<Vec<f32>>() };
        let (la, lb) = (lsm(&a.logits), lsm(&b.logits));
        let mut idx: Vec<usize> = (0..lb.len()).collect();
        idx.sort_by(|&i, &j| lb[j].total_cmp(&lb[i]));
        let top: f32 = idx[..10].iter().map(|&i| (la[i] - lb[i]).abs()).fold(0.0, f32::max);
        eprintln!("logprob dos 10 mais prováveis: diferença máxima {top:.4}");
        // Limite 0,5 (era 0,35, D72): esta medida (máximo sobre 10 tokens, 40 tokens de texto sem sentido) varia de uma compilação para outra
        // em dois binários cujo código de CPU é o mesmo (0,28 e 0,37; só diferem em código não relacionado), de forma determinística por binário; um defeito real (escala ou linha trocada) dá
        // diferenças de vários pontos, e o `argmax` igual continua exigido.
        assert!(top < 0.5, "caminho reempacotado longe do original: {top}");
    }

    /// Lista metadados e tensores de um GGUF (para investigar arquiteturas novas):
    /// `BURITI_TEST_MODEL=... cargo test dump_gguf -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_gguf() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("defina BURITI_TEST_MODEL");
        let g = Gguf::open(Path::new(&path)).expect("abrir GGUF");
        let mut keys: Vec<_> = g.meta.keys().cloned().collect();
        keys.sort();
        for k in keys {
            let v = &g.meta[&k];
            match v {
                Value::A(a) if a.len() > 6 => println!("META {k} = [{} itens]", a.len()),
                _ => println!("META {k} = {v:?}"),
            }
        }
        let mut names: Vec<_> = g.tensors.keys().filter(|n| !n.starts_with("blk.") || n.starts_with("blk.0.")).cloned().collect();
        names.sort();
        for n in names {
            let t = &g.tensors[&n];
            println!("TENSOR {n} tipo={} dims={:?}", t.ggml_type, t.dims);
        }
    }

    /// Imprime os ids de token de uma lista de textos (JSON em BURITI_TEST_FILE) e o prompt de chat
    /// montado, para comparar com o `/tokenize` e o `/apply-template` do llama.cpp:
    /// `BURITI_TEST_MODEL=... BURITI_TEST_FILE=... cargo test dump_tokens -- --ignored --nocapture`
    /// Diagnóstico do vocabulário: `BURITI_TEST_MODEL=... BURITI_TEST_FILE=lista.json cargo test dump_vocab -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_vocab() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("defina BURITI_TEST_MODEL");
        let file = std::env::var("BURITI_TEST_FILE").expect("defina BURITI_TEST_FILE");
        let g = Gguf::open(Path::new(&path)).expect("abrir GGUF");
        let tok = Tokenizer::from_gguf(&g).expect("carregar o tokenizador");
        let items: Vec<String> = serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
        for s in items {
            println!("VOCAB {}", tok.debug_lookup(&s));
        }
    }

    #[test]
    #[ignore]
    fn dump_tokens() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("defina BURITI_TEST_MODEL");
        let file = std::env::var("BURITI_TEST_FILE").expect("defina BURITI_TEST_FILE");
        // só o tokenizador: assim dá para validar a tokenização antes de o motor suportar a arquitetura
        let g = Gguf::open(Path::new(&path)).expect("abrir GGUF");
        let tok = Tokenizer::from_gguf(&g).expect("carregar o tokenizador");
        let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
        for (i, t) in texts.iter().enumerate() {
            let ids = tok.encode_plain(t);
            println!("TOK {i} {}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        }
        let msg = |role: &str, content: &str| crate::engine::ChatMsg { role: role.into(), content: content.into() };
        let convs = [
            vec![msg("user", "Hello! What is 12345 + 678?")],
            vec![msg("system", "  You are a pirate.  "), msg("user", "Say hi"), msg("assistant", "Arr, hi!"), msg("user", "Again, please.")],
            // resposta antiga com raciocínio (Qwen3 o remove do histórico)
            vec![msg("user", "What is 2+2?"), msg("assistant", "<think>\nThe user asks a sum.\n</think>\n\n4."), msg("user", "And 3+3?")],
        ];
        for (i, c) in convs.iter().enumerate() {
            let ids = tok.chat_prompt(c);
            println!("CHAT {i} {}", serde_json::to_string(&tok.render_all(&ids)).unwrap());
            println!("CHATIDS {i} {}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        }
    }

    /// Imprime o top-20 de logprobs do próximo token para um texto (sem template de chat), para
    /// comparar com o llama.cpp: `BURITI_TEST_MODEL=... BURITI_TEST_TEXT=... cargo test dump_logits -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_logits() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("defina BURITI_TEST_MODEL");
        let text = std::env::var("BURITI_TEST_TEXT").unwrap_or_else(|_| "The capital of France is".into());
        let m = Model::load(Path::new(&path)).expect("carregar modelo");
        let mut ids = m.tok.encode_plain(&text);
        if let Some(b) = m.tok.bos() {
            ids.insert(0, b); // Llama 3 espera o BOS no início
        }
        // BURITI_TEST_CTX: contexto configurado (escolhe o conjunto de fatores do longrope: acima do original, o `long`)
        let cfg_ctx = std::env::var("BURITI_TEST_CTX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let mut s = m.new_session((ids.len() + 16).max(512).max(cfg_ctx));
        let mut i = 0;
        while i < ids.len() {
            let end = (i + 32).min(ids.len());
            m.forward_batch(&mut s, &ids[i..end], i, end == ids.len());
            i = end;
        }
        let max = s.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = max + s.logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
        let mut idx: Vec<usize> = (0..s.logits.len()).collect();
        idx.sort_by(|&a, &b| s.logits[b].total_cmp(&s.logits[a]));
        println!("IDS {}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        for &t in idx.iter().take(20) {
            println!("TOP {t} {:.5}", s.logits[t] - lse);
        }
    }

    /// Custo fixo de uma região paralela do rayon (acordar as threads): `cargo test forkjoin -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_forkjoin() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let mut out = vec![0f32; 896];
        pool.install(|| {
            for round in 0..3 {
                let t0 = std::time::Instant::now();
                let n = 5000;
                for _ in 0..n {
                    out.par_chunks_mut(16).enumerate().for_each(|(i, c)| c[0] = i as f32);
                }
                println!("rodada {round}: {:.1} µs por região paralela (trabalho desprezível)", t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
            // mesma coisa, mas com 3 regiões encadeadas por join (q/k/v)
            let (mut a, mut b, mut c) = (vec![0f32; 896], vec![0f32; 128], vec![0f32; 128]);
            let t0 = std::time::Instant::now();
            let n = 5000;
            for _ in 0..n {
                rayon::join(
                    || a.par_chunks_mut(16).enumerate().for_each(|(i, c)| c[0] = i as f32),
                    || rayon::join(
                        || b.par_chunks_mut(16).enumerate().for_each(|(i, c)| c[0] = i as f32),
                        || c.par_chunks_mut(16).enumerate().for_each(|(i, c)| c[0] = i as f32),
                    ),
                );
            }
            println!("3 regiões com join: {:.1} µs por grupo", t0.elapsed().as_secs_f64() * 1e6 / n as f64);
        });
    }
}

/// (bytes de pesos por tarefa, máximo de linhas por tarefa) do `matmat`. Padrão 8192 e 16; `BURITI_MM_BYTES` e
/// `BURITI_MM_ROWS` existem só para experimentos de desempenho (não mudam o resultado numérico).
fn matmat_blocking() -> (usize, usize) {
    static V: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let get = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok()).filter(|&n| n >= 1).unwrap_or(d);
        (get("BURITI_MM_BYTES", 8192), get("BURITI_MM_ROWS", 16))
    })
}

// ------------------------------------------------------------------ read-only view for the GPU loader (task G8)

/// A weight matrix as the GPU loader sees it: the format, the shape and the bytes in the GGUF layout, straight from the mapped file.
pub(crate) struct TensorView<'a> {
    pub dtype: Dtype,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
    /// Offset of the first row in the file: two views with the same offset share the same weights (tied embedding and output).
    pub off: usize,
    pub bytes: &'a [u8],
}

pub(crate) struct LayerView<'a> {
    pub attn_norm: &'a [f32],
    pub wq: TensorView<'a>,
    pub bq: &'a [f32],
    pub wk: TensorView<'a>,
    pub bk: &'a [f32],
    pub wv: TensorView<'a>,
    pub bv: &'a [f32],
    pub wo: TensorView<'a>,
    pub ffn_norm: &'a [f32],
    pub w_gate: TensorView<'a>,
    pub w_up: TensorView<'a>,
    pub w_down: TensorView<'a>,
    /// per-head RMS norms on Q and K (Qwen3, Gemma 3), post-attention and post-FFN norms (Gemma 3): empty = absent
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub post_ffn_norm: &'a [f32],
    pub swa: bool,
    pub local: bool,
    /// StarCoder2: LayerNorm biases, the output and FFN biases (empty = absent)
    pub attn_norm_b: &'a [f32],
    pub ffn_norm_b: &'a [f32],
    pub bo: &'a [f32],
    pub b_up: &'a [f32],
    pub b_down: &'a [f32],
}

pub(crate) struct GpuView<'a> {
    pub cfg: &'a Config,
    pub layers: Vec<LayerView<'a>>,
    pub embd: TensorView<'a>,
    pub output: TensorView<'a>,
    pub out_norm: &'a [f32],
    /// StarCoder2: the output LayerNorm bias (empty = RMSNorm model)
    pub out_norm_b: &'a [f32],
    /// StarCoder2: the FFN has no gate (plain GELU MLP); `w_gate` then repeats `w_up` and must not be uploaded
    pub plain_mlp: bool,
}

impl Tensor {
    fn view<'a>(&self, mm: &'a [u8]) -> TensorView<'a> {
        TensorView { dtype: self.dtype, rows: self.rows, cols: self.cols, row_bytes: self.row_bytes, off: self.off, bytes: &mm[self.off..self.off + self.rows * self.row_bytes] }
    }
}

impl Model {
    /// The weights and norms for the GPU path. StarCoder2 (its own forward) is not covered yet (task G11).
    pub(crate) fn gpu_view(&self) -> Result<GpuView<'_>, String> {
        let mm = &self.gguf.mmap[..];
        if let Some(sc) = &self.sc2 {
            let layers = sc
                .layers
                .iter()
                .map(|l| LayerView {
                    attn_norm: &l.attn_norm,
                    wq: l.wq.view(mm),
                    bq: &l.bq,
                    wk: l.wk.view(mm),
                    bk: &l.bk,
                    wv: l.wv.view(mm),
                    bv: &l.bv,
                    wo: l.wo.view(mm),
                    ffn_norm: &l.ffn_norm,
                    w_gate: l.w_up.view(mm),
                    w_up: l.w_up.view(mm),
                    w_down: l.w_down.view(mm),
                    q_norm: &[],
                    k_norm: &[],
                    post_attn_norm: &[],
                    post_ffn_norm: &[],
                    swa: false,
                    local: false,
                    attn_norm_b: &l.attn_norm_b,
                    ffn_norm_b: &l.ffn_norm_b,
                    bo: &l.bo,
                    b_up: &l.b_up,
                    b_down: &l.b_down,
                })
                .collect();
            return Ok(GpuView { cfg: &self.cfg, layers, embd: self.embd.view(mm), output: self.output.view(mm), out_norm: &self.out_norm, out_norm_b: &sc.out_norm_b, plain_mlp: true });
        }
        let layers = self
            .layers
            .iter()
            .map(|l| LayerView {
                attn_norm: &l.attn_norm,
                wq: l.wq.view(mm),
                bq: &l.bq,
                wk: l.wk.view(mm),
                bk: &l.bk,
                wv: l.wv.view(mm),
                bv: &l.bv,
                wo: l.wo.view(mm),
                ffn_norm: &l.ffn_norm,
                w_gate: l.w_gate.view(mm),
                w_up: l.w_up.view(mm),
                w_down: l.w_down.view(mm),
                q_norm: &l.q_norm,
                k_norm: &l.k_norm,
                post_attn_norm: &l.post_attn_norm,
                post_ffn_norm: &l.post_ffn_norm,
                swa: l.swa,
                local: l.local,
                attn_norm_b: &[],
                ffn_norm_b: &[],
                bo: &[],
                b_up: &[],
                b_down: &[],
            })
            .collect();
        Ok(GpuView { cfg: &self.cfg, layers, embd: self.embd.view(mm), output: self.output.view(mm), out_norm: &self.out_norm, out_norm_b: &[], plain_mlp: false })
    }

    /// The RoPE frequencies the forward pass uses for a session of `ctx` positions (the longrope long set above the original context).
    pub(crate) fn rope_freqs(&self, ctx: usize, local: bool) -> &[f32] {
        if local {
            &self.inv_freq_local
        } else {
            self.inv_global(ctx)
        }
    }
}
