//! StarCoder2 (3B, 7B, 15B e o 15B-instruct) no motor próprio. Submódulo de `model.rs` (para usar os itens
//! privados dele: `Tensor`, `Session`, `matvec`, `matmat`, `rope`...); a passagem direta do Qwen2/Llama/Gemma/Phi
//! não é tocada: `Model::forward*` desviam para cá quando `sc2` existe.
//!
//! Diferenças para as outras arquiteturas (conferidas no `src/models/starcoder2.cpp` do llama.cpp b12606 e nos
//! metadados dos GGUFs reais):
//! - **LayerNorm com peso E viés** (`attn_norm`, `ffn_norm`, `output_norm`; média subtraída, `eps` de
//!   `starcoder2.attention.layer_norm_epsilon`), não RMSNorm;
//! - **viés** em Q, K, V, saída da atenção (`attn_output.bias`) e nas duas matrizes do MLP;
//! - **MLP sem porta**: `c_proj(gelu(c_fc(x)))` (`ffn_up` -> GELU de tangente hiperbólica -> `ffn_down`), sem `ffn_gate`;
//! - RoPE estilo NeoX (pares i, i + head_dim/2), `rope.freq_base` do arquivo, rotação em todas as dimensões;
//! - GQA, escala 1/sqrt(head_dim); `output.weight` ausente = embeddings compartilhados;
//! - **sem janela deslizante**: o llama.cpp b12606 não lê `sliding_window` do StarCoder2 (o HF declara 4096 e o
//!   modelo foi treinado com contexto de 16384); aqui também atenção completa, igual ao llama.cpp.
//! As operações são genéricas (norma, matmul com viés, GELU elemento a elemento, atenção agrupada): um backend de GPU
//! futuro implementa as mesmas.

use super::*;

/// Pesos de uma camada do StarCoder2.
pub(super) struct Sc2Layer {
    pub(super) attn_norm: Vec<f32>,
    pub(super) attn_norm_b: Vec<f32>,
    pub(super) wq: Tensor,
    pub(super) bq: Vec<f32>,
    pub(super) wk: Tensor,
    pub(super) bk: Vec<f32>,
    pub(super) wv: Tensor,
    pub(super) bv: Vec<f32>,
    pub(super) wo: Tensor,
    pub(super) bo: Vec<f32>,
    pub(super) ffn_norm: Vec<f32>,
    pub(super) ffn_norm_b: Vec<f32>,
    pub(super) w_up: Tensor,
    pub(super) b_up: Vec<f32>,
    pub(super) w_down: Tensor,
    pub(super) b_down: Vec<f32>,
}

pub(super) struct Sc2 {
    pub(super) layers: Vec<Sc2Layer>,
    pub(super) out_norm_b: Vec<f32>,
    /// vetor de 1.0 com `n_ff` elementos: o GELU sem porta reaproveita `gelu_mul` (gelu(x)·1 = gelu(x), exato)
    ones: Vec<f32>,
}

/// Só nos testes: defeitos plantados de propósito (princípio 3 do CLAUDE.md: todo teste precisa de dentes). Fora dos
/// testes `planted` é sempre falso e some na compilação.
#[cfg(test)]
pub(super) static PLANT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
#[cfg(test)]
#[inline(always)]
fn planted(id: u32) -> bool {
    PLANT.load(std::sync::atomic::Ordering::Relaxed) == id
}
#[cfg(not(test))]
#[inline(always)]
fn planted(_id: u32) -> bool {
    false
}

/// LayerNorm com viés, a mesma conta do `ggml_norm` + mul + add: média e variância em f64 (como o ggml acumula),
/// `1/sqrt(var + eps)`, depois `(x - média) · inv · w + b`.
pub(crate) fn layernorm(out: &mut [f32], x: &[f32], w: &[f32], b: &[f32], eps: f32) {
    let n = x.len();
    let eps = if planted(9) { eps * 1e4 } else { eps };
    if planted(2) {
        // defeito plantado: RMSNorm no lugar da LayerNorm (sem tirar a média)
        let ss = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
        let inv = 1.0 / (ss as f32 + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] * inv * w[i] + b[i];
        }
        return;
    }
    let mean = (x.iter().map(|&v| v as f64).sum::<f64>() / n as f64) as f32;
    let var = (x.iter().map(|&v| { let d = (v - mean) as f64; d * d }).sum::<f64>() / n as f64) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..n {
        let y = (x[i] - mean) * inv * w[i];
        out[i] = if planted(1) { y } else { y + b[i] };
    }
}

/// v += b (vetor vazio = sem viés). `id` só serve aos defeitos plantados dos testes.
fn add_bias(id: u32, v: &mut [f32], b: &[f32]) {
    if planted(id) {
        return;
    }
    for (a, x) in v.iter_mut().zip(b) {
        *a += x;
    }
}

/// GELU de tangente hiperbólica (`gelu_pytorch_tanh` do HF = `ggml_gelu` do llama.cpp) sobre um trecho.
fn gelu_inplace(g: &mut [f32], ones: &[f32]) {
    if planted(7) {
        // defeito plantado: o GELU "rápido" x·sigmoid(1,702x) no lugar do de tangente hiperbólica
        for v in g.iter_mut() {
            *v /= 1.0 + (-1.702 * *v).exp();
        }
        return;
    }
    gelu_mul(g, &ones[..g.len()]);
}

fn rope_norm_flag() -> bool {
    planted(8) // defeito plantado: pares adjacentes (Llama) em vez de NeoX
}

pub(super) fn load(g: Gguf, repack: bool) -> Result<Model, String> {
    let key = |k: &str| format!("starcoder2.{k}");
    let n_layer = g.u64(&key("block_count"))? as usize;
    let n_embd = g.u64(&key("embedding_length"))? as usize;
    let n_ff = g.u64(&key("feed_forward_length"))? as usize;
    let n_head = g.u64(&key("attention.head_count"))? as usize;
    let n_kv = g.u64(&key("attention.head_count_kv"))? as usize;
    let n_ctx_train = g.u64(&key("context_length")).unwrap_or(16384) as usize;
    let eps = g.f64(&key("attention.layer_norm_epsilon")).unwrap_or(1e-5) as f32;
    let theta = g.f64(&key("rope.freq_base")).unwrap_or(10000.0) as f32;
    if n_layer == 0 || n_head == 0 || n_kv == 0 || n_head % n_kv != 0 || n_embd % n_head != 0 {
        return Err("hiperparâmetros inconsistentes no modelo".into());
    }
    let head_dim = match g.u64(&key("attention.key_length")) {
        Ok(k) => k as usize,
        Err(_) => n_embd / n_head,
    };
    if head_dim == 0 || head_dim % 2 != 0 || head_dim > 256 {
        return Err("head_dim não suportado".into());
    }
    // o llama.cpp exige n_rot == head_dim no StarCoder2 (rotação em todas as dimensões)
    if let Ok(r) = g.u64(&key("rope.dimension_count")) {
        if r as usize != head_dim {
            return Err(format!("starcoder2.rope.dimension_count = {r} difere de head_dim = {head_dim}: não suportado"));
        }
    }
    let tok = Tokenizer::from_gguf(&g)?;
    let n_vocab = tok.vocab_size();

    let t = |name: &str| load_tensor(&g, name);
    let v = |name: &str| load_vec(&g, name);
    let vopt = |name: &str| if g.tensors.contains_key(name) { load_vec(&g, name) } else { Ok(Vec::new()) };

    let embd = t("token_embd.weight")?;
    let output = match t("output.weight") {
        Ok(o) => o,
        Err(_) => t("token_embd.weight")?, // embeddings compartilhados (3B)
    };
    if embd.cols != n_embd || embd.rows < n_vocab || output.cols != n_embd || output.rows < n_vocab {
        return Err("formato de token_embd/output inesperado".into());
    }
    let (q_dim, kv_dim) = (n_head * head_dim, n_kv * head_dim);
    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let p = |s: &str| format!("blk.{i}.{s}");
        let l = Sc2Layer {
            attn_norm: v(&p("attn_norm.weight"))?,
            attn_norm_b: v(&p("attn_norm.bias"))?,
            wq: t(&p("attn_q.weight"))?,
            bq: vopt(&p("attn_q.bias"))?,
            wk: t(&p("attn_k.weight"))?,
            bk: vopt(&p("attn_k.bias"))?,
            wv: t(&p("attn_v.weight"))?,
            bv: vopt(&p("attn_v.bias"))?,
            wo: t(&p("attn_output.weight"))?,
            bo: vopt(&p("attn_output.bias"))?,
            ffn_norm: v(&p("ffn_norm.weight"))?,
            ffn_norm_b: v(&p("ffn_norm.bias"))?,
            w_up: t(&p("ffn_up.weight"))?,
            b_up: vopt(&p("ffn_up.bias"))?,
            w_down: t(&p("ffn_down.weight"))?,
            b_down: vopt(&p("ffn_down.bias"))?,
        };
        let ok = l.wq.cols == n_embd && l.wq.rows == q_dim
            && l.wk.cols == n_embd && l.wk.rows == kv_dim
            && l.wv.cols == n_embd && l.wv.rows == kv_dim
            && l.wo.cols == q_dim && l.wo.rows == n_embd
            && l.w_up.cols == n_embd && l.w_up.rows == n_ff
            && l.w_down.cols == n_ff && l.w_down.rows == n_embd
            && l.attn_norm.len() == n_embd && l.attn_norm_b.len() == n_embd
            && l.ffn_norm.len() == n_embd && l.ffn_norm_b.len() == n_embd
            && (l.bq.is_empty() || l.bq.len() == q_dim)
            && (l.bk.is_empty() || l.bk.len() == kv_dim)
            && (l.bv.is_empty() || l.bv.len() == kv_dim)
            && (l.bo.is_empty() || l.bo.len() == n_embd)
            && (l.b_up.is_empty() || l.b_up.len() == n_ff)
            && (l.b_down.is_empty() || l.b_down.len() == n_embd);
        if !ok {
            return Err(format!("formato inesperado de tensores na camada {i}"));
        }
        layers.push(l);
    }
    let out_norm = v("output_norm.weight")?;
    let out_norm_b = v("output_norm.bias")?;
    if out_norm.len() != n_embd || out_norm_b.len() != n_embd {
        return Err("formato inesperado de output_norm".into());
    }

    // mesmo reempacotamento Q4_K das outras arquiteturas (só os pesos das camadas; ver `Tensor::rep`)
    if repack {
        let mut released: Vec<(usize, usize)> = Vec::new();
        for l in layers.iter_mut() {
            for t in [&mut l.wq, &mut l.wk, &mut l.wv, &mut l.wo, &mut l.w_up, &mut l.w_down] {
                let b = t.try_repack(&g.mmap);
                if b > 0 {
                    released.push((t.off, b));
                }
            }
        }
        for &(o, n) in &released {
            release_pages(&g.mmap, o, n);
        }
    }

    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| theta.powf(-(2.0 * i as f32) / head_dim as f32)).collect();
    Ok(Model {
        cfg: Config { n_layer, n_embd, n_ff, n_head, n_kv, head_dim, eps, n_ctx_train, n_vocab, rope_norm: false, embed_scale: 1.0, gelu: true, window: 0, n_rot: head_dim, rope_mscale: 1.0 },
        tok,
        gguf: g,
        layers: Vec::new(),
        embd,
        output,
        out_norm,
        inv_freq_local: inv_freq.clone(),
        inv_freq,
        inv_freq_long: Vec::new(),
        n_ctx_orig: n_ctx_train,
        sc2: Some(Sc2 { layers, out_norm_b, ones: vec![1.0; n_ff.max(2048)] }),
    })
}

impl Model {
    /// Um token na posição `pos` (escreve o KV dessa posição). As mesmas contas, na mesma ordem, de `forward_batch_sc2`.
    pub(super) fn forward_sc2(&self, sc: &Sc2, s: &mut Session, token: u32, pos: usize, want_logits: bool) {
        let c = &self.cfg;
        let (hd, n_head, n_kv) = (c.head_dim, c.n_head, c.n_kv);
        let kv_dim = n_kv * hd;
        let group = n_head / n_kv;
        let mm = &self.gguf.mmap[..];
        assert!(pos < s.ctx && (token as usize) < c.n_vocab);
        let mut ac = ActCache::default();
        dequant_row(self.embd.dtype, self.embd.row(mm, token as usize), &mut s.x);
        let scale = if planted(10) { 1.0 / hd as f32 } else { 1.0 / (hd as f32).sqrt() };

        for (li, l) in sc.layers.iter().enumerate() {
            // ---- atenção
            layernorm(&mut s.xn, &s.x, &l.attn_norm, &l.attn_norm_b, c.eps);
            ac.clear();
            self.matvec(&l.wq, &s.xn, &mut s.q, &mut ac);
            self.matvec(&l.wk, &s.xn, &mut s.kk, &mut ac);
            self.matvec(&l.wv, &s.xn, &mut s.vv, &mut ac);
            add_bias(3, &mut s.q, &l.bq);
            add_bias(11, &mut s.kk, &l.bk);
            add_bias(12, &mut s.vv, &l.bv);
            for h in 0..n_head {
                rope(&mut s.q[h * hd..(h + 1) * hd], pos, &self.inv_freq, rope_norm_flag(), 1.0);
            }
            for h in 0..n_kv {
                rope(&mut s.kk[h * hd..(h + 1) * hd], pos, &self.inv_freq, rope_norm_flag(), 1.0);
            }
            store_f16(&s.kk, &mut s.k[li][pos * kv_dim..(pos + 1) * kv_dim]);
            store_f16(&s.vv, &mut s.v[li][pos * kv_dim..(pos + 1) * kv_dim]);
            let t_att = std::time::Instant::now();
            for g in 0..n_kv {
                let (a, b) = (g * group * hd, (g + 1) * group * hd);
                attend_group(&s.q[a..b], &s.k[li], &s.v[li], pos, g, group, hd, kv_dim, scale, &mut s.scores, &mut s.att[a..b]);
            }
            PH_NS[1].fetch_add(t_att.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            ac.clear();
            self.matvec(&l.wo, &s.att, &mut s.tmp, &mut ac);
            add_bias(4, &mut s.tmp, &l.bo);
            for (x, d) in s.x.iter_mut().zip(&s.tmp) {
                *x += d;
            }

            // ---- MLP sem porta: down(gelu(up(x) + b_up)) + b_down
            layernorm(&mut s.xn, &s.x, &l.ffn_norm, &l.ffn_norm_b, c.eps);
            ac.clear();
            self.matvec(&l.w_up, &s.xn, &mut s.gate, &mut ac);
            add_bias(5, &mut s.gate, &l.b_up);
            let t_act = std::time::Instant::now();
            gelu_inplace(&mut s.gate, &sc.ones);
            PH_NS[2].fetch_add(t_act.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            ac.clear();
            self.matvec(&l.w_down, &s.gate, &mut s.tmp, &mut ac);
            add_bias(6, &mut s.tmp, &l.b_down);
            for (x, d) in s.x.iter_mut().zip(&s.tmp) {
                *x += d;
            }
        }

        if want_logits {
            layernorm(&mut s.xn, &s.x, &self.out_norm, &sc.out_norm_b, c.eps);
            ac.clear();
            self.matvec(&self.output, &s.xn, &mut s.logits, &mut ac);
        }
    }

    /// `tokens` nas posições `start..start+n` de uma vez (prefill). Idêntico, bit a bit (logits e bytes do KV), a chamar
    /// `forward_sc2` token a token: toda conta por token usa as mesmas funções e a mesma ordem de soma.
    pub(super) fn forward_batch_sc2(&self, sc: &Sc2, s: &mut Session, tokens: &[u32], start: usize, want_logits: bool) {
        let n = tokens.len();
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
        let mut xn = vec![0.0f32; n * n_embd];
        let mut q = vec![0.0f32; n * q_dim];
        let mut kk = vec![0.0f32; n * kv_dim];
        let mut vv = vec![0.0f32; n * kv_dim];
        let mut att = vec![0.0f32; n * q_dim];
        let mut tmp = vec![0.0f32; n * n_embd];
        let mut hid = vec![0.0f32; n * c.n_ff];
        let scale = if planted(10) { 1.0 / hd as f32 } else { 1.0 / (hd as f32).sqrt() };
        let ctx = s.ctx;
        let half = hd / 2;
        let mut cs = vec![(0f32, 0f32); n * half];
        for i in 0..n {
            // defeito plantado 21: posição errada (+1) só no lote
            let p = if planted(21) { start + i + 1 } else { start + i };
            for j in 0..half {
                cs[i * half + j] = (p as f32 * self.inv_freq[j]).sin_cos();
            }
        }

        for (li, l) in sc.layers.iter().enumerate() {
            for i in 0..n {
                layernorm(&mut xn[i * n_embd..(i + 1) * n_embd], &x[i * n_embd..(i + 1) * n_embd], &l.attn_norm, &l.attn_norm_b, c.eps);
            }
            self.matmat(&l.wq, &xn, n, q_dim, &mut q);
            self.matmat(&l.wk, &xn, n, kv_dim, &mut kk);
            self.matmat(&l.wv, &xn, n, kv_dim, &mut vv);
            for i in 0..n {
                let pos = start + i;
                let (qi, ki, vi) = (&mut q[i * q_dim..(i + 1) * q_dim], &mut kk[i * kv_dim..(i + 1) * kv_dim], &mut vv[i * kv_dim..(i + 1) * kv_dim]);
                add_bias(3, qi, &l.bq);
                add_bias(11, ki, &l.bk);
                add_bias(12, vi, &l.bv);
                let cs_i = &cs[i * half..(i + 1) * half];
                for h in 0..n_head {
                    rope_cs(&mut qi[h * hd..(h + 1) * hd], cs_i, rope_norm_flag());
                }
                for h in 0..n_kv {
                    rope_cs(&mut ki[h * hd..(h + 1) * hd], cs_i, rope_norm_flag());
                }
                store_f16(ki, &mut s.k[li][pos * kv_dim..(pos + 1) * kv_dim]);
                store_f16(vi, &mut s.v[li][pos * kv_dim..(pos + 1) * kv_dim]);
            }
            // máscara causal: o token i só enxerga 0..=start+i, então gravar o KV do lote inteiro antes não muda nada
            let (kc, vc) = (&s.k[li], &s.v[li]);
            let t_att = std::time::Instant::now();
            att.par_chunks_mut(group * hd).enumerate().for_each_init(
                || vec![0.0f32; group * ctx],
                |sc_buf, (idx, out)| {
                    let (i, g) = (idx / n_kv, idx % n_kv);
                    let qg = &q[i * q_dim + g * group * hd..i * q_dim + (g + 1) * group * hd];
                    attend_group(qg, kc, vc, start + i, g, group, hd, kv_dim, scale, sc_buf, out);
                },
            );
            PH_NS[1].fetch_add(t_att.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            self.matmat(&l.wo, &att, n, n_embd, &mut tmp);
            for t in tmp.chunks_mut(n_embd) {
                add_bias(4, t, &l.bo);
            }
            for (a, d) in x.iter_mut().zip(&tmp) {
                *a += d;
            }

            for i in 0..n {
                layernorm(&mut xn[i * n_embd..(i + 1) * n_embd], &x[i * n_embd..(i + 1) * n_embd], &l.ffn_norm, &l.ffn_norm_b, c.eps);
            }
            self.matmat(&l.w_up, &xn, n, c.n_ff, &mut hid);
            for h in hid.chunks_mut(c.n_ff) {
                add_bias(5, h, &l.b_up);
            }
            let t_act = std::time::Instant::now();
            hid.par_chunks_mut(2048).for_each(|hc| gelu_inplace(hc, &sc.ones));
            PH_NS[2].fetch_add(t_act.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            self.matmat(&l.w_down, &hid, n, n_embd, &mut tmp);
            for t in tmp.chunks_mut(n_embd) {
                // defeito plantado 20: o viés de c_proj some só no lote
                let b: &[f32] = if planted(20) { &[] } else { &l.b_down };
                add_bias(6, t, b);
            }
            for (a, d) in x.iter_mut().zip(&tmp) {
                *a += d;
            }
        }

        if want_logits {
            layernorm(&mut s.xn, &x[(n - 1) * n_embd..], &self.out_norm, &sc.out_norm_b, c.eps);
            self.matvec(&self.output, &s.xn, &mut s.logits, &mut ActCache::default());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Mutex;

    /// Os defeitos plantados são globais: os testes que rodam o motor se revezam.
    static LOCK: Mutex<()> = Mutex::new(());

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 // [-1, 1)
    }

    // ------------------------------------------------------------------ modelo sintético

    #[derive(Clone, Copy)]
    struct Spec {
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        n_ff: usize,
        tied: bool,
    }
    /// 3B em miniatura (GQA 4:1) e a razão de grupo do 15B (48 cabeças / 4 de KV = 12:1).
    const SMALL: Spec = Spec { n_layer: 3, n_embd: 64, n_head: 8, n_kv: 2, hd: 8, n_ff: 96, tied: true };
    const GROUP12: Spec = Spec { n_layer: 2, n_embd: 96, n_head: 12, n_kv: 1, hd: 8, n_ff: 64, tied: false };
    const THETA: f32 = 100000.0;
    const EPS: f32 = 1e-5;

    struct T {
        name: String,
        dims: Vec<usize>, // GGUF: dims[0] = colunas
        data: Vec<f32>,
    }

    fn byte_chars() -> Vec<char> {
        let mut out = Vec::new();
        let mut n = 0u32;
        for b in 0..=255u32 {
            let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            out.push(if printable { char::from_u32(b).unwrap() } else { n += 1; char::from_u32(255 + n).unwrap() });
        }
        out
    }

    fn vocab() -> (Vec<String>, Vec<String>) {
        let bc = byte_chars();
        let mut tokens: Vec<String> = bc.iter().map(|c| c.to_string()).collect();
        let sp = bc[b' ' as usize];
        tokens.extend(["th".to_string(), "the".to_string(), format!("{sp}the")]);
        let merges = vec!["t h".to_string(), "th e".to_string(), format!("{sp} the")];
        tokens.extend(["<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<file_sep>"].map(String::from));
        (tokens, merges)
    }

    fn make_weights(sp: Spec, seed: u64) -> Vec<T> {
        let mut s = seed;
        let (q_dim, kv_dim) = (sp.n_head * sp.hd, sp.n_kv * sp.hd);
        let nv = vocab().0.len();
        let mut out: Vec<T> = Vec::new();
        let mat = |name: String, rows: usize, cols: usize, out: &mut Vec<T>, s: &mut u64| {
            let k = 1.5 / (cols as f32).sqrt();
            out.push(T { name, dims: vec![cols, rows], data: (0..rows * cols).map(|_| lcg(s) * k * 1.7).collect() });
        };
        let vecf = |name: String, n: usize, base: f32, amp: f32, out: &mut Vec<T>, s: &mut u64| {
            out.push(T { name, dims: vec![n], data: (0..n).map(|_| base + lcg(s) * amp).collect() });
        };
        out.push(T { name: "token_embd.weight".into(), dims: vec![sp.n_embd, nv], data: (0..nv * sp.n_embd).map(|_| lcg(&mut s)).collect() });
        vecf("output_norm.weight".into(), sp.n_embd, 1.0, 0.3, &mut out, &mut s);
        vecf("output_norm.bias".into(), sp.n_embd, 0.0, 0.4, &mut out, &mut s);
        if !sp.tied {
            mat("output.weight".into(), nv, sp.n_embd, &mut out, &mut s);
        }
        for i in 0..sp.n_layer {
            let p = |x: &str| format!("blk.{i}.{x}");
            vecf(p("attn_norm.weight"), sp.n_embd, 1.0, 0.3, &mut out, &mut s);
            vecf(p("attn_norm.bias"), sp.n_embd, 0.0, 0.4, &mut out, &mut s);
            mat(p("attn_q.weight"), q_dim, sp.n_embd, &mut out, &mut s);
            vecf(p("attn_q.bias"), q_dim, 0.0, 0.6, &mut out, &mut s);
            mat(p("attn_k.weight"), kv_dim, sp.n_embd, &mut out, &mut s);
            vecf(p("attn_k.bias"), kv_dim, 0.0, 0.6, &mut out, &mut s);
            mat(p("attn_v.weight"), kv_dim, sp.n_embd, &mut out, &mut s);
            vecf(p("attn_v.bias"), kv_dim, 0.0, 0.6, &mut out, &mut s);
            mat(p("attn_output.weight"), sp.n_embd, q_dim, &mut out, &mut s);
            vecf(p("attn_output.bias"), sp.n_embd, 0.0, 0.6, &mut out, &mut s);
            vecf(p("ffn_norm.weight"), sp.n_embd, 1.0, 0.3, &mut out, &mut s);
            vecf(p("ffn_norm.bias"), sp.n_embd, 0.0, 0.4, &mut out, &mut s);
            mat(p("ffn_up.weight"), sp.n_ff, sp.n_embd, &mut out, &mut s);
            vecf(p("ffn_up.bias"), sp.n_ff, 0.0, 0.8, &mut out, &mut s);
            mat(p("ffn_down.weight"), sp.n_embd, sp.n_ff, &mut out, &mut s);
            vecf(p("ffn_down.bias"), sp.n_embd, 0.0, 0.6, &mut out, &mut s);
        }
        out
    }

    struct W(Vec<u8>);
    impl W {
        fn u32(&mut self, v: u32) {
            self.0.extend(v.to_le_bytes());
        }
        fn u64(&mut self, v: u64) {
            self.0.extend(v.to_le_bytes());
        }
        fn str(&mut self, s: &str) {
            self.u64(s.len() as u64);
            self.0.extend(s.as_bytes());
        }
        fn kv_str(&mut self, k: &str, v: &str) {
            self.str(k);
            self.u32(8);
            self.str(v);
        }
        fn kv_u32(&mut self, k: &str, v: u32) {
            self.str(k);
            self.u32(4);
            self.u32(v);
        }
        fn kv_f32(&mut self, k: &str, v: f32) {
            self.str(k);
            self.u32(6);
            self.0.extend(v.to_le_bytes());
        }
        fn kv_strs(&mut self, k: &str, v: &[String]) {
            self.str(k);
            self.u32(9);
            self.u32(8);
            self.u64(v.len() as u64);
            for s in v {
                self.str(s);
            }
        }
        fn kv_i32s(&mut self, k: &str, v: &[i32]) {
            self.str(k);
            self.u32(9);
            self.u32(5);
            self.u64(v.len() as u64);
            for x in v {
                self.u32(*x as u32);
            }
        }
    }

    /// GGUF sintético do StarCoder2 (pesos F32). `pre`: valor de `tokenizer.ggml.pre` (None = ausente, como nos arquivos reais).
    fn write_gguf(sp: Spec, w: &[T], path: &std::path::Path, pre: Option<&str>, tmpl: Option<&str>) {
        let (tokens, merges) = vocab();
        let mut types = vec![1i32; tokens.len()];
        for t in types.iter_mut().skip(tokens.len() - 5) {
            *t = 3;
        }
        let mut h = W(Vec::new());
        h.0.extend(b"GGUF");
        h.u32(3);
        h.u64(w.len() as u64);
        let mut n_kv = 14u64;
        n_kv += pre.is_some() as u64 + tmpl.is_some() as u64;
        h.u64(n_kv);
        h.kv_str("general.architecture", "starcoder2");
        h.kv_u32("starcoder2.block_count", sp.n_layer as u32);
        h.kv_u32("starcoder2.context_length", 16384);
        h.kv_u32("starcoder2.embedding_length", sp.n_embd as u32);
        h.kv_u32("starcoder2.feed_forward_length", sp.n_ff as u32);
        h.kv_u32("starcoder2.attention.head_count", sp.n_head as u32);
        h.kv_u32("starcoder2.attention.head_count_kv", sp.n_kv as u32);
        h.kv_f32("starcoder2.rope.freq_base", THETA);
        h.kv_f32("starcoder2.attention.layer_norm_epsilon", EPS);
        h.kv_str("tokenizer.ggml.model", "gpt2");
        h.kv_strs("tokenizer.ggml.tokens", &tokens);
        h.kv_strs("tokenizer.ggml.merges", &merges);
        h.kv_i32s("tokenizer.ggml.token_type", &types);
        h.kv_u32("tokenizer.ggml.eos_token_id", (tokens.len() - 5) as u32);
        if let Some(p) = pre {
            h.kv_str("tokenizer.ggml.pre", p);
        }
        if let Some(t) = tmpl {
            h.kv_str("tokenizer.chat_template", t);
        }
        let mut off = 0u64;
        let mut offs = Vec::new();
        for t in w {
            h.str(&t.name);
            h.u32(t.dims.len() as u32);
            for d in &t.dims {
                h.u64(*d as u64);
            }
            h.u32(0); // F32
            h.u64(off);
            offs.push(off);
            off += (t.data.len() as u64 * 4).div_ceil(32) * 32;
        }
        while h.0.len() % 32 != 0 {
            h.0.push(0);
        }
        for t in w {
            for v in &t.data {
                h.0.extend(v.to_le_bytes());
            }
            while h.0.len() % 32 != 0 {
                h.0.push(0);
            }
        }
        std::fs::write(path, &h.0).unwrap();
    }

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        std::env::temp_dir().join(format!("buriti-sc2-{}-{}-{tag}.gguf", std::process::id(), N.fetch_add(1, Relaxed)))
    }

    // ------------------------------------------------------------------ referência ingênua em f64

    fn get<'a>(w: &'a [T], name: &str) -> &'a [f32] {
        &w.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name}")).data
    }

    fn lin(x: &[f64], w: &[f32], b: &[f32], rows: usize) -> Vec<f64> {
        let cols = x.len();
        (0..rows).map(|r| (0..cols).map(|c| w[r * cols + c] as f64 * x[c]).sum::<f64>() + b[r] as f64).collect()
    }

    fn ln_ref(x: &[f64], w: &[f32], b: &[f32]) -> Vec<f64> {
        let n = x.len() as f64;
        let m = x.iter().sum::<f64>() / n;
        let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / n;
        x.iter().enumerate().map(|(i, a)| (a - m) / (v + EPS as f64).sqrt() * w[i] as f64 + b[i] as f64).collect()
    }

    fn gelu_ref(x: f64) -> f64 {
        0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    }

    fn round_f16(v: f64) -> f64 {
        crate::native::quant::f16_to_f32(crate::native::f16::f32_to_f16(v as f32)) as f64
    }

    /// Logits do último token, pela fórmula do HF (Starcoder2ForCausalLM), tudo em f64; só K e V passam por f16
    /// (o cache do motor). `tokens` inteiros, atenção causal completa (sem janela).
    fn ref_logits(sp: Spec, w: &[T], tokens: &[u32]) -> Vec<f64> {
        let (hd, n_head, n_kv) = (sp.hd, sp.n_head, sp.n_kv);
        let (q_dim, kv_dim) = (n_head * hd, n_kv * hd);
        let group = n_head / n_kv;
        let nv = vocab().0.len();
        let embd = get(w, "token_embd.weight");
        let mut ks: Vec<Vec<Vec<f64>>> = vec![Vec::new(); sp.n_layer]; // [camada][pos][kv_dim]
        let mut vs: Vec<Vec<Vec<f64>>> = vec![Vec::new(); sp.n_layer];
        let mut last = Vec::new();
        for (pos, &tk) in tokens.iter().enumerate() {
            let mut x: Vec<f64> = embd[tk as usize * sp.n_embd..(tk as usize + 1) * sp.n_embd].iter().map(|&v| v as f64).collect();
            for li in 0..sp.n_layer {
                let p = |s: &str| get(w, &format!("blk.{li}.{s}"));
                let xn = ln_ref(&x, p("attn_norm.weight"), p("attn_norm.bias"));
                let mut q = lin(&xn, p("attn_q.weight"), p("attn_q.bias"), q_dim);
                let mut k = lin(&xn, p("attn_k.weight"), p("attn_k.bias"), kv_dim);
                let v = lin(&xn, p("attn_v.weight"), p("attn_v.bias"), kv_dim);
                // RoPE NeoX: pares (i, i + hd/2), ângulo pos · theta^(-2i/hd)
                let rot = |vec: &mut [f64], heads: usize| {
                    for h in 0..heads {
                        for i in 0..hd / 2 {
                            let ang = pos as f64 * (THETA as f64).powf(-(2.0 * i as f64) / hd as f64);
                            let (sn, cs) = ang.sin_cos();
                            let (a, b) = (vec[h * hd + i], vec[h * hd + i + hd / 2]);
                            vec[h * hd + i] = a * cs - b * sn;
                            vec[h * hd + i + hd / 2] = a * sn + b * cs;
                        }
                    }
                };
                rot(&mut q, n_head);
                rot(&mut k, n_kv);
                ks[li].push(k.iter().map(|&a| round_f16(a)).collect());
                vs[li].push(v.iter().map(|&a| round_f16(a)).collect());
                let mut att = vec![0f64; q_dim];
                for h in 0..n_head {
                    let g = h / group;
                    let sc: Vec<f64> = (0..=pos).map(|t| (0..hd).map(|d| q[h * hd + d] * ks[li][t][g * hd + d]).sum::<f64>() / (hd as f64).sqrt()).collect();
                    let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for d in 0..hd {
                        att[h * hd + d] = (0..=pos).map(|t| e[t] / z * vs[li][t][g * hd + d]).sum();
                    }
                }
                let o = lin(&att, p("attn_output.weight"), p("attn_output.bias"), sp.n_embd);
                for i in 0..sp.n_embd {
                    x[i] += o[i];
                }
                let xn = ln_ref(&x, p("ffn_norm.weight"), p("ffn_norm.bias"));
                let up: Vec<f64> = lin(&xn, p("ffn_up.weight"), p("ffn_up.bias"), sp.n_ff).into_iter().map(gelu_ref).collect();
                let down = lin(&up, p("ffn_down.weight"), p("ffn_down.bias"), sp.n_embd);
                for i in 0..sp.n_embd {
                    x[i] += down[i];
                }
            }
            if pos == tokens.len() - 1 {
                let xn = ln_ref(&x, get(w, "output_norm.weight"), get(w, "output_norm.bias"));
                let out = if sp.tied { embd } else { get(w, "output.weight") };
                last = (0..nv).map(|r| (0..sp.n_embd).map(|c| out[r * sp.n_embd + c] as f64 * xn[c]).sum::<f64>()).collect();
            }
        }
        last
    }

    fn toks(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32).map(|i| (i * 37 + seed * 11) % 250 + 1).collect()
    }

    /// Maior diferença entre os logits do motor (token a token E em lote) e a referência.
    fn engine_vs_ref(sp: Spec, seed: u64, ntok: usize, plant: u32) -> (f64, f64, f64) {
        let w = make_weights(sp, seed);
        let path = tmp_path("ref");
        write_gguf(sp, &w, &path, Some("starcoder"), None);
        let tk = toks(ntok, seed as u32);
        let want = ref_logits(sp, &w, &tk);
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let (d1, d2);
        {
            let m = Model::load_with(&path, false).expect("carregar o GGUF sintético");
            PLANT.store(plant, Relaxed);
            let mut a = m.new_session(64);
            for (i, &t) in tk.iter().enumerate() {
                m.forward(&mut a, t, i, i == tk.len() - 1);
            }
            let mut b = m.new_session(64);
            let mut i = 0;
            for chunk in [5usize, 4, ntok - 9] {
                m.forward_batch(&mut b, &tk[i..i + chunk], i, i + chunk == tk.len());
                i += chunk;
            }
            PLANT.store(0, Relaxed);
            let diff = |l: &[f32]| l.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max);
            d1 = diff(&a.logits);
            d2 = diff(&b.logits);
        }
        let _ = std::fs::remove_file(&path);
        (d1, d2, scale)
    }

    /// Tolerância: o cache em f16 e o `exp` vetorizado dão ~1e-4 relativo; 3e-3 do maior logit sobra e ainda detecta
    /// cada defeito plantado (ver `planted_bugs_are_caught`).
    fn tol(scale: f64) -> f64 {
        3e-3 * scale.max(1.0)
    }

    #[test]
    fn matches_naive_f64_reference() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (sp, seed) in [(SMALL, 7u64), (GROUP12, 9)] {
            let (d1, d2, scale) = engine_vs_ref(sp, seed, 14, 0);
            eprintln!("n_head={} n_kv={}: dif. token a token {d1:.2e}, em lote {d2:.2e} (maior logit {scale:.2})", sp.n_head, sp.n_kv);
            assert!(d1 < tol(scale), "token a token longe da referência: {d1} (tol {})", tol(scale));
            assert!(d2 < tol(scale), "lote longe da referência: {d2}");
        }
    }

    /// Defeito plantado em cada operação nova: o teste anterior TEM de reprovar (princípio 3).
    #[test]
    fn planted_bugs_are_caught() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let names = [
            (1, "LayerNorm sem viés"),
            (2, "RMSNorm no lugar da LayerNorm"),
            (3, "sem viés de Q"),
            (4, "sem viés de attn_output"),
            (5, "sem viés de c_fc (ffn_up)"),
            (6, "sem viés de c_proj (ffn_down)"),
            (7, "GELU rápido x·sigmoid(1,702x) no lugar do de tangente hiperbólica"),
            (8, "RoPE de pares adjacentes no lugar do NeoX"),
            (9, "eps errado (x10000)"),
            (10, "escala da atenção 1/hd"),
            (11, "sem viés de K"),
            (12, "sem viés de V"),
        ];
        for (id, what) in names {
            let (d1, _d2, scale) = engine_vs_ref(SMALL, 7, 14, id);
            eprintln!("defeito {id} ({what}): dif. {d1:.3e} (tol {:.1e})", tol(scale));
            assert!(d1 > 3.0 * tol(scale), "o defeito '{what}' NÃO foi detectado (dif. {d1})");
        }
    }

    /// Princípio 1: o prefill em lote é idêntico ao token a token, logits e bytes do KV, com GQA 4:1 e 12:1.
    fn strict_batch_check(sp: Spec, ntok: usize) -> (bool, bool) {
        let w = make_weights(sp, 21);
        let path = tmp_path("batch");
        write_gguf(sp, &w, &path, Some("starcoder"), None);
        let r;
        {
            let m = Model::load_with(&path, false).unwrap();
            let tk = toks(ntok, 3);
            let mut a = m.new_session(64);
            for (i, &t) in tk.iter().enumerate() {
                m.forward(&mut a, t, i, i == tk.len() - 1);
            }
            let mut b = m.new_session(64);
            let mut i = 0;
            for chunk in [8usize, 3, ntok - 11] {
                m.forward_batch(&mut b, &tk[i..i + chunk], i, i + chunk == tk.len());
                i += chunk;
            }
            let logits_eq = a.logits.iter().zip(&b.logits).all(|(x, y)| x.to_bits() == y.to_bits());
            let kv = sp.n_kv * sp.hd;
            let kv_eq = (0..sp.n_layer).all(|l| a.k[l][..ntok * kv] == b.k[l][..ntok * kv] && a.v[l][..ntok * kv] == b.v[l][..ntok * kv]);
            r = (logits_eq, kv_eq);
        }
        let _ = std::fs::remove_file(&path);
        r
    }

    #[test]
    fn batch_equals_sequential_strict() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for sp in [SMALL, GROUP12] {
            let (l, kv) = strict_batch_check(sp, 21);
            assert!(l, "logits do lote diferem (bit a bit) do token a token");
            assert!(kv, "bytes do KV do lote diferem do token a token");
        }
    }

    #[test]
    fn batch_planted_bugs_are_caught() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (id, what) in [(20, "viés de c_proj ausente só no lote"), (21, "posição do RoPE errada só no lote")] {
            PLANT.store(id, Relaxed);
            let (l, kv) = strict_batch_check(SMALL, 21);
            PLANT.store(0, Relaxed);
            eprintln!("defeito {id} ({what}): logits iguais={l}, KV igual={kv}");
            assert!(!(l && kv), "o defeito '{what}' NÃO foi detectado pela igualdade estrita");
        }
    }

    /// LayerNorm contra a fórmula ingênua, inclusive com média grande (onde o RMSNorm erra muito).
    #[test]
    fn layernorm_matches_reference() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = 5u64;
        for n in [8usize, 64, 3072] {
            let x: Vec<f32> = (0..n).map(|_| 3.0 + 2.0 * lcg(&mut s)).collect();
            let w: Vec<f32> = (0..n).map(|_| 1.0 + 0.3 * lcg(&mut s)).collect();
            let b: Vec<f32> = (0..n).map(|_| 0.5 * lcg(&mut s)).collect();
            let mut out = vec![0f32; n];
            layernorm(&mut out, &x, &w, &b, EPS);
            let xs: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let want = ln_ref(&xs, &w, &b);
            let d = out.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max);
            assert!(d < 2e-5, "n={n}: dif. {d}");
        }
    }

    /// O GELU sem porta é o de tangente hiperbólica (não o exato com erf nem o rápido), em toda a faixa útil.
    #[test]
    fn plain_gelu_is_tanh_variant() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let xs: Vec<f32> = (0..4099).map(|i| -12.0 + i as f32 * 24.0 / 4099.0).collect();
        let mut g = xs.clone();
        gelu_inplace(&mut g, &vec![1.0; xs.len()]);
        let mut worst_erf = 0f64;
        for (x, y) in xs.iter().zip(&g) {
            let want = gelu_ref(*x as f64);
            assert!((*y as f64 - want).abs() <= 3e-6 * want.abs().max(1e-3) + 1e-6, "gelu({x}) = {y}, esperado {want}");
            // distância ao GELU exato (erf): o teste distingue as duas variantes? (diferem até ~5e-4 perto de |x|~2)
            if x.abs() <= 4.0 {
                let exact = 0.5 * *x as f64 * (1.0 + erf(*x as f64 / 2f64.sqrt()));
                worst_erf = worst_erf.max((exact - want).abs());
            }
        }
        assert!(worst_erf > 1e-4, "o teste não separa a variante tanh da exata ({worst_erf})");
    }

    /// erf (Abramowitz-Stegun 7.1.26 refinado por série): só para o teste distinguir variantes.
    fn erf(x: f64) -> f64 {
        // série de Taylor até convergir (|x| <= ~8.5 aqui basta com a assíntota nas pontas)
        let (mut sum, mut term, mut n) = (x, x, 0.0f64);
        loop {
            n += 1.0;
            term *= -x * x / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() < 1e-17 {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    }

    // ------------------------------------------------------------------ tokenizador e carregamento

    fn load_tokenizer(pre: Option<&str>, tmpl: Option<&str>) -> Result<Tokenizer, String> {
        let w = make_weights(SMALL, 1);
        let path = tmp_path("tok");
        write_gguf(SMALL, &w, &path, pre, tmpl);
        let r = {
            let g = Gguf::open(&path).unwrap();
            Tokenizer::from_gguf(&g)
        };
        let _ = std::fs::remove_file(&path);
        r
    }

    #[test]
    fn pretokenizer_starcoder_rules() {
        let chars = |s: &str| s.chars().collect::<Vec<char>>();
        let split = |s: &str| -> Vec<String> {
            let c = chars(s);
            let (mut out, mut i) = (Vec::new(), 0);
            while i < c.len() {
                let k = crate::native::tokenizer::next_pretoken_starcoder(&c, i);
                out.push(c[i..k].iter().collect());
                i = k;
            }
            out
        };
        assert_eq!(split("def foo(x):"), ["def", " foo", "(", "x", "):"]);
        assert_eq!(split("a  b"), ["a", " ", " b"]); // \s+(?!\S) deixa o último espaço para a palavra
        assert_eq!(split("x\n    return"), ["x", "\n   ", " return"]);
        assert_eq!(split("don't We'll"), ["don", "'t", " We", "'ll"]);
        assert_eq!(split("é!? "), ["é", "!?", " "]);
        assert_eq!(split("a\tb"), ["a", "\t", "b"]);
    }

    #[test]
    fn tokenizer_digits_and_roundtrip() {
        let t = load_tokenizer(None, None).expect("sem `tokenizer.ggml.pre` deve valer o do StarCoder2");
        let ids = t.encode_plain("x = 12345");
        // cada dígito é um token (pré-tokenizador "starcoder"): x, " ", "=", " ", 1,2,3,4,5 -> 5 dígitos separados
        assert_eq!(t.encode_plain("12345").len(), 5);
        assert!(ids.len() >= 5);
        let dec = |ids: &[u32]| String::from_utf8(ids.iter().flat_map(|&i| t.token_bytes(i)).collect()).unwrap();
        for s in ["the theory of the thing", "def f(x):\n\treturn x + 1  # çã 日本 \u{1F600}", "  \n\n a", "12345 6789", ""] {
            assert_eq!(dec(&t.encode_plain(s)), s, "ida e volta falhou para {s:?}");
        }
        let one = t.encode_plain("7");
        let two = t.encode_plain("77");
        assert_eq!(two, [one[0], one[0]], "dígitos têm de ser separados");
        // pré-tokenizador desconhecido é recusado (em vez de tokenizar errado em silêncio)
        assert!(load_tokenizer(Some("gpt-4o"), None).is_err());
        assert!(load_tokenizer(Some("starcoder"), None).is_ok());
    }

    #[test]
    fn fim_prompt_layout_and_no_bos() {
        let t = load_tokenizer(Some("starcoder"), None).unwrap();
        assert!(t.has_fim());
        assert_eq!(t.bos(), None, "o StarCoder2 não leva BOS em prompts crus");
        let p = t.fim_prompt("ab", "cd").unwrap();
        // layout: <fim_prefix> prefixo <fim_suffix> sufixo <fim_middle>
        let vocab_len = t.vocab_size() as u32;
        let (pre_id, mid_id, suf_id) = (vocab_len - 4, vocab_len - 3, vocab_len - 2);
        assert_eq!(p[0], pre_id);
        assert_eq!(*p.last().unwrap(), mid_id);
        assert_eq!(p.iter().filter(|&&x| x == suf_id).count(), 1);
        // os marcadores digitados num prompt de COMPLETAR viram os mesmos tokens
        assert_eq!(t.raw_prompt("<fim_prefix>ab<fim_suffix>cd<fim_middle>"), p);
        assert!(t.code_stop_ids().contains(&(vocab_len - 1)), "<file_sep> deve encerrar um completar");
    }

    const INSTRUCT_TMPL: &str = "{{bos_token}}{{'You are an exceptionally intelligent coding assistant ...'}}{{'### Instruction\n' + message['content']}}{{'### Response\n'}}";

    #[test]
    fn chat_template_instruct_and_base() {
        let msg = |r: &str, c: &str| crate::engine::ChatMsg { role: r.into(), content: c.into() };
        // base (3B): sem template -> texto cru
        let base = load_tokenizer(Some("starcoder"), None).unwrap();
        let ids = base.chat_prompt(&[msg("system", "ignorado"), msg("user", "def f():")]);
        assert_eq!(ids, base.raw_prompt("def f():"));
        // instruct: o template do GGUF é reconhecido (check_template) e o formato é o `### Instruction`
        let ins = load_tokenizer(Some("starcoder"), Some(INSTRUCT_TMPL)).expect("template ### Instruction deve ser aceito");
        let ids = ins.chat_prompt(&[msg("user", "Hi"), msg("assistant", "Yo"), msg("user", "Again")]);
        let eot = ins.eos[0];
        assert_eq!(ids[0], eot, "o template começa com o BOS `<|endoftext|>`");
        assert_eq!(ids.iter().filter(|&&x| x == eot).count(), 2, "BOS + eos depois da resposta do assistente");
        let text = String::from_utf8(ids.iter().flat_map(|&i| ins.token_bytes(i)).collect()).unwrap();
        assert!(text.ends_with("### Instruction\nAgain\n\n### Response\n"), "{text:?}");
        assert!(text.contains("### Instruction\nHi\n\n### Response\nYo<|endoftext|>\n\n### Instruction\nAgain"), "{text:?}");
        // um template de outro formato na mesma arquitetura é recusado
        assert!(load_tokenizer(Some("starcoder"), Some("<|im_start|>user\n{{ m }}<|im_end|>")).is_err());
    }

    /// O 15B-instruct tem 40 camadas, 48 cabeças e 4 de KV (grupo 12), `head_dim` 128: a lógica de formas aceita (o
    /// tensor real não coube na RAM para rodar, ver D51); aqui só as formas, com um arquivo sintético de grupo 12.
    #[test]
    fn rejects_inconsistent_shapes() {
        let mut w = make_weights(SMALL, 3);
        // viés de Q com tamanho errado
        w.iter_mut().find(|t| t.name == "blk.0.attn_q.bias").unwrap().data.pop();
        w.iter_mut().find(|t| t.name == "blk.0.attn_q.bias").unwrap().dims = vec![SMALL.n_head * SMALL.hd - 1];
        let path = tmp_path("bad");
        write_gguf(SMALL, &w, &path, Some("starcoder"), None);
        let r = Model::load_with(&path, false);
        assert!(r.is_err(), "viés com tamanho errado deveria ser recusado");
        let _ = std::fs::remove_file(&path);
    }

    // ------------------------------------------------------------------ modelo real (lento: `--ignored`)

    /// Versão de UMA carga só para modelos grandes (15B-instruct, 9,9 GB: `starcoder2_real_selfchecks` carrega o arquivo
    /// 3 vezes e não cabe). Chat com o template do arquivo, lote == token a token (logits e KV, bit a bit), continuação
    /// gulosa e velocidade (prefill e geração, medidos aqui; condição de medição fica por conta de quem roda).
    /// `BURITI_TEST_SC2=... cargo test starcoder2_real_big -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn starcoder2_real_big() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(path) = real_path() else { eprintln!("modelo ausente, pulando"); return };
        let t0 = std::time::Instant::now();
        let m = Model::load_with(Path::new(&path), true).expect("carregar");
        eprintln!("carga: {:.1} s", t0.elapsed().as_secs_f64());
        let t = &m.tok;
        let dec = |ids: &[u32]| String::from_utf8_lossy(&ids.iter().flat_map(|&i| t.token_bytes(i)).collect::<Vec<u8>>()).into_owned();
        let msgs = vec![crate::engine::ChatMsg { role: "user".into(), content: "Write a Python function that returns the n-th Fibonacci number.".into() }];
        let prompt = t.chat_prompt(&msgs);
        eprintln!("prompt de chat: {} tokens: {:?}", prompt.len(), dec(&prompt));
        let n_rep = m.sc2.as_ref().unwrap().layers.iter().map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_up, &l.w_down].iter().filter(|t| t.rep.is_some()).count()).sum::<usize>();
        eprintln!("tensores reempacotados: {n_rep} de {}", m.cfg.n_layer * 6);
        // lote == token a token nos primeiros 23 tokens
        let tk: Vec<u32> = prompt.iter().take(23.min(prompt.len())).cloned().collect();
        let mut a = m.new_session(64);
        for (i, &id) in tk.iter().enumerate() {
            m.forward(&mut a, id, i, i == tk.len() - 1);
        }
        let mut b = m.new_session(64);
        let mut i = 0;
        for chunk in [8usize, 8, tk.len() - 16] {
            m.forward_batch(&mut b, &tk[i..i + chunk], i, i + chunk == tk.len());
            i += chunk;
        }
        let kv = m.cfg.n_kv * m.cfg.head_dim;
        for li in 0..m.cfg.n_layer {
            assert!(a.k[li][..tk.len() * kv] == b.k[li][..tk.len() * kv], "K difere (camada {li})");
            assert!(a.v[li][..tk.len() * kv] == b.v[li][..tk.len() * kv], "V difere (camada {li})");
        }
        assert!(a.logits.iter().zip(&b.logits).all(|(x, y)| x.to_bits() == y.to_bits()), "logits diferem");
        eprintln!("lote == token a token, bit a bit ({} tokens)", tk.len());
        drop((a, b));
        // prefill em lote do prompt de chat + geração gulosa (tempo)
        let mut s = m.new_session(512);
        let tp = std::time::Instant::now();
        m.forward_batch(&mut s, &prompt, 0, true);
        let dp = tp.elapsed().as_secs_f64();
        assert!(s.logits.iter().all(|v| v.is_finite()), "logits não finitos");
        eprintln!("PREFILL {} tokens em {:.2} s = {:.2} tok/s", prompt.len(), dp, prompt.len() as f64 / dp);
        let mut out = Vec::new();
        let mut pos = prompt.len();
        let tg = std::time::Instant::now();
        let n_gen = 24;
        for _ in 0..n_gen {
            let next = s.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            out.push(next);
            if t.eos.contains(&next) { break; }
            m.forward(&mut s, next, pos, true);
            pos += 1;
        }
        let dg = tg.elapsed().as_secs_f64();
        eprintln!("GERACAO {} tokens em {:.2} s = {:.3} tok/s", out.len(), dg, out.len() as f64 / dg);
        eprintln!("continuação gulosa: {:?}", dec(&out));
    }

    fn real_path() -> Option<String> {
        let p = std::env::var("BURITI_TEST_SC2").unwrap_or_else(|_| format!("{}/buriti-data/models/starcoder2-3b-Q4_K_M.gguf", env!("CARGO_MANIFEST_DIR")));
        std::path::Path::new(&p).exists().then_some(p)
    }

    /// Autochecagens no arquivo real: ida e volta do tokenizador, logits finitos, determinismo, lote == token a token
    /// (logits e bytes do KV, com e sem reempacotamento), completar uma função Python em modo guloso e o layout do FIM.
    /// `cargo test starcoder2_real -- --ignored --nocapture` (lento no 3B: alguns minutos).
    #[test]
    #[ignore]
    fn starcoder2_real_selfchecks() {
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(path) = real_path() else { eprintln!("modelo ausente, pulando"); return };
        let m = Model::load_with(Path::new(&path), true).expect("carregar");
        let t = &m.tok;
        let dec = |ids: &[u32]| String::from_utf8_lossy(&ids.iter().flat_map(|&i| t.token_bytes(i)).collect::<Vec<u8>>()).into_owned();
        // 1) ida e volta
        let texts = [
            "def fibonacci(n):\n\tif n <= 1:\n\t\treturn n\n\treturn fibonacci(n-1) + fibonacci(n-2)\n",
            "class VeryLongIdentifierNameForTestingTokenizationBehaviour_v2_FINAL(BaseHTTPRequestHandlerMixin):",
            "x = [1, 22, 333, 4444, 55555]  # números 3.14159 e 1,000,000",
            "Olá, ação, coração, ñandú, über, naïve café. 日本語 \u{1F600}\u{1F389} русский العربية",
            "  leading,   multiple    spaces   and trailing   \n\n\n\t\tindented\r\nCRLF\r\n",
            "<fim_prefix>def f(x):\n    <fim_suffix>\n    return y<fim_middle>",
        ];
        for s in texts {
            assert_eq!(dec(&t.encode_plain(s)), s, "ida e volta falhou: {s:?}");
        }
        // 2) layout do FIM e das marcas de parada
        let fim = t.fim_prompt("def f(x):\n    ", "\n    return y\n").expect("FIM");
        assert_eq!(fim, t.raw_prompt("<fim_prefix>def f(x):\n    <fim_suffix>\n    return y\n<fim_middle>"));
        eprintln!("FIM ({} tokens): {:?}", fim.len(), fim);
        eprintln!("ids de parada de código: {:?}, eos: {:?}", t.code_stop_ids(), t.eos);
        // 3) logits finitos, determinismo e guloso
        let prompt = t.raw_prompt("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return");
        let run = |m: &Model, batch: bool| -> (Vec<f32>, Vec<u32>) {
            let mut s = m.new_session(256);
            if batch {
                m.forward_batch(&mut s, &prompt, 0, true);
            } else {
                for (i, &id) in prompt.iter().enumerate() {
                    m.forward(&mut s, id, i, i == prompt.len() - 1);
                }
            }
            let first = s.logits.clone();
            let mut out = Vec::new();
            let mut pos = prompt.len();
            for _ in 0..14 {
                let next = s.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
                out.push(next);
                m.forward(&mut s, next, pos, true);
                pos += 1;
            }
            (first, out)
        };
        let (l1, g1) = run(&m, true);
        assert!(l1.iter().all(|v| v.is_finite()), "logits não finitos");
        let (l2, g2) = run(&m, true);
        assert!(l1.iter().zip(&l2).all(|(a, b)| a.to_bits() == b.to_bits()) && g1 == g2, "não determinístico");
        eprintln!("continuação gulosa: {:?}", dec(&g1));
        // 4) lote == token a token (logits e KV), no layout original e no reempacotado
        for repack in [true, false] {
            let m2 = Model::load_with(Path::new(&path), repack).unwrap();
            let tk: Vec<u32> = prompt.iter().take(23).cloned().collect();
            let mut a = m2.new_session(64);
            for (i, &id) in tk.iter().enumerate() {
                m2.forward(&mut a, id, i, i == tk.len() - 1);
            }
            let mut b = m2.new_session(64);
            let mut i = 0;
            for chunk in [8usize, 8, tk.len() - 16] {
                m2.forward_batch(&mut b, &tk[i..i + chunk], i, i + chunk == tk.len());
                i += chunk;
            }
            let kv = m2.cfg.n_kv * m2.cfg.head_dim;
            for li in 0..m2.cfg.n_layer {
                assert!(a.k[li][..tk.len() * kv] == b.k[li][..tk.len() * kv], "K difere (camada {li}, repack={repack})");
                assert!(a.v[li][..tk.len() * kv] == b.v[li][..tk.len() * kv], "V difere (camada {li}, repack={repack})");
            }
            assert!(a.logits.iter().zip(&b.logits).all(|(x, y)| x.to_bits() == y.to_bits()), "logits diferem (repack={repack})");
            eprintln!("lote == token a token, bit a bit (repack={repack}, {} tensores reempacotados)", m2.sc2.as_ref().unwrap().layers.iter().map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_up, &l.w_down].iter().filter(|t| t.rep.is_some()).count()).sum::<usize>());
        }
    }
}
