//! A whole decoder model on the GPU, one token at a time (task G8).
//!
//! `GpuModel::open` reads the weights of a loaded CPU `Model` (through `Model::gpu_view`, no second file parser), uploads every matrix
//! once in its quantized GGUF format (row-aligned chunks that respect the adapter limits, see `matvec`), the norms and biases as f32, and
//! allocates one f16 KV cache per layer (see `attention`). `forward(token, pos)` then records the whole token (embedding gather, every
//! layer, final norm, output projection) as about 20 dispatches per layer inside ONE compute pass, submits it once and reads the logits
//! back: the host does four small parameter writes (token id, RoPE table, KV position, attention position) and nothing else per token.
//!
//! Arithmetic differs from the CPU engine in one known way: the CPU quantizes the activation vector to int8 before every matrix product,
//! the GPU keeps f32 activations (D64). The meaningful reference is therefore the CPU engine with `BURITI_FLOAT_ACT=1`.
//!
//! Covered (G8): Llama/Qwen2-style decoders (RMSNorm, Q/K/V with optional bias, RoPE, GQA attention over the f16 KV cache, SwiGLU or
//! GeGLU, tied or separate output matrix). Not covered yet (task G11): per-head Q/K norms (Qwen3, Gemma 3), post-attention and post-FFN
//! norms and sliding-window layers (Gemma 3), StarCoder2's own forward, fused Q/K/V or gate/up layouts beyond what `gpu_view` already splits.

use super::attention::{Attention, AttnShape, KvCache, KvWrite};
use super::batch::{self, AttnBatch, KvWriteBatch, TokenOp};
use super::device::{GpuBuffer, GpuDevice, RunTiming, Step, DEFAULT_TIMEOUT};
use super::gemm::{self, GemmKernel, GemmParams};
use super::matvec::{self, upload_matrix, GpuMatrix, MvKernel};
use super::ops::{self, Act, Op, OpCall};
use super::pack;
use super::select::AdapterSelector;
use crate::native::gguf::Dtype;
use crate::native::model::{GpuView, Model, TensorView};
use std::sync::Arc;

struct GpuLayer {
    attn_norm: GpuBuffer,
    ffn_norm: GpuBuffer,
    wq: GpuMatrix,
    wk: GpuMatrix,
    wv: GpuMatrix,
    wo: GpuMatrix,
    bq: Option<GpuBuffer>,
    bk: Option<GpuBuffer>,
    bv: Option<GpuBuffer>,
    /// per-head RMS norms on Q and K (Qwen3, Gemma 3): the weight is shared by all heads
    q_norm: Option<GpuBuffer>,
    k_norm: Option<GpuBuffer>,
    /// Gemma 3: RMS norms applied to the attention and FFN outputs before the residual add
    post_attn_norm: Option<GpuBuffer>,
    post_ffn_norm: Option<GpuBuffer>,
    /// sliding-window attention on this layer, and the local RoPE frequency set (Gemma 3 local layers)
    swa: bool,
    local: bool,
    /// StarCoder2: LayerNorm biases and the output / FFN biases (the norms are LayerNorms when `attn_norm_b` is present)
    attn_norm_b: Option<GpuBuffer>,
    ffn_norm_b: Option<GpuBuffer>,
    bo: Option<GpuBuffer>,
    b_up: Option<GpuBuffer>,
    b_down: Option<GpuBuffer>,
    /// `None` in a plain (ungated) MLP
    w_gate: Option<GpuMatrix>,
    w_up: GpuMatrix,
    w_down: GpuMatrix,
    cache: KvCache,
}

pub struct GpuModel {
    dev: GpuDevice,
    pub n_vocab: usize,
    pub ctx: usize,
    embd: GpuMatrix,
    /// `None` when the output projection is the embedding matrix (tied weights): the same buffers serve both.
    output: Option<GpuMatrix>,
    out_norm: GpuBuffer,
    out_norm_b: Option<GpuBuffer>,
    layers: Vec<GpuLayer>,
    /// One kernel and pipeline per weight format in use (a handful of entries: a linear search).
    kernels: Vec<(Dtype, MvKernel, Arc<wgpu::ComputePipeline>)>,
    // activations
    x: GpuBuffer,
    xn: GpuBuffer,
    q: GpuBuffer,
    /// the normalized Q and K when a layer has per-head norms (RMSNorm needs a separate output buffer)
    qn: GpuBuffer,
    kn: GpuBuffer,
    kk: GpuBuffer,
    vv: GpuBuffer,
    att: GpuBuffer,
    tmp: GpuBuffer,
    tmp2: GpuBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    act_out: GpuBuffer,
    logits: GpuBuffer,
    ids: GpuBuffer,
    cs: GpuBuffer,
    cs_local: GpuBuffer,
    // dispatches that are the same for every layer (the parameters are fixed, or rewritten per token)
    gather: Vec<OpCall>,
    rms: OpCall,
    ln: Option<OpCall>,
    add_up: Option<OpCall>,
    add_q: OpCall,
    add_k: OpCall,
    add_v: OpCall,
    rope_q: OpCall,
    rope_k: OpCall,
    rms_q: OpCall,
    rms_k: OpCall,
    add_res: OpCall,
    act: OpCall,
    kvw: KvWrite,
    attn: Attention,
    /// the attention of sliding-window layers (its position parameters carry the window)
    attn_local: Attention,
    inv_freq: Vec<f32>,
    inv_freq_local: Vec<f32>,
    rope_mscale: f32,
    /// The dimensions of the model, kept for the prefill buffers.
    dims: Dims,
    /// Batched prefill state, built by `enable_prefill`.
    prefill: Option<Prefill>,
    /// Bytes of weights resident on the GPU, and of KV cache.
    pub weight_bytes: u64,
    pub kv_bytes: u64,
}

#[derive(Clone, Copy)]
struct Dims {
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    n_rot: usize,
    rope_norm: bool,
    eps: f32,
    embed_scale: f32,
    gelu: bool,
    window: usize,
    plain: bool,
}

/// Target duration of one prefill submission, far below the 2 s after which the Windows driver resets the GPU.
const PASS_BUDGET_S: f64 = 0.4;

/// Everything the batched prefill needs: activation buffers for `t_cap` tokens, one GEMM pipeline per weight format and the batched ops.
struct Prefill {
    t_cap: usize,
    /// Layers per submission. One pass over a large model with a full batch can run longer than the 2 s the Windows driver allows
    /// before it declares the GPU hung and removes the device, so the prefill is submitted in groups of layers sized from the time of
    /// the previous group (it starts at 1).
    group: std::sync::atomic::AtomicUsize,
    tp: GpuBuffer,
    gemm: Vec<(Dtype, GemmKernel, Arc<wgpu::ComputePipeline>)>,
    /// Per layer: wq, wk, wv, wo, gate, up, down.
    gparams: Vec<Vec<GemmParams>>,
    bx: GpuBuffer,
    bxn: GpuBuffer,
    bq: GpuBuffer,
    bqn: GpuBuffer,
    bkn: GpuBuffer,
    bkk: GpuBuffer,
    bvv: GpuBuffer,
    batt: GpuBuffer,
    btmp: GpuBuffer,
    bgate: GpuBuffer,
    bup: GpuBuffer,
    bact: GpuBuffer,
    ids: GpuBuffer,
    cs: GpuBuffer,
    cs_local: GpuBuffer,
    btmp2: GpuBuffer,
    gather: Vec<OpCall>,
    rms: OpCall,
    ln: Option<OpCall>,
    bias_o: Option<TokenOp>,
    bias_up: Option<TokenOp>,
    bias_down: Option<TokenOp>,
    rms_qb: OpCall,
    rms_kb: OpCall,
    bias_q: TokenOp,
    bias_k: TokenOp,
    bias_v: TokenOp,
    rope_q: TokenOp,
    rope_k: TokenOp,
    kvw: KvWriteBatch,
    attn: AttnBatch,
    attn_local: AttnBatch,
    add_res: OpCall,
    act: OpCall,
    copy: OpCall,
}

fn upload(dev: &GpuDevice, t: &TensorView, rows: usize, what: &str) -> Result<GpuMatrix, String> {
    if rows > t.rows {
        return Err(format!("{what}: {rows} rows wanted, the tensor has {}", t.rows));
    }
    let rb = t.row_bytes;
    upload_matrix(dev, t.dtype, rows as u64, t.cols as u64, &|s, c| t.bytes[s as usize * rb..(s + c) as usize * rb].to_vec(), None).map_err(|e| format!("{what}: {e}"))
}

fn expect_shape(what: &str, m: &GpuMatrix, rows: usize, cols: usize) -> Result<(), String> {
    if m.rows as usize != rows || m.cols as usize != cols {
        return Err(format!("{what}: shape {}x{}, expected {rows}x{cols}", m.rows, m.cols));
    }
    Ok(())
}

fn norm_buffer(dev: &GpuDevice, w: &[f32], n: usize, what: &str) -> Result<GpuBuffer, String> {
    if w.len() != n {
        return Err(format!("{what}: {} values, expected {n}", w.len()));
    }
    dev.storage_from_f32(w, what)
}

impl GpuModel {
    /// Opens the adapter chosen by `sel` and loads `model` onto it for sessions of up to `ctx` positions.
    pub fn open(model: &Model, sel: &AdapterSelector, ctx: usize) -> Result<GpuModel, String> {
        let dev = GpuDevice::open(sel)?;
        GpuModel::load(dev, model, ctx)
    }

    pub fn load(dev: GpuDevice, model: &Model, ctx: usize) -> Result<GpuModel, String> {
        let view: GpuView = model.gpu_view()?;
        let c = view.cfg;
        let (n_embd, n_ff, n_head, n_kv, hd, n_vocab) = (c.n_embd, c.n_ff, c.n_head, c.n_kv, c.head_dim, c.n_vocab);
        if ctx == 0 || ctx > c.n_ctx_train.max(ctx) {
            return Err("gpu model: bad context size".into());
        }
        let kv_dim = n_kv * hd;
        let shape = AttnShape { n_head: n_head as u32, n_kv: n_kv as u32, head_dim: hd as u32, max_ctx: ctx as u32 };
        shape.validate()?;

        // weights
        let embd = upload(&dev, &view.embd, n_vocab, "token embedding")?;
        expect_shape("token embedding", &embd, n_vocab, n_embd)?;
        let tied = view.output.off == view.embd.off && view.output.dtype == view.embd.dtype && view.output.cols == view.embd.cols;
        let output = if tied {
            None
        } else {
            let o = upload(&dev, &view.output, n_vocab, "output matrix")?;
            expect_shape("output matrix", &o, n_vocab, n_embd)?;
            Some(o)
        };
        let out_norm = norm_buffer(&dev, view.out_norm, n_embd, "output norm")?;
        let plain = view.plain_mlp;
        let out_norm_b = if view.out_norm_b.is_empty() { None } else { Some(norm_buffer(&dev, view.out_norm_b, n_embd, "output norm bias")?) };
        let mut layers = Vec::new();
        for (i, l) in view.layers.iter().enumerate() {
            let tag = |s: &str| format!("layer {i} {s}");
            let wq = upload(&dev, &l.wq, n_head * hd, &tag("wq"))?;
            let wk = upload(&dev, &l.wk, kv_dim, &tag("wk"))?;
            let wv = upload(&dev, &l.wv, kv_dim, &tag("wv"))?;
            let wo = upload(&dev, &l.wo, n_embd, &tag("wo"))?;
            let w_gate = if plain { None } else { Some(upload(&dev, &l.w_gate, n_ff, &tag("gate"))?) };
            let w_up = upload(&dev, &l.w_up, n_ff, &tag("up"))?;
            let w_down = upload(&dev, &l.w_down, n_embd, &tag("down"))?;
            expect_shape(&tag("wq"), &wq, n_head * hd, n_embd)?;
            expect_shape(&tag("wk"), &wk, kv_dim, n_embd)?;
            expect_shape(&tag("wv"), &wv, kv_dim, n_embd)?;
            expect_shape(&tag("wo"), &wo, n_embd, n_head * hd)?;
            if let Some(g) = &w_gate {
                expect_shape(&tag("gate"), g, n_ff, n_embd)?;
            }
            expect_shape(&tag("up"), &w_up, n_ff, n_embd)?;
            expect_shape(&tag("down"), &w_down, n_embd, n_ff)?;
            let bias = |b: &[f32], n: usize, s: &str| -> Result<Option<GpuBuffer>, String> {
                if b.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(norm_buffer(&dev, b, n, &tag(s))?))
                }
            };
            layers.push(GpuLayer {
                attn_norm: norm_buffer(&dev, l.attn_norm, n_embd, &tag("attn norm"))?,
                ffn_norm: norm_buffer(&dev, l.ffn_norm, n_embd, &tag("ffn norm"))?,
                q_norm: if l.q_norm.is_empty() { None } else { Some(norm_buffer(&dev, l.q_norm, hd, &tag("q norm"))?) },
                k_norm: if l.k_norm.is_empty() { None } else { Some(norm_buffer(&dev, l.k_norm, hd, &tag("k norm"))?) },
                post_attn_norm: if l.post_attn_norm.is_empty() { None } else { Some(norm_buffer(&dev, l.post_attn_norm, n_embd, &tag("post-attention norm"))?) },
                post_ffn_norm: if l.post_ffn_norm.is_empty() { None } else { Some(norm_buffer(&dev, l.post_ffn_norm, n_embd, &tag("post-FFN norm"))?) },
                swa: l.swa,
                local: l.local,
                attn_norm_b: bias(l.attn_norm_b, n_embd, "attn norm bias")?,
                ffn_norm_b: bias(l.ffn_norm_b, n_embd, "ffn norm bias")?,
                bo: bias(l.bo, n_embd, "bo")?,
                b_up: bias(l.b_up, n_ff, "b_up")?,
                b_down: bias(l.b_down, n_embd, "b_down")?,
                bq: bias(l.bq, n_head * hd, "bq")?,
                bk: bias(l.bk, kv_dim, "bk")?,
                bv: bias(l.bv, kv_dim, "bv")?,
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                cache: KvCache::new(&dev, shape)?,
            });
        }

        // one pipeline per weight format in use
        let mut kernels: Vec<(Dtype, MvKernel, Arc<wgpu::ComputePipeline>)> = Vec::new();
        let mut dtypes = vec![embd.dtype];
        if let Some(o) = &output {
            dtypes.push(o.dtype);
        }
        for l in &layers {
            for m in [&l.wq, &l.wk, &l.wv, &l.wo].into_iter().chain(l.w_gate.as_ref()).chain([&l.w_up, &l.w_down]) {
                dtypes.push(m.dtype);
            }
        }
        for d in dtypes {
            if !kernels.iter().any(|(kd, _, _)| *kd == d) {
                let k = MvKernel::new(d, pack::DEFAULT_WG)?;
                let p = k.pipeline(&dev)?;
                kernels.push((d, k, p));
            }
        }

        // activations
        let f = |n: usize, s: &str| dev.create_storage((n * 4) as u64, s);
        let x = f(n_embd, "g.x")?;
        let xn = f(n_embd, "g.xn")?;
        let q = f(n_head * hd, "g.q")?;
        let qn = f(n_head * hd, "g.qn")?;
        let kn = f(kv_dim, "g.kn")?;
        let kk = f(kv_dim, "g.k")?;
        let vv = f(kv_dim, "g.v")?;
        let att = f(n_head * hd, "g.att")?;
        let tmp = f(n_embd.max(n_head * hd), "g.tmp")?;
        let tmp2 = f(n_embd, "g.tmp2")?;
        let gate = f(n_ff, "g.gate")?;
        let up = if plain { dev.storage_from_f32(&vec![1.0f32; n_ff], "g.ones")? } else { f(n_ff, "g.up")? };
        let act_out = f(n_ff, "g.act")?;
        let logits = f(n_vocab, "g.logits")?;
        let ids = dev.create_storage(4, "g.ids")?;
        let inv_freq = model.rope_freqs(ctx, false).to_vec();
        let cs = dev.create_storage((inv_freq.len() * 2 * 4) as u64, "g.rope")?;
        let inv_freq_local = model.rope_freqs(ctx, true).to_vec();
        let cs_local = dev.create_storage((inv_freq_local.len().max(1) * 2 * 4) as u64, "g.rope_local")?;
        if c.n_rot / 2 != inv_freq.len() {
            return Err(format!("rope: {} frequencies for {} rotating dimensions", inv_freq.len(), c.n_rot));
        }

        // dispatches
        let gather = ops::gather(&dev, &Op::gather(embd.dtype)?, &embd, 1, c.embed_scale)?;
        let rms = ops::rmsnorm(&dev, &Op::rmsnorm(), n_embd as u32, 1, c.eps)?;
        let ln = if out_norm_b.is_some() { Some(ops::layernorm(&dev, &Op::layernorm(), n_embd as u32, 1, c.eps)?) } else { None };
        let add = Op::add();
        let add_up = if plain { Some(ops::add(&dev, &add, n_ff as u32)?) } else { None };
        let add_q = ops::add(&dev, &add, (n_head * hd) as u32)?;
        let add_k = ops::add(&dev, &add, kv_dim as u32)?;
        let add_v = ops::add(&dev, &add, kv_dim as u32)?;
        let rope = Op::rope();
        let rms_q = ops::rmsnorm(&dev, &Op::rmsnorm(), hd as u32, n_head as u32, c.eps)?;
        let rms_k = ops::rmsnorm(&dev, &Op::rmsnorm(), hd as u32, n_kv as u32, c.eps)?;
        let rope_q = ops::rope(&dev, &rope, n_head as u32, hd as u32, c.n_rot as u32, c.rope_norm)?;
        let rope_k = ops::rope(&dev, &rope, n_kv as u32, hd as u32, c.n_rot as u32, c.rope_norm)?;
        let add_res = ops::add(&dev, &add, n_embd as u32)?;
        let act = ops::act(&dev, &Op::act(if c.gelu { Act::GeluMul } else { Act::SiluMul }), n_ff as u32)?;
        let kvw = KvWrite::new(&dev, &Op::kv_write(), shape)?;
        let attn = Attention::new(&dev, &Op::attn_decode(), shape)?;
        let attn_local = Attention::new(&dev, &Op::attn_decode(), shape)?;

        let mut weight_bytes = embd.resident_bytes + output.as_ref().map_or(0, |o| o.resident_bytes);
        for l in &layers {
            for m in [&l.wq, &l.wk, &l.wv, &l.wo].into_iter().chain(l.w_gate.as_ref()).chain([&l.w_up, &l.w_down]) {
                weight_bytes += m.resident_bytes;
            }
        }
        let kv_bytes = layers.len() as u64 * 2 * shape.cache_bytes();
        dev.check_health()?;
        Ok(GpuModel {
            n_vocab,
            ctx,
            embd,
            output,
            out_norm,
            out_norm_b,
            layers,
            kernels,
            x,
            xn,
            q,
            qn,
            kn,
            kk,
            vv,
            att,
            tmp,
            tmp2,
            gate,
            up,
            act_out,
            logits,
            ids,
            cs,
            cs_local,
            gather,
            rms,
            ln,
            add_up,
            add_q,
            add_k,
            add_v,
            rope_q,
            rope_k,
            rms_q,
            rms_k,
            add_res,
            act,
            kvw,
            attn,
            attn_local,
            inv_freq,
            inv_freq_local,
            rope_mscale: c.rope_mscale,
            dims: Dims { n_embd, n_ff, n_head, n_kv, hd, n_rot: c.n_rot, rope_norm: c.rope_norm, eps: c.eps, embed_scale: c.embed_scale, gelu: c.gelu, window: c.window, plain },
            prefill: None,
            weight_bytes,
            kv_bytes,
            dev,
        })
    }

    pub fn adapter_name(&self) -> &str {
        self.dev.name()
    }

    fn mv<'a>(&'a self, st: &mut Vec<Step<'a>>, m: &'a GpuMatrix, x: &'a GpuBuffer, y: &'a GpuBuffer) -> Result<(), String> {
        let (_, k, p) = self.kernels.iter().find(|(d, _, _)| *d == m.dtype).ok_or_else(|| format!("no kernel for {}", pack::name(m.dtype)))?;
        st.extend(matvec::steps(&self.dev, k.rows_per_group, p, &[m], x, y)?);
        Ok(())
    }

    /// RMSNorm, or LayerNorm with bias when the model has one (StarCoder2).
    fn norm_step<'a>(&'a self, st: &mut Vec<Step<'a>>, x: &'a GpuBuffer, w: &'a GpuBuffer, b: Option<&'a GpuBuffer>, out: &'a GpuBuffer) {
        match (b, &self.ln) {
            (Some(b), Some(ln)) => st.push(ln.step(&[x, w, b, out])),
            _ => st.push(self.rms.step(&[x, w, out])),
        }
    }

    /// One token at position `pos`: returns the logits (`n_vocab` floats) and the timing of the pass. The KV cache of positions
    /// `0..pos` must come from earlier calls with the same session (a new conversation simply starts again at position 0).
    pub fn forward_timed(&self, token: u32, pos: usize) -> Result<(Vec<f32>, RunTiming), String> {
        if pos >= self.ctx {
            return Err(format!("position {pos} is outside the context of {} positions", self.ctx));
        }
        if token as usize >= self.n_vocab {
            return Err(format!("token {token} is outside the vocabulary of {}", self.n_vocab));
        }
        let dev = &self.dev;
        dev.upload_bytes(&self.ids, 0, &token.to_le_bytes())?;
        dev.upload_f32(&self.cs, 0, &ops::rope_table(pos as u32, &self.inv_freq, self.rope_mscale))?;
        if self.layers.iter().any(|l| l.local) {
            dev.upload_f32(&self.cs_local, 0, &ops::rope_table(pos as u32, &self.inv_freq_local, self.rope_mscale))?;
        }
        self.kvw.set_position(dev, pos as u32)?;
        self.attn.set_position(dev, pos as u32, None)?;
        if self.layers.iter().any(|l| l.swa) {
            self.attn_local.set_position(dev, pos as u32, Some(self.dims.window as u32))?;
        }

        let mut st: Vec<Step> = Vec::with_capacity(self.layers.len() * 24 + 8);
        let mut marks: Vec<(&'static str, usize)> = vec![("gather", 0)];
        for (i, call) in self.gather.iter().enumerate() {
            st.push(call.step(&ops::gather_io(&self.embd, i, &self.ids, &self.x)));
        }
        for l in &self.layers {
            marks.push(("norm", st.len()));
            self.norm_step(&mut st, &self.x, &l.attn_norm, l.attn_norm_b.as_ref(), &self.xn);
            marks.push(("qkv", st.len()));
            self.mv(&mut st, &l.wq, &self.xn, &self.q)?;
            self.mv(&mut st, &l.wk, &self.xn, &self.kk)?;
            self.mv(&mut st, &l.wv, &self.xn, &self.vv)?;
            if let Some(b) = &l.bq {
                st.push(self.add_q.step(&[&self.q, b]));
            }
            if let Some(b) = &l.bk {
                st.push(self.add_k.step(&[&self.kk, b]));
            }
            if let Some(b) = &l.bv {
                st.push(self.add_v.step(&[&self.vv, b]));
            }
            // per-head norms (Qwen3, Gemma 3): the normalized copies replace q and k from here on
            let (qb, kb) = match (&l.q_norm, &l.k_norm) {
                (Some(qw), Some(kw)) => {
                    st.push(self.rms_q.step(&[&self.q, qw, &self.qn]));
                    st.push(self.rms_k.step(&[&self.kk, kw, &self.kn]));
                    (&self.qn, &self.kn)
                }
                _ => (&self.q, &self.kk),
            };
            let cs = if l.local { &self.cs_local } else { &self.cs };
            let at = if l.swa { &self.attn_local } else { &self.attn };
            marks.push(("rope+kv", st.len()));
            st.push(self.rope_q.step(&[qb, cs]));
            st.push(self.rope_k.step(&[kb, cs]));
            st.push(self.kvw.call.step(&[kb, &self.vv, &l.cache.k, &l.cache.v]));
            marks.push(("attention", st.len()));
            st.push(at.call.step(&[qb, &l.cache.k, &l.cache.v, &at.scores, &self.att]));
            marks.push(("wo", st.len()));
            self.mv(&mut st, &l.wo, &self.att, &self.tmp)?;
            marks.push(("add+norm", st.len()));
            if let Some(b) = &l.bo {
                st.push(self.add_res.step(&[&self.tmp, b]));
            }
            let a_out = match &l.post_attn_norm {
                Some(w) => {
                    st.push(self.rms.step(&[&self.tmp, w, &self.tmp2]));
                    &self.tmp2
                }
                None => &self.tmp,
            };
            st.push(self.add_res.step(&[&self.x, a_out]));
            self.norm_step(&mut st, &self.x, &l.ffn_norm, l.ffn_norm_b.as_ref(), &self.xn);
            marks.push(("gate+up", st.len()));
            if let Some(g) = &l.w_gate {
                self.mv(&mut st, g, &self.xn, &self.gate)?;
                self.mv(&mut st, &l.w_up, &self.xn, &self.up)?;
            } else {
                // plain MLP: gate = up(x) + b_up, activation GELU(gate) * ones
                self.mv(&mut st, &l.w_up, &self.xn, &self.gate)?;
                if let (Some(b), Some(a)) = (&l.b_up, &self.add_up) {
                    st.push(a.step(&[&self.gate, b]));
                }
            }
            marks.push(("act", st.len()));
            st.push(self.act.step(&[&self.gate, &self.up, &self.act_out]));
            marks.push(("down", st.len()));
            self.mv(&mut st, &l.w_down, &self.act_out, &self.tmp)?;
            marks.push(("add", st.len()));
            if let Some(b) = &l.b_down {
                st.push(self.add_res.step(&[&self.tmp, b]));
            }
            let f_out = match &l.post_ffn_norm {
                Some(w) => {
                    st.push(self.rms.step(&[&self.tmp, w, &self.tmp2]));
                    &self.tmp2
                }
                None => &self.tmp,
            };
            st.push(self.add_res.step(&[&self.x, f_out]));
        }
        marks.push(("tail+lm_head", st.len()));
        self.norm_step(&mut st, &self.x, &self.out_norm, self.out_norm_b.as_ref(), &self.xn);
        self.mv(&mut st, self.output.as_ref().unwrap_or(&self.embd), &self.xn, &self.logits)?;
        if std::env::var("BURITI_GPU_PROFILE").is_ok() {
            // diagnostic: every step alone in its own pass (GPU time from timestamp queries), summed per kind
            let mut sums: Vec<(&'static str, f64, usize)> = Vec::new();
            for (k, (name, from)) in marks.iter().enumerate() {
                let to = marks.get(k + 1).map_or(st.len(), |m| m.1);
                for step in &st[*from..to] {
                    let r = dev.run_steps(std::slice::from_ref(step), 1, DEFAULT_TIMEOUT)?;
                    let s = r.gpu_seconds.unwrap_or(r.wall_seconds);
                    match sums.iter_mut().find(|e| e.0 == *name) {
                        Some(e) => {
                            e.1 += s;
                            e.2 += 1;
                        }
                        None => sums.push((name, s, 1)),
                    }
                }
            }
            let total: f64 = sums.iter().map(|e| e.1).sum();
            println!("decode profile at position {pos} (GPU time, each step run alone): total {:.2} ms, {} dispatches", total * 1e3, st.len());
            for (name, s, n) in &sums {
                println!("  {name:<14} {:>8.3} ms  {:>5.1}%  ({n} dispatches, {:.0} us each)", s * 1e3, s / total * 100.0, s * 1e6 / *n as f64);
            }
        }
        let timing = dev.run_steps(&st, 1, DEFAULT_TIMEOUT)?;
        let logits = dev.read_back_f32(&self.logits, self.n_vocab, DEFAULT_TIMEOUT)?;
        Ok((logits, timing))
    }

    pub fn forward(&self, token: u32, pos: usize) -> Result<Vec<f32>, String> {
        Ok(self.forward_timed(token, pos)?.0)
    }
}

impl GpuModel {
    /// Builds the batched prefill path for batches of up to `t_cap` tokens (a multiple of 32 is the efficient choice: the GEMM tile is
    /// 32 tokens). The activation buffers, the GEMM pipelines and the batched ops live as long as the model.
    pub fn enable_prefill(&mut self, t_cap: usize) -> Result<(), String> {
        if t_cap == 0 || t_cap > self.ctx {
            return Err(format!("prefill capacity {t_cap}: must be between 1 and the context of {} positions", self.ctx));
        }
        let d = self.dims;
        let dev = &self.dev;
        let (kv_dim, qd) = (d.n_kv * d.hd, d.n_head * d.hd);
        let tc = t_cap as u32;
        let shape = AttnShape { n_head: d.n_head as u32, n_kv: d.n_kv as u32, head_dim: d.hd as u32, max_ctx: self.ctx as u32 };
        let f = |n: usize, s: &str| dev.create_storage((n * 4) as u64, s);
        // one GEMM pipeline per format of the layer matrices
        let mut gemms: Vec<(Dtype, GemmKernel, Arc<wgpu::ComputePipeline>)> = Vec::new();
        let mut gparams = Vec::new();
        for l in &self.layers {
            // the same slot order for every layer: wq, wk, wv, wo, gate (the up matrix again in a plain MLP), up, down
            let ms = [&l.wq, &l.wk, &l.wv, &l.wo, l.w_gate.as_ref().unwrap_or(&l.w_up), &l.w_up, &l.w_down];
            let mut v = Vec::new();
            for m in ms {
                if !gemms.iter().any(|(dt, _, _)| *dt == m.dtype) {
                    let k = GemmKernel::new(m.dtype)?;
                    let p = k.pipeline(dev)?;
                    gemms.push((m.dtype, k, p));
                }
                v.push(gemm::gemm_params(dev, m)?);
            }
            gparams.push(v);
        }
        let cs_len = self.inv_freq.len() * 2;
        let gather = ops::gather(dev, &Op::gather(self.embd.dtype)?, &self.embd, tc, d.embed_scale)?;
        let add = Op::add();
        let pf = Prefill {
            t_cap,
            group: std::sync::atomic::AtomicUsize::new(1),
            tp: gemm::token_count_buffer(dev)?,
            gemm: gemms,
            gparams,
            bx: f(t_cap * d.n_embd, "p.x")?,
            bxn: f(t_cap * d.n_embd, "p.xn")?,
            bq: f(t_cap * qd, "p.q")?,
            bqn: f(t_cap * qd, "p.qn")?,
            bkn: f(t_cap * kv_dim, "p.kn")?,
            bkk: f(t_cap * kv_dim, "p.k")?,
            bvv: f(t_cap * kv_dim, "p.v")?,
            batt: f(t_cap * qd, "p.att")?,
            btmp: f(t_cap * d.n_embd, "p.tmp")?,
            bgate: f(t_cap * d.n_ff, "p.gate")?,
            bup: if d.plain { dev.storage_from_f32(&vec![1.0f32; t_cap * d.n_ff], "p.ones")? } else { f(t_cap * d.n_ff, "p.up")? },
            bact: f(t_cap * d.n_ff, "p.act")?,
            ids: dev.create_storage((t_cap * 4) as u64, "p.ids")?,
            cs: f(t_cap * cs_len, "p.rope")?,
            cs_local: f(t_cap * self.inv_freq_local.len().max(1) * 2, "p.rope_local")?,
            btmp2: f(t_cap * d.n_embd, "p.tmp2")?,
            gather,
            rms: ops::rmsnorm(dev, &Op::rmsnorm(), d.n_embd as u32, tc, d.eps)?,
            ln: if self.out_norm_b.is_some() { Some(ops::layernorm(dev, &Op::layernorm(), d.n_embd as u32, tc, d.eps)?) } else { None },
            bias_o: if d.plain { Some(batch::bias_add(dev, &Op::bias_add(), d.n_embd as u32, tc)?) } else { None },
            bias_up: if d.plain { Some(batch::bias_add(dev, &Op::bias_add(), d.n_ff as u32, tc)?) } else { None },
            bias_down: if d.plain { Some(batch::bias_add(dev, &Op::bias_add(), d.n_embd as u32, tc)?) } else { None },
            rms_qb: ops::rmsnorm(dev, &Op::rmsnorm(), d.hd as u32, tc * d.n_head as u32, d.eps)?,
            rms_kb: ops::rmsnorm(dev, &Op::rmsnorm(), d.hd as u32, tc * d.n_kv as u32, d.eps)?,
            bias_q: batch::bias_add(dev, &Op::bias_add(), qd as u32, tc)?,
            bias_k: batch::bias_add(dev, &Op::bias_add(), kv_dim as u32, tc)?,
            bias_v: batch::bias_add(dev, &Op::bias_add(), kv_dim as u32, tc)?,
            rope_q: batch::rope_batch(dev, &Op::rope_batch(), d.n_head as u32, d.hd as u32, d.n_rot as u32, d.rope_norm, tc)?,
            rope_k: batch::rope_batch(dev, &Op::rope_batch(), d.n_kv as u32, d.hd as u32, d.n_rot as u32, d.rope_norm, tc)?,
            kvw: KvWriteBatch::new(dev, &Op::kv_write_batch(), shape, tc)?,
            attn: AttnBatch::new(dev, &Op::attn_batch(), shape, tc)?,
            attn_local: {
                let mut a = AttnBatch::new(dev, &Op::attn_batch(), shape, tc)?;
                a.window = d.window as u32;
                a
            },
            add_res: ops::add(dev, &add, (t_cap * d.n_embd) as u32)?,
            act: ops::act(dev, &Op::act(if d.gelu { Act::GeluMul } else { Act::SiluMul }), (t_cap * d.n_ff) as u32)?,
            copy: batch::copy_row(dev, &Op::copy_row(), d.n_embd as u32)?,
        };
        dev.check_health()?;
        self.prefill = Some(pf);
        Ok(())
    }

    /// Tokens per batch of the prefill (0 when `enable_prefill` was not called).
    pub fn prefill_capacity(&self) -> usize {
        self.prefill.as_ref().map_or(0, |p| p.t_cap)
    }

    /// One batch of at most `prefill_capacity()` tokens at positions `pos0 ..`; returns the logits of its last token only when
    /// `want_logits`. The engine calls it batch by batch so it can stop between batches when the client leaves.
    pub fn prefill_part(&self, tokens: &[u32], pos0: usize, want_logits: bool) -> Result<Option<Vec<f32>>, String> {
        let pf = self.prefill.as_ref().ok_or("prefill is not enabled (call enable_prefill)")?;
        if tokens.is_empty() || tokens.len() > pf.t_cap {
            return Err(format!("prefill_part: {} tokens do not fit a batch of {}", tokens.len(), pf.t_cap));
        }
        if pos0 + tokens.len() > self.ctx {
            return Err(format!("prefill_part: positions {pos0}..{} are outside the context of {} positions", pos0 + tokens.len(), self.ctx));
        }
        self.prefill_chunk(pf, tokens, pos0, want_logits)
    }

    /// The tokens `tokens` at positions `pos0 ..`, in batches of the prefill capacity; returns the logits of the LAST token. Equivalent to
    /// calling `forward` for each token (the KV cache ends up the same) but each layer's matrix products run as batched GEMMs.
    pub fn prefill(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, String> {
        let pf = self.prefill.as_ref().ok_or("prefill is not enabled (call enable_prefill)")?;
        if tokens.is_empty() {
            return Err("prefill: no tokens".into());
        }
        if pos0 + tokens.len() > self.ctx {
            return Err(format!("prefill: positions {pos0}..{} are outside the context of {} positions", pos0 + tokens.len(), self.ctx));
        }
        let n_chunks = tokens.len().div_ceil(pf.t_cap);
        let mut logits = Vec::new();
        for (ci, chunk) in tokens.chunks(pf.t_cap).enumerate() {
            let last = ci + 1 == n_chunks;
            if let Some(l) = self.prefill_chunk(pf, chunk, pos0 + ci * pf.t_cap, last)? {
                logits = l;
            }
        }
        Ok(logits)
    }

    fn gemm_mv<'a>(&'a self, pf: &'a Prefill, st: &mut Vec<Step<'a>>, layer: usize, which: usize, m: &'a GpuMatrix, x: &'a GpuBuffer, y: &'a GpuBuffer) -> Result<(), String> {
        let (_, _, p) = pf.gemm.iter().find(|(d, _, _)| *d == m.dtype).ok_or_else(|| format!("no GEMM pipeline for {}", pack::name(m.dtype)))?;
        st.extend(gemm::gemm_steps(&self.dev, p, m, &pf.gparams[layer][which], x, y, &pf.tp, pf.t_cap as u32)?);
        Ok(())
    }

    fn pnorm_step<'a>(&'a self, pf: &'a Prefill, st: &mut Vec<Step<'a>>, x: &'a GpuBuffer, w: &'a GpuBuffer, b: Option<&'a GpuBuffer>, out: &'a GpuBuffer) {
        match (b, &pf.ln) {
            (Some(b), Some(ln)) => st.push(ln.step(&[x, w, b, out])),
            _ => st.push(pf.rms.step(&[x, w, out])),
        }
    }

    fn prefill_chunk(&self, pf: &Prefill, chunk: &[u32], p0: usize, want_logits: bool) -> Result<Option<Vec<f32>>, String> {
        let d = self.dims;
        let dev = &self.dev;
        let t = chunk.len();
        if let Some(bad) = chunk.iter().find(|&&id| id as usize >= self.n_vocab) {
            return Err(format!("token {bad} is outside the vocabulary of {}", self.n_vocab));
        }
        let ids: Vec<u8> = chunk.iter().flat_map(|i| i.to_le_bytes()).collect();
        dev.upload_bytes(&pf.ids, 0, &ids)?;
        let table: Vec<f32> = (0..t).flat_map(|i| ops::rope_table((p0 + i) as u32, &self.inv_freq, self.rope_mscale)).collect();
        dev.upload_f32(&pf.cs, 0, &table)?;
        if self.layers.iter().any(|l| l.local) {
            let local: Vec<f32> = (0..t).flat_map(|i| ops::rope_table((p0 + i) as u32, &self.inv_freq_local, self.rope_mscale)).collect();
            dev.upload_f32(&pf.cs_local, 0, &local)?;
        }
        let t32 = t as u32;
        // the parameter words that depend on the batch
        let nb = pack::blocks_per_row(self.embd.dtype, self.embd.cols as usize) as u32;
        for (i, call) in pf.gather.iter().enumerate() {
            let c = &self.embd.chunks[i];
            call.set_params(dev, &[c.start as u32, c.rows as u32, self.embd.cols as u32, nb, t32, d.embed_scale.to_bits()])?;
        }
        pf.rms.set_params(dev, &[d.n_embd as u32, t32, d.eps.to_bits()])?;
        if let Some(ln) = &pf.ln {
            ln.set_params(dev, &[d.n_embd as u32, t32, d.eps.to_bits()])?;
        }
        pf.rms_qb.set_params(dev, &[d.hd as u32, t32 * d.n_head as u32, d.eps.to_bits()])?;
        pf.rms_kb.set_params(dev, &[d.hd as u32, t32 * d.n_kv as u32, d.eps.to_bits()])?;
        for op in [&pf.bias_q, &pf.bias_k, &pf.bias_v, &pf.rope_q, &pf.rope_k].into_iter().chain(pf.bias_o.iter()).chain(pf.bias_up.iter()).chain(pf.bias_down.iter()) {
            op.set_tokens(dev, t32)?;
        }
        pf.kvw.set_batch(dev, p0 as u32, t32)?;
        pf.attn.set_batch(dev, p0 as u32, t32)?;
        if self.layers.iter().any(|l| l.swa) {
            pf.attn_local.set_batch(dev, p0 as u32, t32)?;
        }
        pf.add_res.set_params(dev, &[(t * d.n_embd) as u32])?;
        pf.act.set_params(dev, &[(t * d.n_ff) as u32])?;
        gemm::set_tokens(dev, &pf.tp, t32)?;
        batch::set_copy_row(dev, &pf.copy, d.n_embd as u32, t32 - 1)?;

        let mut st: Vec<Step> = Vec::with_capacity(self.layers.len() * 40 + 8);
        // where each kind of step starts in `st`, for the diagnostic profile (BURITI_GPU_PROFILE=1)
        let mut marks: Vec<(&'static str, usize)> = Vec::new();
        marks.push(("gather", st.len()));
        for (i, call) in pf.gather.iter().enumerate() {
            st.push(call.step(&ops::gather_io(&self.embd, i, &pf.ids, &pf.bx)));
        }
        let mut layer_marks: Vec<usize> = Vec::with_capacity(self.layers.len());
        for (li, l) in self.layers.iter().enumerate() {
            layer_marks.push(st.len());
            marks.push(("rmsnorm", st.len()));
            self.pnorm_step(pf, &mut st, &pf.bx, &l.attn_norm, l.attn_norm_b.as_ref(), &pf.bxn);
            marks.push(("gemm q/k/v", st.len()));
            self.gemm_mv(pf, &mut st, li, 0, &l.wq, &pf.bxn, &pf.bq)?;
            self.gemm_mv(pf, &mut st, li, 1, &l.wk, &pf.bxn, &pf.bkk)?;
            self.gemm_mv(pf, &mut st, li, 2, &l.wv, &pf.bxn, &pf.bvv)?;
            marks.push(("bias+rope+kv", st.len()));
            if let Some(b) = &l.bq {
                st.push(pf.bias_q.call.step(&[&pf.bq, b]));
            }
            if let Some(b) = &l.bk {
                st.push(pf.bias_k.call.step(&[&pf.bkk, b]));
            }
            if let Some(b) = &l.bv {
                st.push(pf.bias_v.call.step(&[&pf.bvv, b]));
            }
            let (qb, kb) = match (&l.q_norm, &l.k_norm) {
                (Some(qw), Some(kw)) => {
                    st.push(pf.rms_qb.step(&[&pf.bq, qw, &pf.bqn]));
                    st.push(pf.rms_kb.step(&[&pf.bkk, kw, &pf.bkn]));
                    (&pf.bqn, &pf.bkn)
                }
                _ => (&pf.bq, &pf.bkk),
            };
            let cs = if l.local { &pf.cs_local } else { &pf.cs };
            let at = if l.swa { &pf.attn_local } else { &pf.attn };
            st.push(pf.rope_q.call.step(&[qb, cs]));
            st.push(pf.rope_k.call.step(&[kb, cs]));
            st.push(pf.kvw.call.step(&[kb, &pf.bvv, &l.cache.k, &l.cache.v]));
            marks.push(("attention", st.len()));
            st.push(at.call.step(&[qb, &l.cache.k, &l.cache.v, &at.scores, &pf.batt]));
            marks.push(("gemm o", st.len()));
            self.gemm_mv(pf, &mut st, li, 3, &l.wo, &pf.batt, &pf.btmp)?;
            if let (Some(b), Some(op)) = (&l.bo, &pf.bias_o) {
                st.push(op.call.step(&[&pf.btmp, b]));
            }
            marks.push(("add+rmsnorm", st.len()));
            let a_out = match &l.post_attn_norm {
                Some(w) => {
                    st.push(pf.rms.step(&[&pf.btmp, w, &pf.btmp2]));
                    &pf.btmp2
                }
                None => &pf.btmp,
            };
            st.push(pf.add_res.step(&[&pf.bx, a_out]));
            self.pnorm_step(pf, &mut st, &pf.bx, &l.ffn_norm, l.ffn_norm_b.as_ref(), &pf.bxn);
            marks.push(("gemm gate/up", st.len()));
            if let Some(g) = &l.w_gate {
                self.gemm_mv(pf, &mut st, li, 4, g, &pf.bxn, &pf.bgate)?;
                self.gemm_mv(pf, &mut st, li, 5, &l.w_up, &pf.bxn, &pf.bup)?;
            } else {
                self.gemm_mv(pf, &mut st, li, 5, &l.w_up, &pf.bxn, &pf.bgate)?;
                if let (Some(b), Some(op)) = (&l.b_up, &pf.bias_up) {
                    st.push(op.call.step(&[&pf.bgate, b]));
                }
            }
            marks.push(("swiglu", st.len()));
            st.push(pf.act.step(&[&pf.bgate, &pf.bup, &pf.bact]));
            marks.push(("gemm down", st.len()));
            self.gemm_mv(pf, &mut st, li, 6, &l.w_down, &pf.bact, &pf.btmp)?;
            if let (Some(b), Some(op)) = (&l.b_down, &pf.bias_down) {
                st.push(op.call.step(&[&pf.btmp, b]));
            }
            marks.push(("add", st.len()));
            let f_out = match &l.post_ffn_norm {
                Some(w) => {
                    st.push(pf.rms.step(&[&pf.btmp, w, &pf.btmp2]));
                    &pf.btmp2
                }
                None => &pf.btmp,
            };
            st.push(pf.add_res.step(&[&pf.bx, f_out]));
        }
        marks.push(("tail", st.len()));
        if want_logits {
            // the last token's row, then the same tail as `forward`
            st.push(pf.copy.step(&[&pf.bx, &self.x]));
            self.norm_step(&mut st, &self.x, &self.out_norm, self.out_norm_b.as_ref(), &self.xn);
            self.mv(&mut st, self.output.as_ref().unwrap_or(&self.embd), &self.xn, &self.logits)?;
        }
        if std::env::var("BURITI_GPU_PROFILE").is_ok() {
            // diagnostic: every step in its own pass, GPU time (timestamp queries) summed per kind; one pass per step adds submission
            // overhead that the GPU time does not include
            let mut sums: Vec<(&'static str, f64, usize)> = Vec::new();
            for (k, (name, from)) in marks.iter().enumerate() {
                let to = marks.get(k + 1).map_or(st.len(), |m| m.1);
                for step in &st[*from..to] {
                    let t = dev.run_steps(std::slice::from_ref(step), 1, DEFAULT_TIMEOUT)?;
                    let s = t.gpu_seconds.unwrap_or(t.wall_seconds);
                    match sums.iter_mut().find(|e| e.0 == *name) {
                        Some(e) => {
                            e.1 += s;
                            e.2 += 1;
                        }
                        None => sums.push((name, s, 1)),
                    }
                }
            }
            let total: f64 = sums.iter().map(|e| e.1).sum();
            println!("prefill profile for a batch of {t} tokens (GPU time, each step run alone): total {:.1} ms", total * 1e3);
            for (name, s, n) in &sums {
                println!("  {name:<14} {:>8.2} ms  {:>5.1}%  ({n} dispatches)", s * 1e3, s / total * 100.0);
            }
        }
        // submit in groups of layers (see `Prefill::group`); every group waits for the previous one, so the order of the steps is kept
        let (mut from, mut i) = (0usize, 0usize);
        while i < self.layers.len() {
            let g = pf.group.load(std::sync::atomic::Ordering::Relaxed).max(1);
            let j = (i + g).min(self.layers.len());
            let to = if j == self.layers.len() { st.len() } else { layer_marks[j] };
            let timing = dev.run_steps(&st[from..to], 1, DEFAULT_TIMEOUT)?;
            let per_layer = timing.wall_seconds / (j - i) as f64;
            pf.group.store(((PASS_BUDGET_S / per_layer.max(1e-6)).floor() as usize).clamp(1, self.layers.len()), std::sync::atomic::Ordering::Relaxed);
            from = to;
            i = j;
        }
        if want_logits {
            Ok(Some(dev.read_back_f32(&self.logits, self.n_vocab, DEFAULT_TIMEOUT)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn selector() -> AdapterSelector {
        std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best)
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
    }

    fn log_softmax(v: &[f32]) -> Vec<f64> {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
        v.iter().map(|&x| x as f64 - m - s.ln()).collect()
    }

    /// GPU against the CPU engine on a real model file (`BURITI_TEST_MODEL`). The meaningful comparison is with `BURITI_FLOAT_ACT=1` (the CPU
    /// engine then keeps f32 activations like the GPU); without it the CPU quantizes activations to int8 and the two differ by that noise
    /// (the D53 floor), so only informative numbers are printed and a looser bound applies.
    #[test]
    #[ignore = "needs a real GPU and BURITI_TEST_MODEL=<model.gguf>; BURITI_TEST_ADAPTER selects the adapter (default: best)"]
    fn gpu_matches_cpu_engine() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("set BURITI_TEST_MODEL to a .gguf file");
        let float_act = std::env::var("BURITI_FLOAT_ACT").is_ok();
        let model = Model::load_with(Path::new(&path), false).expect("the model must load on the CPU engine");
        let t0 = std::time::Instant::now();
        let gm = GpuModel::open(&model, &selector(), 256).expect("the model must load on the GPU");
        println!("adapter {}; GPU load {:.1} s; weights {:.1} MiB, KV cache {:.1} MiB; CPU activations: {}", gm.adapter_name(), t0.elapsed().as_secs_f64(), gm.weight_bytes as f64 / 1048576.0, gm.kv_bytes as f64 / 1048576.0, if float_act { "f32" } else { "int8" });
        let prompt = std::env::var("BURITI_TEST_TEXT").unwrap_or_else(|_| "The capital of France is".into());
        let ids = model.tok.encode_plain(&prompt);
        assert!(!ids.is_empty());

        // CPU
        let mut s = model.new_session(256);
        let mut cpu_logits = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            model.forward(&mut s, t, i, i + 1 == ids.len());
        }
        cpu_logits.extend_from_slice(&s.logits);
        // GPU
        let mut gpu_logits = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            gpu_logits = gm.forward(t, i).unwrap();
        }
        assert_eq!(gpu_logits.len(), cpu_logits.len());
        let (lc, lg) = (log_softmax(&cpu_logits), log_softmax(&gpu_logits));
        let mut order: Vec<usize> = (0..cpu_logits.len()).collect();
        order.sort_by(|&a, &b| cpu_logits[b].partial_cmp(&cpu_logits[a]).unwrap());
        let top10: Vec<usize> = order[..10].to_vec();
        let d_top: Vec<f64> = top10.iter().map(|&i| (lc[i] - lg[i]).abs()).collect();
        let mean_top = d_top.iter().sum::<f64>() / d_top.len() as f64;
        let max_top = d_top.iter().cloned().fold(0.0, f64::max);
        let max_all = lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("prompt {:?} ({} tokens): top-1 CPU {} GPU {}; top-10 logprob diff mean {mean_top:.5} max {max_top:.5}; max over the vocabulary {max_all:.5}", prompt, ids.len(), argmax(&cpu_logits), argmax(&gpu_logits));

        // greedy continuation, both engines, 16 tokens
        let mut cpu_ids = Vec::new();
        let mut next = argmax(&cpu_logits) as u32;
        let mut pos = ids.len();
        for _ in 0..16 {
            cpu_ids.push(next);
            model.forward(&mut s, next, pos, true);
            pos += 1;
            next = argmax(&s.logits) as u32;
        }
        let mut gpu_ids = Vec::new();
        let mut next = argmax(&gpu_logits) as u32;
        let mut pos = ids.len();
        let t1 = std::time::Instant::now();
        let mut gpu_s = 0f64;
        for _ in 0..16 {
            gpu_ids.push(next);
            let (lg, t) = gm.forward_timed(next, pos).unwrap();
            gpu_s += t.gpu_seconds.unwrap_or(t.wall_seconds);
            pos += 1;
            next = argmax(&lg) as u32;
        }
        let wall = t1.elapsed().as_secs_f64();
        let same = cpu_ids.iter().zip(&gpu_ids).take_while(|(a, b)| a == b).count();
        println!("greedy 16 tokens: first {same} identical; CPU {:?}\n                                  GPU {:?}", cpu_ids, gpu_ids);
        println!("GPU decode: {:.1} ms per token pass (GPU time {:.1} ms), {:.1} tokens/s wall clock (includes the logits readback and the host work)", wall / 16.0 * 1e3, gpu_s / 16.0 * 1e3, 16.0 / wall);
        assert_eq!(argmax(&cpu_logits), argmax(&gpu_logits), "the first predicted token differs");
        if float_act {
            assert!(mean_top < 0.01, "mean top-10 logprob difference {mean_top}");
            assert_eq!(same, 16, "the greedy continuation diverges after {same} tokens");
        }
    }

    /// Speed of the own engine on the GPU in the same terms as `llama-bench`: prefill of 512 tokens (`pp512`) and generation of 64 tokens (`tg64`),
    /// one process per call so a script can interleave it with llama.cpp. Prints one `BENCH` line. A warm-up (one batch and four tokens) runs first.
    #[test]
    #[ignore = "benchmark: needs a real GPU and BURITI_TEST_MODEL=<model.gguf>; BURITI_TEST_ADAPTER selects the adapter (default: best)"]
    fn gpu_bench_model() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("set BURITI_TEST_MODEL to a .gguf file");
        let model = Model::load_with(Path::new(&path), false).expect("the model must load on the CPU engine");
        let mut gm = GpuModel::open(&model, &selector(), 1024).expect("the model must load on the GPU");
        gm.enable_prefill(64).expect("prefill must build");
        let mut ids: Vec<u32> = Vec::new();
        while ids.len() < 512 {
            ids.extend(model.tok.encode_plain("The quick brown fox jumps over the lazy dog, and then the cat sat on the warm mat by the window. "));
        }
        ids.truncate(512);
        let _ = gm.prefill(&ids[..64], 0).unwrap();
        for i in 0..4 {
            let _ = gm.forward(ids[i], 64 + i).unwrap();
        }
        let t0 = std::time::Instant::now();
        let first = gm.prefill(&ids, 0).unwrap();
        let pp = ids.len() as f64 / t0.elapsed().as_secs_f64();
        let (mut next, mut pos) = (argmax(&first) as u32, ids.len());
        let t1 = std::time::Instant::now();
        for _ in 0..64 {
            next = argmax(&gm.forward(next, pos).unwrap()) as u32;
            pos += 1;
        }
        let tg = 64.0 / t1.elapsed().as_secs_f64();
        println!("BENCH adapter=\"{}\" pp512={pp:.2} tg64={tg:.2}", gm.adapter_name());
    }

    /// The batched prefill against the token-by-token path of the same GPU model, and against the CPU engine (`BURITI_FLOAT_ACT=1` for the
    /// fair reference), on a prompt of about 100 tokens in batches of 32 (so several batches and a partial one). Also times both paths.
    #[test]
    #[ignore = "needs a real GPU and BURITI_TEST_MODEL=<model.gguf>; BURITI_TEST_ADAPTER selects the adapter (default: best)"]
    fn gpu_prefill_matches_sequential() {
        let path = std::env::var("BURITI_TEST_MODEL").expect("set BURITI_TEST_MODEL to a .gguf file");
        let float_act = std::env::var("BURITI_FLOAT_ACT").is_ok();
        let t_cap: usize = std::env::var("BURITI_TEST_TCAP").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let model = Model::load_with(Path::new(&path), false).expect("the model must load on the CPU engine");
        let ctx: usize = std::env::var("BURITI_TEST_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
        let mut gm = GpuModel::open(&model, &selector(), ctx).expect("the model must load on the GPU");
        let t0 = std::time::Instant::now();
        gm.enable_prefill(t_cap).expect("prefill must build");
        println!("adapter {}; prefill capacity {t_cap}, built in {:.2} s", gm.adapter_name(), t0.elapsed().as_secs_f64());
        let text = std::env::var("BURITI_TEST_TEXT").unwrap_or_else(|_| "The quick brown fox jumps over the lazy dog. ".repeat(11) + "The capital of France is");
        let ids = model.tok.encode_plain(&text);
        let n = ids.len();
        assert!(n > t_cap + 5, "the prompt must span more than one batch ({n} tokens)");

        // sequential GPU path
        let ts = std::time::Instant::now();
        let mut seq = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            seq = gm.forward(t, i).unwrap();
        }
        let seq_s = ts.elapsed().as_secs_f64();
        let greedy = |first: &[f32], gm: &GpuModel, n: usize| -> Vec<u32> {
            let (mut out, mut next, mut pos) = (Vec::new(), argmax(first) as u32, n);
            for _ in 0..8 {
                out.push(next);
                next = argmax(&gm.forward(next, pos).unwrap()) as u32;
                pos += 1;
            }
            out
        };
        let seq_ids = greedy(&seq, &gm, n);

        // batched prefill, twice (the second run is the timed one: the first may include driver warm-up)
        let _ = gm.prefill(&ids, 0).unwrap();
        let tp = std::time::Instant::now();
        let pre = gm.prefill(&ids, 0).unwrap();
        let pre_s = tp.elapsed().as_secs_f64();
        let pre_ids = greedy(&pre, &gm, n);

        let (ls, lp) = (log_softmax(&seq), log_softmax(&pre));
        let max_all = ls.iter().zip(&lp).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let mut order: Vec<usize> = (0..seq.len()).collect();
        order.sort_by(|&a, &b| seq[b].partial_cmp(&seq[a]).unwrap());
        let mean_top = order[..10].iter().map(|&i| (ls[i] - lp[i]).abs()).sum::<f64>() / 10.0;
        println!("prompt {n} tokens: prefill vs sequential GPU: top-1 {} / {}; top-10 logprob diff mean {mean_top:.5}; max over the vocabulary {max_all:.5}", argmax(&seq), argmax(&pre));
        println!("greedy 8 tokens after sequential {:?}\n                      after prefill   {:?}", seq_ids, pre_ids);
        println!("sequential {:.2} s ({:.1} tok/s); batched prefill {:.2} s ({:.1} tok/s); {:.1}x", seq_s, n as f64 / seq_s, pre_s, n as f64 / pre_s, seq_s / pre_s);

        // against the CPU engine
        let mut s = model.new_session(ctx);
        for (i, &t) in ids.iter().enumerate() {
            model.forward(&mut s, t, i, i + 1 == n);
        }
        let lc = log_softmax(&s.logits);
        let mean_cpu = order[..10].iter().map(|&i| (lc[i] - lp[i]).abs()).sum::<f64>() / 10.0;
        println!("prefill vs the CPU engine ({} activations): top-1 CPU {} GPU {}; top-10 logprob diff mean {mean_cpu:.5}", if float_act { "f32" } else { "int8" }, argmax(&s.logits), argmax(&pre));
        assert_eq!(argmax(&seq), argmax(&pre), "the first predicted token differs between prefill and sequential");
        assert!(mean_top < 0.005, "prefill differs from the sequential path by {mean_top} (mean top-10 logprob)");
        assert_eq!(seq_ids, pre_ids, "the greedy continuation differs after prefill");
        if float_act {
            assert!(mean_cpu < 0.01, "prefill differs from the CPU engine by {mean_cpu}");
        }
    }
}
