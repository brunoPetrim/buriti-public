//! Batched versions of the per-token operations, for prefill on the GPU (task G9): RoPE with one table per token, bias added to every
//! token, copy of one row, KV append for a run of positions, and causal attention for a run of query tokens.
//!
//! Every kernel's grid is sized for the capacity `t_cap` of the batch; the number of tokens of the current batch is a parameter word
//! rewritten between submissions (`set_tokens` / `set_batch`), and threads beyond it leave at once.

use super::attention::AttnShape;
use super::device::{GpuBuffer, GpuDevice};
use super::ops::{elem_groups, make_call, Op, OpCall};

impl Op {
    pub fn rope_batch() -> Op {
        Op::from("op_rope_batch", include_str!("shaders/rope_batch.wgsl"))
    }
    pub fn bias_add() -> Op {
        Op::from("op_bias_add", include_str!("shaders/bias_add.wgsl"))
    }
    pub fn copy_row() -> Op {
        Op::from("op_copy_row", include_str!("shaders/copy_row.wgsl"))
    }
    pub fn kv_write_batch() -> Op {
        Op::from("op_kv_write_batch", include_str!("shaders/kv_write_batch.wgsl"))
    }
    pub fn attn_batch() -> Op {
        Op::from("op_attn_batch", include_str!("shaders/attn_batch.wgsl"))
    }
}

/// An op whose parameter words are fixed except for the token count of the batch.
pub struct TokenOp {
    pub call: OpCall,
    words: Vec<u32>,
    slot: usize,
}

impl TokenOp {
    pub fn set_tokens(&self, dev: &GpuDevice, t: u32) -> Result<(), String> {
        let mut w = self.words.clone();
        w[self.slot] = t;
        self.call.set_params(dev, &w)
    }
}

/// RoPE in place on `v` = [token][n_heads][head_dim]; buffers (`io`): v, cs (`t_cap * n_rot` floats: the `ops::rope_table` of every token, concatenated).
pub fn rope_batch(dev: &GpuDevice, op: &Op, n_heads: u32, head_dim: u32, n_rot: u32, interleaved: bool, t_cap: u32) -> Result<TokenOp, String> {
    if n_heads == 0 || t_cap == 0 || n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim {
        return Err(format!("rope_batch: n_rot {n_rot} must be even and within head_dim {head_dim}"));
    }
    let half = n_rot / 2;
    let words = vec![n_heads, head_dim, half, interleaved as u32, 0];
    let call = make_call(dev, op, &words, elem_groups(dev, t_cap as u64 * n_heads as u64 * half as u64, 64)?)?;
    Ok(TokenOp { call, words, slot: 4 })
}

/// x[t * n + i] += b[i]; buffers (`io`): x (t_cap * n), b (n).
pub fn bias_add(dev: &GpuDevice, op: &Op, n: u32, t_cap: u32) -> Result<TokenOp, String> {
    if n == 0 || t_cap == 0 {
        return Err("bias_add: empty".into());
    }
    let words = vec![n, 0];
    let call = make_call(dev, op, &words, elem_groups(dev, n as u64 * t_cap as u64, 256)?)?;
    Ok(TokenOp { call, words, slot: 1 })
}

/// dst = row `row` of `src` (rows of `n` floats); buffers (`io`): src, dst.
pub fn copy_row(dev: &GpuDevice, op: &Op, n: u32) -> Result<OpCall, String> {
    make_call(dev, op, &[n, 0], elem_groups(dev, n as u64, 256)?)
}

pub fn set_copy_row(dev: &GpuDevice, call: &OpCall, n: u32, row: u32) -> Result<(), String> {
    call.set_params(dev, &[n, row])
}

/// The KV append of a batch: buffers (`io`) kk (t_cap * kv_dim), vv (t_cap * kv_dim), cache.k, cache.v.
pub struct KvWriteBatch {
    pub call: OpCall,
    shape: AttnShape,
}

impl KvWriteBatch {
    pub fn new(dev: &GpuDevice, op: &Op, shape: AttnShape, t_cap: u32) -> Result<KvWriteBatch, String> {
        shape.validate()?;
        let half = (shape.kv_dim() / 2) as u64;
        Ok(KvWriteBatch { call: make_call(dev, op, &[shape.kv_dim(), 0, 0], elem_groups(dev, t_cap as u64 * half, 64)?)?, shape })
    }
    /// Tokens `pos0 .. pos0 + t` go to those positions of the cache.
    pub fn set_batch(&self, dev: &GpuDevice, pos0: u32, t: u32) -> Result<(), String> {
        if t == 0 || pos0 as u64 + t as u64 > self.shape.max_ctx as u64 {
            return Err(format!("kv_write_batch: positions {pos0}..{} are outside the cache of {} positions", pos0 as u64 + t as u64, self.shape.max_ctx));
        }
        self.call.set_params(dev, &[self.shape.kv_dim(), pos0, t])
    }
}

/// Causal attention for a batch: buffers (`io`) q (t_cap * n_head * head_dim), cache.k, cache.v, scores, out (same size as q).
pub struct AttnBatch {
    pub call: OpCall,
    pub shape: AttnShape,
    /// Sliding window in positions (0 = none); set before `set_batch` (Gemma 3 local layers).
    pub window: u32,
    /// One scratch row of `max_ctx` floats per (token, head).
    pub scores: GpuBuffer,
}

impl AttnBatch {
    pub fn new(dev: &GpuDevice, op: &Op, shape: AttnShape, t_cap: u32) -> Result<AttnBatch, String> {
        shape.validate()?;
        let max = dev.caps.max_compute_workgroups_per_dimension as u64;
        if shape.n_head as u64 > max || t_cap as u64 > max {
            return Err("attn_batch: the grid does not fit this adapter".into());
        }
        let scores = dev.create_storage(t_cap as u64 * shape.n_head as u64 * shape.max_ctx as u64 * 4, "attn_batch.scores")?;
        let call = make_call(dev, op, &Self::words(&shape, 0, 0, 0), [shape.n_head, t_cap, 1])?;
        Ok(AttnBatch { call, shape, window: 0, scores })
    }
    fn words(shape: &AttnShape, pos0: u32, t: u32, window: u32) -> [u32; 9] {
        let scale = 1.0 / (shape.head_dim as f32).sqrt();
        [shape.n_head, shape.n_kv, shape.head_dim, shape.kv_dim(), pos0, t, shape.max_ctx, scale.to_bits(), window]
    }
    /// The batch's token `i` is at position `pos0 + i` and sees positions `0..=pos0 + i`.
    pub fn set_batch(&self, dev: &GpuDevice, pos0: u32, t: u32) -> Result<(), String> {
        if t == 0 || pos0 as u64 + t as u64 > self.shape.max_ctx as u64 {
            return Err(format!("attn_batch: positions {pos0}..{} are outside the cache of {} positions", pos0 as u64 + t as u64, self.shape.max_ctx));
        }
        self.call.set_params(dev, &Self::words(&self.shape, pos0, t, self.window))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::attention::KvCache;
    use crate::gpu::device::DEFAULT_TIMEOUT;
    use crate::gpu::ops::rope_table;
    use crate::gpu::pack;
    use crate::gpu::select::AdapterSelector;
    use crate::native::f16::f32_to_f16;
    use crate::native::model::rope as cpu_rope;
    use crate::native::quant::f16_to_f32;

    fn open_or_skip(sel: &AdapterSelector, what: &str) -> Option<GpuDevice> {
        match GpuDevice::open(sel) {
            Ok(d) => {
                println!("[{what}] adapter: {} ({}, {})", d.name(), d.report.backend, d.report.device_type);
                Some(d)
            }
            Err(e) => {
                println!("SKIP [{what}]: no GPU device could be created: {e}");
                None
            }
        }
    }

    fn selector() -> AdapterSelector {
        std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best)
    }

    fn replace_once(src: &str, from: &str, to: &str) -> String {
        assert_eq!(src.matches(from).count(), 1, "planted-bug marker '{from}' must appear exactly once");
        src.replace(from, to)
    }

    fn pack_halves(h: &[u16]) -> Vec<u8> {
        h.chunks(2).flat_map(|p| (p[0] as u32 | (p[1] as u32) << 16).to_le_bytes()).collect()
    }

    // ---------------------------------------------------------------- RoPE
    const ROPE_TOL: (f64, f64) = (1e-6, 4e-6);

    /// Worst |gpu - cpu| / (atol + rtol * |cpu|) over a batch of tokens through the CPU engine's `rope` (1 or less passes).
    fn rope_error(dev: &GpuDevice, op: &Op, hd: usize, n_rot: usize, interleaved: bool) -> f64 {
        let (n_heads, t, t_cap, pos0) = (14usize, 37usize, 64usize, 5usize);
        let v0: Vec<f32> = pack::gen_x(t_cap * n_heads * hd, 11);
        let inv: Vec<f32> = crate::gpu::ops::rope_inv_freq(1_000_000.0, n_rot, None);
        let cs: Vec<f32> = (0..t_cap).flat_map(|i| rope_table((pos0 + i) as u32, &inv, 1.0)).collect();
        let r = rope_batch(dev, op, n_heads as u32, hd as u32, n_rot as u32, interleaved, t_cap as u32).unwrap();
        r.set_tokens(dev, t as u32).unwrap();
        let (vb, cb) = (dev.storage_from_f32(&v0, "t.v").unwrap(), dev.storage_from_f32(&cs, "t.cs").unwrap());
        r.call.run(dev, &[&vb, &cb], 1).unwrap();
        let got = dev.read_back_f32(&vb, v0.len(), DEFAULT_TIMEOUT).unwrap();
        let mut worst = 0f64;
        for tok in 0..t_cap {
            let mut want = v0[tok * n_heads * hd..(tok + 1) * n_heads * hd].to_vec();
            if tok < t {
                for h in 0..n_heads {
                    cpu_rope(&mut want[h * hd..(h + 1) * hd], pos0 + tok, &inv, interleaved, 1.0);
                }
            }
            for (i, w) in want.iter().enumerate() {
                let g = got[tok * n_heads * hd + i];
                let e = (g as f64 - *w as f64).abs() / (ROPE_TOL.0 + ROPE_TOL.1 * (*w as f64).abs());
                worst = worst.max(if g.is_nan() { f64::INFINITY } else { e });
            }
        }
        worst
    }

    fn rope_check(dev: &GpuDevice) {
        let op = Op::rope_batch();
        for (hd, n_rot, inter) in [(64usize, 64usize, false), (64, 32, false), (64, 64, true)] {
            let e = rope_error(dev, &op, hd, n_rot, inter);
            println!("  rope_batch head_dim {hd} n_rot {n_rot} interleaved {inter}: worst error / tolerance {e:.3}");
            assert!(e <= 1.0, "rope_batch error {e}");
        }
        let src = op.wgsl.clone();
        for (tag, wgsl) in [
            ("one_table_for_all_tokens", replace_once(&src, "let sn = cs[t * 2u * hp + 2u * i];", "let sn = cs[2u * i];")),
            ("tokens_beyond_the_count_rotated", replace_once(&src, "if (idx >= p[4] * per_tok) { return; }", "if (idx >= 64u * per_tok) { return; }")),
            ("head_stride_wrong", replace_once(&src, "let base = (t * p[0] + h) * p[1];", "let base = (t * p[0] + h) * p[1] + t;")),
        ] {
            let bad = op.with_wgsl(tag, wgsl);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rope_error(dev, &bad, 64, 64, false) > 1.0)).unwrap_or(true);
            assert!(caught, "planted bug '{tag}' was not detected");
            println!("planted rope_batch bug '{tag}' detected");
        }
    }

    // ---------------------------------------------------------------- bias, copy row
    fn bias_copy_check(dev: &GpuDevice) {
        let (n, t, t_cap) = (896usize, 37usize, 64usize);
        let x0 = pack::gen_x(t_cap * n, 21);
        let b = pack::gen_x(n, 22);
        let bo = bias_add(dev, &Op::bias_add(), n as u32, t_cap as u32).unwrap();
        bo.set_tokens(dev, t as u32).unwrap();
        let (xb, bb) = (dev.storage_from_f32(&x0, "t.x").unwrap(), dev.storage_from_f32(&b, "t.b").unwrap());
        bo.call.run(dev, &[&xb, &bb], 1).unwrap();
        let got = dev.read_back_f32(&xb, x0.len(), DEFAULT_TIMEOUT).unwrap();
        for i in 0..t_cap * n {
            let want = if i < t * n { x0[i] + b[i % n] } else { x0[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "bias_add differs at {i}");
        }
        // a broken variant: the bias index ignores the token (b[i] instead of b[i % n]) must be caught by the same comparison
        let bad = Op::bias_add().with_wgsl("no_modulo", replace_once(&Op::bias_add().wgsl, "b[i % p[0]]", "b[i % 7u]"));
        let bo2 = bias_add(dev, &bad, n as u32, t_cap as u32).unwrap();
        bo2.set_tokens(dev, t as u32).unwrap();
        let xb2 = dev.storage_from_f32(&x0, "t.x2").unwrap();
        bo2.call.run(dev, &[&xb2, &bb], 1).unwrap();
        let got2 = dev.read_back_f32(&xb2, x0.len(), DEFAULT_TIMEOUT).unwrap();
        assert!((0..t * n).any(|i| got2[i].to_bits() != (x0[i] + b[i % n]).to_bits()), "planted bias_add bug was not detected");
        // copy_row
        let cr = copy_row(dev, &Op::copy_row(), n as u32).unwrap();
        set_copy_row(dev, &cr, n as u32, 36).unwrap();
        let db = dev.create_storage((n * 4) as u64, "t.dst").unwrap();
        cr.run(dev, &[&xb2, &db], 1).unwrap();
        let _ = xb2;
        let row = dev.read_back_f32(&db, n, DEFAULT_TIMEOUT).unwrap();
        let cur = dev.read_back_f32(&xb, t_cap * n, DEFAULT_TIMEOUT).unwrap();
        let _ = cur;
        let src_row = dev.read_back_f32(&xb2, t_cap * n, DEFAULT_TIMEOUT).unwrap();
        assert_eq!(row, src_row[36 * n..37 * n].to_vec(), "copy_row copied the wrong row");
    }

    // ---------------------------------------------------------------- KV write batch
    fn kv_batch_check(dev: &GpuDevice) {
        let sh = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 80 };
        let kv_dim = sh.kv_dim() as usize;
        let (t, t_cap, pos0) = (19usize, 32usize, 7usize);
        let mut kk = pack::gen_x(t_cap * kv_dim, 31).into_iter().map(|v| v * 3.0).collect::<Vec<f32>>();
        kk[3] = 1.0e6;
        kk[kv_dim + 5] = f32::NAN;
        kk[2 * kv_dim + 9] = 3.0e-8;
        kk[3 * kv_dim + 4] = 2049.0;
        let vv = pack::gen_x(t_cap * kv_dim, 32);
        let run = |op: &Op| -> Vec<u8> {
            let cache = KvCache::new(dev, sh).unwrap();
            let w = KvWriteBatch::new(dev, op, sh, t_cap as u32).unwrap();
            w.set_batch(dev, pos0 as u32, t as u32).unwrap();
            let (kb, vb) = (dev.storage_from_f32(&kk, "t.kk").unwrap(), dev.storage_from_f32(&vv, "t.vv").unwrap());
            w.call.run(dev, &[&kb, &vb, &cache.k, &cache.v], 1).unwrap();
            let mut all = dev.read_back(&cache.k, 0, sh.cache_bytes(), DEFAULT_TIMEOUT).unwrap();
            all.extend(dev.read_back(&cache.v, 0, sh.cache_bytes(), DEFAULT_TIMEOUT).unwrap());
            all
        };
        let differs = |bytes: &[u8]| -> usize {
            let half = bytes.len() / 2;
            let mut n = 0;
            for (c, (src, off)) in [(&kk, 0usize), (&vv, half)].iter().enumerate() {
                let _ = c;
                for p in 0..sh.max_ctx as usize {
                    for i in 0..kv_dim {
                        let at = off + (p * kv_dim + i) * 2;
                        let got = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                        let want = if p >= pos0 && p < pos0 + t { f32_to_f16(src[(p - pos0) * kv_dim + i]) } else { 0 };
                        if got != want {
                            n += 1;
                        }
                    }
                }
            }
            n
        };
        let good = differs(&run(&Op::kv_write_batch()));
        println!("  kv_write_batch: {good} halves differ from the CPU conversion (positions {pos0}..{} written, the rest zero)", pos0 + t);
        assert_eq!(good, 0, "kv_write_batch must be bit-identical to the CPU conversion and leave other positions alone");
        let src = Op::kv_write_batch().wgsl;
        for (tag, wgsl) in [
            ("positions_not_offset", replace_once(&src, "let dst = (p[1] + t) * half_dim + i;", "let dst = t * half_dim + i;")),
            ("batch_count_ignored", replace_once(&src, "if (idx >= p[2] * half_dim) { return; }", "if (idx >= 32u * half_dim) { return; }")),
            ("token_source_stride_wrong", replace_once(&src, "let src = t * p[0] + 2u * i;", "let src = t * p[0] + i;")),
        ] {
            let bad = Op::kv_write_batch().with_wgsl(tag, wgsl);
            assert!(differs(&run(&bad)) > 0, "planted bug '{tag}' was not detected");
            println!("planted kv_write_batch bug '{tag}' detected");
        }
    }

    // ---------------------------------------------------------------- attention batch
    fn exact(q: &[f32], kc: &[u16], vc: &[u16], sh: &AttnShape, h: usize, np: usize) -> Vec<f64> {
        let (hd, kv_dim) = (sh.head_dim as usize, sh.kv_dim() as usize);
        let g = h / sh.group() as usize;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut s: Vec<f64> = (0..np).map(|t| (0..hd).map(|d| q[h * hd + d] as f64 * f16_to_f32(kc[t * kv_dim + g * hd + d]) as f64).sum::<f64>() * scale).collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in s.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        (0..hd).map(|d| (0..np).map(|t| s[t] * f16_to_f32(vc[t * kv_dim + g * hd + d]) as f64).sum::<f64>() / sum).collect()
    }

    fn attn_error(dev: &GpuDevice, op: &Op, sh: AttnShape, pos0: usize, t: usize, t_cap: usize, qs: f32, window: u32) -> f64 {
        let (hd, kv_dim, n_head) = (sh.head_dim as usize, sh.kv_dim() as usize, sh.n_head as usize);
        let q: Vec<f32> = pack::gen_x(t_cap * n_head * hd, 41).into_iter().map(|v| v * qs).collect();
        let kc: Vec<u16> = pack::gen_x(sh.max_ctx as usize * kv_dim, 42).into_iter().map(|v| f32_to_f16(v * 2.0)).collect();
        let vc: Vec<u16> = pack::gen_x(sh.max_ctx as usize * kv_dim, 43).into_iter().map(f32_to_f16).collect();
        let cache = KvCache::new(dev, sh).unwrap();
        dev.upload_bytes(&cache.k, 0, &pack_halves(&kc)).unwrap();
        dev.upload_bytes(&cache.v, 0, &pack_halves(&vc)).unwrap();
        let mut ab = AttnBatch::new(dev, op, sh, t_cap as u32).unwrap();
        ab.window = window;
        ab.set_batch(dev, pos0 as u32, t as u32).unwrap();
        let qb = dev.storage_from_f32(&q, "t.q").unwrap();
        let ob = dev.create_storage((t_cap * n_head * hd * 4) as u64, "t.out").unwrap();
        ab.call.run(dev, &[&qb, &cache.k, &cache.v, &ab.scores, &ob], 1).unwrap();
        let got = dev.read_back_f32(&ob, t_cap * n_head * hd, DEFAULT_TIMEOUT).unwrap();
        let mut worst = 0f64;
        for tok in 0..t {
            for h in 0..n_head {
                let pos = pos0 + tok;
                let t0 = if window > 0 { (pos + 1).saturating_sub(window as usize) } else { 0 };
                let want = exact(&q[tok * n_head * hd..(tok + 1) * n_head * hd], &kc[t0 * kv_dim..], &vc[t0 * kv_dim..], &sh, h, pos + 1 - t0);
                for d in 0..hd {
                    let g = got[(tok * n_head + h) * hd + d];
                    worst = worst.max(if g.is_nan() { f64::INFINITY } else { (g as f64 - want[d]).abs() });
                }
            }
        }
        // tokens beyond the count must stay untouched (zero)
        assert!(got[t * n_head * hd..].iter().all(|&v| v == 0.0), "a token beyond the batch was written");
        worst
    }

    const ATTN_TOL: f64 = 2e-5;
    /// With huge scores (query scale 2000, scores around 1000) the f32 dot product has an absolute error near 1e-4 and the softmax is
    /// nearly one-hot, so the output carries that error: the tolerance follows the magnitude of the scores.
    fn attn_tol(qs: f32) -> f64 {
        if qs > 100.0 {
            5e-4
        } else {
            ATTN_TOL
        }
    }

    fn attn_check(dev: &GpuDevice) {
        let op = Op::attn_batch();
        let qwen = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 200 };
        let wide = AttnShape { n_head: 4, n_kv: 1, head_dim: 256, max_ctx: 150 };
        // (shape, first position, tokens, capacity, query scale)
        for (i, (sh, pos0, t, cap, qs, win)) in [(qwen, 0usize, 1usize, 8usize, 2.0f32, 0u32), (qwen, 0, 37, 64, 2.0, 0), (qwen, 11, 19, 32, 2.0, 0), (qwen, 90, 50, 64, 2.0, 0), (wide, 40, 33, 64, 2.0, 0), (qwen, 5, 20, 32, 2000.0, 0), (qwen, 30, 20, 32, 2.0, 24), (qwen, 0, 40, 64, 2.0, 16)].into_iter().enumerate() {
            let e = attn_error(dev, &op, sh, pos0, t, cap, qs, win);
            println!("  attn_batch case {i}: heads {}/{} hd {} first position {pos0}, {t} tokens, window {win}: worst error {e:.3e}", sh.n_head, sh.n_kv, sh.head_dim);
            assert!(e < attn_tol(qs), "attn_batch error {e:e}");
        }
        let src = op.wgsl.clone();
        for (tag, wgsl) in [
            ("not_causal", replace_once(&src, "let np = pos + 1u - t0;", "let np = p[4] + p[5] - t0;")),
            ("first_position_ignored", replace_once(&src, "let pos = p[4] + tok;", "let pos = tok;")),
            ("window_ignored", replace_once(&src, "if (p[8] > 0u && pos + 1u > p[8]) { t0 = pos + 1u - p[8]; }", "")),
            ("window_off_by_one", replace_once(&src, "t0 = pos + 1u - p[8];", "t0 = pos - p[8];")),
            ("token_row_not_offset", replace_once(&src, "let row = tok * n_head + h;", "let row = h;")),
            ("gqa_modulo", replace_once(&src, "let g = h / group;", "let g = h % n_kv;")),
            ("max_not_subtracted", replace_once(&src, "let e = exp(sc[sb + t] - mx);", "let e = exp(sc[sb + t]);")),
        ] {
            let bad = op.with_wgsl(tag, wgsl);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                [(qwen, 11usize, 19usize, 32usize, 2.0f32, 0u32), (qwen, 5, 20, 32, 2000.0, 0), (qwen, 30, 20, 32, 2.0, 24)].iter().any(|&(sh, p0, t, cap, qs, win)| attn_error(dev, &bad, sh, p0, t, cap, qs, win) >= attn_tol(qs))
            }))
            .unwrap_or(true);
            assert!(caught, "planted bug '{tag}' was not detected");
            println!("planted attn_batch bug '{tag}' detected");
        }
    }

    #[test]
    fn software_adapter_batch_ops() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        bias_copy_check(&dev);
        kv_batch_check(&dev);
    }

    #[test]
    fn software_adapter_rope_batch() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        // only the first configuration and no planted bugs on the (slow) software adapter; the real-adapter test runs everything
        let e = rope_error(&dev, &Op::rope_batch(), 64, 64, false);
        assert!(e <= 1.0, "rope_batch error {e}");
    }

    #[test]
    fn software_adapter_attn_batch() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        let sh = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 200 };
        let e = attn_error(&dev, &Op::attn_batch(), sh, 11, 19, 32, 2.0, 0);
        assert!(e < ATTN_TOL, "attn_batch error {e:e}");
        let w = attn_error(&dev, &Op::attn_batch(), sh, 30, 20, 32, 2.0, 24);
        assert!(w < ATTN_TOL, "windowed attn_batch error {w:e}");
    }

    #[test]
    #[ignore = "needs a real GPU: BURITI_TEST_ADAPTER=<selector> selects the adapter (default: best)"]
    fn real_adapter_batch_ops() {
        let Some(dev) = open_or_skip(&selector(), "real") else { return };
        rope_check(&dev);
        bias_copy_check(&dev);
        kv_batch_check(&dev);
        attn_check(&dev);
    }
}
