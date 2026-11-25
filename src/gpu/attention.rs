//! Attention for one token on the GPU (task G7): the f16 KV cache that stays in video memory, its append kernel and the
//! decode attention kernel with grouped-query attention and an optional sliding window.
//!
//! The cache layout is the CPU engine's (`Session::k/v`): one cache per layer, position `t` and KV head `g` start at
//! element `t * kv_dim + g * head_dim`, elements are f16, stored two per `u32` (no shader-f16 feature needed). The
//! attention kernel is deliberately simple (one workgroup per query head, scores in a global scratch row per head):
//! correctness first, the speed work is task G13.
//!
//! Per token a decoder calls, for each layer: `KvWrite::set_position` + a step of `kv_write`, then `Attention::
//! set_position` + a step of `attention`, all inside one `run_steps` submission.

use super::device::{GpuBuffer, GpuDevice};
use super::kernels::grid_2d;
use super::ops::{make_call, Op, OpCall};

/// The attention shape of one layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnShape {
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    /// Capacity of the cache in positions (the context size the session was opened with).
    pub max_ctx: u32,
}

impl AttnShape {
    pub fn kv_dim(&self) -> u32 {
        self.n_kv * self.head_dim
    }
    pub fn group(&self) -> u32 {
        self.n_head / self.n_kv
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.n_head == 0 || self.n_kv == 0 || self.head_dim == 0 || self.max_ctx == 0 {
            return Err("attention: every dimension must be non-zero".into());
        }
        if self.n_head % self.n_kv != 0 {
            return Err(format!("attention: {} query heads are not a multiple of {} KV heads", self.n_head, self.n_kv));
        }
        if self.head_dim % 2 != 0 {
            return Err("attention: head_dim must be even (the cache stores f16 pairs)".into());
        }
        Ok(())
    }
    /// Bytes of one of the two caches (K or V) of a layer.
    pub fn cache_bytes(&self) -> u64 {
        self.max_ctx as u64 * self.kv_dim() as u64 * 2
    }
}

/// The K and V caches of one layer, zero-initialised.
pub struct KvCache {
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub shape: AttnShape,
}

impl KvCache {
    pub fn new(dev: &GpuDevice, shape: AttnShape) -> Result<KvCache, String> {
        shape.validate()?;
        let bytes = shape.cache_bytes();
        Ok(KvCache { k: dev.create_storage(bytes, "kv.k")?, v: dev.create_storage(bytes, "kv.v")?, shape })
    }
}

impl Op {
    pub fn kv_write() -> Op {
        Op::from("op_kv_write", include_str!("shaders/kv_write.wgsl"))
    }
    pub fn attn_decode() -> Op {
        Op::from("op_attn_decode", include_str!("shaders/attn_decode.wgsl"))
    }
}

/// The KV append: buffers (`io`) kk (kv_dim f32), vv (kv_dim f32), cache.k, cache.v.
pub struct KvWrite {
    pub call: OpCall,
    shape: AttnShape,
}

impl KvWrite {
    pub fn new(dev: &GpuDevice, op: &Op, shape: AttnShape) -> Result<KvWrite, String> {
        shape.validate()?;
        let half = (shape.kv_dim() / 2) as u64;
        let groups = grid_2d(half.div_ceil(64), dev.caps.max_compute_workgroups_per_dimension)?;
        Ok(KvWrite { call: make_call(dev, op, &[shape.kv_dim(), 0], groups)?, shape })
    }
    /// Sets the position the next run writes (must be below `max_ctx`).
    pub fn set_position(&self, dev: &GpuDevice, pos: u32) -> Result<(), String> {
        if pos >= self.shape.max_ctx {
            return Err(format!("kv_write: position {pos} is outside the cache of {} positions", self.shape.max_ctx));
        }
        self.call.set_params(dev, &[self.shape.kv_dim(), pos])
    }
}

/// The attention of one token: buffers (`io`) q (n_head * head_dim f32), cache.k, cache.v, scores, out
/// (n_head * head_dim f32). `scores` comes from `Attention::scores_buffer`.
pub struct Attention {
    pub call: OpCall,
    pub shape: AttnShape,
    /// Scratch rows of `max_ctx` floats, one per query head.
    pub scores: GpuBuffer,
}

impl Attention {
    pub fn new(dev: &GpuDevice, op: &Op, shape: AttnShape) -> Result<Attention, String> {
        shape.validate()?;
        let groups = grid_2d(shape.n_head as u64, dev.caps.max_compute_workgroups_per_dimension)?;
        let scores = dev.create_storage(shape.n_head as u64 * shape.max_ctx as u64 * 4, "attn.scores")?;
        let scale = 1.0 / (shape.head_dim as f32).sqrt();
        let call = make_call(dev, op, &Self::words(&shape, 0, None, scale), groups)?;
        Ok(Attention { call, shape, scores })
    }
    fn words(shape: &AttnShape, pos: u32, window: Option<u32>, scale: f32) -> [u32; 8] {
        let t0 = window.map_or(0, |w| (pos + 1).saturating_sub(w));
        [shape.n_head, shape.n_kv, shape.head_dim, shape.kv_dim(), t0, pos + 1 - t0, shape.max_ctx, scale.to_bits()]
    }
    /// The token at `pos` attends to positions `0..=pos`, or to the last `window` positions on a sliding-window layer
    /// (Gemma 3 local layers), exactly as the CPU engine does.
    pub fn set_position(&self, dev: &GpuDevice, pos: u32, window: Option<u32>) -> Result<(), String> {
        if pos >= self.shape.max_ctx {
            return Err(format!("attention: position {pos} is outside the cache of {} positions", self.shape.max_ctx));
        }
        if window == Some(0) {
            return Err("attention: a sliding window of 0 positions".into());
        }
        let scale = 1.0 / (self.shape.head_dim as f32).sqrt();
        self.call.set_params(dev, &Self::words(&self.shape, pos, window, scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::DEFAULT_TIMEOUT;
    use crate::gpu::pack;
    use crate::gpu::select::AdapterSelector;
    use crate::native::f16::f32_to_f16;
    use crate::native::quant::f16_to_f32;
    use crate::native::vmath::attend_group;

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

    fn replace_once(src: &str, from: &str, to: &str) -> String {
        assert_eq!(src.matches(from).count(), 1, "planted-bug marker '{from}' must appear exactly once");
        src.replace(from, to)
    }

    fn pack_halves(h: &[u16]) -> Vec<u8> {
        assert!(h.len() % 2 == 0);
        h.chunks(2).flat_map(|p| (p[0] as u32 | (p[1] as u32) << 16).to_le_bytes()).collect()
    }

    /// Exact (f64) attention for one query head over f16-decoded caches.
    fn exact_head(q: &[f32], kc: &[u16], vc: &[u16], sh: &AttnShape, h: usize, t0: usize, np: usize) -> Vec<f64> {
        let (hd, kv_dim) = (sh.head_dim as usize, sh.kv_dim() as usize);
        let g = h / sh.group() as usize;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut s: Vec<f64> = (0..np)
            .map(|t| (0..hd).map(|d| q[h * hd + d] as f64 * f16_to_f32(kc[(t0 + t) * kv_dim + g * hd + d]) as f64).sum::<f64>() * scale)
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in s.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        (0..hd).map(|d| (0..np).map(|t| s[t] * f16_to_f32(vc[(t0 + t) * kv_dim + g * hd + d]) as f64).sum::<f64>() / sum).collect()
    }

    struct Case {
        sh: AttnShape,
        pos: u32,
        window: Option<u32>,
        /// scale of the random query (large values give scores far above 88, where exp() without the max overflows)
        qs: f32,
    }

    /// Runs the attention kernel on random caches filled with f16 data (uploaded directly, so only the attention math is
    /// under test) and returns the worst error against the exact reference and the CPU engine's `attend_group`.
    fn run_case(dev: &GpuDevice, op: &Op, c: &Case, seed: u64) -> (f64, f64) {
        let sh = c.sh;
        let (hd, kv_dim, max_ctx) = (sh.head_dim as usize, sh.kv_dim() as usize, sh.max_ctx as usize);
        let q: Vec<f32> = pack::gen_x(sh.n_head as usize * hd, seed).into_iter().map(|v| v * c.qs).collect();
        let kc: Vec<u16> = pack::gen_x(max_ctx * kv_dim, seed + 1).into_iter().map(|v| f32_to_f16(v * 2.0)).collect();
        let vc: Vec<u16> = pack::gen_x(max_ctx * kv_dim, seed + 2).into_iter().map(f32_to_f16).collect();
        let cache = KvCache::new(dev, sh).unwrap();
        dev.upload_bytes(&cache.k, 0, &pack_halves(&kc)).unwrap();
        dev.upload_bytes(&cache.v, 0, &pack_halves(&vc)).unwrap();
        let att = Attention::new(dev, op, sh).unwrap();
        att.set_position(dev, c.pos, c.window).unwrap();
        let qb = dev.storage_from_f32(&q, "t.q").unwrap();
        let ob = dev.create_storage((sh.n_head as usize * hd * 4) as u64, "t.out").unwrap();
        att.call.run(dev, &[&qb, &cache.k, &cache.v, &att.scores, &ob], 1).unwrap();
        let got = dev.read_back_f32(&ob, sh.n_head as usize * hd, DEFAULT_TIMEOUT).unwrap();

        let t0 = c.window.map_or(0, |w| (c.pos as usize + 1).saturating_sub(w as usize));
        let np = c.pos as usize + 1 - t0;
        let mut err_exact = 0f64;
        for h in 0..sh.n_head as usize {
            let want = exact_head(&q, &kc, &vc, &sh, h, t0, np);
            for d in 0..hd {
                let g = got[h * hd + d];
                err_exact = err_exact.max(if g.is_nan() { f64::INFINITY } else { (g as f64 - want[d]).abs() });
            }
        }
        // the CPU engine's own routine, group by group, as `Session` calls it
        let group = sh.group() as usize;
        let mut cpu = vec![0f32; sh.n_head as usize * hd];
        let mut scores = vec![0f32; group * (c.pos as usize + 1)];
        for g in 0..sh.n_kv as usize {
            let (a, b) = (g * group * hd, (g + 1) * group * hd);
            attend_group(&q[a..b], &kc[t0 * kv_dim..], &vc[t0 * kv_dim..], c.pos as usize - t0, g, group, hd, kv_dim, 1.0 / (hd as f32).sqrt(), &mut scores, &mut cpu[a..b]);
        }
        let err_cpu = got.iter().zip(&cpu).map(|(a, b)| if a.is_nan() { f64::INFINITY } else { (*a as f64 - *b as f64).abs() }).fold(0.0, f64::max);
        (err_exact, err_cpu)
    }

    // Shapes: Qwen 2.5 0.5B (GQA 7:1), plain MHA with a head_dim of 128, a head_dim above the 128-thread workgroup (Gemma-like 256),
    // and a head_dim that is not a power of two. Positions cover one position, short, more positions than threads, and a window.
    fn cases() -> Vec<Case> {
        let qwen = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 400 };
        let mha = AttnShape { n_head: 8, n_kv: 8, head_dim: 128, max_ctx: 200 };
        let wide = AttnShape { n_head: 4, n_kv: 1, head_dim: 256, max_ctx: 150 };
        let odd = AttnShape { n_head: 6, n_kv: 3, head_dim: 96, max_ctx: 150 };
        vec![
            Case { sh: qwen, pos: 0, window: None, qs: 2.0 },
            Case { sh: qwen, pos: 6, window: None, qs: 2.0 },
            Case { sh: qwen, pos: 130, window: None, qs: 2.0 },
            Case { sh: qwen, pos: 399, window: None, qs: 2.0 },
            Case { sh: qwen, pos: 300, window: Some(64), qs: 2.0 },
            Case { sh: mha, pos: 77, window: None, qs: 2.0 },
            Case { sh: wide, pos: 140, window: None, qs: 2.0 },
            Case { sh: odd, pos: 129, window: Some(100), qs: 2.0 },
            // huge scores: the softmax must subtract the maximum
            Case { sh: qwen, pos: 200, window: None, qs: 2000.0 },
        ]
    }

    // f16 caches are read exactly on both sides, so the error is float summation order and exp(): the measured worst case
    // is far below this; a wrong kernel is orders of magnitude above.
    const TOL: f64 = 2e-5;

    fn check_all(dev: &GpuDevice, op: &Op) -> (f64, f64) {
        let (mut ex, mut cp) = (0f64, 0f64);
        for (i, c) in cases().iter().enumerate() {
            let (e1, e2) = run_case(dev, op, c, 100 + i as u64 * 10);
            println!("  case {i}: heads {}/{} hd {} pos {} window {:?}: exact {e1:.3e}, cpu {e2:.3e}", c.sh.n_head, c.sh.n_kv, c.sh.head_dim, c.pos, c.window);
            ex = ex.max(e1);
            cp = cp.max(e2);
        }
        (ex, cp)
    }

    fn selector() -> AdapterSelector {
        std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best)
    }

    #[test]
    fn shape_validation_is_pure() {
        assert!(AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 8 }.validate().is_ok());
        assert!(AttnShape { n_head: 14, n_kv: 4, head_dim: 64, max_ctx: 8 }.validate().is_err());
        assert!(AttnShape { n_head: 14, n_kv: 2, head_dim: 63, max_ctx: 8 }.validate().is_err());
        assert!(AttnShape { n_head: 0, n_kv: 2, head_dim: 64, max_ctx: 8 }.validate().is_err());
        assert_eq!(AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 4096 }.cache_bytes(), 4096 * 128 * 2);
        // words(): the window start and the number of positions, as `model.rs` computes t0 and `pos - t0`
        let sh = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 400 };
        let w = Attention::words(&sh, 300, Some(64), 0.125);
        assert_eq!((w[4], w[5]), (237, 64));
        let w = Attention::words(&sh, 10, Some(64), 0.125);
        assert_eq!((w[4], w[5]), (0, 11));
        let w = Attention::words(&sh, 10, None, 0.125);
        assert_eq!((w[4], w[5]), (0, 11));
    }

    #[test]
    fn software_adapter_attention() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        let (ex, cp) = check_all(&dev, &Op::attn_decode());
        assert!(ex < TOL && cp < TOL, "attention error vs exact {ex:e}, vs CPU engine {cp:e}");
    }

    #[test]
    #[ignore = "needs a real GPU: BURITI_TEST_ADAPTER=<selector> selects the adapter (default: best)"]
    fn real_adapter_attention() {
        let Some(dev) = open_or_skip(&selector(), "real") else { return };
        let (ex, cp) = check_all(&dev, &Op::attn_decode());
        assert!(ex < TOL && cp < TOL, "attention error vs exact {ex:e}, vs CPU engine {cp:e}");
    }

    /// Planted bugs: each broken kernel must exceed the tolerance on at least one case.
    fn planted_bugs_fail(dev: &GpuDevice) {
        let base = Op::attn_decode();
        let src = base.wgsl.clone();
        let bugs: Vec<(&str, String)> = vec![
            ("gqa_modulo", replace_once(&src, "let g = h / group;", "let g = h % n_kv;")),
            ("no_scale", replace_once(&src, "sc[sb + t] = tot * scale;", "sc[sb + t] = tot;")),
            ("window_ignored", replace_once(&src, "let t0 = p[4];", "let t0 = 0u;")),
            ("one_position_short", replace_once(&src, "let np = p[5];", "let np = p[5] - 1u;")),
            ("no_normalization", replace_once(&src, "out[qb + 2u * pr] = o0 * inv;", "out[qb + 2u * pr] = o0;")),
            ("no_normalization_odd_half", replace_once(&src, "out[qb + 2u * pr + 1u] = o1 * inv;", "out[qb + 2u * pr + 1u] = o1;")),
            ("wrong_k_half", replace_once(&src, "s = s + q[qb + 2u * w] * kv.x + q[qb + 2u * w + 1u] * kv.y;", "s = s + q[qb + 2u * w] * kv.y + q[qb + 2u * w + 1u] * kv.x;")),
            ("wrong_v_half", replace_once(&src, "o0 = o0 + e * v.x;", "o0 = o0 + e * v.y;")),
            ("score_lane_dropped", replace_once(&src, "for (var l = 0u; l < 8u; l = l + 1u) {", "for (var l = 0u; l < 7u; l = l + 1u) {")),
            ("position_groups_skip", replace_once(&src, "for (var j = jg; j < np; j = j + ng) {", "for (var j = jg; j < (np + 1u) / 2u; j = j + ng) {")),
            ("group_partials_dropped", replace_once(&src, "for (var k = 1u; k < ng; k = k + 1u) {", "for (var k = 1u; k < 1u; k = k + 1u) {")),
            ("v_head_offset", replace_once(&src, "vc[(t0 + j) * kvw + kbw + pr]", "vc[(t0 + j) * kvw + pr]")),
            ("max_not_subtracted", replace_once(&src, "let e = exp(sc[sb + t] - mx);", "let e = exp(sc[sb + t]);")),
        ];
        for (tag, wgsl) in bugs {
            let op = base.with_wgsl(tag, wgsl);
            let (ex, cp) = check_all(dev, &op);
            assert!(ex >= TOL || cp >= TOL, "planted bug '{tag}' was not detected (errors {ex:e} / {cp:e})");
            println!("planted bug '{tag}' detected (errors {ex:.2e} / {cp:.2e})");
        }
    }

    #[test]
    fn software_adapter_planted_bugs() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        planted_bugs_fail(&dev);
    }

    #[test]
    #[ignore = "needs a real GPU: BURITI_TEST_ADAPTER=<selector> selects the adapter (default: best)"]
    fn real_adapter_planted_bugs() {
        let Some(dev) = open_or_skip(&selector(), "real") else { return };
        planted_bugs_fail(&dev);
    }

    /// The KV append: the packed halves must be BIT-IDENTICAL to the CPU conversion (`f32_to_f16`: round to nearest even, saturating, NaN as
    /// 0x7E00), including values above 65504, infinities, NaN, subnormal halves and exact halves.
    fn kv_write_matches(dev: &GpuDevice, op: &Op) -> (usize, usize, usize) {
        let sh = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 40 };
        let kv_dim = sh.kv_dim() as usize;
        let cache = KvCache::new(dev, sh).unwrap();
        let w = KvWrite::new(dev, op, sh).unwrap();
        let (mut total, mut differ, mut far) = (0usize, 0usize, 0usize);
        for pos in [0u32, 1, 7, 39] {
            // include a value above 65504, values below the f16 normal range and exact halves
            let mut kk: Vec<f32> = pack::gen_x(kv_dim, 500 + pos as u64).into_iter().map(|v| v * 3.0).collect();
            kk[3] = 1.0e-6;
            kk[4] = 0.5;
            kk[5] = -0.25;
            kk[6] = 1.0e6;
            kk[7] = -1.0e6;
            kk[8] = f32::INFINITY;
            kk[9] = f32::NEG_INFINITY;
            kk[10] = f32::NAN;
            kk[11] = 3.0e-8;
            kk[12] = 6.0e-8;
            kk[13] = 65519.0;
            kk[14] = 65520.0;
            kk[15] = 2049.0;
            kk[16] = 2051.0;
            kk[17] = -0.0;
            let vv: Vec<f32> = pack::gen_x(kv_dim, 600 + pos as u64).into_iter().map(|v| v * 0.01).collect();
            let kb = dev.storage_from_f32(&kk, "t.kk").unwrap();
            let vb = dev.storage_from_f32(&vv, "t.vv").unwrap();
            w.set_position(dev, pos).unwrap();
            w.call.run(dev, &[&kb, &vb, &cache.k, &cache.v], 1).unwrap();
            let bytes = dev.read_back(&cache.k, 0, sh.cache_bytes(), DEFAULT_TIMEOUT).unwrap();
            let vbytes = dev.read_back(&cache.v, 0, sh.cache_bytes(), DEFAULT_TIMEOUT).unwrap();
            for (src, raw) in [(&kk, &bytes), (&vv, &vbytes)] {
                for i in 0..kv_dim {
                    let off = (pos as usize * kv_dim + i) * 2;
                    let got = u16::from_le_bytes([raw[off], raw[off + 1]]);
                    let want = f32_to_f16(src[i]);
                    total += 1;
                    if got != want {
                        differ += 1;
                        if (got as i32 - want as i32).abs() > 1 {
                            far += 1;
                        }
                    }
                }
            }
        }
        // positions that were never written stay zero
        let bytes = dev.read_back(&cache.k, 0, sh.cache_bytes(), DEFAULT_TIMEOUT).unwrap();
        assert!(bytes[(2 * kv_dim) * 2..(3 * kv_dim) * 2].iter().all(|&b| b == 0), "an unwritten position must stay zero");
        (total, differ, far)
    }

    fn kv_write_check(dev: &GpuDevice) {
        let (total, differ, far) = kv_write_matches(dev, &Op::kv_write());
        println!("kv_write: {total} halves, {differ} differ from the CPU conversion");
        assert_eq!((differ, far), (0, 0), "the GPU conversion must be bit-identical to the CPU one");
        // planted bugs
        let base = Op::kv_write();
        let src = base.wgsl.clone();
        for (tag, wgsl) in [
            ("wrong_position", replace_once(&src, "let dst = p[1] * half_dim + i;", "let dst = p[1] * half_dim + i + 1u;")),
            ("halves_swapped", replace_once(&src, "kc[dst] = f32_to_f16(kk[2u * i]) | (f32_to_f16(kk[2u * i + 1u]) << 16u);", "kc[dst] = f32_to_f16(kk[2u * i + 1u]) | (f32_to_f16(kk[2u * i]) << 16u);")),
            ("v_not_written", replace_once(&src, "vc[dst] = f32_to_f16(vv[2u * i]) | (f32_to_f16(vv[2u * i + 1u]) << 16u);", "")),
            ("truncates_instead_of_rounding", replace_once(&src, "if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) { h = h + 1u; }", "")),
            ("ties_always_round_up", replace_once(&src, "if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) { h = h + 1u; }", "if (rem >= 0x1000u) { h = h + 1u; }")),
            ("subnormals_flushed", replace_once(&src, "if (e < -10) { return sign; }", "if (e < 1) { return sign; }")),
            ("nan_as_infinity", replace_once(&src, "return sign_in | 0x7E00u;", "return sign_in | 0x7BFFu;")),
        ] {
            let (t, d, f) = kv_write_matches_catching(dev, &base.with_wgsl(tag, wgsl));
            assert!(d > 0 || f > 0, "planted bug '{tag}' was not detected (of {t} halves)");
            println!("planted kv_write bug '{tag}' detected ({d} of {t} halves differ)");
        }
    }

    fn kv_write_matches_catching(dev: &GpuDevice, op: &Op) -> (usize, usize, usize) {
        // a planted bug may also break the "unwritten stays zero" check: treat that panic as detection
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kv_write_matches(dev, op))) {
            Ok(r) => r,
            Err(_) => (1, 1, 1),
        }
    }

    #[test]
    fn software_adapter_kv_write() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        kv_write_check(&dev);
    }

    #[test]
    #[ignore = "needs a real GPU: BURITI_TEST_ADAPTER=<selector> selects the adapter (default: best)"]
    fn real_adapter_kv_write() {
        let Some(dev) = open_or_skip(&selector(), "real") else { return };
        kv_write_check(&dev);
    }

    /// End to end: kv_write token by token, then the attention, against the exact reference computed from the f16 values the CPU
    /// conversion produces (the GPU cache is bit-identical to them).
    fn decode_loop(dev: &GpuDevice) -> f64 {
        let sh = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 64 };
        let (hd, kv_dim) = (64usize, sh.kv_dim() as usize);
        let cache = KvCache::new(dev, sh).unwrap();
        let (wop, aop) = (Op::kv_write(), Op::attn_decode());
        let kw = KvWrite::new(dev, &wop, sh).unwrap();
        let att = Attention::new(dev, &aop, sh).unwrap();
        let (mut kc, mut vc) = (vec![0u16; 64 * kv_dim], vec![0u16; 64 * kv_dim]);
        let mut worst = 0f64;
        for pos in 0..40usize {
            let kk = pack::gen_x(kv_dim, 900 + pos as u64);
            let vv = pack::gen_x(kv_dim, 1900 + pos as u64);
            for i in 0..kv_dim {
                kc[pos * kv_dim + i] = f32_to_f16(kk[i]);
                vc[pos * kv_dim + i] = f32_to_f16(vv[i]);
            }
            let q: Vec<f32> = pack::gen_x(sh.n_head as usize * hd, 2900 + pos as u64);
            let (kb, vb, qb) = (dev.storage_from_f32(&kk, "t.kk").unwrap(), dev.storage_from_f32(&vv, "t.vv").unwrap(), dev.storage_from_f32(&q, "t.q").unwrap());
            let ob = dev.create_storage((sh.n_head as usize * hd * 4) as u64, "t.out").unwrap();
            kw.set_position(dev, pos as u32).unwrap();
            att.set_position(dev, pos as u32, None).unwrap();
            // one submission: append, then attend (the order of steps is the order of execution)
            dev.run_steps(&[kw.call.step(&[&kb, &vb, &cache.k, &cache.v]), att.call.step(&[&qb, &cache.k, &cache.v, &att.scores, &ob])], 1, DEFAULT_TIMEOUT).unwrap();
            let got = dev.read_back_f32(&ob, sh.n_head as usize * hd, DEFAULT_TIMEOUT).unwrap();
            for h in 0..sh.n_head as usize {
                let want = exact_head(&q, &kc, &vc, &sh, h, 0, pos + 1);
                for d in 0..hd {
                    worst = worst.max((got[h * hd + d] as f64 - want[d]).abs());
                }
            }
        }
        worst
    }

    /// Informative timing (not a pass/fail gate): the decode attention of a Qwen 2.5 0.5B layer (14 heads, 2 KV heads, head_dim 64)
    /// at several positions. Median of 3 runs of 20 calls inside one pass; GPU time from timestamp queries when the adapter has them,
    /// wall clock otherwise (the wall clock includes the submission overhead, about 0.13 to 0.2 ms per call on the 940MX, see D65).
    #[test]
    #[ignore = "informative timing on a real GPU: BURITI_TEST_ADAPTER=<selector> (default: best)"]
    fn attention_timing() {
        let Some(dev) = open_or_skip(&selector(), "timing") else { return };
        let sh = AttnShape { n_head: 14, n_kv: 2, head_dim: 64, max_ctx: 4096 };
        let kv_dim = sh.kv_dim() as usize;
        let cache = KvCache::new(&dev, sh).unwrap();
        let h: Vec<u16> = pack::gen_x(sh.max_ctx as usize * kv_dim, 7).into_iter().map(f32_to_f16).collect();
        dev.upload_bytes(&cache.k, 0, &pack_halves(&h)).unwrap();
        dev.upload_bytes(&cache.v, 0, &pack_halves(&h)).unwrap();
        let att = Attention::new(&dev, &Op::attn_decode(), sh).unwrap();
        let q = dev.storage_from_f32(&pack::gen_x(14 * 64, 8), "t.q").unwrap();
        let out = dev.create_storage(14 * 64 * 4, "t.out").unwrap();
        println!("attention timing on {} ({} f16 KV, one layer, per call):", dev.name(), kv_dim);
        for pos in [0u32, 127, 511, 2047, 4095] {
            att.set_position(&dev, pos, None).unwrap();
            let mut us: Vec<f64> = (0..3)
                .map(|_| {
                    let t = att.call.run(&dev, &[&q, &cache.k, &cache.v, &att.scores, &out], 20).unwrap();
                    t.gpu_seconds.unwrap_or(t.wall_seconds) / 20.0 * 1e6
                })
                .collect();
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // bytes the kernel reads per call: every query head reads K and V of its group (no sharing across the 7 heads of a group)
            let bytes = 14.0 * (pos as f64 + 1.0) * 64.0 * 2.0 * 2.0;
            println!("  pos {pos:>5}: {:>8.1} us (min {:.1}, max {:.1}); reads {:.2} MB, {:.1} GB/s effective", us[1], us[0], us[2], bytes / 1e6, bytes / (us[1] * 1e-6) / 1e9);
        }
    }

    #[test]
    fn software_adapter_decode_loop() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        let worst = decode_loop(&dev);
        println!("decode loop worst error {worst:.3e}");
        assert!(worst < 1e-3, "decode loop error {worst:e}");
    }

    #[test]
    #[ignore = "needs a real GPU: BURITI_TEST_ADAPTER=<selector> selects the adapter (default: best)"]
    fn real_adapter_decode_loop() {
        let Some(dev) = open_or_skip(&selector(), "real") else { return };
        let worst = decode_loop(&dev);
        println!("decode loop worst error {worst:.3e}");
        assert!(worst < 1e-3, "decode loop error {worst:e}");
    }
}
