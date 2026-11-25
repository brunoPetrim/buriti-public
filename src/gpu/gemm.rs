//! Batched matrix product on the GPU for prefill (task G9): `Y[t] = W X[t]` for T tokens at once, for every weight format.
//!
//! One generic kernel (`shaders/gemm_main.wgsl`) serves all ten formats: the weights are decoded by the same per-format decoder
//! sections the embedding gather uses (`gather_decoders.wgsl`) and staged in shared memory once per tile of 32 weight rows x 32
//! tokens, so the decode cost is shared by 32 tokens. Activations stay f32 (D64). Output layout `Y[t * rows + r]`, token-major.
//! The grid is sized for the capacity of the batch (`t_cap` tokens); the number of tokens of the current call is written to a
//! small shared buffer (`token_count_buffer`) so one recorded sequence serves every batch size up to the capacity.

use super::device::{GpuBuffer, GpuDevice, Step};
use super::matvec::GpuMatrix;
use super::pack;
use crate::native::gguf::Dtype;
use std::sync::Arc;

/// Weight rows and tokens per workgroup tile (the same number on both axes).
pub const TILE: u32 = 64;

#[derive(Clone)]
pub struct GemmKernel {
    pub dtype: Dtype,
    pub key: String,
    pub wgsl: String,
}

impl GemmKernel {
    pub fn new(dtype: Dtype) -> Result<GemmKernel, String> {
        if !pack::supported(dtype) {
            return Err(format!("{} has no GPU kernel", pack::name(dtype)));
        }
        let nbuf = pack::array_row_bytes(dtype, pack::col_multiple(dtype)).len();
        let all = include_str!("shaders/gather_decoders.wgsl").replace("\r\n", "\n");
        let marker = format!("//@@ {}\n", pack::name(dtype));
        let start = all.find(&marker).ok_or_else(|| format!("no decoder for {}", pack::name(dtype)))? + marker.len();
        let end = all[start..].find("//@@ ").map_or(all.len(), |e| start + e);
        let main = include_str!("shaders/gemm_main.wgsl")
            .replace("\r\n", "\n")
            .replace("@X@", &nbuf.to_string())
            .replace("@Y@", &(nbuf + 1).to_string())
            .replace("@P@", &(nbuf + 2).to_string())
            .replace("@T@", &(nbuf + 3).to_string());
        // four consecutive weights per call: Q8_0, Q4_K and Q6_K have a specialized decoder, the other formats decode one by one
        let section = &all[start..end];
        let generic = if section.contains("fn weight4(") { "" } else { "fn weight4(lrow: u32, c: u32) -> vec4<f32> {\n    return vec4<f32>(weight(lrow, c), weight(lrow, c + 1u), weight(lrow, c + 2u), weight(lrow, c + 3u));\n}\n" };
        Ok(GemmKernel { dtype, key: format!("gemm_{}", pack::name(dtype)), wgsl: format!("{}\n{}\n{}", section, generic, main) })
    }

    /// A broken variant for the planted-bug tests.
    pub fn with_wgsl(&self, tag: &str, wgsl: String) -> GemmKernel {
        GemmKernel { dtype: self.dtype, key: format!("{}_{tag}", self.key), wgsl }
    }

    pub fn pipeline(&self, dev: &GpuDevice) -> Result<Arc<wgpu::ComputePipeline>, String> {
        dev.pipeline(&self.key, &self.wgsl, "main")
    }
}

/// The per-chunk parameter buffers of one matrix.
pub struct GemmParams {
    chunks: Vec<GpuBuffer>,
}

pub fn gemm_params(dev: &GpuDevice, m: &GpuMatrix) -> Result<GemmParams, String> {
    let nb = pack::blocks_per_row(m.dtype, m.cols as usize) as u32;
    let mut chunks = Vec::new();
    for c in &m.chunks {
        let mut p = Vec::new();
        for v in [c.start as u32, c.rows as u32, m.cols as u32, nb, m.rows as u32, 0, 0, 0] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        chunks.push(dev.storage_from_bytes(&p, "gemm.params")?);
    }
    Ok(GemmParams { chunks })
}

/// The shared token count `[T, 0, 0, 0]`.
pub fn token_count_buffer(dev: &GpuDevice) -> Result<GpuBuffer, String> {
    dev.storage_from_bytes(&[0u8; 16], "gemm.tokens")
}

pub fn set_tokens(dev: &GpuDevice, tp: &GpuBuffer, t: u32) -> Result<(), String> {
    let mut p = t.to_le_bytes().to_vec();
    p.resize(16, 0);
    dev.upload_bytes(tp, 0, &p)
}

/// The dispatches of `Y = W X` for a batch of up to `t_cap` tokens (one per chunk of the matrix). `x`: t_cap * cols floats,
/// `y`: t_cap * rows floats.
pub fn gemm_steps<'a>(dev: &GpuDevice, pipeline: &'a wgpu::ComputePipeline, m: &'a GpuMatrix, gp: &'a GemmParams, x: &'a GpuBuffer, y: &'a GpuBuffer, tp: &'a GpuBuffer, t_cap: u32) -> Result<Vec<Step<'a>>, String> {
    if t_cap == 0 {
        return Err("gemm: empty batch".into());
    }
    if x.len < t_cap as u64 * m.cols * 4 || y.len < t_cap as u64 * m.rows * 4 {
        return Err("gemm: the x or y buffer is too small for the batch".into());
    }
    let maxg = dev.caps.max_compute_workgroups_per_dimension as u64;
    let mut out = Vec::new();
    for (c, p) in m.chunks.iter().zip(&gp.chunks) {
        let (gx, gy) = (c.rows.div_ceil(TILE as u64), (t_cap as u64).div_ceil(TILE as u64));
        if gx > maxg || gy > maxg {
            return Err(format!("gemm: a {gx} x {gy} grid does not fit this adapter"));
        }
        let mut buffers: Vec<&GpuBuffer> = c.bufs.iter().collect();
        buffers.push(x);
        buffers.push(y);
        buffers.push(p);
        buffers.push(tp);
        out.push(Step { pipeline, buffers, groups: [gx as u32, gy as u32, 1] });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::DEFAULT_TIMEOUT;
    use crate::gpu::matvec::upload_matrix;
    use crate::gpu::pack::ErrAcc;
    use crate::gpu::select::AdapterSelector;

    const ALL: [Dtype; 10] = [Dtype::F32, Dtype::F16, Dtype::Q8_0, Dtype::Q4_0, Dtype::Q4_1, Dtype::Q5_0, Dtype::Q5_1, Dtype::Q4K, Dtype::Q5K, Dtype::Q6K];
    const ROWS: usize = 70; // not a multiple of the tile
    const COLS: usize = 512; // a multiple of 256 (the K-quants)
    const TOKENS: usize = 70; // more than one tile of 64 tokens and not a multiple of it
    const CAP: usize = 128; // the capacity the grid is sized for

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

    /// Runs the kernel on a synthetic matrix (`max_bytes` forces several chunks) and returns the worst error, relative to the sum of
    /// |w x|, against the exact f64 reference of every (token, row) pair.
    fn run(dev: &GpuDevice, k: &GemmKernel, max_bytes: Option<u64>) -> f64 {
        let d = k.dtype;
        let source = move |s: u64, c: u64| pack::gen_rows(d, s, c, COLS, 77);
        let m = upload_matrix(dev, d, ROWS as u64, COLS as u64, &source, max_bytes).unwrap();
        let gp = gemm_params(dev, &m).unwrap();
        let tp = token_count_buffer(dev).unwrap();
        set_tokens(dev, &tp, TOKENS as u32).unwrap();
        let x: Vec<f32> = pack::gen_x(TOKENS * COLS, 5);
        let xb = dev.storage_from_f32(&x, "t.x").unwrap();
        let yb = dev.create_storage((TOKENS * ROWS * 4) as u64, "t.y").unwrap();
        let pipe = k.pipeline(dev).unwrap();
        // the capacity is two tiles of tokens, the call uses 37
        let st = gemm_steps(dev, &pipe, &m, &gp, &xb, &yb, &tp, CAP as u32).err();
        assert!(st.is_some(), "a batch larger than the buffers must be rejected");
        let xb2 = dev.storage_from_f32(&[x.clone(), vec![0.0; (CAP - TOKENS) * COLS]].concat(), "t.x2").unwrap();
        let yb2 = dev.create_storage((CAP * ROWS * 4) as u64, "t.y2").unwrap();
        let steps = gemm_steps(dev, &pipe, &m, &gp, &xb2, &yb2, &tp, CAP as u32).unwrap();
        dev.run_steps(&steps, 1, DEFAULT_TIMEOUT).unwrap();
        let y = dev.read_back_f32(&yb2, CAP * ROWS, DEFAULT_TIMEOUT).unwrap();
        let _ = yb;
        let mut acc = ErrAcc::default();
        for t in 0..TOKENS {
            for r in 0..ROWS {
                let row = pack::gen_rows(d, r as u64, 1, COLS, 77);
                acc.add(y[t * ROWS + r], pack::reference_row(d, &row, &x[t * COLS..(t + 1) * COLS]));
            }
        }
        // tokens past the count must not be written (the output starts zeroed)
        assert!(y[TOKENS * ROWS..].iter().all(|&v| v == 0.0), "{}: a token beyond the batch was written", pack::name(d));
        acc.finish().max_gpu
    }

    const TOL: f64 = 1e-5;

    fn check(dev: &GpuDevice, formats: &[Dtype]) {
        for &d in formats {
            let k = GemmKernel::new(d).unwrap();
            let single = run(dev, &k, None);
            // several chunks: the limit is a few rows of the format
            let small = pack::array_row_bytes(d, COLS).iter().sum::<u64>() * 24;
            let multi = run(dev, &k, Some(small));
            println!("  {:>5}: error vs the exact reference, single chunk {single:.2e}, several chunks {multi:.2e}", pack::name(d));
            assert!(single < TOL && multi < TOL, "{}: gemm error {single:e} / {multi:e}", pack::name(d));
        }
    }

    fn selector() -> AdapterSelector {
        std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best)
    }

    #[test]
    fn kernel_assembly_is_pure() {
        for d in ALL {
            let k = GemmKernel::new(d).unwrap();
            assert!(k.wgsl.contains("fn weight("), "{}: the decoder is missing", pack::name(d));
            assert!(!k.wgsl.contains('@') || !k.wgsl.contains("@X@"), "{}: an unreplaced marker", pack::name(d));
            assert!(k.wgsl.contains("fn main("));
        }
    }

    #[test]
    fn software_adapter_gemm() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        check(&dev, &[Dtype::F16, Dtype::Q8_0, Dtype::Q4K]);
    }

    #[test]
    #[ignore = "needs a real GPU: BURITI_TEST_ADAPTER=<selector> selects the adapter (default: best)"]
    fn real_adapter_gemm() {
        let Some(dev) = open_or_skip(&selector(), "real") else { return };
        check(&dev, &ALL);
    }

    /// Each planted bug must exceed the tolerance. `limit` bounds how many of the bugs run (the software adapter compiles slowly:
    /// the default test runs the first three, the real-adapter test runs all of them).
    fn planted_bugs(dev: &GpuDevice, limit: usize) {
        let base = GemmKernel::new(Dtype::Q8_0).unwrap();
        let src = base.wgsl.clone();
        let bugs: Vec<(&str, String)> = vec![
            ("column_swapped", replace_once(&src, "wv = weight4(row0 + sr, k0 + kq);", "wv = weight4(row0 + sr, k0 + (kq ^ 4u));")),
            ("chunk_start_dropped", replace_once(&src, "yout[t * m_total + base + r] = acc[i * 4u + j];", "yout[t * m_total + r] = acc[i * 4u + j];")),
            ("last_accumulator_uses_wrong_x", replace_once(&src, "acc[15] = acc[15] + w3 * x3;", "acc[15] = acc[15] + w3 * x2;")),
            ("x_column_shifted", replace_once(&src, "xt[(kq + 1u) * 16u + sg][sl] = xin[xb + 1u];", "xt[(kq + 1u) * 16u + sg][sl] = xin[xb + 2u];")),
            ("k_tail_skipped", replace_once(&src, "for (var k0 = 0u; k0 < cols; k0 = k0 + KC) {", "for (var k0 = 0u; k0 + KC < cols; k0 = k0 + KC) {")),
            ("second_row_uses_first_weight", replace_once(&src, "acc[5] = acc[5] + w1 * x1;", "acc[5] = acc[5] + w0 * x1;")),
            ("shared_layout_stride_wrong", replace_once(&src, "wt[(kq + 2u) * 16u + sg][sl] = wv.z;", "wt[(kq + 2u) * 16u + sg][sl] = wv.w;")),
            ("weight_lane_swapped", replace_once(&src, "wt[(kq + 1u) * 16u + sg][sl] = wv.y;", "wt[(kq + 1u) * 16u + sg][sl] = wv.x;")),
            // not planted: the staging checks of the token and row bounds and the bounds on the writes of tokens beyond the count only
            // keep reads inside the buffers and the zero padding intact; removing them changes accumulators that are staged as zero or
            // never read, so those mutants are equivalent (verified: the first attempt at planting them was not detected, as expected).
        ];
        for (tag, wgsl) in bugs.into_iter().take(limit) {
            let k = base.with_wgsl(tag, wgsl);
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let a = run(dev, &k, None);
                let small = pack::array_row_bytes(Dtype::Q8_0, COLS).iter().sum::<u64>() * 24;
                let b = run(dev, &k, Some(small));
                a >= TOL || b >= TOL
            }))
            .unwrap_or(true);
            assert!(caught, "planted bug '{tag}' was not detected");
            println!("planted gemm bug '{tag}' detected");
        }
    }

    #[test]
    fn software_adapter_planted_bugs() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        planted_bugs(&dev, 3);
    }

    #[test]
    #[ignore = "needs a real GPU: BURITI_TEST_ADAPTER=<selector> selects the adapter (default: best)"]
    fn real_adapter_planted_bugs() {
        let Some(dev) = open_or_skip(&selector(), "real") else { return };
        planted_bugs(&dev, usize::MAX);
    }
}
