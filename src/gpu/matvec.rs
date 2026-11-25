//! Matrix-vector product on the GPU, `y = W x` for one token (tasks G4, G5 and G5b, D64, D65): f32, F16, Q8_0,
//! Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K weights, dequantized on the fly in the shader, activations and
//! output in f32.
//!
//! Layout and splitting live in `pack.rs` (pure). Here: the kernels (WGSL in `shaders/mv_*.wgsl`, one
//! workgroup per row with a shared-memory tree reduction, no subgroup or f16 feature needed), the upload of a
//! matrix as row-aligned chunks that respect the adapter limits, the dispatch, and the checks against the
//! exact float reference. The CPU engine quantizes activations to int8; these kernels do not (D64), so the
//! reference is the float one, never the CPU engine's integer path.

#![allow(dead_code)]

use super::device::{GpuBuffer, GpuDevice, RunTiming, Step, DEFAULT_TIMEOUT};
use super::kernels::grid_2d;
use super::pack::{self, ErrAcc, ErrStats};
use crate::native::gguf::Dtype;
#[cfg(test)]
use crate::native::quant;
use std::sync::Arc;

#[allow(unused_imports)]
pub use super::pack::DEFAULT_WG;
const WG_LINE: &str = "const WG: u32 = 64u;";
const ROWS_LINE: &str = "const ROWS: u32 = 1u;";
const UPLOAD_SLICE: usize = 16 << 20;

fn shader_source(d: Dtype) -> Option<&'static str> {
    Some(match d {
        Dtype::F32 => include_str!("shaders/mv_f32.wgsl"),
        Dtype::F16 => include_str!("shaders/mv_f16.wgsl"),
        Dtype::Q8_0 => include_str!("shaders/mv_q8_0.wgsl"),
        Dtype::Q4K => include_str!("shaders/mv_q4_k.wgsl"),
        Dtype::Q6K => include_str!("shaders/mv_q6_k.wgsl"),
        Dtype::Q4_0 => include_str!("shaders/mv_q4_0.wgsl"),
        Dtype::Q4_1 => include_str!("shaders/mv_q4_1.wgsl"),
        Dtype::Q5_0 => include_str!("shaders/mv_q5_0.wgsl"),
        Dtype::Q5_1 => include_str!("shaders/mv_q5_1.wgsl"),
        Dtype::Q5K => include_str!("shaders/mv_q5_k.wgsl"),
        _ => return None,
    })
}

/// A shader with its pipeline-cache key. The tests build broken variants with `with_wgsl`.
#[derive(Clone)]
pub struct MvKernel {
    pub dtype: Dtype,
    pub key: String,
    pub wgsl: String,
    pub wg: u32,
    /// Rows computed by one workgroup (the quantized kernels share the x loads between them).
    pub rows_per_group: u32,
}

impl MvKernel {
    /// The default configuration of the format (`pack::default_rows`).
    pub fn new(dtype: Dtype, wg: u32) -> Result<MvKernel, String> {
        MvKernel::config(dtype, wg, pack::default_rows(dtype))
    }

    pub fn config(dtype: Dtype, wg: u32, rows_per_group: u32) -> Result<MvKernel, String> {
        if !wg.is_power_of_two() || !(16..=256).contains(&wg) {
            return Err(format!("workgroup size {wg}: must be a power of two between 16 and 256"));
        }
        if ![1, 2, 4, 8].contains(&rows_per_group) || (rows_per_group > 1 && !pack::multi_row(dtype)) {
            return Err(format!("{}: {rows_per_group} rows per workgroup is not supported", pack::name(dtype)));
        }
        let src = shader_source(dtype).ok_or_else(|| format!("{} has no GPU matvec kernel yet", pack::name(dtype)))?.replace("\r\n", "\n");
        if src.matches(WG_LINE).count() != 1 {
            return Err("internal error: the shader has no workgroup-size line".into());
        }
        let mut wgsl = src.replace(WG_LINE, &format!("const WG: u32 = {wg}u;"));
        if pack::multi_row(dtype) {
            if wgsl.matches(ROWS_LINE).count() != 1 {
                return Err("internal error: the shader has no rows-per-workgroup line".into());
            }
            wgsl = wgsl.replace(ROWS_LINE, &format!("const ROWS: u32 = {rows_per_group}u;"));
        }
        Ok(MvKernel { dtype, key: format!("mv_{}_wg{wg}_r{rows_per_group}", pack::name(dtype)), wgsl, wg, rows_per_group })
    }

    pub fn with_wgsl(&self, tag: &str, wgsl: String) -> MvKernel {
        MvKernel { dtype: self.dtype, key: format!("{}_{tag}", self.key), wgsl, wg: self.wg, rows_per_group: self.rows_per_group }
    }

    pub fn pipeline(&self, dev: &GpuDevice) -> Result<Arc<wgpu::ComputePipeline>, String> {
        if self.wg > dev.caps.max_compute_invocations_per_workgroup || self.wg > dev.caps.max_compute_workgroup_size_x {
            return Err(format!("workgroup size {} is above this adapter's limits", self.wg));
        }
        dev.pipeline(&self.key, &self.wgsl, "main")
    }
}

pub struct GpuChunk {
    pub start: u64,
    pub rows: u64,
    pub bufs: Vec<GpuBuffer>,
    pub params: GpuBuffer,
}

/// A weight matrix resident on the GPU, as row-aligned chunks.
pub struct GpuMatrix {
    pub dtype: Dtype,
    pub rows: u64,
    pub cols: u64,
    pub chunks: Vec<GpuChunk>,
    /// Bytes of the weight buffers on the GPU (after repacking).
    pub resident_bytes: u64,
    /// The same weights in the GGUF layout.
    pub native_bytes: u64,
}

/// Uploads a `rows x cols` matrix. `source(start, count)` returns the GGUF-layout bytes of those rows (so a
/// matrix bigger than RAM-friendly never has to exist as one `Vec`). `max_bytes` lowers the per-buffer limit
/// (tests use it to force several chunks); it can never raise the adapter's own limit.
pub fn upload_matrix(dev: &GpuDevice, dtype: Dtype, rows: u64, cols: u64, source: &dyn Fn(u64, u64) -> Vec<u8>, max_bytes: Option<u64>) -> Result<GpuMatrix, String> {
    pack::check_shape(dtype, rows as usize, cols as usize)?;
    if rows > u32::MAX as u64 || cols > u32::MAX as u64 {
        return Err("matrix dimensions above 2^32".into());
    }
    let limit = dev.caps.max_buffer_size.min(dev.caps.max_storage_buffer_binding_size);
    let max = max_bytes.map_or(limit, |m| m.min(limit));
    let rbs = pack::array_row_bytes(dtype, cols as usize);
    let plan = pack::plan_chunks(rows, &rbs, max)?;
    pack::check_plan(&plan, rows, &rbs, max)?;
    let nb = pack::blocks_per_row(dtype, cols as usize) as u32;
    let mut chunks = Vec::new();
    let mut resident = 0;
    for c in plan {
        let raw = source(c.start, c.rows);
        let arrays = pack::repack_rows(dtype, &raw, c.rows as usize, cols as usize)?;
        drop(raw);
        let mut bufs = Vec::new();
        for (i, a) in arrays.iter().enumerate() {
            let b = dev.create_storage(a.len() as u64, &format!("mv.w{i}"))?;
            let mut at = 0;
            while at < a.len() {
                let n = UPLOAD_SLICE.min(a.len() - at);
                dev.upload_bytes(&b, at as u64, &a[at..at + n])?;
                at += n;
            }
            resident += a.len() as u64;
            bufs.push(b);
        }
        let mut p = Vec::new();
        for v in [c.rows as u32, cols as u32, c.start as u32, nb, 0, 0, 0, 0] {
            p.extend_from_slice(&v.to_le_bytes());
        }
        let params = dev.storage_from_bytes(&p, "mv.params")?;
        chunks.push(GpuChunk { start: c.start, rows: c.rows, bufs, params });
    }
    dev.check_health()?;
    Ok(GpuMatrix { dtype, rows, cols, chunks, resident_bytes: resident, native_bytes: pack::native_row_bytes(dtype, cols as usize) * rows })
}

/// Makes the buffers for an activation vector and the output of a matrix.
pub fn make_io(dev: &GpuDevice, x: &[f32], rows: u64) -> Result<(GpuBuffer, GpuBuffer), String> {
    Ok((dev.storage_from_f32(x, "mv.x")?, dev.create_storage(rows * 4, "mv.y")?))
}

pub(crate) fn steps<'a>(dev: &GpuDevice, rows_per_group: u32, pipeline: &'a wgpu::ComputePipeline, mats: &[&'a GpuMatrix], x: &'a GpuBuffer, y: &'a GpuBuffer) -> Result<Vec<Step<'a>>, String> {
    let mut out = Vec::new();
    for m in mats {
        if x.len < m.cols * 4 || y.len < m.rows * 4 {
            return Err("matvec: the x or y buffer is too small for the matrix".into());
        }
        for c in &m.chunks {
            let mut buffers: Vec<&GpuBuffer> = c.bufs.iter().collect();
            buffers.push(x);
            buffers.push(y);
            buffers.push(&c.params);
            out.push(Step { pipeline, buffers, groups: grid_2d(c.rows.div_ceil(rows_per_group as u64), dev.caps.max_compute_workgroups_per_dimension)? });
        }
    }
    Ok(out)
}

/// Runs `y = W x` for every matrix in `mats` (all of the kernel's format) `repeat` times inside one pass.
pub fn run_matvec(dev: &GpuDevice, k: &MvKernel, mats: &[&GpuMatrix], x: &GpuBuffer, y: &GpuBuffer, repeat: u32) -> Result<RunTiming, String> {
    if mats.iter().any(|m| m.dtype != k.dtype) {
        return Err("matvec: the matrix and the kernel have different formats".into());
    }
    let pipeline = k.pipeline(dev)?;
    let st = steps(dev, k.rows_per_group, &pipeline, mats, x, y)?;
    dev.run_steps(&st, repeat, DEFAULT_TIMEOUT)
}

/// One `y = W x` with a fresh input and output; returns y.
pub fn matvec(dev: &GpuDevice, k: &MvKernel, m: &GpuMatrix, x: &[f32]) -> Result<Vec<f32>, String> {
    if x.len() as u64 != m.cols {
        return Err(format!("matvec: x has {} elements, the matrix has {} columns", x.len(), m.cols));
    }
    let (bx, by) = make_io(dev, x, m.rows)?;
    run_matvec(dev, k, &[m], &bx, &by, 1)?;
    dev.read_back_f32(&by, m.rows as usize, DEFAULT_TIMEOUT)
}

/// Rows worth checking: the first and last row, both edges of every chunk, and evenly spaced ones up to `n`.
pub fn sample_rows(m: &GpuMatrix, n: usize) -> Vec<u64> {
    let mut v = vec![0, m.rows - 1];
    for c in &m.chunks {
        v.push(c.start);
        v.push(c.start + c.rows - 1);
    }
    let step = (m.rows / n.max(1) as u64).max(1);
    let mut r = 0;
    while r < m.rows {
        v.push(r);
        r += step;
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Error of `y` (the GPU result) against the exact float reference on `rows_to_check`.
pub fn error_stats(d: Dtype, cols: usize, y: &[f32], x: &[f32], source: &dyn Fn(u64, u64) -> Vec<u8>, rows_to_check: &[u64]) -> ErrStats {
    let mut acc = ErrAcc::default();
    for &r in rows_to_check {
        let row = source(r, 1);
        acc.add(y[r as usize], pack::reference_row(d, &row, &x[..cols]));
    }
    acc.finish()
}

/// Timed passes over `mats`: the sequence is dispatched back to back inside one pass (about `target_s`
/// seconds per pass), `passes` times. GB/s counts the resident weight bytes read.
pub struct Timed {
    pub gbps: Vec<f64>,
    pub wall_gbps: Vec<f64>,
    pub iters: u32,
    pub timestamp: bool,
}

pub fn time_matvec(dev: &GpuDevice, k: &MvKernel, mats: &[&GpuMatrix], x: &GpuBuffer, y: &GpuBuffer, passes: usize, target_s: f64) -> Result<Timed, String> {
    let warm = run_matvec(dev, k, mats, x, y, 1)?;
    let t1 = warm.gpu_seconds.unwrap_or(warm.wall_seconds).max(1e-6);
    let iters = ((target_s / t1).ceil() as u32).clamp(1, 400);
    let bytes: f64 = mats.iter().map(|m| m.resident_bytes as f64).sum::<f64>() * iters as f64;
    let (mut g, mut w) = (Vec::new(), Vec::new());
    let mut ts = true;
    for _ in 0..passes {
        let t = run_matvec(dev, k, mats, x, y, iters)?;
        w.push(super::bench::gbps(bytes, t.wall_seconds));
        match t.gpu_seconds {
            Some(s) => g.push(super::bench::gbps(bytes, s)),
            None => {
                ts = false;
                g.push(super::bench::gbps(bytes, t.wall_seconds));
            }
        }
    }
    Ok(Timed { gbps: g, wall_gbps: w, iters, timestamp: ts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::select::AdapterSelector;

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

    /// (rows, cols) used for correctness, per format. The second one is also used for the planted bugs.
    fn shapes(d: Dtype) -> Vec<(u64, u64)> {
        match d {
            Dtype::F32 => vec![(1, 4), (37, 64), (5, 1000), (200, 4096)],
            Dtype::F16 => vec![(1, 8), (37, 64), (5, 1000), (64, 4096)],
            Dtype::Q8_0 | Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 => vec![(1, 32), (37, 96), (50, 896), (100, 4864)],
            Dtype::Q4K | Dtype::Q6K | Dtype::Q5K => vec![(1, 256), (37, 512), (30, 4864)],
            _ => vec![],
        }
    }

    /// Shape for the planted bugs: every thread of the workgroup has work (items >= WG), and for Q8_0 an odd block count (35) so the scale padding matters.
    fn bug_shape(d: Dtype) -> (u64, u64) {
        match d {
            Dtype::F32 | Dtype::F16 => (20, 4096),
            Dtype::Q8_0 | Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 => (20, 1120),
            _ => (10, 4864),
        }
    }

    const FORMATS: [Dtype; 10] = pack::ALL_FORMATS;

    /// GPU result of one matrix against the exact reference, on every row.
    fn stats_for(dev: &GpuDevice, k: &MvKernel, d: Dtype, rows: u64, cols: u64, max_bytes: Option<u64>, seed: u64) -> (ErrStats, usize) {
        let src = move |s: u64, n: u64| pack::gen_rows(d, s, n, cols as usize, seed);
        let m = upload_matrix(dev, d, rows, cols, &src, max_bytes).expect("upload");
        let x = pack::gen_x(cols as usize, seed + 1);
        let y = matvec(dev, k, &m, &x).expect("matvec");
        let all: Vec<u64> = (0..rows).collect();
        (error_stats(d, cols as usize, &y, &x, &src, &all), m.chunks.len())
    }

    fn bugs(d: Dtype) -> Vec<(&'static str, &'static str, &'static str)> {
        let (w_from, w_a, w_b) = if pack::multi_row(d) {
            ("y[p[2] + row0 + lid.x] = scratch[lid.x * WG];", "y[p[2] + row0 + lid.x + 1u] = scratch[lid.x * WG];", "y[row0 + lid.x] = scratch[lid.x * WG];")
        } else {
            ("y[p[2] + row] = scratch[0];", "y[p[2] + row + 1u] = scratch[0];", "y[row] = scratch[0];")
        };
        let mut v = vec![
            ("reduction starts too low (half of the partials dropped)", "var s = WG / 2u;", "var s = WG / 4u;"),
            ("output index shifted by one row", w_from, w_a),
            ("chunk row offset ignored", w_from, w_b),
            ("thread stride doubled (items skipped)", "i += WG;", "i += WG * 2u;"),
        ];
        match d {
            Dtype::F32 => v.push(("wrong row stride", "let base = row * k4;", "let base = row * k4 + 1u;")),
            Dtype::F16 => {
                v.push(("wrong row stride", "let base = row * k8;", "let base = row * k8 + 1u;"));
                v.push(("halves swapped", "dot(vec4<f32>(a, b), x[2u * i])", "dot(vec4<f32>(b, a), x[2u * i])"));
            }
            Dtype::Q8_0 => {
                v.push(("wrong quant row stride", "let q = qs[row * items + i];", "let q = qs[row * items + i + 1u];"));
                v.push(("scale rows not padded to an even block count", "let sw = (nb + 1u) / 2u;", "let sw = nb / 2u;"));
                v.push(("sign lost in the top byte", "bitcast<i32>(v) >> 24u)", "bitcast<i32>(v >> 24u))"));
                v.push(("wrong x vector offset", "acc[r] += d * (dot(i8x4(q.x), x0)", "acc[r] += d * (dot(i8x4(q.x), x1)"));
            }
            Dtype::Q4K => {
                v.push(("wrong block stride", "let rb = row * nb * 9u + b * 9u;", "let rb = row * nb * 8u + b * 9u;"));
                v.push(("wrong high bits of the packed scales", "((sbyte(h, j - 4u) >> 6u) << 4u)", "((sbyte(h, j - 4u) >> 6u) << 3u)"));
                v.push(("min term dropped", "- dd.y * (f32(s1.y) * sl + f32(s2.y) * sh)", ""));
            }
            Dtype::Q4_0 => {
                v.push(("wrong quant row stride", "let q = qs[row * nb + i];", "let q = qs[row * nb + i + 1u];"));
                v.push(("scale rows not padded to an even block count", "let sw = (nb + 1u) / 2u;", "let sw = nb / 2u;"));
                v.push(("high nibbles taken against the low x", "dot(w4(q.x, 4u), h0)", "dot(w4(q.x, 4u), l0)"));
                v.push(("offset 8 wrong", "return nib(v, sh) - vec4<f32>(8.0);", "return nib(v, sh) - vec4<f32>(7.0);"));
                v.push(("nibble byte order", "(v >> (sh + 8u)) & 15u", "(v >> (sh + 16u)) & 15u"));
                v.push(("scale pair order swapped", "select(s2.x, s2.y, (i & 1u) == 1u)", "select(s2.y, s2.x, (i & 1u) == 1u)"));
            }
            Dtype::Q4_1 => {
                v.push(("wrong quant row stride", "let q = qs[row * nb + i];", "let q = qs[row * nb + i + 1u];"));
                v.push(("wrong d/m row stride", "dm[row * nb + i]", "dm[row * nb + i + 1u]"));
                v.push(("min term dropped", "+ s2.y * sx;", ";"));
                v.push(("d and m swapped", "s2.x * (lo + hi) + s2.y * sx", "s2.y * (lo + hi) + s2.x * sx"));
                v.push(("high nibbles taken against the low x", "dot(nib(q.y, 4u), h1)", "dot(nib(q.y, 4u), l1)"));
                v.push(("sum of x misses the high half", "l0 + l1 + l2 + l3 + h0 + h1 + h2 + h3", "l0 + l1 + l2 + l3"));
            }
            Dtype::Q5_0 => {
                v.push(("wrong quant row stride", "let q = qs[row * nb + i];", "let q = qs[row * nb + i + 1u];"));
                v.push(("wrong qh row stride", "let qhw = qh[row * nb + i];", "let qhw = qh[row * nb + i + 1u];"));
                v.push(("scale rows not padded to an even block count", "let sw = (nb + 1u) / 2u;", "let sw = nb / 2u;"));
                v.push(("5th bit of the high half read from the low bits", "bits4(qhw, 16u)", "bits4(qhw, 0u)"));
                v.push(("5th bit of the second quad off by one", "bits4(qhw, 4u)", "bits4(qhw, 5u)"));
                v.push(("offset 16 wrong", "return n + b - vec4<f32>(16.0);", "return n + b - vec4<f32>(15.0);"));
                v.push(("5th bit worth 8 instead of 16", "((v >> sh) & 1u) << 4u,", "((v >> sh) & 1u) << 3u,"));
            }
            Dtype::Q5_1 => {
                v.push(("wrong quant row stride", "let q = qs[row * nb + i];", "let q = qs[row * nb + i + 1u];"));
                v.push(("wrong qh row stride", "let qhw = qh[row * nb + i];", "let qhw = qh[row * nb + i + 1u];"));
                v.push(("wrong d/m row stride", "dm[row * nb + i]", "dm[row * nb + i + 1u]"));
                v.push(("min term dropped", "+ s2.y * sx;", ";"));
                v.push(("5th bit of the high half read from the low bits", "bits4(qhw, 16u)", "bits4(qhw, 0u)"));
                v.push(("5th bit worth 8 instead of 16 (third weight)", "((v >> (sh + 2u)) & 1u) << 4u", "((v >> (sh + 2u)) & 1u) << 3u"));
            }
            Dtype::Q5K => {
                v.push(("wrong block stride", "let rb = row * nb * 11u + b * 11u;", "let rb = row * nb * 10u + b * 11u;"));
                v.push(("wrong qh vector", "let qv = w[rb + 1u + h];", "let qv = w[rb + 1u];"));
                v.push(("5th bit of the high group read from the low group", "let bh = qv >> vec4<u32>(2u * g + 1u);", "let bh = qv >> vec4<u32>(2u * g);"));
                v.push(("wrong group bit for the low group", "let bl = qv >> vec4<u32>(2u * g);", "let bl = qv >> vec4<u32>(g);"));
                v.push(("min term dropped", "- dd.y * (f32(s1.y) * sl + f32(s2.y) * sh)", ""));
                v.push(("wrong high bits of the packed scales", "((sbyte(h, j - 4u) >> 6u) << 4u)", "((sbyte(h, j - 4u) >> 6u) << 3u)"));
                v.push(("wrong quant vector", "let q = w[rb + 3u + g * 2u + h];", "let q = w[rb + 2u + g * 2u + h];"));
            }
            Dtype::Q6K => {
                v.push(("wrong block stride", "return row * nb * 53u + b * 53u;", "return row * nb * 52u + b * 53u;"));
                v.push(("wrong high bits of the second value", "(((hq >> 2u) & m3) << 4u)", "(((hq >> 2u) & m3) << 3u)"));
                v.push(("wrong scale index", "let si = n * 8u + (c >> 1u);", "let si = n * 8u + (c & 1u);"));
            }
            _ => {}
        }
        v
    }

    /// y = W e_c is column c of W: compares every column with the CPU decoder (`quant::dequant_row`). The only
    /// differences allowed are the float rounding of the GPU's order of operations (d * (q * 1) + m against
    /// q * d + m), so the tolerance is tiny against the row's magnitude. Returns the largest error seen.
    fn onehot_decode(dev: &GpuDevice, d: Dtype) -> f32 {
        let cols: usize = if pack::col_multiple(d) == 256 { 256 } else { 96 };
        let rows = 5u64;
        let src = move |s: u64, n: u64| pack::gen_rows(d, s, n, cols, 21);
        let m = upload_matrix(dev, d, rows, cols as u64, &src, Some(pack::array_row_bytes(d, cols).into_iter().max().unwrap() * 2)).expect("upload");
        assert_eq!(m.chunks.len(), 3, "the one-hot matrix is split in chunks of 2 rows");
        let k = MvKernel::new(d, DEFAULT_WG).unwrap();
        let rb = pack::native_row_bytes(d, cols) as usize;
        let raw = src(0, rows);
        let mut want = vec![vec![0f32; cols]; rows as usize];
        for r in 0..rows as usize {
            quant::dequant_row(d, &raw[r * rb..(r + 1) * rb], &mut want[r]);
        }
        let scale: Vec<f32> = want.iter().map(|w| w.iter().fold(0f32, |a, v| a.max(v.abs()))).collect();
        let mut worst = 0f32;
        let mut exact = 0usize;
        for c in 0..cols {
            let mut x = vec![0f32; cols];
            x[c] = 1.0;
            let y = matvec(dev, &k, &m, &x).expect("matvec");
            for r in 0..rows as usize {
                let e = (y[r] - want[r][c]).abs();
                if y[r].to_bits() == want[r][c].to_bits() {
                    exact += 1;
                }
                worst = worst.max(e / scale[r].max(1e-6));
                assert!(e <= 2e-6 * scale[r], "{} row {r} column {c}: GPU {} vs CPU decoder {} (row magnitude {})", pack::name(d), y[r], want[r][c], scale[r]);
            }
        }
        println!("  {}: one-hot decode of {} weights: {} bit-identical, worst error {:.2e} of the row magnitude", pack::name(d), rows as usize * cols, exact, worst);
        worst
    }

    /// The whole battery for one device (software adapter in CI, real adapters when run by hand).
    fn exercise(dev: &GpuDevice) {
        let tol = pack::TOL;
        for d in FORMATS {
            let k = MvKernel::new(d, DEFAULT_WG).unwrap();
            let name = pack::name(d);
            let mut worst = ErrStats::default();
            for (rows, cols) in shapes(d) {
                let (s, n) = stats_for(dev, &k, d, rows, cols, None, 11);
                assert_eq!(n, 1, "no splitting expected without a limit");
                assert!(s.passes(tol), "{name} {rows}x{cols}: GPU error {:.3e} above {tol:.0e} ({s:?})", s.max_gpu);
                // the same matrix split into chunks of 3 rows must give the same answer
                let rb = pack::array_row_bytes(d, cols as usize).into_iter().max().unwrap();
                let (s2, n2) = stats_for(dev, &k, d, rows, cols, Some(rb * 3), 11);
                assert_eq!(n2 as u64, rows.div_ceil(3), "{name}: chunk count");
                assert!(s2.passes(tol), "{name} {rows}x{cols} in chunks: GPU error {:.3e} ({s2:?})", s2.max_gpu);
                if s.max_gpu > worst.max_gpu {
                    worst = s;
                }
            }
            println!("  {name}: worst GPU error {:.2e} (CPU float kernel {:.2e}), rms ratio {:.2}", worst.max_gpu, worst.max_cpu, worst.ratio());

            // planted bugs, on the chunked bug shape: every one must be caught
            let (rows, cols) = bug_shape(d);
            let k = MvKernel::config(d, DEFAULT_WG, if pack::multi_row(d) { 4 } else { 1 }).unwrap(); // bugs are planted in the 4-rows-per-workgroup variant
            let rb = pack::array_row_bytes(d, cols as usize).into_iter().max().unwrap();
            let (ok, _) = stats_for(dev, &k, d, rows, cols, Some(rb * 3), 11);
            assert!(ok.passes(tol), "{name}: the unbroken kernel must pass on the bug shape ({ok:?})");
            for (i, (what, from, to)) in bugs(d).into_iter().enumerate() {
                let bad = k.with_wgsl(&format!("bug{i}"), replace_once(&k.wgsl, from, to));
                let (s, _) = stats_for(dev, &bad, d, rows, cols, Some(rb * 3), 11);
                assert!(!s.passes(tol), "{name}: planted bug '{what}' was NOT detected (error {:.3e})", s.max_gpu);
            }
        }
        // More rows than one grid dimension allows (65535 on most adapters): the 2D row index must hold.
        let big = dev.caps.max_compute_workgroups_per_dimension as u64 + 1000;
        if big <= 200_000 {
            let k = MvKernel::new(Dtype::F32, DEFAULT_WG).unwrap();
            let (s, _) = stats_for(dev, &k, Dtype::F32, big, 8, None, 5);
            assert!(s.passes(pack::TOL), "{big} rows: {s:?}");
        } else {
            println!("  (2D grid not exercised: the adapter allows {} workgroups per dimension)", dev.caps.max_compute_workgroups_per_dimension);
        }
        // Other workgroup sizes give the same answer.
        for wg in [32u32, 128, 256] {
            if wg > dev.caps.max_compute_invocations_per_workgroup {
                continue;
            }
            for d in [Dtype::F16, Dtype::Q4K] {
                let k = MvKernel::new(d, wg).unwrap();
                let (s, _) = stats_for(dev, &k, d, 20, 1024, None, 3);
                assert!(s.passes(pack::TOL), "{} wg {wg}: {s:?}", pack::name(d));
            }
        }
        // Several rows per workgroup (the x loads are shared): every row count, including a last partial group.
        for d in [Dtype::Q8_0, Dtype::Q4K, Dtype::Q6K, Dtype::Q4_0, Dtype::Q4_1, Dtype::Q5_0, Dtype::Q5_1, Dtype::Q5K] {
            // the default (4) is covered by the battery above; the legacy formats and Q5_K check the others on one shape set too
            for rpg in [1u32, 2, 8] {
                let k = MvKernel::config(d, DEFAULT_WG, rpg).unwrap();
                for (rows, cols) in [(1u64, 512u64), (13, 512), (37, 1280)] {
                    let rb = pack::array_row_bytes(d, cols as usize).into_iter().max().unwrap();
                    for max in [None, Some(rb * 3)] {
                        let (s, _) = stats_for(dev, &k, d, rows, cols, max, 7);
                        assert!(s.passes(pack::TOL), "{} rows/group {rpg} {rows}x{cols} {max:?}: {s:?}", pack::name(d));
                    }
                }
            }
        }
        // The GPU decode must be the CPU decoder's, weight by weight: a one-hot x picks one column, so y is that
        // column of the matrix. This is what catches a wrong nibble order or 5th-bit position directly.
        for d in [Dtype::Q4_0, Dtype::Q4_1, Dtype::Q5_0, Dtype::Q5_1, Dtype::Q5K] {
            onehot_decode(dev, d);
        }
        // Errors, not panics: bad shapes and a kernel/matrix mismatch.
        let src = |s: u64, n: u64| pack::gen_rows(Dtype::Q4K, s, n, 896, 1);
        assert!(upload_matrix(dev, Dtype::Q4K, 4, 896, &src, None).is_err(), "896 columns are not a multiple of 256");
        let m = upload_matrix(dev, Dtype::Q8_0, 4, 64, &|s, n| pack::gen_rows(Dtype::Q8_0, s, n, 64, 1), None).unwrap();
        let k = MvKernel::new(Dtype::F32, DEFAULT_WG).unwrap();
        assert!(matvec(dev, &k, &m, &vec![0.0; 64]).is_err(), "format mismatch");
        assert!(matvec(dev, &MvKernel::new(Dtype::Q8_0, 64).unwrap(), &m, &vec![0.0; 63]).is_err(), "wrong x length");
        dev.check_health().expect("healthy at the end");
    }

    /// Informational (not a gate): does the OLD byte-extract form `(w >> ((j & 3u) * 8u)) & 255u` give wrong bytes
    /// on this adapter? It did on WARP for a shift of 24 (D64). Prints, per adapter, how many of the four shifts
    /// went wrong in each form. Run it by hand on every adapter: `BURITI_TEST_ADAPTER=... cargo test
    /// probe_old_byte_extract -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_old_byte_extract() {
        let sel = std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best);
        let dev = open_or_skip(&sel, "probe").expect("no adapter could be opened");
        let n = 1u32 << 16; // threads (bytes); words = n / 4
        let mut st = 0x1234_5678_9ABC_DEF1u64;
        let words: Vec<u32> = (0..n / 4)
            .map(|_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // every byte non-zero, so a lost or shifted byte shows
                let w = (st >> 32) as u32 | 0x0101_0101;
                w
            })
            .collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let bw = dev.storage_from_bytes(&bytes, "probe.words").unwrap();
        let outs: Vec<GpuBuffer> = (0..3).map(|i| dev.create_storage(n as u64 * 4, &format!("probe.out{i}")).unwrap()).collect();
        let params = dev.storage_from_bytes(&n.to_le_bytes().iter().chain(&[0u8; 12]).copied().collect::<Vec<u8>>(), "probe.p").unwrap();
        let pipeline = dev.pipeline("probe_byte_extract", include_str!("shaders/probe_byte_extract.wgsl"), "main").unwrap();
        let groups = grid_2d((n as u64).div_ceil(64), dev.caps.max_compute_workgroups_per_dimension).unwrap();
        // the shader indexes by gid.x only: 1D grid (n / 64 = 1024 groups fits every adapter)
        assert_eq!(groups[1], 1);
        dev.run(&pipeline, &[&bw, &bw, &outs[0], &outs[1], &outs[2], &params], groups, 1, DEFAULT_TIMEOUT).unwrap();
        let mut report = Vec::new();
        for (name, o) in ["old form, u32 array", "old form, vec4<u32> component", "new form (what the kernels use)"].iter().zip(&outs) {
            let got = dev.read_back(o, 0, n as u64 * 4, DEFAULT_TIMEOUT).unwrap();
            let mut wrong = [0u32; 4];
            for t in 0..n as usize {
                let v = u32::from_le_bytes(got[t * 4..t * 4 + 4].try_into().unwrap());
                let want = (words[t >> 2] >> ((t as u32 & 3) * 8)) & 255;
                if v != want {
                    wrong[t & 3] += 1;
                }
            }
            report.push(format!("{name}: wrong bytes per shift (0, 8, 16, 24) = {wrong:?} of {} each", n / 4));
        }
        println!("PROBE {} ({}):", dev.name(), dev.report.backend);
        for r in &report {
            println!("  {r}");
        }
        // The same old form inside the real kernels (the context where WARP failed): swap it back into the Q4_K and Q5_K
        // shaders (their scale bytes use shifts 0 to 24) and look at the error against the exact reference.
        for d in [Dtype::Q4K, Dtype::Q5K] {
            let k = MvKernel::config(d, DEFAULT_WG, 4).unwrap();
            let new_form = "return (h[1u + (j >> 2u)] << (24u - 8u * (j & 3u))) >> 24u;";
            let old_form = "return (h[1u + (j >> 2u)] >> ((j & 3u) * 8u)) & 255u;";
            let bad = k.with_wgsl("old_byte_extract", replace_once(&k.wgsl, new_form, old_form));
            let (s_new, _) = stats_for(&dev, &k, d, 10, 4864, None, 11);
            let (s_old, _) = stats_for(&dev, &bad, d, 10, 4864, None, 11);
            println!("  {} kernel, scale bytes by the old form: error {:.2e} of sum|wx| ({}), by the new form: {:.2e}", pack::name(d), s_old.max_gpu, if s_old.passes(pack::TOL) { "correct" } else { "WRONG" }, s_new.max_gpu);
        }
        dev.check_health().unwrap();
    }

    #[test]
    fn kernel_construction() {
        assert!(MvKernel::new(Dtype::F32, 64).is_ok());
        assert!(MvKernel::new(Dtype::F32, 48).is_err());
        assert!(MvKernel::new(Dtype::F32, 512).is_err());
        assert!(MvKernel::new(Dtype::BF16, 64).is_err());
        assert!(MvKernel::new(Dtype::Q5K, 64).is_ok() && MvKernel::new(Dtype::Q4_0, 64).is_ok());
        let k = MvKernel::new(Dtype::Q4K, 128).unwrap();
        assert!(k.wgsl.contains("const WG: u32 = 128u;") && !k.wgsl.contains("64u;\nvar<workgroup>"));
        assert_ne!(k.key, MvKernel::new(Dtype::Q4K, 64).unwrap().key);
    }

    #[test]
    fn software_adapter_matvec() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        exercise(&dev);
    }

    /// `cargo test real_adapter_matvec -- --ignored --nocapture`; adapter from `BURITI_TEST_ADAPTER` (default best).
    #[test]
    #[ignore]
    fn real_adapter_matvec() {
        let sel = std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best);
        let dev = open_or_skip(&sel, "real").expect("no adapter could be opened");
        exercise(&dev);
    }
}
