//! Elementwise and reduction ops on the GPU (task G6, D65): RMSNorm, LayerNorm, RoPE, gated SiLU / GELU, softmax,
//! residual add and the embedding gather. Each op is a WGSL kernel under `shaders/` and a host function here that
//! builds an `OpCall` (the pipeline, a small parameter buffer and the grid). The caller supplies the data buffers
//! (`io`, in the order documented on each function), so a decoder can put many calls into one compute pass with
//! `OpCall::step` and `GpuDevice::run_steps`.
//!
//! All activations are f32. No subgroup operation and no f16 shader feature is used. Reductions use a 256-thread
//! shared-memory tree, so the summation order differs from the CPU engine's: the tests compare against the CPU
//! implementation (or an exact f64 reference) with an explicit tolerance, never with equality (except the
//! residual add, which is plain IEEE addition and bit-identical).

#![allow(dead_code)]

use super::device::{GpuBuffer, GpuDevice, RunTiming, Step, DEFAULT_TIMEOUT};
use super::kernels::grid_2d;
use super::matvec::GpuMatrix;
use super::pack;
use crate::native::gguf::Dtype;
use std::sync::Arc;

/// Threads per workgroup of the row ops (a power of two: the reductions are trees).
pub const ROW_WG: u32 = 256;
const ELEM_WG: u64 = 256;
const ROPE_WG: u64 = 64;
const GATHER_WG: u64 = 64;
/// 2 pi as four pieces of 8 bits (k * piece is exact for k < 2^16) and pi/2 as two, for the angle reduction of the
/// GPU-angle RoPE (`rope_trig`).
pub const ROPE_PIECES: [f32; 6] = [6.28125, 0.00193023681640625, 0.0000050663948059082031, 3.9683741e-9, 1.5703125, 0.00048382679489];

/// The gated activations the CPU engine has (`vmath::silu_mul`, `vmath::gelu_mul`) plus the ungated GELU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    /// out = silu(g) * u (SwiGLU)
    SiluMul,
    /// out = gelu_tanh(g) * u (GeGLU, Gemma)
    GeluMul,
    /// out = gelu_tanh(g) (StarCoder2's plain MLP); `u` is bound but unused
    Gelu,
}

/// A shader with its pipeline-cache key. The tests build broken variants with `with_wgsl`.
#[derive(Clone)]
pub struct Op {
    pub key: String,
    pub wgsl: String,
}

impl Op {
    pub(crate) fn from(key: &str, src: &str) -> Op {
        Op { key: key.into(), wgsl: src.replace("\r\n", "\n") }
    }
    pub fn rmsnorm() -> Op {
        Op::from("op_rmsnorm", include_str!("shaders/rmsnorm.wgsl"))
    }
    pub fn layernorm() -> Op {
        Op::from("op_layernorm", include_str!("shaders/layernorm.wgsl"))
    }
    /// RoPE with a sin/cos table built by the host (the default; bit-identical angles to the CPU engine).
    pub fn rope() -> Op {
        Op::from("op_rope", include_str!("shaders/rope.wgsl"))
    }
    /// RoPE with the angle computed on the GPU (for a decoder that keeps the position on the GPU).
    pub fn rope_trig() -> Op {
        Op::from("op_rope_trig", include_str!("shaders/rope_trig.wgsl"))
    }
    pub fn softmax() -> Op {
        Op::from("op_softmax", include_str!("shaders/softmax.wgsl"))
    }
    pub fn add() -> Op {
        Op::from("op_add", include_str!("shaders/add.wgsl"))
    }
    pub fn act(a: Act) -> Op {
        let mode = match a {
            Act::SiluMul => 0,
            Act::GeluMul => 1,
            Act::Gelu => 2,
        };
        let src = include_str!("shaders/act.wgsl").replace("\r\n", "\n");
        assert_eq!(src.matches("const MODE: u32 = 0u;").count(), 1);
        Op { key: format!("op_act{mode}"), wgsl: src.replace("const MODE: u32 = 0u;", &format!("const MODE: u32 = {mode}u;")) }
    }
    /// The embedding gather for weights of `dtype`: the decoder section of `gather_decoders.wgsl` plus the common
    /// main, with the binding numbers that follow the format's weight buffers filled in.
    pub fn gather(dtype: Dtype) -> Result<Op, String> {
        if !pack::supported(dtype) {
            return Err(format!("{} has no GPU gather", pack::name(dtype)));
        }
        let nbuf = pack::array_row_bytes(dtype, pack::col_multiple(dtype)).len();
        let all = include_str!("shaders/gather_decoders.wgsl").replace("\r\n", "\n");
        let marker = format!("//@@ {}\n", pack::name(dtype));
        let start = all.find(&marker).ok_or_else(|| format!("no gather decoder for {}", pack::name(dtype)))? + marker.len();
        let end = all[start..].find("//@@ ").map_or(all.len(), |e| start + e);
        let main = include_str!("shaders/gather_main.wgsl")
            .replace("\r\n", "\n")
            .replace("@IDS@", &nbuf.to_string())
            .replace("@OUT@", &(nbuf + 1).to_string())
            .replace("@P@", &(nbuf + 2).to_string());
        Ok(Op { key: format!("op_gather_{}", pack::name(dtype)), wgsl: format!("{}\n{}", &all[start..end], main) })
    }
    pub fn with_wgsl(&self, tag: &str, wgsl: String) -> Op {
        Op { key: format!("{}_{tag}", self.key), wgsl }
    }
    fn pipeline(&self, dev: &GpuDevice) -> Result<Arc<wgpu::ComputePipeline>, String> {
        dev.pipeline(&self.key, &self.wgsl, "main")
    }
}

/// One dispatch ready to run: the pipeline, its parameter buffer (the last binding) and the grid.
pub struct OpCall {
    pipeline: Arc<wgpu::ComputePipeline>,
    pub params: GpuBuffer,
    pub groups: [u32; 3],
}

impl OpCall {
    /// The step for `GpuDevice::run_steps`: the data buffers in `io` order, then the parameter buffer.
    pub fn step<'a>(&'a self, io: &[&'a GpuBuffer]) -> Step<'a> {
        let mut buffers: Vec<&GpuBuffer> = io.to_vec();
        buffers.push(&self.params);
        Step { pipeline: &self.pipeline, buffers, groups: self.groups }
    }
    pub fn run(&self, dev: &GpuDevice, io: &[&GpuBuffer], repeat: u32) -> Result<RunTiming, String> {
        dev.run_steps(&[self.step(io)], repeat, DEFAULT_TIMEOUT)
    }
    /// Overwrites the parameter words (at most 16) between submissions, for ops whose parameters change every token
    /// (the position of the KV write and of the attention). The grid is not changed.
    pub fn set_params(&self, dev: &GpuDevice, params: &[u32]) -> Result<(), String> {
        if params.len() > 16 {
            return Err("an op has at most 16 parameter words".into());
        }
        let mut p: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
        p.resize(64, 0);
        dev.upload_bytes(&self.params, 0, &p)
    }
}

pub(crate) fn make_call(dev: &GpuDevice, op: &Op, params: &[u32], groups: [u32; 3]) -> Result<OpCall, String> {
    let mut p: Vec<u8> = params.iter().flat_map(|v| v.to_le_bytes()).collect();
    p.resize(64, 0);
    Ok(OpCall { pipeline: op.pipeline(dev)?, params: dev.storage_from_bytes(&p, "op.params")?, groups })
}

pub(crate) fn elem_groups(dev: &GpuDevice, n: u64, wg: u64) -> Result<[u32; 3], String> {
    if n == 0 {
        return Err("empty op".into());
    }
    grid_2d(n.div_ceil(wg), dev.caps.max_compute_workgroups_per_dimension)
}

fn check_len(what: &str, b: &GpuBuffer, floats: u64) -> Result<(), String> {
    if b.len < floats * 4 {
        return Err(format!("{what}: the buffer has {} floats, {floats} needed", b.len / 4));
    }
    Ok(())
}

/// Buffers (`io`): x (rows * n), w (n), out (rows * n). In and out must be different buffers.
pub fn rmsnorm(dev: &GpuDevice, op: &Op, n: u32, rows: u32, eps: f32) -> Result<OpCall, String> {
    if n == 0 || rows == 0 {
        return Err("rmsnorm: empty".into());
    }
    make_call(dev, op, &[n, rows, eps.to_bits()], grid_2d(rows as u64, dev.caps.max_compute_workgroups_per_dimension)?)
}

/// Buffers (`io`): x (rows * n), w (n), b (n), out (rows * n). In and out must be different buffers.
pub fn layernorm(dev: &GpuDevice, op: &Op, n: u32, rows: u32, eps: f32) -> Result<OpCall, String> {
    if n == 0 || rows == 0 {
        return Err("layernorm: empty".into());
    }
    make_call(dev, op, &[n, rows, eps.to_bits()], grid_2d(rows as u64, dev.caps.max_compute_workgroups_per_dimension)?)
}

/// RoPE with the host table, in place on `v` (n_heads * head_dim floats); buffers (`io`): v, cs (the table of
/// `rope_table`, 2 * n_rot / 2 floats). `n_rot` is the number of rotating dimensions per head (even, at most
/// `head_dim`). `interleaved`: Llama pairs (2i, 2i + 1); otherwise NeoX pairs (i, i + n_rot / 2).
pub fn rope(dev: &GpuDevice, op: &Op, n_heads: u32, head_dim: u32, n_rot: u32, interleaved: bool) -> Result<OpCall, String> {
    if n_heads == 0 || n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim {
        return Err(format!("rope: n_rot {n_rot} must be even and within head_dim {head_dim}"));
    }
    let half = n_rot / 2;
    make_call(dev, op, &[n_heads, head_dim, half, interleaved as u32], elem_groups(dev, n_heads as u64 * half as u64, ROPE_WG)?)
}

/// The sin/cos table for one position, as the CPU engine forms it: for each frequency i, `sin` and `cos` of the f32
/// angle `pos as f32 * inv_freq[i]`, both times `mscale` (longrope `attn_factor`; 1 otherwise), interleaved as
/// [sin_0, cos_0, sin_1, cos_1, ...]. Upload it (a few hundred bytes) once per token; every layer that uses the
/// same frequency set (the longrope short or long set, the Gemma local set) shares it.
pub fn rope_table(pos: u32, inv_freq: &[f32], mscale: f32) -> Vec<f32> {
    inv_freq
        .iter()
        .flat_map(|f| {
            let (s, c) = (pos as f32 * f).sin_cos();
            [s * mscale, c * mscale]
        })
        .collect()
}

/// RoPE with the angle computed on the GPU (`Op::rope_trig`). In place on `v` (n_heads * head_dim floats); buffers (`io`): v, inv_freq (n_rot / 2 floats). `n_rot` is the
/// number of rotating dimensions per head (even, at most `head_dim`). `interleaved`: Llama pairs (2i, 2i + 1);
/// otherwise NeoX pairs (i, i + n_rot / 2). `mscale` multiplies sin and cos (longrope `attn_factor`, else 1).
pub fn rope_trig(dev: &GpuDevice, op: &Op, n_heads: u32, head_dim: u32, n_rot: u32, interleaved: bool, pos: u32, mscale: f32) -> Result<OpCall, String> {
    if n_heads == 0 || n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim {
        return Err(format!("rope: n_rot {n_rot} must be even and within head_dim {head_dim}"));
    }
    let half = n_rot / 2;
    let mut p = vec![n_heads, head_dim, half, interleaved as u32, pos, mscale.to_bits()];
    p.extend(ROPE_PIECES.iter().map(|v| v.to_bits()));
    make_call(dev, op, &p, elem_groups(dev, n_heads as u64 * half as u64, ROPE_WG)?)
}

/// The RoPE frequencies the CPU engine builds (`Model::load`): theta^(-2i / n_rot), divided by the factor of
/// the set in use (Llama 3.x `rope_freqs.weight`, or the longrope short / long set; `None` = no factors).
pub fn rope_inv_freq(theta: f32, n_rot: usize, factors: Option<&[f32]>) -> Vec<f32> {
    (0..n_rot / 2)
        .map(|i| {
            let base = theta.powf(-(2.0 * i as f32) / n_rot as f32);
            base / factors.map_or(1.0, |f| f[i])
        })
        .collect()
}

/// Buffers (`io`): g, u, out (n floats each; for `Act::Gelu` pass g again as u).
pub fn act(dev: &GpuDevice, op: &Op, n: u32) -> Result<OpCall, String> {
    make_call(dev, op, &[n], elem_groups(dev, n as u64, ELEM_WG)?)
}

/// Buffers (`io`): x (rows * n), out (rows * n). In and out must be different buffers.
pub fn softmax(dev: &GpuDevice, op: &Op, n: u32, rows: u32) -> Result<OpCall, String> {
    if n == 0 || rows == 0 {
        return Err("softmax: empty".into());
    }
    make_call(dev, op, &[n, rows], grid_2d(rows as u64, dev.caps.max_compute_workgroups_per_dimension)?)
}

/// x += y in place; buffers (`io`): x, y.
pub fn add(dev: &GpuDevice, op: &Op, n: u32) -> Result<OpCall, String> {
    make_call(dev, op, &[n], elem_groups(dev, n as u64, ELEM_WG)?)
}

/// The gather of `n_ids` rows of `m`, one `OpCall` per chunk of the matrix (the chunk that holds a row writes it).
/// Buffers for each call: `gather_io(m, chunk_index, ids, out)`. `scale` multiplies every value (Gemma:
/// sqrt(n_embd); 1 otherwise). `ids` holds u32 row numbers; the caller guarantees they are below `m.rows`.
pub fn gather(dev: &GpuDevice, op: &Op, m: &GpuMatrix, n_ids: u32, scale: f32) -> Result<Vec<OpCall>, String> {
    if n_ids == 0 {
        return Err("gather: no ids".into());
    }
    let max = dev.caps.max_compute_workgroups_per_dimension;
    if n_ids > max {
        return Err(format!("gather: {n_ids} ids is above the {max} the grid allows in one dispatch"));
    }
    let nb = pack::blocks_per_row(m.dtype, m.cols as usize) as u32;
    let gx = (m.cols).div_ceil(GATHER_WG);
    if gx > max as u64 {
        return Err("gather: too many columns".into());
    }
    m.chunks
        .iter()
        .map(|c| make_call(dev, op, &[c.start as u32, c.rows as u32, m.cols as u32, nb, n_ids, scale.to_bits()], [gx as u32, n_ids, 1]))
        .collect()
}

/// The buffers of chunk `i` for a gather call: its weight arrays, then ids and out.
pub fn gather_io<'a>(m: &'a GpuMatrix, i: usize, ids: &'a GpuBuffer, out: &'a GpuBuffer) -> Vec<&'a GpuBuffer> {
    let mut v: Vec<&GpuBuffer> = m.chunks[i].bufs.iter().collect();
    v.push(ids);
    v.push(out);
    v
}

/// Gathers `ids` from `m` into a new buffer and returns it read back (`ids.len() * cols` floats).
pub fn embed(dev: &GpuDevice, op: &Op, m: &GpuMatrix, ids: &[u32], scale: f32) -> Result<Vec<f32>, String> {
    if let Some(bad) = ids.iter().find(|&&i| i as u64 >= m.rows) {
        return Err(format!("embedding id {bad} is outside the {} rows of the matrix", m.rows));
    }
    let calls = gather(dev, op, m, ids.len() as u32, scale)?;
    let ib = dev.storage_from_bytes(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), "gather.ids")?;
    let out = dev.create_storage(ids.len() as u64 * m.cols * 4, "gather.out")?;
    let steps: Vec<Step> = calls.iter().enumerate().map(|(i, c)| c.step(&gather_io(m, i, &ib, &out))).collect();
    dev.run_steps(&steps, 1, DEFAULT_TIMEOUT)?;
    dev.read_back_f32(&out, ids.len() * m.cols as usize, DEFAULT_TIMEOUT)
}

/// Convenience for tests and tools: upload `inputs`, run one call, read back `out_len` floats from buffer `out_at`
/// of the io list (inputs first, then one zeroed output buffer per entry of `outs`).
pub fn run_simple(dev: &GpuDevice, call: &OpCall, inputs: &[&[f32]], outs: &[usize]) -> Result<Vec<Vec<f32>>, String> {
    let mut bufs: Vec<GpuBuffer> = Vec::new();
    for (i, x) in inputs.iter().enumerate() {
        bufs.push(dev.storage_from_f32(x, &format!("op.in{i}"))?);
    }
    for (i, &n) in outs.iter().enumerate() {
        bufs.push(dev.create_storage(n as u64 * 4, &format!("op.out{i}"))?);
    }
    let io: Vec<&GpuBuffer> = bufs.iter().collect();
    call.run(dev, &io, 1)?;
    outs.iter().enumerate().map(|(i, &n)| dev.read_back_f32(&bufs[inputs.len() + i], n, DEFAULT_TIMEOUT)).collect()
}

/// In-place variant: the first input is also the output (RoPE, residual add).
pub fn run_inplace(dev: &GpuDevice, call: &OpCall, inputs: &[&[f32]]) -> Result<Vec<f32>, String> {
    let bufs: Vec<GpuBuffer> = inputs.iter().enumerate().map(|(i, x)| dev.storage_from_f32(x, &format!("op.in{i}"))).collect::<Result<_, _>>()?;
    let io: Vec<&GpuBuffer> = bufs.iter().collect();
    call.run(dev, &io, 1)?;
    dev.read_back_f32(&bufs[0], inputs[0].len(), DEFAULT_TIMEOUT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::matvec::upload_matrix;
    use crate::gpu::select::AdapterSelector;
    use crate::native::model::{rmsnorm as cpu_rmsnorm, rope as cpu_rope};
    use crate::native::quant;
    use crate::native::vmath::{gelu_mul, silu_mul, softmax as cpu_softmax};

    // The unit shapes of Qwen 2.5 0.5B.
    const HIDDEN: usize = 896;
    const N_HEAD: usize = 14;
    const N_KV: usize = 2;
    const HEAD_DIM: usize = 64;
    const FFN: usize = 4864;
    const QWEN_THETA: f32 = 1_000_000.0;

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

    fn data(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        pack::gen_x(n, seed).into_iter().map(|v| v * scale).collect()
    }

    /// Largest of |got - want| / (atol + rtol-free scale): the error relative to `atol + rtol * |want|`. <= 1 passes.
    fn rel_err(got: &[f32], want: &[f64], atol: f64, rtol: f64) -> f64 {
        assert_eq!(got.len(), want.len());
        let mut worst = 0f64;
        for (g, w) in got.iter().zip(want) {
            let e = (*g as f64 - w).abs() / (atol + rtol * w.abs());
            if e.is_nan() {
                return f64::INFINITY;
            }
            worst = worst.max(e);
        }
        worst
    }

    /// Largest absolute error.
    fn max_abs(got: &[f32], want: &[f64]) -> f64 {
        got.iter().zip(want).map(|(g, w)| if g.is_nan() { f64::INFINITY } else { (*g as f64 - w).abs() }).fold(0.0, f64::max)
    }

    // ---- tolerances (absolute + relative to the reference magnitude). Chosen after measuring on 940MX Vulkan/DX12, Intel DX12
    // and WARP: the measured errors are 10x or more below these, a wrong op is 1e-2 or more.
    const RMS_TOL: (f64, f64) = (1e-6, 2e-5);
    const LN_TOL: (f64, f64) = (2e-6, 5e-5);
    const ACT_TOL: (f64, f64) = (1e-6, 5e-5);
    const SOFTMAX_TOL: (f64, f64) = (1e-7, 5e-5);
    const ROPE_TOL: (f64, f64) = (1e-6, 4e-6);

    fn rmsnorm_ref(x: &[f32], w: &[f32], eps: f32, n: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for row in x.chunks(n) {
            let ss = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
            let inv = 1.0 / (ss + eps as f64).sqrt();
            out.extend(row.iter().zip(w).map(|(&v, &wv)| v as f64 * inv * wv as f64));
        }
        out
    }

    fn layernorm_ref(x: &[f32], w: &[f32], b: &[f32], eps: f32, n: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for row in x.chunks(n) {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
            let inv = 1.0 / (var + eps as f64).sqrt();
            out.extend((0..n).map(|i| (row[i] as f64 - mean) * inv * w[i] as f64 + b[i] as f64));
        }
        out
    }

    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }
    fn gelu_tanh(x: f64) -> f64 {
        0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044715 * x * x * x)).tanh())
    }

    fn softmax_ref(x: &[f32], n: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for row in x.chunks(n) {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let e: Vec<f64> = row.iter().map(|&v| (v as f64 - m).exp()).collect();
            let s: f64 = e.iter().sum();
            out.extend(e.iter().map(|v| v / s));
        }
        out
    }

    /// RoPE reference in f64 from the same f32 angle the CPU engine forms (pos as f32 * inv_freq): the question is
    /// only whether sin and cos of that angle are right on the GPU.
    fn rope_ref(v: &[f32], n_heads: usize, head_dim: usize, inv: &[f32], interleaved: bool, pos: u32, ms: f32) -> Vec<f64> {
        let half = inv.len();
        let mut out: Vec<f64> = v.iter().map(|&a| a as f64).collect();
        for h in 0..n_heads {
            for i in 0..half {
                let ang = (pos as f32 * inv[i]) as f64;
                let (s, c) = (ang.sin() * ms as f64, ang.cos() * ms as f64);
                let (ia, ib) = if interleaved { (h * head_dim + 2 * i, h * head_dim + 2 * i + 1) } else { (h * head_dim + i, h * head_dim + i + half) };
                let (a, b) = (v[ia] as f64, v[ib] as f64);
                out[ia] = a * c - b * s;
                out[ib] = a * s + b * c;
            }
        }
        out
    }

    // ------------------------------------------------------------------ checks per op: (error in tolerance units, plain max error)

    fn t_rmsnorm(dev: &GpuDevice, op: &Op, n: usize, rows: usize, eps: f32, seed: u64) -> (f64, f64) {
        let x = data(n * rows, seed, 3.0);
        let w = data(n, seed + 1, 2.0);
        let call = rmsnorm(dev, op, n as u32, rows as u32, eps).unwrap();
        let got = run_simple(dev, &call, &[&x, &w], &[n * rows]).unwrap().remove(0);
        let want = rmsnorm_ref(&x, &w, eps, n);
        // the CPU engine's own function must agree with the f64 reference too (it is what the GPU stands in for)
        let mut cpu = vec![0f32; n];
        cpu_rmsnorm(&mut cpu, &x[..n], &w, eps);
        assert!(rel_err(&cpu, &want[..n], RMS_TOL.0, RMS_TOL.1) <= 1.0, "the CPU rmsnorm disagrees with the reference");
        (rel_err(&got, &want, RMS_TOL.0, RMS_TOL.1), max_abs(&got, &want))
    }

    fn t_layernorm(dev: &GpuDevice, op: &Op, n: usize, rows: usize, eps: f32, seed: u64) -> (f64, f64) {
        let x: Vec<f32> = data(n * rows, seed, 3.0).iter().map(|v| v + 1.5).collect(); // a mean away from zero
        let w = data(n, seed + 1, 2.0);
        let b = data(n, seed + 2, 1.0);
        let call = layernorm(dev, op, n as u32, rows as u32, eps).unwrap();
        let got = run_simple(dev, &call, &[&x, &w, &b], &[n * rows]).unwrap().remove(0);
        let want = layernorm_ref(&x, &w, &b, eps, n);
        let mut cpu = vec![0f32; n];
        crate::native::model::starcoder2::layernorm(&mut cpu, &x[..n], &w, &b, eps);
        assert!(rel_err(&cpu, &want[..n], LN_TOL.0, LN_TOL.1) <= 1.0, "the CPU layernorm disagrees with the reference");
        (rel_err(&got, &want, LN_TOL.0, LN_TOL.1), max_abs(&got, &want))
    }

    #[allow(clippy::too_many_arguments)]
    fn t_rope_trig(dev: &GpuDevice, op: &Op, n_heads: usize, head_dim: usize, n_rot: usize, interleaved: bool, pos: u32, ms: f32, inv: &[f32], seed: u64) -> (f64, f64) {
        let v = data(n_heads * head_dim, seed, 2.0);
        let call = rope_trig(dev, op, n_heads as u32, head_dim as u32, n_rot as u32, interleaved, pos, ms).unwrap();
        let got = run_inplace(dev, &call, &[&v, inv]).unwrap();
        // The CPU engine's own angle is an f32 product, so it is only as good as the f32 rounding of the angle
        // (ulp / 2 = 6e-8 * angle). The GPU-angle kernel is allowed 1.5e-7 * angle on top of the base tolerance: the
        // Intel DX12 driver reassociates the range reduction and lands at 1.07 of 1e-7 * angle (measured, D65); a wrong
        // kernel is off by 1e-2 or more.
        let max_angle = pos as f64 * inv.iter().cloned().fold(0f32, f32::max) as f64;
        let atol = ROPE_TOL.0 + 1.5e-7 * max_angle;
        let want = rope_ref(&v, n_heads, head_dim, inv, interleaved, pos, ms);
        let mut cpu = v.clone();
        for h in 0..n_heads {
            cpu_rope(&mut cpu[h * head_dim..(h + 1) * head_dim], pos as usize, inv, interleaved, ms);
        }
        assert!(rel_err(&cpu, &want, ROPE_TOL.0, ROPE_TOL.1) <= 1.0, "the CPU rope disagrees with the reference (pos {pos})");
        (rel_err(&got, &want, atol, ROPE_TOL.1), max_abs(&got, &want))
    }

    #[allow(clippy::too_many_arguments)]
    fn t_rope(dev: &GpuDevice, op: &Op, n_heads: usize, head_dim: usize, n_rot: usize, interleaved: bool, pos: u32, ms: f32, inv: &[f32], seed: u64) -> (f64, f64) {
        let v = data(n_heads * head_dim, seed, 2.0);
        let call = rope(dev, op, n_heads as u32, head_dim as u32, n_rot as u32, interleaved).unwrap();
        let got = run_inplace(dev, &call, &[&v, &rope_table(pos, inv, ms)]).unwrap();
        let want = rope_ref(&v, n_heads, head_dim, inv, interleaved, pos, ms);
        // the CPU engine's rope on the same data
        let mut cpu = v.clone();
        for h in 0..n_heads {
            cpu_rope(&mut cpu[h * head_dim..(h + 1) * head_dim], pos as usize, inv, interleaved, ms);
        }
        assert!(rel_err(&cpu, &want, ROPE_TOL.0, ROPE_TOL.1) <= 1.0, "the CPU rope disagrees with the reference (pos {pos})");
        (rel_err(&got, &want, ROPE_TOL.0, ROPE_TOL.1), max_abs(&got, &want))
    }

    fn t_act(dev: &GpuDevice, a: Act, op: &Op, n: usize, seed: u64) -> (f64, f64) {
        let g = data(n, seed, 12.0); // |g| up to 12: both tails of the sigmoid
        let u = data(n, seed + 1, 2.0);
        let call = act(dev, op, n as u32).unwrap();
        let got = run_simple(dev, &call, &[&g, &u], &[n]).unwrap().remove(0);
        let want: Vec<f64> = g.iter().zip(&u).map(|(&x, &uv)| match a {
            Act::SiluMul => silu(x as f64) * uv as f64,
            Act::GeluMul => gelu_tanh(x as f64) * uv as f64,
            Act::Gelu => gelu_tanh(x as f64),
        }).collect();
        // the CPU engine's function too (its exp is an approximation: looser bound)
        let mut cpu = g.clone();
        match a {
            Act::SiluMul => silu_mul(&mut cpu, &u),
            Act::GeluMul => gelu_mul(&mut cpu, &u),
            Act::Gelu => gelu_mul(&mut cpu, &vec![1.0; n]),
        }
        assert!(rel_err(&cpu, &want, 4e-6, 4e-4) <= 1.0, "the CPU {a:?} disagrees with the reference");
        (rel_err(&got, &want, ACT_TOL.0, ACT_TOL.1), max_abs(&got, &want))
    }

    fn t_softmax(dev: &GpuDevice, op: &Op, n: usize, rows: usize, seed: u64, big: bool, spike: Option<usize>) -> (f64, f64) {
        // `big`: values around 80..120, where exp() overflows f32 unless the maximum is subtracted first
        let mut x: Vec<f32> = data(n * rows, seed, 6.0).iter().map(|v| if big { v + 100.0 } else { *v }).collect();
        if let Some(at) = spike {
            // one value far above the rest in every row (+120): a missing maximum overflows, a maximum that misses it too
            for r in 0..rows {
                x[r * n + at] = 226.0;
            }
        }
        let call = softmax(dev, op, n as u32, rows as u32).unwrap();
        let got = run_simple(dev, &call, &[&x], &[n * rows]).unwrap().remove(0);
        let want = softmax_ref(&x, n);
        let mut cpu = x[..n].to_vec();
        cpu_softmax(&mut cpu);
        assert!(rel_err(&cpu, &want[..n], 1e-6, 1e-4) <= 1.0, "the CPU softmax disagrees with the reference");
        // every row must also sum to 1 (a NaN fails)
        let sums = got.chunks(n).map(|r| { let s: f64 = r.iter().map(|&v| v as f64).sum(); if s.is_nan() { f64::INFINITY } else { (s - 1.0).abs() / 1e-4 } }).fold(0.0, f64::max);
        (rel_err(&got, &want, SOFTMAX_TOL.0, SOFTMAX_TOL.1).max(sums), max_abs(&got, &want))
    }

    /// Reports the worst error seen by a battery; asserts the tolerance.
    struct Worst(String, f64, f64);
    impl Worst {
        fn new(name: &str) -> Worst {
            Worst(name.into(), 0.0, 0.0)
        }
        fn take(&mut self, what: &str, r: (f64, f64)) {
            assert!(r.0 <= 1.0, "{} {what}: error {:.3e} (tolerance units; 1.0 = limit), max abs {:.3e}", self.0, r.0, r.1);
            self.1 = self.1.max(r.0);
            self.2 = self.2.max(r.1);
        }
        fn print(&self) {
            println!("  {}: worst error {:.2} of the tolerance, max abs {:.2e}", self.0, self.1, self.2);
        }
    }

    /// The planted bugs of an op: each must push the error above the tolerance on the bug shape.
    fn expect_caught(what: &str, r: (f64, f64)) {
        assert!(r.0 > 1.0, "planted bug '{what}' was NOT detected (error {:.3e} of the tolerance)", r.0);
    }

    fn exercise(dev: &GpuDevice) {
        // ---------------- RMSNorm
        let op = Op::rmsnorm();
        let mut w = Worst::new("rmsnorm");
        for (n, rows) in [(1usize, 1usize), (255, 1), (HIDDEN, 1), (257, 3), (HEAD_DIM, N_HEAD + N_KV), (4864, 2), (6144, 1)] {
            w.take(&format!("{n}x{rows}"), t_rmsnorm(dev, &op, n, rows, 1e-6, n as u64));
        }
        w.take("big eps", t_rmsnorm(dev, &op, 896, 1, 1e-2, 5));
        w.print();
        assert!(t_rmsnorm(dev, &op, 896, 4, 0.5, 3).0 <= 1.0, "the unbroken rmsnorm must pass the bug shape");
        for (what, from, to) in [
            ("epsilon ignored", "sqrt(ss + bitcast<f32>(p[2]))", "sqrt(ss)"),
            ("mean over n + 1", "scratch[0] / f32(n)", "scratch[0] / f32(n + 1u)"),
            ("weight index shifted", "* inv * w[i];", "* inv * w[i + 1u];"),
            ("weight dropped", "x[base + i] * inv * w[i]", "x[base + i] * inv * (w[i] * 0.0 + 1.0)"),
            ("row stride off by one", "let base = row * n;", "let base = row * n + 1u;"),
            ("reduction starts too low", "var s = 128u;", "var s = 64u;"),
            ("one square term lost per thread", "acc += v * v;", "acc += v * v * select(1.0, 0.0, i >= 256u);"),
        ] {
            let bad = op.with_wgsl("bug", replace_once(&op.wgsl, from, to));
            expect_caught(what, t_rmsnorm(dev, &bad, 896, 4, 0.5, 3)); // 896 values per row: every one of the 256 threads has work
        }

        // ---------------- LayerNorm
        let op = Op::layernorm();
        let mut w = Worst::new("layernorm");
        for (n, rows) in [(1usize, 1usize), (255, 1), (HIDDEN, 1), (257, 3), (6144, 1), (4864, 2)] {
            w.take(&format!("{n}x{rows}"), t_layernorm(dev, &op, n, rows, 1e-5, n as u64));
        }
        w.print();
        assert!(t_layernorm(dev, &op, 896, 4, 0.5, 3).0 <= 1.0, "the unbroken layernorm must pass the bug shape");
        for (what, from, to) in [
            ("mean not subtracted", "let d = x[base + i] - mean;", "let d = x[base + i];"),
            ("mean not subtracted in the output", "out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];", "out[base + i] = x[base + i] * inv * w[i] + b[i];"),
            ("bias dropped", "* inv * w[i] + b[i];", "* inv * w[i] + b[i] * 0.0;"),
            ("bias index shifted", "+ b[i];", "+ b[i + 1u];"),
            ("variance over n - 1", "block_sum(acc2, lid.x) / f32(n)", "block_sum(acc2, lid.x) / f32(n - 1u)"),
            ("epsilon ignored", "sqrt(variance + bitcast<f32>(p[2]))", "sqrt(variance)"),
            ("reduction starts too low", "var s = 128u;", "var s = 64u;"),
            ("row stride off by one", "let base = row * n;", "let base = row * n + 1u;"),
        ] {
            let bad = op.with_wgsl("bug", replace_once(&op.wgsl, from, to));
            expect_caught(what, t_layernorm(dev, &bad, 896, 4, 0.5, 3));
        }

        // ---------------- RoPE with the host table (default): Qwen/Llama standard (NeoX pairs for Qwen2, interleaved for Llama), partial rotary, longrope
        let op = Op::rope();
        let mut w = Worst::new("rope (table)");
        let inv_full = rope_inv_freq(QWEN_THETA, HEAD_DIM, None);
        let positions = [0u32, 1, 2, 17, 255, 1023, 4095, 32767, 300_000];
        for pos in positions {
            w.take(&format!("q 14x64 neox pos {pos}"), t_rope(dev, &op, N_HEAD, HEAD_DIM, HEAD_DIM, false, pos, 1.0, &inv_full, pos as u64));
            w.take(&format!("k 2x64 neox pos {pos}"), t_rope(dev, &op, N_KV, HEAD_DIM, HEAD_DIM, false, pos, 1.0, &inv_full, pos as u64 + 1));
        }
        w.take("interleaved (Llama) pos 700", t_rope(dev, &op, 32, 128, 128, true, 700, 1.0, &rope_inv_freq(500_000.0, 128, None), 3));
        // partial rotary: only the first 32 of 64 dimensions rotate (Phi-style): the rest must stay untouched
        let inv_part = rope_inv_freq(10_000.0, 32, None);
        for pos in [5u32, 3000] {
            w.take("partial 32 of 64 neox", t_rope(dev, &op, 7, 64, 32, false, pos, 1.0, &inv_part, 9));
            w.take("partial 32 of 64 interleaved", t_rope(dev, &op, 7, 64, 32, true, pos, 1.0, &inv_part, 10));
        }
        // longrope (Phi-3.5, Phi-4-mini): the short and the long factor sets, and the attention scaling on sin and cos
        let short: Vec<f32> = (0..48).map(|i| 1.0 + 0.05 * i as f32).collect();
        let long: Vec<f32> = (0..48).map(|i| 2.0 + 0.5 * i as f32).collect();
        let (inv_short, inv_long) = (rope_inv_freq(10_000.0, 96, Some(&short)), rope_inv_freq(10_000.0, 96, Some(&long)));
        assert_ne!(inv_short, inv_long);
        for (set, inv) in [("short", &inv_short), ("long", &inv_long)] {
            for pos in [1u32, 4000, 60000] {
                w.take(&format!("longrope {set} pos {pos} mscale 1.19"), t_rope(dev, &op, 32, 96, 96, false, pos, 1.19, inv, 11));
            }
        }
        w.print();
        // the bug shape: 14 heads x 64, a late position, NeoX pairs (or interleaved, or mscale != 1 for the bugs that need it)
        let bug = |op: &Op| t_rope(dev, op, N_HEAD, HEAD_DIM, HEAD_DIM, false, 1000, 1.0, &inv_full, 7);
        let bug_il = |op: &Op| t_rope(dev, op, N_HEAD, HEAD_DIM, HEAD_DIM, true, 1000, 1.0, &inv_full, 7);
        let bug_ms = |op: &Op| t_rope(dev, op, N_HEAD, HEAD_DIM, HEAD_DIM, false, 1000, 1.19, &inv_full, 7);
        assert!(bug(&op).0 <= 1.0 && bug_il(&op).0 <= 1.0 && bug_ms(&op).0 <= 1.0, "the unbroken rope must pass the bug shapes");
        for (what, from, to, kind) in [
            ("pairs: neox partner off by one", "var ib = base + i + hp;", "var ib = base + i + hp + 1u;", 0),
            ("sin sign flipped", "v[ia] = a * co - b * sn;", "v[ia] = a * co + b * sn;", 0),
            ("cos and sin swapped in the second element", "v[ib] = a * sn + b * co;", "v[ib] = a * co + b * sn;", 0),
            ("head stride wrong", "let base = h * p[1];", "let base = h * p[2];", 0),
            ("table index off by one", "let sn = cs[2u * i];", "let sn = cs[2u * i + 2u];", 0),
            ("sin and cos slots swapped", "let co = cs[2u * i + 1u];", "let co = cs[2u * i];", 0),
            ("table of the wrong position half used", "let co = cs[2u * i + 1u];", "let co = cs[2u * i + 1u + 2u];", 0),
            ("interleaved flag ignored", "if (p[3] == 1u) {", "if (p[3] == 7u) {", 1),
            ("interleaved: odd element wrong", "ib = base + 2u * i + 1u;", "ib = base + 2u * i;", 1),
            ("interleaved: even element wrong", "ia = base + 2u * i;", "ia = base + 2u * i + 1u;", 1),
        ] {
            let bad = op.with_wgsl("bug", replace_once(&op.wgsl, from, to));
            expect_caught(what, if kind == 1 { bug_il(&bad) } else { bug(&bad) });
        }
        // the mscale lives in the table (host side): a table built without it must fail the mscale shape
        {
            let inv = &inv_full;
            let v = data(N_HEAD * HEAD_DIM, 7, 2.0);
            let call = rope(dev, &op, N_HEAD as u32, HEAD_DIM as u32, HEAD_DIM as u32, false).unwrap();
            let no_scale = rope_table(1000, inv, 1.0);
            let got = run_inplace(dev, &call, &[&v, &no_scale]).unwrap();
            let want = rope_ref(&v, N_HEAD, HEAD_DIM, inv, false, 1000, 1.19);
            expect_caught("table built without mscale", (rel_err(&got, &want, ROPE_TOL.0, ROPE_TOL.1), 0.0));
        }

        // ---------------- RoPE with the angle computed on the GPU (variant B, for a decoder that keeps the position on the GPU)
        let op = Op::rope_trig();
        let mut w = Worst::new("rope (GPU angle)");
        for pos in [0u32, 1, 17, 255, 1023, 4095, 32767, 300_000] {
            w.take(&format!("q 14x64 neox pos {pos}"), t_rope_trig(dev, &op, N_HEAD, HEAD_DIM, HEAD_DIM, false, pos, 1.0, &inv_full, pos as u64));
        }
        w.take("interleaved (Llama) pos 700", t_rope_trig(dev, &op, 32, 128, 128, true, 700, 1.0, &rope_inv_freq(500_000.0, 128, None), 3));
        w.take("partial 32 of 64 neox pos 3000", t_rope_trig(dev, &op, 7, 64, 32, false, 3000, 1.0, &inv_part, 9));
        for (set, inv) in [("short", &inv_short), ("long", &inv_long)] {
            w.take(&format!("longrope {set} pos 4000 mscale 1.19"), t_rope_trig(dev, &op, 32, 96, 96, false, 4000, 1.19, inv, 11));
        }
        w.print();
        let tbug = |op: &Op| t_rope_trig(dev, op, N_HEAD, HEAD_DIM, HEAD_DIM, false, 1000, 1.0, &inv_full, 7);
        let tbug_il = |op: &Op| t_rope_trig(dev, op, N_HEAD, HEAD_DIM, HEAD_DIM, true, 1000, 1.0, &inv_full, 7);
        let tbug_ms = |op: &Op| t_rope_trig(dev, op, N_HEAD, HEAD_DIM, HEAD_DIM, false, 1000, 1.19, &inv_full, 7);
        let tbug_far = |op: &Op| t_rope_trig(dev, op, N_HEAD, HEAD_DIM, HEAD_DIM, false, 300_000, 1.0, &inv_full, 7);
        assert!(tbug(&op).0 <= 1.0 && tbug_il(&op).0 <= 1.0 && tbug_ms(&op).0 <= 1.0 && tbug_far(&op).0 <= 1.0, "the unbroken GPU-angle rope must pass the bug shapes");
        for (what, from, to, kind) in [
            ("pairs: neox partner off by one", "var ib = base + i + hp;", "var ib = base + i + hp + 1u;", 0),
            ("sin sign flipped", "v[ia] = a * cs - b * sn;", "v[ia] = a * cs + b * sn;", 0),
            ("cos and sin swapped", "v[ib] = a * sn + b * cs;", "v[ib] = a * cs + b * sn;", 0),
            ("position off by one", "f32(p[4]) * inv_freq[i]", "f32(p[4] + 1u) * inv_freq[i]", 0),
            ("frequency index off by one", "f32(p[4]) * inv_freq[i]", "f32(p[4]) * inv_freq[(i + 1u) % hp]", 0),
            ("head stride wrong", "let base = h * p[1];", "let base = h * p[2];", 0),
            ("mscale ignored on sin", "sn = sn * ms;", "", 2),
            ("mscale ignored on cos", "cs = cs * ms;", "", 2),
            ("range reduction: wrong first piece", "var r = angle - k * bitcast<f32>(p[6]);", "var r = angle - k * bitcast<f32>(p[7]);", 0),
            ("range reduction second piece dropped", "r = r - k * bitcast<f32>(p[7]);", "", 0),
            ("pi/2 piece wrong", "(r - qd * bitcast<f32>(p[10]))", "(r - qd * bitcast<f32>(p[11]))", 0),
            ("quadrant 1 wrong", "sn = cx;\n        cs = -sx;", "sn = cx;\n        cs = sx;", 0),
            ("quadrant 2 wrong", "sn = -sx;\n        cs = -cx;", "sn = sx;\n        cs = -cx;", 0),
            ("quadrant 3 wrong", "sn = -cx;\n        cs = sx;", "sn = cx;\n        cs = sx;", 0),
            ("sin polynomial coefficient wrong", "-0.16666667 + x2", "-0.16 + x2", 0),
            ("cos polynomial coefficient wrong", "x2 * (-0.5 + x2", "x2 * (-0.4 + x2", 0),
            ("interleaved flag ignored", "if (p[3] == 1u) {", "if (p[3] == 7u) {", 1),
            ("no range reduction modulo 2 pi (angles near 3e5)", "let k = round(angle * 0.15915494);", "let k = 0.0;", 3),
        ] {
            let bad = op.with_wgsl("bug", replace_once(&op.wgsl, from, to));
            let r = match kind {
                1 => tbug_il(&bad),
                2 => tbug_ms(&bad),
                3 => tbug_far(&bad),
                _ => tbug(&bad),
            };
            if kind == 3 {
                // informational: with a fused multiply-add the single pi/2 reduction is already exact on some adapters, so this is not a gate
                println!("  rope (GPU angle) \"{what}\": error {:.2} of the tolerance ({})", r.0, if r.0 > 1.0 { "visible on this adapter" } else { "not visible on this adapter" });
                continue;
            }
            expect_caught(what, r);
        }

        // ---------------- activations
        for (a, name) in [(Act::SiluMul, "silu*mul"), (Act::GeluMul, "gelu*mul"), (Act::Gelu, "gelu")] {
            let op = Op::act(a);
            let mut w = Worst::new(name);
            for n in [1usize, 255, 256, 257, FFN, 5000, 100_003, 1_000_003] {
                w.take(&format!("n {n}"), t_act(dev, a, &op, n, n as u64));
            }
            w.print();
        }
        for (a, what, from, to) in [
            (Act::SiluMul, "silu: exp sign", "a = x / (1.0 + exp(-x));", "a = x / (1.0 + exp(x));"),
            (Act::SiluMul, "gate not multiplied", "out[i] = a * uv;", "out[i] = a;"),
            (Act::SiluMul, "wrong gate index", "let uv = u[i];", "let uv = u[i + 1u];"),
            (Act::GeluMul, "gelu: cubic coefficient", "0.044715 * x * x * x", "0.04 * x * x * x"),
            (Act::GeluMul, "gelu: sqrt(2/pi) constant", "const C: f32 = 0.79788456;", "const C: f32 = 0.7978;"),
            (Act::GeluMul, "gelu: the fast sigmoid form", "-2.0 * C * (x + 0.044715 * x * x * x)", "-1.702 * x"),
            (Act::Gelu, "gelu alone multiplied by the gate", "out[i] = a;", "out[i] = a * uv;"),
        ] {
            let op = Op::act(a);
            let bad = op.with_wgsl("bug", replace_once(&op.wgsl, from, to));
            expect_caught(what, t_act(dev, a, &bad, FFN, 3));
        }

        // ---------------- softmax
        let op = Op::softmax();
        let mut w = Worst::new("softmax");
        for (n, rows, big) in [(1usize, 1usize, false), (37, N_HEAD, false), (255, 3, false), (1024, N_HEAD, false), (2055, 2, false), (300, 3, true), (1031, 5, true)] {
            w.take(&format!("{n}x{rows} big {big}"), t_softmax(dev, &op, n, rows, n as u64, big, None));
        }
        for at in [0usize, 300, 1030] {
            w.take(&format!("spike at {at}"), t_softmax(dev, &op, 1031, 3, 9, true, Some(at)));
        }
        w.print();
        for (what, from, to) in [
            ("maximum not subtracted (overflow)", "let e = exp(x[base + i] - mx);", "let e = exp(x[base + i]);"),
            ("not normalized", "out[base + i] = out[base + i] * inv;", "out[base + i] = out[base + i];"),
            ("sum reduction starts too low", "s = 128u;\n    loop {\n        if (s == 0u) { break; }\n        if (lid.x < s) {\n            scratch[lid.x] += scratch[lid.x + s];", "s = 64u;\n    loop {\n        if (s == 0u) { break; }\n        if (lid.x < s) {\n            scratch[lid.x] += scratch[lid.x + s];"),
            ("max over a prefix only", "m = max(m, x[base + i]);\n        i += 256u;", "m = max(m, x[base + i]);\n        i += 512u;"),
            ("row stride off by one", "let base = row * n;", "let base = row * n + 1u;"),
            ("exp of the wrong element", "out[base + i] = e;", "out[base + (i + 1u) % n] = e;"),
        ] {
            let bad = op.with_wgsl("bug", replace_once(&op.wgsl, from, to));
            // two data sets: plain values (normalization, indexing, sums) and a row with one huge value (a missing or partial maximum overflows)
            expect_caught(what, (t_softmax(dev, &bad, 1031, 4, 5, true, None).0.max(t_softmax(dev, &bad, 1031, 4, 5, true, Some(300)).0), 0.0));
        }

        // ---------------- residual add: bit-identical
        let op = Op::add();
        for n in [1usize, 255, 256, 257, HIDDEN, FFN, 1_000_003] {
            let (a, b) = (data(n, 1, 5.0), data(n, 2, 5.0));
            let call = add(dev, &op, n as u32).unwrap();
            let got = run_inplace(dev, &call, &[&a, &b]).unwrap();
            for i in 0..n {
                assert_eq!(got[i].to_bits(), (a[i] + b[i]).to_bits(), "add n={n} element {i}");
            }
        }
        for (what, from, to) in [("wrong index on y", "x[i] = x[i] + y[i];", "x[i] = x[i] + y[i + 1u];"), ("y subtracted", "x[i] = x[i] + y[i];", "x[i] = x[i] - y[i];"), ("last element skipped", "if (i >= p[0]) { return; }", "if (i + 1u >= p[0]) { return; }")] {
            let bad = op.with_wgsl("bug", replace_once(&op.wgsl, from, to));
            let (a, b) = (data(HIDDEN, 1, 5.0), data(HIDDEN, 2, 5.0));
            let call = add(dev, &bad, HIDDEN as u32).unwrap();
            let got = run_inplace(dev, &call, &[&a, &b]).unwrap();
            assert!((0..HIDDEN).any(|i| got[i].to_bits() != (a[i] + b[i]).to_bits()), "planted bug '{what}' was NOT detected");
        }

        // ---------------- gather
        gather_battery(dev);

        // ---------------- errors, not panics
        assert!(rope(dev, &Op::rope(), 4, 64, 66, false).is_err(), "n_rot above head_dim");
        assert!(rope(dev, &Op::rope(), 4, 64, 31, false).is_err(), "odd n_rot");
        assert!(rope_trig(dev, &Op::rope_trig(), 4, 64, 66, false, 0, 1.0).is_err());
        assert!(rmsnorm(dev, &Op::rmsnorm(), 0, 1, 1e-6).is_err());
        assert!(Op::gather(Dtype::BF16).is_err());
        dev.check_health().expect("healthy at the end");
    }

    /// Row-aligned chunks, every format, ids at the chunk edges and repeated, the Gemma scale.
    fn gather_battery(dev: &GpuDevice) {
        let mut worst_all = 0f64;
        for d in pack::ALL_FORMATS {
            let cols: usize = if pack::col_multiple(d) == 256 { 768 } else { 288 }; // odd block counts: 3 blocks of 256, 9 of 32
            let rows = 37u64;
            let src = move |s: u64, n: u64| pack::gen_rows(d, s, n, cols, 31);
            let rb = pack::array_row_bytes(d, cols).into_iter().max().unwrap();
            let m = upload_matrix(dev, d, rows, cols as u64, &src, Some(rb * 5)).expect("upload"); // chunks of 5 rows: 8 chunks
            assert_eq!(m.chunks.len(), 8, "{}", pack::name(d));
            let raw = src(0, rows);
            let nrb = pack::native_row_bytes(d, cols) as usize;
            let ids: Vec<u32> = vec![0, 36, 5, 5, 4, 9, 10, 17, 35, 1, 36, 0, 22];
            let k = Op::gather(d).unwrap();
            for scale in [1.0f32, 29.9333] {
                let got = embed(dev, &k, &m, &ids, scale).unwrap();
                let mut worst = 0f64;
                for (t, &id) in ids.iter().enumerate() {
                    let mut want = vec![0f32; cols];
                    quant::dequant_row(d, &raw[id as usize * nrb..(id as usize + 1) * nrb], &mut want);
                    let rowmax = want.iter().fold(0f32, |a, v| a.max(v.abs())) as f64 * scale as f64;
                    for c in 0..cols {
                        let e = (got[t * cols + c] as f64 - want[c] as f64 * scale as f64).abs();
                        worst = worst.max(e / rowmax.max(1e-6));
                        assert!(e <= 3e-6 * rowmax, "{} token {t} (row {id}) column {c}: GPU {} vs CPU decoder {} * {scale}", pack::name(d), got[t * cols + c], want[c]);
                    }
                }
                worst_all = worst_all.max(worst);
            }
            // planted bugs for this format's decoder, and the common ones
            let one = |op: &Op| -> bool {
                let got = embed(dev, op, &m, &ids, 2.0).unwrap();
                let mut ok = true;
                for (t, &id) in ids.iter().enumerate() {
                    let mut want = vec![0f32; cols];
                    quant::dequant_row(d, &raw[id as usize * nrb..(id as usize + 1) * nrb], &mut want);
                    let rowmax = want.iter().fold(0f32, |a, v| a.max(v.abs())) as f64 * 2.0;
                    for c in 0..cols {
                        if (got[t * cols + c] as f64 - want[c] as f64 * 2.0).abs() > 3e-6 * rowmax {
                            ok = false;
                        }
                    }
                }
                ok
            };
            assert!(one(&k), "{}: the unbroken gather must pass the bug-shape check", pack::name(d));
            let mut bugs: Vec<(&str, &str, &str)> = vec![
                ("scale ignored", "* bitcast<f32>(p[5])", ""),
                ("output token stride wrong", "out[t * cols + c] =", "out[t * cols + c + 1u] ="),
                ("chunk start ignored", "weight(id - p[0], c)", "weight(id, c)"),
                ("row off by one", "weight(id - p[0], c)", "weight(id - p[0] + 1u, c)"),
            ];
            bugs.extend(match d {
                Dtype::F32 => vec![("column off by one", "w[lrow * p[2] + c]", "w[lrow * p[2] + c + 1u]")],
                Dtype::F16 => vec![("halves swapped", "select(v.x, v.y, (idx & 1u) == 1u)", "select(v.y, v.x, (idx & 1u) == 1u)")],
                Dtype::Q8_0 => vec![("sign lost", "let q = sbyte_of(qs[at >> 2u], at & 3u);", "let q = i32(byte_of(qs[at >> 2u], at & 3u));"), ("scale pair swapped", "(b & 1u) == 1u);", "(b & 1u) == 0u);")],
                Dtype::Q4_0 => vec![("nibbles swapped", "select(qb & 15u, qb >> 4u, j >= 16u)", "select(qb >> 4u, qb & 15u, j >= 16u)"), ("offset 8", "f32(i32(nib) - 8) * d", "f32(i32(nib) - 7) * d")],
                Dtype::Q4_1 => vec![("nibbles swapped", "select(qb & 15u, qb >> 4u, j >= 16u)", "select(qb >> 4u, qb & 15u, j >= 16u)"), ("min dropped", "f32(nib) * s.x + s.y", "f32(nib) * s.x")],
                Dtype::Q5_0 => vec![("nibbles swapped", "select(qb & 15u, qb >> 4u, j >= 16u)", "select(qb >> 4u, qb & 15u, j >= 16u)"), ("5th bit of the high half from the low bits", "((qh[lrow * nb + b] >> j) & 1u) << 4u", "((qh[lrow * nb + b] >> (j & 15u)) & 1u) << 4u"), ("offset 16", "- 16) * d", "- 15) * d")],
                Dtype::Q5_1 => vec![("5th bit of the high half from the low bits", "((qh[lrow * nb + b] >> j) & 1u) << 4u", "((qh[lrow * nb + b] >> (j & 15u)) & 1u) << 4u"), ("min dropped", "f32(nib | hi5) * s.x + s.y", "f32(nib | hi5) * s.x")],
                Dtype::Q4K => vec![("nibbles swapped", "select(qb & 15u, qb >> 4u, (g & 1u) == 1u)", "select(qb >> 4u, qb & 15u, (g & 1u) == 1u)"), ("min added instead of subtracted", "- dd.y * f32(sm.y)", "+ dd.y * f32(sm.y)")],
                Dtype::Q5K => vec![("5th bit from the wrong group", "((qhb >> g) & 1u) << 4u", "((qhb >> (g >> 1u)) & 1u) << 4u"), ("nibbles swapped", "select(qb & 15u, qb >> 4u, (g & 1u) == 1u)", "select(qb >> 4u, qb & 15u, (g & 1u) == 1u)")],
                Dtype::Q6K => vec![("high bits of the quant", "(qhb >> (quarter * 2u)) & 3u", "(qhb >> quarter) & 3u"), ("scale index", "let si = n * 8u + (l >> 4u) + quarter * 2u;", "let si = n * 8u + (l >> 4u) + quarter;")],
                _ => vec![],
            });
            for (i, (what, from, to)) in bugs.into_iter().enumerate() {
                let bad = k.with_wgsl(&format!("bug{i}"), replace_once(&k.wgsl, from, to));
                assert!(!one(&bad), "{}: planted bug '{what}' was NOT detected", pack::name(d));
            }
        }
        println!("  gather: all {} formats match the CPU decoder, worst error {:.2e} of the row magnitude", pack::ALL_FORMATS.len(), worst_all);
        // ids outside the matrix are an error
        let m = upload_matrix(dev, Dtype::F32, 4, 8, &|s, n| pack::gen_rows(Dtype::F32, s, n, 8, 1), None).unwrap();
        assert!(embed(dev, &Op::gather(Dtype::F32).unwrap(), &m, &[4], 1.0).is_err());
    }

    /// The cross-op unit shapes of Qwen 2.5 0.5B (hidden 896, 14 heads, 2 KV heads, head_dim 64, FFN 4864) chained
    /// through GPU buffers the way a decoder layer's elementwise part runs, each stage checked against the CPU
    /// engine's own functions run on the previous stage's GPU output (so errors do not pile up in the comparison).
    fn qwen_chain(dev: &GpuDevice) {
        let (rms, rp, ac, sm, ad) = (Op::rmsnorm(), Op::rope(), Op::act(Act::SiluMul), Op::softmax(), Op::add());
        let emb_op = Op::gather(Dtype::Q8_0).unwrap();
        let vocab = 1000u64;
        let src = move |s: u64, n: u64| pack::gen_rows(Dtype::Q8_0, s, n, HIDDEN, 77);
        let m = upload_matrix(dev, Dtype::Q8_0, vocab, HIDDEN as u64, &src, Some(256 * 1024)).unwrap();
        assert!(m.chunks.len() > 1);
        let raw = src(0, vocab);
        let rb = pack::native_row_bytes(Dtype::Q8_0, HIDDEN) as usize;
        let tok = 777u32;
        // 1. embedding of one token
        let x = embed(dev, &emb_op, &m, &[tok], 1.0).unwrap();
        let mut want = vec![0f32; HIDDEN];
        quant::dequant_row(Dtype::Q8_0, &raw[tok as usize * rb..(tok as usize + 1) * rb], &mut want);
        assert_eq!(x.len(), HIDDEN);
        for c in 0..HIDDEN {
            assert!((x[c] - want[c]).abs() <= 1e-6 * want[c].abs().max(0.01), "embedding column {c}");
        }
        // 2. attention norm
        let nw = data(HIDDEN, 5, 1.0).iter().map(|v| v + 1.5).collect::<Vec<f32>>();
        let call = rmsnorm(dev, &rms, HIDDEN as u32, 1, 1e-6).unwrap();
        let xn = run_simple(dev, &call, &[&x, &nw], &[HIDDEN]).unwrap().remove(0);
        let mut cpu = vec![0f32; HIDDEN];
        cpu_rmsnorm(&mut cpu, &x, &nw, 1e-6);
        assert!(rel_err(&xn, &cpu.iter().map(|&v| v as f64).collect::<Vec<_>>(), RMS_TOL.0, RMS_TOL.1) <= 1.0);
        // 3. RoPE on q (14 x 64) and k (2 x 64) at position 321 (the q/k projections are stood in by data)
        let inv = rope_inv_freq(QWEN_THETA, HEAD_DIM, None);
        let (q, k) = (data(N_HEAD * HEAD_DIM, 8, 3.0), data(N_KV * HEAD_DIM, 9, 3.0));
        let (qg, kg) = (
            run_inplace(dev, &rope(dev, &rp, N_HEAD as u32, HEAD_DIM as u32, HEAD_DIM as u32, false).unwrap(), &[&q, &rope_table(321, &inv, 1.0)]).unwrap(),
            run_inplace(dev, &rope(dev, &rp, N_KV as u32, HEAD_DIM as u32, HEAD_DIM as u32, false).unwrap(), &[&k, &rope_table(321, &inv, 1.0)]).unwrap(),
        );
        for (got, orig, heads) in [(&qg, &q, N_HEAD), (&kg, &k, N_KV)] {
            let mut cpu = orig.clone();
            for h in 0..heads {
                cpu_rope(&mut cpu[h * HEAD_DIM..(h + 1) * HEAD_DIM], 321, &inv, false, 1.0);
            }
            assert!(rel_err(got, &cpu.iter().map(|&v| v as f64).collect::<Vec<_>>(), ROPE_TOL.0, ROPE_TOL.1) <= 1.0, "rope {heads} heads");
        }
        // 4. attention scores softmax: 14 heads over a context of 321 positions, scaled by 1/sqrt(64)
        let scores: Vec<f32> = data(N_HEAD * 321, 12, 8.0).iter().map(|v| v / 8.0).collect();
        let call = softmax(dev, &sm, 321, N_HEAD as u32).unwrap();
        let p = run_simple(dev, &call, &[&scores], &[N_HEAD * 321]).unwrap().remove(0);
        for (h, row) in scores.chunks(321).enumerate() {
            let mut c = row.to_vec();
            cpu_softmax(&mut c);
            assert!(rel_err(&p[h * 321..(h + 1) * 321], &c.iter().map(|&v| v as f64).collect::<Vec<_>>(), 1e-7, 1e-4) <= 1.0, "softmax head {h}");
        }
        // 5. SwiGLU over the FFN width, 6. residual add on the hidden state
        let (g, u) = (data(FFN, 14, 4.0), data(FFN, 15, 2.0));
        let call = act(dev, &ac, FFN as u32).unwrap();
        let h = run_simple(dev, &call, &[&g, &u], &[FFN]).unwrap().remove(0);
        let mut c = g.clone();
        silu_mul(&mut c, &u);
        assert!(rel_err(&h, &c.iter().map(|&v| v as f64).collect::<Vec<_>>(), 4e-6, 4e-4) <= 1.0);
        let down = data(HIDDEN, 16, 1.0);
        let call = add(dev, &ad, HIDDEN as u32).unwrap();
        let r = run_inplace(dev, &call, &[&x, &down]).unwrap();
        assert!((0..HIDDEN).all(|i| r[i].to_bits() == (x[i] + down[i]).to_bits()));
        println!("  qwen 2.5 0.5B chain (embedding Q8_0 -> rmsnorm -> rope 14+2 heads -> softmax 14 x 321 -> swiglu 4864 -> add 896): ok");
    }

    #[test]
    fn rope_inv_freq_matches_the_cpu_recipe() {
        let f = rope_inv_freq(10_000.0, 8, None);
        assert_eq!(f.len(), 4);
        assert_eq!(f[0], 1.0);
        assert!((f[1] - 0.1).abs() < 1e-6 && (f[3] - 0.001).abs() < 1e-6);
        let g = rope_inv_freq(10_000.0, 8, Some(&[1.0, 2.0, 4.0, 8.0]));
        assert!((g[1] - 0.05).abs() < 1e-6 && (g[3] - 0.000125).abs() < 1e-7);
    }

    #[test]
    fn act_shader_has_one_mode_line_and_gather_sections_exist() {
        for a in [Act::SiluMul, Act::GeluMul, Act::Gelu] {
            assert!(Op::act(a).wgsl.contains("const MODE: u32 ="));
        }
        assert_ne!(Op::act(Act::SiluMul).wgsl, Op::act(Act::GeluMul).wgsl);
        for d in pack::ALL_FORMATS {
            let g = Op::gather(d).unwrap();
            assert!(g.wgsl.contains("fn weight(") && g.wgsl.contains("fn main(") && !g.wgsl.contains("@IDS@") && !g.wgsl.contains("//@@"), "{}", pack::name(d));
        }
        // the sections are distinct (a wrong marker would give every format the first decoder)
        assert_ne!(Op::gather(Dtype::Q4_0).unwrap().wgsl, Op::gather(Dtype::Q5_0).unwrap().wgsl);
    }

    #[test]
    fn software_adapter_ops() {
        let Some(dev) = open_or_skip(&AdapterSelector::Software, "software") else { return };
        exercise(&dev);
        qwen_chain(&dev);
    }

    /// `cargo test real_adapter_ops -- --ignored --nocapture`; adapter from `BURITI_TEST_ADAPTER` (default best).
    #[test]
    #[ignore]
    fn real_adapter_ops() {
        let sel = std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best);
        let dev = open_or_skip(&sel, "real").expect("no adapter could be opened");
        exercise(&dev);
        qwen_chain(&dev);
    }

    fn median(v: &mut Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// Per-call time of each op for realistic shapes: 3 runs of 20 calls in ONE compute pass (GPU timestamps), plus the wall
    /// time of a single call submitted alone (submit + wait). Informational. `BURITI_TEST_ADAPTER=backend:vulkan cargo test
    /// ops_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn ops_timing() {
        let sel = std::env::var("BURITI_TEST_ADAPTER").map(|s| AdapterSelector::parse(&s)).unwrap_or(AdapterSelector::Best);
        let dev = open_or_skip(&sel, "timing").expect("no adapter could be opened");
        // warm up the pipelines
        let f = |n: usize, s: u64| dev.storage_from_f32(&data(n, s, 1.0), "t").unwrap();
        let mut rows: Vec<(String, f64, f64, f64)> = Vec::new();
        let mut time = |name: &str, step: Step| {
            let mut gpu = Vec::new();
            let mut wall20 = Vec::new();
            let mut single = Vec::new();
            dev.run_steps(std::slice::from_ref(&step), 20, DEFAULT_TIMEOUT).unwrap();
            for _ in 0..3 {
                let t = dev.run_steps(std::slice::from_ref(&step), 20, DEFAULT_TIMEOUT).unwrap();
                gpu.push(t.gpu_seconds.unwrap_or(t.wall_seconds) / 20.0 * 1e6);
                wall20.push(t.wall_seconds / 20.0 * 1e6);
                single.push(dev.run_steps(std::slice::from_ref(&step), 1, DEFAULT_TIMEOUT).unwrap().wall_seconds * 1e6);
            }
            rows.push((name.to_string(), median(&mut gpu), median(&mut wall20), median(&mut single)));
        };
        {
            let (x, w, out) = (f(HIDDEN, 1), f(HIDDEN, 2), dev.create_storage(HIDDEN as u64 * 4, "o").unwrap());
            let c = rmsnorm(&dev, &Op::rmsnorm(), HIDDEN as u32, 1, 1e-6).unwrap();
            time("rmsnorm 896", c.step(&[&x, &w, &out]));
            let (x2, w2, o2) = (f(HEAD_DIM * 16, 1), f(HEAD_DIM, 2), dev.create_storage(HEAD_DIM as u64 * 16 * 4, "o").unwrap());
            let c2 = rmsnorm(&dev, &Op::rmsnorm(), HEAD_DIM as u32, 16, 1e-6).unwrap();
            time("rmsnorm 16 heads x 64 (q/k norm)", c2.step(&[&x2, &w2, &o2]));
            let (xl, wl, bl, ol) = (f(6144, 1), f(6144, 2), f(6144, 3), dev.create_storage(6144 * 4, "o").unwrap());
            let cl = layernorm(&dev, &Op::layernorm(), 6144, 1, 1e-5).unwrap();
            time("layernorm 6144 (StarCoder2)", cl.step(&[&xl, &wl, &bl, &ol]));
            let v = f((N_HEAD + N_KV) * HEAD_DIM, 1);
            let cst = dev.storage_from_f32(&rope_table(100, &rope_inv_freq(QWEN_THETA, HEAD_DIM, None), 1.0), "cs").unwrap();
            let cr = rope(&dev, &Op::rope(), (N_HEAD + N_KV) as u32, HEAD_DIM as u32, HEAD_DIM as u32, false).unwrap();
            time("rope q+k 16 heads x 64 (host table)", cr.step(&[&v, &cst]));
            let inv = f(HEAD_DIM / 2, 2);
            let crt = rope_trig(&dev, &Op::rope_trig(), (N_HEAD + N_KV) as u32, HEAD_DIM as u32, HEAD_DIM as u32, false, 100, 1.0).unwrap();
            time("rope q+k 16 heads x 64 (GPU angle)", crt.step(&[&v, &inv]));
            let (g, u, o) = (f(FFN, 1), f(FFN, 2), dev.create_storage(FFN as u64 * 4, "o").unwrap());
            let ca = act(&dev, &Op::act(Act::SiluMul), FFN as u32).unwrap();
            time("silu*mul 4864", ca.step(&[&g, &u, &o]));
            let cg = act(&dev, &Op::act(Act::GeluMul), FFN as u32).unwrap();
            time("gelu*mul 4864", cg.step(&[&g, &u, &o]));
            let (s, so) = (f(N_HEAD * 1024, 1), dev.create_storage((N_HEAD * 1024) as u64 * 4, "o").unwrap());
            let cs = softmax(&dev, &Op::softmax(), 1024, N_HEAD as u32).unwrap();
            time("softmax 14 heads x 1024", cs.step(&[&s, &so]));
            let (xa, ya) = (f(HIDDEN, 1), f(HIDDEN, 2));
            let cad = add(&dev, &Op::add(), HIDDEN as u32).unwrap();
            time("add 896", cad.step(&[&xa, &ya]));
        }
        for d in [Dtype::Q8_0, Dtype::Q4_0] {
            let vocab = 151_936u64;
            let m = upload_matrix(&dev, d, vocab, HIDDEN as u64, &|s, n| pack::gen_rows(d, s, n, HIDDEN, 1), None).unwrap();
            let op = Op::gather(d).unwrap();
            let calls = gather(&dev, &op, &m, 1, 1.0).unwrap();
            let ids = dev.storage_from_bytes(&123_456u32.to_le_bytes(), "ids").unwrap();
            let out = dev.create_storage(HIDDEN as u64 * 4, "o").unwrap();
            assert_eq!(calls.len(), 1);
            time(&format!("embedding gather {} 151936x896, 1 token", pack::name(d)), calls[0].step(&gather_io(&m, 0, &ids, &out)));
        }
        println!("TIMING {} ({}) per call, microseconds, median of 3 runs of 20 calls in one pass:", dev.name(), dev.report.backend);
        println!("  {:<42} {:>10} {:>14} {:>16}", "op", "GPU us", "wall/20 us", "1 call alone us");
        for (n, g, w, s) in &rows {
            println!("  {n:<42} {g:>10.1} {w:>14.1} {s:>16.1}");
        }
    }
}
