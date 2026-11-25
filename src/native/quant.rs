//! Desquantização e produto escalar para os formatos do GGML (layouts idênticos aos do ggml).
//! Versão de referência: clara e correta, ainda sem SIMD manual (fica para a fase de otimização).

use super::gguf::Dtype;

/// f16 -> f32 por manipulação de bits (o caminho normal é só shifts e um add; roda uma vez por bloco).
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        // zero ou subnormal: m * 2^-24
        let v = m as f32 * (1.0 / 16_777_216.0);
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13)); // inf / NaN
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Desquantiza UM bloco (`dtype.block().1` bytes) para `dtype.block().0` floats.
#[inline(always)]
pub fn dequant_block(dtype: Dtype, blk: &[u8], out: &mut [f32]) {
    match dtype {
        Dtype::Q4_0 => {
            let d = f16_at(blk, 0);
            for j in 0..16 {
                let q = blk[2 + j];
                out[j] = ((q & 0xF) as i32 - 8) as f32 * d;
                out[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
            }
        }
        Dtype::Q5_0 => {
            let d = f16_at(blk, 0);
            let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let q = blk[6 + j] as u32;
                out[j] = (((q & 0xF) | xh0) as i32 - 16) as f32 * d;
                out[j + 16] = (((q >> 4) | xh1) as i32 - 16) as f32 * d;
            }
        }
        Dtype::Q8_0 => {
            let d = f16_at(blk, 0);
            for i in 0..32 {
                out[i] = (blk[2 + i] as i8) as f32 * d;
            }
        }
        Dtype::Q4_1 => {
            let (d, m) = (f16_at(blk, 0), f16_at(blk, 2));
            for j in 0..16 {
                let q = blk[4 + j];
                out[j] = (q & 0xF) as f32 * d + m;
                out[j + 16] = (q >> 4) as f32 * d + m;
            }
        }
        Dtype::Q5_1 => {
            let (d, m) = (f16_at(blk, 0), f16_at(blk, 2));
            let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let q = blk[8 + j] as u32;
                out[j] = ((q & 0xF) | xh0) as f32 * d + m;
                out[j + 16] = ((q >> 4) | xh1) as f32 * d + m;
            }
        }
        Dtype::Q5K => {
            // d, dmin, scales[12], qh[32], qs[128]; o 5º bit do grupo g está no bit g de qh[l]
            let d = f16_at(blk, 0);
            let dmin = f16_at(blk, 2);
            let scales = &blk[4..16];
            let qh = &blk[16..48];
            let qs = &blk[48..176];
            for g in 0..8 {
                let (sc, m) = scale_min_k4(g, scales);
                let (dg, mg) = (d * sc as f32, dmin * m as f32);
                let q = &qs[(g / 2) * 32..(g / 2) * 32 + 32];
                for l in 0..32 {
                    let nib = if g % 2 == 0 { q[l] & 0xF } else { q[l] >> 4 };
                    let hi = ((qh[l] >> g) & 1) << 4;
                    out[g * 32 + l] = dg * (nib | hi) as f32 - mg;
                }
            }
        }
        Dtype::Q4K => {
            let d = f16_at(blk, 0);
            let dmin = f16_at(blk, 2);
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            let mut is = 0;
            let mut o = 0;
            for g in 0..4 {
                let q = &qs[g * 32..g * 32 + 32];
                let (sc1, m1) = scale_min_k4(is, scales);
                let (sc2, m2) = scale_min_k4(is + 1, scales);
                let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
                for l in 0..32 {
                    out[o + l] = d1 * (q[l] & 0xF) as f32 - mn1;
                    out[o + 32 + l] = d2 * (q[l] >> 4) as f32 - mn2;
                }
                o += 64;
                is += 2;
            }
        }
        Dtype::Q6K => {
            let ql = &blk[0..128];
            let qh = &blk[128..192];
            let sc = &blk[192..208];
            let d = f16_at(blk, 208);
            for n in 0..2 {
                let (ql, qh, sc) = (&ql[n * 64..], &qh[n * 32..], &sc[n * 8..]);
                let o = n * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    out[o + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    out[o + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    out[o + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    out[o + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }
        Dtype::F32 | Dtype::F16 | Dtype::BF16 => unreachable!("F32/F16/BF16 não usam blocos"),
    }
}

#[inline(always)]
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Produto de uma linha F16/BF16 por `x`, com 8 acumuladores (vetoriza; `.sum()` em ordem não vetoriza).
#[inline(always)]
fn dot_half(row: &[u8], x: &[f32], conv: fn(u16) -> f32) -> f32 {
    let mut acc = [0f32; 8];
    let rc = row.chunks_exact(16);
    let xc = x.chunks_exact(8);
    let (rr, xr) = (rc.remainder(), xc.remainder());
    for (r, xv) in rc.zip(xc) {
        for i in 0..8 {
            acc[i] += conv(u16::from_le_bytes([r[2 * i], r[2 * i + 1]])) * xv[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (c, v) in rr.chunks_exact(2).zip(xr) {
        s += conv(u16::from_le_bytes([c[0], c[1]])) * v;
    }
    s
}

/// Produto escalar com 8 acumuladores independentes (o compilador vetoriza; somar em ordem
/// estrita impediria a vetorização).
#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Produto de uma linha quantizada (bytes) por um vetor `x` de floats.
/// No x86_64, usa a versão compilada com AVX2+FMA quando a CPU suporta (checado em tempo de
/// execução, então o mesmo binário roda em CPUs antigas). Em ARM64 o NEON já é o padrão.
pub fn dot_row(dtype: Dtype, row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: só chamamos depois de confirmar que a CPU tem AVX2 e FMA.
            return unsafe { dot_row_avx2(dtype, row, x) };
        }
    }
    dot_row_impl(dtype, row, x)
}

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_avx2(dtype: Dtype, row: &[u8], x: &[f32]) -> f32 {
    dot_row_impl(dtype, row, x)
}

#[inline(always)]
fn dot_row_impl(dtype: Dtype, row: &[u8], x: &[f32]) -> f32 {
    match dtype {
        Dtype::F32 => row.chunks_exact(4).zip(x).map(|(c, v)| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * v).sum(),
        Dtype::F16 => dot_half(row, x, f16_to_f32),
        Dtype::BF16 => dot_half(row, x, bf16_to_f32),
        _ => {
            let (be, bb) = dtype.block();
            let mut buf = [0f32; 256];
            let mut acc = 0f32;
            for (i, blk) in row.chunks_exact(bb).enumerate() {
                dequant_block(dtype, blk, &mut buf[..be]);
                acc += dot_f32(&buf[..be], &x[i * be..(i + 1) * be]);
            }
            acc
        }
    }
}

/// Desquantiza uma linha inteira (usado para o embedding de um token).
pub fn dequant_row(dtype: Dtype, row: &[u8], out: &mut [f32]) {
    match dtype {
        Dtype::F32 => {
            for (o, c) in out.iter_mut().zip(row.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        Dtype::F16 => {
            for (o, c) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        Dtype::BF16 => {
            for (o, c) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        _ => {
            let (be, bb) = dtype.block();
            for (i, blk) in row.chunks_exact(bb).enumerate() {
                dequant_block(dtype, blk, &mut out[i * be..(i + 1) * be]);
            }
        }
    }
}
