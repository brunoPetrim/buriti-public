//! Kernels NEON (ARM64) do produto linha de pesos × ativação int8.
//!
//! Cobrem os formatos do catálogo: Q8_0, Q4_0, Q5_0, Q4_K e Q6_K (os demais seguem no caminho
//! genérico). Usam só NEON básico (`vmull_s8` + `vpadalq_s16`), presente em todo ARMv8; a
//! instrução `sdot` (dotprod) ainda não é usada.
//!
//! **Como foi validado sem hardware ARM:** em testes fora do ARM, as instruções vêm de
//! `neon_emu.rs` (emulação em software), então a lógica daqui (índices, deslocamentos, ordem das
//! lanes) é executada e comparada com a referência em float. A velocidade só dá para medir em ARM
//! de verdade. E em ARM de verdade, `self_test()` roda no primeiro uso e compara cada kernel com o
//! caminho genérico; se algo divergir, o motor volta ao caminho genérico (mais lento, mas correto).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(not(target_arch = "aarch64"))]
use super::neon_emu::*;

use super::{f16_at, scale_min_k4, Act, Dtype};

/// 16 produtos int8 × int8 somados em 4 acumuladores int32.
#[inline(always)]
unsafe fn dot16(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let hi = vmull_high_s8(a, b);
    vpadalq_s16(vpadalq_s16(acc, lo), hi)
}

/// Produto de 32 pesos (2 vetores de 16, já int8) por 32 ativações.
#[inline(always)]
unsafe fn dot32(w0: int8x16_t, w1: int8x16_t, x: *const i8) -> int32x4_t {
    dot16(dot16(vdupq_n_s32(0), w0, vld1q_s8(x)), w1, vld1q_s8(x.add(16)))
}

pub unsafe fn dot_q8_0(row: &[u8], a: &Act) -> f32 {
    let nb = row.len() / 34;
    assert!(a.q.len() >= nb * 32 && a.d.len() >= nb);
    let mut acc = vdupq_n_f32(0.0);
    for (i, blk) in row.chunks_exact(34).enumerate() {
        let p = blk.as_ptr().add(2) as *const i8;
        let s = dot32(vld1q_s8(p), vld1q_s8(p.add(16)), a.q.as_ptr().add(i * 32));
        acc = vfmaq_n_f32(acc, vcvtq_f32_s32(s), f16_at(blk, 0) * a.d[i]);
    }
    vaddvq_f32(acc)
}

pub unsafe fn dot_q4_0(row: &[u8], a: &Act) -> f32 {
    let nb = row.len() / 18;
    assert!(a.q.len() >= nb * 32 && a.d.len() >= nb);
    let (m, eight) = (vdupq_n_u8(0x0F), vdupq_n_s8(8));
    let mut acc = vdupq_n_f32(0.0);
    for (i, blk) in row.chunks_exact(18).enumerate() {
        let qs = vld1q_u8(blk.as_ptr().add(2));
        // pesos 0..15 = nibbles baixos, 16..31 = nibbles altos (menos 8)
        let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qs, m)), eight);
        let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs)), eight);
        let s = dot32(lo, hi, a.q.as_ptr().add(i * 32));
        acc = vfmaq_n_f32(acc, vcvtq_f32_s32(s), f16_at(blk, 0) * a.d[i]);
    }
    vaddvq_f32(acc)
}

/// 16 bits (2 bytes) → 16 lanes: 0xFF onde o bit é 1 (lane i testa o bit i).
#[inline(always)]
unsafe fn bits16(b0: u8, b1: u8) -> uint8x16_t {
    const MASK: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    vtstq_u8(vcombine_u8(vdup_n_u8(b0), vdup_n_u8(b1)), vld1q_u8(MASK.as_ptr()))
}

pub unsafe fn dot_q5_0(row: &[u8], a: &Act) -> f32 {
    let nb = row.len() / 22;
    assert!(a.q.len() >= nb * 32 && a.d.len() >= nb);
    let (m, ten, sixteen) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x10), vdupq_n_s8(16));
    let mut acc = vdupq_n_f32(0.0);
    for (i, blk) in row.chunks_exact(22).enumerate() {
        let qs = vld1q_u8(blk.as_ptr().add(6));
        // o 5º bit do peso j está no bit j de qh (bytes 2..6 do bloco): 0..15 nos bytes 2,3; 16..31 nos 4,5
        let h0 = vandq_u8(bits16(blk[2], blk[3]), ten);
        let h1 = vandq_u8(bits16(blk[4], blk[5]), ten);
        let lo = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qs, m), h0)), sixteen);
        let hi = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(qs), h1)), sixteen);
        let s = dot32(lo, hi, a.q.as_ptr().add(i * 32));
        acc = vfmaq_n_f32(acc, vcvtq_f32_s32(s), f16_at(blk, 0) * a.d[i]);
    }
    vaddvq_f32(acc)
}

pub unsafe fn dot_q4k(row: &[u8], a: &Act) -> f32 {
    let nb = row.len() / 144;
    assert!(a.q.len() >= nb * 256 && a.d.len() >= nb && a.bsums.len() >= nb * 16);
    let m4 = vdupq_n_u8(0x0F);
    let mut acc = 0f32;
    for (i, blk) in row.chunks_exact(144).enumerate() {
        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
        let scales = &blk[4..16];
        let qs = blk.as_ptr().add(16);
        let xq = a.q.as_ptr().add(i * 256);
        let bs = &a.bsums[i * 16..i * 16 + 16];
        let mut sumi = vdupq_n_s32(0);
        let mut sum_m = 0i32;
        for g in 0..8 {
            // grupo g: 32 pesos de 4 bits; os pares de grupos (2j, 2j+1) dividem os mesmos 32 bytes
            let q = qs.add((g / 2) * 32);
            let (q0, q1) = (vld1q_u8(q), vld1q_u8(q.add(16)));
            let (w0, w1) = if g % 2 == 0 {
                (vandq_u8(q0, m4), vandq_u8(q1, m4))
            } else {
                (vshrq_n_u8::<4>(q0), vshrq_n_u8::<4>(q1))
            };
            let s = dot32(vreinterpretq_s8_u8(w0), vreinterpretq_s8_u8(w1), xq.add(g * 32));
            let (sc, mn) = scale_min_k4(g, scales);
            sumi = vmlaq_n_s32(sumi, s, sc);
            sum_m += mn * (bs[2 * g] as i32 + bs[2 * g + 1] as i32);
        }
        acc += a.d[i] * (d * vaddvq_s32(sumi) as f32 - dmin * sum_m as f32);
    }
    acc
}

pub unsafe fn dot_q6k(row: &[u8], a: &Act) -> f32 {
    let nb = row.len() / 210;
    assert!(a.q.len() >= nb * 256 && a.d.len() >= nb);
    let (m4, m3, c32) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x03), vdupq_n_s8(32));
    let mut acc = 0f32;
    for (i, blk) in row.chunks_exact(210).enumerate() {
        let d = f16_at(blk, 208);
        let mut sumi = vdupq_n_s32(0);
        for n in 0..2 {
            let ql = blk.as_ptr().add(n * 64);
            let qh = blk.as_ptr().add(128 + n * 32);
            let sc = &blk[192 + n * 8..192 + n * 8 + 8];
            let x = a.q.as_ptr().add(i * 256 + n * 128);
            for half in 0..2 {
                // elementos l = half*16 .. half*16+16 de cada uma das 4 "partes" de 32
                let ql_a = vld1q_u8(ql.add(half * 16));
                let ql_b = vld1q_u8(ql.add(32 + half * 16));
                let qhv = vld1q_u8(qh.add(half * 16));
                let hi2 = |v: uint8x16_t| vshlq_n_u8::<4>(vandq_u8(v, m3));
                let q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_a, m4), hi2(qhv))), c32);
                let q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_b, m4), hi2(vshrq_n_u8::<2>(qhv)))), c32);
                let q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql_a), hi2(vshrq_n_u8::<4>(qhv)))), c32);
                let q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql_b), hi2(vshrq_n_u8::<6>(qhv)))), c32);
                let xo = x.add(half * 16);
                let s0 = dot16(vdupq_n_s32(0), q1, vld1q_s8(xo));
                let s1 = dot16(vdupq_n_s32(0), q2, vld1q_s8(xo.add(32)));
                let s2 = dot16(vdupq_n_s32(0), q3, vld1q_s8(xo.add(64)));
                let s3 = dot16(vdupq_n_s32(0), q4, vld1q_s8(xo.add(96)));
                sumi = vmlaq_n_s32(sumi, s0, sc[half] as i8 as i32);
                sumi = vmlaq_n_s32(sumi, s1, sc[half + 2] as i8 as i32);
                sumi = vmlaq_n_s32(sumi, s2, sc[half + 4] as i8 as i32);
                sumi = vmlaq_n_s32(sumi, s3, sc[half + 6] as i8 as i32);
            }
        }
        acc += d * a.d[i] * vaddvq_s32(sumi) as f32;
    }
    acc
}

/// Formatos que têm kernel NEON.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub fn supports(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Q8_0 | Dtype::Q4_0 | Dtype::Q5_0 | Dtype::Q4K | Dtype::Q6K)
}

/// Despacha para o kernel NEON do formato (só chamar se `supports(dtype)`).
pub unsafe fn dot(dtype: Dtype, row: &[u8], a: &Act) -> f32 {
    match dtype {
        Dtype::Q8_0 => dot_q8_0(row, a),
        Dtype::Q4_0 => dot_q4_0(row, a),
        Dtype::Q5_0 => dot_q5_0(row, a),
        Dtype::Q4K => dot_q4k(row, a),
        Dtype::Q6K => dot_q6k(row, a),
        _ => unreachable!("sem kernel NEON para {dtype:?}"),
    }
}

/// Linha de pesos aleatória (com escalas f16 finitas) e ativação quantizada correspondente.
fn sample(dtype: Dtype, nblocks: usize, seed: u64) -> (Vec<u8>, Act, Vec<f32>) {
    let (be, bb) = dtype.block();
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut row: Vec<u8> = (0..nblocks * bb).map(|_| (next() & 0xFF) as u8).collect();
    let d_offsets: &[usize] = match dtype {
        Dtype::Q4K => &[0, 2],
        Dtype::Q6K => &[208],
        _ => &[0],
    };
    for b in 0..nblocks {
        for &o in d_offsets {
            row[b * bb + o] = 0x00;
            row[b * bb + o + 1] = 0x28; // f16 0x2800 = 0.03125
        }
    }
    let x: Vec<f32> = (0..nblocks * be).map(|_| (next() % 20001) as f32 / 10000.0 - 1.0).collect();
    let act = Act::new(dtype, &x).expect("formato quantizado");
    (row, act, x)
}

/// Compara cada kernel NEON com o caminho genérico em dados aleatórios. Em ARM, roda uma vez no
/// primeiro uso: se algum divergir, o motor usa o caminho genérico em vez do NEON.
pub fn self_test() -> bool {
    use crate::native::quant::dequant_row;
    for dtype in [Dtype::Q8_0, Dtype::Q4_0, Dtype::Q5_0, Dtype::Q4K, Dtype::Q6K] {
        for seed in 1..=4u64 {
            let (row, act, x) = sample(dtype, 6, seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let want = super::dot_row_q_impl(dtype, &row, &act);
            // SAFETY: ARM64 sempre tem NEON; em testes fora do ARM as instruções são emuladas.
            let got = unsafe { dot(dtype, &row, &act) };
            let mut w = vec![0f32; x.len()];
            dequant_row(dtype, &row, &mut w);
            let scale: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            if !got.is_finite() || (got - want).abs() > 1e-4 * scale + 1e-6 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::quant::{dequant_row, dot_row};

    /// Os kernels NEON (emulados fora do ARM) dão o mesmo resultado que o caminho genérico.
    #[test]
    fn matches_generic() {
        assert!(self_test());
    }

    /// E ficam perto da referência em float (como os kernels AVX2), em várias sementes e tamanhos.
    #[test]
    fn matches_float_reference() {
        for dtype in [Dtype::Q8_0, Dtype::Q4_0, Dtype::Q5_0, Dtype::Q4K, Dtype::Q6K] {
            for (nblocks, seed) in [(1usize, 7u64), (3, 11), (6, 13), (10, 17), (6, 19), (6, 23)] {
                let (row, act, x) = sample(dtype, nblocks, seed);
                let want = dot_row(dtype, &row, &x);
                let mut w = vec![0f32; x.len()];
                dequant_row(dtype, &row, &mut w);
                let scale: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
                let got = unsafe { dot(dtype, &row, &act) };
                assert!(
                    (got - want).abs() <= 0.02 * scale + 1e-3,
                    "{dtype:?} blocos={nblocks} semente={seed}: referência {want}, NEON {got}, escala {scale}"
                );
            }
        }
    }

    /// O emulador segue a definição das instruções (conferido em casos pequenos à mão).
    #[cfg(not(target_arch = "aarch64"))]
    #[test]
    fn emulator_semantics() {
        use super::super::neon_emu::*;
        unsafe {
            let a: [i8; 16] = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16];
            let b: [i8; 16] = [2; 16];
            // dot16 sobre zero = soma de a[i]*2
            let s = vaddvq_s32(dot16(vdupq_n_s32(0), a, b));
            assert_eq!(s, a.iter().map(|&v| v as i32 * 2).sum::<i32>());
            // vtst: bit 3 de 0b1000 e de 0b0111
            let t = vtstq_u8(vcombine_u8(vdup_n_u8(0b1000), vdup_n_u8(0b0111)), vdupq_n_u8(0b1000));
            assert_eq!(&t[..8], &[0xFF; 8]);
            assert_eq!(&t[8..], &[0; 8]);
            // vmull_high usa as lanes altas
            let hi = vmull_high_s8(a, b);
            assert_eq!(hi[0], 9 * 2);
            assert_eq!(hi[7], -16 * 2);
            // pares adjacentes
            let r = vpadalq_s16([10, 20, 30, 40], [1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(r, [13, 27, 41, 55]);
        }
    }
}
