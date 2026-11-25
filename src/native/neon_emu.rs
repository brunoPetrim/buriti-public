//! Emulação em software das poucas instruções NEON que os kernels de `neon.rs` usam.
//!
//! Só existe em testes fora do ARM64: permite executar aqui (no x86) a lógica dos kernels NEON —
//! índices, deslocamentos, ordem das lanes, sinais — contra a referência em float, sem ter hardware
//! ARM. Cada função segue a definição da instrução ARM de mesmo nome. Não é código de produção.
#![allow(non_camel_case_types)]

pub type int8x8_t = [i8; 8];
pub type uint8x8_t = [u8; 8];
pub type int8x16_t = [i8; 16];
pub type uint8x16_t = [u8; 16];
pub type int16x8_t = [i16; 8];
pub type int32x4_t = [i32; 4];
pub type float32x4_t = [f32; 4];

pub unsafe fn vld1q_s8(p: *const i8) -> int8x16_t {
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = *p.add(i);
    }
    r
}

pub unsafe fn vld1q_u8(p: *const u8) -> uint8x16_t {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = *p.add(i);
    }
    r
}

pub fn vdupq_n_s32(v: i32) -> int32x4_t {
    [v; 4]
}
pub fn vdupq_n_f32(v: f32) -> float32x4_t {
    [v; 4]
}
pub fn vdupq_n_s8(v: i8) -> int8x16_t {
    [v; 16]
}
pub fn vdupq_n_u8(v: u8) -> uint8x16_t {
    [v; 16]
}
pub fn vdup_n_u8(v: u8) -> uint8x8_t {
    [v; 8]
}

/// `vcombine_u8(lo, hi)`: as 8 lanes de `lo` seguidas das 8 de `hi`.
pub fn vcombine_u8(lo: uint8x8_t, hi: uint8x8_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    r[..8].copy_from_slice(&lo);
    r[8..].copy_from_slice(&hi);
    r
}

pub fn vandq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] & b[i];
    }
    r
}

pub fn vorrq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] | b[i];
    }
    r
}

/// Deslocamento lógico à direita por constante, por lane.
pub fn vshrq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] >> N;
    }
    r
}

/// Deslocamento à esquerda por constante, por lane (descarta os bits que saem).
pub fn vshlq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = a[i] << N;
    }
    r
}

/// Subtração por lane com wraparound (como a instrução).
pub fn vsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i].wrapping_sub(b[i]);
    }
    r
}

pub fn vreinterpretq_s8_u8(a: uint8x16_t) -> int8x16_t {
    let mut r = [0i8; 16];
    for i in 0..16 {
        r[i] = a[i] as i8;
    }
    r
}

/// `vtstq_u8`: cada lane vira 0xFF se `(a & b) != 0`, senão 0.
pub fn vtstq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = [0u8; 16];
    for i in 0..16 {
        r[i] = if a[i] & b[i] != 0 { 0xFF } else { 0 };
    }
    r
}

pub fn vget_low_s8(a: int8x16_t) -> int8x8_t {
    let mut r = [0i8; 8];
    r.copy_from_slice(&a[..8]);
    r
}

/// Multiplicação alargada 8×8 → 16 bits, por lane.
pub fn vmull_s8(a: int8x8_t, b: int8x8_t) -> int16x8_t {
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[i] as i16 * b[i] as i16;
    }
    r
}

/// Como `vmull_s8`, mas sobre as 8 lanes ALTAS dos dois vetores de 16.
pub fn vmull_high_s8(a: int8x16_t, b: int8x16_t) -> int16x8_t {
    let mut r = [0i16; 8];
    for i in 0..8 {
        r[i] = a[8 + i] as i16 * b[8 + i] as i16;
    }
    r
}

/// `vpadalq_s16(acc, b)`: acc[i] += b[2i] + b[2i+1] (soma de pares adjacentes, alargada para 32 bits).
pub fn vpadalq_s16(acc: int32x4_t, b: int16x8_t) -> int32x4_t {
    let mut r = acc;
    for i in 0..4 {
        r[i] = r[i].wrapping_add(b[2 * i] as i32 + b[2 * i + 1] as i32);
    }
    r
}

/// `vmlaq_n_s32(a, b, n)`: a[i] + b[i] * n.
pub fn vmlaq_n_s32(a: int32x4_t, b: int32x4_t, n: i32) -> int32x4_t {
    let mut r = [0i32; 4];
    for i in 0..4 {
        r[i] = a[i].wrapping_add(b[i].wrapping_mul(n));
    }
    r
}

pub fn vaddvq_s32(a: int32x4_t) -> i32 {
    a[0].wrapping_add(a[1]).wrapping_add(a[2]).wrapping_add(a[3])
}

pub fn vcvtq_f32_s32(a: int32x4_t) -> float32x4_t {
    [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32]
}

/// `vfmaq_n_f32(a, b, n)`: a[i] + b[i] * n (multiplicação-soma fundida).
pub fn vfmaq_n_f32(a: float32x4_t, b: float32x4_t, n: f32) -> float32x4_t {
    [b[0].mul_add(n, a[0]), b[1].mul_add(n, a[1]), b[2].mul_add(n, a[2]), b[3].mul_add(n, a[3])]
}

pub fn vaddvq_f32(a: float32x4_t) -> f32 {
    (a[0] + a[1]) + (a[2] + a[3])
}
