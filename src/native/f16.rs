//! KV cache em f16: conversões e produtos com valores f16.
//!
//! O KV cache guarda K e V em f16 (metade da memória e da banda de leitura da atenção em contextos
//! longos; é o padrão do llama.cpp). A conversão satura em ±65504 em vez de virar infinito.
//! No x86_64 usa F16C/AVX2 (escolhido em tempo de execução); nas demais CPUs, a versão escalar.


const F16_MAX: f32 = 65504.0;

/// f32 → f16 (arredonda para o par mais próximo; satura em vez de gerar infinito).
pub fn f32_to_f16(v: f32) -> u16 {
    let v = v.clamp(-F16_MAX, F16_MAX);
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x7F_FFFF;
    if exp == 255 {
        return sign | 0x7E00; // NaN (o infinito já foi saturado acima)
    }
    let e = exp - 127 + 15;
    if e <= 0 {
        // zero ou subnormal em f16
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    if h >= 0x7C00 {
        h = 0x7BFF;
    }
    sign | h as u16
}

#[cfg(target_arch = "x86_64")]
fn has_f16c() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
    })
}

/// dst[i] = f16(src[i]).
pub fn store_f16(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() {
            // SAFETY: só chamamos depois de confirmar AVX2, FMA e F16C.
            return unsafe { store_f16_avx2(src, dst) };
        }
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32_to_f16(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn store_f16_avx2(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let (lo, hi) = (_mm256_set1_ps(-F16_MAX), _mm256_set1_ps(F16_MAX));
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(src.as_ptr().add(i)), lo), hi);
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        i += 8;
    }
    while i < n {
        dst[i] = f32_to_f16(src[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::quant::f16_to_f32;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    #[test]
    fn roundtrip_and_edges() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(1e9), 0x7BFF, "satura em vez de virar infinito");
        assert_eq!(f32_to_f16(-1e9), 0xFBFF);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7BFF);
        assert_eq!(f32_to_f16(6.1035156e-5), 0x0400, "menor normal");
        assert_eq!(f32_to_f16(5.9604645e-8), 0x0001, "menor subnormal");
        assert_eq!(f32_to_f16(1e-9), 0x0000);
        // ida e volta: erro relativo <= 2^-11 na faixa normal
        let mut s = 7u64;
        for _ in 0..20000 {
            let v = lcg(&mut s) * 10f32.powi((lcg(&mut s) * 4.0) as i32);
            if v.abs() < 6.2e-5 || v.abs() > 65000.0 {
                continue;
            }
            let back = f16_to_f32(f32_to_f16(v));
            assert!(((back - v) / v).abs() <= 1.0 / 2048.0 + 1e-7, "{v} -> {back}");
        }
    }

    /// A conversão em hardware (F16C) tem de coincidir bit a bit com a versão escalar.
    #[test]
    fn store_matches_scalar() {
        let mut s = 99u64;
        let src: Vec<f32> = (0..1003).map(|i| lcg(&mut s) * [1e-6f32, 1e-3, 1.0, 1e3, 1e5][i % 5]).collect();
        let mut dst = vec![0u16; src.len()];
        store_f16(&src, &mut dst);
        for (i, (&v, &h)) in src.iter().zip(&dst).enumerate() {
            assert_eq!(h, f32_to_f16(v), "elemento {i}: {v}");
        }
    }
}
