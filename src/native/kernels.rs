//! Produto escalar com ativações quantizadas em inteiros (a técnica do llama.cpp):
//! o vetor de entrada `x` é quantizado uma vez para int8 (blocos de 32, ou de 256 para os
//! formatos K), e cada linha de pesos é multiplicada em aritmética inteira. Muito mais rápido
//! que desquantizar os pesos para float. Validado contra a versão de referência (quant.rs).

use super::{
    gguf::Dtype,
    quant::f16_to_f32,
};

// Kernels NEON (ARM64). Fora do ARM só existem em testes, com as instruções emuladas.
#[cfg(any(target_arch = "aarch64", test))]
#[path = "neon.rs"]
mod neon;
#[cfg(all(test, not(target_arch = "aarch64")))]
#[path = "neon_emu.rs"]
mod neon_emu;

/// Ativação quantizada (int8) com escala por bloco.
pub struct Act {
    pub(super) q: Vec<i8>,
    pub(super) d: Vec<f32>,
    /// soma de cada 16 valores (só para Q4_K)
    pub(super) bsums: Vec<i16>,
    /// só nos formatos K (bloco de 256): por bloco, 4 x i32 com as somas de 32 valores em pares
    /// (soma32[2p] no i16 baixo, soma32[2p+1] no alto). É o termo dos mínimos do kernel reempacotado
    /// (`repack.rs`); derivado de `bsums`, então não muda nenhum valor dos demais campos.
    pub(super) bp: Vec<i32>,
}

impl Act {
    /// `None` para pesos F32/F16 (esses usam o caminho em float).
    pub fn new(dtype: Dtype, x: &[f32]) -> Option<Act> {
        // Só diagnóstico (BURITI_FLOAT_ACT=1): ativações em float, sem a quantização int8. Lento (usa o caminho
        // de referência em float), mas dá o resultado "exato" dos pesos, para separar o ruído da quantização
        // da ativação de um erro de implementação.
        static FLOAT_ACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *FLOAT_ACT.get_or_init(|| std::env::var_os("BURITI_FLOAT_ACT").is_some()) {
            return None;
        }
        // Só diagnóstico (BURITI_ACT_GGML=1, lido uma vez): quantização das ativações idêntica à do ggml
        // (b12606): escala f16 no Q8_0; máximo com sinal, d negativo e `nearest_int` no Q8_K. Padrão: desligado.
        let ggml = act_ggml();
        match dtype {
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => None,
            Dtype::Q4K | Dtype::Q5K | Dtype::Q6K if ggml => Some(Self::quantize_ggml(x, true)),
            Dtype::Q4_0 | Dtype::Q5_0 | Dtype::Q8_0 if ggml => Some(Self::quantize_ggml(x, false)),
            Dtype::Q4K | Dtype::Q5K | Dtype::Q6K => Some(Self::quantize(x, 256, true)),
            // bsums (somas de 16) também aqui: o kernel reempacotado do Q4_0/Q5_0 (repack.rs) precisa da soma
            // dos 32 valores do bloco (correção do deslocamento dos pesos sem sinal); os kernels antigos não as leem.
            Dtype::Q4_0 | Dtype::Q5_0 | Dtype::Q8_0 => Some(Self::quantize(x, 32, true)),
            // Q4_1/Q5_1 têm um mínimo por bloco: precisam da soma da ativação (bsums, de 16 em 16)
            Dtype::Q4_1 | Dtype::Q5_1 => Some(Self::quantize(x, 32, true)),
        }
    }

    /// Classe de ativação do formato (formatos da mesma classe compartilham a mesma `Act`).
    fn class(dtype: Dtype) -> Option<usize> {
        match dtype {
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => None,
            Dtype::Q4K | Dtype::Q5K | Dtype::Q6K => Some(0),
            Dtype::Q4_0 | Dtype::Q5_0 | Dtype::Q8_0 => Some(2),
            Dtype::Q4_1 | Dtype::Q5_1 => Some(3),
        }
    }

    /// Transcrição escalar do ggml b12606, só para diagnóstico (`BURITI_ACT_GGML=1`).
    /// `kq = false`: blocos de 32 como `quantize_row_q8_0` (caminho AVX2 de `arch/x86/quants.c`): `id = 127/amax`
    /// (do máximo em f32, não de `1/d`), `q = nearest_int(x*id)` (par mais próximo), e a escala `d = amax/127`
    /// guardada em **f16** (arredondada), que é o que o produto escalar do ggml usa.
    /// `kq = true`: blocos de 256 como `quantize_row_q8_K_ref` (o x86 chama a própria `_ref`): máximo **com sinal**
    /// (o primeiro de maior módulo), `iscale = -127/max`, `q = min(127, nearest_int(iscale*x))` e `d = 1/iscale`
    /// (f32, **negativo** quando o máximo é positivo); `bsums` somam os `q` já gravados (portanto com o sinal trocado).
    pub(super) fn quantize_ggml(x: &[f32], kq: bool) -> Act {
        // `nearest_int` do ggml: truque do número mágico (arredonda para o par mais próximo, |v| < 2^22)
        fn nearest_int(v: f32) -> i32 {
            let r = v + 12582912.0;
            ((r.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
        }
        let block = if kq { 256 } else { 32 };
        let mut q = vec![0i8; x.len()];
        let mut d = Vec::with_capacity(x.len() / block);
        for (xb, qb) in x.chunks_exact(block).zip(q.chunks_exact_mut(block)) {
            if kq {
                let (mut max, mut amax) = (0f32, 0f32);
                for &v in xb {
                    if v.abs() > amax {
                        amax = v.abs();
                        max = v;
                    }
                }
                if amax == 0.0 {
                    d.push(0.0);
                    continue;
                }
                let iscale = -127.0f32 / max;
                for (o, &v) in qb.iter_mut().zip(xb) {
                    *o = nearest_int(iscale * v).min(127) as i8;
                }
                d.push(1.0 / iscale);
            } else {
                let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
                let id = if amax != 0.0 { 127.0f32 / amax } else { 0.0 };
                d.push(f16_to_f32(super::f16::f32_to_f16(amax / 127.0)));
                for (o, &v) in qb.iter_mut().zip(xb) {
                    *o = nearest_int(v * id).clamp(-127, 127) as i8;
                }
            }
        }
        // somas de 16 também no formato de blocos de 32 (ver `new`)
        let bsums: Vec<i16> = q.chunks_exact(16).map(|c| c.iter().map(|&v| v as i16).sum()).collect();
        let bp = if kq {
            bsums
                .chunks_exact(4)
                .map(|c| {
                    let s = |a: i16, b: i16| (a as i32 + b as i32) as i16 as u16 as u32;
                    (s(c[0], c[1]) | (s(c[2], c[3]) << 16)) as i32
                })
                .collect()
        } else {
            Vec::new()
        };
        Act { q, d, bsums, bp }
    }

    /// Quantização escalar (a única no ARM e a reserva no x86). Empates exatos (v * id = k + 0,5) arredondam para o PAR, como o AVX2
    /// (`cvtps_epi32`, modo padrão) e o `nearest_int` do ggml; `.round()` arredondava para longe do zero e o caminho do ARM divergia dos
    /// outros só nesses empates (o CI do macOS ARM64 mostrou: o dot com a ativação do ggml diferia da padrão em ~1e-4).
    fn quantize_scalar(x: &[f32], block: usize, q: &mut [i8], d: &mut Vec<f32>) {
        for (xb, qb) in x.chunks_exact(block).zip(q.chunks_exact_mut(block)) {
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            d.push(amax / 127.0);
            for (o, v) in qb.iter_mut().zip(xb) {
                *o = (v * id).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
    }

    fn quantize(x: &[f32], block: usize, bsums: bool) -> Act {
        let mut q = vec![0i8; x.len()];
        let mut d = Vec::with_capacity(x.len() / block);
        #[cfg(target_arch = "x86_64")]
        let done = if has_avx2() {
            // SAFETY: só chamamos depois de confirmar AVX2; `block` é múltiplo de 32.
            unsafe { quantize_avx2(x, block, &mut q, &mut d) };
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "x86_64"))]
        let done = false;
        if !done {
            Self::quantize_scalar(x, block, &mut q, &mut d);
        }
        let bsums = if bsums {
            q.chunks_exact(16).map(|c| c.iter().map(|&v| v as i16).sum()).collect()
        } else {
            Vec::new()
        };
        // somas de 32 em pares (|soma| <= 32*128 = 4096 cabe em i16)
        let bp = if block == 256 && bsums.len() == q.len() / 16 {
            bsums
                .chunks_exact(4)
                .map(|c| {
                    let s = |a: i16, b: i16| (a as i32 + b as i32) as i16 as u16 as u32;
                    (s(c[0], c[1]) | (s(c[2], c[3]) << 16)) as i32
                })
                .collect()
        } else {
            Vec::new()
        };
        Act { q, d, bsums, bp }
    }
}

/// `BURITI_ACT_GGML` (só diagnóstico, lido uma vez por processo): ativações quantizadas como o ggml.
pub(super) fn act_ggml() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("BURITI_ACT_GGML").is_some_and(|v| v != "0" && !v.is_empty()))
}

/// Guarda as ativações já quantizadas do vetor de entrada atual, por classe de formato, para que
/// várias multiplicações sobre o mesmo vetor (q/k/v; gate/up) quantizem uma vez só.
/// Chame `clear()` sempre que o vetor de entrada mudar.
#[derive(Default)]
pub struct ActCache {
    slots: [Option<Act>; 4],
}

impl ActCache {
    pub fn clear(&mut self) {
        self.slots = Default::default();
    }

    /// `None` para formatos em float (sem ativação quantizada).
    pub fn get(&mut self, dtype: Dtype, x: &[f32]) -> Option<&Act> {
        let c = Act::class(dtype)?;
        if self.slots[c].is_none() {
            self.slots[c] = Act::new(dtype, x);
        }
        self.slots[c].as_ref()
    }
}

/// Quantização int8 por blocos com AVX2: máximo absoluto, escala e arredondamento (para o par mais
/// próximo, como o `cvtps2dq`) em vetor. `block` é múltiplo de 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn quantize_avx2(x: &[f32], block: usize, q: &mut [i8], d: &mut Vec<f32>) {
    use std::arch::x86_64::*;
    let sign = _mm256_set1_ps(-0.0);
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (xb, qb) in x.chunks_exact(block).zip(q.chunks_exact_mut(block)) {
        let mut m = _mm256_setzero_ps();
        for c in xb.chunks_exact(8) {
            m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(c.as_ptr())));
        }
        let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        let m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        let amax = _mm_cvtss_f32(m4);
        let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        d.push(amax / 127.0);
        let idv = _mm256_set1_ps(id);
        for (xc, qc) in xb.chunks_exact(32).zip(qb.chunks_exact_mut(32)) {
            let p = xc.as_ptr();
            let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p), idv));
            let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(8)), idv));
            let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(16)), idv));
            let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p.add(24)), idv));
            // empacota 4×8 int32 → 32 int8 (com saturação); os pacotes saem entrelaçados por
            // faixa de 128 bits, a permutação final recoloca na ordem original
            let packed = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
            _mm256_storeu_si256(qc.as_mut_ptr() as *mut __m256i, _mm256_permutevar8x32_epi32(packed, order));
        }
    }
}

#[inline(always)]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

#[inline(always)]
fn scale_min_k4(j: usize, q: &[u8]) -> (i32, i32) {
    if j < 4 {
        ((q[j] & 63) as i32, (q[j + 4] & 63) as i32)
    } else {
        (((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as i32, ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as i32)
    }
}

/// Produto de uma linha quantizada de pesos por uma ativação já quantizada.
pub fn dot_row_q(dtype: Dtype, row: &[u8], a: &Act) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: só chamamos depois de confirmar que a CPU tem AVX2 e FMA.
            return unsafe {
                match dtype {
                    Dtype::Q8_0 => x86::dot_q8_0(row, a),
                    Dtype::Q4_0 => x86::dot_q4_0(row, a),
                    Dtype::Q5_0 => x86::dot_q5_0(row, a),
                    Dtype::Q4K => x86::dot_q4k::<1>(row, [a])[0],
                    Dtype::Q6K => x86::dot_q6k::<1>(row, [a])[0],
                    Dtype::Q4_1 => x86::dot_q4_1(row, a),
                    Dtype::Q5_1 => x86::dot_q5_1(row, a),
                    Dtype::Q5K => x86::dot_q5k::<1>(row, [a])[0],
                    _ => dot_row_q_avx2(dtype, row, a),
                }
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon::supports(dtype) && neon_ok() {
            // SAFETY: NEON faz parte do ARMv8 (aarch64).
            return unsafe { neon::dot(dtype, row, a) };
        }
    }
    dot_row_q_impl(dtype, row, a)
}

/// Autoteste dos kernels NEON no primeiro uso: se divergirem do caminho genérico, não são usados.
#[cfg(target_arch = "aarch64")]
fn neon_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        let ok = neon::self_test();
        if !ok {
            eprintln!("[native] aviso: os kernels NEON falharam no autoteste; usando o caminho genérico (mais lento)");
        }
        ok
    })
}

/// Kernels AVX2 com instruções explícitas (mesma técnica do llama.cpp). Cada linha de pesos é
/// percorrida acumulando num vetor de floats; a soma horizontal é feita uma única vez no fim.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::Act;

    /// f16 → f32 em hardware (F16C), sem ramos: a versão por bits custava mais que o resto do bloco.
    #[inline(always)]
    unsafe fn f16_at(b: &[u8], i: usize) -> f32 {
        assert!(i + 2 <= b.len());
        let h = std::ptr::read_unaligned(b.as_ptr().add(i) as *const u16);
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
    }

    #[inline(always)]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// 32 produtos int8 × int8 → 8 somas parciais int32.
    #[inline(always)]
    unsafe fn dot_i8(w: __m256i, x: __m256i) -> __m256i {
        let aw = _mm256_sign_epi8(w, w); // |w|
        let sx = _mm256_sign_epi8(x, w); // x com o sinal de w
        _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), _mm256_set1_epi16(1))
    }

    /// Nibbles de 16 bytes → 32 bytes (0..15): baixos nas posições 0..15, altos em 16..31.
    #[inline(always)]
    unsafe fn nibbles(p: *const u8) -> __m256i {
        let qs = _mm_loadu_si128(p as *const __m128i);
        let m4 = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(qs, m4);
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), m4);
        _mm256_set_m128i(hi, lo)
    }

    /// 32 bits → 32 bytes (0xFF onde o bit é 1).
    #[inline(always)]
    unsafe fn bytes_from_bits_32(bits: u32) -> __m256i {
        let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(bits as i32), shuf);
        let bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
        _mm256_cmpeq_epi8(_mm256_or_si256(bytes, bit_mask), _mm256_set1_epi64x(-1))
    }

    /// Formatos de bloco de 32 sem mínimo (Q8_0/Q4_0/Q5_0), um token. Dois blocos por iteração com
    /// dois acumuladores independentes (esconde a latência do fmadd); `$w` produz os 32 pesos int8.
    macro_rules! kernel32 {
        ($name:ident, $bsz:expr, |$blk:ident| $w:expr) => {
            #[target_feature(enable = "avx2,fma,f16c")]
            pub unsafe fn $name(row: &[u8], a: &Act) -> f32 {
                let nb = row.len() / $bsz;
                assert!(a.q.len() >= nb * 32 && a.d.len() >= nb);
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut i = 0;
                while i + 2 <= nb {
                    let b0 = &row[i * $bsz..(i + 1) * $bsz];
                    let b1 = &row[(i + 1) * $bsz..(i + 2) * $bsz];
                    let w0: __m256i = { let $blk = b0; $w };
                    let w1: __m256i = { let $blk = b1; $w };
                    let x0 = _mm256_loadu_si256(a.q.as_ptr().add(i * 32) as *const __m256i);
                    let x1 = _mm256_loadu_si256(a.q.as_ptr().add(i * 32 + 32) as *const __m256i);
                    let d0 = f16_at(b0, 0) * *a.d.get_unchecked(i);
                    let d1 = f16_at(b1, 0) * *a.d.get_unchecked(i + 1);
                    acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(dot_i8(w0, x0)), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(dot_i8(w1, x1)), acc1);
                    i += 2;
                }
                if i < nb {
                    let b0 = &row[i * $bsz..(i + 1) * $bsz];
                    let w0: __m256i = { let $blk = b0; $w };
                    let x0 = _mm256_loadu_si256(a.q.as_ptr().add(i * 32) as *const __m256i);
                    let d0 = f16_at(b0, 0) * *a.d.get_unchecked(i);
                    acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(dot_i8(w0, x0)), acc0);
                }
                hsum_ps(_mm256_add_ps(acc0, acc1))
            }
        };
    }

    kernel32!(dot_q8_0, 34, |blk| _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i));
    kernel32!(dot_q4_0, 18, |blk| _mm256_sub_epi8(nibbles(blk.as_ptr().add(2)), _mm256_set1_epi8(8)));
    kernel32!(dot_q5_0, 22, |blk| {
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        // bit 4 do peso: se for 0, o valor é nibble-16 (bits altos em 1: 0xF0); se for 1, é o nibble
        let hi5 = _mm256_andnot_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0xF0u8 as i8));
        _mm256_or_si256(nibbles(blk.as_ptr().add(6)), hi5)
    });

    // ---- formatos K (Q4_K, Q5_K, Q6_K): 256 pesos por bloco, escalas por sub-bloco.
    //
    // Antes, cada bloco e cada token refaziam um laço escalar de 8 passos para o termo dos
    // "mínimos" (e o Q6_K gastava 16 instruções vetoriais no "-32·Σx"). Agora: escalas e mínimos
    // saem do bloco já como vetores (uma vez por bloco, não por token) e os dois termos usam as
    // somas parciais da ativação (`bsums`) com UM `madd`, acumulado em vetor.

    /// Escalas e mínimos do Q4_K/Q5_K (12 bytes em blk[4..16]) como vetores: as 8 escalas em
    /// broadcast (i16) e os 8 mínimos duplicados (16 × i16, na ordem das somas de 16 elementos da
    /// ativação). É o mesmo desempacotamento de bits do llama.cpp; o teste contra a referência
    /// em float (que usa `scale_min_k4`) cobre qualquer erro aqui.
    #[inline(always)]
    unsafe fn k4_scales(blk: &[u8]) -> ([__m256i; 8], __m256i) {
        const K1: u32 = 0x3f3f_3f3f;
        const K2: u32 = 0x0f0f_0f0f;
        const K3: u32 = 0x0303_0303;
        let p = blk.as_ptr().add(4) as *const u32;
        let (u0, u1, u2) = (p.read_unaligned(), p.add(1).read_unaligned(), p.add(2).read_unaligned());
        let u3n = ((u2 >> 4) & K2) | (((u1 >> 6) & K3) << 4); // mínimos 4..7
        let u2n = u1 & K1; // mínimos 0..3
        let u1n = (u2 & K2) | (((u0 >> 6) & K3) << 4); // escalas 4..7
        let u0n = u0 & K1; // escalas 0..3
        let sc8 = _mm_cvtsi64_si128((u0n as u64 | ((u1n as u64) << 32)) as i64);
        let mut arr = [0i16; 8];
        _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, _mm_cvtepu8_epi16(sc8));
        let mut scv = [_mm256_setzero_si256(); 8];
        for j in 0..8 {
            scv[j] = _mm256_set1_epi16(arr[j]);
        }
        let mn8 = _mm_cvtsi64_si128((u2n as u64 | ((u3n as u64) << 32)) as i64);
        let m16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mn8, mn8));
        (scv, m16)
    }

    /// Q4_K: os 8 vetores de 32 pesos (0..15): baixos e altos de cada um dos 4 pares de grupos.
    #[inline(always)]
    unsafe fn q4k_weights(blk: &[u8]) -> ([__m256i; 4], [__m256i; 4]) {
        let m4 = _mm256_set1_epi8(0x0F);
        let mut wl = [_mm256_setzero_si256(); 4];
        let mut wh = [_mm256_setzero_si256(); 4];
        for j in 0..4 {
            let q4 = _mm256_loadu_si256(blk.as_ptr().add(16 + 32 * j) as *const __m256i);
            wl[j] = _mm256_and_si256(q4, m4);
            wh[j] = _mm256_and_si256(_mm256_srli_epi16(q4, 4), m4);
        }
        (wl, wh)
    }

    /// Soma inteira ponderada pelas escalas dos 256 pesos do bloco contra 256 ativações int8.
    #[inline(always)]
    unsafe fn k4_sumi(wl: &[__m256i; 4], wh: &[__m256i; 4], scv: &[__m256i; 8], xq: *const i8) -> __m256i {
        let mut sumi = _mm256_setzero_si256();
        for j in 0..4 {
            let q8l = _mm256_loadu_si256(xq.add(64 * j) as *const __m256i);
            let q8h = _mm256_loadu_si256(xq.add(64 * j + 32) as *const __m256i);
            let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(wl[j], q8l), scv[2 * j]);
            let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(wh[j], q8h), scv[2 * j + 1]);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(pl, ph));
        }
        sumi
    }

    /// Q4_K e Q5_K para `N` ativações de uma vez (N = 1, 4 ou 8): as mesmas contas por ativação em
    /// qualquer N, então o resultado é idêntico (o teste exige igualdade). O termo dos mínimos entra
    /// no mesmo acumulador (um vetor por ativação), o que deixa registradores livres para N = 8.
    /// `$w` desempacota os pesos do bloco `$blk` em (baixos[4], altos[4]) de 32 valores sem sinal.
    macro_rules! kernelk {
        ($name:ident, $bsz:expr, |$blk:ident| $w:expr) => {
            #[target_feature(enable = "avx2,fma,f16c")]
            pub unsafe fn $name<const N: usize>(row: &[u8], a: [&Act; N]) -> [f32; N] {
                let nb = row.len() / $bsz;
                for x in &a {
                    assert!(x.q.len() >= nb * 256 && x.d.len() >= nb && x.bsums.len() >= nb * 16);
                }
                let mut acc = [_mm256_setzero_ps(); N];
                for (i, $blk) in row.chunks_exact($bsz).enumerate() {
                    let (dw, dmin_w) = (f16_at($blk, 0), f16_at($blk, 2));
                    let (scv, m16) = k4_scales($blk);
                    let (wl, wh): ([__m256i; 4], [__m256i; 4]) = $w;
                    for k in 0..N {
                        let dx = *a[k].d.get_unchecked(i);
                        let bs = _mm256_loadu_si256(a[k].bsums.as_ptr().add(i * 16) as *const __m256i);
                        let mins = _mm256_cvtepi32_ps(_mm256_madd_epi16(m16, bs));
                        let sumi = k4_sumi(&wl, &wh, &scv, a[k].q.as_ptr().add(i * 256));
                        acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(dw * dx), _mm256_cvtepi32_ps(sumi), acc[k]);
                        acc[k] = _mm256_fnmadd_ps(_mm256_set1_ps(dmin_w * dx), mins, acc[k]);
                    }
                }
                let mut out = [0f32; N];
                for k in 0..N {
                    out[k] = hsum_ps(acc[k]);
                }
                out
            }
        };
    }

    kernelk!(dot_q4k, 144, |blk| q4k_weights(blk));
    kernelk!(dot_q5k, 176, |blk| q5k_weights(blk));

    /// Q6_K: as 4 "partes" de 32 pesos de 6 bits (0..63) da metade `n` do bloco.
    #[inline(always)]
    unsafe fn q6k_parts(blk: &[u8], n: usize) -> [__m256i; 4] {
        let m4 = _mm256_set1_epi8(0x0F);
        let m3 = _mm256_set1_epi8(0x03);
        let ql0 = _mm256_loadu_si256(blk.as_ptr().add(n * 64) as *const __m256i);
        let ql1 = _mm256_loadu_si256(blk.as_ptr().add(n * 64 + 32) as *const __m256i);
        let qh = _mm256_loadu_si256(blk.as_ptr().add(128 + n * 32) as *const __m256i);
        [
            _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qh, m3), 4)),
            _mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m3), 4)),
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m3), 4)),
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m3), 4)),
        ]
    }

    /// As 16 escalas (i8) do Q6_K como i16, e delas os vetores de escala por parte: para a metade
    /// `n` e a parte `p`, 8 lanes com a escala do 1º grupo de 16 e 8 com a do 2º.
    #[inline(always)]
    unsafe fn q6k_scales(blk: &[u8]) -> (__m256i, [[__m256i; 4]; 2]) {
        let sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(blk.as_ptr().add(192) as *const __m128i));
        let mut s = [0i16; 16];
        _mm256_storeu_si256(s.as_mut_ptr() as *mut __m256i, sc16);
        let mut v = [[_mm256_setzero_si256(); 4]; 2];
        for n in 0..2 {
            for p in 0..4 {
                v[n][p] = _mm256_set_m128i(_mm_set1_epi16(s[n * 8 + 2 * p + 1]), _mm_set1_epi16(s[n * 8 + 2 * p]));
            }
        }
        (sc16, v)
    }

    /// Σ (q-32)·x ponderado pelas escalas = Σ q·x·sc − 32·Σ_g sc[g]·bsum[g]: o segundo termo sai de
    /// um único `madd` com as somas parciais da ativação.
    #[inline(always)]
    unsafe fn q6k_sumi(parts: &[[__m256i; 4]; 2], scales: &[[__m256i; 4]; 2], corr: __m256i, xq: *const i8) -> __m256i {
        let mut sumi = _mm256_setzero_si256();
        for n in 0..2 {
            for p in 0..4 {
                let x = _mm256_loadu_si256(xq.add(n * 128 + p * 32) as *const __m256i);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(parts[n][p], x), scales[n][p]));
            }
        }
        _mm256_sub_epi32(sumi, corr)
    }

    /// Q6_K para `N` ativações de uma vez (N = 1, 4 ou 8; mesmas contas por ativação em qualquer N,
    /// então o resultado é idêntico). Os pesos do bloco são desempacotados uma vez só.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q6k<const N: usize>(row: &[u8], a: [&Act; N]) -> [f32; N] {
        let nb = row.len() / 210;
        for x in &a {
            assert!(x.q.len() >= nb * 256 && x.d.len() >= nb && x.bsums.len() >= nb * 16);
        }
        let mut acc = [_mm256_setzero_ps(); N];
        for (i, blk) in row.chunks_exact(210).enumerate() {
            let dw = f16_at(blk, 208);
            let (sc16, scales) = q6k_scales(blk);
            let parts = [q6k_parts(blk, 0), q6k_parts(blk, 1)];
            for k in 0..N {
                let bs = _mm256_loadu_si256(a[k].bsums.as_ptr().add(i * 16) as *const __m256i);
                let corr = _mm256_slli_epi32(_mm256_madd_epi16(bs, sc16), 5);
                let sumi = q6k_sumi(&parts, &scales, corr, a[k].q.as_ptr().add(i * 256));
                acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(dw * *a[k].d.get_unchecked(i)), _mm256_cvtepi32_ps(sumi), acc[k]);
            }
        }
        let mut out = [0f32; N];
        for k in 0..N {
            out[k] = hsum_ps(acc[k]);
        }
        out
    }



    /// Formatos de bloco de 32 com mínimo (Q4_1/Q5_1): valor = d·q + m, com q sem sinal (0..15 ou
    /// 0..31). Produto = d·Σ(q·x) + m·Σx; a soma de x vem dos `bsums` da ativação. Gera a versão
    /// de um token (`$n1`) e a de 4 tokens (`$n4`). `$w` produz os 32 pesos q (u8) do bloco `$blk`.
    macro_rules! kernel32m {
        ($n1:ident, $n4:ident, $bsz:expr, |$blk:ident| $w:expr) => {
            #[target_feature(enable = "avx2,fma,f16c")]
            pub unsafe fn $n1(row: &[u8], a: &Act) -> f32 {
                let mut acc = _mm256_setzero_ps();
                let mut acc_m = 0f32;
                let ones = _mm256_set1_epi16(1);
                for (i, $blk) in row.chunks_exact($bsz).enumerate() {
                    let w: __m256i = $w;
                    let x = _mm256_loadu_si256(a.q.as_ptr().add(i * 32) as *const __m256i);
                    let s = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(f16_at($blk, 0) * a.d[i]), _mm256_cvtepi32_ps(s), acc);
                    acc_m += (f16_at($blk, 2) * a.d[i]) * (a.bsums[2 * i] as i32 + a.bsums[2 * i + 1] as i32) as f32;
                }
                hsum_ps(acc) + acc_m
            }

            #[target_feature(enable = "avx2,fma,f16c")]
            pub unsafe fn $n4(row: &[u8], a: [&Act; 4]) -> [f32; 4] {
                let mut acc = [_mm256_setzero_ps(); 4];
                let mut acc_m = [0f32; 4];
                let ones = _mm256_set1_epi16(1);
                for (i, $blk) in row.chunks_exact($bsz).enumerate() {
                    let w: __m256i = $w;
                    let (dw, mw) = (f16_at($blk, 0), f16_at($blk, 2));
                    for k in 0..4 {
                        let x = _mm256_loadu_si256(a[k].q.as_ptr().add(i * 32) as *const __m256i);
                        let s = _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), ones);
                        acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(dw * a[k].d[i]), _mm256_cvtepi32_ps(s), acc[k]);
                        acc_m[k] += (mw * a[k].d[i]) * (a[k].bsums[2 * i] as i32 + a[k].bsums[2 * i + 1] as i32) as f32;
                    }
                }
                [hsum_ps(acc[0]) + acc_m[0], hsum_ps(acc[1]) + acc_m[1], hsum_ps(acc[2]) + acc_m[2], hsum_ps(acc[3]) + acc_m[3]]
            }
        };
    }

    kernel32m!(dot_q4_1, dot_q4_1_x4, 20, |blk| nibbles(blk.as_ptr().add(4)));
    kernel32m!(dot_q5_1, dot_q5_1_x4, 24, |blk| {
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let hi5 = _mm256_and_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0x10));
        _mm256_or_si256(nibbles(blk.as_ptr().add(8)), hi5)
    });

    /// Q5_K: como o Q4_K, mas cada peso tem o 5º bit em `qh` (o grupo g usa o bit g de cada byte).
    /// Devolve os 8 vetores de 32 pesos (0..31): baixos e altos de cada um dos 4 pares de grupos.
    #[inline(always)]
    unsafe fn q5k_weights(blk: &[u8]) -> ([__m256i; 4], [__m256i; 4]) {
        let m4 = _mm256_set1_epi8(0x0F);
        let qh = _mm256_loadu_si256(blk.as_ptr().add(16) as *const __m256i);
        let mut wl = [_mm256_setzero_si256(); 4];
        let mut wh = [_mm256_setzero_si256(); 4];
        for j in 0..4 {
            let q = _mm256_loadu_si256(blk.as_ptr().add(48 + 32 * j) as *const __m256i);
            wl[j] = _mm256_or_si256(_mm256_and_si256(q, m4), bit_to_4(qh, 2 * j as i32));
            wh[j] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4), bit_to_4(qh, 2 * j as i32 + 1));
        }
        (wl, wh)
    }

    /// Leva o bit `g` de cada byte para a posição 4 (valor 0x10). O que vaza para os bytes
    /// vizinhos (deslocamento de 16 bits) cai fora da máscara.
    #[inline(always)]
    unsafe fn bit_to_4(qh: __m256i, g: i32) -> __m256i {
        let v = if g <= 4 { _mm256_sll_epi16(qh, _mm_cvtsi32_si128(4 - g)) } else { _mm256_srl_epi16(qh, _mm_cvtsi32_si128(g - 4)) };
        _mm256_and_si256(v, _mm256_set1_epi8(0x10))
    }



    // ---- versões "x4": 4 ativações (tokens) por passada. Cada bloco de pesos é desempacotado
    // UMA vez e multiplicado pelas 4 ativações. No prefill o desempacotamento (principalmente o
    // 5º bit do Q5_0) era o custo dominante. A ordem das contas por ativação é a mesma da versão
    // de um token, então o resultado é idêntico.

    /// Formatos de bloco de 32 (Q8_0/Q4_0/Q5_0): `$w` produz os 32 pesos int8 do bloco `$blk`.
    macro_rules! kernel32_x4 {
        ($name:ident, $bsz:expr, |$blk:ident| $w:expr) => {
            #[target_feature(enable = "avx2,fma,f16c")]
            pub unsafe fn $name(row: &[u8], a: [&Act; 4]) -> [f32; 4] {
                // mesma ordem de soma do kernel de um token (blocos pares/ímpares em acumuladores
                // separados): o resultado é idêntico, e o teste compara com igualdade estrita
                let nb = row.len() / $bsz;
                for x in &a {
                    assert!(x.q.len() >= nb * 32 && x.d.len() >= nb);
                }
                let mut acc0 = [_mm256_setzero_ps(); 4];
                let mut acc1 = [_mm256_setzero_ps(); 4];
                let mut i = 0;
                // pares de blocos com acumuladores fixos (a escolha em tempo de execução os jogaria
                // para a memória)
                while i + 2 <= nb {
                    let b0 = &row[i * $bsz..(i + 1) * $bsz];
                    let b1 = &row[(i + 1) * $bsz..(i + 2) * $bsz];
                    let w0: __m256i = { let $blk = b0; $w };
                    let w1: __m256i = { let $blk = b1; $w };
                    step32_x4(w0, f16_at(b0, 0), i, &a, &mut acc0);
                    step32_x4(w1, f16_at(b1, 0), i + 1, &a, &mut acc1);
                    i += 2;
                }
                if i < nb {
                    let b0 = &row[i * $bsz..(i + 1) * $bsz];
                    let w0: __m256i = { let $blk = b0; $w };
                    step32_x4(w0, f16_at(b0, 0), i, &a, &mut acc0);
                }
                let r = |k: usize| hsum_ps(_mm256_add_ps(acc0[k], acc1[k]));
                [r(0), r(1), r(2), r(3)]
            }
        };
    }

    /// Um bloco de 32 pesos já desempacotados (`w`) contra as 4 ativações.
    #[inline(always)]
    unsafe fn step32_x4(w: __m256i, dw: f32, i: usize, a: &[&Act; 4], acc: &mut [__m256; 4]) {
        let aw = _mm256_sign_epi8(w, w); // |w|, uma vez para as 4 ativações
        let ones = _mm256_set1_epi16(1);
        for k in 0..4 {
            let x = _mm256_loadu_si256(a[k].q.as_ptr().add(i * 32) as *const __m256i);
            let sx = _mm256_sign_epi8(x, w);
            let s = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
            acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(dw * *a[k].d.get_unchecked(i)), _mm256_cvtepi32_ps(s), acc[k]);
        }
    }

    kernel32_x4!(dot_q8_0_x4, 34, |blk| _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i));
    kernel32_x4!(dot_q4_0_x4, 18, |blk| _mm256_sub_epi8(nibbles(blk.as_ptr().add(2)), _mm256_set1_epi8(8)));
    kernel32_x4!(dot_q5_0_x4, 22, |blk| {
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let hi5 = _mm256_andnot_si256(bytes_from_bits_32(qh), _mm256_set1_epi8(0xF0u8 as i8));
        _mm256_or_si256(nibbles(blk.as_ptr().add(6)), hi5)
    });


}

/// Mesmo que 4 chamadas de `dot_row_q` (resultado idêntico), mas desempacota os pesos uma vez só.
pub fn dot_row_q_x4(dtype: Dtype, row: &[u8], a: [&Act; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: só chamamos depois de confirmar que a CPU tem AVX2 e FMA.
            return unsafe {
                match dtype {
                    Dtype::Q8_0 => x86::dot_q8_0_x4(row, a),
                    Dtype::Q4_0 => x86::dot_q4_0_x4(row, a),
                    Dtype::Q5_0 => x86::dot_q5_0_x4(row, a),
                    Dtype::Q4K => x86::dot_q4k::<4>(row, a),
                    Dtype::Q6K => x86::dot_q6k::<4>(row, a),
                    Dtype::Q4_1 => x86::dot_q4_1_x4(row, a),
                    Dtype::Q5_1 => x86::dot_q5_1_x4(row, a),
                    Dtype::Q5K => x86::dot_q5k::<4>(row, a),
                    Dtype::F32 | Dtype::F16 | Dtype::BF16 => unreachable!("formatos float usam o caminho em float"),
                }
            };
        }
    }
    [dot_row_q(dtype, row, a[0]), dot_row_q(dtype, row, a[1]), dot_row_q(dtype, row, a[2]), dot_row_q(dtype, row, a[3])]
}

/// 8 ativações por passada (resultado idêntico a 8 chamadas de `dot_row_q`). Nos formatos K os pesos
/// são desempacotados uma vez para as 8; nos demais, são dois grupos de 4.
pub fn dot_row_q_x8(dtype: Dtype, row: &[u8], a: [&Act; 8]) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: só chamamos depois de confirmar que a CPU tem AVX2, FMA e F16C.
            unsafe {
                match dtype {
                    Dtype::Q4K => return x86::dot_q4k::<8>(row, a),
                    Dtype::Q5K => return x86::dot_q5k::<8>(row, a),
                    Dtype::Q6K => return x86::dot_q6k::<8>(row, a),
                    _ => {}
                }
            }
        }
    }
    let lo = dot_row_q_x4(dtype, row, [a[0], a[1], a[2], a[3]]);
    let hi = dot_row_q_x4(dtype, row, [a[4], a[5], a[6], a[7]]);
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_row_q_avx2(dtype: Dtype, row: &[u8], a: &Act) -> f32 {
    dot_row_q_impl(dtype, row, a)
}

#[inline(always)]
fn dot_row_q_impl(dtype: Dtype, row: &[u8], a: &Act) -> f32 {
    let mut acc = 0f32;
    match dtype {
        Dtype::Q8_0 => {
            for (i, blk) in row.chunks_exact(34).enumerate() {
                let xq = &a.q[i * 32..i * 32 + 32];
                let mut s = 0i32;
                for j in 0..32 {
                    s += ((blk[2 + j] as i8 as i16) * xq[j] as i16) as i32;
                }
                acc += f16_at(blk, 0) * a.d[i] * s as f32;
            }
        }
        // Q4_0 e Q5_0: primeiro "desempacota" os 32 pesos para int8, depois um laço simples de
        // produto int16 que o compilador transforma em pmaddwd (AVX2).
        Dtype::Q4_0 => {
            for (i, blk) in row.chunks_exact(18).enumerate() {
                let xq = &a.q[i * 32..i * 32 + 32];
                let qs = &blk[2..18];
                let mut w = [0i8; 32];
                for j in 0..16 {
                    w[j] = (qs[j] & 0xF) as i8 - 8;
                    w[j + 16] = (qs[j] >> 4) as i8 - 8;
                }
                let mut s = 0i32;
                for j in 0..32 {
                    s += (w[j] as i16 * xq[j] as i16) as i32;
                }
                acc += f16_at(blk, 0) * a.d[i] * s as f32;
            }
        }
        Dtype::Q5_0 => {
            for (i, blk) in row.chunks_exact(22).enumerate() {
                let xq = &a.q[i * 32..i * 32 + 32];
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let qs = &blk[6..22];
                let mut w = [0i8; 32];
                for j in 0..16 {
                    w[j] = ((qs[j] & 0xF) | ((((qh >> j) & 1) as u8) << 4)) as i8 - 16;
                    w[j + 16] = ((qs[j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) as i8 - 16;
                }
                let mut s = 0i32;
                for j in 0..32 {
                    s += (w[j] as i16 * xq[j] as i16) as i32;
                }
                acc += f16_at(blk, 0) * a.d[i] * s as f32;
            }
        }
        Dtype::Q4_1 | Dtype::Q5_1 => {
            let bsz = dtype.block().1;
            for (i, blk) in row.chunks_exact(bsz).enumerate() {
                let xq = &a.q[i * 32..i * 32 + 32];
                let mut w = [0u8; 32];
                if dtype == Dtype::Q4_1 {
                    for j in 0..16 {
                        w[j] = blk[4 + j] & 0xF;
                        w[j + 16] = blk[4 + j] >> 4;
                    }
                } else {
                    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                    for j in 0..16 {
                        w[j] = (blk[8 + j] & 0xF) | ((((qh >> j) & 1) as u8) << 4);
                        w[j + 16] = (blk[8 + j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4);
                    }
                }
                let mut s = 0i32;
                for j in 0..32 {
                    s += (w[j] as i16 * xq[j] as i16) as i32;
                }
                let xsum = a.bsums[2 * i] as i32 + a.bsums[2 * i + 1] as i32;
                acc += a.d[i] * (f16_at(blk, 0) * s as f32 + f16_at(blk, 2) * xsum as f32);
            }
        }
        Dtype::Q5K => {
            for (i, blk) in row.chunks_exact(176).enumerate() {
                let d = f16_at(blk, 0);
                let dmin = f16_at(blk, 2);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let xq = &a.q[i * 256..i * 256 + 256];
                let bs = &a.bsums[i * 16..i * 16 + 16];
                let (mut sum_sc, mut sum_m) = (0i32, 0i32);
                for g in 0..8 {
                    let q = &qs[(g / 2) * 32..(g / 2) * 32 + 32];
                    let x = &xq[g * 32..g * 32 + 32];
                    let mut s = 0i32;
                    for l in 0..32 {
                        let nib = if g % 2 == 0 { q[l] & 0xF } else { q[l] >> 4 };
                        let w = (nib | (((qh[l] >> g) & 1) << 4)) as i16;
                        s += (w * x[l] as i16) as i32;
                    }
                    let (sc, m) = scale_min_k4(g, scales);
                    sum_sc += sc * s;
                    sum_m += m * (bs[2 * g] as i32 + bs[2 * g + 1] as i32);
                }
                acc += a.d[i] * (d * sum_sc as f32 - dmin * sum_m as f32);
            }
        }
        Dtype::Q4K => {
            for (i, blk) in row.chunks_exact(144).enumerate() {
                let d = f16_at(blk, 0);
                let dmin = f16_at(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let xq = &a.q[i * 256..i * 256 + 256];
                let bs = &a.bsums[i * 16..i * 16 + 16];
                let (mut sum_sc, mut sum_m) = (0i32, 0i32);
                for g in 0..8 {
                    let q = &qs[(g / 2) * 32..(g / 2) * 32 + 32];
                    let x = &xq[g * 32..g * 32 + 32];
                    let mut w = [0u8; 32];
                    if g % 2 == 0 {
                        for l in 0..32 {
                            w[l] = q[l] & 0xF;
                        }
                    } else {
                        for l in 0..32 {
                            w[l] = q[l] >> 4;
                        }
                    }
                    let mut s = 0i32;
                    for l in 0..32 {
                        s += (w[l] as i16 * x[l] as i16) as i32;
                    }
                    let (sc, m) = scale_min_k4(g, scales);
                    sum_sc += sc * s;
                    sum_m += m * (bs[2 * g] as i32 + bs[2 * g + 1] as i32);
                }
                acc += a.d[i] * (d * sum_sc as f32 - dmin * sum_m as f32);
            }
        }
        Dtype::Q6K => {
            for (i, blk) in row.chunks_exact(210).enumerate() {
                let xq = &a.q[i * 256..i * 256 + 256];
                let d = f16_at(blk, 208);
                let mut sum = 0i32;
                for n in 0..2 {
                    let ql = &blk[n * 64..n * 64 + 64];
                    let qh = &blk[128 + n * 32..128 + n * 32 + 32];
                    let sc = &blk[192 + n * 8..192 + n * 8 + 8];
                    let x = &xq[n * 128..n * 128 + 128];
                    for half in 0..2 {
                        // half = 0: elementos l=0..15 (is=0); half = 1: l=16..31 (is=1)
                        let (mut s0, mut s1, mut s2, mut s3) = (0i32, 0i32, 0i32, 0i32);
                        for l in half * 16..half * 16 + 16 {
                            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i16 - 32;
                            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i16 - 32;
                            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i16 - 32;
                            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i16 - 32;
                            s0 += (q1 * x[l] as i16) as i32;
                            s1 += (q2 * x[l + 32] as i16) as i32;
                            s2 += (q3 * x[l + 64] as i16) as i32;
                            s3 += (q4 * x[l + 96] as i16) as i32;
                        }
                        sum += (sc[half] as i8 as i32) * s0
                            + (sc[half + 2] as i8 as i32) * s1
                            + (sc[half + 4] as i8 as i32) * s2
                            + (sc[half + 6] as i8 as i32) * s3;
                    }
                }
                acc += d * a.d[i] * sum as f32;
            }
        }
        Dtype::F32 | Dtype::F16 | Dtype::BF16 => unreachable!("formatos float usam o caminho em float"),
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::quant::{dequant_row, dot_row};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            (self.next() % 20001) as f32 / 10000.0 - 1.0
        }
    }

    /// Compara o kernel inteiro com a referência em float, para blocos aleatórios.
    fn check(dtype: Dtype, d_offsets: &[usize]) {
        let (be, bb) = dtype.block();
        let nblocks = 6;
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for trial in 0..20 {
            let mut row: Vec<u8> = (0..nblocks * bb).map(|_| (rng.next() & 0xFF) as u8).collect();
            for b in 0..nblocks {
                for &off in d_offsets {
                    // f16 0x2800 = 0.03125 (escala pequena e finita)
                    row[b * bb + off] = 0x00;
                    row[b * bb + off + 1] = 0x28;
                }
            }
            let x: Vec<f32> = (0..nblocks * be).map(|_| rng.f()).collect();
            let reference = dot_row(dtype, &row, &x);
            let mut w = vec![0f32; nblocks * be];
            dequant_row(dtype, &row, &mut w);
            let scale: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            let act = Act::new(dtype, &x).unwrap();
            // o kernel escolhido em tempo de execução (AVX2) e a reserva genérica (usada em ARM)
            for (name, got) in [("dispatch", dot_row_q(dtype, &row, &act)), ("generico", dot_row_q_impl(dtype, &row, &act))] {
                let err = (got - reference).abs();
                assert!(
                    err <= 0.02 * scale + 1e-3,
                    "{dtype:?} {name} trial {trial}: referência {reference}, kernel {got}, erro {err}, escala {scale}"
                );
            }
        }
    }

    /// Micro-benchmark de uma thread: `cargo test bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_kernels() {
        for (dtype, cols) in [(Dtype::Q4_0, 896), (Dtype::Q5_0, 896), (Dtype::Q8_0, 896), (Dtype::Q4K, 1536), (Dtype::Q6K, 1536)] {
            let (be, bb) = dtype.block();
            let rows = 4096;
            let row_bytes = cols / be * bb;
            let mut rng = Rng(0x1234_5678_9ABC_DEF1);
            let mut data: Vec<u8> = (0..rows * row_bytes).map(|_| (rng.next() & 0xFF) as u8).collect();
            for b in 0..rows * cols / be {
                data[b * bb] = 0x00;
                data[b * bb + 1] = 0x28;
            }
            let x: Vec<f32> = (0..cols).map(|_| rng.f()).collect();
            let act = Act::new(dtype, &x).unwrap();
            // melhor de 7 rodadas: pouco sensível a outros programas usando a CPU
            let mut best = f64::MAX;
            let mut sink = 0f32;
            for _ in 0..7 {
                let t0 = std::time::Instant::now();
                for _ in 0..3 {
                    for r in 0..rows {
                        sink += dot_row_q(dtype, &data[r * row_bytes..(r + 1) * row_bytes], &act);
                    }
                }
                best = best.min(t0.elapsed().as_secs_f64());
            }
            let elems = 3.0 * rows as f64 * cols as f64;
            println!("{dtype:?}: {:.2} Gelem/s, {:.2} GB/s (1 thread) [{sink}]", elems / best / 1e9, 3.0 * (rows * row_bytes) as f64 / best / 1e9);

            // versão de 4 tokens: cada elemento de peso vale 4 produtos
            let acts4: Vec<Act> = (0..4).map(|_| Act::new(dtype, &x).unwrap()).collect();
            let g = [&acts4[0], &acts4[1], &acts4[2], &acts4[3]];
            let mut best4 = f64::MAX;
            for _ in 0..7 {
                let t0 = std::time::Instant::now();
                for _ in 0..3 {
                    for r in 0..rows {
                        sink += dot_row_q_x4(dtype, &data[r * row_bytes..(r + 1) * row_bytes], g)[0];
                    }
                }
                best4 = best4.min(t0.elapsed().as_secs_f64());
            }
            println!("{dtype:?} x4: {:.2} Gmac/s (1 thread) [{sink}]", 4.0 * elems / best4 / 1e9);

            // versão de 8 tokens
            let acts8: Vec<Act> = (0..8).map(|_| Act::new(dtype, &x).unwrap()).collect();
            let g8 = [&acts8[0], &acts8[1], &acts8[2], &acts8[3], &acts8[4], &acts8[5], &acts8[6], &acts8[7]];
            let mut best8 = f64::MAX;
            for _ in 0..7 {
                let t0 = std::time::Instant::now();
                for _ in 0..3 {
                    for r in 0..rows {
                        sink += dot_row_q_x8(dtype, &data[r * row_bytes..(r + 1) * row_bytes], g8)[0];
                    }
                }
                best8 = best8.min(t0.elapsed().as_secs_f64());
            }
            println!("{dtype:?} x8: {:.2} Gmac/s (1 thread) [{sink}]", 8.0 * elems / best8 / 1e9);
        }
    }

    /// Teto de vazão com várias threads ao mesmo tempo (o micro-benchmark acima usa uma só):
    /// `cargo test bench_kernels_mt -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_kernels_mt() {
        for dtype in [Dtype::Q4K, Dtype::Q6K] {
            let (be, bb) = dtype.block();
            let (cols, rows) = (2048usize, 2048usize);
            let row_bytes = cols / be * bb;
            let mut rng = Rng(0x1234_5678_9ABC_DEF1);
            let mut data: Vec<u8> = (0..rows * row_bytes).map(|_| (rng.next() & 0xFF) as u8).collect();
            for b in 0..rows * cols / be {
                data[b * bb] = 0x00;
                data[b * bb + 1] = 0x28;
            }
            let x: Vec<f32> = (0..cols).map(|_| rng.f()).collect();
            let acts: Vec<Act> = (0..8).map(|_| Act::new(dtype, &x).unwrap()).collect();
            for threads in [1usize, 2, 3, 4] {
                let t0 = std::time::Instant::now();
                let total: f64 = std::thread::scope(|s| {
                    let hs: Vec<_> = (0..threads)
                        .map(|_| {
                            s.spawn(|| {
                                let g = [&acts[0], &acts[1], &acts[2], &acts[3], &acts[4], &acts[5], &acts[6], &acts[7]];
                                let (mut sink, mut passes) = (0f32, 0usize);
                                while t0.elapsed().as_secs_f64() < 2.0 {
                                    for r in 0..rows {
                                        sink += dot_row_q_x8(dtype, &data[r * row_bytes..(r + 1) * row_bytes], g)[0];
                                    }
                                    passes += 1;
                                }
                                std::hint::black_box(sink);
                                passes as f64 * 8.0 * rows as f64 * cols as f64
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).sum()
                });
                println!("{dtype:?} x8 com {threads} thread(s): {:.1} GMAC/s no total ({:.1} por thread)", total / t0.elapsed().as_secs_f64() / 1e9, total / t0.elapsed().as_secs_f64() / 1e9 / threads as f64);
            }
        }
    }

    /// Empates exatos (v * id = k + 0,5): o AVX2, o escalar de reserva (o único caminho no ARM) e o ggml arredondam para o PAR. Com o
    /// escalar usando `.round()` este teste reprova no x86 (0,5 virava 1, 2,5 virava 3...), o que mostra a divergência sem precisar de ARM.
    #[test]
    fn quantize_ties_round_to_even_in_every_path() {
        for block in [32usize, 256] {
            // amax = 127 -> id = 1 exato: cada elemento já é o valor a arredondar
            let head = [127.0f32, 0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5, -3.5, -4.5, 5.5, 6.5, 100.5, 101.5, 126.5];
            let want: [i8; 16] = [127, 0, 2, 2, 4, 4, 0, -2, -2, -4, -4, 6, 6, 100, 102, 126];
            let mut x = vec![0f32; block * 2];
            for b in 0..2 {
                x[b * block..b * block + head.len()].copy_from_slice(&head);
            }
            let dispatched = Act::quantize(&x, block, false);
            let (mut q, mut d) = (vec![0i8; x.len()], Vec::new());
            Act::quantize_scalar(&x, block, &mut q, &mut d);
            assert_eq!(&q[..16], &want, "escalar, bloco {block}");
            assert_eq!(&dispatched.q[..16], &want, "despachado (AVX2 no x86), bloco {block}");
            assert_eq!(dispatched.q, q, "os dois caminhos têm de dar os mesmos bytes (bloco {block})");
        }
    }

    /// A quantização vetorizada tem de coincidir com a escalar (no máximo ±1 em empates de arredondamento).
    #[test]
    fn quantize_avx2_matches_scalar() {
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for block in [32usize, 256] {
            let x: Vec<f32> = (0..block * 5).map(|_| rng.f() * 3.0).collect();
            let a = Act::quantize(&x, block, false);
            for (bi, xb) in x.chunks_exact(block).enumerate() {
                let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert_eq!(a.d[bi], amax / 127.0, "escala do bloco {bi}");
                let id = 127.0 / amax;
                for (j, v) in xb.iter().enumerate() {
                    let want = (v * id).round_ties_even().clamp(-127.0, 127.0) as i32;
                    let got = a.q[bi * block + j] as i32;
                    assert!((want - got).abs() <= 1, "bloco {bi} elem {j}: esperado {want}, obtido {got}");
                }
            }
        }
    }

    /// O kernel x4 tem de dar o mesmo resultado que 4 chamadas do kernel de um token.
    fn check_x4(dtype: Dtype, d_offsets: &[usize]) {
        let (be, bb) = dtype.block();
        let nblocks = 6;
        let mut rng = Rng(0xD1B54A32D192ED03);
        for trial in 0..20 {
            let mut row: Vec<u8> = (0..nblocks * bb).map(|_| (rng.next() & 0xFF) as u8).collect();
            for b in 0..nblocks {
                for &off in d_offsets {
                    row[b * bb + off] = 0x00;
                    row[b * bb + off + 1] = 0x28;
                }
            }
            let acts: Vec<Act> = (0..4)
                .map(|_| {
                    let x: Vec<f32> = (0..nblocks * be).map(|_| rng.f()).collect();
                    Act::new(dtype, &x).unwrap()
                })
                .collect();
            let got = dot_row_q_x4(dtype, &row, [&acts[0], &acts[1], &acts[2], &acts[3]]);
            for k in 0..4 {
                let single = dot_row_q(dtype, &row, &acts[k]);
                // mesma ordem de soma nas duas versões: tem de ser idêntico, não só próximo
                assert!(
                    got[k] == single,
                    "{dtype:?} trial {trial} ativação {k}: x4 {} vs um token {single}",
                    got[k]
                );
            }
        }
    }

    #[test]
    fn x4_matches_single() {
        check_x4(Dtype::Q8_0, &[0]);
        check_x4(Dtype::Q4_0, &[0]);
        check_x4(Dtype::Q5_0, &[0]);
        check_x4(Dtype::Q4K, &[0, 2]);
        check_x4(Dtype::Q6K, &[208]);
        check_x4(Dtype::Q4_1, &[0, 2]);
        check_x4(Dtype::Q5_1, &[0, 2]);
        check_x4(Dtype::Q5K, &[0, 2]);
    }

    /// O kernel de 8 ativações também tem de ser idêntico ao de um token.
    #[test]
    fn x8_matches_single() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for (dtype, d_offsets) in [(Dtype::Q8_0, vec![0usize]), (Dtype::Q4_0, vec![0]), (Dtype::Q5_0, vec![0]), (Dtype::Q4K, vec![0, 2]), (Dtype::Q5K, vec![0, 2]), (Dtype::Q6K, vec![208]), (Dtype::Q4_1, vec![0, 2]), (Dtype::Q5_1, vec![0, 2])] {
            let (be, bb) = dtype.block();
            let nblocks = 6;
            let mut row: Vec<u8> = (0..nblocks * bb).map(|_| (rng.next() & 0xFF) as u8).collect();
            for b in 0..nblocks {
                for &off in &d_offsets {
                    row[b * bb + off] = 0x00;
                    row[b * bb + off + 1] = 0x28;
                }
            }
            let acts: Vec<Act> = (0..8).map(|_| { let x: Vec<f32> = (0..nblocks * be).map(|_| rng.f()).collect(); Act::new(dtype, &x).unwrap() }).collect();
            let g = [&acts[0], &acts[1], &acts[2], &acts[3], &acts[4], &acts[5], &acts[6], &acts[7]];
            let got = dot_row_q_x8(dtype, &row, g);
            for k in 0..8 {
                assert!(got[k] == dot_row_q(dtype, &row, &acts[k]), "{dtype:?} ativação {k}");
            }
        }
    }

    // ---- BURITI_ACT_GGML: quantização das ativações igual à do ggml b12606 (D53) ----

    /// Arredondamento para o par mais próximo, escrito de outro jeito que o código testado.
    fn rne(v: f32) -> i32 {
        let f = v.floor();
        let diff = v - f;
        let fi = f as i32;
        if diff > 0.5 || (diff == 0.5 && fi % 2 != 0) { fi + 1 } else { fi }
    }

    /// Vetor com blocos de magnitudes bem diferentes (para exercitar o arredondamento da escala em f16).
    fn varied(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|i| rng.f() * [0.013f32, 1.7, 55.3, 0.0004, 9.1][(i / 32) % 5]).collect()
    }

    /// Oráculo escalar do `quantize_row_q8_0` do ggml (caminho AVX2: id = 127/amax; d = amax/127 em f16).
    fn ggml_q8_0_oracle(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let (mut q, mut d) = (Vec::new(), Vec::new());
        for b in x.chunks_exact(32) {
            let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            d.push(f16_to_f32(crate::native::f16::f32_to_f16(amax / 127.0)));
            q.extend(b.iter().map(|v| rne(v * id).clamp(-127, 127) as i8));
        }
        (q, d)
    }

    /// Oráculo escalar do `quantize_row_q8_K_ref` do ggml (máximo com sinal, iscale = -127/max, d = 1/iscale).
    fn ggml_q8k_oracle(x: &[f32]) -> (Vec<i8>, Vec<f32>, Vec<i16>) {
        let (mut q, mut d, mut bs) = (Vec::new(), Vec::new(), Vec::new());
        for b in x.chunks_exact(256) {
            let mut max = 0f32;
            let mut amax = 0f32;
            for &v in b {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            if amax == 0.0 {
                d.push(0.0);
                q.extend(std::iter::repeat(0i8).take(256));
                bs.extend(std::iter::repeat(0i16).take(16));
                continue;
            }
            let iscale = -127.0f32 / max;
            let qb: Vec<i8> = b.iter().map(|&v| rne(iscale * v).min(127) as i8).collect();
            bs.extend(qb.chunks(16).map(|c| c.iter().map(|&v| v as i16).sum::<i16>()));
            q.extend(qb);
            d.push(1.0 / iscale);
        }
        (q, d, bs)
    }

    #[test]
    fn act_ggml_q8_0_matches_reference() {
        let mut rng = Rng(0xA11CE5EED);
        let mut f16_changed = 0;
        for _ in 0..50 {
            let x = varied(&mut rng, 32 * 10);
            let (q, d) = ggml_q8_0_oracle(&x);
            let a = Act::quantize_ggml(&x, false);
            assert_eq!(a.q, q, "q do Q8_0 difere do ggml");
            assert_eq!(a.d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "d (f16) do Q8_0 difere do ggml");
            // dentes: a escala tem de ser mesmo representável em f16 e, em geral, diferente da escala em f32 do padrão
            let plain = Act::quantize(&x, 32, false);
            for (g, p) in a.d.iter().zip(&plain.d) {
                assert_eq!(crate::native::f16::f32_to_f16(*g) as u32, crate::native::f16::f32_to_f16(*p) as u32);
                if g != p {
                    f16_changed += 1;
                }
            }
            // os q do ggml (id = 127/amax em f32) são os mesmos do caminho padrão
            assert_eq!(a.q, plain.q);
        }
        assert!(f16_changed > 100, "o arredondamento f16 quase nunca mudou a escala ({f16_changed}): o teste não exercita nada");
    }

    #[test]
    fn act_ggml_q8_k_matches_reference() {
        let mut rng = Rng(0xB0B5EED);
        let (mut neg_d, mut pos_d) = (0, 0);
        for _ in 0..50 {
            let x = varied(&mut rng, 256 * 4);
            let (q, d, bs) = ggml_q8k_oracle(&x);
            let a = Act::quantize_ggml(&x, true);
            assert_eq!(a.q, q, "q do Q8_K difere do ggml");
            assert_eq!(a.d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "d do Q8_K difere do ggml");
            assert_eq!(a.bsums, bs, "bsums do Q8_K difere do ggml");
            assert_eq!(a.bp.len(), 4 * 4);
            for (dv, blk) in a.d.iter().zip(x.chunks_exact(256)) {
                let max = blk.iter().fold((0f32, 0f32), |(m, am), &v| if v.abs() > am { (v, v.abs()) } else { (m, am) }).0;
                // convenção de sinal: d tem o sinal contrário ao do máximo com sinal
                assert_eq!(*dv < 0.0, max > 0.0);
                if *dv < 0.0 { neg_d += 1 } else { pos_d += 1 }
            }
            // relação com o padrão: |iscale| é o mesmo (127/amax), então q_ggml == -q_padrão nos blocos de máximo
            // positivo e q_ggml == q_padrão nos de máximo negativo (e |d| difere no máximo 1 ulp)
            let plain = Act::quantize(&x, 256, true);
            for (b, blk) in x.chunks_exact(256).enumerate() {
                let max = blk.iter().fold((0f32, 0f32), |(m, am), &v| if v.abs() > am { (v, v.abs()) } else { (m, am) }).0;
                let sgn = if max > 0.0 { -1 } else { 1 };
                let bad = (0..256).filter(|&j| a.q[b * 256 + j] as i32 != sgn * plain.q[b * 256 + j] as i32).count();
                assert_eq!(bad, 0, "bloco {b}: q ggml não é ±q do padrão");
                assert!((a.d[b].abs() - plain.d[b]).abs() <= 2.0 * f32::EPSILON * plain.d[b]);
            }
        }
        assert!(neg_d > 10 && pos_d > 10, "o teste precisa cobrir máximos positivos e negativos ({neg_d}/{pos_d})");
    }

    /// Com as ativações do ggml, os kernels de 4 e 8 ativações continuam idênticos ao de um token, e o
    /// resultado continua perto da referência em float (o sinal trocado do Q8_K e o d negativo não podem quebrar nada).
    #[test]
    fn act_ggml_kernels_still_consistent() {
        let mut rng = Rng(0xC0FFEE);
        for (dtype, d_offsets) in [(Dtype::Q8_0, vec![0usize]), (Dtype::Q4_0, vec![0]), (Dtype::Q5_0, vec![0]), (Dtype::Q4K, vec![0, 2]), (Dtype::Q5K, vec![0, 2]), (Dtype::Q6K, vec![208])] {
            let (be, bb) = dtype.block();
            let nblocks = 6;
            let kq = be == 256;
            let mut row: Vec<u8> = (0..nblocks * bb).map(|_| (rng.next() & 0xFF) as u8).collect();
            for b in 0..nblocks {
                for &off in &d_offsets {
                    row[b * bb + off] = 0x00;
                    row[b * bb + off + 1] = 0x28;
                }
            }
            let xs: Vec<Vec<f32>> = (0..8).map(|_| (0..nblocks * be).map(|_| rng.f()).collect()).collect();
            let acts: Vec<Act> = xs.iter().map(|x| Act::quantize_ggml(x, kq)).collect();
            let g8 = [&acts[0], &acts[1], &acts[2], &acts[3], &acts[4], &acts[5], &acts[6], &acts[7]];
            let got8 = dot_row_q_x8(dtype, &row, g8);
            let got4 = dot_row_q_x4(dtype, &row, [&acts[0], &acts[1], &acts[2], &acts[3]]);
            let mut w = vec![0f32; nblocks * be];
            dequant_row(dtype, &row, &mut w);
            for k in 0..8 {
                let single = dot_row_q(dtype, &row, &acts[k]);
                assert!(got8[k] == single, "{dtype:?} x8 ativação {k}");
                if k < 4 {
                    assert!(got4[k] == single, "{dtype:?} x4 ativação {k}");
                }
                let reference = dot_row(dtype, &row, &xs[k]);
                let scale: f32 = w.iter().zip(&xs[k]).map(|(a, b)| (a * b).abs()).sum();
                assert!((single - reference).abs() <= 0.02 * scale + 1e-3, "{dtype:?} ativação {k}: ref {reference}, ggml-act {single}");
                // o genérico (reserva ARM) também aceita as ativações do ggml
                let generic = dot_row_q_impl(dtype, &row, &acts[k]);
                assert!((generic - single).abs() <= 1e-3 * scale + 1e-3, "{dtype:?} genérico {generic} vs {single}");
            }
        }
    }

    /// Nos formatos K, a ativação do ggml é a padrão com o sinal trocado nos blocos de máximo positivo (e |d| a 1 ulp):
    /// o produto escalar tem de coincidir quase bit a bit (só 1 ulp da escala), senão algum kernel depende do sinal.
    #[test]
    fn act_ggml_q8_k_dot_equals_default_to_ulps() {
        let mut rng = Rng(0xFEED_FACE);
        for (dtype, d_offsets) in [(Dtype::Q4K, vec![0usize, 2]), (Dtype::Q5K, vec![0, 2]), (Dtype::Q6K, vec![208])] {
            let (be, bb) = dtype.block();
            let nblocks = 6;
            let mut row: Vec<u8> = (0..nblocks * bb).map(|_| (rng.next() & 0xFF) as u8).collect();
            for b in 0..nblocks {
                for &off in &d_offsets {
                    row[b * bb + off] = 0x00;
                    row[b * bb + off + 1] = 0x28;
                }
            }
            let (mut worst_ab, mut worst_cd) = (0f32, 0f32);
            for _ in 0..200 {
                let x = varied(&mut rng, nblocks * be);
                let (p, g) = (Act::quantize(&x, 256, true), Act::quantize_ggml(&x, true));
                let (a, b) = (dot_row_q(dtype, &row, &p), dot_row_q(dtype, &row, &g));
                let (c, d) = (dot_row_q_impl(dtype, &row, &p), dot_row_q_impl(dtype, &row, &g));
                let mut w = vec![0f32; nblocks * be];
                dequant_row(dtype, &row, &mut w);
                let scale: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum(); // soma dos |termos| (há cancelamento)
                worst_ab = worst_ab.max((a - b).abs() / scale);
                worst_cd = worst_cd.max((c - d).abs() / scale);
            }
            assert!(
                worst_ab < 1e-5 && worst_cd < 1e-5,
                "{dtype:?}: dot com a ativação do ggml difere da padrão (relativo): kernel despachado {worst_ab:e}, genérico {worst_cd:e}"
            );
        }
    }

    #[test]
    fn q4_1() {
        check(Dtype::Q4_1, &[0, 2]);
    }
    #[test]
    fn q5_1() {
        check(Dtype::Q5_1, &[0, 2]);
    }
    #[test]
    fn q5_k() {
        check(Dtype::Q5K, &[0, 2]);
    }

    /// BF16 e F16: produto em float contra a desquantização de referência.
    #[test]
    fn half_floats() {
        let mut rng = Rng(0xA5A5_5A5A_1234_4321);
        for dtype in [Dtype::BF16, Dtype::F16] {
            let n = 203; // não múltiplo de 8: exercita o resto
            let vals: Vec<f32> = (0..n).map(|_| rng.f() * 4.0).collect();
            let row: Vec<u8> = vals
                .iter()
                .flat_map(|&v| {
                    let bits = v.to_bits();
                    let h: u16 = if dtype == Dtype::BF16 {
                        (bits >> 16) as u16
                    } else {
                        // f16 de uma escala simples: valores pequenos com 10 bits de mantissa
                        let e = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
                        if e <= 0 || e >= 31 { 0 } else { (((bits >> 31) as u16) << 15) | ((e as u16) << 10) | ((bits >> 13) & 0x3FF) as u16 }
                    };
                    h.to_le_bytes()
                })
                .collect();
            let x: Vec<f32> = (0..n).map(|_| rng.f()).collect();
            let mut w = vec![0f32; n];
            dequant_row(dtype, &row, &mut w);
            let want: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
            let got = dot_row(dtype, &row, &x);
            assert!((got - want).abs() < 1e-3 * (1.0 + want.abs()), "{dtype:?}: {got} vs {want}");
        }
    }

    #[test]
    fn q8_0() {
        check(Dtype::Q8_0, &[0]);
    }
    #[test]
    fn q4_0() {
        check(Dtype::Q4_0, &[0]);
    }
    #[test]
    fn q5_0() {
        check(Dtype::Q5_0, &[0]);
    }
    #[test]
    fn q4_k() {
        check(Dtype::Q4K, &[0, 2]);
    }
    #[test]
    fn q6_k() {
        check(Dtype::Q6K, &[208]);
    }
}
