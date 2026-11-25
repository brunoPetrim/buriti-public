//! Operações em `f32` da torre de visão: LayerNorm, GELU, produto de matriz com pesos F16/Q8_0/F32,
//! atenção bidirecional e pixel shuffle. A versão ingênua em `f64` que serve de referência fica em
//! `reference.rs` (só em testes). Tudo é determinístico e **independe do número de threads**: cada
//! valor de saída é calculado por uma única thread, na mesma ordem de soma (e a ordem de soma de um
//! vetor não depende de como os vetores são agrupados de quatro em quatro).
//!
//! Desempenho: o caminho AVX2+FMA+F16C é escolhido em tempo de execução (como em `quant.rs`/`vmath.rs`);
//! há uma versão genérica com a mesma árvore de somas (sem FMA, então não é idêntica bit a bit à
//! AVX2, só dentro da tolerância; os testes cobrem os dois caminhos na mesma máquina). Não usa os
//! kernels int8 do motor de propósito: a torre é pequena (D50) e o f32 evita ruído extra de
//! quantização da ativação na validação contra o llama.cpp.

use rayon::prelude::*;

use crate::native::{gguf::Dtype, quant::dequant_row, vmath};

/// Como calcular o GELU de tangente hiperbólica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeluMode {
    /// `x/(1+exp(-2√(2/π)(x+0,044715x³)))` em f32 (vmath, erro absoluto < 3e-6).
    #[default]
    Exact,
    /// Emula a tabela em f16 que o ggml usa na CPU (`GGML_GELU_FP16`, definido sem condição em
    /// `ggml-cpu/vec.h` do b12606): entrada arredondada para f16, saída também. Só para
    /// diagnóstico de paridade com o llama.cpp: reproduz o ruído dele (~5e-4 relativo por ativação).
    GgmlF16Table,
}

/// LayerNorm por linha de `n_embd`: `(x-média)/sqrt(var+eps)·w + b`, média e variância em f64.
pub fn layer_norm(x: &[f32], n_embd: usize, w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    x.par_chunks(n_embd).zip(out.par_chunks_mut(n_embd)).for_each(|(xr, or)| {
        let n = n_embd as f64;
        let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = xr.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for i in 0..n_embd {
            or[i] = ((xr[i] as f64 - mean) * inv) as f32 * w[i] + b[i];
        }
    });
}

static ONES: [f32; 4096] = [1.0; 4096];

/// GELU (tangente hiperbólica) no lugar.
pub fn gelu(x: &mut [f32], mode: GeluMode) {
    match mode {
        GeluMode::Exact => x.par_chunks_mut(4096).for_each(|c| vmath::gelu_mul(c, &ONES[..c.len()])),
        GeluMode::GgmlF16Table => x.par_iter_mut().for_each(|v| *v = gelu_ggml_f16(*v)),
    }
}

/// Um valor do GELU como o ggml da CPU o calcula com a tabela f16.
pub fn gelu_ggml_f16(x: f32) -> f32 {
    use crate::native::{f16::f32_to_f16, quant::f16_to_f32};
    if x <= -10.0 {
        return 0.0;
    }
    if x >= 10.0 {
        return x;
    }
    let xr = f16_to_f32(f32_to_f16(x));
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    let y = 0.5 * xr * (1.0 + (SQRT_2_OVER_PI * xr * (1.0 + 0.044715 * xr * xr)).tanh());
    f16_to_f32(f32_to_f16(y))
}

// ------------------------------------------------------------------------------ produto escalar

#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c"))
}
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// Um vetor `w` contra 4 vetores `x[t]` (mesmo tamanho). Por vetor: 16 somas parciais (duas de 8
    /// lanes, `lo`/`hi`) e redução em ordem fixa, o que dá o mesmo resultado qualquer que seja o
    /// agrupamento dos vetores em quatro.
    #[inline(always)]
    pub unsafe fn dot4(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
        let n = w.len();
        assert!(x.iter().all(|v| v.len() == n));
        let mut lo = [_mm256_setzero_ps(); 4];
        let mut hi = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 16 <= n {
            let w0 = _mm256_loadu_ps(w.as_ptr().add(i));
            let w1 = _mm256_loadu_ps(w.as_ptr().add(i + 8));
            for t in 0..4 {
                lo[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[t].as_ptr().add(i)), lo[t]);
                hi[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x[t].as_ptr().add(i + 8)), hi[t]);
            }
            i += 16;
        }
        if i + 8 <= n {
            let w0 = _mm256_loadu_ps(w.as_ptr().add(i));
            for t in 0..4 {
                lo[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[t].as_ptr().add(i)), lo[t]);
            }
            i += 8;
        }
        let mut out = [0f32; 4];
        for t in 0..4 {
            out[t] = hsum(_mm256_add_ps(lo[t], hi[t]));
        }
        while i < n {
            for t in 0..4 {
                out[t] += w[i] * x[t][i];
            }
            i += 1;
        }
        out
    }

    /// F16 -> f32 de uma linha inteira com F16C.
    #[inline(always)]
    pub unsafe fn f16_row(src: &[u8], dst: &mut [f32]) {
        let n = dst.len();
        assert!(src.len() >= 2 * n);
        let mut i = 0;
        while i + 8 <= n {
            let h = _mm_loadu_si128(src.as_ptr().add(2 * i) as *const __m128i);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
            i += 8;
        }
        while i < n {
            dst[i] = crate::native::quant::f16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]));
            i += 1;
        }
    }
}

/// Versão genérica de `x86::dot4` (mesma árvore de somas, sem FMA).
#[inline(always)]
fn dot4_generic(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    let n = w.len();
    assert!(x.iter().all(|v| v.len() == n));
    let mut out = [0f32; 4];
    for t in 0..4 {
        let (mut lo, mut hi) = ([0f32; 8], [0f32; 8]);
        let mut i = 0;
        while i + 16 <= n {
            for j in 0..8 {
                lo[j] += w[i + j] * x[t][i + j];
                hi[j] += w[i + 8 + j] * x[t][i + 8 + j];
            }
            i += 16;
        }
        if i + 8 <= n {
            for j in 0..8 {
                lo[j] += w[i + j] * x[t][i + j];
            }
            i += 8;
        }
        let v: [f32; 8] = std::array::from_fn(|j| lo[j] + hi[j]);
        let a = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
        let b = [a[0] + a[2], a[1] + a[3]];
        let mut s = b[0] + b[1];
        while i < n {
            s += w[i] * x[t][i];
            i += 1;
        }
        out[t] = s;
    }
    out
}

#[inline(always)]
fn dot4<const AVX: bool>(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if AVX {
            // SAFETY: só é chamada de dentro de funções com `target_feature(avx2,fma,f16c)`, depois de `has_avx2()`.
            return unsafe { x86::dot4(w, x) };
        }
    }
    dot4_generic(w, x)
}

#[inline(always)]
fn row_to_f32<const AVX: bool>(dtype: Dtype, bytes: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if AVX && dtype == Dtype::F16 {
            // SAFETY: idem `dot4`.
            return unsafe { x86::f16_row(bytes, out) };
        }
    }
    dequant_row(dtype, bytes, out)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn linear_block<const AVX: bool>(wbytes: &[u8], dtype: Dtype, rows: usize, cols: usize, rb: usize, xc: &[f32], bias: Option<&[f32]>, oc: &mut [f32]) {
    let nt = xc.len() / cols;
    let mut wf = vec![0f32; cols];
    for o in 0..rows {
        row_to_f32::<AVX>(dtype, &wbytes[o * rb..(o + 1) * rb], &mut wf);
        let bi = bias.map_or(0.0, |b| b[o]);
        let mut t = 0;
        while t < nt {
            let m = (nt - t).min(4);
            let xs: [&[f32]; 4] = std::array::from_fn(|j| {
                let tj = t + j.min(m - 1);
                &xc[tj * cols..(tj + 1) * cols]
            });
            let r = dot4::<AVX>(&wf, xs);
            for j in 0..m {
                oc[(t + j) * rows + o] = r[j] + bi;
            }
            t += m;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn linear_block_avx2(wbytes: &[u8], dtype: Dtype, rows: usize, cols: usize, rb: usize, xc: &[f32], bias: Option<&[f32]>, oc: &mut [f32]) {
    linear_block::<true>(wbytes, dtype, rows, cols, rb, xc, bias, oc)
}

/// `out[t][o] = Σ_i W[o][i]·x[t][i] + bias[o]` para `n` vetores `x` (cada um com `cols` valores).
/// `wbytes`: `rows` linhas de pesos, `dtype` qualquer um do motor (F32, F16, Q8_0, Q4_K...).
/// Blocos de 32 vetores por tarefa: cada linha de pesos é convertida para f32 **uma vez por bloco** e
/// usada contra os 32 vetores (4 de cada vez, 8 acumuladores AVX2 independentes).
#[allow(clippy::too_many_arguments)]
pub fn linear(wbytes: &[u8], dtype: Dtype, rows: usize, cols: usize, x: &[f32], n: usize, bias: Option<&[f32]>, out: &mut [f32]) {
    linear_isa(has_avx2(), wbytes, dtype, rows, cols, x, n, bias, out)
}

/// Como `linear`, com o caminho escolhido (os testes cobrem o genérico e o AVX2 na mesma máquina).
#[allow(clippy::too_many_arguments)]
pub fn linear_isa(avx: bool, wbytes: &[u8], dtype: Dtype, rows: usize, cols: usize, x: &[f32], n: usize, bias: Option<&[f32]>, out: &mut [f32]) {
    assert_eq!(x.len(), n * cols);
    assert_eq!(out.len(), n * rows);
    let (be, bb) = dtype.block();
    assert!(cols % be == 0 && wbytes.len() >= rows * (cols / be * bb));
    let rb = cols / be * bb;
    const TB: usize = 32;
    out.par_chunks_mut(TB * rows).zip(x.par_chunks(TB * cols)).for_each(|(oc, xc)| {
        #[cfg(target_arch = "x86_64")]
        {
            if avx {
                // SAFETY: `avx` só é verdadeiro quando `has_avx2()` confirmou AVX2, FMA e F16C.
                return unsafe { linear_block_avx2(wbytes, dtype, rows, cols, rb, xc, bias, oc) };
            }
        }
        let _ = avx;
        linear_block::<false>(wbytes, dtype, rows, cols, rb, xc, bias, oc)
    });
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn attn_token<const AVX: bool>(q: &[f32], k: &[f32], v: &[f32], t: usize, n: usize, n_head: usize, d: usize, scale: f32, o: &mut [f32], sc: &mut [f32]) {
    let dim = n_head * d;
    for h in 0..n_head {
        let qh = &q[t * dim + h * d..t * dim + (h + 1) * d];
        let key = |s: usize| &k[s * dim + h * d..s * dim + (h + 1) * d];
        let mut s = 0;
        while s < n {
            let m = (n - s).min(4);
            let r = dot4::<AVX>(qh, std::array::from_fn(|j| key(s + j.min(m - 1))));
            for j in 0..m {
                sc[s + j] = r[j] * scale;
            }
            s += m;
        }
        vmath::softmax(&mut sc[..n]);
        let oh = &mut o[h * d..(h + 1) * d];
        oh.fill(0.0);
        for s in 0..n {
            let p = sc[s];
            let vh = &v[s * dim + h * d..s * dim + (h + 1) * d];
            for i in 0..d {
                oh[i] += p * vh[i];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn attn_token_avx2(q: &[f32], k: &[f32], v: &[f32], t: usize, n: usize, n_head: usize, d: usize, scale: f32, o: &mut [f32], sc: &mut [f32]) {
    attn_token::<true>(q, k, v, t, n, n_head, d, scale, o, sc)
}

/// Atenção bidirecional (sem máscara, como no SigLIP): para cada token e cabeça,
/// `softmax(q·Kᵀ/√d)·V`. `q`, `k`, `v`, `out`: `n × (n_head·d)`, cabeça `h` em `[h·d, (h+1)·d)`.
/// `scale` é `1/√d` (parâmetro só para que os testes com dentes possam plantar um fator errado).
#[allow(clippy::too_many_arguments)]
pub fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, n_head: usize, d: usize, scale: f32, out: &mut [f32]) {
    attention_isa(has_avx2(), q, k, v, n, n_head, d, scale, out)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_isa(avx: bool, q: &[f32], k: &[f32], v: &[f32], n: usize, n_head: usize, d: usize, scale: f32, out: &mut [f32]) {
    let dim = n_head * d;
    assert!(q.len() == n * dim && k.len() == n * dim && v.len() == n * dim && out.len() == n * dim);
    out.par_chunks_mut(dim).enumerate().for_each(|(t, o)| {
        let mut sc = vec![0f32; n];
        #[cfg(target_arch = "x86_64")]
        {
            if avx {
                // SAFETY: `avx` só é verdadeiro quando `has_avx2()` confirmou AVX2, FMA e F16C.
                return unsafe { attn_token_avx2(q, k, v, t, n, n_head, d, scale, o, &mut sc) };
            }
        }
        let _ = avx;
        attn_token::<false>(q, k, v, t, n, n_head, d, scale, o, &mut sc)
    });
}

/// Pixel shuffle do Idefics3 (`build_patch_merge_permute` do llama.cpp, `pixel_shuffle` do HF):
/// `x` tem `side×side` vetores de `c` valores (índice `y·side+x`); a saída tem
/// `(side/s)²` vetores de `c·s²` valores, índice `yo·(side/s)+xo`, e o valor `e` do patch
/// `(yo·s+dy, xo·s+dx)` vai para a posição `(dy·s+dx)·c + e`.
pub fn pixel_shuffle(x: &[f32], side: usize, c: usize, s: usize) -> Vec<f32> {
    assert_eq!(x.len(), side * side * c);
    assert!(s >= 1 && side % s == 0);
    let so = side / s;
    let mut out = vec![0f32; so * so * c * s * s];
    for yo in 0..so {
        for xo in 0..so {
            let dst = (yo * so + xo) * c * s * s;
            for dy in 0..s {
                for dx in 0..s {
                    let src = ((yo * s + dy) * side + (xo * s + dx)) * c;
                    let d0 = dst + (dy * s + dx) * c;
                    out[d0..d0 + c].copy_from_slice(&x[src..src + c]);
                }
            }
        }
    }
    out
}
