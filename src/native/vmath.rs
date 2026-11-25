//! Matemática vetorizada fora das multiplicações de pesos: `exp` (softmax e silu) e a atenção por
//! grupo de cabeças. AVX2/F16C escolhidos em tempo de execução; versão escalar nas demais CPUs.
//! A conta é a mesma em todos os caminhos de chamada (um token ou lote), então os resultados são
//! idênticos entre eles (os testes exigem).

use super::quant::f16_to_f32;

/// Maior `head_dim` suportado pela atenção agrupada (buffers na pilha).
pub const MAX_HEAD_DIM: usize = 256;

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
    })
}

// ------------------------------------------------------------------------------ exp

/// exp(x) em AVX2: x = n·ln2 + r, polinômio de grau 7 em r (|r| ≤ 0.35, erro relativo ~2e-7) e
/// escala por 2^n pelo expoente. Satura em [-87, 88] (sem infinito nem NaN).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn exp256(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0)), _mm256_set1_ps(88.0));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)));
    // r = x - n·ln2, com ln2 dividido em duas partes para não perder precisão
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_375), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
    let mut p = _mm256_set1_ps(1.0 / 5040.0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 720.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 120.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
    let e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    _mm256_mul_ps(p, _mm256_castsi256_ps(e))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hmax(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// Softmax in-place (numericamente estável).
#[allow(dead_code)] // o caminho de produção usa `softmax_avx2` dentro da atenção; esta serve de API e de teste
pub fn softmax(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: só chamamos depois de confirmar AVX2, FMA e F16C.
            return unsafe { softmax_avx2(v) };
        }
    }
    softmax_scalar(v)
}

fn softmax_scalar(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn softmax_avx2(v: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = v.len();
    if n == 0 {
        return;
    }
    let mut mv = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        mv = _mm256_max_ps(mv, _mm256_loadu_ps(v.as_ptr().add(i)));
        i += 8;
    }
    let mut max = hmax(mv);
    for &x in &v[i..] {
        max = max.max(x);
    }
    let maxv = _mm256_set1_ps(max);
    let mut sv = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(v.as_ptr().add(i)), maxv));
        _mm256_storeu_ps(v.as_mut_ptr().add(i), e);
        sv = _mm256_add_ps(sv, e);
        i += 8;
    }
    let mut sum = hsum(sv);
    if i < n {
        // resto (< 8): passa por um vetor completo para usar o mesmo exp
        let mut tmp = [max; 8];
        tmp[..n - i].copy_from_slice(&v[i..]);
        let mut out = [0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), exp256(_mm256_sub_ps(_mm256_loadu_ps(tmp.as_ptr()), maxv)));
        for k in 0..n - i {
            v[i + k] = out[k];
            sum += out[k];
        }
    }
    let inv = _mm256_set1_ps(1.0 / sum);
    i = 0;
    while i + 8 <= n {
        _mm256_storeu_ps(v.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_loadu_ps(v.as_ptr().add(i)), inv));
        i += 8;
    }
    let invs = 1.0 / sum;
    for x in &mut v[i..] {
        *x *= invs;
    }
}

/// g[i] = silu(g[i]) · u[i], com silu(g) = g / (1 + exp(-g)).
pub fn silu_mul(g: &mut [f32], u: &[f32]) {
    debug_assert_eq!(g.len(), u.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: só chamamos depois de confirmar AVX2, FMA e F16C.
            return unsafe { silu_mul_avx2(g, u) };
        }
    }
    for (g, u) in g.iter_mut().zip(u) {
        *g = (*g / (1.0 + (-*g).exp())) * u;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn silu_mul_avx2(g: &mut [f32], u: &[f32]) {
    use std::arch::x86_64::*;
    let n = g.len().min(u.len());
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(g.as_ptr().add(i));
        let s = _mm256_div_ps(x, _mm256_add_ps(one, exp256(_mm256_sub_ps(zero, x))));
        _mm256_storeu_ps(g.as_mut_ptr().add(i), _mm256_mul_ps(s, _mm256_loadu_ps(u.as_ptr().add(i))));
        i += 8;
    }
    if i < n {
        let mut tg = [0f32; 8];
        let mut tu = [0f32; 8];
        tg[..n - i].copy_from_slice(&g[i..n]);
        tu[..n - i].copy_from_slice(&u[i..n]);
        let x = _mm256_loadu_ps(tg.as_ptr());
        let s = _mm256_div_ps(x, _mm256_add_ps(one, exp256(_mm256_sub_ps(zero, x))));
        let mut out = [0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_mul_ps(s, _mm256_loadu_ps(tu.as_ptr())));
        g[i..n].copy_from_slice(&out[..n - i]);
    }
}

/// g[i] = gelu(g[i]) · u[i] (GeGLU do Gemma), com o GELU de tangente hiperbólica
/// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, que é igual a `x / (1 + exp(-2·√(2/π)·(x + 0.044715·x³)))`.
pub fn gelu_mul(g: &mut [f32], u: &[f32]) {
    debug_assert_eq!(g.len(), u.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: só chamamos depois de confirmar AVX2, FMA e F16C.
            return unsafe { gelu_mul_avx2(g, u) };
        }
    }
    for (g, u) in g.iter_mut().zip(u) {
        *g = gelu_scalar(*g) * u;
    }
}

fn gelu_scalar(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // √(2/π)
    x / (1.0 + (-2.0 * C * (x + 0.044_715 * x * x * x)).exp())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gelu_mul_avx2(g: &mut [f32], u: &[f32]) {
    use std::arch::x86_64::*;
    let n = g.len().min(u.len());
    let one = _mm256_set1_ps(1.0);
    let k = _mm256_set1_ps(-2.0 * 0.797_884_56);
    let c3 = _mm256_set1_ps(0.044_715);
    let gelu = |x: __m256| -> __m256 {
        // y = -2·√(2/π)·(x + 0.044715·x³)
        let y = _mm256_mul_ps(k, _mm256_fmadd_ps(c3, _mm256_mul_ps(x, _mm256_mul_ps(x, x)), x));
        _mm256_div_ps(x, _mm256_add_ps(one, exp256(y)))
    };
    let mut i = 0;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(g.as_ptr().add(i));
        _mm256_storeu_ps(g.as_mut_ptr().add(i), _mm256_mul_ps(gelu(x), _mm256_loadu_ps(u.as_ptr().add(i))));
        i += 8;
    }
    if i < n {
        let (mut tg, mut tu) = ([0f32; 8], [0f32; 8]);
        tg[..n - i].copy_from_slice(&g[i..n]);
        tu[..n - i].copy_from_slice(&u[i..n]);
        let mut out = [0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), _mm256_mul_ps(gelu(_mm256_loadu_ps(tg.as_ptr())), _mm256_loadu_ps(tu.as_ptr())));
        g[i..n].copy_from_slice(&out[..n - i]);
    }
}

// -------------------------------------------------------------------------- atenção

/// Atenção de UM token para o grupo de `group` cabeças de consulta que compartilham a cabeça de KV
/// `g` (GQA). `q`: group·hd (as cabeças do grupo são contíguas); `kc`/`vc`: cache em f16 desta
/// camada (posição t, cabeça g em `t·kv_dim + g·hd`); `pos`: posição do token (enxerga 0..=pos);
/// `scores`: espaço de trabalho de pelo menos group·(pos+1); `out`: group·hd.
/// Cada K e cada V é lido e convertido para f32 UMA vez para todo o grupo.
#[allow(clippy::too_many_arguments)]
pub fn attend_group(
    q: &[f32],
    kc: &[u16],
    vc: &[u16],
    pos: usize,
    g: usize,
    group: usize,
    hd: usize,
    kv_dim: usize,
    scale: f32,
    scores: &mut [f32],
    out: &mut [f32],
) {
    assert!(hd <= MAX_HEAD_DIM && hd % 2 == 0);
    assert!(q.len() >= group * hd && out.len() >= group * hd && scores.len() >= group * (pos + 1));
    assert!(kc.len() >= pos * kv_dim + (g + 1) * hd && vc.len() >= pos * kv_dim + (g + 1) * hd);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && hd % 8 == 0 {
            // SAFETY: só chamamos depois de confirmar AVX2, FMA e F16C; os limites foram conferidos acima.
            return unsafe { attend_group_avx2(q, kc, vc, pos, g, group, hd, kv_dim, scale, scores, out) };
        }
    }
    attend_group_scalar(q, kc, vc, pos, g, group, hd, kv_dim, scale, scores, out)
}

#[allow(clippy::too_many_arguments)]
fn attend_group_scalar(q: &[f32], kc: &[u16], vc: &[u16], pos: usize, g: usize, group: usize, hd: usize, kv_dim: usize, scale: f32, scores: &mut [f32], out: &mut [f32]) {
    let np = pos + 1;
    let mut buf = [0f32; MAX_HEAD_DIM];
    for t in 0..np {
        for (b, &k) in buf[..hd].iter_mut().zip(&kc[t * kv_dim + g * hd..t * kv_dim + (g + 1) * hd]) {
            *b = f16_to_f32(k);
        }
        for j in 0..group {
            let mut s = 0.0;
            for (a, b) in q[j * hd..(j + 1) * hd].iter().zip(&buf[..hd]) {
                s += a * b;
            }
            scores[j * np + t] = s * scale;
        }
    }
    for j in 0..group {
        softmax_scalar(&mut scores[j * np..(j + 1) * np]);
    }
    out[..group * hd].fill(0.0);
    for t in 0..np {
        for (b, &v) in buf[..hd].iter_mut().zip(&vc[t * kv_dim + g * hd..t * kv_dim + (g + 1) * hd]) {
            *b = f16_to_f32(v);
        }
        for j in 0..group {
            let p = scores[j * np + t];
            for (o, b) in out[j * hd..(j + 1) * hd].iter_mut().zip(&buf[..hd]) {
                *o += p * b;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn attend_group_avx2(q: &[f32], kc: &[u16], vc: &[u16], pos: usize, g: usize, group: usize, hd: usize, kv_dim: usize, scale: f32, scores: &mut [f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let np = pos + 1;
    let mut buf = [0f32; MAX_HEAD_DIM];
    let load = |src: *const u16, dst: &mut [f32; MAX_HEAD_DIM]| {
        let mut c = 0;
        while c < hd {
            _mm256_storeu_ps(dst.as_mut_ptr().add(c), _mm256_cvtph_ps(_mm_loadu_si128(src.add(c) as *const __m128i)));
            c += 8;
        }
    };
    // 1) pontuações: cada K é convertido uma vez e multiplicado pelas cabeças do grupo
    for t in 0..np {
        load(kc.as_ptr().add(t * kv_dim + g * hd), &mut buf);
        for j in 0..group {
            let qj = q.as_ptr().add(j * hd);
            let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let mut c = 0;
            while c + 16 <= hd {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qj.add(c)), _mm256_loadu_ps(buf.as_ptr().add(c)), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(qj.add(c + 8)), _mm256_loadu_ps(buf.as_ptr().add(c + 8)), a1);
                c += 16;
            }
            if c < hd {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qj.add(c)), _mm256_loadu_ps(buf.as_ptr().add(c)), a0);
            }
            *scores.get_unchecked_mut(j * np + t) = hsum(_mm256_add_ps(a0, a1)) * scale;
        }
    }
    // 2) softmax de cada cabeça
    for j in 0..group {
        softmax_avx2(&mut scores[j * np..(j + 1) * np]);
    }
    // 3) soma ponderada dos V (cada V convertido uma vez)
    out[..group * hd].fill(0.0);
    for t in 0..np {
        load(vc.as_ptr().add(t * kv_dim + g * hd), &mut buf);
        for j in 0..group {
            let p = _mm256_set1_ps(*scores.get_unchecked(j * np + t));
            let o = out.as_mut_ptr().add(j * hd);
            let mut c = 0;
            while c < hd {
                _mm256_storeu_ps(o.add(c), _mm256_fmadd_ps(p, _mm256_loadu_ps(buf.as_ptr().add(c)), _mm256_loadu_ps(o.add(c))));
                c += 8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::f16::f32_to_f16;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// O exp vetorizado fica a ≤ 4e-7 (relativo) do exp da biblioteca padrão em toda a faixa útil.
    #[test]
    fn exp_accuracy() {
        let xs: Vec<f32> = (0..20000).map(|i| -86.0 + (i as f32) * (86.0 + 20.0) / 20000.0).collect();
        let mut g = xs.clone();
        let ones = vec![1.0f32; xs.len()];
        // silu(g)·1 = g/(1+exp(-g)); compara com a conta escalar
        silu_mul(&mut g, &ones);
        for (x, y) in xs.iter().zip(&g) {
            let want = x / (1.0 + (-x).exp());
            assert!((y - want).abs() <= 4e-7 * want.abs() + 1e-30, "silu({x}) = {y}, esperado {want}");
        }
        // softmax de vários tamanhos (inclui o resto < 8)
        let mut s = 3u64;
        for n in [1usize, 5, 8, 13, 64, 77, 301] {
            let v: Vec<f32> = (0..n).map(|_| lcg(&mut s) * 30.0).collect();
            let mut a = v.clone();
            softmax(&mut a);
            let mut b = v.clone();
            softmax_scalar(&mut b);
            let sum: f32 = a.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "n={n}: soma {sum}");
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() <= 1e-6 * y.abs() + 1e-9, "n={n}: {x} vs {y}");
            }
        }
        // valores extremos não geram inf/NaN
        let mut e = vec![-1e30f32, 0.0, 1e30, -1e9];
        softmax(&mut e);
        assert!(e.iter().all(|x| x.is_finite()));
    }

    /// O GELU vetorizado coincide com a fórmula de referência (tanh) em toda a faixa útil.
    #[test]
    fn gelu_accuracy() {
        let xs: Vec<f32> = (0..20003).map(|i| -30.0 + (i as f32) * 60.0 / 20003.0).collect();
        let mut g = xs.clone();
        gelu_mul(&mut g, &vec![1.0; xs.len()]);
        for (x, y) in xs.iter().zip(&g) {
            let want = 0.5 * x * (1.0 + (0.797_884_56f32 * (x + 0.044_715 * x * x * x)).tanh());
            assert!((y - want).abs() <= 2e-6 * want.abs().max(1e-3) + 1e-7, "gelu({x}) = {y}, esperado {want}");
        }
    }

    /// A atenção agrupada (AVX2) coincide com a escalar, para vários grupos, posições e head_dim.
    #[test]
    fn attention_matches_scalar() {
        let mut s = 11u64;
        for (group, hd, pos) in [(4usize, 64usize, 37usize), (7, 64, 100), (3, 128, 20), (6, 64, 0), (1, 64, 5)] {
            let (n_kv, g) = (3usize, 1usize);
            let kv_dim = n_kv * hd;
            let ctx = pos + 4;
            let kc: Vec<u16> = (0..ctx * kv_dim).map(|_| f32_to_f16(lcg(&mut s) * 2.0)).collect();
            let vc: Vec<u16> = (0..ctx * kv_dim).map(|_| f32_to_f16(lcg(&mut s))).collect();
            let q: Vec<f32> = (0..group * hd).map(|_| lcg(&mut s)).collect();
            let scale = 1.0 / (hd as f32).sqrt();
            let mut sc1 = vec![0f32; group * (pos + 1)];
            let mut sc2 = sc1.clone();
            let (mut o1, mut o2) = (vec![0f32; group * hd], vec![0f32; group * hd]);
            attend_group(&q, &kc, &vc, pos, g, group, hd, kv_dim, scale, &mut sc1, &mut o1);
            attend_group_scalar(&q, &kc, &vc, pos, g, group, hd, kv_dim, scale, &mut sc2, &mut o2);
            for (a, b) in o1.iter().zip(&o2) {
                assert!((a - b).abs() <= 2e-5 * (1.0 + b.abs()), "grupo={group} hd={hd} pos={pos}: {a} vs {b}");
            }
        }
    }
}
