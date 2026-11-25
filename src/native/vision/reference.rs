//! Referência **ingênua em f64** da torre de visão (só em testes). Laços literais, sem blocagem, sem
//! SIMD, sem as funções de `ops.rs`: o `exp`/`tanh` são os exatos do f64, o softmax é a definição, a
//! convolução do patch é o laço de 7 níveis e o pixel shuffle é o `reshape`/`permute` do grafo do
//! llama.cpp executado literalmente. Os pesos vêm do mesmo mmap (a desquantização é testada à parte em
//! `quant.rs`), então o que esta referência confere é a **montagem** (ordem, índices, normas, viés,
//! residuais), não a leitura dos bytes.

use rayon::prelude::*;

use super::mmproj::{Layer, Linear, Mat, Mmproj, Norm};
use crate::native::quant::dequant_row;

pub fn dequant_mat(m: &Mmproj, mt: &Mat) -> Vec<f64> {
    let (be, bb) = mt.dtype.block();
    let rb = mt.cols / be * bb;
    let bytes = m.bytes(mt);
    let mut out = Vec::with_capacity(mt.rows * mt.cols);
    let mut row = vec![0f32; mt.cols];
    for r in 0..mt.rows {
        dequant_row(mt.dtype, &bytes[r * rb..(r + 1) * rb], &mut row);
        out.extend(row.iter().map(|&v| v as f64));
    }
    out
}

fn lin(m: &Mmproj, l: &Linear, x: &[f64], n: usize) -> Vec<f64> {
    let w = dequant_mat(m, &l.w);
    let (rows, cols) = (l.w.rows, l.w.cols);
    let mut out = vec![0f64; n * rows];
    out.par_chunks_mut(rows).enumerate().for_each(|(t, o)| {
        for r in 0..rows {
            let mut s = l.b.as_ref().map_or(0.0, |b| b[r] as f64);
            for i in 0..cols {
                s += w[r * cols + i] * x[t * cols + i];
            }
            o[r] = s;
        }
    });
    out
}

fn ln(x: &[f64], nm: &Norm, eps: f32) -> Vec<f64> {
    let n = nm.w.len();
    let mut out = vec![0f64; x.len()];
    for (xr, or) in x.chunks(n).zip(out.chunks_mut(n)) {
        let mean = xr.iter().sum::<f64>() / n as f64;
        let var = xr.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        for i in 0..n {
            or[i] = (xr[i] - mean) / (var + eps as f64).sqrt() * nm.w[i] as f64 + nm.b[i] as f64;
        }
    }
    out
}

pub fn gelu64(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn attn(q: &[f64], k: &[f64], v: &[f64], n: usize, nh: usize, d: usize) -> Vec<f64> {
    let dim = nh * d;
    let mut out = vec![0f64; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(t, o)| {
        for h in 0..nh {
            let mut sc = vec![0f64; n];
            for s in 0..n {
                let mut a = 0.0;
                for i in 0..d {
                    a += q[t * dim + h * d + i] * k[s * dim + h * d + i];
                }
                sc[s] = a / (d as f64).sqrt();
            }
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let es: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
            let z: f64 = es.iter().sum();
            for i in 0..d {
                let mut a = 0.0;
                for s in 0..n {
                    a += es[s] / z * v[s * dim + h * d + i];
                }
                o[h * d + i] = a;
            }
        }
    });
    out
}

/// Uma camada: devolve (`attn_out`, `ffn_out`, `layer_out`) a partir da entrada `x` (`n × n_embd`).
pub fn layer(m: &Mmproj, l: &Layer, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hp = &m.hp;
    let n = x.len() / hp.n_embd;
    let h = ln(x, &l.ln1, hp.eps);
    let (q, k, v) = (lin(m, &l.q, &h, n), lin(m, &l.k, &h, n), lin(m, &l.v, &h, n));
    let a = attn(&q, &k, &v, n, hp.n_head, hp.head_dim());
    let ao = lin(m, &l.o, &a, n);
    let x1: Vec<f64> = x.iter().zip(&ao).map(|(a, b)| a + b).collect();
    let h2 = ln(&x1, &l.ln2, hp.eps);
    let mut up = lin(m, &l.up, &h2, n);
    for v in up.iter_mut() {
        *v = gelu64(*v);
    }
    let fo = lin(m, &l.down, &up, n);
    let x2: Vec<f64> = x1.iter().zip(&fo).map(|(a, b)| a + b).collect();
    (ao, fo, x2)
}

/// `permute(0,2,1,3)` do ggml num tensor de 3 dimensões `[ne0, ne1, ne2]` (ne0 mais rápida):
/// o resultado tem `[ne0, ne2, ne1]`.
pub fn swap12(d: &[f64], ne0: usize, ne1: usize, ne2: usize) -> Vec<f64> {
    let mut out = vec![0f64; d.len()];
    for i2 in 0..ne2 {
        for i1 in 0..ne1 {
            for i0 in 0..ne0 {
                out[i0 + ne0 * (i2 + ne2 * i1)] = d[i0 + ne0 * (i1 + ne1 * i2)];
            }
        }
    }
    out
}

/// `build_patch_merge_permute` do llama.cpp, passo a passo, sobre `x` (`side²` vetores de `c`).
pub fn pixel_shuffle_literal(x: &[f64], side: usize, c: usize, s: usize) -> Vec<f64> {
    // cur = reshape_3d(cur, c, width, height)         ne = [c, W, H]
    // cur = reshape_3d(cur, c*s, W/s, H)              (mesmos dados)
    let (w, h) = (side, side);
    let cur = swap12(x, c * s, w / s, h); // permute(0,2,1,3): ne = [c*s, H, W/s]
    // cont_3d(c*s*s, H/s, W/s): mesmos dados com a 1ª dimensão ampliada
    let cur = swap12(&cur, c * s * s, h / s, w / s); // permute(0,2,1,3): ne = [c*s*s, W/s, H/s]
    cur // cont_2d(c*s*s, (W/s)*(H/s)): mesmos dados
}

/// Convolução do patch (laço literal de 7 níveis) e posição aprendida: (`patch_bias`, `pos_embed`).
pub fn embed(m: &Mmproj, pixels: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let hp = &m.hp;
    let (n, side, p, w) = (hp.n_embd, hp.side(), hp.patch_size, hp.image_size);
    let np = hp.n_patches();
    // convolução do patch: laço literal; o peso (dims [p,p,3,n]) tem índice ((co·3+ci)·p+ky)·p+kx
    let pw = dequant_mat(m, &m.patch_w);
    let mut x = vec![0f64; np * n];
    for oy in 0..side {
        for ox in 0..side {
            for co in 0..n {
                let mut s = m.patch_b[co] as f64;
                for ci in 0..3 {
                    for ky in 0..p {
                        for kx in 0..p {
                            let wv = pw[((co * 3 + ci) * p + ky) * p + kx];
                            s += wv * pixels[((oy * p + ky) * w + ox * p + kx) * 3 + ci] as f64;
                        }
                    }
                }
                x[(oy * side + ox) * n + co] = s;
            }
        }
    }
    let pb = x.clone();
    for (a, b) in x.iter_mut().zip(&m.pos) {
        *a += *b as f64;
    }
    (pb, x)
}

/// Torre inteira em f64 a partir de pixels já normalizados; devolve (nome, valores) por estágio.
pub fn encode(m: &Mmproj, pixels: &[f32]) -> Vec<(String, Vec<f64>)> {
    let hp = &m.hp;
    let (n, side) = (hp.n_embd, hp.side());
    let mut st: Vec<(String, Vec<f64>)> = Vec::new();

    let (pb, pe) = embed(m, pixels);
    st.push(("patch_bias".into(), pb));
    st.push(("pos_embed".into(), pe.clone()));
    let mut x = pe;
    for (il, l) in m.layers.iter().enumerate() {
        let (ao, fo, nx) = layer(m, l, &x);
        st.push((format!("attn_out-{il}"), ao));
        st.push((format!("ffn_out-{il}"), fo));
        st.push((format!("layer_out-{il}"), nx.clone()));
        x = nx;
    }
    if let Some(pl) = &m.post_ln {
        x = ln(&x, pl, hp.eps);
    }
    st.push(("post_ln".into(), x.clone()));
    let sh = pixel_shuffle_literal(&x, side, n, hp.scale_factor);
    st.push(("pixel_shuffle".into(), sh.clone()));
    let fc = Linear { w: m.fc, b: None };
    let out = lin(m, &fc, &sh, hp.n_tokens());
    st.push(("mm_out".into(), out));
    st
}
