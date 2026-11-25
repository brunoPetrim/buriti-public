//! Passagem direta da torre de visão SigLIP + conector Idefics3, em `f32`.
//!
//! Entrada: pixels **já normalizados** `(v/255 - média)/desvio`, `image_size × image_size × 3`,
//! intercalados (`(y·W + x)·3 + canal`). Saída: `n_tokens × proj_dim` valores, prontos para entrar
//! como embeddings no modelo de linguagem (a injeção ainda não existe; ver docs/VISAO-MOTOR-PROPRIO.md).
//!
//! A ordem das etapas e os nomes de cada estágio seguem o grafo `build_vit` do llama.cpp b12606
//! (`pos_embed`, `layer_inp_normed-N`, `Qcur-N`, `attn_out-N`, `ffn_inp-N`, `ffn_out-N`, `layer_out-N`,
//! `pixel_shuffle`), de modo que o dono possa comparar estágio a estágio com o `llama-mtmd-debug`.

use super::{
    mmproj::Mmproj,
    ops::{self, GeluMode},
};

/// Defeitos que os testes plantam de propósito para provar que cada teste tem dentes.
/// Nunca são usados em produção (`Bug::None` é o padrão).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bug {
    #[default]
    None,
    /// atenção com 1/d em vez de 1/√d
    WrongAttnScale,
    /// esquece a soma da posição aprendida
    NoPositionEmbedding,
    /// LayerNorm sem o viés
    NoNormBias,
    /// pixel shuffle com (dx, dy) trocados
    ShuffleSwapped,
    /// usa K no lugar de V na soma ponderada da atenção
    UseKAsV,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeOpts {
    pub gelu: GeluMode,
    pub bug: Bug,
}

/// Callback de estágio: nome (no estilo do llama.cpp) e os valores (linhas de `n_embd` por token).
pub type Sink<'a> = &'a mut dyn FnMut(&str, &[f32]);

/// `pixels` normalizados `image_size²×3`. Devolve `n_tokens × proj_dim`.
pub fn encode(m: &Mmproj, pixels: &[f32], opts: EncodeOpts, mut sink: Option<Sink>) -> Result<Vec<f32>, String> {
    let hp = &m.hp;
    let (n, side, p) = (hp.n_embd, hp.side(), hp.patch_size);
    let np = hp.n_patches();
    if pixels.len() != hp.image_size * hp.image_size * 3 {
        return Err(format!("imagem com {} valores; esperado {}×{}×3 = {}", pixels.len(), hp.image_size, hp.image_size, hp.image_size * hp.image_size * 3));
    }
    if pixels.iter().any(|v| !v.is_finite()) {
        return Err("imagem com valores não finitos".into());
    }
    let has_sink = sink.is_some();
    let mut emit = |name: &str, d: &[f32]| {
        if let Some(s) = sink.as_mut() {
            (**s)(name, d);
        }
    };

    // Entrada "bruta" no estilo do llama.cpp (planar: x mais rápido, depois y, depois o canal).
    if has_sink {
        let mut planar = vec![0f32; pixels.len()];
        let w = hp.image_size;
        for c in 0..3 {
            for y in 0..w {
                for x in 0..w {
                    planar[(c * w + y) * w + x] = pixels[(y * w + x) * 3 + c];
                }
            }
        }
        emit("inp_raw", &planar);
    }

    // 1. patches (im2col: coluna = (canal, y, x), a mesma ordem dos pesos da convolução)
    let k = 3 * p * p;
    let mut cols = vec![0f32; np * k];
    for py in 0..side {
        for px in 0..side {
            let dst = &mut cols[(py * side + px) * k..(py * side + px + 1) * k];
            for c in 0..3 {
                for ky in 0..p {
                    for kx in 0..p {
                        dst[(c * p + ky) * p + kx] = pixels[((py * p + ky) * hp.image_size + px * p + kx) * 3 + c];
                    }
                }
            }
        }
    }
    let mut x = vec![0f32; np * n];
    ops::linear(m.bytes(&m.patch_w), m.patch_w.dtype, n, k, &cols, np, Some(&m.patch_b), &mut x);
    drop(cols);
    emit("patch_bias", &x);

    // 2. posição aprendida
    if opts.bug != Bug::NoPositionEmbedding {
        for (a, b) in x.iter_mut().zip(&m.pos) {
            *a += *b;
        }
    }
    emit("pos_embed", &x);

    // 3. camadas
    let mut normed = vec![0f32; np * n];
    let mut q = vec![0f32; np * n];
    let mut kk = vec![0f32; np * n];
    let mut v = vec![0f32; np * n];
    let mut att = vec![0f32; np * n];
    let mut proj = vec![0f32; np * n];
    let mut hid = vec![0f32; np * hp.n_ff];
    let mut ffn_out = vec![0f32; np * n];
    let scale = if opts.bug == Bug::WrongAttnScale { 1.0 / hp.head_dim() as f32 } else { 1.0 / (hp.head_dim() as f32).sqrt() };
    for (il, l) in m.layers.iter().enumerate() {
        let nm = |s: &str| format!("{s}-{il}");
        norm(&x, &l.ln1, hp.eps, opts, &mut normed);
        emit(&nm("layer_inp_normed"), &normed);
        lin(m, &l.q, &normed, np, &mut q);
        lin(m, &l.k, &normed, np, &mut kk);
        lin(m, &l.v, &normed, np, &mut v);
        emit(&nm("Qcur"), &q);
        emit(&nm("Kcur"), &kk);
        emit(&nm("Vcur"), &v);
        let vsrc = if opts.bug == Bug::UseKAsV { &kk } else { &v };
        ops::attention(&q, &kk, vsrc, np, hp.n_head, hp.head_dim(), scale, &mut att);
        emit(&nm("kqv_out"), &att);
        lin(m, &l.o, &att, np, &mut proj);
        emit(&nm("attn_out"), &proj);
        for (a, b) in x.iter_mut().zip(&proj) {
            *a += *b;
        }
        emit(&nm("ffn_inp"), &x);
        norm(&x, &l.ln2, hp.eps, opts, &mut normed);
        emit(&nm("ffn_inp_normed"), &normed);
        lin(m, &l.up, &normed, np, &mut hid);
        ops::gelu(&mut hid, opts.gelu);
        lin(m, &l.down, &hid, np, &mut ffn_out);
        emit(&nm("ffn_out"), &ffn_out);
        for (a, b) in x.iter_mut().zip(&ffn_out) {
            *a += *b;
        }
        emit(&nm("layer_out"), &x);
    }

    // 4. norma final
    if let Some(pl) = &m.post_ln {
        norm_with(&x, &pl.w, &pl.b, hp.eps, opts, &mut normed);
        std::mem::swap(&mut x, &mut normed);
    }
    emit("post_ln", &x);

    // 5. conector Idefics3: pixel shuffle e projeção linear sem viés
    let s = hp.scale_factor;
    let shuffled = if opts.bug == Bug::ShuffleSwapped { shuffle_swapped(&x, side, n, s) } else { ops::pixel_shuffle(&x, side, n, s) };
    emit("pixel_shuffle", &shuffled);
    let nt = hp.n_tokens();
    let mut out = vec![0f32; nt * hp.proj_dim];
    ops::linear(m.bytes(&m.fc), m.fc.dtype, hp.proj_dim, n * s * s, &shuffled, nt, None, &mut out);
    emit("mm_out", &out);
    if out.iter().any(|v| !v.is_finite()) {
        return Err("a torre de visão produziu valores não finitos".into());
    }
    Ok(out)
}

fn lin(m: &Mmproj, l: &super::mmproj::Linear, x: &[f32], n: usize, out: &mut [f32]) {
    ops::linear(m.bytes(&l.w), l.w.dtype, l.w.rows, l.w.cols, x, n, l.b.as_deref(), out);
}

fn norm(x: &[f32], nm: &super::mmproj::Norm, eps: f32, opts: EncodeOpts, out: &mut [f32]) {
    norm_with(x, &nm.w, &nm.b, eps, opts, out)
}

fn norm_with(x: &[f32], w: &[f32], b: &[f32], eps: f32, opts: EncodeOpts, out: &mut [f32]) {
    if opts.bug == Bug::NoNormBias {
        let z = vec![0f32; b.len()];
        ops::layer_norm(x, w.len(), w, &z, eps, out);
    } else {
        ops::layer_norm(x, w.len(), w, b, eps, out);
    }
}

/// Defeito plantado: (dx, dy) trocados.
fn shuffle_swapped(x: &[f32], side: usize, c: usize, s: usize) -> Vec<f32> {
    let so = side / s;
    let mut out = vec![0f32; so * so * c * s * s];
    for yo in 0..so {
        for xo in 0..so {
            for dy in 0..s {
                for dx in 0..s {
                    let src = ((yo * s + dy) * side + (xo * s + dx)) * c;
                    let d0 = (yo * so + xo) * c * s * s + (dx * s + dy) * c;
                    out[d0..d0 + c].copy_from_slice(&x[src..src + c]);
                }
            }
        }
    }
    out
}

/// `(v/255 - média)/desvio` por canal, de bytes RGB intercalados (o que o navegador envia).
pub fn normalize_u8(rgb: &[u8], mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    rgb.iter().enumerate().map(|(i, &b)| (b as f32 / 255.0 - mean[i % 3]) / std[i % 3]).collect()
}
