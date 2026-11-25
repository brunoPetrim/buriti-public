//! Imagens sintéticas determinísticas e comparação de estágios com o `llama-mtmd-debug`.
//!
//! As imagens (`rainbow`, `gray`, `cb`, ...) repetem **exatamente** as contas de `tools/mtmd/debug/mtmd-debug.cpp`
//! do llama.cpp b12606 (valores em [0,1] enviados direto à torre, sem normalização por média/desvio:
//! o `mtmd_debug_encode_image` não normaliza), para o dono rodar a mesma imagem nos dois motores.
//!
//! O formato de texto do llama.cpp (`common_debug_cb_eval`, `common/debug.cpp`): uma linha
//! `common_debug_cb_eval: <nome> = (f32) <OP>(...{...}) = {ne0, ne1, ne2, ne3}`, o tensor com 3 valores
//! no começo e no fim de cada dimensão e, no fim, `    sum = <valor>` (soma em f32 na ordem da memória).
//! `parse_llama_log` lê isso; `llama_like_dump` escreve algo equivalente. **Ainda não foi confrontado com uma
//! saída real do llama.cpp** (o dono roda esse passo; ver docs/VISAO-MOTOR-PROPRIO.md).

use std::collections::HashMap;
use std::fmt::Write as _;

/// `kind`: "rainbow", "gray", "white", "black", "red", "green", "blue", "cb". `n`: lado em pixels.
/// Retorna `n·n·3` valores intercalados (`(y·n + x)·3 + canal`).
pub fn synthetic_image(kind: &str, n: usize) -> Result<Vec<f32>, String> {
    let mut img = vec![0f32; n * n * 3];
    let set = |img: &mut Vec<f32>, y: usize, x: usize, r: f32, g: f32, b: f32| {
        img[(y * n + x) * 3] = r;
        img[(y * n + x) * 3 + 1] = g;
        img[(y * n + x) * 3 + 2] = b;
    };
    match kind {
        "black" => {}
        "white" => img.fill(1.0),
        "gray" => img.fill(0.5),
        "red" | "green" | "blue" => {
            let c = ["red", "green", "blue"].iter().position(|k| *k == kind).unwrap();
            for i in 0..n * n {
                img[i * 3 + c] = 1.0;
            }
        }
        "cb" => {
            for y in 0..n {
                for x in 0..n {
                    let v = if (x + y) % 2 == 1 { 0.0 } else { 1.0 };
                    set(&mut img, y, x, v, v, v);
                }
            }
        }
        "rainbow" => {
            let (cx, cy) = (n as f32 / 2.0, n as f32 / 2.0);
            let max_dist = (cx * cx + cy * cy).sqrt();
            for y in 0..n {
                for x in 0..n {
                    let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                    let mut hue = dy.atan2(dx) / (2.0 * 3.14159265f32);
                    if hue < 0.0 {
                        hue += 1.0;
                    }
                    let sat = ((dx * dx + dy * dy).sqrt() / max_dist).min(1.0);
                    let h6 = hue * 6.0;
                    let i6 = h6 as i32;
                    let f = h6 - i6 as f32;
                    let p = 1.0 - sat;
                    let q = 1.0 - sat * f;
                    let t = 1.0 - sat * (1.0 - f);
                    let (r, g, b) = match i6 % 6 {
                        0 => (1.0, t, p),
                        1 => (q, 1.0, p),
                        2 => (p, 1.0, t),
                        3 => (p, q, 1.0),
                        4 => (t, p, 1.0),
                        _ => (1.0, p, q),
                    };
                    set(&mut img, y, x, r, g, b);
                }
            }
        }
        // disco vermelho sobre fundo branco (raio 150/512 do lado, centrado)
        "disc" => {
            img.fill(1.0);
            let (c, r) = (n as f32 / 2.0, 150.0 * n as f32 / 512.0);
            for y in 0..n {
                for x in 0..n {
                    let (dx, dy) = (x as f32 - c, y as f32 - c);
                    if dx * dx + dy * dy < r * r {
                        set(&mut img, y, x, 1.0, 0.0, 0.0);
                    }
                }
            }
        }
        // gradiente suave (não existe no llama.cpp; serve só aos testes daqui)
        "gradient" => {
            for y in 0..n {
                for x in 0..n {
                    set(&mut img, y, x, x as f32 / n as f32, y as f32 / n as f32, 0.5 * (x + y) as f32 / n as f32);
                }
            }
        }
        _ => return Err(format!("imagem sintética desconhecida: {kind}")),
    }
    Ok(img)
}

/// Resumo de um tensor `[ne0, ne1]` (ne0 mais rápida = tamanho do vetor por token).
#[derive(Debug, Clone)]
pub struct Stat {
    pub ne: [usize; 2],
    /// soma em f32, sequencial, na ordem da memória (como o `common_debug_print_tensor`)
    pub sum: f32,
    pub sum_abs: f64,
    pub finite: bool,
}

pub fn stat(d: &[f32], ne0: usize) -> Stat {
    let mut sum = 0f32;
    let mut sa = 0f64;
    let mut finite = true;
    for &v in d {
        sum += v;
        sa += (v as f64).abs();
        finite &= v.is_finite();
    }
    Stat { ne: [ne0, d.len() / ne0.max(1)], sum, sum_abs: sa, finite }
}

/// Texto no estilo do `common_debug_cb_eval` (3 valores nas pontas de cada dimensão).
pub fn llama_like_dump(name: &str, d: &[f32], ne0: usize) -> String {
    let ne1 = d.len() / ne0;
    let mut s = String::new();
    let _ = writeln!(s, "common_debug_cb_eval: {:>24} = (f32) NAME(x{{{}}}, }}) = {{{}, {}, 1, 1}}", name, ne0, ne0, ne1);
    let n = 3usize;
    let _ = writeln!(s, "    [\n        [");
    let mut i1 = 0;
    while i1 < ne1 {
        if i1 == n && ne1 > 2 * n {
            let _ = writeln!(s, "            ..., ");
            i1 = ne1 - n;
        }
        let _ = write!(s, "            [");
        let mut i0 = 0;
        while i0 < ne0 {
            if i0 == n && ne0 > 2 * n {
                let _ = write!(s, "   ..., ");
                i0 = ne0 - n;
            }
            let _ = write!(s, "{:12.4}", d[i1 * ne0 + i0]);
            if i0 < ne0 - 1 {
                let _ = write!(s, ", ");
            }
            i0 += 1;
        }
        let _ = writeln!(s, "  ],");
        i1 += 1;
    }
    let _ = writeln!(s, "        ],\n    ]");
    let sum = stat(d, ne0).sum;
    let _ = writeln!(s, "    sum = {:.6}", sum);
    s
}

/// Lê um log do `llama-mtmd-debug` e devolve, por nome de tensor, `(ne0, ne1, sum)`. Se o mesmo nome
/// aparecer mais de uma vez (várias passagens), vale a **última** ocorrência.
pub fn parse_llama_log(text: &str) -> HashMap<String, (usize, usize, f64)> {
    let mut out = HashMap::new();
    let mut cur: Option<(String, usize, usize)> = None;
    for line in text.lines() {
        if let Some(rest) = line.split("common_debug_cb_eval:").nth(1) {
            // "<nome> = (f32) ... = {a, b, c, d}"
            let name = rest.trim_start().split(" = ").next().unwrap_or("").trim().to_string();
            let ne = rest.rsplit('{').next().and_then(|t| t.split('}').next()).map(|t| t.split(',').filter_map(|v| v.trim().parse::<usize>().ok()).collect::<Vec<_>>());
            cur = match ne {
                Some(v) if v.len() == 4 => Some((name, v[0], v[1] * v[2] * v[3])),
                _ => None,
            };
        } else if let Some(v) = line.trim_start().strip_prefix("sum = ") {
            if let (Some((name, a, b)), Ok(s)) = (cur.take(), v.trim().parse::<f64>()) {
                out.insert(name, (a, b, s));
            }
        }
    }
    out
}

/// Resultado da comparação de um estágio com o log do llama.cpp.
#[derive(Debug)]
pub struct StageDiff {
    pub name: String,
    pub shape_ok: bool,
    pub ours: f64,
    pub theirs: f64,
    /// `|Δsoma| / tolerância`; a tolerância é `1e-3·(média |x|)·√N + 1e-3·|soma|` (ver o doc). ≤ 1: compatível.
    pub ratio: f64,
}

pub fn compare_with_log(stages: &[(String, Stat)], log: &HashMap<String, (usize, usize, f64)>) -> Vec<StageDiff> {
    let mut out = Vec::new();
    for (name, st) in stages {
        if let Some(&(a, b, theirs)) = log.get(name) {
            let n = (st.ne[0] * st.ne[1]) as f64;
            let tol = 1e-3 * (st.sum_abs / n) * n.sqrt() + 1e-3 * (st.sum as f64).abs();
            let ratio = (st.sum as f64 - theirs).abs() / tol.max(1e-12);
            out.push(StageDiff { name: name.clone(), shape_ok: a * b == st.ne[0] * st.ne[1], ours: st.sum as f64, theirs, ratio });
        }
    }
    out
}

/// Valores [0,1] -> bytes RGB (arredonda). É a imagem que vai tanto para o PNG do llama-server quanto para a nossa torre.
pub fn to_u8(img: &[f32]) -> Vec<u8> {
    img.iter().map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8).collect()
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// PNG RGB 8 bits sem compressão (blocos "stored" do deflate), escrito à mão: dá ao llama-server a MESMA imagem
/// que a torre recebe, sem nenhuma dependência de imagem (decodificar PNG no Rust continua fora do escopo).
pub fn png_rgb(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), w * h * 3);
    let mut raw = Vec::with_capacity(h * (w * 3 + 1));
    for y in 0..h {
        raw.push(0); // filtro "nenhum"
        raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        z.push(if chunks.peek().is_none() { 1 } else { 0 });
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}
