//! Testes da visão: modelo sintético pequeno (GGUF escrito aqui mesmo) contra a referência ingênua em
//! f64, testes com dentes (defeitos plantados têm de reprovar) e testes `#[ignore]` com o arquivo real
//! (`BURITI_TEST_MMPROJ=caminho cargo test real_ -- --ignored --nocapture`).

use std::{
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use super::{
    diag::{self, synthetic_image},
    mmproj::{Mmproj, Projector},
    prompt,
    ops::{self, GeluMode},
    reference,
    tower::{self, Bug, EncodeOpts},
};
use crate::native::{f16::f32_to_f16, gguf::Gguf};

// ------------------------------------------------------------------ gerador de números e GGUF sintético

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Wt {
    F32,
    F16,
    Q8,
}

fn enc_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn enc_f16(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect()
}
fn enc_q8(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for blk in v.chunks(32) {
        let amax = blk.iter().fold(0f32, |a, x| a.max(x.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend(blk.iter().map(|x| (x * id).round() as i8 as u8));
    }
    out
}
fn enc(v: &[f32], w: Wt) -> (u32, Vec<u8>) {
    match w {
        Wt::F32 => (0, enc_f32(v)),
        Wt::F16 => (1, enc_f16(v)),
        Wt::Q8 => (8, enc_q8(v)),
    }
}

enum Kv {
    U32(u32),
    F32(f32),
    B(bool),
    S(&'static str),
    Arr3(f32),
}

struct Builder {
    kvs: Vec<(String, Kv)>,
    tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)>,
}

fn put_str(o: &mut Vec<u8>, s: &str) {
    o.extend_from_slice(&(s.len() as u64).to_le_bytes());
    o.extend_from_slice(s.as_bytes());
}

impl Builder {
    fn build(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(b"GGUF");
        o.extend_from_slice(&3u32.to_le_bytes());
        o.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        o.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (k, v) in &self.kvs {
            put_str(&mut o, k);
            match v {
                Kv::U32(x) => {
                    o.extend_from_slice(&4u32.to_le_bytes());
                    o.extend_from_slice(&x.to_le_bytes());
                }
                Kv::F32(x) => {
                    o.extend_from_slice(&6u32.to_le_bytes());
                    o.extend_from_slice(&x.to_le_bytes());
                }
                Kv::B(x) => {
                    o.extend_from_slice(&7u32.to_le_bytes());
                    o.push(*x as u8);
                }
                Kv::S(x) => {
                    o.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut o, x);
                }
                Kv::Arr3(x) => {
                    o.extend_from_slice(&9u32.to_le_bytes());
                    o.extend_from_slice(&6u32.to_le_bytes());
                    o.extend_from_slice(&3u64.to_le_bytes());
                    for _ in 0..3 {
                        o.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        let mut off = 0usize;
        for (name, dims, ty, data) in &self.tensors {
            put_str(&mut o, name);
            o.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                o.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            o.extend_from_slice(&ty.to_le_bytes());
            o.extend_from_slice(&(off as u64).to_le_bytes());
            off += data.len().div_ceil(32) * 32;
        }
        while o.len() % 32 != 0 {
            o.push(0);
        }
        for (_, _, _, data) in &self.tensors {
            o.extend_from_slice(data);
            while o.len() % 32 != 0 {
                o.push(0);
            }
        }
        o
    }
}

/// Modelo minúsculo: imagem 32, patch 8 (16 patches), embd 32, 4 cabeças, ff 64, 2 camadas,
/// scale 2 (4 tokens), projeção 24.
#[derive(Clone)]
struct Spec {
    wt: Wt,
    swapped_ffn: bool,
    projector: &'static str,
    use_gelu: bool,
    n_head: u32,
}
impl Default for Spec {
    fn default() -> Self {
        Spec { wt: Wt::F16, swapped_ffn: false, projector: "idefics3", use_gelu: true, n_head: 4 }
    }
}
const IMG: usize = 32;
const P: usize = 8;
const N: usize = 32;
const FF: usize = 64;
const NL: usize = 2;
const PROJ: usize = 24;
const SC: usize = 2;

fn tiny(spec: &Spec, seed: u64) -> Builder {
    let mut r = Rng(seed);
    let mut kvs = vec![
        ("general.architecture".to_string(), Kv::S("clip")),
        ("clip.has_vision_encoder".into(), Kv::B(true)),
        ("clip.projector_type".into(), Kv::S(spec.projector)),
        ("clip.vision.image_size".into(), Kv::U32(IMG as u32)),
        ("clip.vision.patch_size".into(), Kv::U32(P as u32)),
        ("clip.vision.embedding_length".into(), Kv::U32(N as u32)),
        ("clip.vision.feed_forward_length".into(), Kv::U32(FF as u32)),
        ("clip.vision.block_count".into(), Kv::U32(NL as u32)),
        ("clip.vision.attention.head_count".into(), Kv::U32(spec.n_head)),
        ("clip.vision.projection_dim".into(), Kv::U32(PROJ as u32)),
        ("clip.vision.attention.layer_norm_epsilon".into(), Kv::F32(1e-6)),
        ("clip.vision.projector.scale_factor".into(), Kv::U32(SC as u32)),
        ("clip.vision.image_mean".into(), Kv::Arr3(0.5)),
        ("clip.vision.image_std".into(), Kv::Arr3(0.5)),
    ];
    if spec.use_gelu {
        kvs.push(("clip.use_gelu".into(), Kv::B(true)));
    }
    let mut t: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
    let w = spec.wt;
    let mat = |t: &mut Vec<(String, Vec<usize>, u32, Vec<u8>)>, name: String, rows: usize, cols: usize, r: &mut Rng| {
        let (ty, data) = enc(&r.vec(rows * cols, 1.0 / (cols as f32).sqrt()), w);
        t.push((name, vec![cols, rows], ty, data));
    };
    let f32t = |t: &mut Vec<(String, Vec<usize>, u32, Vec<u8>)>, name: String, dims: Vec<usize>, v: Vec<f32>| t.push((name, dims, 0, enc_f32(&v)));
    f32t(&mut t, "v.patch_embd.weight".into(), vec![P, P, 3, N], r.vec(N * 3 * P * P, 1.0 / ((3 * P * P) as f32).sqrt()));
    f32t(&mut t, "v.patch_embd.bias".into(), vec![N], r.vec(N, 0.1));
    f32t(&mut t, "v.position_embd.weight".into(), vec![N, (IMG / P) * (IMG / P)], r.vec(N * 16, 0.3));
    for il in 0..NL {
        let nm = |s: &str| format!("v.blk.{il}.{s}");
        for ln in ["ln1", "ln2"] {
            let wv: Vec<f32> = r.vec(N, 0.2).iter().map(|x| 1.0 + x).collect();
            f32t(&mut t, nm(&format!("{ln}.weight")), vec![N], wv);
            f32t(&mut t, nm(&format!("{ln}.bias")), vec![N], r.vec(N, 0.2));
        }
        for a in ["attn_q", "attn_k", "attn_v", "attn_out"] {
            mat(&mut t, nm(&format!("{a}.weight")), N, N, &mut r);
            f32t(&mut t, nm(&format!("{a}.bias")), vec![N], r.vec(N, 0.1));
        }
        // o "up" verdadeiro (N -> FF) e o "down" verdadeiro (FF -> N); o conversor antigo troca os nomes
        let (un, dn) = if spec.swapped_ffn { ("ffn_down", "ffn_up") } else { ("ffn_up", "ffn_down") };
        mat(&mut t, nm(&format!("{un}.weight")), FF, N, &mut r);
        f32t(&mut t, nm(&format!("{un}.bias")), vec![FF], r.vec(FF, 0.1));
        mat(&mut t, nm(&format!("{dn}.weight")), N, FF, &mut r);
        f32t(&mut t, nm(&format!("{dn}.bias")), vec![N], r.vec(N, 0.1));
    }
    let wv: Vec<f32> = r.vec(N, 0.2).iter().map(|x| 1.0 + x).collect();
    f32t(&mut t, "v.post_ln.weight".into(), vec![N], wv);
    f32t(&mut t, "v.post_ln.bias".into(), vec![N], r.vec(N, 0.2));
    mat(&mut t, "mm.model.fc.weight".into(), PROJ, N * SC * SC, &mut r);
    Builder { kvs, tensors: t }
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn tmp_file(bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("buriti-vision-test-{}-{}.gguf", std::process::id(), COUNTER.fetch_add(1, Ordering::SeqCst)));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn load(b: &Builder) -> Result<Mmproj, String> {
    let p = tmp_file(&b.build());
    let r = Mmproj::open(&p);
    let _ = std::fs::remove_file(&p);
    r
}

fn load_bytes(bytes: &[u8]) -> Result<Mmproj, String> {
    let p = tmp_file(bytes);
    let r = Mmproj::open(&p);
    let _ = std::fs::remove_file(&p);
    r
}

fn pixels(seed: u64) -> Vec<f32> {
    Rng(seed).vec(IMG * IMG * 3, 1.0)
}

fn max_rel(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - y).abs())) / scale
}

/// Roda a torre e a referência e devolve, por estágio comum, o erro relativo ao maior valor do estágio.
fn stage_errors(m: &Mmproj, px: &[f32], opts: EncodeOpts) -> Vec<(String, f64)> {
    let mut got: Vec<(String, Vec<f32>)> = Vec::new();
    let mut sink = |n: &str, d: &[f32]| got.push((n.to_string(), d.to_vec()));
    tower::encode(m, px, opts, Some(&mut sink)).unwrap();
    reference::encode(m, px)
        .into_iter()
        .map(|(name, r)| {
            let g = &got.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("estágio {name} não emitido")).1;
            let e = max_rel(g, &r);
            (name, e)
        })
        .collect()
}

const TOL: f64 = 2e-4;

// --------------------------------------------------------------------------------------------- testes

#[test]
fn tiny_pipeline_matches_reference_all_dtypes() {
    for wt in [Wt::F32, Wt::F16, Wt::Q8] {
        let m = load(&tiny(&Spec { wt, ..Spec::default() }, 11)).unwrap();
        assert_eq!(m.hp.n_tokens(), 4);
        for (name, e) in stage_errors(&m, &pixels(5), EncodeOpts::default()) {
            assert!(e < TOL, "estágio {name}: erro relativo {e:e} (tolerância {TOL:e})");
        }
    }
}

#[test]
fn swapped_ffn_names_give_identical_result() {
    // os mesmos pesos gravados com `ffn_up`/`ffn_down` trocados (conversor antigo) têm de dar o MESMO resultado
    let a = load(&tiny(&Spec::default(), 7)).unwrap();
    let b = load(&tiny(&Spec { swapped_ffn: true, ..Spec::default() }, 7)).unwrap();
    // o gerador sorteia na ordem dos tensores; com a troca a ordem de sorteio muda, então o que se
    // compara aqui é a forma carregada: up = n_embd -> n_ff e down = n_ff -> n_embd nos dois casos
    for m in [&a, &b] {
        for l in &m.layers {
            assert_eq!((l.up.w.rows, l.up.w.cols), (FF, N));
            assert_eq!((l.down.w.rows, l.down.w.cols), (N, FF));
            assert_eq!(l.up.b.as_ref().unwrap().len(), FF);
            assert_eq!(l.down.b.as_ref().unwrap().len(), N);
        }
    }
    // e o resultado do arquivo trocado concorda com a referência (que usa os mesmos `Linear` carregados)
    for (name, e) in stage_errors(&b, &pixels(5), EncodeOpts::default()) {
        assert!(e < TOL, "estágio {name}: {e:e}");
    }
    // forma que não é nenhuma das duas é recusada, nunca adivinhada
    let mut bad = tiny(&Spec::default(), 7);
    for t in bad.tensors.iter_mut() {
        if t.0 == "v.blk.0.ffn_up.weight" {
            t.1 = vec![N, N];
        }
    }
    assert!(load(&bad).is_err());
}

#[test]
fn deterministic_and_independent_of_thread_count() {
    let m = load(&tiny(&Spec::default(), 3)).unwrap();
    let px = pixels(9);
    let a = tower::encode(&m, &px, EncodeOpts::default(), None).unwrap();
    let b = tower::encode(&m, &px, EncodeOpts::default(), None).unwrap();
    assert_eq!(a, b, "duas execuções diferem");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let c = pool.install(|| tower::encode(&m, &px, EncodeOpts::default(), None).unwrap());
    let d = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap().install(|| tower::encode(&m, &px, EncodeOpts::default(), None).unwrap());
    assert_eq!(a, c, "1 thread difere do padrão");
    assert_eq!(a, d, "3 threads diferem do padrão");
    // imagens diferentes dão saídas diferentes (não é uma função constante)
    let e = tower::encode(&m, &pixels(10), EncodeOpts::default(), None).unwrap();
    assert_ne!(a, e);
}

#[test]
fn planted_bugs_are_caught() {
    // cada defeito plantado tem de passar da tolerância em algum estágio (e o estágio certo tem de acusar)
    let m = load(&tiny(&Spec::default(), 11)).unwrap();
    let px = pixels(5);
    let expect: [(Bug, &str); 5] = [
        (Bug::WrongAttnScale, "attn_out-0"),
        (Bug::NoPositionEmbedding, "pos_embed"),
        (Bug::NoNormBias, "attn_out-0"),
        (Bug::ShuffleSwapped, "pixel_shuffle"),
        (Bug::UseKAsV, "attn_out-0"),
    ];
    for (bug, stage) in expect {
        let errs = stage_errors(&m, &px, EncodeOpts { bug, ..Default::default() });
        let e = errs.iter().find(|(n, _)| n == stage).unwrap().1;
        assert!(e > 50.0 * TOL, "defeito {bug:?} não foi detectado em {stage}: erro {e:e}");
    }
    // e sem defeito o mesmo estágio passa
    let ok = stage_errors(&m, &px, EncodeOpts::default());
    assert!(ok.iter().all(|(_, e)| *e < TOL));
}

#[test]
fn pixel_shuffle_matches_literal_reshape_permute_and_markers() {
    for (side, c, s) in [(4usize, 3usize, 2usize), (8, 5, 2), (12, 2, 4), (6, 4, 3), (4, 2, 1)] {
        let x: Vec<f32> = Rng(side as u64).vec(side * side * c, 1.0);
        let got = ops::pixel_shuffle(&x, side, c, s);
        let want = reference::pixel_shuffle_literal(&x.iter().map(|v| *v as f64).collect::<Vec<_>>(), side, c, s);
        assert_eq!(got.iter().map(|v| *v as f64).collect::<Vec<_>>(), want, "side={side} c={c} s={s}");
    }
    // marcador: o valor do patch (y, x) canal e vai para o token (y/s, x/s), posição ((y%s)·s + x%s)·c + e
    let (side, c, s) = (4usize, 3usize, 2usize);
    let x: Vec<f32> = (0..side * side * c).map(|i| i as f32).collect();
    let out = ops::pixel_shuffle(&x, side, c, s);
    let (y, xx, e) = (3usize, 2usize, 1usize);
    let tok = (y / s) * (side / s) + xx / s;
    let pos = ((y % s) * s + xx % s) * c + e;
    assert_eq!(out[tok * c * s * s + pos], x[(y * side + xx) * c + e]);
}

#[test]
fn ops_match_f64_references() {
    // LayerNorm (tamanho que não é múltiplo de 8)
    let (n, rows) = (37usize, 5usize);
    let mut r = Rng(1);
    let x = r.vec(n * rows, 3.0);
    let (w, b) = (r.vec(n, 1.0), r.vec(n, 1.0));
    let mut out = vec![0f32; x.len()];
    ops::layer_norm(&x, n, &w, &b, 1e-6, &mut out);
    for t in 0..rows {
        let xr: Vec<f64> = x[t * n..(t + 1) * n].iter().map(|v| *v as f64).collect();
        let mean = xr.iter().sum::<f64>() / n as f64;
        let var = xr.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        for i in 0..n {
            let want = (xr[i] - mean) / (var + 1e-6).sqrt() * w[i] as f64 + b[i] as f64;
            assert!((out[t * n + i] as f64 - want).abs() < 1e-5, "layer_norm t={t} i={i}");
        }
    }
    // atenção com n, cabeças e d fora do "redondo" (cauda < 4 chaves, d < 16, d = 64), nos dois caminhos
    let isas: Vec<bool> = if ops::has_avx2() { vec![false, true] } else { vec![false] };
    for (nt, nh, d) in [(37usize, 3usize, 8usize), (19, 2, 64), (5, 1, 20), (4, 2, 16)] {
        let (q, k, v) = (r.vec(nt * nh * d, 2.0), r.vec(nt * nh * d, 2.0), r.vec(nt * nh * d, 1.0));
        let f = |s: &[f32]| s.iter().map(|x| *x as f64).collect::<Vec<_>>();
        let (q64, k64, v64) = (f(&q), f(&k), f(&v));
        let mut want = vec![0f64; nt * nh * d];
        for t in 0..nt {
            for h in 0..nh {
                let sc: Vec<f64> = (0..nt).map(|s| (0..d).map(|i| q64[t * nh * d + h * d + i] * k64[s * nh * d + h * d + i]).sum::<f64>() / (d as f64).sqrt()).collect();
                let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                let es: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = es.iter().sum();
                for i in 0..d {
                    want[t * nh * d + h * d + i] = (0..nt).map(|s| es[s] / z * v64[s * nh * d + h * d + i]).sum();
                }
            }
        }
        for &avx in &isas {
            let mut o = vec![0f32; nt * nh * d];
            ops::attention_isa(avx, &q, &k, &v, nt, nh, d, 1.0 / (d as f32).sqrt(), &mut o);
            assert!(max_rel(&o, &want) < 1e-5, "atenção avx={avx} n={nt} h={nh} d={d}: {:e}", max_rel(&o, &want));
        }
    }
    // produto de matriz nos três formatos, com número de vetores que não é múltiplo de 4 nem do bloco
    for wt in [Wt::F32, Wt::F16, Wt::Q8] {
        for (rows, cols, nv) in [(7usize, 64usize, 19usize), (5, 96, 70), (3, 32, 1)] {
            let wf = r.vec(rows * cols, 1.0);
            let (ty, bytes) = enc(&wf, wt);
            let dtype = match ty {
                0 => crate::native::gguf::Dtype::F32,
                1 => crate::native::gguf::Dtype::F16,
                _ => crate::native::gguf::Dtype::Q8_0,
            };
            let xs = r.vec(nv * cols, 1.0);
            let bias = r.vec(rows, 1.0);
            let mut deq = vec![0f32; rows * cols];
            for rr in 0..rows {
                let rb = bytes.len() / rows;
                crate::native::quant::dequant_row(dtype, &bytes[rr * rb..(rr + 1) * rb], &mut deq[rr * cols..(rr + 1) * cols]);
            }
            let mut want = vec![0f64; nv * rows];
            for t in 0..nv {
                for rr in 0..rows {
                    want[t * rows + rr] = bias[rr] as f64 + (0..cols).map(|i| deq[rr * cols + i] as f64 * xs[t * cols + i] as f64).sum::<f64>();
                }
            }
            for &avx in &isas {
                let mut got = vec![0f32; nv * rows];
                ops::linear_isa(avx, &bytes, dtype, rows, cols, &xs, nv, Some(&bias), &mut got);
                assert!(max_rel(&got, &want) < 1e-5, "linear {:?} avx={avx} {rows}x{cols} n={nv}: {:e}", dtype, max_rel(&got, &want));
            }
        }
    }
    // o resultado de um vetor não depende de como os vetores são agrupados de 4 em 4
    let (rows, cols) = (6usize, 96usize);
    let (_, bytes) = enc(&r.vec(rows * cols, 1.0), Wt::F16);
    let xs = r.vec(9 * cols, 1.0);
    let mut all = vec![0f32; 9 * rows];
    ops::linear(&bytes, crate::native::gguf::Dtype::F16, rows, cols, &xs, 9, None, &mut all);
    for t in 0..9 {
        let mut one = vec![0f32; rows];
        ops::linear(&bytes, crate::native::gguf::Dtype::F16, rows, cols, &xs[t * cols..(t + 1) * cols], 1, None, &mut one);
        assert_eq!(&all[t * rows..(t + 1) * rows], &one[..], "vetor {t}: agrupamento muda o resultado");
    }
}

#[test]
fn gelu_exact_and_ggml_table_emulation() {
    let xs: Vec<f32> = (-1200..=1200).map(|i| i as f32 * 0.01).collect();
    let mut e = xs.clone();
    ops::gelu(&mut e, GeluMode::Exact);
    let mut t = xs.clone();
    ops::gelu(&mut t, GeluMode::GgmlF16Table);
    let (mut max_exact, mut max_table, mut differs) = (0f64, 0f64, 0);
    for i in 0..xs.len() {
        let want = reference::gelu64(xs[i] as f64);
        max_exact = max_exact.max((e[i] as f64 - want).abs());
        max_table = max_table.max((t[i] as f64 - want).abs() / (want.abs() + 1.0));
        differs += (e[i] != t[i]) as usize;
        // a saída da tabela é exatamente representável em f16
        if xs[i].abs() < 10.0 {
            assert_eq!(crate::native::quant::f16_to_f32(f32_to_f16(t[i])), t[i]);
        }
    }
    assert!(max_exact < 3e-6, "GELU exato: erro absoluto máximo {max_exact:e}");
    assert!(max_table < 2e-3 && max_table > 1e-5, "emulação da tabela f16: {max_table:e} (deve ser pequena mas visível)");
    assert!(differs > xs.len() / 2, "a emulação da tabela devia diferir do GELU exato");
}

#[test]
fn loader_rejects_malformed_files_without_panicking() {
    let good = tiny(&Spec::default(), 1).build();
    assert!(load_bytes(&good).is_ok());
    // truncado em vários pontos (cabeçalho, tabela de tensores, dados)
    for cut in [0usize, 3, 8, 40, 200, 1000, good.len() / 2, good.len() - 1] {
        let r = load_bytes(&good[..cut.min(good.len())]);
        assert!(r.is_err(), "corte em {cut} deveria falhar");
    }
    // magia errada
    let mut bad = good.clone();
    bad[0] = b'X';
    assert!(load_bytes(&bad).is_err());
    // conector ainda não implementado, ativação desconhecida, cabeças que não dividem n_embd
    assert!(load(&tiny(&Spec { projector: "gemma3", ..Spec::default() }, 1)).unwrap_err().contains("gemma3"));
    assert!(load(&tiny(&Spec { projector: "qwen2vl", ..Spec::default() }, 1)).is_err());
    assert!(load(&tiny(&Spec { use_gelu: false, ..Spec::default() }, 1)).unwrap_err().contains("GELU"));
    assert!(load(&tiny(&Spec { n_head: 5, ..Spec::default() }, 1)).is_err());
    // tensor ausente e forma errada
    let mut b = tiny(&Spec::default(), 1);
    b.tensors.retain(|t| t.0 != "v.blk.1.attn_q.weight");
    assert!(load(&b).unwrap_err().contains("attn_q"));
    let mut b = tiny(&Spec::default(), 1);
    for t in b.tensors.iter_mut() {
        if t.0 == "mm.model.fc.weight" {
            t.1 = vec![N * SC * SC + 1, PROJ];
        }
    }
    assert!(load(&b).is_err());
    // offset de tensor para fora do arquivo: dados de um tensor grande removidos
    let mut b = tiny(&Spec::default(), 1);
    if let Some(t) = b.tensors.iter_mut().find(|t| t.0 == "mm.model.fc.weight") {
        t.3.truncate(10);
    }
    assert!(load(&b).is_err());
    // uma imagem de tamanho errado é recusada pela torre
    let m = load(&tiny(&Spec::default(), 1)).unwrap();
    assert!(tower::encode(&m, &[0.0; 10], EncodeOpts::default(), None).is_err());
    let mut nan = pixels(1);
    nan[5] = f32::NAN;
    assert!(tower::encode(&m, &nan, EncodeOpts::default(), None).is_err());
    assert_eq!(m.hp.projector, Projector::Idefics3);
}

#[test]
fn synthetic_images_follow_mtmd_debug_definitions() {
    let g = synthetic_image("gray", 8).unwrap();
    assert!(g.iter().all(|v| *v == 0.5) && g.len() == 8 * 8 * 3);
    let cb = synthetic_image("cb", 4).unwrap();
    assert_eq!(cb[0], 1.0); // (0,0): (x+y)%2==0 -> 1
    assert_eq!(cb[3], 0.0); // (0,1) -> 0
    let rb = synthetic_image("rainbow", 64).unwrap();
    assert!(rb.iter().all(|v| (0.0..=1.0).contains(v)));
    assert_eq!(rb, synthetic_image("rainbow", 64).unwrap());
    // no centro a saturação é 0: branco (r=g=b=1)
    let c = (32 * 64 + 32) * 3;
    assert!(rb[c] > 0.99 && rb[c + 1] > 0.99 && rb[c + 2] > 0.99);
    assert!(synthetic_image("nao-existe", 4).is_err());
}

#[test]
fn llama_log_roundtrip_and_comparison_has_teeth() {
    let m = load(&tiny(&Spec::default(), 11)).unwrap();
    let mut stages: Vec<(String, diag::Stat)> = Vec::new();
    let mut text = String::new();
    let mut sink = |n: &str, d: &[f32]| {
        let ne0 = if n == "inp_raw" { d.len() } else { d.len() / (if n == "pixel_shuffle" || n == "mm_out" { m.hp.n_tokens() } else { m.hp.n_patches() }) };
        stages.push((n.to_string(), diag::stat(d, ne0)));
        text.push_str(&diag::llama_like_dump(n, d, ne0));
        text.push_str("some unrelated log line\n");
    };
    tower::encode(&m, &pixels(5), EncodeOpts::default(), Some(&mut sink)).unwrap();
    let log = diag::parse_llama_log(&text);
    assert!(log.contains_key("layer_out-1") && log.contains_key("pixel_shuffle"));
    let diffs = diag::compare_with_log(&stages, &log);
    assert!(diffs.len() >= 10);
    assert!(diffs.iter().all(|d| d.shape_ok && d.ratio < 1e-3), "{diffs:?}");
    // com dentes: uma soma 1% diferente num estágio tem de passar da tolerância
    let mut log2 = log.clone();
    let e = log2.get_mut("layer_out-0").unwrap();
    e.2 *= 1.01;
    let d2 = diag::compare_with_log(&stages, &log2);
    let bad = d2.iter().find(|d| d.name == "layer_out-0").unwrap();
    assert!(bad.ratio > 1.0, "perturbação de 1% não detectada: razão {}", bad.ratio);
}

// ------------------------------------------------------------------- arquivo real (ignorados)

fn real_path() -> Option<PathBuf> {
    std::env::var("BURITI_TEST_MMPROJ").ok().map(PathBuf::from)
}

/// Carrega o mmproj real, codifica uma imagem sintética e confere forma, finitude e determinismo.
/// `BURITI_TEST_MMPROJ=...mmproj-SmolVLM-500M-Instruct-f16.gguf cargo test real_encode -- --ignored --nocapture`
#[test]
#[ignore]
fn real_encode() {
    let Some(path) = real_path() else { panic!("defina BURITI_TEST_MMPROJ") };
    let t0 = std::time::Instant::now();
    let m = Mmproj::open(&path).unwrap();
    let hp = m.hp.clone();
    println!("carregado em {:.2?}: {hp:?}", t0.elapsed());
    println!("tokens por imagem: {}, camadas: {}, projeção: {}", hp.n_tokens(), m.layers.len(), hp.proj_dim);
    let kind = std::env::var("BURITI_TEST_IMG").unwrap_or_else(|_| "rainbow".into());
    let img = synthetic_image(&kind, hp.image_size).unwrap();
    let mut stats: Vec<(String, diag::Stat)> = Vec::new();
    let t1 = std::time::Instant::now();
    let a = {
        let mut sink = |n: &str, d: &[f32]| {
            let ne0 = match n {
                "inp_raw" => hp.image_size,
                "pixel_shuffle" => hp.n_embd * hp.scale_factor * hp.scale_factor,
                "mm_out" => hp.proj_dim,
                _ => hp.n_embd,
            };
            stats.push((n.to_string(), diag::stat(d, ne0)));
        };
        tower::encode(&m, &img, EncodeOpts::default(), Some(&mut sink)).unwrap()
    };
    println!("encode (com estágios): {:.2?} [AVISO: máquina compartilhada, tempo com contenção]", t1.elapsed());
    assert_eq!(a.len(), hp.n_tokens() * hp.proj_dim);
    assert!(a.iter().all(|v| v.is_finite()));
    for (n, s) in &stats {
        assert!(s.finite, "estágio {n} com valores não finitos");
        if n.starts_with("layer_out") || n == "post_ln" || n == "mm_out" {
            let mean_abs = s.sum_abs / (s.ne[0] * s.ne[1]) as f64;
            println!("  {n:<18} soma={:>14.4} média|x|={:.4}", s.sum, mean_abs);
        }
    }
    let t2 = std::time::Instant::now();
    let b = tower::encode(&m, &img, EncodeOpts::default(), None).unwrap();
    println!("encode (sem estágios): {:.2?}", t2.elapsed());
    assert_eq!(a, b, "duas execuções do arquivo real diferem");
    // imagens diferentes dão saídas diferentes e finitas
    let other = tower::encode(&m, &synthetic_image("cb", hp.image_size).unwrap(), EncodeOpts::default(), None).unwrap();
    assert_ne!(a, other);
    let dist: f64 = a.iter().zip(&other).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    println!("||rainbow - cb|| / ||rainbow|| = {:.3}", dist / norm);
}

/// Confere camadas do arquivo real (e a convolução do patch) contra a referência ingênua em f64.
/// Camadas escolhidas por `BURITI_TEST_LAYERS` (padrão: 0 e a última); cada uma custa ~10 a 30 s.
#[test]
#[ignore]
fn real_stages_vs_reference() {
    let Some(path) = real_path() else { panic!("defina BURITI_TEST_MMPROJ") };
    let m = Mmproj::open(&path).unwrap();
    let hp = m.hp.clone();
    let img = synthetic_image("rainbow", hp.image_size).unwrap();
    let mut got: std::collections::HashMap<String, Vec<f32>> = Default::default();
    let mut sink = |n: &str, d: &[f32]| {
        got.insert(n.to_string(), d.to_vec());
    };
    tower::encode(&m, &img, EncodeOpts::default(), Some(&mut sink)).unwrap();

    let (pb, pe) = reference::embed(&m, &img);
    let e1 = max_rel(&got["patch_bias"], &pb);
    let e2 = max_rel(&got["pos_embed"], &pe);
    println!("patch_bias: erro relativo máx {e1:e}; pos_embed: {e2:e}");
    assert!(e1 < TOL && e2 < TOL);

    let layers: Vec<usize> = std::env::var("BURITI_TEST_LAYERS")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, hp.n_layer - 1]);
    for il in layers {
        let input: Vec<f64> = if il == 0 { pe.clone() } else { got[&format!("layer_out-{}", il - 1)].iter().map(|v| *v as f64).collect() };
        let t = std::time::Instant::now();
        let (ao, fo, lo) = reference::layer(&m, &m.layers[il], &input);
        let (ea, ef, el) = (max_rel(&got[&format!("attn_out-{il}")], &ao), max_rel(&got[&format!("ffn_out-{il}")], &fo), max_rel(&got[&format!("layer_out-{il}")], &lo));
        println!("camada {il}: attn_out {ea:e}, ffn_out {ef:e}, layer_out {el:e} (referência: {:.1?})", t.elapsed());
        assert!(ea < TOL && ef < TOL && el < TOL, "camada {il} fora da tolerância {TOL:e}");
    }
    // pixel shuffle e projeção finais a partir do post_ln de produção
    let post: Vec<f64> = got["post_ln"].iter().map(|v| *v as f64).collect();
    let sh = reference::pixel_shuffle_literal(&post, hp.side(), hp.n_embd, hp.scale_factor);
    let es = max_rel(&got["pixel_shuffle"], &sh);
    println!("pixel_shuffle: {es:e}");
    assert!(es < 1e-6);
}

/// Imprime os estágios da torre no formato do `common_debug_cb_eval` do llama.cpp e, se
/// `BURITI_TEST_LLAMA_LOG` apontar para o log do `llama-mtmd-debug`, compara as somas estágio a estágio.
/// `BURITI_TEST_IMG` (rainbow|gray|cb|...), `BURITI_TEST_SIZE` (padrão: `image_size`), `BURITI_VISION_GELU=ggml`
/// (emula a tabela f16 do GELU do ggml), `BURITI_TEST_DUMP_OUT` (arquivo de saída do dump).
#[test]
#[ignore]
fn dump_vision() {
    let Some(path) = real_path() else { panic!("defina BURITI_TEST_MMPROJ") };
    let m = Mmproj::open(&path).unwrap();
    let hp = m.hp.clone();
    let kind = std::env::var("BURITI_TEST_IMG").unwrap_or_else(|_| "rainbow".into());
    let size: usize = std::env::var("BURITI_TEST_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(hp.image_size);
    assert_eq!(size, hp.image_size, "a torre aceita só image_size ({}) por causa da posição aprendida", hp.image_size);
    let img = synthetic_image(&kind, size).unwrap();
    let gelu = if std::env::var("BURITI_VISION_GELU").as_deref() == Ok("ggml") { GeluMode::GgmlF16Table } else { GeluMode::Exact };
    let mut text = String::new();
    let mut stats: Vec<(String, diag::Stat)> = Vec::new();
    let mut sink = |n: &str, d: &[f32]| {
        let ne0 = match n {
            "inp_raw" => size,
            "pixel_shuffle" => hp.n_embd * hp.scale_factor * hp.scale_factor,
            "mm_out" => hp.proj_dim,
            "Qcur-0" | "Kcur-0" | "Vcur-0" => hp.head_dim(),
            _ => hp.n_embd,
        };
        text.push_str(&diag::llama_like_dump(n, d, ne0));
        stats.push((n.to_string(), diag::stat(d, ne0)));
    };
    tower::encode(&m, &img, EncodeOpts { gelu, ..Default::default() }, Some(&mut sink)).unwrap();
    match std::env::var("BURITI_TEST_DUMP_OUT") {
        Ok(p) => std::fs::write(p, &text).unwrap(),
        Err(_) => println!("{text}"),
    }
    for (n, s) in &stats {
        println!("STAGE {n:<18} soma={:>16.6} somaabs={:>16.4}", s.sum, s.sum_abs);
    }
    if let Ok(p) = std::env::var("BURITI_TEST_LLAMA_LOG") {
        let log = diag::parse_llama_log(&std::fs::read_to_string(p).unwrap());
        let diffs = diag::compare_with_log(&stats, &log);
        println!("{} estágios em comum com o log do llama.cpp", diffs.len());
        for d in &diffs {
            println!("COMPARE {:<18} forma={} nossa={:>16.4} llama={:>16.4} razao={:.2}", d.name, d.shape_ok, d.ours, d.theirs, d.ratio);
        }
        assert!(!diffs.is_empty(), "nenhum nome de estágio em comum: o log tem o formato esperado?");
    }
}

#[test]
fn gguf_reader_is_the_shared_one() {
    // garantia de que o protótipo reutiliza `native::gguf` (e não uma cópia): o tipo vem de lá
    fn takes(_: &Gguf) {}
    let m = load(&tiny(&Spec::default(), 1)).unwrap();
    takes(&m.gguf);
}

/// Quantifica diferenças esperadas (não são erros): GELU exato x tabela f16 do ggml, e (com
/// `BURITI_TEST_MMPROJ2`) o mmproj Q8_0 x F16. Mede o tamanho do "ruído" que o dono deve esperar
/// ao comparar com o llama.cpp.
#[test]
#[ignore]
fn real_variant_noise() {
    let Some(path) = real_path() else { panic!("defina BURITI_TEST_MMPROJ") };
    let m = Mmproj::open(&path).unwrap();
    let img = synthetic_image(&std::env::var("BURITI_TEST_IMG").unwrap_or_else(|_| "rainbow".into()), m.hp.image_size).unwrap();
    let base = tower::encode(&m, &img, EncodeOpts::default(), None).unwrap();
    let rel = |a: &[f32], b: &[f32]| -> (f64, f64) {
        let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt();
        let n: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let maxd = a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((x - y).abs() as f64));
        let maxv = b.iter().fold(0f64, |m, x| m.max(x.abs() as f64));
        (d / n, maxd / maxv)
    };
    let g = tower::encode(&m, &img, EncodeOpts { gelu: GeluMode::GgmlF16Table, ..Default::default() }, None).unwrap();
    let (l2, mx) = rel(&g, &base);
    println!("GELU tabela f16 (ggml) x exato: L2 relativo {l2:.3e}, máx |Δ|/máx|x| {mx:.3e}");
    if let Ok(p2) = std::env::var("BURITI_TEST_MMPROJ2") {
        let m2 = Mmproj::open(&PathBuf::from(p2)).unwrap();
        let o2 = tower::encode(&m2, &img, EncodeOpts::default(), None).unwrap();
        let (l2, mx) = rel(&o2, &base);
        println!("mmproj 2 x mmproj 1 (ambos GELU exato): L2 relativo {l2:.3e}, máx |Δ|/máx|x| {mx:.3e}");
        let o2g = tower::encode(&m2, &img, EncodeOpts { gelu: GeluMode::GgmlF16Table, ..Default::default() }, None).unwrap();
        let (l2, mx) = rel(&o2g, &base);
        println!("mmproj 2 (GELU tabela) x mmproj 1 (exato): L2 relativo {l2:.3e}, máx {mx:.3e}");
    }
}

/// Ponta a ponta (M4): imagem sintética -> torre -> embeddings no lugar dos `<image>` -> SmolLM2 -> texto (guloso).
/// `BURITI_TEST_MMPROJ`, `BURITI_TEST_LM` (o GGUF do SmolVLM), `BURITI_TEST_GRID=1` (fatia 1x1 + visão geral, como o llama.cpp
/// faz com 512x512), `BURITI_TEST_Q` (pergunta). **Só mostra o que o motor responde**: não é paridade com o llama.cpp.
#[test]
#[ignore]
fn smolvlm_end_to_end() {
    use crate::engine::ChatMsg;
    use crate::native::{model::Model, tokenizer::IMAGE_MARKER};
    let Some(path) = real_path() else { panic!("defina BURITI_TEST_MMPROJ") };
    let lm_path = std::env::var("BURITI_TEST_LM").expect("defina BURITI_TEST_LM");
    let mm = Mmproj::open(&path).unwrap();
    let lm = Model::load(std::path::Path::new(&lm_path)).unwrap();
    let hp = mm.hp.clone();
    assert_eq!(hp.proj_dim, lm.cfg.n_embd, "a projeção do mmproj tem de ter o n_embd do modelo de linguagem");
    let n = hp.image_size;
    // disco vermelho sobre fundo branco
    let mut rgb = vec![255u8; n * n * 3];
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f32 - 256.0, y as f32 - 256.0);
            if dx * dx + dy * dy < 150.0 * 150.0 {
                let i = (y * n + x) * 3;
                rgb[i + 1] = 0;
                rgb[i + 2] = 0;
            }
        }
    }
    let px = tower::normalize_u8(&rgb, hp.mean, hp.std);
    let t = std::time::Instant::now();
    // BURITI_TEST_BUG=shuffle|attn|pos|zero: controles com defeito plantado (a resposta tem de piorar)
    let bug = match std::env::var("BURITI_TEST_BUG").as_deref() {
        Ok("shuffle") => Bug::ShuffleSwapped,
        Ok("attn") => Bug::WrongAttnScale,
        Ok("pos") => Bug::NoPositionEmbedding,
        _ => Bug::None,
    };
    let mut emb = tower::encode(&mm, &px, EncodeOpts { bug, ..Default::default() }, None).unwrap();
    if std::env::var("BURITI_TEST_BUG").as_deref() == Ok("zero") {
        emb.fill(0.0);
    }
    println!("torre: {:.1?} por bloco (máquina compartilhada)", t.elapsed());
    let grid = std::env::var("BURITI_TEST_GRID").ok().as_deref() == Some("1");
    let per = hp.n_tokens();
    let (gx, gy) = if grid { (1, 1) } else { (0, 0) };
    // ordem dos blocos no prompt: fatias (linha a linha), depois a visão geral; aqui todos têm a mesma imagem
    let tiles = if grid { 2 } else { 1 };
    let q = std::env::var("BURITI_TEST_Q").unwrap_or_else(|_| "What shape and what color is in the image?".into());
    let msg = ChatMsg { role: "user".into(), content: format!("{IMAGE_MARKER}{q}") };
    let ids = lm.tok.chat_prompt_images(&[msg], Some(&|_| lm.tok.idefics3_image_ids(gx, gy, per).unwrap()));
    let slot = lm.tok.image_slot_id().unwrap();
    let slots: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == slot).map(|(i, _)| i).collect();
    assert_eq!(slots.len(), tiles * per);
    println!("prompt: {} tokens, {} posições de imagem", ids.len(), slots.len());
    let mut s = lm.new_session(ids.len() + 80);
    let t = std::time::Instant::now();
    let mut i = 0;
    while i < ids.len() {
        let end = (i + 32).min(ids.len());
        let e: Vec<(usize, &[f32])> = slots.iter().enumerate().filter(|(_, &p)| p >= i && p < end).map(|(k, &p)| (p - i, &emb[(k % per) * hp.proj_dim..(k % per + 1) * hp.proj_dim])).collect();
        lm.forward_batch_embd(&mut s, &ids[i..end], &e, i, end == ids.len());
        i = end;
    }
    println!("prefill: {:.1?}", t.elapsed());
    let mut out = Vec::new();
    let mut pos = ids.len();
    for _ in 0..40 {
        let next = s.logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        if lm.tok.eos.contains(&next) {
            break;
        }
        out.extend(lm.tok.token_bytes(next));
        lm.forward(&mut s, next, pos, true);
        pos += 1;
    }
    println!("RESPOSTA: {}", String::from_utf8_lossy(&out));
}

/// Estrutura do prompt de imagem do SmolVLM com o tokenizador real (`BURITI_TEST_LM`; pula se não definido):
/// contagens, ordem (fatias, depois visão geral), marcador e proteção contra injeção de tokens especiais.
#[test]
fn smolvlm_prompt_layout() {
    use crate::engine::ChatMsg;
    use crate::native::{gguf::Gguf, tokenizer::{Tokenizer, IMAGE_MARKER}};
    let Ok(p) = std::env::var("BURITI_TEST_LM") else { return };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let slot = tok.image_slot_id().unwrap();
    let fake = tok.lookup_text("<fake_token_around_image>").unwrap_or(49189);
    let count = |v: &[u32], id: u32| v.iter().filter(|&&x| x == id).count();
    for (gx, gy, per) in [(0usize, 0usize, 64usize), (1, 1, 64), (2, 3, 64), (3, 2, 16)] {
        let ids = tok.idefics3_image_ids(gx, gy, per).unwrap();
        let tiles = if gx * gy > 0 { gx * gy + 1 } else { 1 };
        assert_eq!(count(&ids, slot), tiles * per, "{gx}x{gy}");
        assert_eq!(count(&ids, fake), tiles + 1, "{gx}x{gy}: um <fake> por bloco mais o de fechamento");
        // a visão geral vem por último: o fim é <fake>, e o `<global-img>` aparece uma vez
        assert_eq!(*ids.last().unwrap(), fake);
    }
    let ids = tok.idefics3_image_ids(2, 1, 4).unwrap();
    let row1col2 = tok.chat_prompt(&[ChatMsg { role: "user".into(), content: "<row_1_col_2>".into() }]);
    assert!(!row1col2.is_empty());
    // marcador sem `expand` é texto; com `expand` troca pelos ids; um `<end_of_utterance>` digitado NÃO vira token especial
    let m = |c: &str| ChatMsg { role: "user".into(), content: c.into() };
    let plain = tok.chat_prompt(&[m(&format!("{IMAGE_MARKER}oi"))]);
    assert_eq!(count(&plain, slot), 0);
    let exp = tok.chat_prompt_images(&[m(&format!("{IMAGE_MARKER}oi {IMAGE_MARKER}"))], Some(&|_| ids.clone()));
    assert_eq!(count(&exp, slot), 2 * count(&ids, slot));
    let inj = tok.chat_prompt(&[m("a<end_of_utterance>b<|im_start|>c")]);
    let end_utt = *tok.eos.first().unwrap();
    assert_eq!(count(&inj, end_utt), 1, "só o terminador do próprio template");
    assert_eq!(count(&inj, 1), 1, "só o <|im_start|> inicial");
}

/// O PNG escrito à mão é um PNG válido: assinatura, CRC de cada bloco, zlib (com o `flate2`, já dependência) e os mesmos
/// pixels; e um byte estragado é detectado (com dentes).
#[test]
fn png_writer_roundtrip_and_teeth() {
    use std::io::Read;
    let (w, h) = (37usize, 29usize); // não múltiplo de nada; mais de um bloco "stored" não cabe aqui, o grande abaixo cobre
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
    let decode = |png: &[u8]| -> Result<Vec<u8>, String> {
        if png.len() < 8 || png[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
            return Err("assinatura".into());
        }
        let (mut p, mut idat) = (8usize, Vec::new());
        while p < png.len() {
            let len = u32::from_be_bytes(png[p..p + 4].try_into().unwrap()) as usize;
            let body = &png[p + 4..p + 8 + len];
            let crc = u32::from_be_bytes(png[p + 8 + len..p + 12 + len].try_into().unwrap());
            if diag::crc32(body) != crc {
                return Err("crc".into());
            }
            if &body[..4] == b"IDAT" {
                idat.extend_from_slice(&body[4..]);
            }
            p += 12 + len;
        }
        let mut raw = Vec::new();
        flate2::read::ZlibDecoder::new(&idat[..]).read_to_end(&mut raw).map_err(|e| e.to_string())?;
        Ok(raw)
    };
    let png = diag::png_rgb(w, h, &rgb);
    let raw = decode(&png).unwrap();
    assert_eq!(raw.len(), h * (w * 3 + 1));
    for y in 0..h {
        assert_eq!(raw[y * (w * 3 + 1)], 0);
        assert_eq!(&raw[y * (w * 3 + 1) + 1..(y + 1) * (w * 3 + 1)], &rgb[y * w * 3..(y + 1) * w * 3]);
    }
    // imagem grande (vários blocos stored de 65535 bytes): 512x512
    let big = diag::to_u8(&synthetic_image("disc", 512).unwrap());
    let raw = decode(&diag::png_rgb(512, 512, &big)).unwrap();
    assert!((0..512).all(|y| raw[y * 1537 + 1..(y + 1) * 1537] == big[y * 1536..(y + 1) * 1536]));
    // dentes: um byte de dado estragado (CRC) e um byte do adler estragado (zlib) são detectados
    let mut bad = png.clone();
    bad[40] ^= 1;
    assert!(decode(&bad).is_err());
    let mut z = png.clone();
    let n = z.len();
    // o adler fica nos 4 bytes antes do CRC do IDAT; reconstrói o CRC para só o zlib reprovar
    let idat_end = n - 12 - 4; // antes do bloco IEND (12 bytes) e do CRC (4)
    z[idat_end - 1] ^= 1;
    let body_start = 8 + 12 + 13 + 4 + 4; // assinatura + IHDR(12+13)... + len + kind do IDAT
    let crc = diag::crc32(&z[body_start - 4..idat_end]);
    z[idat_end..idat_end + 4].copy_from_slice(&crc.to_be_bytes());
    assert!(decode(&z).is_err(), "adler estragado não foi detectado");
    // disc: pixel central vermelho, canto branco
    let d = diag::to_u8(&synthetic_image("disc", 512).unwrap());
    assert_eq!(&d[(256 * 512 + 256) * 3..(256 * 512 + 256) * 3 + 3], &[255, 0, 0]);
    assert_eq!(&d[0..3], &[255, 255, 255]);
}

/// M5, nosso lado: grava o PNG da imagem sintética, monta o prompt de chat com imagem (fatia 1x1 + visão geral, como o
/// llama.cpp com 512x512), roda torre + `forward_batch_embd` e imprime `IDS`, `IMGPOS` e o top-20 de logprobs do 1º token
/// no formato `TOP <id> <logprob>` de `tools/compare-logits.ps1`. Variáveis: BURITI_TEST_MMPROJ, BURITI_TEST_LM,
/// BURITI_TEST_IMG (disc|rainbow|gray|cb|...), BURITI_TEST_Q, BURITI_TEST_GRID (1 padrão; 0 = só visão geral),
/// BURITI_TEST_PNG (onde gravar o PNG). Usado por `tools/compare-vision.ps1`.
#[test]
#[ignore]
fn dump_vision_logits() {
    use crate::engine::ChatMsg;
    use crate::native::{model::Model, tokenizer::IMAGE_MARKER};
    let Some(path) = real_path() else { panic!("defina BURITI_TEST_MMPROJ") };
    let lm_path = std::env::var("BURITI_TEST_LM").expect("defina BURITI_TEST_LM");
    let kind = std::env::var("BURITI_TEST_IMG").unwrap_or_else(|_| "disc".into());
    let q = std::env::var("BURITI_TEST_Q").unwrap_or_else(|_| "Describe this image.".into());
    let grid = std::env::var("BURITI_TEST_GRID").ok().as_deref() != Some("0");
    let mm = Mmproj::open(&path).unwrap();
    let lm = Model::load(std::path::Path::new(&lm_path)).unwrap();
    let hp = mm.hp.clone();
    assert_eq!(hp.proj_dim, lm.cfg.n_embd);
    let n = hp.image_size;
    let rgb = diag::to_u8(&synthetic_image(&kind, n).unwrap());
    if let Ok(p) = std::env::var("BURITI_TEST_PNG") {
        std::fs::write(&p, diag::png_rgb(n, n, &rgb)).unwrap();
        println!("PNG {p}");
    }
    let emb = tower::encode(&mm, &tower::normalize_u8(&rgb, hp.mean, hp.std), EncodeOpts::default(), None).unwrap();
    let per = hp.n_tokens();
    let (gx, gy) = if grid { (1, 1) } else { (0, 0) };
    let msg = |c: String| ChatMsg { role: "user".into(), content: c };
    // texto do template com o marcador (para comparar com o /apply-template do llama-server): imagem antes e depois do texto
    for (name, content) in [("IMAGE_FIRST", format!("{IMAGE_MARKER}{q}")), ("TEXT_FIRST", format!("{q}{IMAGE_MARKER}"))] {
        let t = lm.tok.render_all(&lm.tok.chat_prompt(&[msg(content)]));
        println!("TEMPLATE {name} {}", serde_json::to_string(&t).unwrap());
    }
    let ids = lm.tok.chat_prompt_images(&[msg(format!("{IMAGE_MARKER}{q}"))], Some(&|_| lm.tok.idefics3_image_ids(gx, gy, per).unwrap()));
    let slot = lm.tok.image_slot_id().unwrap();
    let slots: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == slot).map(|(i, _)| i).collect();
    assert_eq!(slots.len(), if grid { 2 * per } else { per });
    println!("NTOK {}", ids.len());
    println!("NIMG {}", slots.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < slots.len() {
        let mut j = i;
        while j + 1 < slots.len() && slots[j + 1] == slots[j] + 1 {
            j += 1;
        }
        runs.push(format!("{}-{}", slots[i], slots[j]));
        i = j + 1;
    }
    println!("IMGPOS {}", runs.join(","));
    println!("IDS {}", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    let mut s = lm.new_session(ids.len() + 16);
    let mut i = 0;
    while i < ids.len() {
        let end = (i + 32).min(ids.len());
        let e: Vec<(usize, &[f32])> = slots.iter().enumerate().filter(|(_, &p)| p >= i && p < end).map(|(k, &p)| (p - i, &emb[(k % per) * hp.proj_dim..(k % per + 1) * hp.proj_dim])).collect();
        lm.forward_batch_embd(&mut s, &ids[i..end], &e, i, end == ids.len());
        i = end;
    }
    let max = s.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + s.logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    let mut idx: Vec<usize> = (0..s.logits.len()).collect();
    idx.sort_by(|&a, &b| s.logits[b].total_cmp(&s.logits[a]));
    for &t in idx.iter().take(20) {
        println!("TOP {t} {:.5}", s.logits[t] - lse);
    }
}

fn tile_of(kind: &str) -> Vec<u8> {
    diag::to_u8(&synthetic_image(kind, 512).unwrap())
}

/// Prompt com imagem do servidor (D58): posições, chave, remoção de `<__media__>` digitado, e a chave muda com a imagem.
/// Usa `BURITI_TEST_LM` (pula sem ele).
#[test]
fn server_prompt_build_and_key() {
    use crate::engine::ChatMsg;
    use crate::image::ChatImage;
    use crate::native::{gguf::Gguf, tokenizer::{Tokenizer, IMAGE_MARKER}};
    let Ok(p) = std::env::var("BURITI_TEST_LM") else { return };
    let tok = Tokenizer::from_gguf(&Gguf::open(std::path::Path::new(&p)).unwrap()).unwrap();
    let m = |r: &str, c: &str| ChatMsg { role: r.into(), content: c.into() };
    let img = |k: &str, msg: usize, grid: (usize, usize)| ChatImage { msg, grid, tiles: vec![tile_of(k); if grid.0 * grid.1 > 0 { grid.0 * grid.1 + 1 } else { 1 }] };
    let msgs = [m("user", &format!("{IMAGE_MARKER}{IMAGE_MARKER}oi")), m("assistant", "ok"), m("user", "e agora?")];
    let (ids, ip) = prompt::build_prompt(&tok, 64, &msgs, &[img("disc", 0, (0, 0))]).unwrap();
    assert_eq!(ip.slots.len(), 64, "o <__media__> digitado não pode criar posições de imagem");
    assert!(ids.len() > 64 && ip.slots.windows(2).all(|w| w[1] == w[0] + 1));
    let (_, ip2) = prompt::build_prompt(&tok, 64, &msgs, &[img("disc", 0, (1, 1))]).unwrap();
    assert_eq!(ip2.slots.len(), 128);
    // chave: depende do conteúdo e da grade, é estável e nunca 0
    let k = |im: ChatImage| prompt::image_key(&[im]);
    assert_eq!(k(img("disc", 0, (0, 0))), k(img("disc", 0, (0, 0))));
    assert_ne!(k(img("disc", 0, (0, 0))), k(img("gray", 0, (0, 0))));
    assert_ne!(k(img("disc", 0, (0, 0))), k(img("disc", 0, (1, 1))));
    assert_ne!(k(img("disc", 0, (0, 0))), 0);
    // índice de mensagem inexistente é recusado
    assert!(prompt::build_prompt(&tok, 64, &msgs, &[img("disc", 9, (0, 0))]).is_err());
    // sem imagem em nenhuma mensagem do prompt, o prompt é o de texto
    let (t, _) = prompt::build_prompt(&tok, 64, &[m("user", "oi")], &[]).unwrap();
    assert_eq!(t, tok.chat_prompt(&[m("user", "oi")]));
}

/// Conversa com imagem pelo caminho do servidor (`generate`/`run`): resposta coerente, reuso do KV com a MESMA imagem (a torre
/// nem roda) e, com OUTRA imagem, o cache de imagem é invalidado (a resposta muda). `BURITI_TEST_MMPROJ` + `BURITI_TEST_LM`.
#[test]
#[ignore]
fn server_chat_with_image_and_cache() {
    use crate::engine::{ChatMsg, ChatRequest};
    use crate::image::ChatImage;
    use crate::native::model::Model;
    let Some(path) = real_path() else { panic!("defina BURITI_TEST_MMPROJ") };
    let lm = Model::load(std::path::Path::new(&std::env::var("BURITI_TEST_LM").expect("defina BURITI_TEST_LM"))).unwrap();
    let mm = Mmproj::open(&path).unwrap();
    let mut sess = lm.new_session(1024);
    let ask = |sess: &mut crate::native::model::Session, kind: &str, q: &str| -> (String, f32) {
        let req = ChatRequest {
            messages: vec![ChatMsg { role: "user".into(), content: q.into() }],
            temperature: Some(0.0),
            max_tokens: Some(30),
            stop: vec![],
            sampling: Default::default(),
            images: vec![ChatImage { msg: 0, grid: (0, 0), tiles: vec![tile_of(kind)] }],
            constraint: None,
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel::<axum::body::Bytes>(1024);
        let t = std::time::Instant::now();
        super::super::generate(&lm, Some(&mm), None, sess, &req, &tx, None);
        let dt = t.elapsed().as_secs_f32();
        drop(tx);
        let mut text = String::new();
        while let Ok(b) = rx.try_recv() {
            for line in String::from_utf8_lossy(&b).lines() {
                if let Some(j) = line.strip_prefix("data: ").and_then(|l| serde_json::from_str::<serde_json::Value>(l).ok()) {
                    if let Some(c) = j["choices"][0]["delta"]["content"].as_str() {
                        text.push_str(c);
                    }
                }
            }
        }
        (text, dt)
    };
    let q = "What shape and what color is in the image?";
    let (a, t1) = ask(&mut sess, "disc", q);
    let (b, t2) = ask(&mut sess, "disc", q);
    let (c, t3) = ask(&mut sess, "gray", q);
    println!("disc : {a:?} ({t1:.1}s)\ndisc2: {b:?} ({t2:.1}s)\ngray : {c:?} ({t3:.1}s)");
    assert!(a.to_lowercase().contains("red") || a.to_lowercase().contains("circle"), "resposta sem relação com a imagem: {a:?}");
    assert_eq!(a, b, "mesma imagem e mesmo prompt têm de dar a mesma resposta (guloso)");
    assert!(t2 < t1 / 2.0, "a 2ª chamada devia reaproveitar o KV e pular a torre ({t1:.1}s -> {t2:.1}s)");
    assert!(!c.to_lowercase().contains("red"), "com outra imagem a resposta não pode herdar o KV da anterior: {c:?}");
}
