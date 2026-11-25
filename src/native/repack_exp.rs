//! EXPERIMENTO E4 (só testes; nada disto entra no caminho do modelo): um kernel Q4_K sobre pesos
//! reempacotados em blocos de 8 linhas intercaladas (no estilo `block_q4_Kx8` do llama.cpp), para
//! decidir com micro-benchmark se o reempacotamento vale a integração (critério do relatório de
//! paridade, 2.4/E4: pelo menos +25% de GMAC/s com 4 threads).
//!
//! Layout de um bloco de 8 linhas (1184 bytes; o do llama.cpp tem 1152 porque guarda escalas e
//! mínimos em 96 bytes empacotados, aqui ficam em bytes soltos para simplificar o desempacotamento):
//!   [  0..16)   d    (f16 × 8, na ordem de lanes PERM)
//!   [ 16..32)   dmin (f16 × 8, na ordem de lanes PERM)
//!   [ 32..96)   escalas u8: [sub-bloco 0..8][linha 0..8]
//!   [ 96..160)  mínimos u8: [par p 0..4][lane 0..8 (PERM)][2] = min[2p+e] da linha PERM[lane]
//!   [160..1184) qs: [pedaço k 0..16][linha 0..8][8 bytes]  (pedaço k = bytes 8k..8k+8 dos qs da linha)
//!
//! Ordem de lanes `PERM`: o `hadd` dos acumuladores inteiros das duas metades (linhas 0..3 e 4..7)
//! produz as linhas em [0 1 4 5 2 3 6 7]; os vetores de d/dmin/mínimos já ficam nessa ordem e a
//! permutação só é desfeita uma vez, no fim, em cada vetor de resultado.
//!
//! A ativação também tem layout próprio (`RAct`): o int8 de sempre mais, por bloco, as somas de 32
//! valores em pares (para o termo dos mínimos de 8 linhas com um só `madd`).

use super::gguf::Dtype;
use super::kernels::{dot_row_q_x8, Act};

const BB: usize = 1184;
const OFF_SC: usize = 32;
const OFF_MN: usize = 96;
const OFF_QS: usize = 160;
const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
const QK: usize = 256;
const ROW_BLOCK: usize = 144;

/// Defeitos plantados de propósito, para provar que os testes têm dentes (CLAUDE.md, princípio 3).
#[derive(Clone, Copy, PartialEq)]
enum Bug {
    None,
    /// troca os qs intercalados das linhas 2 e 3
    SwapRows23,
    /// troca as escalas dos sub-blocos 0 e 1 da linha 5
    SwapScales,
}

fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Reempacota 8 linhas Q4_K (cada uma com `nb` blocos de 144 bytes) em `nb` blocos de 1184 bytes.
fn repack_group(rows: [&[u8]; 8], nb: usize, bug: Bug, out: &mut Vec<u8>) {
    for i in 0..nb {
        let start = out.len();
        out.resize(start + BB, 0);
        let o = &mut out[start..];
        let blks: Vec<&[u8]> = rows.iter().map(|r| &r[i * ROW_BLOCK..(i + 1) * ROW_BLOCK]).collect();
        for l in 0..8 {
            let r = PERM[l];
            o[l * 2..l * 2 + 2].copy_from_slice(&blks[r][0..2]);
            o[16 + l * 2..16 + l * 2 + 2].copy_from_slice(&blks[r][2..4]);
        }
        for r in 0..8 {
            for j in 0..8 {
                let (s, _) = scale_min(j, &blks[r][4..16]);
                o[OFF_SC + j * 8 + r] = s;
            }
        }
        if bug == Bug::SwapScales {
            for (a, b) in [(0usize, 8usize)] {
                o.swap(OFF_SC + a + 5, OFF_SC + b + 5);
            }
        }
        for p in 0..4 {
            for l in 0..8 {
                for e in 0..2 {
                    let (_, m) = scale_min(2 * p + e, &blks[PERM[l]][4..16]);
                    o[OFF_MN + p * 16 + l * 2 + e] = m;
                }
            }
        }
        for k in 0..16 {
            for r in 0..8 {
                let src_r = match (bug, r) {
                    (Bug::SwapRows23, 2) => 3,
                    (Bug::SwapRows23, 3) => 2,
                    _ => r,
                };
                let src = &blks[src_r][16 + k * 8..16 + k * 8 + 8];
                o[OFF_QS + k * 64 + r * 8..OFF_QS + k * 64 + r * 8 + 8].copy_from_slice(src);
            }
        }
    }
}

/// Matriz inteira (linhas múltiplas de 8): grupos consecutivos de `nb` blocos de 1184 bytes.
fn repack_matrix(data: &[u8], nrows: usize, row_bytes: usize, bug: Bug) -> Vec<u8> {
    assert!(nrows % 8 == 0);
    let nb = row_bytes / ROW_BLOCK;
    let mut out = Vec::with_capacity(nrows / 8 * nb * BB);
    for g in 0..nrows / 8 {
        let rows: [&[u8]; 8] = std::array::from_fn(|r| &data[(g * 8 + r) * row_bytes..(g * 8 + r + 1) * row_bytes]);
        repack_group(rows, nb, bug, &mut out);
    }
    out
}

/// Ativação para o kernel repacked: a mesma quantização do `Act` (arredondamento ao par mais
/// próximo, como o `cvtps2dq`), mais as somas de 32 em pares para os mínimos.
pub struct RAct {
    q: Vec<i8>,
    d: Vec<f32>,
    /// por bloco, 4 × i32 = (soma32[2p] em i16 baixo, soma32[2p+1] em i16 alto)
    bp: Vec<i32>,
}

impl RAct {
    fn new(x: &[f32]) -> RAct {
        let nb = x.len() / QK;
        let mut q = vec![0i8; x.len()];
        let mut d = Vec::with_capacity(nb);
        let mut bp = Vec::with_capacity(nb * 4);
        for (xb, qb) in x.chunks_exact(QK).zip(q.chunks_exact_mut(QK)) {
            let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
            let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            d.push(amax / 127.0);
            for (o, v) in qb.iter_mut().zip(xb) {
                *o = (v * id).round_ties_even().clamp(-128.0, 127.0) as i8;
            }
            let s32: Vec<i32> = qb.chunks_exact(32).map(|c| c.iter().map(|&v| v as i32).sum()).collect();
            for p in 0..4 {
                bp.push(((s32[2 * p] as i16 as u16 as u32) | ((s32[2 * p + 1] as i16 as u16 as u32) << 16)) as i32);
            }
        }
        RAct { q, d, bp }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{RAct, BB, OFF_MN, OFF_QS, OFF_SC};
    use std::arch::x86_64::*;

    /// 8 linhas × 8 ativações de `nb` blocos. `out[k][r]` = produto da linha r pela ativação k.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemm_8x8(w: *const u8, nb: usize, a: &[&RAct; 8], out: &mut [[f32; 8]; 8]) {
        let m4 = _mm256_set1_epi8(0x0F);
        // máscaras de pshufb: para a metade h (linhas 4h..4h+3), cada escala repetida 4× como i16
        let mut masks = [_mm256_setzero_si256(); 2];
        for h in 0..2 {
            let mut m = [0x80u8; 32];
            for lane in 0..2 {
                for j in 0..8 {
                    let row = 4 * h + 2 * lane + if j < 4 { 0 } else { 1 };
                    m[lane * 16 + 2 * j] = row as u8;
                }
            }
            masks[h] = _mm256_loadu_si256(m.as_ptr() as *const __m256i);
        }
        let qp: [*const i8; 8] = std::array::from_fn(|k| a[k].q.as_ptr());
        let dp: [*const f32; 8] = std::array::from_fn(|k| a[k].d.as_ptr());
        let bpp: [*const i32; 8] = std::array::from_fn(|k| a[k].bp.as_ptr());
        for k in 0..8 {
            assert!(a[k].q.len() >= nb * 256 && a[k].d.len() >= nb && a[k].bp.len() >= nb * 4);
        }
        let mut acc = [_mm256_setzero_ps(); 8];
        for i in 0..nb {
            let blk = w.add(i * BB);
            let dv = _mm256_cvtph_ps(_mm_loadu_si128(blk as *const __m128i));
            let dminv = _mm256_cvtph_ps(_mm_loadu_si128(blk.add(16) as *const __m128i));
            let mut mnv = [_mm256_setzero_si256(); 4];
            for p in 0..4 {
                mnv[p] = _mm256_cvtepu8_epi16(_mm_loadu_si128(blk.add(OFF_MN + 16 * p) as *const __m128i));
            }
            let mut isum = [_mm256_setzero_si256(); 8];
            for g in 0..4 {
                let mut wlo = [[_mm256_setzero_si256(); 2]; 4];
                let mut whi = [[_mm256_setzero_si256(); 2]; 4];
                for step in 0..4 {
                    for h in 0..2 {
                        let q = _mm256_loadu_si256(blk.add(OFF_QS + (g * 4 + step) * 64 + h * 32) as *const __m256i);
                        wlo[step][h] = _mm256_and_si256(q, m4);
                        whi[step][h] = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                    }
                }
                let mut scv = [[_mm256_setzero_si256(); 2]; 2];
                for t in 0..2 {
                    let v = _mm256_broadcastsi128_si256(_mm_loadl_epi64(blk.add(OFF_SC + 8 * (2 * g + t)) as *const __m128i));
                    for h in 0..2 {
                        scv[t][h] = _mm256_shuffle_epi8(v, masks[h]);
                    }
                }
                for k in 0..8 {
                    let xp = qp[k].add(i * 256 + g * 64);
                    let mut tl = [_mm256_setzero_si256(); 2];
                    let mut th = [_mm256_setzero_si256(); 2];
                    for step in 0..4 {
                        let xl = _mm256_set1_epi64x((xp.add(step * 8) as *const i64).read_unaligned());
                        let xh = _mm256_set1_epi64x((xp.add(32 + step * 8) as *const i64).read_unaligned());
                        for h in 0..2 {
                            tl[h] = _mm256_add_epi16(tl[h], _mm256_maddubs_epi16(wlo[step][h], xl));
                            th[h] = _mm256_add_epi16(th[h], _mm256_maddubs_epi16(whi[step][h], xh));
                        }
                    }
                    let s0 = _mm256_add_epi32(_mm256_madd_epi16(tl[0], scv[0][0]), _mm256_madd_epi16(th[0], scv[1][0]));
                    let s1 = _mm256_add_epi32(_mm256_madd_epi16(tl[1], scv[0][1]), _mm256_madd_epi16(th[1], scv[1][1]));
                    isum[k] = _mm256_add_epi32(isum[k], _mm256_hadd_epi32(s0, s1));
                }
            }
            for k in 0..8 {
                let dx = _mm256_set1_ps(*dp[k].add(i));
                acc[k] = _mm256_fmadd_ps(_mm256_mul_ps(dv, dx), _mm256_cvtepi32_ps(isum[k]), acc[k]);
                let bp = bpp[k].add(i * 4);
                let mut m = _mm256_madd_epi16(mnv[0], _mm256_set1_epi32(*bp));
                m = _mm256_add_epi32(m, _mm256_madd_epi16(mnv[1], _mm256_set1_epi32(*bp.add(1))));
                m = _mm256_add_epi32(m, _mm256_madd_epi16(mnv[2], _mm256_set1_epi32(*bp.add(2))));
                m = _mm256_add_epi32(m, _mm256_madd_epi16(mnv[3], _mm256_set1_epi32(*bp.add(3))));
                acc[k] = _mm256_fnmadd_ps(_mm256_mul_ps(dminv, dx), _mm256_cvtepi32_ps(m), acc[k]);
            }
        }
        for k in 0..8 {
            // desfaz a ordem de lanes PERM: qwords [0 2 1 3]
            let v = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(acc[k]), 0xD8));
            _mm256_storeu_ps(out[k].as_mut_ptr(), v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
}
/// Fora do x86-64 não há AVX2: os testes que dependem dele conferem isto antes e pulam.
#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

/// Wrapper seguro: um grupo de 8 linhas repacked × 8 ativações.
#[cfg(target_arch = "x86_64")]
fn gemm8(group: &[u8], nb: usize, a: &[&RAct; 8]) -> [[f32; 8]; 8] {
    assert!(group.len() >= nb * BB && has_avx2());
    let mut out = [[0f32; 8]; 8];
    // SAFETY: AVX2/FMA/F16C confirmados acima; o tamanho do grupo e das ativações foi conferido.
    unsafe { x86::gemm_8x8(group.as_ptr(), nb, a, &mut out) };
    out
}
#[cfg(not(target_arch = "x86_64"))]
fn gemm8(_group: &[u8], _nb: usize, _a: &[&RAct; 8]) -> [[f32; 8]; 8] {
    unreachable!("gemm8 needs x86-64 AVX2; the tests check has_avx2() first")
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Linhas Q4_K aleatórias com d/dmin = 0x2800 (2^-5, finito).
    fn random_rows(rng: &mut Rng, nrows: usize, nb: usize) -> Vec<u8> {
        let mut data: Vec<u8> = (0..nrows * nb * ROW_BLOCK).map(|_| (rng.next() & 0xFF) as u8).collect();
        for b in 0..nrows * nb {
            for off in [0usize, 2] {
                data[b * ROW_BLOCK + off] = 0x00;
                data[b * ROW_BLOCK + off + 1] = 0x28;
            }
        }
        data
    }

    /// Ativações "diádicas": inteiros em [-127,127] × 2^-6 com um ±127 por bloco, de modo que
    /// d = 2^-6 e q = x·64 exatos; com d/dmin = 2^-5, todas as contas em float são exatas e a ordem
    /// de soma não importa: os dois kernels têm de dar o MESMO float, bit a bit.
    fn dyadic_x(rng: &mut Rng, cols: usize) -> Vec<f32> {
        let mut x: Vec<f32> = (0..cols).map(|_| ((rng.next() % 255) as i32 - 127) as f32 / 64.0).collect();
        for b in 0..cols / QK {
            x[b * QK + (rng.next() % QK as u64) as usize] = if rng.next() & 1 == 0 { 127.0 / 64.0 } else { -127.0 / 64.0 };
        }
        x
    }

    struct Outcome {
        max_abs_diff: f32,
        equal: bool,
        max_rel: f64,
    }

    /// Compara repacked × `dot_row_q_x8` para 8 linhas e 8 ativações.
    fn compare(data: &[u8], nb: usize, xs: &[Vec<f32>], bug: Bug) -> Outcome {
        let row_bytes = nb * ROW_BLOCK;
        let rep = repack_matrix(&data[..8 * row_bytes], 8, row_bytes, bug);
        let ras: Vec<RAct> = xs.iter().map(|x| RAct::new(x)).collect();
        let acts: Vec<Act> = xs.iter().map(|x| Act::new(Dtype::Q4K, x).unwrap()).collect();
        let got = gemm8(&rep, nb, &std::array::from_fn(|k| &ras[k]));
        let ga: [&Act; 8] = std::array::from_fn(|k| &acts[k]);
        let (mut equal, mut mad, mut rel) = (true, 0f32, 0f64);
        let mut scale = 0f32;
        for r in 0..8 {
            let want = dot_row_q_x8(Dtype::Q4K, &data[r * row_bytes..(r + 1) * row_bytes], ga);
            for k in 0..8 {
                scale = scale.max(want[k].abs());
                equal &= got[k][r].to_bits() == want[k].to_bits();
                mad = mad.max((got[k][r] - want[k]).abs());
            }
        }
        rel = rel.max(mad as f64 / scale.max(1e-30) as f64);
        Outcome { max_abs_diff: mad, equal, max_rel: rel }
    }

    #[test]
    fn repacked_equals_current_exact_on_dyadic_data() {
        if !has_avx2() {
            return;
        }
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for trial in 0..20 {
            let nb = 1 + trial % 6;
            let data = random_rows(&mut rng, 8, nb);
            let xs: Vec<Vec<f32>> = (0..8).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
            let o = compare(&data, nb, &xs, Bug::None);
            assert!(o.equal, "trial {trial} nb {nb}: diferença máxima {} (esperado igualdade bit a bit)", o.max_abs_diff);
        }
    }

    #[test]
    fn repacked_matches_current_on_random_data() {
        if !has_avx2() {
            return;
        }
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let mut worst = 0f64;
        for trial in 0..20 {
            let nb = 1 + trial % 8;
            let data = random_rows(&mut rng, 8, nb);
            let xs: Vec<Vec<f32>> = (0..8).map(|_| (0..nb * QK).map(|_| rng.f()).collect()).collect();
            let o = compare(&data, nb, &xs, Bug::None);
            worst = worst.max(o.max_rel);
            // só a ordem de soma em float difere (somas inteiras idênticas): muito abaixo de 1e-5
            assert!(o.max_rel < 1e-5, "trial {trial}: erro relativo {} (dif. máx. {})", o.max_rel, o.max_abs_diff);
        }
        println!("dados aleatórios: pior erro relativo ao maior |resultado| = {worst:e}");
    }

    /// Oráculo em f64 da mesma conta (pesos originais Q4_K + ativação quantizada do RAct).
    fn oracle(row: &[u8], ra: &RAct) -> f64 {
        let mut total = 0f64;
        for (i, blk) in row.chunks_exact(ROW_BLOCK).enumerate() {
            let f16 = |o: usize| crate::native::quant::f16_to_f32(u16::from_le_bytes([blk[o], blk[o + 1]])) as f64;
            let (dw, dmin) = (f16(0), f16(2));
            let dx = ra.d[i] as f64;
            let (mut sumi, mut mins) = (0i64, 0i64);
            for j in 0..8 {
                let (s, m) = scale_min(j, &blk[4..16]);
                let (g, hi) = (j / 2, j % 2);
                let (mut ws, mut bs) = (0i64, 0i64);
                for b in 0..32 {
                    let byte = blk[16 + g * 32 + b];
                    let w = if hi == 0 { byte & 15 } else { byte >> 4 } as i64;
                    let x = ra.q[i * QK + j * 32 + b] as i64;
                    ws += w * x;
                    bs += x;
                }
                sumi += s as i64 * ws;
                mins += m as i64 * bs;
            }
            total += dw * dx * sumi as f64 - dmin * dx * mins as f64;
        }
        total
    }

    fn real_q4k_tensor() -> Option<(Vec<u8>, usize, usize)> {
        let path = format!("{}/buriti-data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR"));
        let g = super::super::gguf::Gguf::open(std::path::Path::new(&path)).ok()?;
        for name in ["blk.0.ffn_up.weight", "blk.3.ffn_gate.weight", "blk.5.attn_q.weight"] {
            let info = g.tensors.get(name)?;
            if super::super::gguf::Gguf::dtype_of(info) == Some(Dtype::Q4K) {
                let (cols, rows) = (info.dims[0], info.dims[1]);
                let n = rows.min(64) * cols / QK * ROW_BLOCK;
                return Some((g.mmap[info.offset..info.offset + n].to_vec(), rows.min(64), cols));
            }
        }
        None
    }

    #[test]
    fn repacked_matches_current_on_real_tensor() {
        if !has_avx2() {
            return;
        }
        let Some((data, rows, cols)) = real_q4k_tensor() else {
            println!("modelo qwen2.5-1.5b ausente ou sem tensor Q4_K: teste de tensor real PULADO");
            return;
        };
        let nb = cols / QK;
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        // erros absolutos máximos, depois normalizados pelo maior |resultado| (os resultados perto de
        // zero por cancelamento tornariam um erro relativo por elemento sem sentido)
        let (mut e_rel, mut e_cur, mut e_rep, mut omax) = (0f64, 0f64, 0f64, 0f64);
        for _ in 0..3 {
            // ativações com escala de unidade e alguns valores grandes (como as reais)
            let xs: Vec<Vec<f32>> = (0..8).map(|_| (0..cols).map(|j| rng.f() * if j % 97 == 0 { 20.0 } else { 1.0 }).collect()).collect();
            let ras: Vec<RAct> = xs.iter().map(|x| RAct::new(x)).collect();
            let acts: Vec<Act> = xs.iter().map(|x| Act::new(Dtype::Q4K, x).unwrap()).collect();
            let ga: [&Act; 8] = std::array::from_fn(|k| &acts[k]);
            let rep = repack_matrix(&data, rows, nb * ROW_BLOCK, Bug::None);
            for g in 0..rows / 8 {
                let got = gemm8(&rep[g * nb * BB..], nb, &std::array::from_fn(|k| &ras[k]));
                for r in 0..8 {
                    let row = &data[(g * 8 + r) * nb * ROW_BLOCK..(g * 8 + r + 1) * nb * ROW_BLOCK];
                    let cur = dot_row_q_x8(Dtype::Q4K, row, ga);
                    for k in 0..8 {
                        let o = oracle(row, &ras[k]);
                        omax = omax.max(o.abs());
                        e_cur = e_cur.max((cur[k] as f64 - o).abs());
                        e_rep = e_rep.max((got[k][r] as f64 - o).abs());
                        e_rel = e_rel.max((got[k][r] - cur[k]).abs() as f64);
                    }
                }
            }
        }
        let (worst_rel, worst_cur, worst_rep) = (e_rel / omax, e_cur / omax, e_rep / omax);
        println!("tensor real ({rows} linhas × {cols}): erro relativo ao oráculo f64: atual {worst_cur:e}, repacked {worst_rep:e}; repacked vs atual {worst_rel:e}");
        assert!(worst_rep < 1e-4 && worst_cur < 1e-4, "erro contra o oráculo grande demais");
        assert!(worst_rel < 2e-4);
    }

    /// Testes com dentes: com um defeito plantado no repacker, a comparação exata TEM de falhar.
    #[test]
    fn planted_bugs_are_caught() {
        if !has_avx2() {
            return;
        }
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let nb = 3;
        let data = random_rows(&mut rng, 8, nb);
        let xs: Vec<Vec<f32>> = (0..8).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
        assert!(compare(&data, nb, &xs, Bug::None).equal, "controle sem defeito");
        for bug in [Bug::SwapRows23, Bug::SwapScales] {
            let o = compare(&data, nb, &xs, bug);
            assert!(!o.equal, "defeito plantado NÃO foi detectado");
            println!("defeito plantado detectado, diferença máxima {}", o.max_abs_diff);
        }
    }

    // ---------------------------------------------------------------- benchmark

    struct SyncPtr(*mut f32);
    unsafe impl Send for SyncPtr {}
    unsafe impl Sync for SyncPtr {}
    impl SyncPtr {
        // método (e não o campo): a closure captura o struct inteiro, que é Sync
        fn get(&self) -> *mut f32 {
            self.0
        }
    }

    /// Réplica do `matmat` do modelo (n tokens, tarefas por conjunto de linhas, escrita direta em [token][linha]).
    fn run_current(pool: &rayon::ThreadPool, mats: &[Vec<u8>], rows: usize, row_bytes: usize, acts: &[Act], out: &mut [f32]) {
        use rayon::prelude::*;
        let n = acts.len();
        let rpc = (8192 / row_bytes).clamp(1, 16);
        let ptr = SyncPtr(out.as_mut_ptr());
        pool.install(|| {
            for m in mats {
                (0..rows.div_ceil(rpc)).into_par_iter().for_each(|ci| {
                    let r0 = ci * rpc;
                    let nr = rpc.min(rows - r0);
                    let mut i = 0;
                    while i + 8 <= n {
                        let g: [&Act; 8] = std::array::from_fn(|k| &acts[i + k]);
                        for j in 0..nr {
                            let r = dot_row_q_x8(Dtype::Q4K, &m[(r0 + j) * row_bytes..(r0 + j + 1) * row_bytes], g);
                            for k in 0..8 {
                                // SAFETY: tarefas com intervalos de linhas disjuntos; out tem n*rows elementos
                                unsafe { *ptr.get().add((i + k) * rows + r0 + j) = r[k] };
                            }
                        }
                        i += 8;
                    }
                });
            }
        });
    }

    fn run_repacked(pool: &rayon::ThreadPool, mats: &[Vec<u8>], rows: usize, row_bytes: usize, ras: &[RAct], out: &mut [f32]) {
        use rayon::prelude::*;
        let n = ras.len();
        let nb = row_bytes / ROW_BLOCK;
        let gpt = (8192 / (8 * row_bytes)).max(1);
        let ngroups = rows / 8;
        let ptr = SyncPtr(out.as_mut_ptr());
        pool.install(|| {
            for m in mats {
                (0..ngroups.div_ceil(gpt)).into_par_iter().for_each(|ci| {
                    for g in ci * gpt..(ci * gpt + gpt).min(ngroups) {
                        let mut i = 0;
                        while i + 8 <= n {
                            let a: [&RAct; 8] = std::array::from_fn(|k| &ras[i + k]);
                            let o = gemm8(&m[g * nb * BB..(g + 1) * nb * BB], nb, &a);
                            for k in 0..8 {
                                // SAFETY: cada tarefa é dona de seus grupos de 8 linhas (disjuntos)
                                unsafe { std::ptr::copy_nonoverlapping(o[k].as_ptr(), ptr.get().add((i + k) * rows + g * 8), 8) };
                            }
                            i += 8;
                        }
                    }
                });
            }
        });
    }

    /// `cargo test bench_repack -- --ignored --nocapture` (rode 3×; só interpretar medido em máquina quieta).
    #[test]
    #[ignore]
    fn bench_repack() {
        assert!(has_avx2());
        let n = 32usize; // PREFILL_BATCH
        let nmat = 4usize; // matrizes distintas por passada (> L3: os pesos vêm da RAM, como no modelo)
        for (rows, cols) in [(1536usize, 1536usize), (8960, 1536), (1536, 8960), (256, 1536)] {
            let row_bytes = cols / QK * ROW_BLOCK;
            let mut rng = Rng(0xBEEF_1234_5678_9ABC);
            let mats: Vec<Vec<u8>> = (0..nmat).map(|_| random_rows(&mut rng, rows, cols / QK)).collect();
            let t0 = std::time::Instant::now();
            let rmats: Vec<Vec<u8>> = mats.iter().map(|m| repack_matrix(m, rows, row_bytes, Bug::None)).collect();
            let rep_s = t0.elapsed().as_secs_f64() / nmat as f64;
            let xs: Vec<Vec<f32>> = (0..n).map(|_| (0..cols).map(|_| rng.f()).collect()).collect();
            let acts: Vec<Act> = xs.iter().map(|x| Act::new(Dtype::Q4K, x).unwrap()).collect();
            let ras: Vec<RAct> = xs.iter().map(|x| RAct::new(x)).collect();
            let mut out_a = vec![0f32; n * rows];
            let mut out_b = vec![0f32; n * rows];
            let macs = (nmat * rows * cols * n) as f64;
            println!(
                "forma {rows}x{cols}, {n} tokens, {nmat} matrizes/passada ({:.1} MB atual, {:.1} MB repacked); repack: {:.1} ms por matriz ({:.2} GB/s), memória +{:.1}%",
                (nmat * rows * row_bytes) as f64 / 1e6,
                rmats.iter().map(|m| m.len()).sum::<usize>() as f64 / 1e6,
                rep_s * 1e3,
                (rows * row_bytes) as f64 / rep_s / 1e9,
                (rmats[0].len() as f64 / mats[0].len() as f64 - 1.0) * 100.0
            );
            for threads in [1usize, 2, 3, 4] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                // aquecimento
                run_current(&pool, &mats, rows, row_bytes, &acts, &mut out_a);
                run_repacked(&pool, &rmats, rows, row_bytes, &ras, &mut out_b);
                let (mut best_a, mut best_b) = (f64::MAX, f64::MAX);
                for _ in 0..7 {
                    let t = std::time::Instant::now();
                    run_current(&pool, &mats, rows, row_bytes, &acts, &mut out_a);
                    best_a = best_a.min(t.elapsed().as_secs_f64());
                    let t = std::time::Instant::now();
                    run_repacked(&pool, &rmats, rows, row_bytes, &ras, &mut out_b);
                    best_b = best_b.min(t.elapsed().as_secs_f64());
                }
                let maxd = out_a.iter().zip(&out_b).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let (ga, gb) = (macs / best_a / 1e9, macs / best_b / 1e9);
                println!("  {threads} thread(s): atual {ga:.1} GMAC/s, repacked {gb:.1} GMAC/s, razão {:.2}x (dif. máx. das saídas {maxd:.2e})", gb / ga);
            }
        }
    }
}
