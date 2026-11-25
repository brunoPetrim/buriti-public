//! Pure (no GPU) side of the matvec kernels (tasks G4 and G5, D64): which formats and shapes the kernels
//! accept, the GPU-friendly repacking of GGUF rows, the row-aligned splitting of a matrix across buffers,
//! deterministic synthetic rows for tests and benchmarks, and the exact float reference with its error
//! metric. Everything here is testable without a GPU.
//!
//! The reference is the CPU engine's float path: `quant::dequant_row` (the same decoders the CPU kernels are
//! tested against) accumulated in f64. The CPU engine's int8-activation kernels are NOT the reference: the GPU
//! kernels keep the activations in f32 (D64).

use crate::native::f16::f32_to_f16;
use crate::native::gguf::Dtype;
use crate::native::quant;

/// Default workgroup size of the matvec kernels (a thread count per row; tuned in D64).
pub const DEFAULT_WG: u32 = 32;

/// Rows per workgroup by default (tuned in D64; only the quantized kernels can use more than 1).
pub fn default_rows(d: Dtype) -> u32 {
    match d {
        Dtype::Q6K => 2,
        Dtype::Q8_0 | Dtype::Q4K | Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 | Dtype::Q5K => DEFAULT_ROWS_QUANT,
        _ => 1,
    }
}

pub const DEFAULT_ROWS_QUANT: u32 = 4;

/// The quantized kernels compute several rows per workgroup, sharing the activation loads.
pub fn multi_row(d: Dtype) -> bool {
    matches!(d, Dtype::Q8_0 | Dtype::Q4K | Dtype::Q6K | Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 | Dtype::Q5K)
}

/// Formats with a GPU matvec kernel.
pub fn supported(d: Dtype) -> bool {
    ALL_FORMATS.contains(&d)
}

/// Every format with a GPU matvec kernel (G4, G5, G5b).
pub const ALL_FORMATS: [Dtype; 10] = [Dtype::F32, Dtype::F16, Dtype::Q8_0, Dtype::Q4K, Dtype::Q6K, Dtype::Q4_0, Dtype::Q4_1, Dtype::Q5_0, Dtype::Q5_1, Dtype::Q5K];

pub fn name(d: Dtype) -> &'static str {
    Dtype::NAMES[d as usize]
}

/// Parses the format names (`f32`, `q4_0`, `q5_k`...; case-insensitive).
pub fn parse_dtype(s: &str) -> Option<Dtype> {
    let l = s.trim().to_ascii_lowercase();
    ALL_FORMATS.into_iter().find(|&d| name(d).to_ascii_lowercase() == l)
}

/// The number of columns (K) must be a multiple of this: the kernels load whole vec4s / blocks.
pub fn col_multiple(d: Dtype) -> usize {
    match d {
        Dtype::F32 => 4,
        Dtype::F16 => 8,
        Dtype::Q8_0 | Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 => 32,
        Dtype::Q4K | Dtype::Q6K | Dtype::Q5K => 256,
        _ => usize::MAX,
    }
}

pub fn check_shape(d: Dtype, rows: usize, cols: usize) -> Result<(), String> {
    if !supported(d) {
        return Err(format!("{} has no GPU matvec kernel yet", name(d)));
    }
    if rows == 0 || cols == 0 {
        return Err("empty matrix".into());
    }
    let m = col_multiple(d);
    if cols % m != 0 {
        return Err(format!("{}: {cols} columns is not a multiple of {m}", name(d)));
    }
    Ok(())
}

/// Blocks per row (1 element = 1 block for f32/F16 is not used; the kernels get `cols / block_elems`).
pub fn blocks_per_row(d: Dtype, cols: usize) -> usize {
    cols / d.block().0
}

/// Bytes per row in the GGUF file (what the CPU engine reads).
pub fn native_row_bytes(d: Dtype, cols: usize) -> u64 {
    let (be, bb) = d.block();
    (cols / be * bb) as u64
}

/// Bytes per row of each GPU array after repacking (every array is a multiple of 4 bytes per row, so rows
/// stay u32-aligned in every buffer).
pub fn array_row_bytes(d: Dtype, cols: usize) -> Vec<u64> {
    let nb = blocks_per_row(d, cols) as u64;
    match d {
        Dtype::F32 => vec![cols as u64 * 4],
        Dtype::F16 => vec![cols as u64 * 2],
        Dtype::Q8_0 => vec![nb * 32, nb.div_ceil(2) * 4],
        Dtype::Q4_0 => vec![nb * 16, nb.div_ceil(2) * 4],
        Dtype::Q4_1 => vec![nb * 16, nb * 4],
        Dtype::Q5_0 => vec![nb * 16, nb * 4, nb.div_ceil(2) * 4],
        Dtype::Q5_1 => vec![nb * 16, nb * 4, nb * 4],
        Dtype::Q4K => vec![nb * 144],
        Dtype::Q5K => vec![nb * 176],
        Dtype::Q6K => vec![nb * 212],
        _ => vec![],
    }
}

/// Repacks `rows` consecutive GGUF rows (`src`, `rows * native_row_bytes` bytes) into the GPU arrays:
/// - F32, F16, Q4_K: copied as they are (already multiples of 16/4 bytes; Q4_K blocks are 144 bytes).
/// - Q8_0: array 0 = the 32 int8 quants of each block, array 1 = the block scales as f16 pairs (two blocks
///   per u32, rows padded to an even number of blocks).
/// - Q6_K: 212 bytes per block = ql, qh, scales (the first 208 bytes as they are) + d as an f32.
/// - Q4_0: array 0 = the 16 nibble bytes of each block (byte j holds weight j in its low nibble and weight
///   j + 16 in its high nibble, exactly as in the GGUF block), array 1 = scales as f16 pairs (as Q8_0).
/// - Q4_1: array 0 = the 16 nibble bytes, array 1 = one u32 per block (d in the low half, m in the high half).
/// - Q5_0: array 0 = the 16 nibble bytes, array 1 = the u32 `qh` of each block (bit j = 5th bit of weight j,
///   for j < 16 low nibble of byte j, for j >= 16 high nibble of byte j - 16), array 2 = scales as f16 pairs.
/// - Q5_1: arrays 0 and 1 as Q5_0, array 2 = one u32 per block (d, m).
/// - Q5_K: copied as it is (176 bytes = 11 vec4<u32> per block).
pub fn repack_rows(d: Dtype, src: &[u8], rows: usize, cols: usize) -> Result<Vec<Vec<u8>>, String> {
    check_shape(d, rows, cols)?;
    let rb = native_row_bytes(d, cols) as usize;
    if src.len() != rows * rb {
        return Err(format!("repack: {} bytes given, {} expected for {rows} rows", src.len(), rows * rb));
    }
    let nb = blocks_per_row(d, cols);
    Ok(match d {
        Dtype::F32 | Dtype::F16 | Dtype::Q4K | Dtype::Q5K => vec![src.to_vec()],
        Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 => {
            // (bytes of d and m, offset of qh if any, offset of the quants)
            let (dm_len, qh_off, qs_off) = match d {
                Dtype::Q4_0 => (2usize, None, 2usize),
                Dtype::Q4_1 => (4, None, 4),
                Dtype::Q5_0 => (2, Some(2usize), 6),
                _ => (4, Some(4usize), 8),
            };
            let bb = d.block().1;
            let mut qs = vec![0u8; rows * nb * 16];
            let mut qh = vec![0u8; if qh_off.is_some() { rows * nb * 4 } else { 0 }];
            // d alone: f16 pairs (rows padded to an even number of blocks); d and m: one u32 per block
            let dm_row = if dm_len == 2 { nb.div_ceil(2) * 4 } else { nb * 4 };
            let mut dm = vec![0u8; rows * dm_row];
            for r in 0..rows {
                for b in 0..nb {
                    let blk = &src[r * rb + b * bb..r * rb + (b + 1) * bb];
                    qs[(r * nb + b) * 16..(r * nb + b) * 16 + 16].copy_from_slice(&blk[qs_off..qs_off + 16]);
                    if let Some(o) = qh_off {
                        qh[(r * nb + b) * 4..(r * nb + b) * 4 + 4].copy_from_slice(&blk[o..o + 4]);
                    }
                    if dm_len == 2 {
                        let o = r * dm_row + (b / 2) * 4 + (b % 2) * 2;
                        dm[o..o + 2].copy_from_slice(&blk[0..2]);
                    } else {
                        let o = r * dm_row + b * 4;
                        dm[o..o + 4].copy_from_slice(&blk[0..4]);
                    }
                }
            }
            if qh_off.is_some() {
                vec![qs, qh, dm]
            } else {
                vec![qs, dm]
            }
        }
        Dtype::Q8_0 => {
            let sw = nb.div_ceil(2) * 4;
            let mut qs = vec![0u8; rows * nb * 32];
            let mut sc = vec![0u8; rows * sw];
            for r in 0..rows {
                for b in 0..nb {
                    let blk = &src[r * rb + b * 34..r * rb + b * 34 + 34];
                    qs[(r * nb + b) * 32..(r * nb + b) * 32 + 32].copy_from_slice(&blk[2..34]);
                    let o = r * sw + (b / 2) * 4 + (b % 2) * 2;
                    sc[o..o + 2].copy_from_slice(&blk[0..2]);
                }
            }
            vec![qs, sc]
        }
        Dtype::Q6K => {
            let mut out = vec![0u8; rows * nb * 212];
            for r in 0..rows {
                for b in 0..nb {
                    let blk = &src[r * rb + b * 210..r * rb + b * 210 + 210];
                    let o = (r * nb + b) * 212;
                    out[o..o + 208].copy_from_slice(&blk[0..208]);
                    let dv = quant::f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
                    out[o + 208..o + 212].copy_from_slice(&dv.to_le_bytes());
                }
            }
            vec![out]
        }
        _ => return Err(format!("{} has no GPU matvec kernel yet", name(d))),
    })
}

// ------------------------------------------------------------------ the layout contract, in Rust

/// Decodes one row of the repacked arrays (`arrays` as returned by `repack_rows`, `r` = row inside them) by the
/// SAME addressing the shaders use (which array, which byte, which bit). It is the layout contract in plain
/// Rust: the tests compare it bit for bit with the CPU engine's decoder (`quant::dequant_row`), so a wrong
/// nibble order or 5th-bit position is caught without a GPU. Only for Q4_0, Q4_1, Q5_0 and Q5_1.
#[allow(dead_code)]
pub fn decode_repacked_row(d: Dtype, arrays: &[Vec<u8>], r: usize, cols: usize) -> Vec<f32> {
    let nb = blocks_per_row(d, cols);
    let f16 = |a: &[u8], o: usize| quant::f16_to_f32(u16::from_le_bytes([a[o], a[o + 1]]));
    let u32at = |a: &[u8], o: usize| u32::from_le_bytes([a[o], a[o + 1], a[o + 2], a[o + 3]]);
    let mut out = vec![0f32; cols];
    for b in 0..nb {
        for j in 0..32usize {
            let qbyte = arrays[0][(r * nb + b) * 16 + (j & 15)] as u32;
            let nib = if j < 16 { qbyte & 15 } else { qbyte >> 4 };
            let (qh, dm): (u32, usize) = match d {
                Dtype::Q4_0 | Dtype::Q4_1 => (0, 1),
                _ => (u32at(&arrays[1], (r * nb + b) * 4), 2),
            };
            let bit5 = ((qh >> j) & 1) << 4; // for j >= 16 the bit is j as well (qh bit 16 + (j - 16))
            let (dv, mv) = match d {
                Dtype::Q4_0 | Dtype::Q5_0 => {
                    let row_bytes = nb.div_ceil(2) * 4;
                    (f16(&arrays[dm], r * row_bytes + (b / 2) * 4 + (b % 2) * 2), 0.0)
                }
                _ => (f16(&arrays[dm], (r * nb + b) * 4), f16(&arrays[dm], (r * nb + b) * 4 + 2)),
            };
            out[b * 32 + j] = match d {
                Dtype::Q4_0 => (nib as i32 - 8) as f32 * dv,
                Dtype::Q5_0 => ((nib | bit5) as i32 - 16) as f32 * dv,
                Dtype::Q4_1 => nib as f32 * dv + mv,
                Dtype::Q5_1 => (nib | bit5) as f32 * dv + mv,
                _ => panic!("decode_repacked_row: {} has no repack of this kind", name(d)),
            };
        }
    }
    out
}

// ------------------------------------------------------------------ splitting a matrix across buffers

/// A run of whole rows that lives in one set of buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowChunk {
    pub start: u64,
    pub rows: u64,
}

/// Splits `rows` rows into chunks so that every array of every chunk fits in `max_bytes` (the smaller of the
/// adapter's `max_buffer_size` and `max_storage_buffer_binding_size`). Chunks are row aligned, contiguous,
/// as big as allowed. `row_bytes` has the bytes per row of each array.
pub fn plan_chunks(rows: u64, row_bytes: &[u64], max_bytes: u64) -> Result<Vec<RowChunk>, String> {
    if rows == 0 {
        return Err("empty matrix".into());
    }
    let mut per = u64::MAX;
    for &rb in row_bytes {
        if rb == 0 {
            return Err("a row of 0 bytes".into());
        }
        per = per.min(max_bytes / rb);
    }
    if per == 0 {
        return Err(format!("one row ({} bytes) does not fit in a buffer of {max_bytes} bytes", row_bytes.iter().max().copied().unwrap_or(0)));
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < rows {
        let n = per.min(rows - start);
        out.push(RowChunk { start, rows: n });
        start += n;
    }
    Ok(out)
}

/// The invariants of a plan: contiguous from row 0, covers every row exactly once, every array fits.
pub fn check_plan(plan: &[RowChunk], rows: u64, row_bytes: &[u64], max_bytes: u64) -> Result<(), String> {
    let mut next = 0;
    for (i, c) in plan.iter().enumerate() {
        if c.rows == 0 {
            return Err(format!("chunk {i} is empty"));
        }
        if c.start != next {
            return Err(format!("chunk {i} starts at row {} but {next} was expected (gap or overlap)", c.start));
        }
        for &rb in row_bytes {
            if c.rows * rb > max_bytes {
                return Err(format!("chunk {i} needs {} bytes in one buffer, the limit is {max_bytes}", c.rows * rb));
            }
        }
        next += c.rows;
    }
    if next != rows {
        return Err(format!("the plan covers {next} rows of {rows}"));
    }
    Ok(())
}

// ------------------------------------------------------------------ synthetic data

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [-1, 1).
fn unit(state: &mut u64) -> f32 {
    ((splitmix(state) >> 40) as f32 / (1u64 << 23) as f32) - 1.0
}

/// A deterministic activation vector in [-1, 1).
pub fn gen_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0x1234_5678_9ABC_DEF0;
    (0..cols).map(|_| unit(&mut s)).collect()
}

fn put_f16(dst: &mut [u8], v: f32) {
    dst.copy_from_slice(&f32_to_f16(v).to_le_bytes());
}

/// Deterministic GGUF-layout rows `start..start+count` (each row depends only on `seed` and its index, so a
/// sample of rows can be regenerated for the reference without holding the whole matrix). Quantized formats
/// get random payload bytes (every bit pattern of the quants and of the packed scales) with valid f16 block
/// scales of a magnitude that keeps the dequantized values around 1.
pub fn gen_rows(d: Dtype, start: u64, count: u64, cols: usize, seed: u64) -> Vec<u8> {
    let rb = native_row_bytes(d, cols) as usize;
    let mut out = vec![0u8; rb * count as usize];
    for r in 0..count as usize {
        let row = &mut out[r * rb..(r + 1) * rb];
        let mut s = seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ (start + r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5555;
        match d {
            Dtype::F32 => {
                for c in row.chunks_exact_mut(4) {
                    c.copy_from_slice(&unit(&mut s).to_le_bytes());
                }
            }
            Dtype::F16 => {
                for c in row.chunks_exact_mut(2) {
                    put_f16(c, unit(&mut s));
                }
            }
            _ => {
                let (_, bb) = d.block();
                for blk in row.chunks_exact_mut(bb) {
                    let mut i = 0;
                    while i < bb {
                        let v = splitmix(&mut s).to_le_bytes();
                        let n = (bb - i).min(8);
                        blk[i..i + n].copy_from_slice(&v[..n]);
                        i += n;
                    }
                    // block scales: positive f16, the weights come out around magnitude 1
                    let pos = |s: &mut u64, base: f32| base * (0.5 + 0.5 * (unit(s) + 1.0));
                    match d {
                        Dtype::Q8_0 => put_f16(&mut blk[0..2], pos(&mut s, 1.0 / 64.0)),
                        Dtype::Q4K => {
                            put_f16(&mut blk[0..2], pos(&mut s, 1.0 / 400.0));
                            put_f16(&mut blk[2..4], pos(&mut s, 1.0 / 400.0));
                        }
                        Dtype::Q5K => {
                            put_f16(&mut blk[0..2], pos(&mut s, 1.0 / 800.0));
                            put_f16(&mut blk[2..4], pos(&mut s, 1.0 / 800.0));
                        }
                        Dtype::Q4_0 => put_f16(&mut blk[0..2], pos(&mut s, 1.0 / 8.0)),
                        Dtype::Q5_0 => put_f16(&mut blk[0..2], pos(&mut s, 1.0 / 16.0)),
                        Dtype::Q4_1 => {
                            put_f16(&mut blk[0..2], pos(&mut s, 1.0 / 15.0));
                            put_f16(&mut blk[2..4], unit(&mut s) * 0.5);
                        }
                        Dtype::Q5_1 => {
                            put_f16(&mut blk[0..2], pos(&mut s, 1.0 / 31.0));
                            put_f16(&mut blk[2..4], unit(&mut s) * 0.5);
                        }
                        Dtype::Q6K => put_f16(&mut blk[208..210], pos(&mut s, 1.0 / 2000.0)),
                        _ => {}
                    }
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------ reference and error metric

/// One row against `x`: (exact f64 dot of the f32-decoded weights, sum of |w x| in f64, the CPU engine's
/// float kernel `quant::dot_row` result).
pub fn reference_row(d: Dtype, row: &[u8], x: &[f32]) -> (f64, f64, f32) {
    let mut w = vec![0f32; x.len()];
    quant::dequant_row(d, row, &mut w);
    let (mut dot, mut abs) = (0f64, 0f64);
    for (a, b) in w.iter().zip(x) {
        let p = *a as f64 * *b as f64;
        dot += p;
        abs += p.abs();
    }
    (dot, abs, quant::dot_row(d, row, x))
}

/// Largest and root-mean-square error relative to the row's `sum |w x|` (the natural scale of float
/// rounding in a dot product: a plain relative error would blow up on rows whose dot cancels to ~0).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ErrStats {
    pub rows_checked: u32,
    pub max_gpu: f64,
    pub max_cpu: f64,
    pub rms_gpu: f64,
    pub rms_cpu: f64,
}

impl ErrStats {
    /// RMS error of the GPU divided by the RMS error of the CPU float kernel (the project's "ratio ~1.0").
    pub fn ratio(&self) -> f64 {
        if self.rms_cpu > 0.0 {
            self.rms_gpu / self.rms_cpu
        } else {
            f64::NAN
        }
    }
    /// A NaN error never passes.
    pub fn passes(&self, tol: f64) -> bool {
        self.rows_checked > 0 && self.max_gpu <= tol
    }
}

/// Accumulates `ErrStats` from (gpu value, reference) pairs.
#[derive(Default)]
pub struct ErrAcc {
    n: u32,
    max_g: f64,
    max_c: f64,
    ss_g: f64,
    ss_c: f64,
    nan: bool,
}

impl ErrAcc {
    pub fn add(&mut self, gpu: f32, reference: (f64, f64, f32)) {
        let scale = reference.1.max(1e-30);
        let eg = (gpu as f64 - reference.0).abs() / scale;
        let ec = (reference.2 as f64 - reference.0).abs() / scale;
        if eg.is_nan() {
            self.nan = true;
        }
        self.n += 1;
        self.max_g = self.max_g.max(eg);
        self.max_c = self.max_c.max(ec);
        self.ss_g += eg * eg;
        self.ss_c += ec * ec;
    }
    pub fn finish(&self) -> ErrStats {
        let n = self.n.max(1) as f64;
        ErrStats {
            rows_checked: self.n,
            max_gpu: if self.nan { f64::NAN } else { self.max_g },
            max_cpu: self.max_c,
            rms_gpu: if self.nan { f64::NAN } else { (self.ss_g / n).sqrt() },
            rms_cpu: (self.ss_c / n).sqrt(),
        }
    }
}

/// Tolerance on the error relative to `sum |w x|`. f32 rounding of the decoded weights and a different
/// summation order give ~1e-7; a wrong stride, row or reduction gives 1e-2 or more.
pub const TOL: f64 = 1e-5;

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Dtype; 10] = ALL_FORMATS;

    #[test]
    fn shapes_and_names() {
        for d in ALL {
            assert!(supported(d));
            assert_eq!(parse_dtype(name(d)), Some(d));
        }
        assert!(supported(Dtype::Q5K) && supported(Dtype::Q4_0) && !supported(Dtype::BF16));
        assert!(check_shape(Dtype::Q5K, 4, 896).is_err(), "896 is not a multiple of 256");
        assert!(check_shape(Dtype::Q4_0, 4, 896).is_ok() && check_shape(Dtype::Q5_1, 4, 4864).is_ok());
        assert!(check_shape(Dtype::Q4_0, 4, 48).is_err());
        assert!(check_shape(Dtype::Q4K, 4, 4864).is_ok());
        assert!(check_shape(Dtype::Q4K, 4, 896).is_err(), "896 is not a multiple of 256");
        assert!(check_shape(Dtype::Q8_0, 4, 896).is_ok());
        assert!(check_shape(Dtype::F16, 4, 12).is_err());
        assert!(check_shape(Dtype::F32, 0, 8).is_err());
        assert!(check_shape(Dtype::BF16, 1, 256).is_err());
    }

    #[test]
    fn row_bytes_are_u32_aligned_and_match_the_native_size() {
        for d in ALL {
            for cols in [256usize, 512, 4864] {
                if check_shape(d, 1, cols).is_err() {
                    continue;
                }
                for b in array_row_bytes(d, cols) {
                    assert_eq!(b % 4, 0, "{} {cols}", name(d));
                }
            }
        }
        // Q8_0 keeps 34 bytes per block when the row has an even number of blocks; Q4_K is the GGUF layout; Q6_K pays 2 bytes per block.
        assert_eq!(array_row_bytes(Dtype::Q8_0, 896).iter().sum::<u64>(), native_row_bytes(Dtype::Q8_0, 896));
        assert_eq!(array_row_bytes(Dtype::Q4K, 4864), vec![native_row_bytes(Dtype::Q4K, 4864)]);
        assert_eq!(array_row_bytes(Dtype::Q6K, 512)[0], native_row_bytes(Dtype::Q6K, 512) + 2 * 2);
        // an odd number of blocks pads the scales by one f16
        assert_eq!(array_row_bytes(Dtype::Q8_0, 96), vec![3 * 32, 8]);
    }

    #[test]
    fn repack_q8_0_moves_quants_and_scales() {
        let (rows, cols) = (3usize, 96usize); // 3 blocks per row: odd, so the scale array is padded
        let src = gen_rows(Dtype::Q8_0, 0, rows as u64, cols, 7);
        let arr = repack_rows(Dtype::Q8_0, &src, rows, cols).unwrap();
        assert_eq!(arr[0].len(), rows * 3 * 32);
        assert_eq!(arr[1].len(), rows * 8);
        for r in 0..rows {
            for b in 0..3 {
                let blk = &src[(r * 3 + b) * 34..(r * 3 + b) * 34 + 34];
                assert_eq!(&arr[0][(r * 3 + b) * 32..(r * 3 + b) * 32 + 32], &blk[2..]);
                let o = r * 8 + (b / 2) * 4 + (b % 2) * 2;
                assert_eq!(&arr[1][o..o + 2], &blk[..2]);
            }
            assert_eq!(&arr[1][r * 8 + 6..r * 8 + 8], &[0, 0], "padding must be zero");
        }
        assert!(repack_rows(Dtype::Q8_0, &src[1..], rows, cols).is_err());
    }

    #[test]
    fn repack_q6_k_adds_the_f32_scale() {
        let src = gen_rows(Dtype::Q6K, 0, 2, 512, 3);
        let arr = repack_rows(Dtype::Q6K, &src, 2, 512).unwrap();
        assert_eq!(arr[0].len(), 2 * 2 * 212);
        for blk in 0..4 {
            let s = &src[blk * 210..blk * 210 + 210];
            let o = blk * 212;
            assert_eq!(&arr[0][o..o + 208], &s[..208]);
            let d = quant::f16_to_f32(u16::from_le_bytes([s[208], s[209]]));
            assert_eq!(f32::from_le_bytes(arr[0][o + 208..o + 212].try_into().unwrap()), d);
            assert!(d > 0.0 && d < 1.0);
        }
    }

    #[test]
    fn repacked_legacy_formats_decode_like_the_cpu_decoder() {
        // Q4_0, Q4_1, Q5_0, Q5_1: the repacked arrays, read with the shaders' addressing, give the CPU decoder's
        // values bit for bit (odd block counts included: the d pairs are padded).
        for d in [Dtype::Q4_0, Dtype::Q4_1, Dtype::Q5_0, Dtype::Q5_1] {
            for cols in [32usize, 96, 896, 1120] {
                let rows = 5;
                let src = gen_rows(d, 2, rows as u64, cols, 5);
                let arr = repack_rows(d, &src, rows, cols).unwrap();
                let rb = native_row_bytes(d, cols) as usize;
                for (a, want) in arr.iter().zip(array_row_bytes(d, cols)) {
                    assert_eq!(a.len() as u64, want * rows as u64, "{} {cols}", name(d));
                }
                for r in 0..rows {
                    let mut cpu = vec![0f32; cols];
                    quant::dequant_row(d, &src[r * rb..(r + 1) * rb], &mut cpu);
                    let got = decode_repacked_row(d, &arr, r, cols);
                    for c in 0..cols {
                        assert_eq!(got[c].to_bits(), cpu[c].to_bits(), "{} cols {cols} row {r} col {c}: layout {} vs CPU {}", name(d), got[c], cpu[c]);
                    }
                }
            }
        }
    }

    #[test]
    fn layout_decoder_has_teeth() {
        // A decoder that swaps the nibble halves or reads the 5th bit from the wrong place must differ.
        let d = Dtype::Q5_0;
        let src = gen_rows(d, 0, 1, 64, 3);
        let arr = repack_rows(d, &src, 1, 64).unwrap();
        let mut cpu = vec![0f32; 64];
        quant::dequant_row(d, &src, &mut cpu);
        let mut swapped = arr.clone();
        for b in swapped[0].iter_mut() {
            *b = (*b << 4) | (*b >> 4);
        }
        assert_ne!(decode_repacked_row(d, &swapped, 0, 64), cpu);
        let mut wrong_qh = arr.clone();
        for c in wrong_qh[1].chunks_exact_mut(4) {
            let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]).rotate_left(16);
            c.copy_from_slice(&v.to_le_bytes());
        }
        assert_ne!(decode_repacked_row(d, &wrong_qh, 0, 64), cpu);
        assert_eq!(decode_repacked_row(d, &arr, 0, 64), cpu);
    }

    #[test]
    fn chunk_plan_covers_and_fits() {
        let rb = [100u64, 40];
        for (rows, max) in [(1u64, 100u64), (10, 100), (10, 250), (10, 1000), (151_936, 1 << 30), (7, 299)] {
            let plan = plan_chunks(rows, &rb, max).unwrap();
            check_plan(&plan, rows, &rb, max).unwrap_or_else(|e| panic!("rows={rows} max={max}: {e}"));
        }
        // 10 rows, 250 bytes: 2 rows of 100 bytes per chunk -> 5 chunks of 2
        let plan = plan_chunks(10, &rb, 250).unwrap();
        assert_eq!(plan.len(), 5);
        assert!(plan.iter().all(|c| c.rows == 2));
        // the biggest array decides
        assert_eq!(plan_chunks(10, &[100, 40], 399).unwrap()[0].rows, 3);
        assert!(plan_chunks(10, &[100], 99).is_err(), "a row that does not fit is an error");
        assert!(plan_chunks(0, &[100], 1000).is_err());
        assert!(plan_chunks(5, &[0], 1000).is_err());
    }

    #[test]
    fn chunk_plan_checker_has_teeth() {
        let rb = [100u64];
        let good = plan_chunks(10, &rb, 300).unwrap();
        assert!(check_plan(&good, 10, &rb, 300).is_ok());
        let mut gap = good.clone();
        gap[1].start += 1;
        assert!(check_plan(&gap, 10, &rb, 300).is_err(), "gap not detected");
        let mut short = good.clone();
        short.pop();
        assert!(check_plan(&short, 10, &rb, 300).is_err(), "missing rows not detected");
        let mut big = good.clone();
        big[0].rows += 1;
        big[1].start += 1;
        assert!(check_plan(&big, 10, &rb, 300).is_err(), "oversized chunk not detected");
        assert!(check_plan(&good, 10, &rb, 299).is_err());
    }

    #[test]
    fn rows_are_deterministic_and_independent_of_the_slice() {
        for d in ALL {
            let cols = 512;
            let a = gen_rows(d, 0, 6, cols, 9);
            let b = gen_rows(d, 4, 2, cols, 9);
            let rb = native_row_bytes(d, cols) as usize;
            assert_eq!(&a[4 * rb..], &b[..], "{}", name(d));
            assert_ne!(gen_rows(d, 0, 1, cols, 10), a[..rb].to_vec(), "{}: the seed must matter", name(d));
        }
    }

    #[test]
    fn reference_agrees_with_the_cpu_float_kernel() {
        for d in ALL {
            let cols = 1024;
            let row = gen_rows(d, 3, 1, cols, 1);
            let x = gen_x(cols, 2);
            let (dot, abs, cpu) = reference_row(d, &row, &x);
            assert!(abs > 1.0, "{}: the test data should not be degenerate", name(d));
            assert!(((cpu as f64) - dot).abs() / abs < TOL, "{}: CPU float kernel {cpu} vs exact {dot}", name(d));
        }
    }

    #[test]
    fn error_metric_has_teeth() {
        let r = (10.0f64, 100.0f64, 10.0f32);
        let mut a = ErrAcc::default();
        a.add(10.0, r);
        assert!(a.finish().passes(TOL));
        let mut a = ErrAcc::default();
        a.add(10.01, r); // 1e-4 of the abs sum
        assert!(!a.finish().passes(TOL));
        let mut a = ErrAcc::default();
        a.add(f32::NAN, r);
        assert!(!a.finish().passes(TOL), "NaN must fail");
        assert!(!ErrAcc::default().finish().passes(TOL), "no rows checked must fail");
    }
}
