// Batched matrix product for prefill (task G9): Y[t][r] = sum_c W[r][c] * X[t][c] for the rows of ONE chunk of an uploaded
// matrix and T tokens at once. The weights are decoded by the format's own decoder section of gather_decoders.wgsl
// (`weight(lrow, c)`, the same code the embedding gather uses, so one decoder serves all ten formats); the decoded
// values are staged in shared memory once per tile and reused by every token of the tile, so the cost of decoding is
// divided by the number of tokens (64).
// Tile: 64 weight rows x 64 tokens, K walked in steps of 16 columns (every format's column count is a multiple of 32).
// Workgroup 16 x 16 threads, each thread owns a 4 x 4 block of the tile: rows ty + 16 i and tokens tx + 16 j (interleaved, so the
// 16 lanes of a row read 16 consecutive words of shared memory: no bank conflicts). The shared tiles are stored k-major and padded by
// one word (row stride 65) so that the staging writes, whose consecutive threads walk down k, do not collide either.
// 16 FMA per 8 shared-memory reads. Rows or tokens beyond the matrix or the batch are staged as zero and not written.
// `tp[0]` = number of tokens T of this call (the grid is sized for the capacity; workgroups whose first token is >= T leave at
// once, before any barrier, uniformly).
// p = [chunk_start, chunk_rows, cols, blocks per row, rows of the whole matrix, 0, 0, 0].
// Output layout: Y[t * m_total + chunk_start + r] (token-major, the layout the elementwise kernels expect).
// The markers @X@, @Y@, @P@, @T@ are the binding numbers after the weight buffers, filled in by the host.
@group(0) @binding(@X@) var<storage, read> xin: array<f32>;
@group(0) @binding(@Y@) var<storage, read_write> yout: array<f32>;
@group(0) @binding(@P@) var<storage, read> p: array<u32>;
@group(0) @binding(@T@) var<storage, read> tp: array<u32>;

fn byte_of(word: u32, k: u32) -> u32 {
    return (word << (24u - 8u * k)) >> 24u;
}
fn sbyte_of(word: u32, k: u32) -> i32 {
    return bitcast<i32>(word << (24u - 8u * k)) >> 24u;
}
// the f16 at the low (hi = false) or high half of a word
fn half_pair(word: u32, hi: bool) -> f32 {
    let v = unpack2x16float(word);
    return select(v.x, v.y, hi);
}

const TM: u32 = 64u;
const TN: u32 = 64u;
const KC: u32 = 16u;
// k-major tiles of 64 values per k, stored as vec4 so that the compute loop reads four rows (or four tokens) with ONE load
var<workgroup> wt: array<vec4<f32>, 256>;
var<workgroup> xt: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n_tok = tp[0];
    let tok0 = wid.y * TN;
    if (tok0 >= n_tok) { return; }
    let rows = p[1];
    let cols = p[2];
    let row0 = wid.x * TM;
    if (row0 >= rows) { return; }
    let tid = lid.y * 16u + lid.x;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }
    // staging: thread tid takes row sr = tid % 64 and the four consecutive columns kq .. kq + 3 (kq = 4 * (tid / 64)) of the 16 of this
    // step, so the decoder works on four weights at once (one word of quants, one scale) and the 32 lanes of a warp write 32
    // consecutive words of each shared row (no bank conflicts)
    let sr = tid >> 2u;
    let kq = (tid & 3u) * 4u;
    let sg = sr >> 2u;
    let sl = sr & 3u;
    for (var k0 = 0u; k0 < cols; k0 = k0 + KC) {
        var wv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (row0 + sr < rows) { wv = weight4(row0 + sr, k0 + kq); }
        wt[kq * 16u + sg][sl] = wv.x;
        wt[(kq + 1u) * 16u + sg][sl] = wv.y;
        wt[(kq + 2u) * 16u + sg][sl] = wv.z;
        wt[(kq + 3u) * 16u + sg][sl] = wv.w;
        if (tok0 + sr < n_tok) {
            let xb = (tok0 + sr) * cols + k0 + kq;
            xt[kq * 16u + sg][sl] = xin[xb];
            xt[(kq + 1u) * 16u + sg][sl] = xin[xb + 1u];
            xt[(kq + 2u) * 16u + sg][sl] = xin[xb + 2u];
            xt[(kq + 3u) * 16u + sg][sl] = xin[xb + 3u];
        } else {
            xt[kq * 16u + sg][sl] = 0.0;
            xt[(kq + 1u) * 16u + sg][sl] = 0.0;
            xt[(kq + 2u) * 16u + sg][sl] = 0.0;
            xt[(kq + 3u) * 16u + sg][sl] = 0.0;
        }
        workgroupBarrier();
        for (var k = 0u; k < KC; k = k + 1u) {
            let wq = wt[k * 16u + lid.y];
            let xq = xt[k * 16u + lid.x];
            let w0 = wq.x;
            let w1 = wq.y;
            let w2 = wq.z;
            let w3 = wq.w;
            let x0 = xq.x;
            let x1 = xq.y;
            let x2 = xq.z;
            let x3 = xq.w;
            acc[0] = acc[0] + w0 * x0;
            acc[1] = acc[1] + w0 * x1;
            acc[2] = acc[2] + w0 * x2;
            acc[3] = acc[3] + w0 * x3;
            acc[4] = acc[4] + w1 * x0;
            acc[5] = acc[5] + w1 * x1;
            acc[6] = acc[6] + w1 * x2;
            acc[7] = acc[7] + w1 * x3;
            acc[8] = acc[8] + w2 * x0;
            acc[9] = acc[9] + w2 * x1;
            acc[10] = acc[10] + w2 * x2;
            acc[11] = acc[11] + w2 * x3;
            acc[12] = acc[12] + w3 * x0;
            acc[13] = acc[13] + w3 * x1;
            acc[14] = acc[14] + w3 * x2;
            acc[15] = acc[15] + w3 * x3;
        }
        workgroupBarrier();
    }
    let m_total = p[4];
    let base = p[0];
    for (var i = 0u; i < 4u; i = i + 1u) {
        let r = row0 + lid.y * 4u + i;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let t = tok0 + lid.x * 4u + j;
            if (r < rows && t < n_tok) { yout[t * m_total + base + r] = acc[i * 4u + j]; }
        }
    }
}
