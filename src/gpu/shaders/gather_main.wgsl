// Embedding gather (task G6): out[t][c] = scale * W[ids[t]][c] for the rows of ONE chunk of an uploaded matrix
// (the same repacked buffers the matvec kernels read; `weight(lrow, c)` comes from the format's decoder section
// of gather_decoders.wgsl, which also declares the weight bindings 0..n-1). The three markers below are rewritten
// by the host with the binding numbers that follow the weight buffers. A token whose row is not in this chunk
// is skipped (the host dispatches every chunk; exactly one writes each token). One thread per (token, column).
// p = [chunk_start, chunk_rows, cols, blocks per row, n_ids, scale as f32 bits, 0, 0].
@group(0) @binding(@IDS@) var<storage, read> ids: array<u32>;
@group(0) @binding(@OUT@) var<storage, read_write> out: array<f32>;
@group(0) @binding(@P@) var<storage, read> p: array<u32>;

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

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let cols = p[2];
    let t = wid.y;
    let c = wid.x * 64u + lid.x;
    if (t >= p[4] || c >= cols) { return; }
    let id = ids[t];
    if (id < p[0] || id >= p[0] + p[1]) { return; }
    out[t * cols + c] = weight(id - p[0], c) * bitcast<f32>(p[5]);
}
