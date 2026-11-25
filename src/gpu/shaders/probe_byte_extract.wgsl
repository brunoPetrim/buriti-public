// Informational probe (D64/D65): does the OLD byte-extract form `(w >> ((j & 3u) * 8u)) & 255u`, with a
// data-dependent shift, give the right byte on this adapter? It failed on the WARP software adapter for a
// shift of 24 (Q4_K scale bytes). out_old / out_vec use the old form on a plain u32 array and on a vec4<u32>
// element indexed by data; out_new uses the form the kernels use now, `(w << (24u - 8u * (j & 3u))) >> 24u`.
// Thread t reads word t / 4 and byte t % 4, so every thread count in a multiple of 4 covers all four shifts.
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> words4: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> out_old: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_vec: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_new: array<u32>;
@group(0) @binding(5) var<storage, read> p: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= p[0]) { return; }
    let j = t;
    let wi = j >> 2u;
    out_old[t] = (words[wi] >> ((j & 3u) * 8u)) & 255u;
    // a vec4<u32> element chosen by data: component (j >> 2) & 3 of vec4 j >> 4 (the way Q4_K reads its scale bytes)
    out_vec[t] = (words4[j >> 4u][(j >> 2u) & 3u] >> ((j & 3u) * 8u)) & 255u;
    out_new[t] = (words[wi] << (24u - 8u * (j & 3u))) >> 24u;
}
