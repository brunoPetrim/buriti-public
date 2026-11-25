// RoPE in place with a sin/cos table from the host (task G6, the default): the CPU engine's `rope_cs`. For every
// head h and frequency index i < hp (= n_rot / 2) the pair (a, b) is rotated:  a' = a cos - b sin,
// b' = a sin + b cos, with cs[2 i] = sin and cs[2 i + 1] = cos ALREADY multiplied by mscale (the longrope
// `attn_factor`) and computed by the host as the CPU engine computes them: (pos as f32 * inv_freq[i]).sin_cos().
// The host builds the table once per token (hp pairs, a few hundred bytes) and it serves every layer of the same
// frequency set, so the angles are bit-identical to the CPU engine's on every adapter (the built-in sin and cos of
// the GPU are not: see rope_trig.wgsl). Only the first n_rot = 2 * hp dimensions of each head rotate (partial
// rotary); the rest is left alone. The pair layout is either NeoX / Qwen / Phi style (i, i + hp) or Llama
// style, interleaved (2i, 2i + 1). One thread per (head, pair).
// p = [n_heads, head_dim, hp, interleaved (1) or neox (0)].
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> cs: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
    let hp = p[2];
    if (idx >= p[0] * hp) { return; }
    let h = idx / hp;
    let i = idx % hp;
    let base = h * p[1];
    var ia = base + i;
    var ib = base + i + hp;
    if (p[3] == 1u) {
        ia = base + 2u * i;
        ib = base + 2u * i + 1u;
    }
    let sn = cs[2u * i];
    let co = cs[2u * i + 1u];
    let a = v[ia];
    let b = v[ib];
    v[ia] = a * co - b * sn;
    v[ib] = a * sn + b * co;
}
