// RoPE for a batch of tokens (task G9): the same rotation as rope.wgsl, with one sin/cos table per token. v holds
// [token][head][head_dim] (in place), cs holds [token][2 * hp] (sin, cos pairs already times mscale, built by the host
// as `ops::rope_table(pos0 + t, ...)`), hp = n_rot / 2. One thread per (token, head, pair); tokens >= T are skipped (the
// grid is sized for the capacity of the batch).
// p = [n_heads, head_dim, hp, interleaved (1) or neox (0), T, 0, 0, 0].
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> cs: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
    let hp = p[2];
    let per_tok = p[0] * hp;
    if (idx >= p[4] * per_tok) { return; }
    let t = idx / per_tok;
    let r = idx % per_tok;
    let h = r / hp;
    let i = r % hp;
    let base = (t * p[0] + h) * p[1];
    var ia = base + i;
    var ib = base + i + hp;
    if (p[3] == 1u) {
        ia = base + 2u * i;
        ib = base + 2u * i + 1u;
    }
    let sn = cs[t * 2u * hp + 2u * i];
    let co = cs[t * 2u * hp + 2u * i + 1u];
    let a = v[ia];
    let b = v[ib];
    v[ia] = a * co - b * sn;
    v[ib] = a * sn + b * co;
}
