// Bias added to every token of a batch (task G9): x[t * n + i] += b[i] for t < T. The bias vector is shared by all tokens.
// x holds [token][n] (in place). p = [n, T, 0, 0]. The grid is sized for the capacity of the batch.
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    if (i >= p[0] * p[1]) { return; }
    x[i] = x[i] + b[i % p[0]];
}
