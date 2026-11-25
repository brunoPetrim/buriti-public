// RMSNorm with weight (task G6): out[r][i] = x[r][i] * inv_r * w[i], inv_r = 1 / sqrt(mean(x[r]^2) + eps), the
// CPU engine's `rmsnorm` (same order: (x * inv) * w). One workgroup per row of n values; the same weight vector
// serves every row (rows = the heads of a per-head norm such as Qwen3's q/k norm, or 1 for the hidden state).
// p = [n, rows, eps as f32 bits, 0]. In and out are different buffers.
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = p[0];
    let row = wid.y * nwg.x + wid.x;
    if (row >= p[1]) { return; }
    let base = row * n;
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= n) { break; }
        let v = x[base + i];
        acc += v * v;
        i += 256u;
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            scratch[lid.x] += scratch[lid.x + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let ss = scratch[0] / f32(n);
    let inv = 1.0 / sqrt(ss + bitcast<f32>(p[2]));
    i = lid.x;
    loop {
        if (i >= n) { break; }
        out[base + i] = x[base + i] * inv * w[i];
        i += 256u;
    }
}
