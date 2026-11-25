// LayerNorm with weight and bias (task G6, StarCoder2): out[r][i] = (x[r][i] - mean_r) * inv_r * w[i] + b[i],
// inv_r = 1 / sqrt(var_r + eps), the variance taken around the mean (two passes, as the CPU engine's `layernorm`;
// the CPU accumulates in f64, here the sums are f32 trees). One workgroup per row of n values.
// p = [n, rows, eps as f32 bits, 0]. In and out are different buffers.
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32>;

var<workgroup> scratch: array<f32, 256>;

// Sum of one value per thread over the workgroup; every thread gets the total. Ends with a barrier so the
// scratch array can be reused right away.
fn block_sum(v: f32, lid: u32) -> f32 {
    scratch[lid] = v;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) {
            scratch[lid] += scratch[lid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let total = scratch[0];
    workgroupBarrier();
    return total;
}

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
        acc += x[base + i];
        i += 256u;
    }
    let mean = block_sum(acc, lid.x) / f32(n);
    var acc2 = 0.0;
    i = lid.x;
    loop {
        if (i >= n) { break; }
        let d = x[base + i] - mean;
        acc2 += d * d;
        i += 256u;
    }
    let variance = block_sum(acc2, lid.x) / f32(n);
    let inv = 1.0 / sqrt(variance + bitcast<f32>(p[2]));
    i = lid.x;
    loop {
        if (i >= n) { break; }
        out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
        i += 256u;
    }
}
