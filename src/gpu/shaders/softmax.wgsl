// Softmax over rows (task G6), numerically stable: out[r][i] = exp(x[r][i] - max_r) / sum_r. One workgroup per row
// of n values: a max reduction, then exp and a sum reduction (the exponentials are kept in `out` by the same
// thread that reads them again, so no extra synchronization), then the division. A row of all -inf gives NaN,
// as the CPU engine does. p = [n, rows, 0, 0]. In and out are different buffers.
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = p[0];
    let row = wid.y * nwg.x + wid.x;
    if (row >= p[1]) { return; }
    let base = row * n;
    var m = -3.4028235e38;
    var i = lid.x;
    loop {
        if (i >= n) { break; }
        m = max(m, x[base + i]);
        i += 256u;
    }
    scratch[lid.x] = m;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let mx = scratch[0];
    workgroupBarrier();
    var acc = 0.0;
    i = lid.x;
    loop {
        if (i >= n) { break; }
        let e = exp(x[base + i] - mx);
        out[base + i] = e;
        acc += e;
        i += 256u;
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            scratch[lid.x] += scratch[lid.x + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = 1.0 / scratch[0];
    i = lid.x;
    loop {
        if (i >= n) { break; }
        out[base + i] = out[base + i] * inv;
        i += 256u;
    }
}
