// Sum of squares with a workgroup tree reduction in shared memory. Each workgroup walks the input with a
// grid stride, reduces its 256 partial sums and writes one value to partial[workgroup]. The CPU adds the
// partials (in f64), so the result needs a tolerance against the CPU reference, not equality.
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = arrayLength(&x);
    let stride = nwg.x * 256u;
    var acc = 0.0;
    var i = wid.x * 256u + lid.x;
    loop {
        if (i >= n) { break; }
        let v = x[i];
        acc += v * v;
        i += stride;
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
    if (lid.x == 0u) {
        partial[wid.x] = scratch[0];
    }
}
