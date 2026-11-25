// y = W x for f32 weights (task G4). One workgroup per row: the threads walk the row with a stride of WG
// vec4s, then a shared-memory tree reduction adds the partial sums. Requires cols % 4 == 0.
// params p: [0] rows in this chunk, [1] cols, [2] first output index (row offset of the chunk), [3] blocks per row.
// The WG line is rewritten by the host to tune the workgroup size.
const WG: u32 = 64u;
var<workgroup> scratch: array<f32, WG>;

@group(0) @binding(0) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = p[0];
    let row = wid.y * nwg.x + wid.x;
    var acc = 0.0;
    if (row < rows) {
        let k4 = p[1] / 4u;
        let base = row * k4;
        var i = lid.x;
        loop {
            if (i >= k4) { break; }
            acc += dot(w[base + i], x[i]);
            i += WG;
        }
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            scratch[lid.x] += scratch[lid.x + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid.x == 0u && row < rows) {
        y[p[2] + row] = scratch[0];
    }
}
