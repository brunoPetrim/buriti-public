// y = W x for F16 weights (task G4), WITHOUT the shader f16 feature (the 940MX has none): each u32 holds two
// halves (low half first, little endian) and is unpacked with unpack2x16float. One vec4<u32> = 8 weights.
// Requires cols % 8 == 0. Same params and reduction as mv_f32.
const WG: u32 = 64u;
var<workgroup> scratch: array<f32, WG>;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = p[0];
    let row = wid.y * nwg.x + wid.x;
    var acc = 0.0;
    if (row < rows) {
        let k8 = p[1] / 8u;
        let base = row * k8;
        var i = lid.x;
        loop {
            if (i >= k8) { break; }
            let q = w[base + i];
            let a = unpack2x16float(q.x);
            let b = unpack2x16float(q.y);
            let c = unpack2x16float(q.z);
            let d = unpack2x16float(q.w);
            acc += dot(vec4<f32>(a, b), x[2u * i]) + dot(vec4<f32>(c, d), x[2u * i + 1u]);
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
