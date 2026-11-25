// Memory bandwidth, read only: every thread streams the buffer with a grid stride (vec4 loads, consecutive
// threads read consecutive 16-byte words) and writes one sum, so the loads cannot be optimized away.
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sink: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = arrayLength(&src);
    let total = nwg.x * 256u;
    var acc = vec4<f32>(0.0);
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        acc += src[i];
        i += total;
    }
    sink[gid.x] = acc.x + acc.y + acc.z + acc.w;
}
