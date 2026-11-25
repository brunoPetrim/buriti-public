// Memory bandwidth, copy: dst = src with a grid stride over vec4 words (reads and writes the whole size).
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = arrayLength(&dst);
    let total = nwg.x * 256u;
    var i = gid.x;
    loop {
        if (i >= n) { break; }
        dst[i] = src[i];
        i += total;
    }
}
