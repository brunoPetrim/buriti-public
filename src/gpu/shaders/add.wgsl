// Residual add in place (task G6): x[i] += y[i] (IEEE addition: bit-identical to the CPU). p = [n, 0, 0, 0].
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    if (i >= p[0]) { return; }
    x[i] = x[i] + y[i];
}
