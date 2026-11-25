// Elementwise f32 add: out[i] = a[i] + b[i]. The grid may be 2D (x * y workgroups) so that more than
// max_compute_workgroups_per_dimension * 64 elements can be covered.
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i < arrayLength(&out)) {
        out[i] = a[i] + b[i];
    }
}
