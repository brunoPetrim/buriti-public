// Copies one row of a batch into a vector (task G9): dst[i] = src[row * n + i]. Used to take the last token of a prefill batch
// out of the batch buffers for the final norm and the output projection. p = [n, row, 0, 0].
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    if (i >= p[0]) { return; }
    dst[i] = src[p[1] * p[0] + i];
}
