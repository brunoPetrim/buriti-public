// y = W x for Q4_0 weights (task G5b), repacked at upload time: binding 0 holds the 16 nibble bytes of each block
// (one vec4<u32>; byte j has weight j in the low nibble and weight j + 16 in the high nibble, as in the GGUF
// block), binding 1 the block scales as f16 pairs (two blocks per u32, the row padded to an even number of
// blocks). One item = one block = 32 weights. Weight = (nibble - 8) * d, as the CPU decoder.
// A workgroup computes ROWS consecutive rows: the x values of an item are loaded once and used by all of them.
// The two const lines below are rewritten by the host (workgroup size and rows per workgroup).
const WG: u32 = 64u;
const ROWS: u32 = 1u;
var<workgroup> scratch: array<f32, WG * ROWS>;

@group(0) @binding(0) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> sc: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32>;

fn nib(v: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>((v >> sh) & 15u, (v >> (sh + 8u)) & 15u, (v >> (sh + 16u)) & 15u, (v >> (sh + 24u)) & 15u));
}
fn w4(v: u32, sh: u32) -> vec4<f32> {
    return nib(v, sh) - vec4<f32>(8.0);
}

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = p[0];
    let row0 = (wid.y * nwg.x + wid.x) * ROWS;
    var acc: array<f32, ROWS>;
    let nb = p[3];
    let sw = (nb + 1u) / 2u;
    var i = lid.x;
    loop {
        if (i >= nb) { break; }
        let xi = i * 8u;
        let l0 = x[xi];
        let l1 = x[xi + 1u];
        let l2 = x[xi + 2u];
        let l3 = x[xi + 3u];
        let h0 = x[xi + 4u];
        let h1 = x[xi + 5u];
        let h2 = x[xi + 6u];
        let h3 = x[xi + 7u];
        for (var r = 0u; r < ROWS; r = r + 1u) {
            let row = row0 + r;
            if (row < rows) {
                let q = qs[row * nb + i];
                let s2 = unpack2x16float(sc[row * sw + (i >> 1u)]);
                let d = select(s2.x, s2.y, (i & 1u) == 1u);
                let lo = dot(w4(q.x, 0u), l0) + dot(w4(q.y, 0u), l1) + dot(w4(q.z, 0u), l2) + dot(w4(q.w, 0u), l3);
                let hi = dot(w4(q.x, 4u), h0) + dot(w4(q.y, 4u), h1) + dot(w4(q.z, 4u), h2) + dot(w4(q.w, 4u), h3);
                acc[r] += d * (lo + hi);
            }
        }
        i += WG;
    }
    for (var r = 0u; r < ROWS; r = r + 1u) {
        scratch[r * WG + lid.x] = acc[r];
    }
    workgroupBarrier();
    var s = WG / 2u;
    loop {
        if (s == 0u) { break; }
        if (lid.x < s) {
            for (var r = 0u; r < ROWS; r = r + 1u) {
                scratch[r * WG + lid.x] += scratch[r * WG + lid.x + s];
            }
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid.x < ROWS && row0 + lid.x < rows) {
        y[p[2] + row0 + lid.x] = scratch[lid.x * WG];
    }
}
