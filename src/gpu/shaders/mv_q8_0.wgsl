// y = W x for Q8_0 weights (task G5), repacked at upload time: binding 0 holds the int8 quants of each
// row (32 bytes per block, no scale), binding 1 the block scales as f16 pairs (two blocks per u32, the row
// padded to an even number of blocks). One item = half a block (16 values = one vec4<u32>).
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

fn i8x4(v: u32) -> vec4<f32> {
    return vec4<f32>(vec4<i32>(bitcast<i32>(v << 24u) >> 24u, bitcast<i32>(v << 16u) >> 24u, bitcast<i32>(v << 8u) >> 24u, bitcast<i32>(v) >> 24u));
}

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = p[0];
    let row0 = (wid.y * nwg.x + wid.x) * ROWS;
    var acc: array<f32, ROWS>;
    let nb = p[3];
    let items = nb * 2u;
    let sw = (nb + 1u) / 2u;
    var i = lid.x;
    loop {
        if (i >= items) { break; }
        let b = i >> 1u;
        let xi = i * 4u;
        let x0 = x[xi];
        let x1 = x[xi + 1u];
        let x2 = x[xi + 2u];
        let x3 = x[xi + 3u];
        for (var r = 0u; r < ROWS; r = r + 1u) {
            let row = row0 + r;
            if (row < rows) {
                let q = qs[row * items + i];
                let s2 = unpack2x16float(sc[row * sw + (b >> 1u)]);
                let d = select(s2.x, s2.y, (b & 1u) == 1u);
                acc[r] += d * (dot(i8x4(q.x), x0) + dot(i8x4(q.y), x1) + dot(i8x4(q.z), x2) + dot(i8x4(q.w), x3));
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
