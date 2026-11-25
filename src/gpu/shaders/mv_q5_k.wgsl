// y = W x for Q5_K weights (task G5b), GGUF-native layout (176 bytes = 11 vec4<u32> per block, so no repack):
// vec4 0 = [d and dmin as two f16, 12 scale bytes], vec4 1..2 = qh (32 bytes: the 5th bit of group g of byte l
// is bit g of qh[l]), vec4 3..10 = the 128 nibble bytes. One item = 16 bytes of quants = 32 weights (16 low
// nibbles of scale group 2g and the 16 high nibbles of group 2g + 1, for bytes h*16..h*16+15 of the 32 of group
// pair g); the matching 16 bytes of qh are vec4 1 + h. Weight = d * sc * (nibble | 5th bit << 4) - dmin * m.
// A workgroup computes ROWS consecutive rows: the x values of an item are loaded once and used by all of them.
// The two const lines below are rewritten by the host (workgroup size and rows per workgroup).
const WG: u32 = 64u;
const ROWS: u32 = 1u;
var<workgroup> scratch: array<f32, WG * ROWS>;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;

fn sbyte(h: vec4<u32>, j: u32) -> u32 {
    return (h[1u + (j >> 2u)] << (24u - 8u * (j & 3u))) >> 24u;
}
// get_scale_min_k4 of ggml: 6-bit scale and min of group j (0..7) out of the 12 packed bytes.
fn scale_min(h: vec4<u32>, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(sbyte(h, j) & 63u, sbyte(h, j + 4u) & 63u);
    }
    let a = sbyte(h, j + 4u);
    return vec2<u32>((a & 15u) | ((sbyte(h, j - 4u) >> 6u) << 4u), (a >> 4u) | ((sbyte(h, j) >> 6u) << 4u));
}
fn nib(v: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>((v >> sh) & 15u, (v >> (sh + 8u)) & 15u, (v >> (sh + 16u)) & 15u, (v >> (sh + 24u)) & 15u));
}
// The 5th bit (0.0 or 16.0) of each of the 4 bytes of `t`, where `t` is a word of qh already shifted so that the
// wanted group bit is bit 0 of every byte.
fn hbit(t: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>((t & 1u) << 4u, ((t >> 8u) & 1u) << 4u, ((t >> 16u) & 1u) << 4u, ((t >> 24u) & 1u) << 4u));
}

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let rows = p[0];
    let row0 = (wid.y * nwg.x + wid.x) * ROWS;
    var acc: array<f32, ROWS>;
    let nb = p[3];
    let items = nb * 8u;
    var i = lid.x;
    loop {
        if (i >= items) { break; }
        let b = i >> 3u;
        let g = (i >> 1u) & 3u;
        let h = i & 1u;
        let xi = b * 64u + g * 16u + h * 4u;
        let l0 = x[xi];
        let l1 = x[xi + 1u];
        let l2 = x[xi + 2u];
        let l3 = x[xi + 3u];
        let h0 = x[xi + 8u];
        let h1 = x[xi + 9u];
        let h2 = x[xi + 10u];
        let h3 = x[xi + 11u];
        let ones = vec4<f32>(1.0);
        let sl = dot(l0 + l1 + l2 + l3, ones);
        let sh = dot(h0 + h1 + h2 + h3, ones);
        for (var r = 0u; r < ROWS; r = r + 1u) {
            let row = row0 + r;
            if (row < rows) {
                let rb = row * nb * 11u + b * 11u;
                let hd = w[rb];
                let qv = w[rb + 1u + h];
                let q = w[rb + 3u + g * 2u + h];
                let dd = unpack2x16float(hd.x);
                let s1 = scale_min(hd, 2u * g);
                let s2 = scale_min(hd, 2u * g + 1u);
                let bl = qv >> vec4<u32>(2u * g);
                let bh = qv >> vec4<u32>(2u * g + 1u);
                let lo = dot(nib(q.x, 0u) + hbit(bl.x), l0) + dot(nib(q.y, 0u) + hbit(bl.y), l1) + dot(nib(q.z, 0u) + hbit(bl.z), l2) + dot(nib(q.w, 0u) + hbit(bl.w), l3);
                let hi = dot(nib(q.x, 4u) + hbit(bh.x), h0) + dot(nib(q.y, 4u) + hbit(bh.y), h1) + dot(nib(q.z, 4u) + hbit(bh.z), h2) + dot(nib(q.w, 4u) + hbit(bh.w), h3);
                acc[r] += dd.x * (f32(s1.x) * lo + f32(s2.x) * hi) - dd.y * (f32(s1.y) * sl + f32(s2.y) * sh);
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
