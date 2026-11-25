// y = W x for Q6_K weights (task G5), repacked at upload time: 53 u32 per block = ql (32 words), qh (16),
// the 16 int8 scales (4 words) and the block scale d as an f32 (1 word). GGUF has 210 bytes per block, which
// is not a multiple of 4. One item = 8 of the 32 "l" positions of one 128-value half = 32 weights.
// A workgroup computes ROWS consecutive rows: the x values of an item are loaded once and used by all of them.
// The two const lines below are rewritten by the host (workgroup size and rows per workgroup).
const WG: u32 = 64u;
const ROWS: u32 = 1u;
var<workgroup> scratch: array<f32, WG * ROWS>;

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;

fn b4(v: u32) -> vec4<f32> {
    return vec4<f32>(vec4<u32>(v & 255u, (v >> 8u) & 255u, (v >> 16u) & 255u, v >> 24u)) - vec4<f32>(32.0);
}
fn sc8(wd: u32, k: u32) -> f32 {
    return f32(bitcast<i32>(wd << (24u - 8u * k)) >> 24u);
}
// first word of block b of a row
fn blk(row: u32, nb: u32, b: u32) -> u32 {
    return row * nb * 53u + b * 53u;
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
        let n = (i >> 2u) & 1u;
        let c = i & 3u;
        var t1: array<f32, ROWS>;
        var t2: array<f32, ROWS>;
        var t3: array<f32, ROWS>;
        var t4: array<f32, ROWS>;
        for (var wi = 0u; wi < 2u; wi = wi + 1u) {
            let xi = b * 64u + n * 32u + 2u * c + wi;
            let xa = x[xi];
            let xb = x[xi + 8u];
            let xc = x[xi + 16u];
            let xd = x[xi + 24u];
            for (var r = 0u; r < ROWS; r = r + 1u) {
                let row = row0 + r;
                if (row < rows) {
                    let bb = blk(row, nb, b);
                    let qa = w[bb + n * 16u + 2u * c + wi];
                    let qb = w[bb + n * 16u + 8u + 2u * c + wi];
                    let hq = w[bb + 32u + n * 8u + 2u * c + wi];
                    let m = 0x0F0F0F0Fu;
                    let m3 = 0x03030303u;
                    let v1 = (qa & m) | ((hq & m3) << 4u);
                    let v2 = (qb & m) | (((hq >> 2u) & m3) << 4u);
                    let v3 = ((qa >> 4u) & m) | (((hq >> 4u) & m3) << 4u);
                    let v4 = ((qb >> 4u) & m) | (((hq >> 6u) & m3) << 4u);
                    t1[r] += dot(b4(v1), xa);
                    t2[r] += dot(b4(v2), xb);
                    t3[r] += dot(b4(v3), xc);
                    t4[r] += dot(b4(v4), xd);
                }
            }
        }
        let si = n * 8u + (c >> 1u);
        for (var r = 0u; r < ROWS; r = r + 1u) {
            let row = row0 + r;
            if (row < rows) {
                let bb = blk(row, nb, b);
                let d = bitcast<f32>(w[bb + 52u]);
                let s1 = sc8(w[bb + 48u + (si >> 2u)], si & 3u);
                let s2 = sc8(w[bb + 48u + ((si + 2u) >> 2u)], (si + 2u) & 3u);
                let s3 = sc8(w[bb + 48u + ((si + 4u) >> 2u)], (si + 4u) & 3u);
                let s4 = sc8(w[bb + 48u + ((si + 6u) >> 2u)], (si + 6u) & 3u);
                acc[r] += d * (s1 * t1[r] + s2 * t2[r] + s3 * t3[r] + s4 * t4[r]);
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
