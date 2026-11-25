// Attention for ONE token over the GPU-resident f16 KV cache (task G7), with grouped-query attention (GQA) and an
// optional sliding window. One workgroup per query head h; its KV head is g = h / (n_head / n_kv), as on the CPU
// engine. Per head: (1) scores s[t] = scale * dot(q_h, K[t0 + t][g]) for t in 0..np, kept in the global scratch
// `sc` (one row of `stride` floats per head; shared memory is too small for long contexts); (2) a numerically
// stable softmax: max reduction, e = exp(s - max) written back in place, sum reduction; (3) out[h][d] =
// (sum_t e[t] * V[t0 + t][g][d]) / sum.
// K and V are f16 pairs packed in u32 (no shader-f16 feature): element i of the cache is half (i & 1) of word i >> 1.
// The cache layout is the CPU engine's: position t, head g starts at element t * kv_dim + g * hd (hd and kv_dim are even,
// so a head row starts on a word boundary and every access below reads whole words).
// Step 1 uses 8 lanes per position (16 positions per round): the 8 lanes read consecutive words of the K row, and lane 0 adds
// the 8 partial sums. Step 3 splits the positions over `ng` groups of lanes (a lane owns one pair of output dimensions, so
// the reads of V are coalesced) and adds the group partial sums at the end.
// p = [n_head, n_kv, hd, kv_dim, t0, np, stride, scale (f32 bits)]. np >= 1. hd is even.
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<u32>;
@group(0) @binding(2) var<storage, read> vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> p: array<u32>;

const WG: u32 = 128u;
var<workgroup> red: array<f32, 128>;
var<workgroup> red2: array<f32, 256>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let h = wid.y * nwg.x + wid.x;
    let n_head = p[0];
    if (h >= n_head) { return; }
    let n_kv = p[1];
    let hd = p[2];
    let kv_dim = p[3];
    let t0 = p[4];
    let np = p[5];
    let stride = p[6];
    let scale = bitcast<f32>(p[7]);
    let group = n_head / n_kv;
    let g = h / group;
    let qb = h * hd;
    let sb = h * stride;
    let kbw = (g * hd) >> 1u;
    let kvw = kv_dim >> 1u;
    let hp = hd >> 1u;

    // (1) scores
    let lane = lid.x & 7u;
    let slot = lid.x >> 3u;
    var tb = 0u;
    loop {
        if (tb >= np) { break; }
        let t = tb + slot;
        var s = 0.0;
        if (t < np) {
            let wb = (t0 + t) * kvw + kbw;
            for (var w = lane; w < hp; w = w + 8u) {
                let kv = unpack2x16float(kc[wb + w]);
                s = s + q[qb + 2u * w] * kv.x + q[qb + 2u * w + 1u] * kv.y;
            }
        }
        red[lid.x] = s;
        workgroupBarrier();
        if (lane == 0u && t < np) {
            var tot = 0.0;
            for (var l = 0u; l < 8u; l = l + 1u) {
                tot = tot + red[lid.x + l];
            }
            sc[sb + t] = tot * scale;
        }
        workgroupBarrier();
        tb = tb + 16u;
    }
    storageBarrier();
    workgroupBarrier();

    // (2) softmax: max
    var m = -3.4028235e38;
    var t = lid.x;
    loop {
        if (t >= np) { break; }
        m = max(m, sc[sb + t]);
        t = t + WG;
    }
    red[lid.x] = m;
    workgroupBarrier();
    var st = WG / 2u;
    loop {
        if (st == 0u) { break; }
        if (lid.x < st) { red[lid.x] = max(red[lid.x], red[lid.x + st]); }
        workgroupBarrier();
        st = st >> 1u;
    }
    let mx = red[0];
    workgroupBarrier();

    // softmax: exponentials (in place) and their sum
    var acc = 0.0;
    t = lid.x;
    loop {
        if (t >= np) { break; }
        let e = exp(sc[sb + t] - mx);
        sc[sb + t] = e;
        acc = acc + e;
        t = t + WG;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    st = WG / 2u;
    loop {
        if (st == 0u) { break; }
        if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let inv = 1.0 / red[0];
    storageBarrier();
    workgroupBarrier();

    // (3) weighted sum of V: lane `ln` of position group `jg` owns the output pair pb + ln
    let lanes = min(hp, WG);
    let ng = WG / lanes;
    let jg = lid.x / lanes;
    let ln = lid.x % lanes;
    var pb = 0u;
    loop {
        if (pb >= hp) { break; }
        let pr = pb + ln;
        var o0 = 0.0;
        var o1 = 0.0;
        if (jg < ng && pr < hp) {
            for (var j = jg; j < np; j = j + ng) {
                let e = sc[sb + j];
                let v = unpack2x16float(vc[(t0 + j) * kvw + kbw + pr]);
                o0 = o0 + e * v.x;
                o1 = o1 + e * v.y;
            }
        }
        red2[lid.x * 2u] = o0;
        red2[lid.x * 2u + 1u] = o1;
        workgroupBarrier();
        if (jg == 0u && pr < hp) {
            for (var k = 1u; k < ng; k = k + 1u) {
                o0 = o0 + red2[(k * lanes + ln) * 2u];
                o1 = o1 + red2[(k * lanes + ln) * 2u + 1u];
            }
            out[qb + 2u * pr] = o0 * inv;
            out[qb + 2u * pr + 1u] = o1 * inv;
        }
        workgroupBarrier();
        pb = pb + lanes;
    }
}
