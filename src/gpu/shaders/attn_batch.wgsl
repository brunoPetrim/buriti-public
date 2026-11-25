// Causal attention for a batch of tokens (task G9): token `tok` of the batch sits at position pos0 + tok and attends to positions
// 0 ..= pos0 + tok of the f16 KV cache (which already holds the batch's own K and V: the host appends them first). The same math
// as attn_decode.wgsl (GQA, scale, a stable softmax), one workgroup per (query head, token): x = head, y = token. Scores live in a
// global scratch, one row of `stride` floats per (token, head). q and out hold [token][n_head][head_dim].
// Sliding window (Gemma 3 local layers, task G11): with p[8] = window > 0 a token at position `pos` sees only the last `window` positions,
// `pos + 1 - window ..= pos`, exactly as the CPU engine does; 0 = every earlier position.
// p = [n_head, n_kv, hd, kv_dim, pos0, T, stride, scale (f32 bits), window].
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<u32>;
@group(0) @binding(2) var<storage, read> vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> p: array<u32>;

const WG: u32 = 128u;
var<workgroup> red: array<f32, 128>;

fn k_at(i: u32) -> f32 {
    let v = unpack2x16float(kc[i >> 1u]);
    return select(v.x, v.y, (i & 1u) == 1u);
}

fn v_at(i: u32) -> f32 {
    let v = unpack2x16float(vc[i >> 1u]);
    return select(v.x, v.y, (i & 1u) == 1u);
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.x;
    let tok = wid.y;
    let n_head = p[0];
    if (h >= n_head || tok >= p[5]) { return; }
    let n_kv = p[1];
    let hd = p[2];
    let kv_dim = p[3];
    let pos = p[4] + tok;
    var t0 = 0u;
    if (p[8] > 0u && pos + 1u > p[8]) { t0 = pos + 1u - p[8]; }
    let np = pos + 1u - t0;
    let stride = p[6];
    let scale = bitcast<f32>(p[7]);
    let group = n_head / n_kv;
    let g = h / group;
    let row = tok * n_head + h;
    let qb = row * hd;
    let sb = row * stride;
    let kb = g * hd;

    var t = lid.x;
    loop {
        if (t >= np) { break; }
        let base = (t0 + t) * kv_dim + kb;
        var s = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            s = s + q[qb + d] * k_at(base + d);
        }
        sc[sb + t] = s * scale;
        t = t + WG;
    }
    storageBarrier();
    workgroupBarrier();

    var m = -3.4028235e38;
    t = lid.x;
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

    var d = lid.x;
    loop {
        if (d >= hd) { break; }
        var o = 0.0;
        for (var j = 0u; j < np; j = j + 1u) {
            o = o + sc[sb + j] * v_at((t0 + j) * kv_dim + kb + d);
        }
        out[qb + d] = o * inv;
        d = d + WG;
    }
}
