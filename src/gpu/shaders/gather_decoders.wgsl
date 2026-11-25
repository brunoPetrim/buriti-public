// Per-format weight decoders for the embedding gather (task G6). Each section starts with a `//@@ <FORMAT>` line,
// declares the weight bindings (0..n-1, as the matvec upload lays them out, see `pack::repack_rows`) and defines
// `weight(lrow, c)`: the value of column c of row lrow (the row inside the chunk). `p[2]` = columns,
// `p[3]` = blocks per row. Bytes are taken with `byte_of` (shift left, then right: the form that is correct on
// every adapter, see D64/D65), never with `(w >> (k * 8)) & 255`.
//@@ F32
@group(0) @binding(0) var<storage, read> w: array<f32>;
fn weight(lrow: u32, c: u32) -> f32 {
    return w[lrow * p[2] + c];
}
//@@ F16
@group(0) @binding(0) var<storage, read> w: array<u32>;
fn weight(lrow: u32, c: u32) -> f32 {
    let idx = lrow * p[2] + c;
    let v = unpack2x16float(w[idx >> 1u]);
    return select(v.x, v.y, (idx & 1u) == 1u);
}
//@@ Q8_0
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> sc: array<u32>;
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let b = c >> 5u;
    let at = (lrow * nb + b) * 32u + (c & 31u);
    let q = sbyte_of(qs[at >> 2u], at & 3u);
    let d = half_pair(sc[lrow * ((nb + 1u) / 2u) + (b >> 1u)], (b & 1u) == 1u);
    return f32(q) * d;
}
// Four consecutive columns c..c+3 (c a multiple of 4): one word of quants and one scale. The same arithmetic per element as `weight`.
fn weight4(lrow: u32, c: u32) -> vec4<f32> {
    let nb = p[3];
    let b = c >> 5u;
    let at = (lrow * nb + b) * 32u + (c & 31u);
    let word = qs[at >> 2u];
    let hi = (b & 1u) == 1u;
    let d = half_pair(sc[lrow * ((nb + 1u) / 2u) + (b >> 1u)], hi);
    return vec4<f32>(f32(sbyte_of(word, 0u)) * d, f32(sbyte_of(word, 1u)) * d, f32(sbyte_of(word, 2u)) * d, f32(sbyte_of(word, 3u)) * d);
}
//@@ Q4_0
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> sc: array<u32>;
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let b = c >> 5u;
    let j = c & 31u;
    let qb = byte_of(qs[(lrow * nb + b) * 4u + ((j & 15u) >> 2u)], j & 3u);
    let nib = select(qb & 15u, qb >> 4u, j >= 16u);
    let d = half_pair(sc[lrow * ((nb + 1u) / 2u) + (b >> 1u)], (b & 1u) == 1u);
    return f32(i32(nib) - 8) * d;
}
//@@ Q4_1
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> dm: array<u32>;
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let b = c >> 5u;
    let j = c & 31u;
    let qb = byte_of(qs[(lrow * nb + b) * 4u + ((j & 15u) >> 2u)], j & 3u);
    let nib = select(qb & 15u, qb >> 4u, j >= 16u);
    let s = unpack2x16float(dm[lrow * nb + b]);
    return f32(nib) * s.x + s.y;
}
//@@ Q5_0
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> qh: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<u32>;
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let b = c >> 5u;
    let j = c & 31u;
    let qb = byte_of(qs[(lrow * nb + b) * 4u + ((j & 15u) >> 2u)], j & 3u);
    let nib = select(qb & 15u, qb >> 4u, j >= 16u);
    let hi5 = ((qh[lrow * nb + b] >> j) & 1u) << 4u;
    let d = half_pair(sc[lrow * ((nb + 1u) / 2u) + (b >> 1u)], (b & 1u) == 1u);
    return f32(i32(nib | hi5) - 16) * d;
}
//@@ Q5_1
@group(0) @binding(0) var<storage, read> qs: array<u32>;
@group(0) @binding(1) var<storage, read> qh: array<u32>;
@group(0) @binding(2) var<storage, read> dm: array<u32>;
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let b = c >> 5u;
    let j = c & 31u;
    let qb = byte_of(qs[(lrow * nb + b) * 4u + ((j & 15u) >> 2u)], j & 3u);
    let nib = select(qb & 15u, qb >> 4u, j >= 16u);
    let hi5 = ((qh[lrow * nb + b] >> j) & 1u) << 4u;
    let s = unpack2x16float(dm[lrow * nb + b]);
    return f32(nib | hi5) * s.x + s.y;
}
//@@ Q4_K
@group(0) @binding(0) var<storage, read> w: array<u32>;
fn sc_byte(base: u32, j: u32) -> u32 {
    return byte_of(w[base + 1u + (j >> 2u)], j & 3u);
}
// get_scale_min_k4 of ggml: 6-bit scale and min of group j (0..7).
fn scale_min(base: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(sc_byte(base, j) & 63u, sc_byte(base, j + 4u) & 63u);
    }
    let a = sc_byte(base, j + 4u);
    return vec2<u32>((a & 15u) | ((sc_byte(base, j - 4u) >> 6u) << 4u), (a >> 4u) | ((sc_byte(base, j) >> 6u) << 4u));
}
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let base = (lrow * nb + (c >> 8u)) * 36u;
    let e = c & 255u;
    let g = e >> 5u;
    let off = (g >> 1u) * 32u + (e & 31u);
    let qb = byte_of(w[base + 4u + (off >> 2u)], off & 3u);
    let nib = select(qb & 15u, qb >> 4u, (g & 1u) == 1u);
    let dd = unpack2x16float(w[base]);
    let sm = scale_min(base, g);
    return dd.x * f32(sm.x) * f32(nib) - dd.y * f32(sm.y);
}
// Four consecutive columns c..c+3 (c a multiple of 4, so they share the group, the scale and one word of quants).
fn weight4(lrow: u32, c: u32) -> vec4<f32> {
    let nb = p[3];
    let base = (lrow * nb + (c >> 8u)) * 36u;
    let e = c & 255u;
    let g = e >> 5u;
    let off = (g >> 1u) * 32u + (e & 31u);
    let word = w[base + 4u + (off >> 2u)];
    let hi = (g & 1u) == 1u;
    let dd = unpack2x16float(w[base]);
    let sm = scale_min(base, g);
    let ds = dd.x * f32(sm.x);
    let mn = dd.y * f32(sm.y);
    let q0 = byte_of(word, 0u);
    let q1 = byte_of(word, 1u);
    let q2 = byte_of(word, 2u);
    let q3 = byte_of(word, 3u);
    return vec4<f32>(
        ds * f32(select(q0 & 15u, q0 >> 4u, hi)) - mn,
        ds * f32(select(q1 & 15u, q1 >> 4u, hi)) - mn,
        ds * f32(select(q2 & 15u, q2 >> 4u, hi)) - mn,
        ds * f32(select(q3 & 15u, q3 >> 4u, hi)) - mn);
}
//@@ Q5_K
@group(0) @binding(0) var<storage, read> w: array<u32>;
fn sc_byte(base: u32, j: u32) -> u32 {
    return byte_of(w[base + 1u + (j >> 2u)], j & 3u);
}
fn scale_min(base: u32, j: u32) -> vec2<u32> {
    if (j < 4u) {
        return vec2<u32>(sc_byte(base, j) & 63u, sc_byte(base, j + 4u) & 63u);
    }
    let a = sc_byte(base, j + 4u);
    return vec2<u32>((a & 15u) | ((sc_byte(base, j - 4u) >> 6u) << 4u), (a >> 4u) | ((sc_byte(base, j) >> 6u) << 4u));
}
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let base = (lrow * nb + (c >> 8u)) * 44u;
    let e = c & 255u;
    let g = e >> 5u;
    let l = e & 31u;
    let off = (g >> 1u) * 32u + l;
    let qb = byte_of(w[base + 12u + (off >> 2u)], off & 3u);
    let qhb = byte_of(w[base + 4u + (l >> 2u)], l & 3u);
    let nib = select(qb & 15u, qb >> 4u, (g & 1u) == 1u) | (((qhb >> g) & 1u) << 4u);
    let dd = unpack2x16float(w[base]);
    let sm = scale_min(base, g);
    return dd.x * f32(sm.x) * f32(nib) - dd.y * f32(sm.y);
}
//@@ Q6_K
@group(0) @binding(0) var<storage, read> w: array<u32>;
fn weight(lrow: u32, c: u32) -> f32 {
    let nb = p[3];
    let base = (lrow * nb + (c >> 8u)) * 53u;
    let e = c & 255u;
    let n = e >> 7u;
    let quarter = (e & 127u) >> 5u;
    let l = e & 31u;
    let oq = n * 64u + (quarter & 1u) * 32u + l;
    let qlb = byte_of(w[base + (oq >> 2u)], oq & 3u);
    let nib = select(qlb & 15u, qlb >> 4u, quarter >= 2u);
    let qhb = byte_of(w[base + 32u + ((n * 32u + l) >> 2u)], l & 3u);
    let q = i32(nib | (((qhb >> (quarter * 2u)) & 3u) << 4u)) - 32;
    let si = n * 8u + (l >> 4u) + quarter * 2u;
    let s = sbyte_of(w[base + 48u + (si >> 2u)], si & 3u);
    return bitcast<f32>(w[base + 52u]) * f32(s) * f32(q);
}
// Four consecutive columns c..c+3 (c a multiple of 4): one word of low quants, one of high bits, one scale.
fn q6_val(wl: u32, wh: u32, j: u32, quarter: u32) -> i32 {
    let qlb = byte_of(wl, j);
    let nib = select(qlb & 15u, qlb >> 4u, quarter >= 2u);
    let qhb = byte_of(wh, j);
    let sh = quarter * 2u;
    let hb = (qhb >> sh) & 3u;
    return i32(nib | (hb << 4u)) - 32;
}
fn weight4(lrow: u32, c: u32) -> vec4<f32> {
    let nb = p[3];
    let base = (lrow * nb + (c >> 8u)) * 53u;
    let e = c & 255u;
    let n = e >> 7u;
    let quarter = (e & 127u) >> 5u;
    let l = e & 31u;
    let oq = n * 64u + (quarter & 1u) * 32u + l;
    let wl = w[base + (oq >> 2u)];
    let wh = w[base + 32u + ((n * 32u + l) >> 2u)];
    let si = quarter * 2u + n * 8u + (l >> 4u);
    let s = sbyte_of(w[base + 48u + (si >> 2u)], si & 3u);
    let ds = bitcast<f32>(w[base + 52u]) * f32(s);
    return vec4<f32>(ds * f32(q6_val(wl, wh, 0u, quarter)), ds * f32(q6_val(wl, wh, 1u, quarter)), ds * f32(q6_val(wl, wh, 2u, quarter)), ds * f32(q6_val(wl, wh, 3u, quarter)));
}
