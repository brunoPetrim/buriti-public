// KV cache append (task G7): stores one token's K and V rows (kv_dim f32 values each) into the f16 caches at
// position `pos`, two halves per u32 (the same [position][kv_dim] layout as the CPU engine's `Vec<u16>` cache; kv_dim
// is even). No shader-f16 feature is needed.
// The conversion is written with integer operations instead of pack2x16float: the WGSL spec leaves the rounding of
// pack2x16float to the implementation, and the Microsoft WARP software adapter was measured to differ from the CPU
// conversion in about half of the values (it truncates). This routine is a line-by-line port of `f32_to_f16` in
// src/native/f16.rs (round to nearest even, saturate at +-65504 instead of infinity, NaN as 0x7E00), so the GPU cache
// is bit-identical to the CPU engine's cache on every adapter.
// p = [kv_dim, pos, 0, 0].
@group(0) @binding(0) var<storage, read> kk: array<f32>;
@group(0) @binding(1) var<storage, read> vv: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<u32>;
@group(0) @binding(3) var<storage, read_write> vc: array<u32>;
@group(0) @binding(4) var<storage, read> p: array<u32>;

fn f32_to_f16(v_in: f32) -> u32 {
    let xin = bitcast<u32>(v_in);
    let sign_in = (xin >> 16u) & 0x8000u;
    if (((xin >> 23u) & 255u) == 255u && (xin & 0x7FFFFFu) != 0u) {
        return sign_in | 0x7E00u;
    }
    // not NaN: clamp saturates the infinities as well
    let v = clamp(v_in, -65504.0, 65504.0);
    let x = bitcast<u32>(v);
    let sign = (x >> 16u) & 0x8000u;
    let exp = i32((x >> 23u) & 255u);
    let man = x & 0x7FFFFFu;
    let e = exp - 112;
    if (e <= 0) {
        if (e < -10) { return sign; }
        let m = man | 0x800000u;
        let shift = u32(14 - e);
        var h = m >> shift;
        let rem = m & ((1u << shift) - 1u);
        let half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u) == 1u)) { h = h + 1u; }
        return sign | h;
    }
    var h = (u32(e) << 10u) | (man >> 13u);
    let rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) == 1u)) { h = h + 1u; }
    if (h >= 0x7C00u) { h = 0x7BFFu; }
    return sign | h;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
    let half_dim = p[0] / 2u;
    if (i >= half_dim) { return; }
    let dst = p[1] * half_dim + i;
    kc[dst] = f32_to_f16(kk[2u * i]) | (f32_to_f16(kk[2u * i + 1u]) << 16u);
    vc[dst] = f32_to_f16(vv[2u * i]) | (f32_to_f16(vv[2u * i + 1u]) << 16u);
}
