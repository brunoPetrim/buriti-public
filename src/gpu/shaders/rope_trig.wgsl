// RoPE in place with the angle computed on the GPU (task G6, variant B): the CPU engine's `rope`: for every head h and frequency index i < hp (= n_rot / 2)
// the pair (a, b) is rotated by the angle pos * inv_freq[i]:  a' = a cos - b sin,  b' = a sin + b cos, with sin
// and cos multiplied by mscale (the longrope `attn_factor`; 1 otherwise). Only the first n_rot = 2 * hp
// dimensions of each head rotate (partial rotary); the rest is left alone. The pair layout is either NeoX /
// Qwen / Phi style (i, i + hp) or Llama style, interleaved (2i, 2i + 1). `inv_freq` already holds the
// factors: theta^(-2i/n_rot) divided by the Llama 3 / longrope factor of the set chosen by the host (short or
// long). One thread per (head, pair).
// p = [n_heads, head_dim, hp, interleaved (1) or neox (0), pos, mscale as f32 bits, 2 pi pieces (4 f32 bits),
// pi/2 pieces (2 f32 bits)] (see `ops::ROPE_PIECES`).
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> inv_freq: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
    let hp = p[2];
    if (idx >= p[0] * hp) { return; }
    let h = idx / hp;
    let i = idx % hp;
    let base = h * p[1];
    var ia = base + i;
    var ib = base + i + hp;
    if (p[3] == 1u) {
        ia = base + 2u * i;
        ib = base + 2u * i + 1u;
    }
    let angle = f32(p[4]) * inv_freq[i];
    let ms = bitcast<f32>(p[5]);
    // The built-in sin and cos are not accurate enough on every adapter (measured: 1e-4 at an angle of 1000 on
    // the 940MX Vulkan, 7e-5 even at small angles on the DX12 drivers of the 940MX and the Intel HD 620), so
    // sin and cos are computed here from plain float arithmetic, with the same result on every adapter:
    // 1. reduce modulo 2 pi (Cody-Waite: 2 pi as four pieces of 8 bits, so every k * piece is exact for k < 2^16,
    //    i.e. angles up to about 4e5, which covers a 131072-token context at frequency 1);
    // 2. reduce to [-pi/4, pi/4] with the quadrant q (pi/2 as two pieces);
    // 3. Taylor polynomials of sin and cos on that interval (truncation error below 2e-9), rotated by q.
    // The pieces of 2 pi and pi/2 come in through the parameter buffer, not as constants: the DX12 shader compilers
    // fold `angle - k * C1 - k * C2` into `angle - k * (C1 + C2)`, which rounds the product and loses the exactness
    // (measured: 3e-5 at an angle of 664 on both DX12 adapters; Vulkan was exact).
    let k = round(angle * 0.15915494);
    var r = angle - k * bitcast<f32>(p[6]);
    r = r - k * bitcast<f32>(p[7]);
    r = r - k * bitcast<f32>(p[8]);
    r = r - k * bitcast<f32>(p[9]);
    let qd = round(r * 0.63661977);
    let x = (r - qd * bitcast<f32>(p[10])) - qd * bitcast<f32>(p[11]);
    let x2 = x * x;
    let sx = x + x * x2 * (-0.16666667 + x2 * (0.008333334 + x2 * (-0.0001984127 + x2 * 0.0000027557319)));
    let cx = 1.0 + x2 * (-0.5 + x2 * (0.041666668 + x2 * (-0.0013888889 + x2 * (0.0000248015873 + x2 * -0.00000027557319))));
    let q = i32(qd) & 3;
    var sn = sx;
    var cs = cx;
    if (q == 1) {
        sn = cx;
        cs = -sx;
    } else if (q == 2) {
        sn = -sx;
        cs = -cx;
    } else if (q == 3) {
        sn = -cx;
        cs = sx;
    }
    sn = sn * ms;
    cs = cs * ms;
    let a = v[ia];
    let b = v[ib];
    v[ia] = a * cs - b * sn;
    v[ib] = a * sn + b * cs;
}
