// Gated activations (task G6), elementwise on f32:
//   MODE 0: out = silu(g) * u        silu(g) = g / (1 + exp(-g))                       (SwiGLU: Qwen, Llama, Mistral, Phi)
//   MODE 1: out = gelu(g) * u        the tanh GELU as the CPU engine writes it, g / (1 + exp(-2 C (g + 0.044715 g^3))),
//                                    C = sqrt(2 / pi)                                    (GeGLU: Gemma)
//   MODE 2: out = gelu(g)            the same GELU with no gate (StarCoder2's plain MLP); `u` must still be bound
//                                    (any buffer of at least n floats, for instance g itself) but is not used
// The CPU engine has only this tanh form of GELU; there is no erf variant to match.
// The MODE line is rewritten by the host. p = [n, 0, 0, 0].
const MODE: u32 = 0u;
const C: f32 = 0.79788456;

@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    if (i >= p[0]) { return; }
    let x = g[i];
    let uv = u[i];
    var a: f32;
    if (MODE == 0u) {
        a = x / (1.0 + exp(-x));
    } else {
        a = x / (1.0 + exp(-2.0 * C * (x + 0.044715 * x * x * x)));
    }
    if (MODE == 2u) {
        out[i] = a;
    } else {
        out[i] = a * uv;
    }
}
