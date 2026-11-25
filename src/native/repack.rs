//! Pesos Q4_K reempacotados em blocos de 8 linhas intercaladas (no estilo do `block_q4_Kx8` do
//! llama.cpp) e o kernel AVX2 que os multiplica (experimento E5, decisão D48; o micro-benchmark
//! está em `repack_exp.rs`, decisão D47).
//!
//! Layout de um bloco de 8 linhas (1184 bytes; o do llama.cpp tem 1152 porque guarda escalas e
//! mínimos em 96 bytes empacotados, aqui ficam em bytes soltos para simplificar o desempacotamento):
//!   [  0..16)   d    (f16 x 8, na ordem de lanes PERM)
//!   [ 16..32)   dmin (f16 x 8, na ordem de lanes PERM)
//!   [ 32..96)   escalas u8: [sub-bloco 0..8][linha 0..8]
//!   [ 96..160)  mínimos u8: [par p 0..4][lane 0..8 (PERM)][2] = min[2p+e] da linha PERM[lane]
//!   [160..1184) qs: [pedaço k 0..16][linha 0..8][8 bytes]  (pedaço k = bytes 8k..8k+8 dos qs da linha)
//!
//! Ordem de lanes `PERM`: o `hadd` dos acumuladores inteiros das duas metades (linhas 0..3 e 4..7)
//! produz as linhas em [0 1 4 5 2 3 6 7]; os vetores de d/dmin/mínimos já ficam nessa ordem e a
//! permutação só é desfeita uma vez, no fim, em cada vetor de resultado.
//!
//! **Igualdade estrita entre 1, 4 e 8 ativações (CLAUDE.md, princípio 1):** o kernel é um só código
//! genérico em `N` (`gemm::<N>`); cada saída (linha, ativação) faz exatamente as mesmas contas, na
//! mesma ordem, em qualquer `N`: a soma inteira do bloco (exata) e, por bloco, um `fmadd` com
//! `dw·dx` e um `fnmadd` com `dmin·dx`. O `N` só decide quantas ativações compartilham os pesos já
//! desempacotados. A ordem de soma em float é DIFERENTE da do kernel antigo (lanes = linhas, sem
//! soma horizontal no fim), então um tensor ou usa este layout em todos os caminhos (geração e
//! prefill) ou nenhum: ver `Tensor` em `model.rs`.
//!
//! O Q6_K tem o seu próprio layout e kernel (`gemm6`, decisão D52, descrito mais abaixo), no mesmo
//! estilo e com a mesma regra de igualdade estrita; `Repacked::gemm` escolhe pelo tipo do tensor.
//!
//! Só existe em x86-64 com AVX2+FMA+F16C: `Repacked::new` devolve `None` nas outras CPUs (e em ARM),
//! que seguem no layout original. O próprio valor `Repacked` é a prova de que a CPU tem os recursos.

use super::kernels::Act;
use rayon::prelude::*;

/// bytes de um bloco de 8 linhas reempacotado
pub const BB: usize = 1184;
const OFF_SC: usize = 32;
const OFF_MN: usize = 96;
const OFF_QS: usize = 160;
const PERM: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];
const QK: usize = 256;
const ROW_BLOCK: usize = 144;

#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Chunk([u8; 32]);

/// Matriz Q4_K (ou Q6_K, `q6`) inteira reempacotada: `nrows / 8` grupos, cada um com `nb` blocos de
/// `BB` (Q4_K) ou `BB6` (Q6_K) bytes.
pub struct Repacked {
    buf: Vec<Chunk>,
    nrows: usize,
    nb: usize,
    q6: bool,
    /// Q5_K (D84): o layout do Q4_K mais o fluxo `qh` (`BB5` bytes por bloco); `q6` fica falso
    q5: bool,
    /// formatos de bloco de 32 (D81): `Some` = Q8_0/Q4_0/Q5_0 reempacotado (`nb` = blocos de 32); `q6` fica falso
    k32: Option<Kind32>,
}

// ---------------------------------------------------------------------------------------------
// Q6_K reempacotado (decisão D52): bloco de 8 linhas x 256 colunas em 1696 bytes (os 1680 do
// original, 8 x 210, mais 16 de preenchimento para que os `qs` fiquem alinhados em 32 bytes).
//   [  0..16)   d    (f16 x 8, na ordem de lanes PERM)
//   [ 16..32)   preenchimento
//   [ 32..160)  escalas i8, em PARES: [m 0..8][lane 0..8 (PERM)][2] = (sc[2m], sc[2m+1]) da linha PERM[lane]
//               (o grupo de 16 elementos g = 2m + e usa a escala sc[g] do bloco original)
//   [160..1696) qs: [unidade u = 4n+c, 0..8][fluxo s 0..3][linha 0..8][8 bytes], com, para a metade n
//               (128 elementos) e o pedaço c (8 posições l = 8c..8c+8):
//                 fluxo 0 = ql[64n + l]        fluxo 1 = ql[64n + 32 + l]       fluxo 2 = qh[32n + l]
//               e as 4 "partes" p (elemento 128n + 32p + l) saem, como no original, de
//                 p0 = ql0 & 15 | (qh & 3) << 4      p1 = ql1 & 15 | (qh >> 2 & 3) << 4
//                 p2 = ql0 >> 4 | (qh >> 4 & 3) << 4 p3 = ql1 >> 4 | (qh >> 6 & 3) << 4
// O valor do peso é q - 32 (q = 0..63); o -32 sai por uma correção com as somas parciais da
// ativação (bsums, de 16 em 16): Σ sc·(q-32)·x = Σ sc·q·x - 32·Σ_g sc[g]·bsum[g].
//
// PROVA DE QUE NÃO HÁ ESTOURO (kernel `x86::gemm6`):
//  * `maddubs(w, x)` com w = q em 0..63 (u8) e x em -127..127 (a quantização limita a ±127; vale
//    mesmo com -128): cada i16 soma 2 produtos, |.| <= 2·63·128 = 16128 < 32767: sem saturação.
//  * Somam-se EXATAMENTE 2 resultados de `maddubs` em i16 (os pedaços c = 2cc e 2cc+1, que formam um
//    grupo de 16 elementos com a mesma escala): |.| <= 4·63·128 = 32256 < 32767: sem estouro. (Com 3
//    ou mais passos estouraria: por isso a soma em i16 pára no grupo da escala.)
//  * `madd_epi16(t, escala)` soma 2 produtos i16·i16 em i32: |.| <= 2·32256·128 = 8,3e6.
//  * Total por linha e bloco (i32): |Σ sc·q·x| <= 256·63·128·128 = 2,6e8 e |32·Σ sc·bsum| <=
//    32·16·128·2032 = 1,3e8 (bsum = soma de 16 valores, |.| <= 2032): ambos < 2^31 = 2,1e9.
// A soma inteira é exata; o float por bloco é um só `fmadd` (dw·dx) x soma, na mesma ordem para
// qualquer N (1, 4 ou 8 ativações): igualdade estrita entre as variantes.
// ---------------------------------------------------------------------------------------------

/// bytes de um bloco de 8 linhas Q6_K reempacotado
pub const BB6: usize = 1696;
const OFF6_SC: usize = 32;
const OFF6_Q: usize = 160;
const ROW_BLOCK6: usize = 210;

// ---------------------------------------------------------------------------------------------
// Q5_K reempacotado (decisao D84): o layout do Q4_K (BB = 1184 bytes: d, dmin, escalas, minimos, qs)
// seguido do fluxo dos bits altos, 256 bytes: [passo 0..4][linha 0..8][8 bytes] = qh[8*passo .. 8*passo+8]
// da linha (qh original: 32 bytes; o byte l guarda o 5o bit do elemento l de cada um dos 8 sub-blocos,
// o sub-bloco j no bit j). Bloco de 8 linhas = 1440 bytes (45 x 32; o original ocupa 8 x 176 = 1408).
// O passo s do grupo g (bytes de qs 32g + 8s .. +8) usa os MESMOS 8 bytes de qh para os dois nibbles:
// baixo (sub-bloco 2g) = bit 2g, alto (sub-bloco 2g+1) = bit 2g+1. O peso e q5 = nibble | bit << 4 (0..31).
//
// PROVA DE QUE NAO HA ESTOURO (kernel `x86::gemm::<N, true>`), igual a do Q4_K com o limite dos pesos em 31:
//  * `maddubs(w, x)`, w = q5 em 0..31 (u8), x em -127..127 (a quantizacao nunca emite -128; vale mesmo com
//    -128): cada i16 soma 2 produtos, |.| <= 2·31·128 = 7936 < 32767.
//  * Somam-se, em i16, os 4 passos de UM sub-bloco de 32 elementos (cada lane i16 recebe 2 produtos por
//    passo, logo 8 por sub-bloco): |.| <= 8·31·128 = 31744 < 32767 (com |x| <= 127: 31496). A soma em i16 PARA
//    no sub-bloco (`tl` e `th` sao zerados a cada grupo g); somar dois sub-blocos (16 produtos, 63488)
//    estouraria. Margem pequena (1023): por isso o teste de extremos.
//  * `madd_epi16(t, escala)` (escala u8 0..63 como i16) soma 2 produtos: |.| <= 2·31744·63 = 4,0e6.
//  * Por linha e bloco (i32): |Σ sc·q·x| <= 256·63·31·128 = 6,4e7 e o termo dos minimos |Σ mn·bsum32| <=
//    256·63·128 = 2,1e6 (a mesma conta do Q4_K): ambos < 2^31.
// A soma inteira e exata; o float por bloco e um `fmadd` (dw·dx) x soma e um `fnmadd` (dmin·dx) x minimos,
// na mesma ordem para qualquer N (1, 4 ou 8): igualdade estrita entre as variantes, como no Q4_K.
// ---------------------------------------------------------------------------------------------

/// bytes de um bloco de 8 linhas Q5_K reempacotado
pub const BB5: usize = 1440;
const OFF5_QH: usize = 1184;
const ROW_BLOCK5: usize = 176;

/// O reempacotamento do Q5_K esta ligado? `BURITI_NO_REPACK_Q5K=1` o desliga (chave de DIAGNOSTICO para comparar
/// ligado/desligado no mesmo binario; os Q5_K voltam ao layout e aos kernels originais).
pub fn q5k_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let set = |n: &str| std::env::var_os(n).is_some_and(|v| v != "0" && !v.is_empty());
        !set("BURITI_NO_REPACK_Q5K")
    })
}

/// O reempacotamento do Q6_K está ligado? (`BURITI_NO_REPACK_Q6=1` o desliga e deixa só o Q4_K
/// reempacotado: chave de DIAGNÓSTICO para medir o ganho do Q6_K; não muda o resultado do Q4_K.)
pub fn q6_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("BURITI_NO_REPACK_Q6").is_none())
}

// ---------------------------------------------------------------------------------------------
// Formatos de bloco de 32 reempacotados em 8 linhas (decisão D81): Q8_0, Q4_0 e Q5_0.
//
// Layout de um grupo de 8 linhas (`nb` blocos de 32 colunas), SEM preenchimento por bloco:
//   [0 .. dsec)             escalas: [bloco i][linha 0..8][f16], `dsec` = nb·16 arredondado para cima a múltiplo de 32
//   [dsec + i·QB ..)        pesos do bloco i, QB = 256 (Q8_0) | 128 (Q4_0) | 160 (Q5_0), alinhados em 32 bytes:
//     Q8_0: [pedaço j 0..8][linha r 0..8][4 bytes]   = bytes 4j..4j+4 de qs da linha r (i8)
//     Q4_0: [pedaço c 0..4][linha r 0..8][4 bytes]   = bytes 4c..4c+4 de qs da linha r (os nibbles baixos são os
//           elementos 4c..4c+4 = pedaço c, os altos os elementos 16+4c.. = pedaço c+4)
//     Q5_0: como o Q4_0 (128 bytes) e depois [linha r 0..8][u32]: o `qh` da linha r reorganizado para que o bit
//           (8·b + j) seja o bit original (4·j + b) (elemento 4j+b do pedaço j, byte b do pedaço): assim o bit 4 de
//           cada peso do pedaço j sai com UM deslocamento (4-j) e uma máscara 0x10.
// Em todos os casos o vetor de 32 bytes tem a linha r na faixa de 32 bits r: as 8 saídas já ficam na ordem das
// linhas (sem permutação), e o tamanho é o do original (8·34, 8·18, 8·22 bytes por bloco).
//
// SOMA INTEIRA (exata, igual à dos kernels antigos). `x` = ativação int8 com |x| <= 127 (o quantizador nunca
// produz -128: arredonda v·127/amax com |v| <= amax; no modo ggml há clamp em ±127).
//  * Q8_0 (pesos com sinal -128..127): |w| (u8, 0..128) x (x com o sinal de w) por `maddubs`: cada i16 soma
//    2 produtos, |.| <= 2·128·127 = 32512 < 32767 (sem saturação). Cada `maddubs` vai para i32 (`madd` com 1)
//    antes de somar com o seguinte (dois em i16 estourariam: 65024). Total do bloco, 32 produtos:
//    |.| <= 32·128·127 = 520192 < 2^31.
//  * Q4_0 / Q5_0: o peso sem sinal q (0..15 / 0..31) vai direto no `maddubs` (primeiro operando, u8) e o
//    deslocamento sai por uma correção: Σ (q - off)·x = Σ q·x - off·Σx, com off = 8 / 16 e Σx a soma dos 32 valores
//    do bloco da ativação (bsums[2i] + bsums[2i+1]). `maddubs`: |.| <= 2·15·127 = 3810 (Q4_0) e 2·31·127 = 7874
//    (Q5_0). Somam-se em i16 sem passar por i32: no Q4_0 OITO `maddubs` (<= 8·3810 = 30480 < 32767) e no Q5_0
//    QUATRO (<= 4·7874 = 31496 < 32767), depois `madd` com 1 para i32. (Mesmo com x = -128 caberia: 30720 e 31744.)
//    Total: |Σ q·x| <= 32·31·127 = 125984 e |off·Σx| <= 16·32·127: longe de 2^31.
// O float por bloco é UM `fmadd` ((d_peso·d_ativação) · soma) por saída, na mesma ordem para qualquer N
// (1, 4 ou 8 ativações): igualdade estrita entre as variantes. A ordem difere da dos kernels antigos (dois
// acumuladores e soma horizontal), então o tensor usa este layout em todos os caminhos ou em nenhum.
// ---------------------------------------------------------------------------------------------

/// Qual formato de bloco de 32 está reempacotado.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind32 {
    Q8_0,
    Q4_0,
    Q5_0,
}

impl Kind32 {
    /// bytes de um bloco do formato original
    fn src_block(self) -> usize {
        match self {
            Kind32::Q8_0 => 34,
            Kind32::Q4_0 => 18,
            Kind32::Q5_0 => 22,
        }
    }
    /// bytes de pesos de um bloco de 8 linhas reempacotado
    fn qbytes(self) -> usize {
        match self {
            Kind32::Q8_0 => 256,
            Kind32::Q4_0 => 128,
            Kind32::Q5_0 => 160,
        }
    }
    fn dsec(nb: usize) -> usize {
        (nb * 16).next_multiple_of(32)
    }
    /// bytes de um grupo de 8 linhas com `nb` blocos
    fn group_bytes(self, nb: usize) -> usize {
        Self::dsec(nb) + nb * self.qbytes()
    }
}

/// O reempacotamento deste formato de bloco de 32 está ligado? `BURITI_NO_REPACK_Q8_0` e `BURITI_NO_REPACK_Q5_0`
/// desligam um formato (chaves de DIAGNÓSTICO para comparar ligado/desligado no mesmo binário). O Q4_0 fica DESLIGADO
/// por padrão (D81: nenhum modelo local tem tensor Q4_0, então o ganho e o prefill em lote não foram medidos num modelo
/// real; só o kernel foi provado com matrizes sintéticas): `BURITI_REPACK_Q4_0=1` o liga (e `BURITI_NO_REPACK_Q4_0` o desliga).
pub fn enabled32(k: Kind32) -> bool {
    static V: std::sync::OnceLock<[bool; 3]> = std::sync::OnceLock::new();
    let v = V.get_or_init(|| {
        let set = |n: &str| std::env::var_os(n).is_some_and(|v| v != "0" && !v.is_empty());
        [!set("BURITI_NO_REPACK_Q8_0"), !set("BURITI_NO_REPACK_Q5_0"), set("BURITI_REPACK_Q4_0") && !set("BURITI_NO_REPACK_Q4_0")]
    });
    match k {
        Kind32::Q8_0 => v[0],
        Kind32::Q5_0 => v[1],
        Kind32::Q4_0 => v[2],
    }
}

/// Reempacota 8 linhas de blocos de 32 (`nb` blocos cada) no layout descrito acima (`out`: um grupo).
fn repack_group32(kind: Kind32, rows: [&[u8]; 8], nb: usize, out: &mut [u8]) {
    let dsec = Kind32::dsec(nb);
    let (sb, qb) = (kind.src_block(), kind.qbytes());
    debug_assert_eq!(out.len(), kind.group_bytes(nb));
    for i in 0..nb {
        let blks: [&[u8]; 8] = std::array::from_fn(|r| &rows[r][i * sb..(i + 1) * sb]);
        for r in 0..8 {
            out[i * 16 + r * 2..i * 16 + r * 2 + 2].copy_from_slice(&blks[r][0..2]);
        }
        let o = &mut out[dsec + i * qb..dsec + (i + 1) * qb];
        match kind {
            Kind32::Q8_0 => {
                for j in 0..8 {
                    for r in 0..8 {
                        o[j * 32 + r * 4..j * 32 + r * 4 + 4].copy_from_slice(&blks[r][2 + 4 * j..2 + 4 * j + 4]);
                    }
                }
            }
            Kind32::Q4_0 | Kind32::Q5_0 => {
                let qs0 = if kind == Kind32::Q4_0 { 2 } else { 6 };
                for c in 0..4 {
                    for r in 0..8 {
                        o[c * 32 + r * 4..c * 32 + r * 4 + 4].copy_from_slice(&blks[r][qs0 + 4 * c..qs0 + 4 * c + 4]);
                    }
                }
                if kind == Kind32::Q5_0 {
                    for r in 0..8 {
                        let qh = u32::from_le_bytes([blks[r][2], blks[r][3], blks[r][4], blks[r][5]]);
                        let mut h = 0u32;
                        for j in 0..8 {
                            for b in 0..4 {
                                h |= ((qh >> (4 * j + b)) & 1) << (8 * b + j);
                            }
                        }
                        o[128 + r * 4..128 + r * 4 + 4].copy_from_slice(&h.to_le_bytes());
                    }
                }
            }
        }
    }
}

fn cpu_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// O reempacotamento está ligado neste processo? Decidido uma vez: CPU com AVX2/FMA/F16C e sem as
/// variáveis de diagnóstico `BURITI_NO_REPACK` (volta ao layout original, para comparar) e
/// `BURITI_FLOAT_ACT` (ativações em float: o kernel reempacotado só existe com ativação int8).
pub fn enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| cpu_ok() && std::env::var_os("BURITI_NO_REPACK").is_none() && std::env::var_os("BURITI_FLOAT_ACT").is_none())
}

fn scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Reempacota 8 linhas Q4_K (cada uma com `nb` blocos de 144 bytes) em `nb` blocos de 1184 bytes (`out`).
fn repack_group(rows: [&[u8]; 8], nb: usize, out: &mut [u8]) {
    repack_group_k::<false>(rows, nb, out)
}

/// Q4_K (`Q5 = false`: blocos de 144 bytes -> 1184) ou Q5_K (`Q5 = true`: 176 -> 1440, com o fluxo `qh` no fim).
/// Q5_K original: d, dmin (4), escalas/minimos (12), qh (32), qs (128).
fn repack_group_k<const Q5: bool>(rows: [&[u8]; 8], nb: usize, out: &mut [u8]) {
    let (sb, bb, qs0) = if Q5 { (ROW_BLOCK5, BB5, 48) } else { (ROW_BLOCK, BB, 16) };
    debug_assert_eq!(out.len(), nb * bb);
    for i in 0..nb {
        let o = &mut out[i * bb..(i + 1) * bb];
        let blks: [&[u8]; 8] = std::array::from_fn(|r| &rows[r][i * sb..(i + 1) * sb]);
        if Q5 {
            for s in 0..4 {
                for r in 0..8 {
                    o[OFF5_QH + s * 64 + r * 8..OFF5_QH + s * 64 + r * 8 + 8].copy_from_slice(&blks[r][16 + s * 8..16 + s * 8 + 8]);
                }
            }
        }
        for l in 0..8 {
            let r = PERM[l];
            o[l * 2..l * 2 + 2].copy_from_slice(&blks[r][0..2]);
            o[16 + l * 2..16 + l * 2 + 2].copy_from_slice(&blks[r][2..4]);
        }
        for r in 0..8 {
            for j in 0..8 {
                o[OFF_SC + j * 8 + r] = scale_min(j, &blks[r][4..16]).0;
            }
        }
        for p in 0..4 {
            for l in 0..8 {
                for e in 0..2 {
                    o[OFF_MN + p * 16 + l * 2 + e] = scale_min(2 * p + e, &blks[PERM[l]][4..16]).1;
                }
            }
        }
        for k in 0..16 {
            for r in 0..8 {
                o[OFF_QS + k * 64 + r * 8..OFF_QS + k * 64 + r * 8 + 8].copy_from_slice(&blks[r][qs0 + k * 8..qs0 + k * 8 + 8]);
            }
        }
    }
}

/// Reempacota 8 linhas Q6_K (cada uma com `nb` blocos de 210 bytes) em `nb` blocos de 1696 bytes (`out`).
fn repack_group6(rows: [&[u8]; 8], nb: usize, out: &mut [u8]) {
    debug_assert_eq!(out.len(), nb * BB6);
    for i in 0..nb {
        let o = &mut out[i * BB6..(i + 1) * BB6];
        let blks: [&[u8]; 8] = std::array::from_fn(|r| &rows[r][i * ROW_BLOCK6..(i + 1) * ROW_BLOCK6]);
        for l in 0..8 {
            let r = PERM[l];
            o[l * 2..l * 2 + 2].copy_from_slice(&blks[r][208..210]);
        }
        for m in 0..8 {
            for l in 0..8 {
                for e in 0..2 {
                    o[OFF6_SC + m * 16 + l * 2 + e] = blks[PERM[l]][192 + 2 * m + e];
                }
            }
        }
        for n in 0..2 {
            for c in 0..4 {
                let u = n * 4 + c;
                for r in 0..8 {
                    let src = [n * 64 + c * 8, n * 64 + 32 + c * 8, 128 + n * 32 + c * 8];
                    for (s, &from) in src.iter().enumerate() {
                        let to = OFF6_Q + u * 192 + s * 64 + r * 8;
                        o[to..to + 8].copy_from_slice(&blks[r][from..from + 8]);
                    }
                }
            }
        }
    }
}

impl Repacked {
    /// A forma é elegível? Linhas múltiplas de 8 e colunas múltiplas de 256 (bloco K).
    pub fn eligible(rows: usize, cols: usize) -> bool {
        rows > 0 && rows % 8 == 0 && cols > 0 && cols % QK == 0
    }

    /// Reempacota `rows` linhas Q4_K de `cols` colunas (`data` = as linhas originais, contíguas), em
    /// paralelo por grupo de 8 linhas (rodar no pool do rayon). `None` se a CPU não tem
    /// AVX2/FMA/F16C ou a forma não é elegível. Não consulta as variáveis de ambiente (isso é `enabled`).
    pub fn new(data: &[u8], rows: usize, cols: usize) -> Option<Repacked> {
        if !cpu_ok() || !Self::eligible(rows, cols) {
            return None;
        }
        let nb = cols / QK;
        let row_bytes = nb * ROW_BLOCK;
        assert!(data.len() >= rows * row_bytes);
        let gb = nb * BB; // bytes de um grupo; múltiplo de 32 (1184 = 37 · 32)
        let mut buf = vec![Chunk([0; 32]); rows / 8 * gb / 32];
        // SAFETY: `Chunk` é `repr(C)` de 32 bytes, sem preenchimento; a vista em bytes cobre exatamente o vetor.
        let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 32) };
        bytes.par_chunks_mut(gb).enumerate().for_each(|(g, out)| {
            let rws: [&[u8]; 8] = std::array::from_fn(|r| &data[(g * 8 + r) * row_bytes..(g * 8 + r + 1) * row_bytes]);
            repack_group(rws, nb, out);
        });
        Some(Repacked { buf, nrows: rows, nb, q6: false, q5: false, k32: None })
    }

    /// Como `new`, para linhas Q5_K (176 bytes por bloco de 256). Mesma elegibilidade (D84).
    pub fn new_q5(data: &[u8], rows: usize, cols: usize) -> Option<Repacked> {
        if !cpu_ok() || !Self::eligible(rows, cols) {
            return None;
        }
        let nb = cols / QK;
        let row_bytes = nb * ROW_BLOCK5;
        assert!(data.len() >= rows * row_bytes);
        let gb = nb * BB5; // multiplo de 32 (1440 = 45 · 32)
        let mut buf = vec![Chunk([0; 32]); rows / 8 * gb / 32];
        // SAFETY: igual a `new`.
        let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 32) };
        bytes.par_chunks_mut(gb).enumerate().for_each(|(g, out)| {
            let rws: [&[u8]; 8] = std::array::from_fn(|r| &data[(g * 8 + r) * row_bytes..(g * 8 + r + 1) * row_bytes]);
            repack_group_k::<true>(rws, nb, out);
        });
        Some(Repacked { buf, nrows: rows, nb, q6: false, q5: true, k32: None })
    }

    /// Como `new`, para linhas Q6_K (210 bytes por bloco de 256). Mesma elegibilidade.
    pub fn new_q6(data: &[u8], rows: usize, cols: usize) -> Option<Repacked> {
        if !cpu_ok() || !Self::eligible(rows, cols) {
            return None;
        }
        let nb = cols / QK;
        let row_bytes = nb * ROW_BLOCK6;
        assert!(data.len() >= rows * row_bytes);
        let gb = nb * BB6; // múltiplo de 32 (1696 = 53 · 32)
        let mut buf = vec![Chunk([0; 32]); rows / 8 * gb / 32];
        // SAFETY: igual a `new`.
        let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 32) };
        bytes.par_chunks_mut(gb).enumerate().for_each(|(g, out)| {
            let rws: [&[u8]; 8] = std::array::from_fn(|r| &data[(g * 8 + r) * row_bytes..(g * 8 + r + 1) * row_bytes]);
            repack_group6(rws, nb, out);
        });
        Some(Repacked { buf, nrows: rows, nb, q6: true, q5: false, k32: None })
    }

    /// A forma é elegível para os formatos de bloco de 32? Linhas múltiplas de 8 e colunas múltiplas de 32.
    pub fn eligible32(rows: usize, cols: usize) -> bool {
        rows > 0 && rows % 8 == 0 && cols > 0 && cols % 32 == 0
    }

    /// Como `new`, para linhas Q8_0, Q4_0 ou Q5_0 (`data` = as linhas originais, contíguas). `None` se a CPU não tem
    /// AVX2/FMA/F16C ou a forma não é elegível. Não consulta as variáveis de ambiente (isso é `enabled32`).
    pub fn new32(kind: Kind32, data: &[u8], rows: usize, cols: usize) -> Option<Repacked> {
        if !cpu_ok() || !Self::eligible32(rows, cols) {
            return None;
        }
        let nb = cols / 32;
        let row_bytes = nb * kind.src_block();
        assert!(data.len() >= rows * row_bytes);
        let gb = kind.group_bytes(nb); // múltiplo de 32 por construção
        let mut buf = vec![Chunk([0; 32]); rows / 8 * gb / 32];
        // SAFETY: igual a `new`.
        let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 32) };
        bytes.par_chunks_mut(gb).enumerate().for_each(|(g, out)| {
            let rws: [&[u8]; 8] = std::array::from_fn(|r| &data[(g * 8 + r) * row_bytes..(g * 8 + r + 1) * row_bytes]);
            repack_group32(kind, rws, nb, out);
        });
        Some(Repacked { buf, nrows: rows, nb, q6: false, q5: false, k32: Some(kind) })
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    /// Bytes ocupados (para o relato de memória).
    pub fn size_bytes(&self) -> usize {
        self.buf.len() * 32
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: vista em bytes do próprio vetor (ver `new`).
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.buf.len() * 32) }
    }

    /// Grupo `g` (linhas 8g..8g+8) x `N` ativações; `out[k][r]` = produto da linha 8g+r pela ativação k.
    /// Cada ativação dá o mesmo resultado, bit a bit, qualquer que seja `N` e a posição dela no grupo.
    #[inline]
    pub fn gemm<const N: usize>(&self, g: usize, a: [&Act; N], out: &mut [[f32; 8]; N]) {
        if let Some(k) = self.k32 {
            return self.gemm32::<N>(k, g, a, out);
        }
        let gb = self.nb * if self.q6 { BB6 } else if self.q5 { BB5 } else { BB };
        let w = &self.bytes()[g * gb..(g + 1) * gb]; // confere o limite do grupo
        for x in &a {
            assert!(x.q.len() >= self.nb * QK && x.d.len() >= self.nb, "ativação curta demais");
            if self.q6 {
                assert!(x.bsums.len() >= self.nb * 16, "ativação sem as somas de 16 (bsums)");
            } else {
                assert!(x.bp.len() >= self.nb * 4, "ativação sem as somas pareadas (bp)");
            }
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: um `Repacked` só existe se a CPU tem AVX2/FMA/F16C (`new`/`new_q6`); `w` tem `nb` blocos e as
        // ativações têm, no mínimo, `nb` blocos de q, d e bp (Q4_K) ou bsums (Q6_K) (conferido acima).
        unsafe {
            if self.q6 {
                x86::gemm6::<N>(w.as_ptr(), self.nb, &a, out)
            } else if self.q5 {
                x86::gemm::<N, true>(w.as_ptr(), self.nb, &a, out)
            } else {
                x86::gemm::<N, false>(w.as_ptr(), self.nb, &a, out)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (w, out);
            unreachable!("Repacked não existe fora do x86-64")
        }
    }
}

impl Repacked {
    /// Q8_0/Q4_0/Q5_0 reempacotados (ver o layout e as provas acima): mesmo contrato de `gemm`.
    #[inline]
    fn gemm32<const N: usize>(&self, kind: Kind32, g: usize, a: [&Act; N], out: &mut [[f32; 8]; N]) {
        let gb = kind.group_bytes(self.nb);
        let w = &self.bytes()[g * gb..(g + 1) * gb]; // confere o limite do grupo
        for x in &a {
            assert!(x.q.len() >= self.nb * 32 && x.d.len() >= self.nb, "ativação curta demais");
            if kind != Kind32::Q8_0 {
                assert!(x.bsums.len() >= self.nb * 2, "ativação sem as somas de 16 (bsums)");
            }
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: um `Repacked` só existe se a CPU tem AVX2/FMA/F16C (`new32`); `w` é um grupo inteiro de `nb` blocos e as
        // ativações têm, no mínimo, `nb` blocos de q e d (e `2·nb` bsums nos formatos sem sinal) (conferido acima).
        unsafe {
            match kind {
                Kind32::Q8_0 => x86::gemm32::<N, 0>(w.as_ptr(), self.nb, &a, out),
                Kind32::Q4_0 => x86::gemm32::<N, 1>(w.as_ptr(), self.nb, &a, out),
                Kind32::Q5_0 => x86::gemm32::<N, 2>(w.as_ptr(), self.nb, &a, out),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (w, out);
            unreachable!("Repacked não existe fora do x86-64")
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Act, Kind32, BB, BB5, BB6, OFF5_QH, OFF6_Q, OFF6_SC, OFF_MN, OFF_QS, OFF_SC};
    use std::arch::x86_64::*;

    /// Bit 4 (0x10) de cada peso do pedaço `j` do Q5_0 reempacotado: `h` tem, por linha, o bit (8b + j) = bit do
    /// peso b do pedaço j; um deslocamento (4 - j) o leva à posição 4 do byte b. Sem contaminação entre bytes:
    /// à esquerda (j <= 4) o deslocamento é <= 4 e o bit j' do byte só chega à posição 4 do MESMO byte se j' = j
    /// (j' + s = 4 + 8m com m >= 1 exigiria j' + s >= 12 > 7 + 4); à direita (s = j - 4 <= 3) idem (j' - s = 4 - 8m
    /// com m >= 1 exigiria j' - s <= -4 < -3).
    #[inline(always)]
    unsafe fn hibit(h: __m256i, j: usize) -> __m256i {
        let v = if j <= 4 { _mm256_sll_epi32(h, _mm_cvtsi32_si128(4 - j as i32)) } else { _mm256_srl_epi32(h, _mm_cvtsi32_si128(j as i32 - 4)) };
        _mm256_and_si256(v, _mm256_set1_epi8(0x10))
    }

    /// Q8_0 (K = 0), Q4_0 (K = 1) e Q5_0 (K = 2) reempacotados, `N` ativações por passada (provas no começo do
    /// arquivo). Cada saída (linha, ativação) faz as mesmas contas na mesma ordem em qualquer `N`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemm32<const N: usize, const K: usize>(w: *const u8, nb: usize, a: &[&Act; N], out: &mut [[f32; 8]; N]) {
        let dsec = Kind32::dsec(nb);
        let qbytes = [256usize, 128, 160][K];
        let ones = _mm256_set1_epi16(1);
        let m4 = _mm256_set1_epi8(0x0F);
        let qp: [*const i8; N] = std::array::from_fn(|k| a[k].q.as_ptr());
        let dp: [*const f32; N] = std::array::from_fn(|k| a[k].d.as_ptr());
        let bsp: [*const i16; N] = std::array::from_fn(|k| a[k].bsums.as_ptr());
        let mut acc = [_mm256_setzero_ps(); N];
        for i in 0..nb {
            let dv = _mm256_cvtph_ps(_mm_loadu_si128(w.add(i * 16) as *const __m128i));
            let qb = w.add(dsec + i * qbytes);
            let mut s = [_mm256_setzero_si256(); N];
            if K == 0 {
                for j in 0..8 {
                    let wv = _mm256_loadu_si256(qb.add(32 * j) as *const __m256i);
                    let aw = _mm256_sign_epi8(wv, wv);
                    for k in 0..N {
                        let x = _mm256_set1_epi32((qp[k].add(i * 32 + 4 * j) as *const i32).read_unaligned());
                        let sx = _mm256_sign_epi8(x, wv);
                        s[k] = _mm256_add_epi32(s[k], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones));
                    }
                }
            } else {
                let hq = if K == 2 { _mm256_loadu_si256(qb.add(128) as *const __m256i) } else { _mm256_setzero_si256() };
                let mut t = [_mm256_setzero_si256(); N];
                for c in 0..4 {
                    let v = _mm256_loadu_si256(qb.add(32 * c) as *const __m256i);
                    let mut wlo = _mm256_and_si256(v, m4); // pedaço c
                    let mut whi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4); // pedaço c + 4
                    if K == 2 {
                        wlo = _mm256_or_si256(wlo, hibit(hq, c));
                        whi = _mm256_or_si256(whi, hibit(hq, c + 4));
                    }
                    // Q4_0: um único `madd` no fim (8 maddubs em i16); Q5_0: depois de 4 maddubs (c = 1 e c = 3)
                    let flush = if K == 1 { c == 3 } else { c & 1 == 1 };
                    for k in 0..N {
                        let xl = _mm256_set1_epi32((qp[k].add(i * 32 + 4 * c) as *const i32).read_unaligned());
                        let xh = _mm256_set1_epi32((qp[k].add(i * 32 + 4 * (c + 4)) as *const i32).read_unaligned());
                        t[k] = _mm256_add_epi16(t[k], _mm256_maddubs_epi16(wlo, xl));
                        t[k] = _mm256_add_epi16(t[k], _mm256_maddubs_epi16(whi, xh));
                        if flush {
                            s[k] = _mm256_add_epi32(s[k], _mm256_madd_epi16(t[k], ones));
                            t[k] = _mm256_setzero_si256();
                        }
                    }
                }
                let off = if K == 1 { 8 } else { 16 };
                for k in 0..N {
                    let sx = *bsp[k].add(2 * i) as i32 + *bsp[k].add(2 * i + 1) as i32;
                    s[k] = _mm256_sub_epi32(s[k], _mm256_set1_epi32(off * sx));
                }
            }
            for k in 0..N {
                let dx = _mm256_set1_ps(*dp[k].add(i));
                acc[k] = _mm256_fmadd_ps(_mm256_mul_ps(dv, dx), _mm256_cvtepi32_ps(s[k]), acc[k]);
            }
        }
        for k in 0..N {
            _mm256_storeu_ps(out[k].as_mut_ptr(), acc[k]);
        }
    }

    /// Leva o bit `b` de cada byte de `qh` para a posicao 4 (0x10). Mesma analise de contaminacao do
    /// `hibit`: com deslocamentos de 16 bits o que vaza para o byte vizinho cai fora da mascara 0x10.
    #[inline(always)]
    unsafe fn qbit4(qh: __m256i, b: usize) -> __m256i {
        let v = if b <= 4 { _mm256_sll_epi16(qh, _mm_cvtsi32_si128(4 - b as i32)) } else { _mm256_srl_epi16(qh, _mm_cvtsi32_si128(b as i32 - 4)) };
        _mm256_and_si256(v, _mm256_set1_epi8(0x10))
    }

    /// Q4_K (`Q5 = false`) e Q5_K (`Q5 = true`, D84) reempacotados; so o Q5_K le o fluxo `qh` e o junta ao nibble.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemm<const N: usize, const Q5: bool>(w: *const u8, nb: usize, a: &[&Act; N], out: &mut [[f32; 8]; N]) {
        let m4 = _mm256_set1_epi8(0x0F);
        // máscaras de pshufb: para a metade h (linhas 4h..4h+3), cada escala repetida 4x como i16
        let mut masks = [_mm256_setzero_si256(); 2];
        for h in 0..2 {
            let mut m = [0x80u8; 32];
            for lane in 0..2 {
                for j in 0..8 {
                    let row = 4 * h + 2 * lane + if j < 4 { 0 } else { 1 };
                    m[lane * 16 + 2 * j] = row as u8;
                }
            }
            masks[h] = _mm256_loadu_si256(m.as_ptr() as *const __m256i);
        }
        let qp: [*const i8; N] = std::array::from_fn(|k| a[k].q.as_ptr());
        let dp: [*const f32; N] = std::array::from_fn(|k| a[k].d.as_ptr());
        let bpp: [*const i32; N] = std::array::from_fn(|k| a[k].bp.as_ptr());
        let mut acc = [_mm256_setzero_ps(); N];
        for i in 0..nb {
            let blk = w.add(i * if Q5 { BB5 } else { BB });
            let dv = _mm256_cvtph_ps(_mm_loadu_si128(blk as *const __m128i));
            let dminv = _mm256_cvtph_ps(_mm_loadu_si128(blk.add(16) as *const __m128i));
            let mut mnv = [_mm256_setzero_si256(); 4];
            for p in 0..4 {
                mnv[p] = _mm256_cvtepu8_epi16(_mm_loadu_si128(blk.add(OFF_MN + 16 * p) as *const __m128i));
            }
            let mut isum = [_mm256_setzero_si256(); N];
            for g in 0..4 {
                let mut wlo = [[_mm256_setzero_si256(); 2]; 4];
                let mut whi = [[_mm256_setzero_si256(); 2]; 4];
                for step in 0..4 {
                    for h in 0..2 {
                        let q = _mm256_loadu_si256(blk.add(OFF_QS + (g * 4 + step) * 64 + h * 32) as *const __m256i);
                        wlo[step][h] = _mm256_and_si256(q, m4);
                        whi[step][h] = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                        if Q5 {
                            // 5o bit: bit 2g (sub-bloco 2g, nibble baixo) e bit 2g+1 (nibble alto) dos mesmos 8 bytes
                            let qh = _mm256_loadu_si256(blk.add(OFF5_QH + step * 64 + h * 32) as *const __m256i);
                            wlo[step][h] = _mm256_or_si256(wlo[step][h], qbit4(qh, 2 * g));
                            whi[step][h] = _mm256_or_si256(whi[step][h], qbit4(qh, 2 * g + 1));
                        }
                    }
                }
                let mut scv = [[_mm256_setzero_si256(); 2]; 2];
                for t in 0..2 {
                    let v = _mm256_broadcastsi128_si256(_mm_loadl_epi64(blk.add(OFF_SC + 8 * (2 * g + t)) as *const __m128i));
                    for h in 0..2 {
                        scv[t][h] = _mm256_shuffle_epi8(v, masks[h]);
                    }
                }
                for k in 0..N {
                    let xp = qp[k].add(i * 256 + g * 64);
                    let mut tl = [_mm256_setzero_si256(); 2];
                    let mut th = [_mm256_setzero_si256(); 2];
                    for step in 0..4 {
                        let xl = _mm256_set1_epi64x((xp.add(step * 8) as *const i64).read_unaligned());
                        let xh = _mm256_set1_epi64x((xp.add(32 + step * 8) as *const i64).read_unaligned());
                        for h in 0..2 {
                            tl[h] = _mm256_add_epi16(tl[h], _mm256_maddubs_epi16(wlo[step][h], xl));
                            th[h] = _mm256_add_epi16(th[h], _mm256_maddubs_epi16(whi[step][h], xh));
                        }
                    }
                    let s0 = _mm256_add_epi32(_mm256_madd_epi16(tl[0], scv[0][0]), _mm256_madd_epi16(th[0], scv[1][0]));
                    let s1 = _mm256_add_epi32(_mm256_madd_epi16(tl[1], scv[0][1]), _mm256_madd_epi16(th[1], scv[1][1]));
                    isum[k] = _mm256_add_epi32(isum[k], _mm256_hadd_epi32(s0, s1));
                }
            }
            for k in 0..N {
                let dx = _mm256_set1_ps(*dp[k].add(i));
                acc[k] = _mm256_fmadd_ps(_mm256_mul_ps(dv, dx), _mm256_cvtepi32_ps(isum[k]), acc[k]);
                let bp = bpp[k].add(i * 4);
                let mut m = _mm256_madd_epi16(mnv[0], _mm256_set1_epi32(*bp));
                m = _mm256_add_epi32(m, _mm256_madd_epi16(mnv[1], _mm256_set1_epi32(*bp.add(1))));
                m = _mm256_add_epi32(m, _mm256_madd_epi16(mnv[2], _mm256_set1_epi32(*bp.add(2))));
                m = _mm256_add_epi32(m, _mm256_madd_epi16(mnv[3], _mm256_set1_epi32(*bp.add(3))));
                acc[k] = _mm256_fnmadd_ps(_mm256_mul_ps(dminv, dx), _mm256_cvtepi32_ps(m), acc[k]);
            }
        }
        for k in 0..N {
            // desfaz a ordem de lanes PERM: qwords [0 2 1 3]
            let v = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(acc[k]), 0xD8));
            _mm256_storeu_ps(out[k].as_mut_ptr(), v);
        }
    }

    /// Q6_K reempacotado (ver o layout e a prova de não-estouro no começo do arquivo). `N` ativações
    /// compartilham os pesos já desempacotados; cada saída faz as mesmas contas, na mesma ordem, em
    /// qualquer `N`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn gemm6<const N: usize>(w: *const u8, nb: usize, a: &[&Act; N], out: &mut [[f32; 8]; N]) {
        let m4 = _mm256_set1_epi8(0x0F);
        let m30 = _mm256_set1_epi8(0x30);
        // pshufb: do vetor de escalas em pares (i32 = (sc[2m], sc[2m+1]) por lane PERM) tira, para a
        // metade h (linhas 4h..4h+3) e a paridade e, a escala `e` de cada linha repetida 4x como i16.
        // Em cada faixa de 128 bits as lanes 0..3 são as linhas PERM[0..4] = (0,1,4,5) na faixa baixa e
        // (2,3,6,7) na alta: h = 0 usa as lanes 0 e 1 de cada faixa, h = 1 as lanes 2 e 3.
        let mut masks = [[_mm256_setzero_si256(); 2]; 2];
        for h in 0..2 {
            for e in 0..2 {
                let mut m = [0x80u8; 32];
                for lane in 0..2 {
                    for j in 0..8 {
                        let src = 2 * h + j / 4;
                        m[lane * 16 + 2 * j] = (4 * src + 2 * e) as u8;
                        m[lane * 16 + 2 * j + 1] = (4 * src + 2 * e + 1) as u8;
                    }
                }
                masks[h][e] = _mm256_loadu_si256(m.as_ptr() as *const __m256i);
            }
        }
        let qp: [*const i8; N] = std::array::from_fn(|k| a[k].q.as_ptr());
        let dp: [*const f32; N] = std::array::from_fn(|k| a[k].d.as_ptr());
        let bsp: [*const i32; N] = std::array::from_fn(|k| a[k].bsums.as_ptr() as *const i32);
        let mut acc = [_mm256_setzero_ps(); N];
        for i in 0..nb {
            let blk = w.add(i * BB6);
            let dv = _mm256_cvtph_ps(_mm_loadu_si128(blk as *const __m128i));
            // escalas em pares, i8 -> i16: 8 vetores (um por m), cada i32 = (sc[2m], sc[2m+1]) da lane PERM
            let mut sp = [_mm256_setzero_si256(); 8];
            for m in 0..8 {
                sp[m] = _mm256_cvtepi8_epi16(_mm_loadu_si128(blk.add(OFF6_SC + 16 * m) as *const __m128i));
            }
            // somas inteiras das duas metades de linhas (h = 0, 1), por ativação
            let mut s = [[_mm256_setzero_si256(); 2]; N];
            for n in 0..2 {
                for cc in 0..2 {
                    // 2 pedaços de 8 posições (c = 2cc, 2cc+1) x 4 partes x 2 metades de linhas
                    let mut wv = [[[_mm256_setzero_si256(); 2]; 4]; 2];
                    for ci in 0..2 {
                        let u = n * 4 + cc * 2 + ci;
                        let base = blk.add(OFF6_Q + u * 192);
                        for h in 0..2 {
                            let q0 = _mm256_loadu_si256(base.add(h * 32) as *const __m256i);
                            let q1 = _mm256_loadu_si256(base.add(64 + h * 32) as *const __m256i);
                            let qh = _mm256_loadu_si256(base.add(128 + h * 32) as *const __m256i);
                            wv[ci][0][h] = _mm256_or_si256(_mm256_and_si256(q0, m4), _mm256_and_si256(_mm256_slli_epi16(qh, 4), m30));
                            wv[ci][1][h] = _mm256_or_si256(_mm256_and_si256(q1, m4), _mm256_and_si256(_mm256_slli_epi16(qh, 2), m30));
                            wv[ci][2][h] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q0, 4), m4), _mm256_and_si256(qh, m30));
                            wv[ci][3][h] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q1, 4), m4), _mm256_and_si256(_mm256_srli_epi16(qh, 2), m30));
                        }
                    }
                    // escala do grupo g = 8n + 2p + cc = 2m + e, com m = 4n + p e e = cc
                    let mut scv = [[_mm256_setzero_si256(); 2]; 4];
                    for p in 0..4 {
                        for h in 0..2 {
                            scv[p][h] = _mm256_shuffle_epi8(sp[n * 4 + p], masks[h][cc]);
                        }
                    }
                    for k in 0..N {
                        let xp = qp[k].add(i * 256 + n * 128);
                        for p in 0..4 {
                            let x0 = _mm256_set1_epi64x((xp.add(p * 32 + (2 * cc) * 8) as *const i64).read_unaligned());
                            let x1 = _mm256_set1_epi64x((xp.add(p * 32 + (2 * cc + 1) * 8) as *const i64).read_unaligned());
                            for h in 0..2 {
                                // exatamente 2 maddubs somados em i16 (ver a prova: <= 32256), depois a escala
                                let t = _mm256_add_epi16(_mm256_maddubs_epi16(wv[0][p][h], x0), _mm256_maddubs_epi16(wv[1][p][h], x1));
                                s[k][h] = _mm256_add_epi32(s[k][h], _mm256_madd_epi16(t, scv[p][h]));
                            }
                        }
                    }
                }
            }
            for k in 0..N {
                // soma das duas lanes de cada linha (ordem PERM, como no Q4_K) e correção do -32
                let mut isum = _mm256_hadd_epi32(s[k][0], s[k][1]);
                let bs = bsp[k].add(i * 8);
                let mut corr = _mm256_madd_epi16(sp[0], _mm256_set1_epi32(bs.read_unaligned()));
                for m in 1..8 {
                    corr = _mm256_add_epi32(corr, _mm256_madd_epi16(sp[m], _mm256_set1_epi32(bs.add(m).read_unaligned())));
                }
                isum = _mm256_sub_epi32(isum, _mm256_slli_epi32(corr, 5));
                let dx = _mm256_set1_ps(*dp[k].add(i));
                acc[k] = _mm256_fmadd_ps(_mm256_mul_ps(dv, dx), _mm256_cvtepi32_ps(isum), acc[k]);
            }
        }
        for k in 0..N {
            let v = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(acc[k]), 0xD8));
            _mm256_storeu_ps(out[k].as_mut_ptr(), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::gguf::{Dtype, Gguf};
    use crate::native::kernels::{dot_row_q, dot_row_q_x4, dot_row_q_x8};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            (self.next() % 20001) as f32 / 10000.0 - 1.0
        }
    }

    /// Linhas Q4_K aleatórias com d/dmin = 0x2800 (2^-5, finito).
    fn random_rows(rng: &mut Rng, nrows: usize, nb: usize) -> Vec<u8> {
        let mut data: Vec<u8> = (0..nrows * nb * ROW_BLOCK).map(|_| (rng.next() & 0xFF) as u8).collect();
        for b in 0..nrows * nb {
            for off in [0usize, 2] {
                data[b * ROW_BLOCK + off] = 0x00;
                data[b * ROW_BLOCK + off + 1] = 0x28;
            }
        }
        data
    }

    /// Ativações "diádicas": inteiros em [-127,127] x 2^-6 com um +-127 por bloco, de modo que
    /// d = 2^-6 e q = x·64 exatos; com d/dmin = 2^-5 todas as contas em float são exatas e a ordem
    /// de soma não importa: o kernel antigo e o reempacotado têm de dar o MESMO float, bit a bit.
    fn dyadic_x(rng: &mut Rng, cols: usize) -> Vec<f32> {
        let mut x: Vec<f32> = (0..cols).map(|_| ((rng.next() % 255) as i32 - 127) as f32 / 64.0).collect();
        for b in 0..cols / QK {
            x[b * QK + (rng.next() % QK as u64) as usize] = if rng.next() & 1 == 0 { 127.0 / 64.0 } else { -127.0 / 64.0 };
        }
        x
    }

    fn acts(xs: &[Vec<f32>]) -> Vec<Act> {
        xs.iter().map(|x| Act::new(Dtype::Q4K, x).unwrap()).collect()
    }

    /// Resultado de um grupo (8 linhas) para as ativações `ac`, pelas três variantes (8, 4 e 1 por
    /// passada, com o tamanho do lote dividido como o `matmat`): `[ativação][linha]`.
    fn gemm_all(rp: &Repacked, g: usize, ac: &[Act]) -> Vec<[f32; 8]> {
        let n = ac.len();
        let mut res = vec![[0f32; 8]; n];
        let mut i = 0;
        while i + 8 <= n {
            let mut o = [[0f32; 8]; 8];
            rp.gemm::<8>(g, std::array::from_fn(|k| &ac[i + k]), &mut o);
            res[i..i + 8].copy_from_slice(&o);
            i += 8;
        }
        while i + 4 <= n {
            let mut o = [[0f32; 8]; 4];
            rp.gemm::<4>(g, std::array::from_fn(|k| &ac[i + k]), &mut o);
            res[i..i + 4].copy_from_slice(&o);
            i += 4;
        }
        while i < n {
            let mut o = [[0f32; 8]; 1];
            rp.gemm::<1>(g, [&ac[i]], &mut o);
            res[i] = o[0];
            i += 1;
        }
        res
    }

    fn bits(v: &[[f32; 8]]) -> Vec<u32> {
        v.iter().flatten().map(|f| f.to_bits()).collect()
    }

    /// Troca os qs intercalados das linhas 2 e 3 do primeiro bloco (defeito plantado: repacker errado).
    fn plant_swap_rows(rp: &mut Repacked) {
        let b = unsafe { std::slice::from_raw_parts_mut(rp.buf.as_mut_ptr() as *mut u8, rp.buf.len() * 32) };
        for k in 0..16 {
            for e in 0..8 {
                b.swap(OFF_QS + k * 64 + 2 * 8 + e, OFF_QS + k * 64 + 3 * 8 + e);
            }
        }
    }

    /// Troca as escalas dos sub-blocos 0 e 1 da linha 5 do primeiro bloco.
    fn plant_swap_scales(rp: &mut Repacked) {
        let b = unsafe { std::slice::from_raw_parts_mut(rp.buf.as_mut_ptr() as *mut u8, rp.buf.len() * 32) };
        b.swap(OFF_SC + 5, OFF_SC + 8 + 5);
    }

    #[test]
    fn eligibility() {
        assert!(Repacked::eligible(16, 256));
        assert!(!Repacked::eligible(12, 256), "linhas não múltiplas de 8");
        assert!(!Repacked::eligible(16, 128), "colunas não múltiplas de 256");
        assert!(!Repacked::eligible(0, 256));
        assert!(Repacked::new(&[], 12, 256).is_none());
    }

    /// Os três kernels têm de dar o MESMO resultado que o kernel antigo em dados diádicos (float
    /// exato), e o mesmo resultado entre si em dados arbitrários (igualdade estrita).
    #[test]
    fn x1_x4_x8_equal_old_on_dyadic_and_each_other_on_random() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for trial in 0..12 {
            let nb = 1 + trial % 6;
            let data = random_rows(&mut rng, 8, nb);
            let rp = Repacked::new(&data, 8, nb * QK).unwrap();
            let n = 13; // 8 + 4 + 1: as três variantes
            let xs: Vec<Vec<f32>> = (0..n).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
            let ac = acts(&xs);
            let got = gemm_all(&rp, 0, &ac);
            let rb = nb * ROW_BLOCK;
            for r in 0..8 {
                for k in 0..n {
                    let want = dot_row_q(Dtype::Q4K, &data[r * rb..(r + 1) * rb], &ac[k]);
                    assert_eq!(got[k][r].to_bits(), want.to_bits(), "dados diádicos: trial {trial} linha {r} ativação {k}: {} contra {want}", got[k][r]);
                }
            }
            // dados arbitrários: cada ativação sozinha (x1) = em grupo de 4 = em grupo de 8, em qualquer posição
            let xs: Vec<Vec<f32>> = (0..n).map(|_| (0..nb * QK).map(|_| rng.f()).collect()).collect();
            let ac = acts(&xs);
            let mixed = gemm_all(&rp, 0, &ac); // 8 + 4 + 1
            let ones: Vec<[f32; 8]> = (0..n).map(|k| { let mut o = [[0f32; 8]; 1]; rp.gemm::<1>(0, [&ac[k]], &mut o); o[0] }).collect();
            assert_eq!(bits(&mixed), bits(&ones), "trial {trial}: x8/x4/x1 diferem (ordem de soma diferente entre as variantes)");
            let mut o4 = [[0f32; 8]; 4];
            rp.gemm::<4>(0, [&ac[3], &ac[0], &ac[7], &ac[11]], &mut o4);
            for (s, k) in [3usize, 0, 7, 11].iter().enumerate() {
                assert_eq!(bits(&o4[s..s + 1]), bits(&ones[*k..*k + 1]), "trial {trial}: x4 em outra posição difere do x1");
            }
            // o x8 antigo é só tolerância (ordem de soma diferente)
            let g8: [&Act; 8] = std::array::from_fn(|k| &ac[k]);
            let old = dot_row_q_x8(Dtype::Q4K, &data[..rb], g8);
            let _ = dot_row_q_x4(Dtype::Q4K, &data[..rb], [&ac[0], &ac[1], &ac[2], &ac[3]]);
            for k in 0..8 {
                let scale = old.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
                assert!((mixed[k][0] - old[k]).abs() / scale < 1e-5, "trial {trial}: repacked longe do kernel antigo");
            }
        }
    }

    /// Testes com dentes: com um defeito plantado no repacker, a comparação exata TEM de falhar.
    #[test]
    fn planted_bugs_are_caught() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let nb = 3;
        let data = random_rows(&mut rng, 8, nb);
        let xs: Vec<Vec<f32>> = (0..9).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
        let ac = acts(&xs);
        let rb = nb * ROW_BLOCK;
        let equal_old = |rp: &Repacked| {
            let got = gemm_all(rp, 0, &ac);
            (0..8).all(|r| (0..9).all(|k| got[k][r].to_bits() == dot_row_q(Dtype::Q4K, &data[r * rb..(r + 1) * rb], &ac[k]).to_bits()))
        };
        assert!(equal_old(&Repacked::new(&data, 8, nb * QK).unwrap()), "controle sem defeito");
        let mut a = Repacked::new(&data, 8, nb * QK).unwrap();
        plant_swap_rows(&mut a);
        assert!(!equal_old(&a), "troca de linhas NÃO foi detectada");
        let mut b = Repacked::new(&data, 8, nb * QK).unwrap();
        plant_swap_scales(&mut b);
        assert!(!equal_old(&b), "troca de escalas NÃO foi detectada");
    }

    /// Oráculo em f64 da mesma conta (pesos originais Q4_K + a ativação quantizada do `Act`).
    fn oracle(row: &[u8], a: &Act) -> f64 {
        let mut total = 0f64;
        for (i, blk) in row.chunks_exact(ROW_BLOCK).enumerate() {
            let f16 = |o: usize| crate::native::quant::f16_to_f32(u16::from_le_bytes([blk[o], blk[o + 1]])) as f64;
            let (dw, dmin) = (f16(0), f16(2));
            let dx = a.d[i] as f64;
            let (mut sumi, mut mins) = (0i64, 0i64);
            for j in 0..8 {
                let (s, m) = scale_min(j, &blk[4..16]);
                let (g, hi) = (j / 2, j % 2);
                let (mut ws, mut bs) = (0i64, 0i64);
                for b in 0..32 {
                    let byte = blk[16 + g * 32 + b];
                    let w = if hi == 0 { byte & 15 } else { byte >> 4 } as i64;
                    let x = a.q[i * QK + j * 32 + b] as i64;
                    ws += w * x;
                    bs += x;
                }
                sumi += s as i64 * ws;
                mins += m as i64 * bs;
            }
            total += dw * dx * sumi as f64 - dmin * dx * mins as f64;
        }
        total
    }

    /// Com `BURITI_ACT_GGML` as ativações têm o sinal trocado e d negativo (Q8_K do ggml): o kernel reempacotado
    /// (que consome `bp`) tem de continuar idêntico entre x1/x4/x8 e perto do oráculo em f64.
    #[test]
    fn ggml_style_acts_work_with_repacked_kernel() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x5EED_0A17);
        let nb = 3;
        let data = random_rows(&mut rng, 8, nb);
        let rp = Repacked::new(&data, 8, nb * QK).unwrap();
        let rb = nb * ROW_BLOCK;
        let n = 13;
        let ac: Vec<Act> = (0..n).map(|_| { let x: Vec<f32> = (0..nb * QK).map(|_| rng.f()).collect(); Act::quantize_ggml(&x, true) }).collect();
        assert!(ac.iter().any(|a| a.d.iter().any(|&d| d < 0.0)), "o teste precisa de d negativo");
        let mixed = gemm_all(&rp, 0, &ac);
        for k in 0..n {
            let mut o1 = [[0f32; 8]; 1];
            rp.gemm::<1>(0, [&ac[k]], &mut o1);
            assert_eq!(bits(&mixed[k..k + 1]), bits(&o1), "ativação {k}: x8/x4 e x1 diferem");
            for r in 0..8 {
                let o = oracle(&data[r * rb..(r + 1) * rb], &ac[k]);
                let scale = o.abs().max(1.0);
                assert!((mixed[k][r] as f64 - o).abs() < 1e-4 * scale, "ativação {k} linha {r}: {} contra o oráculo {o}", mixed[k][r]);
            }
        }
    }

    /// Primeiro tensor Q4_K (ordem alfabética) de um modelo local com >= 64 linhas: (bytes das 64 primeiras linhas, linhas, colunas).
    fn real_q4k_tensor() -> Option<(Vec<u8>, usize, usize)> {
        let path = std::env::var("BURITI_TEST_MODEL").unwrap_or_else(|_| format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR")));
        let g = Gguf::open(std::path::Path::new(&path)).ok()?;
        let mut names: Vec<&String> = g.tensors.keys().collect();
        names.sort();
        for name in names {
            let info = &g.tensors[name];
            if Gguf::dtype_of(info) == Some(Dtype::Q4K) && info.dims.len() == 2 && info.dims[0] % QK == 0 && info.dims[1] >= 64 {
                let (cols, rows) = (info.dims[0], 64);
                let n = rows * cols / QK * ROW_BLOCK;
                return Some((g.mmap[info.offset..info.offset + n].to_vec(), rows, cols));
            }
        }
        None
    }

    #[test]
    fn real_tensor_matches_oracle_old_and_variants_agree() {
        if !cpu_ok() {
            return;
        }
        let Some((data, rows, cols)) = real_q4k_tensor() else {
            println!("modelo de teste ausente ou sem tensor Q4_K: teste de tensor real PULADO");
            return;
        };
        let nb = cols / QK;
        let rb = nb * ROW_BLOCK;
        let rp = Repacked::new(&data, rows, cols).unwrap();
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let (mut e_old, mut e_rep, mut e_diff, mut omax) = (0f64, 0f64, 0f64, 0f64);
        for _ in 0..2 {
            // ativações com escala de unidade e alguns valores grandes (como as reais); 13 = 8 + 4 + 1
            let xs: Vec<Vec<f32>> = (0..13).map(|_| (0..cols).map(|j| rng.f() * if j % 97 == 0 { 20.0 } else { 1.0 }).collect()).collect();
            let ac = acts(&xs);
            for g in 0..rows / 8 {
                let mixed = gemm_all(&rp, g, &ac);
                for k in 0..13 {
                    let mut o1 = [[0f32; 8]; 1];
                    rp.gemm::<1>(g, [&ac[k]], &mut o1);
                    assert_eq!(bits(&mixed[k..k + 1]), bits(&o1), "tensor real: x8/x4 e x1 diferem (grupo {g}, ativação {k})");
                    for r in 0..8 {
                        let row = &data[(g * 8 + r) * rb..(g * 8 + r + 1) * rb];
                        let old = dot_row_q(Dtype::Q4K, row, &ac[k]);
                        let o = oracle(row, &ac[k]);
                        omax = omax.max(o.abs());
                        e_old = e_old.max((old as f64 - o).abs());
                        e_rep = e_rep.max((mixed[k][r] as f64 - o).abs());
                        e_diff = e_diff.max((mixed[k][r] - old).abs() as f64);
                    }
                }
            }
        }
        println!("tensor real ({rows} linhas x {cols}): erro máx./maior |resultado| contra o oráculo f64: antigo {:e}, repacked {:e}; repacked contra antigo {:e}", e_old / omax, e_rep / omax, e_diff / omax);
        assert!(e_rep / omax < 1e-4 && e_old / omax < 1e-4, "erro contra o oráculo grande demais");
        assert!(e_diff / omax < 2e-4, "repacked longe do caminho antigo");
        // dentes: com as linhas 2 e 3 trocadas o erro contra o oráculo explode
        let mut bad = Repacked::new(&data, rows, cols).unwrap();
        plant_swap_rows(&mut bad);
        let xs: Vec<Vec<f32>> = vec![(0..cols).map(|_| rng.f()).collect()];
        let ac = acts(&xs);
        let got = gemm_all(&bad, 0, &ac);
        let worst = (0..8).map(|r| (got[0][r] as f64 - oracle(&data[r * rb..(r + 1) * rb], &ac[0])).abs()).fold(0f64, f64::max);
        assert!(worst / omax > 1e-3, "defeito plantado (linhas trocadas) não foi detectado no tensor real");
    }

    // ------------------------------------------------------------------ Q6_K (D52)

    /// Linhas Q6_K aleatórias (210 bytes por bloco) com d = 0x2800 (2^-5, finito) em [208..210).
    fn random_rows6(rng: &mut Rng, nrows: usize, nb: usize) -> Vec<u8> {
        let mut data: Vec<u8> = (0..nrows * nb * ROW_BLOCK6).map(|_| (rng.next() & 0xFF) as u8).collect();
        for b in 0..nrows * nb {
            data[b * ROW_BLOCK6 + 208] = 0x00;
            data[b * ROW_BLOCK6 + 209] = 0x28;
        }
        data
    }

    fn raw_mut(rp: &mut Repacked) -> &mut [u8] {
        // SAFETY: vista em bytes do próprio vetor (ver `new`).
        unsafe { std::slice::from_raw_parts_mut(rp.buf.as_mut_ptr() as *mut u8, rp.buf.len() * 32) }
    }

    /// Defeito plantado: troca os `qs` das linhas 2 e 3 (os 3 fluxos) da unidade 0 do primeiro bloco.
    fn plant6_swap_rows(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for s in 0..3 {
            for e in 0..8 {
                b.swap(OFF6_Q + s * 64 + 2 * 8 + e, OFF6_Q + s * 64 + 3 * 8 + e);
            }
        }
    }

    /// Defeito plantado: troca as escalas sc[4] e sc[5] da lane 3 (linha 5) do primeiro bloco.
    fn plant6_swap_scales(rp: &mut Repacked) {
        let b = raw_mut(rp);
        b.swap(OFF6_SC + 2 * 16 + 3 * 2, OFF6_SC + 2 * 16 + 3 * 2 + 1);
    }

    /// Defeito plantado: troca os bits altos (fluxo qh) das linhas 1 e 6 na unidade 5 do primeiro bloco.
    fn plant6_swap_high_bits(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for e in 0..8 {
            b.swap(OFF6_Q + 5 * 192 + 2 * 64 + 8 + e, OFF6_Q + 5 * 192 + 2 * 64 + 6 * 8 + e);
        }
    }

    /// Defeito plantado: troca os fluxos ql0 e ql1 inteiros da unidade 2 do primeiro bloco.
    fn plant6_swap_ql_streams(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for e in 0..64 {
            b.swap(OFF6_Q + 2 * 192 + e, OFF6_Q + 2 * 192 + 64 + e);
        }
    }

    /// Defeito plantado: apaga os bits altos (qh) de todas as linhas da unidade 7 do último bloco.
    fn plant6_zero_qh(rp: &mut Repacked, nb: usize) {
        let b = raw_mut(rp);
        let base = (nb - 1) * BB6 + OFF6_Q + 7 * 192 + 128;
        b[base..base + 64].fill(0);
    }

    #[test]
    fn eligibility6() {
        assert_eq!(BB6, 8 * ROW_BLOCK6 + 16, "bloco = 8 linhas + 16 bytes de preenchimento");
        assert_eq!(BB6 % 32, 0);
        assert_eq!(OFF6_Q % 32, 0);
        assert_eq!(OFF6_Q + 8 * 192, BB6);
        assert!(Repacked::new_q6(&[], 12, 256).is_none(), "linhas não múltiplas de 8");
        assert!(Repacked::new_q6(&[0u8; 8 * 210], 8, 128).is_none(), "colunas não múltiplas de 256");
    }

    /// x1 = x4 = x8 (estrito) em dados arbitrários; = kernel antigo (estrito) em dados diádicos.
    #[test]
    fn q6_x1_x4_x8_equal_old_on_dyadic_and_each_other_on_random() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for trial in 0..12 {
            let nb = 1 + trial % 6;
            let data = random_rows6(&mut rng, 8, nb);
            let rp = Repacked::new_q6(&data, 8, nb * QK).unwrap();
            let n = 13; // 8 + 4 + 1
            let xs: Vec<Vec<f32>> = (0..n).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
            let ac = acts(&xs);
            let got = gemm_all(&rp, 0, &ac);
            let rb = nb * ROW_BLOCK6;
            for r in 0..8 {
                for k in 0..n {
                    let want = dot_row_q(Dtype::Q6K, &data[r * rb..(r + 1) * rb], &ac[k]);
                    let o = oracle6(&data[r * rb..(r + 1) * rb], &ac[k]);
                    // O reempacotado tem de dar EXATAMENTE o oráculo f64 (dados diádicos: tudo exato). O kernel
                    // antigo não: em alguns casos ele arredonda (medido: trial 9 dá 2 ulp de diferença, e o
                    // oráculo coincide com o reempacotado), então contra ele só há tolerância apertada.
                    assert_eq!(got[k][r].to_bits(), (o as f32).to_bits(), "Q6_K diádico: trial {trial} linha {r} ativação {k}: {} contra oráculo f64 {o} (antigo {want})", got[k][r]);
                    assert!((got[k][r] - want).abs() <= 4e-7 * o.abs().max(1.0) as f32, "Q6_K diádico: longe do antigo");
                }
            }
            let xs: Vec<Vec<f32>> = (0..n).map(|_| (0..nb * QK).map(|_| rng.f()).collect()).collect();
            let ac = acts(&xs);
            let mixed = gemm_all(&rp, 0, &ac);
            let ones: Vec<[f32; 8]> = (0..n).map(|k| { let mut o = [[0f32; 8]; 1]; rp.gemm::<1>(0, [&ac[k]], &mut o); o[0] }).collect();
            assert_eq!(bits(&mixed), bits(&ones), "Q6_K trial {trial}: x8/x4/x1 diferem");
            let mut o4 = [[0f32; 8]; 4];
            rp.gemm::<4>(0, [&ac[3], &ac[0], &ac[7], &ac[11]], &mut o4);
            for (s, k) in [3usize, 0, 7, 11].iter().enumerate() {
                assert_eq!(bits(&o4[s..s + 1]), bits(&ones[*k..*k + 1]), "Q6_K trial {trial}: x4 em outra posição difere do x1");
            }
            // contra o kernel antigo em dados arbitrários: só ordem de soma em float (tolerância)
            for k in 0..8 {
                for r in 0..8 {
                    let old = dot_row_q(Dtype::Q6K, &data[r * rb..(r + 1) * rb], &ac[k]);
                    assert!((mixed[k][r] - old).abs() <= 2e-4 * old.abs().max(1.0), "Q6_K trial {trial}: longe do kernel antigo ({} contra {old})", mixed[k][r]);
                }
            }
        }
    }

    /// Valores extremos: pesos q = 63 e q = 0 (a metade cada), escalas +-127/-128 e ativações +-127: é o
    /// pior caso da prova de não-estouro em i16 (todos os produtos com o mesmo sinal).
    #[test]
    fn q6_extremes_do_not_overflow() {
        if !cpu_ok() {
            return;
        }
        let nb = 2;
        for (qbyte, scale, xv) in [(0xFFu8, 127u8, 127i8), (0xFF, 128, 127), (0xFF, 127, -127), (0xFF, 128, -127), (0x00, 127, -127), (0xFF, 128, -128)] {
            let mut data = vec![qbyte; 8 * nb * ROW_BLOCK6];
            for b in 0..8 * nb {
                for g in 0..16 {
                    data[b * ROW_BLOCK6 + 192 + g] = scale;
                }
                data[b * ROW_BLOCK6 + 208] = 0x00;
                data[b * ROW_BLOCK6 + 209] = 0x28;
            }
            let rp = Repacked::new_q6(&data, 8, nb * QK).unwrap();
            // ativação construída à mão: q = xv em tudo, bsums coerentes, d = 2^-6
            let mut a = Act::new(Dtype::Q6K, &vec![1.0f32; nb * QK]).unwrap();
            a.q.iter_mut().for_each(|v| *v = xv);
            a.d.iter_mut().for_each(|v| *v = 1.0 / 64.0);
            a.bsums.iter_mut().for_each(|v| *v = 16 * xv as i16);
            let mut o = [[0f32; 8]; 1];
            rp.gemm::<1>(0, [&a], &mut o);
            let rb = nb * ROW_BLOCK6;
            for r in 0..8 {
                let want = dot_row_q(Dtype::Q6K, &data[r * rb..(r + 1) * rb], &a);
                assert_eq!(o[0][r].to_bits(), want.to_bits(), "extremos q={qbyte:#x} sc={scale} x={xv}: linha {r}");
                // valor exato esperado: 256 · nb · (q - 32) · sc · x · d · dx, com q = 63 (0xFF: ql = 15, qh = 3) ou 0
                let q = (if qbyte == 0xFF { 63i64 } else { 0 }) - 32;
                let sc = scale as i8 as i64;
                let exact = (256 * nb as i64 * q * sc * xv as i64) as f64 / 32.0 / 64.0;
                assert_eq!(o[0][r] as f64, exact, "extremos: valor exato, linha {r}");
            }
        }
    }

    /// Testes com dentes: cada defeito plantado no repacker TEM de reprovar a comparação exata.
    #[test]
    fn q6_planted_bugs_are_caught() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x0BAD_5EED_1234_5678);
        let nb = 3;
        let data = random_rows6(&mut rng, 8, nb);
        let xs: Vec<Vec<f32>> = (0..9).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
        let ac = acts(&xs);
        let rb = nb * ROW_BLOCK6;
        let equal_old = |rp: &Repacked| {
            let got = gemm_all(rp, 0, &ac);
            (0..8).all(|r| (0..9).all(|k| got[k][r].to_bits() == dot_row_q(Dtype::Q6K, &data[r * rb..(r + 1) * rb], &ac[k]).to_bits()))
        };
        assert!(equal_old(&Repacked::new_q6(&data, 8, nb * QK).unwrap()), "controle sem defeito");
        let plants: [(&str, Box<dyn Fn(&mut Repacked)>); 5] = [
            ("linhas trocadas", Box::new(plant6_swap_rows)),
            ("escalas trocadas", Box::new(plant6_swap_scales)),
            ("bits altos trocados", Box::new(plant6_swap_high_bits)),
            ("fluxos ql trocados", Box::new(plant6_swap_ql_streams)),
            ("qh apagado", Box::new(move |rp: &mut Repacked| plant6_zero_qh(rp, nb))),
        ];
        for (name, plant) in plants {
            let mut bad = Repacked::new_q6(&data, 8, nb * QK).unwrap();
            plant(&mut bad);
            assert!(!equal_old(&bad), "defeito plantado NÃO foi detectado: {name}");
        }
    }

    /// Oráculo em f64 INDEPENDENTE do kernel antigo: decodifica o formato Q6_K elemento a elemento
    /// (posição e -> metade n, parte p, posição l; escala sc[e/16]) contra a ativação quantizada do `Act`.
    fn oracle6(row: &[u8], a: &Act) -> f64 {
        let mut total = 0f64;
        for (i, blk) in row.chunks_exact(ROW_BLOCK6).enumerate() {
            let d = crate::native::quant::f16_to_f32(u16::from_le_bytes([blk[208], blk[209]])) as f64;
            let mut sumi = 0i64;
            for e in 0..256 {
                let (n, p, l) = (e / 128, (e % 128) / 32, e % 32);
                let lo = blk[n * 64 + if p % 2 == 0 { l } else { 32 + l }];
                let nib = if p < 2 { lo & 15 } else { lo >> 4 };
                let hi = (blk[128 + n * 32 + l] >> (2 * p)) & 3;
                let q = (nib | (hi << 4)) as i64 - 32;
                let sc = blk[192 + e / 16] as i8 as i64;
                sumi += sc * q * a.q[i * QK + e] as i64;
            }
            total += d * a.d[i] as f64 * sumi as f64;
        }
        total
    }

    /// Modelos locais para os testes de tensor real: `BURITI_TEST_MODELS` (separados por `;`), senão
    /// `BURITI_TEST_MODEL`, senão os Qwen 0,5B e 1,5B de buriti-data. Só os que existem.
    fn test_models() -> Vec<String> {
        let list: Vec<String> = match (std::env::var("BURITI_TEST_MODELS"), std::env::var("BURITI_TEST_MODEL")) {
            (Ok(v), _) => v.split(';').map(str::to_string).collect(),
            (_, Ok(v)) => vec![v],
            _ => ["qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-1.5b-instruct-q4_k_m.gguf"].iter().map(|m| format!("{}/buriti-data/models/{m}", env!("CARGO_MANIFEST_DIR"))).collect(),
        };
        list.into_iter().filter(|p| std::path::Path::new(p).exists()).collect()
    }

    /// Tensor Q6_K de camada (`blk.*`) com >= 64 linhas de um dos modelos: (bytes das 64 primeiras linhas, linhas, colunas, nome).
    fn real_q6k_tensors() -> Vec<(Vec<u8>, usize, usize, String)> {
        let mut found = Vec::new();
        for path in test_models() {
            let Ok(g) = Gguf::open(std::path::Path::new(&path)) else { continue };
            let mut names: Vec<&String> = g.tensors.keys().filter(|n| n.starts_with("blk.")).collect();
            names.sort();
            for name in names {
                let info = &g.tensors[name];
                if Gguf::dtype_of(info) == Some(Dtype::Q6K) && info.dims.len() == 2 && info.dims[0] % QK == 0 && info.dims[1] >= 64 {
                    let (cols, rows) = (info.dims[0], 64);
                    let n = rows * cols / QK * ROW_BLOCK6;
                    found.push((g.mmap[info.offset..info.offset + n].to_vec(), rows, cols, format!("{path}:{name}")));
                    break;
                }
            }
        }
        found
    }

    #[test]
    fn q6_real_tensor_matches_oracle_old_and_variants_agree() {
        if !cpu_ok() {
            return;
        }
        let tensors = real_q6k_tensors();
        if tensors.is_empty() {
            println!("nenhum modelo de teste com tensor Q6_K de camada: teste de tensor real Q6_K PULADO");
            return;
        }
        for (data, rows, cols, name) in tensors {
            let nb = cols / QK;
            let rb = nb * ROW_BLOCK6;
            let rp = Repacked::new_q6(&data, rows, cols).unwrap();
            let mut rng = Rng(0xA5A5_1234_DEAD_BEEF);
            let (mut e_old, mut e_rep, mut e_diff, mut omax) = (0f64, 0f64, 0f64, 0f64);
            for _ in 0..2 {
                let xs: Vec<Vec<f32>> = (0..13).map(|_| (0..cols).map(|j| rng.f() * if j % 97 == 0 { 20.0 } else { 1.0 }).collect()).collect();
                let ac = acts(&xs);
                for g in 0..rows / 8 {
                    let mixed = gemm_all(&rp, g, &ac);
                    for k in 0..13 {
                        let mut o1 = [[0f32; 8]; 1];
                        rp.gemm::<1>(g, [&ac[k]], &mut o1);
                        assert_eq!(bits(&mixed[k..k + 1]), bits(&o1), "{name}: x8/x4 e x1 diferem (grupo {g}, ativação {k})");
                        for r in 0..8 {
                            let row = &data[(g * 8 + r) * rb..(g * 8 + r + 1) * rb];
                            let old = dot_row_q(Dtype::Q6K, row, &ac[k]);
                            let o = oracle6(row, &ac[k]);
                            omax = omax.max(o.abs());
                            e_old = e_old.max((old as f64 - o).abs());
                            e_rep = e_rep.max((mixed[k][r] as f64 - o).abs());
                            e_diff = e_diff.max((mixed[k][r] - old).abs() as f64);
                        }
                    }
                }
            }
            println!("{name} ({rows} linhas x {cols}): erro máx./maior |resultado| contra o oráculo f64: antigo {:e}, repacked {:e}; repacked contra antigo {:e}", e_old / omax, e_rep / omax, e_diff / omax);
            assert!(e_rep / omax < 1e-4 && e_old / omax < 1e-4, "{name}: erro contra o oráculo grande demais");
            assert!(e_diff / omax < 2e-4, "{name}: repacked longe do caminho antigo");
            // dentes: com os defeitos plantados o erro contra o oráculo explode
            let xs: Vec<Vec<f32>> = vec![(0..cols).map(|_| rng.f()).collect()];
            let ac = acts(&xs);
            for (what, plant) in [("linhas", plant6_swap_rows as fn(&mut Repacked)), ("escalas", plant6_swap_scales), ("bits altos", plant6_swap_high_bits), ("fluxos ql", plant6_swap_ql_streams)] {
                let mut bad = Repacked::new_q6(&data, rows, cols).unwrap();
                plant(&mut bad);
                let got = gemm_all(&bad, 0, &ac);
                let worst = (0..8).map(|r| (got[0][r] as f64 - oracle6(&data[r * rb..(r + 1) * rb], &ac[0])).abs()).fold(0f64, f64::max);
                assert!(worst > 30.0 * e_rep.max(1e-12), "{name}: defeito plantado ({what}) não foi detectado no tensor real: erro {worst:e} contra o ruído {e_rep:e}");
            }
        }
    }

    // ------------------------------------------------------------------ Q8_0 / Q4_0 / Q5_0 (D81)

    const KINDS32: [Kind32; 3] = [Kind32::Q8_0, Kind32::Q4_0, Kind32::Q5_0];

    fn dt32(kind: Kind32) -> Dtype {
        match kind {
            Kind32::Q8_0 => Dtype::Q8_0,
            Kind32::Q4_0 => Dtype::Q4_0,
            Kind32::Q5_0 => Dtype::Q5_0,
        }
    }

    /// Linhas aleatórias do formato, com `d` (f16) escolhido por bloco entre 2^-6, 2^-5 e 2^-4 (finitos, e assim os
    /// dados diádicos continuam exatos em float; escalas diferentes por linha deixam a troca de escalas detectável).
    fn random_rows32(rng: &mut Rng, kind: Kind32, nrows: usize, nb: usize) -> Vec<u8> {
        let sb = kind.src_block();
        let mut data: Vec<u8> = (0..nrows * nb * sb).map(|_| (rng.next() & 0xFF) as u8).collect();
        for b in 0..nrows * nb {
            let h: u16 = [0x2400, 0x2800, 0x2C00][(rng.next() % 3) as usize];
            data[b * sb..b * sb + 2].copy_from_slice(&h.to_le_bytes());
        }
        data
    }

    /// Como `dyadic_x`, com um +-127 em cada bloco de 32 (d = 2^-6 e q = x·64 exatos em todos os blocos).
    fn dyadic_x32(rng: &mut Rng, cols: usize) -> Vec<f32> {
        let mut x: Vec<f32> = (0..cols).map(|_| ((rng.next() % 255) as i32 - 127) as f32 / 64.0).collect();
        for b in 0..cols / 32 {
            x[b * 32 + (rng.next() % 32) as usize] = if rng.next() & 1 == 0 { 127.0 / 64.0 } else { -127.0 / 64.0 };
        }
        x
    }

    fn acts32(kind: Kind32, xs: &[Vec<f32>]) -> Vec<Act> {
        xs.iter().map(|x| Act::new(dt32(kind), x).unwrap()).collect()
    }

    /// Oráculo em f64 INDEPENDENTE dos kernels (decodifica elemento a elemento, como o ggml) contra a ativação quantizada.
    fn oracle32(kind: Kind32, row: &[u8], a: &Act) -> f64 {
        let mut total = 0f64;
        for (i, blk) in row.chunks_exact(kind.src_block()).enumerate() {
            let d = crate::native::quant::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
            let mut sumi = 0i64;
            for e in 0..32 {
                let w: i64 = match kind {
                    Kind32::Q8_0 => blk[2 + e] as i8 as i64,
                    Kind32::Q4_0 => {
                        let b = blk[2 + e % 16];
                        (if e < 16 { b & 15 } else { b >> 4 }) as i64 - 8
                    }
                    Kind32::Q5_0 => {
                        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                        let b = blk[6 + e % 16];
                        let nib = if e < 16 { b & 15 } else { b >> 4 } as i64;
                        (nib | ((((qh >> e) & 1) as i64) << 4)) - 16
                    }
                };
                sumi += w * a.q[i * 32 + e] as i64;
            }
            total += d * a.d[i] as f64 * sumi as f64;
        }
        total
    }

    /// Defeitos plantados no repacker (um grupo, bloco 0): linhas 2 e 3 trocadas num pedaço, escalas das linhas 1 e 5
    /// trocadas, pedaços 0 e 1 trocados (todas as linhas) e, no Q5_0, os bits altos da linha 3 invertidos.
    fn plants32(kind: Kind32, nb: usize) -> Vec<(&'static str, Box<dyn Fn(&mut Repacked)>)> {
        let dsec = Kind32::dsec(nb);
        let mut v: Vec<(&'static str, Box<dyn Fn(&mut Repacked)>)> = vec![
            ("linhas trocadas", Box::new(move |rp: &mut Repacked| {
                let b = raw_mut(rp);
                for e in 0..4 {
                    b.swap(dsec + 2 * 4 + e, dsec + 3 * 4 + e);
                }
            })),
            ("escalas trocadas", Box::new(move |rp: &mut Repacked| {
                let b = raw_mut(rp);
                for i in 0..nb {
                    b.swap(i * 16 + 2, i * 16 + 10);
                    b.swap(i * 16 + 3, i * 16 + 11);
                }
            })),
            ("pedaços trocados", Box::new(move |rp: &mut Repacked| {
                let b = raw_mut(rp);
                for e in 0..32 {
                    b.swap(dsec + e, dsec + 32 + e);
                }
            })),
        ];
        if kind == Kind32::Q5_0 {
            v.push(("bits altos invertidos", Box::new(move |rp: &mut Repacked| {
                let b = raw_mut(rp);
                for e in 0..4 {
                    b[dsec + 128 + 3 * 4 + e] ^= 0xFF;
                }
            })));
            v.push(("bits altos trocados entre pedaços", Box::new(move |rp: &mut Repacked| {
                // o bit do pedaço 1 passa para a posição do pedaço 2 em todos os bytes da linha 6
                let b = raw_mut(rp);
                for e in 0..4 {
                    let x = b[dsec + 128 + 6 * 4 + e];
                    let (b1, b2) = ((x >> 1) & 1, (x >> 2) & 1);
                    b[dsec + 128 + 6 * 4 + e] = (x & !0b110) | (b2 << 1) | (b1 << 2);
                }
            })));
        }
        v
    }

    #[test]
    fn b32_layout_and_eligibility() {
        for kind in KINDS32 {
            for nb in [1usize, 2, 3, 7, 28, 48] {
                assert_eq!(kind.group_bytes(nb) % 32, 0, "{kind:?} nb={nb}: grupo desalinhado");
                assert_eq!(Kind32::dsec(nb) % 32, 0);
                assert_eq!(kind.qbytes() % 32, 0);
                // nunca maior que o original mais o preenchimento das escalas (< 32 bytes)
                assert!(kind.group_bytes(nb) < 8 * nb * kind.src_block() + 32, "{kind:?} nb={nb}");
                assert!(kind.group_bytes(nb) >= 8 * nb * kind.src_block());
            }
            assert!(Repacked::new32(kind, &[], 12, 32).is_none(), "linhas não múltiplas de 8");
            assert!(Repacked::new32(kind, &[0u8; 8 * 34], 8, 48).is_none(), "colunas não múltiplas de 32");
        }
    }

    /// x1 = x4 = x8 (estrito) em dados arbitrários; = kernel antigo (estrito) e = oráculo f64 em dados diádicos.
    #[test]
    fn b32_x1_x4_x8_equal_old_on_dyadic_and_each_other_on_random() {
        if !cpu_ok() {
            return;
        }
        for kind in KINDS32 {
            let mut rng = Rng(0x2545_F491_4F6C_DD1D ^ kind as u64);
            for trial in 0..12 {
                let nb = 1 + trial % 5;
                let data = random_rows32(&mut rng, kind, 8, nb);
                let rp = Repacked::new32(kind, &data, 8, nb * 32).unwrap();
                let n = 13; // 8 + 4 + 1
                let xs: Vec<Vec<f32>> = (0..n).map(|_| dyadic_x32(&mut rng, nb * 32)).collect();
                let ac = acts32(kind, &xs);
                let got = gemm_all(&rp, 0, &ac);
                let rb = nb * kind.src_block();
                for r in 0..8 {
                    for k in 0..n {
                        let row = &data[r * rb..(r + 1) * rb];
                        let want = dot_row_q(dt32(kind), row, &ac[k]);
                        let o = oracle32(kind, row, &ac[k]);
                        assert_eq!(got[k][r].to_bits(), want.to_bits(), "{kind:?} diádico: trial {trial} linha {r} ativação {k}: {} contra o kernel antigo {want}", got[k][r]);
                        assert_eq!(got[k][r] as f64, o, "{kind:?} diádico: trial {trial} linha {r} ativação {k}: contra o oráculo f64");
                    }
                }
                let xs: Vec<Vec<f32>> = (0..n).map(|_| (0..nb * 32).map(|_| rng.f()).collect()).collect();
                let ac = acts32(kind, &xs);
                let mixed = gemm_all(&rp, 0, &ac);
                let ones: Vec<[f32; 8]> = (0..n).map(|k| { let mut o = [[0f32; 8]; 1]; rp.gemm::<1>(0, [&ac[k]], &mut o); o[0] }).collect();
                assert_eq!(bits(&mixed), bits(&ones), "{kind:?} trial {trial}: x8/x4/x1 diferem");
                let mut o4 = [[0f32; 8]; 4];
                rp.gemm::<4>(0, [&ac[3], &ac[0], &ac[7], &ac[11]], &mut o4);
                for (s, k) in [3usize, 0, 7, 11].iter().enumerate() {
                    assert_eq!(bits(&o4[s..s + 1]), bits(&ones[*k..*k + 1]), "{kind:?} trial {trial}: x4 em outra posição difere do x1");
                }
                // contra o kernel antigo e o oráculo em dados arbitrários: só ordem de soma em float
                for k in 0..n {
                    for r in 0..8 {
                        let row = &data[r * rb..(r + 1) * rb];
                        let old = dot_row_q(dt32(kind), row, &ac[k]);
                        let o = oracle32(kind, row, &ac[k]);
                        let scale = o.abs().max(1.0);
                        assert!((mixed[k][r] as f64 - o).abs() <= 1e-4 * scale, "{kind:?} trial {trial}: longe do oráculo ({} contra {o})", mixed[k][r]);
                        assert!((mixed[k][r] - old).abs() as f64 <= 2e-4 * scale, "{kind:?} trial {trial}: longe do kernel antigo ({} contra {old})", mixed[k][r]);
                    }
                }
            }
        }
    }

    /// Valores extremos: o pior caso das provas de não-estouro (todos os produtos com o mesmo sinal). Compara com o valor
    /// exato calculado à mão (e, com |x| <= 127, com o kernel antigo). Q4_0/Q5_0 também com x = -128 (cabe: 30720 e 31744).
    #[test]
    fn b32_extremes_do_not_overflow() {
        if !cpu_ok() {
            return;
        }
        let nb = 3;
        for kind in KINDS32 {
            // (byte dos pesos, byte do qh, valor exato do peso, x)
            let mut cases: Vec<(u8, u8, i64, i8)> = Vec::new();
            for &x in &[127i8, -127, -128] {
                if kind == Kind32::Q8_0 {
                    if x != -128 {
                        cases.extend([(0x7F, 0, 127, x), (0x80, 0, -128, x), (0x00, 0, 0, x), (0x81, 0, -127, x)]);
                    }
                } else if kind == Kind32::Q4_0 {
                    cases.extend([(0xFF, 0, 15 - 8, x), (0x00, 0, -8, x), (0x88, 0, 0, x)]);
                } else {
                    cases.extend([(0xFF, 0xFF, 31 - 16, x), (0x00, 0x00, -16, x), (0xFF, 0x00, 15 - 16, x), (0x00, 0xFF, 16 - 16, x)]);
                }
            }
            for (wbyte, qhbyte, wval, xv) in cases {
                let sb = kind.src_block();
                let mut data = vec![wbyte; 8 * nb * sb];
                for b in 0..8 * nb {
                    data[b * sb..b * sb + 2].copy_from_slice(&0x2800u16.to_le_bytes());
                    if kind == Kind32::Q5_0 {
                        data[b * sb + 2..b * sb + 6].fill(qhbyte);
                    }
                }
                let rp = Repacked::new32(kind, &data, 8, nb * 32).unwrap();
                let mut a = Act::new(dt32(kind), &vec![1.0f32; nb * 32]).unwrap();
                a.q.iter_mut().for_each(|v| *v = xv);
                a.d.iter_mut().for_each(|v| *v = 1.0 / 64.0);
                a.bsums.iter_mut().for_each(|v| *v = 16 * xv as i16);
                for n in [1usize, 4, 8] {
                    let ac: Vec<Act> = (0..n).map(|_| Act { q: a.q.clone(), d: a.d.clone(), bsums: a.bsums.clone(), bp: Vec::new() }).collect();
                    let got = gemm_all(&rp, 0, &ac);
                    let exact = (32 * nb as i64 * wval * xv as i64) as f64 / 32.0 / 64.0;
                    for r in 0..8 {
                        for k in 0..n {
                            assert_eq!(got[k][r] as f64, exact, "{kind:?} extremos w={wval} x={xv} N={n}: linha {r}");
                        }
                        if xv != -128 {
                            let want = dot_row_q(dt32(kind), &data[r * nb * sb..(r + 1) * nb * sb], &a);
                            assert_eq!(got[0][r].to_bits(), want.to_bits(), "{kind:?} extremos w={wval} x={xv}: kernel antigo, linha {r}");
                        }
                    }
                }
            }
        }
    }

    /// Testes com dentes: cada defeito plantado no repacker TEM de reprovar a comparação exata com o kernel antigo.
    #[test]
    fn b32_planted_bugs_are_caught() {
        if !cpu_ok() {
            return;
        }
        for kind in KINDS32 {
            let mut rng = Rng(0x0BAD_5EED_1234_5678 ^ kind as u64);
            let nb = 3;
            let data = random_rows32(&mut rng, kind, 8, nb);
            let xs: Vec<Vec<f32>> = (0..9).map(|_| dyadic_x32(&mut rng, nb * 32)).collect();
            let ac = acts32(kind, &xs);
            let rb = nb * kind.src_block();
            let equal_old = |rp: &Repacked| {
                let got = gemm_all(rp, 0, &ac);
                (0..8).all(|r| (0..9).all(|k| got[k][r].to_bits() == dot_row_q(dt32(kind), &data[r * rb..(r + 1) * rb], &ac[k]).to_bits()))
            };
            assert!(equal_old(&Repacked::new32(kind, &data, 8, nb * 32).unwrap()), "{kind:?}: controle sem defeito");
            for (name, plant) in plants32(kind, nb) {
                let mut bad = Repacked::new32(kind, &data, 8, nb * 32).unwrap();
                plant(&mut bad);
                assert!(!equal_old(&bad), "{kind:?}: defeito plantado NÃO foi detectado: {name}");
            }
        }
    }

    /// Modelos para os testes de tensor real dos formatos de bloco de 32: os de `test_models()` mais o SmolVLM Q8_0 de
    /// buriti-data e, se existirem, os arquivos de `BURITI_TEST_MODELS32` (separados por `;`).
    fn test_models32() -> Vec<String> {
        let mut v = test_models();
        v.push(format!("{}/buriti-data/models/smolvlm-500m-instruct-q8_0.gguf", env!("CARGO_MANIFEST_DIR")));
        if let Ok(extra) = std::env::var("BURITI_TEST_MODELS32") {
            v.extend(extra.split(';').map(str::to_string));
        }
        v.dedup();
        v.into_iter().filter(|p| std::path::Path::new(p).exists()).collect()
    }

    /// Tensores de camada (`blk.*`) do formato com >= 64 linhas: (bytes das 64 primeiras linhas, linhas, colunas, nome).
    fn real_tensors32(kind: Kind32) -> Vec<(Vec<u8>, usize, usize, String)> {
        let mut found = Vec::new();
        for path in test_models32() {
            let Ok(g) = Gguf::open(std::path::Path::new(&path)) else { continue };
            let mut names: Vec<&String> = g.tensors.keys().filter(|n| n.starts_with("blk.")).collect();
            names.sort();
            for name in names {
                let info = &g.tensors[name];
                if Gguf::dtype_of(info) == Some(dt32(kind)) && info.dims.len() == 2 && info.dims[0] % 32 == 0 && info.dims[1] >= 64 {
                    let (cols, rows) = (info.dims[0], 64);
                    let n = rows * cols / 32 * kind.src_block();
                    found.push((g.mmap[info.offset..info.offset + n].to_vec(), rows, cols, format!("{path}:{name}")));
                    break;
                }
            }
        }
        found
    }

    #[test]
    fn b32_real_tensor_matches_oracle_old_and_variants_agree() {
        if !cpu_ok() {
            return;
        }
        for kind in KINDS32 {
            let tensors = real_tensors32(kind);
            if tensors.is_empty() {
                println!("{kind:?}: nenhum modelo de teste com tensor de camada: teste de tensor real PULADO");
                continue;
            }
            for (data, rows, cols, name) in tensors {
                let rb = cols / 32 * kind.src_block();
                let rp = Repacked::new32(kind, &data, rows, cols).unwrap();
                let mut rng = Rng(0xA5A5_1234_DEAD_BEEF);
                let (mut e_old, mut e_rep, mut e_diff, mut omax) = (0f64, 0f64, 0f64, 0f64);
                for _ in 0..2 {
                    let xs: Vec<Vec<f32>> = (0..13).map(|_| (0..cols).map(|j| rng.f() * if j % 97 == 0 { 20.0 } else { 1.0 }).collect()).collect();
                    let ac = acts32(kind, &xs);
                    for g in 0..rows / 8 {
                        let mixed = gemm_all(&rp, g, &ac);
                        for k in 0..13 {
                            let mut o1 = [[0f32; 8]; 1];
                            rp.gemm::<1>(g, [&ac[k]], &mut o1);
                            assert_eq!(bits(&mixed[k..k + 1]), bits(&o1), "{name}: x8/x4 e x1 diferem (grupo {g}, ativação {k})");
                            for r in 0..8 {
                                let row = &data[(g * 8 + r) * rb..(g * 8 + r + 1) * rb];
                                let old = dot_row_q(dt32(kind), row, &ac[k]);
                                let o = oracle32(kind, row, &ac[k]);
                                omax = omax.max(o.abs());
                                e_old = e_old.max((old as f64 - o).abs());
                                e_rep = e_rep.max((mixed[k][r] as f64 - o).abs());
                                e_diff = e_diff.max((mixed[k][r] - old).abs() as f64);
                            }
                        }
                    }
                }
                println!("{kind:?} {name} ({rows} linhas x {cols}): erro máx./maior |resultado| contra o oráculo f64: antigo {:e}, repacked {:e}; repacked contra antigo {:e}", e_old / omax, e_rep / omax, e_diff / omax);
                assert!(e_rep / omax < 1e-4 && e_old / omax < 1e-4, "{name}: erro contra o oráculo grande demais");
                assert!(e_diff / omax < 2e-4, "{name}: repacked longe do caminho antigo");
                // dentes: com os defeitos plantados o erro contra o oráculo explode
                let xs: Vec<Vec<f32>> = vec![(0..cols).map(|_| rng.f()).collect()];
                let ac = acts32(kind, &xs);
                for (what, plant) in plants32(kind, cols / 32) {
                    let mut bad = Repacked::new32(kind, &data, rows, cols).unwrap();
                    plant(&mut bad);
                    let got = gemm_all(&bad, 0, &ac);
                    let worst = (0..8).map(|r| (got[0][r] as f64 - oracle32(kind, &data[r * rb..(r + 1) * rb], &ac[0])).abs()).fold(0f64, f64::max);
                    assert!(worst > 30.0 * e_rep.max(1e-12), "{name}: defeito plantado ({what}) não foi detectado no tensor real: erro {worst:e} contra o ruído {e_rep:e}");
                }
            }
        }
    }

    // ------------------------------------------------------------------ Q5_K (D84)

    /// Linhas Q5_K aleatorias (176 bytes por bloco) com d/dmin = 0x2800 (2^-5, finito).
    fn random_rows5(rng: &mut Rng, nrows: usize, nb: usize) -> Vec<u8> {
        let mut data: Vec<u8> = (0..nrows * nb * ROW_BLOCK5).map(|_| (rng.next() & 0xFF) as u8).collect();
        for b in 0..nrows * nb {
            for off in [0usize, 2] {
                data[b * ROW_BLOCK5 + off] = 0x00;
                data[b * ROW_BLOCK5 + off + 1] = 0x28;
            }
        }
        data
    }

    fn acts5(xs: &[Vec<f32>]) -> Vec<Act> {
        xs.iter().map(|x| Act::new(Dtype::Q5K, x).unwrap()).collect()
    }

    /// Oraculo em f64 INDEPENDENTE do kernel antigo: decodifica o Q5_K elemento a elemento
    /// (sub-bloco j, posicao b: nibble de qs[32*(j/2) + b], 5o bit = bit j de qh[b]) contra a ativacao do `Act`.
    fn oracle5(row: &[u8], a: &Act) -> f64 {
        let mut total = 0f64;
        for (i, blk) in row.chunks_exact(ROW_BLOCK5).enumerate() {
            let f16 = |o: usize| crate::native::quant::f16_to_f32(u16::from_le_bytes([blk[o], blk[o + 1]])) as f64;
            let (dw, dmin) = (f16(0), f16(2));
            let dx = a.d[i] as f64;
            let (mut sumi, mut mins) = (0i64, 0i64);
            for j in 0..8 {
                let (s, m) = scale_min(j, &blk[4..16]);
                let (g, hi) = (j / 2, j % 2);
                let (mut ws, mut bs) = (0i64, 0i64);
                for b in 0..32 {
                    let byte = blk[48 + g * 32 + b];
                    let nib = if hi == 0 { byte & 15 } else { byte >> 4 } as i64;
                    let h = ((blk[16 + b] >> j) & 1) as i64;
                    let x = a.q[i * QK + j * 32 + b] as i64;
                    ws += (nib | (h << 4)) * x;
                    bs += x;
                }
                sumi += s as i64 * ws;
                mins += m as i64 * bs;
            }
            total += dw * dx * sumi as f64 - dmin * dx * mins as f64;
        }
        total
    }

    /// Defeito plantado: troca os `qh` das linhas 2 e 3 (os 4 passos) do primeiro bloco.
    fn plant5_swap_qh_rows(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for s in 0..4 {
            for e in 0..8 {
                b.swap(OFF5_QH + s * 64 + 2 * 8 + e, OFF5_QH + s * 64 + 3 * 8 + e);
            }
        }
    }

    /// Defeito plantado: troca os passos 0 e 1 do `qh` do primeiro bloco (os 64 bytes de cada).
    fn plant5_swap_qh_steps(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for e in 0..64 {
            b.swap(OFF5_QH + e, OFF5_QH + 64 + e);
        }
    }

    /// Defeito plantado: gira os bits de cada byte do `qh` do primeiro bloco (bit j vai para j+1: bit errado).
    fn plant5_rotate_qh_bits(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for v in &mut b[OFF5_QH..OFF5_QH + 256] {
            *v = v.rotate_left(1);
        }
    }

    /// Defeito plantado: inverte os bits do passo 2 do `qh` do primeiro bloco.
    fn plant5_invert_qh_step(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for v in &mut b[OFF5_QH + 128..OFF5_QH + 192] {
            *v = !*v;
        }
    }

    /// Defeito plantado: apaga o `qh` de todas as linhas do ultimo bloco.
    fn plant5_zero_qh(rp: &mut Repacked, nb: usize) {
        let b = raw_mut(rp);
        let base = (nb - 1) * BB5 + OFF5_QH;
        b[base..base + 256].fill(0);
    }

    /// Defeito plantado: troca o minimo do sub-bloco 3 da linha 4 pelo do sub-bloco 6 (desempacotamento errado).
    fn plant5_swap_mins(rp: &mut Repacked) {
        let b = raw_mut(rp);
        // minimos: [par p][lane (PERM)][2]: sub-bloco 3 = par 1, e = 1; sub-bloco 6 = par 3, e = 0; linha 4 = lane 2
        b.swap(OFF_MN + 16 + 2 * 2 + 1, OFF_MN + 3 * 16 + 2 * 2);
    }

    /// Defeito plantado: troca as escalas dos sub-blocos 0 e 7 de TODAS as linhas do primeiro bloco.
    fn plant5_swap_scales_all(rp: &mut Repacked) {
        let b = raw_mut(rp);
        for r in 0..8 {
            b.swap(OFF_SC + r, OFF_SC + 7 * 8 + r);
        }
    }

    fn plants5(nb: usize) -> Vec<(&'static str, Box<dyn Fn(&mut Repacked)>)> {
        vec![
            ("linhas (qs) trocadas", Box::new(plant_swap_rows)),
            ("escalas trocadas", Box::new(plant5_swap_scales_all)),
            ("minimos trocados", Box::new(plant5_swap_mins)),
            ("linhas do qh trocadas", Box::new(plant5_swap_qh_rows)),
            ("passos do qh trocados", Box::new(plant5_swap_qh_steps)),
            ("bits do qh girados", Box::new(plant5_rotate_qh_bits)),
            ("passo do qh invertido", Box::new(plant5_invert_qh_step)),
            ("qh apagado", Box::new(move |rp: &mut Repacked| plant5_zero_qh(rp, nb))),
        ]
    }

    #[test]
    fn eligibility5() {
        assert_eq!(BB5, 8 * ROW_BLOCK5 + 32, "bloco = 8 linhas + 32 bytes das escalas/minimos expandidos");
        assert_eq!(BB5 % 32, 0);
        assert_eq!(OFF5_QH, BB, "o qh vem logo depois do bloco do Q4_K");
        assert_eq!(OFF5_QH + 4 * 64, BB5);
        assert!(Repacked::new_q5(&[], 12, 256).is_none(), "linhas nao multiplas de 8");
        assert!(Repacked::new_q5(&[0u8; 8 * 176], 8, 128).is_none(), "colunas nao multiplas de 256");
        if cpu_ok() {
            let rp = Repacked::new_q5(&[0u8; 8 * 176], 8, 256).unwrap();
            assert_eq!(rp.size_bytes(), BB5);
        }
    }

    /// x1 = x4 = x8 (estrito) em dados arbitrarios; = kernel antigo (estrito) em dados diadicos (float exato).
    #[test]
    fn q5_x1_x4_x8_equal_old_on_dyadic_and_each_other_on_random() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x7F4A_7C15_9E37_79B9);
        for trial in 0..12 {
            let nb = 1 + trial % 6;
            let data = random_rows5(&mut rng, 8, nb);
            let rp = Repacked::new_q5(&data, 8, nb * QK).unwrap();
            let n = 13; // 8 + 4 + 1
            let xs: Vec<Vec<f32>> = (0..n).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
            let ac = acts5(&xs);
            let got = gemm_all(&rp, 0, &ac);
            let rb = nb * ROW_BLOCK5;
            for r in 0..8 {
                for k in 0..n {
                    let row = &data[r * rb..(r + 1) * rb];
                    let want = dot_row_q(Dtype::Q5K, row, &ac[k]);
                    let o = oracle5(row, &ac[k]);
                    assert_eq!(got[k][r].to_bits(), (o as f32).to_bits(), "Q5_K diadico: trial {trial} linha {r} ativacao {k}: {} contra o oraculo f64 {o} (antigo {want})", got[k][r]);
                    assert_eq!(got[k][r].to_bits(), want.to_bits(), "Q5_K diadico: trial {trial} linha {r} ativacao {k}: diferente do kernel antigo {want}");
                }
            }
            let xs: Vec<Vec<f32>> = (0..n).map(|_| (0..nb * QK).map(|_| rng.f()).collect()).collect();
            let ac = acts5(&xs);
            let mixed = gemm_all(&rp, 0, &ac);
            let ones: Vec<[f32; 8]> = (0..n).map(|k| { let mut o = [[0f32; 8]; 1]; rp.gemm::<1>(0, [&ac[k]], &mut o); o[0] }).collect();
            assert_eq!(bits(&mixed), bits(&ones), "Q5_K trial {trial}: x8/x4/x1 diferem");
            let mut o4 = [[0f32; 8]; 4];
            rp.gemm::<4>(0, [&ac[3], &ac[0], &ac[7], &ac[11]], &mut o4);
            for (s, k) in [3usize, 0, 7, 11].iter().enumerate() {
                assert_eq!(bits(&o4[s..s + 1]), bits(&ones[*k..*k + 1]), "Q5_K trial {trial}: x4 em outra posicao difere do x1");
            }
            // contra o kernel antigo em dados arbitrarios: so ordem de soma em float (tolerancia)
            for k in 0..8 {
                for r in 0..8 {
                    let old = dot_row_q(Dtype::Q5K, &data[r * rb..(r + 1) * rb], &ac[k]);
                    let scale = old.abs().max(1.0);
                    assert!((mixed[k][r] - old).abs() <= 2e-4 * scale, "Q5_K trial {trial}: longe do kernel antigo ({} contra {old})", mixed[k][r]);
                }
            }
        }
    }

    /// Com `BURITI_ACT_GGML` (d negativo, sinal trocado) o kernel continua estrito entre x1/x4/x8 e perto do oraculo.
    #[test]
    fn q5_ggml_style_acts() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x5EED_5A17);
        let nb = 3;
        let data = random_rows5(&mut rng, 8, nb);
        let rp = Repacked::new_q5(&data, 8, nb * QK).unwrap();
        let rb = nb * ROW_BLOCK5;
        let n = 13;
        let ac: Vec<Act> = (0..n).map(|_| { let x: Vec<f32> = (0..nb * QK).map(|_| rng.f()).collect(); Act::quantize_ggml(&x, true) }).collect();
        let mixed = gemm_all(&rp, 0, &ac);
        for k in 0..n {
            let mut o1 = [[0f32; 8]; 1];
            rp.gemm::<1>(0, [&ac[k]], &mut o1);
            assert_eq!(bits(&mixed[k..k + 1]), bits(&o1), "ativacao {k}: x8/x4 e x1 diferem");
            for r in 0..8 {
                let o = oracle5(&data[r * rb..(r + 1) * rb], &ac[k]);
                assert!((mixed[k][r] as f64 - o).abs() < 1e-4 * o.abs().max(1.0), "ativacao {k} linha {r}: {} contra o oraculo {o}", mixed[k][r]);
            }
        }
    }

    /// Valores extremos (pior caso da prova de nao-estouro em i16): pesos 31 (e 16 e 15), escalas e minimos 63,
    /// ativacoes +-127 e -128, tudo com o mesmo sinal; o resultado tem de ser EXATAMENTE o valor analitico.
    #[test]
    fn q5_extremes_do_not_overflow() {
        if !cpu_ok() {
            return;
        }
        let nb = 2;
        for (qs, qh, xv) in [(0xFFu8, 0xFFu8, 127i8), (0xFF, 0xFF, -127), (0xFF, 0xFF, -128), (0x00, 0xFF, 127), (0x00, 0xFF, -128), (0xFF, 0x00, -128), (0xFF, 0xFF, 1)] {
            let mut data = vec![0u8; 8 * nb * ROW_BLOCK5];
            for b in 0..8 * nb {
                let blk = &mut data[b * ROW_BLOCK5..(b + 1) * ROW_BLOCK5];
                blk[0] = 0x00;
                blk[1] = 0x28;
                blk[2] = 0x00;
                blk[3] = 0x28;
                blk[4..16].fill(0xFF); // todas as escalas e minimos = 63
                blk[16..48].fill(qh);
                blk[48..176].fill(qs);
            }
            let rp = Repacked::new_q5(&data, 8, nb * QK).unwrap();
            let mut a = Act::new(Dtype::Q5K, &vec![1.0f32; nb * QK]).unwrap();
            a.q.iter_mut().for_each(|v| *v = xv);
            a.d.iter_mut().for_each(|v| *v = 1.0 / 64.0);
            a.bsums.iter_mut().for_each(|v| *v = 16 * xv as i16);
            let s32 = (32 * xv as i16) as u16 as u32;
            a.bp.iter_mut().for_each(|v| *v = (s32 | (s32 << 16)) as i32);
            let q = (if qs == 0xFF { 15i64 } else { 0 }) + if qh == 0xFF { 16 } else { 0 };
            let (mut o, mut o2) = ([[0f32; 8]; 1], [[0f32; 8]; 2]);
            rp.gemm::<1>(0, [&a], &mut o);
            rp.gemm::<2>(0, [&a, &a], &mut o2);
            // 256 elementos por bloco: d*dx*soma(sc*q*x) - dmin*dx*soma(mn*x), com d = dmin = 2^-5, dx = 2^-6, sc = mn = 63
            let exact = nb as f64 * (256 * 63 * (q - 1) * xv as i64) as f64 / 32.0 / 64.0;
            for r in 0..8 {
                assert_eq!(o[0][r] as f64, exact, "extremos qs={qs:#x} qh={qh:#x} x={xv}: linha {r}");
                assert_eq!(o2[1][r].to_bits(), o[0][r].to_bits());
            }
        }
    }

    /// Testes com dentes: cada defeito plantado no repacker TEM de reprovar a comparacao exata com o kernel antigo.
    #[test]
    fn q5_planted_bugs_are_caught() {
        if !cpu_ok() {
            return;
        }
        let mut rng = Rng(0x0BAD_5EED_5555_1234);
        let nb = 3;
        let data = random_rows5(&mut rng, 8, nb);
        let xs: Vec<Vec<f32>> = (0..9).map(|_| dyadic_x(&mut rng, nb * QK)).collect();
        let ac = acts5(&xs);
        let rb = nb * ROW_BLOCK5;
        let equal_old = |rp: &Repacked| {
            let got = gemm_all(rp, 0, &ac);
            (0..8).all(|r| (0..9).all(|k| got[k][r].to_bits() == dot_row_q(Dtype::Q5K, &data[r * rb..(r + 1) * rb], &ac[k]).to_bits()))
        };
        assert!(equal_old(&Repacked::new_q5(&data, 8, nb * QK).unwrap()), "controle sem defeito");
        for (name, plant) in plants5(nb) {
            let mut bad = Repacked::new_q5(&data, 8, nb * QK).unwrap();
            plant(&mut bad);
            assert!(!equal_old(&bad), "defeito plantado NAO foi detectado: {name}");
        }
    }

    fn real_q5k_models() -> Vec<String> {
        let mut m = test_models();
        let phi = std::env::var("BURITI_TEST_PHI").unwrap_or_else(|_| format!("{}/buriti-data/models/Phi-3.5-mini-instruct-Q4_K_M.gguf", env!("CARGO_MANIFEST_DIR")));
        if std::env::var("BURITI_TEST_MODELS").is_err() && std::path::Path::new(&phi).exists() {
            m.push(phi);
        }
        m
    }

    /// Tensores Q5_K de camada (`blk.*`) com >= 64 linhas (ate 3 por modelo): (bytes das 64 primeiras linhas, linhas, colunas, nome).
    fn real_q5k_tensors() -> Vec<(Vec<u8>, usize, usize, String)> {
        let mut found = Vec::new();
        for path in real_q5k_models() {
            let Ok(g) = Gguf::open(std::path::Path::new(&path)) else { continue };
            let mut names: Vec<&String> = g.tensors.keys().filter(|n| n.starts_with("blk.")).collect();
            names.sort();
            let mut per_model = 0;
            for name in names {
                let info = &g.tensors[name];
                if Gguf::dtype_of(info) == Some(Dtype::Q5K) && info.dims.len() == 2 && info.dims[0] % QK == 0 && info.dims[1] >= 64 && per_model < 3 {
                    let (cols, rows) = (info.dims[0], 64);
                    let n = rows * cols / QK * ROW_BLOCK5;
                    found.push((g.mmap[info.offset..info.offset + n].to_vec(), rows, cols, format!("{path}:{name}")));
                    per_model += 1;
                }
            }
        }
        found
    }

    #[test]
    fn q5_real_tensor_matches_oracle_old_and_variants_agree() {
        if !cpu_ok() {
            return;
        }
        let tensors = real_q5k_tensors();
        if tensors.is_empty() {
            println!("nenhum modelo de teste com tensor Q5_K de camada: teste de tensor real Q5_K PULADO");
            return;
        }
        for (data, rows, cols, name) in tensors {
            let nb = cols / QK;
            let rb = nb * ROW_BLOCK5;
            let rp = Repacked::new_q5(&data, rows, cols).unwrap();
            let mut rng = Rng(0xA5A5_5A5A_DEAD_BEEF);
            let (mut e_old, mut e_rep, mut e_diff, mut omax) = (0f64, 0f64, 0f64, 0f64);
            for _ in 0..2 {
                let xs: Vec<Vec<f32>> = (0..13).map(|_| (0..cols).map(|j| rng.f() * if j % 97 == 0 { 20.0 } else { 1.0 }).collect()).collect();
                let ac = acts5(&xs);
                for g in 0..rows / 8 {
                    let mixed = gemm_all(&rp, g, &ac);
                    for k in 0..13 {
                        let mut o1 = [[0f32; 8]; 1];
                        rp.gemm::<1>(g, [&ac[k]], &mut o1);
                        assert_eq!(bits(&mixed[k..k + 1]), bits(&o1), "{name}: x8/x4 e x1 diferem (grupo {g}, ativacao {k})");
                        for r in 0..8 {
                            let row = &data[(g * 8 + r) * rb..(g * 8 + r + 1) * rb];
                            let old = dot_row_q(Dtype::Q5K, row, &ac[k]);
                            let o = oracle5(row, &ac[k]);
                            omax = omax.max(o.abs());
                            e_old = e_old.max((old as f64 - o).abs());
                            e_rep = e_rep.max((mixed[k][r] as f64 - o).abs());
                            e_diff = e_diff.max((mixed[k][r] - old).abs() as f64);
                        }
                    }
                }
            }
            println!("{name} ({rows} linhas x {cols}): erro max./maior |resultado| contra o oraculo f64: antigo {:e}, repacked {:e}; repacked contra antigo {:e}", e_old / omax, e_rep / omax, e_diff / omax);
            assert!(e_rep / omax < 1e-4 && e_old / omax < 1e-4, "{name}: erro contra o oraculo grande demais");
            assert!(e_diff / omax < 2e-4, "{name}: repacked longe do caminho antigo");
            let xs: Vec<Vec<f32>> = vec![(0..cols).map(|_| rng.f()).collect()];
            let ac = acts5(&xs);
            for (what, plant) in plants5(nb) {
                let mut bad = Repacked::new_q5(&data, rows, cols).unwrap();
                plant(&mut bad);
                // os defeitos tocam so um bloco do grupo 0: o resultado tem de diferir, bit a bit, do repacking correto
                // (que e deterministico); uma troca de uma escala em 12 blocos fica abaixo do ruido de f32 contra o oraculo
                let good = Repacked::new_q5(&data, rows, cols).unwrap();
                let (gb, gg) = (gemm_all(&bad, 0, &ac), gemm_all(&good, 0, &ac));
                assert_ne!(bits(&gb), bits(&gg), "{name}: defeito plantado ({what}) nao foi detectado no tensor real");
            }
        }
    }
}
