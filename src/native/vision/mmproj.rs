//! Leitor do arquivo `mmproj` (GGUF com `general.architecture = "clip"`): hiperparâmetros da torre de
//! visão e os tensores, com checagem de limites em tudo (arquivo malformado dá erro, nunca pânico).
//! Reaproveita o leitor GGUF do motor (`native/gguf.rs`); os pesos das matrizes ficam no mmap
//! (F16 ou Q8_0 etc.) e só vêm para `f32` os vetores pequenos (vieses, normas, posições).
//!
//! Nomes e formas conferidos contra o `mmproj-SmolVLM-500M-Instruct-f16.gguf` real (D50) e contra o
//! cabeçalho do `mmproj-model-f16.gguf` do Gemma 3 4B; as chaves e a regra de "ffn_up/ffn_down
//! trocados" seguem o `tools/mtmd/mtmd-model.cpp` do llama.cpp b12606 (lido como texto).

use std::path::Path;

use crate::native::{
    gguf::{Dtype, Gguf, Value},
    quant::dequant_row,
};

/// Qual conector liga a torre ao modelo de linguagem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projector {
    /// SmolVLM / Idefics3: pixel shuffle (`scale_factor`²) + uma camada linear sem viés.
    Idefics3,
    /// Gemma 3: média 2D (`scale_factor`×`scale_factor`) + RMSNorm + projeção (ainda não implementado).
    Gemma3,
}

#[derive(Debug, Clone)]
pub struct VisionHparams {
    pub projector: Projector,
    pub image_size: usize,
    pub patch_size: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub n_head: usize,
    /// dimensão de saída do conector = `n_embd` do modelo de linguagem
    pub proj_dim: usize,
    pub eps: f32,
    /// lado do bloco de fusão (pixel shuffle no Idefics3, média 2D no Gemma 3)
    pub scale_factor: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl VisionHparams {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
    /// patches por lado
    pub fn side(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn n_patches(&self) -> usize {
        self.side() * self.side()
    }
    /// vetores que a imagem entrega ao modelo de linguagem
    pub fn n_tokens(&self) -> usize {
        let s = self.side() / self.scale_factor;
        s * s
    }
}

/// Matriz de pesos no mmap: `rows` saídas × `cols` entradas, uma linha por saída (como no GGUF:
/// `dims = [cols, rows]`, a dimensão mais rápida primeiro).
#[derive(Debug, Clone, Copy)]
pub struct Mat {
    pub off: usize,
    pub dtype: Dtype,
    pub rows: usize,
    pub cols: usize,
}

pub struct Linear {
    pub w: Mat,
    pub b: Option<Vec<f32>>,
}

pub struct Norm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct Layer {
    pub ln1: Norm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub ln2: Norm,
    /// n_embd -> n_ff
    pub up: Linear,
    /// n_ff -> n_embd
    pub down: Linear,
}

pub struct Mmproj {
    pub gguf: Gguf,
    pub hp: VisionHparams,
    /// convolução do patch como matriz `n_embd × (3·patch²)`, colunas na ordem (canal, y, x)
    pub patch_w: Mat,
    pub patch_b: Vec<f32>,
    /// `n_patches × n_embd`
    pub pos: Vec<f32>,
    pub layers: Vec<Layer>,
    pub post_ln: Option<Norm>,
    /// `proj_dim × (n_embd·scale_factor²)` (Idefics3)
    pub fc: Mat,
}

impl std::fmt::Debug for Mmproj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mmproj({:?}, {} camadas)", self.hp, self.layers.len())
    }
}

impl Mmproj {
    pub fn open(path: &Path) -> Result<Mmproj, String> {
        Mmproj::from_gguf(Gguf::open(path)?)
    }

    /// Bytes de uma matriz (já validados em `from_gguf`).
    pub fn bytes(&self, m: &Mat) -> &[u8] {
        let (be, bb) = m.dtype.block();
        &self.gguf.mmap[m.off..m.off + m.rows * (m.cols / be * bb)]
    }

    pub fn from_gguf(g: Gguf) -> Result<Mmproj, String> {
        let hp = read_hparams(&g)?;
        if hp.projector != Projector::Idefics3 {
            return Err("conector 'gemma3' ainda não implementado no motor de visão (só 'idefics3', SmolVLM)".into());
        }
        let (n, ff) = (hp.n_embd, hp.n_ff);
        let p = hp.patch_size;

        let patch_w = mat_raw(&g, "v.patch_embd.weight", &[p, p, 3, n], n, 3 * p * p)?;
        if !matches!(patch_w.dtype, Dtype::F32 | Dtype::F16) {
            return Err("v.patch_embd.weight deve ser F32 ou F16".into());
        }
        let patch_b = vec_f32(&g, "v.patch_embd.bias", n)?;
        let pos = vec_f32(&g, "v.position_embd.weight", hp.n_patches() * n)?;

        let mut layers = Vec::with_capacity(hp.n_layer);
        for il in 0..hp.n_layer {
            let t = |s: &str| format!("v.blk.{il}.{s}");
            let lin = |name: &str, rows: usize, cols: usize| -> Result<Linear, String> {
                let w = mat(&g, &t(&format!("{name}.weight")), rows, cols)?;
                let b = opt_vec_f32(&g, &t(&format!("{name}.bias")), rows)?;
                Ok(Linear { w, b })
            };
            let norm = |name: &str| -> Result<Norm, String> {
                Ok(Norm { w: vec_f32(&g, &t(&format!("{name}.weight")), n)?, b: vec_f32(&g, &t(&format!("{name}.bias")), n)? })
            };
            // O conversor antigo grava `ffn_up`/`ffn_down` trocados (o llama.cpp corrige olhando a
            // forma: o "up" verdadeiro tem `dims = [n_embd, n_ff]`). Aqui também decidimos pela forma.
            let (a, b) = (g.tensors.get(&t("ffn_up.weight")), g.tensors.get(&t("ffn_down.weight")));
            let (a, b) = match (a, b) {
                (Some(a), Some(b)) => (a, b),
                _ => return Err(format!("tensores ffn_up/ffn_down ausentes na camada {il}")),
            };
            let swapped = if a.dims == [n, ff] && b.dims == [ff, n] {
                false
            } else if a.dims == [ff, n] && b.dims == [n, ff] {
                true
            } else {
                return Err(format!("formas inesperadas de ffn_up/ffn_down na camada {il}: {:?} e {:?}", a.dims, b.dims));
            };
            let (up_name, down_name) = if swapped { ("ffn_down", "ffn_up") } else { ("ffn_up", "ffn_down") };
            layers.push(Layer {
                ln1: norm("ln1")?,
                q: lin("attn_q", n, n)?,
                k: lin("attn_k", n, n)?,
                v: lin("attn_v", n, n)?,
                o: lin("attn_out", n, n)?,
                ln2: norm("ln2")?,
                up: lin(up_name, ff, n)?,
                down: lin(down_name, n, ff)?,
            });
        }

        let post_ln = if g.tensors.contains_key("v.post_ln.weight") {
            Some(Norm { w: vec_f32(&g, "v.post_ln.weight", n)?, b: vec_f32(&g, "v.post_ln.bias", n)? })
        } else {
            None
        };
        let fc = mat(&g, "mm.model.fc.weight", hp.proj_dim, n * hp.scale_factor * hp.scale_factor)?;
        Ok(Mmproj { gguf: g, hp, patch_w, patch_b, pos, layers, post_ln, fc })
    }
}

fn get_u(g: &Gguf, key: &str) -> Result<usize, String> {
    let v = g.u64(key)?;
    usize::try_from(v).map_err(|_| format!("valor enorme em {key}"))
}

fn read_hparams(g: &Gguf) -> Result<VisionHparams, String> {
    if g.meta.get("general.architecture").and_then(Value::as_str) != Some("clip") {
        return Err("não é um arquivo mmproj (general.architecture != clip)".into());
    }
    if g.meta.get("clip.has_vision_encoder").and_then(Value::as_bool) != Some(true) {
        return Err("o mmproj não tem codificador de visão".into());
    }
    let projector = match g.meta.get("clip.projector_type").and_then(Value::as_str) {
        Some("idefics3") => Projector::Idefics3,
        Some("gemma3") => Projector::Gemma3,
        Some(o) => return Err(format!("conector de visão '{o}' não suportado")),
        None => return Err("metadado ausente: clip.projector_type".into()),
    };
    // O llama.cpp usa GELU "quick" quando nenhuma das duas chaves existe; aqui só o GELU de tangente
    // hiperbólica (SigLIP) existe, então recusamos o resto em vez de calcular errado.
    if g.meta.get("clip.use_gelu").and_then(Value::as_bool) != Some(true) {
        return Err("ativação da torre de visão não suportada (só GELU de tangente hiperbólica, clip.use_gelu)".into());
    }
    let scale_factor = match g.meta.get("clip.vision.projector.scale_factor").and_then(Value::as_u64) {
        Some(v) => v as usize,
        None if projector == Projector::Gemma3 => 4, // padrão do llama.cpp para todo o Gemma 3
        None => return Err("metadado ausente: clip.vision.projector.scale_factor".into()),
    };
    let rgb = |key: &str| -> Result<[f32; 3], String> {
        let a = g.meta.get(key).and_then(Value::as_arr).ok_or_else(|| format!("metadado ausente: {key}"))?;
        if a.len() < 3 {
            return Err(format!("{key} precisa de 3 valores"));
        }
        let mut o = [0f32; 3];
        for (i, v) in a.iter().take(3).enumerate() {
            o[i] = v.as_f64().ok_or_else(|| format!("{key} inválido"))? as f32;
        }
        Ok(o)
    };
    let hp = VisionHparams {
        projector,
        image_size: get_u(g, "clip.vision.image_size")?,
        patch_size: get_u(g, "clip.vision.patch_size")?,
        n_embd: get_u(g, "clip.vision.embedding_length")?,
        n_ff: get_u(g, "clip.vision.feed_forward_length")?,
        n_layer: get_u(g, "clip.vision.block_count")?,
        n_head: get_u(g, "clip.vision.attention.head_count")?,
        proj_dim: get_u(g, "clip.vision.projection_dim")?,
        eps: g.f64("clip.vision.attention.layer_norm_epsilon").ok_or("metadado ausente: clip.vision.attention.layer_norm_epsilon")? as f32,
        scale_factor,
        mean: rgb("clip.vision.image_mean")?,
        std: rgb("clip.vision.image_std")?,
    };
    // limites que evitam alocação absurda e divisões por zero num arquivo malformado
    let ok = (1..=2048).contains(&hp.image_size)
        && (1..=128).contains(&hp.patch_size)
        && hp.image_size % hp.patch_size == 0
        && (1..=8192).contains(&hp.n_embd)
        && (1..=65536).contains(&hp.n_ff)
        && (1..=128).contains(&hp.n_layer)
        && (1..=128).contains(&hp.n_head)
        && hp.n_embd % hp.n_head == 0
        && (1..=65536).contains(&hp.proj_dim)
        && (1..=16).contains(&hp.scale_factor)
        && hp.side() % hp.scale_factor == 0
        && hp.eps > 0.0
        && hp.std.iter().all(|s| *s > 0.0);
    if !ok {
        return Err("hiperparâmetros da torre de visão inválidos ou fora dos limites".into());
    }
    Ok(hp)
}

/// Localiza o tensor, confere `dims` e que os bytes cabem no arquivo; devolve a matriz
/// `rows × cols` (linhas contíguas, `cols` múltiplo do bloco da quantização).
fn mat_raw(g: &Gguf, name: &str, dims: &[usize], rows: usize, cols: usize) -> Result<Mat, String> {
    let info = g.tensors.get(name).ok_or_else(|| format!("tensor ausente no mmproj: {name}"))?;
    if info.dims != dims {
        return Err(format!("forma inesperada de {name}: {:?} (esperado {:?})", info.dims, dims));
    }
    let dtype = Gguf::dtype_of(info).ok_or_else(|| format!("tipo de tensor não suportado em {name} (ggml {})", info.ggml_type))?;
    let (be, bb) = dtype.block();
    if cols % be != 0 {
        return Err(format!("{name}: {cols} colunas não é múltiplo do bloco {be}"));
    }
    let bytes = rows.checked_mul(cols / be * bb).ok_or("tensor enorme")?;
    match info.offset.checked_add(bytes) {
        Some(end) if end <= g.mmap.len() => {}
        _ => return Err(format!("tensor {name} fora do arquivo (truncado?)")),
    }
    Ok(Mat { off: info.offset, dtype, rows, cols })
}

fn mat(g: &Gguf, name: &str, rows: usize, cols: usize) -> Result<Mat, String> {
    mat_raw(g, name, &[cols, rows], rows, cols)
}

/// Lê um vetor pequeno (F32/F16/BF16) como `f32`, conferindo o tamanho.
fn vec_f32(g: &Gguf, name: &str, len: usize) -> Result<Vec<f32>, String> {
    opt_vec_f32(g, name, len)?.ok_or_else(|| format!("tensor ausente no mmproj: {name}"))
}

fn opt_vec_f32(g: &Gguf, name: &str, len: usize) -> Result<Option<Vec<f32>>, String> {
    let Some(info) = g.tensors.get(name) else { return Ok(None) };
    let total: usize = info.dims.iter().product();
    if total != len {
        return Err(format!("tamanho inesperado de {name}: {total} (esperado {len})"));
    }
    let dtype = Gguf::dtype_of(info).ok_or_else(|| format!("tipo de tensor não suportado em {name}"))?;
    if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16) {
        return Err(format!("{name} deve ser F32, F16 ou BF16"));
    }
    let bytes = len * dtype.block().1;
    match info.offset.checked_add(bytes) {
        Some(end) if end <= g.mmap.len() => {}
        _ => return Err(format!("tensor {name} fora do arquivo (truncado?)")),
    }
    let mut out = vec![0f32; len];
    dequant_row(dtype, &g.mmap[info.offset..info.offset + bytes], &mut out);
    Ok(Some(out))
}
