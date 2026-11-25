//! Leitor de arquivos GGUF (v2/v3): metadados + tabela de tensores, com os dados mapeados em memória.
//! Todo acesso é checado contra o tamanho do arquivo (um GGUF malformado dá erro, nunca pânico/OOB).

use std::{collections::HashMap, fs::File, path::Path};

use memmap2::Mmap;

#[derive(Debug, Clone)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    A(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::B(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::S(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::A(a) => Some(a),
            _ => None,
        }
    }
}

/// Tipos de tensor suportados pelo motor próprio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Q4_0,
    Q5_0,
    Q8_0,
    Q4K,
    Q6K,
    // a ordem importa: o índice (`as usize`) é usado nos contadores de diagnóstico (`Dtype::COUNT`)
    Q4_1,
    Q5_1,
    Q5K,
    BF16,
}

impl Dtype {
    pub const COUNT: usize = 11;
    pub const NAMES: [&'static str; Dtype::COUNT] =
        ["F32", "F16", "Q4_0", "Q5_0", "Q8_0", "Q4_K", "Q6_K", "Q4_1", "Q5_1", "Q5_K", "BF16"];

    fn from_ggml(t: u32) -> Option<Dtype> {
        Some(match t {
            0 => Dtype::F32,
            1 => Dtype::F16,
            2 => Dtype::Q4_0,
            3 => Dtype::Q4_1,
            6 => Dtype::Q5_0,
            7 => Dtype::Q5_1,
            8 => Dtype::Q8_0,
            12 => Dtype::Q4K,
            13 => Dtype::Q5K,
            14 => Dtype::Q6K,
            30 => Dtype::BF16,
            _ => return None,
        })
    }

    /// (elementos por bloco, bytes por bloco)
    pub fn block(self) -> (usize, usize) {
        match self {
            Dtype::F32 => (1, 4),
            Dtype::F16 | Dtype::BF16 => (1, 2),
            Dtype::Q4_0 => (32, 18),
            Dtype::Q4_1 => (32, 20),
            Dtype::Q5_0 => (32, 22),
            Dtype::Q5_1 => (32, 24),
            Dtype::Q8_0 => (32, 34),
            Dtype::Q4K => (256, 144),
            Dtype::Q5K => (256, 176),
            Dtype::Q6K => (256, 210),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dims: Vec<usize>,
    pub ggml_type: u32,
    /// offset absoluto dentro do arquivo
    pub offset: usize,
}

pub struct Gguf {
    pub mmap: Mmap,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).filter(|e| *e <= self.b.len()).ok_or("GGUF truncado ou corrompido")?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        if n > 1 << 28 {
            return Err("string enorme no GGUF".into());
        }
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    fn value(&mut self, ty: u32, depth: u8) -> Result<Value, String> {
        Ok(match ty {
            0 => Value::U(self.u8()? as u64),
            1 => Value::I(self.u8()? as i8 as i64),
            2 => Value::U(self.u16()? as u64),
            3 => Value::I(self.u16()? as i16 as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_bits(self.u32()?) as f64),
            7 => Value::B(self.u8()? != 0),
            8 => Value::S(self.string()?),
            9 => {
                if depth > 2 {
                    return Err("arrays aninhados demais no GGUF".into());
                }
                let et = self.u32()?;
                let n = self.u64()? as usize;
                if n > 50_000_000 {
                    return Err("array enorme no GGUF".into());
                }
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::A(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_bits(self.u64()?)),
            _ => return Err(format!("tipo de metadado GGUF desconhecido: {ty}")),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let file = File::open(path).map_err(|e| crate::errcode::coded("err_open_model", &[&e.to_string()], format!("could not open the model: {e}")))?;
        // SAFETY: o arquivo é só lido; se outro processo o truncar durante o uso, o SO pode
        // gerar um erro de acesso — risco aceito (mesmo comportamento do llama.cpp).
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap falhou: {e}"))?;

        let (meta, tensors) = {
            let mut r = Rd { b: &mmap, p: 0 };
            if r.take(4)? != b"GGUF" {
                return Err(crate::errcode::coded("err_not_gguf_file", &[], "not a GGUF file"));
            }
            let version = r.u32()?;
            if !(2..=3).contains(&version) {
                return Err(crate::errcode::coded("err_gguf_version", &[&version.to_string()], format!("unsupported GGUF version: {version}")));
            }
            let n_tensors = r.u64()? as usize;
            let n_kv = r.u64()? as usize;
            if n_tensors > 1_000_000 || n_kv > 1_000_000 {
                return Err("cabeçalho GGUF inválido".into());
            }

            let mut meta = HashMap::new();
            for _ in 0..n_kv {
                let key = r.string()?;
                let ty = r.u32()?;
                meta.insert(key, r.value(ty, 0)?);
            }

            let mut infos = Vec::with_capacity(n_tensors);
            for _ in 0..n_tensors {
                let name = r.string()?;
                let nd = r.u32()? as usize;
                if nd == 0 || nd > 4 {
                    return Err("tensor com dimensões inválidas".into());
                }
                let mut dims = Vec::with_capacity(nd);
                for _ in 0..nd {
                    dims.push(r.u64()? as usize);
                }
                let ggml_type = r.u32()?;
                let off = r.u64()? as usize;
                infos.push((name, dims, ggml_type, off));
            }

            let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32).max(1) as usize;
            let data_start = r.p.div_ceil(align) * align;
            let tensors = infos
                .into_iter()
                .map(|(n, dims, ggml_type, off)| (n, TensorInfo { dims, ggml_type, offset: data_start + off }))
                .collect::<HashMap<_, _>>();
            (meta, tensors)
        };

        Ok(Gguf { mmap, meta, tensors })
    }

    pub fn u64(&self, key: &str) -> Result<u64, String> {
        self.meta.get(key).and_then(Value::as_u64).ok_or_else(|| format!("metadado ausente: {key}"))
    }
    pub fn f64(&self, key: &str) -> Option<f64> {
        self.meta.get(key).and_then(Value::as_f64)
    }

    pub fn dtype_of(info: &TensorInfo) -> Option<Dtype> {
        Dtype::from_ggml(info.ggml_type)
    }
}
