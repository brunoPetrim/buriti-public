//! Visão no motor próprio (protótipo, D50): leitor do `mmproj`, torre SigLIP em f32 e conector
//! Idefics3 (SmolVLM). **Ainda não está ligado ao modelo de linguagem nem ao servidor**: nada daqui é
//! chamado pelo caminho de conversa (por isso o `allow(dead_code)`). O desenho completo, o plano de
//! validação contra o llama.cpp e o que falta estão em `docs/VISAO-MOTOR-PROPRIO.md`.
//!
//! Contrato com o navegador: a decodificação, o redimensionamento e a conversão para RGB acontecem no
//! navegador (canvas); o Rust recebe `image_size × image_size × 3` bytes (RGB intercalado, 8 bits) e só
//! normaliza (`tower::normalize_u8`). Nenhum decodificador de imagem existe no servidor.
#![allow(dead_code)]

pub mod diag;
pub mod mmproj;
pub mod ops;
pub mod prompt;
pub mod tower;

#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use mmproj::Mmproj;
#[allow(unused_imports)]
pub use tower::{encode, normalize_u8, Bug, EncodeOpts};
