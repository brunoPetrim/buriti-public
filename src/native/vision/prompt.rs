//! Prompt de chat com imagem para o motor próprio (D58, M6): marcador `<__media__>` -> ids do bloco de imagem (fatias e visão
//! geral, `Tokenizer::idefics3_image_ids`), posições dos vetores de imagem, chave (hash) da imagem para o KV cache e os
//! embeddings (torre) calculados sob demanda.

use sha2::{Digest, Sha256};

use super::{
    mmproj::Mmproj,
    tower::{encode, normalize_u8, EncodeOpts},
};
use crate::{
    engine::ChatMsg,
    image::{ChatImage, VisionCaps},
    native::tokenizer::{Tokenizer, IMAGE_MARKER},
};

pub struct ImagePrompt {
    /// posição (no prompt) de cada vetor de imagem, na ordem dos blocos
    pub slots: Vec<usize>,
    /// identifica o conteúdo das imagens; 0 nunca é uma chave válida
    pub key: u64,
    images: Vec<ChatImage>,
}

/// Capacidade que o motor anuncia à interface para este mmproj.
pub fn caps(mm: &Mmproj, max_grid: usize) -> VisionCaps {
    VisionCaps { tile: mm.hp.image_size, per_tile: mm.hp.n_tokens(), max_grid }
}

/// Chave (64 bits do SHA-256) do conteúdo: grade e bytes de todos os blocos, em ordem.
pub fn image_key(images: &[ChatImage]) -> u64 {
    let mut h = Sha256::new();
    for im in images {
        h.update([im.grid.0 as u8, im.grid.1 as u8, im.msg as u8]);
        for t in &im.tiles {
            h.update(t);
        }
    }
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().unwrap()) | 1
}

/// Monta os ids do prompt. A imagem vai ANTES do texto da mensagem (como a interface do llama.cpp); qualquer `<__media__>`
/// digitado pelo usuário é removido (não pode criar posições de imagem sem embedding).
pub fn build_prompt(tok: &Tokenizer, per_tile: usize, msgs: &[ChatMsg], images: &[ChatImage]) -> Result<(Vec<u32>, ImagePrompt), String> {
    let slot = tok.image_slot_id().ok_or("este modelo não tem formato de imagem")?;
    if images.iter().any(|im| im.msg >= msgs.len()) {
        return Err("the image points to a message that does not exist".into());
    }
    let mut order: Vec<&ChatImage> = images.iter().collect();
    order.sort_by_key(|im| im.msg); // o k-ésimo marcador do prompt é a k-ésima imagem na ordem das mensagens
    let clean: Vec<ChatMsg> = msgs
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let c = m.content.replace(IMAGE_MARKER, "");
            ChatMsg { role: m.role.clone(), content: if order.iter().any(|im| im.msg == i) { format!("{IMAGE_MARKER}{c}") } else { c } }
        })
        .collect();
    let expand = |k: usize| -> Vec<u32> {
        let im = order[k];
        tok.idefics3_image_ids(im.grid.0, im.grid.1, per_tile).unwrap_or_default()
    };
    let ids = tok.chat_prompt_images(&clean, Some(&expand));
    let slots: Vec<usize> = ids.iter().enumerate().filter(|(_, &t)| t == slot).map(|(i, _)| i).collect();
    let want: usize = order.iter().map(|im| im.n_tiles() * per_tile).sum();
    if slots.len() != want {
        return Err(format!("the prompt has {} image positions, expected {want}", slots.len()));
    }
    Ok((ids, ImagePrompt { slots, key: image_key(images), images: order.into_iter().cloned().collect() }))
}

impl ImagePrompt {
    /// Embeddings de todos os vetores de imagem (`slots.len() × proj_dim`), bloco a bloco, na ordem das posições.
    pub fn embeddings(&self, mm: &Mmproj) -> Result<Vec<f32>, String> {
        let hp = &mm.hp;
        let mut out = Vec::with_capacity(self.slots.len() * hp.proj_dim);
        for im in &self.images {
            for t in &im.tiles {
                out.extend(encode(mm, &normalize_u8(t, hp.mean, hp.std), EncodeOpts::default(), None)?);
            }
        }
        Ok(out)
    }
}
