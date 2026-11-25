//! Imagens do chat (D58, M6): contrato entre o navegador e o motor próprio.
//!
//! O navegador (canvas) decodifica e redimensiona; o servidor NUNCA decodifica PNG/JPEG/WebP. Em `POST /api/chat`, uma
//! mensagem de usuário pode trazer
//! `"images": [{"grid": [gx, gy], "tiles": ["<base64 de RGB 8 bits, tile×tile×3 bytes>", ...]}]`
//! com os blocos NESTA ORDEM: as fatias linha a linha (`gx·gy` blocos; `grid` [0,0] = sem fatias) e, por último, a visão
//! geral. Cada bloco tem exatamente `tile`×`tile` pixels (o tamanho que o modelo declara em `VisionCaps`, 512 no
//! SmolVLM). Nada disso é guardado: só o nome, o tamanho e o tipo "image" vão para o histórico (`history::AttachMeta`).
//! O conteúdo é DADO não confiável: limites de contagem e de tamanho são checados ANTES de decodificar.

use serde_json::Value;

/// Limite do arquivo ORIGINAL escolhido pelo usuário (checado no navegador, como nos demais anexos).
#[allow(dead_code)] // checado no navegador; fica aqui como contrato
pub const MAX_FILE_BYTES: usize = 5 * 1024 * 1024;
/// Uma imagem por pedido nesta versão (o KV cache identifica a imagem por um só hash; ver `native::vision::prompt`).
pub const MAX_IMAGES_PER_REQUEST: usize = 1;

/// O que o motor ativo sabe fazer com imagens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionCaps {
    /// lado de cada bloco, em pixels (512 no SmolVLM)
    pub tile: usize,
    /// vetores de imagem por bloco (64 no SmolVLM)
    pub per_tile: usize,
    /// maior grade de fatias por lado (a visão geral vem a mais)
    pub max_grid: usize,
}

/// Imagem de uma mensagem, já validada: blocos RGB (fatias linha a linha e, por último, a visão geral).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatImage {
    /// índice da mensagem (em `ChatRequest::messages`) a que a imagem pertence
    pub msg: usize,
    pub grid: (usize, usize),
    pub tiles: Vec<Vec<u8>>,
}

impl ChatImage {
    pub fn n_tiles(&self) -> usize {
        self.tiles.len()
    }
}

fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Base64 padrão (RFC 4648), estrito: sem espaços, com `=` de preenchimento e bits finais zerados.
/// `expect_len` é o tamanho exato esperado da saída; o tamanho do texto é conferido antes de alocar.
pub fn b64_decode_exact(s: &str, expect_len: usize) -> Result<Vec<u8>, String> {
    let b = s.as_bytes();
    if b.len() != expect_len.div_ceil(3) * 4 {
        return Err("tamanho do base64 não confere".into());
    }
    let mut out = Vec::with_capacity(expect_len);
    for (ci, chunk) in b.chunks(4).enumerate() {
        let last = ci == b.len() / 4 - 1;
        let pad = chunk.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 || (pad > 0 && !last) {
            return Err("preenchimento inválido no base64".into());
        }
        let mut v = [0u8; 4];
        for (i, &c) in chunk[..4 - pad].iter().enumerate() {
            v[i] = b64_val(c).ok_or("caractere inválido no base64")?;
        }
        let n = ((v[0] as u32) << 18) | ((v[1] as u32) << 12) | ((v[2] as u32) << 6) | v[3] as u32;
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        let take = 3 - pad;
        // bits que sobram no último grupo têm de ser zero (forma canônica)
        if pad > 0 && bytes[take..].iter().any(|&x| x != 0) {
            return Err("bits finais do base64 não são zero".into());
        }
        out.extend_from_slice(&bytes[..take]);
    }
    if out.len() != expect_len {
        return Err("tamanho decodificado não confere".into());
    }
    Ok(out)
}

/// Lê `messages[i].images` do corpo de `/api/chat`. `caps` = o que o motor ativo aceita (`None` = não aceita imagem).
/// Sem nenhuma imagem no corpo devolve lista vazia (e não exige `caps`).
pub fn parse_request_images(messages: &Value, caps: Option<VisionCaps>) -> Result<Vec<ChatImage>, String> {
    let Some(arr) = messages.as_array() else { return Ok(vec![]) };
    let mut out: Vec<ChatImage> = Vec::new();
    for (mi, m) in arr.iter().enumerate() {
        let Some(imgs) = m.get("images") else { continue };
        let Some(imgs) = imgs.as_array() else { return Err(crate::errcode::coded("err_img_invalid", &[], "invalid images: a list was expected")) };
        if imgs.is_empty() {
            continue;
        }
        let Some(caps) = caps else { return Err(crate::errcode::coded("err_img_novision", &[], "The active model does not accept images.")) };
        if m.get("role").and_then(Value::as_str) != Some("user") {
            return Err(crate::errcode::coded("err_img_invalid", &[], "images are only accepted in user messages"));
        }
        for im in imgs {
            if out.len() >= MAX_IMAGES_PER_REQUEST {
                return Err(crate::errcode::coded("err_img_one", &[], format!("Only {MAX_IMAGES_PER_REQUEST} image per request is allowed.")));
            }
            let g = im.get("grid").and_then(Value::as_array).ok_or_else(|| crate::errcode::coded("err_img_invalid", &[], "the image has no `grid`"))?;
            let (gx, gy) = match (g.first().and_then(Value::as_u64), g.get(1).and_then(Value::as_u64), g.len()) {
                (Some(x), Some(y), 2) => (x as usize, y as usize),
                _ => return Err(crate::errcode::coded("err_img_invalid", &[], "invalid `grid`")),
            };
            if gx > caps.max_grid || gy > caps.max_grid || (gx == 0) != (gy == 0) {
                return Err(crate::errcode::coded("err_img_invalid", &[], format!("invalid tile grid ({gx}x{gy}); the maximum is {0}x{0}", caps.max_grid)));
            }
            let want = if gx * gy > 0 { gx * gy + 1 } else { 1 };
            let tiles = im.get("tiles").and_then(Value::as_array).ok_or_else(|| crate::errcode::coded("err_img_invalid", &[], "the image has no `tiles`"))?;
            if tiles.len() != want {
                return Err(crate::errcode::coded("err_img_invalid", &[], format!("the {gx}x{gy} grid needs {want} tiles, got {}", tiles.len())));
            }
            let size = caps.tile * caps.tile * 3;
            let mut dec = Vec::with_capacity(want);
            for t in tiles {
                let s = t.as_str().ok_or_else(|| crate::errcode::coded("err_img_invalid", &[], "an image tile is not base64 text"))?;
                dec.push(b64_decode_exact(s, size).map_err(|e| crate::errcode::coded("err_img_invalid", &[], format!("invalid image tile: {e}")))?);
            }
            out.push(ChatImage { msg: mi, grid: (gx, gy), tiles: dec });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn b64(data: &[u8]) -> String {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for c in data.chunks(3) {
            let n = ((c[0] as u32) << 16) | ((*c.get(1).unwrap_or(&0) as u32) << 8) | *c.get(2).unwrap_or(&0) as u32;
            s.push(T[(n >> 18) as usize & 63] as char);
            s.push(T[(n >> 12) as usize & 63] as char);
            s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
            s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
        }
        s
    }

    #[test]
    fn base64_vectors_and_strictness() {
        // vetores da RFC 4648
        for (plain, enc) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")] {
            assert_eq!(b64_decode_exact(enc, plain.len()).unwrap(), plain.as_bytes(), "{enc}");
        }
        // ida e volta com todos os bytes
        let all: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        assert_eq!(b64_decode_exact(&b64(&all), all.len()).unwrap(), all);
        // estrito: tamanho errado, caractere inválido, espaço, preenchimento no meio, bits finais não zero
        assert!(b64_decode_exact("Zm9v", 4).is_err());
        assert!(b64_decode_exact("Zm9", 2).is_err());
        assert!(b64_decode_exact("Zm9*", 3).is_err());
        assert!(b64_decode_exact("Zm9 ", 3).is_err());
        assert!(b64_decode_exact("Zg==Zg==", 2).is_err());
        assert!(b64_decode_exact("Zh==", 1).is_err()); // 'h' deixa bits não zero
        assert!(b64_decode_exact("Zm8=", 3).is_err()); // saída de 2 bytes não é 3
    }

    fn caps() -> VisionCaps {
        VisionCaps { tile: 4, per_tile: 1, max_grid: 3 }
    }
    fn tile_b64() -> String {
        b64(&vec![7u8; 4 * 4 * 3])
    }

    #[test]
    fn parse_images_accepts_valid_and_rejects_the_rest() {
        let t = tile_b64();
        let ok = json!([{"role":"user","content":"x","images":[{"grid":[0,0],"tiles":[t]}]}]);
        let r = parse_request_images(&ok, Some(caps())).unwrap();
        assert_eq!((r.len(), r[0].msg, r[0].grid, r[0].tiles.len(), r[0].tiles[0].len()), (1, 0, (0, 0), 1, 48));
        let grid = json!([{"role":"assistant","content":"a"},{"role":"user","content":"x","images":[{"grid":[2,1],"tiles":[t,t,t]}]}]);
        let r = parse_request_images(&grid, Some(caps())).unwrap();
        assert_eq!((r[0].msg, r[0].n_tiles()), (1, 3));
        // sem imagem: vazio e não exige capacidade; com lista vazia também
        assert!(parse_request_images(&json!([{"role":"user","content":"x"}]), None).unwrap().is_empty());
        assert!(parse_request_images(&json!([{"role":"user","content":"x","images":[]}]), None).unwrap().is_empty());
        let en = |s: &str| crate::errcode::english(s);
        let bad = |v: Value, c: Option<VisionCaps>| en(&parse_request_images(&v, c).unwrap_err());
        // motor sem visão
        assert!(en(&bad(ok.clone(), None)).contains("does not accept images"));
        // papel errado
        assert!(bad(json!([{"role":"assistant","content":"x","images":[{"grid":[0,0],"tiles":[t]}]}]), Some(caps())).contains("user messages"));
        // contagem de blocos, grade, tamanho do bloco, texto não base64, imagens demais
        assert!(bad(json!([{"role":"user","content":"x","images":[{"grid":[2,1],"tiles":[t,t]}]}]), Some(caps())).contains("needs 3 tiles"));
        assert!(bad(json!([{"role":"user","content":"x","images":[{"grid":[4,1],"tiles":[t]}]}]), Some(caps())).contains("grid"));
        assert!(bad(json!([{"role":"user","content":"x","images":[{"grid":[1,0],"tiles":[t]}]}]), Some(caps())).contains("grid"));
        assert!(bad(json!([{"role":"user","content":"x","images":[{"grid":[0,0],"tiles":[b64(&[1u8; 45])]}]}]), Some(caps())).contains("invalid"));
        assert!(bad(json!([{"role":"user","content":"x","images":[{"grid":[0,0],"tiles":[5]}]}]), Some(caps())).contains("base64"));
        assert!(bad(json!([{"role":"user","content":"x","images":[{"grid":[0,0],"tiles":[t]},{"grid":[0,0],"tiles":[t]}]}]), Some(caps())).contains("per request"));
        assert!(bad(json!([{"role":"user","content":"x","images":{"grid":[0,0]}}]), Some(caps())).contains("list"));
        assert!(bad(json!([{"role":"user","content":"x","images":[{"tiles":[t]}]}]), Some(caps())).contains("grid"));
    }
}
