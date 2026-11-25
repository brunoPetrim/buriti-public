//! Histórico de conversas criptografado em disco.
//!
//! Cada conversa é um arquivo `history/<id>.chat`: `"BRT1" + nonce(12) + ChaCha20-Poly1305(json)`.
//! O id da conversa entra como dado associado (AAD), então um arquivo renomeado não decifra.
//! A chave (32 bytes aleatórios) é gerada no primeiro uso e guardada no perfil do usuário.

use std::path::{Path, PathBuf};

use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305, Nonce,
};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"BRT1";

/// Metadados de um arquivo anexado a uma mensagem do usuário. O CONTEÚDO do arquivo nunca é guardado
/// (nem o caminho: o navegador não o expõe). Campos desconhecidos (como um `text` enviado por engano)
/// são descartados na leitura.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AttachMeta {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    /// "text" | "code" | "pdf" | "image"
    #[serde(default)]
    pub kind: String,
}

pub const MAX_ATTACH_PER_MSG: usize = 20;
pub const MAX_ATTACH_NAME: usize = 200;

#[derive(Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Ausente nos arquivos antigos (`default`) e omitido quando vazio, para manter o formato antigo.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<AttachMeta>,
}

impl Message {
    /// Limita a quantidade e o tamanho dos metadados e normaliza o tipo.
    pub fn sanitize(&mut self) {
        self.attachments.truncate(MAX_ATTACH_PER_MSG);
        for a in &mut self.attachments {
            a.name = a.name.chars().filter(|c| !c.is_control()).take(MAX_ATTACH_NAME).collect();
            if !matches!(a.kind.as_str(), "text" | "code" | "pdf" | "image") {
                a.kind = "text".into();
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Chat {
    pub id: String,
    pub title: String,
    pub created: u64,
    pub updated: u64,
    pub messages: Vec<Message>,
}

#[derive(Serialize)]
pub struct ChatMeta {
    pub id: String,
    pub title: String,
    pub updated: u64,
}

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub struct HistoryStore {
    dir: PathBuf,
    cipher: ChaCha20Poly1305,
}

impl HistoryStore {
    pub fn open(dir: PathBuf, key_file: &Path) -> Result<Self, String> {
        let key = load_or_create_key(key_file)?;
        let cipher = ChaCha20Poly1305::new_from_slice(&key).map_err(|e| e.to_string())?;
        Ok(Self { dir, cipher })
    }

    pub fn new_id() -> String {
        let mut b = [0u8; 8];
        getrandom::fill(&mut b).expect("aleatoriedade");
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// Só aceita ids de 16 hex (impede path traversal).
    fn path(&self, id: &str) -> Result<PathBuf, String> {
        if id.len() == 16 && id.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
            Ok(self.dir.join(format!("{id}.chat")))
        } else {
            Err(crate::errcode::coded("err_chat_id", &[], "invalid conversation id"))
        }
    }

    pub fn save(&self, chat: &Chat) -> Result<(), String> {
        let path = self.path(&chat.id)?;
        let plain = serde_json::to_vec(chat).map_err(|e| e.to_string())?;
        let mut nonce = [0u8; 12];
        getrandom::fill(&mut nonce).map_err(|e| e.to_string())?;
        let ct = self
            .cipher
            .encrypt(Nonce::from_slice(&nonce), Payload { msg: &plain, aad: chat.id.as_bytes() })
            .map_err(|_| crate::errcode::coded("err_encrypt", &[], "failed to encrypt"))?;
        let mut out = Vec::with_capacity(4 + 12 + ct.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ct);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, out).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
    }

    pub fn load(&self, id: &str) -> Result<Chat, String> {
        let data = std::fs::read(self.path(id)?).map_err(|e| e.to_string())?;
        if data.len() < 4 + 12 + 16 || &data[..4] != MAGIC {
            return Err(crate::errcode::coded("err_chat_file", &[], "invalid conversation file"));
        }
        let plain = self
            .cipher
            .decrypt(Nonce::from_slice(&data[4..16]), Payload { msg: &data[16..], aad: id.as_bytes() })
            .map_err(|_| crate::errcode::coded("err_decrypt", &[], "could not decrypt (different key or the file was changed)"))?;
        serde_json::from_slice(&plain).map_err(|e| e.to_string())
    }

    pub fn list(&self) -> Vec<ChatMeta> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if let Some(id) = name.strip_suffix(".chat") {
                    if let Ok(c) = self.load(id) {
                        out.push(ChatMeta { id: c.id, title: c.title, updated: c.updated });
                    }
                }
            }
        }
        out.sort_by(|a, b| b.updated.cmp(&a.updated));
        out
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        std::fs::remove_file(self.path(id)?).map_err(|e| e.to_string())
    }
}

fn load_or_create_key(path: &Path) -> Result<[u8; 32], String> {
    if let Ok(bytes) = std::fs::read(path) {
        return <[u8; 32]>::try_from(bytes.as_slice()).map_err(|_| format!("chave inválida em {}", path.display()));
    }
    let mut key = [0u8; 32];
    getrandom::fill(&mut key).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, key).map_err(|e| format!("não foi possível gravar a chave: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(tag: &str) -> (HistoryStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("buriti-hist-test-{tag}-{}", HistoryStore::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store = HistoryStore::open(dir.clone(), &dir.join("key.bin")).unwrap();
        (store, dir)
    }

    #[test]
    fn arquivo_antigo_sem_anexos_ainda_carrega() {
        let old = r#"{"id":"0123456789abcdef","title":"t","created":1,"updated":2,"messages":[{"role":"user","content":"oi"}]}"#;
        let c: Chat = serde_json::from_str(old).unwrap();
        assert!(c.messages[0].attachments.is_empty());
        // sem anexos o JSON gravado continua idêntico ao formato antigo
        let out = serde_json::to_string(&c.messages[0]).unwrap();
        assert_eq!(out, r#"{"role":"user","content":"oi"}"#);
    }

    #[test]
    fn anexos_ida_e_volta_sem_conteudo() {
        let raw = r#"{"role":"user","content":"veja","attachments":[{"name":"a.rs","size":12,"kind":"code","text":"SEGREDO-NO-ARQUIVO"}]}"#;
        let m: Message = serde_json::from_str(raw).unwrap();
        let out = serde_json::to_string(&m).unwrap();
        assert!(!out.contains("SEGREDO"), "conteúdo vazou: {out}");
        assert!(out.contains("a.rs") && out.contains("\"size\":12"));
        // pelo cofre criptografado também
        let (store, dir) = temp_store("anexo");
        let chat = Chat { id: HistoryStore::new_id(), title: "x".into(), created: 1, updated: 1, messages: vec![m] };
        store.save(&chat).unwrap();
        let back = store.load(&chat.id).unwrap();
        assert_eq!(back.messages[0].attachments, vec![AttachMeta { name: "a.rs".into(), size: 12, kind: "code".into() }]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sanitize_limita_e_normaliza() {
        let many = (0..30).map(|i| AttachMeta { name: format!("f{i}\n"), size: 1, kind: "exe".into() }).collect();
        let mut m = Message { role: "user".into(), content: String::new(), attachments: many };
        m.attachments[0].name = "x".repeat(500);
        m.sanitize();
        assert_eq!(m.attachments.len(), MAX_ATTACH_PER_MSG);
        assert_eq!(m.attachments[0].name.chars().count(), MAX_ATTACH_NAME);
        assert_eq!(m.attachments[1].name, "f1"); // caractere de controle removido
        assert!(m.attachments.iter().all(|a| a.kind == "text"));
        let mut ok = Message { role: "user".into(), content: String::new(), attachments: vec![AttachMeta { name: "a.pdf".into(), size: 1, kind: "pdf".into() }] };
        ok.sanitize();
        assert_eq!(ok.attachments[0].kind, "pdf");
    }
}
