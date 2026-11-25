//! Anexos do chat (texto, código e PDF).
//!
//! Os arquivos NUNCA são guardados nem enviados para fora: o navegador lê texto/código sozinho e manda
//! só os bytes de PDF para o servidor local (loopback), que devolve o texto extraído. Nada vai para o
//! histórico além do nome, tamanho e tipo (ver `history::AttachMeta`).
//!
//! O conteúdo de um anexo é TEXTO NÃO CONFIÁVEL: a interface o põe na mensagem como dado, dentro de um
//! bloco cercado ("### Arquivo: nome"), e nunca como instrução do sistema.

use serde::Serialize;

use crate::engine::ChatMsg;

/// Limite por arquivo (igual ao da interface).
pub const MAX_FILE_BYTES: usize = 5 * 1024 * 1024;
/// Teto de caracteres do texto extraído de um PDF (acima disso o texto é cortado e marcado).
pub const MAX_PDF_TEXT_CHARS: usize = 2_000_000;
/// Teto do corpo de `/api/chat` (anexos de texto grandes; o limite padrão do axum é 2 MB).
pub const CHAT_BODY_MAX: usize = 16 * 1024 * 1024;

/// Estimativa de tokens: caracteres / 3,5 (arredondado para cima). Heurística, igual à da interface.
pub fn estimate_tokens(s: &str) -> usize {
    (s.chars().count() * 2 + 6) / 7
}

/// Extrai o texto de um PDF (bytes já validados: tamanho e cabeçalho `%PDF-`). Roda num PROCESSO FILHO do próprio
/// binário (`crate::pdf`): pânico, aborto, laço infinito ou saída gigante do leitor não derrubam o servidor.
pub fn extract_pdf_text(bytes: &[u8]) -> Result<String, String> {
    crate::pdf::extract_via_child(bytes)
}

#[derive(Serialize, Debug, PartialEq)]
pub struct PdfText {
    pub text: String,
    pub chars: usize,
    pub truncated: bool,
}

/// Valida o resultado da extração: vazio = PDF sem texto (digitalizado, sem OCR); texto enorme é cortado.
pub fn finish_pdf_text(r: Result<String, String>) -> Result<PdfText, String> {
    let text = match r {
        Ok(t) => t,
        Err(e) if crate::pdf::is_final_message(&e) => return Err(e),
        Err(e) => return Err(crate::errcode::coded("err_attach_pdf", &[&e], format!("Could not read the PDF: {e}"))),
    };
    if text.trim().is_empty() {
        return Err(crate::errcode::coded("err_attach_nopdftext", &[], "The PDF has no selectable text (it may be scanned); there is no OCR in this version."));
    }
    let total = text.chars().count();
    if total > MAX_PDF_TEXT_CHARS {
        let cut: String = text.chars().take(MAX_PDF_TEXT_CHARS).collect();
        return Ok(PdfText { chars: MAX_PDF_TEXT_CHARS, text: cut, truncated: true });
    }
    Ok(PdfText { text, chars: total, truncated: false })
}

/// Caminho completo de um PDF recebido: limites, cabeçalho e extração.
pub fn process_pdf(bytes: &[u8]) -> Result<PdfText, String> {
    process_pdf_with(bytes, extract_pdf_text)
}

/// Igual a `process_pdf`, com o extrator injetável (os testes unitários não podem reexecutar o binário de teste).
pub fn process_pdf_with(bytes: &[u8], extract: impl FnOnce(&[u8]) -> Result<String, String>) -> Result<PdfText, String> {
    if bytes.is_empty() {
        return Err(crate::errcode::coded("err_attach_empty", &[], "The file is empty."));
    }
    if bytes.len() > MAX_FILE_BYTES {
        return Err(too_big());
    }
    // alguns PDFs têm lixo antes do cabeçalho; o padrão tolera até 1024 bytes
    let head = &bytes[..bytes.len().min(1024)];
    if !head.windows(5).any(|w| w == b"%PDF-") {
        return Err(crate::errcode::coded("err_attach_notpdf", &[], "The file does not look like a PDF."));
    }
    finish_pdf_text(extract(bytes))
}

pub fn too_big() -> String {
    let mb = (MAX_FILE_BYTES / (1024 * 1024)).to_string();
    crate::errcode::coded("err_attach_big", &[&mb], format!("The file is too large (limit of {mb} MB)."))
}

/// Maior mensagem (a última, com os blocos de anexo) aceita para um contexto de `ctx_size` tokens: 90% do contexto.
/// Os 10% restantes ficam para a resposta e para o modelo de chat (marcadores de papel); não usamos `max_tokens`
/// porque a interface não o envia e precisa do mesmo número para bloquear (interface e servidor usam esta regra).
pub fn message_limit(ctx_size: u32) -> usize {
    ctx_size as usize - ctx_size as usize / 10
}

/// Rede de segurança do servidor: recusa o pedido só se a ÚLTIMA mensagem (a que não dá para cortar) passa do limite.
/// As mensagens antigas NÃO contam: os motores já descartam as mais velhas quando a conversa passa do contexto
/// (o motor próprio em `native::generate`), então conversas longas sem anexo continuam funcionando.
pub fn check_context(messages: &[ChatMsg], ctx_size: u32) -> Result<(), String> {
    let Some(last) = messages.last() else { return Ok(()) };
    let used = estimate_tokens(&last.content);
    let limit = message_limit(ctx_size);
    if used > limit {
        return Err(crate::errcode::coded(
            "err_ctx_over",
            &[&used.to_string(), &limit.to_string()],
            format!("The message is too large for the configured context (about {used} tokens; the limit is {limit}). Remove or shorten the attachments, or raise the context in Settings."),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(s: &str) -> ChatMsg {
        ChatMsg { role: "user".into(), content: s.into() }
    }

    #[test]
    fn estimativa_de_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcd"), 2); // 4 / 3,5 = 1,14 arredonda para 2
        assert_eq!(estimate_tokens(&"a".repeat(7)), 2); // 7 / 3,5
        assert_eq!(estimate_tokens(&"a".repeat(8)), 3); // arredonda para cima
        assert_eq!(estimate_tokens(&"é".repeat(7)), 2); // conta caracteres, não bytes
    }

    #[test]
    fn pdf_limites_e_cabecalho() {
        let ex = |_: &[u8]| Ok("texto".to_string());
        assert_eq!(crate::errcode::english(&process_pdf_with(b"", ex).unwrap_err()), "The file is empty.");
        let big = vec![b'a'; MAX_FILE_BYTES + 1];
        assert_eq!(process_pdf_with(&big, ex).unwrap_err(), too_big());
        assert_eq!(crate::errcode::english(&process_pdf_with(b"isto nao e pdf", ex).unwrap_err()), "The file does not look like a PDF.");
        // exatamente no limite passa da checagem de tamanho (e cai na extração)
        let mut ok = b"%PDF-1.4\n".to_vec();
        ok.resize(MAX_FILE_BYTES, b' ');
        assert_eq!(process_pdf_with(&ok, ex).unwrap().text, "texto");
    }

    #[test]
    fn pdf_texto_vazio_e_corte() {
        let e = finish_pdf_text(Ok("  \n\t ".into())).unwrap_err();
        assert!(crate::errcode::english(&e).contains("no selectable text"), "{e}");
        let ok = finish_pdf_text(Ok("olá mundo".into())).unwrap();
        assert_eq!(ok, PdfText { text: "olá mundo".into(), chars: 9, truncated: false });
        let big = "é".repeat(MAX_PDF_TEXT_CHARS + 10);
        let cut = finish_pdf_text(Ok(big)).unwrap();
        assert!(cut.truncated);
        assert_eq!(cut.chars, MAX_PDF_TEXT_CHARS);
        assert_eq!(cut.text.chars().count(), MAX_PDF_TEXT_CHARS);
        let e = finish_pdf_text(Err("xref quebrada".into())).unwrap_err();
        assert_eq!(crate::errcode::english(&e), "Could not read the PDF: xref quebrada");
        assert_eq!(crate::errcode::parse(&e).unwrap().1, vec!["xref quebrada"]);
        // mensagens finais do leitor isolado passam sem o prefixo
        for m in [crate::pdf::msg_encrypted(), crate::pdf::msg_crash(), crate::pdf::msg_pages()] {
            assert_eq!(finish_pdf_text(Err(m.clone())).unwrap_err(), m);
        }
    }

    #[test]
    fn contexto_so_a_ultima_mensagem_conta() {
        assert_eq!(message_limit(1000), 900);
        // 3150 caracteres = 900 tokens estimados
        let m = [msg(&"a".repeat(3150))];
        assert!(check_context(&m, 1000).is_ok()); // no limite exato passa
        assert!(check_context(&[msg(&"a".repeat(3151))], 1000).is_err());
        let e = check_context(&m, 512).unwrap_err();
        let en = crate::errcode::english(&e);
        assert!(en.contains("about 900 tokens") && en.contains("limit is 461"), "{e}");
        // conversa longa com última mensagem curta: passa (os motores cortam as antigas)
        let longa = [msg(&"a".repeat(20000)), msg("ok"), msg(&"b".repeat(20000)), msg("e agora?")];
        assert!(check_context(&longa, 1000).is_ok());
        // o anexo grande está na última mensagem: recusa
        let com_anexo = [msg("oi"), msg(&"c".repeat(20000))];
        assert!(check_context(&com_anexo, 1000).is_err());
        assert!(check_context(&[], 1000).is_ok());
    }
}
