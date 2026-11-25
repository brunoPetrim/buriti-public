//! Stable error codes for the web UI (D83).
//!
//! The UI used to translate server messages by matching their Portuguese text with regular expressions: change one word
//! of a message and its translation silently disappeared. Now an error that must reach the screen carries a stable
//! `code` (the key of the web translations, e.g. `err_sha`, translated into the 34 languages), the positional `args` that
//! fill `{0}`, `{1}`, ... in that text, and `en`, the English message every other client sees and the UI falls back to.
//!
//! The error stays a `String` (the engine, the downloads and the settings all pass `Result<_, String>` around): `coded`
//! returns a small JSON object as text, and the UI's `tm()` recognises it. An error that wraps another one (`"cannot open
//! the model: {inner}"`) puts the inner error in `args`; the UI translates arguments that are themselves coded.
//! Exception errors that are never shown on screen (the `/v1` API, logs, tests) stay plain English text.

use serde_json::{json, Value};

/// A translatable error. `code` is an `err_*` key present in every language of `web/i18n*.js` (`tools/check-errors.js`
/// checks it), `args` fill `{0}`, `{1}`, ... and `en` is the English message.
pub fn coded(code: &str, args: &[&str], en: impl AsRef<str>) -> String {
    json!({ "code": code, "args": args, "en": en.as_ref() }).to_string()
}

/// Like `coded`, with arguments that are already owned strings (inner errors, paths, numbers).
pub fn coded_owned(code: &str, args: Vec<String>, en: impl AsRef<str>) -> String {
    json!({ "code": code, "args": args, "en": en.as_ref() }).to_string()
}

/// The English text of an error: the `en` field of a coded one, or the string itself. For logs, for messages built
/// around another error, and for tests that look for words in a message.
pub fn english(s: &str) -> String {
    match parse(s) {
        Some((_, _, en)) => en,
        None => s.to_string(),
    }
}

/// `(code, args, en)` of a coded error.
pub fn parse(s: &str) -> Option<(String, Vec<String>, String)> {
    let t = s.trim_start();
    if !t.starts_with('{') {
        return None;
    }
    let v: Value = serde_json::from_str(t).ok()?;
    let code = v.get("code")?.as_str()?.to_string();
    let en = v.get("en")?.as_str()?.to_string();
    let args = v.get("args")?.as_array()?.iter().map(|a| a.as_str().map(str::to_string).unwrap_or_else(|| a.to_string())).collect();
    Some((code, args, en))
}

/// The code of a coded error, if it is one.
pub fn code_of(s: &str) -> Option<String> {
    parse(s).map(|(c, _, _)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let e = coded("err_sha", &["abc", "def"], "SHA-256 does not match (expected abc, got def). File discarded.");
        let (code, args, en) = parse(&e).unwrap();
        assert_eq!(code, "err_sha");
        assert_eq!(args, vec!["abc", "def"]);
        assert!(en.starts_with("SHA-256 does not match"));
        assert_eq!(code_of(&e).as_deref(), Some("err_sha"));
    }

    #[test]
    fn plain_text_is_not_coded() {
        assert!(parse("some plain error").is_none());
        assert!(parse("{not json").is_none());
        assert!(parse(r#"{"other": 1}"#).is_none());
        assert_eq!(english("plain"), "plain");
    }

    #[test]
    fn arguments_with_quotes_and_unicode_survive() {
        let e = coded("err_model_missing", &["C:\\modelos\\quote\"d é.gguf"], "Model not found");
        let (_, args, _) = parse(&e).unwrap();
        assert_eq!(args, vec!["C:\\modelos\\quote\"d é.gguf"]);
    }

    #[test]
    fn an_error_can_wrap_another() {
        let inner = coded("err_not_gguf_file", &[], "not a GGUF file");
        let outer = coded_owned("err_open_model", vec![inner.clone()], format!("cannot open the model: {}", english(&inner)));
        let (code, args, en) = parse(&outer).unwrap();
        assert_eq!(code, "err_open_model");
        assert_eq!(code_of(&args[0]).as_deref(), Some("err_not_gguf_file"));
        assert_eq!(en, "cannot open the model: not a GGUF file");
    }
}
