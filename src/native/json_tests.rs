//! Constrained JSON output with a real model (D82): the generated text must obey the schema, whatever the model would have said.
//! Uses the Qwen 2.5 0.5B of `buriti-data/models` (or `BURITI_TEST_MODEL`); skipped when the file is missing.

use super::*;
use crate::grammar::{Constraint, TokenTrie};
use serde_json::{json, Value};

fn model_path() -> Option<String> {
    let path = std::env::var("BURITI_TEST_MODEL").unwrap_or_else(|_| format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR")));
    if !std::path::Path::new(&path).exists() {
        eprintln!("test model missing, skipping: {path}");
        return None;
    }
    Some(path)
}

/// The text and the finish reason of a chat stream (`data: {json}` lines).
fn read_stream(rx: &mut mpsc::Receiver<Bytes>) -> (String, String) {
    let mut raw = String::new();
    while let Ok(b) = rx.try_recv() {
        raw.push_str(&String::from_utf8_lossy(&b));
    }
    let (mut text, mut finish) = (String::new(), String::new());
    for line in raw.lines() {
        let Some(j) = line.strip_prefix("data: ") else { continue };
        let Ok(v) = serde_json::from_str::<Value>(j) else { continue };
        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
            text.push_str(c);
        }
        if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
            finish = f.to_string();
        }
    }
    (text, finish)
}

struct Run {
    text: String,
    finish: String,
    seconds: f32,
}

/// One chat turn with (or without) the constraint of `schema` (`None` schema with `constrained` = any JSON object).
fn ask(model: &Model, trie: &Arc<TokenTrie>, schema: Option<&Value>, constrained: bool, user: &str, temperature: f32, seed: u64) -> Run {
    let grammar = Arc::new(match schema {
        Some(s) => crate::jsonschema::grammar_for_schema(s).expect("schema"),
        None => crate::jsonschema::grammar_json_object().expect("json_object"),
    });
    let constraint = constrained.then(|| Constraint::new(grammar.clone(), trie.clone(), model.tok.eos.clone()).expect("constraint"));
    let req = ChatRequest {
        messages: vec![ChatMsg { role: "user".into(), content: user.into() }],
        temperature: Some(temperature),
        max_tokens: Some(160),
        stop: vec![],
        sampling: SamplingOpts { seed: Some(seed), ..Default::default() },
        images: vec![],
        constraint: constrained.then(|| grammar.clone()),
    };
    let mut sess = model.new_session(1024);
    let (tx, mut rx) = mpsc::channel::<Bytes>(4096);
    let t = Instant::now();
    generate(model, None, None, &mut sess, &req, &tx, constraint);
    let seconds = t.elapsed().as_secs_f32();
    drop(tx);
    let (text, finish) = read_stream(&mut rx);
    Run { text, finish, seconds }
}

fn person_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
        },
        "required": ["name", "age"]
    })
}

fn trie_for(model: &Model) -> Arc<TokenTrie> {
    let t = Instant::now();
    let trie = Arc::new(TokenTrie::build(model.tok.vocab_size(), |id| model.tok.text_token_bytes(id)));
    eprintln!("token trie of {} tokens built in {:.2} s", model.tok.vocab_size(), t.elapsed().as_secs_f32());
    trie
}

const PROMPTS: [&str; 4] = [
    "Describe a person called Ana who is 31 years old and likes chess and rowing, as JSON.",
    "Invent a character: give a name and an age, and a couple of tags. Answer with JSON only.",
    "Who is the oldest of the three: Rui (70), Bia (65), Leo (12)? Return the person as JSON.",
    "Write the profile of a pirate named Barnaby, age 45, as JSON.",
];

/// Every constrained answer must be valid JSON that matches the schema and must end by itself (finish_reason "stop"), with a
/// greedy run and with sampling; the same prompts WITHOUT the constraint are only reported (a small model often breaks the format).
#[test]
fn constrained_answers_obey_the_schema() {
    let Some(path) = model_path() else { return };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("load model");
    let trie = trie_for(&model);
    let schema = person_schema();
    let grammar = crate::jsonschema::grammar_for_schema(&schema).unwrap();
    let (mut ok, mut total, mut free_ok) = (0, 0, 0);
    for (i, p) in PROMPTS.iter().enumerate() {
        for (temp, seed) in [(0.0f32, 1u64), (0.9, 100 + i as u64)] {
            let r = ask(&model, &trie, Some(&schema), true, p, temp, seed);
            total += 1;
            let v: Value = serde_json::from_str(&r.text).unwrap_or_else(|e| panic!("not JSON ({e}): {:?} [{}]", r.text, r.finish));
            assert!(grammar.matches(r.text.as_bytes()), "grammar rejects its own output: {:?}", r.text);
            assert!(v["name"].is_string() && (v["age"].is_i64() || v["age"].is_u64()), "schema violated: {}", r.text);
            if let Some(t) = v.get("tags") {
                assert!(t.as_array().is_some_and(|a| a.len() <= 3 && a.iter().all(Value::is_string)), "tags violated: {}", r.text);
            }
            assert_eq!(r.finish, "stop", "must end by itself: {:?}", r.text);
            ok += 1;
            eprintln!("constrained [{temp}] {:.1}s: {}", r.seconds, r.text.replace('\n', " "));
            let f = ask(&model, &trie, Some(&schema), false, p, temp, seed);
            let (n_c, n_f) = (model.tok.encode_plain(&r.text).len(), model.tok.encode_plain(&f.text).len());
            eprintln!("  speed: constrained {:.1} s for ~{n_c} tokens ({:.1} tok/s incl. prefill), free {:.1} s for ~{n_f} tokens ({:.1} tok/s incl. prefill)", r.seconds, n_c as f32 / r.seconds, f.seconds, n_f as f32 / f.seconds);
            if serde_json::from_str::<Value>(f.text.trim()).is_ok_and(|v| v["name"].is_string() && v["age"].is_number()) {
                free_ok += 1;
            }
        }
    }
    eprintln!("RESULT: constrained {ok}/{total} valid; the same prompts without the constraint: {free_ok}/{total} valid and exact (no fences, no extra text)");
}

#[test]
fn json_object_mode_gives_an_object() {
    let Some(path) = model_path() else { return };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("load model");
    let trie = trie_for(&model);
    for (i, p) in ["List three fruits as JSON.", "Say hello in JSON."].iter().enumerate() {
        let r = ask(&model, &trie, None, true, p, 0.7, 5 + i as u64);
        let v: Value = serde_json::from_str(&r.text).unwrap_or_else(|e| panic!("not JSON ({e}): {:?} [{}]", r.text, r.finish));
        assert!(v.is_object(), "{}", r.text);
        assert_eq!(r.finish, "stop", "{:?}", r.text);
    }
}

#[test]
fn enum_answers_are_one_of_the_values() {
    let Some(path) = model_path() else { return };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("load model");
    let trie = trie_for(&model);
    let schema = json!({"enum": ["red", "green", "blue"]});
    for (i, p) in ["Pick the colour of the sky.", "Pick the colour of grass.", "Pick a colour at random."].iter().enumerate() {
        let r = ask(&model, &trie, Some(&schema), true, p, 0.8, 11 + i as u64);
        assert!(["\"red\"", "\"green\"", "\"blue\""].contains(&r.text.trim()), "{:?}", r.text);
    }
}

/// With teeth (CLAUDE.md, principle 3): a constraint that does nothing is caught. The same test body, with the mask skipped,
/// must NOT reproduce the guarantee on every prompt: here a deliberately wrong grammar (an enum that cannot hold the answer's
/// first word) shows that the output follows the grammar and not the model's own preference.
#[test]
fn the_output_follows_the_grammar_not_the_model() {
    let Some(path) = model_path() else { return };
    let model = Model::load_with(std::path::Path::new(&path), false).expect("load model");
    let trie = trie_for(&model);
    // the model wants to answer about a person; the grammar only allows the number 42
    let schema = json!({"enum": [42]});
    let r = ask(&model, &trie, Some(&schema), true, "Write a short poem about the sea.", 0.9, 3);
    assert_eq!(r.text.trim(), "42", "{:?}", r.text);
}
