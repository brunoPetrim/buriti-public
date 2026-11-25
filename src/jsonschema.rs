//! Constrained decoding, part 2 (D82): JSON Schema to a byte grammar (`grammar.rs`).
//!
//! Supported: `type` (string, number, integer, boolean, null, object, array, or a list of types), `enum`, `const`,
//! `properties` / `required`, `items`, `minItems` / `maxItems`, `anyOf` / `oneOf`, `$ref` (`#`, `#/$defs/..`,
//! `#/definitions/..`, with recursion), `true` (anything) and the annotation keywords (title, description, default, ...).
//! A schema with no `type` and no other keyword accepts any JSON value.
//!
//! Choices that differ from the full JSON Schema language (all stated in the docs; none is silent):
//! - Object members come out in a FIXED order: the required ones first, then the optional ones, each group in the order
//!   `serde_json` gives (alphabetical by name, because the crate's `preserve_order` feature is not enabled). A document
//!   with the same members in another order is valid JSON Schema but is not generated. `additionalProperties` is treated
//!   as false when `properties` is present.
//! - Keywords that restrict values and cannot be enforced here are REFUSED with an error instead of ignored (`pattern`,
//!   `minimum`, `maximum`, `minLength`, `maxLength`, `allOf`, `not`, `if`, `uniqueItems`, ...): ignoring them would let
//!   the model break a rule the caller believes is enforced.
//! - Whitespace between tokens is bounded (one space, or a newline plus up to 20 spaces) so that the model cannot stall on
//!   blanks; numbers are not range-checked.

use crate::grammar::{Builder, ByteSet, Grammar, Sym};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

const UNSUPPORTED: &[&str] = &[
    "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "uniqueItems", "allOf", "not", "if", "then", "else",
    "patternProperties", "propertyNames", "minProperties", "maxProperties", "dependentRequired", "dependentSchemas", "dependencies", "contains", "minContains", "maxContains",
    "prefixItems", "unevaluatedProperties", "unevaluatedItems", "additionalItems",
];
const MAX_DEPTH: usize = 64;
const MAX_ITEMS: usize = 1000;

fn rule(r: u32) -> Sym {
    Sym::Rule(r)
}

struct Conv<'a> {
    b: Builder,
    root: &'a Value,
    refs: HashMap<String, u32>,
    ws: u32,
    string: u32,
    number: u32,
    integer: u32,
    boolean: u32,
    null: u32,
    any: u32,
    object_any: u32,
}

impl<'a> Conv<'a> {
    fn new(root: &'a Value) -> Self {
        let mut b = Builder::new();
        // whitespace: nothing, one space, or a newline and up to 20 spaces
        let spaces = b.up_to(Builder::lit(" "), 20);
        let mut nl = Builder::lit("\n");
        nl.push(rule(spaces));
        let ws = b.rule_of(vec![Vec::new(), Builder::lit(" "), nl]);
        let with_ws = |b: &mut Builder, raw: u32| b.rule_of(vec![vec![rule(raw), rule(ws)]]);

        let digit = Sym::Set(ByteSet::range(b'0', b'9'));
        let digits = b.star(vec![vec![digit]]);
        let minus = b.opt(Builder::lit("-"));
        let int_body = b.rule_of(vec![vec![Sym::Set(ByteSet::single(b'0'))], vec![Sym::Set(ByteSet::range(b'1', b'9')), rule(digits)]]);
        let int_raw = b.rule_of(vec![vec![rule(minus), rule(int_body)]]);
        let frac = b.opt(vec![Sym::Set(ByteSet::single(b'.')), digit, rule(digits)]);
        let sign = b.opt(vec![Sym::Set(ByteSet::of(b"+-"))]);
        let exp = b.opt(vec![Sym::Set(ByteSet::of(b"eE")), rule(sign), digit, rule(digits)]);
        let num_raw = b.rule_of(vec![vec![rule(int_raw), rule(frac), rule(exp)]]);

        let hex = Sym::Set(ByteSet::range(b'0', b'9').union(ByteSet::range(b'a', b'f')).union(ByteSet::range(b'A', b'F')));
        let esc = b.rule_of(vec![vec![Sym::Set(ByteSet::of(b"\"\\/bfnrt"))], vec![Sym::Set(ByteSet::single(b'u')), hex, hex, hex, hex]]);
        let plain = Sym::Set(ByteSet::of(b"\"\\").union(ByteSet::range(0, 0x1F)).negate());
        let chars = b.star(vec![vec![plain], vec![Sym::Set(ByteSet::single(b'\\')), rule(esc)]]);
        let quote = Sym::Set(ByteSet::single(b'"'));
        let str_raw = b.rule_of(vec![vec![quote, rule(chars), quote]]);

        let bool_raw = b.rule_of(vec![Builder::lit("true"), Builder::lit("false")]);
        let null_raw = b.rule_of(vec![Builder::lit("null")]);

        let string = with_ws(&mut b, str_raw);
        let number = with_ws(&mut b, num_raw);
        let integer = with_ws(&mut b, int_raw);
        let boolean = with_ws(&mut b, bool_raw);
        let null = with_ws(&mut b, null_raw);

        // any JSON value (recursive through the `any` rule)
        let any = b.rule();
        let comma_ws = vec![Sym::Set(ByteSet::single(b',')), rule(ws)];
        let mut kv = vec![rule(string)];
        kv.extend(Builder::lit(":"));
        kv.push(rule(ws));
        kv.push(rule(any));
        let mut more_kv = comma_ws.clone();
        more_kv.extend(kv.clone());
        let more_kvs = b.star(vec![more_kv]);
        let mut members = kv;
        members.push(rule(more_kvs));
        let members_opt = b.opt(members);
        let mut obj = Builder::lit("{");
        obj.push(rule(ws));
        obj.push(rule(members_opt));
        obj.extend(Builder::lit("}"));
        obj.push(rule(ws));
        let object_any = b.rule_of(vec![obj]);
        let mut more_items = comma_ws;
        more_items.push(rule(any));
        let more = b.star(vec![more_items]);
        let items_opt = b.opt(vec![rule(any), rule(more)]);
        let mut arr = Builder::lit("[");
        arr.push(rule(ws));
        arr.push(rule(items_opt));
        arr.extend(Builder::lit("]"));
        arr.push(rule(ws));
        let array_any = b.rule_of(vec![arr]);
        for alt in [object_any, array_any, string, number, boolean, null] {
            b.alt(any, vec![rule(alt)]);
        }
        Conv { b, root, refs: HashMap::new(), ws, string, number, integer, boolean, null, any, object_any }
    }

    fn literal(&mut self, values: &[Value]) -> Result<u32, String> {
        if values.is_empty() {
            return Err("JSON Schema: an empty 'enum' accepts no value".into());
        }
        let mut alts = Vec::new();
        for v in values {
            let mut s = Builder::lit(&serde_json::to_string(v).map_err(|e| format!("JSON Schema: invalid enum value: {e}"))?);
            s.push(rule(self.ws));
            alts.push(s);
        }
        Ok(self.b.rule_of(alts))
    }

    fn visit(&mut self, schema: &Value, depth: usize) -> Result<u32, String> {
        if depth > MAX_DEPTH {
            return Err("JSON Schema: nesting is too deep".into());
        }
        match schema {
            Value::Bool(true) => Ok(self.any),
            Value::Bool(false) => Err("JSON Schema: the `false` schema accepts no value".into()),
            Value::Object(m) => self.visit_obj(m, depth),
            _ => Err("JSON Schema: each schema must be an object (or true)".into()),
        }
    }

    fn visit_obj(&mut self, m: &Map<String, Value>, depth: usize) -> Result<u32, String> {
        if let Some(k) = m.keys().find(|k| UNSUPPORTED.contains(&k.as_str())) {
            return Err(format!("JSON Schema: the keyword '{k}' is not supported by constrained generation yet (it cannot be enforced); remove it or validate the answer afterwards."));
        }
        if let Some(r) = m.get("$ref") {
            let r = r.as_str().ok_or("JSON Schema: '$ref' must be a string")?;
            return self.visit_ref(r, depth);
        }
        if let Some(c) = m.get("const") {
            return self.literal(std::slice::from_ref(c));
        }
        if let Some(e) = m.get("enum") {
            let list = e.as_array().ok_or("JSON Schema: 'enum' must be a list")?;
            return self.literal(list);
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(v) = m.get(key) {
                let list = v.as_array().ok_or_else(|| format!("JSON Schema: '{key}' must be a list"))?;
                if list.is_empty() {
                    return Err(format!("JSON Schema: an empty '{key}' accepts no value"));
                }
                let mut alts = Vec::new();
                for s in list {
                    let r = self.visit(s, depth + 1)?;
                    alts.push(vec![rule(r)]);
                }
                return Ok(self.b.rule_of(alts));
            }
        }
        match m.get("type") {
            Some(Value::String(t)) => self.visit_type(t, m, depth),
            Some(Value::Array(ts)) => {
                if ts.is_empty() {
                    return Err("JSON Schema: empty 'type'".into());
                }
                let mut alts = Vec::new();
                for t in ts {
                    let t = t.as_str().ok_or("JSON Schema: the items of 'type' must be strings")?;
                    let r = self.visit_type(t, m, depth)?;
                    alts.push(vec![rule(r)]);
                }
                Ok(self.b.rule_of(alts))
            }
            None => {
                if m.contains_key("properties") || m.contains_key("required") {
                    self.object(m, depth)
                } else if m.contains_key("items") {
                    self.array(m, depth)
                } else {
                    Ok(self.any)
                }
            }
            _ => Err("JSON Schema: 'type' must be a string or a list of strings".into()),
        }
    }

    fn visit_type(&mut self, t: &str, m: &Map<String, Value>, depth: usize) -> Result<u32, String> {
        match t {
            "string" => Ok(self.string),
            "number" => Ok(self.number),
            "integer" => Ok(self.integer),
            "boolean" => Ok(self.boolean),
            "null" => Ok(self.null),
            "object" => self.object(m, depth),
            "array" => self.array(m, depth),
            other => Err(format!("JSON Schema: unknown type '{other}'")),
        }
    }

    fn visit_ref(&mut self, r: &str, depth: usize) -> Result<u32, String> {
        if let Some(&id) = self.refs.get(r) {
            return Ok(id);
        }
        let root = self.root;
        let Some(ptr) = r.strip_prefix('#') else {
            return Err(format!("JSON Schema: only local references ('#/...') are supported, got '{r}'"));
        };
        let target = if ptr.is_empty() { root } else { root.pointer(ptr).ok_or_else(|| format!("JSON Schema: the reference '{r}' does not exist"))? };
        let id = self.b.rule();
        self.refs.insert(r.to_string(), id);
        let inner = self.visit(target, depth + 1)?;
        self.b.alt(id, vec![rule(inner)]);
        Ok(id)
    }

    fn object(&mut self, m: &Map<String, Value>, depth: usize) -> Result<u32, String> {
        let Some(props) = m.get("properties").and_then(Value::as_object) else {
            return Ok(self.object_any);
        };
        if props.is_empty() {
            return Ok(self.object_any);
        }
        let required: HashSet<&str> = match m.get("required") {
            None => HashSet::new(),
            Some(Value::Array(a)) => a.iter().map(|v| v.as_str().ok_or("JSON Schema: the items of 'required' must be strings")).collect::<Result<_, _>>()?,
            Some(_) => return Err("JSON Schema: 'required' must be a list".into()),
        };
        if let Some(missing) = required.iter().find(|n| !props.contains_key(**n)) {
            return Err(format!("JSON Schema: the required property '{missing}' is not in 'properties'"));
        }
        let comma = vec![Sym::Set(ByteSet::single(b',')), rule(self.ws)];
        let (mut req, mut opt): (Vec<Vec<Sym>>, Vec<Vec<Sym>>) = (Vec::new(), Vec::new());
        for (name, sub) in props {
            let v = self.visit(sub, depth + 1)?;
            let mut member = Builder::lit(&serde_json::to_string(name).map_err(|e| e.to_string())?);
            member.push(rule(self.ws));
            member.extend(Builder::lit(":"));
            member.push(rule(self.ws));
            member.push(rule(v));
            if required.contains(name.as_str()) {
                req.push(member);
            } else {
                opt.push(member);
            }
        }
        let n = opt.len();
        // rest[k]: optional members k.. , each preceded by a comma, in order (any of them may be skipped)
        let mut rest = vec![0u32; n + 1];
        rest[n] = self.b.rule_of(vec![Vec::new()]);
        for k in (0..n).rev() {
            let mut with = comma.clone();
            with.extend(opt[k].clone());
            with.push(rule(rest[k + 1]));
            rest[k] = self.b.rule_of(vec![with, vec![rule(rest[k + 1])]]);
        }
        let body: Vec<Sym> = if req.is_empty() {
            // first[k]: no member written yet, so the first one written has no comma
            let mut first = vec![0u32; n + 1];
            first[n] = self.b.rule_of(vec![Vec::new()]);
            for k in (0..n).rev() {
                let mut with = opt[k].clone();
                with.push(rule(rest[k + 1]));
                first[k] = self.b.rule_of(vec![with, vec![rule(first[k + 1])]]);
            }
            vec![rule(first[0])]
        } else {
            let mut body = req[0].clone();
            for r in &req[1..] {
                body.extend(comma.clone());
                body.extend(r.clone());
            }
            body.push(rule(rest[0]));
            body
        };
        let mut seq = Builder::lit("{");
        seq.push(rule(self.ws));
        seq.extend(body);
        seq.extend(Builder::lit("}"));
        seq.push(rule(self.ws));
        Ok(self.b.rule_of(vec![seq]))
    }

    fn array(&mut self, m: &Map<String, Value>, depth: usize) -> Result<u32, String> {
        let item = match m.get("items") {
            None => self.any,
            Some(Value::Array(_)) => return Err("JSON Schema: 'items' as a list (tuples) is not supported".into()),
            Some(s) => self.visit(s, depth + 1)?,
        };
        let min = m.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
        let max = m.get("maxItems").and_then(Value::as_u64).map(|x| x as usize);
        if min > MAX_ITEMS || max.is_some_and(|x| x > MAX_ITEMS) {
            return Err(format!("JSON Schema: 'minItems' and 'maxItems' above {MAX_ITEMS} are not supported"));
        }
        if max.is_some_and(|x| x < min) {
            return Err("JSON Schema: 'maxItems' is smaller than 'minItems'".into());
        }
        let mut comma_item = vec![Sym::Set(ByteSet::single(b',')), rule(self.ws)];
        comma_item.push(rule(item));
        let list: Vec<Sym> = if max == Some(0) {
            Vec::new()
        } else {
            let m1 = min.max(1);
            let mut s = vec![rule(item)];
            for _ in 1..m1 {
                s.extend(comma_item.clone());
            }
            match max {
                None => {
                    let tail = self.b.star(vec![comma_item.clone()]);
                    s.push(rule(tail));
                }
                Some(mx) => {
                    let tail = self.b.up_to(comma_item.clone(), mx - m1);
                    s.push(rule(tail));
                }
            }
            if min == 0 {
                vec![rule(self.b.opt(s))]
            } else {
                s
            }
        };
        let mut seq = Builder::lit("[");
        seq.push(rule(self.ws));
        seq.extend(list);
        seq.extend(Builder::lit("]"));
        seq.push(rule(self.ws));
        Ok(self.b.rule_of(vec![seq]))
    }
}

/// The grammar of a JSON Schema (the root value is the whole generated text).
pub fn grammar_for_schema(schema: &Value) -> Result<Grammar, String> {
    let mut c = Conv::new(schema);
    let r = c.visit(schema, 0)?;
    c.b.build(r)
}

/// The grammar requested by the `response_format` of a chat request (`None` when there is none or it is `text`).
/// `json_object` is any JSON object; `json_schema` is `{"json_schema": {"schema": {...}}}` as in the OpenAI API.
pub fn from_response_format(body: &Value) -> Result<Option<std::sync::Arc<Grammar>>, String> {
    let Some(rf) = body.get("response_format").filter(|v| !v.is_null()) else { return Ok(None) };
    let ty = rf.get("type").and_then(Value::as_str).ok_or("response_format.type must be a string")?;
    let g = match ty {
        "text" => return Ok(None),
        "json_object" => grammar_json_object()?,
        "json_schema" => {
            let schema = rf.get("json_schema").and_then(|j| j.get("schema")).ok_or("response_format.json_schema.schema is required")?;
            grammar_for_schema(schema)?
        }
        other => return Err(format!("response_format.type '{other}' is not supported (use 'text', 'json_object' or 'json_schema')")),
    };
    if g.start().is_empty() {
        return Err("response_format: the grammar accepts no text".into());
    }
    Ok(Some(std::sync::Arc::new(g)))
}

/// `response_format: {"type": "json_object"}`: any JSON object.
pub fn grammar_json_object() -> Result<Grammar, String> {
    let dummy = Value::Null;
    let c = Conv::new(&dummy);
    let root = c.object_any;
    c.b.build(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ok(g: &Grammar, s: &str) {
        assert!(g.matches(s.as_bytes()), "must match: {s}");
    }
    fn no(g: &Grammar, s: &str) {
        assert!(!g.matches(s.as_bytes()), "must NOT match: {s}");
    }

    #[test]
    fn required_members_come_in_order() {
        let g = grammar_for_schema(&json!({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]})).unwrap();
        ok(&g, r#"{"age":3,"name":"x"}"#);
        ok(&g, "{\"age\": 30,\n \"name\": \"Ana\"}");
        no(&g, r#"{"name":"x","age":3}"#); // other order: valid JSON Schema, not generated (documented)
        no(&g, r#"{"age":3}"#);
        no(&g, r#"{"age":"3","name":"x"}"#);
        no(&g, r#"{"age":3,"name":"x","extra":1}"#);
    }

    #[test]
    fn optional_members_may_be_skipped_in_order() {
        let g = grammar_for_schema(&json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},"required":["a"]})).unwrap();
        for s in [r#"{"a":1}"#, r#"{"a":1,"b":2}"#, r#"{"a":1,"c":3}"#, r#"{"a":1,"b":2,"c":3}"#] {
            ok(&g, s);
        }
        no(&g, r#"{"a":1,"c":3,"b":2}"#);
        no(&g, r#"{"b":2}"#);
        let all_opt = grammar_for_schema(&json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}}})).unwrap();
        for s in ["{}", r#"{"b":1}"#, r#"{"a":1,"c":3}"#, r#"{"a":1,"b":2,"c":3}"#, r#"{"c":3}"#] {
            ok(&all_opt, s);
        }
        no(&all_opt, r#"{"b":1,"a":1}"#);
        no(&all_opt, r#"{,"b":1}"#);
    }

    #[test]
    fn arrays_with_bounds() {
        let g = grammar_for_schema(&json!({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3})).unwrap();
        for s in ["[1]", "[1,2,3]", "[ 1 , 2 ]"] {
            ok(&g, s);
        }
        for s in ["[]", "[1,2,3,4]", "[1,]", "[\"a\"]"] {
            no(&g, s);
        }
        let any_len = grammar_for_schema(&json!({"type":"array","items":{"type":"string"}})).unwrap();
        ok(&any_len, "[]");
        ok(&any_len, r#"["a","b","c","d","e"]"#);
        let exact = grammar_for_schema(&json!({"type":"array","items":{"type":"boolean"},"minItems":2,"maxItems":2})).unwrap();
        ok(&exact, "[true,false]");
        no(&exact, "[true]");
        no(&exact, "[true,false,true]");
    }

    #[test]
    fn enums_consts_unions_and_type_lists() {
        let e = grammar_for_schema(&json!({"enum":["red","green",3,null]})).unwrap();
        for s in [r#""red""#, r#""green""#, "3", "null"] {
            ok(&e, s);
        }
        no(&e, r#""blue""#);
        no(&e, "4");
        let u = grammar_for_schema(&json!({"anyOf":[{"type":"integer"},{"type":"string"}]})).unwrap();
        ok(&u, "12");
        ok(&u, r#""x""#);
        no(&u, "true");
        let t = grammar_for_schema(&json!({"type":["string","null"]})).unwrap();
        ok(&t, "null");
        ok(&t, r#""s""#);
        no(&t, "1");
        let c = grammar_for_schema(&json!({"const":"fixed"})).unwrap();
        ok(&c, r#""fixed""#);
        no(&c, r#""other""#);
    }

    #[test]
    fn recursive_references() {
        let g = grammar_for_schema(&json!({
            "$defs": {"node": {"type":"object","properties":{"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["kids"]}},
            "$ref": "#/$defs/node"
        }))
        .unwrap();
        ok(&g, r#"{"kids":[]}"#);
        ok(&g, r#"{"kids":[{"kids":[]},{"kids":[{"kids":[]}]}]}"#);
        no(&g, r#"{"kids":[{}]}"#);
        assert!(grammar_for_schema(&json!({"$ref":"#/$defs/missing"})).is_err());
        assert!(grammar_for_schema(&json!({"$ref":"#"})).is_err(), "a schema that is only a reference to itself is left recursion");
        assert!(grammar_for_schema(&json!({"$ref":"http://example.com/x"})).is_err());
    }

    #[test]
    fn numbers_and_strings_follow_json() {
        let n = grammar_for_schema(&json!({"type":"number"})).unwrap();
        for s in ["0", "-1.5e3", "10", "2E+4", "-0.25"] {
            ok(&n, s);
        }
        for s in ["01", "-", "1.", ".5", "1e", "+1"] {
            no(&n, s);
        }
        let i = grammar_for_schema(&json!({"type":"integer"})).unwrap();
        ok(&i, "-12");
        no(&i, "1.5");
        let s = grammar_for_schema(&json!({"type":"string"})).unwrap();
        for t in [r#""a""#, r#""a\"b""#, r#""é""#, "\"é\"", r#""\n""#] {
            ok(&s, t);
        }
        for t in ["\"a\nb\"", r#""\x""#, r#""abc"#, r#"abc"#, r#""\u12""#] {
            no(&s, t);
        }
    }

    #[test]
    fn whitespace_is_bounded() {
        let g = grammar_for_schema(&json!({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]})).unwrap();
        ok(&g, &format!("{{\n{}\"a\": 1\n}}", " ".repeat(20)));
        no(&g, &format!("{{{}\"a\":1}}", " ".repeat(100)));
    }

    #[test]
    fn json_object_mode_and_free_values() {
        let g = grammar_json_object().unwrap();
        for s in ["{}", r#"{"a":[1,{"b":null},"x",true,-2.5e1]}"#, "{ \"k\" : { \"n\" : [ ] } }"] {
            ok(&g, s);
        }
        for s in ["[1]", r#"{"a":}"#, r#"{"a" 1}"#, "{", r#"{"a":1,}"#, "\"x\""] {
            no(&g, s);
        }
        let any = grammar_for_schema(&json!({})).unwrap();
        ok(&any, "[1,2]");
        ok(&any, "\"x\"");
        ok(&any, "{}");
    }

    #[test]
    fn unenforceable_keywords_are_refused_not_ignored() {
        for kw in [json!({"type":"string","pattern":"^a"}), json!({"type":"integer","minimum":1}), json!({"allOf":[{"type":"string"}]}), json!({"type":"string","maxLength":3})] {
            let e = grammar_for_schema(&kw).err().expect("must be refused");
            assert!(e.contains("is not supported"), "{e}");
        }
        // annotations are fine
        assert!(grammar_for_schema(&json!({"type":"string","title":"t","description":"d","default":"x","format":"email"})).is_ok());
        assert!(grammar_for_schema(&json!(false)).is_err());
        assert!(grammar_for_schema(&json!({"enum":[]})).is_err());
        assert!(grammar_for_schema(&json!({"type":"object","properties":{"a":{"type":"integer"}},"required":["b"]})).is_err());
        assert!(grammar_for_schema(&json!({"type":"array","items":[{"type":"string"}]})).is_err());
        assert!(grammar_for_schema(&json!({"type":"array","minItems":3,"maxItems":2})).is_err());
    }

    #[test]
    fn response_format_selects_the_grammar() {
        assert!(from_response_format(&json!({})).unwrap().is_none());
        assert!(from_response_format(&json!({"response_format": null})).unwrap().is_none());
        assert!(from_response_format(&json!({"response_format": {"type": "text"}})).unwrap().is_none());
        let obj = from_response_format(&json!({"response_format": {"type": "json_object"}})).unwrap().unwrap();
        assert!(obj.matches(b"{}") && !obj.matches(b"[]"));
        let sch = from_response_format(&json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}}})).unwrap().unwrap();
        assert!(sch.matches(br#"{"ok":true}"#) && !sch.matches(br#"{"ok":1}"#));
        for bad in [json!({"response_format": {"type": "xml"}}), json!({"response_format": {}}), json!({"response_format": {"type": "json_schema"}}), json!({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "string", "pattern": "a"}}}})] {
            assert!(from_response_format(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn nesting_a_schema_in_a_schema() {
        let g = grammar_for_schema(&json!({
            "type":"object",
            "properties":{
                "city":{"type":"string"},
                "tags":{"type":"array","items":{"enum":["a","b"]},"maxItems":2},
                "geo":{"type":"object","properties":{"lat":{"type":"number"},"lon":{"type":"number"}},"required":["lat","lon"]}
            },
            "required":["city","geo"]
        }))
        .unwrap();
        ok(&g, r#"{"city":"Rio","geo":{"lat":-22.9,"lon":-43.2},"tags":["a","b"]}"#);
        ok(&g, r#"{"city":"Rio","geo":{"lat":1,"lon":2}}"#);
        no(&g, r#"{"city":"Rio","geo":{"lat":1}}"#);
        no(&g, r#"{"city":"Rio","geo":{"lat":1,"lon":2},"tags":["a","b","a"]}"#);
        no(&g, r#"{"city":"Rio","geo":{"lat":1,"lon":2},"tags":["z"]}"#);
    }
}
