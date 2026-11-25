//! Chamada de funções (`tools`) em `/v1/chat/completions` (D79).
//!
//! Entrada: `tools`, `tool_choice` e o histórico (`assistant.tool_calls`, papel `tool`) viram mensagens de chat comuns
//! (`ChatMsg`) escritas EXATAMENTE como o template do modelo as escreveria (Hermes/Qwen, Llama 3.1+); o tokenizador do
//! motor próprio só as envolve nos delimitadores de turno. Saída: o texto gerado é lido de volta como `tool_calls`.
//! Não há decodificação restrita: o modelo pode errar o formato e o JSON, e então o texto cru volta como `content`
//! com o aviso `buriti_tool_call_error` (nunca inventamos uma chamada).
//!
//! JSON com ORDEM: o `serde_json` do projeto ordena as chaves (sem `preserve_order`), mas o template do modelo faz
//! `tool | tojson` na ordem do cliente. `J` guarda a ordem e escreve com os separadores do `tojson` do template.

use serde_json::{json, Value};

use crate::engine::ChatMsg;

/// Papel interno (nunca vem da API: `parse_messages` só deixa passar os papéis da OpenAI) da mensagem de sistema do
/// Llama 3 quando há ferramentas: o tokenizador escreve `Environment: ipython` antes da data e não apara o texto.
pub const TOOLS_SYSTEM_ROLE: &str = "system+tools";

// ------------------------------------------------------------------ formato do modelo

/// Como o template do modelo escreve e lê chamadas de função.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolFormat {
    /// ChatML do Qwen2.5/Qwen3 (Hermes): `<tools>` no sistema, `<tool_call>{json}</tool_call>` na resposta,
    /// `<tool_response>` num turno `user`. `double_braces`: o template deste GGUF escreve `{{"name": ...}}` (chaves
    /// duplicadas, como nos GGUFs oficiais do Qwen2.5); `default_system`: o sistema padrão do template.
    Hermes { qwen3: bool, double_braces: bool, default_system: Option<&'static str> },
    /// Llama 3.1+ (JSON `{"name": ..., "parameters": {...}}`, papel `ipython` para o resultado).
    Llama3,
}

/// O que o modelo carregado sabe fazer com ferramentas.
#[derive(Clone, Debug)]
pub enum ToolSupport {
    Yes(ToolFormat),
    /// motivo, em português (vai para o cliente)
    No(String),
}

const RECOMMEND: &str = "Use um modelo Instruct com ferramentas: Qwen2.5 ou Qwen3 (família Qwen) ou Llama 3.1/3.2. Modelos pequenos (abaixo de 3B) erram o formato com frequência; para agentes prefira 7B/8B ou mais.";

impl ToolSupport {
    pub fn no_template(arch: &str) -> ToolSupport {
        ToolSupport::No(format!("Este modelo (arquitetura '{arch}') não tem chamada de funções no template de chat do arquivo. {RECOMMEND}"))
    }
    pub fn not_implemented(what: &str) -> ToolSupport {
        ToolSupport::No(format!("O template deste modelo ({what}) tem ferramentas, mas esse formato ainda não está implementado no motor próprio do Buriti. {RECOMMEND}"))
    }
    pub fn llama_cpp() -> ToolSupport {
        ToolSupport::No(format!("Chamada de funções só funciona no motor próprio do Buriti (o llama-server é iniciado sem `--jinja`). Troque o motor em Configurações. {RECOMMEND}"))
    }

    /// Resumo para `GET /v1/models` (extensão do Buriti, campo `buriti_tools`).
    pub fn describe(&self) -> Value {
        match self {
            ToolSupport::Yes(f) => json!({
                "supported": true,
                "format": match f { ToolFormat::Hermes { .. } => "hermes", ToolFormat::Llama3 => "llama3" },
                "valid_json_guaranteed": false,
                "note": "O formato é o do template do modelo, sem decodificação restrita: o JSON pode vir inválido (o texto cru volta em `content` com `buriti_tool_call_error`). Modelos abaixo de 3B erram com frequência; prefira Qwen2.5/Qwen3 ou Llama 3.1/3.2 de 7B/8B ou mais."
            }),
            ToolSupport::No(r) => json!({ "supported": false, "reason": r }),
        }
    }
}

// ------------------------------------------------------------------ JSON com ordem

/// Valor JSON que preserva a ordem das chaves.
#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    /// o texto do número como veio
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

/// Separadores ao escrever: `Compact` (`,` `:`), `Spaced` (`, ` `: `, o `tojson` do template) e `Indent(n)` (o `tojson(indent=n)`).
#[derive(Clone, Copy, Debug)]
pub enum Style {
    Compact,
    Spaced,
    Indent(usize),
}

const MAX_DEPTH: usize = 64;

struct P<'a> {
    b: &'a [u8],
    s: &'a str,
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("JSON inválido (posição {}): {m}", self.i))
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    /// Lê uma string JSON (o cursor está na aspa de abertura). Sem `build` só confere a sintaxe (sem alocar).
    fn string(&mut self, build: bool) -> Result<String, String> {
        let start = self.i;
        self.i += 1;
        loop {
            match self.b.get(self.i) {
                None => return self.err("texto não terminado"),
                Some(b'\\') => self.i += 2,
                Some(b'"') => {
                    self.i += 1;
                    break;
                }
                Some(_) => self.i += 1,
            }
        }
        let lit = &self.s[start..self.i];
        if build {
            serde_json::from_str::<String>(lit).map_err(|e| format!("JSON inválido: {e}"))
        } else {
            serde_json::from_str::<serde::de::IgnoredAny>(lit).map(|_| String::new()).map_err(|e| format!("JSON inválido: {e}"))
        }
    }
    fn object(&mut self, build: bool, keep: Option<&[&str]>, depth: usize) -> Result<J, String> {
        self.i += 1; // '{'
        let mut v = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(if build { J::Obj(v) } else { J::Null });
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return self.err("esperava o nome de uma chave");
            }
            let k = self.string(true)?;
            self.ws();
            if self.peek() != Some(b':') {
                return self.err("esperava ':'");
            }
            self.i += 1;
            let keep_it = build && keep.map_or(true, |ks| ks.contains(&k.as_str()));
            let val = self.value(keep_it, depth + 1)?;
            if keep_it {
                v.push((k, val));
            }
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return self.err("esperava ',' ou '}'"),
            }
        }
        Ok(if build { J::Obj(v) } else { J::Null })
    }
    fn value(&mut self, build: bool, depth: usize) -> Result<J, String> {
        if depth > MAX_DEPTH {
            return self.err("aninhamento profundo demais");
        }
        self.ws();
        match self.peek() {
            None => self.err("fim inesperado"),
            Some(b'{') => self.object(build, None, depth),
            Some(b'[') => {
                self.i += 1;
                let mut v = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(if build { J::Arr(v) } else { J::Null });
                }
                loop {
                    let x = self.value(build, depth + 1)?;
                    if build {
                        v.push(x);
                    }
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            break;
                        }
                        _ => return self.err("esperava ',' ou ']'"),
                    }
                }
                Ok(if build { J::Arr(v) } else { J::Null })
            }
            Some(b'"') => {
                let s = self.string(build)?;
                Ok(if build { J::Str(s) } else { J::Null })
            }
            Some(b't') if self.s[self.i..].starts_with("true") => {
                self.i += 4;
                Ok(J::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with("false") => {
                self.i += 5;
                Ok(J::Bool(false))
            }
            Some(b'n') if self.s[self.i..].starts_with("null") => {
                self.i += 4;
                Ok(J::Null)
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let lit = &self.s[start..self.i];
                serde_json::from_str::<serde::de::IgnoredAny>(lit).map_err(|_| format!("JSON inválido: número `{lit}`"))?;
                Ok(if build { J::Num(lit.to_string()) } else { J::Null })
            }
            Some(_) => self.err("valor inesperado"),
        }
    }
}

impl J {
    /// Lê um documento JSON inteiro (texto depois do valor é erro).
    pub fn parse(s: &str) -> Result<J, String> {
        let mut p = P { b: s.as_bytes(), s, i: 0 };
        let v = p.value(true, 0)?;
        p.ws();
        if p.i != p.b.len() {
            return p.err("texto depois do JSON");
        }
        Ok(v)
    }

    /// Lê um objeto JSON e guarda só os membros de nível 1 cujo nome está em `keys` (o resto é conferido e descartado
    /// sem alocar: o corpo de um pedido de editor pode ter megabytes de código).
    pub fn parse_members(s: &str, keys: &[&str]) -> Result<Vec<(String, J)>, String> {
        let mut p = P { b: s.as_bytes(), s, i: 0 };
        p.ws();
        if p.peek() != Some(b'{') {
            return p.err("o corpo deve ser um objeto");
        }
        let v = p.object(true, Some(keys), 0)?;
        p.ws();
        if p.i != p.b.len() {
            return p.err("texto depois do JSON");
        }
        match v {
            J::Obj(m) => Ok(m),
            _ => Ok(vec![]),
        }
    }

    /// Membro de um objeto (se a chave se repete, vale a última, como nos analisadores comuns).
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(m) => m.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn render(&self, st: Style) -> String {
        let mut out = String::new();
        self.write(&mut out, st, 0);
        out
    }

    fn write(&self, out: &mut String, st: Style, lvl: usize) {
        let nl = |out: &mut String, lvl: usize| {
            if let Style::Indent(n) = st {
                out.push('\n');
                out.extend(std::iter::repeat(' ').take(n * lvl));
            }
        };
        let (sep, colon) = match st {
            Style::Compact => (",", ":"),
            Style::Spaced => (", ", ": "),
            Style::Indent(_) => (",", ": "),
        };
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Num(n) => out.push_str(n),
            J::Str(s) => out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())),
            J::Arr(a) if a.is_empty() => out.push_str("[]"),
            J::Arr(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(sep);
                    }
                    nl(out, lvl + 1);
                    x.write(out, st, lvl + 1);
                }
                nl(out, lvl);
                out.push(']');
            }
            J::Obj(m) if m.is_empty() => out.push_str("{}"),
            J::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(sep);
                    }
                    nl(out, lvl + 1);
                    out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into()));
                    out.push_str(colon);
                    v.write(out, st, lvl + 1);
                }
                nl(out, lvl);
                out.push('}');
            }
        }
    }
}

// ------------------------------------------------------------------ pedido

/// Uma ferramenta oferecida: o nome e o objeto inteiro como o cliente mandou (`{"type":"function","function":{...}}`).
#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub json: J,
}

const MAX_TOOLS: usize = 128;

fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// Lê `tools` do corpo bruto do pedido (para manter a ordem das chaves) e valida cada ferramenta.
pub fn parse_tools(raw: &str) -> Result<Vec<Tool>, String> {
    let members = J::parse_members(raw, &["tools"])?;
    let items = match members.iter().rev().find(|(k, _)| k == "tools") {
        Some((_, J::Arr(a))) => a,
        Some((_, J::Null)) | None => return Ok(vec![]),
        _ => return Err("`tools` deve ser uma lista.".into()),
    };
    if items.len() > MAX_TOOLS {
        return Err(format!("`tools` tem {} itens; o máximo é {MAX_TOOLS}.", items.len()));
    }
    let mut out: Vec<Tool> = Vec::new();
    for (i, t) in items.iter().enumerate() {
        if t.get("type").and_then(J::as_str) != Some("function") {
            return Err(format!("tools[{i}]: só `type: \"function\"` é suportado."));
        }
        let Some(f @ J::Obj(_)) = t.get("function") else { return Err(format!("tools[{i}]: falta o objeto `function`.")) };
        let name = f.get("name").and_then(J::as_str).unwrap_or("");
        if !valid_name(name) {
            return Err(format!("tools[{i}]: `function.name` inválido (use letras, números, `_`, `-` ou `.`, até 64 caracteres)."));
        }
        if let Some(p) = f.get("parameters") {
            if !matches!(p, J::Obj(_) | J::Null) {
                return Err(format!("tools[{i}]: `function.parameters` deve ser um objeto (JSON Schema)."));
            }
        }
        if out.iter().any(|x| x.name == name) {
            return Err(format!("tools[{i}]: o nome `{name}` está repetido."));
        }
        out.push(Tool { name: name.to_string(), json: t.clone() });
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

pub fn parse_choice(v: Option<&Value>) -> Result<ToolChoice, String> {
    match v {
        None | Some(Value::Null) => Ok(ToolChoice::Auto),
        Some(Value::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" => Ok(ToolChoice::Required),
            other => Err(format!("`tool_choice` desconhecido: `{other}` (use auto, none, required ou {{\"type\":\"function\",\"function\":{{\"name\":...}}}}).")),
        },
        Some(o) if o.get("type").and_then(Value::as_str) == Some("function") => match o.pointer("/function/name").and_then(Value::as_str) {
            Some(n) if !n.is_empty() => Ok(ToolChoice::Named(n.to_string())),
            _ => Err("`tool_choice` com `type: function` precisa de `function.name`.".into()),
        },
        Some(_) => Err("`tool_choice` inválido.".into()),
    }
}

fn nonempty_array(v: Option<&Value>) -> bool {
    v.and_then(Value::as_array).is_some_and(|a| !a.is_empty())
}

/// O pedido usa ferramentas? `tools` não vazio, ou histórico com `tool_calls`/papel `tool`.
pub fn wants_tools(body: &Value) -> bool {
    if body.get("tools").is_some_and(|t| !t.is_null() && t.as_array().map(|a| !a.is_empty()).unwrap_or(true)) {
        return true;
    }
    body.get("messages").and_then(Value::as_array).is_some_and(|ms| ms.iter().any(|m| m.get("role").and_then(Value::as_str) == Some("tool") || nonempty_array(m.get("tool_calls"))))
}

/// O pedido já preparado: as mensagens no formato do modelo e o que a resposta pode chamar.
#[derive(Debug)]
pub struct Plan {
    pub messages: Vec<ChatMsg>,
    pub fmt: ToolFormat,
    /// nomes que o modelo pode chamar (vazio = não ler chamadas na resposta)
    pub offered: Vec<String>,
    pub choice: ToolChoice,
}

impl Plan {
    pub fn parses(&self) -> bool {
        !self.offered.is_empty()
    }
}

/// Prepara um pedido com ferramentas. `raw` é o corpo bruto (para a ordem das chaves de `tools`).
pub fn plan(body: &Value, raw: &[u8], support: &ToolSupport) -> Result<Plan, String> {
    let fmt = match support {
        ToolSupport::Yes(f) => f.clone(),
        ToolSupport::No(why) => return Err(why.clone()),
    };
    let choice = parse_choice(body.get("tool_choice"))?;
    let has_tools = body.get("tools").is_some_and(|t| !t.is_null() && t.as_array().map(|a| !a.is_empty()).unwrap_or(true));
    let mut tools = if has_tools { parse_tools(std::str::from_utf8(raw).map_err(|_| "corpo não é UTF-8.".to_string())?)? } else { vec![] };
    match &choice {
        ToolChoice::None => tools.clear(), // o modelo nem vê as ferramentas
        ToolChoice::Required if tools.is_empty() => return Err("`tool_choice: required` precisa de `tools`.".into()),
        ToolChoice::Named(n) => {
            tools.retain(|t| &t.name == n);
            if tools.is_empty() {
                return Err(format!("`tool_choice` pede a função `{n}`, que não está em `tools`."));
            }
        }
        _ => {}
    }
    let messages = render_messages(&fmt, body.get("messages").unwrap_or(&Value::Null), &tools)?;
    Ok(Plan { messages, fmt, offered: tools.iter().map(|t| t.name.clone()).collect(), choice })
}

// ------------------------------------------------------------------ mensagens -> texto do template

enum M {
    Sys(String),
    User(String),
    Asst(String, Vec<RCall>),
    Tool(String),
}

struct RCall {
    name: String,
    /// `arguments` como o cliente mandou (texto JSON)
    args: String,
}

fn normalize(messages: &Value) -> Result<Vec<M>, String> {
    let arr = messages.as_array().ok_or("`messages` ausente.")?;
    let mut out = Vec::new();
    for m in arr {
        let content = crate::openai::content_text(m.get("content").unwrap_or(&Value::Null))?;
        out.push(match m.get("role").and_then(Value::as_str).unwrap_or("") {
            "system" | "developer" => M::Sys(content),
            "user" => M::User(content),
            "tool" => M::Tool(content),
            "assistant" => {
                let mut calls = Vec::new();
                for c in m.get("tool_calls").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]) {
                    let f = c.get("function").ok_or("`tool_calls[].function` ausente.")?;
                    let name = f.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).ok_or("`tool_calls[].function.name` ausente.")?;
                    let args = match f.get("arguments") {
                        None | Some(Value::Null) => "{}".to_string(),
                        Some(Value::String(s)) => s.clone(),
                        Some(other) => other.to_string(),
                    };
                    calls.push(RCall { name: name.to_string(), args });
                }
                M::Asst(content, calls)
            }
            other => return Err(format!("papel `{other}` não é suportado (use system, user, assistant ou tool).")),
        });
    }
    if out.is_empty() {
        return Err("`messages` vazio.".into());
    }
    Ok(out)
}

fn msg(role: &str, content: String) -> ChatMsg {
    ChatMsg { role: role.to_string(), content }
}

/// `arguments` (texto JSON do cliente) como o `tojson` do template o escreve: um objeto, com `, ` e `: `.
fn args_spaced(args: &str, call: &str) -> Result<String, String> {
    let a = args.trim();
    if a.is_empty() {
        return Ok("{}".into());
    }
    J::parse(a).map(|j| j.render(Style::Spaced)).map_err(|e| format!("`arguments` da chamada `{call}` não é JSON válido: {e}"))
}

/// Escreve as mensagens como o template do modelo. Com `tools` vazio, só o histórico (sem o bloco de ferramentas).
pub fn render_messages(fmt: &ToolFormat, messages: &Value, tools: &[Tool]) -> Result<Vec<ChatMsg>, String> {
    let msgs = normalize(messages)?;
    match fmt {
        ToolFormat::Hermes { qwen3, double_braces, default_system } => render_hermes(&msgs, tools, *qwen3, *double_braces, *default_system),
        ToolFormat::Llama3 => render_llama3(&msgs, tools),
    }
}

fn render_hermes(msgs: &[M], tools: &[Tool], qwen3: bool, double: bool, default_system: Option<&str>) -> Result<Vec<ChatMsg>, String> {
    let (sys, rest) = match msgs.first() {
        Some(M::Sys(s)) => (Some(s.as_str()), &msgs[1..]),
        _ => (None, msgs),
    };
    let mut out = Vec::new();
    if tools.is_empty() {
        // sem ferramentas o template só escreve o sistema (o padrão, se faltar, o tokenizador já põe)
        if let Some(s) = sys {
            out.push(msg("system", s.to_string()));
        }
    } else {
        let mut s = String::new();
        if let Some(p) = sys.or(default_system).filter(|p| !p.is_empty()) {
            s.push_str(p);
            s.push_str("\n\n");
        }
        s.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for t in tools {
            s.push('\n');
            s.push_str(&t.json.render(Style::Spaced));
        }
        s.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n");
        s.push_str(if double { "{{\"name\": <function-name>, \"arguments\": <args-json-object>}}" } else { "{\"name\": <function-name>, \"arguments\": <args-json-object>}" });
        s.push_str("\n</tool_call>");
        out.push(msg("system", s));
    }
    let mut i = 0;
    while i < rest.len() {
        match &rest[i] {
            M::Sys(s) => out.push(msg("system", s.clone())),
            M::User(s) => out.push(msg("user", s.clone())),
            M::Asst(c, calls) if calls.is_empty() => out.push(msg("assistant", c.clone())),
            M::Asst(c, calls) => {
                let mut s = c.clone();
                for call in calls {
                    if !s.is_empty() {
                        s.push('\n');
                    }
                    // o Qwen2.5 faz `arguments | tojson` (um objeto); o Qwen3 escreve o texto de `arguments` como veio
                    let args = if qwen3 { call.args.trim().to_string() } else { args_spaced(&call.args, &call.name)? };
                    let args = if args.is_empty() { "{}".to_string() } else { args };
                    s.push_str(&format!("<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>", call.name, args));
                }
                out.push(msg("assistant", s));
            }
            M::Tool(_) => {
                // respostas seguidas de ferramenta formam UM turno `user`
                let mut s = String::new();
                while let Some(M::Tool(c)) = rest.get(i) {
                    if !s.is_empty() {
                        s.push('\n');
                    }
                    s.push_str(&format!("<tool_response>\n{c}\n</tool_response>"));
                    i += 1;
                }
                out.push(msg("user", s));
                continue;
            }
        }
        i += 1;
    }
    Ok(out)
}

fn render_llama3(msgs: &[M], tools: &[Tool]) -> Result<Vec<ChatMsg>, String> {
    let (sys, rest) = match msgs.first() {
        Some(M::Sys(s)) => (Some(s.as_str()), &msgs[1..]),
        _ => (None, msgs),
    };
    let mut out = Vec::new();
    if tools.is_empty() {
        if let Some(s) = sys {
            out.push(msg("system", s.to_string()));
        }
    } else {
        // As ferramentas vão no bloco de sistema (como o llama.cpp renderiza o template do Llama 3.x: `tools_in_user_message=false`),
        // não na 1ª mensagem do usuário: não somem quando o contexto corta mensagens antigas e não exige uma mensagem de usuário.
        let mut s = String::from("You have access to the following functions. To call a function, please respond with JSON for a function call.Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name to its value}.Do not use variables.\n\n");
        for t in tools {
            s.push_str(&t.json.render(Style::Indent(4)));
            s.push_str("\n\n");
        }
        s.push_str(sys.unwrap_or("").trim());
        out.push(msg(TOOLS_SYSTEM_ROLE, s));
    }
    for m in rest {
        match m {
            M::Sys(s) => out.push(msg("system", s.clone())),
            M::User(s) => out.push(msg("user", s.clone())),
            M::Asst(c, calls) if calls.is_empty() => out.push(msg("assistant", c.clone())),
            M::Asst(_, calls) => {
                if calls.len() != 1 {
                    return Err("O template do Llama 3 só aceita UMA chamada de função por mensagem do assistente no histórico.".into());
                }
                let c = &calls[0];
                out.push(msg("assistant", format!("{{\"name\": \"{}\", \"parameters\": {}}}", c.name, args_spaced(&c.args, &c.name)?)));
            }
            // `message.content is iterable` vale também para texto no jinja: o template escreve `content | tojson` (texto entre aspas)
            M::Tool(c) => out.push(msg("ipython", serde_json::to_string(c).unwrap_or_default())),
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------ texto gerado -> tool_calls

#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub id: String,
    pub name: String,
    /// texto JSON (um objeto), como a OpenAI manda em `function.arguments`
    pub arguments: String,
}

/// O que o texto gerado virou. `calls` não vazio: chamadas válidas (e `content` é o texto fora delas).
/// `error`: o texto parecia uma chamada mas não passou na validação; `content` é o texto cru.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub content: Option<String>,
    pub calls: Vec<Call>,
    pub error: Option<String>,
}

impl Outcome {
    fn plain(text: &str) -> Outcome {
        Outcome { content: Some(text.to_string()), calls: vec![], error: None }
    }
    fn invalid(text: &str, why: String) -> Outcome {
        Outcome { content: Some(text.to_string()), calls: vec![], error: Some(why) }
    }
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const PYTHON_TAG: &str = "<|python_tag|>";

/// Uma chamada em JSON: `{"name": ..., "arguments"|"parameters": {...}}`. Confere o nome e que os argumentos são um objeto.
fn one_call(seg: &str, offered: &[String], new_id: &mut dyn FnMut() -> String) -> Result<Call, String> {
    let j = J::parse(seg).map_err(|e| format!("a chamada não é JSON válido ({e})"))?;
    if !matches!(j, J::Obj(_)) {
        return Err("a chamada não é um objeto JSON".into());
    }
    let name = j.get("name").and_then(J::as_str).ok_or("a chamada não tem `name`")?;
    if !offered.iter().any(|o| o == name) {
        return Err(format!("a função `{name}` não está em `tools`"));
    }
    let arguments = match j.get("arguments").or_else(|| j.get("parameters")) {
        None | Some(J::Null) => "{}".to_string(),
        Some(a @ J::Obj(_)) => a.render(Style::Compact),
        Some(J::Str(s)) => match J::parse(s) {
            Ok(a @ J::Obj(_)) => a.render(Style::Compact),
            _ => return Err(format!("`arguments` de `{name}` não é um objeto JSON válido")),
        },
        Some(_) => return Err(format!("`arguments` de `{name}` deve ser um objeto JSON")),
    };
    Ok(Call { id: new_id(), name: name.to_string(), arguments })
}

fn parse_hermes(text: &str, offered: &[String], new_id: &mut dyn FnMut() -> String) -> Outcome {
    let Some(first) = text.find(OPEN) else { return Outcome::plain(text) };
    let mut content = text[..first].trim_end().to_string();
    let mut calls = Vec::new();
    let mut rest = &text[first..];
    while let Some(p) = rest.find(OPEN) {
        let between = rest[..p].trim();
        if !between.is_empty() {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(between);
        }
        let after = &rest[p + OPEN.len()..];
        // sem `</tool_call>` (fim de geração logo depois do JSON) o JSON completo ainda vale
        let (seg, next) = match after.find(CLOSE) {
            Some(e) => (&after[..e], &after[e + CLOSE.len()..]),
            None => (after, ""),
        };
        match one_call(seg.trim(), offered, new_id) {
            Ok(c) => calls.push(c),
            Err(e) => return Outcome::invalid(text, e),
        }
        rest = next;
    }
    let tail = rest.trim();
    if !tail.is_empty() {
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(tail);
    }
    Outcome { content: if content.is_empty() { None } else { Some(content) }, calls, error: None }
}

fn parse_llama3(text: &str, offered: &[String], new_id: &mut dyn FnMut() -> String) -> Outcome {
    let mut t = text.trim();
    let tagged = t.starts_with(PYTHON_TAG);
    if tagged {
        t = t[PYTHON_TAG.len()..].trim_start();
    }
    for end in ["<|eom_id|>", "<|eot_id|>"] {
        t = t.strip_suffix(end).unwrap_or(t).trim_end();
    }
    if !t.starts_with('{') {
        return if tagged { Outcome::invalid(text, "depois de `<|python_tag|>` não veio um objeto JSON (ferramentas embutidas do Llama não são suportadas)".into()) } else { Outcome::plain(text) };
    }
    match J::parse(t) {
        Err(e) => Outcome::invalid(text, format!("a chamada não é JSON válido ({e})")),
        // JSON válido sem `name`: é só texto (por exemplo, o usuário pediu JSON)
        Ok(j) if !tagged && j.get("name").is_none() => Outcome::plain(text),
        Ok(_) => match one_call(t, offered, new_id) {
            Ok(c) => Outcome { content: None, calls: vec![c], error: None },
            Err(e) => Outcome::invalid(text, e),
        },
    }
}

/// Lê o texto gerado como chamadas de função do formato do modelo. Só aceita nomes de `offered`.
pub fn parse_output(fmt: &ToolFormat, text: &str, offered: &[String], new_id: &mut dyn FnMut() -> String) -> Outcome {
    match fmt {
        ToolFormat::Hermes { .. } => parse_hermes(text, offered, new_id),
        ToolFormat::Llama3 => parse_llama3(text, offered, new_id),
    }
}

/// `tool_calls` no formato da OpenAI. No fluxo (`stream`) cada item leva `index`.
pub fn calls_value(calls: &[Call], stream: bool) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut v = json!({ "id": c.id, "type": "function", "function": { "name": c.name, "arguments": c.arguments } });
                if stream {
                    v["index"] = json!(i);
                }
                v
            })
            .collect(),
    )
}

// ------------------------------------------------------------------ fluxo

#[derive(PartialEq, Clone, Copy, Debug)]
enum Mode {
    /// ainda não dá para saber se é texto ou chamada (Llama: só espaços até agora)
    Undecided,
    /// é texto: sai na hora
    Pass,
    /// é (ou pode ser) uma chamada: guarda até o fim
    Buffer,
}

/// Monta a resposta em fluxo: o texto comum sai ao vivo; a partir do início de uma chamada (`<tool_call>` no Qwen; `{`
/// ou `<|python_tag|>` no começo da resposta do Llama) o texto fica guardado até o fim, e a chamada sai INTEIRA num bloco
/// (o `arguments` completo, sem fatiar em pedaços) quando o modelo termina. Nada é emitido antes de saber se o JSON vale.
pub struct ToolStream {
    fmt: ToolFormat,
    offered: Vec<String>,
    all: String,
    /// quantos bytes de `all` já saíram como `content`
    sent: usize,
    mode: Mode,
}

pub struct Done {
    /// texto que ainda falta sair como `content`
    pub tail: String,
    pub outcome: Outcome,
}

impl ToolStream {
    pub fn new(fmt: &ToolFormat, offered: &[String]) -> ToolStream {
        let mode = match fmt {
            ToolFormat::Hermes { .. } => Mode::Pass,
            ToolFormat::Llama3 => Mode::Undecided,
        };
        ToolStream { fmt: fmt.clone(), offered: offered.to_vec(), all: String::new(), sent: 0, mode }
    }

    /// Recebe um pedaço de texto; devolve o que já pode sair como `content`.
    pub fn feed(&mut self, t: &str) -> String {
        self.all.push_str(t);
        match self.fmt {
            ToolFormat::Hermes { .. } => self.feed_hermes(),
            ToolFormat::Llama3 => self.feed_llama(),
        }
    }

    fn feed_hermes(&mut self) -> String {
        if self.mode == Mode::Buffer {
            return String::new();
        }
        let pending = &self.all[self.sent..];
        if let Some(p) = pending.find(OPEN) {
            let emit = pending[..p].trim_end().to_string();
            self.sent += emit.len();
            self.mode = Mode::Buffer;
            return emit;
        }
        // segura o fim que ainda pode virar `<tool_call>` e os espaços finais (se vier uma chamada, o texto antes dela é aparado)
        let mut keep = 0;
        for n in (1..=OPEN.len().min(pending.len())).rev() {
            let from = pending.len() - n;
            if pending.is_char_boundary(from) && OPEN.starts_with(&pending[from..]) {
                keep = n;
                break;
            }
        }
        let emit = pending[..pending.len() - keep].trim_end().to_string();
        self.sent += emit.len();
        emit
    }

    fn feed_llama(&mut self) -> String {
        if self.mode == Mode::Undecided {
            let t = self.all.trim_start();
            if t.is_empty() {
                return String::new();
            }
            self.mode = if t.starts_with('{') || t.starts_with(PYTHON_TAG) || PYTHON_TAG.starts_with(t) { Mode::Buffer } else { Mode::Pass };
        }
        if self.mode == Mode::Pass {
            let emit = self.all[self.sent..].to_string();
            self.sent = self.all.len();
            return emit;
        }
        String::new()
    }

    /// O modelo terminou: lê o texto inteiro e diz o que falta emitir.
    pub fn finish(&mut self, new_id: &mut dyn FnMut() -> String) -> Done {
        let outcome = parse_output(&self.fmt, &self.all, &self.offered, new_id);
        let tail = if outcome.calls.is_empty() {
            self.all[self.sent..].to_string()
        } else {
            let c = outcome.content.as_deref().unwrap_or("");
            if c.starts_with(&self.all[..self.sent]) {
                c[self.sent..].to_string()
            } else {
                String::new()
            }
        };
        Done { tail, outcome }
    }
}

// ------------------------------------------------------------------ testes

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN25: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

    fn hermes(qwen3: bool, double: bool) -> ToolFormat {
        ToolFormat::Hermes { qwen3, double_braces: double, default_system: if qwen3 { None } else { Some(QWEN25) } }
    }

    /// o que o tokenizador faz com ChatML (`chat_prompt`), em texto
    fn chatml(msgs: &[ChatMsg], default_sys: Option<&str>) -> String {
        let mut s = String::new();
        if let (Some(d), true) = (default_sys, msgs.first().map(|m| m.role.as_str()) != Some("system")) {
            s.push_str(&format!("<|im_start|>system\n{d}<|im_end|>\n"));
        }
        for m in msgs {
            s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
        }
        s.push_str("<|im_start|>assistant\n");
        s
    }

    fn weather_body() -> String {
        // chaves em ordem "não alfabética" de propósito: o template as escreve na ordem do cliente
        r#"{"tools":[{"type":"function","function":{"name":"get_weather","description":"Clima de uma cidade","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"messages":[{"role":"user","content":"Clima em Paris?"}]}"#.to_string()
    }

    const WEATHER_TOOL_JSON: &str = r#"{"type": "function", "function": {"name": "get_weather", "description": "Clima de uma cidade", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}"#;

    fn ids() -> impl FnMut() -> String {
        let mut n = 0;
        move || {
            n += 1;
            format!("call_{n}")
        }
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // ---- JSON com ordem

    #[test]
    fn ordered_json_roundtrip_and_styles() {
        let src = r#"{"z": 1, "a": [true, null, "x\"yé"], "m": {"k": 1.50, "e": []}, "o": {}}"#;
        let j = J::parse(src).unwrap();
        assert_eq!(j.render(Style::Compact), r#"{"z":1,"a":[true,null,"x\"yé"],"m":{"k":1.50,"e":[]},"o":{}}"#);
        assert_eq!(j.render(Style::Spaced), r#"{"z": 1, "a": [true, null, "x\"yé"], "m": {"k": 1.50, "e": []}, "o": {}}"#);
        // tojson(indent=4)
        assert_eq!(
            J::parse(r#"{"b":[1,2],"a":{"c":null},"e":[],"f":{}}"#).unwrap().render(Style::Indent(4)),
            "{\n    \"b\": [\n        1,\n        2\n    ],\n    \"a\": {\n        \"c\": null\n    },\n    \"e\": [],\n    \"f\": {}\n}"
        );
        assert_eq!(J::parse("[]").unwrap().render(Style::Indent(4)), "[]");
        // a última chave repetida vale
        assert_eq!(J::parse(r#"{"a":1,"a":2}"#).unwrap().get("a"), Some(&J::Num("2".into())));
    }

    #[test]
    fn ordered_json_rejects_garbage() {
        for bad in ["", "{", "{\"a\":}", "{\"a\":1,}", "[1,]", "{\"a\":1} x", "'a'", "{\"a\":\"x}", "01a", "{\"a\" 1}", "tru", "[1 2]"] {
            assert!(J::parse(bad).is_err(), "devia recusar {bad:?}");
        }
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        assert!(J::parse(&deep).is_err(), "aninhamento profundo demais");
        // `parse_members` guarda só o que pediu, mas confere o resto
        let m = J::parse_members(r#"{"messages":[{"content":"é"}],"tools":[1],"n":2}"#, &["tools"]).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, "tools");
        assert!(J::parse_members(r#"{"messages":[{"content":"x"}"#, &["tools"]).is_err());
    }

    // ---- tools e tool_choice

    #[test]
    fn parses_tools_and_rejects_bad_ones() {
        let t = parse_tools(&weather_body()).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].name, "get_weather");
        // ordem do cliente preservada (o serde_json do projeto ordenaria as chaves)
        assert_eq!(t[0].json.render(Style::Spaced), WEATHER_TOOL_JSON);
        for bad in [
            r#"{"tools":[{"type":"retrieval"}]}"#,
            r#"{"tools":[{"type":"function"}]}"#,
            r#"{"tools":[{"type":"function","function":{"name":"has space"}}]}"#,
            r#"{"tools":[{"type":"function","function":{"name":"a","parameters":[1]}}]}"#,
            r#"{"tools":[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"a"}}]}"#,
            r#"{"tools":"x"}"#,
        ] {
            assert!(parse_tools(bad).is_err(), "devia recusar {bad}");
        }
        assert!(parse_tools(r#"{"tools":null}"#).unwrap().is_empty());
        assert!(parse_tools(r#"{"messages":[]}"#).unwrap().is_empty());
    }

    #[test]
    fn tool_choice_variants() {
        let v = |s: &str| serde_json::from_str::<Value>(s).unwrap();
        assert_eq!(parse_choice(None).unwrap(), ToolChoice::Auto);
        assert_eq!(parse_choice(Some(&v("null"))).unwrap(), ToolChoice::Auto);
        assert_eq!(parse_choice(Some(&v("\"auto\""))).unwrap(), ToolChoice::Auto);
        assert_eq!(parse_choice(Some(&v("\"none\""))).unwrap(), ToolChoice::None);
        assert_eq!(parse_choice(Some(&v("\"required\""))).unwrap(), ToolChoice::Required);
        assert_eq!(parse_choice(Some(&v(r#"{"type":"function","function":{"name":"f"}}"#))).unwrap(), ToolChoice::Named("f".into()));
        for bad in ["\"sempre\"", "7", r#"{"type":"function"}"#, r#"{"type":"x","function":{"name":"f"}}"#] {
            assert!(parse_choice(Some(&v(bad))).is_err(), "devia recusar {bad}");
        }
    }

    fn two_tools_body(choice: &str) -> (Value, String) {
        let raw = format!(
            r#"{{"tools":[{{"type":"function","function":{{"name":"a","parameters":{{"type":"object"}}}}}},{{"type":"function","function":{{"name":"b"}}}}],{choice}"messages":[{{"role":"user","content":"oi"}}]}}"#
        );
        (serde_json::from_str(&raw).unwrap(), raw)
    }

    #[test]
    fn plan_handles_tool_choice() {
        let sup = ToolSupport::Yes(hermes(false, true));
        let run = |choice: &str| {
            let (b, raw) = two_tools_body(choice);
            plan(&b, raw.as_bytes(), &sup)
        };
        let auto = run("").unwrap();
        assert_eq!(auto.offered, names(&["a", "b"]));
        assert!(auto.parses());
        assert!(auto.messages[0].content.contains("\"name\": \"a\"") && auto.messages[0].content.contains("\"name\": \"b\""));
        // none: o modelo nem vê as ferramentas e a resposta não é lida como chamada
        let none = run(r#""tool_choice":"none","#).unwrap();
        assert!(none.offered.is_empty() && !none.parses());
        assert!(none.messages.iter().all(|m| !m.content.contains("<tools>")));
        // required: as ferramentas valem (obrigar não dá, sem decodificação restrita)
        assert_eq!(run(r#""tool_choice":"required","#).unwrap().offered, names(&["a", "b"]));
        // nomeada: só essa ferramenta aparece no prompt
        let named = run(r#""tool_choice":{"type":"function","function":{"name":"b"}},"#).unwrap();
        assert_eq!(named.offered, names(&["b"]));
        assert!(!named.messages[0].content.contains("\"name\": \"a\""));
        assert!(run(r#""tool_choice":{"type":"function","function":{"name":"zzz"}},"#).unwrap_err().contains("zzz"));
        assert!(run(r#""tool_choice":"talvez","#).is_err());
        // required sem ferramentas
        let b: Value = serde_json::from_str(r#"{"tool_choice":"required","messages":[{"role":"tool","content":"x"}]}"#).unwrap();
        assert!(plan(&b, b"{}", &sup).is_err());
    }

    #[test]
    fn plan_refuses_models_without_tools_with_a_recommendation() {
        let (b, raw) = two_tools_body("");
        let e = plan(&b, raw.as_bytes(), &ToolSupport::no_template("gemma3")).unwrap_err();
        assert!(e.contains("gemma3") && e.contains("Qwen2.5") && e.contains("Llama 3"), "{e}");
        let e = plan(&b, raw.as_bytes(), &ToolSupport::llama_cpp()).unwrap_err();
        assert!(e.contains("motor próprio") && e.contains("Qwen2.5"), "{e}");
        assert!(ToolSupport::not_implemented("Mistral").describe()["supported"] == false);
        assert_eq!(ToolSupport::Yes(ToolFormat::Llama3).describe()["valid_json_guaranteed"], false);
    }

    #[test]
    fn wants_tools_detection() {
        let v = |s: &str| serde_json::from_str::<Value>(s).unwrap();
        assert!(!wants_tools(&v(r#"{"messages":[{"role":"user","content":"x"}]}"#)));
        assert!(!wants_tools(&v(r#"{"tools":[],"messages":[]}"#)));
        assert!(!wants_tools(&v(r#"{"tools":null}"#)));
        assert!(wants_tools(&v(r#"{"tools":[{"type":"function"}]}"#)));
        assert!(wants_tools(&v(r#"{"messages":[{"role":"tool","content":"x"}]}"#)));
        assert!(wants_tools(&v(r#"{"messages":[{"role":"assistant","tool_calls":[{}]}]}"#)));
        assert!(!wants_tools(&v(r#"{"messages":[{"role":"assistant","tool_calls":[]}]}"#)));
    }

    // ---- renderização: o texto que o template do Qwen2.5 (GGUF do Qwen) produz

    const QWEN_TOOLS_HEAD: &str = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
    const QWEN_TOOLS_TAIL_DOUBLE: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n";

    #[test]
    fn qwen25_first_turn_matches_the_template() {
        let body: Value = serde_json::from_str(&weather_body()).unwrap();
        let p = plan(&body, weather_body().as_bytes(), &ToolSupport::Yes(hermes(false, true))).unwrap();
        let expected = format!("{QWEN_TOOLS_HEAD}{WEATHER_TOOL_JSON}{QWEN_TOOLS_TAIL_DOUBLE}<|im_start|>user\nClima em Paris?<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(chatml(&p.messages, Some(QWEN25)), expected);
        // com chaves simples (template do Hugging Face) só muda o exemplo do formato
        let p1 = plan(&body, weather_body().as_bytes(), &ToolSupport::Yes(hermes(false, false))).unwrap();
        assert!(chatml(&p1.messages, Some(QWEN25)).contains("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>"));
    }

    #[test]
    fn qwen25_uses_the_client_system_and_two_tools() {
        let raw = r#"{"tools":[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"b"}}],"messages":[{"role":"system","content":"Seja breve."},{"role":"user","content":"oi"}]}"#;
        let body: Value = serde_json::from_str(raw).unwrap();
        let p = plan(&body, raw.as_bytes(), &ToolSupport::Yes(hermes(false, false))).unwrap();
        let t = chatml(&p.messages, Some(QWEN25));
        assert!(t.starts_with("<|im_start|>system\nSeja breve.\n\n# Tools\n\n"), "{t}");
        assert!(t.contains("<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"a\"}}\n{\"type\": \"function\", \"function\": {\"name\": \"b\"}}\n</tools>"), "{t}");
        assert_eq!(p.messages.len(), 2, "o sistema do cliente não aparece duas vezes");
    }

    #[test]
    fn qwen3_has_no_default_system() {
        let raw = r#"{"tools":[{"type":"function","function":{"name":"a"}}],"messages":[{"role":"user","content":"oi"}]}"#;
        let body: Value = serde_json::from_str(raw).unwrap();
        let p = plan(&body, raw.as_bytes(), &ToolSupport::Yes(hermes(true, false))).unwrap();
        assert!(p.messages[0].content.starts_with("# Tools\n\nYou may call"), "{}", p.messages[0].content);
    }

    #[test]
    fn qwen25_multi_turn_history() {
        // user -> assistant (2 chamadas, com texto antes) -> 2 respostas de ferramenta -> user
        let raw = r#"{"tools":[{"type":"function","function":{"name":"f"}}],"messages":[
            {"role":"user","content":"u1"},
            {"role":"assistant","content":"Vou ver.","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"f","arguments":"{\"b\":1,\"a\":{\"x\":[1,2]}}"}},
                {"id":"c2","type":"function","function":{"name":"f","arguments":""}}]},
            {"role":"tool","tool_call_id":"c1","content":"r1"},
            {"role":"tool","tool_call_id":"c2","content":"r2"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c3","type":"function","function":{"name":"f","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c3","content":"r3"},
            {"role":"assistant","content":"pronto"},
            {"role":"user","content":"u2"}]}"#;
        let body: Value = serde_json::from_str(raw).unwrap();
        let p = plan(&body, raw.as_bytes(), &ToolSupport::Yes(hermes(false, true))).unwrap();
        let t = chatml(&p.messages[1..], None); // sem o bloco de sistema (testado acima)
        let want = "<|im_start|>user\nu1<|im_end|>\n\
<|im_start|>assistant\nVou ver.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"b\": 1, \"a\": {\"x\": [1, 2]}}}\n</tool_call>\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><|im_end|>\n\
<|im_start|>user\n<tool_response>\nr1\n</tool_response>\n<tool_response>\nr2\n</tool_response><|im_end|>\n\
<|im_start|>assistant\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><|im_end|>\n\
<|im_start|>user\n<tool_response>\nr3\n</tool_response><|im_end|>\n\
<|im_start|>assistant\npronto<|im_end|>\n\
<|im_start|>user\nu2<|im_end|>\n\
<|im_start|>assistant\n";
        assert_eq!(t, want);
        // Qwen3 escreve `arguments` como veio (texto), sem reformatar
        let p3 = plan(&body, raw.as_bytes(), &ToolSupport::Yes(hermes(true, false))).unwrap();
        assert!(p3.messages.iter().any(|m| m.content.contains("\"arguments\": {\"b\":1,\"a\":{\"x\":[1,2]}}}")), "{:?}", p3.messages);
    }

    #[test]
    fn history_without_tools_and_bad_history() {
        // histórico com resposta de ferramenta, sem `tools` (ou com tool_choice none): sem bloco <tools>, histórico ok
        let raw = r#"{"tool_choice":"none","messages":[{"role":"user","content":"u"},{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","content":"r"}]}"#;
        let body: Value = serde_json::from_str(raw).unwrap();
        let p = plan(&body, raw.as_bytes(), &ToolSupport::Yes(hermes(false, true))).unwrap();
        assert!(!p.parses());
        assert_eq!(p.messages[0].role, "user", "sem ferramentas o sistema padrão fica para o tokenizador");
        assert!(p.messages[1].content.starts_with("<tool_call>"));
        // `arguments` que não é JSON no histórico: erro claro, não um prompt torto
        let bad = r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{nao"}}]}]}"#;
        let b: Value = serde_json::from_str(bad).unwrap();
        let e = plan(&b, bad.as_bytes(), &ToolSupport::Yes(hermes(false, true))).unwrap_err();
        assert!(e.contains("arguments") && e.contains("`f`"), "{e}");
        // papel desconhecido
        let b: Value = serde_json::from_str(r#"{"messages":[{"role":"function","content":"x"}]}"#).unwrap();
        assert!(normalize(&b["messages"]).is_err());
    }

    #[test]
    fn llama3_rendering() {
        let body: Value = serde_json::from_str(&weather_body()).unwrap();
        let p = plan(&body, weather_body().as_bytes(), &ToolSupport::Yes(ToolFormat::Llama3)).unwrap();
        assert_eq!(p.messages[0].role, TOOLS_SYSTEM_ROLE);
        let pretty = J::parse(WEATHER_TOOL_JSON).unwrap().render(Style::Indent(4));
        assert_eq!(
            p.messages[0].content,
            format!("You have access to the following functions. To call a function, please respond with JSON for a function call.Respond in the format {{\"name\": function name, \"parameters\": dictionary of argument name to its value}}.Do not use variables.\n\n{pretty}\n\n")
        );
        assert!(pretty.starts_with("{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"get_weather\""), "{pretty}");
        assert_eq!(p.messages[1].role, "user");
        // histórico: uma chamada do assistente e o resultado como `ipython` (texto entre aspas, como `content | tojson`)
        let raw = r#"{"tools":[{"type":"function","function":{"name":"f"}}],"messages":[{"role":"system","content":" Seja breve. "},{"role":"user","content":"u"},
            {"role":"assistant","content":"ignorado","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{\"q\":\"a\"}"}}]},
            {"role":"tool","tool_call_id":"1","content":"{\"t\":72}"}]}"#;
        let b: Value = serde_json::from_str(raw).unwrap();
        let p = plan(&b, raw.as_bytes(), &ToolSupport::Yes(ToolFormat::Llama3)).unwrap();
        assert!(p.messages[0].content.ends_with("}\n\nSeja breve."), "o sistema do cliente é aparado e vem depois das ferramentas: {:?}", p.messages[0].content);
        assert_eq!((p.messages[2].role.as_str(), p.messages[2].content.as_str()), ("assistant", "{\"name\": \"f\", \"parameters\": {\"q\": \"a\"}}"));
        assert_eq!((p.messages[3].role.as_str(), p.messages[3].content.as_str()), ("ipython", "\"{\\\"t\\\":72}\""));
        // duas chamadas numa mensagem: o template do Llama não aceita
        let two = r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"a","arguments":"{}"}},{"function":{"name":"b","arguments":"{}"}}]}]}"#;
        let b: Value = serde_json::from_str(two).unwrap();
        assert!(plan(&b, two.as_bytes(), &ToolSupport::Yes(ToolFormat::Llama3)).is_err());
    }

    // ---- análise da saída

    fn parse(fmt: &ToolFormat, text: &str, offered: &[&str]) -> Outcome {
        parse_output(fmt, text, &names(offered), &mut ids())
    }

    #[test]
    fn hermes_parses_good_calls() {
        let f = hermes(false, true);
        let o = parse(&f, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"u\": 1.5}}\n</tool_call>", &["get_weather"]);
        assert_eq!(o.error, None);
        assert_eq!(o.content, None);
        assert_eq!(o.calls, vec![Call { id: "call_1".into(), name: "get_weather".into(), arguments: "{\"city\":\"Paris\",\"u\":1.5}".into() }]);
        // texto antes, duas chamadas, texto depois
        let o = parse(&f, "Vou consultar.\n<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call>\n<tool_call>\n{\"name\":\"b\",\"arguments\":{\"x\":[1]}}\n</tool_call>\nFim.", &["a", "b"]);
        assert_eq!(o.error, None);
        assert_eq!(o.content.as_deref(), Some("Vou consultar.\nFim."));
        assert_eq!(o.calls.iter().map(|c| (c.id.as_str(), c.name.as_str(), c.arguments.as_str())).collect::<Vec<_>>(), vec![("call_1", "a", "{}"), ("call_2", "b", "{\"x\":[1]}")]);
        // `arguments` como texto JSON, ou ausente
        let o = parse(&f, "<tool_call>{\"name\":\"a\",\"arguments\":\"{\\\"k\\\": 2}\"}</tool_call>", &["a"]);
        assert_eq!(o.calls[0].arguments, "{\"k\":2}");
        let o = parse(&f, "<tool_call>{\"name\":\"a\"}</tool_call>", &["a"]);
        assert_eq!(o.calls[0].arguments, "{}");
        // sem `</tool_call>` mas com o JSON completo (o motor parou ali): vale
        let o = parse(&f, "<tool_call>\n{\"name\":\"a\",\"arguments\":{\"k\":1}}", &["a"]);
        assert_eq!((o.error, o.calls.len()), (None, 1));
        // texto sem chamada
        let o = parse(&f, "Paris está a 20 graus.", &["a"]);
        assert_eq!((o.content.as_deref(), o.calls.len(), o.error), (Some("Paris está a 20 graus."), 0, None));
    }

    #[test]
    fn hermes_never_invents_a_call() {
        let f = hermes(false, true);
        let cases = [
            ("<tool_call>\n{\"name\": \"a\", \"arguments\": {\"city\": \n</tool_call>", "JSON inválido"),
            ("<tool_call>\n{\"name\": \"a\", \"arguments\": {\"city\": \"Pa", "JSON inválido"), // cortado (max_tokens)
            ("<tool_call>\n{\"name\": \"outra\", \"arguments\": {}}\n</tool_call>", "não está em `tools`"),
            ("<tool_call>\n{\"arguments\": {}}\n</tool_call>", "name"),
            ("<tool_call>\n{\"name\": \"a\", \"arguments\": [1]}\n</tool_call>", "objeto"),
            ("<tool_call>\n{\"name\": \"a\", \"arguments\": \"nao json\"}\n</tool_call>", "objeto JSON válido"),
            ("<tool_call>\n[1]\n</tool_call>", "objeto"),
            // uma boa e uma ruim: tudo ou nada
            ("<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{ruim}</tool_call>", "JSON inválido"),
        ];
        for (text, why) in cases {
            let o = parse(&f, text, &["a"]);
            assert!(o.calls.is_empty(), "{text}");
            assert_eq!(o.content.as_deref(), Some(text), "o texto cru volta como content");
            assert!(o.error.as_deref().unwrap_or("").contains(why), "{text}: {:?}", o.error);
        }
    }

    #[test]
    fn llama3_parses_calls() {
        let f = ToolFormat::Llama3;
        let o = parse(&f, "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}", &["get_weather"]);
        assert_eq!((o.content.clone(), o.error.clone()), (None, None));
        assert_eq!(o.calls[0].arguments, "{\"city\":\"Paris\"}");
        // etiqueta de python e marcadores de fim em texto
        let o = parse(&f, "<|python_tag|>{\"name\":\"a\",\"parameters\":{}}<|eom_id|>", &["a"]);
        assert_eq!((o.calls.len(), o.error), (1, None));
        let o = parse(&f, "  {\"name\":\"a\",\"arguments\":{\"k\":1}}  \n", &["a"]);
        assert_eq!(o.calls[0].arguments, "{\"k\":1}");
        // texto comum, e JSON válido que não é chamada (o usuário pediu JSON): texto, sem aviso
        for text in ["Claro! Paris faz 20 graus.", "{\"city\": \"Paris\"}", ""] {
            let o = parse(&f, text, &["a"]);
            assert_eq!((o.content.as_deref(), o.calls.len(), o.error.is_some()), (Some(text), 0, false), "{text}");
        }
        // inválidos: texto cru + aviso
        for text in ["{\"name\": \"a\", \"parameters\": {\"x\": ", "{\"name\": \"zzz\", \"parameters\": {}}", "{\"name\": \"a\", \"parameters\": 3}", "<|python_tag|>brave_search.call(query=\"x\")"] {
            let o = parse(&f, text, &["a"]);
            assert!(o.calls.is_empty() && o.error.is_some(), "{text}");
            assert_eq!(o.content.as_deref(), Some(text));
        }
    }

    #[test]
    fn calls_value_shapes() {
        let c = vec![Call { id: "call_1".into(), name: "f".into(), arguments: "{}".into() }];
        assert_eq!(calls_value(&c, false), json!([{ "id": "call_1", "type": "function", "function": { "name": "f", "arguments": "{}" } }]));
        assert_eq!(calls_value(&c, true)[0]["index"], 0);
    }

    // ---- fluxo

    /// Alimenta o `ToolStream` com `pieces` e junta tudo o que sairia como `content` (ao vivo + o resto do fim).
    fn run_stream(fmt: &ToolFormat, pieces: &[&str], offered: &[&str]) -> (String, Vec<Vec<String>>, Outcome) {
        let mut s = ToolStream::new(fmt, &names(offered));
        let mut emitted = Vec::new();
        let mut live = String::new();
        for p in pieces {
            let e = s.feed(p);
            live.push_str(&e);
            emitted.push(vec![e]);
        }
        let d = s.finish(&mut ids());
        live.push_str(&d.tail);
        (live, emitted, d.outcome)
    }

    fn pieces_at<'a>(text: &'a str, cut: usize) -> Vec<&'a str> {
        vec![&text[..cut], &text[cut..]]
    }

    #[test]
    fn stream_assembly_is_independent_of_how_the_text_is_cut() {
        let f = hermes(false, true);
        let text = "Vou ver.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {\"k\": \"é\"}}\n</tool_call>";
        for cut in (0..=text.len()).filter(|c| text.is_char_boundary(*c)) {
            let (content, emitted, o) = run_stream(&f, &pieces_at(text, cut), &["a"]);
            assert_eq!(content, "Vou ver.", "corte em {cut}");
            assert_eq!(o.calls.len(), 1, "corte em {cut}");
            assert!(emitted.iter().flatten().all(|e| !e.contains("<tool") && !e.contains("</")), "a marca da chamada nunca vaza ao vivo (corte {cut}): {emitted:?}");
        }
        // um token por vez (cada caractere)
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
        let (content, emitted, o) = run_stream(&f, &pieces, &["a"]);
        assert_eq!((content.as_str(), o.calls.len()), ("Vou ver.", 1));
        // o texto antes da chamada saiu ao vivo (antes do fim), sem o "\n" final
        assert_eq!(emitted.iter().flatten().cloned().collect::<String>(), "Vou ver.");
    }

    #[test]
    fn stream_plain_text_flows_live_and_keeps_whitespace() {
        let f = hermes(false, true);
        let text = "Olá <mundo>, 1 < 2 e <tool> não é chamada.\n\nFim ";
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
        let (content, emitted, o) = run_stream(&f, &pieces, &["a"]);
        assert_eq!(content, text, "nada se perde nem se duplica");
        assert!(o.calls.is_empty() && o.error.is_none());
        // ao vivo: a maior parte já saiu antes do fim
        assert!(emitted.iter().flatten().map(String::len).sum::<usize>() > text.len() - 12);
    }

    #[test]
    fn stream_invalid_call_comes_back_as_raw_content() {
        let f = hermes(false, true);
        let text = "Certo.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {\"k\": ";
        for cut in (0..=text.len()).filter(|c| text.is_char_boundary(*c)) {
            let (content, _, o) = run_stream(&f, &pieces_at(text, cut), &["a"]);
            assert_eq!(content, text, "corte em {cut}: o texto cru inteiro volta, uma vez só");
            assert!(o.calls.is_empty() && o.error.is_some());
        }
    }

    #[test]
    fn stream_llama_decides_by_the_first_characters() {
        let f = ToolFormat::Llama3;
        // chamada: nada sai ao vivo
        let text = "{\"name\": \"a\", \"parameters\": {\"k\": 1}}";
        for cut in 0..=text.len() {
            let (content, emitted, o) = run_stream(&f, &pieces_at(text, cut), &["a"]);
            assert_eq!((content.as_str(), o.calls.len()), ("", 1), "corte em {cut}");
            assert!(emitted.iter().flatten().all(String::is_empty));
        }
        // com etiqueta, mesmo cortada no meio dela
        let (content, _, o) = run_stream(&f, &["<|pyth", "on_tag|>{\"name\":\"a\",\"parameters\":{}}"], &["a"]);
        assert_eq!((content.as_str(), o.calls.len()), ("", 1));
        // texto: sai ao vivo (inclusive um "<" que podia ser a etiqueta, no fim)
        let (content, emitted, o) = run_stream(&f, &["  ", "Claro", ", sim ", "<b>"], &["a"]);
        assert_eq!(content, "  Claro, sim <b>");
        assert!(o.calls.is_empty() && emitted.iter().flatten().filter(|e| !e.is_empty()).count() >= 3);
        // JSON inválido: tudo volta como texto
        let (content, _, o) = run_stream(&f, &["{\"name\": \"a\", ", "\"para"], &["a"]);
        assert_eq!(content, "{\"name\": \"a\", \"para");
        assert!(o.error.is_some());
    }

    // ---- ponta a ponta com o modelo (ignorado: precisa do GGUF e demora)

    /// Pergunta ao modelo real e devolve o que a API devolveria. `BURITI_TEST_MODEL` (padrão: o Qwen2.5 0.5B da pasta de modelos).
    async fn ask_model(engine: &crate::native::NativeEngine, body: &Value, raw: &str) -> (String, Outcome) {
        use crate::engine::{sse_events, ChatRequest, Engine};
        use futures_util::StreamExt;
        let sup = engine.tool_support();
        let p = plan(body, raw.as_bytes(), &sup).expect("plano");
        let req = ChatRequest { messages: p.messages.clone(), temperature: Some(0.0), max_tokens: Some(160), stop: vec![], sampling: Default::default(), images: vec![], constraint: None };
        let mut ev = Box::pin(sse_events(engine.chat(req).await.expect("chat")));
        let mut text = String::new();
        while let Some(v) = ev.next().await {
            if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                text.push_str(t);
            }
        }
        let o = parse_output(&p.fmt, &text, &p.offered, &mut ids());
        (text, o)
    }

    #[test]
    #[ignore]
    fn e2e_real_model_tool_calling() {
        use crate::engine::Engine;
        let path = std::env::var("BURITI_TEST_MODEL").unwrap_or_else(|_| format!("{}/buriti-data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", env!("CARGO_MANIFEST_DIR")));
        if !std::path::Path::new(&path).exists() {
            eprintln!("modelo ausente, pulando: {path}");
            return;
        }
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let engine = crate::native::NativeEngine::new(crate::config::Paths::from_env());
            let s = crate::config::Settings { model_path: path.clone(), engine_kind: "native".into(), processor: "cpu".into(), ctx_size: 2048, ..Default::default() };
            engine.start(&s).await.expect("start");
            let t0 = std::time::Instant::now();
            while engine.status().await.state != "ready" {
                assert!(t0.elapsed().as_secs() < 300, "modelo não carregou");
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
            println!("modelo: {path}\nformato: {:?}", engine.tool_support());
            let tools = r#"[{"type":"function","function":{"name":"get_weather","description":"Get the current weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"}},"required":["city"]}}},{"type":"function","function":{"name":"add","description":"Add two numbers","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}}]"#;
            // (pergunta, função esperada, argumentos esperados em texto)
            let cases = [
                ("What is the weather in Paris?", Some("get_weather"), "paris"),
                ("How is the weather in Tokyo right now?", Some("get_weather"), "tokyo"),
                ("What is 17 plus 25?", Some("add"), "17"),
                ("Add 3 and 4 using the tool.", Some("add"), "3"),
                ("Is it raining in Lisbon?", Some("get_weather"), "lisbon"),
                ("Tell me a short joke.", None, ""),
            ];
            let (mut valid, mut right_fn, mut right_args, mut n_call) = (0, 0, 0, 0);
            for (q, want, arg) in cases {
                let raw = format!(r#"{{"tools":{tools},"messages":[{{"role":"user","content":{}}}]}}"#, serde_json::to_string(q).unwrap());
                let body: Value = serde_json::from_str(&raw).unwrap();
                let (text, o) = ask_model(&engine, &body, &raw).await;
                println!("\nQ: {q}\nraw: {text:?}\n=> calls={:?} error={:?}", o.calls, o.error);
                if want.is_some() {
                    n_call += 1;
                    if o.error.is_none() && !o.calls.is_empty() {
                        valid += 1;
                    }
                    if o.calls.first().map(|c| c.name.as_str()) == want {
                        right_fn += 1;
                        if o.calls[0].arguments.to_lowercase().contains(arg) {
                            right_args += 1;
                        }
                    }
                }
            }
            // volta com o resultado da ferramenta: o modelo deve responder em texto
            let raw = format!(r#"{{"tools":{tools},"messages":[{{"role":"user","content":"What is the weather in Paris?"}},{{"role":"assistant","content":null,"tool_calls":[{{"id":"call_1","type":"function","function":{{"name":"get_weather","arguments":"{{\"city\":\"Paris\"}}"}}}}]}},{{"role":"tool","tool_call_id":"call_1","content":"18 C, cloudy"}}]}}"#);
            let body: Value = serde_json::from_str(&raw).unwrap();
            let (text, o) = ask_model(&engine, &body, &raw).await;
            println!("\nvolta da ferramenta: raw={text:?} calls={} error={:?}", o.calls.len(), o.error);
            println!("\nRESULTADO ({n_call} pedidos que pedem ferramenta): JSON/chamada valida {valid}/{n_call}, funcao certa {right_fn}/{n_call}, argumento certo {right_args}/{n_call}");
        });
    }
}
