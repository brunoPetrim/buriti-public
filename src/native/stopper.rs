//! Filtro de texto de saída da geração: textos de parada (`stop`) e, no preenchimento no meio (FIM), o corte
//! quando o modelo começa a repetir o sufixo que já existe no arquivo. Lógica pura (sem modelo), testável.
//!
//! O filtro segura o final do texto que ainda pode virar o começo de um padrão; só o que já não pode ser
//! parte de nenhum padrão sai. Quando um padrão aparece, o texto anterior a ele sai e a geração acaba.

/// Quantos caracteres do começo do sufixo valem como "o modelo voltou a copiar o sufixo".
const SUFFIX_HEAD_CHARS: usize = 24;
/// Cabeça de sufixo com menos de tantos caracteres não-brancos é comum demais (`}`, `)`, `end`): não vale como parada.
const SUFFIX_HEAD_MIN_VISIBLE: usize = 8;

pub struct Stopper {
    patterns: Vec<String>,
    hold_max: usize,
    held: String,
    guard: Option<BracketGuard>,
}

impl Stopper {
    /// `stops`: textos de parada do pedido. `suffix`: o texto depois do cursor (FIM), se houver.
    pub fn new(stops: &[String], suffix: Option<&str>) -> Self {
        let mut patterns: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        if let Some(h) = suffix.and_then(suffix_head) {
            patterns.push(h);
        }
        let hold_max = patterns.iter().map(String::len).max().unwrap_or(0).saturating_sub(1);
        let guard = suffix.and_then(|s| s.trim_start().chars().next()).filter(|c| matches!(c, ')' | ']' | '}')).map(|c| BracketGuard { stop_on: c, ..Default::default() });
        Stopper { patterns, hold_max, held: String::new(), guard }
    }

    /// Recebe o próximo trecho gerado. Devolve o texto que já pode sair e se um padrão foi encontrado
    /// (nesse caso o texto devolvido é só o que vem antes do padrão e a geração deve terminar).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        let (released, pattern_hit) = self.match_patterns(text);
        match &mut self.guard {
            None => (released, pattern_hit),
            Some(g) => {
                let (out, cut) = g.feed(&released);
                (out, pattern_hit || cut)
            }
        }
    }

    fn match_patterns(&mut self, text: &str) -> (String, bool) {
        if self.patterns.is_empty() {
            return (text.to_string(), false);
        }
        self.held.push_str(text);
        if let Some(pos) = self.patterns.iter().filter_map(|p| self.held.find(p.as_str())).min() {
            self.held.truncate(pos);
            return (std::mem::take(&mut self.held), true);
        }
        if self.held.len() > self.hold_max {
            let mut cut = self.held.len() - self.hold_max;
            while !self.held.is_char_boundary(cut) {
                cut -= 1;
            }
            return (self.held.drain(..cut).collect(), false);
        }
        (String::new(), false)
    }

    /// Fim da geração sem padrão encontrado: o que ficou segurado não era parada e pode sair.
    pub fn flush(&mut self) -> String {
        let rest = std::mem::take(&mut self.held);
        match &mut self.guard {
            None => rest,
            Some(g) => {
                let (mut out, cut) = g.feed(&rest);
                if !cut {
                    out.push_str(&g.take_pending_ws());
                }
                out
            }
        }
    }
}

/// Guarda de parênteses do FIM: quando o texto depois do cursor começa fechando um bloco (`}`, `)`, `]`), a lacuna
/// é o final de uma construção que o SUFIXO fecha. Se o modelo gerar esse mesmo fecho sem tê-lo aberto na própria
/// lacuna, ele duplicou o fecho do sufixo e saiu da construção (passa a escrever o que vem depois): corta antes dele.
/// Fechos de OUTRO tipo que a lacuna não abriu passam (o `)` de `while (i <cursor>` está no prefixo, não no sufixo).
/// Os espaços em branco finais ficam segurados até aparecer texto visível, para não sobrar indentação do fecho cortado.
/// Reconhece aspas (`"`, `'`, crase) e comentários `//` de uma linha; não entende `/* */`, `#`, nem aspas triplas.
#[derive(Default)]
struct BracketGuard {
    /// o fecho que o sufixo começa por (o que o modelo não deve repetir)
    stop_on: char,
    depth: usize,
    quote: Option<char>,
    escape: bool,
    line_comment: bool,
    pending_ws: String,
    prev: Option<char>,
}

impl BracketGuard {
    fn take_pending_ws(&mut self) -> String {
        std::mem::take(&mut self.pending_ws)
    }

    /// Devolve o texto que pode sair e se encontrou o fecho a cortar (o texto dele e o seguinte são descartados).
    fn feed(&mut self, text: &str) -> (String, bool) {
        let mut out = String::new();
        for c in text.chars() {
            if c.is_whitespace() {
                if c == '\n' {
                    self.line_comment = false;
                    if self.quote.is_some_and(|q| q != '`') {
                        self.quote = None; // aspas simples/duplas não passam da linha
                        self.escape = false;
                    }
                }
                self.pending_ws.push(c);
                self.prev = Some(c);
                continue;
            }
            if let Some(q) = self.quote {
                if self.escape {
                    self.escape = false;
                } else if c == '\\' {
                    self.escape = true;
                } else if c == q {
                    self.quote = None;
                }
            } else if !self.line_comment {
                match c {
                    '"' | '\'' | '`' => self.quote = Some(c),
                    '/' if self.prev == Some('/') => self.line_comment = true,
                    '(' | '[' | '{' => self.depth += 1,
                    ')' | ']' | '}' => {
                        if self.depth > 0 {
                            self.depth -= 1;
                        } else if c == self.stop_on {
                            self.pending_ws.clear();
                            return (out, true);
                        }
                    }
                    _ => {}
                }
            }
            out.push_str(&std::mem::take(&mut self.pending_ws));
            out.push(c);
            self.prev = Some(c);
        }
        (out, false)
    }
}

/// Começo do sufixo (literal, com a quebra de linha e a indentação como no arquivo) usado para perceber que o
/// modelo terminou a lacuna e passou a reescrever o que já existe depois do cursor.
fn suffix_head(suffix: &str) -> Option<String> {
    let head: String = suffix.chars().take(SUFFIX_HEAD_CHARS).collect();
    if head.chars().filter(|c| !c.is_whitespace()).count() < SUFFIX_HEAD_MIN_VISIBLE {
        return None;
    }
    Some(head)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str], stops: &[&str], suffix: Option<&str>) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut st = Stopper::new(&stops, suffix);
        let mut out = String::new();
        for c in chunks {
            let (o, hit) = st.push(c);
            out.push_str(&o);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&st.flush());
        (out, false)
    }

    #[test]
    fn no_patterns_passes_through() {
        assert_eq!(run(&["ab", "cd"], &[], None), ("abcd".into(), false));
        assert_eq!(run(&["ab", "cd"], &[], Some("}")), ("abcd".into(), false)); // sufixo curto: sem guarda
    }

    #[test]
    fn stop_text_split_across_chunks() {
        let text = "x = 1;\n\nnext";
        // todos os cortes possíveis do texto em dois blocos: o resultado tem de ser o mesmo
        for cut in 0..=text.len() {
            if !text.is_char_boundary(cut) {
                continue;
            }
            let (a, b) = text.split_at(cut);
            assert_eq!(run(&[a, b], &["\n\n"], None), ("x = 1;".into(), true), "corte em {cut}");
        }
        // um caractere por bloco
        let ch: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = ch.iter().map(String::as_str).collect();
        assert_eq!(run(&refs, &["\n\n"], None), ("x = 1;".into(), true));
    }

    #[test]
    fn partial_stop_is_flushed_when_generation_ends() {
        assert_eq!(run(&["abc", "<|end"], &["<|endoftext|>"], None), ("abc<|end".into(), false));
    }

    #[test]
    fn earliest_pattern_wins_and_multibyte_is_safe() {
        assert_eq!(run(&["ação ", "b END ", "c STOP"], &["STOP", "END"], None), ("ação b ".into(), true));
        // texto com caracteres de vários bytes perto do limite de retenção
        let (o, hit) = run(&["é", "ã", "ç", "õ", "STOP"], &["STOP"], None);
        assert_eq!((o.as_str(), hit), ("éãçõ", true));
    }

    #[test]
    fn suffix_regeneration_is_cut() {
        let suffix = "\n  return result;\n}\n\nmodule.exports = {};";
        // o modelo escreve a lacuna e continua copiando o sufixo
        let chunks = ["if (a) {", " b();", " }", "\n  retu", "rn result;", "\n}\n\nmod"];
        assert_eq!(run(&chunks, &[], Some(suffix)), ("if (a) { b(); }".into(), true));
    }

    #[test]
    fn short_or_absent_suffix_does_not_cut() {
        assert!(suffix_head("}\n").is_none());
        assert!(suffix_head("   ").is_none());
        assert!(suffix_head("\n  return result;\n}").is_some());
        // sem a guarda, o "}" gerado dentro da lacuna é preservado
        assert_eq!(run(&["if (a) {\n  b();\n}"], &[], Some("}\n")), ("if (a) {\n  b();\n}".into(), false));
    }

    #[test]
    fn closing_bracket_of_the_suffix_cuts_the_overrun() {
        let suffix = "\n  }\n\n  has(key) {\n";
        // o modelo escreve a lacuna, FECHA o bloco que o sufixo já fecha e continua com outro método
        let text = "// TODO\n    throw new Error(\"x}\"); // fecho } no comentário\n  }\n\n  // other\n  has(key) {";
        let want = "// TODO\n    throw new Error(\"x}\"); // fecho } no comentário";
        for cut in 0..=text.len() {
            if !text.is_char_boundary(cut) {
                continue;
            }
            let (a, b) = text.split_at(cut);
            assert_eq!(run(&[a, b], &[], Some(suffix)), (want.into(), true), "corte em {cut}");
        }
        // blocos aninhados que a própria lacuna abre e fecha não são cortados, e o texto sai igual
        let ok = "if (a) {\n  b([1, 2]);\n}\n";
        assert_eq!(run(&[ok], &[], Some(suffix)), (ok.to_string(), false));
        // um `)` que fecha o `(` aberto ANTES do cursor não é cortado (o sufixo começa por `}`); o `}` duplicado, sim
        assert_eq!(run(&["< s.length) {
 x;
}"], &[], Some(suffix)), ("< s.length) {
 x;
}".into(), false));
        // (o corte pode vir no `flush`, quando o fecho ainda estava segurado: só o texto importa)
        assert_eq!(run(&["< s.length) {
 x;
}
}

more"], &[], Some(suffix)).0, "< s.length) {
 x;
}");
        // sem guarda quando o sufixo não começa fechando algo
        assert_eq!(run(&["a}"], &[], Some("\n  next();")), ("a}".into(), false));
    }

    #[test]
    fn bracket_guard_ignores_strings_and_comments() {
        let suffix = ")";
        for text in ["f(\")\", ')', `)`)", "x // )\ny", "s = \"a\\\")\""] {
            let (o, hit) = run(&[text], &[], Some(suffix));
            assert!(!hit, "{text}");
            assert_eq!(o, text);
        }
        assert_eq!(run(&["a, b) c"], &[], Some(suffix)), ("a, b".into(), true));
    }

    #[test]
    fn middle_that_merely_resembles_suffix_start_is_kept() {
        // só a cópia literal do começo (24 caracteres) corta; uma linha parecida não
        let suffix = "\n  return result;\n}\n\nmodule.exports = {};";
        let (o, hit) = run(&["x = 1;\n  return r;"], &[], Some(suffix));
        assert!(!hit);
        assert_eq!(o, "x = 1;\n  return r;");
    }
}
