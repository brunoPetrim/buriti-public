//! Constrained decoding, part 1 (D82): a byte-level grammar matcher (a pushdown automaton run as a set of stacks, in the
//! style of llama.cpp's grammar sampler) and the token mask built from it. `jsonschema.rs` turns a JSON Schema into a
//! `Grammar`; the engine asks `Constraint::mask` before sampling and `Constraint::accept` after.
//!
//! How it works:
//! - A grammar is a set of rules; a rule is a list of alternatives; an alternative is a sequence of symbols; a symbol is
//!   either a set of bytes or a reference to a rule. Matching is on BYTES (no code points): strings accept every byte
//!   above 0x1F except quote and backslash, so a UTF-8 sequence split across tokens is fine. The cost is that the matcher
//!   does not validate UTF-8 itself; it trusts the model (the engine's `Utf8Stream` still decodes the output).
//! - The state is a set of stacks. The top of a stack is the next terminal expected; an empty stack means "the grammar
//!   is complete here". Rules are expanded lazily: a rule reference is replaced by one stack per alternative, and when
//!   the reference is the last symbol of its alternative the continuation is dropped (tail call), so right-recursive
//!   repetition (`rep ::= x rep | ""`) keeps a constant stack depth. Left recursion is refused when the grammar is built
//!   (`Builder::build`); the step budget of an expansion stays as a last defence against a hang.
//! - The token mask walks a trie of the vocabulary's byte strings with the current stacks: a branch whose next byte no
//!   stack accepts is cut, so most of the 150k tokens are never looked at. The mask of a state is cached by the state.
//!   Special tokens are not in the trie (a token such as `<|im_end|>` must never be matched as text); the end-of-text
//!   tokens are allowed exactly when the grammar is complete.

use std::collections::HashMap;
use std::sync::Arc;

/// A set of bytes (256 bits).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ByteSet([u64; 4]);

impl ByteSet {
    pub const fn empty() -> Self {
        ByteSet([0; 4])
    }
    pub fn single(b: u8) -> Self {
        let mut s = Self::empty();
        s.insert(b);
        s
    }
    pub fn range(lo: u8, hi: u8) -> Self {
        let mut s = Self::empty();
        for b in lo..=hi {
            s.insert(b);
        }
        s
    }
    pub fn of(bytes: &[u8]) -> Self {
        let mut s = Self::empty();
        for &b in bytes {
            s.insert(b);
        }
        s
    }
    pub fn insert(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1u64 << (b & 63);
    }
    pub fn contains(&self, b: u8) -> bool {
        (self.0[(b >> 6) as usize] >> (b & 63)) & 1 == 1
    }
    pub fn union(mut self, o: Self) -> Self {
        for i in 0..4 {
            self.0[i] |= o.0[i];
        }
        self
    }
    pub fn negate(mut self) -> Self {
        for i in 0..4 {
            self.0[i] = !self.0[i];
        }
        self
    }
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sym {
    Set(ByteSet),
    Rule(u32),
}

/// Grammar under construction. A rule is created first (so it can refer to itself) and its alternatives are added later.
#[derive(Default)]
pub struct Builder {
    rules: Vec<Vec<Vec<Sym>>>,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }
    /// A new, still empty rule.
    pub fn rule(&mut self) -> u32 {
        self.rules.push(Vec::new());
        (self.rules.len() - 1) as u32
    }
    /// Adds an alternative (a sequence of symbols; an empty sequence matches the empty string) to a rule.
    pub fn alt(&mut self, rule: u32, seq: Vec<Sym>) {
        self.rules[rule as usize].push(seq);
    }
    /// A rule with the given alternatives.
    pub fn rule_of(&mut self, alts: Vec<Vec<Sym>>) -> u32 {
        let r = self.rule();
        for a in alts {
            self.alt(r, a);
        }
        r
    }
    /// The literal text as a sequence of single-byte symbols.
    pub fn lit(s: &str) -> Vec<Sym> {
        s.bytes().map(|b| Sym::Set(ByteSet::single(b))).collect()
    }
    /// `seq?`
    pub fn opt(&mut self, seq: Vec<Sym>) -> u32 {
        self.rule_of(vec![seq, Vec::new()])
    }
    /// `(a | b | ...)*` as a right-recursive rule.
    pub fn star(&mut self, alts: Vec<Vec<Sym>>) -> u32 {
        let r = self.rule();
        for mut a in alts {
            a.push(Sym::Rule(r));
            self.alt(r, a);
        }
        self.alt(r, Vec::new());
        r
    }
    /// At most `n` repetitions of `seq` (nested optionals).
    pub fn up_to(&mut self, seq: Vec<Sym>, n: usize) -> u32 {
        let mut inner = self.rule_of(vec![Vec::new()]);
        for _ in 0..n {
            let mut s = seq.clone();
            s.push(Sym::Rule(inner));
            inner = self.rule_of(vec![s, Vec::new()]);
        }
        inner
    }
    /// A rule that can start with itself (directly or through other rules, skipping empty prefixes) would expand forever.
    fn reject_left_recursion(&self) -> Result<(), String> {
        let n = self.rules.len();
        let mut nullable = vec![false; n];
        loop {
            let mut changed = false;
            for r in 0..n {
                if !nullable[r] && self.rules[r].iter().any(|alt| alt.iter().all(|s| matches!(s, Sym::Rule(x) if nullable[*x as usize]))) {
                    nullable[r] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // edges: a rule to the rules that can come first in one of its alternatives
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (r, alts) in self.rules.iter().enumerate() {
            for alt in alts {
                for s in alt {
                    match s {
                        Sym::Set(_) => break,
                        Sym::Rule(x) => {
                            if (*x as usize) < n {
                                edges[r].push(*x as usize);
                                if !nullable[*x as usize] {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        // cycle search (colors: 0 new, 1 on the path, 2 done)
        fn visit(r: usize, edges: &[Vec<usize>], color: &mut [u8]) -> bool {
            color[r] = 1;
            for &x in &edges[r] {
                if color[x] == 1 || (color[x] == 0 && visit(x, edges, color)) {
                    return true;
                }
            }
            color[r] = 2;
            false
        }
        let mut color = vec![0u8; n];
        for r in 0..n {
            if color[r] == 0 && visit(r, &edges, &mut color) {
                return Err("grammar: left recursion is not supported (a rule can start with itself)".into());
            }
        }
        Ok(())
    }

    pub fn build(self, root: u32) -> Result<Grammar, String> {
        if root as usize >= self.rules.len() {
            return Err("grammar: the start rule does not exist".into());
        }
        self.reject_left_recursion()?;
        let mut alts: Vec<Vec<Sym>> = Vec::new();
        let mut rule_alts: Vec<Vec<u32>> = Vec::with_capacity(self.rules.len());
        for r in &self.rules {
            let mut ids = Vec::with_capacity(r.len());
            for a in r {
                for s in a {
                    if let Sym::Rule(x) = s {
                        if *x as usize >= self.rules.len() {
                            return Err("grammar: reference to a rule that does not exist".into());
                        }
                    }
                }
                ids.push(alts.len() as u32);
                alts.push(a.clone());
            }
            rule_alts.push(ids);
        }
        Ok(Grammar { alts, rule_alts, root })
    }
}

/// A position: (alternative, index of the next symbol in it). A stack keeps the continuation of every open rule; its top
/// (last element) is what comes next.
type Pos = (u32, u32);
pub type Stack = Vec<Pos>;

/// Steps allowed for one expansion before the grammar is declared too ambiguous (or left-recursive).
const EXPAND_BUDGET: usize = 20_000;

pub struct Grammar {
    alts: Vec<Vec<Sym>>,
    rule_alts: Vec<Vec<u32>>,
    root: u32,
}

impl Grammar {
    /// Expands `start` until every stack has a terminal (or nothing) on top; pushes the results to `out`.
    fn expand(&self, start: Stack, out: &mut Vec<Stack>) {
        let mut budget = EXPAND_BUDGET;
        let mut work = vec![start];
        while let Some(mut st) = work.pop() {
            if budget == 0 {
                out.clear();
                return;
            }
            budget -= 1;
            loop {
                match st.last().copied() {
                    None => {
                        out.push(st);
                        break;
                    }
                    Some((a, i)) => {
                        let alt = &self.alts[a as usize];
                        if i as usize >= alt.len() {
                            st.pop();
                            continue;
                        }
                        match alt[i as usize] {
                            Sym::Set(_) => {
                                out.push(st);
                                break;
                            }
                            Sym::Rule(r) => {
                                if i as usize + 1 >= alt.len() {
                                    st.pop(); // tail call: nothing to come back to
                                } else if let Some(top) = st.last_mut() {
                                    top.1 = i + 1;
                                }
                                for &ra in &self.rule_alts[r as usize] {
                                    let mut ns = st.clone();
                                    ns.push((ra, 0));
                                    work.push(ns);
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    fn finish(mut stacks: Vec<Stack>) -> Vec<Stack> {
        stacks.sort();
        stacks.dedup();
        stacks
    }

    /// The state before any byte.
    pub fn start(&self) -> Vec<Stack> {
        let mut out = Vec::new();
        for &a in &self.rule_alts[self.root as usize] {
            self.expand(vec![(a, 0)], &mut out);
        }
        Self::finish(out)
    }

    /// The state after the byte `b`; empty when no stack accepts it.
    pub fn step(&self, stacks: &[Stack], b: u8) -> Vec<Stack> {
        let mut out = Vec::new();
        for st in stacks {
            if let Some(&(a, i)) = st.last() {
                if let Sym::Set(s) = self.alts[a as usize][i as usize] {
                    if s.contains(b) {
                        let mut ns = st.clone();
                        if let Some(top) = ns.last_mut() {
                            top.1 = i + 1;
                        }
                        self.expand(ns, &mut out);
                    }
                }
            }
        }
        Self::finish(out)
    }

    /// The bytes some stack accepts next.
    pub fn allowed(&self, stacks: &[Stack]) -> ByteSet {
        let mut set = ByteSet::empty();
        for st in stacks {
            if let Some(&(a, i)) = st.last() {
                if let Sym::Set(s) = self.alts[a as usize][i as usize] {
                    set = set.union(s);
                }
            }
        }
        set
    }

    /// Can the text end here?
    pub fn is_complete(stacks: &[Stack]) -> bool {
        stacks.iter().any(|s| s.is_empty())
    }

    /// Whole-string match (tests and validation of finished output).
    pub fn matches(&self, text: &[u8]) -> bool {
        let mut st = self.start();
        for &b in text {
            st = self.step(&st, b);
            if st.is_empty() {
                return false;
            }
        }
        Self::is_complete(&st)
    }
}

/// A trie of the byte strings of the vocabulary.
pub struct TokenTrie {
    nodes: Vec<TrieNode>,
    pub n_vocab: usize,
}

#[derive(Default)]
struct TrieNode {
    children: Vec<(u8, u32)>,
    tokens: Vec<u32>,
}

impl TokenTrie {
    /// `bytes_of(id)` gives the text bytes of a token, or `None` for tokens that must never be produced as text (special
    /// and control tokens, which includes the end-of-text tokens: `Constraint` handles those on its own).
    pub fn build(n_vocab: usize, bytes_of: impl Fn(u32) -> Option<Vec<u8>>) -> TokenTrie {
        let mut nodes = vec![TrieNode::default()];
        for id in 0..n_vocab as u32 {
            let Some(bytes) = bytes_of(id) else { continue };
            if bytes.is_empty() {
                continue;
            }
            let mut at = 0usize;
            for &b in &bytes {
                let next = match nodes[at].children.iter().find(|(c, _)| *c == b) {
                    Some(&(_, n)) => n as usize,
                    None => {
                        nodes.push(TrieNode::default());
                        let n = nodes.len() - 1;
                        nodes[at].children.push((b, n as u32));
                        n
                    }
                };
                at = next;
            }
            nodes[at].tokens.push(id);
        }
        TokenTrie { nodes, n_vocab }
    }

    /// The set of token ids (as a bitset) whose bytes keep the grammar alive from `stacks`.
    ///
    /// The walk interns every distinct grammar state and remembers the transition (state, byte) -> state, so the grammar step runs
    /// once per distinct pair and not once per trie node: inside a string, for instance, almost every byte leads back to the same
    /// state, and the 150k nodes of the vocabulary cost a hash lookup each instead of a stack copy and an expansion.
    fn allowed_bits(&self, g: &Grammar, stacks: &[Stack]) -> Vec<u64> {
        const DEAD: u32 = u32::MAX;
        let mut bits = vec![0u64; self.n_vocab.div_ceil(64)];
        let mut ids: HashMap<Vec<Stack>, u32> = HashMap::new();
        let mut states: Vec<Vec<Stack>> = Vec::new();
        let mut allowed: Vec<ByteSet> = Vec::new();
        let mut trans: HashMap<(u32, u8), u32> = HashMap::new();
        let intern = |st: Vec<Stack>, ids: &mut HashMap<Vec<Stack>, u32>, states: &mut Vec<Vec<Stack>>, allowed: &mut Vec<ByteSet>| -> u32 {
            if let Some(&i) = ids.get(&st) {
                return i;
            }
            let i = states.len() as u32;
            allowed.push(g.allowed(&st));
            ids.insert(st.clone(), i);
            states.push(st);
            i
        };
        let start = intern(stacks.to_vec(), &mut ids, &mut states, &mut allowed);
        let mut work: Vec<(usize, u32)> = vec![(0, start)];
        while let Some((node, s)) = work.pop() {
            for &(b, child) in &self.nodes[node].children {
                if !allowed[s as usize].contains(b) {
                    continue;
                }
                let next = match trans.get(&(s, b)) {
                    Some(&n) => n,
                    None => {
                        let ns = g.step(&states[s as usize], b);
                        let n = if ns.is_empty() { DEAD } else { intern(ns, &mut ids, &mut states, &mut allowed) };
                        trans.insert((s, b), n);
                        n
                    }
                };
                if next == DEAD {
                    continue;
                }
                for &tk in &self.nodes[child as usize].tokens {
                    bits[(tk >> 6) as usize] |= 1u64 << (tk & 63);
                }
                if !self.nodes[child as usize].children.is_empty() {
                    work.push((child as usize, next));
                }
            }
        }
        bits
    }
}

/// The constraint of one request: the grammar state plus the cache of masks.
pub struct Constraint {
    grammar: Arc<Grammar>,
    trie: Arc<TokenTrie>,
    stacks: Vec<Stack>,
    eos: Vec<u32>,
    cache: HashMap<Vec<Stack>, Arc<Vec<u64>>>,
}

/// Masks cached per request (each is `n_vocab / 8` bytes).
const MAX_CACHED_MASKS: usize = 64;

impl Constraint {
    pub fn new(grammar: Arc<Grammar>, trie: Arc<TokenTrie>, eos: Vec<u32>) -> Result<Constraint, String> {
        let stacks = grammar.start();
        if stacks.is_empty() {
            return Err("grammar: it accepts no text (or it is too ambiguous)".into());
        }
        Ok(Constraint { grammar, trie, stacks, eos, cache: HashMap::new() })
    }

    pub fn is_complete(&self) -> bool {
        Grammar::is_complete(&self.stacks)
    }

    /// Sets the logit of every token the grammar forbids to -infinity. Returns false when no token is allowed at all (a
    /// dead end: the caller should stop).
    pub fn mask(&mut self, logits: &mut [f32]) -> bool {
        let bits = match self.cache.get(&self.stacks) {
            Some(b) => b.clone(),
            None => {
                let b = Arc::new(self.trie.allowed_bits(&self.grammar, &self.stacks));
                if self.cache.len() >= MAX_CACHED_MASKS {
                    self.cache.clear();
                }
                self.cache.insert(self.stacks.clone(), b.clone());
                b
            }
        };
        let complete = self.is_complete();
        let mut any = false;
        for (i, l) in logits.iter_mut().enumerate() {
            let ok = i < self.trie.n_vocab && (bits[i >> 6] >> (i & 63)) & 1 == 1 || (complete && self.eos.contains(&(i as u32)));
            if ok {
                any = true;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
        any
    }

    /// Advances over the bytes of an accepted token. Returns false (and keeps the state) if the grammar rejects them.
    pub fn accept(&mut self, token_bytes: &[u8]) -> bool {
        let mut st = self.stacks.clone();
        for &b in token_bytes {
            st = self.grammar.step(&st, b);
            if st.is_empty() {
                return false;
            }
        }
        self.stacks = st;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn abc_grammar() -> Grammar {
        // root ::= "a" ("b" | "c")* "d"
        let mut b = Builder::new();
        let root = b.rule();
        let tail = b.star(vec![Builder::lit("b"), Builder::lit("c")]);
        let mut seq = Builder::lit("a");
        seq.push(Sym::Rule(tail));
        seq.extend(Builder::lit("d"));
        b.alt(root, seq);
        b.build(root).unwrap()
    }

    #[test]
    fn matches_the_language_and_nothing_else() {
        let g = abc_grammar();
        for ok in ["ad", "abd", "acbcd", "abbbbbbd"] {
            assert!(g.matches(ok.as_bytes()), "{ok} must match");
        }
        for bad in ["", "a", "abx", "abdd", "bd", "ada", "abcdc"] {
            assert!(!g.matches(bad.as_bytes()), "{bad} must not match");
        }
    }

    #[test]
    fn repetition_keeps_the_stack_depth_constant() {
        let g = abc_grammar();
        let mut st = g.start();
        st = g.step(&st, b'a');
        let depth0 = st.iter().map(|s| s.len()).max().unwrap();
        for _ in 0..5000 {
            st = g.step(&st, b'b');
        }
        assert_eq!(st.iter().map(|s| s.len()).max().unwrap(), depth0, "right recursion must not grow the stack");
        assert!(st.len() <= 4);
    }

    #[test]
    fn left_recursion_is_refused_when_the_grammar_is_built() {
        // r ::= r "a" | "a"
        let mut b = Builder::new();
        let r = b.rule();
        let mut s = vec![Sym::Rule(r)];
        s.extend(Builder::lit("a"));
        b.alt(r, s);
        b.alt(r, Builder::lit("a"));
        let e = b.build(r).err().expect("must be refused");
        assert!(e.contains("left recursion"), "{e}");
        // through another rule, and through an empty prefix: a ::= b "x"; b ::= "" | a
        let mut b = Builder::new();
        let (a, c) = (b.rule(), b.rule());
        b.alt(a, vec![Sym::Rule(c), Sym::Set(ByteSet::single(b'x'))]);
        b.alt(c, Vec::new());
        b.alt(c, vec![Sym::Rule(a)]);
        assert!(b.build(a).is_err());
        // right recursion is fine, and so is a rule that merely refers to another before a terminal
        assert!(abc_grammar().matches(b"abcd"));
    }

    #[test]
    fn a_schema_that_refers_to_itself_without_progress_is_refused() {
        // the expansion budget is still there as a last defence, but this is caught before: {"$ref":"#"} is r ::= r
        let mut b = Builder::new();
        let r = b.rule();
        b.alt(r, vec![Sym::Rule(r)]);
        assert!(b.build(r).is_err());
    }

    #[test]
    fn byte_sets() {
        let s = ByteSet::range(b'0', b'9');
        assert!(s.contains(b'5') && !s.contains(b'a'));
        let n = ByteSet::of(b"\"\\").negate();
        assert!(!n.contains(b'"') && !n.contains(b'\\') && n.contains(b'x') && n.contains(0xC3));
        assert!(ByteSet::empty().is_empty() && !s.is_empty());
    }

    /// A toy vocabulary: single bytes plus a few multi-byte tokens and one special token.
    fn toy_trie() -> (TokenTrie, Vec<Vec<u8>>) {
        let toks: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec(), b"ab".to_vec(), b"bcd".to_vec(), b"abd".to_vec(), b"dd".to_vec(), b"x".to_vec(), b"<eos>".to_vec()];
        let t2 = toks.clone();
        let trie = TokenTrie::build(toks.len(), move |id| if id == 9 { None } else { Some(t2[id as usize].clone()) });
        (trie, toks)
    }

    #[test]
    fn token_mask_follows_the_grammar() {
        let (trie, toks) = toy_trie();
        let mut c = Constraint::new(Arc::new(abc_grammar()), Arc::new(trie), vec![9]).unwrap();
        let allowed = |c: &mut Constraint| -> Vec<usize> {
            let mut l = vec![0.0f32; 10];
            assert!(c.mask(&mut l));
            (0..10).filter(|i| l[*i].is_finite()).collect()
        };
        // at the start only "a" and the tokens that begin with "a" (and stay valid) can come: a, ab, abd
        assert_eq!(allowed(&mut c), vec![0, 4, 6]);
        assert!(c.accept(&toks[0]));
        // after "a": b, c, d, bcd (valid), and not "dd" (past the end) nor "x" nor the end-of-text token (incomplete)
        assert_eq!(allowed(&mut c), vec![1, 2, 3, 5]);
        assert!(!c.accept(b"x"), "a rejected token must not change the state");
        assert!(c.accept(&toks[5]));
        assert!(c.is_complete());
        // complete: only the end-of-text token is left
        assert_eq!(allowed(&mut c), vec![9]);
    }

    #[test]
    fn mask_cache_is_reused_for_the_same_state() {
        let (trie, _) = toy_trie();
        let mut c = Constraint::new(Arc::new(abc_grammar()), Arc::new(trie), vec![9]).unwrap();
        assert!(c.accept(b"a"));
        let mut l = vec![0.0f32; 10];
        c.mask(&mut l);
        let n = c.cache.len();
        assert!(c.accept(b"b"));
        let mut l2 = vec![0.0f32; 10];
        c.mask(&mut l2);
        assert_eq!(c.cache.len(), n, "after a b the state is the same: no new mask");
        assert_eq!(l.iter().map(|x| x.is_finite()).collect::<Vec<_>>(), l2.iter().map(|x| x.is_finite()).collect::<Vec<_>>());
    }
}
