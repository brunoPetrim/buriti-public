#!/usr/bin/env node
// Traducoes da pagina do site (index.html). Uso (rode na raiz do repositorio):
//   node tools/i18n-site.js sync    marca o HTML (data-i18n="<hash>"), regrava i18n/_source_en.json e lista o que falta por idioma
//   node tools/i18n-site.js build   gera assets/i18n/<lang>.js a partir de i18n/<lang>.json
//   node tools/i18n-site.js check   confere os 34 idiomas (chaves, tags, entidades); sai com 1 se algo falta
// O ingles e o proprio texto do index.html. A chave de cada texto e o hash do ingles: se o ingles muda,
// a chave muda e o idioma passa a "faltar" ate ser retraduzido (a pagina cai para o ingles nesse trecho).
// Elementos com translate="no" ficam de fora. Blocos <pre> (codigo) nunca sao traduzidos.
"use strict";
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const ROOT = path.resolve(__dirname, "..");
const HTML = path.join(ROOT, "index.html");
const SRC_DIR = path.join(ROOT, "i18n");
const OUT_DIR = path.join(ROOT, "assets", "i18n");
const LANGS = ["pt", "es", "zh", "ru", "ja", "ko", "ar", "de", "fr", "sw", "it", "id", "fil", "hi", "th", "vi", "km", "yo", "am", "om", "ha", "zu", "ig", "sn", "bn", "ms", "ur", "pa", "ht", "pap", "pl", "ro", "nl"];
// textos da interface que o JavaScript monta (menu, tema, seletor); chaves fixas
const EXTRA = {
  x_menu: "Menu",
  x_language: "Language",
  x_to_light: "light",
  x_to_dark: "dark",
  x_switch_light: "Switch to light theme",
  x_switch_dark: "Switch to dark theme",
};
const NO_TRANSLATE = new Set(["HTTP", "Rust, axum", ".gguf", "trait Engine", "Continue, JetBrains..."]);
const INLINE = new Set(["strong", "em", "code", "a", "span", "br", "kbd", "b", "i"]);
const UNIT = new Set(["p", "li", "h2", "h3", "td", "th", "caption", "div", "text", "title", "desc", "a"]);
const VOID = new Set(["img", "br", "meta", "link", "hr", "input"]);

const norm = (s) => s.replace(/\s+/g, " ").trim();
const hash = (s) => crypto.createHash("sha1").update(s).digest("hex").slice(0, 8);
const hasLetters = (inner) => (inner.replace(/<code>[\s\S]*?<\/code>/g, "").replace(/<[^>]*>/g, "").replace(/&[a-z#0-9]+;/g, "").match(/\p{L}/gu) || []).length >= 2;

function parse(html) {
  const re = /<!--[\s\S]*?-->|<(\/?)([a-zA-Z][\w:-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>/g;
  const root = { tag: "#root", children: [], attrs: "", opaque: false };
  let cur = root, m;
  while ((m = re.exec(html))) {
    if (m[0].startsWith("<!--")) continue;
    const closing = m[1] === "/", tag = m[2].toLowerCase();
    if (closing) {
      let n = cur;
      while (n && n.tag !== tag) n = n.parent;
      if (n && n.parent) { n.close = [m.index, m.index + m[0].length]; cur = n.parent; }
      continue;
    }
    const node = { tag, attrs: m[3], open: [m.index, m.index + m[0].length], children: [], parent: cur };
    node.opaque = cur.opaque || ["script", "style", "pre"].includes(tag);
    cur.children.push(node);
    if (!(VOID.has(tag) || m[4] === "/")) cur = node;
  }
  return root;
}
const attr = (n, name) => { const m = n.attrs.match(new RegExp("(?:^|\\s)" + name + "\\s*=\\s*\"([^\"]*)\"")); return m ? m[1] : null; };
const allInline = (n) => n.children.every((c) => INLINE.has(c.tag) && allInline(c));

function collect(html) {
  const root = parse(html);
  const items = []; // {node, kind: "html"|"attr", text}
  const warn = [];
  (function walk(n) {
    for (const c of n.children) {
      if (c.opaque && c.tag !== "pre") continue;
      if (c.tag === "pre" || c.tag === "script" || c.tag === "style") continue;
      if (attr(c, "translate") === "no") continue;
      // atributos de texto
      for (const [t, a] of [["img", "alt"], ["meta", "content"], [null, "aria-label"]]) {
        if ((t === null || c.tag === t) && attr(c, a)) {
          if (t === "meta" && attr(c, "name") !== "description") continue;
          items.push({ node: c, kind: "attr", name: a, text: norm(attr(c, a)) });
        }
      }
      const parentOk = c.tag !== "a" || ["nav", "body", "div"].includes(c.parent.tag);
      if (UNIT.has(c.tag) && c.close && allInline(c) && parentOk) {
        const inner = html.slice(c.open[1], c.close[0]);
        const text = norm(inner);
        if (text && hasLetters(inner) && !NO_TRANSLATE.has(text) && !(c.tag === "div" && c.children.length === 0 && !/\p{L}/u.test(text))) {
          items.push({ node: c, kind: "html", text });
          continue;
        }
        if (c.tag !== "div") continue;
      }
      // texto solto num elemento que nao virou unidade
      if (c.close && !["title"].includes(c.tag)) {
        let pos = c.open[1], own = "";
        for (const k of c.children) { own += html.slice(pos, k.open[0]); pos = k.close ? k.close[1] : k.open[1]; }
        own += html.slice(pos, c.close[0]);
        if (c.tag !== "html" && /\p{L}{2,}/u.test(own.replace(/<[^>]*>/g, "")) && !["head", "svg", "button", "select", "option", "label"].includes(c.tag))
          warn.push(`texto solto em <${c.tag}${c.attrs.slice(0, 40)}>: ${norm(own).slice(0, 60)}`);
      }
      walk(c);
    }
  })(root);
  return { items, warn };
}

function sync() {
  let html = fs.readFileSync(HTML, "utf8");
  const { items, warn } = collect(html);
  // regrava data-i18n / data-i18n-a, de tras para frente
  const edits = [];
  for (const it of items) {
    const key = hash(it.text);
    const name = it.kind === "html" ? "data-i18n" : "data-i18n-a";
    const m = it.node.attrs.match(new RegExp("\\s" + name + "=\"[^\"]*\""));
    const tagStart = it.node.open[0] + 1 + it.node.tag.length;
    if (m) {
      const s = it.node.open[0] + 1 + it.node.tag.length + m.index;
      edits.push([s, s + m[0].length, ` ${name}="${key}"`]);
    } else edits.push([tagStart, tagStart, ` ${name}="${key}"`]);
    it.key = key;
  }
  edits.sort((a, b) => b[0] - a[0]).forEach(([s, e, t]) => { html = html.slice(0, s) + t + html.slice(e); });
  fs.writeFileSync(HTML, html);
  // fonte em ingles (le de novo: as posicoes mudaram)
  const again = collect(html).items;
  const src = {};
  for (const it of again) src[hash(it.text)] = it.text;
  Object.assign(src, EXTRA);
  fs.mkdirSync(SRC_DIR, { recursive: true });
  fs.writeFileSync(path.join(SRC_DIR, "_source_en.json"), JSON.stringify(src, null, 1) + "\n");
  console.log(`${Object.keys(src).length} textos (${Object.keys(EXTRA).length} extras de interface)`);
  warn.forEach((w) => console.log("AVISO", w));
  report(src, false);
}

function loadLang(l) {
  const f = path.join(SRC_DIR, l + ".json");
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : null;
}
const skeleton = (s) => (s.match(/<[^>]+>|&[a-z#0-9]+;/gi) || []).map((x) => x.replace(/\s+/g, " ")).sort().join("|");

function report(src, strict) {
  let bad = 0;
  for (const l of LANGS) {
    const tr = loadLang(l);
    if (!tr) { console.log(`${l}: SEM ARQUIVO`); bad++; continue; }
    const miss = Object.keys(src).filter((k) => !(k in tr) || !String(tr[k]).trim());
    const extra = Object.keys(tr).filter((k) => !(k in src));
    const skel = Object.keys(src).filter((k) => k in tr && skeleton(src[k]) !== skeleton(String(tr[k])));
    if (miss.length || skel.length || (strict && extra.length)) bad++;
    if (miss.length || skel.length || extra.length)
      console.log(`${l}: faltam ${miss.length}, tags/entidades diferentes ${skel.length}${extra.length ? ", sobrando " + extra.length : ""}` + (skel.length ? "  ex: " + skel.slice(0, 3).join(",") : ""));
  }
  console.log(bad ? `\nPENDENTE em ${bad} idioma(s)` : "\nOK: 33 idiomas completos (mais o ingles do proprio HTML)");
  return bad;
}

function build() {
  const src = JSON.parse(fs.readFileSync(path.join(SRC_DIR, "_source_en.json"), "utf8"));
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const l of LANGS) {
    const tr = loadLang(l);
    if (!tr) continue;
    const out = {};
    for (const k of Object.keys(src)) if (k in tr && String(tr[k]).trim()) out[k] = tr[k];
    fs.writeFileSync(path.join(OUT_DIR, l + ".js"), `window.BURITI_I18N=window.BURITI_I18N||{};window.BURITI_I18N[${JSON.stringify(l)}]=${JSON.stringify(out)};\n`);
  }
  console.log("gerado em assets/i18n/");
}

const cmd = process.argv[2];
if (cmd === "sync") sync();
else if (cmd === "build") build();
else if (cmd === "check") {
  // confere o HTML marcado e os arquivos
  const html = fs.readFileSync(HTML, "utf8");
  const items = collect(html).items;
  const stale = items.filter((it) => {
    const n = it.kind === "html" ? "data-i18n" : "data-i18n-a";
    return attr(it.node, n) !== hash(it.text);
  });
  if (stale.length) console.log(`index.html desatualizado: ${stale.length} texto(s) sem marca ou com hash velho; rode "sync"`);
  const src = JSON.parse(fs.readFileSync(path.join(SRC_DIR, "_source_en.json"), "utf8"));
  const bad = report(src, true);
  const built = LANGS.filter((l) => !fs.existsSync(path.join(OUT_DIR, l + ".js")));
  if (built.length) console.log('sem assets/i18n/*.js para: ' + built.join(", ") + ' (rode "build")');
  process.exit(bad || stale.length || built.length ? 1 : 0);
} else console.log("uso: node tools/i18n-site.js sync|build|check");
