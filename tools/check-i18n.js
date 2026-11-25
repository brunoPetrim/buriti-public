// Confere as traducoes da interface (somente le web/*.js e web/index.html; nao executa nada do projeto alem de avaliar os objetos de traducao).
// Uso:  node tools/check-i18n.js          resumo (falha, exit 1, se faltar chave usada pela pagina em en/pt/es)
//       node tools/check-i18n.js --full   lista as chaves que faltam em cada lingua
// Regras: (1) toda chave usada pela pagina existe em en, pt e es; (2) toda chave de en existe nas 34 linguas
// (inclusive os textos por modelo m_<modelo>_*); (3) marcadores {x} iguais aos do ingles.
// Nas linguas alem de en/pt/es a falta de chave e AVISO no resumo; use --strict para falhar (exit 1) por qualquer chave faltando.
const fs = require("fs");
const path = require("path");
const web = path.join(__dirname, "..", "web");
const full = process.argv.includes("--full");

let src = "";
for (const f of ["i18n.js", "i18n2.js", "i18n3.js", "i18n4.js", "i18n5.js"]) {
  const p = path.join(web, f);
  if (fs.existsSync(p)) src += fs.readFileSync(p, "utf8") + "\n";
}
const { I18N, LANG_NAMES } = new Function(src + "; return { I18N, LANG_NAMES };")();
const html = fs.readFileSync(path.join(web, "index.html"), "utf8");

const used = new Set();
for (const m of html.matchAll(/\bt\(\s*"([a-z0-9_]+)"\s*\)/gi)) used.add(m[1]);
for (const m of html.matchAll(/data-i18n(?:-ph)?="([a-z0-9_]+)"/gi)) used.add(m[1]);

const langs = Object.keys(LANG_NAMES);
const ph = (s) => (String(s).match(/\{\w+\}/g) || []).sort().join(",");
let errors = 0;

console.log("Linguas: " + langs.length + " | chaves em en: " + Object.keys(I18N.en).length + " | chaves usadas pela pagina: " + used.size);

const core = ["en", "pt", "es"];
for (const k of used) for (const l of core) if (!(I18N[l] && I18N[l][k])) { console.log("ERRO: chave usada pela pagina falta em " + l + ": " + k); errors++; }

for (const l of langs) {
  const miss = [];
  for (const k of Object.keys(I18N.en)) {
    if (I18N[l] && I18N[l][k]) { if (ph(I18N[l][k]) !== ph(I18N.en[k])) { console.log("ERRO: marcadores diferentes em " + l + ": " + k); errors++; } continue; }
    miss.push(k); // todas as chaves (inclusive m_<modelo>_*) existem nas 34 linguas
  }
  if ((core.includes(l) || process.argv.includes("--strict")) && miss.length) errors += miss.length;
  if (miss.length) {
    console.log((core.includes(l) || process.argv.includes("--strict") ? "ERRO " : "falta ") + l + " (" + LANG_NAMES[l] + "): " + miss.length + " chave(s)" + (full ? ":\n    " + miss.join("\n    ") : ""));
  }
}
console.log(errors ? "\nFALHOU: " + errors + " problema(s)." : "\nOK: nenhuma chave faltando.");
process.exit(errors ? 1 : 0);
