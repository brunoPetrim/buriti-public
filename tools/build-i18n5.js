// Gera web/i18n5.js a partir de uma pasta com um JSON por lingua (<lingua>.json: chave -> texto traduzido).
// Uso:  node tools/build-i18n5.js <pasta com os .json>
// Le so JSON (dados) e escreve so web/i18n5.js. Depois rode: node tools/check-i18n.js
const fs = require("fs");
const path = require("path");
const dir = process.argv[2];
if (!dir) { console.error("uso: node tools/build-i18n5.js <pasta>"); process.exit(2); }
const out = {};
for (const f of fs.readdirSync(dir).filter((x) => x.endsWith(".json") && !x.startsWith("_")).sort()) {
  out[f.replace(/\.json$/, "")] = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
}
let js = "// Quinta parte das traduções: textos dos modelos do catálogo e da página \"API\" nas demais línguas (as 31 além de en, pt e es).\n";
js += "// GERADO por tools/build-i18n5.js a partir de traduções de IA: a revisão por falantes nativos é bem-vinda (veja o README). Mescla sobre I18N.\n";
js += "const I18N5 = {\n";
for (const [lang, obj] of Object.entries(out)) {
  js += "  " + JSON.stringify(lang) + ": {\n";
  js += Object.entries(obj).map(([k, v]) => "    " + JSON.stringify(k) + ": " + JSON.stringify(v)).join(",\n") + "\n  },\n";
}
js += "};\nfor (const [lang, extra] of Object.entries(I18N5)) Object.assign((I18N[lang] ||= {}), extra);\n";
fs.writeFileSync(path.join(__dirname, "..", "web", "i18n5.js"), js, "utf8");
console.log("i18n5.js: " + Object.keys(out).length + " linguas");
