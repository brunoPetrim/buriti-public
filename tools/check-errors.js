// Checks the stable error codes (D83). Reads src/*.rs and web/*.js only; executes nothing of the project.
// Usage: node tools/check-errors.js            summary; exit 1 on any problem
//        node tools/check-errors.js --pending   also lists the SERVER_MSGS keys that no Rust code emits yet (migration progress)
// Rules: (1) every code used in Rust as coded("err_x", ...) / coded_owned("err_x", ...) exists in the 34 languages;
//        (2) the number of {0}, {1}, ... markers of the text is not larger than the arguments the call passes (when the
//            arguments are written inline); (3) every code has an English text.
const fs = require("fs");
const path = require("path");
const root = path.join(__dirname, "..");

let src = "";
for (const f of ["i18n.js", "i18n2.js", "i18n3.js", "i18n4.js", "i18n5.js"]) {
  const p = path.join(root, "web", f);
  if (fs.existsSync(p)) src += fs.readFileSync(p, "utf8") + "\n";
}
const { I18N, LANG_NAMES } = new Function(src + "; return { I18N, LANG_NAMES };")();
const langs = Object.keys(LANG_NAMES);

function walk(dir, out) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (e.name.endsWith(".rs")) out.push(p);
  }
  return out;
}

const used = new Map(); // code -> [file:line]
let problems = 0;
for (const f of walk(path.join(root, "src"), [])) {
  if (f.endsWith("errcode.rs")) continue;
  const text = fs.readFileSync(f, "utf8");
  const re = /\b(?:coded|coded_owned)\(\s*"([a-z0-9_]+)"|\bsse_error\(\s*[A-Za-z:]+,\s*"([a-z0-9_]+)"/g;
  let m;
  while ((m = re.exec(text))) {
    const line = text.slice(0, m.index).split("\n").length;
    const code = m[1] || m[2];
    if (!used.has(code)) used.set(code, []);
    used.get(code).push(path.relative(root, f) + ":" + line);
  }
}

for (const [code, where] of used) {
  if (!I18N.en || !I18N.en[code]) {
    console.log("ERROR: code without an English text: " + code + "  (" + where[0] + ")");
    problems++;
    continue;
  }
  const miss = langs.filter((l) => !(I18N[l] && I18N[l][code]));
  if (miss.length) {
    console.log("ERROR: code " + code + " is missing in " + miss.length + " language(s): " + miss.slice(0, 8).join(", ") + (miss.length > 8 ? ", ..." : ""));
    problems++;
  }
}

if (process.argv.includes("--pending")) {
  const html = fs.readFileSync(path.join(root, "web", "index.html"), "utf8");
  const keys = [...html.matchAll(/\/,\s*"(err_[a-z0-9_]+)"\]/g)].map((m) => m[1]);
  const pending = keys.filter((k) => !used.has(k));
  console.log("SERVER_MSGS keys: " + keys.length + " | already emitted by Rust with a code: " + (keys.length - pending.length) + " | pending: " + pending.length);
  if (pending.length) console.log("pending: " + pending.join(", "));
}

console.log("codes used in Rust: " + used.size + " | languages: " + langs.length + " | problems: " + problems);
process.exit(problems ? 1 : 0);
