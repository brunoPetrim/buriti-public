/* Theme, menu and language. Loaded in <head>. No network (translations are local files). */
(function () {
  var root = document.documentElement;
  root.className = (root.className ? root.className + " " : "") + "js";
  var THEME_KEY = "buriti-site-theme", LANG_KEY = "buriti-site-lang";
  var NAMES = {
    en: "English", pt: "Português (BR)", es: "Español", zh: "中文 (普通话)", ru: "Русский", ja: "日本語", ko: "한국어",
    ar: "العربية", de: "Deutsch", fr: "Français", sw: "Kiswahili", it: "Italiano", id: "Bahasa Indonesia",
    fil: "Filipino", hi: "हिन्दी", th: "ไทย", vi: "Tiếng Việt", km: "ភាសាខ្មែរ", yo: "Yorùbá", am: "አማርኛ",
    om: "Afaan Oromoo", ha: "Hausa", zu: "isiZulu", ig: "Igbo", sn: "chiShona", bn: "বাংলা", ms: "Bahasa Melayu",
    ur: "اردو", pa: "ਪੰਜਾਬੀ", ht: "Kreyòl ayisyen", pap: "Papiamentu", pl: "Polski", ro: "Română", nl: "Nederlands"
  };
  var RTL = { ar: 1, ur: 1 };
  var cur = "en", T = {}; // T = textos do idioma atual (vazio = ingles do proprio HTML)
  window.BURITI_I18N = window.BURITI_I18N || {};

  function store(get, k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { /* storage bloqueado */ } return null; }

  /* ---- tema: claro por padrao, o sistema operacional nao decide; a escolha do botao fica salva ---- */
  var saved = store(0, THEME_KEY);
  root.setAttribute("data-theme", saved === "dark" ? "dark" : "light");
  function x(key, fallback) { return T[key] || fallback; }

  /* ---- idioma ---- */
  var decode = function (s) { return s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&amp;/g, "&"); };
  function apply(lang, strings) {
    cur = lang; T = strings || {};
    root.setAttribute("lang", lang === "pt" ? "pt-BR" : lang);
    root.setAttribute("dir", RTL[lang] ? "rtl" : "ltr");
    var nodes = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.__en === undefined) el.__en = el.innerHTML;
      var v = T[el.getAttribute("data-i18n")];
      if (el.tagName === "TITLE") el.textContent = v ? decode(v) : decode(el.__en);
      else el.innerHTML = v || el.__en;
    }
    var attrs = document.querySelectorAll("[data-i18n-a]");
    for (var j = 0; j < attrs.length; j++) {
      var a = attrs[j], name = a.tagName === "IMG" ? "alt" : a.tagName === "META" ? "content" : "aria-label";
      if (a.__enA === undefined) a.__enA = a.getAttribute(name);
      var w = T[a.getAttribute("data-i18n-a")];
      a.setAttribute(name, w ? decode(w) : a.__enA);
    }
    var sel = document.getElementById("lang-select");
    if (sel) { sel.value = lang; sel.setAttribute("aria-label", x("x_language", "Language")); }
    var mb = document.getElementById("menu-btn");
    if (mb) mb.querySelector(".lbl").textContent = x("x_menu", "Menu");
    themeLabel();
  }
  function setLang(lang, remember) {
    if (remember) store(0, LANG_KEY, lang);
    if (lang === "en") return apply("en", null);
    var ready = window.BURITI_I18N[lang];
    if (ready) return apply(lang, ready);
    var s = document.createElement("script");
    s.src = "assets/i18n/" + lang + ".js";
    s.onload = function () { if (cur !== lang || true) apply(lang, window.BURITI_I18N[lang]); };
    s.onerror = function () { apply("en", null); };
    document.head.appendChild(s);
  }
  function pick() {
    var s = store(0, LANG_KEY);
    if (s && NAMES[s]) return s;
    var list = (navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || "en"]);
    for (var i = 0; i < list.length; i++) {
      var c = String(list[i]).toLowerCase(), base = c.split("-")[0];
      if (NAMES[base]) return base;
      if (base === "tl") return "fil";
    }
    return "en";
  }

  /* ---- tema: botao ---- */
  function effective() { return root.getAttribute("data-theme") === "light" ? "light" : "dark"; }
  function themeLabel() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    var now = effective();
    btn.querySelector(".ico").textContent = now === "light" ? "☾" : "☀";
    btn.querySelector(".lbl").textContent = now === "light" ? x("x_to_dark", "dark") : x("x_to_light", "light");
    btn.setAttribute("aria-label", now === "light" ? x("x_switch_dark", "Switch to dark theme") : x("x_switch_light", "Switch to light theme"));
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.addEventListener("click", function () {
      var next = effective() === "light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      store(0, THEME_KEY, next);
      themeLabel();
    });

    /* menu hamburguer */
    var mb = document.getElementById("menu-btn"), nav = document.getElementById("site-nav");
    function toggle(open) {
      nav.classList.toggle("open", open);
      mb.setAttribute("aria-expanded", open ? "true" : "false");
    }
    if (mb && nav) {
      mb.addEventListener("click", function (e) { e.stopPropagation(); toggle(!nav.classList.contains("open")); });
      nav.addEventListener("click", function (e) { if (e.target.closest && e.target.closest("a")) toggle(false); });
      document.addEventListener("click", function (e) { if (nav.classList.contains("open") && !nav.contains(e.target)) toggle(false); });
      document.addEventListener("keydown", function (e) { if (e.key === "Escape" && nav.classList.contains("open")) { toggle(false); mb.focus(); } });
    }

    /* seletor de idioma */
    var sel = document.getElementById("lang-select");
    if (sel) {
      Object.keys(NAMES).forEach(function (k) {
        var o = document.createElement("option"); o.value = k; o.textContent = NAMES[k]; sel.appendChild(o);
      });
      sel.addEventListener("change", function () { setLang(sel.value, true); });
    }
    var first = pick();
    if (sel) sel.value = first;
    themeLabel();
    if (first !== "en") setLang(first, false);
  });
})();
