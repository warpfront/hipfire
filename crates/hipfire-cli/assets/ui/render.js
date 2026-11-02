/* hipfire chat UI — DOM helpers, icons, markdown, syntax highlighting.
 *
 * Model output is untrusted: everything here builds nodes with
 * createElement + textContent. innerHTML only ever receives the static
 * ICONS strings below, never anything derived from input.
 */
"use strict";

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

const ICONS = (() => {
  const line = (d) => `<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
  const solid = (d) => `<svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">${d}</svg>`;
  return {
    flame: solid('<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.15.43-2.29 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>'),
    sidebar: line('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/>'),
    edit: line('<path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.4 2.6a2.1 2.1 0 1 1 3 3L12 15l-4 1 1-4Z"/>'),
    pencil: line('<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>'),
    search: line('<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>'),
    database: line('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>'),
    upload: line('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>'),
    download: line('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>'),
    trash: line('<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'),
    sun: line('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>'),
    moon: line('<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>'),
    monitor: line('<rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>'),
    chevDown: line('<path d="m6 9 6 6 6-6"/>'),
    chevLeft: line('<path d="m15 18-6-6 6-6"/>'),
    chevRight: line('<path d="m9 18 6-6-6-6"/>'),
    file: line('<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v5h6M16 13H8M16 17H8M10 9H8"/>'),
    keyboard: line('<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M6 9h.01M10 9h.01M14 9h.01M18 9h.01M6 13h.01M18 13h.01M10 13h4M7 16h10"/>'),
    sliders: line('<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>'),
    arrowDown: line('<path d="M12 5v14M19 12l-7 7-7-7"/>'),
    arrowUp: line('<path d="M12 19V5M5 12l7-7 7 7"/>'),
    clip: line('<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>'),
    bulb: line('<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6M10 22h4"/>'),
    stop: solid('<rect x="6" y="6" width="12" height="12" rx="2"/>'),
    image: line('<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>'),
    x: line('<path d="M18 6 6 18M6 6l12 12"/>'),
    copy: line('<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'),
    check: line('<path d="M20 6 9 17l-5-5"/>'),
    retry: line('<path d="M3 12a9 9 0 1 0 2.64-6.36L3 8"/><path d="M3 3v5h5"/>'),
    pin: line('<path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1z"/>'),
    more: solid('<circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/>'),
    wrench: line('<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>'),
    alert: line('<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>'),
    speaker: line('<path d="M11 5 6 9H2v6h4l5 4V5Z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07M19.07 4.93a10 10 0 0 1 0 14.14"/>'),
    wrap: line('<path d="M3 6h18M3 12h15a3 3 0 1 1 0 6h-4"/><path d="m16 16-2 2 2 2"/><path d="M3 18h7"/>'),
    code: line('<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>'),
    sparkles: line('<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>'),
    mail: line('<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 6L2 7"/>'),
    message: line('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
  };
})();

function icon(name) {
  const s = el("span");
  s.dataset.icon = name;
  s.innerHTML = ICONS[name] || "";
  return s;
}

function hydrateIcons(root) {
  for (const n of root.querySelectorAll("[data-icon]")) {
    if (!n.firstChild) n.innerHTML = ICONS[n.dataset.icon] || "";
  }
}

function miniBtn(name, label, text) {
  const b = el("button", "mini-btn");
  b.type = "button";
  b.title = label;
  b.setAttribute("aria-label", label);
  b.append(icon(name));
  if (text) b.append(el("span", "mini-label", text));
  return b;
}

async function copyText(text, btn) {
  let ok = true;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // navigator.clipboard needs a secure context; serve is often reached
    // over plain http on a LAN address.
    const ta = el("textarea", "clip-ta");
    ta.value = text;
    ta.setAttribute("readonly", "");
    document.body.append(ta);
    ta.select();
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    ta.remove();
  }
  if (btn && ok) flashCopied(btn);
  return ok;
}

function flashCopied(btn) {
  const ic = btn.querySelector("[data-icon]");
  const label = btn.querySelector(".mini-label");
  if (label) btn.dataset.label ??= label.textContent;
  if (ic) ic.innerHTML = ICONS.check;
  if (label) label.textContent = "Copied";
  btn.classList.add("ok");
  clearTimeout(btn._copyTimer);
  btn._copyTimer = setTimeout(() => {
    if (ic) ic.innerHTML = ICONS[ic.dataset.icon];
    if (label) label.textContent = btn.dataset.label;
    btn.classList.remove("ok");
  }, 1500);
}

function downloadText(name, text, type = "text/plain;charset=utf-8") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

/* ================= syntax highlighting ================= */

const Highlight = (() => {
  const words = (s) => new Set(s.split(/\s+/).filter(Boolean));
  const END = String.raw`(?![\s\S])`;
  const C_COM = String.raw`\/\/[^\n]*|\/\*[\s\S]*?(?:\*\/|${END})`;
  const HASH_COM = String.raw`(?<![^\s])#[^\n]*`;
  const DQ = String.raw`"(?:[^"\\\n]|\\.)*"?`;
  const SQ = String.raw`'(?:[^'\\\n]|\\.)*'?`;
  const CHAR = String.raw`'(?:[^'\\\n]|\\(?:[nrt0\\'"]|x[\da-fA-F]{2}|u\{[\da-fA-F]{1,6}\}))'`;
  const BT = String.raw`\x60(?:[^\x60\\]|\\[\s\S])*\x60?`;
  const TRIPLE = String.raw`"""[\s\S]*?(?:"""|${END})|'''[\s\S]*?(?:'''|${END})`;
  const NUM = String.raw`\b(?:0[xX][\da-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)(?:[a-zA-Z]\w*)?\b`;
  const WORD = String.raw`[A-Za-z_$][\w$]*`;

  const JS = "async await break case catch class const continue debugger default delete do else export extends finally for from function get if import in instanceof let new of return set static switch throw try typeof var void while with yield";
  const LIT = words("true false null undefined None True False nil NULL nullptr NaN Infinity");
  const SELF = words("self this Self super cls");
  const PLAIN = words("text txt plain plaintext output log none md markdown");

  const DEFS = {
    js: { com: C_COM, str: [BT, DQ, SQ], kw: JS, ty: true },
    ts: { com: C_COM, str: [BT, DQ, SQ], ty: true,
      kw: JS + " abstract as declare enum implements interface keyof namespace private protected public readonly satisfies type" },
    py: { com: HASH_COM, str: [TRIPLE, DQ, SQ], meta: String.raw`@[\w.]+`, ty: true,
      kw: "and as assert async await break class continue def del elif else except finally for from global if import in is lambda match case nonlocal not or pass raise return try while with yield" },
    rust: { com: C_COM, str: [DQ, CHAR], extra: [[String.raw`'[a-z_]\w*\b(?!')`, "meta"]], ty: true,
      meta: String.raw`#!?\[[^\]\n]*\]|\b[a-z_]\w*!`,
      kw: "as async await break const continue crate dyn else enum extern fn for if impl in let loop match mod move mut pub ref return static struct trait type unsafe use where while" },
    go: { com: C_COM, str: [BT, DQ, CHAR], ty: true,
      kw: "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var" },
    c: { com: C_COM, str: [DQ, CHAR], meta: String.raw`^[ \t]*#[ \t]*[a-z]+`, ty: true,
      kw: "auto break case char class const constexpr continue default delete do double else enum explicit extern float for friend goto if inline int long namespace new noexcept operator private protected public register return short signed sizeof static static_cast struct switch template typedef typename union unsigned using virtual void volatile while bool __global__ __device__ __host__ __shared__" },
    java: { com: C_COM, str: [String.raw`"""[\s\S]*?(?:"""|${END})`, DQ, CHAR], meta: String.raw`@\w+`, ty: true,
      kw: "abstract boolean break byte case catch char class continue default do double else enum extends final finally float for if implements import instanceof int interface long native new package private protected public return short static switch synchronized throw throws transient try void volatile while var val fun when object override suspend data sealed internal" },
    sh: { com: HASH_COM, str: [DQ, String.raw`'[^']*'?`], var: String.raw`\$\{[^}\n]*\}?|\$(?:\w+|[@#?$!*-])`,
      kw: "if then else elif fi for while until do done case esac function in return export local readonly declare echo exit set unset source alias cd shift trap eval exec sudo" },
    sql: { com: String.raw`--[^\n]*|\/\*[\s\S]*?(?:\*\/|${END})`, str: [SQ, DQ], ci: true,
      kw: "select from where and or not insert into values update set delete create table drop alter add column join left right inner outer full cross on group by order having limit offset as distinct union all is in like between exists primary key foreign references index view case when then else end begin commit rollback returning with asc desc default unique check constraint" },
    css: { com: String.raw`\/\*[\s\S]*?(?:\*\/|${END})`, str: [DQ, SQ], meta: String.raw`@[\w-]+`, kw: "",
      extra: [[String.raw`#[\da-fA-F]{3,8}\b`, "num"], [String.raw`--[\w-]+`, "var"], [String.raw`[\w-]+(?=\s*:[^{};\n]*[;}])`, "fn"]] },
    json: { str: [DQ], kw: "", jsonKey: true },
    yaml: { com: HASH_COM, str: [DQ, SQ], kw: "",
      key: String.raw`(?<=^[ \t]*(?:-[ \t]+)?)[\w.\-/]+(?=[ \t]*:(?:\s|$))` },
    toml: { com: HASH_COM, str: [TRIPLE, DQ, SQ], kw: "",
      meta: String.raw`^[ \t]*\[\[?[^\]\n]*\]\]?`, key: String.raw`(?<=^[ \t]*)[\w.\-"]+(?=[ \t]*=)` },
    html: { com: String.raw`<!--[\s\S]*?(?:-->|${END})`, str: [DQ, SQ], kw: "", nums: false, noWords: true,
      extra: [[String.raw`(?<=<\/?)[\w:-]+`, "kw"], [String.raw`[\w:-]+(?==)`, "fn"]] },
    generic: { com: C_COM + "|" + HASH_COM, str: [DQ, SQ, BT], ty: true,
      kw: "if else elif for while do return function fn def class struct let const var import from export public private static new break continue switch case match try catch throw end then local" },
  };

  const ALIAS = {
    javascript: "js", jsx: "js", mjs: "js", cjs: "js", node: "js",
    typescript: "ts", tsx: "ts",
    python: "py", python3: "py", py3: "py",
    rs: "rust", golang: "go",
    h: "c", cpp: "c", "c++": "c", cc: "c", cxx: "c", hpp: "c", cuda: "c", cu: "c", hip: "c",
    objc: "c", glsl: "c", hlsl: "c", metal: "c", zig: "c",
    cs: "java", csharp: "java", "c#": "java", kotlin: "java", kt: "java", scala: "java", swift: "java", dart: "java",
    bash: "sh", shell: "sh", zsh: "sh", console: "sh", shellsession: "sh", fish: "sh", powershell: "sh", ps1: "sh",
    dockerfile: "sh", docker: "sh", make: "sh", makefile: "sh", cmake: "sh",
    yml: "yaml", ini: "toml", conf: "toml", cfg: "toml",
    xml: "html", svg: "html", vue: "html", htm: "html",
    jsonc: "json", json5: "json", jsonl: "json",
    scss: "css", sass: "css", less: "css",
    mysql: "sql", postgres: "sql", postgresql: "sql", sqlite: "sql",
    patch: "diff",
  };

  const EXT = {
    js: "js", ts: "ts", py: "py", rust: "rs", go: "go", c: "c", java: "java", sh: "sh",
    sql: "sql", css: "css", json: "json", yaml: "yaml", toml: "toml", html: "html", diff: "diff",
  };

  function compile(d) {
    const parts = [];
    const kinds = [];
    const add = (src, kind) => {
      if (!src) return;
      parts.push(`(${src})`);
      kinds.push(kind);
    };
    add(d.meta, "meta");
    add(d.com, "com");
    for (const s of d.str || []) add(s, "str");
    for (const [src, kind] of d.extra || []) add(src, kind);
    add(d.key, "key");
    add(d.var, "var");
    if (d.nums !== false) add(NUM, "num");
    if (!d.noWords) add(WORD, "word");
    d.rx = new RegExp(parts.join("|"), "gm");
    d.kinds = kinds;
    d.kwSet = words(d.ci ? (d.kw || "").toLowerCase() : d.kw || "");
  }

  function diff(code) {
    const frag = document.createDocumentFragment();
    code.split("\n").forEach((line, i) => {
      if (i) frag.append("\n");
      const cls = /^(\+\+\+|---)/.test(line) ? "tok-meta"
        : line[0] === "+" ? "tok-add"
        : line[0] === "-" ? "tok-del"
        : line.startsWith("@@") ? "tok-hunk" : null;
      frag.append(cls ? el("span", cls, line) : line);
    });
    return frag;
  }

  function resolve(lang) {
    const raw = (lang || "").toLowerCase();
    return ALIAS[raw] || raw;
  }

  function highlight(code, lang) {
    const key = resolve(lang);
    if (key === "diff") return diff(code);
    const frag = document.createDocumentFragment();
    const d = DEFS[key] || (key && !PLAIN.has(key) ? DEFS.generic : null);
    if (!d || code.length > 150000) {
      frag.append(code);
      return frag;
    }
    if (!d.rx) compile(d);
    const rx = d.rx;
    rx.lastIndex = 0;
    let last = 0;
    let m;
    while ((m = rx.exec(code))) {
      if (!m[0]) { rx.lastIndex++; continue; }
      let g = 1;
      while (g <= d.kinds.length && m[g] === undefined) g++;
      let kind = d.kinds[g - 1];
      const tok = m[0];
      if (kind === "word") {
        if (d.kwSet.has(d.ci ? tok.toLowerCase() : tok)) kind = "kw";
        else if (LIT.has(tok)) kind = "num";
        else if (SELF.has(tok)) kind = "var";
        else if (code[rx.lastIndex] === "(") kind = "fn";
        else if (d.ty && /^[A-Z][a-z\d]/.test(tok)) kind = "ty";
        else continue;
      } else if (kind === "str" && d.jsonKey && /^\s*:/.test(code.slice(rx.lastIndex, rx.lastIndex + 16))) {
        kind = "key";
      }
      if (m.index > last) frag.append(code.slice(last, m.index));
      frag.append(el("span", "tok-" + kind, tok));
      last = rx.lastIndex;
    }
    if (last < code.length) frag.append(code.slice(last));
    return frag;
  }

  function fileName(lang) {
    const key = resolve(lang);
    const ext = EXT[key] || (/^[a-z0-9]{1,6}$/.test(key) && !PLAIN.has(key) ? key : "txt");
    return "snippet." + ext;
  }

  return { highlight, fileName };
})();

/* ================= markdown ================= */

const Markdown = (() => {
  const FENCE = /^( {0,3})(`{3,}|~{3,})[ \t]*([^\s`]*)[^`]*$/;
  const HEADING = /^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$/;
  const HR = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
  const QUOTE = /^ {0,3}>[ \t]?/;
  const ITEM = /^([ \t]*)([-*+]|\d{1,9}[.)])(?:[ \t]+|$)/;
  // Blockquote/list recursion ceiling. Real markdown nests a handful of
  // levels; 24 is far above that and far below the ~10k frames that
  // overflow the stack on a crafted `>>> …` reply.
  const MAX_DEPTH = 24;
  const MATH_OPEN = /^[ \t]*(\$\$|\\\[)/;
  const TABLE_SEP = /^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$/;

  const INLINE = new RegExp([
    String.raw`(\x60+)([^\x60]|[^\x60][\s\S]*?[^\x60])\1(?!\x60)`,
    String.raw`\\\(([\s\S]+?)\\\)`,
    String.raw`\$\$([^$]+?)\$\$`,
    String.raw`\\([\\\x60*_{}\[\]()#+\-.!~|$<>])`,
    String.raw`\*\*(?=\S)([\s\S]*?\S)\*\*`,
    String.raw`(?<!\w)__(?=\S)([\s\S]*?\S)__(?!\w)`,
    String.raw`~~(?=\S)([\s\S]*?\S)~~`,
    String.raw`\*(?=[^\s*])((?:\*\*[\s\S]+?\*\*|[^*])*?[^\s*])\*(?!\*)`,
    String.raw`(?<!\w)_(?=[^\s_])([\s\S]*?[^\s_])_(?!\w)`,
    String.raw`\[((?:[^\[\]\n]|\[[^\[\]\n]*\])+)\]\([ \t]*<?((?:[^()\s<>]|\([^()\s]*\))+)>?(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'))?[ \t]*\)`,
    String.raw`<((?:https?:\/\/|mailto:)[^>\s]+)>`,
    String.raw`(https?:\/\/[^\s<>]+)`,
    String.raw`\$(?=[^\s$])([^$\n]*?[^\s$\\])\$(?![\w$])`,
    String.raw`(\x60+)`,
  ].join("|"), "g");

  const detab = (s) => s.replace(/^[ \t]+/, (w) => w.replace(/\t/g, "    "));
  const indentOf = (s) => detab(s).match(/^ */)[0].length;
  const dedent = (s, n) => detab(s).replace(new RegExp(`^ {0,${n}}`), "");
  const isBlockStart = (l) =>
    FENCE.test(l) || HEADING.test(l) || HR.test(l) || QUOTE.test(l) || ITEM.test(l) || MATH_OPEN.test(l);
  const isTableStart = (lines, i) =>
    lines[i].includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]) &&
    lines[i + 1].includes("-") && (lines[i + 1].includes("|") || lines[i].trim().startsWith("|"));

  /* ---------- TeX → readable unicode (no math library) ---------- */
  const TEX = {
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", zeta: "ζ", eta: "η",
    theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π",
    rho: "ρ", sigma: "σ", tau: "τ", upsilon: "υ", phi: "φ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
    Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
    cdot: "·", times: "×", div: "÷", pm: "±", mp: "∓", leq: "≤", le: "≤", geq: "≥", ge: "≥", neq: "≠", ne: "≠",
    approx: "≈", equiv: "≡", sim: "∼", simeq: "≃", propto: "∝", infty: "∞", sum: "∑", prod: "∏", int: "∫",
    oint: "∮", partial: "∂", nabla: "∇", in: "∈", notin: "∉", subset: "⊂", subseteq: "⊆", supset: "⊃",
    supseteq: "⊇", cup: "∪", cap: "∩", emptyset: "∅", forall: "∀", exists: "∃", neg: "¬", land: "∧",
    lor: "∨", to: "→", rightarrow: "→", leftarrow: "←", Rightarrow: "⇒", Leftarrow: "⇐", iff: "⇔",
    leftrightarrow: "↔", mapsto: "↦", ldots: "…", cdots: "⋯", dots: "…", circ: "∘", angle: "∠",
    perp: "⊥", parallel: "∥", hbar: "ℏ", ell: "ℓ", log: "log", ln: "ln", sin: "sin", cos: "cos",
    tan: "tan", exp: "exp", lim: "lim", max: "max", min: "min", det: "det", quad: "  ", qquad: "    ",
    ",": " ", ";": " ", ":": " ", "!": "", "{": "{", "}": "}", "%": "%", "_": "_", "&": "&", "#": "#", "\\": "\n",
  };
  const SUP = mapChars("0123456789+-=()nia", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃ");
  const SUB = mapChars("0123456789+-=()aeoxijkmnt", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓᵢⱼₖₘₙₜ");
  function mapChars(from, to) {
    const a = [...from], b = [...to], m = new Map();
    a.forEach((c, i) => m.set(c, b[i]));
    return m;
  }
  const script = (s, table, mark) =>
    [...s].every((c) => table.has(c)) ? [...s].map((c) => table.get(c)).join("") : `${mark}(${s})`;
  const group = (s) => (s.length > 1 && !/^[\w.]+$/.test(s) ? `(${s})` : s);

  function prettyTex(src) {
    return src
      .replace(/\\(?:begin|end)\{[^}]*\}/g, "")
      .replace(/\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|operatorname|mathcal|mathbb|boldsymbol)\{([^{}]*)\}/g, "$1")
      .replace(/\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}/g, (_, a, b) => `${group(a)}/${group(b)}`)
      .replace(/\\sqrt\{([^{}]*)\}/g, (_, a) => `√${group(a)}`)
      .replace(/\\(?:left|right|big|Big|bigg|Bigg)\b/g, "")
      .replace(/\\([A-Za-z]+|[,;:!{}%_&#\\])/g, (m, w) => TEX[w] ?? m)
      .replace(/\^\{([^{}]*)\}|\^([\w+-])/g, (_, a, b) => script(a ?? b, SUP, "^"))
      .replace(/_\{([^{}]*)\}|_([\w])/g, (_, a, b) => script(a ?? b, SUB, "_"))
      .replace(/&/g, " ")
      .replace(/[{}]/g, "");
  }

  function mathNode(src, display) {
    const n = el(display ? "div" : "span", display ? "math-display" : "math-inline", prettyTex(src.trim()));
    n.title = src.trim();
    return n;
  }

  /* ---------- inline ---------- */
  const safeHref = (u) => (/^(https?:|mailto:)/i.test(u) ? u : null);
  const count = (s, c) => s.split(c).length - 1;

  function anchor(href) {
    const a = el("a");
    a.href = href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  }

  function inline(parent, text) {
    const rx = new RegExp(INLINE.source, "g");
    let last = 0;
    let m;
    const wrap = (tag, inner) => {
      const n = el(tag);
      inline(n, inner);
      parent.append(n);
    };
    while ((m = rx.exec(text))) {
      if (m.index > last) parent.append(text.slice(last, m.index));
      last = rx.lastIndex;
      if (m[1] != null) {
        let c = m[2];
        if (c.length > 2 && c.startsWith(" ") && c.endsWith(" ") && c.trim()) c = c.slice(1, -1);
        parent.append(el("code", "code-inline", c));
      } else if (m[3] != null) parent.append(mathNode(m[3], false));
      else if (m[4] != null) parent.append(mathNode(m[4], false));
      else if (m[5] != null) parent.append(m[5]);
      else if (m[6] != null || m[7] != null) wrap("strong", m[6] ?? m[7]);
      else if (m[8] != null) wrap("del", m[8]);
      else if (m[9] != null || m[10] != null) wrap("em", m[9] ?? m[10]);
      else if (m[11] != null) {
        const href = safeHref(m[12]);
        const a = href ? anchor(href) : el("span");
        inline(a, m[11]);
        parent.append(a);
      } else if (m[13] != null) {
        const a = anchor(m[13]);
        a.textContent = m[13].replace(/^mailto:/i, "");
        parent.append(a);
      } else if (m[14] != null) {
        let url = m[14];
        let tail = "";
        while (url.length > 8 && (/[.,;:!?'"*_~]$/.test(url) ||
            (url.endsWith(")") && count(url, "(") < count(url, ")")) ||
            (url.endsWith("]") && count(url, "[") < count(url, "]")))) {
          tail = url.slice(-1) + tail;
          url = url.slice(0, -1);
        }
        const a = anchor(url);
        a.textContent = url;
        parent.append(a);
        if (tail) parent.append(tail);
      } else if (m[15] != null) parent.append(mathNode(m[15], false));
      else if (m[16] != null) parent.append(m[16]);
    }
    if (last < text.length) parent.append(text.slice(last));
  }

  /* ---------- blocks ---------- */
  function codeBlock(lang, code) {
    const wrap = el("div", "code-block");
    const head = el("div", "code-head");
    head.append(el("span", "code-lang", lang || "text"));
    const wrapBtn = miniBtn("wrap", "Toggle word wrap");
    wrapBtn.addEventListener("click", () => {
      wrapBtn.classList.toggle("on", wrap.classList.toggle("wrap"));
    });
    const dl = miniBtn("download", "Download");
    dl.addEventListener("click", () => downloadText(Highlight.fileName(lang), code));
    const cp = miniBtn("copy", "Copy code", "Copy");
    cp.addEventListener("click", () => copyText(code, cp));
    head.append(wrapBtn, dl, cp);
    const pre = el("pre");
    const c = el("code");
    c.append(Highlight.highlight(code, lang));
    pre.append(c);
    wrap.append(head, pre);
    return wrap;
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    const cells = [];
    let cur = "";
    let tick = false;
    for (let k = 0; k < s.length; k++) {
      const ch = s[k];
      if (ch === "\\" && s[k + 1] === "|") { cur += "|"; k++; continue; }
      if (ch === "`") tick = !tick;
      if (ch === "|" && !tick) { cells.push(cur.trim()); cur = ""; continue; }
      cur += ch;
    }
    cells.push(cur.trim());
    return cells;
  }

  function cell(tag, text, align) {
    const c = el(tag);
    if (align) c.style.textAlign = align;
    inline(c, text.replace(/<br\s*\/?>/gi, "\n"));
    return c;
  }

  function table(lines, i) {
    const head = splitRow(lines[i]);
    const aligns = splitRow(lines[i + 1]).map((c) =>
      c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "");
    const t = el("table");
    const thead = el("thead");
    const tbody = el("tbody");
    const tr = el("tr");
    head.forEach((c, k) => tr.append(cell("th", c, aligns[k])));
    thead.append(tr);
    i += 2;
    while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
      const cells = splitRow(lines[i]);
      const row = el("tr");
      for (let k = 0; k < head.length; k++) row.append(cell("td", cells[k] ?? "", aligns[k]));
      tbody.append(row);
      i++;
    }
    t.append(thead, tbody);
    const wrap = el("div", "table-wrap");
    wrap.append(t);
    return { node: wrap, next: i };
  }

  function list(lines, i, depth) {
    const first = lines[i].match(ITEM);
    const base = indentOf(first[1]);
    const ordered = /\d/.test(first[2]);
    const node = el(ordered ? "ol" : "ul");
    if (ordered) {
      const n = parseInt(first[2], 10);
      if (n !== 1) node.start = n;
    }
    const items = [];
    let loose = false;
    let cur = null;
    while (i < lines.length) {
      const line = lines[i];
      const m = line.match(ITEM);
      if (m && indentOf(m[1]) <= base + 1) {
        if (indentOf(m[1]) < base || /\d/.test(m[2]) !== ordered) break;
        cur = { lines: [line.slice(m[0].length)], pad: indentOf(m[1]) + m[0].length - m[1].length };
        items.push(cur);
        i++;
        continue;
      }
      if (!line.trim()) {
        let j = i + 1;
        while (j < lines.length && !lines[j].trim()) j++;
        if (j >= lines.length) break;
        const nm = lines[j].match(ITEM);
        if (nm && indentOf(nm[1]) >= base && indentOf(nm[1]) <= base + 1 && /\d/.test(nm[2]) === ordered) {
          loose = true;
          i = j;
          continue;
        }
        if (indentOf(lines[j]) >= cur.pad) {
          for (; i < j; i++) cur.lines.push("");
          continue;
        }
        break;
      }
      if (indentOf(line) >= Math.min(cur.pad, base + 2)) {
        cur.lines.push(dedent(line, cur.pad));
        i++;
        continue;
      }
      if (!isBlockStart(line) && cur.lines[cur.lines.length - 1].trim()) {
        cur.lines.push(line.trim());
        i++;
        continue;
      }
      break;
    }
    for (const it of items) {
      const li = el("li");
      const task = it.lines[0].match(/^\[([ xX])\][ \t]+/);
      if (task) {
        it.lines[0] = it.lines[0].slice(task[0].length);
        li.className = "task";
        const done = task[1] !== " ";
        const box = el("span", "task-box" + (done ? " done" : ""));
        if (done) box.append(icon("check"));
        li.append(box);
      }
      const kids = blocks(it.lines, depth + 1);
      if (!loose && kids[0]?.tagName === "P") li.append(...kids.shift().childNodes);
      li.append(...kids);
      node.append(li);
    }
    return { node, next: i };
  }

  function blocks(lines, depth = 0) {
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      let m;

      if ((m = line.match(FENCE))) {
        const [, ind, fence, lang] = m;
        const close = new RegExp(`^ {0,3}${fence[0] === "`" ? "`" : "~"}{${fence.length},}[ \\t]*$`);
        const buf = [];
        i++;
        while (i < lines.length && !close.test(lines[i])) {
          buf.push(ind ? dedent(lines[i], ind.length) : lines[i]);
          i++;
        }
        if (i < lines.length) i++;
        out.push(codeBlock(lang, buf.join("\n")));
        continue;
      }

      if ((m = line.match(MATH_OPEN))) {
        const closer = m[1] === "$$" ? "$$" : "\\]";
        const rest = line.slice(line.indexOf(m[1]) + 2);
        const buf = [];
        let j = rest.indexOf(closer);
        i++;
        if (j >= 0) buf.push(rest.slice(0, j));
        else {
          buf.push(rest);
          while (i < lines.length) {
            j = lines[i].indexOf(closer);
            if (j >= 0) { buf.push(lines[i].slice(0, j)); i++; break; }
            buf.push(lines[i]);
            i++;
          }
        }
        out.push(mathNode(buf.join("\n"), true));
        continue;
      }

      if ((m = line.match(HEADING))) {
        const h = el("h" + m[1].length);
        inline(h, m[2]);
        out.push(h);
        i++;
        continue;
      }

      if (HR.test(line)) {
        out.push(el("hr"));
        i++;
        continue;
      }

      if (QUOTE.test(line)) {
        const buf = [];
        while (i < lines.length && lines[i].trim() && QUOTE.test(lines[i])) {
          buf.push(lines[i].replace(QUOTE, ""));
          i++;
        }
        const bq = el("blockquote");
        // Depth-capped: `>>>>…` (and deep list nesting) costs one frame per
        // level otherwise, so a crafted reply can blow the stack and brick
        // the UI (the saved conversation would fail to boot, too).
        bq.append(...(depth < MAX_DEPTH ? blocks(buf, depth + 1) : [plainPara(buf)]));
        out.push(bq);
        continue;
      }

      if (isTableStart(lines, i)) {
        const t = table(lines, i);
        out.push(t.node);
        i = t.next;
        continue;
      }

      if (ITEM.test(line)) {
        const l = list(lines, i, depth);
        out.push(l.node);
        i = l.next;
        continue;
      }

      const buf = [line.trim()];
      i++;
      while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i]) && !isTableStart(lines, i)) {
        buf.push(lines[i].trim());
        i++;
      }
      const p = el("p");
      inline(p, buf.join("\n"));
      out.push(p);
    }
    return out;
  }

  function plainPara(lines) {
    const p = el("p");
    p.append(lines.join("\n"));
    return p;
  }

  function render(root, src) {
    // Model output is untrusted; rendering must never take the page down.
    // A depth-capped `blocks` keeps crafted nesting from blowing the stack,
    // and this catch keeps any other renderer defect from bricking the
    // conversation (the raw text is still shown).
    try {
      root.replaceChildren(...blocks(src.replace(/\r\n?/g, "\n").split("\n")));
    } catch (e) {
      console.warn("markdown render failed, falling back to plain text", e);
      root.replaceChildren(plainPara([src]));
    }
  }

  return { render };
})();
