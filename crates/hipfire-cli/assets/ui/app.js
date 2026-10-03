/* hipfire embedded chat UI — vanilla JS over the OpenAI gateway.
 *
 * Wire contract (same as every other client of this gateway):
 *   POST /v1/chat/completions, {"stream": true, "stream_options":
 *   {"include_usage": true}} → SSE `data:` frames whose delta carries
 *   `content` (visible tokens) and `reasoning_content` (thinking). Tool
 *   calls are released in terminal chunks as `delta.tool_calls` with
 *   finish_reason "tool_calls"; the final choice chunk carries `timings`
 *   and a `hipfire` extension with the *resolved* reasoning mode/effort.
 *   Client abort → server-side cancellation is wired engine-side, so
 *   AbortController.abort() really does stop GPU work.
 *
 * History persists in IndexedDB (db.js). DOM helpers, icons and the
 * markdown renderer live in render.js; model output only ever reaches the
 * page through textContent.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const mobileMQ = matchMedia("(max-width: 860px)");
const coarseMQ = matchMedia("(pointer: coarse)");
const isMobile = () => mobileMQ.matches;

const LS = {
  theme: "hipfire.theme",
  settings: "hipfire.settings.v2",
  model: "hipfire.model",
  side: "hipfire.sidebar",
};

function lsGet(k) {
  try { return localStorage.getItem(k); } catch { return null; }
}
function lsSet(k, v) {
  try {
    if (v == null) localStorage.removeItem(k);
    else localStorage.setItem(k, v);
  } catch { /* storage disabled */ }
}

function applyTheme(t) {
  if (t === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  for (const b of document.querySelectorAll("[data-theme-opt]")) {
    b.setAttribute("aria-checked", String(b.dataset.themeOpt === t));
  }
}
applyTheme(lsGet(LS.theme) || "system");

const els = {
  sidebar: $("sidebar"), scrim: $("scrim"), main: $("main"),
  collapseSidebar: $("collapse-sidebar"), openSidebar: $("open-sidebar"),
  sidebarNew: $("sidebar-new"), convSearch: $("conv-search"), convList: $("conv-list"),
  storageText: $("storage-text"), importChats: $("import-chats"), importFile: $("import-file"),
  exportAll: $("export-all"), clearAll: $("clear-all"),
  model: $("model"), badges: $("model-badges"),
  statusPill: $("status-pill"), statusDot: $("status-dot"), status: $("status"),
  exportConv: $("export-conv"), showShortcuts: $("show-shortcuts"), toggleSettings: $("toggle-settings"),
  scroller: $("scroller"), greeting: $("greeting"), emptySub: $("empty-sub"), suggestions: $("suggestions"),
  log: $("log"), scrollBottom: $("scroll-bottom"),
  composer: $("composer"), composerWrap: $("composer-wrap"), attachments: $("attachments"), input: $("input"),
  attach: $("attach"), file: $("file"), thinkPill: $("think-pill"), thinkLabel: $("think-label"),
  charCount: $("char-count"), send: $("send"), stopBtn: $("stop-btn"), drop: $("drop-overlay"),
  settings: $("settings"), closeSettings: $("close-settings"),
  resetSettings: $("reset-settings"), doneSettings: $("done-settings"),
  system: $("system"), stopSeq: $("stop_seq"), reasoningGroup: $("reasoning-group"),
  thinking: $("thinking"), thinkingWrap: $("thinking-wrap"),
  effort: $("effort"), hint: $("settings-hint"),
  liveStats: $("live-stats"), liveTtft: $("live-ttft"), liveRateEl: $("live-rate"),
  tools: $("tools"), toolsWrap: $("tools-wrap"),
  toolChoiceRow: $("tool-choice-row"), toolChoice: $("tool_choice"),
  responseFormat: $("response_format"),
  menu: $("menu"), toasts: $("toasts"),
  confirmDlg: $("confirm-dlg"), confirmTitle: $("confirm-title"),
  confirmBody: $("confirm-body"), confirmOk: $("confirm-ok"),
  shortcutsDlg: $("shortcuts-dlg"), shortcutList: $("shortcut-list"),
  lightbox: $("lightbox"), lightboxImg: $("lightbox-img"),
};

const state = {
  convs: new Map(),      // id → conversation record, mirrors IndexedDB
  conv: null,            // open conversation; null = fresh, unsaved chat
  streams: new Map(),   // conversation id → in-flight generation (parallel chats)
  models: [],            // /v1/models entries
  health: null,          // last /health payload
  stats: null,           // last /stats payload
  offline: false,
  polling: false,
  attachment: null,      // {dataUrl, name} — at most one image per request
  drafts: new Map(),     // conversation id ("" = new chat) → unsent text
  stick: true,           // auto-follow the bottom while streaming
  dbOk: true,
  errorCard: null,       // {conv, text} transient failure shown under the log
  saved: {},             // persisted settings snapshot
  effortKey: null,
  speaking: null,
  unread: false,
  menuAnchor: null,
};

const streamOf = (conv) => (conv ? state.streams.get(conv.id) : undefined);
const curStream = () => streamOf(state.conv);
const anyStreaming = () => state.streams.size > 0;

const bc = "BroadcastChannel" in window ? new BroadcastChannel("hipfire-chat") : null;

/* ================= small utilities ================= */

const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
const num = (x) => (typeof x === "number" && Number.isFinite(x) ? x : undefined);
const plural = (n, w) => `${n.toLocaleString()} ${w}${n === 1 ? "" : "s"}`;
const shortModel = (id) => String(id || "").split(/[\\/]/).pop();
const SAFE_IMG = /^data:image\/(?:png|jpeg);base64,/;

function opt(value, label) {
  const o = el("option", null, label);
  o.value = value;
  return o;
}

function fmtClock(ts) {
  return ts ? new Date(ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "";
}
function fmtDur(ms) {
  const s = Math.max(1, Math.round(ms / 1000));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}
function fmtMs(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}
function fmtBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(0)} KB`;
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MB`;
  return `${(b / 1024 ** 3).toFixed(2)} GB`;
}
function fmtCtx(n) {
  return n >= 1024 ? `${Math.round(n / 1024)}K` : String(n);
}
function prettyJSON(s) {
  try { return JSON.stringify(JSON.parse(s), null, 2); } catch { return s ?? ""; }
}
function fileSlug(s) {
  return (s || "chat").replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "chat";
}
function autoTitle(text) {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 60 ? t.slice(0, 57).trimEnd() + "…" : t || "New chat";
}
function keyLabel(k) {
  return isMac ? ({ Ctrl: "⌘", Shift: "⇧", Alt: "⌥" }[k] || k) : k;
}

/* ================= persistence ================= */

let lastSaveError = 0;
function reportSaveError(e) {
  if (Date.now() - lastSaveError < 30000) return;
  lastSaveError = Date.now();
  toast(e?.name === "QuotaExceededError"
    ? "Browser storage is full — delete or export some chats."
    : `Couldn't save chat history: ${e?.message || e}`, { err: true, timeout: 7000 });
}

async function saveConv(conv) {
  state.convs.set(conv.id, conv);
  if (!state.dbOk) return;
  try {
    await ChatDB.put(conv);
    bc?.postMessage({ type: "put", id: conv.id });
    updateStorageInfo();
  } catch (e) {
    reportSaveError(e);
  }
}

const pendingSaves = new Map();  // conv id → timer; parallel streams each save independently
function saveSoon(conv) {
  if (pendingSaves.has(conv.id)) return;
  pendingSaves.set(conv.id, setTimeout(() => {
    pendingSaves.delete(conv.id);
    if (state.convs.has(conv.id)) saveConv(conv);
  }, 1500));
}

function touch(conv) {
  conv.updated = Date.now();
}

function normalizeMsg(m) {
  // "system" is deliberately absent: the UI's system prompt is a setting,
  // not a message, so a system row could only arrive via a crafted chat
  // import — where it would sit invisible in the transcript while being
  // re-sent to the model on every turn.
  if (!m || !["user", "assistant", "tool"].includes(m.role)) return null;
  const out = { ...m, id: typeof m.id === "string" && m.id ? m.id : uid() };
  if (Array.isArray(m.content)) {
    out.content = m.content.filter((p) => p?.type === "text").map((p) => p.text).join("\n");
    const im = m.content.find((p) => p?.type === "image_url");
    if (im?.image_url?.url) out.image = im.image_url.url;
  } else {
    out.content = typeof m.content === "string" ? m.content : "";
  }
  if (out.image && !SAFE_IMG.test(out.image)) delete out.image;
  if (Array.isArray(m.variants)) {
    out.variants = m.variants.map((v) =>
      v && { ...v, tail: (v.tail || []).map(normalizeMsg).filter(Boolean) });
    out.vi = Math.min(Math.max(0, m.vi | 0), out.variants.length - 1);
  } else {
    delete out.variants;
    delete out.vi;
  }
  return out;
}

function normalizeConv(raw) {
  if (!raw || !Array.isArray(raw.messages)) return null;
  const now = Date.now();
  return {
    id: typeof raw.id === "string" && raw.id ? raw.id : uid(),
    title: String(raw.title || "").slice(0, 200) || "New chat",
    titled: !!raw.titled,
    pinned: !!raw.pinned,
    created: num(raw.created) ?? now,
    updated: num(raw.updated) ?? now,
    model: typeof raw.model === "string" ? raw.model : "",
    messages: raw.messages.map(normalizeMsg).filter(Boolean),
  };
}

async function loadHistory() {
  try {
    await ChatDB.open();
    for (const c of await ChatDB.all()) state.convs.set(c.id, c);
  } catch (e) {
    state.dbOk = false;
    toast(`Chat history is unavailable (${e.message}). Chats won't be saved.`, { err: true, timeout: 9000 });
  }
}

let storageTimer = 0;
function updateStorageInfo() {
  clearTimeout(storageTimer);
  storageTimer = setTimeout(async () => {
    const n = state.convs.size;
    if (!state.dbOk) {
      els.storageText.textContent = "History not saved";
      return;
    }
    let text = `${plural(n, "chat")} saved locally`;
    try {
      const est = await navigator.storage?.estimate?.();
      if (est?.usage) text = `${plural(n, "chat")} · ${fmtBytes(est.usage)}`;
    } catch { /* estimate needs a secure context */ }
    els.storageText.textContent = text;
  }, 300);
}

bc?.addEventListener("message", async ({ data }) => {
  if (!data || !state.dbOk) return;
  if (data.type === "put") {
    if (state.streams.has(data.id)) return;
    const c = await ChatDB.get(data.id).catch(() => null);
    if (!c) return;
    state.convs.set(c.id, c);
    if (state.conv?.id === c.id) {
      state.conv = c;
      renderConv({ keepScroll: true });
    }
  } else if (data.type === "del") {
    state.streams.get(data.id)?.abort.abort();
    state.convs.delete(data.id);
    if (state.conv?.id === data.id) openConv(null);
  } else if (data.type === "reload") {
    for (const s of state.streams.values()) s.abort.abort();
    state.convs.clear();
    for (const c of await ChatDB.all().catch(() => [])) state.convs.set(c.id, c);
    if (state.conv && !state.convs.has(state.conv.id)) openConv(null);
  }
  renderConvList();
  updateStorageInfo();
});

/* ================= routing ================= */

function routeId() {
  try { return decodeURIComponent(location.hash.slice(1)) || null; } catch { return null; }
}

function setRoute(id, replace) {
  const next = id ? "#" + encodeURIComponent(id) : "";
  if (location.hash === next) return;
  const url = next || location.pathname + location.search;
  if (replace) history.replaceState(null, "", url);
  else history.pushState(null, "", url);
}

/* ================= sidebar ================= */

function sidebarOpen() {
  return isMobile()
    ? document.body.classList.contains("side-open")
    : !document.body.classList.contains("side-collapsed");
}

function setSidebar(open) {
  if (isMobile()) {
    document.body.classList.toggle("side-open", open);
  } else {
    document.body.classList.toggle("side-collapsed", !open);
    lsSet(LS.side, open ? null : "collapsed");
  }
  syncScrim();
}

function syncScrim() {
  const settingsOpen = els.settings.classList.contains("open");
  els.scrim.hidden = !(settingsOpen || (isMobile() && document.body.classList.contains("side-open")));
}

function dateGroup(ts) {
  const d = new Date(ts);
  const now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(now) - day(d)) / 864e5);
  if (diff <= 0) return "Today";
  if (diff === 1) return "Yesterday";
  if (diff < 7) return "Previous 7 days";
  if (diff < 30) return "Previous 30 days";
  return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function highlightInto(node, text, q) {
  if (!q) {
    node.textContent = text;
    return;
  }
  const lower = text.toLowerCase();
  let at = 0;
  for (;;) {
    const j = lower.indexOf(q, at);
    if (j < 0) break;
    if (j > at) node.append(text.slice(at, j));
    node.append(el("mark", null, text.slice(j, j + q.length)));
    at = j + q.length;
  }
  if (at < text.length) node.append(text.slice(at));
}

function searchConv(c, q) {
  if (c.title.toLowerCase().includes(q)) return { snippet: null };
  for (const m of c.messages) {
    const t = m.content || "";
    const j = t.toLowerCase().indexOf(q);
    if (j < 0) continue;
    const start = Math.max(0, j - 28);
    const snippet = (start ? "…" : "") + t.slice(start, j + q.length + 50).replace(/\s+/g, " ");
    return { snippet };
  }
  return null;
}

function emptyList(iconName, text) {
  const p = el("div", "conv-empty");
  p.append(icon(iconName), el("div", null, text));
  return p;
}

function convGroup(label, items) {
  const g = el("section", "conv-group");
  g.append(el("h4", null, label), ...items);
  return g;
}

function convItem(c, q, hit) {
  const item = el("div", "conv-item" + (state.conv?.id === c.id ? " active" : ""));
  item.dataset.id = c.id;
  const link = el("button", "conv-link");
  link.type = "button";
  link.title = c.title;
  const name = el("span", "conv-name");
  if (q && c.pinned) {
    const pin = icon("pin");
    pin.classList.add("conv-pin");
    name.append(pin);
  }
  highlightInto(name, c.title, q);
  link.append(name);
  if (hit?.snippet) {
    const sn = el("span", "conv-snippet");
    highlightInto(sn, hit.snippet, q);
    link.append(sn);
  }
  link.addEventListener("click", () => openConv(c.id));
  link.addEventListener("dblclick", () => startRename(c, name));
  item.append(link);
  if (state.streams.has(c.id)) item.append(el("span", "conv-streaming"));
  const del = el("button", "icon-btn conv-del");
  del.type = "button";
  del.title = "Delete chat";
  del.setAttribute("aria-label", "Delete chat");
  del.append(icon("trash"));
  del.addEventListener("click", (e) => {
    e.stopPropagation();
    deleteConv(c);
  });
  item.append(del);
  const more = el("button", "icon-btn conv-more");
  more.type = "button";
  more.title = "Chat options";
  more.setAttribute("aria-label", "Chat options");
  more.setAttribute("aria-haspopup", "menu");
  more.append(icon("more"));
  more.addEventListener("click", (e) => {
    e.stopPropagation();
    if (state.menuAnchor === more) closeMenu();
    else openMenu(more, convMenuItems(c));
  });
  item.append(more);
  item.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    openMenu({ x: e.clientX, y: e.clientY }, convMenuItems(c));
  });
  return item;
}

function renderConvList() {
  const raw = els.convSearch.value.trim();
  const q = raw.toLowerCase();
  const list = [...state.convs.values()].sort((a, b) => b.updated - a.updated);
  const frag = document.createDocumentFragment();
  if (q) {
    const hits = [];
    for (const c of list) {
      const hit = searchConv(c, q);
      if (hit) hits.push(convItem(c, q, hit));
    }
    frag.append(hits.length
      ? convGroup(plural(hits.length, "result"), hits)
      : emptyList("search", `No chats match “${raw}”`));
  } else if (!list.length) {
    frag.append(emptyList("message", "No chats yet. Conversations you start are saved here."));
  } else {
    const groups = new Map([["Pinned", []]]);
    for (const c of list) {
      const g = c.pinned ? "Pinned" : dateGroup(c.updated);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(convItem(c));
    }
    for (const [label, items] of groups) if (items.length) frag.append(convGroup(label, items));
  }
  els.convList.replaceChildren(frag);
}

function markActive() {
  for (const n of els.convList.querySelectorAll(".conv-item")) {
    n.classList.toggle("active", n.dataset.id === state.conv?.id);
  }
}

function convMenuItems(c) {
  return [
    { icon: "pencil", label: "Rename", fn: () => {
      const n = els.convList.querySelector(`.conv-item[data-id="${CSS.escape(c.id)}"] .conv-name`);
      if (!n) return;
      if (!sidebarOpen()) setSidebar(true);
      startRename(c, n);
    } },
    { icon: "pin", label: c.pinned ? "Unpin" : "Pin to top", fn: () => {
      c.pinned = !c.pinned;
      saveConv(c);
      renderConvList();
    } },
    { icon: "file", label: "Export as Markdown", fn: () => exportMarkdown(c) },
    { icon: "download", label: "Export as JSON", fn: () => exportJSON([c], `${fileSlug(c.title)}.json`) },
    "sep",
    { icon: "trash", label: "Delete", danger: true, fn: () => deleteConv(c) },
  ];
}

function startRename(c, anchor) {
  const target = anchor.closest(".conv-link");
  const input = el("input", "conv-rename");
  input.value = c.title;
  input.maxLength = 120;
  input.setAttribute("aria-label", "Chat title");
  target.hidden = true;
  target.after(input);
  input.focus();
  input.select();
  let done = false;
  const finish = (commit) => {
    if (done) return;
    done = true;
    const v = input.value.trim();
    input.remove();
    target.hidden = false;
    if (commit && v && v !== c.title) {
      c.title = v;
      c.titled = true;
      saveConv(c);
    }
    renderTopbar();
    renderConvList();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

function deleteConv(c) {
  state.streams.get(c.id)?.abort.abort();
  state.convs.delete(c.id);
  if (state.dbOk) ChatDB.remove(c.id).catch(reportSaveError);
  bc?.postMessage({ type: "del", id: c.id });
  if (state.conv?.id === c.id) openConv(null);
  renderConvList();
  updateStorageInfo();
  toast(`Deleted “${c.title}”`, {
    action: { label: "Undo", fn: () => { saveConv(c); renderConvList(); } },
  });
}

async function clearAll() {
  const n = state.convs.size;
  if (!n) { toast("There are no chats to delete"); return; }
  const ok = await confirmDialog({
    title: "Delete all chats?",
    body: `This permanently removes ${plural(n, "chat")} from this browser. Export them first if you want a backup.`,
    ok: "Delete all",
  });
  if (!ok) return;
  for (const s of state.streams.values()) s.abort.abort();
  state.convs.clear();
  if (state.dbOk) await ChatDB.clear().catch(reportSaveError);
  bc?.postMessage({ type: "reload" });
  openConv(null);
  renderConvList();
  updateStorageInfo();
  toast("All chats deleted");
}

/* ================= conversation view ================= */

function newChat() {
  if (!state.conv) {
    els.input.focus();
    return;
  }
  openConv(null);
}

function openConv(id, opts = {}) {
  const c = id ? state.convs.get(id) || null : null;
  state.drafts.set(state.conv?.id || "", els.input.value);
  const leaving = curStream();
  if (leaving) leaving.view = null;   // stop view updates on the conversation we leave
  state.conv = c;
  state.errorCard = null;
  els.input.value = state.drafts.get(c?.id || "") || "";
  autosize();
  clearAttachment();
  stopSpeaking();
  renderConv();
  markActive();
  renderLiveStats();
  if (opts.push !== false) setRoute(c?.id);
  if (isMobile()) setSidebar(false);
  if (!coarseMQ.matches) els.input.focus();
}

function renderTopbar() {
  els.exportConv.hidden = !state.conv;
  updateDocTitle();
}

function updateDocTitle() {
  const base = state.conv ? `${state.conv.title} · hipfire` : "hipfire";
  document.title = anyStreaming() && document.hidden ? `● Generating… · ${base}`
    : state.unread ? `✓ Response ready · ${base}` : base;
}

function renderEmpty() {
  const h = new Date().getHours();
  const part = h < 5 ? "tonight" : h < 12 ? "this morning" : h < 18 ? "this afternoon" : "this evening";
  els.greeting.textContent = `How can I help ${part}?`;
  const model = els.model.value || state.health?.model;
  els.emptySub.textContent = model
    ? `${shortModel(model)} · running locally on your GPU`
    : "Running locally on your GPU";
}

const SUGGESTIONS = [
  { icon: "code", title: "Write a script", sub: "that renames photos by the date they were taken",
    prompt: "Write a Python script that renames every photo in a folder by the date it was taken (from EXIF), keeping the original extension." },
  { icon: "bulb", title: "Explain a concept", sub: "how KV caching speeds up LLM inference",
    prompt: "Explain how KV caching speeds up LLM inference. Keep it intuitive first, then show the memory math for a 7B model." },
  { icon: "mail", title: "Draft a message", sub: "asking for a deadline extension",
    prompt: "Draft a short, polite message to my manager asking for a one-week extension on a project deadline." },
  { icon: "sparkles", title: "Brainstorm", sub: "weekend projects for a home GPU",
    prompt: "Brainstorm 8 fun weekend projects I could build with a local LLM running on my home GPU." },
];

function renderSuggestions() {
  els.suggestions.replaceChildren(...SUGGESTIONS.map((s) => {
    const b = el("button", "suggestion");
    b.type = "button";
    const text = el("span");
    text.append(el("b", null, s.title), el("small", null, s.sub));
    b.append(icon(s.icon), text);
    b.addEventListener("click", () => {
      els.input.value = s.prompt;
      autosize();
      send();
    });
    return b;
  }));
}

function renderConv(opts = {}) {
  const c = state.conv;
  const prevTop = els.scroller.scrollTop;
  if (state.conv) { const s = curStream(); if (s) s.view = null; }
  const frag = document.createDocumentFragment();
  if (c) for (const m of c.messages) {
    const n = renderMessage(c, m);
    if (n) frag.append(n);
  }
  if (state.errorCard && state.errorCard.conv === c) frag.append(errorNode(state.errorCard));
  els.log.replaceChildren(frag);
  markLast();
  document.body.classList.toggle("has-msgs", els.log.childElementCount > 0);
  renderTopbar();
  renderEmpty();
  if (opts.anchor) {
    els.log.querySelector(`.msg[data-id="${CSS.escape(opts.anchor)}"]`)?.scrollIntoView({ block: "nearest" });
  } else if (opts.keepScroll && !state.stick) {
    els.scroller.scrollTop = prevTop;
  } else {
    state.stick = true;
    scrollToBottom();
  }
}

function markLast() {
  for (const n of els.log.querySelectorAll(".msg.last")) n.classList.remove("last");
  const last = els.log.lastElementChild;
  if (last?.classList.contains("assistant")) last.classList.add("last");
}

function scrollToBottom() {
  els.scroller.scrollTop = els.scroller.scrollHeight;
}

// CSS honors prefers-reduced-motion for transitions; JS-driven smooth
// scrolling must check it itself.
function scrollBehavior() {
  return matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}
function followBottom() {
  if (state.stick) scrollToBottom();
}

function renderMessage(conv, m) {
  if (m.role === "user") return userNode(conv, m);
  if (m.role === "assistant") {
    const v = assistantView(conv, m);
    if (streamOf(conv)?.msg === m) streamOf(conv).view = v;
    return v.node;
  }
  if (m.role === "tool") return toolNode(m);
  return null;
}

function imageEl(src) {
  const img = el("img", "att-img");
  img.src = src;
  img.alt = "Attached image";
  img.loading = "lazy";
  img.addEventListener("click", () => {
    els.lightboxImg.src = src;
    els.lightbox.showModal();
  });
  return img;
}

function timeEl(ts) {
  const t = el("span", "msg-time", fmtClock(ts));
  if (ts) t.title = new Date(ts).toLocaleString();
  return t;
}

function userNode(conv, m) {
  const node = el("div", "msg user");
  node.dataset.id = m.id;
  if (m.image && SAFE_IMG.test(m.image)) node.append(imageEl(m.image));
  const bubble = el("div", "bubble", m.content);
  const acts = el("div", "msg-actions");
  variantNav(acts, conv, m);
  const edit = miniBtn("pencil", "Edit message");
  edit.addEventListener("click", () => startEdit(conv, m, node, bubble, acts));
  const copy = miniBtn("copy", "Copy");
  copy.addEventListener("click", () => copyText(m.content, copy));
  acts.append(timeEl(m.ts), edit, copy);
  node.append(bubble, acts);
  return node;
}

function toolNode(m) {
  const node = el("div", "msg tool");
  node.dataset.id = m.id;
  const avatar = el("div", "avatar");
  avatar.append(icon("wrench"));
  const body = el("div", "msg-body");
  const box = el("div", "tool-result");
  const head = el("header");
  head.append(el("span", null, "Tool result"));
  if (m.name) head.append(el("b", null, m.name));
  box.append(head, el("pre", null, prettyJSON(m.content)));
  body.append(box);
  node.append(avatar, body);
  return node;
}

function errorNode(card) {
  const node = el("div", "msg error");
  const c = el("div", "err-card");
  const t = el("div", "err-text");
  t.append(el("b", null, "Something went wrong"), card.text);
  const acts = el("div", "err-acts");
  const retry = el("button", "btn btn-sm");
  retry.type = "button";
  retry.append(icon("retry"), "Retry");
  retry.addEventListener("click", () => retryAfterError(card.conv));
  const dismiss = el("button", "btn btn-sm btn-ghost", "Dismiss");
  dismiss.type = "button";
  dismiss.addEventListener("click", () => {
    state.errorCard = null;
    node.remove();
    markLast();
    document.body.classList.toggle("has-msgs", els.log.childElementCount > 0);
  });
  acts.append(retry, dismiss);
  c.append(icon("alert"), t, acts);
  node.append(c);
  return node;
}

function chip(text, kind, iconName) {
  const c = el("span", "chip" + (kind ? " " + kind : ""));
  if (iconName) c.append(icon(iconName));
  c.append(text);
  return c;
}

function statsText(m) {
  const x = m.meta || {};
  const u = x.usage || {};
  const short = [];
  const long = [];
  if (m.model) long.push(`Model: ${m.model}`);
  if (x.tps) {
    short.push(`${x.tps.toFixed(1)} tok/s`);
    long.push(x.tpsIncludesPrefill
      ? `${x.tps.toFixed(1)} tok/s (wall — includes prefill)`
      : `Decode: ${x.tps.toFixed(1)} tok/s`);
  }
  if (u.completion_tokens) short.push(plural(u.completion_tokens, "token"));
  if (x.ttft != null) long.push(`Time to first token: ${fmtMs(x.ttft)}`);
  if (x.prefillTps) long.push(`Prefill: ${Math.round(x.prefillTps)} tok/s`);
  if (u.prompt_tokens != null) {
    const cached = u.prompt_tokens_details?.cached_tokens;
    long.push(`Tokens: ${u.prompt_tokens} prompt + ${u.completion_tokens ?? 0} completion` +
      (cached ? ` (${cached} cached)` : ""));
  }
  if (x.tau) long.push(`Speculative τ: ${x.tau.toFixed(2)}`);
  if (x.effort) long.push(`Reasoning: ${x.effort}`);
  if (m.thinkOffFallback) long.push("Thinking auto-disabled (unclosable within token budget)");
  if (m.thinkMs) long.push(`Thinking time: ${fmtDur(m.thinkMs)}`);
  if (m.ts) long.push(new Date(m.ts).toLocaleString());
  return { short: short.join(" · ") || fmtClock(m.ts), long: long.join("\n") };
}

function assistantView(conv, m) {
  const node = el("div", "msg assistant");
  node.dataset.id = m.id;
  const avatar = el("div", "avatar");
  avatar.append(icon("flame"));
  const body = el("div", "msg-body");
  const md = el("div", "md bubble");
  const tools = el("div", "toolcalls");
  const meta = el("div", "meta");
  const acts = el("div", "msg-actions");
  body.append(md, tools, meta, acts);
  node.append(avatar, body);

  let think = null;
  let thinkBody = null;
  let thinkLabel = null;
  let userToggled = false;
  let renderedLen = -1;

  function ensureThink() {
    if (think) return;
    think = el("details", "think");
    const sum = el("summary");
    thinkLabel = el("span", "think-label");
    const chev = icon("chevDown");
    chev.classList.add("chev");
    sum.append(icon("bulb"), thinkLabel, chev);
    sum.addEventListener("click", () => { userToggled = true; });
    thinkBody = el("div", "think-body");
    think.append(sum, thinkBody);
    body.prepend(think);
  }

  function waitLabel() {
    const h = state.health;
    if (h?.loading_model) return `Loading ${shortModel(h.loading_model)}…`;
    if (state.stats?.queue_depth > 1) return "Waiting in queue…";
    return "";
  }

  function update() {
    const live = streamOf(conv)?.msg === m;
    node.classList.toggle("live", live);

    if (m.reasoning) {
      ensureThink();
      thinkBody.textContent = m.reasoning;
      const thinking = live && !m.content;
      think.classList.toggle("live", thinking);
      thinkLabel.textContent = thinking ? "Thinking…"
        : m.thinkMs ? `Thought for ${fmtDur(m.thinkMs)}` : "Thoughts";
      if (!userToggled) think.open = thinking;
      if (thinking && think.open) thinkBody.scrollTop = thinkBody.scrollHeight;
    }

    if (!m.content) {
      renderedLen = 0;
      md.classList.remove("streaming");
      if (live && !m.reasoning) {
        let t = md.querySelector(".typing");
        if (!t) {
          t = el("div", "typing");
          const dots = el("i");
          dots.append(el("span"));
          t.append(dots, el("span", "typing-label"));
          md.replaceChildren(t);
        }
        t.querySelector(".typing-label").textContent = waitLabel();
      } else if (!live && !m.tool_calls?.length) {
        md.replaceChildren(el("p", "muted-note", m.stopped ? "Stopped before any output." : "No response."));
      } else {
        md.replaceChildren();
      }
    } else if (m.content.length !== renderedLen || !live) {
      renderedLen = m.content.length;
      Markdown.render(md, m.content);
      md.classList.toggle("streaming", live);
    }

    if (live) {
      meta.hidden = true;
      return;
    }
    tools.replaceChildren(...(m.tool_calls || []).map((tc) => toolCallNode(conv, m, tc)));
    meta.replaceChildren();
    if (m.stopped) meta.append(chip("Stopped", "warn", "stop"));
    if (m.finish === "length") meta.append(chip("Cut off at the token limit", "warn", "alert"));
    if (m.error) meta.append(chip(m.error, "warn", "alert"));
    for (const w of m.meta?.warnings || []) meta.append(chip(w, "warn", "alert"));
    meta.hidden = !meta.childElementCount;

    acts.replaceChildren();
    variantNav(acts, conv, m);
    const copy = miniBtn("copy", "Copy response");
    copy.addEventListener("click", () => copyText(m.content, copy));
    const regen = miniBtn("retry", "Regenerate");
    regen.addEventListener("click", () => regenerate(conv, m));
    acts.append(copy, regen);
    if ("speechSynthesis" in window && m.content) {
      const sp = miniBtn("speaker", "Read aloud");
      sp.classList.toggle("on", state.speaking === m);
      sp.addEventListener("click", () => speak(m, sp, md));
      acts.append(sp);
    }
    const stats = statsText(m);
    const s = el("span", "msg-stats", stats.short);
    s.title = stats.long;
    acts.append(s);
  }

  update();
  return { node, update };
}

function toolCallNode(conv, m, tc) {
  const name = tc.function?.name || "?";
  const box = el("div", "toolcall");
  const head = el("header");
  head.append(icon("wrench"), el("span", null, "Tool call"), el("b", null, name));
  box.append(head, el("pre", null, prettyJSON(tc.function?.arguments)));

  const idx = conv.messages.indexOf(m);
  const after = conv.messages.slice(idx + 1);
  const answered = after.some((x) => x.role === "tool" && tc.id && x.tool_call_id === tc.id);
  if (answered || !after.every((x) => x.role === "tool") || streamOf(conv)) return box;

  const reply = el("div", "tc-reply");
  const ta = el("textarea");
  ta.placeholder = `Paste the result of ${name}…  (Ctrl+Enter to send)`;
  ta.spellcheck = false;
  const row = el("div", "tc-row");
  const sendB = el("button", "btn btn-accent btn-sm", "Send result");
  sendB.type = "button";
  row.append(sendB);
  reply.append(ta, row);
  box.append(reply);

  const submit = () => {
    if (!ta.value.trim()) { ta.focus(); return; }
    if (streamOf(conv)) { busyToast(); return; }
    conv.messages.push({
      id: uid(), role: "tool", tool_call_id: tc.id || undefined,
      name: tc.function?.name, content: ta.value, ts: Date.now(),
    });
    touch(conv);
    const pending = (m.tool_calls || []).filter((t) =>
      t.id && !conv.messages.some((x) => x.role === "tool" && x.tool_call_id === t.id));
    if (pending.length && tc.id) {
      saveConv(conv);
      renderConv();
    } else {
      startAssistant(conv);
    }
  };
  sendB.addEventListener("click", submit);
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); }
  });
  return box;
}

/* ================= branching =================
 * Regenerate and edit fork a message into variants. Each inactive variant
 * keeps its own `tail` (the messages that followed it), so ‹ › restores a
 * whole branch. The active variant's slot is null; its data lives on the
 * message and in conv.messages. */

const VKEYS = ["content", "image", "reasoning", "thinkMs", "tool_calls", "meta",
  "finish", "stopped", "error", "ts", "model"];

function snapshot(conv, idx) {
  const m = conv.messages[idx];
  const v = { tail: conv.messages.slice(idx + 1) };
  for (const k of VKEYS) if (m[k] !== undefined) v[k] = m[k];
  return v;
}

function restore(conv, idx, v) {
  const m = conv.messages[idx];
  for (const k of VKEYS) delete m[k];
  for (const k of VKEYS) if (v[k] !== undefined) m[k] = v[k];
  m.content ??= "";
  conv.messages.splice(idx + 1, Infinity, ...(v.tail || []));
}

function selectVariant(conv, idx, vi) {
  const m = conv.messages[idx];
  m.variants[m.vi] = snapshot(conv, idx);
  restore(conv, idx, m.variants[vi]);
  m.variants[vi] = null;
  m.vi = vi;
}

function newVariant(conv, idx, fields) {
  const m = conv.messages[idx];
  if (!m.variants) {
    m.variants = [null];
    m.vi = 0;
  }
  m.variants[m.vi] = snapshot(conv, idx);
  restore(conv, idx, { ...fields, tail: [] });
  m.vi = m.variants.length;
  m.variants.push(null);
}

function dropVariant(conv, idx) {
  const m = conv.messages[idx];
  m.variants.pop();
  const vi = m.variants.length - 1;
  restore(conv, idx, m.variants[vi]);
  m.variants[vi] = null;
  m.vi = vi;
  if (m.variants.length === 1) {
    delete m.variants;
    delete m.vi;
  }
}

function variantNav(parent, conv, m) {
  const n = m.variants?.length || 0;
  if (n < 2) return;
  const nav = el("span", "variant-nav");
  const prev = miniBtn("chevLeft", "Previous version");
  const next = miniBtn("chevRight", "Next version");
  prev.disabled = m.vi === 0;
  next.disabled = m.vi === n - 1;
  prev.addEventListener("click", () => switchVariant(conv, m, m.vi - 1));
  next.addEventListener("click", () => switchVariant(conv, m, m.vi + 1));
  nav.append(prev, el("span", null, `${m.vi + 1} / ${n}`), next);
  parent.append(nav);
}

function switchVariant(conv, m, vi) {
  if (streamOf(conv)) { busyToast(); return; }
  const idx = conv.messages.indexOf(m);
  if (idx < 0 || vi < 0 || vi >= m.variants.length || vi === m.vi) return;
  selectVariant(conv, idx, vi);
  saveConv(conv);
  renderConv({ anchor: m.id });
}

/* ================= chat actions ================= */

function busyToast() {
  toast("This chat is still generating — press Esc to stop it, or keep working in another chat.");
}

function canGenerate(conv, extraImage) {
  const err = preflight(!!extraImage || !!conv?.messages.some((m) => m.role === "user" && m.image));
  if (err) {
    toast(err, { err: true, timeout: 6000 });
    openSettings();
    return false;
  }
  return true;
}

function send() {
  const text = els.input.value.trim();
  if (!text) return;
  if (streamOf(state.conv)) { busyToast(); return; }
  if (!canGenerate(state.conv, state.attachment)) return;
  let c = state.conv;
  if (!c) {
    c = {
      id: uid(), title: autoTitle(text), titled: false, pinned: false,
      created: Date.now(), updated: Date.now(), model: els.model.value, messages: [],
    };
    state.convs.set(c.id, c);
    state.conv = c;
    setRoute(c.id, true);
  }
  const m = { id: uid(), role: "user", content: text, ts: Date.now() };
  if (state.attachment) m.image = state.attachment.dataUrl;
  c.messages.push(m);
  if (!c.titled && c.messages.filter((x) => x.role === "user").length === 1) c.title = autoTitle(text);
  els.input.value = "";
  state.drafts.delete(c.id);
  autosize();
  clearAttachment();
  touch(c);
  startAssistant(c);
}

function startAssistant(conv) {
  const m = { id: uid(), role: "assistant", content: "", ts: Date.now() };
  conv.messages.push(m);
  return generate(conv, m);
}

function regenerate(conv, m) {
  if (streamOf(conv)) { busyToast(); return; }
  if (!canGenerate(conv)) return;
  const idx = conv.messages.indexOf(m);
  if (idx < 0) return;
  newVariant(conv, idx, { content: "", ts: Date.now() });
  touch(conv);
  generate(conv, m);
}

function retryAfterError(conv) {
  state.errorCard = null;
  if (streamOf(conv)) { busyToast(); return; }
  const last = conv.messages[conv.messages.length - 1];
  if (last?.role === "assistant") regenerate(conv, last);
  else if (canGenerate(conv)) startAssistant(conv);
}

function startEdit(conv, m, node, bubble, acts) {
  if (streamOf(conv)) { busyToast(); return; }
  const box = el("div", "edit-box");
  const ta = el("textarea");
  ta.value = m.content;
  ta.setAttribute("aria-label", "Edit message");
  const foot = el("div", "edit-foot");
  const hint = el("small", null, "Sending starts a new branch; the original stays available with ‹ ›.");
  const cancel = el("button", "btn btn-ghost btn-sm", "Cancel");
  const save = el("button", "btn btn-accent btn-sm", "Send");
  cancel.type = save.type = "button";
  foot.append(hint, cancel, save);
  box.append(ta, foot);
  bubble.replaceWith(box);
  acts.hidden = true;
  const fit = () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight + 2, innerHeight * 0.5) + "px";
  };
  fit();
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  const close = () => {
    box.replaceWith(bubble);
    acts.hidden = false;
  };
  const commit = () => {
    const text = ta.value.trim();
    if (!text) return;
    if (streamOf(conv)) { busyToast(); return; }
    if (!canGenerate(conv)) return;
    const idx = conv.messages.indexOf(m);
    if (idx < 0) return;
    newVariant(conv, idx, { content: text, image: m.image, ts: Date.now() });
    touch(conv);
    startAssistant(conv);
  };
  cancel.addEventListener("click", close);
  save.addEventListener("click", commit);
  ta.addEventListener("input", fit);
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); commit(); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); }
  });
}

function speak(m, btn, mdNode) {
  const synth = window.speechSynthesis;
  if (state.speaking === m) {
    stopSpeaking();
    return;
  }
  stopSpeaking();
  const clone = mdNode.cloneNode(true);
  for (const n of clone.querySelectorAll(".code-head")) n.remove();
  const u = new SpeechSynthesisUtterance(clone.textContent.slice(0, 32000));
  state.speaking = m;
  btn.classList.add("on");
  u.onend = u.onerror = () => {
    if (state.speaking === m) state.speaking = null;
    btn.classList.remove("on");
  };
  synth.speak(u);
}

function stopSpeaking() {
  if (!state.speaking) return;
  state.speaking = null;
  window.speechSynthesis?.cancel();
  for (const b of els.log.querySelectorAll('.mini-btn.on[aria-label="Read aloud"]')) b.classList.remove("on");
}

/* ================= generation ================= */

function scheduleView(s) {
  if (s.pending) return;
  s.pending = true;
  const run = () => {
    s.pending = false;
    if (state.streams.get(s.conv.id) !== s) return;
    if (state.conv === s.conv) {
      s.view?.update();
      followBottom();
    }
    if (performance.now() - s.statusAt > 500) {
      s.statusAt = performance.now();
      renderStatus();
    }
  };
  if (s.msg.content.length > 30000) setTimeout(() => requestAnimationFrame(run), 150);
  else requestAnimationFrame(run);
}

function liveRate() {
  const s = curStream();
  if (!s?.firstAt || s.deltas < 4) return null;
  const secs = (performance.now() - s.firstAt) / 1000;
  return secs > 0.3 ? (s.deltas / secs).toFixed(1) : null;
}

/* Strip above the composer: TTFT + token rate. While streaming it counts
 * client-side (first delta timestamp vs now); afterwards it pins the
 * server's reported values for the last assistant message in view. */
function renderLiveStats() {
  const s = curStream();
  if (s) {
    const ttft = s.firstAt ? `${fmtMs(s.firstAt - s.t0)}` : "…";
    const r = liveRate();
    els.liveTtft.textContent = `TTFT ${ttft}`;
    els.liveRateEl.textContent = r ? `${r} tok/s` : "";
    els.liveStats.hidden = false;
    return;
  }
  const last = state.conv?.messages.findLast((m) => m.role === "assistant" && (m.meta?.ttft != null || m.meta?.tps));
  if (!last) {
    els.liveStats.hidden = true;
    return;
  }
  const parts = [];
  if (last.meta.ttft != null) parts.push(`TTFT ${fmtMs(last.meta.ttft)}`);
  if (last.meta.tps) parts.push(`${last.meta.tps.toFixed(1)} tok/s`);
  els.liveTtft.textContent = parts[0] || "";
  els.liveRateEl.textContent = parts.slice(1).join(" · ");
  els.liveStats.hidden = !parts.length;
}
function collectMeta(m, chunk) {
  const t = chunk.timings || {};
  const hip = chunk.hipfire || {};
  const r = hip.reasoning;
  // Only a genuine daemon-reported decode_tok_s may be labelled "decode".
  // The slots route's done carries just the wall-inclusive tok_s; keeping
  // the number but marking it means the UI never presents prefill time as
  // decode speed (it was 3x off measured on gfx1101 multi-slot).
  const decodeTps = num(t.decode_tok_s) ?? num(hip.decode_tok_s);
  const wallTps = decodeTps == null ? num(hip.tok_s) : null;
  m.meta = {
    ...(m.meta || {}),
    ttft: num(t.ttft_ms),
    tps: decodeTps ?? wallTps,
    ...(wallTps != null ? { tpsIncludesPrefill: true } : {}),
    prefillTps: num(t.prefill_tok_s),
    tau: num(t.tau),
    effort: r?.mode === "enabled" ? r.effort || "on" : undefined,
    warnings: hip.config_warnings?.length ? hip.config_warnings : undefined,
  };
}

async function readSSE(body, onChunk) {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") {
          reader.cancel().catch(() => {});
          return;
        }
        let chunk;
        try { chunk = JSON.parse(payload); } catch { continue; }
        onChunk(chunk);
      }
    }
  }
}

async function generate(conv, m) {
  const idx = conv.messages.indexOf(m);
  // buildBody JSON.parses the Tools / Response-format fields; a throw here
  // must surface as the stream error card, not an unhandled rejection
  // (the tool-result "Send result" path skips the preflight validation).
  let body = null;
  try {
    body = buildBody(conv.messages.slice(0, idx));
  } catch { body = null; }
  m.model = body?.model || "";
  m.ts = Date.now();
  if (m.model) conv.model = m.model;
  const s = {
    conv, msg: m, abort: new AbortController(), view: null,
    t0: performance.now(), firstAt: 0, thinkAt: 0, deltas: 0, pending: false, statusAt: 0,
  };
  state.streams.set(conv.id, s);
  if (state.conv === conv) state.stick = true;
  setBusy();
  if (state.conv === conv) renderConv();
  renderConvList();
  saveConv(conv);

  const toolCalls = new Map();
  let failed = null;
  let attemptBody = body;
  // Two passes max. Pass 2 only happens when the first failed with an
  // unclosable think span (`open_think`): the multi-slot route neither
  // honors think budgets nor accepts reasoning_effort, so a reasoning
  // turn under any max_tokens cap fails closed there. Degrade once to
  // thinking-off — visibly — instead of leaving the user an error.
  for (let pass = 0; pass < 2; pass++) {
    try {
      if (!attemptBody) throw new Error("Settings contain invalid JSON (Tools or Response format).");
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(attemptBody),
        signal: s.abort.signal,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        throw new Error(err?.error?.message || `HTTP ${resp.status} ${resp.statusText}`.trim());
      }
      await readSSE(resp.body, (chunk) => {
        if (chunk.error) throw new Error(chunk.error.message || "the stream reported an error");
        const choice = chunk.choices?.[0];
        const d = choice?.delta;
        const now = performance.now();
        if (d?.reasoning_content) {
          if (!s.thinkAt) s.thinkAt = now;
          if (!s.firstAt) s.firstAt = now;
          m.reasoning = (m.reasoning || "") + d.reasoning_content;
          s.deltas++;
        }
        if (d?.content) {
          if (s.thinkAt && !m.thinkMs) m.thinkMs = Math.round(now - s.thinkAt);
          if (!s.firstAt) s.firstAt = now;
          m.content += d.content;
          s.deltas++;
        }
        for (const tc of d?.tool_calls || []) {
          const prev = toolCalls.get(tc.index) || { index: tc.index, id: "", name: "", arguments: "" };
          if (tc.id) prev.id = tc.id;
          if (tc.function?.name) prev.name += tc.function.name;
          if (tc.function?.arguments) prev.arguments += tc.function.arguments;
          toolCalls.set(tc.index, prev);
        }
        if (choice?.finish_reason) {
          m.finish = choice.finish_reason;
          collectMeta(m, chunk);
        }
        if (chunk.usage) m.meta = { ...(m.meta || {}), usage: chunk.usage };
        scheduleView(s);
        saveSoon(conv);
      });
      failed = null;
      break;
    } catch (e) {
      if (e.name === "AbortError") { m.stopped = true; break; }
      failed = e.message === "Failed to fetch" || e.name === "TypeError"
        ? "Can't reach hipfire serve. Is it still running?"
        : e.message;
      const unclosedThink = /open_think/.test(failed || "");
      if (pass === 0 && unclosedThink && attemptBody?.enable_thinking !== false) {
        // Reset the partial reasoning from the failed pass so the retry
        // starts from a clean bubble (timers included).
        m.reasoning = "";
        m.content = "";
        delete m.thinkMs;
        m.meta = undefined;
        s.thinkAt = 0;
        s.firstAt = 0;
        s.deltas = 0;
        toolCalls.clear();
        attemptBody = { ...attemptBody, enable_thinking: false };
        delete attemptBody.reasoning_effort;
        delete attemptBody.max_think_tokens;
        m.thinkOffFallback = true;
        toast("Thinking couldn't close within the token budget — retrying with thinking off.", { timeout: 6000 });
        continue;
      }
      break;
    }
  }

  if (toolCalls.size) {
    m.tool_calls = [...toolCalls.values()].sort((a, b) => a.index - b.index).map((tc) => ({
      id: tc.id, type: "function", function: { name: tc.name, arguments: tc.arguments },
    }));
  }
  if (s.thinkAt && !m.thinkMs) m.thinkMs = Math.round(performance.now() - s.thinkAt);
  // The slots route's done carries no ttft_ms; pin the client-measured
  // value (first delta minus request send — includes queueing and prefill,
  // i.e. exactly what the user waited) so TTFT survives stream end and is
  // persisted with the message instead of silently disappearing.
  if (m.meta && m.meta.ttft == null && s.firstAt) {
    m.meta.ttft = Math.round(s.firstAt - s.t0);
  }
  const u = m.meta?.usage;
  if (!m.meta?.tps && u?.completion_tokens && s.firstAt) {
    const secs = (performance.now() - s.firstAt) / 1000;
    if (secs > 0) m.meta.tps = u.completion_tokens / secs;
  }

  let structural = false;
  if (failed) {
    state.errorCard = { conv, text: failed };
    structural = true;
    const at = conv.messages.indexOf(m);
    if (!m.content && !m.reasoning && !m.tool_calls) {
      if (m.variants) dropVariant(conv, at);
      else if (at >= 0) conv.messages.splice(at, 1);
    } else {
      m.error = failed;
    }
  }

  if (state.streams.get(conv.id) === s) state.streams.delete(conv.id);
  clearTimeout(pendingSaves.get(conv.id));
  pendingSaves.delete(conv.id);
  if (state.convs.has(conv.id)) {
    touch(conv);
    await saveConv(conv);
  }
  setBusy();
  if (state.conv === conv) {
    if (structural) renderConv({ keepScroll: true });
    else {
      s.view?.update();
      markLast();
      followBottom();
    }
  }
  renderConvList();
  if (document.hidden) state.unread = true;
  updateDocTitle();
  pollStatus();
}

function setBusy() {
  updateComposer();
  renderStatus();
  updateDocTitle();
}

/* ================= request shaping ================= */

function toWire(m, withImage) {
  if (m.role === "user") {
    return {
      role: "user",
      content: withImage && m.image
        ? [{ type: "text", text: m.content }, { type: "image_url", image_url: { url: m.image } }]
        : m.content,
    };
  }
  if (m.role === "assistant") {
    if (!m.content && !m.tool_calls?.length) return null;
    const w = { role: "assistant", content: m.content || "" };
    if (m.tool_calls?.length) w.tool_calls = m.tool_calls;
    return w;
  }
  if (m.role === "tool") {
    const w = { role: "tool", content: m.content };
    if (m.tool_call_id) w.tool_call_id = m.tool_call_id;
    if (m.name) w.name = m.name;
    return w;
  }
  if (m.role === "system") return { role: "system", content: m.content };
  return null;
}

/* Off wins over effort: "reasoning_effort": "off" is the effort-native
 * spelling of disable and the resolver normalises it through
 * disabled-wins. */
function applyThinking(body) {
  if (els.thinkingWrap.hidden) return;
  const t = els.thinking.value;
  if (t === "on") body.enable_thinking = true;
  else if (t === "off") body.enable_thinking = false;
  if (!els.effort.hidden && els.effort.value && t !== "off") body.reasoning_effort = els.effort.value;
}

function buildBody(history) {
  // The gateway takes one image per request: only the latest attachment
  // travels, earlier image turns send their text.
  const lastImage = history.findLast((m) => m.role === "user" && m.image);
  const messages = [];
  const sys = els.system.value.trim();
  if (sys) messages.push({ role: "system", content: sys });
  for (const m of history) {
    const w = toWire(m, m === lastImage);
    if (w) messages.push(w);
  }
  const body = {
    model: els.model.value || state.health?.model || undefined,
    stream: true,
    stream_options: { include_usage: true },
    messages,
  };
  for (const id of ["temperature", "top_p", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty"]) {
    const v = parseFloat($(id).value);
    if (Number.isFinite(v)) body[id] = v;
  }
  for (const id of ["max_tokens", "top_k", "seed"]) {
    const v = parseInt($(id).value, 10);
    if (Number.isFinite(v)) body[id] = v;
  }
  const stops = els.stopSeq.value.split(",").map((s) => s.trim()).filter(Boolean);
  if (stops.length) body.stop = stops;
  applyThinking(body);
  // A reasoning turn that exhausts max_tokens inside <think> fails the
  // whole request closed ("unsafe multi_slot terminal: open_think") —
  // with thinking on by default, any user-set token cap used to turn the
  // next message into an error. Cap thinking to leave the answer the rest
  // of the budget; the engine force-closes the think span at the cap.
  if (body.max_tokens && !els.thinkingWrap.hidden && els.thinking.value !== "off"
      && !body.max_think_tokens) {
    body.max_think_tokens = Math.max(64, Math.floor(body.max_tokens * 0.6));
  }
  const tools = els.tools.value.trim();
  if (tools && !els.toolsWrap.hidden) {
    body.tools = JSON.parse(tools);
    if (els.toolChoice.value !== "auto") body.tool_choice = els.toolChoice.value;
  }
  const rf = els.responseFormat.value.trim();
  if (rf) body.response_format = JSON.parse(rf);
  return body;
}

function validateTools() {
  const raw = els.tools.value.trim();
  let ok = true;
  if (raw) {
    try { ok = Array.isArray(JSON.parse(raw)); } catch { ok = false; }
  }
  els.tools.classList.toggle("invalid", !ok);
  els.toolChoiceRow.hidden = !raw || !ok;
  return ok;
}

function validateResponseFormat() {
  const raw = els.responseFormat.value.trim();
  let ok = true;
  if (raw) {
    try {
      const v = JSON.parse(raw);
      ok = typeof v === "object" && v !== null && !Array.isArray(v);
    } catch { ok = false; }
  }
  els.responseFormat.classList.toggle("invalid", !ok);
  return ok;
}

function preflight(hasImage) {
  const toolsOn = !!els.tools.value.trim() && !els.toolsWrap.hidden;
  if (toolsOn && !validateTools()) return "The Tools field isn't a valid JSON array.";
  if (!validateResponseFormat()) return "The Response format field isn't a valid JSON object.";
  const refused = state.health?.capabilities?.refused_request_fields || [];
  if (toolsOn && hasImage && refused.includes("tools+image")) {
    return "This serve route can't combine tools with an image. Clear the Tools field or start a chat without the image.";
  }
  return null;
}

/* ================= settings ================= */

const SETTING_IDS = [
  "system", "temperature", "top_p", "min_p", "top_k", "max_tokens", "seed",
  "repeat_penalty", "presence_penalty", "frequency_penalty", "stop_seq",
  "thinking", "effort", "tools", "tool_choice", "response_format",
];

function persistSettings() {
  const o = {};
  for (const id of SETTING_IDS) {
    const v = $(id).value;
    if (v) o[id] = v;
  }
  state.saved = o;
  lsSet(LS.settings, JSON.stringify(o));
}

function restoreSettings() {
  try { state.saved = JSON.parse(lsGet(LS.settings) || "{}") || {}; } catch { state.saved = {}; }
  for (const id of SETTING_IDS) {
    const n = $(id);
    const v = state.saved[id];
    // thinking/effort options arrive with /health; they restore from state.saved then
    if (v != null && (n.tagName !== "SELECT" || n.options.length)) n.value = v;
    n.addEventListener(n.tagName === "SELECT" ? "change" : "input", persistSettings);
  }
  validateTools();
  validateResponseFormat();
}

function resetSettings() {
  for (const id of SETTING_IDS) {
    const n = $(id);
    if (n.tagName !== "SELECT") n.value = "";
  }
  els.toolChoice.value = "auto";
  if (els.thinking.options.length) els.thinking.value = "auto";
  if (els.effort.options.length) els.effort.value = "";
  persistSettings();
  validateTools();
  validateResponseFormat();
  syncThinkPill();
  toast("Settings reset to server defaults");
}

function openSettings() {
  els.settings.classList.add("open");
  syncScrim();
}
function closeSettings() {
  els.settings.classList.remove("open");
  syncScrim();
}
function toggleSettings() {
  if (els.settings.classList.contains("open")) closeSettings();
  else openSettings();
}

/* Thinking controls are driven by what the *loaded* model advertises on
 * /health (reasoning_contract + reasoning_efforts, probed from the chat
 * path itself — a /v1/models entry can't tell whether the engine
 * resolves enable_thinking inline, upstream, or not at all). */
function renderThinkingControls() {
  const h = state.health || {};
  const contract = h.reasoning_contract || "unsupported";
  const efforts = h.reasoning_efforts || [];
  const refused = h.capabilities?.refused_request_fields || [];
  const togglable = contract !== "unsupported" && contract !== "muse_glimmer";

  els.thinkingWrap.hidden = !togglable;
  if (togglable && !els.thinking.options.length) {
    els.thinking.replaceChildren(opt("auto", "Auto (model default)"), opt("on", "On"), opt("off", "Off"));
    els.thinking.value = state.saved.thinking || "auto";
  }

  // Showable only when the model's template actually reads the variable.
  // `reasoning_effort_native` is the daemon's render probe: it is true only
  // when setting `reasoning_effort` CHANGES the rendered prompt. Advertising
  // rungs is not enough — a template can accept the rung and ignore its
  // value, which would offer a control that changes nothing. The route-level
  // refusal is still honoured, so a route that drops the field hides the
  // picker again.
  const effortNative = h.reasoning_effort_native === true;
  const showEffort = togglable && effortNative && efforts.length > 0
    && !refused.includes("reasoning_effort");
  els.effort.hidden = !showEffort;
  const key = efforts.join(",");
  if (showEffort && state.effortKey !== key) {
    state.effortKey = key;
    const want = els.effort.value || state.saved.effort || "";
    els.effort.replaceChildren(opt("", "Auto"), ...efforts.map((e) => opt(e, e)), opt("off", "Off"));
    if ([...els.effort.options].some((o) => o.value === want)) els.effort.value = want;
  }

  els.hint.textContent = contract === "muse_glimmer"
    ? "This model always reasons; thinking can't be turned off."
    : togglable ? `Reasoning contract: ${contract}${efforts.length ? ` · efforts: ${efforts.join(", ")}` : ""}` : "";
  els.reasoningGroup.hidden = !togglable && !els.hint.textContent;
  syncThinkPill();
}

function syncThinkPill() {
  const show = !els.thinkingWrap.hidden;
  els.thinkPill.hidden = !show;
  if (!show) return;
  const v = els.thinking.value || "auto";
  els.thinkLabel.textContent = v === "on" ? "Thinking" : v === "off" ? "No thinking" : "Think: auto";
  els.thinkPill.classList.toggle("on", v === "on");
  els.thinkPill.classList.toggle("off", v === "off");
}

function cycleThinking() {
  const order = ["auto", "on", "off"];
  const cur = els.thinking.value || "auto";
  els.thinking.value = order[(order.indexOf(cur) + 1) % order.length];
  persistSettings();
  syncThinkPill();
}

/* ================= discovery & status ================= */

const hasOpt = (v) => [...els.model.options].some((o) => o.value === v);

async function loadModels() {
  try {
    const j = await (await fetch("/v1/models")).json();
    state.models = j.data || [];
    const keep = els.model.value;
    els.model.replaceChildren(...state.models.map((m) => opt(m.id, m.id)));
    if (keep && !hasOpt(keep)) els.model.append(opt(keep, keep));
    if (keep) els.model.value = keep;
    syncModelSelect();
    renderModelBadges();
  } catch { /* retried by pollStatus */ }
}

/* Follow the served model unless the user picked one. The loaded model may
 * be a path that /v1/models doesn't list; it's injected as an option so
 * `model` is always sent (the gateway 400s without it). */
function syncModelSelect() {
  const served = state.health?.model;
  if (served && !hasOpt(served)) els.model.append(opt(served, served));
  const picked = lsGet(LS.model);
  const want = picked && hasOpt(picked) ? picked : served;
  if (want && els.model.value !== want) els.model.value = want;
  renderEmpty();
}

function renderModelBadges() {
  const h = state.health;
  if (!h) return;
  // v0.4.0 wire: per-model facts live on the resident entry's
  // architecture.input_modalities; route facts on /health.capabilities.
  const resident = state.models.find((m) => m.id === h.model);
  const canSee = !!resident?.architecture?.input_modalities?.includes("image");
  const route = h.capabilities || {};
  const badges = [];
  if (canSee) badges.push(el("span", "badge on", "Vision"));
  if (route.multi_slot) badges.push(el("span", "badge", "Multi-slot"));
  if (h.n_ctx) badges.push(el("span", "badge", `${fmtCtx(h.n_ctx)} context`));
  els.badges.replaceChildren(...badges);
  els.attach.hidden = !canSee;
  if (!canSee && state.attachment) clearAttachment();
  els.toolsWrap.hidden = (route.refused_request_fields || []).includes("tools");
}

async function pollStatus() {
  if (state.polling) return;
  state.polling = true;
  try {
    const init = AbortSignal.timeout ? { signal: AbortSignal.timeout(5000) } : {};
    const [health, stats] = await Promise.all([
      fetch("/health", init).then((r) => r.json()),
      fetch("/stats", init).then((r) => r.json()).catch(() => null),
    ]);
    state.health = health;
    state.stats = stats;
    state.offline = false;
    if (!state.models.length) await loadModels();
    syncModelSelect();
    renderModelBadges();
    renderThinkingControls();
  } catch {
    state.health = null;
    state.offline = true;
  } finally {
    state.polling = false;
    renderStatus();
  }
}

function renderStatus() {
  const h = state.health;
  const st = state.stats;
  let cls = "ok";
  let text;
  if (!h) {
    cls = state.offline ? "err" : "";
    text = state.offline ? "serve unreachable" : "connecting…";
  } else if (h.loading_model) {
    cls = "busy";
    text = `loading ${shortModel(h.loading_model)}…`;
  } else if (h.status === "restarting") {
    cls = "busy";
    text = "engine restarting…";
  } else if (h.status === "unhealthy") {
    cls = "err";
    text = "engine down";
  } else if (anyStreaming()) {
    cls = "busy";
    text = "generating…";
  } else {
    text = st?.queue_depth ? `ready · ${st.queue_depth} queued` : "ready";
  }
  els.statusDot.className = "dot" + (cls ? " " + cls : "");
  els.status.textContent = text;
  els.statusPill.title = [
    h?.model && `Model: ${h.model}`,
    h?.n_ctx && `Context: ${h.n_ctx.toLocaleString()} tokens`,
    st?.requests_served != null && `Requests served: ${st.requests_served}`,
  ].filter(Boolean).join("\n") || "hipfire serve";
  renderLiveStats();
}

/* ================= composer & attachments ================= */

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, innerHeight * 0.4) + "px";
  updateComposer();
}

function updateComposer() {
  const len = els.input.value.length;
  els.send.disabled = !els.input.value.trim();
  els.send.hidden = !!curStream();
  els.stopBtn.hidden = !curStream();
  els.charCount.textContent = len > 400 ? `≈${Math.ceil(len / 4).toLocaleString()} tokens` : "";
}

function attachFile(f) {
  if (els.attach.hidden) {
    toast("The loaded model doesn't accept images.", { err: true });
    return;
  }
  if (!/^image\/(png|jpeg)$/.test(f.type)) {
    toast("Only PNG and JPEG images are supported.", { err: true });
    return;
  }
  if (f.size > 20 * 1024 * 1024) {
    toast("That image is larger than 20 MB.", { err: true });
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.attachment = { dataUrl: reader.result, name: f.name };
    renderAttachment();
    els.input.focus();
  };
  reader.readAsDataURL(f);
}

function renderAttachment() {
  els.attachments.replaceChildren();
  if (!state.attachment) return;
  const t = el("div", "thumb");
  const img = el("img");
  img.src = state.attachment.dataUrl;
  img.alt = state.attachment.name || "Attached image";
  const x = el("button");
  x.type = "button";
  x.title = "Remove image";
  x.setAttribute("aria-label", "Remove image");
  x.append(icon("x"));
  x.addEventListener("click", () => {
    clearAttachment();
    els.input.focus();
  });
  t.append(img, x);
  els.attachments.append(t);
}

function clearAttachment() {
  state.attachment = null;
  renderAttachment();
}

/* ================= menus, toasts, dialogs ================= */

function openMenu(anchor, items) {
  closeMenu();
  const m = els.menu;
  m.replaceChildren();
  for (const it of items) {
    if (it === "sep") {
      m.append(el("hr"));
      continue;
    }
    const b = el("button", it.danger ? "danger" : null);
    b.type = "button";
    b.setAttribute("role", "menuitem");
    b.append(icon(it.icon), el("span", null, it.label));
    b.addEventListener("click", () => {
      closeMenu();
      it.fn();
    });
    m.append(b);
  }
  m.hidden = false;
  const r = anchor.getBoundingClientRect
    ? anchor.getBoundingClientRect()
    : { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y };
  let x = r.left;
  let y = r.bottom + 4;
  if (x + m.offsetWidth > innerWidth - 8) x = Math.max(8, r.right - m.offsetWidth);
  if (y + m.offsetHeight > innerHeight - 8) y = Math.max(8, r.top - m.offsetHeight - 4);
  m.style.left = x + "px";
  m.style.top = y + "px";
  if (anchor.setAttribute) {
    state.menuAnchor = anchor;
    anchor.setAttribute("aria-expanded", "true");
  }
  m.querySelector("button")?.focus();
}

function closeMenu() {
  if (els.menu.hidden) return;
  els.menu.hidden = true;
  state.menuAnchor?.setAttribute("aria-expanded", "false");
  state.menuAnchor = null;
}

function toast(text, opts = {}) {
  const t = el("div", "toast" + (opts.err ? " err" : ""));
  t.setAttribute("role", opts.err ? "alert" : "status");
  if (opts.err) {
    const ic = icon("alert");
    ic.classList.add("t-icon");
    t.append(ic);
  }
  t.append(el("span", null, text));
  let timer = 0;
  const close = () => {
    clearTimeout(timer);
    t.classList.add("leaving");
    setTimeout(() => t.remove(), 200);
  };
  if (opts.action) {
    const b = el("button", null, opts.action.label);
    b.type = "button";
    b.addEventListener("click", () => {
      close();
      opts.action.fn();
    });
    t.append(b);
  }
  els.toasts.append(t);
  while (els.toasts.childElementCount > 3) els.toasts.firstElementChild.remove();
  const ms = opts.timeout ?? (opts.action ? 6000 : 3200);
  timer = setTimeout(close, ms);
  t.addEventListener("mouseenter", () => clearTimeout(timer));
  t.addEventListener("mouseleave", () => { timer = setTimeout(close, 2000); });
}

function confirmDialog({ title, body, ok = "Delete" }) {
  return new Promise((resolve) => {
    const d = els.confirmDlg;
    els.confirmTitle.textContent = title;
    els.confirmBody.textContent = body;
    els.confirmOk.textContent = ok;
    d.returnValue = "";
    d.addEventListener("close", () => resolve(d.returnValue === "ok"), { once: true });
    d.showModal();
  });
}

const SHORTCUTS = [
  ["New chat", ["Ctrl", "Shift", "O"]],
  ["Search chats", ["Ctrl", "K"]],
  ["Toggle sidebar", ["Ctrl", "B"]],
  ["Settings", ["Ctrl", "."]],
  ["Delete current chat", ["Ctrl", "Shift", "⌫"]],
  ["Stop generating", ["Esc"]],
  ["Edit last message", ["↑"]],
  ["Focus the composer", ["/"]],
  ["New line", ["Shift", "Enter"]],
  ["Show shortcuts", ["?"]],
];

function showShortcuts() {
  if (!els.shortcutList.childElementCount) {
    for (const [label, keys] of SHORTCUTS) {
      const dd = el("dd");
      for (const k of keys) dd.append(el("kbd", null, keyLabel(k)));
      els.shortcutList.append(el("dt", null, label), dd);
    }
  }
  els.shortcutsDlg.showModal();
}

/* ================= export / import ================= */

function exportPayload(convs) {
  return { format: "hipfire-chat-export", version: 1, exported: new Date().toISOString(), conversations: convs };
}

function exportJSON(convs, name) {
  downloadText(name, JSON.stringify(exportPayload(convs), null, 2), "application/json");
}

function toMarkdown(c) {
  const out = [`# ${c.title}`, "", `_Exported from hipfire on ${new Date().toLocaleString()}_`, ""];
  for (const m of c.messages) {
    if (m.role === "user") {
      out.push("## You", "", m.content, "");
      if (m.image) out.push("_[image attached]_", "");
    } else if (m.role === "assistant") {
      out.push(`## Assistant${m.model ? ` · ${shortModel(m.model)}` : ""}`, "");
      if (m.content) out.push(m.content, "");
      for (const tc of m.tool_calls || []) {
        out.push(`**Tool call:** \`${tc.function?.name}\``, "", "```json", prettyJSON(tc.function?.arguments), "```", "");
      }
    } else if (m.role === "tool") {
      out.push(`### Tool result${m.name ? ` · ${m.name}` : ""}`, "", "```", m.content, "```", "");
    }
  }
  return out.join("\n");
}

function exportMarkdown(c) {
  downloadText(`${fileSlug(c.title)}.md`, toMarkdown(c), "text/markdown;charset=utf-8");
}

async function importChats(file) {
  let data;
  try {
    data = JSON.parse(await file.text());
  } catch {
    toast("That file isn't valid JSON.", { err: true });
    return;
  }
  const raw = Array.isArray(data) ? data : Array.isArray(data?.conversations) ? data.conversations : [data];
  const convs = raw.map(normalizeConv).filter(Boolean);
  if (!convs.length) {
    toast("No chats found in that file.", { err: true });
    return;
  }
  const fresh = convs.filter((c) => !(state.convs.get(c.id)?.updated >= c.updated));
  for (const c of fresh) state.convs.set(c.id, c);
  if (state.dbOk && fresh.length) {
    await ChatDB.putMany(fresh).catch(reportSaveError);
    bc?.postMessage({ type: "reload" });
  }
  renderConvList();
  updateStorageInfo();
  toast(fresh.length ? `Imported ${plural(fresh.length, "chat")}` : "Those chats are already here");
}

/* ================= events ================= */

function wireEvents() {
  els.collapseSidebar.addEventListener("click", () => setSidebar(false));
  els.openSidebar.addEventListener("click", () => setSidebar(true));
  els.scrim.addEventListener("click", () => {
    closeSettings();
    if (isMobile()) setSidebar(false);
  });
  mobileMQ.addEventListener("change", () => {
    document.body.classList.remove("side-open");
    syncScrim();
  });
  els.sidebarNew.addEventListener("click", newChat);
  els.convSearch.addEventListener("input", renderConvList);
  els.convSearch.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && els.convSearch.value) {
      e.stopPropagation();
      els.convSearch.value = "";
      renderConvList();
    } else if (e.key === "Enter") {
      els.convList.querySelector(".conv-link")?.click();
    }
  });
  els.convList.addEventListener("scroll", closeMenu, { passive: true });
  els.exportAll.addEventListener("click", () => {
    if (!state.convs.size) { toast("There are no chats to export"); return; }
    exportJSON([...state.convs.values()], `hipfire-chats-${new Date().toISOString().slice(0, 10)}.json`);
  });
  els.importChats.addEventListener("click", () => els.importFile.click());
  els.importFile.addEventListener("change", () => {
    const f = els.importFile.files[0];
    els.importFile.value = "";
    if (f) importChats(f);
  });
  els.clearAll.addEventListener("click", clearAll);
  for (const b of document.querySelectorAll("[data-theme-opt]")) {
    b.addEventListener("click", () => {
      const t = b.dataset.themeOpt;
      lsSet(LS.theme, t === "system" ? null : t);
      applyTheme(t);
    });
  }

  els.model.addEventListener("change", () => {
    lsSet(LS.model, els.model.value);
    renderEmpty();
  });
  els.exportConv.addEventListener("click", () => state.conv && exportMarkdown(state.conv));
  els.showShortcuts.addEventListener("click", showShortcuts);
  els.toggleSettings.addEventListener("click", toggleSettings);
  els.closeSettings.addEventListener("click", closeSettings);
  els.doneSettings.addEventListener("click", closeSettings);
  els.resetSettings.addEventListener("click", resetSettings);
  els.tools.addEventListener("input", validateTools);
  els.responseFormat.addEventListener("input", validateResponseFormat);
  els.thinking.addEventListener("change", syncThinkPill);
  els.effort.addEventListener("change", syncThinkPill);
  els.thinkPill.addEventListener("click", cycleThinking);

  els.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });
  els.input.addEventListener("input", autosize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !coarseMQ.matches) {
      e.preventDefault();
      send();
    } else if (e.key === "ArrowUp" && !els.input.value && state.conv && !curStream()) {
      const last = state.conv.messages.findLast((m) => m.role === "user");
      const btn = last && els.log.querySelector(`.msg[data-id="${CSS.escape(last.id)}"] [aria-label="Edit message"]`);
      if (btn) {
        e.preventDefault();
        btn.click();
      }
    }
  });
  els.input.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
    if (f) {
      e.preventDefault();
      attachFile(f);
    }
  });
  els.stopBtn.addEventListener("click", () => curStream()?.abort.abort());
  els.attach.addEventListener("click", () => els.file.click());
  els.file.addEventListener("change", () => {
    const f = els.file.files[0];
    els.file.value = "";
    if (f) attachFile(f);
  });

  let dragDepth = 0;
  const hasFiles = (e) => e.dataTransfer?.types?.includes("Files");
  els.main.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    els.drop.querySelector("p").textContent = els.attach.hidden
      ? "The loaded model doesn't accept images" : "Drop an image to attach";
    els.drop.hidden = false;
  });
  els.main.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  els.main.addEventListener("dragleave", () => {
    if (--dragDepth <= 0) {
      dragDepth = 0;
      els.drop.hidden = true;
    }
  });
  els.main.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    els.drop.hidden = true;
    const f = e.dataTransfer?.files?.[0];
    if (f) attachFile(f);
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  els.scroller.addEventListener("scroll", () => {
    const gap = els.scroller.scrollHeight - els.scroller.scrollTop - els.scroller.clientHeight;
    if (gap <= 8) state.stick = true;
    els.scrollBottom.hidden = gap < 240;
  }, { passive: true });
  els.scroller.addEventListener("wheel", (e) => { if (e.deltaY < 0) state.stick = false; }, { passive: true });
  els.scroller.addEventListener("touchmove", () => { state.stick = false; }, { passive: true });
  els.scrollBottom.addEventListener("click", () => {
    state.stick = true;
    els.scroller.scrollTo({ top: els.scroller.scrollHeight, behavior: scrollBehavior() });
  });

  // Fab rides just above whatever height the composer+stats strip currently
  // occupies (it grows as the textarea autosizes).
  new ResizeObserver(() => {
    els.scrollBottom.style.bottom = (els.composerWrap.offsetHeight + 12) + "px";
  }).observe(els.composerWrap);

  els.lightbox.addEventListener("click", () => els.lightbox.close());

  document.addEventListener("pointerdown", (e) => {
    if (els.menu.hidden || els.menu.contains(e.target) || state.menuAnchor?.contains(e.target)) return;
    closeMenu();
  }, true);
  els.menu.addEventListener("keydown", (e) => {
    const items = [...els.menu.querySelectorAll("button")];
    const i = items.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length]?.focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length]?.focus(); }
    else if (e.key === "Tab") closeMenu();
  });
  window.addEventListener("resize", closeMenu);

  document.addEventListener("keydown", onGlobalKey);
  window.addEventListener("popstate", () => openConv(routeId(), { push: false }));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    state.unread = false;
    updateDocTitle();
    pollStatus();
  });
  window.addEventListener("beforeunload", (e) => {
    if (!anyStreaming()) return;
    for (const s of state.streams.values()) saveConv(s.conv);
    e.preventDefault();
    e.returnValue = "";
  });
}

function onGlobalKey(e) {
  const mod = isMac ? e.metaKey : e.ctrlKey;
  const k = e.key.toLowerCase();
  const a = document.activeElement;
  const typing = a && (/^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName) || a.isContentEditable);
  if (document.querySelector("dialog[open]")) return;

  if (e.key === "Escape") {
    if (!els.menu.hidden) closeMenu();
    else if (els.settings.classList.contains("open")) closeSettings();
    else if (isMobile() && document.body.classList.contains("side-open")) setSidebar(false);
    else if (curStream()) curStream().abort.abort();
    else return;
    e.preventDefault();
    return;
  }
  if (mod && e.shiftKey && k === "o") {
    e.preventDefault();
    newChat();
  } else if (mod && !e.shiftKey && k === "k") {
    e.preventDefault();
    setSidebar(true);
    els.convSearch.focus();
    els.convSearch.select();
  } else if (mod && !e.shiftKey && k === "b") {
    e.preventDefault();
    setSidebar(!sidebarOpen());
  } else if (mod && e.key === ".") {
    e.preventDefault();
    toggleSettings();
  } else if (mod && e.shiftKey && e.key === "Backspace" && state.conv) {
    e.preventDefault();
    deleteConv(state.conv);
  } else if (!typing && !mod && !e.altKey) {
    if (e.key === "?") {
      e.preventDefault();
      showShortcuts();
    } else if (e.key === "/") {
      e.preventDefault();
      els.input.focus();
    }
  }
}

/* ================= boot ================= */

async function boot() {
  hydrateIcons(document);
  if (isMac) {
    for (const n of document.querySelectorAll(".kbd-hint")) n.textContent = n.textContent.replace(/Ctrl ?/g, "⌘ ");
    for (const n of document.querySelectorAll("[title*='Ctrl+']")) {
      n.title = n.title.replace(/Ctrl\+/g, "⌘").replace(/Shift\+/g, "⇧");
    }
  }
  if (!isMobile() && lsGet(LS.side) === "collapsed") document.body.classList.add("side-collapsed");
  restoreSettings();
  wireEvents();
  renderSuggestions();
  updateComposer();
  renderStatus();

  await loadHistory();
  const id = routeId();
  openConv(id && state.convs.has(id) ? id : null, { push: false });
  if (id && !state.conv) history.replaceState(null, "", location.pathname + location.search);
  renderConvList();
  updateStorageInfo();

  await Promise.all([loadModels(), pollStatus()]);
  setInterval(() => { if (!document.hidden) pollStatus(); }, 2500);
}

boot();
