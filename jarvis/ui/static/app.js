"use strict";
const $ = (id) => document.getElementById(id);
let cursor = 0, booted = false, revealPending = false;
let lastConvSig = "", lastMemSig = "";

// ── clock ─────────────────────────────────────────────────────────────────
setInterval(() => {
  const d = new Date();
  const c = $("clock");
  if (c) c.textContent = d.toLocaleTimeString("en-GB", { hour12: false });
}, 1000);

// ── boot / welcome sequence ────────────────────────────────────────────────
const BOOT_LINES = [
  "> initializing arc reactor .............. <span class='ok'>OK</span>",
  "> loading neural core .................. <span class='ok'>OK</span>",
  "> mounting memory bank ................. <span class='ok'>OK</span>",
  "> syncing conversation log ............. <span class='ok'>OK</span>",
  "> voice interface ..................... <span class='ok'>ACTIVE</span>",
];
function runBoot(user) {
  return new Promise((resolve) => {
    const log = $("bootlog");
    log.innerHTML = "";
    let i = 0;
    const tick = () => {
      if (i < BOOT_LINES.length) {
        const div = document.createElement("div");
        div.innerHTML = BOOT_LINES[i++];
        log.appendChild(div);
        setTimeout(tick, 260);
      } else {
        const w = $("welcome");
        w.innerHTML = `WELCOME BACK, <span class="accent">${(user || "").toUpperCase()}</span>`;
        w.classList.add("show");
        setTimeout(resolve, 1500);
      }
    };
    setTimeout(tick, 300);
  });
}
async function reveal(user) {
  if (booted) { return; }
  booted = true;
  await runBoot(user);
  $("boot").classList.add("hidden");
  const hud = $("hud");
  hud.classList.remove("hidden");
  requestAnimationFrame(() => hud.classList.add("ready"));
}

// ── rendering ───────────────────────────────────────────────────────────────
function renderProfile(state) {
  const name = state.user || "User";
  $("user-name").textContent = name;
  $("user-initial").textContent = (name[0] || "A").toUpperCase();
  document.title = `J.A.R.V.I.S. · ${name}`;

  const sig = JSON.stringify(state.memories);
  if (sig === lastMemSig) return;
  lastMemSig = sig;
  const list = $("mem-list");
  list.innerHTML = "";
  $("mem-count").textContent = state.memories.length;
  if (!state.memories.length) {
    list.innerHTML = `<li class="empty">No memories stored yet, sir.</li>`;
    return;
  }
  state.memories.forEach((m, idx) => {
    const li = document.createElement("li");
    li.style.animationDelay = `${idx * 0.05}s`;
    const span = document.createElement("span");
    span.className = "mem-text";
    span.textContent = m;
    const del = document.createElement("button");
    del.className = "mem-del";
    del.title = "Delete memory";
    del.textContent = "×";
    del.addEventListener("click", () => deleteMemory(m));
    li.appendChild(span);
    li.appendChild(del);
    list.appendChild(li);
  });
}

// Add / delete memories from the HUD, then re-render immediately from the reply.
async function addMemory(text) {
  text = (text || "").trim();
  if (!text) return;
  try {
    const r = await fetch("/api/memory/add", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await r.json();
    if (data && data.memories) applyMemories(data.memories);
  } catch (_) {}
}

async function deleteMemory(text) {
  try {
    const r = await fetch("/api/memory/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await r.json();
    if (data && data.memories) applyMemories(data.memories);
  } catch (_) {}
}

function applyMemories(memories) {
  lastMemSig = null;            // force renderProfile to redraw the memory list
  if (lastState) { lastState.memories = memories; renderProfile(lastState); }
}

function fmtWhen(ts) {
  try { return new Date(ts * 1000).toLocaleString("en-GB", { day:"2-digit", month:"short", hour:"2-digit", minute:"2-digit" }); }
  catch { return ""; }
}
function renderConversations(state) {
  const sig = JSON.stringify(state.conversations.map(s => [s.id, s.turns.length]));
  if (sig === lastConvSig) return;
  lastConvSig = sig;
  const list = $("conv-list");
  list.innerHTML = "";
  $("conv-count").textContent = state.conversations.length;
  if (!state.conversations.length) {
    list.innerHTML = `<li class="empty">No past conversations yet.</li>`;
    return;
  }
  state.conversations.forEach((s, idx) => {
    const li = document.createElement("li");
    li.dataset.id = s.id;
    li.style.animationDelay = `${idx * 0.05}s`;
    const turns = s.turns.map(t =>
      `<div class="turn ${t.role}"><span class="who">${t.role === "user" ? "YOU" : "JARVIS"}</span>${escapeHtml(t.text)}</div>`
    ).join("");
    li.innerHTML =
      `<div class="conv-head"><span>${escapeHtml(s.title)}</span><span class="conv-when">${fmtWhen(s.started)}</span></div>
       <div class="conv-turns">${turns}</div>`;
    li.querySelector(".conv-head").addEventListener("click", () => li.classList.toggle("open"));
    list.appendChild(li);
  });
}
function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[c]));
}

// ── to-dos + agenda ──────────────────────────────────────────────────────────
let lastTodoSig = "", lastAgendaSig = "";
function renderTodos(state) {
  const todos = state.todos || [];
  const sig = JSON.stringify(todos.map(t => t.id));
  if (sig === lastTodoSig) return;
  lastTodoSig = sig;
  const list = $("todo-list"); list.innerHTML = "";
  $("todo-count").textContent = todos.length;
  if (!todos.length) { list.innerHTML = `<li class="empty">Nothing to do, sir.</li>`; return; }
  todos.forEach((t, i) => {
    const li = document.createElement("li");
    li.style.animationDelay = `${i * 0.05}s`;
    li.innerHTML = `<span class="box"></span><span>${escapeHtml(t.text)}</span>`;
    list.appendChild(li);
  });
}
function fmtAgendaWhen(ts) {
  try {
    const d = new Date(ts * 1000), now = new Date();
    const t = d.toLocaleTimeString("en-GB", { hour:"2-digit", minute:"2-digit" });
    if (d.toDateString() === now.toDateString()) return t;
    const tm = new Date(now); tm.setDate(now.getDate() + 1);
    if (d.toDateString() === tm.toDateString()) return `tmrw ${t}`;
    return d.toLocaleDateString("en-GB", { day:"2-digit", month:"short" }) + ` ${t}`;
  } catch { return ""; }
}
function renderAgenda(state) {
  const items = state.agenda || [];
  const sig = JSON.stringify(items.map(x => [x.id, x.kind]));
  if (sig === lastAgendaSig) return;
  lastAgendaSig = sig;
  const list = $("agenda-list"); list.innerHTML = "";
  $("agenda-count").textContent = items.length;
  if (!items.length) { list.innerHTML = `<li class="empty">Your schedule is clear.</li>`; return; }
  items.forEach((x, i) => {
    const li = document.createElement("li");
    li.className = x.kind === "reminder" ? "reminder" : "";
    li.style.animationDelay = `${i * 0.05}s`;
    const icon = x.kind === "reminder" ? "⏰" : "📅";
    li.innerHTML = `<span class="icon">${icon}</span><span>${escapeHtml(x.title)}</span><span class="when">${fmtAgendaWhen(x.start)}</span>`;
    list.appendChild(li);
  });
}

// ── alarm overlay ────────────────────────────────────────────────────────────
let currentAlarm = null;
function showAlarm(alarm) {
  currentAlarm = alarm;
  $("alarm-text").textContent = alarm.text || "Reminder";
  $("alarm-time").textContent = alarm.due ? `set for ${fmtAgendaWhen(alarm.due)}` : "";
  $("alarm").classList.remove("hidden");
}
function hideAlarm() { currentAlarm = null; $("alarm").classList.add("hidden"); }
$("alarm-stop").addEventListener("click", () => {
  if (currentAlarm) wsSend({ cmd: "dismiss_alarm", id: currentAlarm.id });
  hideAlarm();
});
$("alarm-done").addEventListener("click", () => {
  if (currentAlarm) wsSend({ cmd: "complete_todo", id: currentAlarm.id, text: currentAlarm.text });
  hideAlarm();
});

// ── hide button (dismiss the dashboard window; voice re-opens it) ────────────
$("hide-btn").addEventListener("click", () => {
  window.close();                    // works for the chromeless --app window
  setTimeout(() => document.body.classList.add("dimmed"), 60);  // fallback if not closable
});
$("focus-end").addEventListener("click", () => wsSend({ cmd: "end_focus" }));

// ── focus mode card (technique + live countdown + phase) ─────────────────────
let focusState = null;
function renderFocus(f) {
  const card = $("focus-hud");
  if (!card) return;
  if (!f || !f.active) { card.classList.add("hidden"); focusState = null; return; }
  focusState = f;
  card.classList.remove("hidden");
  $("focus-tech").textContent = (f.technique || "").toUpperCase();
  const phase = (f.phase || "").toLowerCase();
  card.classList.toggle("break", phase === "break");
  $("focus-phase").textContent =
    phase === "break" ? "BREAK" : phase === "starting" ? "STARTING…" : "FOCUS";
  const n = f.cycles || 0;
  $("focus-cycles").textContent = `${n} sprint${n === 1 ? "" : "s"}`;
  tickFocus();
}
function tickFocus() {
  if (!focusState || !focusState.active) return;
  const el = $("focus-timer");
  if (!focusState.ends_at) { el.textContent = "··:··"; return; }
  let rem = Math.max(0, Math.round(focusState.ends_at * 1000 - Date.now()) / 1000);
  const m = Math.floor(rem / 60), s = Math.floor(rem % 60);
  el.textContent = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
setInterval(tickFocus, 1000);

// ── live Q&A + analysis feed (every question + answer + screen analysis) ─────
let typeToken = 0, feedSeen = 0, feedCount = 0;
function feedEl() { return $("feed"); }
function scrollFeed() { const f = feedEl(); f.scrollTop = f.scrollHeight; }
function bumpCount() { $("feed-count").textContent = ++feedCount; }

function addQA(entry) {
  $("analysis-idle").classList.add("hidden");
  const div = document.createElement("div");
  div.className = `qa ${entry.role === "user" ? "user" : "assistant"}`;
  const who = entry.role === "user" ? "YOU" : "JARVIS";
  div.innerHTML = `<span class="who">${who}</span>${escapeHtml(entry.text || "")}`;
  feedEl().appendChild(div);
  bumpCount(); scrollFeed();
}

function addExplanation(entry, animate) {
  $("analysis-idle").classList.add("hidden");
  const card = document.createElement("div");
  card.className = "expl" + (animate ? " flash" : "");
  const img = entry.image
    ? `<img src="data:image/png;base64,${entry.image}" alt="captured screen" />` : "";
  card.innerHTML =
    `<div class="etitle">${escapeHtml(entry.title || "Analysis")}</div>${img}` +
    `<div class="etext"></div>` +
    (entry.source ? `<div class="esrc">— ${escapeHtml(entry.source)}</div>` : "");
  feedEl().appendChild(card);
  bumpCount();
  const textEl = card.querySelector(".etext");
  if (animate) typeText(textEl, entry.text || "");
  else { textEl.textContent = entry.text || ""; }
  scrollFeed();
}

function appendFeedEntry(entry, animate) {
  if (!entry || (entry.fid && entry.fid <= feedSeen)) return;
  if (entry.fid) feedSeen = entry.fid;
  if (entry.kind === "explanation") addExplanation(entry, animate);
  else addQA(entry);
}

function syncFeed(state) {
  // append only entries we haven't shown yet (fid guard) — no flicker on the
  // periodic snapshot; live "feed" events keep it current between snapshots.
  (state.feed || []).forEach(e => appendFeedEntry(e, false));
}
function resetFeed() {
  [...feedEl().querySelectorAll(".qa, .expl")].forEach(n => n.remove());
  feedSeen = 0; feedCount = 0; $("feed-count").textContent = 0;
  $("analysis-idle").classList.remove("hidden");
}

function typeText(el, text) {
  const my = ++typeToken;
  el.textContent = "";
  const cur = document.createElement("span"); cur.className = "cursor"; el.appendChild(cur);
  let i = 0;
  const step = () => {
    if (my !== typeToken) return;
    if (i <= text.length) {
      cur.remove();
      el.textContent = text.slice(0, i);
      el.appendChild(cur);
      scrollFeed();
      i += Math.max(1, Math.round(text.length / 240));
      setTimeout(step, 12);
    } else { cur.remove(); }
  };
  step();
}

// ── navigation (JARVIS directs the user) ────────────────────────────────────
function navigate(ev) {
  const map = { about:"panel-about", memories:"panel-memories", conversations:"panel-conversations", analysis:"panel-analysis" };
  const el = $(map[ev.section]);
  if (!el) return;
  el.scrollIntoView({ behavior:"smooth", block:"nearest" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  if (ev.section === "conversations" && ev.item != null) {
    const li = document.querySelector(`.conv-list li[data-id="${ev.item}"]`);
    if (li) { li.classList.add("open","highlight"); li.scrollIntoView({ behavior:"smooth", block:"center" });
      setTimeout(() => li.classList.remove("highlight"), 2500); }
  }
}

// ── event application ────────────────────────────────────────────────────────
const pendingFeed = [];
function applyEvent(ev) {
  switch (ev.type) {
    case "reveal": document.body.classList.remove("dimmed"); revealPending = true; maybeReveal(lastState); break;
    case "hide": break; // the parent window is closed by voice; page stays loaded
    case "feed":
      if (booted) appendFeedEntry(ev.entry, true);
      else { revealPending = true; pendingFeed.push(ev.entry); }
      break;
    case "identity":
      if (lastState) { lastState.user = ev.user; lastState.needs_name = false; }
      if (booted) renderProfile(lastState);
      else { hideNamePrompt(); revealPending = true; maybeReveal(lastState); }
      break;
    case "navigate": if (booted) navigate(ev); break;
    case "alarm": revealPending = true; maybeReveal(lastState); showAlarm(ev.alarm); break;
    case "alarm_cleared": hideAlarm(); break;
    case "focus":
      if (ev.focus && ev.focus.active) { revealPending = true; maybeReveal(lastState); }
      if (booted) renderFocus(ev.focus);
      break;
  }
}

let lastState = { user: "" };  // placeholder until the server sends the real name
async function maybeReveal(state) {
  if (booted) return;
  // First launch with no saved name: ask for it BEFORE the welcome animation, then
  // persist it (server writes ~/.jarvis/user.json) for every session after.
  if (state && state.needs_name) { showNamePrompt(state); return; }
  if (!(state && (state.revealed || revealPending))) return;
  await reveal(state.user);
  if (state) { syncFeed(state); renderFocus(state.focus); }
  while (pendingFeed.length) appendFeedEntry(pendingFeed.shift(), false);
}

// ── first-launch name prompt ─────────────────────────────────────────────────
let namePromptShown = false;
function showNamePrompt(state) {
  const box = $("name-prompt");
  if (!box || namePromptShown) return;
  namePromptShown = true;
  const input = $("name-input");
  if (input && state && state.name_suggestion && !input.value) input.value = state.name_suggestion;
  const w = $("welcome"); if (w) w.classList.remove("show");   // hide the greeting while asking
  box.classList.add("show");
  setTimeout(() => { if (input) { input.focus(); input.select(); } }, 60);
}
function hideNamePrompt() { const b = $("name-prompt"); if (b) b.classList.remove("show"); }
async function submitName() {
  const input = $("name-input");
  const name = ((input && input.value) || "").trim();
  if (!name) { if (input) input.focus(); return; }
  let finalName = name;
  try {
    const r = await fetch("/api/user/name", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await r.json();
    if (data && data.user) finalName = data.user;
  } catch (_) { /* offline: still greet locally with what they typed */ }
  if (lastState) { lastState.user = finalName; lastState.needs_name = false; }
  else lastState = { user: finalName, memories: [], conversations: [], feed: [] };
  hideNamePrompt();
  renderProfile(lastState);
  revealPending = true;
  await maybeReveal(lastState);   // now run the welcome animation with the entered name
}

function applySnapshot(state) {
  lastState = state;
  renderProfile(state);
  renderConversations(state);
  renderTodos(state);
  renderAgenda(state);
  if (booted) syncFeed(state);
  cursor = state.cursor || cursor;
  // a late-joining client should still see a currently ringing alarm
  if (booted && (state.alarms || []).length && !currentAlarm) showAlarm(state.alarms[0]);
  if (booted) renderFocus(state.focus);   // and an in-progress focus session
  maybeReveal(state);
}

// ── WebSocket transport (instant push, with reconnect) ───────────────────────
let ws = null, backoff = 400;
function wsSend(obj) { try { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); } catch (_) {} }
function connect() {
  try {
    ws = new WebSocket(`ws://${location.host}/ws`);
  } catch (_) { setTimeout(connect, backoff); return; }
  ws.onopen = () => { backoff = 400; if (booted) resetFeed(); };  // rebuild cleanly from the fresh snapshot
  ws.onmessage = (m) => {
    let msg; try { msg = JSON.parse(m.data); } catch { return; }
    if (msg.type === "snapshot") applySnapshot(msg.state);
    else if (msg.type === "event") { const ev = msg.event; cursor = Math.max(cursor, ev.id || 0); applyEvent(ev); }
  };
  ws.onclose = () => { ws = null; setTimeout(connect, backoff); backoff = Math.min(backoff * 1.6, 4000); };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };
}

// initial render via a one-shot fetch (so the page isn't blank before the first
// snapshot arrives), then live over the socket.
fetch("/api/state?since=0", { cache: "no-store" })
  .then((r) => r.json()).then(applySnapshot).catch(() => {});
connect();

// first-launch name form
const _nameForm = $("name-prompt");
if (_nameForm) _nameForm.addEventListener("submit", (e) => { e.preventDefault(); submitName(); });

// add-a-memory form
const _memForm = $("mem-add-form");
if (_memForm) _memForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("mem-add-input");
  const text = input.value;
  input.value = "";
  addMemory(text);
});
