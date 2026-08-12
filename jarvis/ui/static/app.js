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
    li.textContent = m;
    li.style.animationDelay = `${idx * 0.05}s`;
    list.appendChild(li);
  });
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

// ── explanation (the key feature) ───────────────────────────────────────────
let typeToken = 0;
function showExplanation(e) {
  $("analysis-idle").classList.add("hidden");
  $("analysis-title").textContent = (e.title || "ANALYSIS").toUpperCase();
  const img = $("analysis-image");
  if (e.image) { img.src = `data:image/png;base64,${e.image}`; img.classList.remove("hidden"); }
  else { img.classList.add("hidden"); img.removeAttribute("src"); }
  $("analysis-src").textContent = e.source ? `— ${e.source}` : "";
  $("panel-analysis").classList.remove("flash"); void $("panel-analysis").offsetWidth;
  $("panel-analysis").classList.add("flash");
  typeText($("analysis-text"), e.body || "");
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
      el.parentElement.scrollTop = el.parentElement.scrollHeight;
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
let pendingExpl = null;
function applyEvent(ev) {
  switch (ev.type) {
    case "reveal": revealPending = true; maybeReveal(lastState); break;
    case "hide": break; // the parent window is closed by voice; page stays loaded
    case "explanation": if (booted) showExplanation(ev); else { revealPending = true; pendingExpl = ev; } break;
    case "navigate": if (booted) navigate(ev); break;
    case "alarm": revealPending = true; maybeReveal(lastState); showAlarm(ev.alarm); break;
    case "alarm_cleared": hideAlarm(); break;
  }
}

let lastState = { user: "Akshath" };
async function maybeReveal(state) {
  if (booted) return;
  if (!(state && (state.revealed || revealPending))) return;
  await reveal(state.user);
  if (pendingExpl) { showExplanation(pendingExpl); pendingExpl = null; }
  else if (state && state.last_explanation) showExplanation(state.last_explanation);
}

function applySnapshot(state) {
  lastState = state;
  renderProfile(state);
  renderConversations(state);
  renderTodos(state);
  renderAgenda(state);
  cursor = state.cursor || cursor;
  // a late-joining client should still see a currently ringing alarm
  if (booted && (state.alarms || []).length && !currentAlarm) showAlarm(state.alarms[0]);
  maybeReveal(state);
}

// ── WebSocket transport (instant push, with reconnect) ───────────────────────
let ws = null, backoff = 400;
function wsSend(obj) { try { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); } catch (_) {} }
function connect() {
  try {
    ws = new WebSocket(`ws://${location.host}/ws`);
  } catch (_) { setTimeout(connect, backoff); return; }
  ws.onopen = () => { backoff = 400; };
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
