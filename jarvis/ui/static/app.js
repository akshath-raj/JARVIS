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

// ── event application + polling ─────────────────────────────────────────────
function applyEvent(ev) {
  switch (ev.type) {
    case "reveal": revealPending = true; break;
    case "hide": /* handled by parent close; keep page */ break;
    case "explanation": if (booted) showExplanation(ev); else { revealPending = true; pendingExpl = ev; } break;
    case "navigate": if (booted) navigate(ev); break;
  }
}
let pendingExpl = null;

async function poll() {
  try {
    const res = await fetch(`/api/state?since=${cursor}`, { cache: "no-store" });
    const state = await res.json();
    renderProfile(state);
    renderConversations(state);
    (state.events || []).forEach(applyEvent);
    cursor = state.cursor || cursor;

    if ((state.revealed || revealPending) && !booted) {
      await reveal(state.user);
      if (pendingExpl) { showExplanation(pendingExpl); pendingExpl = null; }
      else if (state.last_explanation) showExplanation(state.last_explanation);
    }
  } catch (_) { /* server not up yet */ }
}
setInterval(poll, 350);
poll();
