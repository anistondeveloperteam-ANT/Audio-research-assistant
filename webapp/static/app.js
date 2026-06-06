/* Audio Research Assistant — web UI logic (vanilla JS, no build step). */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = {
    config: () => fetch("/api/config").then((r) => r.json()),
    sessions: () => fetch("/api/sessions").then((r) => r.json()),
    createSession: () => fetch("/api/sessions", { method: "POST" }).then((r) => r.json()),
    renameSession: (id, title) =>
      fetch(`/api/sessions/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) }),
    deleteSession: (id) => fetch(`/api/sessions/${id}`, { method: "DELETE" }),
    turns: (id) => fetch(`/api/sessions/${id}/turns`).then((r) => r.json()),
    deleteTurn: (id, idx) => fetch(`/api/sessions/${id}/turns/${idx}`, { method: "DELETE" }).then((r) => r.json()),
    truncateTurns: (id, idx) => fetch(`/api/sessions/${id}/turns/${idx}/truncate`, { method: "POST" }).then((r) => r.json()),
    library: () => fetch("/api/library").then((r) => r.json()),
    papers: () => fetch("/api/papers").then((r) => r.json()),
    deletePaper: (id) => fetch(`/api/papers/${id}`, { method: "DELETE" }).then((r) => r.json()),
    models: () => fetch("/api/models").then((r) => r.json()),
    setModel: (provider, model) =>
      fetch("/api/model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider, model }) }).then((r) => r.json()),
    upload: (file) => {
      const fd = new FormData(); fd.append("file", file);
      return fetch("/api/upload", { method: "POST", body: fd }).then((r) => r.json());
    },
  };

  const state = {
    cfg: { modes: ["Fast", "Balanced", "Deep"], default_mode: "Balanced", default_top_k: 8, provider: "" },
    sessions: [],
    currentId: null,
    streaming: false,
    ingesting: false,
    currentSources: [],
    abort: null,
    autoStick: true,
    nextTurnIndex: 0,
    mode: "Default",    // single optimized retrieval mode (no Fast/Balanced/Deep)
    topk: 8,            // hint only; the server selects sources adaptively
  };

  // Icons for the per-question action buttons (copy / edit / delete).
  const ICON_COPY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
  const ICON_EDIT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
  const ICON_TRASH = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>';

  const SEND_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 11l5-5 5 5M12 6v13"/></svg>';
  const STOP_ICON = '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2.5" fill="currentColor"/></svg>';

  const EXAMPLES = [
    ["How does MVDR beamforming work?", "…and how does it compare to delay-and-sum?"],
    ["Explain acoustic echo cancellation", "with the key signal-processing steps."],
    ["Which metrics evaluate speech enhancement?", "PESQ, STOI, SDR — what do they mean?"],
    ["Summarize deep-learning denoising", "approaches across my papers."],
  ];

  const esc = (s) => (s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Tidy a paper / file name for display: drop ".pdf", underscores -> spaces,
  // and Title-Case anything that's ALL CAPS (e.g. "REVEREBERATION" -> "Reverberation").
  const prettyName = (s) => {
    let t = (s || "").replace(/\.pdf$/i, "").replace(/_+/g, " ").replace(/\s+/g, " ").trim();
    if (t && !/[a-z]/.test(t)) t = t.toLowerCase().replace(/\b[a-z]/g, (c) => c.toUpperCase());
    return t;
  };

  // ---------- Toasts ----------
  function toast(msg, kind) {
    const t = document.createElement("div");
    t.className = "toast" + (kind ? " " + kind : "");
    t.textContent = msg;
    $("toasts").appendChild(t);
    setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .3s"; setTimeout(() => t.remove(), 320); }, 3400);
  }

  // ---------- Markdown + citations ----------
  function renderMarkdown(el, text) {
    el.innerHTML = marked.parse(text || "", { breaks: true, gfm: true });
    linkifyCitations(el);
    enhanceCodeBlocks(el);
  }

  function enhanceCodeBlocks(root) {
    root.querySelectorAll("pre").forEach((pre) => {
      if (pre.querySelector(".code-copy")) return;
      const btn = document.createElement("button");
      btn.className = "code-copy";
      btn.textContent = "Copy";
      btn.addEventListener("click", () => {
        const code = pre.querySelector("code");
        navigator.clipboard.writeText((code || pre).innerText).then(() => {
          btn.textContent = "Copied"; setTimeout(() => (btn.textContent = "Copy"), 1200);
        });
      });
      pre.appendChild(btn);
    });
  }

  // Citation hover preview
  function showCitePop(chip, n) {
    const s = (state.currentSources || []).find((x) => String(x.n) === String(n));
    if (!s) return;
    const pop = $("citePop");
    const pages = s.page_start ? ` · pp. ${s.page_start}${s.page_end && s.page_end !== s.page_start ? "–" + s.page_end : ""}` : "";
    pop.innerHTML = `<div class="cp-title">[${s.n}] ${esc(prettyName(s.title))}</div>` +
      `<div class="cp-meta">${esc(s.section || "")}${pages}</div>` +
      `<div>${esc((s.text || "").slice(0, 170))}…</div>`;
    const r = chip.getBoundingClientRect();
    pop.style.left = Math.min(r.left, window.innerWidth - 350) + "px";
    pop.style.top = (r.bottom + 8) + "px";
    pop.classList.add("show");
  }
  function hideCitePop() { $("citePop").classList.remove("show"); }
  function linkifyCitations(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => {
        const p = n.parentElement;
        if (!p) return NodeFilter.FILTER_REJECT;
        if (p.closest("pre, code, a, .cite")) return NodeFilter.FILTER_REJECT;
        return /\[\d+\]/.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    const targets = [];
    while (walker.nextNode()) targets.push(walker.currentNode);
    for (const node of targets) {
      const frag = document.createDocumentFragment();
      let last = 0;
      const s = node.nodeValue;
      s.replace(/\[(\d+)\]/g, (m, n, idx) => {
        if (idx > last) frag.appendChild(document.createTextNode(s.slice(last, idx)));
        const b = document.createElement("button");
        b.className = "cite"; b.textContent = n; b.dataset.n = n;
        b.addEventListener("click", () => focusSource(parseInt(n, 10)));
        b.addEventListener("mouseenter", () => showCitePop(b, n));
        b.addEventListener("mouseleave", hideCitePop);
        frag.appendChild(b);
        last = idx + m.length;
        return m;
      });
      if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  // ---------- Transcript rendering ----------
  const inner = () => $("transcriptInner");

  function showWelcome() {
    inner().innerHTML = `
      <div class="welcome">
        <div class="hero-mark"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zm7 9a7 7 0 0 1-14 0H3a9 9 0 0 0 8 8.94V23h2v-2.06A9 9 0 0 0 21 12h-2z"/></svg></div>
        <h1>What do your papers say?</h1>
        <p>Ask anything about your audio &amp; speech-enhancement library. Every answer is grounded only in your papers — with each claim cited to its source, section, and page.</p>
        <div class="examples" id="examples"></div>
      </div>`;
    const box = $("examples");
    EXAMPLES.forEach(([k, sub]) => {
      const b = document.createElement("button");
      b.className = "example";
      b.innerHTML = `<span class="ex-k">${esc(k)}</span><span>${esc(sub)}</span>`;
      b.addEventListener("click", () => { $("input").value = k + " " + sub; autosize(); send(); });
      box.appendChild(b);
    });
  }

  function addUserMessage(text, turnIndex) {
    const m = document.createElement("div");
    m.className = "msg user";
    if (turnIndex != null) m.dataset.turnIndex = String(turnIndex);
    m.innerHTML = `<div class="u-wrap"></div>`;
    fillUserWrap(m, text);
    // Delegated so the handler survives the wrap being re-rendered (edit mode).
    m.addEventListener("click", (e) => {
      const b = e.target.closest(".ua-btn");
      if (!b || !m.contains(b)) return;
      if (b.dataset.act === "copy") copyUserMessage(m);
      else if (b.dataset.act === "edit") startEditUserMessage(m);
      else if (b.dataset.act === "delete") deleteUserMessage(m);
    });
    inner().appendChild(m);
    scrollToBottom(true);
    return m;
  }

  // (Re)build the normal bubble + hover actions for a user message.
  function fillUserWrap(m, text) {
    const wrap = m.querySelector(".u-wrap");
    wrap.innerHTML = `
      <div class="bubble"></div>
      <div class="msg-actions">
        <button class="ua-btn" data-act="copy" title="Copy question" aria-label="Copy question">${ICON_COPY}</button>
        <button class="ua-btn" data-act="edit" title="Edit & resend" aria-label="Edit question">${ICON_EDIT}</button>
        <button class="ua-btn danger" data-act="delete" title="Delete question" aria-label="Delete question">${ICON_TRASH}</button>
      </div>`;
    wrap.querySelector(".bubble").textContent = text;
  }

  function copyUserMessage(m) {
    const text = (m.querySelector(".bubble") || {}).textContent || "";
    navigator.clipboard.writeText(text).then(() => toast("Question copied"));
  }

  async function deleteUserMessage(m) {
    if (state.streaming) { toast("Please wait for the answer to finish."); return; }
    const idx = m.dataset.turnIndex;
    try {
      if (idx != null) await api.deleteTurn(state.currentId, idx);
    } catch { toast("Couldn't delete the question.", "error"); return; }
    await reloadTurns();   // re-render from the DB so turn indices + sources stay correct
    toast("Question deleted");
  }

  function startEditUserMessage(m) {
    if (state.streaming) { toast("Please wait for the answer to finish."); return; }
    const wrap = m.querySelector(".u-wrap");
    if (wrap.querySelector(".u-edit")) return;   // already editing
    const text = (m.querySelector(".bubble") || {}).textContent || "";
    wrap.innerHTML = `
      <div class="u-edit">
        <textarea class="u-edit-area" rows="1"></textarea>
        <div class="u-edit-actions">
          <button class="ue-btn ue-cancel">Cancel</button>
          <button class="ue-btn ue-save">Save &amp; resend</button>
        </div>
      </div>`;
    const ta = wrap.querySelector(".u-edit-area");
    ta.value = text;
    const fit = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 200) + "px"; };
    ta.addEventListener("input", fit);
    fit();
    ta.focus();
    ta.setSelectionRange(text.length, text.length);
    wrap.querySelector(".ue-cancel").addEventListener("click", () => fillUserWrap(m, text));
    wrap.querySelector(".ue-save").addEventListener("click", () => saveEditUserMessage(m, text));
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); saveEditUserMessage(m, text); }
      else if (e.key === "Escape") { e.preventDefault(); fillUserWrap(m, text); }
    });
  }

  async function saveEditUserMessage(m, originalText) {
    if (state.streaming) return;
    const ta = m.querySelector(".u-edit-area");
    const edited = (ta ? ta.value : "").trim();
    if (!edited) { toast("The question can't be empty."); return; }
    if (edited === originalText) { fillUserWrap(m, originalText); return; }
    const idx = m.dataset.turnIndex;
    try {
      if (idx != null) await api.truncateTurns(state.currentId, idx);   // drop this Q + everything after
    } catch { toast("Couldn't edit the question.", "error"); fillUserWrap(m, originalText); return; }
    // Remove this message and all later ones from the view; send() re-adds the edited one.
    let n = m.nextElementSibling;
    while (n) { const after = n.nextElementSibling; n.remove(); n = after; }
    m.remove();
    if (idx != null) state.nextTurnIndex = parseInt(idx, 10);
    $("input").value = edited;
    autosize();
    send();
  }

  // Returns handles to drive a streaming assistant message.
  function addAssistantMessage() {
    const m = document.createElement("div");
    m.className = "msg assistant";
    m.innerHTML = `
      <div class="body">
        <div class="bubble assistant">
          <div class="statusline"><span class="typing"><span></span><span></span><span></span></span><span class="status-text">Thinking…</span><span class="elapsed"></span></div>
          <div class="md" style="display:none"></div>
        </div>
        <div class="msg-tools" style="display:none"></div>
      </div>`;
    inner().appendChild(m);
    scrollToBottom(true);
    return {
      el: m,
      statusEl: m.querySelector(".statusline"),
      statusText: m.querySelector(".status-text"),
      elapsed: m.querySelector(".elapsed"),
      md: m.querySelector(".md"),
      tools: m.querySelector(".msg-tools"),
    };
  }

  function renderHistoryMessage(turn) {
    if (turn.role === "user") { addUserMessage(turn.content, turn.turn_index); return; }
    const h = addAssistantMessage();
    h.statusEl.style.display = "none";
    h.md.style.display = "";
    renderMarkdown(h.md, turn.content);
    finalizeTools(h, turn.sources || []);
  }

  function finalizeTools(h, sources, meta) {
    h.tools.style.display = "flex";
    h.tools.innerHTML = "";
    // Speed + model badge (live answers only) — makes the response feel measured, not dull.
    if (meta && meta.seconds != null) {
      const badge = document.createElement("span");
      badge.className = "speed-badge";
      const model = (meta.model || "").split("/").pop();
      badge.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M13 2 4.5 13H11l-1 9 8.5-11H12z"/></svg> ${meta.seconds.toFixed(1)}s${model ? " · " + esc(model) : ""}`;
      badge.title = "Answer time" + (meta.model ? " · " + esc(meta.model) : "");
      h.tools.appendChild(badge);
    }
    const copy = document.createElement("button");
    copy.className = "tool-btn";
    copy.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg> Copy`;
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(h.md.innerText).then(() => toast("Answer copied"));
    });
    h.tools.appendChild(copy);
    if (sources && sources.length) {
      const sc = document.createElement("button");
      sc.className = "src-count";
      sc.innerHTML = `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg> Sources`;
      sc.addEventListener("click", () => { state.currentSources = sources; renderSources(sources); openDrawer(); });
      h.tools.appendChild(sc);
    }
  }

  // ---------- Sources drawer ----------
  function renderSources(sources) {
    state.currentSources = sources || [];
    const body = $("drawerBody");
    if (!state.currentSources.length) {
      body.innerHTML = `<div class="drawer-empty">No sources for this answer.</div>`;
      return;
    }
    body.innerHTML = "";
    state.currentSources.forEach((s) => {
      const pages = s.page_start ? `pp. ${s.page_start}${s.page_end && s.page_end !== s.page_start ? "–" + s.page_end : ""}` : "";
      const card = document.createElement("div");
      card.className = "source-card";
      card.id = "src-card-" + s.n;
      card.innerHTML = `
        <div class="sc-head">
          <span class="sc-n">${s.n}</span>
          <span class="sc-title">${esc(prettyName(s.title))}</span>
        </div>
        <div class="sc-meta">
          ${s.score ? `<span class="chip relevance">${Math.min(100, Math.round(s.score * 100))}% match</span>` : ""}
          ${s.section ? `<span class="chip">${esc(s.section)}</span>` : ""}
          ${pages ? `<span class="chip">${pages}</span>` : ""}
        </div>
        <div class="sc-text">${esc(s.text)}</div>
        ${s.text && s.text.length > 240 ? `<span class="sc-more">Show more</span>` : ""}`;
      const more = card.querySelector(".sc-more");
      if (more) more.addEventListener("click", () => {
        const t = card.querySelector(".sc-text");
        const ex = t.classList.toggle("expanded");
        more.textContent = ex ? "Show less" : "Show more";
      });
      body.appendChild(card);
    });
  }
  function openDrawer() { $("drawer").classList.add("open"); $("scrim").classList.add("show"); }
  function closeDrawer() { $("drawer").classList.remove("open"); $("scrim").classList.remove("show"); }
  function focusSource(n) {
    renderSources(state.currentSources);
    openDrawer();
    const card = $("src-card-" + n);
    if (card) {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1400);
    }
  }

  // ---------- Sessions ----------
  function renderSessions() {
    const box = $("sessions");
    box.innerHTML = "";
    state.sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = "session" + (s.id === state.currentId ? " active" : "");
      item.innerHTML = `
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" style="flex:none;opacity:.6"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="s-title">${esc(s.title || "Untitled")}</span>
        <span class="s-actions">
          <button class="icon-btn" data-act="rename" title="Rename"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg></button>
          <button class="icon-btn danger" data-act="delete" title="Delete"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg></button>
        </span>`;
      item.addEventListener("click", (e) => {
        const act = e.target.closest("[data-act]");
        if (act) { e.stopPropagation(); act.dataset.act === "rename" ? renameSession(s) : deleteSession(s); return; }
        selectSession(s.id);
      });
      box.appendChild(item);
    });
  }

  async function loadSessions(selectId) {
    state.sessions = await api.sessions();
    if (!state.sessions.length) { await newChat(); return; }
    renderSessions();
    await selectSession(selectId || state.sessions[0].id);
  }

  async function selectSession(id) {
    if (state.streaming) return;
    state.currentId = id;
    renderSessions();
    renderTurns(await api.turns(id));
    if (window.innerWidth <= 880) $("sidebar").classList.remove("open");
  }

  // Render a full turn list into the transcript and track the next turn index
  // (so freshly-sent questions get the same index the server will assign them).
  function renderTurns(turns) {
    inner().innerHTML = "";
    if (!turns.length) {
      showWelcome();
      state.currentSources = []; renderSources([]);
      state.nextTurnIndex = 0;
      return;
    }
    turns.forEach(renderHistoryMessage);
    state.nextTurnIndex = turns.reduce((mx, t) => Math.max(mx, t.turn_index), -1) + 1;
    const lastAssist = [...turns].reverse().find((t) => t.role === "assistant" && t.sources);
    renderSources(lastAssist ? lastAssist.sources : []);
    scrollToBottom(true);
  }

  async function reloadTurns() {
    if (!state.currentId) return;
    renderTurns(await api.turns(state.currentId));
  }

  async function newChat() {
    if (state.streaming) return;
    const s = await api.createSession();
    state.sessions.unshift(s);
    renderSessions();
    await selectSession(s.id);
    $("input").focus();
  }

  async function renameSession(s) {
    const title = prompt("Rename conversation:", s.title || "");
    if (title == null) return;
    await api.renameSession(s.id, title.trim() || "Untitled");
    s.title = title.trim() || "Untitled";
    renderSessions();
  }
  async function deleteSession(s) {
    if (!confirm(`Delete "${s.title || "this conversation"}"? This cannot be undone.`)) return;
    await api.deleteSession(s.id);
    state.sessions = state.sessions.filter((x) => x.id !== s.id);
    if (s.id === state.currentId) {
      state.currentId = null;
      if (state.sessions.length) await selectSession(state.sessions[0].id);
      else await newChat();
    } else renderSessions();
  }

  // ---------- Sending + streaming ----------
  function setStreaming(on) {
    state.streaming = on;
    const btn = $("sendBtn");
    if (on) {
      btn.disabled = false;
      btn.classList.add("stop");
      btn.innerHTML = STOP_ICON;
      btn.setAttribute("aria-label", "Stop generating");
    } else {
      btn.classList.remove("stop");
      btn.innerHTML = SEND_ICON;
      btn.setAttribute("aria-label", "Send");
      btn.disabled = !$("input").value.trim();
    }
    $("input").disabled = on;
  }

  function currentModelName() {
    try { return JSON.parse($("modelSel").value).model || ""; } catch { return ""; }
  }

  async function send() {
    const text = $("input").value.trim();
    if (!text || state.streaming || !state.currentId) return;

    // First message in a fresh session -> title it from the question.
    const sess = state.sessions.find((s) => s.id === state.currentId);
    const wasEmpty = sess && (sess.title === "New conversation" || !sess.title);

    if (inner().querySelector(".welcome")) inner().innerHTML = "";
    $("input").value = ""; autosize();
    const userIndex = state.nextTurnIndex;
    addUserMessage(text, userIndex);
    state.nextTurnIndex = userIndex + 2;   // server appends user(+0) then assistant(+1)
    setStreaming(true);
    const h = addAssistantMessage();

    // Live elapsed timer so the user always sees it's working (not frozen).
    const genStart = performance.now();
    const timer = setInterval(() => {
      if (h.elapsed) h.elapsed.textContent = ((performance.now() - genStart) / 1000).toFixed(1) + "s";
    }, 100);

    let answer = "";
    let renderScheduled = false;
    const scheduleRender = () => {
      if (renderScheduled) return;
      renderScheduled = true;
      requestAnimationFrame(() => {
        renderScheduled = false;
        if (h.md.style.display === "none") { h.md.style.display = ""; h.statusEl.style.display = "none"; }
        renderMarkdown(h.md, answer + " ▍");
        scrollToBottom();
      });
    };

    const controller = new AbortController();
    state.abort = controller;
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.currentId, question: text, mode: state.mode, top_k: state.topk }),
        signal: controller.signal,
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let ev; try { ev = JSON.parse(line); } catch { continue; }
          handleEvent(ev, h, () => answer, (v) => { answer = v; }, scheduleRender);
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        answer = (answer || "").trim() + "\n\n_⏹ Stopped._";
      } else {
        toast("Connection error: " + err.message, "error");
        if (!answer) answer = "_Something went wrong. Please try again._";
      }
      h.statusEl.style.display = "none";
      h.md.style.display = "";
      renderMarkdown(h.md, answer);
    } finally {
      state.abort = null;
      clearInterval(timer);
      const secs = (performance.now() - genStart) / 1000;
      // Final clean render (drop the streaming caret).
      h.md.style.display = ""; h.statusEl.style.display = "none";
      renderMarkdown(h.md, answer || "_(no answer)_");
      finalizeTools(h, state.currentSources, { seconds: secs, model: currentModelName() });
      setStreaming(false);
      scrollToBottom();
      $("input").focus();
      if (wasEmpty) {
        const title = text.length > 48 ? text.slice(0, 48) + "…" : text;
        await api.renameSession(state.currentId, title);
        if (sess) sess.title = title;
        renderSessions();
      }
    }
  }

  function handleEvent(ev, h, getAns, setAns, scheduleRender) {
    switch (ev.type) {
      case "status":
        h.statusText.textContent = ev.message || "Working…";
        break;
      case "sanity":
        h.statusEl.style.display = "none"; h.md.style.display = "";
        renderMarkdown(h.md, "⚠️ " + (ev.message || "Please rephrase your question."));
        break;
      case "sources": {
        const n = (ev.sources || []).length;
        renderSources(ev.sources || []);
        if (n) h.statusText.textContent = `Found ${n} relevant passage${n > 1 ? "s" : ""} — writing the answer…`;
        h.el.classList.add("has-sources");
        break;
      }
      case "token":
        setAns(getAns() + (ev.text || ""));
        scheduleRender();
        break;
      case "error":
        toast(ev.message || "Error", "error");
        setAns(getAns() + "\n\n_" + (ev.message || "error") + "_");
        scheduleRender();
        break;
      case "done":
        break;
    }
  }

  // ---------- Composer behaviour ----------
  function autosize() {
    const t = $("input");
    t.style.height = "auto";
    t.style.height = Math.min(t.scrollHeight, 200) + "px";
  }
  function nearBottom() {
    const tr = $("transcript");
    return tr.scrollHeight - tr.scrollTop - tr.clientHeight < 120;
  }
  function updateToBottomBtn() {
    const tr = $("transcript");
    const show = tr.scrollHeight - tr.scrollTop - tr.clientHeight > 220 && !!inner().querySelector(".msg");
    $("toBottom").classList.toggle("show", show);
  }
  function scrollToBottom(force) {
    const tr = $("transcript");
    if (force) state.autoStick = true;
    if (force || state.autoStick) tr.scrollTop = tr.scrollHeight;
    updateToBottomBtn();
  }

  // ---------- Library + upload + ingest ----------
  async function loadLibrary() {
    try {
      const lib = await api.library();
      const p = lib.papers != null ? lib.papers : lib.pdfs;
      $("libLabel").textContent = `${p} paper${p === 1 ? "" : "s"} indexed`;
    } catch { $("libLabel").textContent = "library unavailable"; }
  }

  // ---------- Your papers (manage / delete) ----------
  async function openPapers() {
    $("papersModal").classList.add("show");
    $("papersScrim").classList.add("show");
    const body = $("pmBody");
    body.innerHTML = `<div class="pm-empty">Loading…</div>`;
    let list = [];
    try { list = await api.papers(); } catch {}
    renderPapers(list);
  }
  function closePapers() { $("papersModal").classList.remove("show"); $("papersScrim").classList.remove("show"); }

  function renderPapers(list) {
    $("pmCount").textContent = String(list.length);
    const body = $("pmBody");
    if (!list.length) {
      body.innerHTML = `<div class="pm-empty">No papers yet. Click <b>Add papers</b> in the sidebar to upload one or more PDFs.</div>`;
      return;
    }
    body.innerHTML = "";
    list.forEach((p) => {
      const row = document.createElement("div");
      row.className = "paper-row";
      row.innerHTML = `
        <div class="pr-main">
          <div class="pr-title">${esc(prettyName(p.title))}</div>
          <div class="pr-meta">${p.chunks} chunk${p.chunks === 1 ? "" : "s"}</div>
        </div>
        <button class="pr-del">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>
          Delete
        </button>`;
      const btn = row.querySelector(".pr-del");
      btn.addEventListener("click", async () => {
        if (!confirm(`Delete "${p.title}"? This removes the PDF, its chunks, and all embeddings — permanently.`)) return;
        btn.disabled = true; btn.textContent = "Deleting…";
        try {
          const res = await api.deletePaper(p.id);
          if (res.error) { toast(res.error, "error"); btn.disabled = false; btn.textContent = "Delete"; return; }
          row.remove();
          const remaining = $("pmBody").querySelectorAll(".paper-row").length;
          $("pmCount").textContent = String(remaining);
          if (!remaining) renderPapers([]);
          if (res.library) { const n = res.library.papers != null ? res.library.papers : res.library.pdfs; $("libLabel").textContent = `${n} paper${n === 1 ? "" : "s"} indexed`; }
          toast("Paper deleted.");
        } catch (e) { toast("Delete failed.", "error"); btn.disabled = false; btn.textContent = "Delete"; }
      });
      body.appendChild(row);
    });
  }

  function pickPdf() {
    if (state.ingesting) return;
    $("pdfInput").value = "";
    $("pdfInput").click();
  }

  function isPdf(file) {
    return (file.type === "application/pdf") || file.name.toLowerCase().endsWith(".pdf");
  }

  async function onPdfChosen(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    $("addPaperBtn").classList.add("busy");
    const saved = [], dups = [], errs = [];
    for (const file of files) {
      if (!isPdf(file)) { errs.push(file.name); continue; }
      try {
        const res = await api.upload(file);
        if (res.status === "saved") saved.push(res.filename);
        else if (res.status === "duplicate") dups.push(res.filename);
        else errs.push(file.name);
      } catch { errs.push(file.name); }
    }
    $("addPaperBtn").classList.remove("busy");

    if (dups.length) toast(`${dups.length} already indexed — skipped.`);
    if (errs.length) toast(`${errs.length} file${errs.length > 1 ? "s" : ""} couldn't be added.`, "error");
    if (!saved.length) return;  // nothing new to index

    const label = saved.length === 1 ? saved[0] : `${saved.length} papers`;
    startIngest(label, saved);
  }

  function openIngestModal(label) {
    $("imTitle").textContent = "Adding " + (label || "your papers") + "…";
    $("imStage").textContent = "Starting…";
    $("imLog").textContent = "";
    $("imSpinner").hidden = false;
    $("imCheck").hidden = true;
    $("imFoot").hidden = true;
    $("ingestModal").classList.add("show");
    $("ingestScrim").classList.add("show");
  }
  function closeIngestModal() { $("ingestModal").classList.remove("show"); $("ingestScrim").classList.remove("show"); }

  function logLine(text, cls) {
    const log = $("imLog");
    const span = document.createElement("span");
    if (cls) span.className = cls;
    span.textContent = text + "\n";
    log.appendChild(span);
    log.scrollTop = log.scrollHeight;
  }

  async function startIngest(label, saved) {
    state.ingesting = true;
    $("addPaperBtn").classList.add("busy");
    openIngestModal(label);
    const n = (saved && saved.length) || 1;
    logLine(`→ Saved ${n} file${n > 1 ? "s" : ""}. Indexing now — the first paper also warms up the models.`, "stage");
    if (saved && saved.length) saved.forEach((f) => logLine("   • " + f));
    try {
      const resp = await fetch("/api/ingest", { method: "POST" });
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "", ok = true;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
          if (!line) continue;
          let ev; try { ev = JSON.parse(line); } catch { continue; }
          if (ev.type === "stage") { $("imStage").textContent = ev.label; logLine("◆ " + ev.label, "stage"); }
          else if (ev.type === "log") { logLine(ev.line, /skip/i.test(ev.line) ? "warn" : null); }
          else if (ev.type === "error") { ok = false; logLine("✗ " + ev.message, "warn"); }
          else if (ev.type === "done") {
            logLine("✓ " + ev.message, "ok");
            if (ev.library) { const p = ev.library.papers != null ? ev.library.papers : ev.library.pdfs; $("libLabel").textContent = `${p} papers indexed`; }
          }
        }
      }
      finishIngest(ok, label);
    } catch (err) {
      logLine("✗ " + err.message, "warn");
      finishIngest(false, label);
    }
  }

  function finishIngest(ok, label) {
    state.ingesting = false;
    $("addPaperBtn").classList.remove("busy");
    $("imSpinner").hidden = true;
    $("imCheck").hidden = !ok;
    $("imTitle").textContent = ok ? "Done — indexed" : "Indexing failed";
    $("imStage").textContent = ok ? "You can now ask questions about your papers." : "See the log above. The files were saved but not fully indexed.";
    $("imFoot").hidden = false;
    if (ok) toast(`${label} added to your library.`);
    loadLibrary();
  }

  // ---------- Theme ----------
  const ICON_SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const ICON_MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    $("themeBtn").innerHTML = t === "dark" ? ICON_SUN : ICON_MOON;
    $("themeBtn").title = t === "dark" ? "Switch to light theme" : "Switch to dark theme";
  }
  function toggleTheme() {
    const next = (document.documentElement.getAttribute("data-theme") === "dark") ? "light" : "dark";
    try { localStorage.setItem("ara-theme", next); } catch {}
    applyTheme(next);
  }

  // ---------- Model switcher ----------
  function setProviderLabel(label) {
    $("provLabel").textContent = label;
    $("provDot").style.background = "var(--ok)";
  }
  async function loadModels() {
    try {
      const data = await api.models();
      const sel = $("modelSel");
      sel.innerHTML = "";
      (data.options || []).forEach((o) => {
        const opt = document.createElement("option");
        opt.value = JSON.stringify({ provider: o.provider, model: o.model });
        opt.textContent = o.label;
        sel.appendChild(opt);
      });
      const cur = data.current || {};
      sel.value = JSON.stringify({ provider: cur.provider, model: cur.model });
      setProviderLabel(`${cur.provider} · ${cur.model}`);
    } catch { $("modelSel").innerHTML = '<option>unavailable</option>'; }
  }
  async function onModelChange() {
    let v; try { v = JSON.parse($("modelSel").value); } catch { return; }
    try {
      const res = await api.setModel(v.provider, v.model);
      if (res.error) { toast(res.error, "error"); return; }
      setProviderLabel(res.label);
      toast("Model switched to " + res.model);
    } catch { toast("Could not switch model.", "error"); }
  }

  // ---------- Init ----------
  async function init() {
    applyTheme(document.documentElement.getAttribute("data-theme") || "light");
    try { if (localStorage.getItem("ara-sidebar") === "collapsed" && window.innerWidth > 880) $("app").classList.add("collapsed"); } catch {}
    try { state.cfg = await api.config(); } catch {}
    // One optimized retrieval mode now; the server selects how many sources to
    // use adaptively, so there's nothing to configure here.
    $("provLabel").textContent = state.cfg.provider || "ready";
    if (!state.cfg.provider || state.cfg.provider === "unknown") $("provDot").style.background = "var(--amber)";

    loadLibrary();
    loadModels();
    await loadSessions();

    // Events
    $("newChatBtn").addEventListener("click", newChat);
    $("sendBtn").addEventListener("click", () => {
      if (state.streaming) { if (state.abort) state.abort.abort(); }
      else send();
    });
    $("transcript").addEventListener("scroll", () => { state.autoStick = nearBottom(); updateToBottomBtn(); });
    $("toBottom").addEventListener("click", () => scrollToBottom(true));
    $("sourcesBtn").addEventListener("click", () => { renderSources(state.currentSources); openDrawer(); });
    $("drawerClose").addEventListener("click", closeDrawer);
    $("scrim").addEventListener("click", closeDrawer);
    $("menuBtn").addEventListener("click", () => {
      if (window.innerWidth > 880) {
        const collapsed = $("app").classList.toggle("collapsed");
        try { localStorage.setItem("ara-sidebar", collapsed ? "collapsed" : "open"); } catch {}
      } else {
        $("sidebar").classList.toggle("open");
      }
    });
    $("addPaperBtn").addEventListener("click", pickPdf);
    $("pdfInput").addEventListener("change", onPdfChosen);
    $("imDone").addEventListener("click", closeIngestModal);
    $("themeBtn").addEventListener("click", toggleTheme);
    $("modelSel").addEventListener("change", onModelChange);
    $("manageBtn").addEventListener("click", openPapers);
    $("pmClose").addEventListener("click", closePapers);
    $("papersScrim").addEventListener("click", closePapers);

    const input = $("input");
    input.addEventListener("input", () => { autosize(); $("sendBtn").disabled = state.streaming || !input.value.trim(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
      if (e.key === "Escape") { closeDrawer(); closePapers(); }
    });
  }

  init();
})();
