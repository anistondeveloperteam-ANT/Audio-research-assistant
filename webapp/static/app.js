/* Research Assistant — web UI logic (vanilla JS, no build step). */
(() => {
  "use strict";

  // When the session is missing/expired the API replies 401 — send the user to
  // the login page instead of leaving the app in a broken, empty state. This also
  // recovers a stale page cached from before login was enabled.
  const _origFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const res = await _origFetch(...args);
    if (res.status === 401) {
      window.location.replace("/login");
      return new Promise(() => {});   // halt callers; the page is navigating away
    }
    return res;
  };

  const $ = (id) => document.getElementById(id);
  const api = {
    me: () => fetch("/api/me").then((r) => r.json()),
    logout: () => fetch("/api/logout", { method: "POST" }),
    review: (text) => fetch("/api/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }).then((r) => r.json()),
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
    srcSets: [],        // [{el, question, sources}] for per-query drawer navigation
    srcIndex: 0,
    abort: null,
    autoStick: true,
    nextTurnIndex: 0,
    mode: "Default",    // single optimized retrieval mode (no Fast/Balanced/Deep)
    topk: 8,            // hint only; the server selects sources adaptively
  };

  // Auto-routing: a clear "build / run / solve code" task goes to the autonomous
  // agent (write -> run in Docker -> verify -> refine). Everything else uses the
  // chat path, which already verifies its answer and runs any code it writes.
  function looksLikeCodingTask(t) {
    const s = " " + (t || "").toLowerCase().replace(/[^a-z0-9+ ]/g, " ") + " ";
    return /\b(implement|benchmark|simulate|simulation|leetcode|refactor|debug|optimi[sz]e)\b/.test(s)
      || /\bwrite\s+(a|an|me|the)?\s*(python|program|code|script|function|class)\b/.test(s)
      || /\b(code|program|script|function)\s+(to|that|for|which)\b/.test(s)
      || /\bsolve\b[^.]*\b(problem|equation|puzzle|leetcode|sudoku|maze)\b/.test(s)
      || /\b(find|compute|calculate|build)\b[^.]*\b(fastest|most efficient|optimal|best)\b[^.]*\b(algorithm|code|program|way|method)\b/.test(s);
  }

  // Icons for the per-question action buttons (copy / edit / delete).
  const ICON_COPY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
  const ICON_EDIT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
  const ICON_TRASH = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>';
  const ICON_REPEAT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 2v6h6"/><path d="M3 13a9 9 0 1 0 3-7.7L3 8"/></svg>';

  const SEND_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 11l5-5 5 5M12 6v13"/></svg>';
  const STOP_ICON = '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2.5" fill="currentColor"/></svg>';

  const EXAMPLES = [
    ["How does transformer attention work?", "…and why it scales better than RNNs."],
    ["What's the latest research on", "diffusion models? Summarize recent papers."],
    ["Implement and benchmark", "quicksort vs mergesort on 100k integers."],
    ["Find the best algorithm for", "shortest paths in a weighted graph."],
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

  // ---------- Markdown + math + citations ----------
  function renderMarkdown(el, text) {
    // 1) Protect $…$ / $$…$$ math from the markdown parser (so underscores/backslashes
    //    survive), 2) parse markdown, 3) restore the raw math, 4) render it with KaTeX.
    const math = [];
    const src = (text || "").replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g, (m) => {
      math.push(m);
      return "@@MATH" + (math.length - 1) + "@@";
    });
    let html = marked.parse(src, { breaks: true, gfm: true });
    html = html.replace(/@@MATH(\d+)@@/g, (_, i) => esc(math[+i]));
    el.innerHTML = html;
    stripCitations(el);        // drop ugly inline [n] markers (never inside code)
    renderMath(el);            // KaTeX
    enhanceCodeBlocks(el);
  }

  function renderMath(el) {
    if (!window.renderMathInElement) return;
    try {
      window.renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      });
    } catch (e) {}
  }

  // Remove inline bracketed citations ([2], [12], [3, 4]) from the rendered answer —
  // they look noisy. Skips code/links so it never mangles things like arr[12].
  // Remove noisy inline [n] citations AND emojis from the rendered answer (kept out
  // of code blocks/links), for a clean, professional look.
  const EMOJI_RE = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{2190}-\u{21FF}\u{2300}-\u{23FF}\u{FE00}-\u{FE0F}\u{200D}]/gu;
  function stripCitations(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => {
        const p = n.parentElement;
        return (!p || p.closest("pre, code, a")) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      node.nodeValue = node.nodeValue
        .replace(/[ \t]*\[\d+(?:\s*,\s*\d+)*\]/g, "")
        .replace(EMOJI_RE, "")
        .replace(/[ \t]{2,}/g, " ");
    }
  }

  function enhanceCodeBlocks(root) {
    root.querySelectorAll("pre > code").forEach((code) => {
      const pre = code.parentElement;
      if (!pre || (pre.parentElement && pre.parentElement.classList.contains("code-card"))) return;

      // Language label from the ```lang fence (marked adds language-xxx).
      const m = (code.className || "").match(/language-([\w+#.-]+)/i);
      const lang = (m ? m[1] : "code").toLowerCase();

      // Syntax highlight (highlight.js). Auto-detects when the language is unknown.
      if (window.hljs) { try { hljs.highlightElement(code); } catch (e) {} }

      // Wrap in an IDE-style card: header (dots + language + copy) over the code.
      const card = document.createElement("div");
      card.className = "code-card";
      const head = document.createElement("div");
      head.className = "code-head";
      head.innerHTML = '<span class="code-dots"><i></i><i></i><i></i></span>'
                     + '<span class="code-lang">' + esc(lang) + '</span>';
      const copy = document.createElement("button");
      copy.className = "code-copy"; copy.type = "button"; copy.textContent = "Copy";
      copy.addEventListener("click", () => {
        navigator.clipboard.writeText(code.innerText).then(() => {
          copy.textContent = "Copied ✓"; setTimeout(() => (copy.textContent = "Copy"), 1300);
        }).catch(() => {});
      });
      head.appendChild(copy);

      pre.parentNode.insertBefore(card, pre);
      card.appendChild(head);
      card.appendChild(pre);
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
    return;   // Sources panel removed — leave [n] markers as plain text.
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
        b.addEventListener("click", () => focusSource(parseInt(n, 10), b));
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
    const localRag = !!(state.cfg && state.cfg.local_rag_enabled);
    const heading = localRag ? "What do your papers say?" : "What would you like to research?";
    const blurb = localRag
      ? "Ask anything about your library. Every answer is grounded in your papers — each claim cited to its source, section, and page."
      : "Ask anything, or give it a coding task. It searches the web, papers, patents &amp; code, verifies its answer, and cites every source — or writes and runs code to prove the result.";
    inner().innerHTML = `
      <div class="welcome">
        <div class="hero-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg></div>
        <h1>${heading}</h1>
        <p>${blurb}</p>
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
      else if (b.dataset.act === "repeat") repeatUserMessage(m);
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
        <button class="ua-btn" data-act="repeat" title="Ask again (regenerate)" aria-label="Ask again">${ICON_REPEAT}</button>
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

  // "Ask again" — re-run this exact question, regenerating its answer in place
  // (drops this Q + everything after it, then re-sends the same text).
  async function repeatUserMessage(m) {
    if (state.streaming) { toast("Please wait for the answer to finish."); return; }
    const text = (m.querySelector(".bubble") || {}).textContent || "";
    if (!text.trim()) return;
    const idx = m.dataset.turnIndex;
    try {
      if (idx != null) await api.truncateTurns(state.currentId, idx);
    } catch { toast("Couldn't reprocess the question.", "error"); return; }
    let n = m.nextElementSibling;
    while (n) { const after = n.nextElementSibling; n.remove(); n = after; }
    m.remove();
    if (idx != null) state.nextTurnIndex = parseInt(idx, 10);
    $("input").value = text;
    autosize();
    send();
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
          <details class="thinking" style="display:none">
            <summary><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V17h6v-.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2z"/></svg><span class="th-label">Thinking…</span><span class="th-caret">▸</span></summary>
            <div class="th-body"></div>
          </details>
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
      thinking: m.querySelector(".thinking"),
      thinkBody: m.querySelector(".th-body"),
      thinkLabel: m.querySelector(".th-label"),
      md: m.querySelector(".md"),
      tools: m.querySelector(".msg-tools"),
    };
  }

  function appendThinking(h, text) {
    if (!h.thinking || !text) return;
    if (h.thinking.style.display === "none") {
      h.thinking.style.display = "";
      h.thinking.classList.add("live");
    }
    h._thinkRaw = (h._thinkRaw || "") + text;
    h.thinkBody.textContent = h._thinkRaw;
    if (h.thinking.open) h.thinkBody.scrollTop = h.thinkBody.scrollHeight;
    if (state.autoStick) scrollToBottom();
  }

  function finishThinking(h) {
    if (!h || !h.thinking || !h._thinkRaw) return;
    h.thinking.classList.remove("live");
    if (h.thinkLabel) h.thinkLabel.textContent = "Thought process";
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
    // "From memory" badge when the answer was reused from the saved-answer cache.
    if (h._cached) {
      const mem = document.createElement("span");
      mem.className = "speed-badge mem-badge";
      mem.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/></svg> From memory${h._cachedPct ? " · " + h._cachedPct + "%" : ""}`;
      mem.title = "Reused a saved answer" + (h._cachedKind ? " (" + h._cachedKind + " match)" : "");
      h.tools.appendChild(mem);
    }
    const copy = document.createElement("button");
    copy.className = "tool-btn";
    copy.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg> Copy`;
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(h.md.innerText).then(() => toast("Answer copied"));
    });
    h.tools.appendChild(copy);
    // The answer is peer-reviewed automatically (AUTO_REVIEW) — no manual button.
  }

  // ----- per-query source navigation -----
  function precedingUser(el) {
    let p = el && el.previousElementSibling;
    while (p) { if (p.classList.contains("user")) return p; p = p.previousElementSibling; }
    return null;
  }
  function questionForAnswer(el) {
    const u = precedingUser(el);
    const b = u && u.querySelector(".bubble");
    return b ? b.textContent : "";
  }
  function collectSourceSets() {
    const sets = [];
    inner().querySelectorAll(".msg.assistant").forEach((el) => {
      if (el._sources && el._sources.length) {
        sets.push({ el, sources: el._sources, question: el._question || questionForAnswer(el) });
      }
    });
    return sets;
  }
  function openSourcesForEl(el) {
    state.srcSets = collectSourceSets();
    let i = state.srcSets.findIndex((s) => s.el === el);
    if (i < 0) {
      state.srcSets.push({ el, sources: el._sources || [], question: el._question || questionForAnswer(el) });
      i = state.srcSets.length - 1;
    }
    openSourcesAt(i);
  }
  function openSourcesAt(i) {
    const sets = state.srcSets || [];
    if (!sets.length) { state.currentSources = []; renderSources([]); updateSourceNav(); openDrawer(); return; }
    state.srcIndex = Math.max(0, Math.min(sets.length - 1, i));
    const set = sets[state.srcIndex];
    state.currentSources = set.sources;
    renderSources(set.sources);
    updateSourceNav();
    openDrawer();
  }
  function updateSourceNav() {
    const sets = state.srcSets || [];
    const nav = $("drawerNav");
    if (!nav) return;
    nav.style.display = sets.length ? "flex" : "none";
    if (!sets.length) return;
    const i = state.srcIndex || 0;
    $("srcQuestion").textContent = sets[i].question || "This answer";
    $("srcPos").textContent = (i + 1) + " / " + sets.length;
    $("srcPrev").disabled = i <= 0;
    $("srcNext").disabled = i >= sets.length - 1;
  }

  // ---------- Sources drawer ----------
  function renderSources(sources) {
    state.currentSources = sources || [];
    const body = $("drawerBody");
    if (!body) return;   // Sources panel removed.
    if (!state.currentSources.length) {
      body.innerHTML = `<div class="drawer-empty">No sources for this answer.</div>`;
      return;
    }
    body.innerHTML = "";
    const TYPE = { local_pdf: "Paper", web: "Web", github_repo: "GitHub", github_code: "GitHub",
                   online_pdf: "PDF", research_paper: "Research", patent: "Patent" };
    state.currentSources.forEach((s) => {
      const st = s.source_type || "local_pdf";
      const titleInner = esc(prettyName(s.title));
      const titleEl = s.url
        ? `<a class="sc-title" href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${titleInner}</a>`
        : `<span class="sc-title">${titleInner}</span>`;
      const pages = s.page_start ? `pp. ${s.page_start}${s.page_end && s.page_end !== s.page_start ? "–" + s.page_end : ""}` : "";
      let meta = "";
      if (s.score) meta += `<span class="chip relevance">${Math.min(100, Math.round(s.score * 100))}% match</span>`;
      meta += `<span class="chip type type-${st}">${TYPE[st] || "Source"}</span>`;
      if (s.published) meta += `<span class="chip date" title="Published / updated">🗓 ${esc(String(s.published))}</span>`;
      if (st === "local_pdf") {
        if (s.section) meta += `<span class="chip">${esc(s.section)}</span>`;
        if (pages) meta += `<span class="chip">${pages}</span>`;
        if (s.graph_reason) meta += `<span class="chip">Graph: ${esc(s.graph_reason)}</span>`;
      } else {
        if (s.file_path) {
          const loc = esc(s.file_path) + (s.line_start ? ":" + s.line_start + (s.line_end ? "-" + s.line_end : "") : "");
          meta += `<span class="chip">${loc}</span>`;
        }
        if (s.page) meta += `<span class="chip">p.${s.page}</span>`;
        if (s.license) meta += `<span class="chip">${esc(s.license)}</span>`;
      }
      const card = document.createElement("div");
      card.className = "source-card";
      card.id = "src-card-" + s.n;
      card.innerHTML = `
        <div class="sc-head"><span class="sc-n">${s.n}</span>${titleEl}</div>
        <div class="sc-meta">${meta}</div>
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
  function openDrawer() { const d = $("drawer"); if (d) { d.classList.add("open"); $("scrim").classList.add("show"); } }
  function closeDrawer() { const d = $("drawer"); if (d) { d.classList.remove("open"); $("scrim").classList.remove("show"); } }
  function focusSource(n, chip) {
    // Open the sources for the answer this citation belongs to (with nav), then
    // jump to source [n]. Falls back to the current set if we can't find the message.
    const msg = chip && chip.closest && chip.closest(".msg.assistant");
    if (msg && msg._sources && msg._sources.length) openSourcesForEl(msg);
    else { renderSources(state.currentSources); updateSourceNav(); openDrawer(); }
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

  let _currentModel = "";
  function currentModelName() { return _currentModel; }

  async function send() {
    const text = $("input").value.trim();
    if (!text || state.streaming || !state.currentId) return;
    if (looksLikeCodingTask(text)) { sendAgent(text); return; }

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

  // ---------- Agent mode (write code -> run in Docker -> verify) ----------
  async function sendAgent(text) {
    if (inner().querySelector(".welcome")) inner().innerHTML = "";
    $("input").value = ""; autosize();
    addUserMessage(text, state.nextTurnIndex);   // view-only; agent runs aren't persisted
    setStreaming(true);
    const h = addAssistantMessage();
    h.statusText.textContent = "Agent working…";

    const genStart = performance.now();
    const timer = setInterval(() => {
      if (h.elapsed) h.elapsed.textContent = ((performance.now() - genStart) / 1000).toFixed(1) + "s";
    }, 100);

    h.md.style.display = ""; h.statusEl.style.display = "none";
    const handle = makeAgentUI(h.md);

    const controller = new AbortController();
    state.abort = controller;
    try {
      const resp = await fetch("/api/agent", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }), signal: controller.signal,
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
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
          if (!line) continue;
          let e; try { e = JSON.parse(line); } catch { continue; }
          handle(e);
        }
      }
    } catch (err) {
      handle({ type: "error", message: err.name === "AbortError" ? "Stopped." : ("Connection error: " + (err.message || "")) });
    } finally {
      state.abort = null;
      clearInterval(timer);
      finalizeTools(h, [], { seconds: (performance.now() - genStart) / 1000, model: "agent" });
      setStreaming(false);
      scrollToBottom();
      $("input").focus();
    }
  }

  // Claude-style agent timeline: each event becomes a step card with an icon, a
  // live spinner -> checkmark status, badges, and an expandable body.
  const A_ICON = {
    code:   '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6l-5 6 5 6M16 6l5 6-5 6"/></svg>',
    run:    '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 4l14 8-14 8z"/></svg>',
    review: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/></svg>',
    shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></svg>',
    error:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>',
  };

  function makeAgentUI(root) {
    root.innerHTML = "";
    root.classList.add("agent-run");
    const lead = document.createElement("div");
    lead.className = "agent-lead";
    lead.innerHTML = '<span class="al-dot"></span> Coding task — writing code, running it in a sandbox, and verifying.';
    root.appendChild(lead);
    let runCard = null;

    const note = (t) => { const d = document.createElement("div"); d.className = "agent-note"; d.textContent = t; root.appendChild(d); };
    const setState = (card, s) => {
      card.dataset.status = s;
      card.querySelector(".astep-state").innerHTML =
        s === "running" ? '<span class="astep-spin"></span>' : (s === "fail" ? "✕" : "✓");
    };
    const step = (icon, title, status) => {
      const card = document.createElement("div");
      card.className = "astep";
      card.innerHTML =
        '<div class="astep-head"><span class="astep-icon">' + icon + '</span>'
        + '<span class="astep-title"></span><span class="astep-badge" style="display:none"></span>'
        + '<span class="astep-state"></span></div><div class="astep-body" style="display:none"></div>';
      card.querySelector(".astep-title").textContent = title;
      setState(card, status || "done");
      card.querySelector(".astep-head").addEventListener("click", () => {
        const b = card.querySelector(".astep-body");
        if (b.innerHTML.trim()) b.style.display = (b.style.display === "none" ? "" : "none");
      });
      root.appendChild(card);
      return card;
    };
    const body = (card, html, open) => { const b = card.querySelector(".astep-body"); b.innerHTML = html; b.style.display = open ? "" : "none"; return b; };
    const badge = (card, txt, kind) => { const b = card.querySelector(".astep-badge"); b.textContent = txt; b.style.display = ""; b.className = "astep-badge" + (kind ? " " + kind : ""); };
    const codeInto = (parent, code, lang) => {
      const w = document.createElement("div");
      w.innerHTML = '<pre><code class="language-' + (lang || "python") + '">' + esc(code || "") + "</code></pre>";
      parent.appendChild(w); enhanceCodeBlocks(w);
    };
    const outPre = (txt, err) => { const p = document.createElement("pre"); p.className = "astep-out" + (err ? " err" : ""); p.textContent = txt; return p; };

    return function handle(e) {
      switch (e.type) {
        case "status": note(e.message); break;
        case "context": if (e.chars) note("Gathered " + e.chars + " chars of background"); break;
        case "warning": note("⚠ " + e.message); break;
        case "directive": note("🧭 " + e.text); break;
        case "think": {
          const r = document.createElement("div"); r.className = "agent-round";
          r.innerHTML = "<span>Attempt " + e.iteration + "</span>"; root.appendChild(r); break;
        }
        case "code": { const c = step(A_ICON.code, "Wrote a program", "done"); codeInto(body(c, "", true), e.code); break; }
        case "run": { runCard = step(A_ICON.run, "Running in Docker sandbox", "running"); break; }
        case "run_result": {
          const c = runCard || step(A_ICON.run, "Ran in sandbox", "done");
          setState(c, e.ok ? "done" : "fail");
          c.querySelector(".astep-title").textContent = e.ok ? "Ran successfully" : "Run failed";
          if (e.summary) badge(c, e.summary, e.ok ? "ok" : "bad");
          const b = body(c, "", !e.ok);
          if (e.stdout) b.appendChild(outPre(e.stdout));
          if (!e.ok && e.stderr) b.appendChild(outPre(e.stderr.split("\n").slice(-8).join("\n"), true));
          if (e.error) b.appendChild(outPre(e.error, true));
          runCard = null; break;
        }
        case "reflect": {
          const v = e.verdict || {};
          const c = step(A_ICON.review, v.done ? "Reviewed — good to go" : "Reviewed — needs another pass", "done");
          if (v.score != null) badge(c, "score " + v.score, v.done ? "ok" : "");
          if (v.feedback) body(c, '<div class="astep-note">' + esc(v.feedback) + "</div>", false);
          break;
        }
        case "blocked": { const c = step(A_ICON.shield, "Blocked by policy", "fail"); body(c, '<div class="astep-note err">' + esc(e.reason || "") + "</div>", true); break; }
        case "error": { const c = step(A_ICON.error, "Error", "fail"); body(c, '<div class="astep-note err">' + esc(e.message || "") + "</div>", true); break; }
        case "final": {
          const card = document.createElement("div");
          card.className = "agent-final " + (e.success ? "ok" : "warn");
          const head = document.createElement("div"); head.className = "af-head";
          head.textContent = e.success ? "✓ Best result — verified" : "⚠ Best attempt (not fully verified)";
          card.appendChild(head);
          if (e.answer) { const a = document.createElement("div"); a.className = "af-answer"; a.textContent = e.answer; card.appendChild(a); }
          if (e.output) { const o = document.createElement("div"); o.className = "af-block"; o.innerHTML = '<div class="af-label">Output</div>'; o.appendChild(outPre(e.output)); card.appendChild(o); }
          if (e.code) { const cc = document.createElement("div"); cc.className = "af-block"; cc.innerHTML = '<div class="af-label">Program</div>'; codeInto(cc, e.code); card.appendChild(cc); }
          root.appendChild(card); break;
        }
      }
      scrollToBottom();
    };
  }


  function handleEvent(ev, h, getAns, setAns, scheduleRender) {
    switch (ev.type) {
      case "status":
        h.statusText.textContent = ev.message || "Working…";
        break;
      case "thinking":
        appendThinking(h, ev.text || "");
        break;
      case "sanity":
        h.statusEl.style.display = "none"; h.md.style.display = "";
        renderMarkdown(h.md, "⚠️ " + (ev.message || "Please rephrase your question."));
        break;
      case "sources": {
        const n = (ev.sources || []).length;
        if (n) h.statusText.textContent = `Found ${n} relevant source${n > 1 ? "s" : ""} — writing the answer…`;
        break;
      }
      case "token":
        finishThinking(h);
        setAns(getAns() + (ev.text || ""));
        scheduleRender();
        break;
      case "warning":
        toast(ev.message || "Heads up", "warn");
        break;
      case "error":
        toast(ev.message || "Error", "error");
        setAns(getAns() + "\n\n_" + (ev.message || "error") + "_");
        scheduleRender();
        break;
      case "done":
        finishThinking(h);
        if (ev.cached) { h._cached = true; h._cachedPct = ev.similarity || 0; h._cachedKind = ev.match_kind || ""; }
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
    const pl = $("provLabel"); if (pl) pl.textContent = label;
    const pd = $("provDot"); if (pd) pd.style.background = "var(--ok)";
  }
  function closeModelMenu() {
    const p = $("modelPick"); if (p) p.classList.remove("open");
    const b = $("modelBtn"); if (b) b.setAttribute("aria-expanded", "false");
  }
  function toggleModelMenu() {
    const p = $("modelPick"); if (!p) return;
    const open = p.classList.toggle("open");
    const b = $("modelBtn"); if (b) b.setAttribute("aria-expanded", open ? "true" : "false");
  }
  async function loadModels() {
    try {
      const data = await api.models();
      const menu = $("modelMenu");
      menu.innerHTML = "";
      const cur = data.current || {};
      let curLabel = "";
      (data.options || []).forEach((o) => {
        if (o.model === cur.model) curLabel = o.label;
        const row = document.createElement("button");
        row.type = "button";
        row.className = "mp-opt" + (o.model === cur.model ? " active" : "");
        row.setAttribute("role", "option");
        const vendor = o.vendor || "?";
        const pill = o.available ? (o.free ? '<span class="mp-pill free">Free</span>'
                                           : '<span class="mp-pill paid">Paid</span>')
                                 : '<span class="mp-pill needkey">Add key</span>';
        row.innerHTML =
          `<span class="mp-badge" data-v="${esc(vendor)}">${esc(vendor.slice(0, 2))}</span>` +
          `<span class="mp-opt-main"><span class="mp-opt-name">${esc(o.name || o.model)}</span>` +
          `<span class="mp-opt-vendor">${esc(vendor)}</span></span>` + pill +
          `<svg class="mp-opt-tick" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>`;
        row.addEventListener("click", () => selectModel(o, row));
        menu.appendChild(row);
      });
      _currentModel = cur.model || "";
      $("modelLabel").textContent = curLabel || `${cur.provider} · ${cur.model}`;
      setProviderLabel(`${cur.provider} · ${cur.model}`);
    } catch { const ml = $("modelLabel"); if (ml) ml.textContent = "unavailable"; }
  }
  async function selectModel(o, row) {
    closeModelMenu();
    try {
      const res = await api.setModel(o.provider, o.model);
      if (res.error) { toast(res.error, "error"); return; }
      _currentModel = o.model;
      $("modelLabel").textContent = o.label;
      $("modelMenu").querySelectorAll(".mp-opt").forEach((el) => el.classList.remove("active"));
      if (row) row.classList.add("active");
      setProviderLabel(res.label);
      toast("Model switched to " + res.model);
    } catch { toast("Could not switch model.", "error"); }
  }

  // ---------- Init ----------
  async function init() {
    applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
    try { if (localStorage.getItem("ara-sidebar") === "collapsed" && window.innerWidth > 880) $("app").classList.add("collapsed"); } catch {}
    try { state.cfg = await api.config(); } catch {}
    // Auth: when login is enabled, show the signed-in user + a sign-out button.
    try {
      const me = await api.me();
      if (me && me.auth) {
        if (!me.user_id) { window.location.href = "/login"; return; }
        $("userChip").style.display = "";
        $("userName").textContent = me.user_id;
        $("userAvatar").textContent = (me.user_id || "?").trim().charAt(0).toUpperCase();
        $("logoutBtn").addEventListener("click", async () => {
          try { await api.logout(); } catch {}
          window.location.href = "/login";
        });
      }
    } catch {}
    // Web search is automatic (no toggle): the server falls back to web / research
    // papers / patents / GitHub whenever the local papers don't have the answer.
    { const pl = $("provLabel"); if (pl) pl.textContent = state.cfg.provider || "ready"; }
    if (!state.cfg.provider || state.cfg.provider === "unknown") { const pd = $("provDot"); if (pd) pd.style.background = "var(--amber)"; }

    // Web-search assistant mode: hide local-paper UI when local RAG is off.
    if (state.cfg.local_rag_enabled) {
      loadLibrary();
    } else {
      ["addPaperBtn", "manageBtn"].forEach((id) => { const el = $(id); if (el) el.style.display = "none"; });
      const lib = $("libLabel"); if (lib) lib.textContent = "Web search mode";
    }
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
    $("modelBtn").addEventListener("click", (e) => { e.stopPropagation(); toggleModelMenu(); });
    document.addEventListener("click", (e) => { if (!e.target.closest("#modelPick")) closeModelMenu(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModelMenu(); });
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
      if (e.key === "Escape") { closePapers(); }
    });
  }

  init();
})();
