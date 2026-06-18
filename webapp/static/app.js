/* Research Assistant — Workspace UI logic (vanilla JS, no build step).
   Wires the monochrome-glass mockup to the live backend: sessions, streaming
   chat + code agent, version tree, sources drawer, model picker, paper upload
   + library, auth, theme. */
(() => {
  "use strict";

  // Expired/missing session -> the API replies 401; send the user to login
  // instead of leaving an empty, broken shell.
  const _origFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const res = await _origFetch(...args);
    if (res.status === 401) { window.location.replace("/login"); return new Promise(() => {}); }
    return res;
  };

  const $ = (id) => document.getElementById(id);
  const api = {
    me: () => fetch("/api/me").then((r) => r.json()),
    logout: () => fetch("/api/logout", { method: "POST" }),
    config: () => fetch("/api/config").then((r) => r.json()),
    sessions: () => fetch("/api/sessions").then((r) => r.json()),
    createSession: () => fetch("/api/sessions", { method: "POST" }).then((r) => r.json()),
    renameSession: (id, title) =>
      fetch(`/api/sessions/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title }) }),
    deleteSession: (id) => fetch(`/api/sessions/${id}`, { method: "DELETE" }),
    tree: (id) => fetch(`/api/sessions/${id}/tree`).then((r) => r.json()),
    getVersion: (id, turnId) => fetch(`/api/sessions/${id}/versions/${turnId}`).then((r) => r.json()),
    setActiveVersion: (id, payload) =>
      fetch(`/api/sessions/${id}/versions/active`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }).then((r) => r.json()),
    deleteNode: (id, nodeId) => fetch(`/api/sessions/${id}/nodes/${nodeId}`, { method: "DELETE" }).then((r) => r.json()),
    library: () => fetch("/api/library").then((r) => r.json()),
    papers: () => fetch("/api/papers").then((r) => r.json()),
    deletePaper: (id) => fetch(`/api/papers/${id}`, { method: "DELETE" }).then((r) => r.json()),
    removeIncomplete: () => fetch("/api/papers/remove-incomplete", { method: "POST" }).then((r) => r.json()),
    models: () => fetch("/api/models").then((r) => r.json()),
    setModel: (provider, model) =>
      fetch("/api/model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider, model }) }).then((r) => r.json()),
    upload: (files) => {
      const fd = new FormData();
      (Array.isArray(files) ? files : [files]).forEach((f) => fd.append("files", f));
      return fetch("/api/upload", { method: "POST", body: fd }).then((r) => r.json());
    },
    cancelIngest: () => fetch("/api/ingest/cancel", { method: "POST" }).then((r) => r.json()),
  };

  const state = {
    cfg: { local_rag_enabled: true, provider: "" },
    sessions: [],
    currentId: null,
    streaming: false,
    ingesting: false,
    currentSources: [],
    srcSets: [],
    srcIndex: 0,
    abort: null,
    autoStick: true,
    mode: "fast",
    topk: 8,
    tree: [],
  };

  // Auto-route an obvious "build / run / solve code" task to the autonomous agent.
  // Mirrors backend/answering/code_intent.py::is_code_intent — keep in sync.
  function looksLikeCodingTask(t) {
    const s = " " + (t || "").toLowerCase().replace(/[^a-z0-9+# ]/g, " ").replace(/\s+/g, " ") + " ";
    return /\b(implement|simulate|simulation|benchmark|refactor|debug|optimi[sz]e|leetcode)\b/.test(s)
      || /\b(write|give|gen|generate|show|build|create|make|provide|produce|need|want)\b.{0,40}\b(code|script|program|function|implementation|snippet)\b/.test(s)
      || /\bpython\b.{0,40}\b(code|script|program|function|implementation|snippet|class)\b/.test(s)
      || /\b(code|script|program|function|implementation|snippet|class)\b.{0,40}\bpython\b/.test(s)
      || /\b(code|script|snippet)\s+(for|to|that|which)\b/.test(s)
      || /\bimplementation\s+(of|for)\b/.test(s);
  }

  // ---------- icons ----------
  const ICON_EDIT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
  const ICON_COPY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';
  const ICON_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
  const ICON_REGEN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5M21 12a9 9 0 0 1-15 6.7L3 16M3 21v-5h5"/></svg>';
  const ICON_TRASH = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6"/></svg>';
  const ICON_PREV = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M15 18l-6-6 6-6"/></svg>';
  const ICON_NEXT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M9 18l6-6-6-6"/></svg>';
  const ICON_CHAT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
  const ICON_PENCIL_MINI = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>';
  const ICON_TRASH_MINI = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>';
  const ICON_DOC = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>';
  const SEND_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
  const STOP_ICON = '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2.5" fill="currentColor"/></svg>';

  const EXAMPLES = [
    ["How does transformer attention work?", "…and why it scales better than RNNs."],
    ["What's the latest research on", "diffusion models? Summarize recent papers."],
    ["Implement and benchmark", "quicksort vs mergesort on 100k integers."],
    ["Find the best algorithm for", "shortest paths in a weighted graph."],
  ];

  const esc = (s) => (s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const prettyName = (s) => {
    let t = (s || "").replace(/\.pdf$/i, "").replace(/_+/g, " ").replace(/\s+/g, " ").trim();
    if (t && !/[a-z]/.test(t)) t = t.toLowerCase().replace(/\b[a-z]/g, (c) => c.toUpperCase());
    return t;
  };

  // ---------- toast ----------
  let _toastT = null;
  function toast(msg, kind) {
    const t = $("toast"); if (!t) return;
    t.querySelector(".ttext").textContent = msg;
    t.classList.toggle("error", kind === "error");
    t.classList.add("show");
    clearTimeout(_toastT);
    _toastT = setTimeout(() => t.classList.remove("show"), 3600);
  }

  // ---------- markdown + math + citations ----------
  function renderMarkdown(el, text) {
    const math = [];
    const src = (text || "").replace(
      /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\$([^$\n]+?)\$|\\\(([\s\S]+?)\\\)/g, (m) => {
        math.push(m); return "@@MATH" + (math.length - 1) + "@@";
      });
    let html = (window.marked ? marked.parse(src, { breaks: true, gfm: true }) : esc(src));
    html = html.replace(/@@MATH(\d+)@@/g, (_, i) => esc(math[+i]));
    html = html.replace(/\s*\[not in (?:the )?sources?\]/gi,
      ' <sup class="ungrounded" title="Not supported by the retrieved sources">unverified</sup>');
    el.innerHTML = html;
    stripEmptySourcesColumn(el);
    cleanText(el);
    linkifyCitations(el);
    renderMath(el);
    enhanceCodeBlocks(el);
  }
  function renderMath(el) {
    if (!window.renderMathInElement) return;
    try {
      window.renderMathInElement(el, {
        delimiters: [
          { left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      });
    } catch (e) {}
  }
  const EMOJI_RE = /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}\u{200D}]/gu;
  function cleanText(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => { const p = n.parentElement; return (!p || p.closest("pre, code, a")) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT; },
    });
    const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) node.nodeValue = node.nodeValue.replace(EMOJI_RE, "").replace(/[ \t]{2,}/g, " ");
  }
  function stripEmptySourcesColumn(root) {
    root.querySelectorAll("table").forEach((table) => {
      const ths = Array.from(table.querySelectorAll("thead th"));
      const bodyRows = Array.from(table.querySelectorAll("tbody tr"));
      for (let col = ths.length - 1; col >= 0; col--) {
        if (!/^sources?$/i.test((ths[col].textContent || "").trim())) continue;
        const empty = bodyRows.every((tr) => !(((tr.children[col] || {}).textContent) || "").trim());
        if (!empty) continue;
        ths[col].remove();
        bodyRows.forEach((tr) => { if (tr.children[col]) tr.children[col].remove(); });
      }
    });
  }
  // Wrap markdown fenced code in an IDE-style card. Uses unique `.mdcode*` classes
  // so it never collides with the agent timeline's `.code` block.
  function enhanceCodeBlocks(root) {
    root.querySelectorAll("pre > code").forEach((code) => {
      const pre = code.parentElement;
      if (!pre || (pre.parentElement && pre.parentElement.classList.contains("mdcode"))) return;
      const m = (code.className || "").match(/language-([\w+#.-]+)/i);
      const lang = (m ? m[1] : "code").toLowerCase();
      if (window.hljs) { try { hljs.highlightElement(code); } catch (e) {} }
      const card = document.createElement("div"); card.className = "mdcode";
      const head = document.createElement("div"); head.className = "mdcode-bar";
      head.innerHTML = '<span class="dots"><i></i><i></i><i></i></span><span class="mdcode-lang">' + esc(lang) + '</span>';
      const copy = document.createElement("button");
      copy.className = "mdcode-copy"; copy.type = "button"; copy.textContent = "Copy";
      copy.addEventListener("click", () => {
        navigator.clipboard.writeText(code.innerText).then(() => { copy.textContent = "Copied ✓"; setTimeout(() => (copy.textContent = "Copy"), 1300); }).catch(() => {});
      });
      head.appendChild(copy);
      pre.parentNode.insertBefore(card, pre);
      card.appendChild(head); card.appendChild(pre);
    });
  }
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
        if (p.closest("pre, code, a, .chip")) return NodeFilter.FILTER_REJECT;
        return /\[\d+\]/.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      },
    });
    const targets = []; while (walker.nextNode()) targets.push(walker.currentNode);
    const owner = root.closest && root.closest(".ai-card");
    const srcList = (owner && owner._sources) || state.currentSources || [];
    const nSources = srcList.length;
    for (const node of targets) {
      const frag = document.createDocumentFragment();
      let last = 0; const s = node.nodeValue;
      s.replace(/\[(\d+)\]/g, (m, n, idx) => {
        if (idx > last) frag.appendChild(document.createTextNode(s.slice(last, idx)));
        const num = parseInt(n, 10);
        if (num >= 1 && num <= nSources) {
          const src = srcList[num - 1];
          const cls = "chip " + (src && src.source_type === "local_pdf" ? "chip-pdf" : "chip-web");
          let el;
          if (src && src.url) { el = document.createElement("a"); el.href = src.url; el.target = "_blank"; el.rel = "noopener noreferrer"; }
          else { el = document.createElement("button"); el.type = "button"; el.style.cssText = "border:none;font-family:inherit"; el.addEventListener("click", () => focusSource(num, el)); }
          el.className = cls; el.textContent = n; el.dataset.n = n;
          el.addEventListener("mouseenter", () => showCitePop(el, n));
          el.addEventListener("mouseleave", hideCitePop);
          frag.appendChild(el);
        } else if (nSources === 0) {
          frag.appendChild(document.createTextNode(m));
        }
        last = idx + m.length; return m;
      });
      if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  // ---------- transcript ----------
  const thread = () => $("thread");

  function showWelcome() {
    const localRag = !!(state.cfg && state.cfg.local_rag_enabled);
    const heading = localRag ? "What do your papers say?" : "What would you like to research?";
    const blurb = localRag
      ? "Ask anything about your library. Every answer is grounded in your papers — each claim cited to its source, section, and page."
      : "Ask anything, or give it a coding task. It searches the web, papers, patents &amp; code, verifies its answer, and cites every source — or writes and runs code to prove the result.";
    const w = document.createElement("div");
    w.className = "welcome";
    w.innerHTML = `
      <div class="hero-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg></div>
      <h1>${heading}</h1>
      <p>${blurb}</p>
      <div class="examples" id="examples"></div>`;
    thread().appendChild(w);
    const box = w.querySelector("#examples");
    EXAMPLES.forEach(([k, sub]) => {
      const b = document.createElement("button");
      b.className = "example";
      b.innerHTML = `<span class="ex-k">${esc(k)}</span><span>${esc(sub)}</span>`;
      b.addEventListener("click", () => { $("composerInput").value = k + " " + sub; autosize(); send(); });
      box.appendChild(b);
    });
  }

  // ---- user message ----
  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "row user";
    row.innerHTML = `<div class="user-col">
        <div class="qver" style="display:none"></div>
        <div class="user-msg">
          <button class="user-edit" title="Edit &amp; resend" aria-label="Edit question">${ICON_EDIT}</button>
          <div class="user-bubble"></div>
        </div>
      </div>`;
    row.querySelector(".user-bubble").textContent = text;
    row.querySelector(".user-edit").addEventListener("click", () => startEdit(row));
    thread().appendChild(row);
    scrollToBottom(true);
    return row;
  }
  function fitArea(ta) { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 200) + "px"; }
  function startEdit(row) {
    if (state.streaming) { toast("Please wait for the answer to finish."); return; }
    const msg = row.querySelector(".user-msg");
    const bubble = row.querySelector(".user-bubble");
    if (bubble.classList.contains("editing")) return;
    const text = bubble.textContent;
    msg.classList.add("editing"); bubble.classList.add("editing");
    bubble.innerHTML = `<textarea class="edit-area"></textarea>
      <div class="edit-actions"><button class="edit-btn cancel">Cancel</button><button class="edit-btn save">Save &amp; resend</button></div>`;
    const ta = bubble.querySelector(".edit-area"); ta.value = text;
    ta.addEventListener("input", () => fitArea(ta)); fitArea(ta); ta.focus(); ta.setSelectionRange(text.length, text.length);
    bubble.querySelector(".cancel").addEventListener("click", () => endEdit(row, text));
    bubble.querySelector(".save").addEventListener("click", () => saveEdit(row, text));
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); saveEdit(row, text); }
      else if (e.key === "Escape") { e.preventDefault(); endEdit(row, text); }
    });
  }
  function endEdit(row, text) {
    const msg = row.querySelector(".user-msg"), bubble = row.querySelector(".user-bubble");
    msg.classList.remove("editing"); bubble.classList.remove("editing");
    bubble.textContent = text;
  }
  async function saveEdit(row, original) {
    const ta = row.querySelector(".edit-area");
    const edited = (ta ? ta.value : "").trim();
    if (!edited) { toast("The question can't be empty."); return; }
    if (edited === original) { endEdit(row, original); return; }
    const node = row.dataset.nodeId;
    endEdit(row, original);
    if (!node) { toast("Couldn't edit the question.", "error"); return; }
    $("composerInput").value = edited; autosize();
    send({ editNodeId: node });
  }

  // ---- assistant message ----
  // Each answer is ONE row holding a vertical stack: a live "thinking" cube,
  // an interactive "reason" steps panel, then the answer card.
  const CUBE = '<div class="think-cube"><div class="cube"><span class="face fr"></span><span class="face bk"></span><span class="face rt"></span><span class="face lf"></span><span class="face tp"></span><span class="face bm"></span></div></div>';
  function addAssistantMessage() {
    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = `
      <div class="astack">
        <div class="thinking">
          ${CUBE}
          <div class="think-body"><div class="think-label">Thinking…</div><div class="think-sub">Reading your question</div></div>
        </div>
        <div class="reason" style="display:none">
          <div class="reason-head">
            <div class="reason-orb"><div class="orbit"><div class="ring"></div></div><div class="core"></div></div>
            <div class="reason-check"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg></div>
            <span class="reason-title">Reasoning…</span>
            <span class="reason-time"></span>
            <svg class="reason-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg>
          </div>
          <div class="reason-steps"></div>
        </div>
        <div class="ai-card" style="display:none">
          <div class="badge-slot"></div>
          <div class="answer-body"></div>
          <div class="ai-acts" style="display:none"></div>
        </div>
      </div>`;
    thread().appendChild(row);
    scrollToBottom(true);
    const card = row.querySelector(".ai-card");
    const reason = row.querySelector(".reason");
    const h = {
      el: row, stack: row.querySelector(".astack"), card,
      thinkingEl: row.querySelector(".thinking"),
      reason, reasonSteps: reason.querySelector(".reason-steps"),
      reasonTitle: reason.querySelector(".reason-title"),
      reasonTime: reason.querySelector(".reason-time"),
      badge: card.querySelector(".badge-slot"),
      md: card.querySelector(".answer-body"),
      acts: card.querySelector(".ai-acts"),
    };
    reason.querySelector(".reason-head").addEventListener("click", () => { if (reason.classList.contains("done")) reason.classList.toggle("collapsed"); });
    card._h = h;
    return h;
  }
  // reveal helpers
  function revealCard(h) { if (h.thinkingEl) h.thinkingEl.style.display = "none"; if (h.card) h.card.style.display = ""; }
  function revealReason(h) {
    if (h._reasonStart == null) h._reasonStart = performance.now();
    if (h.thinkingEl) h.thinkingEl.style.display = "none";
    if (h.reason) h.reason.style.display = "";
  }
  function setStepDone(step) { if (step && step.classList.contains("running")) { step.classList.remove("running"); step.classList.add("done"); step.querySelector(".rnode").innerHTML = ICON_CHECK; } }
  // a human-readable pipeline step (status events)
  function appendProcess(h, text) {
    if (!h.reason || !text) return;
    if (h._lastProc === text) return; h._lastProc = text;
    revealReason(h);
    setStepDone(h._curStep);
    const step = document.createElement("div"); step.className = "rstep running";
    step.innerHTML = `<div class="rnode"><span class="spin"></span></div><div class="rstep-body"><div class="rstep-label"></div><div class="rstep-detail"></div></div>`;
    step.querySelector(".rstep-label").textContent = text;
    h.reasonSteps.appendChild(step);
    h._curStep = step;
    h.reasonTitle.textContent = text;
    if (state.autoStick) scrollToBottom();
  }
  // a small chip under the current step (e.g. "Found 8 sources")
  function reasonChip(h, text, onClick) {
    if (!h._curStep) appendProcess(h, "Working…");
    const detail = h._curStep.querySelector(".rstep-detail");
    const chip = document.createElement("div");
    chip.className = "rchip" + (onClick ? " link" : "");
    chip.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg><span></span>`;
    chip.querySelector("span").textContent = text;
    if (onClick) chip.addEventListener("click", onClick);
    detail.appendChild(chip);
  }
  // raw model reasoning tokens -> a muted growing block inside a dedicated step
  function appendThinking(h, text) {
    if (!h.reason || !text) return;
    revealReason(h);
    if (!h._rawStep) {
      setStepDone(h._curStep);
      h._rawStep = document.createElement("div"); h._rawStep.className = "rstep running";
      h._rawStep.innerHTML = `<div class="rnode"><span class="spin"></span></div><div class="rstep-body"><div class="rstep-label">Reasoning</div><div class="rstep-detail"><div class="rraw"></div></div></div>`;
      h.reasonSteps.appendChild(h._rawStep);
      h._curStep = h._rawStep;
      h._rawEl = h._rawStep.querySelector(".rraw");
    }
    h._thinkRaw = (h._thinkRaw || "") + text;
    h._rawEl.textContent = h._thinkRaw;
    h._rawEl.scrollTop = h._rawEl.scrollHeight;
    if (state.autoStick) scrollToBottom();
  }
  // close the reason panel (collapse to a "Reasoned for Xs" summary), or remove it if empty
  function finishReason(h) {
    if (!h || !h.reason) return;
    if (h.thinkingEl) h.thinkingEl.remove();
    setStepDone(h._curStep);
    if (!h.reasonSteps.children.length) { h.reason.remove(); return; }
    const secs = h._reasonStart != null ? (performance.now() - h._reasonStart) / 1000 : 0;
    h.reasonTitle.textContent = secs ? `Reasoned for ${secs.toFixed(1)}s` : "Reasoning";
    h.reasonTime.textContent = "";
    h.reason.classList.add("done", "collapsed");
  }
  // history / version render: no live bits, just the answer card
  function staticCard(h) {
    if (h.thinkingEl) h.thinkingEl.remove();
    if (h.reason) h.reason.remove();
    h.card.style.display = "";
  }

  const GRADE_CLASS = { strong: "badge-lib", partial: "badge-mix", none: "badge-web" };
  function renderBadges(h, meta) {
    h.badge.innerHTML = "";
    if (h._grade) {
      const g = document.createElement("span");
      g.className = "badge " + (GRADE_CLASS[(h._grade || "").toLowerCase()] || "badge-web");
      g.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg> ${esc(h._gradeLabel || h._grade)}`;
      g.title = h._gradeMsg || "";
      h.badge.appendChild(g);
    }
    if (meta && meta.seconds != null) {
      const b = document.createElement("span"); b.className = "speed-badge";
      const model = (meta.model || "").split("/").pop();
      b.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 4.5 13H11l-1 9 8.5-11H12z"/></svg> ${meta.seconds.toFixed(1)}s${model ? " · " + esc(model) : ""}`;
      h.badge.appendChild(b);
    }
    if (h._cached) {
      const m = document.createElement("span"); m.className = "speed-badge";
      m.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/></svg> From memory${h._cachedPct ? " · " + h._cachedPct + "%" : ""}`;
      h.badge.appendChild(m);
    }
  }

  function finalizeActs(h, sources, meta) {
    revealCard(h);
    h.card._sources = sources || [];
    h.card._question = questionForAnswer(h.card);
    renderBadges(h, meta);
    h.acts.style.display = "flex";
    h.acts.innerHTML = "";
    const copy = document.createElement("button");
    copy.className = "act"; copy.title = "Copy answer"; copy.innerHTML = ICON_COPY;
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(h.md.innerText).then(() => {
        copy.classList.add("copied"); copy.innerHTML = ICON_CHECK;
        toast("Answer copied");
        setTimeout(() => { copy.classList.remove("copied"); copy.innerHTML = ICON_COPY; }, 1300);
      }).catch(() => {});
    });
    const regen = document.createElement("button");
    regen.className = "act"; regen.title = "Regenerate"; regen.innerHTML = ICON_REGEN;
    regen.addEventListener("click", () => regenerateAnswer(h.card));
    const del = document.createElement("button");
    del.className = "act"; del.title = "Delete this question & its answers"; del.innerHTML = ICON_TRASH;
    del.addEventListener("click", () => deleteExchange(h.card));
    const versions = document.createElement("div");
    versions.className = "versions"; versions.style.display = "none";
    versions.innerHTML = `<button class="vbtn vprev" aria-label="Previous version">${ICON_PREV}</button><span class="vnum"></span><button class="vbtn vnext" aria-label="Next version">${ICON_NEXT}</button>`;
    versions.querySelector(".vprev").addEventListener("click", () => switchAnswerVersion(h.card, -1));
    versions.querySelector(".vnext").addEventListener("click", () => switchAnswerVersion(h.card, 1));
    h.acts.append(copy, regen, del, versions);
    updateAnswerSwitcher(h.card);
  }

  function precedingUserRow(cardEl) {
    let p = cardEl.closest(".row");
    p = p && p.previousElementSibling;
    while (p) { if (p.classList.contains("user")) return p; p = p.previousElementSibling; }
    return null;
  }
  function questionForAnswer(cardEl) {
    const u = precedingUserRow(cardEl);
    const b = u && u.querySelector(".user-bubble");
    return b ? b.textContent : "";
  }

  // ---- versions ----
  function updateQuestionSwitcher(row) {
    const qv = row.querySelector(".qver");
    const slot = row._slot;
    const total = slot ? slot.version_total : 1;
    if (!slot || total <= 1) { qv.style.display = "none"; qv.innerHTML = ""; return; }
    qv.style.display = "inline-flex";
    qv.innerHTML = `<button class="vbtn vprev" aria-label="Previous question">${ICON_PREV}</button><span class="vnum">${row._qIndex} / ${total}</span><button class="vbtn vnext" aria-label="Next question">${ICON_NEXT}</button>`;
    const prev = qv.querySelector(".vprev"), next = qv.querySelector(".vnext");
    if (row._qIndex <= 1) prev.classList.add("disabled");
    if (row._qIndex >= total) next.classList.add("disabled");
    prev.addEventListener("click", () => switchQuestionVersion(row, -1));
    next.addEventListener("click", () => switchQuestionVersion(row, 1));
  }
  function updateAnswerSwitcher(cardEl) {
    const h = cardEl._h; if (!h) return;
    const versions = h.acts.querySelector(".versions"); if (!versions) return;
    const qv = cardEl._qv;
    const total = qv ? qv.answer_total : 1;
    if (!qv || total <= 1) { versions.style.display = "none"; return; }
    versions.style.display = "flex";
    versions.querySelector(".vnum").textContent = cardEl._aIndex + " / " + total;
    versions.querySelector(".vprev").classList.toggle("disabled", cardEl._aIndex <= 1);
    versions.querySelector(".vnext").classList.toggle("disabled", cardEl._aIndex >= total);
  }

  function renderTree(tree) {
    thread().innerHTML = "";
    state.tree = tree || [];
    if (!state.tree.length) { showWelcome(); state.currentSources = []; renderSources([]); return; }
    state.tree.forEach(renderSlot);
    let lastSrc = [];
    for (const slot of state.tree) {
      const qv = slot.versions.find((v) => v.version_index === slot.active_version_index);
      const a = qv && qv.answers.find((x) => x.version_index === qv.active_answer_index);
      if (a && a.sources) lastSrc = a.sources;
    }
    renderSources(lastSrc);
    scrollToBottom(true);
  }
  function renderSlot(slot) {
    const qv = slot.versions.find((v) => v.version_index === slot.active_version_index) || slot.versions[slot.versions.length - 1];
    const row = addUserMessage((qv && qv.content) || "");
    row.dataset.nodeId = slot.node_id;
    row._slot = slot;
    row._qIndex = slot.active_version_index;
    updateQuestionSwitcher(row);
    if (!qv) return;
    const a = qv.answers.find((x) => x.version_index === qv.active_answer_index) || qv.answers[qv.answers.length - 1];
    if (!a) return;
    const h = addAssistantMessage();
    staticCard(h);
    h.card._sources = a.sources || [];
    renderMarkdown(h.md, a.content || "");
    finalizeActs(h, a.sources || []);
    h.card._qv = qv;
    h.card._aIndex = qv.active_answer_index;
    h.card._nodeId = slot.node_id;
    h.card.dataset.qversionId = String(qv.turn_id);
    row._answerEl = h.card;
    updateAnswerSwitcher(h.card);
  }
  async function reloadTree() {
    if (!state.currentId) return;
    renderTree(await api.tree(state.currentId));
  }
  async function showAnswerVersion(cardEl, qv, aIndex) {
    const h = cardEl._h;
    const a = qv && qv.answers.find((x) => x.version_index === aIndex);
    if (!h || !a) return;
    let content = a.content, sources = a.sources;
    if (content == null) {
      try { const v = await api.getVersion(state.currentId, a.turn_id); content = v.content || ""; sources = v.sources || []; a.content = content; a.sources = sources; }
      catch { toast("Couldn't load that answer.", "error"); return; }
    }
    cardEl._qv = qv; cardEl._aIndex = aIndex; cardEl.dataset.qversionId = String(qv.turn_id);
    cardEl._sources = sources || [];
    h.md.style.display = "";
    renderMarkdown(h.md, content || "");
    finalizeActs(h, sources || []);
    qv.active_answer_index = aIndex;
  }
  async function switchQuestionVersion(row, delta) {
    const slot = row._slot;
    if (!slot || state.streaming) return;
    const total = slot.version_total;
    const next = Math.min(total, Math.max(1, row._qIndex + delta));
    if (next === row._qIndex) return;
    const qv = slot.versions.find((v) => v.version_index === next); if (!qv) return;
    let content = qv.content;
    if (content == null) { try { content = (await api.getVersion(state.currentId, qv.turn_id)).content || ""; qv.content = content; } catch { toast("Couldn't load that version.", "error"); return; } }
    row.querySelector(".user-bubble").textContent = content;
    row._qIndex = next; updateQuestionSwitcher(row);
    slot.active_version_index = next;
    if (row._answerEl) {
      const ai = qv.active_answer_index || (qv.answers.length ? qv.answers[qv.answers.length - 1].version_index : 0);
      if (ai) await showAnswerVersion(row._answerEl, qv, ai);
    }
    api.setActiveVersion(state.currentId, { scope: "question", node_id: slot.node_id, version_index: next }).catch(() => {});
  }
  async function switchAnswerVersion(cardEl, delta) {
    const qv = cardEl._qv;
    if (!qv || state.streaming) return;
    const total = qv.answer_total;
    const next = Math.min(total, Math.max(1, cardEl._aIndex + delta));
    if (next === cardEl._aIndex) return;
    await showAnswerVersion(cardEl, qv, next);
    api.setActiveVersion(state.currentId, { scope: "answer", qversion_id: qv.turn_id, version_index: next }).catch(() => {});
  }
  function regenerateAnswer(cardEl) {
    if (state.streaming) { toast("Please wait for the answer to finish."); return; }
    const qvId = cardEl.dataset.qversionId;
    if (!qvId) { toast("Can't regenerate this answer yet."); return; }
    send({ regenQversionId: parseInt(qvId, 10) });
  }
  async function deleteExchange(cardEl) {
    if (state.streaming) { toast("Please wait for the answer to finish."); return; }
    const nodeId = cardEl._nodeId || (precedingUserRow(cardEl) && precedingUserRow(cardEl).dataset.nodeId);
    if (!nodeId) { toast("Can't delete this yet."); return; }
    if (!confirm("Delete this question and all its answers? This cannot be undone.")) return;
    try { await api.deleteNode(state.currentId, nodeId); } catch { toast("Couldn't delete.", "error"); return; }
    await reloadTree();
    toast("Deleted.");
  }

  // ---------- sources drawer ----------
  const SRC_TYPE = { local_pdf: "Paper", web: "Web", github_repo: "GitHub", github_code: "GitHub", online_pdf: "PDF", research_paper: "Research", patent: "Patent" };
  function collectSourceSets() {
    const sets = [];
    thread().querySelectorAll(".ai-card").forEach((el) => {
      if (el._sources && el._sources.length) sets.push({ el, sources: el._sources, question: el._question || questionForAnswer(el) });
    });
    return sets;
  }
  function openSourcesForEl(el) {
    state.srcSets = collectSourceSets();
    let i = state.srcSets.findIndex((s) => s.el === el);
    if (i < 0) { state.srcSets.push({ el, sources: el._sources || [], question: el._question || questionForAnswer(el) }); i = state.srcSets.length - 1; }
    openSourcesAt(i);
  }
  function openSourcesAt(i) {
    const sets = state.srcSets || [];
    if (!sets.length) { state.currentSources = []; renderSources([]); updateSourceNav(); openDrawer(); return; }
    state.srcIndex = Math.max(0, Math.min(sets.length - 1, i));
    const set = sets[state.srcIndex];
    state.currentSources = set.sources;
    renderSources(set.sources); updateSourceNav(); openDrawer();
  }
  function updateSourceNav() {
    const sets = state.srcSets || [];
    const nav = $("drawerNav"); if (!nav) return;
    nav.style.display = sets.length > 1 ? "flex" : "none";
    if (sets.length <= 1) return;
    const i = state.srcIndex || 0;
    $("srcQuestion").textContent = sets[i].question || "This answer";
    $("srcPos").textContent = (i + 1) + " / " + sets.length;
    $("srcPrev").disabled = i <= 0;
    $("srcNext").disabled = i >= sets.length - 1;
  }
  function renderSources(sources) {
    state.currentSources = sources || [];
    const body = $("drawerBody"); if (!body) return;
    $("drawerCount").textContent = String(state.currentSources.length);
    if (!state.currentSources.length) { body.innerHTML = `<div class="dr-empty">No sources for this answer yet.</div>`; return; }
    body.innerHTML = "";
    state.currentSources.forEach((s) => {
      const st = s.source_type || "local_pdf";
      const isLocal = st === "local_pdf";
      const titleInner = esc(prettyName(s.title));
      const pages = s.page_start ? `pp. ${s.page_start}${s.page_end && s.page_end !== s.page_start ? "–" + s.page_end : ""}` : "";
      let meta = [];
      if (isLocal) { if (s.section) meta.push(esc(s.section)); if (pages) meta.push(pages); }
      else {
        if (s.published) meta.push(esc(String(s.published)));
        if (s.file_path) meta.push(esc(s.file_path) + (s.line_start ? ":" + s.line_start : ""));
        if (s.page) meta.push("p." + s.page);
      }
      const score = s.score ? `<span class="src-score">${Math.min(100, Math.round(s.score * 100))}%</span>` : "";
      const text = (s.text || "").trim();
      const card = document.createElement("div");
      card.className = "source" + (s.url ? " clickable" : "");
      card.id = "src-card-" + s.n;
      card.innerHTML = `
        <div class="src-top">
          <span class="src-num ${isLocal ? "pdf" : "web"}">${s.n}</span>
          <span class="src-type">${isLocal ? ICON_DOC : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18"/></svg>'} ${SRC_TYPE[st] || "Source"}</span>
          ${score}
        </div>
        <div class="src-ttl">${titleInner}</div>
        ${meta.length ? `<div class="src-meta">${meta.join(" · ")}</div>` : ""}
        ${text ? `<div class="src-text">${esc(text)}</div>${text.length > 200 ? '<span class="src-more">Show more</span>' : ""}` : ""}
        ${s.url ? `<a class="src-link" href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">Open source <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 17L17 7M7 7h10v10"/></svg></a>` : ""}`;
      const more = card.querySelector(".src-more");
      if (more) more.addEventListener("click", (e) => { e.stopPropagation(); const t = card.querySelector(".src-text"); const ex = t.classList.toggle("expanded"); more.textContent = ex ? "Show less" : "Show more"; });
      if (s.url) card.addEventListener("click", (e) => {
        if (e.target.closest("a") || e.target.closest(".src-more")) return;
        const sel = window.getSelection && window.getSelection();
        if (sel && !sel.isCollapsed && sel.toString().trim()) return;
        window.open(s.url, "_blank", "noopener,noreferrer");
      });
      body.appendChild(card);
    });
  }
  function openDrawer() { const d = $("drawer"); if (d) d.classList.remove("closed"); }
  function closeDrawer() { const d = $("drawer"); if (d) d.classList.add("closed"); }
  function focusSource(n, chip) {
    const card = chip && chip.closest && chip.closest(".ai-card");
    if (card && card._sources && card._sources.length) openSourcesForEl(card);
    else { renderSources(state.currentSources); updateSourceNav(); openDrawer(); }
    const el = $("src-card-" + n);
    if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1200); }
  }

  // ---------- sessions ----------
  function dayStart(sec) { const x = new Date(sec * 1000); x.setHours(0, 0, 0, 0); return x.getTime() / 1000; }
  function bucketLabel(ts) {
    if (!ts) return "Earlier";
    const today = dayStart(Date.now() / 1000), d = dayStart(ts);
    if (d >= today) return "Today";
    if (d >= today - 86400) return "Yesterday";
    if (d >= today - 7 * 86400) return "Previous 7 days";
    return "Earlier";
  }
  function renderSessions() {
    const box = $("history"); box.innerHTML = "";
    if (!state.sessions.length) { box.innerHTML = `<div class="history-empty">No conversations yet. Start one with <b>New chat</b>.</div>`; }
    let lastBucket = null;
    state.sessions.forEach((s) => {
      const b = bucketLabel(s.updated_at || s.created_at);
      if (b !== lastBucket) { lastBucket = b; const lab = document.createElement("div"); lab.className = "grp-label"; lab.textContent = b; box.appendChild(lab); }
      const item = document.createElement("div");
      item.className = "conv" + (s.id === state.currentId ? " active" : "");
      item.innerHTML = `<span class="conv-dot">${ICON_CHAT}</span><span class="conv-name">${esc(s.title || "Untitled")}</span>
        <span class="conv-acts"><button class="mini" data-act="rename" title="Rename">${ICON_PENCIL_MINI}</button><button class="mini danger" data-act="delete" title="Delete">${ICON_TRASH_MINI}</button></span>`;
      item.addEventListener("click", (e) => {
        const act = e.target.closest("[data-act]");
        if (act) { e.stopPropagation(); act.dataset.act === "rename" ? renameSession(s) : deleteSession(s); return; }
        selectSession(s.id);
      });
      box.appendChild(item);
    });
    const cur = state.sessions.find((s) => s.id === state.currentId);
    $("convTitle").textContent = cur ? (cur.title || "Untitled") : "Research workspace";
  }
  async function loadSessions(selectId) {
    state.sessions = await api.sessions();
    if (!state.sessions.length) { await newChat(); return; }
    renderSessions();
    let target = selectId;
    if (!target) { let saved = null; try { saved = localStorage.getItem("ara-session"); } catch {} target = (saved && state.sessions.some((s) => s.id === saved)) ? saved : state.sessions[0].id; }
    await selectSession(target);
  }
  async function selectSession(id) {
    if (state.streaming) return;
    state.currentId = id;
    try { localStorage.setItem("ara-session", id); } catch {}
    renderSessions();
    closeDrawer();
    renderTree(await api.tree(id));
    if (window.innerWidth <= 820) $("sidebar").classList.add("collapsed");
  }
  async function newChat() {
    if (state.streaming) return;
    const s = await api.createSession();
    state.sessions.unshift(s);
    renderSessions();
    await selectSession(s.id);
    $("composerInput").focus();
  }
  async function renameSession(s) {
    const title = prompt("Rename conversation:", s.title || "");
    if (title == null) return;
    await api.renameSession(s.id, title.trim() || "Untitled");
    s.title = title.trim() || "Untitled";
    renderSessions();
  }
  function isUnnamed(s) { return s && (s.title === "New conversation" || !s.title); }
  async function autoTitleSession(s, text) {
    if (!isUnnamed(s) || !(text || "").trim()) return;
    const t = text.trim(); const title = t.length > 48 ? t.slice(0, 48) + "…" : t;
    try { await api.renameSession(s.id, title); } catch { return; }
    s.title = title; renderSessions();
  }
  async function deleteSession(s) {
    if (!confirm(`Delete "${s.title || "this conversation"}"? This cannot be undone.`)) return;
    await api.deleteSession(s.id);
    state.sessions = state.sessions.filter((x) => x.id !== s.id);
    if (s.id === state.currentId) { state.currentId = null; if (state.sessions.length) await selectSession(state.sessions[0].id); else await newChat(); }
    else renderSessions();
  }

  // ---------- sending + streaming ----------
  function setStreaming(on) {
    state.streaming = on;
    const btn = $("sendBtn");
    if (on) { btn.disabled = false; btn.classList.add("stop"); btn.innerHTML = STOP_ICON; btn.setAttribute("aria-label", "Stop"); }
    else { btn.classList.remove("stop"); btn.innerHTML = SEND_ICON; btn.setAttribute("aria-label", "Send"); btn.disabled = !$("composerInput").value.trim(); }
    $("composerInput").disabled = on;
  }
  let _currentModel = "";
  function currentModelName() { return _currentModel; }

  const AGENT_ONLY_EVENTS = new Set([
    "context", "directive", "think", "code", "run", "run_result", "reflect", "blocked", "final",
    "requirements", "task_type", "reference", "tests", "test_validation", "deliverables", "heldout", "gate_fail", "output",
  ]);

  async function send(opts) {
    opts = opts || {};
    const editNodeId = opts.editNodeId || null;
    const regenQversionId = (opts.regenQversionId != null) ? opts.regenQversionId : null;
    const isRegen = regenQversionId != null;
    const isVersionOp = isRegen || !!editNodeId;

    let text = "";
    if (!isRegen) { text = $("composerInput").value.trim(); if (!text) return; }
    if (state.streaming || !state.currentId) return;
    if (!isVersionOp && looksLikeCodingTask(text)) { sendAgent(text); return; }

    const sess = state.sessions.find((s) => s.id === state.currentId);
    const wasEmpty = !isVersionOp && isUnnamed(sess);

    const wel = thread().querySelector(".welcome"); if (wel) wel.remove();
    if (!isRegen) { $("composerInput").value = ""; autosize(); addUserMessage(text); }
    setStreaming(true);
    const h = addAssistantMessage();
    if (isRegen) { const lbl = h.thinkingEl.querySelector(".think-label"); if (lbl) lbl.textContent = "Regenerating…"; }

    const genStart = performance.now();
    const timer = setInterval(() => { if (h.reasonTime) h.reasonTime.textContent = ((performance.now() - genStart) / 1000).toFixed(1) + "s"; }, 100);

    let answer = "", agentHandle = null, doneMeta = null, renderScheduled = false;
    const scheduleRender = () => {
      if (renderScheduled) return; renderScheduled = true;
      requestAnimationFrame(() => {
        renderScheduled = false;
        revealCard(h);
        renderMarkdown(h.md, answer + " ▍"); scrollToBottom();
      });
    };

    const controller = new AbortController(); state.abort = controller;
    const reqBody = { session_id: state.currentId, question: text, mode: state.mode, top_k: state.topk };
    if (editNodeId) reqBody.edit_node_id = editNodeId;
    if (isRegen) reqBody.regen_qversion_id = regenQversionId;
    try {
      const resp = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(reqBody), signal: controller.signal });
      const reader = resp.body.getReader(); const decoder = new TextDecoder(); let buf = "";
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += decoder.decode(value, { stream: true }); let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
          if (!line) continue;
          let ev; try { ev = JSON.parse(line); } catch { continue; }
          if (ev.type === "done") doneMeta = ev;
          if (agentHandle || AGENT_ONLY_EVENTS.has(ev.type)) {
            if (!agentHandle) { agentHandle = makeAgentUI(h.md); if (h.reason) h.reason.remove(); revealCard(h); }
            if (ev.type !== "done") agentHandle(ev);
          } else { handleEvent(ev, h, () => answer, (v) => { answer = v; }, scheduleRender); }
        }
      }
    } catch (err) {
      const msg = err.name === "AbortError" ? "Stopped." : ("Connection error: " + (err.message || ""));
      if (agentHandle) agentHandle({ type: "error", message: msg });
      else {
        if (err.name === "AbortError") answer = (answer || "").trim() + "\n\n_⏹ Stopped._";
        else { toast("Connection error: " + err.message, "error"); if (!answer) answer = "_Something went wrong. Please try again._"; }
        finishReason(h); revealCard(h); renderMarkdown(h.md, answer);
      }
    } finally {
      state.abort = null; clearInterval(timer);
      const secs = (performance.now() - genStart) / 1000;
      finishReason(h);
      if (agentHandle) finalizeActs(h, [], { seconds: secs, model: "agent" });
      else { revealCard(h); renderMarkdown(h.md, answer || "_(no answer)_"); finalizeActs(h, state.currentSources, { seconds: secs, model: currentModelName() }); }
      if (doneMeta) {
        if (doneMeta.qversion_id != null) h.card.dataset.qversionId = String(doneMeta.qversion_id);
        if (doneMeta.node_id) {
          h.card._nodeId = doneMeta.node_id;
          // Tag the question row too so editing it right after sending works without a full reload.
          const urow = precedingUserRow(h.card);
          if (urow && !urow.dataset.nodeId) urow.dataset.nodeId = doneMeta.node_id;
        }
      }
      setStreaming(false); scrollToBottom(); $("composerInput").focus();
      if (wasEmpty) await autoTitleSession(sess, text);
      if (isVersionOp) { try { await reloadTree(); } catch {} }
    }
  }

  async function sendAgent(text) {
    const wel = thread().querySelector(".welcome"); if (wel) wel.remove();
    $("composerInput").value = ""; autosize();
    const sess = state.sessions.find((s) => s.id === state.currentId);
    const wasEmpty = isUnnamed(sess);
    addUserMessage(text);
    setStreaming(true);
    const h = addAssistantMessage();
    if (h.reason) h.reason.remove();
    revealCard(h);
    const genStart = performance.now();
    const handle = makeAgentUI(h.md);
    const controller = new AbortController(); state.abort = controller;
    try {
      const resp = await fetch("/api/agent", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, session_id: state.currentId }), signal: controller.signal });
      const reader = resp.body.getReader(); const decoder = new TextDecoder(); let buf = "";
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += decoder.decode(value, { stream: true }); let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1); if (!line) continue;
          let e; try { e = JSON.parse(line); } catch { continue; }
          handle(e);
        }
      }
    } catch (err) {
      handle({ type: "error", message: err.name === "AbortError" ? "Stopped." : ("Connection error: " + (err.message || "")) });
    } finally {
      state.abort = null;
      finalizeActs(h, [], { seconds: (performance.now() - genStart) / 1000, model: "agent" });
      setStreaming(false); scrollToBottom(); $("composerInput").focus();
      if (wasEmpty) await autoTitleSession(sess, text);
      if (state.currentId) { try { await reloadTree(); } catch {} }
    }
  }

  // ---- agent timeline (coding-task answer): step timeline + IDE code + console + verified footer ----
  const SVG_SPARK = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.4H22l-6 4.5 2.3 7.1L12 16.6 5.7 21l2.3-7.1-6-4.5h7.6z"/></svg>';
  const SVG_X = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>';
  const SVG_CHEV = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg>';
  const SVG_WARN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>';
  const COPY_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';

  function makeAgentUI(root) {
    root.innerHTML = "";
    const panel = document.createElement("div"); panel.className = "agent";
    panel.innerHTML = `<button class="agent-head" type="button">
        <span class="agent-spark">${SVG_SPARK}</span>
        <span class="agent-title">Agent — writing &amp; verifying code</span>
        <span class="agent-attempt" style="display:none"></span>
        <span class="agent-chevron">${SVG_CHEV}</span>
      </button>
      <div class="agent-steps"></div>`;
    root.appendChild(panel);
    panel.querySelector(".agent-head").addEventListener("click", () => panel.classList.toggle("collapsed"));
    const stepsBox = panel.querySelector(".agent-steps");
    const attemptEl = panel.querySelector(".agent-attempt");
    let cur = null, runStep = null, consoleShown = false, lastOutput = "";

    const nodeHTML = (s) => s === "running" ? '<span class="spin"></span>' : (s === "fail" ? SVG_X : (s === "pending" ? "" : ICON_CHECK));
    const setStatus = (step, s) => { step.className = "astep " + s; step.querySelector(".anode").innerHTML = nodeHTML(s); };
    const addStep = (label, status, detail) => {
      if (cur && cur.classList.contains("running")) setStatus(cur, "done");
      const step = document.createElement("div"); step.className = "astep " + (status || "done");
      step.innerHTML = `<div class="anode">${nodeHTML(status || "done")}</div><div class="astep-main"><div class="alabel"></div><div class="adetail"></div></div>`;
      step.querySelector(".alabel").textContent = label;
      if (detail) step.querySelector(".adetail").textContent = detail;
      stepsBox.appendChild(step); cur = step; scrollToBottom(); return step;
    };
    const codeBlock = (code, file, lang) => {
      const wrap = document.createElement("div"); wrap.className = "code";
      wrap.innerHTML = `<div class="code-head"><span class="win-dots"><i class="d-r"></i><i class="d-y"></i><i class="d-g"></i></span><span class="code-file">${esc(file || "solution.py")}</span><span class="code-badge">${esc(lang || "python")}</span><button class="code-copy" type="button">${COPY_SVG} Copy</button></div><pre class="code-body"><code class="language-${esc(lang || "python")}"></code></pre>`;
      const codeEl = wrap.querySelector("code"); codeEl.textContent = code || "";
      if (window.hljs) { try { hljs.highlightElement(codeEl); } catch (e) {} }
      const copy = wrap.querySelector(".code-copy");
      copy.addEventListener("click", () => { navigator.clipboard.writeText(code || "").then(() => { copy.innerHTML = ICON_CHECK + " Copied"; setTimeout(() => (copy.innerHTML = COPY_SVG + " Copy"), 1300); }).catch(() => {}); });
      root.appendChild(wrap); scrollToBottom();
    };
    const consoleBlock = (stdout, stderr, ok) => {
      consoleShown = true;
      const wrap = document.createElement("div"); wrap.className = "console";
      wrap.innerHTML = `<div class="console-head"><span class="console-dot"></span><span class="console-dot"></span><span class="console-dot"></span><span class="console-title">SANDBOX OUTPUT</span></div>`;
      const body = document.createElement("pre"); body.className = "console-body";
      if (stdout) { const s = document.createElement("span"); s.textContent = stdout; body.appendChild(s); }
      if (stderr) { const s = document.createElement("span"); s.className = "fail"; s.textContent = (stdout ? "\n" : "") + stderr; body.appendChild(s); }
      if (!stdout && !stderr) body.textContent = ok ? "(no output)" : "(failed)";
      wrap.appendChild(body); root.appendChild(wrap); scrollToBottom();
    };
    const codeFoot = (success) => {
      const foot = document.createElement("div"); foot.className = "code-foot";
      const tag = document.createElement("span"); tag.className = "status-tag " + (success ? "verified" : "partial");
      tag.innerHTML = (success ? ICON_CHECK : SVG_WARN) + " " + (success ? "Verified" : "Best attempt");
      foot.appendChild(tag);
      const again = document.createElement("button"); again.className = "run-again"; again.type = "button";
      again.innerHTML = ICON_REGEN + " Run again";
      again.addEventListener("click", () => { const card = root.closest(".ai-card"); if (card) regenerateAnswer(card); });
      foot.appendChild(again);
      root.appendChild(foot); scrollToBottom();
    };

    return function handle(e) {
      switch (e.type) {
        case "status": addStep(e.message || "Working…", "done"); break;
        case "context": if (e.chars) addStep("Gathered background (" + e.chars + " chars)", "done"); break;
        case "warning": addStep("⚠ " + (e.message || ""), "done"); break;
        case "directive": addStep("🧭 " + (e.text || ""), "done"); break;
        case "requirements": addStep("Read the requirements", "done", (e.text || "").slice(0, 600)); break;
        case "task_type": addStep("Verification mode: " + (e.task_type || ""), "done"); break;
        case "reference": addStep(e.scope === "validation" ? "Built an independent validation oracle" : "Built a reference oracle", "done"); break;
        case "tests": addStep("Wrote " + (e.count || 0) + " correctness test" + (e.count === 1 ? "" : "s"), "done"); if (e.code) codeBlock(e.code, "tests.py", "python"); break;
        case "test_validation": if (e.message) addStep("⚠ " + e.message, "done"); break;
        case "deliverables": if (e.items && e.items.length) addStep("Deliverables: " + e.items.map(String).join(", "), "done"); break;
        case "heldout": addStep("Verifying against " + (e.count || 0) + " held-out check" + (e.count === 1 ? "" : "s") + (e.strict ? " (strict)" : ""), "done"); break;
        case "think": attemptEl.style.display = ""; attemptEl.textContent = "Attempt " + e.iteration; break;
        case "code": addStep("Wrote the program", "done"); codeBlock(e.code, "solution.py", "python"); break;
        case "run": runStep = addStep("Running in the sandbox", "running"); break;
        case "run_result": {
          const s = runStep || addStep("Ran in the sandbox", "done");
          setStatus(s, e.ok ? "done" : "fail");
          s.querySelector(".alabel").textContent = e.ok ? "Ran successfully" : "Run failed";
          if (e.summary) s.querySelector(".adetail").textContent = e.summary;
          const tail = (e.stderr || "").split("\n").slice(-12).join("\n");
          consoleBlock(e.stdout || "", e.ok ? "" : (tail || e.error || ""), e.ok);
          runStep = null; break;
        }
        case "reflect": { const v = e.verdict || {}; addStep(v.done ? "Reviewed — good to go" : "Reviewed — needs another pass", "done", v.feedback || (v.score != null ? "score " + v.score : "")); break; }
        case "gate_fail": addStep("Output gate failed — refining", "fail", e.reason || ""); break;
        case "output": lastOutput = e.text || ""; break;
        case "blocked": addStep("Blocked by policy", "fail", e.reason || ""); break;
        case "error": addStep("Error", "fail", e.message || ""); break;
        case "final": {
          if (cur && cur.classList.contains("running")) setStatus(cur, e.success ? "done" : "fail");
          if (e.code) codeBlock(e.code, "solution.py", "python");
          if (!consoleShown && (e.output || lastOutput)) consoleBlock(e.output || lastOutput, "", true);
          if (e.answer) { const a = document.createElement("div"); a.style.cssText = "margin:12px 0 2px;font-size:14px;line-height:1.6;color:var(--text-strong);white-space:pre-wrap"; a.textContent = e.answer; root.appendChild(a); }
          codeFoot(!!e.success);
          panel.querySelector(".agent-title").textContent = e.success ? "Agent — solved & verified" : "Agent — best attempt";
          break;
        }
        default: break;
      }
      scrollToBottom();
    };
  }

  function handleEvent(ev, h, getAns, setAns, scheduleRender) {
    switch (ev.type) {
      case "status": appendProcess(h, ev.message || "Working…"); break;
      case "thinking": appendThinking(h, ev.text || ""); break;
      case "sanity": finishReason(h); revealCard(h); renderMarkdown(h.md, "⚠️ " + (ev.message || "Please rephrase your question.")); break;
      case "sources": {
        const list = ev.sources || []; state.currentSources = list; h.card._sources = list;
        if (list.length) reasonChip(h, `Found ${list.length} relevant source${list.length > 1 ? "s" : ""}`, () => openSourcesForEl(h.card));
        break;
      }
      case "grade": h._grade = ev.grade || ""; h._gradeLabel = ev.label || ""; h._gradeMsg = ev.message || ""; if (h._gradeLabel) reasonChip(h, h._gradeLabel); break;
      case "token": setAns(getAns() + (ev.text || "")); scheduleRender(); break;
      case "warning": toast(ev.message || "Heads up", "warn"); break;
      case "error": toast(ev.message || "Error", "error"); setAns(getAns() + "\n\n_" + (ev.message || "error") + "_"); scheduleRender(); break;
      case "low_confidence": {
        if (h._lowConf) break; h._lowConf = true;
        const w = document.createElement("div"); w.className = "low-conf";
        w.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/></svg>';
        const span = document.createElement("span"); span.textContent = ev.message || "Low confidence."; w.appendChild(span);
        h.md.insertAdjacentElement("afterend", w); break;
      }
      case "done": if (ev.cached) { h._cached = true; h._cachedPct = ev.similarity || 0; } break;
    }
  }

  // ---------- composer ----------
  function autosize() { const t = $("composerInput"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; }
  function nearBottom() { const tr = $("thread"); return tr.scrollHeight - tr.scrollTop - tr.clientHeight < 120; }
  function updateToBottomBtn() { const tr = $("thread"); const show = tr.scrollHeight - tr.scrollTop - tr.clientHeight > 220 && !!thread().querySelector(".row"); $("toBottom").classList.toggle("show", show); }
  function scrollToBottom(force) { const tr = $("thread"); if (force) state.autoStick = true; if (force || state.autoStick) tr.scrollTop = tr.scrollHeight; updateToBottomBtn(); }

  // ---------- library count ----------
  async function loadLibrary() {
    // Show the DB-indexed paper count (the same source as the library modal's list) — NOT the
    // number of PDF files on disk, which can differ when the DB is empty/unreachable.
    try {
      const lib = await api.library();
      const n = lib.papers;
      if (n == null) { $("paperCount").textContent = "—"; $("libWord").textContent = "library unavailable"; return; }
      $("paperCount").textContent = String(n);
      $("libWord").textContent = n === 1 ? "paper indexed" : "papers indexed";
    } catch { $("paperCount").textContent = "—"; $("libWord").textContent = "library unavailable"; }
  }

  // ---------- library modal ----------
  function openLibrary() {
    $("libOverlay").classList.add("open");
    const list = $("libList"); list.innerHTML = `<div class="lib-empty">Loading…</div>`;
    Promise.all([api.papers().catch(() => []), api.library().catch(() => ({}))])
      .then(([items, stats]) => renderLibrary(items, stats))
      .catch(() => { list.innerHTML = `<div class="lib-empty">Couldn't load your library.</div>`; });
  }
  function closeLibrary() { $("libOverlay").classList.remove("open"); }
  function renderLibrary(items, stats) {
    items = items || []; stats = stats || {};
    const onDisk = stats.pdfs || 0;
    $("libSub").textContent = items.length + (items.length === 1 ? " paper" : " papers")
      + (onDisk > items.length ? ` · ${onDisk} PDF${onDisk === 1 ? "" : "s"} on disk` : "");
    const list = $("libList"); list.innerHTML = "";
    if (!items.length) {
      // Files can sit in data/papers/ on disk yet not be indexed (DB offline / indexing unfinished).
      if (onDisk > 0) {
        list.innerHTML = `<div class="lib-empty">None of your papers are indexed yet, but <b>${onDisk} PDF${onDisk === 1 ? "" : "s"}</b> ${onDisk === 1 ? "is" : "are"} saved on disk in <code>data/papers/</code>.<br><br>They're not searchable until they're embedded into the database. If the database is offline, start it and re-upload (or run the indexer) — your files aren't lost.</div>`;
      } else {
        list.innerHTML = `<div class="lib-empty">No papers yet. Click <b>Add papers</b> to upload PDFs.</div>`;
      }
      return;
    }
    const nIncomplete = items.filter((p) => p.incomplete).length;
    if (nIncomplete) {
      const banner = document.createElement("div"); banner.className = "lib-banner";
      banner.innerHTML = `<span>⚠ ${nIncomplete} paper${nIncomplete === 1 ? "" : "s"} half-done (parsed, not embedded).</span>`;
      const rm = document.createElement("button"); rm.textContent = "Remove half-done";
      rm.addEventListener("click", async () => {
        if (!confirm(`Remove ${nIncomplete} half-done paper${nIncomplete === 1 ? "" : "s"}? Their PDFs are deleted so you can re-upload.`)) return;
        rm.disabled = true; rm.textContent = "Removing…";
        try { const res = await api.removeIncomplete(); if (res.error) { toast(res.error, "error"); return; } toast(`Removed ${res.count} half-done paper${res.count === 1 ? "" : "s"}.`); loadLibrary(); const [it, st] = await Promise.all([api.papers().catch(() => []), api.library().catch(() => ({}))]); renderLibrary(it, st); }
        catch { toast("Remove failed.", "error"); }
      });
      banner.appendChild(rm); list.appendChild(banner);
    }
    items.forEach((p) => {
      const row = document.createElement("div"); row.className = "lib-row" + (p.incomplete ? " incomplete" : "");
      row.innerHTML = `<div class="lib-ic">${ICON_DOC}</div>
        <div class="lib-main"><div class="lib-name">${esc(prettyName(p.title))}</div>
        <div class="lib-meta">${p.chunks} chunk${p.chunks === 1 ? "" : "s"}${p.incomplete ? ' · <span class="lib-warn">not embedded</span>' : ""}</div></div>
        <div class="lib-acts"><button class="lib-mini danger" title="Delete">${ICON_TRASH_MINI}</button></div>`;
      row.querySelector(".lib-mini").addEventListener("click", async () => {
        if (!confirm(`Delete "${prettyName(p.title)}"? This removes the PDF, its chunks, and embeddings — permanently.`)) return;
        row.classList.add("removing");
        try { const res = await api.deletePaper(p.id); if (res.error) { toast(res.error, "error"); row.classList.remove("removing"); return; } setTimeout(() => row.remove(), 220); const remaining = items.filter((x) => x.id !== p.id); $("libSub").textContent = remaining.length + (remaining.length === 1 ? " paper" : " papers"); loadLibrary(); toast("Paper deleted."); }
        catch { toast("Delete failed.", "error"); row.classList.remove("removing"); }
      });
      list.appendChild(row);
    });
  }

  // ---------- upload + ingest ----------
  const isPdf = (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
  const fmtSize = (n) => n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB";
  function openUpload() {
    if (state.ingesting) { $("uploadOverlay").classList.add("open"); return; }
    $("upList").innerHTML = ""; $("upLog").classList.remove("show"); $("upLog").textContent = "";
    $("upSummary").textContent = "Drop or browse to add PDFs.";
    $("upDone").disabled = false;
    $("uploadOverlay").classList.add("open");
  }
  function closeUpload() {
    if (state.ingesting) { if (!confirm("Cancel this upload? Its partial data will be removed.")) return; cancelIngest(); }
    $("uploadOverlay").classList.remove("open");
  }
  function upRow(name, size) {
    const row = document.createElement("div"); row.className = "up-file";
    row.innerHTML = `<div class="up-ic">${ICON_DOC}</div>
      <div class="up-main"><div class="up-row1"><span class="up-name">${esc(name)}</span><span class="up-size">${size ? fmtSize(size) : ""}</span></div>
      <div class="up-bar"><div class="up-fill"></div></div><div class="up-status">Queued…</div></div>
      <div class="up-check">${ICON_CHECK}</div>`;
    $("upList").appendChild(row); return row;
  }
  function upLog(line, cls) { const log = $("upLog"); log.classList.add("show"); const s = document.createElement("span"); if (cls) s.className = cls; s.textContent = line + "\n"; log.appendChild(s); log.scrollTop = log.scrollHeight; }

  async function handleFiles(files) {
    files = Array.from(files || []).filter(Boolean); if (!files.length) return;
    if (state.ingesting) { toast("Already indexing — please wait."); return; }
    const pdfs = files.filter(isPdf); let badCount = files.length - pdfs.length;
    $("uploadOverlay").classList.add("open");
    if (!pdfs.length) { toast("Please choose PDF files.", "error"); return; }
    $("upDone").disabled = true; $("addPapersBtn").classList.add("busy");
    $("upSummary").textContent = "Uploading…";
    // show a row per chosen PDF
    const rows = new Map(); pdfs.forEach((f) => rows.set(f.name, upRow(f.name, f.size)));
    let saved = [], dups = [];
    try {
      const res = await api.upload(pdfs);
      (res.results || []).forEach((r) => {
        const row = rows.get(r.filename) || rows.get(r.name);
        if (r.status === "saved") { saved.push(r.filename); if (row) row.querySelector(".up-status").textContent = "Indexing…"; }
        else if (r.status === "duplicate") { dups.push(r.filename || r.name); if (row) { row.classList.add("done"); row.querySelector(".up-status").textContent = "Already indexed"; } }
        else { badCount += 1; if (row) { row.classList.add("err"); row.querySelector(".up-status").textContent = "Failed to save"; } }
      });
    } catch { badCount += pdfs.length; rows.forEach((row) => { row.classList.add("err"); row.querySelector(".up-status").textContent = "Upload failed"; }); }
    $("addPapersBtn").classList.remove("busy");
    if (dups.length) toast(`${dups.length} already indexed — skipped.`);
    if (!saved.length) { $("upSummary").textContent = badCount ? `${badCount} file${badCount > 1 ? "s" : ""} couldn't be added.` : "Nothing new to index."; $("upDone").disabled = false; return; }
    await runIngest(saved, rows);
  }

  async function runIngest(saved, rows) {
    state.ingesting = true; $("addPapersBtn").classList.add("busy");
    const ac = new AbortController(); state.ingestAbort = ac;
    $("upSummary").textContent = `Indexing ${saved.length} paper${saved.length > 1 ? "s" : ""}…`;
    $("upDone").disabled = true;
    upLog(`→ Saved ${saved.length} file(s). Indexing now — the first paper also warms up the models.`, "stage");
    let ok = true;
    try {
      const resp = await fetch("/api/ingest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filenames: saved }), signal: ac.signal });
      const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = "";
      while (true) {
        const { done, value } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true }); let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1); if (!line) continue;
          let ev; try { ev = JSON.parse(line); } catch { continue; }
          if (ev.type === "stage") { $("upSummary").textContent = ev.label; upLog("◆ " + ev.label, "stage"); }
          else if (ev.type === "log") { upLog(ev.line, /WARNING|NOT indexed/i.test(ev.line) ? "warn" : null); }
          else if (ev.type === "error") { ok = false; upLog("✗ " + ev.message, "warn"); }
          else if (ev.type === "cancelled") { upLog("✕ " + ev.message, "warn"); finishIngest(false, rows, true); return; }
          else if (ev.type === "done") { upLog("✓ " + ev.message, "ok"); }
        }
      }
      finishIngest(ok, rows);
    } catch (err) {
      if (err && err.name === "AbortError") return;
      upLog("✗ " + err.message, "warn"); finishIngest(false, rows);
    }
  }
  function finishIngest(ok, rows, cancelled) {
    state.ingesting = false; state.ingestAbort = null; $("addPapersBtn").classList.remove("busy");
    if (rows) rows.forEach((row) => { if (!row.classList.contains("err") && !row.classList.contains("done")) { row.classList.add(ok && !cancelled ? "done" : "err"); row.querySelector(".up-status").textContent = cancelled ? "Cancelled" : (ok ? "Indexed ✓" : "Not indexed"); } });
    $("upSummary").textContent = cancelled ? "Cancelled — partial data removed." : (ok ? "Done — added to your library." : "Indexing finished with warnings (see log).");
    $("upDone").disabled = false;
    loadLibrary();
  }
  async function cancelIngest() {
    if (!state.ingesting) return;
    upLog("✕ Cancelling — removing this upload…", "warn");
    if (state.ingestAbort) { try { state.ingestAbort.abort(); } catch {} }
    try { await api.cancelIngest(); } catch {}
    state.ingesting = false; state.ingestAbort = null; $("addPapersBtn").classList.remove("busy");
    toast("Upload cancelled — its data was removed.");
    loadLibrary();
  }

  // ---------- model picker ----------
  function closeModelMenu() { const m = $("model"); if (m) m.classList.remove("open"); const b = $("modelBtn"); if (b) b.setAttribute("aria-expanded", "false"); }
  function toggleModelMenu() { const m = $("model"); if (!m) return; const open = m.classList.toggle("open"); $("modelBtn").setAttribute("aria-expanded", open ? "true" : "false"); }
  async function loadModels() {
    try {
      const data = await api.models();
      const menu = $("modelMenu"); menu.innerHTML = "";
      const cur = data.current || {}; let curLabel = "", curVendor = "";
      (data.options || []).forEach((o) => {
        if (o.model === cur.model) { curLabel = o.name || o.label || o.model; curVendor = o.vendor || o.label || ""; }
        const row = document.createElement("div");
        row.className = "model-opt" + (o.model === cur.model ? " sel" : "") + (o.available ? "" : " na");
        row.setAttribute("role", "option");
        row.innerHTML = `<span>${esc(o.name || o.model)}${o.vendor ? `<small>${esc(o.vendor)}</small>` : ""}</span><svg class="ck" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M20 6L9 17l-5-5"/></svg>`;
        row.addEventListener("click", () => selectModel(o));
        menu.appendChild(row);
      });
      _currentModel = cur.model || "";
      setModelText(curLabel || (cur.provider + " · " + cur.model), curVendor || cur.provider || "");
    } catch { setModelText("unavailable", ""); }
  }
  function setModelText(name, sub) { $("modelText").innerHTML = esc(name) + (sub ? `<small>${esc(sub)}</small>` : ""); }
  async function selectModel(o) {
    closeModelMenu();
    try {
      const res = await api.setModel(o.provider, o.model);
      if (res.error) { toast(res.error, "error"); return; }
      _currentModel = o.model;
      setModelText(o.name || o.model, o.vendor || o.label || o.provider || "");
      $("modelMenu").querySelectorAll(".model-opt").forEach((el) => el.classList.remove("sel"));
      toast("Model switched to " + (res.model || o.model));
      loadModels();
    } catch { toast("Could not switch model.", "error"); }
  }

  // ---------- theme + collapse ----------
  function applyThemeFromStorage() { try { document.documentElement.classList.toggle("light", localStorage.getItem("ara-theme") === "light"); } catch {} }
  function toggleTheme() {
    const light = document.documentElement.classList.toggle("light");
    try { localStorage.setItem("ara-theme", light ? "light" : "dark"); } catch {}
  }
  function applyCollapseFromStorage() { try { if (localStorage.getItem("ara-sidebar") === "collapsed") $("sidebar").classList.add("collapsed"); } catch {} }
  function toggleCollapse() {
    const c = $("sidebar").classList.toggle("collapsed");
    try { localStorage.setItem("ara-sidebar", c ? "collapsed" : "open"); } catch {}
  }

  // ---------- init ----------
  async function init() {
    applyThemeFromStorage();
    applyCollapseFromStorage();
    try { state.cfg = await api.config(); } catch {}

    try {
      const me = await api.me();
      if (me && me.auth) {
        if (!me.user_id) { window.location.href = "/login"; return; }
        const uname = (me.user_id || "").trim();
        $("acctName").textContent = uname ? uname.charAt(0).toUpperCase() + uname.slice(1) : "Account";
        const initials = (uname.replace(/[^a-zA-Z0-9]/g, " ").trim().split(/\s+/).map((w) => w[0]).join("").slice(0, 2) || "U").toUpperCase();
        $("avatar").firstChild.textContent = initials;
        const lo = $("logoutBtn"); lo.style.display = ""; lo.addEventListener("click", async () => { try { await api.logout(); } catch {} window.location.href = "/login"; });
      } else { $("acctName").textContent = "Guest"; $("avatar").firstChild.textContent = "G"; }
    } catch {}

    if (state.cfg.local_rag_enabled) { loadLibrary(); }
    else { ["addPapersBtn", "libBtn"].forEach((id) => { const el = $(id); if (el) el.style.display = "none"; }); }

    loadModels();
    await loadSessions();

    // events
    $("newChatBtn").addEventListener("click", newChat);
    $("collapseBtn").addEventListener("click", toggleCollapse);
    $("menuToggle").addEventListener("click", toggleCollapse);
    $("themeBtn").addEventListener("click", toggleTheme);
    $("themeBtn").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleTheme(); } });

    $("addPapersBtn").addEventListener("click", openUpload);
    $("libBtn").addEventListener("click", openLibrary);
    $("libClose").addEventListener("click", closeLibrary);
    $("libOverlay").addEventListener("click", (e) => { if (e.target === $("libOverlay")) closeLibrary(); });
    $("pdfInput").addEventListener("change", (e) => { handleFiles(e.target.files); e.target.value = ""; });
    $("dropzone").addEventListener("click", () => $("pdfInput").click());
    $("dropzone").addEventListener("dragover", (e) => { e.preventDefault(); $("dropzone").classList.add("drag"); });
    $("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag"));
    $("dropzone").addEventListener("drop", (e) => { e.preventDefault(); $("dropzone").classList.remove("drag"); handleFiles(e.dataTransfer.files); });
    $("uploadClose").addEventListener("click", closeUpload);
    $("upDone").addEventListener("click", () => { if (!state.ingesting) $("uploadOverlay").classList.remove("open"); });

    $("modelBtn").addEventListener("click", (e) => { e.stopPropagation(); toggleModelMenu(); });
    document.addEventListener("click", (e) => { if (!e.target.closest("#model")) closeModelMenu(); });

    $("srcToggle").addEventListener("click", () => { const sets = collectSourceSets(); if (!sets.length) { toast("No sources for this conversation yet."); return; } openSourcesForEl(sets[sets.length - 1].el); });
    $("drawerClose").addEventListener("click", closeDrawer);
    $("srcPrev").addEventListener("click", () => openSourcesAt((state.srcIndex || 0) - 1));
    $("srcNext").addEventListener("click", () => openSourcesAt((state.srcIndex || 0) + 1));

    // mode segment (Fast / Deep)
    try { if (localStorage.getItem("ara-mode") === "deep") state.mode = "deep"; } catch {}
    const seg = $("modeSeg");
    const paintSeg = () => seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.mode === state.mode));
    paintSeg();
    seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; try { localStorage.setItem("ara-mode", state.mode); } catch {} paintSeg(); }));

    const input = $("composerInput");
    input.addEventListener("input", () => { autosize(); $("sendBtn").disabled = state.streaming || !input.value.trim(); });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
    $("sendBtn").addEventListener("click", () => { if (state.streaming) { if (state.abort) state.abort.abort(); } else send(); });
    $("thread").addEventListener("scroll", () => { state.autoStick = nearBottom(); updateToBottomBtn(); });
    $("toBottom").addEventListener("click", () => scrollToBottom(true));

    // subtle 3D tilt on answer cards (skipped for reduced-motion)
    if (!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) {
      const thr = $("thread");
      thr.addEventListener("mousemove", (e) => {
        const card = e.target.closest(".ai-card"); if (!card) return;
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - 0.5, py = (e.clientY - r.top) / r.height - 0.5;
        card.classList.add("tilting");
        card.style.transform = `perspective(1200px) rotateY(${px * 2}deg) rotateX(${-py * 2}deg)`;
      });
      thr.addEventListener("mouseout", (e) => {
        const card = e.target.closest(".ai-card"); if (!card || (e.relatedTarget && card.contains(e.relatedTarget))) return;
        card.classList.remove("tilting"); card.style.transform = "";
      });
    }

    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
      if (e.key === "Escape") { closeModelMenu(); closeLibrary(); if (!state.ingesting) $("uploadOverlay").classList.remove("open"); closeDrawer(); }
    });
  }

  init();
})();
