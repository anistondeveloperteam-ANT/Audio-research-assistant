/* Shared visual engine for the auth pages (login + reset): a drifting gradient (CSS), a 2D
 * node-network canvas with travelling pulses + mouse parallax, and a Three.js wireframe torus-knot,
 * plus the monochrome card tilt and the dark/light theme toggle (persisted). Uses the global THREE
 * (r128 UMD). Degrades gracefully if WebGL/THREE is unavailable. No external assets. */
(function () {
  var $ = function (s) { return document.querySelector(s); };
  var root = document.documentElement;

  /* ---- theme (persisted under ara-theme; default dark) ---- */
  try { if (localStorage.getItem("ara-theme") === "light") root.classList.add("light"); } catch (e) {}
  function css(v) { return getComputedStyle(root).getPropertyValue(v).trim(); }

  /* ---- shared mouse state ---- */
  var mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };
  window.addEventListener("mousemove", function (e) {
    mouse.tx = e.clientX / window.innerWidth; mouse.ty = e.clientY / window.innerHeight;
  });
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- 1. node-network (2D canvas) ---- */
  var canvas = $("#network"), ctx = canvas && canvas.getContext("2d");
  var W, H, nodes = [], pulses = [], C = {};
  var NODE_COUNT = 64, LINK;
  function readColors() { C = { node: css("--net-node"), line: css("--net-line"), pulse: css("--net-pulse") }; }
  function resize() {
    if (!canvas) return;
    W = canvas.width = window.innerWidth * devicePixelRatio;
    H = canvas.height = window.innerHeight * devicePixelRatio;
    canvas.style.width = window.innerWidth + "px"; canvas.style.height = window.innerHeight + "px";
    LINK = 150 * devicePixelRatio;
  }
  function initNodes() {
    nodes = [];
    for (var i = 0; i < NODE_COUNT; i++) nodes.push({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.18 * devicePixelRatio, vy: (Math.random() - 0.5) * 0.18 * devicePixelRatio,
      r: (Math.random() * 1.4 + 0.8) * devicePixelRatio
    });
  }
  function spawnPulse() {
    var a = nodes[(Math.random() * nodes.length) | 0], b = null, best = LINK;
    for (var k = 0; k < nodes.length; k++) {
      var n = nodes[k]; if (n === a) continue;
      var d = Math.hypot(n.x - a.x, n.y - a.y); if (d < best) { best = d; b = n; }
    }
    if (b) pulses.push({ a: a, b: b, t: 0, speed: 0.006 + Math.random() * 0.01 });
  }
  function draw() {
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    var px = (mouse.x - 0.5) * 40 * devicePixelRatio, py = (mouse.y - 0.5) * 40 * devicePixelRatio;
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i]; n.x += n.vx; n.y += n.vy;
      if (n.x < 0) n.x = W; if (n.x > W) n.x = 0; if (n.y < 0) n.y = H; if (n.y > H) n.y = 0;
    }
    ctx.lineWidth = 1 * devicePixelRatio;
    for (var a = 0; a < nodes.length; a++) {
      for (var b = a + 1; b < nodes.length; b++) {
        var p = nodes[a], q = nodes[b], d = Math.hypot(p.x - q.x, p.y - q.y);
        if (d < LINK) {
          var o = (1 - d / LINK), m = C.line.match(/[\d.]+\)$/);
          ctx.strokeStyle = m ? C.line.replace(/[\d.]+\)$/, (parseFloat(m) * o).toFixed(3) + ")") : C.line;
          ctx.beginPath(); ctx.moveTo(p.x + px * o, p.y + py * o); ctx.lineTo(q.x + px * o, q.y + py * o); ctx.stroke();
        }
      }
    }
    ctx.fillStyle = C.node;
    for (var j = 0; j < nodes.length; j++) {
      var nd = nodes[j]; ctx.beginPath(); ctx.arc(nd.x + px * 0.6, nd.y + py * 0.6, nd.r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = C.pulse;
    for (var u = pulses.length - 1; u >= 0; u--) {
      var pl = pulses[u]; pl.t += pl.speed;
      if (pl.t >= 1) { pulses.splice(u, 1); continue; }
      var x = pl.a.x + (pl.b.x - pl.a.x) * pl.t + px * 0.6, y = pl.a.y + (pl.b.y - pl.a.y) * pl.t + py * 0.6;
      ctx.beginPath(); ctx.arc(x, y, 2.1 * devicePixelRatio, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 0.25; ctx.beginPath(); ctx.arc(x, y, 5 * devicePixelRatio, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
    }
    if (pulses.length < 14 && Math.random() < 0.08) spawnPulse();
  }

  /* ---- 2. Three.js wireframe torus-knot ---- */
  var renderer, scene, camera, knot, wireMat, hasThree = (typeof THREE !== "undefined");
  function initThree() {
    if (!hasThree) return;
    var el = $("#three"); if (!el) return;
    try { renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true }); }
    catch (e) { hasThree = false; return; }
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.z = 30;
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    var geo = new THREE.TorusKnotGeometry(8.4, 2.4, 140, 12, 2, 3);
    wireMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.16 });
    knot = new THREE.Mesh(geo, wireMat); scene.add(knot);
  }
  function applyWireTheme() {
    if (!wireMat) return;
    var light = root.classList.contains("light");
    wireMat.color.set(light ? 0x111111 : 0xffffff);
    wireMat.opacity = light ? 0.10 : 0.16;
  }

  /* ---- master loop ---- */
  function loop() {
    mouse.x += (mouse.tx - mouse.x) * 0.05; mouse.y += (mouse.ty - mouse.y) * 0.05;
    if (!reduceMotion) draw();
    if (knot && renderer) {
      knot.rotation.x += 0.0016; knot.rotation.y += 0.0022;
      knot.rotation.z = (mouse.x - 0.5) * 0.4;
      knot.position.x = (mouse.x - 0.5) * 6; knot.position.y = -(mouse.y - 0.5) * 6;
      renderer.render(scene, camera);
    }
    requestAnimationFrame(loop);
  }

  /* ---- 3. card tilt ---- */
  var card = $("#card");
  function tiltLoop() {
    if (card) {
      var rx = (mouse.y - 0.5) * -10, ry = (mouse.x - 0.5) * 10;
      card.style.transform = "rotateX(" + rx + "deg) rotateY(" + ry + "deg)";
    }
    requestAnimationFrame(tiltLoop);
  }

  /* ---- 4. theme toggle ---- */
  var themeBtn = $("#themeBtn");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    root.classList.toggle("light");
    try { localStorage.setItem("ara-theme", root.classList.contains("light") ? "light" : "dark"); } catch (e) {}
    readColors(); applyWireTheme();
  });

  /* ---- boot ---- */
  resize(); readColors(); initNodes(); initThree(); applyWireTheme();
  loop(); if (!reduceMotion) tiltLoop();
  window.addEventListener("resize", function () {
    resize(); initNodes();
    if (renderer) { camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }
  });
})();
