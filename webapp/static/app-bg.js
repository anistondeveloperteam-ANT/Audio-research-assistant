/* Neural-network background for the workspace. Prefers a 3D Three.js connectome (global THREE,
 * r128 UMD); if THREE didn't load or WebGL is unavailable, it falls back to a self-contained
 * Canvas-2D neural network so a background ALWAYS renders (no blank dark screen). Reads the theme
 * from the documentElement `light` class and re-themes on toggle. No external assets. */
(function () {
  var host = document.getElementById("three");
  if (!host) return;

  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var mouse = { x: .5, y: .5, tx: .5, ty: .5 };
  addEventListener("mousemove", function (e) { mouse.tx = e.clientX / innerWidth; mouse.ty = e.clientY / innerHeight; });
  function isLight() { return document.documentElement.classList.contains("light"); }

  /* ------------------------------------------------------------------ 3D (Three.js) */
  var renderer, scene, camera, group, nodePoints, lineSeg, pulsePoints, spinT = 0;
  var NODES = [], EDGES = [], PULSES = [];
  var NODE_N = innerWidth < 760 ? 130 : 210;

  function makeDot() {
    var c = document.createElement("canvas"); c.width = c.height = 64;
    var g = c.getContext("2d"), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, "rgba(255,255,255,1)");
    grd.addColorStop(.25, "rgba(255,255,255,.85)");
    grd.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grd; g.beginPath(); g.arc(32, 32, 32, 0, Math.PI * 2); g.fill();
    return new THREE.CanvasTexture(c);
  }
  var rnd = function () { return Math.random() * 2 - 1; };

  function build3D() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(62, innerWidth / innerHeight, 0.1, 500);
    camera.position.z = 120;
    try { renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true }); }
    catch (e) { return false; }
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    host.appendChild(renderer.domElement);
    group = new THREE.Group(); scene.add(group);

    NODES = [];
    for (var i = 0; i < NODE_N; i++) {
      var v = new THREE.Vector3(rnd(), rnd(), rnd());
      v.normalize().multiplyScalar(Math.pow(Math.random(), .65) * 108);
      v.x *= 1.3; v.y *= 0.82;
      NODES.push(v);
    }
    var k = 3;
    for (var a = 0; a < NODE_N; a++) {
      var d = [];
      for (var j = 0; j < NODE_N; j++) { if (a !== j) d.push({ j: j, dist: NODES[a].distanceTo(NODES[j]) }); }
      d.sort(function (x, y) { return x.dist - y.dist; });
      for (var n = 0; n < k; n++) EDGES.push([a, d[n].j]);
    }
    var seen = {};
    EDGES = EDGES.filter(function (e) { var key = e[0] < e[1] ? e[0] + "_" + e[1] : e[1] + "_" + e[0]; if (seen[key]) return false; seen[key] = 1; return true; });

    var dot = makeDot();
    var ng = new THREE.BufferGeometry(), np = new Float32Array(NODE_N * 3);
    NODES.forEach(function (p, i) { np[i * 3] = p.x; np[i * 3 + 1] = p.y; np[i * 3 + 2] = p.z; });
    ng.setAttribute("position", new THREE.BufferAttribute(np, 3));
    nodePoints = new THREE.Points(ng, new THREE.PointsMaterial({ size: 5.6, map: dot, transparent: true, depthWrite: false, opacity: .98, sizeAttenuation: true, blending: THREE.AdditiveBlending, color: 0xffffff }));
    group.add(nodePoints);

    var lg = new THREE.BufferGeometry(), lp = new Float32Array(EDGES.length * 6);
    EDGES.forEach(function (e, i) { var p = NODES[e[0]], q = NODES[e[1]]; lp[i * 6] = p.x; lp[i * 6 + 1] = p.y; lp[i * 6 + 2] = p.z; lp[i * 6 + 3] = q.x; lp[i * 6 + 4] = q.y; lp[i * 6 + 5] = q.z; });
    lg.setAttribute("position", new THREE.BufferAttribute(lp, 3));
    lineSeg = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: .14 }));
    group.add(lineSeg);

    var PN = 28, pg = new THREE.BufferGeometry(), pp = new Float32Array(PN * 3);
    for (var z = 0; z < PN; z++) PULSES.push({ e: EDGES[(Math.random() * EDGES.length) | 0], t: Math.random(), s: .004 + Math.random() * .011 });
    pg.setAttribute("position", new THREE.BufferAttribute(pp, 3));
    pulsePoints = new THREE.Points(pg, new THREE.PointsMaterial({ size: 7.5, map: dot, transparent: true, depthWrite: false, opacity: 1, sizeAttenuation: true, blending: THREE.AdditiveBlending, color: 0xffffff }));
    group.add(pulsePoints);

    // light fog so distant nodes recede a touch — but not so much they vanish
    scene.fog = new THREE.FogExp2(0x070708, 0.0011);
    applyTheme3D();
    return true;
  }

  function updatePulses() {
    var arr = pulsePoints.geometry.attributes.position.array;
    for (var i = 0; i < PULSES.length; i++) {
      var p = PULSES[i]; p.t += p.s;
      if (p.t >= 1) { p.e = EDGES[(Math.random() * EDGES.length) | 0]; p.t = 0; p.s = .004 + Math.random() * .011; }
      var a = NODES[p.e[0]], b = NODES[p.e[1]];
      arr[i * 3] = a.x + (b.x - a.x) * p.t; arr[i * 3 + 1] = a.y + (b.y - a.y) * p.t; arr[i * 3 + 2] = a.z + (b.z - a.z) * p.t;
    }
    pulsePoints.geometry.attributes.position.needsUpdate = true;
  }

  function applyTheme3D() {
    if (!nodePoints) return;
    var light = isLight();
    nodePoints.material.color.set(light ? 0x1a1a1a : 0xffffff);
    nodePoints.material.opacity = light ? .85 : .95;
    nodePoints.material.blending = light ? THREE.NormalBlending : THREE.AdditiveBlending;
    lineSeg.material.color.set(light ? 0x2a2a2a : 0xffffff);
    lineSeg.material.opacity = light ? .16 : .22;
    pulsePoints.material.color.set(light ? 0x000000 : 0xffffff);
    pulsePoints.material.opacity = light ? .55 : 1;
    pulsePoints.material.blending = light ? THREE.NormalBlending : THREE.AdditiveBlending;
    nodePoints.material.needsUpdate = lineSeg.material.needsUpdate = pulsePoints.material.needsUpdate = true;
    if (scene.fog) scene.fog.color.set(light ? 0xffffff : 0x070708);
  }

  function loop3D() {
    mouse.x += (mouse.tx - mouse.x) * .04; mouse.y += (mouse.ty - mouse.y) * .04;
    if (group) {
      spinT += 1;
      group.rotation.y += 0.0024;                                   // always-on continuous spin
      var bob = Math.sin(spinT * 0.004) * 0.10;                     // gentle tumble so the spin is obvious
      group.rotation.x += ((((mouse.y - .5) * 0.45) + bob) - group.rotation.x) * 0.04;
      camera.position.x += (((mouse.x - .5) * 56) - camera.position.x) * 0.04;
      camera.position.y += ((-(mouse.y - .5) * 34) - camera.position.y) * 0.04;
      camera.lookAt(0, 0, 0);
      updatePulses();                                               // always-on travelling pulses
      renderer.render(scene, camera);
    }
    requestAnimationFrame(loop3D);
  }

  /* ------------------------------------------------------------------ 2D fallback */
  function start2D() {
    var cv = document.createElement("canvas");
    cv.style.width = "100%"; cv.style.height = "100%";
    host.appendChild(cv);
    var ctx = cv.getContext("2d"); if (!ctx) return;
    var W = 0, H = 0, dpr = Math.min(devicePixelRatio || 1, 2);
    function resize() { W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
    resize(); addEventListener("resize", resize);
    var N = innerWidth < 760 ? 90 : 170, nodes = [];
    for (var i = 0; i < N; i++) nodes.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .25, vy: (Math.random() - .5) * .25 });
    var LINK = 200;
    function draw() {
      var light = isLight();
      var px = (mouse.x - .5) * 26, py = (mouse.y - .5) * 18;
      mouse.x += (mouse.tx - mouse.x) * .05; mouse.y += (mouse.ty - mouse.y) * .05;
      ctx.clearRect(0, 0, W, H);
      var col = light ? "10,10,11" : "255,255,255";
      for (var i = 0; i < N; i++) {
        var a = nodes[i];
        a.x += a.vx; a.y += a.vy; if (a.x < 0 || a.x > W) a.vx *= -1; if (a.y < 0 || a.y > H) a.vy *= -1;
        for (var j = i + 1; j < N; j++) {
          var b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = Math.sqrt(dx * dx + dy * dy);
          if (d < LINK) {
            ctx.strokeStyle = "rgba(" + col + "," + ((1 - d / LINK) * (light ? .10 : .16)).toFixed(3) + ")";
            ctx.lineWidth = 1; ctx.beginPath();
            ctx.moveTo(a.x + px, a.y + py); ctx.lineTo(b.x + px, b.y + py); ctx.stroke();
          }
        }
      }
      ctx.fillStyle = "rgba(" + col + "," + (light ? .55 : .9) + ")";
      for (var k = 0; k < N; k++) { var n = nodes[k]; ctx.beginPath(); ctx.arc(n.x + px, n.y + py, 2.2, 0, 6.2832); ctx.fill(); }
      requestAnimationFrame(draw);
    }
    draw();
  }

  /* ------------------------------------------------------------------ bootstrap */
  var ok3D = false;
  if (typeof THREE !== "undefined") { try { ok3D = build3D(); } catch (e) { ok3D = false; } }
  if (ok3D) {
    loop3D();
    new MutationObserver(applyTheme3D).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  } else {
    start2D();   // Three.js unavailable / WebGL failed -> always show the 2D network
  }
})();
