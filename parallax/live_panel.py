"""Browser-side attitude panel.

Everything here runs inside a single sandboxed iframe that Streamlit mounts
once. It polls the loopback feed for the newest sample and renders at the
display's refresh rate, predicting forward from the last known angular rate so
motion stays continuous between the (much slower) arriving frames.
"""

from __future__ import annotations

PANEL_HEIGHT = 470

# Poll well above the eye's flicker threshold but far below the render rate —
# each response is a few hundred bytes over loopback.
POLL_INTERVAL_MS = 50


def build_panel_html(port: int, palette: dict, face_colors: dict, axis_colors: dict) -> str:
    """Self-contained HTML for the live orientation panel.

    Substitution is plain token replacement rather than %- or str.format-style
    templating: the body is CSS and JavaScript, which are full of bare `%`
    (modulo) and `{}` (blocks) that either scheme would choke on.
    """
    tokens = {
        "__PORT__": str(port),
        "__POLL_MS__": str(POLL_INTERVAL_MS),
        "__BG__": palette["BG"],
        "__WHITE__": palette["WHITE"],
        "__BORDER__": palette["BORDER"],
        "__BORDER_S__": palette["BORDER_S"],
        "__TEXT__": palette["TEXT"],
        "__TEXT_M__": palette["TEXT_M"],
        "__TEXT_D__": palette["TEXT_D"],
        "__GREEN__": palette["GREEN"],
        "__AMBER__": palette["AMBER"],
        "__RED__": palette["RED"],
        "__FX_POS__": face_colors["+X"], "__FX_NEG__": face_colors["-X"],
        "__FY_POS__": face_colors["+Y"], "__FY_NEG__": face_colors["-Y"],
        "__FZ_POS__": face_colors["+Z"], "__FZ_NEG__": face_colors["-Z"],
        "__AXIS_X__": axis_colors["X"], "__AXIS_Y__": axis_colors["Y"],
        "__AXIS_Z__": axis_colors["Z"],
    }
    html = _TEMPLATE
    for token, value in tokens.items():
        html = html.replace(token, value)
    return html


_TEMPLATE = r"""
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: __BG__; color: __TEXT__; }
  .tiles { display: grid; grid-template-columns: repeat(8, 1fr); gap: 6px; margin-bottom: 6px; }
  .tiles-4 { grid-template-columns: repeat(4, 1fr); margin-bottom: 10px; }
  .muted { color: __TEXT_D__; }
  .tile { background: __WHITE__; border: 1px solid __BORDER__; border-radius: 8px;
          padding: 7px 6px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .tile .k { font-size: 9px; font-weight: 700; color: __TEXT_D__; letter-spacing: 0.08em; }
  .tile .v { font-size: 13px; font-weight: 700; font-family: ui-monospace, Menlo, monospace;
             margin-top: 2px; }
  .row { display: flex; gap: 12px; align-items: stretch; }
  .stage { flex: 1.15; background: __WHITE__; border: 1px solid __BORDER__; border-radius: 8px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.05); position: relative; min-height: 300px; }
  .side { flex: 1; display: flex; flex-direction: column; gap: 6px; }
  .att { background: __WHITE__; border: 1px solid __BORDER__; border-left: 4px solid; border-radius: 8px;
         padding: 9px 14px; display: flex; justify-content: space-between; align-items: center;
         box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
  .att .k { font-size: 11px; font-weight: 700; color: __TEXT_D__; letter-spacing: 0.08em; }
  .att .v { font-size: 17px; font-weight: 700; font-family: ui-monospace, Menlo, monospace; }
  button { background: __WHITE__; border: 1.5px solid __BORDER_S__; color: __TEXT_M__;
           font-size: 12px; font-weight: 600; padding: 9px; border-radius: 8px; cursor: pointer;
           transition: all .15s; font-family: inherit; }
  button:hover { border-color: __AXIS_X__; color: __AXIS_X__; }
  .note { font-size: 10px; color: __TEXT_D__; line-height: 1.5; }
  .status { position: absolute; top: 8px; left: 10px; font-size: 10px; font-weight: 700;
            letter-spacing: 0.06em; color: __TEXT_D__; font-family: ui-monospace, Menlo, monospace; }
  canvas { display: block; width: 100%; height: 100%; }
</style>

<div class="tiles" id="tiles"></div>
<div class="tiles tiles-4" id="tiles2"></div>
<div class="row">
  <div class="stage"><div class="status" id="status">CONNECTING…</div><canvas id="cube"></canvas></div>
  <div class="side">
    <div class="att" id="att-roll"  style="border-left-color:__AXIS_X__"><span class="k">ROLL</span><span class="v">—</span></div>
    <div class="att" id="att-pitch" style="border-left-color:__AXIS_Y__"><span class="k">PITCH</span><span class="v">—</span></div>
    <div class="att" id="att-yaw"   style="border-left-color:__AXIS_Z__"><span class="k">YAW</span><span class="v">—</span></div>
    <button id="relevel">⟲ Re-level</button>
    <div class="note">Rendered in the browser at display refresh rate and predicted forward from the
    last angular rate, so motion stays smooth between arriving frames. Roll and pitch are held to
    gravity; yaw is gyro-only and drifts.</div>
  </div>
</div>

<script>
(function () {
  const FEED = "http://127.0.0.1:__PORT__";
  const FACES = ["__FX_POS__", "__FX_NEG__", "__FY_POS__", "__FY_NEG__", "__FZ_POS__", "__FZ_NEG__"];
  const AXIS = ["__AXIS_X__", "__AXIS_Y__", "__AXIS_Z__"];

  // ---- state -------------------------------------------------------------
  // `target` is where the board is believed to be right now; `shown` chases it
  // so a late or corrected frame eases in instead of snapping.
  let target = {roll: 0, pitch: 0, yaw: 0};
  let shown  = {roll: 0, pitch: 0, yaw: 0};
  let rates  = [0, 0, 0];
  let snap = null, haveFix = false, lastFetchAt = 0, inFlight = false;
  let lastFrameAt = performance.now();

  const wrap = (d) => { d = (d + 180) % 360; return (d < 0 ? d + 360 : d) - 180; };

  // ---- data feed ---------------------------------------------------------
  async function poll() {
    if (inFlight) return;
    inFlight = true;
    try {
      const res = await fetch(FEED + "/orientation", {cache: "no-store"});
      snap = await res.json();
      if (snap && snap.live) {
        rates = snap.gyro || [0, 0, 0];
        // Roll the reading forward by however stale it already is.
        const age = snap.age_s || 0;
        target = {
          roll:  wrap(snap.roll  + rates[0] * age),
          pitch: wrap(snap.pitch + rates[1] * age),
          yaw:   wrap(snap.yaw   + rates[2] * age),
        };
        lastFetchAt = performance.now();
        if (!haveFix) { shown = Object.assign({}, target); haveFix = true; }
      }
    } catch (err) {
      snap = null;
    } finally {
      inFlight = false;
    }
  }
  poll();
  setInterval(poll, __POLL_MS__);

  // ---- geometry ----------------------------------------------------------
  function rotate(p, roll, pitch, yaw) {
    const r = roll * Math.PI / 180, q = pitch * Math.PI / 180, y = yaw * Math.PI / 180;
    const sr = Math.sin(r), cr = Math.cos(r), sp = Math.sin(q), cp = Math.cos(q),
          sy = Math.sin(y), cy = Math.cos(y);
    return [
      (cy*cp)*p[0] + (cy*sp*sr - sy*cr)*p[1] + (cy*sp*cr + sy*sr)*p[2],
      (sy*cp)*p[0] + (sy*sp*sr + cy*cr)*p[1] + (sy*sp*cr - cy*sr)*p[2],
      (-sp)*p[0]   + (cp*sr)*p[1]            + (cp*cr)*p[2],
    ];
  }

  // Fixed three-quarter view matching the rest of the dashboard.
  const EYE = [1.75, -1.75, 1.15];
  const norm = (v) => { const m = Math.hypot(v[0], v[1], v[2]); return [v[0]/m, v[1]/m, v[2]/m]; };
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const FWD = norm([-EYE[0], -EYE[1], -EYE[2]]);
  const RIGHT = norm(cross(FWD, [0, 0, 1]));
  const UP = cross(RIGHT, FWD);

  const CORNERS = [];
  for (const z of [-1, 1]) for (const y of [-1, 1]) for (const x of [-1, 1]) CORNERS.push([x, y, z]);
  //          0(-,-,-) 1(+,-,-) 2(-,+,-) 3(+,+,-) 4(-,-,+) 5(+,-,+) 6(-,+,+) 7(+,+,+)
  const QUADS = [
    {v: [1, 3, 7, 5], c: 0},   // +X
    {v: [0, 2, 6, 4], c: 1},   // -X
    {v: [2, 3, 7, 6], c: 2},   // +Y
    {v: [0, 1, 5, 4], c: 3},   // -Y
    {v: [4, 5, 7, 6], c: 4},   // +Z
    {v: [0, 1, 3, 2], c: 5},   // -Z
  ];

  // ---- rendering ---------------------------------------------------------
  const canvas = document.getElementById("cube");
  const ctx = canvas.getContext("2d");

  // `depth` grows with distance from the eye, since FWD points from the eye
  // toward the origin. Perspective must therefore shrink with growing depth.
  const FOCAL = 9;

  function project(p, cx, cy, scale) {
    const depth = dot(p, FWD);
    const persp = FOCAL / (FOCAL + depth);
    return [cx + dot(p, RIGHT) * scale * persp, cy - dot(p, UP) * scale * persp, depth];
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2, cy = h / 2, scale = Math.min(w, h) / 4.8;
    const pts = CORNERS.map((p) => project(rotate(p, shown.roll, shown.pitch, shown.yaw), cx, cy, scale));

    // Painter's algorithm: draw the furthest face first, so nearer faces cover
    // it. Larger depth is further away, hence the descending sort.
    const faces = QUADS.map((q) => {
      const depth = q.v.reduce((acc, i) => acc + pts[i][2], 0) / 4;
      return {q, depth};
    }).sort((a, b) => b.depth - a.depth);

    // Body axes. An axis pointing away from the eye is drawn before the cube so
    // the solid hides it, matching how a real object occludes its far side.
    const units = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const names = ["X", "Y", "Z"];
    const origin = project([0, 0, 0], cx, cy, scale);
    const axes = units.map((u, i) => ({
      i,
      tip: project(rotate(u.map((c) => c * 1.85), shown.roll, shown.pitch, shown.yaw), cx, cy, scale),
      lab: project(rotate(u.map((c) => c * 2.16), shown.roll, shown.pitch, shown.yaw), cx, cy, scale),
    }));

    function drawAxis(a) {
      ctx.beginPath();
      ctx.moveTo(origin[0], origin[1]);
      ctx.lineTo(a.tip[0], a.tip[1]);
      ctx.strokeStyle = AXIS[a.i];
      ctx.lineWidth = 2.5;
      ctx.stroke();
      ctx.fillStyle = AXIS[a.i];
      ctx.font = "600 12px ui-monospace, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(names[a.i], a.lab[0], a.lab[1]);
    }

    axes.filter((a) => a.tip[2] > 0).forEach(drawAxis);

    for (const {q} of faces) {
      ctx.beginPath();
      q.v.forEach((i, n) => (n ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1])));
      ctx.closePath();
      ctx.fillStyle = FACES[q.c];
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    axes.filter((a) => a.tip[2] <= 0).forEach(drawAxis);
  }

  // ---- readouts ----------------------------------------------------------
  const TILES = [
    ["GYRO X", (s) => s.gyro[0].toFixed(2), " °/s"],
    ["GYRO Y", (s) => s.gyro[1].toFixed(2), " °/s"],
    ["GYRO Z", (s) => s.gyro[2].toFixed(2), " °/s"],
    ["RATE",   (s) => s.rate.toFixed(2),    " °/s"],
    ["ACCEL X", (s) => s.accel[0].toFixed(2), " g"],
    ["ACCEL Y", (s) => s.accel[1].toFixed(2), " g"],
    ["ACCEL Z", (s) => s.accel[2].toFixed(2), " g"],
    ["TILT",   (s) => s.tilt.toFixed(1),    "°"],
  ];
  // Sensors that are absent report a sentinel upstream and arrive as null.
  const TILES2 = [
    ["EXT TEMP", (s) => s.temp_c === null ? "no sensor" : s.temp_c.toFixed(1) + " °C"],
    ["RANGE",    (s) => s.distance_cm === null ? "no echo" : s.distance_cm.toFixed(1) + " cm"],
    ["FRAMES",   (s) => String(s.frames)],
    ["IN RATE",  (s) => s.rate_hz.toFixed(1) + " Hz"],
  ];

  const tilesEl = document.getElementById("tiles");
  tilesEl.innerHTML = TILES.map(([k]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">—</div></div>`).join("");
  const tileValues = [...tilesEl.querySelectorAll(".v")];

  const tiles2El = document.getElementById("tiles2");
  tiles2El.innerHTML = TILES2.map(([k]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">—</div></div>`).join("");
  const tile2Values = [...tiles2El.querySelectorAll(".v")];

  const attEls = {
    roll: document.querySelector("#att-roll .v"),
    pitch: document.querySelector("#att-pitch .v"),
    yaw: document.querySelector("#att-yaw .v"),
  };
  const statusEl = document.getElementById("status");

  function updateText() {
    if (snap && snap.live) {
      TILES.forEach(([, fn, unit], i) => { tileValues[i].textContent = fn(snap) + unit; });
      TILES2.forEach(([, fn], i) => {
        const text = fn(snap);
        tile2Values[i].textContent = text;
        tile2Values[i].classList.toggle("muted", text === "no sensor" || text === "no echo");
      });
      statusEl.textContent = `● LIVE · ${snap.rate_hz.toFixed(1)} Hz IN · ${fps.toFixed(0)} FPS OUT`;
      statusEl.style.color = "__GREEN__";
    } else if (snap && snap.connected) {
      statusEl.textContent = "◑ CONNECTED — NO FRAMES";
      statusEl.style.color = "__AMBER__";
    } else {
      statusEl.textContent = snap ? "○ DISCONNECTED" : "○ LINK UNAVAILABLE";
      statusEl.style.color = "__TEXT_D__";
    }
    attEls.roll.textContent  = (shown.roll  >= 0 ? "+" : "") + shown.roll.toFixed(1) + "°";
    attEls.pitch.textContent = (shown.pitch >= 0 ? "+" : "") + shown.pitch.toFixed(1) + "°";
    attEls.yaw.textContent   = (shown.yaw   >= 0 ? "+" : "") + shown.yaw.toFixed(1) + "°";
  }

  // ---- animation loop ----------------------------------------------------
  // Chase the predicted attitude with a time-based ease, so the result is
  // frame-rate independent and free of the steps a raw 5 Hz feed would show.
  const EASE_TAU = 0.07;
  let fps = 0, textAccum = 0;

  function frame(now) {
    const dt = Math.min(0.1, (now - lastFrameAt) / 1000);
    lastFrameAt = now;
    fps = fps ? fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1 : 1 / Math.max(dt, 1e-3);

    if (haveFix && snap && snap.live) {
      // Keep extrapolating from the newest frame while we wait for the next.
      const since = (now - lastFetchAt) / 1000;
      const predicted = {
        roll:  wrap(target.roll  + rates[0] * since),
        pitch: wrap(target.pitch + rates[1] * since),
        yaw:   wrap(target.yaw   + rates[2] * since),
      };
      const k = 1 - Math.exp(-dt / EASE_TAU);
      for (const key of ["roll", "pitch", "yaw"]) {
        shown[key] = wrap(shown[key] + wrap(predicted[key] - shown[key]) * k);
      }
    }

    draw();
    textAccum += dt;
    if (textAccum > 0.1) { updateText(); textAccum = 0; }   // 10 Hz is plenty for digits
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  document.getElementById("relevel").addEventListener("click", () => {
    fetch(FEED + "/relevel", {method: "POST"}).then(() => { haveFix = false; });
  });
})();
</script>
"""
