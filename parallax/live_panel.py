"""Browser-side attitude panel — Asteria-7 satellite orientation display."""

from __future__ import annotations

PANEL_HEIGHT = 490
POLL_INTERVAL_MS = 50


def build_panel_html(port: int, palette: dict, face_colors: dict, axis_colors: dict) -> str:
    tokens = {
        "__PORT__":    str(port),
        "__POLL_MS__": str(POLL_INTERVAL_MS),
        "__BG__":      palette["BG"],
        "__WHITE__":   palette["WHITE"],
        "__BORDER__":  palette["BORDER"],
        "__BORDER_S__":palette["BORDER_S"],
        "__TEXT__":    palette["TEXT"],
        "__TEXT_M__":  palette["TEXT_M"],
        "__TEXT_D__":  palette["TEXT_D"],
        "__GREEN__":   palette["GREEN"],
        "__AMBER__":   palette["AMBER"],
        "__RED__":     palette["RED"],
        "__AXIS_X__":  axis_colors["X"],
        "__AXIS_Y__":  axis_colors["Y"],
        "__AXIS_Z__":  axis_colors["Z"],
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
  .tiles   { display: grid; grid-template-columns: repeat(8,1fr); gap: 6px; margin-bottom: 6px; }
  .tiles-4 { grid-template-columns: repeat(4,1fr); margin-bottom: 10px; }
  .tile { background: __WHITE__; border: 1px solid __BORDER__; border-radius: 8px;
          padding: 7px 6px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .tile .k { font-size: 9px; font-weight: 700; color: __TEXT_D__; letter-spacing: 0.08em; }
  .tile .v { font-size: 13px; font-weight: 700; font-family: ui-monospace,Menlo,monospace; margin-top: 2px; }
  .muted   { color: __TEXT_D__; }
  .row  { display: flex; gap: 12px; align-items: stretch; }
  .stage{ flex: 1.4; background: __WHITE__; border: 1px solid __BORDER__; border-radius: 10px;
          box-shadow: 0 1px 6px rgba(0,0,0,0.06); position: relative; min-height: 330px; }
  .side { flex: 1; display: flex; flex-direction: column; gap: 6px; }
  .att  { background: __WHITE__; border: 1px solid __BORDER__; border-left: 4px solid;
          border-radius: 8px; padding: 9px 14px;
          display: flex; justify-content: space-between; align-items: center;
          box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
  .att .k { font-size: 11px; font-weight: 700; color: __TEXT_D__; letter-spacing: 0.08em; }
  .att .v { font-size: 18px; font-weight: 700; font-family: ui-monospace,Menlo,monospace; }
  button  { background: __WHITE__; border: 1.5px solid __BORDER_S__; color: __TEXT_M__;
            font-size: 12px; font-weight: 600; padding: 9px; border-radius: 8px;
            cursor: pointer; transition: all .15s; font-family: inherit; }
  button:hover { border-color: __AXIS_X__; color: __AXIS_X__; }
  .note   { font-size: 10px; color: __TEXT_D__; line-height: 1.55; }
  .status { position: absolute; top: 9px; left: 12px; font-size: 10px; font-weight: 700;
            letter-spacing: 0.07em; color: __TEXT_D__;
            font-family: ui-monospace,Menlo,monospace; }
  canvas  { display: block; width: 100%; height: 100%; border-radius: 10px; }
</style>

<div class="tiles"   id="tiles"></div>
<div class="tiles tiles-4" id="tiles2"></div>
<div class="row">
  <div class="stage">
    <div class="status" id="status">CONNECTING…</div>
    <canvas id="sat"></canvas>
  </div>
  <div class="side">
    <div class="att" id="att-roll"  style="border-left-color:__AXIS_X__">
      <span class="k">ROLL</span><span class="v" id="v-roll">—</span></div>
    <div class="att" id="att-pitch" style="border-left-color:__AXIS_Y__">
      <span class="k">PITCH</span><span class="v" id="v-pitch">—</span></div>
    <div class="att" id="att-yaw"   style="border-left-color:__AXIS_Z__">
      <span class="k">YAW</span><span class="v" id="v-yaw">—</span></div>
    <button id="relevel">⟲ Re-level</button>
    <div class="note">
      Asteria-7 model rendered at display refresh rate, predicted forward from the
      last angular-rate sample between frames. Roll and pitch are gravity-corrected;
      yaw is gyro-only and drifts.
    </div>
  </div>
</div>

<script>
(function () {
  const FEED    = "http://127.0.0.1:__PORT__";
  const AXIS    = ["__AXIS_X__", "__AXIS_Y__", "__AXIS_Z__"];

  // ── State ────────────────────────────────────────────────────────────────
  let target = {roll:0,pitch:0,yaw:0};
  let shown  = {roll:0,pitch:0,yaw:0};
  let rates  = [0,0,0];
  let snap = null, haveFix = false, lastFetchAt = 0, inFlight = false;
  let lastFrameAt = performance.now();
  const wrap = (d) => { d=(d+180)%360; return (d<0?d+360:d)-180; };

  // ── Feed ─────────────────────────────────────────────────────────────────
  async function poll() {
    if (inFlight) return;
    inFlight = true;
    try {
      const res = await fetch(FEED+"/orientation",{cache:"no-store"});
      snap = await res.json();
      if (snap && snap.live) {
        rates = snap.gyro||[0,0,0];
        const age = snap.age_s||0;
        target = {
          roll:  wrap(snap.roll  + rates[0]*age),
          pitch: wrap(snap.pitch + rates[1]*age),
          yaw:   wrap(snap.yaw   + rates[2]*age),
        };
        lastFetchAt = performance.now();
        if (!haveFix) { shown = Object.assign({},target); haveFix = true; }
      }
    } catch(_) { snap = null; } finally { inFlight = false; }
  }
  poll();
  setInterval(poll, __POLL_MS__);

  // ── Math helpers ─────────────────────────────────────────────────────────
  const dot3  = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross3 = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const norm3  = (v) => { const m=Math.hypot(v[0],v[1],v[2]); return [v[0]/m,v[1]/m,v[2]/m]; };
  const add3   = (a,b) => [a[0]+b[0],a[1]+b[1],a[2]+b[2]];

  function rotate(p, roll, pitch, yaw) {
    const r=roll*Math.PI/180, q=pitch*Math.PI/180, y=yaw*Math.PI/180;
    const sr=Math.sin(r),cr=Math.cos(r),sp=Math.sin(q),cp=Math.cos(q),sy=Math.sin(y),cy=Math.cos(y);
    return [
      (cy*cp)*p[0]+(cy*sp*sr-sy*cr)*p[1]+(cy*sp*cr+sy*sr)*p[2],
      (sy*cp)*p[0]+(sy*sp*sr+cy*cr)*p[1]+(sy*sp*cr-cy*sr)*p[2],
      (-sp)*p[0]  +(cp*sr)*p[1]          +(cp*cr)*p[2],
    ];
  }

  // ── Camera ───────────────────────────────────────────────────────────────
  const EYE   = [1.75,-1.75,1.15];
  const EYE_N = norm3(EYE);
  const FWD   = norm3(EYE.map(x=>-x));
  const RIGHT = norm3(cross3(FWD,[0,0,1]));
  const UP    = cross3(RIGHT,FWD);
  const FOCAL = 7;

  function project(p, cx, cy, sc) {
    const depth = dot3(p,FWD);
    const persp = FOCAL/(FOCAL+depth);
    return [cx+dot3(p,RIGHT)*sc*persp, cy-dot3(p,UP)*sc*persp, depth];
  }

  // ── Shading ───────────────────────────────────────────────────────────────
  const LIGHT = norm3([0.65,-0.45,0.90]);
  const HALF  = norm3(add3(LIGHT, EYE_N));

  function shade(rgb, wn) {
    const d  = Math.max(0, dot3(wn, LIGHT));
    const sp = Math.pow(Math.max(0, dot3(wn, HALF)), 22) * 0.18;
    const f  = 0.36 + 0.64*d + sp;
    return `rgb(${Math.min(255,~~(rgb[0]*f))},${Math.min(255,~~(rgb[1]*f))},${Math.min(255,~~(rgb[2]*f))})`;
  }

  // ── Geometry builders ────────────────────────────────────────────────────
  // Colors [R,G,B]
  const C_BUS = [148,178,204];
  const C_PNL = [28, 56, 128];
  const C_RAD = [210,222,234];
  const C_HGA = [215,225,238];
  const C_INS = [84, 122,156];
  const C_THR = [116,148,172];

  function boxFaces(cx,cy,cz, hx,hy,hz, color, tag) {
    tag = tag||null;
    return [
      {verts:[[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx+hx,cy+hy,cz+hz],[cx+hx,cy-hy,cz+hz]], norm:[1,0,0],  color, tag},
      {verts:[[cx-hx,cy+hy,cz-hz],[cx-hx,cy-hy,cz-hz],[cx-hx,cy-hy,cz+hz],[cx-hx,cy+hy,cz+hz]], norm:[-1,0,0], color, tag},
      {verts:[[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz+hz],[cx+hx,cy+hy,cz+hz]], norm:[0,1,0],  color, tag},
      {verts:[[cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz+hz],[cx-hx,cy-hy,cz+hz]], norm:[0,-1,0], color, tag},
      {verts:[[cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz]], norm:[0,0,1],  color, tag},
      {verts:[[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],[cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz]], norm:[0,0,-1], color, tag},
    ];
  }

  function diskFace(cx,cy,cz, r, n, normDir, color, tag) {
    const verts=[];
    for (let i=0;i<n;i++) {
      const a=2*Math.PI*i/n;
      if (Math.abs(normDir[2])>0.9) verts.push([cx+r*Math.cos(a), cy+r*Math.sin(a), cz]);
      else if (Math.abs(normDir[1])>0.9) verts.push([cx+r*Math.cos(a), cy, cz+r*Math.sin(a)]);
      else verts.push([cx, cy+r*Math.cos(a), cz+r*Math.sin(a)]);
    }
    return [{verts, norm:normDir, color, tag:tag||null}];
  }

  // ── Asteria-7 satellite faces ─────────────────────────────────────────────
  // Solar panel large faces (±Y) get tag "panel" for grid overlay
  function panelBox(cx,cy,cz, hx,hy,hz, color) {
    return boxFaces(cx,cy,cz,hx,hy,hz,color).map((f,i)=>({...f,tag:(i===2||i===3)?"panel":null}));
  }

  const SAT = [
    // Main bus
    ...boxFaces(0,0,0, 1.05,0.85,0.70, C_BUS),
    // Solar panel +X
    ...panelBox(3.3,0,0, 2.10,0.04,0.62, C_PNL),
    // Solar panel -X
    ...panelBox(-3.3,0,0, 2.10,0.04,0.62, C_PNL),
    // Bus-to-panel struts
    ...boxFaces(1.13,0,0, 0.08,0.07,0.055, C_BUS),
    ...boxFaces(-1.13,0,0, 0.08,0.07,0.055, C_BUS),
    // Radiator (+Y side)
    ...boxFaces(0,0.91,0, 0.84,0.04,0.52, C_RAD),
    // PCU (-Y side)
    ...boxFaces(0,-0.97,0, 0.42,0.12,0.35, C_INS),
    // Spectrometer
    ...boxFaces(0.36,0,0.82, 0.35,0.32,0.12, C_INS),
    // Camera
    ...boxFaces(-0.36,0,0.82, 0.24,0.25,0.12, C_INS),
    // HGA rim (drawn first so dish occludes it)
    ...diskFace(0,0,0.845, 0.60,18, [0,0,1], C_INS),
    // HGA dish
    ...diskFace(0,0,0.860, 0.55,18, [0,0,1], C_HGA, "hga"),
    // HGA support post
    ...boxFaces(0,0,0.77, 0.055,0.055,0.07, C_INS),
    // LGA (bottom)
    ...diskFace(0,0,-0.74, 0.15,10, [0,0,-1], C_INS),
    // Thrusters (4 corners of bottom)
    ...boxFaces( 0.62, 0.46,-0.74, 0.085,0.085,0.038, C_THR),
    ...boxFaces(-0.62, 0.46,-0.74, 0.085,0.085,0.038, C_THR),
    ...boxFaces( 0.62,-0.46,-0.74, 0.085,0.085,0.038, C_THR),
    ...boxFaces(-0.62,-0.46,-0.74, 0.085,0.085,0.038, C_THR),
  ];

  // ── Overlays ──────────────────────────────────────────────────────────────
  function drawPanelGrid(pv) {
    if (pv.length !== 4) return;
    const ROWS=5, COLS=13;
    ctx.save();
    ctx.beginPath();
    pv.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    ctx.closePath();
    ctx.clip();
    ctx.strokeStyle="rgba(100,160,255,0.22)";
    ctx.lineWidth=0.75;
    // horizontal (constant Z lines)
    for (let r=1;r<ROWS;r++) {
      const t=r/ROWS;
      const ax=pv[0][0]+(pv[3][0]-pv[0][0])*t, ay=pv[0][1]+(pv[3][1]-pv[0][1])*t;
      const bx=pv[1][0]+(pv[2][0]-pv[1][0])*t, by=pv[1][1]+(pv[2][1]-pv[1][1])*t;
      ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();
    }
    // vertical (constant X lines)
    for (let c=1;c<COLS;c++) {
      const t=c/COLS;
      const ax=pv[0][0]+(pv[1][0]-pv[0][0])*t, ay=pv[0][1]+(pv[1][1]-pv[0][1])*t;
      const bx=pv[3][0]+(pv[2][0]-pv[3][0])*t, by=pv[3][1]+(pv[2][1]-pv[3][1])*t;
      ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();
    }
    ctx.restore();
  }

  function drawHGASpokes(pv) {
    const cx=pv.reduce((s,p)=>s+p[0],0)/pv.length;
    const cy=pv.reduce((s,p)=>s+p[1],0)/pv.length;
    ctx.save();
    ctx.strokeStyle="rgba(130,175,215,0.55)";
    ctx.lineWidth=1.4;
    const step=Math.floor(pv.length/6);
    for (let i=0;i<6;i++) {
      const p=pv[i*step];
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(p[0],p[1]); ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(cx,cy,3,0,2*Math.PI);
    ctx.fillStyle="rgba(100,155,210,0.75)";
    ctx.fill();
    ctx.restore();
  }

  // ── Renderer ──────────────────────────────────────────────────────────────
  const canvas = document.getElementById("sat");
  const ctx    = canvas.getContext("2d");

  function draw() {
    const dpr=window.devicePixelRatio||1;
    const w=canvas.clientWidth, h=canvas.clientHeight;
    if (canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)) {
      canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr);
    }
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,w,h);

    const cx=w/2, cy=h/2;
    const sc=Math.min(w,h)/10.5;   // satellite spans ~10.8 model units tip-to-tip

    const {roll,pitch,yaw}=shown;

    // Project a single model-space point
    const proj=(p)=>project(rotate(p,pitch,roll,yaw), cx,cy,sc);

    // Back-face cull → project → collect
    const visible=[];
    for (const face of SAT) {
      const wn=rotate(face.norm,pitch,roll,yaw);
      if (dot3(wn,EYE_N)<-0.06) continue;
      const pv=face.verts.map(v=>proj(v));
      const depth=pv.reduce((s,p)=>s+p[2],0)/pv.length;
      visible.push({pv,wn,rgb:face.color,tag:face.tag,depth});
    }
    visible.sort((a,b)=>b.depth-a.depth);

    // Body axes
    const o=proj([0,0,0]);
    const AL=1.55;   // axis length in model units
    const axes=[
      {i:0, tip:proj([AL,0,0]),  lab:proj([AL*1.22,0,0])},
      {i:1, tip:proj([0,AL,0]),  lab:proj([0,AL*1.22,0])},
      {i:2, tip:proj([0,0,AL]),  lab:proj([0,0,AL*1.22])},
    ];
    function drawAxis(a) {
      ctx.beginPath(); ctx.moveTo(o[0],o[1]); ctx.lineTo(a.tip[0],a.tip[1]);
      ctx.strokeStyle=AXIS[a.i]; ctx.lineWidth=2.4; ctx.stroke();
      ctx.fillStyle=AXIS[a.i];
      ctx.font="600 11px ui-monospace,Menlo,monospace";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(["X","Y","Z"][a.i], a.lab[0], a.lab[1]);
    }

    // Axes behind satellite
    axes.filter(a=>a.tip[2]>0).forEach(drawAxis);

    // Faces
    for (const {pv,wn,rgb,tag} of visible) {
      ctx.beginPath();
      pv.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
      ctx.closePath();
      ctx.fillStyle=shade(rgb,wn);
      ctx.fill();
      ctx.strokeStyle="rgba(255,255,255,0.22)";
      ctx.lineWidth=0.8;
      ctx.stroke();
      if (tag==="panel") drawPanelGrid(pv);
      if (tag==="hga")   drawHGASpokes(pv);
    }

    // Axes in front
    axes.filter(a=>a.tip[2]<=0).forEach(drawAxis);
  }

  // ── Readouts ──────────────────────────────────────────────────────────────
  const TILES=[
    ["GYRO X", s=>s.gyro[0].toFixed(2)," °/s"],
    ["GYRO Y", s=>s.gyro[1].toFixed(2)," °/s"],
    ["GYRO Z", s=>s.gyro[2].toFixed(2)," °/s"],
    ["RATE",   s=>s.rate.toFixed(2),   " °/s"],
    ["ACCEL X",s=>s.accel[0].toFixed(2)," g"],
    ["ACCEL Y",s=>s.accel[1].toFixed(2)," g"],
    ["ACCEL Z",s=>s.accel[2].toFixed(2)," g"],
    ["TILT",   s=>s.tilt.toFixed(1),   "°"],
  ];
  const TILES2=[
    ["EXT TEMP",s=>s.temp_c===null?"no sensor":s.temp_c.toFixed(1)+" °C"],
    ["RANGE",   s=>s.distance_cm===null?"no echo":s.distance_cm.toFixed(1)+" cm"],
    ["FRAMES",  s=>String(s.frames)],
    ["IN RATE", s=>s.rate_hz.toFixed(1)+" Hz"],
  ];

  const tilesEl=document.getElementById("tiles");
  tilesEl.innerHTML=TILES.map(([k])=>`<div class="tile"><div class="k">${k}</div><div class="v">—</div></div>`).join("");
  const tileVals=[...tilesEl.querySelectorAll(".v")];

  const tiles2El=document.getElementById("tiles2");
  tiles2El.innerHTML=TILES2.map(([k])=>`<div class="tile"><div class="k">${k}</div><div class="v">—</div></div>`).join("");
  const tile2Vals=[...tiles2El.querySelectorAll(".v")];

  const vRoll=document.getElementById("v-roll");
  const vPitch=document.getElementById("v-pitch");
  const vYaw=document.getElementById("v-yaw");
  const statusEl=document.getElementById("status");

  let fps=0, textAccum=0;

  function updateText() {
    if (snap && snap.live) {
      TILES.forEach(([,fn,unit],i)=>{ tileVals[i].textContent=fn(snap)+unit; });
      TILES2.forEach(([,fn],i)=>{
        const t=fn(snap);
        tile2Vals[i].textContent=t;
        tile2Vals[i].classList.toggle("muted",t==="no sensor"||t==="no echo");
      });
      statusEl.textContent=`● LIVE · ${snap.rate_hz.toFixed(1)} Hz IN · ${fps.toFixed(0)} FPS OUT`;
      statusEl.style.color="__GREEN__";
    } else if (snap && snap.connected) {
      statusEl.textContent="◑ CONNECTED — NO FRAMES";
      statusEl.style.color="__AMBER__";
    } else {
      statusEl.textContent=snap?"○ DISCONNECTED":"○ LINK UNAVAILABLE";
      statusEl.style.color="__TEXT_D__";
    }
    vRoll.textContent =(shown.roll >=0?"+":"")+shown.roll.toFixed(1)+"°";
    vPitch.textContent=(shown.pitch>=0?"+":"")+shown.pitch.toFixed(1)+"°";
    vYaw.textContent  =(shown.yaw  >=0?"+":"")+shown.yaw.toFixed(1)+"°";
  }

  // ── Animation loop ────────────────────────────────────────────────────────
  const EASE_TAU=0.07;

  function frame(now) {
    const dt=Math.min(0.1,(now-lastFrameAt)/1000);
    lastFrameAt=now;
    fps=fps?fps*0.9+(1/Math.max(dt,1e-3))*0.1:1/Math.max(dt,1e-3);

    if (haveFix && snap && snap.live) {
      const since=(now-lastFetchAt)/1000;
      const predicted={
        roll:  wrap(target.roll  + rates[0]*since),
        pitch: wrap(target.pitch + rates[1]*since),
        yaw:   wrap(target.yaw   + rates[2]*since),
      };
      const k=1-Math.exp(-dt/EASE_TAU);
      for (const key of ["roll","pitch","yaw"])
        shown[key]=wrap(shown[key]+wrap(predicted[key]-shown[key])*k);
    }

    draw();
    textAccum+=dt;
    if (textAccum>0.1) { updateText(); textAccum=0; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  document.getElementById("relevel").addEventListener("click",()=>{
    fetch(FEED+"/relevel",{method:"POST"}).then(()=>{ haveFix=false; });
  });
})();
</script>
"""
