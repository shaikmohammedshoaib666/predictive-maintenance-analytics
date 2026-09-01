"""Layer 3 / Upgrade 1 — offline 3D digital twin (pack-aware).

Renders a rotatable three.js twin whose **hotspot** (bearing / piston engine /
engine block / SRP gearbox) turns red and blinks on High risk (amber Medium,
green Low). Meshes switch with the industry pack:

- plant_motor — generic rotating machine (default)
- aviation_uav — MALE UAV airframe + aero piston engine (SIH26054)
- auto_car — one generic ICE car + powertrain (not OEM / trim / EV catalogs)
- oil_srp — beam pump / sucker-rod well (SIH26120)

three.js + OrbitControls are vendored locally (see ``src/vendor/three``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_VENDOR_DIR = Path(__file__).parent / "vendor" / "three"

TWIN_KINDS = ("plant_motor", "aviation_uav", "auto_car", "oil_srp")
DEFAULT_TWIN_KIND = "plant_motor"

RISK_COLORS = {
    "High": {"hex": "#e74c3c", "blink": True, "speed": 6.0},
    "Medium": {"hex": "#f39c12", "blink": True, "speed": 2.2},
    "Low": {"hex": "#27ae60", "blink": False, "speed": 0.0},
    "Unknown": {"hex": "#7f8c8d", "blink": False, "speed": 0.0},
}

_CAMERA = {
    "plant_motor": {"pos": [6.5, 3.6, 7.5], "target": [0, 0.4, 0]},
    "aviation_uav": {"pos": [9.5, 4.8, 10.5], "target": [0, 0.3, 0]},
    "auto_car": {"pos": [7.2, 3.4, 8.0], "target": [0, 0.5, 0]},
    "oil_srp": {"pos": [12.0, 7.5, 12.5], "target": [0, 2.2, 0]},
}


def normalize_risk(risk: Optional[str]) -> str:
    """Map arbitrary risk strings onto High / Medium / Low / Unknown."""
    r = (risk or "").strip().lower()
    if r.startswith("high") or r in {"critical", "severe"}:
        return "High"
    if r.startswith("med") or r in {"warn", "warning", "elevated"}:
        return "Medium"
    if r.startswith("low") or r in {"ok", "healthy", "normal"}:
        return "Low"
    return "Unknown"


def normalize_twin_kind(kind: Optional[str]) -> str:
    k = (kind or "").strip().lower()
    if k in TWIN_KINDS:
        return k
    aliases = {
        "plant": "plant_motor",
        "motor": "plant_motor",
        "plant_rotating": "plant_motor",
        "aviation": "aviation_uav",
        "aviation_uav_piston": "aviation_uav",
        "uav": "aviation_uav",
        "auto": "auto_car",
        "automotive": "auto_car",
        "automotive_powertrain": "auto_car",
        "car": "auto_car",
        "oil": "oil_srp",
        "srp": "oil_srp",
        "oil_well": "oil_srp",
    }
    return aliases.get(k, DEFAULT_TWIN_KIND)


def asset_states_from_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce pipeline predictions to the minimal state the twin needs."""
    states: list[dict[str, Any]] = []
    for p in predictions or []:
        states.append(
            {
                "machine_id": str(p.get("machine_id", "asset")),
                "risk_level": normalize_risk(p.get("risk_level")),
                "predicted_rul_days": p.get("predicted_rul_days"),
            }
        )
    return states


@lru_cache(maxsize=1)
def _vendor_js() -> tuple[str, str]:
    """Read the vendored three.js + OrbitControls once (cached)."""
    three = (_VENDOR_DIR / "three.min.js").read_text(encoding="utf-8")
    orbit = (_VENDOR_DIR / "OrbitControls.js").read_text(encoding="utf-8")
    return three, orbit


def vendor_available() -> bool:
    return (_VENDOR_DIR / "three.min.js").exists() and (_VENDOR_DIR / "OrbitControls.js").exists()


# Scene builders are plain three.js. Tokens __THREE_JS__ / __ORBIT_JS__ / __DATA__
# / __COLORS__ / __HEIGHT__ are substituted in Python.
_TEMPLATE = r"""
<div id="twin-wrap" style="position:relative;width:100%;height:__HEIGHT__px;
     border-radius:12px;overflow:hidden;background:radial-gradient(circle at 50% 20%,#1b2735,#0b1016);">
  <div id="twin-badge" style="position:absolute;top:12px;left:14px;z-index:5;
       font-family:system-ui,Segoe UI,Roboto,sans-serif;color:#e8eef5;
       background:rgba(10,16,22,.55);backdrop-filter:blur(4px);padding:8px 12px;border-radius:10px;">
    <div style="font-weight:700;font-size:15px" id="twin-name">asset</div>
    <div style="font-size:12px;opacity:.85" id="twin-sub">risk</div>
    <div style="font-size:11px;opacity:.7;margin-top:2px" id="twin-kind"></div>
  </div>
  <div id="twin-hint" style="position:absolute;bottom:10px;right:14px;z-index:5;
       font-family:system-ui;color:#9fb3c8;font-size:11px;opacity:.8">
    drag to orbit · scroll to zoom · offline (bundled three.js)
  </div>
  <div id="twin-err" style="display:none;position:absolute;inset:0;z-index:6;
       font-family:system-ui;color:#e8eef5;padding:24px;font-size:13px"></div>
</div>
<script>__THREE_JS__</script>
<script>__ORBIT_JS__</script>
<script>
(function(){
  const DATA = __DATA__;
  const COLORS = __COLORS__;
  const KIND = DATA.kind || 'plant_motor';
  function showError(msg){
    const e = document.getElementById('twin-err');
    e.style.display='block';
    e.innerHTML = '<b>3D twin could not initialise.</b><br>'+msg;
  }
  if (typeof THREE === 'undefined'){ showError('three.js global not found'); return; }
  const OrbitControls = THREE.OrbitControls;

  const wrap = document.getElementById('twin-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 120);
  const cam = DATA.camera || {pos:[6.5,3.6,7.5], target:[0,0.4,0]};
  camera.position.set(cam.pos[0], cam.pos[1], cam.pos[2]);

  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio||1));
  wrap.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(cam.target[0], cam.target[1], cam.target[2]);

  scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x20303a, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(5, 8, 6); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.5);
  rim.position.set(-6, 3, -4); scene.add(rim);

  const steel = new THREE.MeshStandardMaterial({color:0x8d97a3, metalness:0.85, roughness:0.35});
  const darkSteel = new THREE.MeshStandardMaterial({color:0x5c656f, metalness:0.9, roughness:0.4});
  const copper = new THREE.MeshStandardMaterial({color:0xb87333, metalness:0.7, roughness:0.5});
  const paint = new THREE.MeshStandardMaterial({color:0x3d5a73, metalness:0.35, roughness:0.55});
  const white = new THREE.MeshStandardMaterial({color:0xdfe7ee, metalness:0.2, roughness:0.6});
  const asphalt = new THREE.MeshStandardMaterial({color:0x2a3038, metalness:0.1, roughness:0.9});
  const sand = new THREE.MeshStandardMaterial({color:0x6b5a3e, metalness:0.05, roughness:0.95});

  function makeHotspot(geometry, extra){
    const mat = new THREE.MeshStandardMaterial({color:0x27ae60, emissive:0x000000, metalness:0.6, roughness:0.3});
    const mesh = new THREE.Mesh(geometry, mat);
    const glowMat = new THREE.MeshBasicMaterial({color:0xff0000, transparent:true, opacity:0.0});
    const glow = new THREE.Mesh(geometry.clone ? geometry.clone() : geometry, glowMat);
    glow.scale.set(1.18, 1.18, 1.18);
    if (extra) extra(mesh, glow);
    return {mesh: mesh, glow: glow, mat: mat, glowMat: glowMat};
  }

  function addGround(y, radius, mat){
    const disc = new THREE.Mesh(new THREE.CircleGeometry(radius, 48),
      new THREE.MeshBasicMaterial({color:0x000000, transparent:true, opacity:0.22}));
    disc.rotation.x = -Math.PI/2; disc.position.y = y; scene.add(disc);
    if (mat){
      const pad = new THREE.Mesh(new THREE.CylinderGeometry(radius*0.55, radius*0.55, 0.08, 36), mat);
      pad.position.y = y + 0.04; scene.add(pad);
    }
  }

  /* scene:plant_motor */
  function buildPlantMotor(){
    const root = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(1.25, 1.25, 3.6, 48), steel);
    body.rotation.z = Math.PI/2; root.add(body);
    for (let i=0;i<10;i++){
      const fin = new THREE.Mesh(new THREE.TorusGeometry(1.26, 0.05, 8, 40), darkSteel);
      fin.rotation.y = Math.PI/2; fin.position.x = -1.6 + i*0.36; root.add(fin);
    }
    const bellGeo = new THREE.CylinderGeometry(1.05, 1.05, 0.5, 48);
    const bellA = new THREE.Mesh(bellGeo, darkSteel); bellA.rotation.z=Math.PI/2; bellA.position.x= 2.0; root.add(bellA);
    const bellB = new THREE.Mesh(bellGeo, darkSteel); bellB.rotation.z=Math.PI/2; bellB.position.x=-2.0; root.add(bellB);
    const tbox = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.7, 0.9), copper);
    tbox.position.set(0, 1.25, 0); root.add(tbox);
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 5.4, 32), steel);
    shaft.rotation.z = Math.PI/2; root.add(shaft);
    const hs = makeHotspot(new THREE.TorusGeometry(0.5, 0.17, 20, 48), function(m,g){
      m.rotation.y = Math.PI/2; m.position.x = 2.35;
      g.rotation.y = Math.PI/2; g.position.x = 2.35;
    });
    root.add(hs.mesh); root.add(hs.glow);
    const base = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.3, 2.4), darkSteel);
    base.position.y = -1.5; root.add(base);
    root.position.y = 0.4;
    scene.add(root);
    addGround(-1.66, 6);
    return {
      hotspotMat: hs.mat, glowMat: hs.glowMat,
      tick: function(){ shaft.rotation.x += 0.04; hs.mesh.rotation.z += 0.04; }
    };
  }

  /* scene:aviation_uav  SIH26054 MALE UAV + aero piston */
  function buildAviationUav(){
    const uav = new THREE.Group();
    const fuse = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.34, 5.4, 28), white);
    fuse.rotation.z = Math.PI/2; uav.add(fuse);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.28, 0.7, 20), white);
    nose.rotation.z = -Math.PI/2; nose.position.x = 3.05; uav.add(nose);
    const wing = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.07, 7.2), paint);
    wing.position.set(-0.2, 0.02, 0); uav.add(wing);
    const wingletL = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.55, 0.35), paint);
    wingletL.position.set(-0.2, 0.28, 3.5); uav.add(wingletL);
    const wingletR = wingletL.clone(); wingletR.position.z = -3.5; uav.add(wingletR);
    const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 2.2, 12), darkSteel);
    boom.rotation.z = Math.PI/2; boom.position.x = -3.4; uav.add(boom);
    const vL = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.7, 0.08), paint);
    vL.position.set(-4.4, 0.35, 0.25); vL.rotation.x = 0.4; uav.add(vL);
    const vR = vL.clone(); vR.position.z = -0.25; vR.rotation.x = -0.4; uav.add(vR);
    const payload = new THREE.Mesh(new THREE.SphereGeometry(0.28, 16, 12), darkSteel);
    payload.position.set(0.6, -0.38, 0); uav.add(payload);

    const engine = new THREE.Group();
    const crank = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.45, 0.7), darkSteel);
    engine.add(crank);
    for (let i=0;i<4;i++){
      const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.45, 14), steel);
      cyl.position.set(-0.28 + (i%2)*0.55, 0.35, (i<2?0.22:-0.22));
      engine.add(cyl);
    }
    const hs = makeHotspot(new THREE.BoxGeometry(1.0, 0.7, 0.85), function(m,g){
      m.position.set(0, 0.05, 0); g.position.set(0, 0.05, 0);
    });
    engine.add(hs.mesh); engine.add(hs.glow);
    engine.position.set(2.35, -0.15, 0);
    uav.add(engine);

    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.2, 12), steel);
    hub.rotation.z = Math.PI/2; hub.position.x = 3.45; uav.add(hub);
    const prop = new THREE.Group();
    for (let i=0;i<3;i++){
      const blade = new THREE.Mesh(new THREE.BoxGeometry(0.06, 1.35, 0.12), darkSteel);
      blade.rotation.z = i * Math.PI*2/3; prop.add(blade);
    }
    prop.position.x = 3.55; uav.add(prop);

    uav.position.y = 0.9;
    scene.add(uav);
    addGround(-1.2, 8);
    return {
      hotspotMat: hs.mat, glowMat: hs.glowMat,
      tick: function(){ prop.rotation.x += 0.28; }
    };
  }

  /* scene:auto_car  generic ICE — not OEM / trim / EV */
  function buildAutoCar(){
    const car = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.75, 1.85), paint);
    body.position.y = 0.55; car.add(body);
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.7, 1.7), white);
    cabin.position.set(-0.25, 1.2, 0); car.add(cabin);
    const hood = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.12, 1.7), paint);
    hood.position.set(1.45, 0.98, 0); car.add(hood);
    function wheel(x,z){
      const w = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.28, 22), asphalt);
      w.rotation.x = Math.PI/2; w.position.set(x, 0.38, z); car.add(w);
      return w;
    }
    const wheels = [wheel(1.45, 0.95), wheel(1.45, -0.95), wheel(-1.45, 0.95), wheel(-1.45, -0.95)];

    const block = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.55, 0.7), darkSteel);
    block.position.set(1.45, 0.85, 0); car.add(block);
    for (let i=0;i<4;i++){
      const cyl = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.22, 10), steel);
      cyl.position.set(1.2 + (i%2)*0.45, 1.18, (i<2?0.16:-0.16)); car.add(cyl);
    }
    const trans = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.28, 0.4), copper);
    trans.position.set(0.7, 0.55, 0); car.add(trans);

    const hs = makeHotspot(new THREE.BoxGeometry(1.05, 0.62, 0.78), function(m,g){
      m.position.set(1.45, 0.88, 0); g.position.set(1.45, 0.88, 0);
    });
    car.add(hs.mesh); car.add(hs.glow);

    scene.add(car);
    addGround(-0.05, 7, asphalt);
    return {
      hotspotMat: hs.mat, glowMat: hs.glowMat,
      tick: function(){ wheels.forEach(function(w){ w.rotation.z += 0.08; }); }
    };
  }

  /* scene:oil_srp  SIH26120 beam pump / sucker rod */
  function buildOilSrp(){
    const rig = new THREE.Group();
    const pad = new THREE.Mesh(new THREE.BoxGeometry(8, 0.18, 4.2), sand);
    pad.position.y = -0.05; rig.add(pad);
    const samson = new THREE.Mesh(new THREE.BoxGeometry(0.28, 4.2, 0.28), darkSteel);
    samson.position.set(-0.2, 2.1, 0); rig.add(samson);
    const brace = new THREE.Mesh(new THREE.BoxGeometry(0.12, 3.4, 0.12), steel);
    brace.position.set(-1.1, 1.5, 0); brace.rotation.z = 0.45; rig.add(brace);

    const beam = new THREE.Group();
    const beamBar = new THREE.Mesh(new THREE.BoxGeometry(5.6, 0.22, 0.28), steel);
    beam.add(beamBar);
    const horse = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.9, 0.18), darkSteel);
    horse.position.set(2.7, -0.35, 0); beam.add(horse);
    const equalizer = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.55, 0.2), copper);
    equalizer.position.set(-2.55, -0.15, 0); beam.add(equalizer);
    beam.position.set(-0.2, 4.15, 0);
    rig.add(beam);

    const pitman = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 2.4, 10), steel);
    pitman.position.set(-2.4, 2.4, 0); pitman.rotation.z = 0.35; rig.add(pitman);
    const crank = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.14, 0.14), darkSteel);
    crank.position.set(-2.6, 1.15, 0.4); rig.add(crank);
    const gearbox = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.8, 0.9), darkSteel);
    gearbox.position.set(-2.5, 0.55, 0); rig.add(gearbox);
    const prime = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 1.1, 20), steel);
    prime.rotation.z = Math.PI/2; prime.position.set(-3.5, 0.55, 0); rig.add(prime);

    const wellhead = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.28, 0.7, 16), steel);
    wellhead.position.set(2.55, 0.4, 0); rig.add(wellhead);
    const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 3.4, 8), steel);
    rod.position.set(2.55, 2.2, 0); rig.add(rod);

    const hs = makeHotspot(new THREE.BoxGeometry(1.4, 0.9, 1.0), function(m,g){
      m.position.set(-2.5, 0.55, 0); g.position.set(-2.5, 0.55, 0);
    });
    rig.add(hs.mesh); rig.add(hs.glow);

    scene.add(rig);
    addGround(-0.2, 10, sand);
    return {
      hotspotMat: hs.mat, glowMat: hs.glowMat,
      tick: function(t){
        const a = 0.18 * Math.sin(t * 1.4);
        beam.rotation.z = a;
        rod.position.y = 2.2 - 0.55 * Math.sin(t * 1.4);
      }
    };
  }

  const builders = {
    plant_motor: buildPlantMotor,
    aviation_uav: buildAviationUav,
    auto_car: buildAutoCar,
    oil_srp: buildOilSrp
  };
  const rig = (builders[KIND] || buildPlantMotor)();

  let current = {risk:"Unknown", speed:0, blink:false, hex:"#7f8c8d"};
  function applyState(st){
    const spec = COLORS[st.risk_level] || COLORS["Unknown"];
    current = {risk: st.risk_level, speed: spec.speed, blink: spec.blink, hex: spec.hex};
    const col = new THREE.Color(spec.hex);
    rig.hotspotMat.color.set(col); rig.hotspotMat.emissive.set(col); rig.glowMat.color.set(col);
    const rul = (st.predicted_rul_days===null||st.predicted_rul_days===undefined) ? "n/a" : st.predicted_rul_days;
    document.getElementById('twin-name').textContent = st.machine_id;
    document.getElementById('twin-sub').innerHTML =
       'risk <b style="color:'+spec.hex+'">'+st.risk_level+'</b> · predicted RUL '+rul+' d';
    const kindEl = document.getElementById('twin-kind');
    kindEl.textContent = (DATA.pack_label || KIND) + ' · hotspot: ' + (DATA.hotspot || 'health');
  }
  applyState(DATA.selected || {machine_id:"asset", risk_level:"Unknown", predicted_rul_days:null});

  const clock = new THREE.Clock();
  function animate(){
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();
    if (rig.tick) rig.tick(t);
    if (current.blink && current.speed>0){
      const pulse = 0.5 + 0.5*Math.sin(t*current.speed);
      rig.hotspotMat.emissiveIntensity = 0.15 + 1.15*pulse;
      rig.glowMat.opacity = 0.15 + 0.5*pulse;
    } else {
      rig.hotspotMat.emissiveIntensity = 0.25; rig.glowMat.opacity = 0.0;
    }
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', ()=>{
    const w = wrap.clientWidth, h = wrap.clientHeight;
    camera.aspect = w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h);
  });
})();
</script>
"""


def build_twin_html(
    assets: list[dict[str, Any]],
    selected_id: Optional[str] = None,
    height: int = 520,
    kind: Optional[str] = None,
    hotspot: Optional[str] = None,
    pack_label: Optional[str] = None,
) -> str:
    """Return a self-contained (offline) HTML/JS 3D twin for st.components.v1.html."""
    twin_kind = normalize_twin_kind(kind)
    norm: list[dict[str, Any]] = []
    for a in assets or []:
        norm.append(
            {
                "machine_id": str(a.get("machine_id", "asset")),
                "risk_level": normalize_risk(a.get("risk_level")),
                "predicted_rul_days": a.get("predicted_rul_days"),
            }
        )
    selected = None
    if selected_id is not None:
        selected = next((a for a in norm if a["machine_id"] == str(selected_id)), None)
    if selected is None:
        selected = norm[0] if norm else {
            "machine_id": "demo-asset",
            "risk_level": "Unknown",
            "predicted_rul_days": None,
        }
    payload = {
        "assets": norm,
        "selected": selected,
        "kind": twin_kind,
        "hotspot": hotspot or "health",
        "pack_label": pack_label or twin_kind,
        "camera": _CAMERA.get(twin_kind, _CAMERA[DEFAULT_TWIN_KIND]),
    }
    three_js, orbit_js = _vendor_js()
    return (
        _TEMPLATE.replace("__THREE_JS__", three_js)
        .replace("__ORBIT_JS__", orbit_js)
        .replace("__DATA__", json.dumps(payload))
        .replace("__COLORS__", json.dumps(RISK_COLORS))
        .replace("__HEIGHT__", str(int(height)))
    )
