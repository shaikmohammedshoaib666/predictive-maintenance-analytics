"""Layer 3 / Upgrade 1 — offline 3D digital twin.

Renders a rotatable 3D motor (three.js) whose drive-end bearing turns red and
blinks when the selected asset's predicted risk is High (amber for Medium,
green for Low). three.js + OrbitControls are **vendored locally** (see
``src/vendor/three``) and inlined into the component, so the twin needs no
internet/CDN access. The twin is asset-agnostic: it is driven by whatever risk
/ RUL the pipeline produced for the machine the user selects.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_VENDOR_DIR = Path(__file__).parent / "vendor" / "three"

RISK_COLORS = {
    "High": {"hex": "#e74c3c", "blink": True, "speed": 6.0},
    "Medium": {"hex": "#f39c12", "blink": True, "speed": 2.2},
    "Low": {"hex": "#27ae60", "blink": False, "speed": 0.0},
    "Unknown": {"hex": "#7f8c8d", "blink": False, "speed": 0.0},
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


_TEMPLATE = """
<div id="twin-wrap" style="position:relative;width:100%;height:__HEIGHT__px;
     border-radius:12px;overflow:hidden;background:radial-gradient(circle at 50% 20%,#1b2735,#0b1016);">
  <div id="twin-badge" style="position:absolute;top:12px;left:14px;z-index:5;
       font-family:system-ui,Segoe UI,Roboto,sans-serif;color:#e8eef5;
       background:rgba(10,16,22,.55);backdrop-filter:blur(4px);padding:8px 12px;border-radius:10px;">
    <div style="font-weight:700;font-size:15px" id="twin-name">asset</div>
    <div style="font-size:12px;opacity:.85" id="twin-sub">risk</div>
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
  const camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 100);
  camera.position.set(6.5, 3.6, 7.5);

  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio||1));
  wrap.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0.4, 0);

  scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x20303a, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(5, 8, 6); scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.5);
  rim.position.set(-6, 3, -4); scene.add(rim);

  const motor = new THREE.Group();
  const steel = new THREE.MeshStandardMaterial({color:0x8d97a3, metalness:0.85, roughness:0.35});
  const darkSteel = new THREE.MeshStandardMaterial({color:0x5c656f, metalness:0.9, roughness:0.4});
  const copper = new THREE.MeshStandardMaterial({color:0xb87333, metalness:0.7, roughness:0.5});

  const body = new THREE.Mesh(new THREE.CylinderGeometry(1.25, 1.25, 3.6, 48), steel);
  body.rotation.z = Math.PI/2; motor.add(body);
  for (let i=0;i<10;i++){
    const fin = new THREE.Mesh(new THREE.TorusGeometry(1.26, 0.05, 8, 40), darkSteel);
    fin.rotation.y = Math.PI/2; fin.position.x = -1.6 + i*0.36; motor.add(fin);
  }
  const bellGeo = new THREE.CylinderGeometry(1.05, 1.05, 0.5, 48);
  const bellA = new THREE.Mesh(bellGeo, darkSteel); bellA.rotation.z=Math.PI/2; bellA.position.x= 2.0; motor.add(bellA);
  const bellB = new THREE.Mesh(bellGeo, darkSteel); bellB.rotation.z=Math.PI/2; bellB.position.x=-2.0; motor.add(bellB);
  const tbox = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.7, 0.9), copper);
  tbox.position.set(0, 1.25, 0); motor.add(tbox);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 5.4, 32), steel);
  shaft.rotation.z = Math.PI/2; motor.add(shaft);

  const bearingMat = new THREE.MeshStandardMaterial({color:0x27ae60, emissive:0x000000, metalness:0.6, roughness:0.3});
  const bearing = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.17, 20, 48), bearingMat);
  bearing.rotation.y = Math.PI/2; bearing.position.x = 2.35; motor.add(bearing);
  const glowMat = new THREE.MeshBasicMaterial({color:0xff0000, transparent:true, opacity:0.0});
  const glow = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.26, 20, 48), glowMat);
  glow.rotation.y = Math.PI/2; glow.position.x = 2.35; motor.add(glow);

  const base = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.3, 2.4), darkSteel);
  base.position.y = -1.5; motor.add(base);
  motor.position.y = 0.4;
  scene.add(motor);

  const disc = new THREE.Mesh(new THREE.CircleGeometry(6, 48),
    new THREE.MeshBasicMaterial({color:0x000000, transparent:true, opacity:0.22}));
  disc.rotation.x = -Math.PI/2; disc.position.y = -1.66; scene.add(disc);

  let current = {risk:"Unknown", speed:0, blink:false, hex:"#7f8c8d"};
  function applyState(st){
    const spec = COLORS[st.risk_level] || COLORS["Unknown"];
    current = {risk: st.risk_level, speed: spec.speed, blink: spec.blink, hex: spec.hex};
    const col = new THREE.Color(spec.hex);
    bearingMat.color.set(col); bearingMat.emissive.set(col); glowMat.color.set(col);
    const rul = (st.predicted_rul_days===null||st.predicted_rul_days===undefined) ? "n/a" : st.predicted_rul_days;
    document.getElementById('twin-name').textContent = st.machine_id;
    document.getElementById('twin-sub').innerHTML =
       'risk <b style="color:'+spec.hex+'">'+st.risk_level+'</b> · predicted RUL '+rul+' d';
  }
  applyState(DATA.selected || {machine_id:"asset", risk_level:"Unknown", predicted_rul_days:null});

  const clock = new THREE.Clock();
  function animate(){
    requestAnimationFrame(animate);
    const t = clock.getElapsedTime();
    shaft.rotation.x += 0.04; bearing.rotation.z += 0.04;
    if (current.blink && current.speed>0){
      const pulse = 0.5 + 0.5*Math.sin(t*current.speed);
      bearingMat.emissiveIntensity = 0.15 + 1.15*pulse;
      glowMat.opacity = 0.15 + 0.5*pulse;
    } else {
      bearingMat.emissiveIntensity = 0.25; glowMat.opacity = 0.0;
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
) -> str:
    """Return a self-contained (offline) HTML/JS 3D twin for st.components.v1.html."""
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
    payload = {"assets": norm, "selected": selected}
    three_js, orbit_js = _vendor_js()
    return (
        _TEMPLATE.replace("__THREE_JS__", three_js)
        .replace("__ORBIT_JS__", orbit_js)
        .replace("__DATA__", json.dumps(payload))
        .replace("__COLORS__", json.dumps(RISK_COLORS))
        .replace("__HEIGHT__", str(int(height)))
    )
