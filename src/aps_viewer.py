"""Upgrade 3 — Autodesk Platform Services (APS / Forge) CAD viewer.

Embeds the APS Viewer to show a *real* translated CAD model (Revit/Fusion/IFC →
SVF) of an asset, and tints/pulses it red when the selected asset's predicted
risk is High. This is **availability-gated**: without `APS_CLIENT_ID` +
`APS_CLIENT_SECRET` (and a translated model `URN`) the feature simply shows
setup instructions, so the app and the Render deploy are never affected.

Credentials are read from environment secrets and never written to disk or the
page beyond the short-lived viewer access token that the APS Viewer requires.
"""

from __future__ import annotations

import os
from typing import Any

APS_AUTH_URL = "https://developer.api.autodesk.com/authentication/v2/token"


def aps_available() -> tuple[bool, str]:
    """True when APS client credentials are present in the environment."""
    cid = (os.getenv("APS_CLIENT_ID") or "").strip()
    sec = (os.getenv("APS_CLIENT_SECRET") or "").strip()
    if not cid or not sec:
        return False, "APS_CLIENT_ID / APS_CLIENT_SECRET not set"
    return True, "APS credentials detected"


def get_access_token(scope: str = "viewables:read data:read") -> dict[str, Any]:
    """2-legged OAuth token for the APS Viewer (client_credentials)."""
    import requests

    cid = (os.getenv("APS_CLIENT_ID") or "").strip()
    sec = (os.getenv("APS_CLIENT_SECRET") or "").strip()
    if not cid or not sec:
        raise RuntimeError("APS_CLIENT_ID / APS_CLIENT_SECRET not set")
    resp = requests.post(
        APS_AUTH_URL,
        data={"grant_type": "client_credentials", "scope": scope},
        auth=(cid, sec),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


_VIEWER_TEMPLATE = """
<link rel="stylesheet"
  href="https://developer.api.autodesk.com/modelderivative/v2/viewers/7.*/style.min.css" type="text/css">
<div id="aps-wrap" style="position:relative;width:100%;height:__HEIGHT__px;border-radius:12px;overflow:hidden;">
  <div id="aps-badge" style="position:absolute;top:10px;left:12px;z-index:5;font-family:system-ui;
       color:#fff;background:rgba(10,16,22,.6);padding:6px 10px;border-radius:8px;">
    <b id="aps-name">__ASSET__</b> · risk
    <b id="aps-risk" style="color:__RISK_HEX__">__RISK__</b>
  </div>
  <div id="apsViewer" style="width:100%;height:100%"></div>
  <div id="aps-err" style="display:none;position:absolute;inset:0;z-index:6;color:#fff;
       font-family:system-ui;padding:20px;font-size:13px;background:#101822"></div>
</div>
<script src="https://developer.api.autodesk.com/modelderivative/v2/viewers/7.*/viewer3D.min.js"></script>
<script>
(function(){
  var TOKEN="__TOKEN__", URN="__URN__", RISK="__RISK__";
  function fail(m){var e=document.getElementById('aps-err');e.style.display='block';
    e.innerHTML='<b>APS Viewer error.</b><br>'+m;}
  if (typeof Autodesk==='undefined'){ fail('APS Viewer script failed to load (needs internet to Autodesk).'); return; }
  Autodesk.Viewing.Initializer({env:'AutodeskProduction', accessToken:TOKEN}, function(){
    var viewer = new Autodesk.Viewing.GuiViewer3D(document.getElementById('apsViewer'));
    viewer.start();
    Autodesk.Viewing.Document.load('urn:'+URN, function(doc){
      var node = doc.getRoot().getDefaultGeometry();
      viewer.loadDocumentNode(doc, node).then(function(){
        if (RISK==='High' || RISK==='Medium'){
          viewer.addEventListener(Autodesk.Viewing.GEOMETRY_LOADED_EVENT, function(){
            try{
              var color = RISK==='High' ? new THREE.Vector4(0.9,0.1,0.1,0.7)
                                        : new THREE.Vector4(0.95,0.6,0.1,0.55);
              var tree = viewer.model.getInstanceTree();
              var ids=[]; tree.enumNodeChildren(tree.getRootId(), function(id){ids.push(id);}, true);
              var on=true;
              setInterval(function(){
                ids.forEach(function(id){ on ? viewer.setThemingColor(id,color) : viewer.clearThemingColor(id); });
                viewer.impl.invalidate(true); on=!on;
              }, RISK==='High'?450:900);
            }catch(e){/* theming best-effort */}
          });
        }
      });
    }, function(err){ fail('Model load failed: '+JSON.stringify(err)); });
  });
})();
</script>
"""

_RISK_HEX = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60", "Unknown": "#7f8c8d"}


def build_viewer_html(token: str, urn: str, *, asset: str, risk: str, height: int = 560) -> str:
    """Embed the APS Viewer for a translated model URN, tinting red/amber on risk."""
    risk = risk if risk in _RISK_HEX else "Unknown"
    return (
        _VIEWER_TEMPLATE.replace("__TOKEN__", token)
        .replace("__URN__", urn)
        .replace("__ASSET__", str(asset))
        .replace("__RISK__", risk)
        .replace("__RISK_HEX__", _RISK_HEX[risk])
        .replace("__HEIGHT__", str(int(height)))
    )
