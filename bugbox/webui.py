"""Small dashboard (standard library only): live counts, latest annotated image, mode switch, capture button."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .util import log

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>bugbox</title><style>
:root{--bg:#f6f7f4;--fg:#1d2321;--card:#fff;--mut:#667;--acc:#2f7d4f;--bad:#b3261e;--line:#dde2dc}
@media (prefers-color-scheme:dark){:root{--bg:#161a18;--fg:#e8ece9;--card:#202623;--mut:#9aa6a0;--acc:#5fc28a;--bad:#ff8a80;--line:#333b37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}.pill{padding:3px 10px;border-radius:999px;background:var(--acc);color:#fff;font-size:12px;text-transform:uppercase}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:16px;grid-template-columns:1fr}
@media(min-width:900px){main{grid-template-columns:1.2fr 1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.counts{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px}
.c{text-align:center;padding:10px;border-radius:10px;border:1px solid var(--line)}.c b{display:block;font-size:40px;line-height:1.1}
.c span{color:var(--mut);font-size:12px}.c.total{background:var(--acc);color:#fff;border:0}.c.total span{color:#fff}
img{width:100%;border-radius:10px;border:1px solid var(--line);display:block;background:#0002;min-height:120px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}button:disabled{opacity:.5}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0}
dt{color:var(--mut)}dd{margin:0;word-break:break-word}.err{color:var(--bad);margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
</style></head><body>
<header><h1 id="title">bugbox</h1><span class="pill" id="mode">...</span></header>
<main><div>
 <div class="card"><div class="counts" id="counts"></div><div class="err" id="err"></div></div>
 <div class="card" style="margin-top:16px"><img id="img" alt="latest annotated image">
 <div class="row"><button id="cap">Capture and count now</button>
 <button id="m_standalone">Standalone (all on the Pi)</button><button id="m_hybrid">Hybrid (Pi detects, cloud names)</button><button id="m_cloud" style="display:none">Cloud (all in Roboflow)</button>
 <button id="lt">Test lights</button></div></div></div>
<div><div class="card"><dl id="info"></dl></div>
 <div class="card" style="margin-top:16px"><b>Recent</b><table id="recent"></table></div></div></main>
<script>
const $=id=>document.getElementById(id);let lastUpd="",lights=false;
async function j(u,o){const r=await fetch(u,o);return r.json()}
function post(u,b){return j(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b||{})})}
function render(s){
 $("title").textContent="bugbox - "+s.name;$("mode").textContent=s.mode+" / "+s.style;
 $("m_standalone").className=s.mode==="standalone"?"on":"";$("m_hybrid").className=s.mode==="hybrid"?"on":"";$("m_cloud").style.display=s.mode==="cloud"?"":"none";$("m_cloud").className=s.mode==="cloud"?"on":"";
 let h="";for(const t of s.targets){h+=`<div class="c"><b>${s.counts[t]??"-"}</b><span>${t}</span></div>`}
 h+=`<div class="c"><b>${s.counts[s.other_label]??"-"}</b><span>${s.other_label}</span></div>`;
 h+=`<div class="c total"><b>${s.total??"-"}</b><span>TOTAL</span></div>`;$("counts").innerHTML=h;
 $("err").textContent=s.error?("! "+s.error):"";
 const usb=s.usb?`${s.usb.label} (${s.usb.free_gb} GB free)`:(s.usb_required?"NOT FOUND (required)":"none");
 const rows=[["Updated",s.updated||"-"],["Backend",(s.backend||"-")+" "+(s.model_id||"")],["Latency",s.latency_s!=null?s.latency_s+" s":"-"],
 ["USB drive",usb],["Photos saved",s.photos_saved],["Camera",s.camera],["Lights",s.lights_on?"on":"off"],
 ["Schedule",s.schedule_active?("running"+(s.schedule_end?" until "+s.schedule_end:"")):("waiting, next "+(s.next_session||"?"))],
 ["Uptime",Math.round(s.uptime_s/60)+" min"],["Errors",s.errors]];
 const k=s.classifier;if(k)rows.splice(3,0,["Cloud classifier",k.enabled?(k.calls_today+" calls today, "+k.budget_left+" left, "+k.cache_hits+" cache hits, "+k.relabelled+" relabelled"):("off"+(k.warning?(" ("+k.warning+")"):""))]);
 if(s.simulated)rows.unshift(["SIMULATION","fake camera and detector"]);
 $("info").innerHTML=rows.map(r=>`<dt>${r[0]}</dt><dd>${r[1]}</dd>`).join("");
 $("recent").innerHTML="<tr><th>time</th><th>src</th><th>total</th></tr>"+(s.recent||[]).map(r=>`<tr><td>${r.when.slice(11)}</td><td>${r.source}</td><td>${r.total}</td></tr>`).join("");
 if(s.updated&&s.updated!==lastUpd){lastUpd=s.updated;$("img").src="/latest.jpg?"+Date.now()}
}
async function tick(){try{render(await j("/api/status"))}catch(e){$("err").textContent="no connection"}}
$("cap").onclick=async()=>{$("cap").disabled=true;$("cap").textContent="Working...";
 const r=await post("/api/capture");$("cap").disabled=false;$("cap").textContent="Capture and count now";if(r.error)$("err").textContent="! "+r.error;tick()};
$("m_standalone").onclick=async()=>{await post("/api/mode",{mode:"standalone"});tick()};
$("m_hybrid").onclick=async()=>{await post("/api/mode",{mode:"hybrid"});tick()};
$("m_cloud").onclick=async()=>{await post("/api/mode",{mode:"cloud"});tick()};
$("lt").onclick=async()=>{lights=!lights;await post("/api/lights",{on:lights});$("lt").className=lights?"on":"";tick()};
tick();setInterval(tick,1500);
</script></body></html>"""


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return {}

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send(200, engine.status())
            elif path == "/latest.jpg":
                jpg = engine.latest_jpeg()
                if jpg:
                    self._send(200, jpg, "image/jpeg")
                else:
                    self._send(204, b"", "image/jpeg")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            body = self._body()
            try:
                if path == "/api/capture":
                    r = engine.capture_and_count("manual")
                    self._send(200, {k: v for k, v in r.to_dict().items() if k != "detections"})
                elif path == "/api/mode":
                    engine.set_mode(body.get("mode", ""))
                    self._send(200, {"mode": engine.current_mode()})
                elif path == "/api/lights":
                    engine.lights.enabled = True
                    engine.lights.on() if body.get("on") else engine.lights.off()
                    self._send(200, {"lights_on": engine.lights.is_on})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                log.error("web: %s failed: %s", path, e)
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def start(engine, host: str, port: int):
    srv = ThreadingHTTPServer((host, port), make_handler(engine))
    th = threading.Thread(target=srv.serve_forever, daemon=True, name="web")
    th.start()
    log.info("dashboard on http://%s:%d", host if host != "0.0.0.0" else "<this-pi>", port)
    return srv
