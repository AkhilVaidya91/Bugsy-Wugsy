"""
Bugsy-Wugsy analytics dashboard.

Reads the directory written by server/receiver.py (counts.csv + JPEGs) and
serves a live dashboard. Whenever counts.csv or the directory changes, the
browser is told over Server-Sent Events and refreshes itself.

    pip install flask
    python dashboard.py --dir received --port 5001

Then open http://localhost:5001
"""
import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

app = Flask(__name__)
OUT_DIR = Path("received")
CSV_NAME = "counts.csv"
IMG_EXT = (".jpg", ".jpeg", ".png")

_cache = {"sig": None, "data": None}


def signature():
    """Cheap fingerprint of the directory state: changes on any append or new file."""
    try:
        st = (OUT_DIR / CSV_NAME).stat()
        csv_sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        csv_sig = (0, 0)
    try:
        dir_sig = OUT_DIR.stat().st_mtime_ns
    except OSError:
        dir_sig = 0
    return csv_sig, dir_sig


def load():
    sig = signature()
    if _cache["sig"] == sig and _cache["data"] is not None:
        return _cache["data"]

    classes, rows = [], []
    path = OUT_DIR / CSV_NAME
    if path.exists():
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            classes = [c for c in (reader.fieldnames or []) if c.startswith("count_")]
            for r in reader:
                try:
                    ts = datetime.strptime(r["timestamp"], "%Y%m%d_%H%M%S_%f")
                    counts = [int(float(r[c] or 0)) for c in classes]
                    rows.append([ts.isoformat(timespec="milliseconds"),
                                 r["image_filename"], counts])
                except (KeyError, TypeError, ValueError):
                    continue  # partial line while the receiver is mid-write
    rows.sort(key=lambda x: x[0])

    files, size = 0, 0
    if OUT_DIR.exists():
        with os.scandir(OUT_DIR) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(IMG_EXT):
                    files += 1
                    size += e.stat().st_size

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0

    data = {
        "classes": [c[len("count_"):] for c in classes],
        "rows": rows,
        "files": files,
        "mb": round(size / 1048576, 1),
        "mtime": mtime,
        "dir": str(OUT_DIR.resolve()),
    }
    _cache["sig"], _cache["data"] = sig, data
    return data


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/data")
def api_data():
    d = dict(load())
    d["now"] = time.time()
    resp = jsonify(d)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/images/<path:name>")
def image(name):
    return send_from_directory(OUT_DIR.resolve(), name)


@app.route("/events")
def events():
    def stream():
        last = signature()
        yield "retry: 2000\n\ndata: hello\n\n"
        idle = 0
        while True:
            time.sleep(0.5)
            cur = signature()
            if cur != last:
                last = cur
                idle = 0
                yield "data: change\n\n"
            else:
                idle += 1
                if idle >= 30:  # keep-alive every ~15 s
                    idle = 0
                    yield ": ping\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bugsy-Wugsy Analytics</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e5e7eb;--ink:#1a1f2b;--mute:#6b7280;--accent:#2f5bea}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:24px 24px 56px}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:20px}
h1{font-size:19px;font-weight:650;margin:0;letter-spacing:-.01em}
.sub{color:var(--mute);font-size:13px}
.right{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.live{display:inline-flex;align-items:center;gap:8px;color:var(--mute);font-size:13px}
.dot{width:8px;height:8px;border-radius:50%;background:#aab0bb}
.dot.on{background:#16a34a;box-shadow:0 0 0 3px rgba(22,163,74,.16)}
.dot.idle{background:#d97706}
.seg{display:inline-flex;background:#eceef2;border-radius:8px;padding:3px}
.seg button{border:0;background:none;padding:5px 12px;border-radius:6px;font:inherit;color:var(--mute);cursor:pointer}
.seg button:hover{color:var(--ink)}
.seg button.on{background:#fff;color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.08)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
main{display:grid;gap:16px}
.grid{display:grid;gap:16px}
.kpis{grid-template-columns:repeat(6,minmax(0,1fr))}
.g21{grid-template-columns:minmax(0,2fr) minmax(0,1fr)}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
.g12{grid-template-columns:minmax(0,1fr) minmax(0,2fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}
.ch{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px}
h2{font-size:13px;font-weight:600;margin:0}
.hint{color:var(--mute);font-size:12px}
.kpi .l{color:var(--mute);font-size:12px}
.kpi .v{font-size:26px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums;margin:2px 0}
.kpi .s{color:var(--mute);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chart{position:relative;height:280px}
.chart.short{height:190px}
.notice{background:#fff8e6;border:1px solid #f3dfae;color:#7a5b0a;border-radius:10px;padding:12px 16px}
.donutwrap{position:relative;height:170px}
.center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none}
.center .big{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.center .sm{color:var(--mute);font-size:12px}
.legend{margin-top:14px;display:grid;gap:6px}
.legend div{display:grid;grid-template-columns:12px 1fr auto auto;gap:8px;align-items:center;font-variant-numeric:tabular-nums}
.sw{width:10px;height:10px;border-radius:3px}
.legend .p{color:var(--mute);min-width:44px;text-align:right}
.frame{aspect-ratio:4/3;background:#eceef2;border-radius:8px;overflow:hidden;cursor:pointer}
.frame img{width:100%;height:100%;object-fit:contain;display:block}
.cap{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;color:var(--mute);font-size:12px}
.chip{display:inline-flex;align-items:center;gap:5px;color:var(--ink);font-variant-numeric:tabular-nums}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}
.tile{position:relative;aspect-ratio:4/3;border-radius:8px;overflow:hidden;background:#eceef2;cursor:pointer;border:0;padding:0}
.tile img{width:100%;height:100%;object-fit:cover;display:block}
.tile .b{position:absolute;top:6px;right:6px;background:rgba(26,31,43,.78);color:#fff;font-size:11px;border-radius:10px;padding:1px 7px;font-variant-numeric:tabular-nums}
.tile .t{position:absolute;left:0;right:0;bottom:0;padding:14px 8px 5px;font-size:11px;color:#fff;background:linear-gradient(transparent,rgba(0,0,0,.55))}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--mute);font-weight:500;font-size:12px;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:8px;border-bottom:1px solid #f0f1f4;white-space:nowrap}
th.n,td.n{text-align:right}
tbody tr{cursor:pointer}
tbody tr:hover{background:#f8f9fb}
td.f{color:var(--mute);max-width:260px;overflow:hidden;text-overflow:ellipsis}
td.z{color:#b6bcc7}
.scroll{overflow-x:auto}
dialog{border:0;border-radius:10px;padding:0;max-width:min(92vw,1100px);overflow:hidden}
dialog::backdrop{background:rgba(20,24,32,.55)}
dialog img{display:block;max-width:100%;max-height:78vh;margin:0 auto}
#lbcap{padding:12px 16px;color:var(--mute);font-size:13px}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.g4{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:760px){.wrap{padding:16px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.g21,.g12,.g4{grid-template-columns:minmax(0,1fr)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Bugsy-Wugsy</h1>
      <div class="sub">Insect detection analytics</div>
    </div>
    <div class="right">
      <div class="live"><span id="dot" class="dot"></span><span id="status">Connecting</span></div>
      <div class="seg" id="range">
        <button data-r="1h">1h</button><button data-r="6h">6h</button><button data-r="24h">24h</button><button data-r="7d">7d</button><button data-r="all">All</button>
      </div>
    </div>
  </header>

  <main>
    <div id="empty" class="notice" hidden></div>
    <section class="grid kpis" id="kpis"></section>

    <section class="grid g21">
      <div class="card">
        <div class="ch"><h2>Insects over time</h2><span class="hint" id="bucketHint"></span></div>
        <div class="chart"><canvas id="cTime"></canvas></div>
      </div>
      <div class="card">
        <div class="ch"><h2>Class mix</h2><span class="hint" id="mixHint"></span></div>
        <div class="donutwrap"><canvas id="cMix"></canvas>
          <div class="center"><div class="big" id="mixTotal">0</div><div class="sm">insects</div></div>
        </div>
        <div class="legend" id="legend"></div>
      </div>
    </section>

    <section class="grid g4">
      <div class="card"><div class="ch"><h2>Cumulative insects</h2></div><div class="chart short"><canvas id="cCum"></canvas></div></div>
      <div class="card"><div class="ch"><h2>Frames saved</h2><span class="hint">capture activity</span></div><div class="chart short"><canvas id="cFrames"></canvas></div></div>
      <div class="card"><div class="ch"><h2>Avg per frame by hour</h2></div><div class="chart short"><canvas id="cHour"></canvas></div></div>
      <div class="card"><div class="ch"><h2>Insects per frame</h2><span class="hint">number of frames</span></div><div class="chart short"><canvas id="cHist"></canvas></div></div>
    </section>

    <section class="grid g12">
      <div class="card">
        <div class="ch"><h2>Latest frame</h2></div>
        <div class="frame" id="latest"></div>
        <div class="cap" id="latestCap"></div>
      </div>
      <div class="card">
        <div class="ch"><h2>Recent frames</h2><span class="hint">click to enlarge</span></div>
        <div class="gallery" id="gallery"></div>
      </div>
    </section>

    <section class="card">
      <div class="ch">
        <h2>Frame log</h2>
        <div class="seg" id="tabs"><button data-t="recent" class="on">Recent</button><button data-t="busiest">Busiest</button></div>
      </div>
      <div class="scroll"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
    </section>
  </main>
</div>

<dialog id="lb"><img id="lbimg" alt=""><div id="lbcap"></div></dialog>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
(function () {
  var PAL = ['#2f5bea', '#14a38b', '#f2a33a', '#c2538f', '#7c5cf0', '#0ea5e9'];
  var RANGES = { '1h': 36e5, '6h': 216e5, '24h': 864e5, '7d': 6048e5, 'all': Infinity };
  var RLABEL = { '1h': 'last hour', '6h': 'last 6 hours', '24h': 'last 24 hours', '7d': 'last 7 days', 'all': 'all time' };
  var STEPS = [60e3, 300e3, 600e3, 900e3, 1800e3, 36e5, 108e5, 216e5, 432e5, 864e5];
  var $ = function (s) { return document.querySelector(s); };

  var D = null, fetchedAt = 0, connected = false, tab = 'recent', meta = [], byFile = {}, keys = {};
  var range = 'all';
  try { range = localStorage.getItem('range') || '24h'; } catch (e) { range = '24h'; }
  if (!(range in RANGES)) range = '24h';

  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#6b7280';
  Chart.defaults.animation = false;

  /* ---------- helpers ---------- */
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function fmt(n) { return Number(n).toLocaleString(); }
  function p2(n) { return n < 10 ? '0' + n : '' + n; }
  function dur(ms) {
    var s = Math.floor(ms / 1000);
    if (s < 60) return s + 's';
    var m = Math.floor(s / 60); if (m < 60) return m + 'm';
    var h = Math.floor(m / 60); if (h < 48) return h + 'h ' + (m % 60) + 'm';
    return Math.floor(h / 24) + 'd ' + (h % 24) + 'h';
  }
  function stepName(ms) { return ms >= 864e5 ? '1 day' : ms >= 36e5 ? (ms / 36e5) + ' h' : (ms / 6e4) + ' min'; }
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function stamp(t) { var d = new Date(t); return MON[d.getMonth()] + ' ' + d.getDate() + ', ' + p2(d.getHours()) + ':' + p2(d.getMinutes()) + ':' + p2(d.getSeconds()); }
  function label(t, step, span) {
    var d = new Date(t);
    if (step >= 864e5) return MON[d.getMonth()] + ' ' + d.getDate();
    var hm = p2(d.getHours()) + ':' + p2(d.getMinutes());
    return span > 864e5 ? MON[d.getMonth()] + ' ' + d.getDate() + ' ' + hm : hm;
  }
  function pickStep(span) { for (var i = 0; i < STEPS.length; i++) if (span / STEPS[i] <= 60) return STEPS[i]; return STEPS[STEPS.length - 1]; }
  function imgUrl(f) { return '/images/' + encodeURIComponent(f); }

  /* ---------- charts ---------- */
  var charts = {};
  var grid = { color: '#eef0f3' };
  function xAxis(extra) { return Object.assign({ grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0, autoSkip: true } }, extra || {}); }
  function yAxis(extra) { return Object.assign({ beginAtZero: true, grid: grid, border: { display: false }, ticks: { precision: 0 } }, extra || {}); }
  function put(id, type, labels, datasets, options) {
    var c = charts[id];
    if (!c) {
      options = options || {};
      options.responsive = true;
      options.maintainAspectRatio = false;
      options.interaction = options.interaction || { mode: 'index', intersect: false };
      options.plugins = Object.assign({ legend: { display: false } }, options.plugins || {});
      charts[id] = new Chart(document.getElementById(id), { type: type, data: { labels: labels, datasets: datasets }, options: options });
    } else {
      c.data.labels = labels; c.data.datasets = datasets; c.update('none');
    }
  }
  function bars(color, data) { return [{ data: data, backgroundColor: color, borderRadius: 2, maxBarThickness: 28 }]; }

  /* ---------- render ---------- */
  function render() {
    if (!D) return;
    var all = D.rows, K = D.classes.length;
    var tEnd = all.length ? all[all.length - 1].t : Date.now();
    var lim = RANGES[range];
    var rows = lim === Infinity ? all : all.filter(function (r) { return r.t >= tEnd - lim; });
    var tStart = rows.length ? rows[0].t : tEnd;
    var span = Math.max(tEnd - tStart, 0);

    var empty = $('#empty');
    if (!all.length) { empty.hidden = false; empty.textContent = 'No data yet. Waiting for ' + D.dir + '/counts.csv'; }
    else empty.hidden = true;

    /* totals */
    var totals = new Array(K).fill(0), sum = 0, withIns = 0, peak = null;
    rows.forEach(function (r) {
      r.c.forEach(function (v, i) { totals[i] += v; });
      sum += r.n;
      if (r.n > 0) withIns++;
      if (!peak || r.n > peak.n) peak = r;
    });
    var top = 0; totals.forEach(function (v, i) { if (v > totals[top]) top = i; });
    var n = rows.length;

    var kp = [
      ['Insects detected', fmt(sum), RLABEL[range]],
      ['Frames saved', fmt(n), fmt(D.files) + ' images, ' + D.mb + ' MB on disk'],
      ['Average per frame', n ? (sum / n).toFixed(2) : '0', peak ? 'peak ' + peak.n + ' in one frame' : 'no frames'],
      ['Frames with insects', n ? Math.round(100 * withIns / n) + '%' : '0%', fmt(withIns) + ' of ' + fmt(n)],
      ['Top class', sum ? meta[top].label : '-', sum ? Math.round(100 * totals[top] / sum) + '% of insects' : 'no insects'],
      ['Insects per hour', span >= 6e4 ? (sum / (span / 36e5)).toFixed(1) : fmt(sum), 'over ' + dur(span) + ' of data']
    ];
    $('#kpis').innerHTML = kp.map(function (k) {
      return '<div class="card kpi"><div class="l">' + esc(k[0]) + '</div><div class="v">' + esc(k[1]) + '</div><div class="s">' + esc(k[2]) + '</div></div>';
    }).join('');

    /* time buckets */
    var step = pickStep(Math.max(span, 6e4));
    var off = -new Date(tStart).getTimezoneOffset() * 6e4;
    var b0 = Math.floor((tStart + off) / step) * step - off;
    var nb = Math.floor((tEnd - b0) / step) + 1;
    var per = []; for (var k = 0; k < K; k++) per.push(new Array(nb).fill(0));
    var frames = new Array(nb).fill(0);
    var hSum = new Array(24).fill(0), hN = new Array(24).fill(0), hist = new Array(6).fill(0);
    rows.forEach(function (r) {
      var i = Math.floor((r.t - b0) / step);
      r.c.forEach(function (v, j) { per[j][i] += v; });
      frames[i]++;
      var h = new Date(r.t).getHours(); hSum[h] += r.n; hN[h]++;
      hist[Math.min(r.n, 5)]++;
    });
    var labels = []; for (var i = 0; i < nb; i++) labels.push(label(b0 + i * step, step, span));
    var cum = [], run = 0;
    for (i = 0; i < nb; i++) { var s = 0; for (k = 0; k < K; k++) s += per[k][i]; run += s; cum.push(run); }
    $('#bucketHint').textContent = 'per ' + stepName(step);

    put('cTime', 'bar', labels, meta.map(function (m, j) {
      return { label: m.label, data: per[j], backgroundColor: m.color, stack: 's', borderRadius: 2, maxBarThickness: 28 };
    }), {
      scales: { x: xAxis({ stacked: true }), y: yAxis({ stacked: true }) },
      plugins: { legend: { display: true, position: 'top', align: 'end', labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8 } } }
    });

    put('cCum', 'line', labels, [{ data: cum, borderColor: '#2f5bea', backgroundColor: 'rgba(47,91,234,.08)', fill: true, tension: .2, pointRadius: 0, borderWidth: 2 }],
      { scales: { x: xAxis({ ticks: { maxTicksLimit: 4, maxRotation: 0 } }), y: yAxis() } });

    put('cFrames', 'bar', labels, bars('#9db4f7', frames),
      { scales: { x: xAxis({ ticks: { maxTicksLimit: 4, maxRotation: 0 } }), y: yAxis() } });

    var hl = []; for (i = 0; i < 24; i++) hl.push(p2(i));
    put('cHour', 'bar', hl, bars('#f2a33a', hSum.map(function (v, j) { return hN[j] ? +(v / hN[j]).toFixed(2) : null; })),
      { scales: { x: xAxis({ ticks: { maxTicksLimit: 6, maxRotation: 0 } }), y: yAxis({ ticks: { precision: 1 } }) } });

    put('cHist', 'bar', ['0', '1', '2', '3', '4', '5+'], bars('#14a38b', hist),
      { scales: { x: xAxis(), y: yAxis() } });

    /* class mix */
    put('cMix', 'doughnut', meta.map(function (m) { return m.label; }),
      [{ data: totals, backgroundColor: meta.map(function (m) { return m.color; }), borderWidth: 2, borderColor: '#fff' }],
      { cutout: '70%', interaction: { mode: 'nearest', intersect: true } });
    $('#mixTotal').textContent = fmt(sum);
    $('#mixHint').textContent = RLABEL[range];
    $('#legend').innerHTML = meta.map(function (m, j) {
      return '<div><span class="sw" style="background:' + m.color + '"></span><span>' + esc(m.label) + '</span><span>' + fmt(totals[j]) + '</span><span class="p">' + (sum ? (100 * totals[j] / sum).toFixed(1) : '0.0') + '%</span></div>';
    }).join('');

    /* latest frame + gallery (rebuilt only when the newest file changes) */
    var last = all.length ? all[all.length - 1] : null;
    var lk = last ? last.file : '';
    if (keys.latest !== lk) {
      keys.latest = lk;
      $('#latest').innerHTML = last ? '<img src="' + imgUrl(last.file) + '" alt="" data-f="' + esc(last.file) + '">' : '';
      var gal = all.slice(-12).reverse();
      $('#gallery').innerHTML = gal.map(function (r) {
        return '<button class="tile" data-f="' + esc(r.file) + '"><img loading="lazy" src="' + imgUrl(r.file) + '" alt=""><span class="b">' + r.n + '</span><span class="t">' + esc(stamp(r.t).slice(-8)) + '</span></button>';
      }).join('');
    }
    $('#latestCap').innerHTML = last
      ? '<span>' + esc(stamp(last.t)) + '</span>' + meta.map(function (m, j) { return '<span class="chip"><span class="sw" style="background:' + m.color + '"></span>' + esc(m.label) + ' ' + last.c[j] + '</span>'; }).join('')
      : '<span>No frames yet</span>';

    /* frame log */
    $('#thead').innerHTML = '<tr><th>Time</th><th>File</th>' + meta.map(function (m) { return '<th class="n">' + esc(m.label) + '</th>'; }).join('') + '<th class="n">Total</th></tr>';
    var list = tab === 'recent' ? rows.slice(-15).reverse()
      : rows.slice().sort(function (a, b) { return b.n - a.n || b.t - a.t; }).slice(0, 15);
    $('#tbody').innerHTML = list.map(function (r) {
      return '<tr data-f="' + esc(r.file) + '"><td>' + esc(stamp(r.t)) + '</td><td class="f">' + esc(r.file) + '</td>' +
        r.c.map(function (v) { return '<td class="n' + (v ? '' : ' z') + '">' + (v || '-') + '</td>'; }).join('') +
        '<td class="n"><b>' + r.n + '</b></td></tr>';
    }).join('');
  }

  /* ---------- data ---------- */
  function refresh() {
    return fetch('/api/data', { cache: 'no-store' }).then(function (r) { return r.json(); }).then(function (d) {
      d.rows = d.rows.map(function (r) {
        var n = 0; r[2].forEach(function (v) { n += v; });
        return { t: Date.parse(r[0]), file: r[1], c: r[2], n: n };
      });
      D = d; fetchedAt = Date.now();
      var pi = 0;
      meta = d.classes.map(function (name) {
        return { name: name, label: /^\d+$/.test(name) ? 'Class ' + name : name.charAt(0).toUpperCase() + name.slice(1), color: name === 'other' ? '#aab0bb' : PAL[pi++ % PAL.length] };
      });
      byFile = {}; d.rows.forEach(function (r) { byFile[r.file] = r; });
      render(); status();
    }).catch(function () { });
  }

  function status() {
    var dot = $('#dot'), el = $('#status');
    if (!connected) { dot.className = 'dot'; el.textContent = 'Reconnecting'; return; }
    if (!D || !D.mtime) { dot.className = 'dot'; el.textContent = 'Waiting for data'; return; }
    var age = (D.now - D.mtime) + (Date.now() - fetchedAt) / 1000;
    var live = age < 120;
    dot.className = 'dot ' + (live ? 'on' : 'idle');
    el.textContent = (live ? 'Live' : 'Idle') + ', updated ' + dur(age * 1000) + ' ago';
  }

  function connect() {
    var es = new EventSource('/events'), t;
    es.onopen = function () { connected = true; status(); };
    es.onmessage = function () { connected = true; clearTimeout(t); t = setTimeout(refresh, 120); };
    es.onerror = function () { connected = false; status(); };
  }

  /* ---------- lightbox + controls ---------- */
  var lb = $('#lb');
  function openFrame(f) {
    var r = byFile[f]; if (!r) return;
    $('#lbimg').src = imgUrl(f);
    $('#lbcap').textContent = stamp(r.t) + '   |   ' + meta.map(function (m, j) { return m.label + ': ' + r.c[j]; }).join(', ') + '   |   ' + f;
    lb.showModal();
  }
  lb.addEventListener('click', function () { lb.close(); });
  document.addEventListener('click', function (e) {
    var el = e.target.closest('[data-f]');
    if (el) openFrame(el.getAttribute('data-f'));
  });

  function mark(sel, attr, val) {
    document.querySelectorAll(sel + ' button').forEach(function (b) { b.classList.toggle('on', b.getAttribute(attr) === val); });
  }
  $('#range').addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    range = b.getAttribute('data-r');
    try { localStorage.setItem('range', range); } catch (x) { }
    mark('#range', 'data-r', range); render();
  });
  $('#tabs').addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    tab = b.getAttribute('data-t'); mark('#tabs', 'data-t', tab); render();
  });
  mark('#range', 'data-r', range);

  setInterval(status, 1000);
  setInterval(function () { if (!connected) refresh(); }, 5000);
  refresh(); connect();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bugsy-Wugsy analytics dashboard")
    ap.add_argument("--dir", default="received", help="directory written by receiver.py")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from other devices")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()
    OUT_DIR = Path(args.dir)
    print(f"Watching {OUT_DIR.resolve()}  ->  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)