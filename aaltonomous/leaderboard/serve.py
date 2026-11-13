"""Live sim leaderboard for the venue screen (stdlib only, + PyYAML for the config if installed).

    python -m aaltonomous.leaderboard.serve --runs runs/eval --port 8090      # http://<host>:8090/
    python -m aaltonomous.leaderboard.serve --runs runs/eval --once           # just write leaderboard.html

Routes: `/` (auto-refreshing page), `/leaderboard.json`, `/healthz`. Every request rescans the eval dirs, so new
results appear on the next refresh. Each scan also rewrites the static page (`--html-out`, no auto-refresh) for
posting to the team chat.
"""
import argparse
import html
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aaltonomous.leaderboard import ingest

CSS = """
:root{--bg:#0b0e14;--panel:#141925;--line:#232a3a;--fg:#e8ecf4;--dim:#8a93a8;--gold:#f5c542;--ok:#2ecc71;
--bad:#ff5c5c;--warn:#ffb347;--accent:#5aa9ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
font-size:clamp(16px,1.35vw,28px);line-height:1.25}
header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:.4em 2em;
padding:.8em 1.2em .4em}
h1{margin:0;font-size:2em;letter-spacing:.01em}
.meta{color:var(--dim);font-size:.8em}
main{padding:0 1.2em 1em}
table{width:100%;border-collapse:separate;border-spacing:0 .25em}
th{color:var(--dim);font-weight:600;font-size:.75em;text-transform:uppercase;letter-spacing:.06em;text-align:left;
padding:.2em .6em}
td{background:var(--panel);padding:.3em .6em;vertical-align:middle}
td:first-child{border-radius:.4em 0 0 .4em}td:last-child{border-radius:0 .4em .4em 0}
.num{text-align:right;font-variant-numeric:tabular-nums}
.rank{width:3.2em;font-size:1.3em;font-weight:800;text-align:center;color:var(--dim)}
.team{font-size:1.25em;font-weight:700;overflow-wrap:anywhere}.team .tag{white-space:nowrap}
.total{font-size:1.6em;font-weight:800}
tr.fin td{background:#1d2232}tr.fin td:first-child{box-shadow:inset .3em 0 0 var(--gold)}
tr.fin .rank{color:var(--gold)}
tr.ref td,tr.out td{opacity:.55}
.tag{display:inline-block;font-size:.55em;font-weight:800;letter-spacing:.08em;padding:.15em .5em;border-radius:.3em;
vertical-align:middle;margin-left:.5em}
.t-fin{background:var(--gold);color:#111}.t-ref{border:1px solid var(--dim);color:var(--dim)}
.badge{display:inline-block;font-weight:800;font-size:.8em;padding:.15em .55em;border-radius:.35em}
.g-pass{background:var(--ok);color:#06210f}.g-fail{background:var(--bad);color:#2a0606}
.g-partial{background:var(--warn);color:#2a1a02}.g-none{border:1px solid var(--dim);color:var(--dim)}
.cell small{display:block;color:var(--dim);font-size:.6em;font-weight:400}
.cell.nc{color:var(--dim);text-decoration:line-through}
.when{color:var(--dim);font-size:.8em;white-space:nowrap}
.strip{margin-top:.8em;color:var(--dim);font-size:.8em}
.strip b{color:var(--accent)}
.empty{padding:3em 0;text-align:center;color:var(--dim);font-size:1.4em}
footer{padding:.4em 1.2em 1em;color:var(--dim);font-size:.65em}
@media (max-width:700px){.hide-s{display:none}body{font-size:15px}header,main{padding-left:16px;padding-right:16px}}
"""


def _e(x):
    return html.escape(str(x), quote=True)


def _ago(ts, now):
    if not ts:
        return "–"
    d = max(0, now - ts)
    rel = f"{int(d)} s ago" if d < 60 else f"{int(d // 60)} min ago" if d < 3600 else \
        f"{d / 3600:.1f} h ago" if d < 86400 else f"{int(d // 86400)} d ago"
    return f"{time.strftime('%a %H:%M', time.localtime(ts))} <small>({rel})</small>"


def _fmt(v, metric):
    if v is None:
        return "–"
    return f"{100 * v:.0f}%" if metric in ("score_frac", "object_success") else f"{v:+.2f}"


def render_html(b, refresh_s=None, max_rows=None):
    now = b["generated"]
    tasks = b["tasks"]
    def note(w):
        return " · unranked" if w <= 0 else "" if w == 1 else f" · ×{w:g}"
    head = "".join(f"<th class='num hide-s'>{_e(t['label'])}<br><small>{_e(k)}{note(t['weight'])}</small></th>"
                   for k, t in tasks.items())
    rows = []
    for t in b["teams"][: max_rows or None]:
        cls = "fin" if t["finalist"] else "ref" if t["reference"] else "" if t["eligible"] else "out"
        tag = "<span class='tag t-fin'>FINALIST</span>" if t["finalist"] else \
            "<span class='tag t-ref'>REFERENCE</span>" if t["reference"] else ""
        cells = []
        for k in tasks:
            c = t["tasks"].get(k)
            if not c:
                cells.append("<td class='num hide-s cell'>–</td>")
                continue
            sub = f"{c['trials']} trials" + (f" · {c['runs']} runs" if c["runs"] > 1 else "")
            if c["short"]:
                sub = c["short"]
            elif c["gate"] is not True:
                sub = "gate: " + (", ".join(c["gate_why"]) or "fail")
            cells.append(f"<td class='num hide-s cell{'' if c['counts'] else ' nc'}'>{_fmt(c['value'], b['metric'])}"
                         f"<small>{_e(sub)}</small></td>")
        gate = {"pass": "PASS", "fail": "FAIL", "partial": "PARTIAL", "none": "–"}[t["gate"]]
        rows.append(
            f"<tr class='{cls}'><td class='rank'>{t['rank'] or '·'}</td>"
            f"<td class='team'>{_e(t['team'])}{tag}</td>"
            f"<td class='num total'>{_fmt(t['total'], b['metric'])}</td>{''.join(cells)}"
            f"<td><span class='badge g-{t['gate']}'>{gate}</span></td>"
            f"<td class='when hide-s'>{_ago(t['last_submission'], now)}</td></tr>")
    body = (f"<table><thead><tr><th>#</th><th>Team</th><th class='num'>Score</th>{head}<th>Sim gate</th>"
            f"<th class='hide-s'>Last submission</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            if rows else "<div class='empty'>No scored submissions yet.</div>")
    strip = ""
    live = [r for r in b["running"] if r["status"] in ("running", "writing")]
    if live:
        strip += "<div class='strip'>Evaluating now: " + ", ".join(
            f"<b>{_e(r['team'])}</b> {_e(r['task'] or '')} ({r['trials'] or 0} trials done)" for r in live) + "</div>"
    if b["errors"]:
        strip += f"<div class='strip'>{len(b['errors'])} unreadable eval dir(s) skipped (see /leaderboard.json)</div>"
    cfg = b["config"]
    weights = " + ".join(f"{t['weight']:g}×{k}" for k, t in tasks.items() if t["weight"] > 0)
    rule = (f"Score = {weights} ({b['metric']}, {b['select']} gate-passing run of ≥{cfg['min_trials']} trials). "
            f"Gate: 0 safety aborts, ≤{cfg['gate_max_clips']} clipped steps, ≥{cfg['gate_min_hz']:g} Hz. "
            f"Top {b['finalists']} eligible teams = finalists (plus a code check). Real-robot finals decide the winner.")
    refresh = f"<meta http-equiv='refresh' content='{int(refresh_s)}'>" if refresh_s else ""
    variant = f"variant <b>{_e(b['variant'])}</b> · " if b.get("variant") else ""
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>{refresh}"
            f"<title>Sim Leaderboard</title><style>{CSS}</style></head><body>"
            f"<header><h1>{_e(b['title'])}</h1><div class='meta'>{variant}updated "
            f"{time.strftime('%a %H:%M:%S', time.localtime(now))}</div></header>"
            f"<main>{body}{strip}</main><footer>{_e(rule)}</footer></body></html>")


class Board:
    """Rescans on demand (at most every `min_interval` s) and rewrites the static page."""

    def __init__(self, runs, config=None, html_out=None, min_interval=2.0):
        self.runs, self.config, self.html_out, self.min_interval = runs, config, html_out, min_interval
        self.lock = threading.Lock()
        self.t, self.data, self.cfg = 0.0, None, None

    def get(self):
        with self.lock:
            if self.data is None or time.time() - self.t > self.min_interval:
                self.cfg = ingest.load_config(self.config)
                self.data = ingest.leaderboard(self.runs, self.cfg)
                self.t = time.time()
                if self.html_out:
                    self._write_static()
            return self.data, self.cfg

    def _write_static(self):
        try:
            p = Path(self.html_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(render_html(self.data))
            os.replace(tmp, p)
        except OSError as e:
            print(f"could not write {self.html_out}: {e}")


def make_handler(board):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html", "/leaderboard.html"):
                    b, cfg = board.get()
                    self._send(200, render_html(b, cfg.get("refresh_s"), cfg.get("max_rows")), "text/html; charset=utf-8")
                elif path == "/leaderboard.json":
                    b, _ = board.get()
                    self._send(200, json.dumps(b, indent=1), "application/json")
                elif path == "/healthz":
                    self._send(200, "ok\n", "text/plain")
                else:
                    self._send(404, "not found\n", "text/plain")
            except Exception as e:                       # never take the venue screen down
                self._send(500, f"leaderboard error: {e!r}\n", "text/plain")

        def log_message(self, fmt, *args):
            pass

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs/eval"], help="eval result dir(s) to scan")
    ap.add_argument("--config", default=None, help="leaderboard.yaml (default: next to this file)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--html-out", default="runs/leaderboard.html", help="static page for the team chat ('' = off)")
    ap.add_argument("--once", action="store_true", help="write --html-out (and print the table) and exit")
    a = ap.parse_args(argv)
    board = Board(a.runs, a.config, a.html_out or None)
    if a.once:
        board.get()
        ingest.main(["--runs", *a.runs] + (["--config", a.config] if a.config else []))
        print(f"-> {a.html_out}")
        return
    board.get()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(board))
    print(f"leaderboard on http://{a.host}:{a.port}/  (json: /leaderboard.json, static: {a.html_out})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
