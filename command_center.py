#!/usr/bin/env python3
"""
Sports Predictor — Command Center
=================================

A self-contained, dependency-free command console for this project.

It serves a card-based web dashboard where you can type (or paste) ANY command
you would run in the terminal, run it, and watch its output stream live into a
card. Commands run inside a single persistent `bash` session rooted at this
project folder, so `cd`, environment variables, and shell state persist between
commands exactly like a real terminal.

Run it:

    python3 command_center.py            # starts server, opens your browser
    python3 command_center.py --no-open  # don't auto-open the browser
    python3 command_center.py --port 9000

Then visit http://127.0.0.1:8799  (or whatever port you chose).

Only the Python standard library is used. The server binds to localhost only.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import secrets
import signal
import socketserver
import subprocess
import threading
import time
import urllib.parse
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
HOST = "127.0.0.1"
DEFAULT_PORT = 8799

# Unique per-boot marker used to detect when a command has finished and to read
# back its exit code. Randomised so command output can't accidentally match it.
SENTINEL = "__CC_DONE_%s__" % secrets.token_hex(8)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# ---------------------------------------------------------------------------
# Persistent shell
# ---------------------------------------------------------------------------
_shell_lock = threading.Lock()   # only one command runs at a time (a terminal is sequential)
_shell = None                    # the live bash Popen, created lazily
_shell_mgmt = threading.Lock()   # guards creation / teardown of _shell


def get_shell() -> subprocess.Popen:
    """Return a live bash process, (re)creating it if needed."""
    global _shell
    with _shell_mgmt:
        if _shell is None or _shell.poll() is not None:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["TERM"] = "dumb"          # discourage tools from emitting cursor tricks
            _shell = subprocess.Popen(
                ["bash"],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,   # own process group -> we can kill the whole tree
            )
        return _shell


def stop_shell() -> bool:
    """Kill the current shell and everything it spawned. Returns True if a
    running shell was actually terminated."""
    global _shell
    with _shell_mgmt:
        sh = _shell
        _shell = None
        if sh is None or sh.poll() is not None:
            return False
        try:
            os.killpg(os.getpgid(sh.pid), signal.SIGKILL)
        except Exception:
            try:
                sh.kill()
            except Exception:
                pass
        return True


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    # Quieter logging
    def log_message(self, fmt, *args):
        pass

    # -- helpers ----------------------------------------------------------
    def _send_html(self, body: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse(self, event: str, payload) -> bool:
        try:
            msg = "event: %s\ndata: %s\n\n" % (event, json.dumps(payload))
            self.wfile.write(msg.encode("utf-8"))
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    # -- routing ----------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self._send_html(PAGE)
        elif path == "/run":
            qs = urllib.parse.parse_qs(parsed.query)
            cmd = (qs.get("cmd", [""])[0]).strip()
            self.handle_run(cmd)
        elif path == "/stop":
            killed = stop_shell()
            body = json.dumps({"stopped": killed}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/cwd":
            body = json.dumps({"cwd": current_cwd()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404, "Not found")

    # -- command execution ------------------------------------------------
    def handle_run(self, cmd: str):
        self._sse_headers()
        if not cmd:
            self._sse("done", {"code": "0", "duration": 0.0, "note": "empty command"})
            return

        got_lock = _shell_lock.acquire(timeout=0)
        if not got_lock:
            self._sse("line", "[a command is already running — Stop it or wait]\n")
            self._sse("done", {"code": "busy", "duration": 0.0})
            return

        start = time.time()
        try:
            sh = get_shell()
            # Send the command, then a sentinel line carrying the exit code.
            # A leading newline guarantees the sentinel starts on its own line
            # even if the command's last output had no trailing newline.
            sh.stdin.write(cmd + "\n")
            sh.stdin.write("printf '%s%%s\\n' \"$?\"\n" % SENTINEL)
            sh.stdin.flush()

            for raw in sh.stdout:
                if SENTINEL in raw:
                    pre_text, _, rest = raw.partition(SENTINEL)
                    if pre_text:
                        self._sse("line", ANSI_RE.sub("", pre_text))
                    code = rest.strip() or "?"
                    self._sse("done", {"code": code, "duration": round(time.time() - start, 2),
                                        "cwd": current_cwd()})
                    return
                clean = ANSI_RE.sub("", raw)
                if not self._sse("line", clean):
                    # client disconnected; keep the shell alive, just stop streaming
                    return
            # Reached EOF without a sentinel -> shell was killed (Stop) or died.
            self._sse("done", {"code": "stopped", "duration": round(time.time() - start, 2)})
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._sse("done", {"code": "error", "duration": round(time.time() - start, 2)})
        finally:
            _shell_lock.release()


def current_cwd() -> str:
    """Best-effort read of the persistent shell's working directory."""
    global _shell
    sh = _shell
    if sh is None or sh.poll() is not None:
        return ROOT
    try:
        pid = sh.pid
        # On macOS/Linux the shell's cwd is readable via lsof-free proc trick.
        if os.path.isdir("/proc"):
            return os.readlink("/proc/%d/cwd" % pid)
    except Exception:
        pass
    return ""


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sports Predictor — Command Center</title>
<style>
  :root{
    --bg:#0f1419; --panel:#161b22; --border:#243140; --muted:#8a96a3;
    --text:#e6e6e6; --accent:#4c9aff; --green:#2fbf71; --red:#e5484d; --amber:#f5a623;
  }
  *{box-sizing:border-box}
  body{background:var(--bg);color:var(--text);
    font:14px/1.5 -apple-system,Segoe UI,Arial,sans-serif;margin:0;padding:0}
  header{padding:18px 24px 10px}
  h1{font-size:19px;margin:0}
  .sub{color:var(--muted);margin:2px 0 0;font-size:12px}
  .sub code{color:var(--accent)}
  .bar{position:sticky;top:0;z-index:5;background:var(--bg);
    padding:12px 24px;border-bottom:1px solid var(--border);
    display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .prompt{color:var(--green);font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  #cmd{flex:1;min-width:240px;background:#0c1015;color:var(--text);
    border:1px solid var(--border);border-radius:8px;padding:10px 12px;
    font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;outline:none}
  #cmd:focus{border-color:var(--accent)}
  button{border:1px solid var(--border);background:var(--panel);color:var(--text);
    border-radius:8px;padding:9px 14px;font-size:13px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button:disabled{opacity:.45;cursor:not-allowed}
  #run{background:var(--accent);border-color:var(--accent);color:#04122b;font-weight:600}
  #stop{background:#2a1416;border-color:#5c2b2f;color:#ffb4b4}
  .cwd{color:var(--muted);font-size:11px;padding:6px 24px 0;
    font-family:ui-monospace,Menlo,Consolas,monospace}
  .grid{padding:16px 24px 60px;display:grid;gap:14px;
    grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr))}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;
    overflow:hidden;display:flex;flex-direction:column;min-width:0}
  .card.wide{grid-column:1 / -1}
  .chead{display:flex;align-items:center;gap:10px;padding:10px 12px;
    border-bottom:1px solid var(--border);background:#12171e}
  .chead .cmdtext{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;font:12px ui-monospace,Menlo,Consolas,monospace;color:#cdd6df}
  .chead .cmdtext b{color:var(--green)}
  .badge{font-size:11px;padding:2px 8px;border-radius:999px;white-space:nowrap;
    border:1px solid var(--border);color:var(--muted)}
  .badge.run{color:var(--accent);border-color:var(--accent)}
  .badge.ok{color:var(--green);border-color:#1f6b45}
  .badge.err{color:var(--red);border-color:#5c2b2f}
  .badge.warn{color:var(--amber);border-color:#6b5320}
  .meta{font-size:11px;color:var(--muted);white-space:nowrap}
  .iconbtn{padding:4px 8px;font-size:11px}
  pre.out{margin:0;padding:12px;max-height:460px;overflow:auto;
    font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    white-space:pre-wrap;word-break:break-word;color:#dbe3ec}
  pre.out:empty::after{content:"(no output)";color:var(--muted)}
  .dot{display:inline-block;width:7px;height:7px;border-radius:50%;
    background:var(--accent);margin-right:2px;animation:pulse 1s infinite}
  @keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
  .hint{color:var(--muted);font-size:12px;padding:2px 24px 0}
  .hint code{color:var(--accent);cursor:pointer}
  .hint code:hover{text-decoration:underline}
</style>
</head>
<body>
<header>
  <h1>Sports Predictor — Command Center</h1>
  <p class="sub">Runs <b>any</b> command in a live <code>bash</code> session rooted at the project folder. Output streams into cards below.</p>
</header>

<div class="bar">
  <span class="prompt">$</span>
  <input id="cmd" list="examples" autocomplete="off" spellcheck="false"
         placeholder="type any command and press Enter…  (↑/↓ for history)">
  <datalist id="examples">
    <option value="./daily_card.sh --card-only">
    <option value="./daily_card.sh --fast">
    <option value="python3 predictor.py &quot;Brazil&quot; &quot;Morocco&quot;">
    <option value="python3 predictor.py --worldcup">
    <option value="python3 simulate.py -n 50000">
    <option value="python3 edge.py --calibrated --market-blend --context">
    <option value="python3 -m golf.simulate --sims 50000">
    <option value="python3 -m nhl.predictor &quot;Toronto Maple Leafs&quot; &quot;Boston Bruins&quot;">
    <option value="python3 run_checks.py">
    <option value="python3 preflight.py">
    <option value="python3 validate_all.py --gate --sims 4000">
    <option value="git status">
    <option value="ls -la">
  </datalist>
  <button id="run">Run</button>
  <button id="stop" disabled>Stop</button>
  <button id="clear">Clear</button>
</div>
<div class="cwd" id="cwd"></div>
<p class="hint">Try:
  <code data-cmd="python3 preflight.py">preflight</code> ·
  <code data-cmd="./daily_card.sh --card-only">daily card</code> ·
  <code data-cmd="python3 run_checks.py">run tests</code> ·
  <code data-cmd="git status">git status</code>
</p>

<div class="grid" id="grid"></div>

<script>
(function(){
  const cmdEl = document.getElementById('cmd');
  const runBtn = document.getElementById('run');
  const stopBtn = document.getElementById('stop');
  const clearBtn = document.getElementById('clear');
  const grid = document.getElementById('grid');
  const cwdEl = document.getElementById('cwd');
  let history = [];
  let hidx = -1;
  let current = null;   // {es, pre, card, badge, meta, t0, timer, cmd}

  function refreshCwd(){
    fetch('/cwd').then(r=>r.json()).then(d=>{
      if(d.cwd){ cwdEl.textContent = 'cwd: ' + d.cwd; }
    }).catch(()=>{});
  }
  refreshCwd();

  function esc(s){ return s.replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

  function newCard(cmd){
    const card = document.createElement('section');
    card.className = 'card';
    const isWide = cmd.length > 60;
    if(isWide) card.classList.add('wide');
    card.innerHTML =
      '<div class="chead">' +
        '<span class="cmdtext"><b>$</b> ' + esc(cmd) + '</span>' +
        '<span class="meta"></span>' +
        '<span class="badge run"><span class="dot"></span>running</span>' +
        '<button class="iconbtn copy" title="Copy output">copy</button>' +
        '<button class="iconbtn wide" title="Toggle full width">↔</button>' +
        '<button class="iconbtn close" title="Remove card">✕</button>' +
      '</div>' +
      '<pre class="out"></pre>';
    grid.prepend(card);
    const pre = card.querySelector('pre.out');
    const badge = card.querySelector('.badge');
    const meta = card.querySelector('.meta');
    card.querySelector('.close').onclick = ()=> card.remove();
    card.querySelector('.wide').onclick = ()=> card.classList.toggle('wide');
    card.querySelector('.copy').onclick = ()=>{
      navigator.clipboard && navigator.clipboard.writeText(pre.textContent);
    };
    return {card, pre, badge, meta};
  }

  function setRunning(on){
    runBtn.disabled = on;
    stopBtn.disabled = !on;
    cmdEl.disabled = false;  // keep typing next command
  }

  function finish(state, code, duration){
    if(!current) return;
    clearInterval(current.timer);
    const b = current.badge;
    b.classList.remove('run');
    b.innerHTML = '';
    if(code === '0'){ b.classList.add('ok'); b.textContent = 'exit 0'; }
    else if(code === 'stopped'){ b.classList.add('warn'); b.textContent = 'stopped'; }
    else if(code === 'busy'){ b.classList.add('warn'); b.textContent = 'busy'; }
    else if(code === 'error'){ b.classList.add('err'); b.textContent = 'error'; }
    else { b.classList.add('err'); b.textContent = 'exit ' + code; }
    if(duration != null){ current.meta.textContent = duration + 's'; }
    setRunning(false);
    current = null;
    refreshCwd();
    cmdEl.focus();
  }

  function run(cmd){
    cmd = cmd.trim();
    if(!cmd) return;
    if(current){ return; }               // one at a time
    if(history[history.length-1] !== cmd){ history.push(cmd); }
    hidx = history.length;

    const {card, pre, badge, meta} = newCard(cmd);
    const t0 = Date.now();
    const timer = setInterval(()=>{
      meta.textContent = ((Date.now()-t0)/1000).toFixed(1) + 's';
    }, 200);
    const es = new EventSource('/run?cmd=' + encodeURIComponent(cmd));
    current = {es, pre, card, badge, meta, t0, timer, cmd};
    setRunning(true);

    es.addEventListener('line', ev=>{
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
      pre.textContent += JSON.parse(ev.data);
      if(atBottom) pre.scrollTop = pre.scrollHeight;
    });
    es.addEventListener('done', ev=>{
      const d = JSON.parse(ev.data);
      es.close();
      finish('done', String(d.code), d.duration);
    });
    es.onerror = ()=>{
      if(!current) return;
      es.close();
      finish('done', 'error', ((Date.now()-t0)/1000).toFixed(1));
    };
  }

  runBtn.onclick = ()=> { run(cmdEl.value); cmdEl.value=''; };
  stopBtn.onclick = ()=> { fetch('/stop').catch(()=>{}); };
  clearBtn.onclick = ()=> { grid.innerHTML=''; };

  cmdEl.addEventListener('keydown', e=>{
    if(e.key === 'Enter'){ e.preventDefault(); run(cmdEl.value); cmdEl.value=''; }
    else if(e.key === 'ArrowUp'){
      if(history.length===0) return;
      e.preventDefault();
      hidx = Math.max(0, hidx-1);
      cmdEl.value = history[hidx] || '';
      cmdEl.setSelectionRange(cmdEl.value.length, cmdEl.value.length);
    } else if(e.key === 'ArrowDown'){
      if(history.length===0) return;
      e.preventDefault();
      hidx = Math.min(history.length, hidx+1);
      cmdEl.value = history[hidx] || '';
    }
  });

  document.querySelectorAll('.hint code').forEach(c=>{
    c.onclick = ()=>{ cmdEl.value = c.getAttribute('data-cmd'); cmdEl.focus(); };
  });

  cmdEl.focus();
})();
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="Sports Predictor Command Center")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((HOST, args.port), Handler)
    url = "http://%s:%d/" % (HOST, args.port)
    print("Sports Predictor — Command Center")
    print("Serving %s (project root: %s)" % (url, ROOT))
    print("Press Ctrl-C to stop the server.")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        stop_shell()
        httpd.server_close()


if __name__ == "__main__":
    main()
