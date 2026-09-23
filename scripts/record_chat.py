"""Record the actual supplied terminal interface, including prompts and replies."""
import html
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
run = json.loads((ROOT / "experiment_runs.json").read_text())["expanded"]["run_dir"]
out = ROOT / "evidence"
out.mkdir(exist_ok=True)
transcript = out / "chat_transcript.json"
if transcript.exists():
    raise SystemExit("Refusing to overwrite an existing chat recording.")
command = [sys.executable, "chat.py", "--model", f"{run}/model.pt", "--transcript", "evidence/chat_transcript.json"]
master, slave = pty.openpty()
started = time.monotonic()
process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave, cwd=ROOT)
os.close(slave)
prompts = ["the customer", "the nurse", "the small cups", "explain quantum teleportation", "/quit"]
events, stream, pending = [], "", ""
sent = 0
try:
    while time.monotonic() - started < 90:
        ready, _, _ = select.select([master], [], [], .2)
        if ready:
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunk = data.decode("utf-8", errors="replace")
            events.append([round(time.monotonic() - started, 6), "o", chunk])
            stream += chunk
            pending += chunk
            if pending.endswith("You: ") and sent < len(prompts):
                os.write(master, (prompts[sent] + "\n").encode())
                sent += 1
                pending = ""
        if process.poll() is not None and not ready:
            break
    else:
        process.terminate()
        raise RuntimeError("Chat recording timed out")
finally:
    os.close(master)
code = process.wait(timeout=10)
if code:
    raise RuntimeError(f"Chat failed: {code}\n{stream}")
record = json.loads(transcript.read_text())
assert len(record["turns"]) == 4
header = {"version": 2, "width": 110, "height": 28, "timestamp": int(time.time()),
          "title": "Actual tiny nanoGPT terminal chat", "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
          "model_sha256": record["model_sha256"], "run_dir": run}
(out / "chat.cast").write_text("\n".join(json.dumps(x) for x in [header, *events]) + "\n")
(out / "chat_terminal.txt").write_text(stream)
page = '''<!doctype html><html><head><meta charset="utf-8"><title>Actual nanoGPT chat recording</title>
<style>body{background:#101827;color:#e9eef6;font:17px system-ui;margin:40px auto;max-width:1120px;padding:0 25px}h1{font-size:28px}p{color:#b7c9de;line-height:1.5}pre{background:#070d16;padding:24px;border:1px solid #31445e;border-radius:12px;white-space:pre-wrap;line-height:1.5;font:16px ui-monospace,monospace}button{padding:12px 20px;margin-right:12px;font-size:16px;cursor:pointer}code{overflow-wrap:anywhere}.meta{font-size:13px}</style></head><body>
<h1>Actual tiny nanoGPT chat recording</h1><p>Four real interactions with the saved expanded-corpus model. Every prompt starts fresh. Temperature 0.8; context 48 tokens; maximum reply 24 tokens. This page replays recorded terminal output; launch chat.py to interact with the model.</p>
<p class="meta">Run: RUN<br>Model SHA-256: HASH</p><button id="play">Replay recording</button><button id="all">Show complete recording</button><pre id="terminal"></pre>
<script>const events=EVENTS;let timers=[];const terminal=document.getElementById('terminal');function clear(){timers.forEach(clearTimeout);timers=[];terminal.textContent=''}function all(){clear();terminal.textContent=events.map(e=>e[2]).join('')}document.getElementById('all').onclick=all;document.getElementById('play').onclick=()=>{clear();const offset=events[0][0];events.forEach(e=>timers.push(setTimeout(()=>terminal.textContent+=e[2],(e[0]-offset)*1000)))};all();</script></body></html>'''
page = page.replace("RUN", html.escape(run)).replace("HASH", record["model_sha256"]).replace("EVENTS", json.dumps(events).replace("</", "<\\/"))
(out / "chat_recording.html").write_text(page)
print(stream)
print("Saved real PTY recording, transcript, and self-contained HTML playback.")
