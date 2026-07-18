"""Local web UI for the voice cloner and the deepfake detector.

    python tools/webui/app.py          # then open http://localhost:7860

Runs on YOUR machine and drives the lab GPU over SSH. That design is forced by the
containers: the synthesizer needs the PyTorch container (coqui-tts) and the detector
needs the TensorFlow one (Keras back-end), and neither container has the other
framework — so no single server-side process can serve both. Orchestrating from
outside sidesteps the split entirely, and needs no SSH tunnel or held GPU allocation.

Cost: each action is a Slurm job, so expect ~30-60 s round trips (queue + model load).
For a live demo, warm it up once before you present.

Requires: `ssh shenkar` working (see docs/server_runbook.md) and the VPN up.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, request

HOST_ALIAS = "shenkar"          # matches the Host block in ~/.ssh/config
REMOTE_REPO = "~/adf"
POLL_SECONDS = 5
JOB_TIMEOUT = 600

app = Flask(__name__)
# Uploads are scp'd to a box with a ~50 GB home quota whose exhaustion already killed
# one job (see docs/runtimes.md). Cap the request body.
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


def _ssh(cmd: str, timeout: int = 120) -> str:
    r = subprocess.run(["ssh", HOST_ALIAS, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"ssh exited {r.returncode}")
    return r.stdout.strip()


def _wait_for_job(job_id: str) -> None:
    """Block until the job leaves the queue, then check it actually succeeded."""
    deadline = time.time() + JOB_TIMEOUT
    while time.time() < deadline:
        if not _ssh(f"squeue -h -j {job_id} -o %T"):
            break
        time.sleep(POLL_SECONDS)
    else:
        raise RuntimeError(f"job {job_id} still queued after {JOB_TIMEOUT}s")
    state = _ssh(f"sacct -j {job_id} --format=State -n | head -1").strip()
    if not state.startswith("COMPLETED"):
        raise RuntimeError(f"job {job_id} finished as {state or 'UNKNOWN'}")


def _submit(env: str, script: str) -> str:
    out = _ssh(f"cd {REMOTE_REPO} && {env} sbatch --parsable {script}")
    m = re.search(r"\d+", out)
    if not m:
        raise RuntimeError(f"could not parse job id from: {out!r}")
    return m.group(0)


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


@app.post("/api/synthesize")
def synthesize():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify(error="no text"), 400
    stamp = str(int(time.time()))
    outdir = f"data/generated/webui/{stamp}"
    job = _submit(f"TEXT={_shquote(text)} NAME=web OUT={outdir}", "scripts/say.slurm")
    _wait_for_job(job)
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "web_00.wav"
        subprocess.run(["scp", "-q", f"{HOST_ALIAS}:{REMOTE_REPO}/{outdir}/web_00.wav", str(local)],
                       check=True, timeout=120)
        b64 = base64.b64encode(local.read_bytes()).decode()
    return jsonify(job=job, audio=f"data:audio/wav;base64,{b64}")


@app.post("/api/detect")
def detect():
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file"), 400
    stamp = str(int(time.time()))
    remote = f"{REMOTE_REPO}/data/webui_uploads/{stamp}.wav"
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "upload.wav"
        f.save(local)
        if local.read_bytes()[:4] != b"RIFF":
            return jsonify(error="not a WAV file (expected a RIFF header)"), 400
        _ssh(f"mkdir -p {REMOTE_REPO}/data/webui_uploads")
        subprocess.run(["scp", "-q", str(local), f"{HOST_ALIAS}:{remote}"], check=True, timeout=300)
    outdir = f"reports/webui_detect/{stamp}"
    job = _submit(f"FILE={remote} OUT={outdir}", "scripts/detect_one.slurm")
    _wait_for_job(job)
    raw = _ssh(f"cat {REMOTE_REPO}/{outdir}/detect_one.json")
    return jsonify(job=job, **json.loads(raw))


@app.get("/")
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    print("Voice clone + detector UI  ->  http://localhost:7860")
    print("Requires the VPN up and `ssh shenkar` working.")
    app.run(host="127.0.0.1", port=7860, debug=False)
