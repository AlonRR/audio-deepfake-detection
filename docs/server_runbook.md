# Server runbook — training on the Shenkar Lab (NVIDIA L4)

How to launch the **detection baseline** training on the lab GPU. You drive your own
JupyterHub session (your account / one-time first login); this repo gives you the code +
exact commands. Paste the output back and I'll debug.

> Status of the code: modules syntax-compile and the EER math is unit-tested, but the
> TensorFlow training path has **not** run against real data/GPU yet — expect the first
> run to surface a fix or two. That's the point of the smoke run in step 4.

## 1. Connect
1. Bring up the **WireGuard** tunnel (import `credentials_server/<LAB_USER>.conf`, Activate).
2. Open **https://$LAB_HOST** → accept the cert warning → log in; **change the
   one-time password** on first login.
3. Open **JupyterHub**: `https://$LAB_HOST/jupyter/hub/login`. Launch a session,
   then open a **Terminal**.

## 2. Get the code onto the server (pick one)
**A — Upload the zip (no GitHub needed).** In JupyterHub's file browser click **Upload**,
send `audio-deepfake-detection.zip`, then in the Terminal:
```bash
cd ~ && unzip -o audio-deepfake-detection.zip -d audio-deepfake-detection && cd audio-deepfake-detection
```
**B — Git (better for iterating on fixes).** Re-auth locally first (`gh auth refresh -h
github.com`) so I can push a private repo; then on the server:
```bash
cd ~ && git clone https://github.com/AlonRR/audio-deepfake-detection && cd audio-deepfake-detection
```

## 3. Find ASVspoof 2019 LA + install deps
```bash
ls /datasets ; ls /shenkar-data/fg/datasets 2>/dev/null      # is ASVspoof already staged?
# note the LA root — it must contain ASVspoof2019_LA_cm_protocols/ and ASVspoof2019_LA_{train,dev,eval}/
bash scripts/setup_env.sh          # installs librosa/soundfile/matplotlib into ~/.local (TF is in the container)
```
If ASVspoof isn't on `/datasets`, download LA from the Edinburgh DataShare (ASVspoof 2019)
into `~/data/LA` and use that path below.

## 4. Smoke run first (tiny subset — proves the pipeline in ~minutes)
```bash
LA=/datasets/ASVspoof2019/LA        # <-- set to the real path from step 3
apptainer exec --nv /opt/containers/tensorflow-25.02.sif \
    python -m src.detection.train --data-root "$LA" --limit 512 --epochs 2 --out reports/smoke
```
Expect it to finish, print a `DEV EER`, and write `reports/smoke/{learning_curves.png,
det_dev.png,result.json}`. If it errors, paste the traceback to me.

## 5. Full run (via Slurm)
```bash
DATA=/datasets/ASVspoof2019/LA OUT=reports/run1 sbatch scripts/train_detection.slurm
squeue -u "$(whoami)"            # watch it
tail -f slurm-<jobid>.out        # progress + final DEV EER
```
Artifacts land in `reports/run1/` (learning curves, DET curve, `dev_scores.npz`,
`result.json`). Pull those back (download from JupyterHub, or `git add`/commit if using B)
and we compare EER against the published ASVspoof baseline.

## What's next after this baseline trains
- Swap `--feat lfcc` ↔ `logmel`; sweep LR/filters/dropout (the "multiple tries" + learning
  curves the rubric wants) — keep the runs that fail, we analyse them.
- Then the SSL detector (frozen wav2vec2-XLS-R features in the **PyTorch** container →
  Keras back-end) and the creation half (record your voice → XTTS + Keras TTS).
