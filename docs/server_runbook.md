# Server runbook — training on the university GPU lab (NVIDIA L4)

How to launch the **detection baseline** training on the lab GPU. You drive your own
JupyterHub session (your account / one-time first login); this repo gives you the code +
exact commands. Paste the output back and I'll debug.

> Host names, account names and VPN profiles are intentionally left as placeholders
> (`$LAB_HOST`, `$LAB_USER`) — this repo is public, and internal infrastructure names
> do not belong in it. Substitute your own from the lab's connection instructions.

> Status of the code: modules syntax-compile and the EER math is unit-tested, but the
> TensorFlow training path has **not** run against real data/GPU yet — expect the first
> run to surface a fix or two. That's the point of the smoke run in step 4.

## 1. Connect
1. Bring up the **WireGuard** tunnel (import your `<LAB_USER>.conf` profile, Activate).
   Keep that file and any one-time password **out of this repo** — `.gitignore` blocks
   `*.conf` / `*credential*` for exactly this reason.
2. Open **`https://$LAB_HOST`** → accept the cert warning → log in; **change the
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

## 6. SSL detector (the SOTA-family main system)
Two stages, two containers — features are cached to disk between them.
```bash
bash scripts/setup_env_pytorch.sh                                   # torch-side deps
DATA=$LA OUT=/models/$USER/ssl_xlsr sbatch scripts/extract_ssl.slurm   # PyTorch: cache frozen XLS-R features
FEATS=/models/$USER/ssl_xlsr OUT=reports/ssl_run1 sbatch scripts/train_ssl.slurm  # TF: train the Keras back-end
```
Compare `reports/ssl_run1/result.json` EER against the CNN/RawNet2 baselines.

## 7. Creation half (needs your recording in data/raw/source.wav)
```bash
# segment + transcribe your 1-5 min into an LJSpeech-style dataset (PyTorch container):
apptainer exec --nv /opt/containers/pytorch-25.04.sif \
    python -m src.creation.xtts_finetune.prepare --source data/raw/source.wav --out data/raw

# A1 Keras baseline (TF container): train, then synthesize the >=10 prompts
apptainer exec --nv /opt/containers/tensorflow-25.02.sif \
    python -m src.creation.keras_tts.train --data-root data/raw --epochs 300 --out reports/tts_baseline
apptainer exec --nv /opt/containers/tensorflow-25.02.sif \
    python -m src.creation.keras_tts.synthesize --weights reports/tts_baseline/tts.weights.h5 --out data/generated/keras_tts

# A2 XTTS-v2 (PyTorch container): zero-shot, then fine-tune, then synthesize
apptainer exec --nv /opt/containers/pytorch-25.04.sif \
    python -m src.creation.xtts_finetune.synthesize --mode zero_shot --out data/generated/xtts_zeroshot
DATA=data/raw OUT=models/xtts_ft sbatch scripts/finetune_xtts.slurm
apptainer exec --nv /opt/containers/pytorch-25.04.sif \
    python -m src.creation.xtts_finetune.synthesize --mode finetuned --ckpt-dir models/xtts_ft/run/ --out data/generated/xtts_ft
```

## 8. Evaluation (the creation-quality metrics)
`src/evaluation/`: `spectrogram_sim` (MCD/SSIM), `speaker_sim` (ECAPA cosine, PyTorch
container), `mos` (build a blind rating sheet with `build_test`, collect ratings, then
`aggregate`). Compare real vs Keras-TTS vs XTTS.

## 9. Cross-generator test (the contribution)
```bash
# spectrogram detector vs your own fakes + real reference:
python -m src.detection.cross_test --model reports/run1/model.keras --model-type cnn \
    --feat lfcc --fakes data/generated --real data/raw/wavs --out reports/cross
```
For the SSL detector, first cache your clips' features with `features_ssl.py`, then
`cross_test --model-type ssl --ssl-manifest ...`.

> Reminder: the TF/torch training paths are unrun until this server session — the metrics
> math is unit-tested, but expect to fix a config detail or two on first run (I'm on SSH
> to debug). Keep failed runs — they're the "show the process" evidence.
