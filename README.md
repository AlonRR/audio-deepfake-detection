# Audio Deep Fakes — Creation & Recognition

Final project for **Neural Networks** (Shenkar College, Dr. Eran Kaufman). Two halves:

1. **Creation** — clone a target speaker's voice from 1–5 min of speech and synthesize
   ≥10 clips, using a **Keras RNN+attention TTS baseline** (built from scratch) *and* a
   fine-tuned pretrained **XTTS‑v2**. Evaluate with **MOS**, **spectrogram similarity**,
   and **speaker‑embedding cosine similarity**.
2. **Detection** — an anti‑spoofing detector using a **frozen self‑supervised frontend**
   (wav2vec2‑XLS‑R / WavLM) feeding a **back‑end trained in Keras**, benchmarked on
   ASVspoof 2019 LA (EER / min t‑DCF), then **cross‑tested on our own generated clips**.

> Status: **scaffold + proposal**. The full method, rationale, experiments, and timeline
> live in [`docs/proposal.md`](docs/proposal.md); the reading list is in
> [`docs/literature.md`](docs/literature.md). No model code is implemented yet.

## Layout

```
docs/            proposal.md (main), literature.md
data/            raw/ (your recording)  generated/ (fakes)  datasets/ (ASVspoof)  ← gitignored
src/creation/    keras_tts/ (baseline TTS)   xtts_finetune/ (XTTS-v2)
src/evaluation/  spectrogram_sim.py  speaker_sim.py  mos.py
src/detection/   features_ssl.py  backend_keras.py  baseline_cnn.py  train.py  evaluate.py
src/common/      shared audio/config helpers
notebooks/       01_creation  02_evaluation  03_detection  04_cross_test
models/  reports/  presentation/                                  ← models/ gitignored
```

## Environment (`uv`)

Torch and TensorFlow must **not** share one environment. Create two:

```bash
# Part A (voice cloning) + SSL frontend + speaker-embedding metric — PyTorch side
uv venv .venv-creation && uv sync --extra creation --extra ssl

# Part B (Keras detection back-end + Keras TTS baseline) — TensorFlow side
uv venv .venv-detection && uv sync --extra detection
```

On the **Shenkar Virtual Lab** these two map to the ready-made GPU containers
(`pytorch-25.04.sif` / `tensorflow-25.02.sif`) on the **NVIDIA L4 (24 GB)** node — see
`docs/proposal.md` § Feasibility & compute for the JupyterHub / Slurm workflow.

## OneDrive sync

This repo lives at `~/code/audio-deepfake-detection` and syncs into
`OneDrive/Shenkar/Neural Networks/Final_project_audio_detection` via the `ods` tool
(two-way, git-aware). Datasets, model checkpoints, and virtualenvs are gitignored, so
they stay local; only code, docs, notebooks, and small figures sync.

## License / ethics

Voice cloning is limited to the author's **own voice** (clean rights). This is a
defensive-research project (detecting synthetic speech); see the ethics note in
`docs/proposal.md`.
