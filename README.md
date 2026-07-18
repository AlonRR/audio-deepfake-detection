# Audio Deep Fakes — Creation & Recognition

Final project for **Neural Networks** (Shenkar College, Dr. Eran Kaufman). Two halves:

1. **Creation** — clone a target speaker's voice from 1–5 min of speech and synthesize
   ≥10 clips, using a **Keras RNN+attention TTS baseline** (built from scratch) *and* a
   fine-tuned pretrained **XTTS‑v2**. Evaluate with **MOS**, **spectrogram similarity**,
   and **speaker‑embedding cosine similarity**.
2. **Detection** — an anti‑spoofing detector using a **frozen self‑supervised frontend**
   (wav2vec2‑XLS‑R / WavLM) feeding a **back‑end trained in Keras**, benchmarked on
   ASVspoof 2019 LA (EER / min t‑DCF), then **cross‑tested on our own generated clips**.

> **Status: both halves complete.** Headline numbers:
> **detection** — SSL frontend + Keras back-end reaches **0.67% EER** on unseen ASVspoof
> eval attacks vs **18.55%** for the CNN-LFCC baseline;
> **creation** — fine-tuned XTTS-v2 reaches **0.449** speaker cosine against a
> real-vs-real ceiling of **0.848**, while the from-scratch Keras baseline fails
> completely (and *diagnosably*);
> **cross-generator test** — both detectors collapse on our own recordings, scoring
> *genuine* speech at **0.0009** bona-fide (**n=4 real clips — directional, not precise**).
> That failure is the project's main finding.
>
> Results: [`docs/results.md`](docs/results.md) · method: [`docs/proposal.md`](docs/proposal.md)
> · references: [`docs/literature.md`](docs/literature.md).
> **MOS is built but unrated** — it needs human listeners and no number is invented.

## Layout

```
docs/            results.md (headline numbers), proposal.md (method), literature.md,
                 runtimes.md (wall-times), server_runbook.md, goal.md
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
UV_PROJECT_ENVIRONMENT=.venv-creation  uv sync --extra creation --extra ssl

# Part B (Keras detection back-end + Keras TTS baseline) — TensorFlow side
UV_PROJECT_ENVIRONMENT=.venv-detection uv sync --extra detection
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
