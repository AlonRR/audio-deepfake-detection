# Results & Analysis

Data-driven results for the two halves. **Detection is complete**; **creation is scaffolded**
(waiting on the target-voice recording). Method/background: `proposal.md`; references:
`literature.md`; wall-times + failure timeline: `runtimes.md`; raw data + figures:
`../reports/`. Regenerate figures: `scripts/plot_detection.py`.

---

## Part B — Detection *(complete)*

### B.1 Setup
- **Data:** ASVspoof 2019 LA — train 25,380 / dev 24,844 / eval 71,237 utterances.
  dev reuses train's attacks **A01–A06**; eval uses **unseen A07–A19** → eval is the real
  generalization test.
- **Metrics:** EER, min t-DCF (ASVspoof2019 legacy cost model), DET curves.
- **Front-ends:** log-mel (80-bin) · LFCC (60-dim, ASVspoof standard) · SSL = frozen
  **wav2vec2-XLS-R-300M** (1024-d, layer-averaged, 200 frames).
- **Back-ends (own, Keras):** 2-D CNN (baseline) · RawNet2 (raw waveform, SincConv + FMS
  + GRU) · attentive-statistics-pooling classifier over SSL features.
- **Compute:** NVIDIA L4 via Slurm + apptainer (TensorFlow container). Feature caching
  cut the full CNN run from ~6 h to 17 min; SSL features extracted once (PyTorch) and
  cached for the TF back-end (no cross-framework env clash).

### B.2 Main results
| Detector (full LA) | Dev EER | Eval EER (unseen) | Eval min t-DCF |
|---|---:|---:|---:|
| CNN · log-mel | 18.99% | — | — |
| CNN · LFCC | 12.99% | 18.55% | 0.3835 |
| RawNet2 (raw) | _running_ | _todo_ | _todo_ |
| **SSL XLS-R + Keras back-end** | **0.04%** | **0.668%** | **0.0189** |

Figures: `reports/figures/eer_tdcf_comparison.png`, `det_eval_overlay.png`,
`det_dev_overlay.png`, and per-run learning curves.

### B.3 Analysis
- **The SSL frontend dominates and *generalizes*.** On unseen attacks it holds at
  **0.67% EER** vs the LFCC-CNN's **18.55%** — a **~28×** gap. Its dev→eval degradation is
  tiny (0.04 → 0.67%), whereas the hand-crafted CNN degrades sharply (12.99 → 18.55%):
  spectrogram CNNs latch onto attack-specific artifacts, while the self-supervised
  representation captures attack-agnostic cues.
- **Front-end matters for the baseline:** LFCC (12.99%) beats log-mel (18.99%) on dev,
  consistent with LFCC being the ASVspoof reference front-end.
- **min t-DCF tracks EER**, confirming the ranking under the ASV-aware cost.

### B.4 "Show the process" — tries, failures, fixes *(kept as evidence)*
1. **Pipeline validation on synthetic data** (`smoke_*` runs): all three model types train
   + emit EER/DET on a tiny generated set before touching real data. The *first* smoke
   **failed** — a numpy ABI clash from `pip --target` shadowing the container numpy — fixed
   with `--user` installs + a pinned `numpy==1.26.4` constraint.
2. **The subset trap** (`base_cnn_lfcc_sub`): a 4k biased subset gave a **misleading 0.0%
   EER** (first-2000 spoof = one attack, trivially separable; train acc ~1.0 / val ~0.76 —
   overfit). This is *why* we evaluate on the full set + eval. Textbook "don't trust an
   easy subset."
3. **Infra failures & fixes** (see `runtimes.md`): OOM on the 7 GB default memory (log-mel
   peak 7.16 GB; XLS-R extractor leaks ~7 GB / 27k utts) → `--mem` 16–32 GB; home **~50 GB
   quota** exceeded mid eval-extraction → prune caches + resumable extraction.
4. **Hyper-parameter search + a deliberate failed run** — _SCAFFOLD (next):_ sweep the SSL
   back-end (learning rate, pooling type, dropout, projection width) and include one
   over-/under-fit run with analysis. Requires re-extracting SSL features (freed for quota).

### B.5 Innovation — cross-generator test *(scaffold; needs creation half)*
Score the trained detector on our **own** XTTS / Keras-TTS clips — an unseen, 2025-era
synthesizer — and compare its score distribution vs real speech and vs its ASVspoof
operating point. Tool ready: `src/detection/cross_test.py`. _Results TBD._

---

## Part A — Creation *(scaffold — waiting on the target-voice recording)*

> Blocked only on the 1–5 min voice recording (`data/raw/reading_script.txt`). Once
> provided, the XTTS fine-tune (checkpoint/resume across the 2 h QOS cap) and the Keras
> Tacotron2-lite baseline run, then the evaluation harness fills the tables below.

### A.1 Setup *(planned)*
- Target voice: student's own, 1–5 min → faster-whisper segmentation → LJSpeech-style set.
- Systems: **A1** from-scratch Keras Tacotron2-lite (weak baseline, Griffin-Lim) · **A2**
  fine-tuned **XTTS-v2** (Coqui). Synthesize **≥10 clips** of varying length/complexity.

### A.2 Quality metrics *(tables to fill)*
| Metric | Real (ref) | A1 Keras-TTS | A2 XTTS-v2 |
|---|---|---|---|
| MCD (dB) ↓ | — | _TBD_ | _TBD_ |
| Log-mel SSIM ↑ | — | _TBD_ | _TBD_ |
| Speaker cosine (ECAPA) ↑ | 1.00 | _TBD_ | _TBD_ |
| MOS (1–5, panel) ↑ | _TBD_ | _TBD_ | _TBD_ |

Figures to generate: mel-spectrogram comparison (real vs A1 vs A2), MCD bars, speaker-cosine
bars, MOS bars, TTS training curves, and the **attention-alignment** plot (A1 attention
collapse = the creation-side "show the process" story).

### A.3 Analysis *(scaffold)*
Expected arc: A1 rough (too little data, Griffin-Lim artifacts, exposure bias / attention
collapse) → A2 markedly better (pretrained Transformer + speaker embedding + HiFi-GAN).
The A1→A2 quality gap *is* the "what failed / how I improved" narrative.

---

## Reproducibility
- **Numbers/data:** `reports/detection/<run>/{result.json,history.csv,*_scores.npz}`.
- **Figures:** `reports/figures/` — rebuild with
  `uv run --no-project --with numpy --with matplotlib --with scipy python scripts/plot_detection.py`.
- **Refresh from lab:** `bash scripts/fetch_results.sh` (then commit).
- **Timeline/wall-times:** `docs/runtimes.md`.
