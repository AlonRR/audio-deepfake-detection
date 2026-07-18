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
| RawNet2 (raw, from scratch) | collapsed † | — | — |
| **SSL XLS-R + Keras back-end** | **0.04%** | **0.668%** | **0.0189** |

† RawNet2 trained from scratch **failed to converge** on our budget — see B.4(4).

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
4. **RawNet2 collapsed (documented failed baseline).** Trained from scratch, `val_accuracy`
   pinned at **0.897 — exactly the majority-class (spoof) rate** — while training accuracy
   *drifted down*; it never learned to discriminate (EER ≈ chance). A known RawNet2
   from-scratch sensitivity (learning-rate / SincConv init). Cancelled before the 2 h cap.
   Kept as evidence (`base_rawnet2_full/history.csv`); the SSL result stands as the strong
   model, CNN-LFCC as the working hand-crafted baseline.
5. **SSL back-end hyper-parameter search + a deliberate failed run** *(complete)*. Swept the
   back-end learning rate on the cached features (proj=256, dropout=0.3):

   | lr | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 † | 1e-1 ‡ |
   |----|-----:|-----:|-----:|-----:|-----:|-----:|
   | dev EER | **0.033%** | **0.033%** | 0.040% | 0.040% | 1.338% | diverged |
   | min t-DCF | **0.00027** | **0.00027** | 0.00126 | 0.00045 | 0.03075 | — |

   **Read:** robust for lr ≤ 3e-3 (dev EER ~0.03–0.04%). **lr=1e-4 and 3e-4 tie for
   best** (0.033% / 0.00027 on both metrics) — a small consistent gain over the 1e-3
   default; dev's resolution can't separate the two.
   † lr=1e-2 is already past the stability edge, not just "30× worse": training collapsed
   after epoch 3 (val-accuracy 0.99 → ~0.10–0.20, oscillating between the all-spoof and
   all-bonafide poles — `ssl_hp_lr1e2/history.csv`); the 1.338% is the epoch-2 best
   checkpoint restored by early stopping, not a stable operating point.
   ‡ At **lr=1e-1 training diverges immediately** — loss spikes to 9.11 at epoch 0 and
   val-accuracy flips between the class poles (0.10 / 0.90) in the 2 epochs recorded
   before the job hit the 2 h cap; it never reached scoring, so **no EER exists** (the
   figure marks it at the 50% chance line). That is the **deliberate failed run**
   (`ssl_hp_lr1e1_FAIL/history.csv`). Figure: `reports/figures/ssl_lr_sweep.png`.
   (Architecture variants proj=128 / dropout=0.5 were left un-run — confirmed never
   started on the server. Completed configs took ~21–32 min each: early stopping ends
   them at 15–24 epochs and `data_ssl` re-reads all 25k features per epoch; dev is
   saturated near 0, so LR was the informative axis. Ran as two 2 h jobs, 267+268, both
   hitting the cap — the sweep resumes via a per-config `result.json` skip-guard, and
   `*_FAIL` configs are skipped once their `history.csv` exists so a resume cannot
   overwrite the divergence evidence.)

### B.5 Innovation — cross-generator test *(scaffold; needs creation half)*
Score the trained detector on our **own** XTTS / Keras-TTS clips — an unseen, 2025-era
synthesizer — and compare its score distribution vs real speech and vs its ASVspoof
operating point. Tool ready: `src/detection/cross_test.py`. _Results TBD._

---

## Part A — Creation *(dataset ready — training pending)*

> The target-voice recording is **in hand** and segmented. Next: the XTTS fine-tune
> (checkpoint/resume across the 2 h QOS cap) and the Keras Tacotron2-lite baseline,
> then the evaluation harness fills the tables below.

### A.1 Setup *(done)*

**Source recording** — `data/raw/source.wav`, read from `data/raw/reading_script.txt`
via the browser recorder in `tools/` (raw capture: browser AGC / noise-suppression /
echo-cancellation disabled, since they smear the spectral detail cloning depends on).

| | |
|---|---|
| Format | 48 kHz · 16-bit · mono |
| Duration | 4:36 (276 s) |
| Peak / clipped samples | −0.7 dBFS / **0** |
| Noise floor | −69.4 dBFS |
| Estimated SNR | **40.8 dB** |
| Speech / silence | 68.5% / 31.5% |

**Segmented dataset** — `data/raw/{metadata.csv,wavs/}`, LJSpeech-style, via
faster-whisper (`small`, VAD) in `src/creation/xtts_finetune/prepare.py`:

- **34 clips · 4.27 min** of speech at 22.05 kHz; 1.8 / 7.4 / 12.2 s (min/median/max);
  577 words.
- Segment window widened to **1.5–13.0 s**. The original 2.0–12.0 s window dropped two
  boundary segments (13.9 s), one of them the numbers-and-proper-nouns line the script
  includes specifically for phonetic coverage.
- **Spoken-form transcript fixes** (`data/raw/transcript_fixes.json`, 6 applied):
  Whisper emits numerals for spoken numbers (`$89.50`, `1145`, `2026`), which misaligns
  text and audio for the *character-level* A1 baseline. A2/XTTS normalizes internally
  and is largely immune; the fix protects A1 from a confound that would otherwise be
  mistaken for attention failure.
- Three clips contain genuine **re-reads** (the speaker restarting a sentence).
  Verified against Whisper segment timings as non-overlapping in time, i.e. real
  repeated audio with correct matching text — valid training pairs, not transcription
  loops.

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
