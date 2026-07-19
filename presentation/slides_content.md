# Slide content — paste-ready

Every number here is copied from a committed artifact. The source is named under each
slide so you can defend it. Existing deck slides are marked **KEEP**, **FIX** or **ADD**.

Target order (16 slides). Your current deck supplies 1, 2, 3–6 and 15.

---

## 1. Title — **KEEP**

SYSTEM: NEURAL NETWORKS AUDIO PROCESSING
Audio Deepfake Voice Cloning & Anti-Spoofing Detection

---

## 2. Problem & motivation — **ADD** (goes before your Objective slide)

**Why this matters**

- Cloning a voice convincingly now takes **minutes of audio**, not hours.
- Detection is an arms race: countermeasures are trained and scored on benchmarks
  built from **2019-era** synthesis.
- **The gap:** no public benchmark contains deepfakes from 2024–25 TTS systems.
- So the question this project asks is not "can I clone a voice" — it is
  **"does a detector that scores 0.67% EER on the benchmark actually work on
  audio from outside it?"**

*Source: docs/proposal.md § Problem & motivation*

---

## 3. Objective — **KEEP** (one correction)

Keep as-is, but change *"Compare a custom Keras RNN baseline against a
state-of-the-art pretrained model"* → add the delivered counts so the promise is
visibly met later:

- Part 1: clone from **1–5 min** → delivered **4:36** · synthesize **≥10 clips** →
  delivered **36** (12 each from three systems)
- Part 2: detection paper from 2020+ → **Tak et al. 2022** (wav2vec2 + AASIST recipe)

---

## 4–6. XTTS literature — **KEEP, with four line-edits**

| Slide | Currently says | Change to |
|---|---|---|
| 4 | "Converts **raw audio** into Discrete Audio Tokens using a **neural audio codec**" | "Converts a **mel-spectrogram** into discrete audio tokens using a **VQ-VAE**" (your own next slide says mel-spectrogram — the deck contradicts itself) |
| 4 | "Zero-Shot cloning from just **minutes** of source audio" | "from only a few **seconds** of reference audio (3–8 s in the paper's evaluation)" — minutes is your *fine-tuning* budget, not the zero-shot prompt |
| 5 | "which **significantly** improved expressiveness" | "which the authors report improved expressiveness in **preliminary experiments**" (no ablation in the paper) |
| 5 | Perceiver Resampler "**drastically** improves speaker cloning" | "replaces a single speaker embedding with 32; the authors found a *single* embedding **degraded** cloning once training scaled to many languages" |

---

## 7–8. SSL frontend — **FIX: retarget from WavLM to XLS-R**

You never ran WavLM. `src/detection/features_ssl.py:24` sets
`DEFAULT_FRONTEND = "facebook/wav2vec2-xls-r-300m"`; every Slurm script hardcodes
XLS-R; every result directory is `ssl_xlsr_*`. Retitle:

**Paper 2: wav2vec 2.0 / XLS-R** — Baevski et al. 2020 (arXiv:2006.11477);
Babu et al. 2021 (arXiv:2111.09296)

- **Concept:** self-supervised speech representation learned without transcripts —
  used here as a **frozen** feature extractor, exactly like pretrained word
  embeddings (`word embedding.pptx`).
- **Method:** contrastive masked prediction over a quantized codebook.
  XLS-R scales this to **436k hours across 128 languages**.
- **Relevance:** the representation preserves the micro-artifacts that generative
  vocoders leave behind — which a hand-crafted spectrogram front-end discards.

**Slide 8 survives almost unchanged** — the conv feature encoder spec is identical
in wav2vec2/XLS-R (7 blocks, 512 channels, strides (5,2,2,2,2,2,2), kernels
(10,3,3,3,3,2,2), ~25 ms receptive field / 20 ms stride). Only two items are
WavLM-exclusive and must go: *masked speech **denoising*** and ***gated*** relative
position bias.

**Also add the paper you actually reproduced:** Tak et al. 2022
(arXiv:2202.12233) — "ASV spoofing and deepfake detection using wav2vec 2.0 and
data augmentation". That is the 2020+ detection paper the brief requires.

*If you want to keep WavLM: one line on the future-work slide, worded as
docs/literature.md already does — "the second frontend to try".*

---

## 9. Approach — Detection — **ADD**

**Three tries, escalating**

1. **First try:** 2-D CNN over log-mel / MFCC-60 spectrograms — the obvious baseline.
2. **Second:** RawNet2 from raw waveform (SincConv → residual + FMS → GRU).
3. **Main system:** **frozen XLS-R** → *my* Keras back-end
   (Conv1D → **attentive statistics pooling** → classifier).

**What is mine:** the entire back-end architecture, training and tuning; the
evaluation harness; the cross-generator experiment. XLS-R is a pretrained tool,
used frozen and stated plainly.

**Course link:** attentive statistics pooling is the same softmax-weighted sum as
seq2seq attention (`attention.pptx`), repurposed to aggregate a variable-length
utterance into one fixed vector.

*Source: docs/results.md §B.1*

---

## 10. Detection results — **ADD**

| Detector (full LA) | Dev EER | **Eval EER (unseen A07–A19)** | Eval min t-DCF |
|---|---:|---:|---:|
| CNN · log-mel | 19.00% | — | — |
| CNN · MFCC-60 | 12.99% | 18.55% | 0.3835 |
| RawNet2 (from scratch) | collapsed | — | — |
| **XLS-R + my Keras back-end** | **0.04%** | **0.668%** | **0.0189** |

- **~28× better** than the hand-crafted baseline on *unseen* attacks.
- Dev→eval degradation is tiny for SSL (0.04 → 0.67%) but sharp for the CNN
  (12.99 → 18.55%): the spectrogram CNN latches onto attack-specific artifacts;
  the self-supervised representation captures attack-agnostic cues.

**Against published numbers**

| System | Eval EER | Source |
|---|---:|---|
| RawNet2 (official baseline) | ~4.7% | Tak et al. 2021 |
| wav2vec2 + AASIST | ~0.2% | Tak et al. 2022 |
| **This work (frozen frontend, no augmentation)** | **0.668%** | `ssl_xlsr_eval/result.json` |

*Figures: `eer_tdcf_comparison.png`, `det_eval_overlay.png`*

---

## 11. Show the process — **ADD** (this is a graded criterion)

**Every one of these is a committed artifact, not a story.**

1. **First smoke test failed** — numpy ABI clash from `pip --target` shadowing the
   container numpy. Fixed with `--user` + a pinned `numpy==1.26.4` constraint.
2. **The subset trap** — a 4k biased subset reported a *perfect* **0.0% EER**.
   Train **and** validation both hit ~1.00. Not over-fitting — the subset's spoof
   half was effectively one attack type, so it carried **no signal at all**.
   That is why everything is scored on the full set plus unseen-attack eval.
3. **RawNet2 collapsed** — `val_accuracy` pinned at **0.897**, exactly the
   majority-class rate, while train accuracy drifted *down*. Kept as evidence.
4. **Infrastructure** — OOM on 7 GB default; ~50 GB home quota exhausted mid-run.

*Figure: `base_cnn_lfcc_sub_learning_curves.png` — the 0% artifact*

---

## 12. Hyper-parameter search + a deliberate failure — **ADD** (graded criterion)

| lr | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 1e-1 |
|---|---:|---:|---:|---:|---:|---:|
| dev EER | **0.033%** | **0.033%** | 0.040% | 0.040% | 1.338% | **diverged** |

- **Robust across two orders of magnitude**, then breaks at 1e-2 (training collapses
  after epoch 3, oscillating between the all-spoof and all-bonafide poles) and
  **diverges at 1e-1** (loss 9.11 at epoch 0 — the deliberate failed run).
- **Honest note for the viva:** the headline 0.668% comes from the **pre-sweep
  default lr=1e-3**. The sweep ran afterwards; its winner was never re-scored on
  eval. The gap is 0.007 pp on 25k trials — inside run-to-run noise.

*Figure: `ssl_lr_sweep.png`*

---

## 13. Approach — Creation — **ADD**

- **A1 — from scratch, in the course framework:** Keras Tacotron2-lite
  (text encoder → attention → autoregressive mel decoder → Griffin-Lim).
  Trained on 4 minutes. **Expected to fail — that is its job.**
- **A2 — pretrained:** XTTS-v2 fine-tuned on the same 4 minutes.
- **Data:** 4:36 recording, 40.8 dB SNR → 34 clips (**30 train / 4 held out**).
  The held-out clips are excluded from training *by construction*.

*Course links: RNN (`ML4.pptx`), attention (`attention.pptx`), speaker embedding
(`word embedding.pptx`), HiFi-GAN vocoder (`L18_gan__slides.pdf`)*

---

## 14. Creation results — **ADD**

| System | MCD ↓ | log-mel L2 ↓ | log-mel corr ↑ | **Speaker cosine ↑** |
|---|---:|---:|---:|---:|
| A1 Keras TTS | 39.43 | 26.63 | 0.011 | **−0.020** |
| A2 XTTS zero-shot | 27.66 | 22.72 | 0.092 | 0.430 |
| **A2 XTTS fine-tuned** | **25.25** | **22.37** | **0.180** | **0.449** |
| *real (ceiling)* | — | — | — | **0.848** |

- **Every metric orders the systems identically:** A1 ≪ zero-shot < fine-tuned.
- **A1's cosine is −0.020 — indistinguishable from zero.** It is not a bad clone;
  it is not a clone.
- **Fine-tuning worked:** eval loss 3.825 → 3.485 (best epoch 9; epoch 10 rose, so
  the trainer early-stopped), log-mel correlation doubled.
- **Honest ceiling:** real-vs-real is 0.848; the clone reaches 0.449. Four minutes
  gives a recognisable voice that a speaker-verification model still separates.

**MCD caveat:** ~25 is *not* comparable to published 4–8 dB — those use MGC-based
cepstral analysis; this is a DCT of the log-mel spectrum. Used to **rank**, not grade.

---

## 15. Why A1 failed — **ADD** (strongest single slide in the deck)

**Its output is provably independent of its input.**

- All 12 clips are **exactly 9.28 s** — the decoder's `max_steps` cap — whether the
  prompt is one word or thirteen. **Duration std = 0.00 s** (real speech: 2.35 s).
- The stop token **never fired once**.
- 7/12 clips near-silent, 5/12 saturated. Mean ZCR **0.241** vs **0.154** for real
  speech — noise, not voiced audio.

**Why:** 30 clips cannot train a seq2seq TTS from scratch. Attention never converged
to a monotonic text↔mel alignment, and the stop token is a *single positive per
sequence* — far too rare to learn from 30 examples.

*Source: `reports/evaluation/signal_stats.json`*

---

## 16. Innovation — the cross-generator test — **ADD**

**Does a 0.67%-EER detector work on audio from outside its benchmark?**

| Detector | ASVspoof eval | **On my own clips** | my **fakes** | my **real** |
|---|---:|---:|---:|---:|
| XLS-R + Keras | 0.67% | **29.17%** | 0.137 | **0.0009** |
| CNN-MFCC | 18.55% | **70.83%** | 0.086 | 0.057 |

**Both collapse — and not the way I predicted.**

The decisive number is the last column: the detector scores my **genuine** speech at
**0.0009** bona-fide — *more confident my real voice is fake than my clones are.*

**Why:** channel shift, not generator novelty. ASVspoof's bona-fide class is VCTK —
one studio mic, one chain. My recording is a browser capture in a bedroom. The model
learned **the benchmark's recording conditions**, not authenticity. And XTTS output,
being smooth and denoised, is *closer* to VCTK than my real audio is.

> **Caveat, stated up front: n = 4 real clips.** Directional, not precise. And this
> experiment does **not** isolate channel from generator — that needs a controlled
> follow-up (In-the-Wild, or re-encoding ASVspoof audio through my capture chain).

---

## 17. Live demo — **ADD**

`tools/webui/run.cmd` → http://localhost:7860

1. Type text → the cloned voice speaks it → **"Send to detector"**
2. **Record my own real voice** → analyze

**Say this before pressing Analyze:** *"This will call my real voice synthetic —
here's why."* The UI deliberately **declines to give a verdict**, because at any
threshold the ranking is inverted (real 0.0009 < fakes 0.137). It shows the score
against both measured reference points instead.

*Fallback if the VPN fails: pre-generated clips in `data/generated/`.*

---

## 18. Limitations & future work — **ADD**

- Single speaker; **n=4** held-out real clips.
- Dev EERs are **selection-optimistic** — dev is simultaneously validation data,
  early-stopping monitor and the reported split. Eval is the only clean number.
- Frozen frontend, no augmentation — the published 0.2% fine-tunes and augments.
- The proposal promised a five-axis sweep; **one axis ran** (2 h QOS cap, 1 GPU).
- **In-the-Wild was never run** — it is the cheapest experiment that would separate
  channel shift from generator shift.
- MCD is not comparable to published values (see slide 14).
- MOS: blind test **built**, ratings **pending** — 40 clips, token-named, key held
  separately. No MOS number is invented.

**Next:** SSL fine-tune · AASIST graph back-end · In-the-Wild · augmented training
so the model cannot use channel as a shortcut.

---

## 19. Ethics — **ADD**

- **Own voice only** — no third-party consent issue.
- The source recording and every cloned clip are **deliberately kept out of the
  public GitHub repo** (private backup instead). Zero audio files are tracked.
- Defensive framing: the project's finding is that **current detectors under-perform
  out of domain** — which argues for caution about deployment claims, not for
  building better fakes.

---

## 20. Engineering standards — **FIX slide 9**

Replace the tree with the real one:

```
audio-deepfake-detection/
├── notebooks/   01_creation · 02_evaluation · 03_detection · 04_cross_test
├── src/
│   ├── common/      audio I/O, features, metadata      (14 import sites)
│   ├── creation/    keras_tts/ · xtts_finetune/
│   ├── detection/   features_ssl · backend_keras · baseline_cnn · cross_test
│   └── evaluation/  spectrogram_sim · speaker_sim · mos · signal_stats
├── scripts/     27 Slurm jobs — every result regenerable
├── tests/       10 passing (EER, t-DCF, MCD, correlation)
├── reports/     result.json per run + 30 figures
└── docs/        proposal · results · literature · runtimes · runbook
```

**The engineering-standards argument, stated:** *every headline number is a
committed `result.json`, produced by a checked-in Slurm script, and cross-checked
against the prose.*

Fix the three factual errors on the current slide:
- notebooks are `01_creation / 02_evaluation / 03_detection / 04_cross_test`
  (three of your four names don't exist)
- the root is `audio-deepfake-detection`, not `research_codebase`
- *"calculating subjective metrics (MOS)"* → "MOS harness (blind test + aggregation),
  **ratings pending**"; the objective metrics are MCD, log-mel L2, log-mel
  correlation, ECAPA speaker cosine
