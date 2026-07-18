# Results & Analysis

Data-driven results for the two halves — **both complete and run end-to-end on the lab
L4**. The one outstanding item is the **MOS listening panel**, which needs human raters
(the blind test is built; no score is invented). Method/background: `proposal.md`; references:
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

> **Which model produced 0.67%?** `reports/ssl_xlsr_run1` — trained at the **default
> `lr=1e-3`** (`train_ssl.slurm` does not override `train_ssl.py`'s default), dev EER
> 0.040%. The LR sweep in B.4(5) ran *afterwards* and found lr=1e-4 / 3e-4 marginally
> better on dev (0.033%), but **that config was never re-scored on eval** — the headline
> eval number therefore comes from the default LR, not the sweep winner. The gap is
> 0.033% vs 0.040% on ~25k dev trials, i.e. on the order of a couple of decisions and
> well inside run-to-run noise, so re-running eval would be unlikely to move 0.67%
> meaningfully. Stated explicitly because "you swept LR, so which model is the headline?"
> is a fair question and the honest answer is "the pre-sweep default."

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

### B.5 Innovation — cross-generator test *(done — job 520)*

Both detectors, scored on **our own clips**: 4 held-out real recordings vs fakes from a
2025-era synthesizer they have never seen.

| Detector | EER on ASVspoof **eval** | **EER on our clips** | mean score, our **fakes** | mean score, our **real** |
|---|---|---|---|---|
| SSL (XLS-R + Keras back-end) | **0.67%** | **29.17%** | 0.137 | **0.0009** |
| CNN-LFCC baseline | 18.55% | **70.83%** | 0.086 | 0.057 |

*(score = P(bona fide); n_fake = 36 SSL / 12 CNN, n_real = 4)*

**Both detectors collapse — and the failure is not the one we predicted.**

The proposal expected the detectors to miss *new fakes*. Instead the decisive number is
the last column: the SSL detector scores our **genuine** held-out speech at **0.0009**
bona-fide — it is *more* confident that the real recording is fake than that the XTTS
clones are. Its headline "88.9% spoof detection rate" is therefore meaningless: it flags
essentially everything. The CNN is worse, at **70.83% EER** — materially worse than
chance, i.e. its ranking is inverted.

**Why: channel shift, not synthesis quality.** ASVspoof 2019 LA's bona-fide class is
VCTK — studio recordings through one microphone and one processing chain. Our real audio
is a browser capture on a different mic, in a different room, resampled. The
countermeasures learned *what the ASVspoof bona-fide channel sounds like*, and treat
anything outside it as spoof. That the fakes score *higher* than real speech follows
directly: XTTS output is smooth, denoised, studio-like — closer to VCTK than a real
bedroom recording is.

**This is a stronger claim than the one the proposal set out to make.** The documented
gap in the literature is "benchmarks contain no 2024–25 TTS." What this shows is sharper
and more uncomfortable: a detector reporting **0.67% EER** does not merely miss modern
fakes, it **fails to recognise out-of-domain real speech as real**. The 0.67% is a
property of the benchmark's recording conditions as much as of the model. Any deployment
claim resting on ASVspoof EER alone is unsafe.

> **Limitations, stated plainly.** `n_real = 4` — far too few for a stable EER, so treat
> these as directional, not precise. And per A.1d, A1's clips are noise, so their
> detection is trivial and inflates the SSL detector's spoof-detection rate; the
> meaningful fake signal is the 24 XTTS clips. A proper follow-up would hold recording
> conditions constant — e.g. re-record the real reference through a VCTK-like chain, or
> score ASVspoof bona-fide audio re-encoded through our capture path — to separate
> *channel* shift from *generator* shift. That experiment is the natural next step and
> is not claimed here.

---

## Part A — Creation *(complete; MOS ratings pending)*

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

**Train / eval split — 30 train · 4 held out** (`--holdout 4`):

`metadata.csv` (train) · `metadata_eval.csv` (held out) · `metadata_all.csv`. The
trainers read `metadata.csv`, so the held-out clips are excluded *by construction* —
they cannot be trained on by accident.

The holdout gives speaker-cosine a **real-vs-real ceiling** and MOS a **real anchor**
drawn from audio the model never saw; scoring those against training audio would
flatter both metrics. Selection is constrained three ways:

| Constraint | Why |
|---|---|
| No phrase shared with a neighbour | The re-reads mean adjacent clips can be near-duplicates. Holding out one half while its twin trains would leak the eval audio into the model. |
| ≥ 4 s of speech | ECAPA embeddings are unstable on very short clips, and two words cannot be MOS-rated. (The first pass picked the 1.8 s `"stone chimney."` — hence the floor.) |
| Spread across the recording | Keeps all script sections represented instead of sampling one region. |

Held out: `clip_0004` (6.6 s, narration) · `clip_0013` (7.4 s, expressive) ·
`clip_0019` (12.2 s, numbers/names) · `clip_0030` (6.5 s, phonetic). **Verified: no
5-gram of any held-out transcript occurs anywhere in the training set.**

- Systems: **A1** from-scratch Keras Tacotron2-lite (weak baseline, Griffin-Lim) · **A2**
  fine-tuned **XTTS-v2** (Coqui). Synthesize **≥10 clips** of varying length/complexity.

### A.1b Getting the creation half to run at all — the environment log

The creation pipeline had never been executed before this pass. Seven distinct
blockers surfaced, each hidden behind the previous one; **none were in the model
code**. Recorded because the brief rewards showing the process, and because
several are traps any student on this cluster would hit.

| # | Symptom | Actual cause | Fix |
|---|---|---|---|
| 1 | Job rejected at submit | `--time=08:00:00` but the `bsc` QOS caps **MaxWall at 2 h** | 2 h; the run needs ~7 min anyway |
| 2 | `ResolutionImpossible`, blamed `numpy>=2` | Misleading. The container sets `PIP_CONSTRAINT=/etc/pip/constraint.txt` pinning `tensorboard==2.16.2`, blocking `coqui-tts-trainer` (needs ≥2.17). Pip backtracked to old coqui versions and reported *their* numpy pin instead | override `PIP_CONSTRAINT` with our own (numpy-only) file |
| 3 | `No module named 'torchaudio'` | NGC PyTorch container ships torch but not torchaudio | install it |
| 4 | torchaudio import RuntimeError | torch built on **CUDA 12.9**, PyPI torchaudio on **12.6**; no cu129 wheel exists for 2.7.0 | use the **CPU** torchaudio wheel — it carries no CUDA version to clash, and torchaudio is only used for audio I/O here |
| 5 | `cannot import name 'isin_mps_friendly'` | **Upstream bug**: `coqui-tts` declares `transformers>=4.57` with no upper bound but calls an API **transformers 5 removed** | pin `<5` in a *separate* user-site (`.pyuser_xtts`) so the detection env, which needs 5.x for XLS-R, is untouched |
| 6 | `IndexError: LJSpeech format expects 3 pipe-delimited columns` | We wrote `id\|text`; the formatter wants `id\|text\|normalized_text` | emit 3 columns |
| 7 | Job `OUT_OF_MEMORY` on synthesis | `best_model.pth` is **5.6 GB** — a trainer checkpoint carrying optimizer state, not 1.9 GB of weights | strip to `{"model": ...}` before inference |

Ties to `L18_gan__slides.pdf` only at the vocoder level; the debugging itself is
infrastructure, but items 2, 4 and 5 are worth an oral answer because they show
the difference between *what an error says* and *what is actually wrong*.

### A.1c XTTS-v2 fine-tune *(done — job 509, 7 min 11 s on the L4)*

30 train clips, batch 3, lr 5e-6, 10 epochs ≈ **100 optimizer steps**.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval loss | 3.825 | 3.707 | 3.634 | 3.573 | 3.525 | 3.504 | 3.499 | 3.487 | **3.485** | 3.488 |

Monotonic decline for 9 epochs (−8.9%), then epoch 10 **rises** — the trainer
selected epoch 9 (`best_model_81.pth`). Two honest observations:

- The model *did* move, but plateaued hard by epoch 7. More epochs at this
  learning rate buys very little; the gain would have to come from a higher lr,
  which risks damaging a 1.9 B-parameter pretrained model on 4 minutes of data.
- The predicted outcome (that ~100 steps would barely shift a model this size)
  was **partly wrong** — the loss did fall meaningfully — and partly right, in
  that it converged almost immediately. The zero-shot-vs-fine-tuned comparison
  in A.2 is what settles whether that loss drop is audible.

### A.1d A1 Keras Tacotron2-lite *(done — job 512, 1 h 07 m, 300 epochs)*

Train loss fell 0.167 → 0.113. **The model still failed completely**, in a specific and
diagnosable way — which is what this baseline exists to demonstrate.

| Symptom | Evidence |
|---|---|
| **Stop token never learned** | All 12 clips are **9.28 s** — exactly the `max_steps=800` cap — whether the prompt is 1 word (`"hello."`) or 13. Duration **std = 0.00 s**; real speech is 2.35 s. |
| **Output is independent of the text** | Same corollary: nothing about the input changes the output length. Attention never aligned, so the decoder is not conditioned on the text at all. |
| **Bimodal collapse** | 6/12 clips near-silent (rms ≤ 0.008, peak ≤ 0.13); the other 6 saturate (peak = 1.000). |
| **Not speech** | Mean ZCR **0.241** vs **0.154** for real speech — noise-like, not voiced. |

**Why:** 30 clips / ~4 min cannot train a seq2seq TTS from scratch. Two failures compound —
attention never converges to a monotonic text↔mel alignment (`attention.pptx`), and the
stop token is a *single positive example per sequence*, far too rare a signal to learn from
30 sequences. Griffin-Lim then adds phase artifacts on top of an already-broken mel.

This is the intended contrast with A2, not a bug to fix. Fixing it would need
orders-of-magnitude more data — i.e. it would stop being the "from scratch on your own
voice" baseline the assignment asks to compare against.

### A.1e Preliminary signal-level comparison *(crude proxies — not the evaluation metrics proper)*

| System | n | dur mean | dur std | rms | ZCR | peak |
|---|---|---|---|---|---|---|
| **real (held-out)** | 4 | 8.14 s | 2.35 | 0.054 | **0.154** | 0.531 |
| A1 keras_tts | 12 | 9.28 s | **0.00** | 0.027 | 0.241 | 0.452 |
| A2 XTTS zero-shot | 12 | 3.65 s | 1.43 | 0.126 | 0.084 | **1.000** |
| A2 XTTS fine-tuned | 12 | 4.55 s | 1.18 | 0.037 | **0.150** | 0.368 |

Read with care — ZCR and RMS are cheap proxies, not perceptual measures; MCD / SSIM /
speaker-cosine in A.2 are authoritative. But two things already stand out:

1. **A1's zero duration variance** is the cleanest single number showing the baseline
   ignores its input entirely.
2. **Fine-tuning moved XTTS toward the real speaker**: ZCR 0.084 → **0.150** against a real
   0.154, and peak 1.000 → 0.368 against a real 0.531. That is preliminary evidence the
   −8.9% eval-loss drop in A.1c produced an audible change rather than a numerical one —
   the question A.1c explicitly left open.

> **Caveat that matters for the cross-generator test:** A1's clips are noise, so *any*
> detector will flag them trivially. The meaningful cross-generator signal comes from the
> **A2** clips, which are real speech from a 2025-era synthesizer. A1's detection rate
> should not be read as evidence the detector generalizes.

### A.2 Quality metrics *(done — jobs 523/524)*

Scored against the **4 held-out clips** (`metadata_eval.csv`), never against training
audio. MCD / L2 / SSIM use the **parallel** clips (same transcript as the real
reference); speaker cosine uses the 12 prompt clips (identity, not content).

| System | MCD ↓ | log-mel L2 ↓ | log-mel SSIM ↑ | Speaker cosine ↑ |
|---|---|---|---|---|
| A1 keras_tts | 93.39 | 26.631 | 0.011 | **−0.020** |
| A2 XTTS zero-shot | 56.00 | 22.721 | 0.092 | 0.430 |
| **A2 XTTS fine-tuned** | **51.68** | **22.370** | **0.180** | **0.449** |
| *real (ceiling)* | — | — | — | **0.915** |

**Every metric orders the systems identically: A1 ≪ zero-shot < fine-tuned.** Three
things worth saying out loud:

1. **A1's speaker cosine is −0.020 — statistically indistinguishable from zero.** The
   baseline produces nothing speaker-like at all, corroborating the signal analysis in
   A.1d (noise, not speech). It is not "a bad clone"; it is not a clone.
2. **Fine-tuning helped on every axis**, most clearly on log-mel SSIM, which *doubled*
   (0.092 → 0.180). Together with the ZCR shift in A.1e this answers the question A.1c
   left open: the −8.9% eval-loss drop was real, not cosmetic.
3. **The ceiling is honest and it is far away.** Real-vs-real cosine is **0.915**; the
   best clone reaches **0.449**. Four minutes of audio produces a recognisable but
   clearly imperfect clone — worth stating plainly rather than implying the clone
   "fooled" a speaker model.

#### Caveat on MCD — read before quoting the number

The MCD column is **not comparable to published MCD figures** (typically 4–8 dB for good
TTS). Published values come from MGC-based mel-cepstral analysis (SPTK/pysptk with alpha
warping); this implementation is a DCT of the log-mel spectrum, which lands about an
order of magnitude higher. It is valid for **ranking systems on parallel utterances**,
not as an absolute quality figure.

The first implementation was worse still and was fixed during this pass — two bugs:

| Bug | Effect | Fix |
|---|---|---|
| `librosa.feature.mfcc` applies `power_to_db` (10·log₁₀), but the MCD constant `10/ln10·√2` already assumes natural-log cepstra | scaling applied twice → MCD 476–665 | take natural log of the mel spectrum and DCT it directly |
| Silence frames included | `log(~0)` is hugely negative and swings with any noise, so silent frames dominated the mean | drop frames > 35 dB below the utterance peak |

Sanity checks after the fix: identical signals → **0.00**; a same-text synthesis (51.9)
scores *below* a different-text real recording of the same speaker (77.2), i.e. the
metric tracks content as MCD should. It remains noise-sensitive, which is the honest
reason to lean on SSIM and speaker cosine as the primary evidence.

### A.3 MOS — blind test built, **ratings pending** *(a suggested metric, not a required one)*

The brief says *"evaluate the quality using metrics such as: MOS · spectrogram
similarity · voice conversion similarity"* — **"such as"**, i.e. these are examples, not
a fixed checklist. The two objective ones are scored above; MOS is the only *perceptual*
measure, so it is worth having, but its absence is not a missing requirement.

`reports/evaluation/mos/` contains a genuine blind test: **40 clips** (4 real + 12 from
each of the three systems), copied under opaque `clip_NNN` tokens so the filename cannot
leak the system, shuffled with a fixed seed.

- `mos_sheet.csv` — `token,rating` only. **This is the file listeners get.**
- `mos_key.csv` — `token,path,system`. Kept back until ratings are collected.
- `audio/clip_NNN.wav` — the clips.

Aggregate with `src.evaluation.mos.aggregate([sheets], key_csv)` → per-system mean and
95% CI. **No MOS number is reported anywhere in this repo until real listeners produce
one.** Roughly 5–10 raters, ~15 minutes each.

### A.4 Analysis — what the creation half showed

The predicted arc held, and the measurements pin down *why* rather than just *that*:

- **A1 did not merely sound rough — it never learned the task.** Output duration is
  constant at the decoder step cap (std **0.00 s**) regardless of input length, and
  speaker cosine is **−0.020**. Attention never aligned and the stop token was never
  learned from 30 sequences (§A.1d).
- **A2 is a working clone but not an identity match.** Fine-tuning improved every metric
  over zero-shot (SSIM 0.092 → 0.180, cosine 0.430 → 0.449), yet the real-vs-real ceiling
  is **0.915** — four minutes of audio yields a recognisable voice that a speaker-
  verification model still separates easily.
- The A1 → A2 gap is the "what failed / how I improved" narrative the brief asks for, and
  the fix was *not* more tuning of A1: it was recognising that a from-scratch seq2seq TTS
  cannot be trained on four minutes at all, and that the leverage lies in pretraining.

*Figures still worth adding if time allows:* mel-spectrogram comparison (real vs A1 vs A2)
and the A1 attention-alignment plot — both would make the collapse visible rather than
tabular.

---

## Reproducibility
- **Numbers/data:** `reports/detection/<run>/{result.json,history.csv,*_scores.npz}`.
- **Figures:** `reports/figures/` — rebuild with
  `uv run --no-project --with numpy --with matplotlib --with scipy python scripts/plot_detection.py`.
- **Refresh from lab:** `bash scripts/fetch_results.sh` (then commit).
- **Timeline/wall-times:** `docs/runtimes.md`.
