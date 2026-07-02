# Oral presentation — outline & examiner Q&A

Structure per `skills/nn-final-project/SKILL.md` § 5. **Detection slides + answers are
filled** (data in hand); **creation slides are scaffolded** (pending the voice recording).
Numbers: `docs/results.md`; figures: `reports/figures/`.

## Slide outline (problem → approach → results → demo → limitations)

1. **Title** — Audio deep fakes: creation & recognition.
2. **Problem & motivation** — few-minute voice cloning; the detection arms race; the
   "latest-TTS not in benchmarks" gap.
3. **Approach — detection** — first-try CNN (log-mel/LFCC) → RawNet2 → **frozen SSL
   frontend (wav2vec2-XLS-R) + my Keras back-end** (Conv1D → attentive-stats pooling →
   classifier). Hyper-parameter sweep + a deliberate failed run.
4. **Detection results** *(filled)* — headline: on **unseen** eval attacks,
   **SSL 0.67% EER vs CNN-LFCC 18.55% (~28×)**; dev-vs-eval bars + DET overlay.
   Figures: `eer_tdcf_comparison.png`, `det_eval_overlay.png`.
5. **Detection — show the process** *(filled)* — smoke tests (first one failed: numpy
   ABI) → the **4k-subset 0% EER trap** → full-data + eval; OOM & 50 GB-quota fixes.
   Figures: `base_cnn_lfcc_sub_learning_curves.png`, per-run curves.
6. **Approach — creation** *(scaffold)* — A1 Keras Tacotron2-lite (fails on purpose) →
   A2 XTTS-v2 fine-tune. Ties to RNN / attention / GAN course material.
7. **Creation results** *(scaffold)* — MOS / MCD / speaker-cosine table (real vs A1 vs
   A2); mel-spectrograms; the attention-collapse "how I improved" story.
8. **Innovation — cross-test** *(scaffold)* — detector on my own clips; the
   generalization gap that closes the creation↔detection loop.
9. **Demo** — play real vs cloned clips; live detector call on a held-out clip.
10. **Limitations & future work** — single speaker, one GPU / 2 h cap, frozen frontend;
    next: SSL fine-tune, AASIST graph back-end, In-the-Wild test.
11. **Ethics** — defensive framing, own-voice-only (clean rights), misuse note.

## Key figures
- **Ready:** `eer_tdcf_comparison.png`, `det_eval_overlay.png`, `det_dev_overlay.png`,
  per-run learning curves, `base_cnn_lfcc_sub_*` (the 0% artifact).
- **Pending creation:** XTTS fine-tune loss; A1-vs-A2 mel-spectrograms; attention
  alignment; MCD/cosine/MOS bars; cross-test score-distribution plot.

## Examiner Q&A — draft answers

**What are EER and min t-DCF; why both?** EER = the threshold where false-accept =
false-reject (one threshold-free number). min t-DCF = ASVspoof's detection cost combining
the countermeasure's errors with a fixed ASV system's errors — it reflects real
deployment cost, whereas EER ignores the downstream ASV. Ours (SSL, eval): EER 0.67%,
min t-DCF 0.0189; they track each other, confirming the ranking.

**Why freeze the SSL frontend instead of end-to-end?** Three reasons: (1) 300 M params on
a few-hour dataset would over-fit; (2) the 1-GPU / 2 h QOS makes extract-once-then-train-a-
light-back-end far cheaper (features cached, back-end trains in minutes); (3) **the
contribution is my Keras back-end** — the frozen encoder is used like pretrained
embeddings. Fine-tuning would likely push EER lower but at compute/over-fit risk (future
work).

**How does the pooling relate to course attention (`attention.pptx`)?** Attentive
statistics pooling learns softmax weights over the time axis, then takes a weighted
mean + std — the same softmax-weighted-sum as seq2seq attention, repurposed for temporal
aggregation into a fixed-length utterance vector.

**Why did the 4k subset give 0% EER?** The subset's spoof half was effectively one attack
type, trivially separable → the CNN overfit (train acc ~1.0, val ~0.76). It proved subset
eval is deceptive and motivated full-data + unseen-eval evaluation.

**Why does SSL generalize but the CNN doesn't?** dev shares attacks A01–A06 with train;
eval has unseen A07–A19. SSL barely degrades (0.04 → 0.67%) while the CNN drops sharply
(12.99 → 18.55%): the spectrogram CNN latches onto attack-specific artifacts; the
self-supervised representation captures attack-agnostic speech cues.

**Why LFCC > log-mel?** LFCC's linear filterbank keeps high-frequency detail where
vocoder/synthesis artifacts concentrate; log-mel compresses the high end.

**What is RawNet2?** End-to-end from raw waveform: learnable SincConv band-pass filters →
residual blocks with filter-wise feature-map scaling → GRU → classifier. No hand-crafted
features — the baseline that motivates learned front-ends.

**How is this "your own work" given pretrained tools?** The Keras back-end (architecture,
training, tuning), the Keras TTS baseline, the whole evaluation harness, and the
cross-generator innovation are mine; XTTS and XLS-R are used as pretrained tools, stated
plainly.

**(Creation, pending)** Why A1 worse than A2 (data size, attention collapse, Griffin-Lim);
what MCD / speaker-cosine measure; where GAN material appears (HiFi-GAN vocoder). — draft
once results are in.
