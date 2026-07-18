# Oral presentation — outline & examiner Q&A

Structure per `skills/nn-final-project/SKILL.md` § 5. **All slides and examiner answers
are filled** — both halves ran end-to-end. The only pending item is the MOS panel.
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
6. **Approach — creation** *(filled)* — A1 Keras Tacotron2-lite trained from scratch on
   4 min (`ML4.pptx` RNN, `attention.pptx`) → A2 XTTS-v2 fine-tune (Transformer +
   speaker embedding + HiFi-GAN vocoder; `word embedding.pptx`, `L18_gan__slides.pdf`).
7. **Creation results** *(filled)* — headline: **every metric orders A1 ≪ zero-shot <
   fine-tuned**; speaker cosine **0.449** vs a real-vs-real ceiling of **0.848**; SSIM
   doubled 0.092 → 0.180 with fine-tuning. A1 cosine **−0.020** = not a clone at all.
   XTTS loss curve 3.825 → 3.485, best epoch 9 (epoch 10 rises → early stop).
8. **Innovation — cross-test** *(filled)* — **both detectors collapse**: SSL 0.67% →
   **29.17%** EER, CNN 18.55% → **70.83%** (worse than chance). The decisive number:
   the SSL detector scores my **real** held-out speech at **0.0009** bona-fide. Most
   consistent with channel shift, not generator novelty. **State the caveat on the slide:
   n=4 real clips — directional, and the experiment does not isolate channel from
   generator.**
9. **Demo** — `tools/webui` (run `tools/webui/run.cmd`): type text → cloned voice speaks
   it → "Send to detector". **Record your own real voice live and let it be flagged
   SYNTHETIC** — that is §B.5 demonstrating itself. Say it *before* it happens.
   Fallback if the VPN fails: pre-generated clips in `data/generated/`.
10. **Limitations & future work** — single speaker, one GPU / 2 h cap, frozen frontend;
    next: SSL fine-tune, AASIST graph back-end, In-the-Wild test.
11. **Ethics** — defensive framing; **own voice only**, so no third-party consent issue;
    the cloned audio and the source recording are deliberately kept out of the public
    GitHub repo (private OneDrive backup instead); misuse note.

## Key figures
- **Ready:** `eer_tdcf_comparison.png`, `det_eval_overlay.png`, `det_dev_overlay.png`,
  per-run learning curves, `base_cnn_lfcc_sub_*` (the 0% artifact).
- **Numbers ready, plots optional:** XTTS loss curve (§A.1c table), MCD/corr/cosine
  (§A.2 table), cross-test (§B.5 table). Worth plotting if time allows: A1-vs-A2
  mel-spectrograms and the A1 attention-alignment map — they make the collapse visible.
- **Not available:** MOS bars (panel not yet run).

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

**You swept the learning rate — is the headline 0.67% from the best config?** No, and the
report says so. 0.67% comes from `ssl_xlsr_run1` at the **default lr=1e-3** (dev 0.040%).
The sweep ran afterwards and lr=1e-4/3e-4 edged it on dev (0.033%), but I never re-scored
that config on eval. The difference is ~0.007 percentage points on 25k dev trials — a
couple of decisions, inside run-to-run noise — so I would not expect eval to move
materially. The honest statement is that the headline is the pre-sweep default, and the
sweep's value was showing the back-end is *insensitive* to LR across two orders of
magnitude, then breaking at 1e-2 and diverging at 1e-1.

**Why did the 4k subset give 0% EER?** The subset's spoof half was effectively one attack
type, trivially separable → **both** train and val accuracy hit ~1.00 (final val 0.9998).
That is not classic over-fitting; the subset simply carries no signal. It proved subset
eval is deceptive and motivated full-data + unseen-eval evaluation.

**Why does SSL generalize but the CNN doesn't?** dev shares attacks A01–A06 with train;
eval has unseen A07–A19. SSL barely degrades (0.04 → 0.67%) while the CNN drops sharply
(12.99 → 18.55%): the spectrogram CNN latches onto attack-specific artifacts; the
self-supervised representation captures attack-agnostic speech cues.

**Why does your cepstral front-end beat log-mel?** Careful — I have to retract the answer
I first wrote. I labelled it "LFCC" and explained it as a linear filterbank preserving
high-frequency detail. On checking the code, `src/common/audio.py` builds an **HTK-mel**
filterbank: a comment claimed mel spacing was replaced with linear, and it never was
(measured peaks 31/312/719/1281/2031/3094/4594/6688 Hz — mel-spaced). So it is an
**MFCC-60**, both arms are mel, and the real contrast is cepstral-plus-CMVN vs raw
filterbank bins — not a frequency-scale effect. The 12.99% / 18.55% numbers are real;
the name and the explanation were not. The true LFCC comparison remains un-run.

**What is RawNet2?** End-to-end from raw waveform: learnable SincConv band-pass filters →
residual blocks with filter-wise feature-map scaling → GRU → classifier. No hand-crafted
features — the baseline that motivates learned front-ends.

**How is this "your own work" given pretrained tools?** The Keras back-end (architecture,
training, tuning), the Keras TTS baseline, the whole evaluation harness, and the
cross-generator innovation are mine; XTTS and XLS-R are used as pretrained tools, stated
plainly.

**Why did A1 fail, and how do you know it's not just "bad quality"?** Because its output
is provably independent of its input. All 12 clips are exactly 9.28 s — the `max_steps`
cap — whether the prompt is one word or thirteen; duration **std = 0.00 s** against 2.35 s
for real speech. The stop token never fired once. Speaker cosine is **−0.020**, i.e. zero.
Two causes compound: attention never converged to a monotonic text↔mel alignment
(`attention.pptx`), and the stop token is a *single positive per sequence* — far too rare
a signal to learn from 30 sequences. Griffin-Lim then adds phase artifacts on top of an
already-broken mel. Fixing it needs orders of magnitude more data, at which point it stops
being the "from scratch on my own voice" baseline the brief asks me to compare against.

**Did fine-tuning XTTS actually do anything, or did you just run it?** Independent
evidence on three axes. Eval loss fell 3.825 → 3.485 over 9 epochs and *rose* at epoch 10,
so the trainer early-stopped at 9 — I did not pick a round number. Log-mel SSIM doubled
(0.092 → 0.180). Zero-crossing rate moved from 0.084 to **0.150** against real speech at
0.154. Speaker cosine rose 0.430 → 0.449. Small but consistent in the same direction on
metrics that don't share a failure mode.

**Your MCD is ~52 — published MCD is 4–8 dB. Explain.** It is not comparable, and I say so
in the report. Published MCD uses MGC-based cepstral analysis (SPTK, alpha warping); mine
is a DCT of the log-mel spectrum, which lands about an order of magnitude higher. I use it
to *rank* systems on parallel utterances, not to grade them. I also found and fixed two
real bugs in it: `librosa.feature.mfcc` applies `power_to_db`, so the standard MCD constant
was scaling a second time (giving 476–665), and silent frames dominated the mean because
`log(~0)` swings wildly. After the fix, identical signals give exactly 0.00, and a
same-text synthesis (51.9) scores below a *different-text real recording of the same
speaker* (77.2) — so it tracks content, as MCD should.

**Why is speaker cosine only 0.449 — didn't the clone work?** It worked partially, and
overstating that would be dishonest. Real-vs-real is **0.848**; the clone reaches 0.449 —
recognisably the right speaker to a listener, but clearly separable to an embedding model.
Four minutes is simply not much data. The honest claim is "a convincing-sounding clone
that does not fool a speaker-verification system," not "a clone that defeats ASV."

**Your detector got 0.67% EER but 29% on your own clips. Which is the real number?**
Neither in isolation — and the interesting part isn't the fakes. My *genuine* held-out
speech scores **0.0009** bona-fide: the detector is more certain my real voice is fake than
that the XTTS clips are. That is channel shift. ASVspoof's bona-fide class is VCTK, one
studio mic and chain; my recording is a browser capture in a bedroom. The model learned the
benchmark's recording conditions, not authenticity — and XTTS output, being smooth and
denoised, is *closer* to VCTK than my real audio is. So 0.67% is partly a property of the
benchmark. The caveat is n_real = 4, so this is directional; the controlled follow-up is to
hold the channel constant and vary only the generator.

**Where is MOS?** Built, not scored. `reports/evaluation/mos/` holds a genuine blind test —
40 clips under opaque tokens, listener sheet (`token,rating`) separated from the hidden key.
I did not invent ratings; MOS requires human listeners and the panel is pending. An earlier
version of the harness leaked the system name into the listener's own sheet, which would
have invalidated the scores — that is fixed.
