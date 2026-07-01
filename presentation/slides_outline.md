# Oral presentation — outline & examiner Q&A (stub)

*Fill in during Week 3. Structure per `skills/nn-final-project/SKILL.md` § 5.*

## Slide outline (problem → approach → results → demo → limitations)

1. **Title** — Audio deep fakes: creation & recognition.
2. **Problem & motivation** — few-minute voice cloning; the detection arms race; the
   "no latest-TTS in benchmarks" gap.
3. **Approach — creation** — A1 Keras TTS baseline (fails on purpose) → A2 XTTS-v2
   fine-tune. Tie to RNN/attention/GAN course material.
4. **Creation results** — MOS / MCD / speaker-cosine table (real vs A1 vs A2);
   spectrograms; the "how I improved" story.
5. **Approach — detection** — first-try CNN → RawNet2 → **SSL frontend + my Keras
   back-end**; hyper-parameter sweep + a failed run.
6. **Detection results** — EER / min t-DCF / DET curve vs published baselines.
7. **Innovation — cross-test** — detector on my own clips; the generalization gap.
8. **Demo** — play real vs cloned clips; live/there-recorded detector call.
9. **Limitations & future work** — data size, single speaker, one GPU; next steps.
10. **Ethics** — defensive framing, own-voice-only, misuse note.

## Key figures to show
- XTTS fine-tuning loss curve; A1 vs A2 spectrograms.
- Detection learning curves + the hyper-parameter sweep table.
- DET curve; the cross-test EER bar (ASVspoof vs In-the-Wild vs my clips).

## Likely examiner questions (draft answers in Week 3)
- Why does A1 (Keras TTS) sound worse than A2? — data size, attention alignment,
  vocoder; what specifically failed and how I diagnosed it.
- Why freeze the SSL frontend instead of training end-to-end? — data/compute, transfer
  learning, defensibility; what fine-tuning would change.
- What is EER / min t-DCF and why report both?
- How does attention here relate to attention in the course (`attention.pptx`)?
- Why did the detector's EER rise on your own clips — and what does that prove?
- Where does the VAE/GAN material (`L16`/`L18`) actually appear in your pipeline?
