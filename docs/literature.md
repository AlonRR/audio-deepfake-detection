# Literature — Audio Deep Fakes: Creation & Recognition

Curated reading list for the project. Each entry has a one-line **why read it**.
Entries marked **✓ verified** had their title/venue/ID confirmed via web search on
2026-07-01; entries marked **⟳ verify** are from memory — confirm the arXiv ID before
citing in the final report (per the course's source-checking rule).

---

## A. Voice cloning / TTS (creation)

- **Tacotron 2** — Shen et al., 2018, arXiv:1712.05884. *⟳ verify.*
  The canonical seq2seq-with-attention TTS (encoder → attention → autoregressive mel
  decoder). The architecture my Keras baseline (A1) is modeled on; maps to `ML4.pptx`
  (RNN) + `attention.pptx`.
- **HiFi-GAN** — Kong et al., NeurIPS 2020, arXiv:2010.05646. *⟳ verify.*
  The GAN vocoder that turns mel-spectrograms into waveforms; the concrete link between
  this project and the course GAN lecture (`L18_gan`).
- **VITS** — Kim et al., ICML 2021, arXiv:2106.06103. *⟳ verify.*
  End-to-end TTS as a conditional **VAE** + adversarial training + flows — the exact
  VAE (`L16_autoencoder`) + GAN (`L18_gan`) combination taught in the course, in one
  system. Basis of YourTTS and many cloners.
- **YourTTS** — Casanova et al., ICML 2022, arXiv:2112.02418. *⟳ verify.*
  Zero-shot multi-speaker TTS + voice conversion built on VITS; the direct predecessor
  to XTTS and a good related-work anchor.
- **XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model** — Casanova et al.,
  Interspeech 2024, [arXiv:2406.04904](https://arxiv.org/abs/2406.04904). **✓ verified.**
  The system I fine-tune for A2: a Tortoise-derived GPT-style token model, 16 languages,
  strong voice cloning. Read for the exact architecture I build on.

## B. Self-supervised speech representations (detection frontend)

- **wav2vec 2.0** — Baevski et al., NeurIPS 2020, arXiv:2006.11477. *⟳ verify.*
  The self-supervised speech encoder whose frozen features power the detector; the
  "pretrained representation → small head" idea, applied to audio.
- **XLS-R** — Babu et al., 2021, arXiv:2111.09296. *⟳ verify.*
  Cross-lingual, large-scale wav2vec2; the specific frontend used by SOTA anti-spoofing.
- **WavLM** — Chen et al., 2022, arXiv:2110.13900. *⟳ verify.*
  Alternative SSL frontend (denoising + speaker-aware pretraining); often the strongest
  frontend for spoof detection — the second frontend to try.

## C. Anti-spoofing / deepfake detection (2020+)

- **End-to-End anti-spoofing with RawNet2** — Tak et al., ICASSP 2021, arXiv:2011.01108.
  *⟳ verify.*
  Raw-waveform SincNet → residual CNN → GRU. My pure-Keras comparison baseline (B2);
  official ASVspoof baseline with published numbers. CNN + GRU = both course topics.
- **AASIST** — Jung et al., ICASSP 2022, arXiv:2110.01200. *⟳ verify.*
  Spectro-temporal graph-attention back-end; the light head I adapt on top of the SSL
  frontend. Ties `attention.pptx`.
- **Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and
  data augmentation** — Tak et al., Odyssey 2022,
  [arXiv:2202.12233](https://arxiv.org/abs/2202.12233). **✓ verified.**
  The recipe my main system (B3) reproduces: SSL frontend + AASIST back-end + augmentation.
  The single most important detection reference.
- **RawBoost** — Tak et al., ICASSP 2022, arXiv:2111.04433. *⟳ verify.*
  Data-only augmentation for anti-spoofing; the augmentation I sweep in the
  hyper-parameter search.
- **AASIST3** — 2024 (ASVspoof 5-era),
  [arXiv:2408.17352](https://arxiv.org/abs/2408.17352). **✓ verified.**
  KAN-enhanced AASIST with SSL features; the "very newest" option and a stretch target.
- **Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier** — 2024,
  [OpenReview acJMIXJg2u](https://openreview.net/pdf?id=acJMIXJg2u). **✓ verified.**
  Treats the SSL model as a feature pyramid (SLS); an alternative strong back-end idea.

## D. Datasets & benchmarks

- **ASVspoof 2019** — Todisco et al., Interspeech 2019, arXiv:1904.05441 (+ Wang et al.
  2020 journal). *⟳ verify.*
  The LA subset is my primary detection train/eval set and the source of comparison EERs.
- **Does Audio Deepfake Detection Generalize? (In-the-Wild)** — Müller et al.,
  Interspeech 2022, [arXiv:2203.16263](https://arxiv.org/abs/2203.16263). **✓ verified.**
  Motivates the whole cross-test: models that look great on ASVspoof collapse on real
  in-the-wild fakes. My optional generalization set.
- **ASVspoof 5** — Wang, Yamagishi et al., 2024,
  [arXiv:2408.08739](https://arxiv.org/abs/2408.08739). **✓ verified.**
  Newest, hardest, crowdsourced + adversarial benchmark; shows old baselines (RawNet2,
  AASIST) collapsing (EER >29%) and SSL ensembles winning — context for my choices.

## E. Metrics & evaluation

- **t-DCF / min t-DCF** — Kinnunen et al., 2020 (fundamentals), arXiv:2007.05979
  (and the 2018 t-DCF paper). *⟳ verify.*
  The tandem detection cost function reported alongside EER in ASVspoof.
- **ECAPA-TDNN** — Desplanques et al., Interspeech 2020, arXiv:2005.07143. *⟳ verify.*
  Speaker-embedding network used for the **speaker-cosine-similarity** creation metric.
- **Mel-Cepstral Distortion (MCD)** — Kubichek, 1993 (classic).
  The objective spectral-distance metric for the **spectrogram-similarity** creation
  metric; pair with log-mel L2/SSIM.
- *(Outside course materials, optional)* **UTMOS / NISQA** — automatic MOS predictors;
  a cheap proxy to sanity-check the human MOS panel. Flag clearly as non-course, and as
  a proxy for (not a replacement of) human MOS.

---

*Verification note:* the **✓ verified** links were confirmed 2026-07-01; the **⟳ verify**
arXiv IDs are recalled and should be re-checked (a wrong digit is easy) before they go in
the submitted report. Prefer the official challenge / ISCA archive pages where they exist.
