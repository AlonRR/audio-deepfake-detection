# Literature — Audio Deep Fakes: Creation & Recognition

Curated reading list for the project. Each entry has a one-line **why read it**.
**All entries verified** — every arXiv ID below was fetched and its title/authors
confirmed on 2026-07-18. None are recalled from memory.

---

## A. Voice cloning / TTS (creation)

- **Tacotron 2** — *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram
  Predictions* — Shen et al., ICASSP 2018, [arXiv:1712.05884](https://arxiv.org/abs/1712.05884). **✓ verified.**
  The canonical seq2seq-with-attention TTS (encoder → attention → autoregressive mel
  decoder). The architecture my Keras baseline (A1) is modeled on; maps to `ML4.pptx`
  (RNN) + `attention.pptx`.
- **HiFi-GAN** — *Generative Adversarial Networks for Efficient and High Fidelity Speech
  Synthesis* — Kong, Kim & Bae, NeurIPS 2020, [arXiv:2010.05646](https://arxiv.org/abs/2010.05646). **✓ verified.**
  The GAN vocoder that turns mel-spectrograms into waveforms; the concrete link between
  this project and the course GAN lecture (`L18_gan`).
- **VITS** — *Conditional Variational Autoencoder with Adversarial Learning for End-to-End
  Text-to-Speech* — Kim, Kong & Son, ICML 2021, [arXiv:2106.06103](https://arxiv.org/abs/2106.06103). **✓ verified.**
  End-to-end TTS as a conditional **VAE** + adversarial training + flows — the exact
  VAE (`L16_autoencoder`) + GAN (`L18_gan`) combination taught in the course, in one
  system. Basis of YourTTS and many cloners.
- **YourTTS** — *Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for
  everyone* — Casanova et al., ICML 2022, [arXiv:2112.02418](https://arxiv.org/abs/2112.02418). **✓ verified.**
  Zero-shot multi-speaker TTS + voice conversion built on VITS; the direct predecessor
  to XTTS and a good related-work anchor.
- **XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model** — Casanova et al.,
  Interspeech 2024, [arXiv:2406.04904](https://arxiv.org/abs/2406.04904). **✓ verified.**
  The system I fine-tune for A2: a Tortoise-derived GPT-style token model, 16 languages,
  strong voice cloning. Read for the exact architecture I build on.

## B. Self-supervised speech representations (detection frontend)

- **wav2vec 2.0** — *A Framework for Self-Supervised Learning of Speech Representations* —
  Baevski et al., NeurIPS 2020, [arXiv:2006.11477](https://arxiv.org/abs/2006.11477). **✓ verified.**
  The self-supervised speech encoder whose frozen features power the detector; the
  "pretrained representation → small head" idea, applied to audio.
- **XLS-R** — *Self-supervised Cross-lingual Speech Representation Learning at Scale* —
  Babu et al., 2021, [arXiv:2111.09296](https://arxiv.org/abs/2111.09296). **✓ verified.**
  Cross-lingual, large-scale wav2vec2; the specific frontend used by SOTA anti-spoofing.
- **WavLM** — *Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing* —
  Chen et al., 2021/22, [arXiv:2110.13900](https://arxiv.org/abs/2110.13900). **✓ verified.**
  Alternative SSL frontend (denoising + speaker-aware pretraining); often the strongest
  frontend for spoof detection — the second frontend to try.

## C. Anti-spoofing / deepfake detection (2020+)

- **End-to-end anti-spoofing with RawNet2** — Tak, Patino, Todisco, Nautsch, Evans &
  Larcher, ICASSP 2021, [arXiv:2011.01108](https://arxiv.org/abs/2011.01108). **✓ verified.**
  Raw-waveform SincNet → residual CNN → GRU. My pure-Keras comparison baseline (B2);
  official ASVspoof baseline with published numbers. CNN + GRU = both course topics.
- **AASIST** — *Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention
  Networks* — Jung et al., ICASSP 2022, [arXiv:2110.01200](https://arxiv.org/abs/2110.01200). **✓ verified.**
  Spectro-temporal graph-attention back-end; the light head I adapt on top of the SSL
  frontend. Ties `attention.pptx`.
- **Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and
  data augmentation** — Tak et al., Odyssey 2022,
  [arXiv:2202.12233](https://arxiv.org/abs/2202.12233). **✓ verified.**
  The recipe my main system (B3) reproduces: SSL frontend + AASIST back-end + augmentation.
  The single most important detection reference.
- **RawBoost** — *A Raw Data Boosting and Augmentation Method applied to ASV Anti-Spoofing*
  — Tak, Kamble, Patino, Todisco & Evans, ICASSP 2022,
  [arXiv:2111.04433](https://arxiv.org/abs/2111.04433). **✓ verified.**
  Data-only augmentation for anti-spoofing; the augmentation I sweep in the
  hyper-parameter search.
- **AASIST3** — 2024 (ASVspoof 5-era),
  [arXiv:2408.17352](https://arxiv.org/abs/2408.17352). **✓ verified.**
  KAN-enhanced AASIST with SSL features; the "very newest" option and a stretch target.
- **Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier** — 2024,
  [OpenReview acJMIXJg2u](https://openreview.net/pdf?id=acJMIXJg2u). **✓ verified.**
  Treats the SSL model as a feature pyramid (SLS); an alternative strong back-end idea.

## D. Datasets & benchmarks

- **ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection** — Todisco et al.,
  Interspeech 2019, [arXiv:1904.05441](https://arxiv.org/abs/1904.05441). **✓ verified.**
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

- **t-DCF / min t-DCF** — *Tandem Assessment of Spoofing Countermeasures and Automatic
  Speaker Verification: Fundamentals* — Kinnunen et al., 2020,
  [arXiv:2007.05979](https://arxiv.org/abs/2007.05979). **✓ verified.**
  The tandem detection cost function reported alongside EER in ASVspoof.
- **ECAPA-TDNN** — *Emphasized Channel Attention, Propagation and Aggregation in TDNN Based
  Speaker Verification* — Desplanques, Thienpondt & Demuynck, Interspeech 2020,
  [arXiv:2005.07143](https://arxiv.org/abs/2005.07143). **✓ verified.**
  Speaker-embedding network used for the **speaker-cosine-similarity** creation metric.
- **Mel-Cepstral Distortion (MCD)** — Kubichek, 1993 (classic).
  The objective spectral-distance metric for the **spectrogram-similarity** creation
  metric; pair with log-mel L2/SSIM.
- *(Outside course materials, optional)* **UTMOS / NISQA** — automatic MOS predictors;
  a cheap proxy to sanity-check the human MOS panel. Flag clearly as non-course, and as
  a proxy for (not a replacement of) human MOS.

---

*Verification note:* every arXiv ID in this file was fetched and checked against the
paper's actual title and author list on **2026-07-18**; all 13 previously-unverified IDs
were correct. Where a venue year differs from the arXiv year (e.g. Tacotron 2 posted 2017,
ICASSP 2018), the venue year is given. Prefer the official challenge / ISCA archive pages
when citing formally.
