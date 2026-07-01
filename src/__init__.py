"""Audio deep fakes — creation & recognition (Shenkar Neural Networks final project).

Packages
--------
creation    Voice cloning: a from-scratch Keras TTS baseline + XTTS-v2 fine-tuning.
evaluation  The three required creation metrics (MOS, spectrogram sim, speaker cosine).
detection   Anti-spoofing: SSL frontend + Keras back-end, plus baselines.
common      Shared audio I/O / feature / config helpers.

Nothing is implemented yet — this pass is the proposal + scaffold. See docs/proposal.md.
"""
