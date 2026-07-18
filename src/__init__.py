"""Audio deep fakes — creation & recognition (Shenkar Neural Networks final project).

Packages
--------
creation    Voice cloning: a from-scratch Keras TTS baseline + XTTS-v2 fine-tuning.
evaluation  Creation-quality metrics (MOS, spectrogram sim, speaker cosine).
detection   Anti-spoofing: SSL frontend + Keras back-end, plus baselines.
common      Shared audio I/O / feature / config helpers.

Both halves are implemented and have been run end-to-end on the lab GPU.
Results live in docs/results.md; wall-times and the failure log in docs/runtimes.md.
"""
