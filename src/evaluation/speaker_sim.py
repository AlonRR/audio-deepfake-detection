"""Speaker-embedding cosine similarity (required creation metric #3).

Course anchor: embeddings (word embedding.pptx) — same idea, applied to voice.
Reference: ECAPA-TDNN, arXiv:2005.07143 (SpeechBrain). Runs in the PyTorch container.

TODO (implementation pass):
    - embed(wav, sr) -> np.ndarray              # ECAPA-TDNN (SpeechBrain) or Resemblyzer
    - cosine(a, b) -> float
    - score(real_wavs, cloned_wavs) -> summary  # mean/std cosine, real-vs-real baseline
"""


def cosine(a, b):
    """Cosine similarity between two speaker embeddings. TODO: implement."""
    raise NotImplementedError
