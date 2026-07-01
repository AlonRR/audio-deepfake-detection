"""Speaker-embedding cosine similarity (required creation metric #3).

Uses SpeechBrain's ECAPA-TDNN (arXiv:2005.07143), the same "embedding" idea as
word embedding.pptx applied to voice. Runs in the PyTorch container. The cosine helper is
numpy-only and unit-tested; embedding needs torch + speechbrain.
"""
from __future__ import annotations

import numpy as np

from src.common.audio import TARGET_SR

_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
_encoder = None  # lazy singleton


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embeddings."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def _get_encoder():
    global _encoder
    if _encoder is None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        _encoder = EncoderClassifier.from_hparams(
            source=_MODEL, run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
    return _encoder


def embed(wav_path: str) -> np.ndarray:
    """ECAPA-TDNN speaker embedding (192-d) for a wav file."""
    import torchaudio

    enc = _get_encoder()
    sig, sr = torchaudio.load(wav_path)
    if sr != TARGET_SR:
        sig = torchaudio.functional.resample(sig, sr, TARGET_SR)
    emb = enc.encode_batch(sig).squeeze().detach().cpu().numpy()
    return emb


def score(real_paths, cloned_paths) -> dict:
    """Mean cosine(cloned_i, real_centroid) and a real-vs-real baseline for context."""
    real = [embed(p) for p in real_paths]
    centroid = np.mean(real, axis=0)
    cloned_cos = [cosine(embed(p), centroid) for p in cloned_paths]
    real_cos = [cosine(e, centroid) for e in real]  # upper-bound reference
    return {"cloned_cosine_mean": float(np.mean(cloned_cos)),
            "cloned_cosine_std": float(np.std(cloned_cos)),
            "real_vs_centroid_mean": float(np.mean(real_cos)),
            "n_cloned": len(cloned_cos)}
