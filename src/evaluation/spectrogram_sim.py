"""Spectrogram-similarity metric (required creation metric #2).

MCD (mel-cepstral distortion) is the headline number; log-mel L2 and a light SSIM are
supporting evidence. Pure numpy + librosa (no TensorFlow/torch), so it runs anywhere and
the core math is unit-tested (identical signals -> MCD 0, SSIM 1).
"""
from __future__ import annotations

import numpy as np

from src.common.audio import TARGET_SR, load_wav

_MCD_K = 10.0 / np.log(10.0) * np.sqrt(2.0)  # standard MCD constant


def _mfcc(wav: np.ndarray, sr: int, n_mfcc: int = 25) -> np.ndarray:
    import librosa

    m = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
    return m[1:]  # drop the 0th (energy) coefficient, MCD convention -> (n_mfcc-1, T)


def mcd(ref_wav: np.ndarray, syn_wav: np.ndarray, sr: int = TARGET_SR) -> float:
    """DTW-aligned mel-cepstral distortion (dB) between reference and synthesized audio."""
    import librosa

    ref, syn = _mfcc(ref_wav, sr), _mfcc(syn_wav, sr)
    _, wp = librosa.sequence.dtw(X=ref, Y=syn, metric="euclidean")
    diff = ref[:, wp[:, 0]] - syn[:, wp[:, 1]]
    return float(_MCD_K * np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


def logmel_l2(ref_wav: np.ndarray, syn_wav: np.ndarray, sr: int = TARGET_SR) -> float:
    """Frame-wise L2 between log-mel spectrograms (length-matched by cropping)."""
    import librosa

    def lm(w):
        return librosa.power_to_db(
            librosa.feature.melspectrogram(y=w, sr=sr, n_fft=512, hop_length=160, n_mels=80)
        )

    a, b = lm(ref_wav), lm(syn_wav)
    t = min(a.shape[1], b.shape[1])
    return float(np.sqrt(np.mean((a[:, :t] - b[:, :t]) ** 2)))


def logmel_ssim(ref_wav: np.ndarray, syn_wav: np.ndarray, sr: int = TARGET_SR) -> float:
    """Global SSIM between the two log-mel "images" (crop to common length)."""
    import librosa

    def lm(w):
        x = librosa.power_to_db(
            librosa.feature.melspectrogram(y=w, sr=sr, n_fft=512, hop_length=160, n_mels=80)
        )
        return (x - x.mean()) / (x.std() + 1e-8)

    a, b = lm(ref_wav), lm(syn_wav)
    t = min(a.shape[1], b.shape[1])
    a, b = a[:, :t], b[:, :t]
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def score_pairs(pairs, sr: int = TARGET_SR) -> dict:
    """Aggregate metrics over (ref_path, syn_path) pairs -> mean MCD / L2 / SSIM."""
    mcds, l2s, ssims = [], [], []
    for ref_path, syn_path in pairs:
        r, s = load_wav(ref_path, sr), load_wav(syn_path, sr)
        mcds.append(mcd(r, s, sr)); l2s.append(logmel_l2(r, s, sr)); ssims.append(logmel_ssim(r, s, sr))
    return {"mcd_mean": float(np.mean(mcds)), "logmel_l2_mean": float(np.mean(l2s)),
            "ssim_mean": float(np.mean(ssims)), "n": len(mcds)}
