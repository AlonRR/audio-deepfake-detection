"""Spectrogram-similarity metric (one of the metrics the brief suggests).

MCD (mel-cepstral distortion) is the headline number; log-mel L2 and a light SSIM are
supporting evidence. Pure numpy + librosa (no TensorFlow/torch), so it runs anywhere and
the core math is unit-tested (identical signals -> MCD 0, SSIM 1).
"""
from __future__ import annotations

import numpy as np

from src.common.audio import TARGET_SR, load_wav

_MCD_K = 10.0 / np.log(10.0) * np.sqrt(2.0)  # standard MCD constant


def _mfcc(wav: np.ndarray, sr: int, n_mfcc: int = 13, n_mels: int = 40,
          fmin: float = 80.0, fmax: float = 7600.0, top_db: float = 35.0) -> np.ndarray:
    """Mel-cepstral coefficients for MCD, with two corrections over librosa's default.

    1. **Natural log, not dB.** `librosa.feature.mfcc` runs `power_to_db` (10*log10)
       first, but the MCD constant 10/ln(10)*sqrt(2) already assumes natural-log
       cepstra — using librosa's output applies the scaling twice.
    2. **Silence frames are dropped.** log of a near-zero mel bin is hugely negative
       and swings wildly with any noise, so silent frames dominated the average.

    Returns (n_mfcc-1, T_voiced); the 0th (energy) coefficient is dropped per MCD
    convention.
    """
    import librosa
    import scipy.fftpack

    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256,
                                         n_mels=n_mels, fmin=fmin, fmax=fmax)
    voiced = librosa.power_to_db(mel, ref=np.max).max(axis=0) > -top_db
    logmel = np.log(np.maximum(mel, 1e-8))              # natural log
    c = scipy.fftpack.dct(logmel, axis=0, type=2, norm="ortho")[1:n_mfcc]
    return c[:, voiced] if voiced.sum() > 5 else c


def mcd(ref_wav: np.ndarray, syn_wav: np.ndarray, sr: int = TARGET_SR) -> float:
    """DTW-aligned mel-cepstral distortion between reference and synthesized audio.

    IMPORTANT — the absolute scale is **not** comparable to published MCD figures
    (which are typically 4-8 dB for good TTS). Those use MGC-based mel-cepstral
    analysis (pysptk/SPTK, alpha warping); this is a DCT of the log-mel spectrum,
    which lands roughly an order of magnitude higher. Use it to RANK systems on
    parallel utterances, not as an absolute quality figure.

    Sanity-checked: identical signals -> 0.00; same-text synthesis scores lower than
    a different-text real recording of the same speaker (i.e. it tracks content, as
    MCD should).
    """
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
