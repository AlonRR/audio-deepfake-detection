"""Audio I/O and feature helpers shared across creation, evaluation, and detection.

Pure numpy / librosa / soundfile — no TensorFlow or torch here, so this module imports
cheaply in either container. Feature parameters live in one place (``FeatureConfig``) so
the detector, the creation baseline, and the evaluation metrics all stay consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_SR = 16_000  # ASVspoof + most SSL frontends operate at 16 kHz


@dataclass(frozen=True)
class FeatureConfig:
    """Front-end feature settings (shared by detection baselines + creation)."""

    sr: int = TARGET_SR
    n_fft: int = 512
    hop_length: int = 160          # 10 ms at 16 kHz
    win_length: int = 400          # 25 ms
    n_mels: int = 80
    n_lfcc: int = 60
    fmin: float = 0.0
    fmax: float | None = None      # None -> sr / 2
    max_frames: int = 400          # fixed-length crop/pad (~4 s) for batching


def load_wav(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sr``.

    Uses soundfile for robust flac/wav reading, then librosa for resampling.
    """
    import librosa
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:                       # stereo -> mono
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def save_wav(path: str, wav: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write a mono float32 waveform to ``path`` (16-bit PCM)."""
    import soundfile as sf

    sf.write(path, np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")


def log_mel(wav: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, T). Powers -> dB, per-utterance normalized."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=wav, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.win_length, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
    )
    log = librosa.power_to_db(mel, ref=np.max)
    return _cmvn(log)


def lfcc(wav: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Linear-frequency cepstral coefficients (LFCC), shape (n_lfcc, T).

    LFCC is the standard ASVspoof front-end. Approximated here as a DCT over a
    linear-frequency (not mel) log power spectrum. Per-utterance mean/var normalized.
    """
    import librosa
    from scipy.fftpack import dct

    spec = np.abs(
        librosa.stft(wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length)
    ) ** 2
    # NOTE: this is an HTK-**MEL** filterbank, not a linear one. An earlier comment here
    # claimed the mel spacing was replaced with linear spacing; it never was, so this
    # front-end computes 60 MEL-cepstral coefficients (MFCC-60), NOT LFCC. Measured filter
    # peaks are [31, 312, 719, 1281, 2031, 3094, 4594, 6688] Hz - mel-spaced.
    # The published numbers under the "lfcc" label were produced by THIS code, so they are
    # real; only the name is wrong. Changing the filterbank now would invalidate them, so
    # the label is corrected in the docs instead. See docs/results.md B.3.
    fb = librosa.filters.mel(sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_lfcc,
                             fmin=cfg.fmin, fmax=cfg.fmax, htk=True, norm=None)
    feat = np.log(np.maximum(fb @ spec, 1e-10))
    cep = dct(feat, type=2, axis=0, norm="ortho")[: cfg.n_lfcc]
    return _cmvn(cep)


def fix_length(feat: np.ndarray, max_frames: int) -> np.ndarray:
    """Crop or wrap-pad a (F, T) feature to exactly ``max_frames`` columns."""
    t = feat.shape[1]
    if t >= max_frames:
        return feat[:, :max_frames]
    reps = int(np.ceil(max_frames / t))
    return np.tile(feat, (1, reps))[:, :max_frames]


def _cmvn(feat: np.ndarray) -> np.ndarray:
    """Per-utterance cepstral mean/variance normalization along time."""
    mu = feat.mean(axis=1, keepdims=True)
    sd = feat.std(axis=1, keepdims=True) + 1e-8
    return ((feat - mu) / sd).astype(np.float32)
