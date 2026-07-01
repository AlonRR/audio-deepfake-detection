"""TTS audio front-end for the Keras baseline — invertible log-mel + Griffin-Lim.

Unlike the detection features (dB + CMVN, not invertible), TTS needs a mel we can turn
back into a waveform. We store natural-log magnitude mel and invert with librosa's
Griffin-Lim. Separate sample rate/params from detection on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TTSAudio:
    sr: int = 22_050
    n_fft: int = 1024
    hop: int = 256
    win: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0
    ref_level_db: float = 20.0
    min_level_db: float = -100.0


CFG = TTSAudio()


def wav_to_mel(wav: np.ndarray, cfg: TTSAudio = CFG) -> np.ndarray:
    """Waveform -> normalized log-mel, shape (T, n_mels) in ~[0, 1]."""
    import librosa

    S = librosa.feature.melspectrogram(
        y=wav, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=1.0)
    db = 20.0 * np.log10(np.maximum(1e-5, S)) - cfg.ref_level_db
    norm = np.clip((db - cfg.min_level_db) / -cfg.min_level_db, 0.0, 1.0)
    return norm.T.astype(np.float32)


def mel_to_wav(mel_norm: np.ndarray, cfg: TTSAudio = CFG, n_iter: int = 60) -> np.ndarray:
    """Normalized log-mel (T, n_mels) -> waveform via Griffin-Lim."""
    import librosa

    db = mel_norm.T * -cfg.min_level_db + cfg.min_level_db + cfg.ref_level_db
    S = np.power(10.0, db / 20.0)
    return librosa.feature.inverse.mel_to_audio(
        S, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop, win_length=cfg.win,
        fmin=cfg.fmin, fmax=cfg.fmax, power=1.0, n_iter=n_iter).astype(np.float32)
