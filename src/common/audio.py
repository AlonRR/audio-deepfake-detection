"""Audio I/O and feature helpers shared across creation, evaluation, and detection.

TODO (implementation pass):
    - load_wav(path, target_sr) -> (np.ndarray, sr)   # soundfile + librosa.resample
    - save_wav(path, wav, sr)
    - trim_silence(wav, sr)
    - log_mel(wav, sr, n_mels, ...) -> np.ndarray      # librosa.feature.melspectrogram
    - lfcc(wav, sr, ...) -> np.ndarray                  # LFCC features for baselines
Keep feature params in one place so creation/detection/eval stay consistent.
"""

TARGET_SR = 16_000  # ASVspoof + most SSL frontends operate at 16 kHz


def load_wav(path, target_sr: int = TARGET_SR):
    """Load an audio file as mono float32 at ``target_sr``. TODO: implement."""
    raise NotImplementedError
