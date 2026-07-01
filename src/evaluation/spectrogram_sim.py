"""Spectrogram-similarity metric (required creation metric #2).

TODO (implementation pass):
    - mcd(ref_wav, syn_wav, sr) -> float        # mel-cepstral distortion (dtw-aligned)
    - logmel_l2(ref, syn) -> float
    - logmel_ssim(ref, syn) -> float
Report MCD as the headline number; L2/SSIM as supporting evidence. Uses librosa.
"""


def mcd(ref_wav, syn_wav, sr):
    """Mel-cepstral distortion between a reference and a synthesized clip. TODO."""
    raise NotImplementedError
