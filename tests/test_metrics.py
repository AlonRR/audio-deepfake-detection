"""Unit tests for the EER/DET math — pure numpy, no TensorFlow needed.

Run standalone:   uv run --with numpy python tests/test_metrics.py
Or with pytest:   uv run --with 'numpy,pytest' pytest tests/test_metrics.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.evaluate import compute_eer, eer_from_scores  # noqa: E402
from src.evaluation.speaker_sim import cosine  # noqa: E402


def test_tdcf_perfect_cm_is_zero():
    from src.detection.evaluate import compute_tdcf
    bona = np.linspace(2.0, 3.0, 200)
    spoof = np.linspace(-3.0, -2.0, 200)
    v = compute_tdcf(bona, spoof, pfa_asv=0.05, pmiss_asv=0.01, pmiss_spoof_asv=0.01)
    assert 0.0 <= v < 1e-6, v


def test_cosine_identity_and_opposite():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(cosine(v, v) - 1.0) < 1e-9
    assert abs(cosine(v, -v) + 1.0) < 1e-9
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9


def test_perfect_separation_is_zero_eer():
    target = np.linspace(2.0, 3.0, 100)     # bona fide: high scores
    nontarget = np.linspace(-3.0, -2.0, 100)  # spoof: low scores
    eer, _ = compute_eer(target, nontarget)
    assert eer < 1e-6, eer


def test_identical_distributions_is_half_eer():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)
    eer, _ = compute_eer(x, y)
    assert abs(eer - 0.5) < 0.03, eer


def test_known_small_case():
    # 2 bona fide (scores 0.9, 0.8), 2 spoof (0.2, 0.1) -> separable -> EER 0
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    eer, _ = eer_from_scores(scores, labels)
    assert eer < 1e-6, eer


def test_eer_in_unit_range():
    rng = np.random.default_rng(1)
    target = rng.normal(1.0, 1.0, 1000)
    nontarget = rng.normal(-0.5, 1.0, 1000)
    eer, thr = compute_eer(target, nontarget)
    assert 0.0 <= eer <= 1.0
    assert np.isfinite(thr)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} metric tests passed.")


# --- spectrogram-similarity metrics -----------------------------------------
# The module docstring claimed these were unit-tested when they were not; these
# tests make the claim true and pin the two properties the report relies on.

def test_mcd_identical_signals_is_zero():
    import numpy as np
    from src.evaluation.spectrogram_sim import mcd
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype("float32") * 0.1
    assert mcd(x, x) == 0.0


def test_mcd_tracks_content_not_just_difference():
    """A same-text pair must score below a different-text pair from the same speaker.

    This is the property that justifies using MCD to RANK systems on parallel
    utterances (see docs/results.md A.2).
    """
    import os
    from src.common.audio import TARGET_SR, load_wav
    from src.evaluation.spectrogram_sim import mcd
    real, same, other = ("data/raw/heldout/clip_0004.wav",
                         "data/generated/xtts_ft_parallel/clip_0004.wav",
                         "data/raw/heldout/clip_0013.wav")
    if not all(os.path.isfile(p) for p in (real, same, other)):
        pytest.skip("generated clips not present (they are gitignored)")
    r = load_wav(real, TARGET_SR)
    assert mcd(r, load_wav(same, TARGET_SR)) < mcd(r, load_wav(other, TARGET_SR))


def test_logmel_corr_identical_is_one():
    import numpy as np
    from src.evaluation.spectrogram_sim import logmel_corr
    rng = np.random.default_rng(1)
    x = rng.standard_normal(16000).astype("float32") * 0.1
    assert logmel_corr(x, x) == pytest.approx(1.0, abs=1e-6)


def test_logmel_corr_is_pearson_not_ssim():
    """Guards the rename: the implementation IS the correlation coefficient.

    Z-normalising collapses the SSIM luminance term to 1 and reduces the rest to
    Pearson r. Documented in spectrogram_sim.logmel_corr.
    """
    import numpy as np
    import librosa
    from src.common.audio import TARGET_SR
    from src.evaluation.spectrogram_sim import logmel_corr
    rng = np.random.default_rng(2)
    a = rng.standard_normal(16000).astype("float32")
    b = rng.standard_normal(16000).astype("float32")

    def lm(w):
        v = librosa.power_to_db(librosa.feature.melspectrogram(
            y=w, sr=TARGET_SR, n_fft=512, hop_length=160, n_mels=80))
        return (v - v.mean()) / (v.std() + 1e-8)

    A, B = lm(a), lm(b)
    t = min(A.shape[1], B.shape[1])
    r = float(np.corrcoef(A[:, :t].ravel(), B[:, :t].ravel())[0, 1])
    assert logmel_corr(a, b) == pytest.approx(r, abs=1e-3)
