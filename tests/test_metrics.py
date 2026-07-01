"""Unit tests for the EER/DET math — pure numpy, no TensorFlow needed.

Run standalone:   uv run --with numpy python tests/test_metrics.py
Or with pytest:   uv run --with 'numpy,pytest' pytest tests/test_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detection.evaluate import compute_eer, eer_from_scores  # noqa: E402


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
