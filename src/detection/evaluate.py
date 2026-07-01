"""Detection evaluation — EER / DET curve (+ min t-DCF hook) and the cross-generator test.

The EER/DET math is pure numpy (unit-tested in tests/test_metrics.py), so it imports and
runs without TensorFlow. The canonical ASVspoof EER algorithm is used (cumulative
target/non-target sums over sorted scores), matching the challenge's official scoring.
"""
from __future__ import annotations

import numpy as np


def compute_det(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """Return (frr, far, thresholds) sweeping every score as a threshold.

    Convention: higher score = more "bona fide". `target` = bona fide, `nontarget` = spoof.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    n_scores = target_scores.size + nontarget_scores.size

    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    idx = np.argsort(all_scores, kind="mergesort")
    labels = labels[idx]

    tar_sums = np.cumsum(labels)
    nontar_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_sums)

    frr = np.concatenate((np.atleast_1d(0.0), tar_sums / max(target_scores.size, 1)))
    far = np.concatenate((np.atleast_1d(1.0), nontar_sums / max(nontarget_scores.size, 1)))
    thr = np.concatenate((np.atleast_1d(all_scores[idx[0]] - 1e-3), all_scores[idx]))
    return frr, far, thr


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """Equal Error Rate (fraction, 0..1) and the threshold at which it occurs."""
    frr, far, thr = compute_det(target_scores, nontarget_scores)
    i = np.nanargmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2.0), float(thr[i])


def eer_from_scores(scores: np.ndarray, labels: np.ndarray):
    """EER from a flat score vector + binary labels (1 = bona fide, 0 = spoof)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    return compute_eer(scores[labels == 1], scores[labels == 0])


def save_det_curve(target_scores, nontarget_scores, out_path: str, title: str = "DET") -> float:
    """Plot the DET curve to ``out_path``; return the EER. TF-free (matplotlib only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frr, far, _ = compute_det(target_scores, nontarget_scores)
    eer, _ = compute_eer(target_scores, nontarget_scores)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(far * 100, frr * 100, lw=2)
    ax.plot([0, 100], [0, 100], "k--", lw=0.7)
    ax.scatter([eer * 100], [eer * 100], c="r", zorder=5, label=f"EER = {eer * 100:.2f}%")
    ax.set(xlabel="False Alarm rate (%)", ylabel="Miss rate (%)", title=title,
           xlim=(0, 40), ylim=(0, 40))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return eer


def min_tdcf(*args, **kwargs):  # noqa: D401
    """min t-DCF (needs paired ASV scores). TODO: port the ASVspoof2019 tDCF module."""
    raise NotImplementedError("min t-DCF requires the ASV score files; add in the eval pass.")
