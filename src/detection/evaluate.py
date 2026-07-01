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


# ASVspoof 2019 legacy cost model (Kinnunen et al. 2018).
DEFAULT_COST = {"Pspoof": 0.05, "Ptar": 0.95 * 0.99, "Pnon": 0.95 * 0.01,
                "Cmiss_asv": 1.0, "Cfa_asv": 10.0, "Cmiss_cm": 1.0, "Cfa_cm": 10.0}


def load_asv_scores(path: str):
    """Parse an ASVspoof ASV score file -> (target, nontarget, spoof) score arrays.

    Each line ends with `<key> <score>` where key is target|nontarget|spoof.
    """
    tar, non, spoof = [], [], []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 2:
                continue
            key, score = p[-2], float(p[-1])
            (tar if key == "target" else non if key == "nontarget" else spoof).append(score)
    return np.asarray(tar), np.asarray(non), np.asarray(spoof)


def _asv_error_rates(tar, non, spoof, thr):
    return (float(np.mean(non >= thr)),      # Pfa_asv
            float(np.mean(tar < thr)),       # Pmiss_asv
            float(np.mean(spoof < thr)))     # Pmiss_spoof_asv


def compute_tdcf(bonafide_cm, spoof_cm, pfa_asv, pmiss_asv, pmiss_spoof_asv, cost=DEFAULT_COST):
    """Normalized t-DCF curve minimum (official ASVspoof2019 formulation)."""
    pmiss_cm, pfa_cm, _ = compute_det(np.asarray(bonafide_cm), np.asarray(spoof_cm))
    c1 = cost["Ptar"] * (cost["Cmiss_cm"] - cost["Cmiss_asv"] * pmiss_asv) \
        - cost["Pnon"] * cost["Cfa_asv"] * pfa_asv
    c2 = cost["Cfa_cm"] * cost["Pspoof"] * (1 - pmiss_spoof_asv)
    if c1 < 0 or c2 < 0:
        raise ValueError("degenerate t-DCF cost (check the cost model / ASV rates)")
    tdcf_norm = (c1 * pmiss_cm + c2 * pfa_cm) / min(c1, c2)
    return float(np.min(tdcf_norm))


def min_tdcf(bonafide_cm, spoof_cm, asv_score_file: str, cost=DEFAULT_COST) -> float:
    """min t-DCF from CM scores + the challenge's ASV score file (at the ASV EER threshold)."""
    tar, non, spoof = load_asv_scores(asv_score_file)
    _, asv_thr = compute_eer(tar, non)
    pfa, pmiss, pmiss_spoof = _asv_error_rates(tar, non, spoof, asv_thr)
    return compute_tdcf(bonafide_cm, spoof_cm, pfa, pmiss, pmiss_spoof, cost)
