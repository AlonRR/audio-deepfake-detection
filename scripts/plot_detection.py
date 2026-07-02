"""Regenerate all detection report figures from the committed score files.

Run from the repo root:
    uv run --no-project --with numpy --with matplotlib --with scipy python scripts/plot_detection.py

Reads reports/detection/<run>/{dev,eval}_scores.npz (+ result.json for EER labels),
writes figures to reports/figures/. Everything needed to rebuild the graphs is tracked.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = os.path.join(ROOT, "reports", "detection")
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)


def load(run, which):
    d = np.load(os.path.join(DET, run, f"{which}_scores.npz"))
    s, l = d["scores"], d["labels"]
    return s[l == 1], s[l == 0]


def eer_label(run):
    with open(os.path.join(DET, run, "result.json")) as fh:
        r = json.load(fh)
    return r.get("eer_pct", r.get("dev_eer_pct"))


def det_points(bona, spoof, n=600):
    lo, hi = min(bona.min(), spoof.min()), max(bona.max(), spoof.max())
    thr = np.linspace(lo, hi, n)
    sb, ss = np.sort(bona), np.sort(spoof)
    pmiss = np.searchsorted(sb, thr, "left") / len(bona)
    pfa = 1.0 - np.searchsorted(ss, thr, "left") / len(spoof)
    eps = 1e-5
    return np.clip(pfa, eps, 1 - eps), np.clip(pmiss, eps, 1 - eps)


TICKS = [0.1, 0.5, 1, 2, 5, 10, 20, 40]


def setup_det(ax):
    t = norm.ppf(np.array(TICKS) / 100)
    ax.set_xticks(t); ax.set_xticklabels([str(x) for x in TICKS])
    ax.set_yticks(t); ax.set_yticklabels([str(x) for x in TICKS])
    lim = norm.ppf(np.array([0.05, 45]) / 100)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False-alarm rate (%)"); ax.set_ylabel("Miss rate (%)")
    ax.grid(True, ls=":", alpha=0.5)


def det_overlay(specs, title, fname):
    fig, ax = plt.subplots(figsize=(6, 6))
    for run, which, c, name in specs:
        b, s = load(run, which)
        pfa, pmiss = det_points(b, s)
        ax.plot(norm.ppf(pfa), norm.ppf(pmiss), color=c, lw=2.2,
                label=f"{name}  (EER {eer_label(run):.2f}%)")
    setup_det(ax); ax.set_title(title); ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)
    print("wrote", fname)


# DET overlays (skip runs whose scores aren't present yet, e.g. rawnet2 mid-run)
def present(run, which):
    return os.path.exists(os.path.join(DET, run, f"{which}_scores.npz"))

eval_specs = [(r, "eval", c, n) for r, c, n in [
    ("base_cnn_lfcc_eval", "tab:red", "CNN-LFCC"),
    ("base_rawnet2_eval", "tab:purple", "RawNet2"),
    ("ssl_xlsr_eval", "tab:green", "SSL XLS-R")] if present(r, "eval")]
det_overlay(eval_specs, "DET - ASVspoof2019 LA eval (unseen attacks A07-A19)", "det_eval_overlay.png")

dev_specs = [(r, "dev", c, n) for r, c, n in [
    ("base_cnn_logmel_full", "tab:orange", "CNN log-mel"),
    ("base_cnn_lfcc_full", "tab:red", "CNN-LFCC"),
    ("base_rawnet2_full", "tab:purple", "RawNet2"),
    ("ssl_xlsr_run1", "tab:green", "SSL XLS-R")] if present(r, "dev")]
det_overlay(dev_specs, "DET - ASVspoof2019 LA dev", "det_dev_overlay.png")

# --- EER + min t-DCF comparison bars (data-driven from result.json, log scale) ---
def metrics(run):
    p = os.path.join(DET, run, "result.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return r.get("eer_pct", r.get("dev_eer_pct")), r.get("min_tdcf")


MODELS = [("CNN\nlog-mel", "base_cnn_logmel_full", None),
          ("CNN\nLFCC", "base_cnn_lfcc_full", "base_cnn_lfcc_eval"),
          ("RawNet2", "base_rawnet2_full", "base_rawnet2_eval"),
          ("SSL\nXLS-R", "ssl_xlsr_run1", "ssl_xlsr_eval")]
rows = [(n, metrics(dv), metrics(ev) if ev else None) for n, dv, ev in MODELS]
rows = [r for r in rows if r[1] is not None]
names = [r[0] for r in rows]
eer_dev = [r[1][0] for r in rows]; eer_ev = [r[2][0] if r[2] else np.nan for r in rows]
td_dev = [r[1][1] for r in rows]; td_ev = [r[2][1] if r[2] else np.nan for r in rows]
x = np.arange(len(names)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, dv, ev, ttl, ylab, floor in [
        (axes[0], eer_dev, eer_ev, "Detection EER - dev vs eval", "EER (%)", 0.01),
        (axes[1], td_dev, td_ev, "min t-DCF - dev vs eval", "min t-DCF", 1e-4)]:
    ax.bar(x - w/2, dv, w, label="dev", color="tab:blue")
    ax.bar(x + w/2, [floor if (v is None or np.isnan(v)) else v for v in ev], w,
           label="eval (unseen)", color="tab:red")
    ax.set_yscale("log"); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.legend()
    for i, v in enumerate(dv):
        ax.annotate(f"{v:g}", (x[i]-w/2, v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(ev):
        miss = v is None or np.isnan(v)
        ax.annotate("n/a" if miss else f"{v:g}", (x[i]+w/2, floor if miss else v),
                    ha="center", va="bottom", fontsize=8, color="gray" if miss else "black")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "eer_tdcf_comparison.png"), dpi=150); plt.close(fig)
print("wrote eer_tdcf_comparison.png")

print("done ->", FIG)
