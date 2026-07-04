"""Regenerate all detection report figures from the committed score files.

Run from the repo root:
    uv run --no-project --with numpy --with matplotlib --with scipy python scripts/plot_detection.py

Reads reports/detection/<run>/{dev,eval}_scores.npz (+ result.json for EER labels),
writes figures to reports/figures/. Everything needed to rebuild the graphs is tracked.
"""
import csv
import glob
import json
import os
import re

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

# --- SSL back-end learning-rate sweep (dev) ---
# The LR sweep holds proj=256 / dropout=0.3 fixed (scripts/hp_ssl.slurm). *_FAIL dirs
# are deliberate failures identified by name — never sweep points, even if a resumed
# job someday drops a result.json into one.
SWEEP_PROJ, SWEEP_DROPOUT = 256, 0.3
SWEEP_FLOOR = 0.01  # log-axis floor (same as the bar chart); an exact-0 EER still plots
sweep, sweep_skipped = [], []
for p in glob.glob(os.path.join(DET, "ssl_hp_lr*", "result.json")):
    run = os.path.basename(os.path.dirname(p))
    if run.endswith("_FAIL"):
        continue
    try:
        with open(p) as fh:
            r = json.load(fh)
    except (OSError, json.JSONDecodeError):
        sweep_skipped.append(run); continue
    lr, eer, proj, drop = (r.get(k) for k in ("lr", "dev_eer_pct", "proj", "dropout"))
    if (None in (lr, eer, proj, drop) or proj != SWEEP_PROJ
            or abs(drop - SWEEP_DROPOUT) > 1e-6):
        sweep_skipped.append(run); continue
    sweep.append((lr, eer))
sweep.sort()
if sweep_skipped:
    print("sweep: skipped (unreadable or off-axis):", ", ".join(sweep_skipped))

# Deliberate failures: lr parsed from the dir name (lr<m>e<x> = m*10^-x), peak loss
# from the tracked history.csv. No EER was ever scored for these.
fails = []
for d in glob.glob(os.path.join(DET, "ssl_hp_lr*_FAIL")):
    m = re.fullmatch(r"ssl_hp_lr(\d)e(\d)_FAIL", os.path.basename(d))
    hist = os.path.join(d, "history.csv")
    if not (m and os.path.exists(hist)):
        continue
    with open(hist) as fh:
        peak = max(float(row["loss"]) for row in csv.DictReader(fh))
    fails.append((int(m.group(1)) * 10.0 ** -int(m.group(2)), peak))

if sweep:
    lrs = [s[0] for s in sweep]; eers = [max(s[1], SWEEP_FLOOR) for s in sweep]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(lrs, eers, "o-", color="tab:green", lw=2, label="dev EER (%)")
    for (lr, eer), y in zip(sweep, eers):
        ax.annotate(f"{eer:.3f}", (lr, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    for lr, peak in fails:  # marked at the 50% chance line, not a measured EER
        ax.plot([lr], [50.0], "X", color="tab:red", markersize=13,
                label=f"lr={lr:g} diverged (loss {peak:.1f}, no EER)")
        ax.annotate("diverged", (lr, 50.0), textcoords="offset points",
                    xytext=(0, -16), ha="center", color="tab:red", fontsize=8)
    ax.set_xlabel("learning rate (log)"); ax.set_ylabel("dev EER %  (log)")
    ax.set_title("SSL back-end: learning-rate sweep (dev)")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ssl_lr_sweep.png"), dpi=150); plt.close(fig)
    print("wrote ssl_lr_sweep.png")

print("done ->", FIG)
