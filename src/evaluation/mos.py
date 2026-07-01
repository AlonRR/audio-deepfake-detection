"""MOS (Mean Opinion Score) harness (required creation metric #1).

MOS is a human judgement (naturalness, 1-5); this module only prepares the blind test and
aggregates the collected ratings — it never fabricates scores. An optional automatic-MOS
proxy (UTMOS) is provided but must be reported as a proxy, clearly flagged as outside the
course materials.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np


def build_test(clips: dict[str, list[str]], out_csv: str, seed: int = 0) -> None:
    """Write a shuffled, blinded rating sheet.

    ``clips`` maps system name (e.g. "real", "keras_tts", "xtts") -> list of wav paths.
    Output CSV columns: token, path, system(hidden key), rating(blank for the listener).
    """
    rows = []
    for system, paths in clips.items():
        for p in paths:
            rows.append({"path": p, "system": system})
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["token"] = f"clip_{i:03d}"
        r["rating"] = ""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "path", "system", "rating"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} blinded clips -> {out_csv}  (listeners fill the 'rating' column 1-5)")


def aggregate(rating_csvs: list[str]) -> dict:
    """Aggregate one-or-more filled rating sheets into per-system mean MOS + 95% CI."""
    by_system: dict[str, list[float]] = {}
    for path in rating_csvs:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not row.get("rating"):
                    continue
                by_system.setdefault(row["system"], []).append(float(row["rating"]))
    out = {}
    for system, vals in by_system.items():
        a = np.asarray(vals)
        ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        out[system] = {"mos": float(a.mean()), "ci95": float(ci), "n": len(a)}
    return out


def auto_mos(wav_path: str) -> float:
    """Automatic MOS proxy via UTMOS (torch.hub). OUTSIDE course materials — a proxy only."""
    import torch

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    import torchaudio

    wav, sr = torchaudio.load(wav_path)
    return float(predictor(wav, sr).item())


def write_summary(agg: dict, out_json: str) -> None:
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)
