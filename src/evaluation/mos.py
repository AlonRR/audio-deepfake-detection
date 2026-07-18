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


def build_test(clips: dict[str, list[str]], out_dir: str, seed: int = 0) -> None:
    """Build a genuinely blind listening test.

    ``clips`` maps system name (e.g. "real", "keras_tts", "xtts_ft") -> wav paths.

    Writes THREE things, and the separation is the point:
      <out_dir>/audio/<token>.wav   clips copied under opaque token names, so the
                                    filename cannot leak the system;
      <out_dir>/mos_sheet.csv       token,rating - the ONLY file a listener sees;
      <out_dir>/mos_key.csv         token,path,system - the hidden key, kept back
                                    until ratings are collected.

    The earlier version put the `system` column in the listener's own sheet and
    kept source filenames, so a rater could see they were scoring "xtts_00.wav".
    That is not a blind test and the resulting MOS would not be defensible.
    """
    import shutil

    rows = []
    for system, paths in clips.items():
        for p in paths:
            rows.append({"path": p, "system": system})
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)

    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    for i, r in enumerate(rows):
        r["token"] = f"clip_{i:03d}"
        shutil.copyfile(r["path"], os.path.join(audio_dir, f"{r['token']}.wav"))

    sheet = os.path.join(out_dir, "mos_sheet.csv")
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "rating"])
        w.writeheader()
        w.writerows([{"token": r["token"], "rating": ""} for r in rows])

    key = os.path.join(out_dir, "mos_key.csv")
    with open(key, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["token", "path", "system"])
        w.writeheader()
        w.writerows([{k: r[k] for k in ("token", "path", "system")} for r in rows])

    n_sys = len(clips)
    print(f"blind MOS test -> {out_dir}")
    print(f"  audio/      {len(rows)} clips from {n_sys} systems, token-named")
    print(f"  mos_sheet.csv  give THIS to listeners (rate each token 1-5)")
    print(f"  mos_key.csv    keep hidden until ratings are in")


def aggregate(rating_csvs: list[str], key_csv: str) -> dict:
    """Join filled sheets against the hidden key -> per-system mean MOS + 95% CI."""
    key: dict[str, str] = {}
    with open(key_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key[row["token"]] = row["system"]

    by_system: dict[str, list[float]] = {}
    for path in rating_csvs:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not row.get("rating"):
                    continue
                system = key.get(row["token"])
                if system is None:
                    continue
                by_system.setdefault(system, []).append(float(row["rating"]))
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
