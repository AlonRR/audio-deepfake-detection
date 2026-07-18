"""Signal-level statistics for the generated clips (reproduces results.md §A.1d / §A.1e).

    python -m src.evaluation.signal_stats --out reports/evaluation

These are deliberately CRUDE proxies — duration, RMS, peak, zero-crossing rate — not
perceptual measures. They exist for one argument that the perceptual metrics cannot make:

    A1's output duration has **zero variance**. Every clip lands on the decoder's
    max_steps cap regardless of how long the input text is, which proves the baseline's
    output is independent of its input rather than merely low quality.

Written as a script (rather than computed ad hoc) so every number quoted in §A.1d/§A.1e
is regenerable from the repo.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import wave

import numpy as np

SYSTEMS = {
    "real_heldout":  "data/raw/heldout",
    "keras_tts":     "data/generated/keras_tts",
    "xtts_zeroshot": "data/generated/xtts_zeroshot",
    "xtts_ft":       "data/generated/xtts_ft",
}


def _read(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def clip_stats(path: str) -> dict:
    x, sr = _read(path)
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if x.size > 1 else 0.0
    return {
        "file": os.path.basename(path),
        "dur_s": round(x.size / sr, 3),
        "rms": round(float(np.sqrt((x ** 2).mean())), 5),
        "peak": round(float(np.abs(x).max()), 4),
        "zcr": round(zcr, 4),
    }


def system_stats(folder: str) -> dict:
    clips = [clip_stats(p) for p in sorted(glob.glob(os.path.join(folder, "*.wav")))]
    if not clips:
        return {"error": f"no wavs in {folder}"}
    dur = np.array([c["dur_s"] for c in clips])
    return {
        "n": len(clips),
        "dur_mean": round(float(dur.mean()), 3),
        # The headline number for A1: 0.00 means output length is independent of input.
        "dur_std": round(float(dur.std()), 4),
        "rms_mean": round(float(np.mean([c["rms"] for c in clips])), 5),
        "zcr_mean": round(float(np.mean([c["zcr"] for c in clips])), 4),
        "peak_mean": round(float(np.mean([c["peak"] for c in clips])), 4),
        "n_near_silent": int(sum(c["rms"] <= 0.008 for c in clips)),
        "n_clipping": int(sum(c["peak"] >= 0.999 for c in clips)),
        "clips": clips,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Signal-level stats for generated clips.")
    ap.add_argument("--out", default="reports/evaluation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {name: system_stats(folder) for name, folder in SYSTEMS.items()}
    path = os.path.join(args.out, "signal_stats.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    hdr = f"{'system':<16}{'n':>4}{'dur_mean':>10}{'dur_std':>9}{'rms':>9}{'zcr':>8}{'peak':>8}"
    print(hdr)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<16} {r['error']}")
            continue
        print(f"{name:<16}{r['n']:>4}{r['dur_mean']:>10.2f}{r['dur_std']:>9.2f}"
              f"{r['rms_mean']:>9.4f}{r['zcr_mean']:>8.3f}{r['peak_mean']:>8.3f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
