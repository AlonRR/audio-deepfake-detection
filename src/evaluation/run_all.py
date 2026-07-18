"""Run the full creation-evaluation harness and write the results table.

    python -m src.evaluation.run_all --out reports/evaluation

Covers the metrics the brief suggests (it says "metrics such as", so they are
examples, not a fixed requirement), each scored the way it is actually valid:

* **Spectrogram similarity (MCD, log-mel L2/SSIM)** - needs PARALLEL utterances,
  i.e. the same words on both sides. Scored on clips synthesized from the four
  HELD-OUT transcripts against the real held-out audio. Comparing a synthesized
  "hello." against a real "Somewhere beyond the ridge..." would be meaningless.
* **Speaker-embedding cosine (ECAPA-TDNN)** - measures speaker identity, not
  content, so it is scored on the 12 varying-length prompt clips against the real
  held-out centroid. A real-vs-centroid figure is reported as the ceiling.
* **MOS** - human listeners. This script only BUILDS the blind test; it never
  invents ratings. See `src.evaluation.mos`.

Everything is scored against the held-out clips in metadata_eval.csv, never
against audio the models trained on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from src.common.metadata import eval_pairs

# system -> (parallel dir for MCD/SSIM, prompt dir for cosine + MOS)
SYSTEMS = {
    "keras_tts":     ("data/generated/keras_tts_parallel",     "data/generated/keras_tts"),
    "xtts_zeroshot": ("data/generated/xtts_zeroshot_parallel", "data/generated/xtts_zeroshot"),
    "xtts_ft":       ("data/generated/xtts_ft_parallel",       "data/generated/xtts_ft"),
}


def _spectrogram(pairs, parallel_dir):
    """MCD / L2 / SSIM over (real_held_out, synthesized_same_text) pairs."""
    from src.evaluation import spectrogram_sim

    matched = []
    for cid, _text, real_wav in pairs:
        syn = os.path.join(parallel_dir, f"{cid}.wav")
        if os.path.isfile(syn):
            matched.append((real_wav, syn))
    if not matched:
        return {"error": f"no parallel clips in {parallel_dir}"}
    return spectrogram_sim.score_pairs(matched)


def _speaker(pairs, prompt_dir):
    """ECAPA cosine of prompt clips vs the real held-out centroid."""
    from src.evaluation import speaker_sim

    cloned = sorted(glob.glob(os.path.join(prompt_dir, "*.wav")))
    if not cloned:
        return {"error": f"no clips in {prompt_dir}"}
    return speaker_sim.score([p[2] for p in pairs], cloned)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the creation evaluation harness.")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="reports/evaluation")
    ap.add_argument("--skip-speaker", action="store_true",
                    help="skip ECAPA (needs speechbrain + torchaudio)")
    ap.add_argument("--build-mos", action="store_true",
                    help="also build the blind MOS listening test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs = eval_pairs(args.data_root)
    if not pairs:
        raise SystemExit("no held-out clips found - run prepare.py with --holdout")
    print(f"held-out reference clips: {[c for c, _, _ in pairs]}")

    results: dict[str, dict] = {}
    for name, (parallel_dir, prompt_dir) in SYSTEMS.items():
        entry: dict = {}
        if os.path.isdir(parallel_dir):
            entry["spectrogram"] = _spectrogram(pairs, parallel_dir)
        else:
            entry["spectrogram"] = {"error": f"missing {parallel_dir}"}
        if args.skip_speaker:
            entry["speaker"] = {"skipped": True}
        elif os.path.isdir(prompt_dir):
            try:
                entry["speaker"] = _speaker(pairs, prompt_dir)
            except Exception as e:                      # noqa: BLE001 - report, don't crash
                entry["speaker"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            entry["speaker"] = {"error": f"missing {prompt_dir}"}
        results[name] = entry
        print(f"  {name}: {json.dumps(entry)[:160]}")

    if args.build_mos:
        from src.evaluation import mos
        clips = {"real": [p[2] for p in pairs]}
        for name, (_par, prompt_dir) in SYSTEMS.items():
            found = sorted(glob.glob(os.path.join(prompt_dir, "*.wav")))
            if found:
                clips[name] = found
        if len(clips) > 1:
            mos.build_test(clips, os.path.join(args.out, "mos"))

    out_json = os.path.join(args.out, "creation_results.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"held_out": [c for c, _, _ in pairs], "systems": results}, fh, indent=2)
    _write_table(results, os.path.join(args.out, "creation_results.md"))
    print(f"\nwrote {out_json}")


def _write_table(results: dict, path: str) -> None:
    def g(d, *keys, fmt="{:.3f}"):
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return fmt.format(d) if isinstance(d, (int, float)) else "—"

    lines = [
        "| System | MCD (dB) ↓ | log-mel L2 ↓ | log-mel SSIM ↑ | Speaker cosine ↑ |",
        "|---|---|---|---|---|",
    ]
    for name, e in results.items():
        lines.append(
            f"| {name} "
            f"| {g(e, 'spectrogram', 'mcd_mean', fmt='{:.2f}')} "
            f"| {g(e, 'spectrogram', 'logmel_l2_mean')} "
            f"| {g(e, 'spectrogram', 'ssim_mean')} "
            f"| {g(e, 'speaker', 'cloned_cosine_mean')} |"
        )
    any_spk = next((e["speaker"] for e in results.values()
                    if isinstance(e.get("speaker"), dict)
                    and "real_vs_centroid_mean" in e["speaker"]), None)
    if any_spk:
        lines.append(f"| *real (ceiling)* | — | — | — "
                     f"| {any_spk['real_vs_centroid_mean']:.3f} |")
    lines.append("")
    lines.append("MOS: pending human listening panel (see reports/evaluation/mos/).")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
