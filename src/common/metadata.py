"""Read the LJSpeech-style metadata written by prepare.py.

    <clip_id>|<transcription>|<normalized_transcription>

Used two ways:
  * training (metadata.csv, the 30-clip train split);
  * parallel synthesis (metadata_eval.csv) - synthesizing the HELD-OUT transcripts
    so MCD / log-mel SSIM compare like with like. Those metrics need the same
    spoken content on both sides; scoring a synthesized "hello." against a real
    "Somewhere beyond the ridge..." would be meaningless.
"""
from __future__ import annotations

import os


def load_texts(csv_path: str) -> list[tuple[str, str]]:
    """Return [(clip_id, transcript), ...] from an LJSpeech-style metadata file."""
    rows: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2 or not parts[0]:
                continue
            rows.append((parts[0], parts[1]))
    return rows


def eval_pairs(data_root: str = "data/raw",
               eval_csv: str = "metadata_eval.csv") -> list[tuple[str, str, str]]:
    """Return [(clip_id, transcript, real_wav_path), ...] for the held-out clips."""
    rows = load_texts(os.path.join(data_root, eval_csv))
    out = []
    for cid, text in rows:
        wav = os.path.join(data_root, "wavs", f"{cid}.wav")
        if os.path.isfile(wav):
            out.append((cid, text, wav))
    return out
