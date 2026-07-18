"""Prepare the target-speaker dataset from a single recording (PyTorch container).

Segments a long recording into short clips and transcribes them with Whisper, writing an
LJSpeech-style dataset that BOTH creation systems consume:

    <out>/metadata.csv     lines: "<clip_id>|<transcript>"
    <out>/wavs/<clip_id>.wav   (22.05 kHz mono)

    python -m src.creation.xtts_finetune.prepare --source data/raw/source.wav --out data/raw
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.common.audio import load_wav, save_wav
from src.creation.keras_tts.audio_tts import CFG


def _load_fixes(out: str) -> dict[str, str]:
    """Spoken-form rewrites for transcripts (see data/raw/transcript_fixes.json).

    Whisper emits numerals for spoken numbers, which misaligns text and audio for a
    character-level TTS. Keyed by substring so the map survives re-segmentation.
    """
    import json

    path = os.path.join(out, "transcript_fixes.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("fixes", {})


def segment(source: str, out: str, model_size: str = "small",
            min_s: float = 2.0, max_s: float = 12.0) -> int:
    """Transcribe with faster-whisper and cut on segment boundaries."""
    from faster_whisper import WhisperModel

    wavs_dir = os.path.join(out, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    wav = load_wav(source, CFG.sr)

    device = _device()
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, _ = model.transcribe(source, language="en", vad_filter=True)

    fixes = _load_fixes(out)
    n, n_fixed = 0, 0
    with open(os.path.join(out, "metadata.csv"), "w", encoding="utf-8") as meta:
        for seg in segments:
            dur = seg.end - seg.start
            text = seg.text.strip()
            if dur < min_s or dur > max_s or not text:
                continue
            for wrong, right in fixes.items():
                if wrong in text:
                    text = text.replace(wrong, right)
                    n_fixed += 1
            a, b = int(seg.start * CFG.sr), int(seg.end * CFG.sr)
            clip = wav[a:b]
            if clip.size < int(min_s * CFG.sr):
                continue
            cid = f"clip_{n:04d}"
            save_wav(os.path.join(wavs_dir, f"{cid}.wav"), clip, CFG.sr)
            meta.write(f"{cid}|{text}\n")
            n += 1
    total_min = wav.size / CFG.sr / 60.0
    print(f"segmented {n} clips from {total_min:.1f} min -> {out}")
    if fixes:
        print(f"applied {n_fixed} spoken-form transcript fixes")
    return n


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment + transcribe a recording into a TTS dataset.")
    ap.add_argument("--source", required=True, help="a single wav/mp3 of the target speaker")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--whisper", default="small")
    # Defaults are deliberately a touch wider than Whisper's typical segment spread:
    # on a single short recording, clips that fall just outside the window are pure
    # data loss, and with only ~4 min of speech every second counts.
    ap.add_argument("--min-s", type=float, default=1.5, help="drop clips shorter than this")
    ap.add_argument("--max-s", type=float, default=13.0, help="drop clips longer than this")
    args = ap.parse_args()
    n = segment(args.source, args.out, args.whisper, args.min_s, args.max_s)
    if n < 10:
        print(f"WARNING: only {n} usable clips — record more speech or lower --whisper thresholds.")


if __name__ == "__main__":
    main()
