"""Prepare the target-speaker dataset from a single recording (PyTorch container).

Segments a long recording into short clips and transcribes them with Whisper, writing an
LJSpeech-style dataset that BOTH creation systems consume:

    <out>/metadata.csv        TRAIN split - lines: "<clip_id>|<transcript>"
    <out>/metadata_eval.csv   held-out clips, unseen real reference for evaluation
    <out>/metadata_all.csv    every clip, for reference
    <out>/wavs/<clip_id>.wav  (22.05 kHz mono)

    python -m src.creation.xtts_finetune.prepare --source data/raw/source.wav --out data/raw

The holdout exists so speaker-cosine has a real-vs-real ceiling and MOS has a real
anchor drawn from audio the model never saw. Trainers read metadata.csv, so the
held-out clips are excluded by construction.
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


def _shares_phrase(a: str, b: str, min_words: int = 5) -> bool:
    """True if two transcripts share a run of >= min_words words.

    The speaker re-read several sentences, so consecutive clips can contain near
    duplicate content. Those must not be split across train/eval: holding out one
    half while its twin stays in training leaks the eval audio into the model.
    """
    wa, wb = a.lower().split(), set()
    tb = b.lower().split()
    for i in range(len(tb) - min_words + 1):
        wb.add(" ".join(tb[i:i + min_words]))
    return any(" ".join(wa[i:i + min_words]) in wb
               for i in range(len(wa) - min_words + 1))


def _pick_holdout(rows: list[tuple[str, str, float]], k: int,
                  min_ref_s: float = 4.0) -> set[str]:
    """Choose k eval-reference clips spread evenly across the recording.

    Three constraints, all of which matter for the evaluation to mean anything:
      * skip clips sharing a phrase with a neighbour (a re-read) — holding out one
        half while its twin stays in training leaks the eval audio into the model;
      * require >= min_ref_s of speech — ECAPA speaker embeddings are unstable on
        very short clips, and two words cannot be MOS-rated;
      * spread the picks so all five script sections are represented.
    """
    if k <= 0 or len(rows) <= k:
        return set()
    tainted = set()
    for i in range(len(rows) - 1):
        if _shares_phrase(rows[i][1], rows[i + 1][1]):
            tainted.add(i)
            tainted.add(i + 1)
    eligible = [i for i in range(len(rows))
                if i not in tainted and rows[i][2] >= min_ref_s]
    if len(eligible) < k:                       # fall back to even spacing over all
        eligible = list(range(len(rows)))
    step = len(eligible) / k
    return {rows[eligible[min(int(j * step + step / 2), len(eligible) - 1)]][0]
            for j in range(k)}


def segment(source: str, out: str, model_size: str = "small",
            min_s: float = 2.0, max_s: float = 12.0, holdout: int = 0) -> int:
    """Transcribe with faster-whisper and cut on segment boundaries."""
    from faster_whisper import WhisperModel

    wavs_dir = os.path.join(out, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    wav = load_wav(source, CFG.sr)

    device = _device()
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, _ = model.transcribe(source, language="en", vad_filter=True)

    fixes = _load_fixes(out)
    rows: list[tuple[str, str, float]] = []
    n, n_fixed = 0, 0
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
        rows.append((cid, text, dur))
        n += 1

    # metadata.csv is what the trainers read, so the holdout is excluded from it by
    # construction — you cannot accidentally train on the evaluation reference.
    eval_ids = _pick_holdout(rows, holdout)
    _write(os.path.join(out, "metadata_all.csv"), rows)
    _write(os.path.join(out, "metadata.csv"), [r for r in rows if r[0] not in eval_ids])
    if eval_ids:
        _write(os.path.join(out, "metadata_eval.csv"), [r for r in rows if r[0] in eval_ids])

    total_min = wav.size / CFG.sr / 60.0
    print(f"segmented {n} clips from {total_min:.1f} min -> {out}")
    if fixes:
        print(f"applied {n_fixed} spoken-form transcript fixes")
    if eval_ids:
        print(f"held out {len(eval_ids)} unseen eval-reference clips: {', '.join(sorted(eval_ids))}")
        print(f"train set: {n - len(eval_ids)} clips -> metadata.csv")
    return n


def _write(path: str, rows: list[tuple[str, str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for cid, text, _dur in rows:
            fh.write(f"{cid}|{text}\n")


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
    ap.add_argument("--holdout", type=int, default=4,
                    help="clips reserved as UNSEEN real reference for speaker-cosine / MOS")
    args = ap.parse_args()
    n = segment(args.source, args.out, args.whisper, args.min_s, args.max_s, args.holdout)
    if n < 10:
        print(f"WARNING: only {n} usable clips — record more speech or lower --whisper thresholds.")


if __name__ == "__main__":
    main()
