"""Synthesize the demo clips with XTTS-v2 — zero-shot or fine-tuned (PyTorch container).

Uses the SAME prompt set as the Keras baseline so MOS / spectrogram / speaker-similarity
comparisons are apples-to-apples.

    # zero-shot (reference wav only, no training):
    python -m src.creation.xtts_finetune.synthesize --mode zero_shot \
        --ref data/raw/wavs/clip_0000.wav --out data/generated/xtts_zeroshot

    # fine-tuned checkpoint (from finetune.py):
    python -m src.creation.xtts_finetune.synthesize --mode finetuned \
        --ckpt-dir models/xtts_ft/run/ --ref data/raw/wavs/clip_0000.wav \
        --out data/generated/xtts_ft
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from src.common.audio import save_wav
from src.creation.keras_tts.synthesize import PROMPTS

XTTS_SR = 24_000


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _zero_shot(items, ref, out_dir):
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device())
    for i, (name, text) in enumerate(items):
        tts.tts_to_file(text=text, speaker_wav=ref, language="en",
                        file_path=os.path.join(out_dir, f"{name}.wav"))
        print(f"[{i:02d}] {name} {text[:40]!r}")


def _finetuned(items, ref, ckpt_dir, out_dir):
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    cfg = XttsConfig()
    cfg.load_json(os.path.join(ckpt_dir, "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=ckpt_dir, use_deepspeed=False)
    model.to(_device())

    gpt_lat, spk_emb = model.get_conditioning_latents(audio_path=[ref])
    for i, (name, text) in enumerate(items):
        with torch.no_grad():
            out = model.inference(text, "en", gpt_lat, spk_emb, temperature=0.7)
        save_wav(os.path.join(out_dir, f"{name}.wav"),
                 np.asarray(out["wav"], dtype=np.float32), XTTS_SR)
        print(f"[{i:02d}] {name} {text[:40]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize demo clips with XTTS-v2.")
    ap.add_argument("--mode", choices=["zero_shot", "finetuned"], required=True)
    ap.add_argument("--ref", help="speaker reference wav (defaults to first data/raw/wavs clip)")
    ap.add_argument("--ckpt-dir", help="fine-tuned checkpoint dir (finetuned mode)")
    ap.add_argument("--out", required=True)
    # Parallel mode: speak the HELD-OUT transcripts instead of PROMPTS so MCD /
    # log-mel SSIM compare identical content against the real held-out audio.
    ap.add_argument("--texts-csv", help="LJSpeech metadata to speak instead of PROMPTS")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = args.ref or next(iter(sorted(glob.glob("data/raw/wavs/*.wav"))), None)
    if not ref:
        raise SystemExit("no speaker reference wav (pass --ref or run prepare.py first)")

    if args.texts_csv:
        from src.common.metadata import load_texts
        items = load_texts(args.texts_csv)
    else:
        items = [(f"xtts_{i:02d}", t) for i, t in enumerate(PROMPTS)]

    if args.mode == "zero_shot":
        _zero_shot(items, ref, args.out)
    else:
        if not args.ckpt_dir:
            raise SystemExit("--ckpt-dir required for finetuned mode")
        _finetuned(items, ref, args.ckpt_dir, args.out)
    print(f"wrote {len(items)} clips -> {args.out}")


if __name__ == "__main__":
    main()
