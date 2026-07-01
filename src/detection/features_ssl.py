"""Frozen SSL frontend feature extraction (PyTorch container).

A pretrained wav2vec2-XLS-R / WavLM encoder is used as a FIXED feature extractor (weights
frozen, no grad) — analogous to using pretrained word embeddings. Per-utterance frame
features are cropped/padded to a fixed length and cached to disk as float16, so the Keras
back-end (TensorFlow container) trains without ever importing torch.

CLI (run in the PyTorch container as a Slurm job):
    python -m src.detection.features_ssl \
        --data-root /datasets/ASVspoof2019/LA --subset train \
        --out /models/$USER/ssl_xlsr/train --frontend facebook/wav2vec2-xls-r-300m
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.common.audio import TARGET_SR, load_wav
from src.detection import data as D

DEFAULT_FRONTEND = "facebook/wav2vec2-xls-r-300m"  # or "microsoft/wavlm-large"


class SSLExtractor:
    """Frozen HF SSL encoder -> frame-level features (T, hidden_size)."""

    def __init__(self, frontend: str = DEFAULT_FRONTEND, layer: str | int = "avg"):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.layer = layer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(frontend, output_hidden_states=True).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size

    def extract(self, wav: np.ndarray) -> np.ndarray:
        """Return (T, D) frame features for one utterance (single layer or layer-average)."""
        torch = self.torch
        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            hs = self.model(x).hidden_states  # tuple(L+1) of (1, T, D)
        if self.layer == "avg":
            feat = torch.stack(hs, dim=0).mean(dim=0)
        else:
            feat = hs[int(self.layer)]
        return feat.squeeze(0).cpu().numpy().astype(np.float16)


def _fix_frames(feat: np.ndarray, max_frames: int) -> np.ndarray:
    t = feat.shape[0]
    if t >= max_frames:
        return feat[:max_frames]
    pad = np.zeros((max_frames - t, feat.shape[1]), dtype=feat.dtype)
    return np.concatenate([feat, pad], axis=0)


def _npy_path(out_dir: str, item) -> str:
    """Stable per-utterance cache name (keyed on the flac id, so runs are resumable)."""
    utt = os.path.splitext(os.path.basename(item.path))[0]
    return os.path.join(out_dir, f"{utt}.npy")


def cache_subset(data_root, subset, out_dir, frontend, layer, max_frames, limit=None) -> str:
    """Extract + cache features for one ASVspoof subset; write a manifest.jsonl.

    Resumable: utterances already cached (their .npy exists) are skipped, so a job
    killed at the 2 h QOS cap can simply be resubmitted to continue. The manifest is
    rebuilt from every item that has a cached array, so it is always complete/correct.
    """
    os.makedirs(out_dir, exist_ok=True)
    items = D.parse_protocol(data_root, subset, limit)
    todo = [it for it in items if not os.path.exists(_npy_path(out_dir, it))]
    print(f"[ssl] {subset}: {len(items)} total, {len(items) - len(todo)} cached, {len(todo)} to do")

    if todo:
        ext = SSLExtractor(frontend, layer)
        for k, it in enumerate(todo):
            try:
                feat = _fix_frames(ext.extract(load_wav(it.path, TARGET_SR)), max_frames)
            except Exception as exc:
                print(f"[ssl] skip {it.path}: {exc}")
                continue
            np.save(_npy_path(out_dir, it), feat)
            if k % 500 == 0:
                print(f"[ssl] {subset} {k}/{len(todo)}", flush=True)
        print(f"[ssl] extracted D={ext.hidden_size}, T={max_frames}")

    manifest = os.path.join(out_dir, "manifest.jsonl")
    n = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        for it in items:
            npy = _npy_path(out_dir, it)
            if os.path.exists(npy):
                mf.write(json.dumps({"npy": npy, "label": it.label}) + "\n")
                n += 1
    print(f"[ssl] wrote manifest -> {manifest}  ({n}/{len(items)} entries)")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache frozen SSL features for ASVspoof LA.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subset", choices=["train", "dev", "eval"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frontend", default=DEFAULT_FRONTEND)
    ap.add_argument("--layer", default="avg", help='"avg" or an int layer index')
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cache_subset(args.data_root, args.subset, args.out, args.frontend, args.layer,
                 args.max_frames, args.limit)


if __name__ == "__main__":
    main()
