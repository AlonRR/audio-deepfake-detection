"""ASVspoof 2019 LA data loading for the detection baseline.

Standard LA layout (as shipped / as staged on the lab's /datasets share):

    <root>/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
    <root>/ASVspoof2019_LA_train/flac/<utt>.flac
    <root>/ASVspoof2019_LA_dev/flac/<utt>.flac
    <root>/ASVspoof2019_LA_eval/flac/<utt>.flac

Protocol line: `SPEAKER  UTT_ID  -  SYSTEM_ID  {bonafide|spoof}`  (label is the last field).
Labels: bona fide -> 1, spoof -> 0. `-` and system id are ignored by the baseline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from src.common.audio import FeatureConfig, fix_length, load_wav, log_mel, lfcc

_SUBSET = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}


@dataclass
class Item:
    path: str
    label: int  # 1 = bona fide, 0 = spoof


def parse_protocol(root: str, subset: str, limit: int | None = None) -> list[Item]:
    """Parse an LA protocol file into (flac_path, label) items."""
    audio_dir, proto_name = _SUBSET[subset]
    proto = os.path.join(root, "ASVspoof2019_LA_cm_protocols", proto_name)
    flac_dir = os.path.join(root, audio_dir, "flac")

    items: list[Item] = []
    with open(proto, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            utt, label = parts[1], parts[-1]
            items.append(Item(os.path.join(flac_dir, f"{utt}.flac"), 1 if label == "bonafide" else 0))
    if limit:
        # keep both classes when sub-sampling for a smoke run
        bona = [i for i in items if i.label == 1][: limit // 2]
        spoof = [i for i in items if i.label == 0][: limit - len(bona)]
        items = bona + spoof
    return items


def extract(item: Item, feat: str, cfg: FeatureConfig) -> np.ndarray:
    """Load one utterance -> fixed-size (F, T, 1) feature map."""
    wav = load_wav(item.path, cfg.sr)
    f = log_mel(wav, cfg) if feat == "logmel" else lfcc(wav, cfg)
    f = fix_length(f, cfg.max_frames)
    return f[..., np.newaxis].astype(np.float32)


def make_dataset(items, feat: str, cfg: FeatureConfig, batch_size: int, shuffle: bool):
    """Build a batched tf.data.Dataset that lazily extracts features via a generator."""
    import tensorflow as tf

    n_feat = cfg.n_mels if feat == "logmel" else cfg.n_lfcc

    def gen():
        order = np.random.permutation(len(items)) if shuffle else range(len(items))
        for i in order:
            it = items[i]
            try:
                yield extract(it, feat, cfg), np.int32(it.label)
            except Exception as exc:  # a corrupt/missing flac shouldn't kill training
                print(f"[data] skip {it.path}: {exc}")

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(n_feat, cfg.max_frames, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(1024)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(items) -> dict[int, float]:
    """Inverse-frequency class weights (ASVspoof train is ~90% spoof)."""
    n1 = sum(i.label for i in items)
    n0 = len(items) - n1
    total = max(len(items), 1)
    return {0: total / (2 * max(n0, 1)), 1: total / (2 * max(n1, 1))}
