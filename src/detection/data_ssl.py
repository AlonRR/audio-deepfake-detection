"""Dataset over cached SSL features (manifest.jsonl from features_ssl.py).

Each manifest line: {"npy": <path to (T, D) float16>, "label": 0|1}. Builds a batched
tf.data.Dataset that memory-maps the .npy files, so training the Keras back-end never
touches torch or raw audio.
"""
from __future__ import annotations

import json

import numpy as np


def read_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def feature_dim(entries: list[dict]) -> tuple[int, int]:
    """Return (max_frames, D) inferred from the first cached array."""
    arr = np.load(entries[0]["npy"], mmap_mode="r")
    return int(arr.shape[0]), int(arr.shape[1])


def make_dataset(entries: list[dict], batch_size: int, shuffle: bool):
    import tensorflow as tf

    max_frames, d = feature_dim(entries)

    def gen():
        order = np.random.permutation(len(entries)) if shuffle else range(len(entries))
        for i in order:
            e = entries[i]
            feat = np.asarray(np.load(e["npy"]), dtype=np.float32)
            yield feat, np.int32(e["label"])

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(max_frames, d), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(entries: list[dict]) -> dict[int, float]:
    n1 = sum(e["label"] for e in entries)
    n0 = len(entries) - n1
    total = max(len(entries), 1)
    return {0: total / (2 * max(n0, 1)), 1: total / (2 * max(n1, 1))}
