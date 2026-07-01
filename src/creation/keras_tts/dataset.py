"""Build (text, mel) training pairs for the Keras TTS baseline.

Expects an LJSpeech-style layout under the data root:
    <root>/metadata.csv      lines: "<clip_id>|<transcript>"   ('|'-separated)
    <root>/wavs/<clip_id>.wav
(produced from your 1-5 min recording by src/creation/xtts_finetune/prepare.py, which
segments + transcribes; the same manifest feeds both creation systems.)
"""
from __future__ import annotations

import os

import numpy as np

from src.common.audio import load_wav
from src.creation.keras_tts.audio_tts import CFG, wav_to_mel
from src.creation.keras_tts.text import PAD_ID, encode


def load_metadata(root: str) -> list[tuple[str, str]]:
    items = []
    with open(os.path.join(root, "metadata.csv"), "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            clip_id, text = parts[0], parts[1]
            wav = os.path.join(root, "wavs", f"{clip_id}.wav")
            if os.path.isfile(wav):
                items.append((wav, text))
    return items


def make_dataset(items, batch_size: int, shuffle: bool, cfg=CFG):
    import tensorflow as tf

    def gen():
        order = np.random.permutation(len(items)) if shuffle else range(len(items))
        for i in order:
            wav_path, text = items[i]
            wav = load_wav(wav_path, cfg.sr)
            mel = wav_to_mel(wav, cfg)                       # (Tm, n_mels)
            ids = np.asarray(encode(text), dtype=np.int32)   # (Tt,)
            yield ids, mel.astype(np.float32), np.int32(mel.shape[0])

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None, cfg.n_mels), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(256)
    return ds.padded_batch(
        batch_size,
        padded_shapes=([None], [None, cfg.n_mels], []),
        padding_values=(PAD_ID, 0.0, 0),
    ).prefetch(tf.data.AUTOTUNE)
