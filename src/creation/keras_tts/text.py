"""Minimal character-level text processing for the Keras TTS baseline.

Lowercase char vocabulary (letters, digits, common punctuation, space) + special tokens.
Good enough for a single-speaker English baseline; no phonemizer dependency.
"""
from __future__ import annotations

_PAD, _EOS = "_", "~"
_SYMBOLS = _PAD + _EOS + "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-:;"
_CH2ID = {c: i for i, c in enumerate(_SYMBOLS)}

PAD_ID = _CH2ID[_PAD]
EOS_ID = _CH2ID[_EOS]
VOCAB_SIZE = len(_SYMBOLS)


def encode(text: str) -> list[int]:
    """Text -> list of symbol ids (unknown chars dropped), with a trailing EOS."""
    ids = [_CH2ID[c] for c in text.strip().lower() if c in _CH2ID]
    return ids + [EOS_ID]


def decode(ids) -> str:
    return "".join(_SYMBOLS[i] for i in ids if 0 <= i < VOCAB_SIZE)
