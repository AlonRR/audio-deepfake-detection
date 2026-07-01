"""Cross-generator test — the project's own contribution.

Score a trained detector on the student's OWN generated clips (an unseen 2025-era
synthesizer) and report how well it flags them, vs its ASVspoof operating point. Supports
both the spectrogram baseline (extract features from audio here) and the SSL back-end
(consume a pre-cached SSL manifest).

    # spectrogram CNN detector, own clips (fakes) + real reference:
    python -m src.detection.cross_test --model reports/run1/model.keras --model-type cnn \
        --feat lfcc --fakes data/generated --real data/raw --out reports/cross

    # SSL back-end (cache own-clip features with features_ssl.py first):
    python -m src.detection.cross_test --model reports/ssl_run1/model.keras --model-type ssl \
        --ssl-manifest /models/$USER/ssl_xlsr/own/manifest.jsonl --out reports/cross_ssl
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from src.common.audio import FeatureConfig
from src.detection.evaluate import compute_eer


def _list_wavs(folder: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))


def _cnn_scores(model, wavs, feat, cfg):
    from src.detection.data import Item, extract

    xs = []
    for w in wavs:
        xs.append(extract(Item(w, 0), feat, cfg))
    x = np.stack(xs) if xs else np.zeros((0, 1))
    return model.predict(x, verbose=0)[:, 1] if len(xs) else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-test a detector on your own fakes.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", choices=["cnn", "ssl"], required=True)
    ap.add_argument("--feat", choices=["logmel", "lfcc"], default="lfcc")
    ap.add_argument("--fakes", help="folder of generated (spoof) wavs (cnn mode)")
    ap.add_argument("--real", help="folder of real (bona fide) wavs (cnn mode)")
    ap.add_argument("--ssl-manifest", help="cached SSL manifest for own clips (ssl mode)")
    ap.add_argument("--out", default="reports/cross")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import keras
    model = keras.models.load_model(args.model)

    if args.model_type == "cnn":
        cfg = FeatureConfig()
        fake_wavs, real_wavs = _list_wavs(args.fakes or ""), _list_wavs(args.real or "")
        fake_scores = _cnn_scores(model, fake_wavs, args.feat, cfg)
        real_scores = _cnn_scores(model, real_wavs, args.feat, cfg)
    else:
        from src.detection import data_ssl as DS
        entries = DS.read_manifest(args.ssl_manifest)
        ds = DS.make_dataset(entries, batch_size=32, shuffle=False)
        probs, labels = [], []
        for x, y in ds:
            probs.append(model(x, training=False).numpy()[:, 1]); labels.append(y.numpy())
        probs, labels = np.concatenate(probs), np.concatenate(labels)
        real_scores, fake_scores = probs[labels == 1], probs[labels == 0]

    # A spoof is "caught" when its bona-fide score is below 0.5.
    caught = float(np.mean(fake_scores < 0.5)) if len(fake_scores) else float("nan")
    result = {"n_fake": int(len(fake_scores)), "n_real": int(len(real_scores)),
              "spoof_detect_rate@0.5": caught,
              "fake_score_mean": float(np.mean(fake_scores)) if len(fake_scores) else None,
              "real_score_mean": float(np.mean(real_scores)) if len(real_scores) else None}
    if len(fake_scores) and len(real_scores):
        eer, thr = compute_eer(real_scores, fake_scores)
        result["cross_eer_pct"] = round(eer * 100, 3)
        result["cross_threshold"] = thr
    with open(os.path.join(args.out, "cross_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
