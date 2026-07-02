"""Train the SSL-frontend detection back-end on cached features, report EER.

Runs in the TensorFlow container. Expects features already cached by features_ssl.py
(run first in the PyTorch container).

    python -m src.detection.train_ssl \
        --train /models/$USER/ssl_xlsr/train/manifest.jsonl \
        --dev   /models/$USER/ssl_xlsr/dev/manifest.jsonl \
        --epochs 40 --batch-size 64 --out reports/ssl_run1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.detection import data_ssl as DS
from src.detection.backend_keras import build_backend
from src.detection.evaluate import eer_from_scores, min_tdcf, save_det_curve
from src.detection.train import _plot_curves, _score  # reuse curve + scoring helpers


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SSL-feature detection back-end (Keras).")
    ap.add_argument("--train", required=True, help="train manifest.jsonl")
    ap.add_argument("--dev", required=True, help="dev manifest.jsonl")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--proj", type=int, default=256, help="Conv1D projection width")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--out", default="reports/ssl_run")
    ap.add_argument("--asv-scores", default=None, help="ASVspoof ASV dev score file -> min t-DCF")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import keras

    train_e = DS.read_manifest(args.train)
    dev_e = DS.read_manifest(args.dev)
    max_frames, d = DS.feature_dim(train_e)
    print(f"train={len(train_e)} dev={len(dev_e)}  feat=(T={max_frames}, D={d})")

    train_ds = DS.make_dataset(train_e, args.batch_size, shuffle=True)
    dev_ds = DS.make_dataset(dev_e, args.batch_size, shuffle=False)

    model = build_backend(d, max_frames, proj=args.proj, dropout=args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.out, "history.csv")),
        keras.callbacks.ModelCheckpoint(os.path.join(args.out, "model.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]
    history = model.fit(train_ds, validation_data=dev_ds, epochs=args.epochs,
                        class_weight=DS.class_weights(train_e), callbacks=callbacks)

    _plot_curves(history, args.out)
    scores, labels = _score(model, dev_ds)
    eer, thr = eer_from_scores(scores, labels)
    save_det_curve(scores[labels == 1], scores[labels == 0],
                   os.path.join(args.out, "det_dev.png"), title=f"SSL DET (dev) EER={eer*100:.2f}%")
    np.savez(os.path.join(args.out, "dev_scores.npz"), scores=scores, labels=labels)
    result = {"dev_eer_pct": round(eer * 100, 3), "threshold": thr,
              "n_train": len(train_e), "feat_dim": d,
              "lr": args.lr, "proj": args.proj, "dropout": args.dropout,
              "epochs": args.epochs, "batch_size": args.batch_size}
    if args.asv_scores:
        result["min_tdcf"] = round(min_tdcf(scores[labels == 1], scores[labels == 0], args.asv_scores), 5)
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSSL DEV EER = {eer*100:.2f}%  ->  {args.out}/result.json")


if __name__ == "__main__":
    main()
