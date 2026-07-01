"""Train the baseline CNN detector on ASVspoof 2019 LA and report EER.

Runs in the lab's TensorFlow GPU container. Produces the "show the process" artifacts:
a learning-curve PNG, a DET curve, the dev EER, and a saved model + score file.

Example (inside the TF container on the L4):
    python -m src.detection.train \
        --data-root /datasets/ASVspoof2019/LA --feat lfcc \
        --epochs 30 --batch-size 32 --out reports/run1

Smoke run (tiny subset, to prove the pipeline before the full run):
    python -m src.detection.train --data-root <LA> --limit 512 --epochs 2 --out reports/smoke
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.common.audio import FeatureConfig
from src.detection import data as D
from src.detection.baseline_cnn import build_cnn
from src.detection.evaluate import eer_from_scores, save_det_curve


def _plot_curves(history, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["loss"], label="train")
    if "val_loss" in h:
        ax[0].plot(h["val_loss"], label="val")
    ax[0].set(title="loss", xlabel="epoch"); ax[0].legend()
    acc = next((k for k in h if "accuracy" in k and not k.startswith("val")), None)
    if acc:
        ax[1].plot(h[acc], label="train")
        if f"val_{acc}" in h:
            ax[1].plot(h[f"val_{acc}"], label="val")
        ax[1].set(title="accuracy", xlabel="epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150)
    plt.close(fig)


def _score(model, ds) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores = P(bona fide), labels) aligned by iterating the dataset."""
    scores, labels = [], []
    for x, y in ds:
        p = model(x, training=False).numpy()[:, 1]
        scores.append(p)
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train baseline CNN spoof detector on ASVspoof LA.")
    ap.add_argument("--data-root", required=True, help="ASVspoof2019 LA root folder")
    ap.add_argument("--feat", choices=["logmel", "lfcc"], default="lfcc")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="cap #train/#dev items (smoke)")
    ap.add_argument("--out", default="reports/run", help="output dir for curves/model/scores")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = FeatureConfig()

    import keras

    train_items = D.parse_protocol(args.data_root, "train", args.limit)
    dev_items = D.parse_protocol(args.data_root, "dev", args.limit)
    print(f"train={len(train_items)}  dev={len(dev_items)}  feat={args.feat}")

    train_ds = D.make_dataset(train_items, args.feat, cfg, args.batch_size, shuffle=True)
    dev_ds = D.make_dataset(dev_items, args.feat, cfg, args.batch_size, shuffle=False)

    n_feat = cfg.n_mels if args.feat == "logmel" else cfg.n_lfcc
    model = build_cnn((n_feat, cfg.max_frames, 1))
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt = os.path.join(args.out, "model.keras")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.out, "history.csv")),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    history = model.fit(
        train_ds, validation_data=dev_ds, epochs=args.epochs,
        class_weight=D.class_weights(train_items), callbacks=callbacks,
    )

    _plot_curves(history, args.out)
    scores, labels = _score(model, dev_ds)
    eer, thr = eer_from_scores(scores, labels)
    save_det_curve(scores[labels == 1], scores[labels == 0],
                   os.path.join(args.out, "det_dev.png"), title=f"DET (dev) EER={eer*100:.2f}%")
    np.savez(os.path.join(args.out, "dev_scores.npz"), scores=scores, labels=labels)

    result = {"dev_eer_pct": round(eer * 100, 3), "threshold": thr,
              "feat": args.feat, "epochs": args.epochs, "n_train": len(train_items)}
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nDEV EER = {eer*100:.2f}%   ->  {args.out}/result.json")


if __name__ == "__main__":
    main()
