"""Train the Keras Tacotron2-lite baseline on the target speaker's clips.

    python -m src.creation.keras_tts.train --data-root data/raw --epochs 300 --out reports/tts_baseline

Expects data/raw/metadata.csv + data/raw/wavs/*.wav (see dataset.py / prepare.py). Saves
weights + a loss curve; expect the audio to be rough — that's the point (we analyse why).
"""
from __future__ import annotations

import argparse
import os

from src.creation.keras_tts.dataset import load_metadata, make_dataset
from src.creation.keras_tts.model import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Keras TTS baseline (Tacotron2-lite).")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="reports/tts_baseline")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import keras

    items = load_metadata(args.data_root)
    if not items:
        raise SystemExit(f"no clips found under {args.data_root} (need metadata.csv + wavs/)")
    print(f"training on {len(items)} clips")
    ds = make_dataset(items, args.batch_size, shuffle=True)

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0))

    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.out, "history.csv")),
        keras.callbacks.ModelCheckpoint(os.path.join(args.out, "tts.weights.h5"),
                                        monitor="loss", save_best_only=True, save_weights_only=True),
    ]
    history = model.fit(ds, epochs=args.epochs, callbacks=callbacks)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for k in ("loss", "mel", "stop"):
        if k in history.history:
            plt.plot(history.history[k], label=k)
    plt.xlabel("epoch"); plt.legend(); plt.title("Keras TTS baseline training")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "loss_curve.png"), dpi=150)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
