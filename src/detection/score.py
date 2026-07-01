"""Score a trained detector on a held-out set (e.g. ASVspoof eval) -> EER + min t-DCF.

The dev protocol reuses train's attacks (A01-A06), so dev EER is optimistic; the *eval*
protocol (A07-A19, unseen attacks) is the real generalization test. This loads a saved
model and scores it on such a set.

    # SSL back-end on cached eval features (cache them first with features_ssl.py --subset eval):
    python -m src.detection.score --mode ssl --model reports/ssl_xlsr_run1/model.keras \
        --ssl-manifest features/ssl_xlsr/eval/manifest.jsonl \
        --asv-scores <LA>/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt \
        --out reports/ssl_xlsr_eval

    # spectrogram CNN on the eval protocol (features extracted on the fly):
    python -m src.detection.score --mode cnn --model reports/base_cnn_lfcc_full/model.keras \
        --data-root <LA> --subset eval --feat lfcc \
        --asv-scores <...asv.eval...> --out reports/base_cnn_lfcc_eval
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from src.common.audio import FeatureConfig
from src.detection.evaluate import eer_from_scores, min_tdcf, save_det_curve


def _score(model, ds) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    for x, y in ds:
        scores.append(model(x, training=False).numpy()[:, 1])
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a trained detector on a held-out set.")
    ap.add_argument("--mode", choices=["cnn", "ssl"], required=True)
    ap.add_argument("--model", required=True, help="saved .keras model")
    ap.add_argument("--data-root", help="ASVspoof LA root (cnn mode)")
    ap.add_argument("--subset", default="eval", help="protocol subset (cnn mode)")
    ap.add_argument("--feat", choices=["logmel", "lfcc"], default="lfcc")
    ap.add_argument("--ssl-manifest", help="cached SSL manifest (ssl mode)")
    ap.add_argument("--asv-scores", default=None, help="ASV score file -> also report min t-DCF")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="reports/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import keras
    # importing backend_keras registers AttentiveStatsPooling so the SSL model reloads
    from src.detection import backend_keras  # noqa: F401
    model = keras.models.load_model(
        args.model, custom_objects={"AttentiveStatsPooling": backend_keras.AttentiveStatsPooling})

    if args.mode == "ssl":
        from src.detection import data_ssl as DS
        entries = DS.read_manifest(args.ssl_manifest)
        ds = DS.make_dataset(entries, args.batch_size, shuffle=False)
    else:
        from src.detection import data as D
        items = D.parse_protocol(args.data_root, args.subset, None)
        ds = D.make_dataset(items, args.feat, FeatureConfig(), args.batch_size, shuffle=False)

    scores, labels = _score(model, ds)
    eer, thr = eer_from_scores(scores, labels)
    save_det_curve(scores[labels == 1], scores[labels == 0],
                   os.path.join(args.out, "det_eval.png"),
                   title=f"{args.subset} DET  EER={eer*100:.2f}%")
    np.savez(os.path.join(args.out, "eval_scores.npz"), scores=scores, labels=labels)
    result = {"subset": args.subset, "mode": args.mode, "eer_pct": round(eer * 100, 3),
              "threshold": float(thr), "n": int(len(labels)), "n_bonafide": int(labels.sum())}
    if args.asv_scores:
        result["min_tdcf"] = round(min_tdcf(scores[labels == 1], scores[labels == 0], args.asv_scores), 5)
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
