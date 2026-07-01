"""Detection training + hyper-parameter search (Keras/TensorFlow).

Implements the required "show the process": multiple runs, a hyper-parameter sweep, and
saved learning curves — including at least one deliberately-failed configuration.

TODO (implementation pass):
    - data pipeline over cached SSL features (train/dev split of ASVspoof 2019 LA)
    - train(model, cfg): fit + log loss/EER curves to reports/
    - sweep(grid): LR, SSL layer(s), pooling, width/depth, augmentation (RawBoost)
    - checkpoint best model to models/
CLI: `python -m src.detection.train --config ...` (added in the implementation pass).
"""


def train(model, cfg):
    """Fit a detector and log learning curves. TODO: implement."""
    raise NotImplementedError
