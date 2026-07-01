"""Frozen SSL frontend feature extraction (PyTorch container).

Uses a pretrained wav2vec2-XLS-R / WavLM encoder as a FIXED feature extractor (weights
frozen), analogous to using pretrained word embeddings. Features are cached to disk
(/models/$(whoami) or /projects) so the Keras back-end never needs torch in its env.

TODO (implementation pass):
    - load_frontend(name="facebook/wav2vec2-xls-r-300m" | "microsoft/wavlm-base-plus")
    - extract(wav, layer="all"|int) -> np.ndarray   # frozen forward pass, hidden states
    - cache_dataset(manifest, out_dir): batch-extract ASVspoof + own clips to .npy
The choice of which SSL layer(s) to pool is a hyper-parameter swept in train.py.
"""

DEFAULT_FRONTEND = "facebook/wav2vec2-xls-r-300m"


def extract(wav, sr, layer="all"):
    """Frozen SSL forward pass -> hidden-state features. TODO: implement."""
    raise NotImplementedError
