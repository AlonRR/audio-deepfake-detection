"""First-try detection baseline (B1) — a light spectrogram CNN in Keras.

Deliberately simple (Conv -> BN -> ReLU -> Pool stacks, in the style of exs/ex4) so it
establishes the ASVspoof pipeline and a weak EER to improve on. Input is a fixed-size
2-D feature map (log-mel or LFCC), shape (F, T, 1); output is P(bona fide).

Import is lazy-TF-free at module load (keras imported inside the builder) so the module
can be introspected in either container.
"""
from __future__ import annotations


def build_cnn(input_shape, n_filters: int = 32, dropout: float = 0.3):
    """A 4-block 2-D CNN detector. ``input_shape`` = (F, T, 1). Returns a keras.Model.

    Two logits (spoof, bona fide); use softmax[:, 1] as the detection score.
    """
    import keras
    from keras import layers

    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        return x

    inp = keras.Input(shape=input_shape, name="feat")
    x = block(inp, n_filters)
    x = block(x, n_filters * 2)
    x = block(x, n_filters * 4)
    x = block(x, n_filters * 4)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(2, activation="softmax", name="cm")(x)
    return keras.Model(inp, out, name="baseline_cnn")


def build_rawnet2(*args, **kwargs):
    """Pure-Keras RawNet2 baseline (B2, SincNet -> res-blocks -> GRU). TODO: implement."""
    raise NotImplementedError("RawNet2 baseline lands in the detection-baseline pass.")
