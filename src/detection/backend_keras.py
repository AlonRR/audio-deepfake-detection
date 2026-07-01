"""Trainable detection back-end (Keras/TensorFlow) over frozen SSL features.

Consumes cached (T, D) SSL frame features from features_ssl.py and outputs P(bona fide).
Architecture: frame projection (Conv1D) -> attentive statistics pooling -> classifier.
Course anchors: attention.pptx (attentive pooling), transfer learning over a frozen
pretrained representation (word embedding.pptx). Runs in the TensorFlow GPU container.
"""
from __future__ import annotations


def build_backend(input_dim: int, max_frames: int, proj: int = 256, dropout: float = 0.3):
    """SSL frame features (T, D) -> attentive-stats-pooled classifier. Returns keras.Model."""
    import keras
    from keras import layers, ops

    class AttentiveStatsPooling(layers.Layer):
        """Weighted mean + std over time (Okabe et al. 2018), a strong utterance pooler."""

        def build(self, input_shape):
            d = input_shape[-1]
            self.w1 = layers.Dense(d, activation="tanh")
            self.w2 = layers.Dense(d)  # per-channel attention logits

        def call(self, x):                       # x: (B, T, D)
            a = ops.softmax(self.w2(self.w1(x)), axis=1)   # (B, T, D)
            mean = ops.sum(a * x, axis=1)                  # (B, D)
            var = ops.sum(a * ops.square(x), axis=1) - ops.square(mean)
            std = ops.sqrt(ops.maximum(var, 1e-8))
            return ops.concatenate([mean, std], axis=-1)   # (B, 2D)

    inp = keras.Input(shape=(max_frames, input_dim), name="ssl_feat")
    x = layers.Masking()(inp)
    x = layers.Conv1D(proj, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(proj, 3, padding="same", activation="relu")(x)
    x = AttentiveStatsPooling()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(2, activation="softmax", name="cm")(x)
    return keras.Model(inp, out, name="ssl_backend")
