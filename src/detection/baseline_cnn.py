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


def build_rawnet2(n_samples: int = 64_000, sr: int = 16_000, sinc_channels: int = 128):
    """Pure-Keras RawNet2 (B2): SincConv front-end -> residual blocks (FMS) -> GRU -> FC.

    Raw waveform input (n_samples, 1) -> 2-way softmax. Port of Tak et al. 2021
    (ICASSP), the official ASVspoof end-to-end baseline. TF-based (runs in the TF
    container). Input is a fixed-length raw waveform (~4 s at 16 kHz).
    """
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import layers

    class SincConv(layers.Layer):
        """Parametrized sinc band-pass filterbank (SincNet), mel-initialized."""

        def __init__(self, out_ch=128, kernel_size=251, sample_rate=16_000,
                     min_low_hz=50.0, min_band_hz=50.0, **kw):
            super().__init__(**kw)
            self.out_ch = out_ch
            self.kernel_size = kernel_size + (1 - kernel_size % 2)  # force odd
            self.sr = sample_rate
            self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz

        def _hz2mel(self, hz):
            return 2595 * np.log10(1 + hz / 700)

        def _mel2hz(self, mel):
            return 700 * (10 ** (mel / 2595) - 1)

        def build(self, _):
            low_hz, high_hz = 30.0, self.sr / 2 - (self.min_low_hz + self.min_band_hz)
            hz = self._mel2hz(np.linspace(self._hz2mel(low_hz), self._hz2mel(high_hz), self.out_ch + 1))
            self.low_hz_ = self.add_weight(
                shape=(self.out_ch, 1), name="low_hz",
                initializer=keras.initializers.Constant(hz[:-1].reshape(-1, 1)))
            self.band_hz_ = self.add_weight(
                shape=(self.out_ch, 1), name="band_hz",
                initializer=keras.initializers.Constant(np.diff(hz).reshape(-1, 1)))
            n = (self.kernel_size - 1) / 2.0
            self.n_ = tf.constant((2 * np.pi * np.arange(-n, 0) / self.sr).reshape(1, -1), tf.float32)
            win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(int(n)) / self.kernel_size)
            self.window_ = tf.constant(win.reshape(1, -1), tf.float32)

        def call(self, x):  # x: (B, T, 1)
            low = self.min_low_hz + tf.abs(self.low_hz_)
            high = tf.clip_by_value(low + self.min_band_hz + tf.abs(self.band_hz_),
                                    self.min_low_hz, self.sr / 2)
            band = (high - low)[:, 0]
            ftl, fth = tf.matmul(low, self.n_), tf.matmul(high, self.n_)
            left = ((tf.sin(fth) - tf.sin(ftl)) / (self.n_ / 2)) * self.window_
            center = 2 * tf.reshape(band, (-1, 1))
            bp = tf.concat([left, center, tf.reverse(left, [1])], axis=1)
            bp = bp / (2 * band[:, None])
            filt = tf.transpose(bp)[:, None, :]  # (k, 1, out_ch)
            return tf.nn.conv1d(x, filt, stride=1, padding="SAME")

    def res_fms(x, ch):
        y = layers.BatchNormalization()(x)
        y = layers.LeakyReLU(0.3)(y)
        y = layers.Conv1D(ch, 3, padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(0.3)(y)
        y = layers.Conv1D(ch, 3, padding="same")(y)
        if x.shape[-1] != ch:
            x = layers.Conv1D(ch, 1, padding="same")(x)
        y = layers.add([x, y])
        y = layers.MaxPooling1D(3)(y)
        s = layers.GlobalAveragePooling1D()(y)          # filter-wise feature-map scaling
        s = layers.Dense(ch, activation="sigmoid")(s)
        return layers.multiply([y, layers.Reshape((1, ch))(s)])

    inp = keras.Input(shape=(n_samples, 1), name="waveform")
    x = SincConv(sinc_channels, sample_rate=sr)(inp)
    x = layers.MaxPooling1D(3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.3)(x)
    for ch in (128, 128, 256, 256):
        x = res_fms(x, ch)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.3)(x)
    x = layers.GRU(256)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(2, activation="softmax", name="cm")(x)
    return keras.Model(inp, out, name="rawnet2")
