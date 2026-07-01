"""First-try detection baseline (B1) — a plain spectrogram CNN in Keras.

Deliberately simple (in the style of exs/ex4: Conv -> BN -> ReLU -> Dropout stacks) so
it establishes the pipeline and a weak EER to improve on. Also hosts the pure-Keras
RawNet2 / LFCC-LCNN comparison baseline (B2) with published numbers.

TODO (implementation pass):
    - build_cnn(input_shape) -> keras.Model        # log-mel / LFCC -> CNN -> score
    - build_rawnet2() -> keras.Model               # SincNet -> res-blocks -> GRU (B2)
"""


def build_cnn(input_shape):
    """Small mel/LFCC-spectrogram CNN detector. TODO: implement."""
    raise NotImplementedError
