"""Trainable detection back-end (Keras/TensorFlow) — the student's own model.

Consumes cached SSL features from features_ssl.py and outputs a bona-fide/spoof score.
Course anchors: attention.pptx (attentive pooling / graph attention), ML3.pptx (CNN).

TODO (implementation pass):
    - build_backend(input_dim, pooling="attentive"|"mean", ...) -> keras.Model
        SSL frame features -> (optional light AASIST-style graph module) ->
        attentive statistics pooling -> Dense -> 2-way / score
    - a config surface for the hyper-parameter search (LR, layer(s), width, dropout).
Runs in the TensorFlow GPU container (tensorflow-25.02.sif).
"""


def build_backend(input_dim, pooling="attentive"):
    """Build the Keras detection back-end over frozen SSL features. TODO: implement."""
    raise NotImplementedError
