"""Baseline TTS (A1) — Tacotron2-style seq2seq built in Keras/TensorFlow.

Course anchors: ML4.pptx (RNN/LSTM), attention.pptx (location-sensitive attention),
exs/ex5 (Embedding -> GRU -> Dense) as the starting scaffold.

TODO (implementation pass):
    - text/phoneme encoder (Embedding -> BiLSTM)
    - location-sensitive attention
    - autoregressive LSTM mel-decoder + stop-token
    - Griffin-Lim (baseline) / small neural vocoder -> waveform
    - train() on the 1-5 min recording; log the learning curve; document failure modes
      (attention collapse, exposure bias, Griffin-Lim artifacts).
"""
