"""Baseline TTS (A1) — Tacotron2-style seq2seq built from scratch in Keras/TensorFlow.

Course anchors: ML4.pptx (RNN/LSTM), attention.pptx (attention), exs/ex5 (seq model).
The deliberately weak baseline whose failure modes (attention collapse, exposure bias,
Griffin-Lim artifacts) we analyse before moving to XTTS.

Modules:
    audio_tts   invertible log-mel + Griffin-Lim vocoder
    text        char-level vocab / encode-decode
    dataset     (text, mel) pairs from an LJSpeech-style folder
    model       encoder + Bahdanau-attention LSTM decoder + postnet (teacher-forced)
    train       training CLI (+ loss curve)
    synthesize  autoregressive inference -> wav
"""
