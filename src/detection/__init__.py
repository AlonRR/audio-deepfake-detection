"""Part B — Detection (anti-spoofing / synthetic-speech recognition).

    baseline_cnn.py   First try: mel/LFCC-spectrogram CNN in Keras (like exs/ex4).
    features_ssl.py   Frozen wav2vec2-XLS-R / WavLM feature extraction (PyTorch).
    backend_keras.py  The trainable back-end built in Keras (main system).
    train.py          Training loop + hyper-parameter search + learning curves.
    evaluate.py       EER / min t-DCF / DET curve; the cross-generator test.

Main recipe (B3): frozen SSL frontend -> cached features -> Keras back-end.
Reference: Tak et al. 2022, arXiv:2202.12233.
"""
