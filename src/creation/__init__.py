"""Part A — Creation (voice cloning).

Two systems, on purpose:
    keras_tts/      A weak Tacotron2-style TTS built from scratch in Keras (baseline
                    that is expected to fail, motivating the pretrained approach).
    xtts_finetune/  Fine-tune pretrained Coqui XTTS-v2 on the target speaker (main).
"""
