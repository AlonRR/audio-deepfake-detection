"""Main cloner (A2) — fine-tune pretrained Coqui XTTS-v2 (PyTorch container).

Course anchors: attention.pptx (GPT-style Transformer decoder), word embedding.pptx
(speaker embedding), L18_gan (HiFi-GAN vocoder). Reference: XTTS, arXiv:2406.04904.

Modules:
    prepare     segment + Whisper-transcribe a recording -> LJSpeech-style dataset
    finetune    Coqui XTTS GPT fine-tuning recipe on the target speaker
    synthesize  zero-shot and fine-tuned synthesis of the demo prompts
"""
