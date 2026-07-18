"""Fine-tune Coqui XTTS-v2 on the target speaker (PyTorch container, GPU).

A wrapper around Coqui's official XTTS GPT fine-tuning recipe (train_gpt_xtts.py). Downloads
the base XTTS-v2 checkpoint + DVAE + tokenizer on first run, then fine-tunes the GPT on the
LJSpeech-style dataset produced by prepare.py.

    python -m src.creation.xtts_finetune.finetune --data-root data/raw --out models/xtts_ft --epochs 10

NOTE: Coqui internals shift between versions — pin coqui-tts and validate on the server;
expect to adjust config field names on the first run.
"""
from __future__ import annotations

import argparse
import os


def run(data_root: str, out_dir: str, epochs: int, batch_size: int, lr: float) -> None:
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs, GPTTrainer, GPTTrainerConfig)
    # coqui-tts 0.27.x defines XttsAudioConfig in TTS.tts.models.xtts; older
    # versions re-exported it from gpt_trainer, which is where this used to import it.
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.utils.manage import ModelManager

    os.makedirs(out_dir, exist_ok=True)

    # --- base XTTS-v2 assets (downloaded once) ---
    base = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main"
    files = {
        "dvae": f"{base}/dvae.pth",
        "mel": f"{base}/mel_stats.pth",
        "tok": f"{base}/vocab.json",
        "ckpt": f"{base}/model.pth",
        "cfg": f"{base}/config.json",
    }
    ckpt_dir = os.path.join(out_dir, "xtts_base")
    os.makedirs(ckpt_dir, exist_ok=True)
    local = {}
    for k, url in files.items():
        dst = os.path.join(ckpt_dir, os.path.basename(url))
        if not os.path.isfile(dst):
            ModelManager._download_model_files([url], ckpt_dir, progress_bar=True)
        local[k] = dst

    dataset = BaseDatasetConfig(formatter="ljspeech", dataset_name="target",
                                path=data_root, meta_file_train="metadata.csv", language="en")

    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        max_wav_length=255995, max_text_length=200,
        mel_norm_file=local["mel"], dvae_checkpoint=local["dvae"],
        xtts_checkpoint=local["ckpt"], tokenizer_file=local["tok"],
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True)

    config = GPTTrainerConfig(
        epochs=epochs, output_path=out_dir, model_args=model_args,
        run_name="xtts_ft", batch_size=batch_size, batch_group_size=48,
        eval_batch_size=batch_size, num_loader_workers=4, lr=lr,
        optimizer="AdamW", optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        save_step=1000, save_n_checkpoints=1, save_checkpoints=True, print_step=50)

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples([dataset], eval_split=True, eval_split_size=0.1)

    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False),
        config, output_path=out_dir, model=model,
        train_samples=train_samples, eval_samples=eval_samples)
    trainer.fit()
    print(f"fine-tuning done -> {out_dir} (best checkpoint under the run subfolder)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune XTTS-v2 on the target speaker.")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="models/xtts_ft")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()
    run(args.data_root, args.out, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
