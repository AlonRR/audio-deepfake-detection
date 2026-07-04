# Detection runs — index for the report

Each `<run>/` holds `result.json` (metrics), `history.csv` (learning-curve data), and
`*_scores.npz` (raw scores+labels for DET curves). Exception: deliberately-failed runs
(`*_FAIL`) hold `history.csv` only — they never reached scoring, so no result.json or
scores exist by design. Figures live in `../figures/` (prefixed by run). Regenerate with
`scripts/plot_detection.py`; refresh from the lab with `scripts/fetch_results.sh`.
Full timeline + wall-times + failures: `docs/runtimes.md`.

## Final results (the comparison table)
| Run | Model / front-end | Set | EER | min t-DCF |
|-----|-------------------|-----|----:|----------:|
| `base_cnn_logmel_full` | CNN · log-mel | dev | 18.99% | 0.4505 |
| `base_cnn_lfcc_full` | CNN · LFCC | dev | 12.99% | 0.2817 |
| `base_cnn_lfcc_eval` | CNN · LFCC | eval | 18.55% | 0.3835 |
| `base_rawnet2_full` | RawNet2 (raw) | dev | collapsed (≈chance) | — |
| `ssl_xlsr_run1` | **SSL XLS-R + Keras** | dev | **0.04%** | 0.0004 |
| `ssl_xlsr_eval` | **SSL XLS-R + Keras** | eval | **0.668%** | 0.0189 |

## Hyper-parameter search (SSL back-end LR sweep — jobs 267/268)
`ssl_hp_lr1e4` `ssl_hp_lr3e4` `ssl_hp_lr1e3` `ssl_hp_lr3e3` `ssl_hp_lr1e2` — one dir per
LR config (proj=256, dropout=0.3). Best: **lr=1e-4 / 3e-4 tie at 0.033% dev EER**.
Figure: `../figures/ssl_lr_sweep.png`; analysis: `docs/results.md` B.4(5).

## "Show the process" runs (kept deliberately)
| Run | What it shows |
|-----|---------------|
| `smoke_cnn_lfcc`, `smoke_cnn_logmel`, `smoke_rawnet2` | First end-to-end pipeline validation on tiny **synthetic** data — all three model types train + emit EER/DET before touching real data. (The *first* smoke attempt failed on a numpy ABI clash from `pip --target` shadowing the container numpy — see `docs/runtimes.md` job 239; fixed via `--user` + pinned numpy.) |
| `base_cnn_lfcc_sub` | The **4k biased subset → 0.0% EER artifact.** The first-2000 spoof are one attack type, trivially separable, so a tiny CNN "wins" (train acc ~1.0, val ~0.76 — overfit). This misleading result is *why* we moved to full-data + eval-set evaluation. A textbook "don't trust an easy subset" lesson. |
| `ssl_hp_lr1e1_FAIL` | The **deliberate failed run** from the hp-sweep: at lr=1e-1 training diverges immediately (loss 9.11 at epoch 0, val-accuracy flipping between the class poles) and the job hit the 2 h cap mid-epoch-3 — `history.csv` only, no EER was ever scored. |

Other documented failures behind the results (in `docs/runtimes.md`): log-mel & SSL-eval
**OOM** on the 7 GB default memory, and the home **~50 GB quota** exceeded mid eval-extraction.
