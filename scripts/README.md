# Server scripts — Shenkar Virtual Lab (verified working)

Slurm + apptainer harness that actually runs on the lab. See `docs/runtimes.md` for
measured wall-times and `docs/server_runbook.md` for the connection walkthrough.

## Lab realities these scripts encode
- **Access:** `ssh` to the submit node (Ubuntu VM). Only `$HOME` is writable — `/datasets`,
  `/projects`, `/models` are **read-only**. So code lives in `~/adf`, data in `~/data`,
  outputs in `~/adf/reports`, feature caches in `~/adf/features`.
- **GPU only via Slurm:** submit node has no GPU. `sbatch --gres=gpu:1` runs on node
  `slurm-gpu` (4× NVIDIA L4). `apptainer` exists only on that node.
- **QOS `bsc`:** **2 h max wall per job**, **1 GPU concurrently** (or 2 MPS shards). So
  jobs serialize — chain them with `--dependency=afterok:<jid>` rather than expecting
  parallelism. Long jobs (XTTS) must checkpoint + resume to cross the 2 h cap.
- **Containers:** `/opt/containers/{tensorflow-25.02,pytorch-25.04}.sif`. numpy is
  **1.26.4** in both. TF container is missing librosa/soundfile/matplotlib; PyTorch
  container has those but is missing `transformers`.
- **Deps without breaking the container numpy:** `pip install --user` into a
  `PYTHONUSERBASE` under `$HOME` with `-c constraints.txt` (pins `numpy==1.26.4`), then
  put that dir on `PYTHONPATH`. Codified in `lab_env.sh` (TF) / `lab_env_pt.sh` (PyTorch).
  Never `pip --target` ahead of the container's numpy (ABI clash).

## Scripts
| Script | Container | Purpose |
|--------|-----------|---------|
| `lab_env.sh` | TF | source to add librosa/soundfile/matplotlib for detection |
| `lab_env_pt.sh` | PyTorch | source to add `transformers` for SSL extraction |
| `make_smoke_data.py` | — | tiny synthetic ASVspoof-layout set to smoke-test the pipeline |
| `full_baseline.slurm` | TF | CNN-LFCC full baseline (EER + min t-DCF) |
| `logmel_full.slurm` | TF | CNN-log-mel full baseline (comparison row) |
| `prep_ssl.slurm` | PyTorch | install `transformers` + cache XLS-R weights once |
| `extract_ssl.slurm` | PyTorch | freeze XLS-R, cache per-utterance features (resumable; `SUBSET=train|dev`) |
| `backend_ssl.slurm` | TF | train the Keras back-end on cached SSL features |

## Reproduce (detection)
```bash
# code + data onto the server
git archive HEAD | ssh shenkar 'mkdir -p ~/adf && tar -x -C ~/adf'
# (ASVspoof 2019 LA -> ~/data/LA ; see server_runbook.md)

# baselines
sbatch scripts/full_baseline.slurm            # CNN-LFCC   -> reports/base_cnn_lfcc_full
sbatch scripts/logmel_full.slurm              # CNN-logmel -> reports/base_cnn_logmel_full

# SSL detector (chained; serialize on the single GPU)
P=$(sbatch --parsable scripts/prep_ssl.slurm)
T=$(sbatch --parsable --dependency=afterok:$P --export=ALL,SUBSET=train scripts/extract_ssl.slurm)
D=$(sbatch --parsable --dependency=afterok:$T --export=ALL,SUBSET=dev   scripts/extract_ssl.slurm)
sbatch      --dependency=afterok:$D scripts/backend_ssl.slurm  # -> reports/ssl_xlsr_run1
```
