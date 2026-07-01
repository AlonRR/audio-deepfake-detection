# Run-time Log

Wall-clock timings for each stage, on the **Shenkar Virtual Lab** (NVIDIA **L4** 24 GB;
Slurm node `slurm-gpu`, 4× L4). GPU-job times come from `sacct -X` (authoritative).
Non-job stages (downloads, unzips, transfers) are wall-clock and marked `~approx`
where not separately instrumented.

**Environment:** jobs run via `sbatch --gres=gpu:1` inside apptainer
(`/opt/containers/tensorflow-25.02.sif`). Audio deps (librosa/soundfile/matplotlib)
installed once into `~/adf/.pyuser` — see `scripts/lab_env.sh`. QOS cap = **2 h/job**
(`bsc`); default account has only `bsc` + `debug` (30 m).

## Detection — GPU jobs (2026-07-01)

| Job | Stage | Data scale | Epochs | Wall time | Result / note |
|----:|-------|-----------|:------:|:---------:|---------------|
| 238 | Env + GPU validation | — | — | **0:25** | apptainer + TF 2.17 + L4 + 6/6 metric tests |
| 239 | Smoke train v1 | 40 synth | 2 | 2:22 | ❌ numpy ABI clash (pip `--target` shadowed numpy) |
| 240 | Container deps probe | — | — | 0:12 | numpy 1.26.4; librosa/soundfile/mpl missing |
| 241 | Smoke train v2 | 40 synth | 2 | **1:39** | ✅ CNN-LFCC + CNN-logmel + RawNet2 all run on GPU |
| 242 | Baseline (subset) | 4 k | 12 | 8:37 | EER 0.0% = biased-subset artifact; per-epoch re-extract |
| 243 | Full baseline (rejected) | — | — | — | QOS rejected 3 h `--time` (cap 2 h) |
| 244 | **Baseline (full)** | 25 380 / 24 844 | 30 | **17:14** | **Dev EER 12.99% · min t-DCF 0.2817** (feature cache on) |

Job 244 per-epoch (cache on): epoch 1 ≈ extract+cache ~50 k utterances; epochs 2–30 ≈
~25 s each on the L4. Without the cache the same run was projected at ~6 h.

## Data preparation

| Stage | Scale | Wall time | Note |
|-------|-------|:---------:|------|
| ASVspoof 2019 LA download | 7.64 GB | see below `~approx` | Edinburgh DataShare bitstream API; ran concurrently with smoke jobs |
| Unzip (train+dev ready) | ~50 k flac | `~approx` | NFS small-file I/O; eval (71 k) extracted afterward |
| Code push (`git archive` → ssh `tar -x`) | 31 files | < 5 s | tracked files only |

Download started **14:42:33**; train+dev were extracted and the first real subset job
ran by **15:31** — so *download + train/dev unzip ≤ ~49 min wall* (14:42→15:31),
overlapping other jobs. Exact download-complete time was not separately instrumented;
future large transfers will be timed explicitly.

## Key result progression (dev EER)

| Setup | Dev EER | Note |
|-------|:-------:|------|
| Subset 4 k (biased) | 0.0% | misleading — one attack type, trivially separable |
| **Full LA · CNN-LFCC baseline** | **12.99%** | first defensible baseline (min t-DCF 0.2817) |
| SSL (XLS-R) + Keras back-end | _todo_ | target: substantially lower (SSL SOTA < 1%) |

## How timings are recorded
- **GPU jobs:** pull with `sacct -X -j <id> --format=JobName,Elapsed,State,Start,End`
  and append a row here.
- **Long jobs** (e.g. XTTS fine-tune, > 2 h): checkpoint + chain across the 2 h QOS cap;
  log each segment separately.
- **Data/transfer stages:** wrap with explicit start/end capture and record here.
