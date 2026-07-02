# Run-time Log

Wall-clock timings for each stage, on the **Shenkar Virtual Lab** (NVIDIA **L4** 24 GB;
Slurm node `slurm-gpu`, 4× L4). GPU-job times come from `sacct -X` (authoritative).
Non-job stages (downloads, unzips, transfers) are wall-clock and marked `~approx`
where not separately instrumented.

**Environment:** jobs run via `sbatch --gres=gpu:1` inside apptainer
(`/opt/containers/tensorflow-25.02.sif`). Audio deps (librosa/soundfile/matplotlib)
installed once into `~/adf/.pyuser` — see `scripts/lab_env.sh`. QOS cap = **2 h/job**,
**1 GPU concurrent** (`bsc`; jobs serialize — chain with `--dependency=afterok`).
**Default job memory = 7 GB** (`cpu=1`); heavier jobs need `--mem` (log-mel ≥ 16 GB,
SSL back-end + eval extraction 32 GB — the XLS-R extractor leaks ~7 GB per ~27k utts).
**Home has a ~50 GB quota**, so delete zips/feature-caches after use (eval SSL features
are ~21 GB — deleted after scoring; train/dev caches deleted, re-extract if retraining).

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
| 246 | CNN-log-mel full (try 1) | 25 380 / 24 844 | 30 | 27:19 | ❌ OUT_OF_MEMORY — default 7 GB too small (peak 7.16 GB) |
| 247 | SSL prep (transformers + XLS-R cache) | — | — | 1:46 | PyTorch container; transformers 5.3.0, hidden_size 1024 |
| 248 | SSL XLS-R extract (train) | 25 380 | — | **17:08** | frozen XLS-R-300M → (200,1024) f16; 8.1 GB cache |
| 249 | SSL XLS-R extract (dev) | 24 844 | — | **15:12** | 8.0 GB cache |
| 250 | **SSL back-end (Keras)** | 25 380 / 24 844 | 40 | **22:37** | **Dev EER 0.04% · min t-DCF 0.0004** |
| 251 | CNN-log-mel full (retry, 16 GB) | 25 380 / 24 844 | 30 | **19:59** | Dev EER 18.99% · min t-DCF 0.4505 |
| 252 | SSL eval extract (try 1, 7 GB) | 71 237 | — | 20:12 | ❌ RAM OOM at ~27.5k, then home-quota exceeded |
| 261 | SSL eval extract (32 GB, resumed) | 71 237 | — | 31:44 | frozen XLS-R → (200,1024) f16 (deleted after scoring) |
| 262 | **SSL eval score** | 71 237 | — | **4:35** | **Eval EER 0.668% · min t-DCF 0.0189** |
| 263 | **CNN-LFCC eval score** | 71 237 | — | **9:34** | **Eval EER 18.55% · min t-DCF 0.3835** |
| 264 | RawNet2 full (from scratch) | 25 380 / 24 844 | 25 (cap) | 1:37 (cancelled) | ❌ **collapsed** — val acc pinned at 0.897 (majority class), no convergence; documented failed baseline |
| 265 | SSL XLS-R re-extract (train) | 25 380 | — | _running_ | features re-extracted for the hp-search (freed earlier for quota) |
| 266 | SSL XLS-R re-extract (dev) | 24 844 | — | _pending_ | `afterok:265` |
| 267 | SSL back-end hp-sweep | 25 380 / 24 844 | 25 × 8 cfg | _pending_ | `afterok:266`; LR sweep {1e-4…1e-1} + proj/dropout; lr=1e-1 = deliberate failure |

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

| Detector (full LA) | Dev EER | Eval EER | Eval min t-DCF | Note |
|-------|:-------:|:-------:|:-------:|------|
| CNN · log-mel | 18.99% | — | — | weakest hand-crafted front-end |
| CNN · LFCC | 12.99% | **18.55%** | 0.3835 | degrades on unseen attacks (poor generalization) |
| **SSL XLS-R + Keras back-end** | **0.04%** | **0.668%** | 0.0189 | barely degrades — SOTA-competitive generalization |

Dev shares its 6 attack types (A01–A06) with train, so dev EER is optimistic; **eval**
(A07–A19, *unseen*) is the real test. The hand-crafted CNN degrades (12.99 → 18.55 %),
while the SSL frontend holds (0.04 → 0.67 %) — **~28× lower eval EER**. This gap is the
core detection result and the "show the process" arc.

## How timings are recorded
- **GPU jobs:** pull with `sacct -X -j <id> --format=JobName,Elapsed,State,Start,End`
  and append a row here.
- **Long jobs** (e.g. XTTS fine-tune, > 2 h): checkpoint + chain across the 2 h QOS cap;
  log each segment separately.
- **Data/transfer stages:** wrap with explicit start/end capture and record here.
