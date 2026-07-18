# Goal: complete the Shenkar NN final project (audio deepfakes)

Repo: `~/code/audio-deepfake-detection` (synced to OneDrive via `ods`)
Server: `ssh shenkar` → `~/adf`, Slurm `bsc` QOS (1 GPU, MaxWall **2h**), L4 24 GB

## Objective

Finish both halves of the assignment end-to-end and leave the project in a state
where the only remaining work is (a) collecting human MOS ratings and (b) the
oral defense.

## Done when

1. **A2:** XTTS-v2 fine-tuned on `data/raw/metadata.csv`; loss curve saved.
2. **A2:** ≥10 clips of varying length/complexity synthesized, zero-shot **and**
   fine-tuned, in `data/generated/`.
3. **A1:** Keras Tacotron2-lite trained + synthesized; failure modes documented
   (this baseline is *meant* to underperform — that contrast is the deliverable).
4. Evaluation harness run: MCD + log-mel correlation, ECAPA speaker cosine, scored
   against the 4 held-out clips in `metadata_eval.csv` (never against train audio).
5. **MOS:** blind randomized rating sheet generated + aggregation script ready.
6. Cross-generator test: trained detector scored on my own A1/A2 fakes; the
   generalization gap reported against its ASVspoof operating point.
7. `docs/results.md` has every number filled or explicitly marked pending;
   `presentation/slides_outline.md` creation slides + examiner Q&A completed.
8. All committed (`type(scope)` messages, noreply email, no `Co-Authored-By`)
   and synced to OneDrive.

## Non-goals / hard limits

- **NEVER fabricate MOS scores or any experimental number.** MOS requires real
  listeners; build the harness, mark it pending, and stop there.
- Do not change hyperparameters to make results look better without recording
  the change and the before/after in `results.md`.
- Do not touch `.pyuser_pt` (the detection env). XTTS deps live in `.pyuser_xtts`;
  they conflict (coqui-tts needs `transformers<5`, XLS-R detection uses 5.x).
- Do not re-run the detection half; those results are final.

## Working rules

- Every failed run gets written up in `results.md`. The brief explicitly rewards
  *"showing the process"* — failures and their analysis are graded material, not
  noise to hide.
- Every method choice cites the course lecture it draws on (oral defensibility).
- Report at each milestone; don't go silent through long job queues.
