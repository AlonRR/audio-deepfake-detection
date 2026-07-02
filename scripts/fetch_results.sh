#!/bin/bash
# Pull vital run artifacts (metrics + raw scores + learning curves) from the lab into
# the repo, so they are version-controlled and synced to OneDrive (durable, not just on
# the server). Run this periodically after runs finish, then commit. Regenerate figures
# with scripts/plot_detection.py.
#
# Usage:  bash scripts/fetch_results.sh        (needs the `shenkar` ssh alias)
set -u
REMOTE="${REMOTE:-shenkar}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DET="$ROOT/reports/detection"
FIG="$ROOT/reports/figures"
mkdir -p "$FIG"

RUNS_DEV="base_cnn_lfcc_full base_cnn_logmel_full ssl_xlsr_run1 base_rawnet2_full"
RUNS_EVAL="base_cnn_lfcc_eval ssl_xlsr_eval base_rawnet2_eval"
# "show the process" runs: synthetic-data smoke tests + the biased 4k subset (0% EER artifact)
RUNS_PROCESS="base_cnn_lfcc_sub smoke_cnn_lfcc smoke_cnn_logmel smoke_rawnet2"

pull() {  # $1=run  $2=score file name
  local run="$1" score="$2"
  mkdir -p "$DET/$run"
  for f in result.json history.csv "$score"; do
    scp -q "$REMOTE:~/adf/reports/$run/$f" "$DET/$run/$f" 2>/dev/null && echo "  $run/$f" || true
  done
  for fig in learning_curves.png det_dev.png det_eval.png; do
    scp -q "$REMOTE:~/adf/reports/$run/$fig" "$FIG/${run}_${fig}" 2>/dev/null && echo "  fig ${run}_${fig}" || true
  done
}

echo "== dev runs =="; for r in $RUNS_DEV;  do pull "$r" dev_scores.npz;  done
echo "== eval runs =="; for r in $RUNS_EVAL; do pull "$r" eval_scores.npz; done
echo "== process runs =="; for r in $RUNS_PROCESS; do pull "$r" dev_scores.npz; done
# hp-sweep runs are dynamic (ssl_hp_<tag>) -> discover them on the server
echo "== hp-sweep runs =="
for r in $(ssh "$REMOTE" 'ls -d ~/adf/reports/ssl_hp_* 2>/dev/null | xargs -n1 basename' 2>/dev/null); do
  pull "$r" dev_scores.npz
done
echo
echo "Next: regenerate figures ->"
echo "  uv run --no-project --with numpy --with matplotlib --with scipy python scripts/plot_detection.py"
echo "Then commit (force-add the ignored types):"
echo "  git add reports/detection scripts/*.py scripts/fetch_results.sh"
echo "  git add -f reports/detection/**/*.npz reports/figures/*.png && git commit -m 'chore(reports): snapshot run artifacts + figures'"
