#!/bin/bash
set -e
PID=$(cat "$HOME/data/la_download.pid")
echo "[autorun] waiting for download pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
sz=$(stat -c %s "$HOME/data/LA.zip" 2>/dev/null || echo 0)
echo "[autorun] download ended; LA.zip=$sz bytes"
if ! unzip -l "$HOME/data/LA.zip" >/dev/null 2>&1; then
  echo "[autorun] ZIP INVALID — last log:"; tail -5 "$HOME/data/la_download.log"; exit 1
fi
if [ ! -d "$HOME/data/LA/ASVspoof2019_LA_train/flac" ]; then
  echo "[autorun] extracting..."
  cd "$HOME/data" && unzip -q -o LA.zip
fi
echo "[autorun] contents:"; ls "$HOME/data/LA"
cd "$HOME/adf" && JID=$(sbatch --parsable scripts/base_cnn.slurm)
echo "[autorun] submitted base_cnn as job $JID"
echo "$JID" > "$HOME/adf/.base.jid"
