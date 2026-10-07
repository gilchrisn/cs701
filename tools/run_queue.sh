#!/usr/bin/env bash
# Runs the lines of $QUEUE one at a time ("<run name> <train.py arguments>"); waits for new lines; stops at a line "STOP".
# Every run directory gets provenance.txt: commit, dirty files, full command, start/end time.
QUEUE=${QUEUE:-$HOME/cs701-runs/queue.txt}
RUNS=${RUNS:-$HOME/cs701-runs}
cd "$(dirname "$0")/.."
touch "$QUEUE"
while true; do
  line=$(head -n 1 "$QUEUE")
  if [ -z "$line" ]; then sleep 30; continue; fi
  sed -i '1d' "$QUEUE"
  [ "$line" = "STOP" ] && break
  name=${line%% *}; args=${line#* }
  out="$RUNS/$name"; mkdir -p "$out"
  { echo "commit $(git rev-parse HEAD)"; echo "dirty: $(git status --porcelain | tr '\n' ' ')";
    echo "command: python train.py --data $HOME/cs701-data/holdout15_seed0 --out $out $args"; echo "start $(date -Is)"; } > "$out/provenance.txt"
  ~/cs701-venv/bin/python train.py --data "$HOME/cs701-data/holdout15_seed0" --out "$out" $args > "$out/stdout.txt" 2>&1
  echo "exit $? end $(date -Is)" >> "$out/provenance.txt"
  echo "$name" >> "$RUNS/done.txt"
done
