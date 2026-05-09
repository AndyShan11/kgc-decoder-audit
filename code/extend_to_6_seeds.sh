#!/usr/bin/env bash
#
# Extend the UMLS / Kinship 36-run lock-in pass from 3 seeds to 6 seeds.
# Designed to run on a server where you have data/UMLS/, data/Kinship/, and
# server_wn18rr.py (or train.py — auto-detected) inside one directory.
#
# Default working directory: /home/shanxh/decoder
# Override with: WORKDIR=/some/other/path bash extend_to_6_seeds.sh ...
#
# Total runtime on a single 2080 Ti: about 5-10 minutes.
# (UMLS L=2 d=200 ~ 30 s/seed; Kinship L=2 d=200 ~ 45 s/seed; 12 runs total.)
#
# Recommended: launch in the background with nohup so the run survives logout.
#
#   # If the script lives in WORKDIR:
#   cd /home/shanxh/decoder
#   nohup bash extend_to_6_seeds.sh > extend_to_6_seeds.log 2>&1 &
#   echo "PID: $!"     # save the pid; tail -f extend_to_6_seeds.log to watch
#
#   # Or one-liner from any directory:
#   nohup bash /home/shanxh/decoder/extend_to_6_seeds.sh \
#       > /home/shanxh/decoder/extend_to_6_seeds.log 2>&1 &
#
# After the runs finish, scp the 12 new JSONs back to your local
#   kgc-decoder-audit/data/decoder_diag_jsons/
# and re-run code/aggregate_runs.py to refresh data/rerun_summary.json.

set -euo pipefail

# --- Working directory ---
WORKDIR=${WORKDIR:-/home/shanxh/decoder}
if [ ! -d "$WORKDIR" ]; then
  echo "ERROR: WORKDIR=$WORKDIR does not exist."
  echo "       Set WORKDIR=/path/to/your/project before running."
  exit 1
fi
cd "$WORKDIR"
echo "WORKDIR: $(pwd)"

# --- Three seeds in addition to the existing {42, 123, 456} lock-in. ---
NEW_SEEDS=(7 99 2024)

# Datasets and decoders to extend. UMLS reversal is the priority; Kinship is
# the validation data point in the opposite direction.
DATASETS=(UMLS Kinship)
SCORERS=(DistMult ComplEx)

# Lock-in configuration (matches the 3-seed runs already in
# data/decoder_diag_jsons/).
D_H=200
LAYERS=2
LR=5e-4
EPOCHS=300
BATCH=4096
LS=0.3

# --- Auto-detect training script ---
PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-}
if [ -z "$SCRIPT" ]; then
  if   [ -f "server_wn18rr.py" ]; then SCRIPT=server_wn18rr.py
  elif [ -f "train.py" ];          then SCRIPT=train.py
  else
    echo "ERROR: neither server_wn18rr.py nor train.py found in $WORKDIR."
    echo "       Set SCRIPT=path/to/your/training_script.py before running."
    exit 1
  fi
fi
echo "SCRIPT:  $SCRIPT"
echo "PYTHON:  $PYTHON  (set PYTHON=/path/to/python to override)"

echo "================================================================="
echo "Extending UMLS + Kinship to 6 seeds (12 new runs)"
echo "  new seeds: ${NEW_SEEDS[*]}"
echo "  config:    d_h=$D_H L=$LAYERS lr=$LR epochs=$EPOCHS bs=$BATCH ls=$LS"
echo "  expected:  ~10 min total on a single 2080 Ti"
echo "================================================================="

START=$(date +%s)

for seed in "${NEW_SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for scorer in "${SCORERS[@]}"; do
      echo
      echo "--- $ds / $scorer / seed=$seed ---"
      "$PYTHON" "$SCRIPT" \
        --dataset "$ds" \
        --seed "$seed" \
        --d_h "$D_H" \
        --layers "$LAYERS" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --label_smooth "$LS" \
        --scorer "$scorer" \
        --val_every 5
    done
  done
done

END=$(date +%s)
echo
echo "================================================================="
echo "All 12 runs done in $((END-START)) seconds."
echo "New JSONs (in current directory):"
for seed in "${NEW_SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for scorer in "${SCORERS[@]}"; do
      stag=""
      [ "$scorer" != "DistMult" ] && stag="_$scorer"
      echo "  ${ds}_d${D_H}_L${LAYERS}_ls${LS}${stag}_seed${seed}.json"
    done
  done
done
echo
echo "Next step: scp these 12 JSONs to your local"
echo "  kgc-decoder-audit/data/decoder_diag_jsons/"
echo "and run:"
echo "  cd kgc-decoder-audit/code && python aggregate_runs.py"
echo "to refresh data/rerun_summary.json with 6-seed numbers."
echo "================================================================="
