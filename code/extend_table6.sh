#!/usr/bin/env bash
#
# Fill in the two missing cells of Table 6 (WN18RR encoder-depth x
# decoder-choice interaction):
#
#   - L=0 DistMult on WN18RR (the cell whose absence reviewers will flag)
#   - L=1 ComplEx  on WN18RR (the symmetric counterpart)
#
# 3 seeds each (matches the rest of Table 6) -> 6 runs total.
# Total runtime on a single 2080 Ti: ~30-40 minutes
# (WN18RR L=0/1 d=200 ~ 5-6 min/seed).
#
# Default WORKDIR: /home/shanxh/decoder
# Override: WORKDIR=/some/other/path bash extend_table6.sh
#
# Recommended: launch with nohup so it survives logout.
#
#   cd /home/shanxh/decoder
#   nohup bash extend_table6.sh > extend_table6.log 2>&1 &
#   echo "PID: $!"
#   tail -f extend_table6.log
#
# After the runs finish, scp the 6 new JSONs back to your local
#   kgc-decoder-audit/data/decoder_diag_jsons/
# and re-run code/aggregate_runs.py (the new cells go straight into
# Table 6 once we update main.tex with the means).

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

# --- Three seeds matching the rest of Table 6 ---
SEEDS=(42 123 456)

# --- The two missing (decoder, L) cells ---
# (decoder, layers) tuples
CELLS=(
  "DistMult 0"
  "ComplEx  1"
)

# --- WN18RR config (matches existing Table 6 cells) ---
DATASET=WN18RR
D_H=200
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
    exit 1
  fi
fi
echo "SCRIPT:  $SCRIPT"
echo "PYTHON:  $PYTHON"

echo "================================================================="
echo "Filling Table 6 missing cells on WN18RR ($DATASET)"
echo "  cells:  L=0 DistMult, L=1 ComplEx"
echo "  seeds:  ${SEEDS[*]}"
echo "  config: d_h=$D_H lr=$LR epochs=$EPOCHS bs=$BATCH ls=$LS"
echo "  expected total runtime: ~30-40 min on a single 2080 Ti"
echo "================================================================="

START=$(date +%s)

for seed in "${SEEDS[@]}"; do
  for cell in "${CELLS[@]}"; do
    read -r scorer layers <<< "$cell"
    echo
    echo "--- $DATASET / $scorer / L=$layers / seed=$seed ---"
    "$PYTHON" "$SCRIPT" \
      --dataset "$DATASET" \
      --seed "$seed" \
      --d_h "$D_H" \
      --layers "$layers" \
      --lr "$LR" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH" \
      --label_smooth "$LS" \
      --scorer "$scorer" \
      --val_every 5
  done
done

END=$(date +%s)
echo
echo "================================================================="
echo "All 6 runs done in $((END-START)) seconds."
echo "New JSONs (in current directory):"
for seed in "${SEEDS[@]}"; do
  for cell in "${CELLS[@]}"; do
    read -r scorer layers <<< "$cell"
    stag=""
    [ "$scorer" != "DistMult" ] && stag="_$scorer"
    echo "  ${DATASET}_d${D_H}_L${layers}_ls${LS}${stag}_seed${seed}.json"
  done
done

echo
echo "Pack them up for scp:"
echo "  tar -czvf table6_results.tar.gz \\"
echo "    WN18RR_d200_L0_ls0.3_seed*.json \\"
echo "    WN18RR_d200_L1_ls0.3_ComplEx_seed*.json \\"
echo "    extend_table6.log"
echo "================================================================="
