#!/usr/bin/env bash
# 8-task tunnel chain, n=3. Pre-reg: TUNNEL_CHAIN8_PREREG_2026-07-15.md (+ AMENDMENT 1).
# Roster locked by the amended rule (first four passers by completion order), saved in
# ~/dv3_logs/ROSTER.txt at launch time. Single new variable vs tunl4b: chain length 8.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
LOGS=~/dv3_logs
ROSTER=$(cat $LOGS/ROSTER.txt)
echo "[tunl8] roster: $ROSTER" >> $LOGS/CHAIN8_STATUS.txt
for s in 1 2 3; do
  python -u orchestrator_chain_nm512.py --tasks $ROSTER --device cuda \
    --tunnel_rehearsal --cont_grading --rehearsal_updates 50 \
    --eval_every 2000 --phase_max_steps 150000 \
    --logdir $LOGS/tunl8_s$s --seed $s > $LOGS/tunl8_s$s.log 2>&1
done
echo '=== TUNL8 SWEEP DONE ===' >> $LOGS/CHAIN8_STATUS.txt
for s in 1 2 3; do
  [ -f $LOGS/tunl8_s$s/chain_summary.json ] && python3 -c "import json;x=json.load(open('$LOGS/tunl8_s$s/chain_summary.json'));print('tunl8_s$s:','PASS' if x['success_all_tasks'] else 'FAIL', {k.replace('minigrid_','')[:12]: round(v,2) for k,v in x['final_retention'].items()})" >> $LOGS/CHAIN8_STATUS.txt
done
