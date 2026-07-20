#!/usr/bin/env bash
# CRADLE v1 confirmation, amended protocol: seeds 2R/4/5, phase cap 150k.
# Pre-reg: COMPOSITE_CHAIN_AMENDMENT_PREREG_2026-07-10.md. Seeds 1,3 carry forward from comp4_s{1,3}.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
TASKS=minigrid_DoorKey-5x5,minigrid_SimpleCrossingS9N1,minigrid_LavaGapS5,minigrid_MultiRoom-N2-S4
for s in 2 4 5; do
  python -u orchestrator_chain_nm512.py --tasks $TASKS --device cuda --composite \
    --eval_every 2000 --phase_max_steps 150000 \
    --logdir ~/dv3_logs/comp4b_s$s --seed $s > ~/dv3_logs/comp4b_s$s.log 2>&1
done
echo '=== COMPOSITE SWEEP 2 DONE ==='
for s in 2 4 5; do
  [ -f ~/dv3_logs/comp4b_s$s/chain_summary.json ] && python3 -c "import json;x=json.load(open('~/dv3_logs/comp4b_s$s/chain_summary.json'));print('comp4b_s$s:','ROUTED-ALL' if x['success_all_tasks_routed'] else 'FAIL', x['final_retention_routed'], '| router', round(x['router_accuracy_final'],3))"
done
