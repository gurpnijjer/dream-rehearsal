#!/usr/bin/env bash
# Chain test A->B->C->D, n=3 (CHAIN_TEST_DESIGN_2026-07-03, pre-reg frozen).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
TASKS=minigrid_DoorKey-5x5,minigrid_SimpleCrossingS9N1,minigrid_LavaGapS5,minigrid_MultiRoom-N2-S4
for s in 1 2 3; do
  python -u orchestrator_chain_nm512.py --tasks $TASKS --device cuda     --eval_every 2000 --phase_max_steps 100000     --logdir ~/dv3_logs/chain4_s$s --seed $s > ~/dv3_logs/chain4_s$s.log 2>&1
done
echo '=== CHAIN SWEEP DONE ==='
for s in 1 2 3; do
  [ -f ~/dv3_logs/chain4_s$s/chain_summary.json ] && python3 -c "import json;x=json.load(open('~/dv3_logs/chain4_s$s/chain_summary.json'));print('chain4_s$s:','ALL' if x['success_all_tasks'] else 'FAIL', x['final_retention'])"
done
