#!/usr/bin/env bash
# Frozen-head-A actor-protection arm (pre-reg STAGE_AB_FROZENHEAD_PREREG_2026-07-01).
# 3 seeds, isolated (replay_fraction=0) so actor protection is unconfounded by replay.
# joint baseline = existing ab_fl_s{1,2,3} (f=0, no freeze) -- not re-run here.
# Launch (Spark): sed -i 's/\r$//' frozen_head_sweep.sh; nohup bash frozen_head_sweep.sh > ~/dv3_logs/frozen_head_sweep.log 2>&1 &
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
COMMON="--expl_behavior greedy --eval_every 2000 --b_max_steps 100000 --device cuda"
run () { python -u orchestrator_ab_nm512.py $COMMON "$@"; }

for s in 1 2 3; do
  run --task_a minigrid_DoorKey-5x5 --task_b minigrid_SimpleCrossingS9N1 \
      --actor_mode frozen_a_head --replay_fraction 0 \
      --logdir ~/dv3_logs/ab_frz_s$s --seed $s > ~/dv3_logs/ab_frz_s$s.log 2>&1
done

echo "=== FROZEN-HEAD ARM DONE ==="
for s in 1 2 3; do
  d=~/dv3_logs/ab_frz_s$s
  [ -f $d/ab_summary.json ] && python3 -c "import json;x=json.load(open('$d/ab_summary.json'));print('ab_frz_s$s: A_after',round(x['A_after'],3),'B_peak',round(x['B_peak'],2),'BOTH',x['success_both_bars'])"
done
