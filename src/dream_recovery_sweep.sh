#!/usr/bin/env bash
# Dream-recovery race: Arm A (rl) vs Arm B (tunnel) x 3 seeds.
# Pre-reg: DREAM_RECOVERY_PREREG_2026-07-10.md. Waits for the corrected re-read to finish first.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
while pgrep -f corrected_routed_reread.py > /dev/null; do sleep 60; done
for d in comp4_s1 comp4b_s2 comp4_s3; do
  for arm in rl tunnel; do
    echo "=== $d / $arm ==="
    OMP_NUM_THREADS=3 timeout 14400 python -u dream_recovery.py \
      --logdir ~/dv3_logs/$d --arm $arm > ~/dv3_logs/dream_${arm}_${d}.log 2>&1
  done
done
echo '=== DREAM RECOVERY SWEEP DONE ==='
for d in comp4_s1 comp4b_s2 comp4_s3; do
  for arm in rl tunnel; do
    f=~/dv3_logs/$d/dream_${arm}_result.json
    [ -f $f ] && python3 -c "import json;x=json.load(open('$f'));print('$d/$arm:','PASS@'+str(x['passed_at_update']) if x['primary_pass'] else 'FAIL', '| best', x['t1_best'], '| start', x['t1_start'])"
  done
done
