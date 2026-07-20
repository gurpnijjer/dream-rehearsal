#!/usr/bin/env bash
# Task auditions (SCALEUP_LOCAL_DESIGN stage 1): can each candidate learn SOLO within 100k
# steps under greedy DreamerV3? Acquisition-feasibility gates only — no retention claims.
# Single-task chain = solo run with the all-good early stop. Deployed orchestrator (v2.2/lex).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
CANDS="minigrid_LavaCrossingS9N1 minigrid_MultiRoom-N4-S5 minigrid_SimpleCrossingS11N5 minigrid_FourRooms minigrid_DistShift2 minigrid_DoorKey-6x6"
for t in $CANDS; do
  echo "=== audition: $t ==="
  python -u orchestrator_chain_nm512.py --tasks $t --device cuda \
    --eval_every 2000 --phase_max_steps 100000 \
    --logdir ~/dv3_logs/audit_${t#minigrid_} --seed 7 > ~/dv3_logs/audit_${t#minigrid_}.log 2>&1
done
echo '=== AUDITIONS DONE ==='
for t in $CANDS; do
  d=~/dv3_logs/audit_${t#minigrid_}
  [ -f $d/chain_summary.json ] && python3 -c "
import json; x = json.load(open('$d/chain_summary.json'))
p = x['phases'][0]
print('${t#minigrid_}:', 'LEARNS' if p['all_good'] else 'FAILS-TO-BOOTSTRAP', '| steps', p['steps_used'], '| final', round(list(x['final_retention'].values())[0], 3))"
done
