#!/usr/bin/env bash
# Tier-2 auditions (Gurp 2026-07-15): cheap-BUT-DISTINCT tasks in our proven fast-learning
# families — length-scaling roster fillers that keep task VARIETY high (the paper needs 8
# distinct tasks, not 8 flavors of one). Difficulty-scaling (the failing long-horizon tasks)
# waits for external compute. Runs after tier-1 backups.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
until grep -q "BACKUP AUDITIONS DONE" ~/dv3_logs/audition_backup.log 2>/dev/null; do sleep 180; done
# Empty-8x8: pure open navigation (distinct look). LavaGapS6: mid lethal gap.
# SimpleCrossingS9N2: denser wall maze. DistShift1: lava-strip navigation.
CANDS="minigrid_Empty-8x8 minigrid_LavaGapS6 minigrid_SimpleCrossingS9N2 minigrid_DistShift1"
for t in $CANDS; do
  echo "=== tier2 audition: $t ==="
  python -u orchestrator_chain_nm512.py --tasks $t --device cuda \
    --eval_every 2000 --phase_max_steps 100000 \
    --logdir ~/dv3_logs/audit_${t#minigrid_} --seed 7 > ~/dv3_logs/audit_${t#minigrid_}.log 2>&1
done
echo '=== TIER2 AUDITIONS DONE ==='
for t in $CANDS; do
  d=~/dv3_logs/audit_${t#minigrid_}
  [ -f $d/chain_summary.json ] && python3 -c "
import json; x = json.load(open('$d/chain_summary.json'))
p = x['phases'][0]
print('${t#minigrid_}:', 'LEARNS' if p['all_good'] else 'FAILS', '| steps', p['steps_used'], '| final', round(list(x['final_retention'].values())[0], 3))"
done
