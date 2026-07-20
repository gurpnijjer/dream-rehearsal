#!/usr/bin/env bash
# Backup auditions (Gurp 2026-07-15): MultiRoom-N4 failing to bootstrap -> audition three
# replacements so the 8-task roster can still fill. Waits for the primary sweep to finish.
# Candidates biased toward learnable-under-greedy while adding variety:
#   LavaGapS7  — bigger lethal gap (family proven: S5 learns in 8-10k)
#   Unlock     — key+door interaction, short horizon
#   Empty-16x16 — large open navigation (guaranteed-floor candidate)
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
until grep -q "AUDITIONS DONE" ~/dv3_logs/audition_sweep.log 2>/dev/null; do sleep 120; done
CANDS="minigrid_LavaGapS7 minigrid_Unlock minigrid_Empty-16x16"
for t in $CANDS; do
  echo "=== backup audition: $t ==="
  python -u orchestrator_chain_nm512.py --tasks $t --device cuda \
    --eval_every 2000 --phase_max_steps 100000 \
    --logdir ~/dv3_logs/audit_${t#minigrid_} --seed 7 > ~/dv3_logs/audit_${t#minigrid_}.log 2>&1
done
echo '=== BACKUP AUDITIONS DONE ==='
for t in $CANDS; do
  d=~/dv3_logs/audit_${t#minigrid_}
  [ -f $d/chain_summary.json ] && python3 -c "
import json; x = json.load(open('$d/chain_summary.json'))
p = x['phases'][0]
print('${t#minigrid_}:', 'LEARNS' if p['all_good'] else 'FAILS-TO-BOOTSTRAP', '| steps', p['steps_used'], '| final', round(list(x['final_retention'].values())[0], 3))"
done
