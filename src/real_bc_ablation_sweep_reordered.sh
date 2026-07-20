#!/usr/bin/env bash
# Real-BC ablation: does imagination beat cloning the stored real episodes?
# Pre-reg: REAL_BC_ABLATION_PREREG_2026-07-16.md. DREAM control = banked tunl4b (not re-run).
# Waits for the tunl8 sweep to finish (GPU serialized).
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dv3 && cd ~/projects/dreamerv3-torch
LOGS=~/dv3_logs
TASKS=minigrid_DoorKey-5x5,minigrid_SimpleCrossingS9N1,minigrid_LavaGapS5,minigrid_MultiRoom-N2-S4


for arm in real_filtered real; do
  [ "$arm" = "real" ] && tag=realbc || tag=realbcf
  for s in 1 2 3; do
    echo "=== $tag seed $s ==="
    OMP_NUM_THREADS=4 python -u orchestrator_chain_nm512.py --tasks $TASKS --device cuda \
      --tunnel_rehearsal --cont_grading --rehearsal_updates 50 \
      --rehearsal_source $arm \
      --eval_every 2000 --phase_max_steps 150000 \
      --logdir $LOGS/${tag}_s$s --seed $s > $LOGS/${tag}_s$s.log 2>&1
  done
done
echo '=== REAL-BC ABLATION DONE ===' >> $LOGS/ABLATION_STATUS.txt
python3 - << 'EOF' >> ~/dv3_logs/ABLATION_STATUS.txt
import json, glob
T1 = "minigrid_SimpleCrossingS9N1"
def arm(tag):
    v, p = [], 0
    for f in sorted(glob.glob(f"~/dv3_logs/{tag}_s*/chain_summary.json")):
        x = json.load(open(f))
        v.append(x["final_retention"][T1]); p += bool(x["success_all_tasks"])
    return (sum(v)/len(v) if v else None), p, len(v)
for tag, name in (("tunl4b", "DREAM (banked)"), ("realbc", "REAL-BC"),
                  ("realbcf", "REAL-BC-FILTERED")):
    m, p, n = arm(tag)
    print(f"{name:20s} T1 mean {m if m is None else round(m,3)} | all-four pass {p}/{n}")
EOF
cat ~/dv3_logs/ABLATION_STATUS.txt
