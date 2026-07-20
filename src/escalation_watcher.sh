#!/usr/bin/env bash
# Escalation watcher (ESCALATION_LADDER_DESIGN §B): waits for tunl4b to finish, evaluates the
# REGISTERED bar (TUNNEL_CHAIN_V2_PREREG), then either declares PASS or deploys the staged
# orchestrator and launches the ARMED Fix C rung (TUNNEL_CHAIN_REGRAD_PREREG). The watcher only
# executes protocol committed in advance; on guardrail breaks it STOPS and launches nothing.
set -u
LOGS=~/dv3_logs
CODE=~/projects/dreamerv3-torch
VERDICT=$LOGS/LADDER_VERDICT.txt

while true; do
  done_ct=0
  for s in 1 2 3; do [ -f $LOGS/tunl4b_s$s/chain_summary.json ] && done_ct=$((done_ct+1)); done
  pgrep -f "logdir.*tunl4b" > /dev/null && running=1 || running=0
  [ $done_ct -eq 3 ] && break
  if [ $running -eq 0 ] && [ $done_ct -lt 3 ]; then
    echo "STALLED: sweep not running, only $done_ct/3 summaries — human needed" > $VERDICT
    exit 1
  fi
  sleep 600
done

python3 - << 'EOF' > $VERDICT
import json
res = {}
for s in (1, 2, 3):
    x = json.load(open(f"~/dv3_logs/tunl4b_s{s}/chain_summary.json"))
    a, b = map(int, x["solve_stability"]["minigrid_LavaGapS5"].split("/"))
    t1 = x["final_retention"]["minigrid_SimpleCrossingS9N1"]
    res[s] = {"t2_stab": a / b if b else None, "pass": x["success_all_tasks"], "t1": t1}
t2_ok = sum(1 for r in res.values() if r["t2_stab"] is not None and r["t2_stab"] >= 0.70)
guard_pass = sum(1 for r in res.values() if r["pass"])
t1_mean = sum(r["t1"] for r in res.values()) / 3
primary = t2_ok >= 2
guards = guard_pass >= 2 and t1_mean >= 0.75
for s, r in res.items():
    print(f"s{s}: t2_stability={r['t2_stab']:.2f} pass={r['pass']} t1={r['t1']:.3f}")
print(f"t1_mean={t1_mean:.3f} | PRIMARY(T2>=0.70 in >=2/3): {primary} | GUARDRAILS: {guards}")
print("VERDICT:", "PASS" if (primary and guards) else ("ESCALATE" if guards else "STOP_GUARDRAILS"))
EOF
cat $VERDICT

if grep -q "VERDICT: ESCALATE" $VERDICT; then
  echo "[watcher] deploying staged orchestrator + launching armed Fix C rung (tunl4c)" >> $VERDICT
  cp $CODE/orchestrator_chain_nm512.py.NEXT $CODE/orchestrator_chain_nm512.py
  cd $CODE && nohup bash tunnel_chain_regrad_sweep.sh > $LOGS/tunnel_chain_regrad_sweep.log 2>&1 < /dev/null &
elif grep -q "VERDICT: PASS" $VERDICT; then
  echo "[watcher] Fix A cleared the bar — deploying staged orchestrator for future runs, launching NOTHING" >> $VERDICT
  cp $CODE/orchestrator_chain_nm512.py.NEXT $CODE/orchestrator_chain_nm512.py
else
  echo "[watcher] guardrails broken or stall — STOP, human required" >> $VERDICT
fi
