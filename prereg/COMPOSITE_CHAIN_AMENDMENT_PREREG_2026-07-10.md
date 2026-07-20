# COMPOSITE_CHAIN_AMENDMENT_PREREG_2026-07-10.md — outcome record + amended protocol (seeds 2R, 4, 5)

**Type:** OUTCOME RECORD of COMPOSITE_INTEGRATED_PREREG_2026-07-06 + pre-registered AMENDMENT
(acquisition-budget change and seed extension). Written and committed BEFORE launching the new runs.

## 1. Outcome of the 2026-07-06 pre-reg (comp4_s1..s3, 100k phase cap)

**Primary read (all four tasks ≥0.6 routed, n=3): 2/3. NOT confirmed as pre-registered.**

| Seed | T0 DoorKey | T1 SimpleCross | T2 LavaGap | T3 MultiRoom | Router final / in-phase | Verdict |
|------|-----------:|---------------:|-----------:|-------------:|------------------------:|:-------:|
| s1   | 0.961      | 0.672          | 0.941      | 0.794        | 0.775 / 0.866           | PASS    |
| s2   | 0.511      | 0.478          | 0.930      | 0.784        | 0.813 / 0.905           | FAIL    |
| s3   | 0.956      | 0.677          | 0.939      | 0.644        | 0.900 / 0.959           | PASS    |

(final routed retention; logdirs `~/dv3_logs/comp4_s{1,2,3}` on Spark)

**Secondary (router acc ≥0.9): held only for s3 (final-round read).** In-phase accuracy 0.87–0.96.
Final-round accuracy runs lower than in-phase; both are logged. Bar retained as diagnostic, not
amended — recording honestly that s1 passed primary while missing secondary (strong heads +
arbitration mask a fraction of misroutes).

**Controlled comparison vs banked plain chain (chain4_s1..s3, same tasks/seeds/budget, single
variable `--composite`): plain chain 0/3** — T1 SimpleCrossing forgotten every seed
(0.347 / 0.509 / 0.247); T0/T2/T3 held. Composite rescues exactly T1 (0.672 / 0.677 routed on
passing seeds). The mechanism earns its keep on the task the shared arm loses.

**s2 failure diagnosis (not a row in the original interpretation matrix):** T3 MultiRoom-N2-S4
never bootstrapped — T3 = 0.0 for the entire 100k P3, phase ended at cap without all-good. The
live actor thrashed on the unrewarded task and dragged T0 from 0.92 into a 0.27–0.86 chaotic band;
final T0 routed 0.511. Root cause is **acquisition variance on the current task**, not
retention failure of the protected tasks and not router/head defect. Evidence it is a dead-zone
crossing problem: s3 showed T3 = 0.0 until ~34k P3-steps then broke through (first nonzero 36k,
all-good ~62k); s1 needed 58k. s2 simply did not cross within 100k.

**Known telemetry gap (non-gating):** `solve_stability` reports "0/0" for the terminal task in all
seeds (counter does not track the final phase's task). Fix opportunistically; retention numbers are
unaffected.

## 2. Amendment (locked before launch)

**Single protocol change: `--phase_max_steps` 100000 → 150000.**
Justification: observed T3 first-reward dead zone of ~34–46k phase-steps (s1, s3); s2 hit the cap
still at zero. This is **acquisition budget for the current task, not retention help** — no
retention mechanism (buffer, heads, router, arbitration) changes. Any future plain-chain
comparison gets the same cap.

**Carry-forward is valid:** s1 and s3 never hit the 100k cap in any phase (max phase length 62k),
so their trajectories are identical under the 150k protocol. They carry forward unchanged.

**New runs:** seed 2 re-run under the amended protocol (fresh logdir, no reuse of comp4_s2), plus
seeds 4 and 5. Logdirs `comp4b_s2`, `comp4b_s4`, `comp4b_s5`.

**PRIMARY (locked): final routed retention ≥0.6 for ALL FOUR tasks in ≥4/5 of seeds {1, 2R, 3, 4, 5}.**
- 5/5 → CRADLE v1 composite CONFIRMED at 4 tasks, task-agnostic.
- 4/5 → CONFIRMED with one recorded failure; failure must be diagnosed and typed (see below).
- ≤3/5 → NOT confirmed; stop and diagnose per matrix before any further protocol change.

**SECONDARY:** router in-phase and final-round accuracy logged per seed (≥0.9 diagnostic bar);
head-vs-live arbitration picks logged.

**Failure typing (added to interpretation matrix):** a seed where the CURRENT task never reaches
the bar within the 150k cap is recorded as **ACQUISITION-FAIL**, distinct from RETENTION-FAIL of a
protected task. If ≥2/5 seeds acquisition-fail on T3 even at 150k, MultiRoom-N2-S4 is a bad
chain-terminal task at this model scale/budget → pre-register a terminal-task swap; do NOT push
the budget further.

## 3. Launch spec (exact)

```bash
# composite_chain_sweep2.sh on Spark, seeds run sequentially, nohup
TASKS=minigrid_DoorKey-5x5,minigrid_SimpleCrossingS9N1,minigrid_LavaGapS5,minigrid_MultiRoom-N2-S4
for s in 2 4 5; do
  python -u orchestrator_chain_nm512.py --tasks $TASKS --device cuda --composite \
    --eval_every 2000 --phase_max_steps 150000 \
    --logdir ~/dv3_logs/comp4b_s$s --seed $s > ~/dv3_logs/comp4b_s$s.log 2>&1
done
```

Estimated wall-clock ~8–14 h/seed sequential (per comp4_s1..s3). Orchestrator file on Spark
unchanged (`orchestrator_chain_nm512.py`, no code edits in this amendment).
