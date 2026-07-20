# DREAM_RECOVERY_RESULT_2026-07-12.md — Arm B (tunnel) 3/3 CONFIRMED; Arm A (rl) 0/3

**Pre-reg:** DREAM_RECOVERY_PREREG_2026-07-10.md (3ae70beb; Arm B amendment 0920d182 — both
committed before any run). **Primary (per arm): T1 real-env ≥0.75 for 3 consecutive evals within
20k imagination-only updates, ≥2/3 seeds.**
**Artifacts:** `<logdir>/dream_{rl,tunnel}_result.json` + `dream_*_curve.jsonl` in comp4_s1,
comp4b_s2, comp4_s3 (Spark). Gates: WM param-sum identical before/after (assert passed, all 6
runs); T1-window purity 0.90/0.95/0.90.

## Result

| Seed | Arm A (rl practice) | Arm B (tunnel: grade dreams, clone top 25%) |
|---|---|---|
| comp4_s1  | FAIL (best 0.84, thrash — never 3-consecutive) | **PASS @ 2,000 updates** (0.38 → 0.85) |
| comp4b_s2 | FAIL (best 0.36; T1 = 0.0 at end; collateral wrecked: T3 → 0.08) | **PASS @ 3,500** (0.66 → 0.92) |
| comp4_s3  | FAIL (best 0.57; T3 collateral → 0.40) | **PASS @ 7,500** (0.66 → 0.85) |

**Arm B: 3/3 — CONFIRMED. Arm A: 0/3 — NOT confirmed.**

Zero new environment steps in training for either arm. Same frozen WM, same T1-phase replay
starts, same budget — the ONLY difference is the teaching signal: policy-gradient practice vs
supervised cloning of trajectories the frozen reward head + critic graded into the top 25%.

## Readings (per the pre-registered matrix)

1. **The world model is a sufficient behavioral memory at this scale.** Dreams alone restore a
   lost skill to above its original frozen-head delivery (tunnel-recovered T1 0.85–0.92 vs
   head_1 delivery 0.62 ± 0.13 measured across the corrected re-reads).
2. **Arm-comparison row hit: "only B passes → tug-of-war mechanism gains direct support."**
   With identical dream material, RL practice thrashes (s1 touches 0.84 then swings to 0.38)
   and can actively destroy other behaviors (s2R collateral collapse); imitation of graded
   dreams is fast and stable, with collateral intact where measured (s3 @6k: 0.958/0.940/0.809).
   Actor forgetting was never signal absence — it is instability of the policy-gradient channel;
   bypassing it with a supervised channel fixes recovery.
3. Collateral drift under single-task rehearsal is real but mild for the tunnel (early stop
   helps). The full arm must rehearse ALL prior tasks, not one.

## Next (per matrix; requires its own pre-reg before any run)

**Interleaved tunnel-rehearsal DURING the chain:** every phase, a fraction of actor updates =
tunnel-cloning on graded dreams from ALL prior tasks' replay starts. Candidate replacement for
frozen-heads + router (storage: none beyond the existing buffer; no task labels at eval; single
live actor). Bar to beat, locked by the corrected re-read: routed-composite T1 delivery
0.62 ± 0.13 → target all-four ≥0.6 with T1 ≥ 0.75 mean, n≥3, plus the plain-chain 0/3 control.
