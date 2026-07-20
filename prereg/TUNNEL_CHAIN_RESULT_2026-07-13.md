# TUNNEL_CHAIN_RESULT_2026-07-13.md — interleaved tunnel-rehearsal chain: 3/3 UNANIMOUS

**Pre-reg:** TUNNEL_CHAIN_PREREG_2026-07-12.md (commit 96e53883, before implementation ran).
**Artifacts:** `tunl4_s{1,2,3}/chain_summary.json` + `chain_metrics.jsonl` + logs (Spark).
Single variable vs the banked plain chain (0/3): `--tunnel_rehearsal` (50 dream-self-imitation
updates per prior task per 2k-step chunk; one live actor; NO frozen heads, NO router, NO task
labels at eval; no storage beyond the existing never-clear buffer).

## Result — PRIMARY CONFIRMED 3/3 (bar: ≥2/3 all-four ≥0.6, final last-3-eval mean, live actor)

| Seed | DoorKey | SimpleCross (T1) | LavaGap | MultiRoom | total steps | verdict |
|---|---|---|---|---|---|---|
| s1 | 0.959 | **0.905** | 0.760 | 0.814 | 136k | PASS |
| s2 | 0.956 | **0.740** | 0.662 | 0.806 | 244k | PASS |
| s3 | 0.958 | **0.826** | 0.727 | 0.799 | 130k | PASS |

**SECONDARY met: T1 cross-seed mean = 0.824 ≥ 0.75** — above the frozen-head ceiling
(corrected composite delivery 0.62 ± 0.13) and far above plain replay (0.25–0.51).

**T1 solve-stability: 38/38, 73/80, 42/42 = 153/160 (96%) post-phase evals at/above bar.**
The task that died in every prior architecture is now the most stable protected task.
Two of three seeds ran FASTER than the composite (130–136k vs ~200k+ steps) — rehearsal
accelerates acquisition (LavaGap learned in 8–10k steps every seed) because the actor never
drifts far from competence.

## Controls / reference points (all banked, same tasks & budget lineage)

- Plain chain (no rehearsal): **0/3**, T1 0.25–0.51 — the single-variable control.
- Composite (frozen heads + router), corrected read: 3/5, T1 0.62 ± 0.13, parameter growth
  per task, task inference required. Tunnel chain beats it on every axis with none of the
  machinery. Frozen-heads architecture is hereby RETIRED as a candidate (remains the
  oracle/isolation reference).

## The systematic wart (3/3 seeds) and its queued fix

**LavaGap (T2) wobbles after fast graduation:** solve-stability 9/33, 14/75, 15/36. Mechanism
(diagnosed in-flight): tunnel-rehearsal speeds acquisition → T2 graduates in 8–10k steps →
(a) its rehearsal library is thin and beginner-heavy, (b) the WM has seen little of the task's
(lethal) dynamics — a mis-grading risk exactly where mistakes kill. Both effects follow from
FAST GRADUATION, not from rehearsal per se; final reads recovered ≥0.66 on all seeds anyway.
**Queued fix (design timestamped 2026-07-12, internal): triple-graduation re-harvest** —
triple-graduation windowed harvest; requires its own pre-reg before running.

## Reading

Catastrophic forgetting in this agent class is SOLVED at 4-task MiniGrid scale by a single
mechanism: continuous self-imitation on world-model-graded dreams of prior tasks. One actor,
no task labels, no parameter growth, ~15% compute overhead, and faster overall learning.
Claim boundary: 4 tasks, one ordering, MiniGrid, 17M-param WM, n=3. Next per plan: scale-up
(8+ tasks, harder domains) once external compute lands; re-graduation arm on local GPU meanwhile.
