# STAGE_AB_SHARED_RESULT_2026-07-03.md — first 3/3 both-bars result

**Date:** 2026-07-03. **Arm:** shared never-clear buffer, joint actor (= Continual-Dreamer recipe at our
scale — no eviction below 1M cap, uniform-over-transitions sampling; verified vs tools.py).
**Runs:** `ab_shared_s{1,2,3}`, greedy, eval_every 2000, b_max 100k. Bars pre-registered (≥0.6 both).

## VERDICT — the two-task problem is SOLVED at this scale

| Seed | A_after | B_peak | Both bars | A solve-stability | Path taken |
|---|---|---|---|---|---|
| s1 | 0.962 | 0.95 | ✅ | 34/50 | 16-eval dead zone (both ~0) → both lock in |
| s2 | 0.959 | 0.94 | ✅ | 24/50 | see-saw: A held 13, traded for B ~20 evals, A returns ~eval 41 |
| s3 | 0.963 | 0.96 | ✅ | 49/50 | clean: A pinned 0.96 throughout, B climbs under it |

**3/3 both-bars, both tasks ~0.95–0.96 at the end, every seed.** First arm in the project to achieve
simultaneous near-max competence on both tasks. Task-agnostic at eval (single joint actor, no task ID).

## Key findings

1. **The see-saw/flicker is a TRANSIENT, not a failure mode.** All three seeds took different routes
   (dead zone / see-saw / clean) to the same destination. Basin variance moved from determining the
   OUTCOME (as in fixed-fraction replay: 1/3, 2/3) to only determining the ROUTE. Mid-run reads
   mislead — s2 looked dead at eval 32 and passed at eval 50 (third mid-run reversal this week).
2. **Mechanism confirmed by margins mid-run (s2):** during its A-dead phase, live≈frozen margin
   (representation intact — shared buffer doing its job); frozen margin dropped (actor tugged to B);
   A then RE-LEARNED from the intact rep + persistent A-data. Recovery is why never-clear beats
   fixed-fraction: the actor can always find A again because the WM never lost it.
3. **vs fixed-fraction replay (banked 2026-07-01):** f=0.25 clear-mode got 2/3 with A_after {0, 0.64,
   0.96} and permanent losses. Shared: 3/3 with A_after ~0.96 uniform. The difference is not dose —
   it's persistence + natural curriculum (A-fraction decays as B grows, but never to zero).
4. **Endpoint vs stability:** A solve-stability varies (24–49/50) but endpoints converge ~0.96.
   Both matter: report both. The transient dead zones are the cost; they may grow with task count.

## What this does NOT show
- Anything beyond 2 tasks, one pair, one direction, MiniGrid scale.
- Storage-free operation (buffer holds all raw A+B experience — fine at 160k transitions).
- Transient-free learning (dead zones of 15–20 evals in 2 of 3 seeds).

## Next (in order)
1. **Chain test A→B→C→D** — the binding question: does the transient/capacity hold at 4 tasks?
2. Reversed-pair check (B→A) — generalization of the characterization.
3. **Dream-rehearsal** (spec: DREAM_REHEARSAL_DESIGN_2026-07-02.md) — same stability, ~zero storage;
   bars now finalize against THIS result (beat/match 3/3 @ ~0.96/0.95).
4. Composite (frozen A-head + shared) — RETIRED to bench; not needed at n=3.

*Banked 2026-07-03. Predecessors: STAGE_AB_REPLAY_RESULT_2026-07-01 (fixed-fraction), frozen-head +
frozen-rep arms (mechanism bracket). The mechanism story (rep-drift total, coupled co-drift, transient
interference) is what turned 1/3 into 3/3 in five days.*

## CORRECTION (2026-07-10 audit — B column was PEAK, not endpoint)

The table's "B_peak" column and the headline line "both tasks ~0.95–0.96 at the end" conflict:
the ~0.95 values are `B_peak` (max over all evals); the ENDPOINT values from
`nm512/results/ab_shared_s{1,2,3}/ab_summary.json` are:

| Seed | B_final (last-K mean) | B_peak |
|---|---|---|
| s1 | **0.864** | 0.949 |
| s2 | **0.772** | 0.945 |
| s3 | **0.890** | 0.956 |

The 3/3 both-bars verdict STANDS (all B_final ≥ 0.6, all A_after ≈ 0.96). The corrected headline:
**3/3 both-bars; A ends ~0.96 every seed; B ends 0.77–0.89 (peaks ~0.95).** Finding 1's
"transient, not a failure mode" is correspondingly weakened for B: B gives back 0.06–0.18 from
peak by run end (see-saw not fully settled at the 100k budget). Note also the code asymmetry
behind the original conflation: `b_learns` gates on B_peak while `a_recovers` gates on last-K
mean (orchestrator_ab_nm512.py:386-389). Downstream docs citing "B≈0.95 every seed" inherit
this correction. Audit trail: five-expert audit 2026-07-10.
