# ROUTER_ALBUM_BUG_2026-07-10.md — album ordering bug: diagnosis, evidence, corrected-read pre-reg

## The bug (one line)

`orchestrator_chain_nm512.py` phase-end snapshot:

```python
disk_eps = tools.load_episodes(task_evaldirs[i], limit=10**9)   # returns NEWEST-FIRST
albums[i] = album_src[-20:]                                     # -> the OLDEST 20 episodes
```

`tools.load_episodes` defaults to `reverse=True` (newest-first iteration). `[-20:]` therefore
selected the **oldest** 20 eval episodes of each task — random-policy rollouts from the start of
each task's own phase — instead of the pre-registered "last 20 task-i eval episodes"
(COMPOSITE_INTEGRATED_PREREG_2026-07-06). Every composite run to date (comp4_s1, comp4_s3,
comp4b_s2; comp4b_s4/s5 in flight) deployed centroids built from incompetent episodes.
Fixed to `album_src[:20]` (local repo commit; deliberately NOT deployed to Spark until the
running sweep finishes — single protocol within the sweep).

## Evidence (routing_confusion_posthoc.py, comp4b_s2, offline, final encoder)

Probe = the actual final-read rollouts stored on disk (60 eps/task), routed offline at the
deployed commit_k=10 against centroids built two ways:

| Albums | T0 | T1 | T2 | T3 | mean |
|---|---|---|---|---|---|
| Intended (newest 20, competent) | 1.00 | 1.00 | 1.00 | 0.95 | ~0.99 |
| Deployed (oldest 20, bug) | 0.40 | 1.00 | 0.62 | 0.98 | **0.75** |

Recorded in-run `router_acc_final` = 0.75 — **exact replication**. Misroute pattern under the
bug: T0→T3 (36/60), T2→T0 (23/60). T1 routes 1.00 under BOTH album rules.

## Reinterpretation of comp4b_s2 (and the standing seeds)

1. **T1's routed failure (0.556) is NOT routing.** T1 routed perfectly even with buggy albums.
   head_1 through the final WM is genuinely marginal on SimpleCrossing: ~0.6–0.7 with high
   round-to-round variance (last in-phase rows 0.72–0.82; final routed rounds 0.556; bandit
   head-arm mean 0.641). Matches s1 (0.672) and s3 (0.677): T1 retention sits AT the bar on
   every seed — the real weakest link is T1 retention margin, not the router.
2. **T0/T2 misroutes were masked by the safe opening.** DoorKey/LavaGap episodes run ~12 policy
   steps; commit_k=10 means the (wrong) head controlled only the last ~2 steps, so routed
   returns stayed 0.94–0.96 despite 40–60% misroutes. Two follow-ups this raises, deferred:
   (a) commit_k=10 also means CORRECT heads barely act on short tasks — the routed read on
   T0/T2 largely evaluates the live actor's opening; offline sweep says k=5 would route
   0.95–1.00 with intended albums, giving heads ~7 of 12 steps. (b) log per-task routed
   accuracy, not just the pooled number, and per-episode policy-control fraction.
3. Deployed `router_accuracy_*` numbers for s1/s3/2R are artifacts of the bug (pooled, and
   inflated in-phase by having only 1–3 centroids live). Secondary reads must use corrected
   per-task accuracy.

## Pre-registered corrected read (locked before running)

After comp4b_s4/s5 finish (unchanged deployed code): re-execute ONLY the final routed read for
all five seeds {comp4_s1, comp4b_s2, comp4_s3, comp4b_s4, comp4b_s5} offline from their saved
`chain_phase{0..3}.pt` + eval dirs, with albums per the REGISTERED rule (newest 20 at phase-end
mtime cutoff): K=3 rounds × 10 episodes per task, routed, bandit arms fixed to each seed's
converged arm (head unless the seed's recorded bandit favored live), commit_k=10 unchanged.

**PRIMARY (unchanged bar): corrected final routed retention ≥0.6 on ALL FOUR tasks, ≥4/5 seeds.**
Both corrected and deployed-buggy numbers reported side by side. No other parameter changes; any
further change (e.g. commit_k=5) is a separate pre-reg.

**Prediction (falsifiable):** corrected T0/T2/T3 ≈ their oracle retention (routing ~0.99 removes
only a masked error); corrected T1 changes little (routing was never T1's problem) — seeds pass
or fail on T1's retention margin alone.

## OUTCOME (2026-07-11, corrected re-read executed — artifacts: <logdir>/corrected_reread.json ×5)

Per-task routing with corrected albums: ~1.00 everywhere (s1 T1: 0.80). Corrected routed T1:
s1 0.474 / s2R 0.676 / s3 0.633 / s4 0.811 / s5 0.494 → **corrected tally 3/5, primary NOT
confirmed** — same count as deployed, with s1 flipping PASS→FAIL and s2R flipping FAIL→PASS.
Prediction held: T0/T2/T3 ≈ oracle on every seed; verdicts decided by T1 alone. The ±0.2
same-seed swing between reads confirms the audit's noise warning: T1 delivery ≈ 0.62 ± 0.13,
and 0.6-bar crossings are round-noise. Fix path unchanged: head quality/maintenance
(dream-recovery arms, launched 2026-07-11).
