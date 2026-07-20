# COMPOSITE_POSTHOC_PREREG_2026-07-06.md — per-task heads through the final chain WM, post-hoc

**Type:** decisive cheap test of the composite hypothesis BEFORE the integrated build.
**Insight:** chain checkpoints already contain every per-task head — the actor inside
`chain_phase{j}.pt` IS head_j frozen at task-j competence. The composite's core claim is therefore
testable post-hoc: load the FINAL world model (`chain_phase3.pt`), swap in actor_j, eval task j in
the real env. No retraining. Router validated separately (1.00 crux accuracy) — this isolates the
OTHER half of the composite: do frozen heads + the shared-buffer-preserved representation retain?

## Design
For each seed s∈{1,2,3}, each task j∈{0..3}: agent = WM(chain_phase3) + actor(chain_phase{j});
eval task j, 10 episodes × 3 rounds. Compare against banked live-actor final_retention.

## Pre-registered predictions (locked; conditioned on banked evidence)
- **The decisive cell: T1 (SimpleCrossing) with head_1.** Live-actor T1 = 0.35/0.51/0.25.
  Frozen-head arm says: frozen head + DRIFTED rep = death; shared-arm margins say: shared buffer
  keeps rep intact (live≈frozen). If the chain's buffer preserved T1's rep, head_1 should recover
  T1 to ≥0.6 (predict ~0.8+). If the chain's rep for T1 is itself degraded, head_1 stays low.
- T0/T2: heads ≈ live (both ~0.95) — heads shouldn't hurt.
- T3: head_3 ≈ live (~0.8).

## Interpretation matrix (locked)
| Outcome (n=3) | Reading | Next |
|---|---|---|
| T1-with-head_1 ≥0.6, others hold | **Composite validated post-hoc → 4/4 achievable.** Chain's T1 loss was pure policy-side. | Integrated composite run (in-loop router + heads) as confirmation; then CRADLE v1 banked at 4 tasks. |
| T1-with-head_1 <0.6 | T1's REPRESENTATION degraded in-chain (unlike 2-task shared). Heads can't fix a rep-side loss. | Rep-side fix first: rehearsal-quality (success-biased) or dream-rehearsal; composite after. |
| T0/T2 heads WORSE than live | Head-snapshot timing or rep-drift-since-snapshot issue. | Diagnose per-task margins (m{j}_frozen vs m{j}_live already logged in chain metrics). |

## Gate
head_j on task j must ≈ the task-j retention recorded AT phase-j end (the head is a copy of the
actor that produced that score through the then-current WM; only WM drift since separates them).
