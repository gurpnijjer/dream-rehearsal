# COMPOSITE_INTEGRATED_PREREG_2026-07-06.md — the full system, end-to-end, no oracle

**Type:** CONFIRMATION run — assembly of parts individually measured at n=3. The system: shared WM +
never-clear shared buffer (rep preservation, proven) + frozen per-task heads snapshotted at phase
boundaries (policy preservation, proven post-hoc) + prototype router picking the task from stored
frames (1.00/0.99/0.97, proven) + margin-arbitrated head-vs-live policy selection (best-of cleared
4/4 in all three seeds post-hoc). No task labels anywhere at eval time.

## Mechanism (implementation-level, locked)
- At phase-i end: snapshot frozen head_i (deep-copied actor) + album_i (last 20 task-i eval episodes;
  raw frames stored, re-embedded under the CURRENT encoder each eval round — drift-proof).
- Every eval round: recompute centroids (per-episode mean frame-embedding → mean); recompute
  policy_pick[j] = argmax over {head_j, live} of mean log-likelihood of album_j's recorded competent
  actions under re-encoded album latents (label-free arbitration; no env rollouts).
- RoutedEvalPolicy (completed tasks only; current task evals live as before): per step, embed the
  current frame, running-mean within episode, nearest centroid = provisional task ĵ; choice COMMITS
  at step 10 (probe v4: overall 0.75@1 → 0.91@10; crux confusion 0.000 at all k). Act with
  policy_pick[ĵ]. try/finally actor restore (gate 5.3 pattern).
- Final read: after the last phase, K=3 extra eval rounds of ALL tasks, fully routed.

## Pre-registered read (locked)
**PRIMARY: final routed retention ≥0.6 for ALL FOUR tasks, n=3 seeds.** (Post-hoc best-of numbers
predict ~0.96/0.65–0.87/0.88–0.93/0.81.)
**SECONDARY:** router accuracy vs ground truth (we know which env we evaluate) ≥0.9 overall;
head-vs-live pick pattern logged; centroid-margin telemetry per round.

## Interpretation matrix (locked)
| Outcome (n=3) | Reading | Next |
|---|---|---|
| 4/4 routed, router acc ≥0.9 | **CRADLE v1 CONFIRMED at 4 tasks, task-agnostic.** | Final report; grants/outreach; then dream-rehearsal + evil-twin. |
| 4/4 misses ONLY via router errors (acc <0.9 in-loop) | Probe→deployment gap in routing (running-mean/commit rule). | Fix integration (windowing, feat-vs-embed); parts remain valid. |
| T1 fails with router CORRECT | Arbitration miscalibrated (album-likelihood ≠ real-env quality). | Arbitrate on margin-probe flips instead of action-likelihood; post-hoc data can pre-validate. |
| Retention WORSE than plain chain on held tasks | Routed eval harness bug (swap leakage etc.). | Fix before interpreting; structural gates should catch. |

## Gates
1. `--composite` off ≡ current chain behavior (structural no-op).
2. Smoke: routed path executes end-to-end; router accuracy printed; no actor leakage
   (post-eval, live actor must be training-active).
3. Router accuracy is computed against ground truth per eval — continuous wiring check.
4. n=3, seeds 1–3, same budget/config as the banked chain (single variable: --composite).
