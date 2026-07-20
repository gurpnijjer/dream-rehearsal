# T2_WOBBLE_PROBES_PREREG_2026-07-13.md — convict the mechanism before choosing the fix

**Context:** tunl4 3/3 banked with one systematic wart: T2 LavaGap post-graduation stability
9/33, 14/75, 15/36. Re-evaluation (2026-07-13) weakened the original "fast graduation" story —
T0 also sprint-graduates and is ~perfectly stable — and surfaced a suspect in OUR OWN grading
code: `tunnel_update` scores dreams as Σγ^t·r + γ^H·V(last) **ignoring the cont (termination)
head**. On the one lethal task, dreams that die in imagined lava keep accruing post-death value
from latents the WM never trained on. Offline probes on existing tunl4 artifacts; GPU idle.

## Probes and locked predictions

**P1 — Grade audit (Fix A: death-aware grading).** Re-grade imagined T2 rollouts (starts from
each seed's deployed T2 library, final actor+WM) under (a) deployed scoring and (b) cont-aware
scoring (survival-weighted rewards + survival-weighted bootstrap). Measure the death rate
(any imagined step with cont < 0.5) inside each scoring's top-25%.
*Prediction:* deployed top-25% contains ≥10% death-dreams; cont-aware top-25% near 0%.
If confirmed → Fix A is load-bearing.

**P2 — Library audit (Fix B: rolling refresh).** For each seed: T2's deployed rehearsal library
(own-phase train episodes) — size and fraction competent (episode reward > 0.05), vs T1's
library as contrast.
*Prediction:* T2 libraries < 50% competent episodes and ≪ T1's episode count.
If confirmed → Fix B relevant.

**P3 — Cont-head autopsy (was the WM's death-physics young at graduation?).** On real T2
episodes, terminal-step discrimination D_cont = mean(cont @ nonterminal) − mean(cont @ terminal),
under ckpt p2 (T2 graduation) vs p3 (final). T0 as control.
*Prediction:* D_cont(T2) materially lower at p2 than p3 (young physics → mis-grading window),
consistent with the observed late-run self-healing.

## Decision rule (locked)

- P1 confirmed → pre-reg Fix A (cont-aware grading) in the fix arm regardless of the rest.
- P2 confirmed → add Fix B (rolling library refresh from newest competent eval episodes).
- Fix C (triple-graduation, pre-registered internally) is the pre-registered ESCALATION
  if an A(+B) arm fails to lift T2 solve-stability ≥ 70% at n=3 — it is the only fix that buys
  new real experience, and the only one with an env-step cost.
- If P1 AND P2 both refuted → stop; the mechanism is not understood; no fix arm until it is.

## OUTCOME (2026-07-13, artifacts: tunl4_s*/t2_wobble_probes.json + t2_probe_p1b.json)

**P1 CONFIRMED (sharpened):** ~99.7% of imagined T2 trajectories terminate in-horizon (episodes
~10 steps, horizon 15) — the deployed scorer graded post-terminal imagination on EVERY dream.
Disambiguated composition of the deployed top-25% (what rehearsal cloned): death-dreams
15.3% / 10.1% / 37.8% per seed. Cont-aware scoring: 0.6% / 0.7% / 1.6% (success composition
96–98%). The deployed ranking is barely better than random on this task.
**P2 REFUTED, inverted:** T2's library is the largest and MOST competent (372–735 eps, 67–81%
competent; T1: 340–457 eps, 45–62%) — episode length, not phase length, sets library size.
Beginner content demonstrably harmless (T1 rehearses at 96% stability from a ~50% library).
**P3 REFUTED:** cont head terminal discrimination 0.95–1.0 already at p2. Physics was never young.

**Convicted mechanism: post-terminal score contamination, maximal on short-episode tasks.**
Fix A (cont-aware grading) proceeds per the decision rule; Fix B dropped; Fix C
(triple-graduation) demoted to escalation-only. Note: dream_recovery.py's Arm B shares the
flawed formula — it passed 3/3 because T1's long episodes rarely terminate in-horizon
(consistency check, not luck); future recovery runs inherit the fix.
