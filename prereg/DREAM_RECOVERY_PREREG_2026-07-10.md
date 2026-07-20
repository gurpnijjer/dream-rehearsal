# DREAM_RECOVERY_PREREG_2026-07-10.md — can the world model re-teach a forgotten behavior, dreams only?

**Status: PRE-REGISTERED, QUEUED behind the comp4b sweep + corrected routed re-read (GPU busy).**
**Amended same-day (before any run): added Arm B (dream self-imitation, "the tunnel") — race
two teaching methods, same budget, same seeds.**

## Setup (the sharpest question the autopsy allows)

Established (n=3, REWARD_HEAD_AUTOPSY result): after the 4-task chain, the WM retains T1's
representations (margin probes), reward knowledge (R(T1) = 0.99/1.06/1.01), and value estimates —
while the live actor has drifted off T1. The training signal inside T1 dreams is intact; the
behavior is gone. If the WM is a *sufficient* memory, it should be able to re-teach the actor T1
**purely in imagination — zero new environment steps**.

## Method

Per seed (comp4_s1, comp4b_s2, comp4_s3): load chain_phase3.pt (final WM + drifted live actor +
critic). **Freeze the entire WM** (encoder, RSSM, all heads). Identify T1-phase episodes in the
shared train buffer by mtime window (phase0_end, phase1_end]. Training loop = standard DreamerV3
actor-critic update, except batches are sampled ONLY from T1-phase episodes: encode → posterior
states → imagine H (config default) → actor/critic losses. No env interaction for training.
Real-env eval of T1 every 500 updates (10 episodes); also eval T0/T2/T3 every 2000 updates
(collateral telemetry, non-gating). Budget: 20k updates max.

## Two arms (same seeds, same 20k-update budget, same eval cadence)

**Arm A — practice in dreams (RL-in-imagination):** as above; standard actor-critic on imagined
rollouts from T1 starts. The body learns by trial-and-error inside the brain's simulator.

**Arm B — the tunnel (dream self-imitation):** the brain demonstrates, the body copies.
Loop: from T1 posterior starts, roll the CURRENT actor in imagination with exploration noise
(sampled actions, optionally raised temperature); score each imagined trajectory with the frozen
reward head (+ frozen critic bootstrap for truncation); keep the top-scoring fraction (top-k
per batch, k pre-set to 25%); behavior-clone the actor on the kept state→action pairs (cross-
entropy, no policy gradient, no critic update). Rationale: the autopsy shows the WM can GRADE
dreams perfectly (R(T1)≈1.0) — self-imitation converts that grading into a demonstration signal,
avoiding the gradient tug-of-war that caused the forgetting. Risk to watch: imitation locks in
imagined success that doesn't transfer if the WM's T1 dynamics are off — the real-env eval
cadence catches this (imagined score up, real-env flat = imagination-reality gap row below).

## Registered read

**PRIMARY (per arm): T1 real-env return ≥ 0.75 for 3 consecutive evals within 20k
imagination-only updates, in ≥2/3 seeds.** (0.75 ≈ own-phase T1 competence band 0.7–0.9; above
every seed's final drifted value 0.25–0.56.)
**ARM COMPARISON (secondary, locked):** updates-to-first-pass and area-under-recovery-curve;
whichever arm wins ≥2/3 seeds is the teaching method carried into the interleaved-rehearsal
design. If both pass, prefer the simpler/faster; if only B passes, the tug-of-war mechanism
story gains direct support.

| Outcome | Reading | Next |
|---|---|---|
| ≥2/3 recover | **WM is a sufficient behavioral memory at our scale.** Dreams alone restore lost skills. | Build the full arm: interleaved dream-rehearsal DURING the chain (every phase rehearses all prior tasks in imagination) — candidate replacement for heads+router. |
| 0–1/3 recover, imagined returns rise but real-env doesn't | Imagination-reality gap (WM good at scoring, bad at simulating T1 dynamics). | Measure open-loop prediction error on T1; WM-side fix before any rehearsal arm. |
| Actor can't even raise imagined returns | Actor optimization/plasticity problem, not memory. | Investigate actor loss landscape / entropy; consider per-task gradient balancing. |

## Guards

- WM frozen is structural (requires_grad False + optimizer only over actor/critic params);
  assert no WM param changes (hash before/after).
- The T1-episode mtime selection must yield ≥80% actual T1 episodes (spot-check 20 by env
  identity in the image observations); else fix selection before running.
- Eval uses the plain live actor being trained — no router, no heads, no arbitration.
- n=3 seeds, same budget, no per-seed tuning. Any hyperparameter change = new pre-reg.
