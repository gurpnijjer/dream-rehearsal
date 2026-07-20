# REWARD_HEAD_AUTOPSY_RESULT_2026-07-10.md — H-reward REFUTED; replay-maintained WM heads, actor-side forgetting

**Pre-reg:** REWARD_HEAD_AUTOPSY_PREREG_2026-07-10.md (committed c5f69699, before any run).
**Registered prediction (H-reward): R(1) < 0.5 — REFUTED on all three seeds.**
**Artifacts:** `<logdir>/reward_head_autopsy.json` in comp4b_s2, comp4_s1, comp4_s3 (Spark).

## Primary result (n=3): reward discrimination retention R(j) = D(j, p3) / D(j, pj)

| Seed | R(T0) | R(T1) | R(T2) | D(T1, own) → D(T1, final) |
|---|---|---|---|---|
| comp4b_s2 | 1.003 | **0.991** | 1.005 | 0.909 → 0.901 |
| comp4_s1  | 1.002 | **1.056** | 1.018 | 0.835 → 0.883 |
| comp4_s3  | 0.999 | **1.006** | 0.996 | 0.830 → 0.835 |

The shared reward head does not forget any task's success signature across the chain. Critic
(exploratory): T1 mean value rises across phases (s2: 0.837 → 0.905). Sanity gate passed (T0
healthy everywhere). Metric anomaly noted for the record: comp4_s3 critic_mean T0_p2 = 1.19 > 1
(value head unbounded above max return) — exploratory read only, not gating.

## Addendum (2026-07-10 audit): degraded-distribution read — selection effect CLOSED

Objection: the primary D is measured on competent-era episodes that replay keeps training on.
Addendum probe on comp4b_s2's FINAL-ERA episodes (written after the phase-3 checkpoint — states
visited by the drifted/routed policy, including outright failures), scored under the final ckpt:

| Task | n success / fail eps | D_success (final-era) | False-reward on failed eps (mean / max) |
|---|---|---|---|
| T0 | 60 / 0  | 0.962 | — |
| T1 | 40 / 20 | **0.873** | **0.0 / 0.0** |
| T2 | 60 / 0  | 0.937 | — |
| T3 | 50 / 10 | 0.762 | 0.0 / 0.0 |

Off the curated path, the head still detects success (0.873 vs 0.901 curated) and hallucinates
zero reward on failure trajectories. The training signal available inside T1 dreams is intact on
the distribution that matters for re-teaching.

## Scope limits (what this does and does not establish)

1. **"Replay-maintained," not "intrinsic memory."** The never-clear buffer keeps training the WM
   heads on old-task data throughout; this result shows those heads track drift under replay —
   not that a replay-free WM would retain anything (frozen-head arm shows it would not).
2. **The latent space itself still drifts.** Same run's margin probe: m1_frozen = −0.25 at run
   end (T0: +4.5, T2: +2.7). Frozen actor heads decay because their input space moves under
   them; co-trained heads (reward, critic) ride the drift. "WM heads co-adapt while frozen
   actors decay" is the precise statement; "the WM remembers everything" is looser than the
   evidence.
3. Sharpened puzzle this banks: actor forgetting proceeds UNDER an intact, correctly-scored
   training signal in replay-anchored imagination. Gradient-imbalance explanation remains
   INFERENCE — untested. Direct test = DREAM_RECOVERY_PREREG_2026-07-10.md (Arms A/B, queued).
