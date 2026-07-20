# TUNNEL_CHAIN_PREREG_2026-07-12.md — interleaved tunnel-rehearsal chain (heads/router replacement candidate)

**Type:** FIX-ARM test. Committed BEFORE implementation runs. GPU free (comp4b sweep + dream
recovery complete).

## Hypothesis

Dream-recovery banked (DREAM_RECOVERY_RESULT_2026-07-12): tunnel self-imitation (clone the top
25% of WM-graded imagined trajectories) restores a lost skill 3/3, zero env steps, while the
policy-gradient channel thrashes 0/3. If the instability of the RL channel IS the forgetting
mechanism, then continuously rehearsing every PRIOR task tunnel-style DURING the chain should
prevent the loss in the first place — one live actor, no frozen heads, no router, no task labels
at eval, no storage beyond the existing never-clear buffer.

## Mechanism (single variable: `--tunnel_rehearsal`)

Plain chain (banked 0/3 control) plus: during phase i>0, after every eval_every=2000-step
training chunk, run `--rehearsal_updates` (locked: 50) tunnel updates per prior task j<i:
sample a batch from task-j's own-phase episodes (keys recorded at phase end from the shared
buffer — no extra storage); encode posterior starts (no grad); imagine with the current sampling
actor (live WM — its heads are replay-maintained, banked 2026-07-10); grade each trajectory with
the live reward head + critic bootstrap; behavior-clone the actor on the top 25%. Actor BC step
only; critic and WM train exactly as in the plain chain. `--tunnel_rehearsal` off ≡ banked chain
behavior (structural no-op gate).

Grading trustworthiness is load-bearing and pre-verified: reward-head retention R ≈ 1.0 and
zero false-reward on failure paths (REWARD_HEAD_AUTOPSY_RESULT_2026-07-10).

## Protocol

Tasks/order/budget identical to the amended composite protocol: DoorKey-5x5 → SimpleCrossingS9N1
→ LavaGapS5 → MultiRoom-N2-S4, phase_max 150k, eval_every 2000, bar 0.6, seeds 1–3, logdirs
`tunl4_s{1,2,3}`. Eval: plain live actor on all learned tasks every round (NO routing, NO heads).
Smoke gate first (--smoke + flag: rehearsal path executes, WM/critic training unchanged).

## Registered read

**PRIMARY: final retention (last-3-eval mean, live actor) ≥ 0.6 on ALL FOUR tasks in ≥2/3
seeds.** Controls it must beat: plain chain 0/3 (banked). Reference it aims to match/beat with
less machinery: composite corrected 3/5 with T1 = 0.62 ± 0.13.
**SECONDARY (locked):** cross-seed mean of final T1 ≥ 0.75 ("beats the frozen-head ceiling");
per-task solve-stability; wall-clock overhead vs plain chain (report, not gate).

| Outcome | Reading | Next |
|---|---|---|
| ≥2/3 pass, T1 mean ≥0.75 | Tunnel-rehearsal REPLACES heads+router at this scale. | 8-task chain; revisit test; write-up. |
| ≥2/3 pass, T1 mean <0.75 | Prevention works but weaker than recovery; acceptable fix. | Tune R (new pre-reg); write-up. |
| <2/3 pass, T1 fails as in plain chain | Online interference beats 50 rehearsal updates/chunk — dose or timing wrong. | One dose escalation (R=200) as a pre-registered follow-up; if that fails, tunnel is recovery-only. |
| <2/3 pass, NEW failure mode (e.g. current task can't learn) | BC channel fights acquisition. | Diagnose; possibly gate rehearsal off during current task's dead zone. |

n=3, seeds 1–3, no per-seed tuning; any parameter change is a new pre-reg.
