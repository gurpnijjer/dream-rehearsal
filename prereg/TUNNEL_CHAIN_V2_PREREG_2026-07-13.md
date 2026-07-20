# TUNNEL_CHAIN_V2_PREREG_2026-07-13.md — cont-aware dream grading (Fix A), single variable

**Type:** fix arm for the one systematic wart in TUNNEL_CHAIN_RESULT_2026-07-13 (3/3 pass, but
T2 LavaGap post-graduation solve-stability only 27% / 19% / 42%). Mechanism convicted by
pre-registered probes (T2_WOBBLE_PROBES outcome): the tunnel grader scored imagination PAST the
dream's own termination — on a ~10-step task with a 15-step horizon that contaminated every
trajectory's score, and 10–38% of what rehearsal cloned were walk-into-lava dreams.

## Change (single variable: `--cont_grading`)

In `tunnel_update` scoring only: weight rewards by imagined survival and the bootstrap by
terminal survival —
`surv_t = Π_{u≤t} cont_u`;  `score = Σ γ^t·surv_t·r_t + γ^H·surv_H·V(feat_H)`
(matches how DreamerV3's own value targets discount by the cont head). Nothing else changes:
same tasks, order, 150k cap, eval_every 2000, 50 rehearsal updates/prior task, top-25% clone,
same bar 0.6, seeds 1–3, logdirs `tunl4b_s{1,2,3}`. Offline replay of the fix on banked
libraries flips cloned-material composition from 62–90% success to 96–98% success
(death-dreams 10–38% → 0.6–1.6%).

## Registered read

**PRIMARY (fix-specific): T2 solve-stability ≥ 70% in ≥2/3 seeds** (baseline 27/19/42%).
**GUARDRAILS (must hold, else the fix is rejected regardless of T2):** all-four ≥ 0.6 final in
≥2/3 seeds (non-inferiority vs tunl4's 3/3), and T1 cross-seed mean ≥ 0.75.
**SECONDARY:** T1/T0 solve-stability not degraded (≥ their tunl4 bands); phase lengths logged
(prediction: T2's own phase unchanged — grading fix, not acquisition change).

| Outcome | Reading | Next |
|---|---|---|
| T2 ≥70% in ≥2/3, guardrails hold | Wart closed with a grading correction. | Scale-up when compute lands; paper. |
| T2 improves but <70% | Partial — residual instability has another component. | ESCALATE: Fix C re-graduation arm (pre-registered internally). |
| Guardrails break | The deployed mis-grading was accidentally load-bearing elsewhere — revert, diagnose. | Stop; no further arms until understood. |

n=3, no per-seed tuning; any other change is a new pre-reg.

## AMENDMENT v2.1 (2026-07-14, before any completed v2 run — sweep aborted at seed 1, phase 1)

The offline grader gauge (dream_picker_eval.py, built per ESCALATION_LADDER_DESIGN §A.1 on
banked tunl4 batches) caught the v2 scorer INVERTED on long-horizon tasks before any seed
finished: survival-weighting through step t zeroes the terminal goal reward itself —
T1 AUC 0.004 (selects 0.5% successes); T2 unaffected by luck of timing. tunl4b partial run
(seed 1, ~P1) aborted per the guardrail row; partial logdir deleted, no results read.

**Corrected formula (single change vs v2): rewards weighted by REACH probability
(surv shifted one step); bootstrap still survival-weighted.** Offline gauge after correction:
T2 AUC 0.992/0.998 (top-25% purity 1.0), T1 top-25% enrichment 1.8x vs base against a myopic
within-horizon label. Referee test (dream-recovery on comp4b_s2's broken T1, scorer=shift vs
banked dep PASS@3.5k): result to be recorded below before relaunch.

Consensus variant evaluated and DROPPED (adds ≤0.005 AUC over shift — not worth machinery).
Bars, guardrails, seeds, logdirs (tunl4b_s{1,2,3}) unchanged. Relaunch is a fresh sweep.

## AMENDMENT v2.2 (2026-07-14, before relaunch — referee outcomes recorded)

Shift-scorer referee REFUTED: recovery on comp4b_s2 T1 flat at 0.0 through 8,500 updates
(banked dep: PASS@3,500); aborted, curve banked (dream_tunnel_shift_curve.jsonl). Diagnosis:
termination-aware scoring under-ranks realized late successes vs the critic's optimistic
value on non-terminating dreams — promises outbid results. λ-return variant offline-ties
shift (same blind spot).

**Adopted scorer "lex" (realized-first): dreams that actually achieved reward (reach-weighted
discounted sum > 0.3) outrank all value promises; within groups, shift score orders.**
Offline gauge: AUC 1.0 / top-25% purity 1.0 on T1 and T2, both seeds probed. Referee:
recovery PASS at 2,000 updates (vs dep 3,500), peak 0.944 (curve: dream_tunnel_lex_curve.jsonl).
Both failure modes closed: no post-terminal contamination (T2), realized-success priority (T1).
Bars, guardrails, seeds, logdirs unchanged. tunl4b relaunches fresh under lex.

## OUTCOME (2026-07-14, watcher-declared — LADDER_VERDICT.txt; artifacts tunl4b_s{1,2,3})

**PASS.** All three seeds pass all-four (s1 0.960/0.868/0.899/0.751; s2 0.961/0.751/0.944/0.800;
s3 0.964/0.825/0.784/0.768). T2 solve-stability 95% / 100% / 67% vs pre-fix 27% / 19% / 42% —
primary (≥70% in ≥2/3) met; s3 individually below 70% (residual per-seed variance, noted
honestly). Guardrails held: T1 mean 0.815 ≥ 0.75. Escalation rung (re-graduation) NOT fired.
The verdict was evaluated and declared by the escalation watcher autonomously per
ESCALATION_LADDER_DESIGN §B — first end-to-end run of the self-executing ladder.
