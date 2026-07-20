# STAGE_AB_FROZENREP_PREREG_2026-07-02.md

**Date registered:** 2026-07-02 (BEFORE any frozen-representation data collected)
**Arm:** frozen representation + frozen A-head (the "mirror" / substrate-freeze composite)
**Predecessors:** STAGE_AB_FROZENHEAD_PREREG_2026-07-01 + its result (n=2 so far: frozen actor through
drifting WM → A_ret TOTAL collapse to 0, gates PASS — representation overwrite alone destroys A).
**Type:** PLASTICITY TEST. Retention of A is guaranteed BY CONSTRUCTION in this arm (see §3) — the open
question is whether task B can still be learned through task A's frozen representation.

---

## 1. The question

The frozen-head arm showed (n=2, s3 pending) that a perfect frozen A-policy dies anyway because the
world model's representation drifts out from under it. So the representation is the substrate that must
be pinned. The bluntest pin is: **freeze the representation (encoder + RSSM dynamics) at A-competence.**

**This arm asks the only question that freeze leaves open: does B still learn?**
Can SimpleCrossing's policy be learned in imagination through DoorKey's frozen encoder/dynamics, with
only the reward/cont heads, critic, and actor allowed to adapt?

---

## 2. Design (single swept variable vs the frozen-head arm)

| Parameter | Value |
|---|---|
| Task A / Task B | `minigrid_DoorKey-5x5` / `minigrid_SimpleCrossingS9N1` |
| **New flag** | `--wm_mode frozen_rep`: at the A→B switch, set `requires_grad=False` on `wm.encoder` + `wm.dynamics`. Reward head, cont head, decoder, critic, actor stay trainable. |
| actor_mode | `frozen_a_head` (A-eval via the frozen A-actor, as in the previous arm) |
| replay_fraction | 0 (no rehearsal — freeze must stand alone) |
| Everything else | identical to frozen-head arm (greedy, eval_every 2000, b_max 100000, seeds 1–3) |

**Why the composite (frozen rep + frozen A-head):** with BOTH frozen, the entire A-eval pathway
(obs → frozen encoder → frozen dynamics → frozen A-actor) contains no trainable component. A_ret ≈ 0.96
is then *structural*, and any deviation is a wiring bug, not a finding. The experiment cleanly isolates
plasticity: only B's learning capacity is under test.

**Why the reward head must NOT be frozen:** the actor trains in imagination against
`wm.heads["reward"]`. Freezing it would leave B optimizing A's reward function — B could never learn,
and the arm would be meaningless. Freeze = encoder + dynamics ONLY.

---

## 3. Validity gates (VERIFY THE GAUGE)

1. **A_ret constancy gate (structural):** A_ret must sit at ~A_before (±eval noise) at EVERY B-eval.
   This is not a finding — it is the wiring check. Any A_ret collapse in this arm = bug (frozen path
   has no trainable parameters), run INVALID.
2. **Representation fingerprint gate:** SHA-256 over encoder+dynamics parameter bytes computed at the
   switch and again at run end MUST be identical. Recorded in summary as `wm_rep_fingerprint_{switch,end}`.
3. **live==frozen probe gate (automatic):** with dynamics+encoder frozen, the margin probe's `live`
   re-encoding equals its `frozen` cached-latent path throughout — `live_margin ≈ frozen_margin` at
   every eval. Persistent divergence = the freeze is leaking.
4. **Swap sanity gate** (inherited from frozen-head arm): frozen A-eval == A_before pre-B.
5. **n=3, consistent direction** before any verdict.

---

## 4. PRIMARY read (locked)

**P1 — B_peak.** Does SimpleCrossing reach competence (≥0.6) through the frozen representation?
**P2 — B trajectory shape.** If B learns but slower than baseline (~0.91–0.96 peak unmixed), quantify
the plasticity tax: B_peak and evals-to-first-0.6 vs the `ab_fl` baseline.

Honest prior: genuinely uncertain. Shared MiniGrid visual grammar (walls/floor/agent/goal) argues yes;
DoorKey-5x5's encoder never saw 9x9 layouts, wall-gap crossings, or B's objects, and the RSSM's
dynamics were fit to A's transition structure — argues no. This uncertainty is why the arm is worth
running.

---

## 5. Interpretation matrix (LOCKED before data)

| Outcome (n=3 consistent) | Reading | Next move |
|---|---|---|
| B_peak ≥ 0.6, A_ret ~0.96 throughout | **Two-task continual learning SOLVED on this pair by substrate-freeze + per-task heads.** 3/3 both-bars, beats replay baseline (2/3 @ 72% stability) outright. | Bank CRADLE v1. Then the real frontier: >2 tasks, task-agnostic head selection, partial-plasticity (distillation) so the WM can still grow. |
| B_peak ≥ 0.6 only some seeds | Frozen substrate is *marginally* sufficient — B learnability is basin-dependent. | Partial-freeze / distillation on the WM (pin A-relevant features only, leave the rest plastic). |
| B_peak < 0.6 every seed | DoorKey's representation cannot express SimpleCrossing. Substrate-freeze is too rigid — plasticity is the binding constraint, and WM protection must be SOFT. | Representation distillation on stored A-observations (LwF-style on the encoder/dynamics), or EWC-on-WM (careful strength sweep per Stage-1 history). Replay 0.25 composite as fallback. |
| A_ret deviates from ~0.96 (any seed) | WIRING BUG (structural gate 3.1 violated). | Fix before interpreting anything. |

Every branch picks the next build. No branch is wasted GPU.

---

## 6. Scope lock

- Freeze granularity is encoder+dynamics as a unit. No finer ablation (encoder-only vs dynamics-only)
  in this arm — that's a follow-up only if the coarse result demands it.
- No replay, no distillation, no EWC in this arm — freeze must be measured alone.
- Task ID at eval time (same v1 limitation as frozen-head arm; noted, deferred).
- Decoder left trainable (it only serves the WM loss; irrelevant to the frozen A-eval path).

---

## 7. Launch

- `--task_a minigrid_DoorKey-5x5 --task_b minigrid_SimpleCrossingS9N1 --wm_mode frozen_rep
  --actor_mode frozen_a_head --replay_fraction 0 --expl_behavior greedy --eval_every 2000
  --b_max_steps 100000 --seed {1,2,3}`
- Logdirs: `~/dv3_logs/ab_frzrep_s{1,2,3}`. Driver: `frozen_rep_sweep.sh`.
- **Launch gate:** only after frozen-head s3 completes (n=3 read of the predecessor arm first) — the
  GPU is sequential and the predecessor's verdict is required context for reading this arm.

---

*Registered 2026-07-02 before implementation. Predictions locked. Read against §5 on n=3.*
