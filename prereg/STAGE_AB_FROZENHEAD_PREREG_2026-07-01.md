# STAGE_AB_FROZENHEAD_PREREG_2026-07-01.md

**Date registered:** 2026-07-01 (BEFORE any frozen-head data collected)
**Arm:** frozen-head-A actor protection (isolated, no replay)
**Predecessors:** STAGE_AB_LOCATE_RESULT_2026-06-21 (Q2 actor-dominant 70/30), STAGE_AB_REPLAY_RESULT_2026-07-01 (replay baseline: 2/3 both-bars @ f=0.25, dose-independent flicker).
**Type:** MECHANISM TEST — not a fix candidate. Judged on its prediction, not a pass/fail fix bar.

---

## 1. The question

Q2 attributed ~70% of DoorKey forgetting to actor drift and ~30% to representation overwrite, measured with the frozen-vs-live margin probe (held-state preference). The replay sweep showed a **dose-independent flicker** (A_ret oscillates 0↔0.96 at every replay fraction) that the probe could not explain (probe read healthy while real-env A_ret=0).

**This arm asks:** if we remove actor drift entirely — freeze the task-A actor, let only the world model continue training on B — does the flicker collapse and does A retention stabilize at the level Q2's 30% representation-residual predicts?

---

## 2. Design (single swept variable)

| Parameter | Value |
|---|---|
| Task A | `minigrid_DoorKey-5x5` |
| Task B | `minigrid_SimpleCrossingS9N1` |
| **Swept variable** | `--actor_mode`: `joint` (baseline, current behavior) vs `frozen_a_head` |
| replay_fraction | **0** (isolate actor protection from replay — no confound) |
| replay_mode | clear (irrelevant at f=0) |
| expl_behavior | greedy |
| eval_every / b_max_steps | 2000 / 100000 |
| Seeds | 1, 2, 3 (n=3 from the start — the phenomenon IS seed-variance) |

**Baseline for comparison:** the existing f=0 no-replay runs `ab_fl_s{1,2,3}` (A_after 0, flat forgetting) AND the replay sweep (2/3 @ f=0.25).

### What `frozen_a_head` does (mechanism)
- At A→B switch: deep-copy the A-competent actor → `frozen_a_actor`, freeze (`requires_grad=False`, `.eval()`).
- B-training: the **live** actor + WM + critic train on B exactly as in `joint` (no change to the training loop).
- **A-eval only:** temporarily swap `agent._task_behavior.actor → frozen_a_actor`, run the A-eval, restore in a `finally`. So A-eval = obs → **live (B-drifted) WM** → **frozen A-actor** → action.
- B-eval and B-training use the live actor throughout.

This isolates: *with actor drift removed, does the drifted WM's representation still support task-A navigation?* The residual A-loss is pure representation overwrite (closed-loop).

---

## 3. PRIMARY read (locked)

**The primary metric is STABILITY, not endpoint.** Specifically:

**P1 — Flicker collapse.** Does A_ret stop oscillating? Measured as: solve-stability (frac of B-evals with A_ret ≥ 0.6) AND the count of 0↔high transitions across the B-phase. Prediction: **flicker collapses** (near-monotone A_ret, few/no transitions) if forgetting is actor-side.

**P2 — Retention level.** Where does steady A_ret settle? Prediction from Q2 70/30: **~0.65–0.70 steady** (the 30% representation residual).

**Secondary:** B_peak (must still reach ≥0.6 — freezing the A-actor must not block B, since the live actor is a separate trainable path).

---

## 4. Interpretation matrix (LOCKED before data)

| Outcome (n=3 consistent) | Reading | Next move |
|---|---|---|
| A_ret **steady ~0.65–0.70**, flicker collapses, B learns | **Q2 70/30 confirmed in closed loop.** Actor is the dominant, unstable half. | Composite: frozen_a_head + replay 0.25 (cover the WM's 30%) = CRADLE v1 fix candidate. |
| A_ret **steady ~0.30–0.50**, flicker collapses, B learns | Actor-side confirmed as the *unstable* element, but representation matters more than probe showed (~50/50). | Composite arm + probe re-examination (probe under-weighted rep). |
| A_ret **still flickers** 0↔0.96, B learns | **Flicker is representation-side, not actor-side.** Q2 mis-attributed the *instability* (probe measured preference, not closed-loop). | MAJOR redirect: WM-side protection (EWC-on-WM, WM distillation). Actor protection alone insufficient. |
| A_ret **~0 steady**, B learns | Representation overwrite is total once actor can't compensate. | Rethink: shared WM may be untenable; consider protected/partitioned WM or per-task WM. |
| B_peak < 0.6 (any A outcome) | Freeze implementation is wrong (live actor should be free) — B must be able to learn. | Debug swap wiring; do not interpret A. |

**Every non-buggy branch is decisive.** This is the point of the arm.

---

## 5. Success/validity gates (VERIFY THE GAUGE)

1. **Swap sanity gate (pre-B):** immediately after the switch, before ANY B-training step, run one frozen-head A-eval. It MUST equal A_before (~0.96) within eval noise — nothing has changed yet. If it doesn't, the swap is mis-wired → run is INVALID, do not interpret. (Same class as the probe's `live==frozen@competence` gate.)
2. **joint ≡ current baseline gate:** `--actor_mode joint` must reproduce the existing f=0 forgetting (A_after ~0). Structural: `joint` takes no swap path at all.
3. **try/finally on the swap:** an exception mid-A-eval must never leave the frozen actor live for B-training. Restore is in `finally`.
4. **n≥3, consistent direction.** No verdict off n=1 (already fooled once this week — replay f=0.25 s1).

---

## 6. Scope lock (what v1 does NOT do)

- No critic split (shared critic drifts to B; irrelevant to A-eval which goes through frozen actor).
- No task-agnostic head selection — **uses task ID at eval time** (v1 limitation; the real open research problem hides here).
- B-actor is the live continuation of A's weights (not fresh init) — cleanest delta from current setup.
- Frozen means frozen: no exploration/gradient touches `frozen_a_actor`.

---

## 7. What would make this arm WRONG to trust

- If B fails to learn → wiring bug, not a result (gate 5.5).
- If the pre-B sanity gate fails → swap mis-wired (gate 5.1).
- If only n=1 or n=2 → hypothesis, not verdict.
- If we read A_after endpoint instead of solve-stability → we already know that misleads on a flickering signal.

---

## 8. Files / launch

- Orchestrator flag: `--actor_mode {joint,frozen_a_head}` (default joint = unchanged).
- Launch: `--task_a minigrid_DoorKey-5x5 --task_b minigrid_SimpleCrossingS9N1 --actor_mode frozen_a_head --replay_fraction 0 --expl_behavior greedy --eval_every 2000 --b_max_steps 100000 --seed {1,2,3}`
- Logdirs: `~/dv3_logs/ab_frz_s{1,2,3}`.

---

*Registered 2026-07-01 before implementation. Predictions locked. Read against §4 matrix on n=3.*
