# Stage A→B — Cross-task forgetting benchmark + mechanism test: PRE-REGISTRATION

**Status:** PRE-REGISTERED / DRAFT-FOR-LOCK 2026-06-19, BEFORE any A→B run.
**Discipline:** single-variable, multi-seed (n≥3), real-env eval, learnability-gated, sealed read table
committed before data. Same standard that ran the whole Failure Localization Program.
**Why now:** the single-task substrate is SEALED (Stage-3: critic refuted causally n=2; actor-drift /
argmax-flip; three cheap actor-side fixes all refuted). The localization is banked. The forward move is
the test v1.0 is actually defined by — **does the agent forget task A after learning task B**, and
**does the localized mechanism explain it.**

---

## 1. The two questions

- **Q1 (benchmark):** Does the agent catastrophically forget task A after training on task B in a
  continual (sequential A→B, no A-replay) setting — a drop in A-performance beyond seed noise?
- **Q2 (mechanism, the on-thesis payoff):** IF forgetting occurs, is it the **same mechanism the
  single-task work LOCALIZED** — **actor-drift / argmax-flip** ("representation survives, behavior dies",
  WM-on-A healthy while A's policy degrades) — or a **different** mechanism (representation/WM overwrite)?
  **Q2 is about the *localized* mechanism, NOT a *fixable* one — independent of fix success.** Stage-3
  established the single-task mechanism is **real-but-UNFIXED** (critic refuted causally n=2; actor-drift;
  all three cheap actor-side fixes refuted). A failed fix does NOT weaken Q2 — the question is whether the
  *localized* mechanism transfers to cross-task forgetting, which is well-posed regardless of whether
  anything stabilizes it. Q2 is what makes this more than a benchmark: it tests whether the localization
  *generalizes* from single-task oscillation to cross-task forgetting.

## 2. Substrate (fixed)

- MiniGrid discrete suite, P2E intrinsic ON (`--intrinsic --intr_scale 0.1`), entropy_target 1.0 — the
  same trainer stack as Stages 0–3 (carries forward the entropy-fix + the known-learnable regime).
- **Harness:** adapt Spark `cradle/training/orchestrator_forgetting.py` (built for the DMC
  walker_walk→run test, which found NO forgetting — §3 snapshot) to: (a) MiniGrid A→B, (b) periodic
  A-retention eval throughout B-training, (c) the §5 mechanism instrumentation. Reuse, don't rebuild.

## 3. Task pair — candidate + selection gate

A forgetting test needs A and B **distinct enough to interfere** (the DMC walker pair failed precisely
because walk→run positively transferred — no interference, no forgetting). Candidate:
- **A = `MiniGrid-Empty-6x6-v0`** — KNOWN learnable (Stage-0, with P2E), pure navigation.
- **B = `MiniGrid-DoorKey-5x5-v0`** — distinct skill (pick up key → open door → navigate); the
  Continual-Dreamer direction.
- **Interference caveat (pre-registered):** DoorKey *contains* navigation, so it may *build on* rather
  than *overwrite* A → weak forgetting. If Phase-0 + a pilot show A→B transfers (no interference), swap B
  for a genuinely conflicting task (e.g. `LavaGapS5-v0` / `SimpleCrossingS9N1-v0` — different hazard /
  path behavior). Final pair is the **Phase-0 deliverable**, chosen for *learnable AND interfering*.

## 4. Phase 0 — learnability gate (cheap, FIRST; fail-fast)

Before any forgetting run, confirm BOTH tasks are independently learnable on this substrate:
- 1-seed fail-fast: does each of A, B reach a competence bar **(success_rate ≥ 0.6 over 20 eps)** within
  **80k steps**? (B may need the full budget; DoorKey is exploration-harder.)
- If a task doesn't clear → it's not a valid benchmark arm → swap (per §3) or STOP and reassess.
- Then n=3 per task to confirm reliability (basin-lottery: this substrate is fragile — may need to
  over-launch and **screen for seeds that learn BOTH A and B**, the §8 confound).
- **Stability qualifier (carried from Stage-0):** report best-checkpoint AND a stability flag — the
  single-task oscillation means "peaked once" ≠ "competent"; A_before must be a *stable* competence, not
  a transient spike.

## 5. Phase 1 — forgetting protocol + mechanism instrumentation

Per seed:
1. **Train A** to stable competence. Record **A_before** = mean A success over the last K evals at
   competence (stable, not peak-spike). Snapshot the A-policy + a set of **A-critical-states** (states on
   A's successful path; reuse the rewind-probe capture).
2. **Train B** (continue the same agent, no A in the buffer) to competence.
3. **Throughout B-training:** every N steps, (a) **re-eval on A** (retention curve), (b) instrument at
   the A-critical-states: **A-policy argmax-margin** (logp(A-action) − logp(2nd)) and **WM-on-A health**
   (recon / next-state prediction error on A-states).
4. **A_after** = mean A success over the last K evals of B-training.

## 6. Metrics (pre-registered)

- **Forgetting (primary, Q1):** `A_before − A_after`, judged against the **cross-seed noise band**
  (forgetting = drop beyond the band, measured from the screen's own seeds — the Stage-0 rule; NO fixed
  absolute threshold). Within band = NO forgetting.
- **Retention curve:** A-success vs B-training step (gradual decay vs cliff vs none).
- **Mechanism (Q2):** trajectories of (i) A argmax-margin and (ii) WM-on-A error across B-training —
  *which moves first / at all* as A forgets.

## 7. SEALED READ — verdict table (locked NOW, before data)

**Q1 — does forgetting occur?**
- A drop **beyond** the cross-seed band → forgetting PRESENT (benchmark valid, proceed to Q2).
- A drop **within** the band → NO forgetting → MiniGrid A→B is not a forgetting benchmark either
  (DMC-walker-like). Important negative → reconsider the task pair (§3) or the whole continual framing.

**Q2 — IF forgetting present, which mechanism (read the instrumentation):**

| Pattern at A-critical-states during B-training | Verdict |
|---|---|
| **A argmax-margin collapses while WM-on-A stays healthy** | **actor-drift / argmax-flip** → SAME mechanism as single-task → **the localization unifies single-task collapse and cross-task forgetting** (strong on-thesis result) |
| **WM-on-A error grows** (recon/prediction degrades) | **representation overwrite** → DIFFERENT mechanism → localization does NOT transfer → fix direction = representation protection (anchoring/EWC/replay — the project has infra) |
| **both move** | co-mechanism — report the leader (earliest departure), same first-mover method as Stage-1 |

**Anti-spin (locked):** A_before is *stable* competence, not a peak spike (the oscillation trap). n=1
confirms nothing — Q1 and Q2 both require ≥2/3 seeds. "A dipped then recovered" is not forgetting unless
the end-state A_after is down beyond the band.

## 8. Confounds / caveats (pre-registered)

- **Basin lottery (known, severe on this substrate):** need seeds that learn BOTH A and B. Screen via
  Phase-0; over-launch and keep qualifiers. Do NOT assume "confirmed under one config" transfers (Stage-3
  lesson — it didn't).
- **Interference vs transfer:** if B builds on A (positive transfer), forgetting won't show — that's a
  task-pair problem, not a null result about forgetting. §3 swap rule handles it.
- **Stability qualifier:** the single-task oscillation contaminates A_before/A_after if read as a spike;
  read stable windows.
- **Eval is argmax (deterministic)** — small A-policy drift can cliff A-eval (the argmax-flip we
  localized); the argmax-margin instrument is exactly the right lens.

## 9. Build sequence (gated; pre-reg is step 0)

0. **Lock this doc** (sign-off) before code. ← we are here
1. **PREREQUISITE — build the Q2 instrumentation (does NOT exist yet; Q2 is UNMEASURABLE without it).**
   The trainer logs entropy/grad-norm, NOT action-margin at specific states. Build + CPU-smoke a logger
   that, at a fixed set of **A-critical-states**, records each eval-interval: (i) **A argmax-margin** =
   logp(current_actor, A-good-action) − logp(2nd-best), and whether A-good-action is still the argmax;
   (ii) **WM-on-A error** = recon / next-state prediction error on A-states. Reuse the rewind-probe
   capture (A-critical-states + A-good-actions) + eval helpers. **Gate:** logger runs, both signals
   compute on a held A-policy, RNG-neutral. STOP if it can't be built — Q2 falls back to Q1-only.
2. Adapt `orchestrator_forgetting.py` → MiniGrid A→B + A-retention curve, wiring in the §9.1 logger.
3. **Phase 0** learnability gate (1-seed fail-fast each task → n=3). STOP + report; finalize the task pair.
4. **Phase 1** forgetting runs, n≥3 (screened seeds). Read against §7.
5. Bank the sealed verdict (supersede-in-place; re-hash; sync).

*Note: §9.1 (argmax-margin logger) is the make-or-break build item — a locked pre-reg whose headline Q2
can't be measured is worthless. Build and smoke it BEFORE Phase 0; if it proves infeasible, Q2 degrades
to Q1-only (benchmark without mechanism) — a deliberate downgrade, not a silent one.*

---

*Drafted 2026-06-19. Changes to §3–§7 after lock invalidate the pre-registration (new date, not an edit).
Carries the Failure-Localization findings as the Q2 hypotheses; this is the test v1.0 is defined by.*
