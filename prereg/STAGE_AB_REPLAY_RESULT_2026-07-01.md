# STAGE_AB_REPLAY_RESULT_2026-07-01.md

**Date:** 2026-07-01
**Status:** Replay-fraction sweep COMPLETE (n=3 × f∈{0.1, 0.25}). One-off f=0.5 s1 in flight. Higher-fraction and shared-mode sweeps CANCELLED (see §7).
**Source data:** `~/dv3_logs/ab_rp{0.1,0.25}_s{1,2,3}/ab_summary.json` on Spark. Baseline (no replay) = `ab_fl_s{1,2,3}` (banked in STAGE_AB_LOCATE_RESULT_2026-06-21).
**Pre-reg:** `docs/STAGE_AB_LOCATE_PREREG_NM512.md` §12.
**Predecessor:** `docs/STAGE_AB_LOCATE_RESULT_2026-06-21.md` (Q1 catastrophic-forgetting + Q2 actor-dominant mechanism).

---

## 0. One-line verdict

**Replay works, but not reliably: at f=0.25 both bars pass in 2/3 seeds. Retention is dose-dependent in *probability* but dose-independent in *stability* — the DoorKey policy flickers between fully-solved and fully-failed at every fraction tested. Replay is fighting the symptom; the actor is the load-bearing failure.**

---

## 1. Setup

| Parameter | Value |
|---|---|
| Task A | `minigrid_DoorKey-5x5` (dense reward, learns instantly under greedy) |
| Task B | `minigrid_SimpleCrossingS9N1` |
| Replay mode | `clear` — clear B replay at switch, mix in raw task-A episodes at fixed fraction (§9) |
| Fractions swept | f ∈ {0.1, 0.25}, n=3 each; f=0 baseline = existing `ab_fl_s{1,2,3}` |
| Exploration | greedy |
| eval_every / b_max_steps | 2000 / 100000 |
| Seeds | {1, 2, 3} per fraction |
| Success bar (pre-reg §12, BOTH required) | A_recovers (A_after ≥ 0.6) AND B_learns (B_peak ≥ 0.6) |
| Solve-stability metric | fraction of B-evals with A_ret ≥ 0.6 (catches the flicker, not just the endpoint) |

---

## 2. Full results table

| Fraction | Seed | A_after | Solve-stability | B_peak | B_final | Both bars? |
|---:|---:|---:|---:|---:|---:|:---:|
| **0** (baseline) | s1 | 0.000 | 0/50 | 0.91–0.95 | — | ❌ (A) |
| **0** | s2 | 0.000 | 0/50 | 0.91–0.95 | — | ❌ (A) |
| **0** | s3 | 0.000 | 0/50 | 0.91–0.95 | — | ❌ (A) |
| **0.1** | s1 | 0.511 | 29/50 (58%) | 0.954 | 0.600 | ❌ (A) |
| **0.1** | s2 | 0.957 | 50/50 (100%) | 0.869 | 0.511 | ✅ |
| **0.1** | s3 | 0.000 | 13/50 (26%) | 0.957 | ~0.95 | ❌ (A) |
| **0.25** | s1 | 0.000 | 27/50 (54%) | 0.951 | — | ❌ (A) |
| **0.25** | s2 | 0.642 | 36/50 (72%) | 0.960 | — | ✅ |
| **0.25** | s3 | 0.962 | 45/50 (90%) | 0.920 | — | ✅ |
| **0.5** | s1 | *KILLED at 33/50 evals (partial, no endpoint): 14/33 A-solved (42%), 2 flicker transitions, B_peak 0.863, last-5 A_ret ≈ 0.96. Consistent with dose-independent flicker; killed to free GPU for the frozen-head arm.* |

### Summary by fraction

| Fraction | Both-bars pass | Mean solve-stability | Mean A_after | Mean B_peak |
|---:|:---:|---:|---:|---:|
| 0.0 | 0/3 | 0% | 0.000 | ~0.93 |
| 0.1 | **1/3** | 61% | 0.489 | 0.927 |
| 0.25 | **2/3** | 72% | 0.535 | 0.944 |
| 0.5 (n=1 partial) | *TBD* | *34% so far* | *TBD* | *TBD* |

---

## 3. What the data says

### (a) Replay is real recovery, not an artifact
Every f>0 run recovers to A_ret ≥ 0.9 at some point during B-training (versus flat 0 for baseline). The recovered runs hit the real-env goal, not just probe-healthy latents.

### (b) More replay raises the probability of holding A
- Both-bars pass rate: **0.1 → 0.25 = 1/3 → 2/3**
- Mean solve-stability: **61% → 72%**
- Mean A_after climbs monotonically across seeds at f=0.25: {0.00, 0.64, 0.96}

### (c) More replay does NOT eliminate the flicker
At every fraction tested, at least one seed shows A_ret oscillating between fully-solved (~0.96) and fully-failed (~0.00) eval-to-eval:
- f=0.1: s3 flickered (26% stable, ended at 0)
- f=0.25: s1 flickered (54% stable, ended at 0)
- f=0.5 s1 (partial): went to 0 for a stretch, then recovered to 0.96 — same shape

**The flicker signature is dose-independent.** That's the load-bearing finding for choosing the next arm.

### (d) B does NOT starve at peak level
Feared "B down to ~0.6" from early n=1 read did not materialize:
- B_peak stays healthy across all seeds and fractions (0.87–0.96)
- No monotonic decline in B_peak as f rises (f=0.25 mean B_peak 0.944 ≥ f=0.1 mean 0.927)
- B_final drifts down in some seeds (plasticity cost shows here, not at peak)

**Peak-level plasticity is preserved. Cost is in end-of-training retention of B, not in reaching B competence.**

### (e) One n=1 read was refuted by n=3
Off f=0.25 s1 alone (A_after 0.000, solve-stability 54%), I wrote: *"doubling replay didn't tighten the basin."* The n=3 result overturned that — {54, 72, 90}% solve-stability with 2/3 both-bars passing. Discipline #2 (n=1 is hypothesis) re-earned its keep. Do not read replay verdicts off endpoints alone.

---

## 4. The flicker as a phenomenon

At every f tested, seeds fall into two visible regimes:

- **Held basin:** A_ret ≈ 0.96 rock-steady across all B-evals (e.g. f=0.1 s2)
- **Flicker basin:** A_ret oscillates between ~0.96 and 0.00 eval-to-eval, sometimes ending at 0 (e.g. f=0.1 s3, f=0.25 s1, f=0.5 s1 so far)

Same code, same fraction, different seed → different regime. Basin outcome is determined early in B-training and persists (once flickering, stays flickering; once held, stays held) — matches the "riding the edge of competence" reading. Consistent with actor-dominant mechanism from Q2: the actor is drifting under B's gradient, and even 25–50% task-A rehearsal doesn't produce enough policy signal to hold DoorKey preferences against SimpleCrossing overwrites in unlucky basins.

**Prediction from Q2 (70% actor / 30% representation):** if we remove actor drift by protecting the actor half, the flicker should collapse. Frozen-head-A + fresh-head-B is the direct test.

---

## 5. Probe cross-check (secondary; not the bar)

Margin probe stayed healthy at evals where real-env A_ret = 0:
- Baseline frozen margin ~+6, degraded to +3.3 (f=0.1 s1) or +4.4 (f=0.1 s2) — small drop
- Baseline wm_on_A ~0.22 → ~0.08 (f=0.1 s1) or ~0.14 (f=0.1 s2) — probe reads representation as *healthier* under replay
- These signals do not track real-env retention seed-to-seed

**Probe blind-spot confirmed at n=6. Real-env A_ret is the fix bar. Probe stays diagnostic-only.**

---

## 6. Verified against code (VERIFY THE GAUGE)

Replay implementation confirmed by reading `orchestrator_ab_nm512.py::make_mixed_dataset` (lines 125–147):
- f>0: yields batches of exactly `n_a = round(f × batch_size)` task-A sequences + `batch_size − n_a` task-B sequences, per batch, every step
- Task-A sequences are **raw episodes** from `tools.sample_episodes(a_eps, ...)`, re-encoded live through the current WM — NOT cached latents
- f=0: structural delegation to unchanged `make_dataset(b_eps)` — the f=0 ≡ baseline gate is code-level, not just asserted
- effective_fraction (`n_a / batch_size`) is logged per run; f_requested 0.1 → eff 0.125 (batch_size 16, n_a=2)

Concern "replay stale cached latents while WM drifts under B" is **moot** on this substrate — replay is raw-transition-re-encoded-live.

---

## 7. What we did NOT run, and why (killed queue)

`next_sweep.sh` had queued 9 more runs behind `ab_rp0.5_s1`:
- Arm 1: f=0.5 s{2,3} (finish the fraction extension)
- Arm 2: `replay_mode=shared` × 3 seeds (Continual-Dreamer never-clear protocol; A-fraction decays naturally)
- Arm 3: **Reversed pair** SimpleCrossing→DoorKey, no replay × 3 seeds (generalization: does the mechanism story replicate?)

**Killed the bash driver (PID 4035863) at 2026-07-01 ~18:35**, kept `ab_rp0.5_s1` Python process running. Rationale:

- **f=0.5 × n=3** would confirm the dose curve (already shape-known: probability rises, flicker persists) — low new information for ~9h GPU-time.
- **Shared mode × n=3** — a nicer replay design, but it's still replay. Answers a question we've already answered.
- **Reversed pair × n=3** — genuinely different (generalization of Q1+Q2 to another pair). *Should* be run, but AFTER the actor-protection arm, so we're testing generalization of a mechanism story that has a fix, not without one.

Discipline #6 ("don't queue fixes aimlessly"): bank the replay result, then let evidence pick the next single arm.

---

## 8. What replay is (and isn't)

**IS:**
- A validated real-recovery mechanism on this substrate (n=6, gate_ok, code-verified)
- The baseline BAR for cheaper/better methods: 2/3 both-bars at f=0.25, 72% mean solve-stability, B_peak preserved
- Sufficient to hold the world-model half of the mechanism (30% representation loss)

**IS NOT:**
- A reliable fix (1/3 seeds still lose A at f=0.25; possibly 1/3 at f=0.5)
- A fix to the actor-dominant half (70% of the collapse) — the flicker is dose-independent
- A contribution — this is the bar, per pre-reg. Anything we build must beat 2/3 both-bars @ f=0.25.

---

## 9. Open questions → next arm

**The question the data now asks:** if we hold the actor's task-A weights constant during B-training, does the flicker collapse?

**Prediction from Q2:** ~70% of the collapse (actor drift) vanishes → seeds that flickered under replay become held. If flicker persists, our 70/30 split is wrong in closed-loop and representation is worse than Q2 suggested.

**Proposed next arm — frozen-head-A + fresh-head-B (multi-head actor):**
- Shared world model (as now) + shared encoder
- Actor split: head_A frozen at A-competence, head_B trained on B
- Head selection: task ID at eval time (v1 limitation — noted)
- Same two-bar test as replay (A_recovers AND B_learns) + solve-stability read
- n=3, replay_fraction=0 (isolate actor protection from replay), same DoorKey→SimpleCrossing pair

**Success bar vs replay baseline:** ≥ 2/3 both-bars AND mean solve-stability > 72%. Preferably 3/3 with rock-steady A retention (no flicker).

**If flicker collapses** → CRADLE v1 concept validated: shared WM + protected policy heads.
**If flicker persists** → representation half is >30% in closed loop; the 70/30 split was probe-inflated (WM overwrite matters more than the held-latent probe showed). Redirect to WM-side protection.

Either result is a real move.

**Deferred until after this arm:**
- Reversed-pair generalization (still cheap, still needs to happen)
- `shared` replay mode (n=3)
- EWC / weight-anchoring (only if actor-protection also flickers)

---

## 10. Files

- **Orchestrator:** `orchestrator_ab_nm512.py` (with `--replay_fraction`, `--replay_mode {clear,shared}`)
- **Probe:** `nm512_margin_probe.py` (MarginLogger, live==frozen@competence gate)
- **Killed driver:** `next_sweep.sh` on Spark (PID 4035863, killed 2026-07-01 ~18:35)
- **Result logdirs:** `~/dv3_logs/ab_rp{0.1,0.25}_s{1,2,3}/`, `ab_rp0.5_s1/` (partial)
- **Baseline logdirs:** `~/dv3_logs/ab_fl_s{1,2,3}/`

---

## 11. Discipline notes

- **VERIFY THE GAUGE:** replay implementation code-read (§6). f=0 ≡ baseline gate structural, not asserted.
- **n=1 is hypothesis:** off f=0.25 s1 wrote "doubling replay didn't help" — refuted by s2/s3 (§3e). Kept in the doc as an honesty log.
- **Pre-registered read:** two-bar success bar locked before data, per §12 of pre-reg. Solve-stability metric caveat noted in pre-reg §12 addendum and applied.
- **BUILDING framing:** the doc frames replay as "does it fix forgetting" (it partially does) and "what does the residual tell us to build next" (actor protection). Not framed as novelty.
- **Real-env eval is the bar:** all verdict decisions off A_ret closed-loop, probe demoted to diagnostic (§5).

---

*Result doc written 2026-07-01. Next: pre-register frozen-head-A actor arm before launching.*
