# TUNNEL_CHAIN8_PREREG_2026-07-15.md — 8-task tunnel chain (scale-up axis 1)

**Committed BEFORE audition results are complete — the roster SELECTION RULE is locked here so
no cherry-picking is possible. Launch is automated (launcher waits for auditions, applies the
rule, verifies, fires) per ESCALATION_LADDER_DESIGN §B: automation only executes protocol
registered in advance.**

## Roster rule (locked)

Tasks 1–4: the incumbents, same order (DoorKey-5x5, SimpleCrossingS9N1, LavaGapS5,
MultiRoom-N2-S4) — comparability with all banked 4-task results.
Tasks 5–8: of the audition passers (all_good within the 100k solo budget; pool = 6 primary +
3 backup candidates, artifacts audit_*/chain_summary.json), take the FOUR with the fewest
steps_used, appended in ascending order (easiest first). Ties break alphabetically.
If fewer than 4 passers exist: DO NOT LAUNCH — human decision (7-task variant needs a fresh
amendment).

## Protocol (single new variable vs tunl4b: chain length 8)

tunl4b recipe exactly: tunnel rehearsal, lex grading (--tunnel_rehearsal --cont_grading),
50 rehearsal updates per prior task per 2k-step chunk (UNIFORM — the allocator is explicitly
NOT used; its trigger is a failure row below), bar 0.6, phase cap 150k, eval_every 2000,
seeds 1–3 sequential, logdirs tunl8_s{1,2,3}.

## Registered read

**PRIMARY: final retention ≥ 0.6 on ALL EIGHT tasks (last-3-eval mean, live actor, no labels)
in ≥ 2/3 seeds.**
**SECONDARY (locked):** per-task solve-stability; incumbent-four retention vs their tunl4b
band (dilution check); rehearsal + eval wall-clock share vs task index (linearity check);
acquisition-fail vs retention-fail typing per the established rule (a task that never reaches
bar within cap = ACQUISITION-FAIL, distinct from forgetting).

| Outcome | Reading | Next |
|---|---|---|
| ≥2/3 pass all eight | Recipe scales 2× in task count. | Paper §7; harder-domain axis. |
| Fails via RETENTION sag on early tasks as count grows | Rehearsal dilution — the predicted scale failure. | Pre-reg stability-weighted allocation (ESCALATION_LADDER §A.4). |
| Fails via ACQUISITION on new tasks in-chain (that passed solo) | Interference with acquisition — new phenomenon. | Stop; diagnose before any arm. |
| Guardrail: incumbent quartet degrades vs tunl4b band even in passing seeds | Silent dilution. | Report honestly; allocator arm. |

n=3, no per-seed tuning. Wall-clock estimate 18–30 h/seed. Eval episodes retained on disk
(they double as press-kit GIF material — no protocol effect).

## AMENDMENT 1 (2026-07-15, WITH partial audition results visible — disclosed)

**Change:** roster tasks 5–8 = the FIRST FOUR passers by audition-completion time (not the four
with fewest steps_used). Launch fires as soon as the 4th passer exists; remaining auditions are
killed to free the GPU.

**Disclosure:** this rule was changed after 6 of 13 auditions had reported (passers: DistShift2
@12k/0.95, LavaCrossing @38k/0.69, DoorKey-6x6 @44k/0.90; failures: MultiRoom-N4,
SimpleCrossingS11N5, FourRooms). Pre-registration exists to prevent exactly this, so the
justification must stand on its own:

1. **The original rule was self-defeating.** "Fewest steps" would have displaced the two hardest
   passers (LavaCrossing, DoorKey-6x6) with the trivial tier-2 fillers (Empty-8x8 learns in
   ~2k steps). The roster would have been engineered easy — a weaker test dressed as a scale-up.
2. **The new rule is STRICTLY HARDER, i.e. the anti-cherry-picking direction.** It retains
   LavaCrossing, whose own solo competence was 0.69 — barely above the 0.6 retention bar it must
   now clear after four more tasks of interference. No outcome measure, bar, or seed count
   changes; only the roster gets harder. A change that makes one's own claim more difficult to
   achieve cannot inflate it.
3. **Known risk, pre-registered now:** if LavaCrossing fails retention, the ACQUISITION-MARGINAL
   confound applies — a task whose solo competence is ~0.69 has little margin to retain. That
   failure must be reported as "marginal-acquisition task," not as clean forgetting, and does not
   by itself refute length-scaling. Same typing rule as the banked acquisition/retention split.

Everything else (protocol, primary bar of all-eight ≥0.6 in ≥2/3 seeds, secondaries, failure
matrix, seeds 1–3) is unchanged.
