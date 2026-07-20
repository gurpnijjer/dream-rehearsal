# REAL_BC_ABLATION_PREREG_2026-07-16.md — why dream at all? (the reviewer's question)

**Committed before the tunl8 sweep finishes (s1 PASS 8/8; s2 mid-P4; s3 queued). This arm is
adversarial toward our own claim 3 and is registered as such.**

## The question

The never-clear buffer ALREADY holds every prior task's real episodes, with the actions the
agent actually took. So: **why imagine anything?** Behavior-cloning the stored real episodes is
simpler, cheaper, and needs no world model at eval time. If it works as well, "dream rehearsal"
collapses toward **CLEAR** (Rolnick et al. 2019), which already does behavior cloning on a replay
buffer for continual RL — and claim 3's novelty largely evaporates (NOVELTY_SCAN_2026-07-16
ranks this the most exposed claim). Every reviewer will ask this. We ask it first.

## Arms (single variable: what the actor imitates; everything else identical to tunl4b)

Protocol = the confirmed 4-task recipe (DoorKey-5x5 → SimpleCrossingS9N1 → LavaGapS5 →
MultiRoom-N2-S4; bar 0.6; eval_every 2000; phase cap 150k; 50 rehearsal updates per prior task
per 2k-step chunk; seeds 1–3). 4 tasks, not 8 — this isolates the mechanism against our
best-characterized banked baseline (tunl4b: 3/3, T1 mean 0.815, T2 stability 95/100/67%).

- **DREAM (banked control, no re-run):** tunl4b. Imagine from prior-task starts with the current
  actor, grade with lex (realized-first), clone top-25%.
- **REAL-BC (new arm, `--rehearsal_source real`):** identical schedule and update count, but
  clone the RECORDED actions from the SAME prior-task episodes the dream arm starts from — no
  imagination, no grading. This is the CLEAR-style rehearsal, matched update-for-update.
- **REAL-BC-FILTERED (new arm, `--rehearsal_source real_filtered`):** as above, but only from
  episodes whose realized episode return > 0.05 (competent-only). This is the STRONGEST fair
  version of the real-data alternative and the one a reviewer will actually champion —
  it removes the "you cloned the agent's beginner flailing" objection.
  Logdirs: `realbc_s{1,2,3}`, `realbcf_s{1,2,3}`.

## Registered reads

**PRIMARY (the claim-3 test): does DREAM beat the better real arm on T1 cross-seed mean
retention?** Bar for a distinctness claim: DREAM T1 mean (banked 0.815) exceeds
max(REAL-BC, REAL-BC-FILTERED) T1 mean by **≥ 0.10**, with ≥2/3 seeds passing all-four in the
dream arm.
**SECONDARY:** all-four pass counts per arm; T2 (lethal, short-episode) solve-stability per arm;
wall-clock per arm (real-BC is cheaper — if it ties, it WINS on cost and we must say so).

## Interpretation matrix (locked — note that two of four rows are bad for us)

| Outcome | Reading | Consequence for the paper |
|---|---|---|
| DREAM > both real arms by ≥0.10 | Imagination contributes beyond the supervised channel — fresh on-policy coverage from the current WM + gradability. | Claim 3 stands as "dream rehearsal"; §5 gains its key ablation; CLEAR distinguished empirically, not rhetorically. |
| DREAM ≈ real arms (within 0.10) | **The supervised CHANNEL is the whole story; imagination is optional.** Claim 2 is vindicated and STRENGTHENED; claim 3 must be reframed. | Retitle the method honestly (e.g. "self-imitation rehearsal"), report real-BC as the simpler equal, cite CLEAR as near-prior-art, and lead the paper harder on claims 1–2. Simpler method = better paper, weaker novelty. Report either way. |
| Real arm > DREAM | We built the wrong thing; the simpler method wins outright. | Report it. The mechanism story (claims 1–2) survives intact and is still the contribution; dream rehearsal becomes a negative result we publish. |
| Both real arms collapse (< plain-chain 0/3 band) | BC on real off-policy data destabilizes the actor. | Interesting in itself; strengthens the imagination argument. |

**Pre-commitment:** we publish this table's outcome whichever row lands, in the paper's main
body — not an appendix. The result that makes our method look worse is the one most worth
reporting, since we'd otherwise be advertising an unnecessary mechanism.

n=3, no per-seed tuning, no post-hoc bar changes. Any parameter change is a new pre-reg.
Runs after tunl8 completes (GPU serialized).

## AMENDMENT 1 (2026-07-17) — decision rule strengthened BEFORE any ablation data exists

**Provenance: zero ablation runs have started** (the sweep is still queued behind tunl8 seed 3;
verified: no `realbc*/chain_summary.json` exists, no arm process running). This amendment is
therefore a pre-data protocol change, not a goalpost move. Prompted by external critique.

**The flaw in the original rule.** The ≥0.10 margin is inside the noise it must resolve. The
dream arm's own banked T1 values are 0.868 / 0.751 / 0.825 — spread **0.117**, s ≈ 0.059,
SEM(n=3) ≈ 0.034; the difference-of-means SE for two n=3 arms is ≈ **0.048**. A 0.10 threshold is
~2 SE: noise clears it too often, and a true 0.10 effect is missed too often. This reproduces
exactly the pathology the 2026-07-10 five-expert audit identified in the 0.6 competence bar
(T1 = 0.62 ± 0.13 → "bar-crossings are round-noise"). A rule that cannot decide the question it
exists to decide is not a rule.

**Amended read (replaces the PRIMARY above; bars for the ARMS themselves unchanged):**
1. **Report per-seed values, arm means, and a paired bootstrap 95% CI on the difference**
   (DREAM − best real arm), resampling seeds with replacement, 10k draws. Point estimates alone
   decide nothing.
2. **DISTINCT** (imagination contributes): the difference CI **excludes zero** AND the point
   estimate ≥ 0.10.
3. **TIE** (channel is the story; imagination optional): CI includes zero AND |difference| < 0.10.
   → Reframe per the original matrix: rename the method, report real-BC as the simpler equal,
   cite CLEAR as near-prior-art, lead on claims 1–2.
4. **INCONCLUSIVE** (new row, and the honest one at n=3): CI includes zero AND |difference| ≥ 0.10
   — i.e. the arms may differ but this experiment cannot tell. → **Occam governs: we do NOT claim
   imagination helps.** Report as underpowered, state the seeds needed for 80% power at the
   observed variance, and either run them or ship the paper with the simpler framing. An
   unresolved comparison defaults AGAINST our own mechanism, never toward it.
5. **Real arm > DREAM** (CI excludes zero, negative): unchanged — we built the wrong thing;
   report it.

**Standing consequence:** the paper's mechanism framing is PROVISIONAL until this reads. The
title's localization claim (§3–4) does not depend on it; the §5 mechanism narrative does. If TIE
or INCONCLUSIVE lands, §5 gets rewritten, not annotated.

## AMENDMENT 2 (2026-07-18) — reorder: run the DECISIVE (filtered) arm first

**Scheduling change only, no result affected. Disclosed.** Prompted by a mid-run observation (a
scheduling input, NOT a verdict): unfiltered real-BC seed 1 drove DoorKey (T0) to 0.0 by step
48k of P1 and held it there through 74k — persistent collapse, not a flicker. Mechanistic (this
is reasoning, not a claim): unfiltered real-BC clones ALL recorded phase-0 actions including the
pre-competence random-policy episodes, actively teaching bad play. This is the expected failure
of naive replay-cloning and is exactly why the FILTERED (competent-only) arm exists — and why
filtered, not unfiltered, is the decisive comparison (verdict = DREAM vs max(real, filtered), and
filtered ⊇-dominates unfiltered in cleanliness).

**Actions:**
1. Killed the running unfiltered sweep (realbc). realbc_s1.log preserved as
   realbc_s1_partial_killed.log for the "naive-BC destabilizes" observation (DoorKey 0.0 @48–74k).
2. Run REAL-BC-FILTERED ×3 next (the decisive arm), concurrent with another queued sweep.
3. Complete unfiltered ×3 AFTER filtered. Once the filtered verdict is read against the CI,
   reassess whether the full unfiltered n=3 is worth its (capped, multi-day) cost or whether the
   preserved seed-1 observation suffices to report "naive cloning destabilizes."

**Discipline note:** reordering reacts to a mid-run read but changes NO registered outcome — the
filtered verdict is identical regardless of run order, and unfiltered is non-decisive. This is a
scheduling optimization, explicitly not a verdict or a bar change. The amended read (bootstrap CI,
Occam-defaults-against-us) is untouched.

## AMENDMENT 3 (2026-07-18) — real_bc_update had a TEMPORAL-ALIGNMENT BUG; all prior ablation data VOID

**The "real-BC collapses everything" result (both arms, DoorKey→0.0) was an ARTIFACT, not a
finding — caught by a verify-the-gauge isolation test, exactly the discipline for a favorable
extreme result (it made our method look NECESSARY; that is when to scrutinize hardest).**

Isolation test (realbc_isolation_test.py): a DoorKey-COMPETENT agent (0.965), WM frozen, hammered
with real_bc updates on DoorKey's OWN competent episodes. It DESTROYED DoorKey (0.965→0.0 within
~10 updates — a cliff, not a slope), while the dream update on the same states PRESERVED it. A
sound BC on a skill's own success cannot destroy it → the update was buggy.

**Root cause (measured):** the RSSM's `observe()` action convention lags the actor's decision by
one step — the actor's argmax at feat[t] matches the RECORDED action at t+1 (agreement 0.96
shifted vs 0.30 unshifted; actor is 98% peaky, so this is misalignment, not stochasticity).
Unshifted, real_bc cloned the actor toward actions it never took (0.29 prob) → destructive
gradients → collapse. tunnel_update (dream) never hit this because imagined feat+action are
generated together, aligned by construction — which is ALSO why the banked dream results
(tunl4b 3/3, tunl8 3/3) are UNAFFECTED: only the real-BC comparison arm was broken.

**Fix:** BC feat[t] toward act[t+1] (feat[:, :-1] vs act[:, 1:]). Isolation test after fix:
real_bc PRESERVES DoorKey (0.965→0.961 over 300 updates). Update is now correct.

**Consequence:** all realbc*/realbcf* data deleted. Ablation RE-RUN with the fixed update. And a
sobering note: the bug FLATTERED us — with real-BC now working correctly, the honest comparison
may well be a TIE (real-BC preserves skills fine), which per AMENDMENT 1 forces the method to be
renamed and claim 3 reframed. We are back to genuine uncertainty about the framing, which is
correct. The bug had been hiding a possibly-unflattering answer.

## RESULT (2026-07-20) — REGISTERED VERDICT: DISTINCT (imagination contributes)

Per-seed T1 (SimpleCrossing) retention, FIXED update (Amendment 3):
- DREAM (tunl4b banked): 0.868 / 0.751 / 0.825 (mean 0.815), all-four 3/3
- REAL-BC-FILTERED:      0.630 / 0.678 / 0.743 (mean 0.684), all-four 3/3

Registered read (Amendment 1): paired diffs +0.238/+0.073/+0.082; point estimate **+0.131**;
paired bootstrap 95% CI **[0.073, 0.238]** (10k draws, seed 20260720) — CI excludes zero AND
point ≥ 0.10 → **DISTINCT**. Stronger than the rule requires: COMPLETE SEPARATION — every dream
seed exceeds every filtered-copying seed (min dream 0.751 > max real 0.743).

**Both honest halves:** (1) fixed real-BC WORKS — 3/3 all-four; competent-episode cloning is a
viable, cheaper continual method and we report it as such. (2) It retains LESS than dream
rehearsal, consistently. Imagination is contributing beyond the supervised channel — plausibly
fresh on-policy coverage from the current WM + gradability. Claim 3 stands as "dream rehearsal";
§5 framing UNFREEZES with CLEAR distinguished empirically at n=3.

**Unfiltered arm decision (per Amendment 2 reassessment):** NOT RUN with the fixed update (all
bug-era data void per Amendment 3). Rationale: filtered clones a strictly cleaner subset of the
same data; unfiltered adds pre-competence flailing and cannot plausibly exceed filtered. Cost of
closing it: ~2-3 GPU days against other queued experiments. DISCLOSED as unrun in the paper; the
verdict is vs the STRONGEST real arm, which is the comparison that matters.
