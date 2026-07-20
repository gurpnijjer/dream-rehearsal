# TUNNEL_CHAIN8_RESULT_2026-07-17.md — 8-task chain CONFIRMED (≥2/3 bar met; s3 pending)

**Pre-reg:** TUNNEL_CHAIN8_PREREG_2026-07-15.md (+ AMENDMENT 1, disclosed roster change —
first-four-passers, the stricter direction). **Primary bar: all EIGHT tasks ≥0.6 final in ≥2/3
seeds. MET at 2/2 completed seeds.** Seed 3 in P5 at time of writing; it can raise the tally to
3/3 but cannot change the verdict. Artifacts: `tunl8_s{1,2}/chain_summary.json`.

Roster (six distinct mechanics, three lethal): DoorKey-5x5 → SimpleCrossingS9N1 → LavaGapS5 →
MultiRoom-N2-S4 → LavaCrossingS9N1 → DistShift2 → DoorKey-6x6 → Unlock.
Single new variable vs tunl4b: chain length 8. Recipe unchanged (tunnel rehearsal, lex grading,
50 updates/prior task/2k steps, bar 0.6, cap 150k, uniform allocation — allocator NOT used).

## Result

| Seed | DoorKey5 | SimpleX | LavaGap | MultiRoom | LavaCross | DistShift | DoorKey6 | Unlock | verdict |
|---|---|---|---|---|---|---|---|---|---|
| s1 | 0.964 | 0.943 | 0.947 | 0.759 | 0.861 | 0.960 | 0.915 | 0.861 | **PASS 8/8** |
| s2 | 0.963 | 0.919 | 0.943 | 0.746 | 0.865 | 0.929 | 0.962 | 0.888 | **PASS 8/8** |
| s3 | 0.901 | 0.876 | 0.942 | 0.754 | 0.794 | 0.959 | 0.962 | 0.740 | **PASS 8/8** |

**FINAL: 3/3 UNANIMOUS.** All three seeds retain all eight tasks. T1 (SimpleCrossing) mean across
3 seeds = 0.913 (0.943/0.919/0.876). Every task's min across seeds ≥ 0.74 (Unlock s3). s3 is the
slow-draw seed (as at 4 tasks) and its values run slightly lower but all clear the bar.

One live actor. No task labels at training or eval. No frozen policies, no router, no parameter
growth. By the final phase, rehearsal runs 350 dream-imitation updates per 2k-step chunk across
seven prior tasks.

## Secondary reads (pre-registered)

**1. Cross-seed consistency is high — and far better than the isolation reference.** Per-task
s1↔s2 agreement is within ~0.03 on 6/8 tasks (max gap 0.047, DoorKey-6x6). Contrast the
frozen-heads composite at FOUR tasks: T1 delivery 0.62 ± 0.13, with two seeds flipping
PASS↔FAIL between two reads of the same trained agent (ROUTER_ALBUM_BUG outcome). At 2× the
task count, dream rehearsal is materially more stable than isolation was at 1×.

**2. The rehearsal-dilution guardrail did NOT trip — it inverted.** The pre-reg's predicted
scale-failure was "incumbent quartet degrades vs the tunl4b band as rehearsal spreads thinner."
Observed for the historically-doomed T1 (SimpleCrossing):

| | 4-task (tunl4b, lex) | 8-task (tunl8) |
|---|---|---|
| T1 per seed | 0.868 / 0.751 / 0.825 | 0.943 / 0.919 (s3 pending) |
| T1 mean | 0.815 | **0.931** (n=2) |

Doubling the task count *improved* the hardest incumbent by ~0.12. The whole incumbent quartet
holds at or above its 4-task band (DoorKey ~0.96 both; LavaGap 0.947/0.943 vs 0.899/0.944/0.784;
MultiRoom 0.759/0.746 vs 0.751/0.800/0.768). **Rehearsal dilution is not present at n=8**, so the
stability-weighted rehearsal allocation is NOT triggered and stays parked.
Interpretation (INFERENCE, not measured): later tasks' rehearsal keeps the actor near a
multi-task competent region, and a longer chain simply means more rehearsal passes over T1
before the final read. A dedicated test would be needed to claim backward transfer.

**3. Ceiling behavior, not uniform decay.** Each task settles near its own difficulty ceiling:
easy tasks ~0.96, MultiRoom ~0.75 (its band in every prior chain), LavaCrossing ~0.86. Note
LavaCrossing's solo-audition stop was 0.69 — but that audition used the same early-stop rule, so
0.69 is where training halted, not a ceiling. No clean claim available; a matched-budget solo
control would be required.

## Reading

**Length-scaling holds: the recipe doubles from 4 to 8 tasks with no modification, no dilution,
and improved consistency.** Controls unchanged: plain replay 0/3 at four tasks; isolation
reference 3/5 at four tasks with coin-flip verdicts. Paper §7 gains its table.

Claim boundary (unchanged): MiniGrid, 17M-param WM, one ordering, greedy exploration; the four
audition failures (MultiRoom-N4, SimpleCrossingS11N5, FourRooms — long-horizon sparse) mark the
acquisition frontier of this study and are an exploration problem, not a retention one.

**Next:** s3 completes the n=3 record; the real-BC ablation (REAL_BC_ABLATION_PREREG_2026-07-16)
fires automatically after and answers "why dream at all?" — the last science before arXiv.
