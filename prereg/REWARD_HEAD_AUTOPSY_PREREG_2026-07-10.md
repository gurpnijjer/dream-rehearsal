# REWARD_HEAD_AUTOPSY_PREREG_2026-07-10.md — where does T1 forgetting live? (offline, pre-registered)

## Puzzle

The live actor forgets T1 (plain chain 0/3, always T1) even though the never-clear buffer keeps
replaying T1 data and actor training imagines from replayed (incl. T1) states. Practice that
doesn't stick ⇒ the training signal inside those imagined rollouts is the suspect. The margin
probe agrees something T1-specific decays WM-side (final comp4b_s2 row: m1_frozen = −0.25 vs
m0 = +4.5, m2 = +2.7). Prime suspect: **the shared reward head forgets what T1 success looks
like**, turning T1 dreams into no-gradient noise. Secondary suspect: the (live, shared) critic.

## Method (offline; checkpoints + stored episodes; no new rollouts; sweep untouched)

Data: per task j, the 20 competent eval episodes at phase-j end (newest-20 mtime rule).
Checkpoints: chain_phase{p}.pt for every p ≥ j (agent state at each later phase end).
Encode: deterministic `_encode_feats` (posterior mode; the margin-probe gate path).

**Registered metric** — reward discrimination of ckpt-p WM on task-j episodes:
D(j,p) = mean(predicted reward at true-success steps [actual > 0.05])
       − mean(predicted reward at zero-reward steps).
**Retention ratio R(j) = D(j,3) / D(j,j).** Exploratory (not gating): critic mean value on the
same episodes across p, same construction.

## Predictions (locked)

- **H-reward:** R(1) < 0.5 while R(0), R(2) ≥ 0.7 → T1 reward-knowledge is selectively
  overwritten; the actor's dream-practice on T1 is unscored. Fix path = protect/rehearse
  reward knowledge (dream-rehearsal arm).
- If instead R(1) ≥ ~0.8: reward head is NOT the bottleneck → suspicion moves to critic
  (exploratory read), then actor-side plasticity. Do NOT proceed to a reward-protection arm.
- Sanity gate: T0 must show healthy D(0,3) (its routed retention is 0.96 — if the metric says
  its reward signal is gone, the METRIC is broken, stop and fix before interpreting).

Seed for this read: comp4b_s2 (bug-era albums are irrelevant here — no routing involved).
n=1 seed is a DIRECTION-FINDING read, not a claim; if H-reward holds, replicate on s1/s3 before
building anything.
