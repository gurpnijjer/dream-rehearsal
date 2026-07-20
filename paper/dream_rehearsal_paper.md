# The World Model Remembers, the Actor Forgets: Dream Rehearsal for Continual Model-Based RL

**v1 full-prose draft, 2026-07-16; tunl8 3/3 and the DISTINCT ablation verdict filled
2026-07-20. Sole remaining slot: https://github.com/gurpnijjer/dream-rehearsal. All numbers final, from banked
pre-registered runs.**

## Abstract

Model-based reinforcement-learning agents in the DreamerV3 family forget catastrophically:
trained on a sequence of tasks, they lose earlier policies almost completely, even when an
unbounded replay buffer preserves every earlier experience. We ask a question the continual-RL
literature has assumed an answer to but never measured: *which component forgets?* Using
pre-registered probes and interventions on four-task MiniGrid chains (n=3 seeds throughout), we
find that under never-clear replay every measurable component of the world model retains
old-task knowledge — representations, reward predictions, and value estimates all remain intact
(reward-head retention ratio ≈ 1.0, including on the degraded policy's own failure states) —
while the actor's behavior collapses regardless. The training signal inside the agent's
imagination is intact; the policy-gradient channel that consumes it is what fails. We
demonstrate this by intervention: with the world model frozen and identical imagined rollouts
from replayed old-task states, re-teaching a lost skill by reinforcement learning in
imagination fails on 3/3 seeds, while supervised self-imitation on the world model's own graded
dreams recovers the skill on 3/3 seeds within 2,000–7,500 updates — using zero environment
interaction — to performance above what frozen-policy storage delivers. Interleaving this
*dream rehearsal* during sequential training yields a task-label-free, parameter-constant
continual learner: 3/3 four-task chains retained (hardest-task mean 0.82; 96% of
post-acquisition evaluations above bar) where plain replay passes 0/3 and a frozen-heads+router
isolation reference passes 3/5 — and holds across eight-task chains spanning six distinct task
mechanics (3/3 seeds retain all eight, unanimously; per-task cross-seed agreement within
~0.03), and outperforms matched competent-episode real-data cloning (paired difference +0.13,
bootstrap 95% CI [0.07, 0.24], complete seed separation) — imagination contributes beyond the
supervised channel. We further show the dream-grading step is load-bearing and characterize its failure
modes with an offline selection gauge: naive trajectory scoring is near-random on
short-horizon tasks, and termination-aware corrections invert selection on long-horizon ones;
a realized-first rule — imitate only what actually succeeded in imagination, an
imagination-safe form of the self-imitation advantage gate — achieves selection AUC 1.0 on
both task profiles and lifts the affected task's post-acquisition stability from 19–42% to
67–100%. All experiments were pre-registered with git-committed protocols before running; we
report every refuted hypothesis, including two scoring bugs our own gauges caught before they
contaminated results.

## 1. Introduction

An agent that learns task B by destroying its competence at task A cannot learn continually,
and reinforcement-learning agents do exactly this. The standard remedies treat the network as
the thing to protect: regularize its parameters (EWC and successors), grow it (progressive
networks, per-task heads), or replay its past data (CLEAR; reservoir buffers). In model-based
RL, the replay remedy has a natural home — the world model trains on buffered experience — and
the resulting recipe (a Dreamer-class agent with a never-clear buffer) is the field's current
working answer [Continual-Dreamer; WMAR; ARROW].

That line of work carries an assumption stated explicitly but, to our knowledge, never tested.
WMAR: *"We hypothesise that maintaining the world model's accuracy across different
environments will help preserve performance."* ARROW infers world-model fidelity backward from
behavioral metrics. The premise — protect the world model and the policy will follow — has not
been checked at the component level, because the standard evaluation measures only episodic
return, which cannot distinguish a forgotten representation from a forgotten behavior.

We checked. The first half of the premise is right: under never-clear replay the world model
retains essentially everything we can measure about old tasks. The second half is wrong: the
actor forgets anyway. The training signal for old tasks — accurate rewards, calibrated values,
intact dynamics over old-task states — is present inside the agent's own imagination throughout
training, and the policy decays regardless. Forgetting, in this regime, is not a memory
problem. It is a channel problem: policy-gradient learning fails to convert an intact signal
into retained behavior, and we show a supervised channel converts the same signal successfully.

This paper makes four contributions, in decreasing order of claimed novelty:

1. **A component-level localization of forgetting** in world-model agents (§3): representations,
   reward heads, and critics survive; only the actor's behavior decays. This contradicts the
   untested premise underlying replay-based continual MBRL and reframes what needs fixing.
2. **A channel isolation result** (§4): with a frozen world model and identical imagined data,
   RL-in-imagination fails to recover a lost skill (0/3) where supervised self-imitation
   succeeds (3/3, zero environment steps). The instability is the policy-gradient channel.
3. **Dream rehearsal** (§5): interleaving supervised self-imitation on world-model-graded
   imagined rollouts of prior tasks during sequential training. One actor, no task labels at
   any time, no parameter growth, ~15% compute overhead. It converts the isolation
   *upper bound* into a *mechanism*: 3/3 four-task chains and 3/3 eight-task
   chains, against 0/3 for the plain-replay recipe.
4. **The grader is load-bearing** (§6): we characterize two failure modes of imagined-trajectory
   scoring (post-terminal contamination; promises outbidding realized successes), give a
   realized-first rule that closes both — an imagination-safe port of the self-imitation
   advantage gate [Oh et al. 2018] — and introduce an offline gauge that measures selection
   quality directly, which caught both failure modes before they cost environment interaction.

Everything was pre-registered before running (protocols, bars, and interpretation matrices
git-committed; off-machine timestamped bundles), and we report the refuted hypotheses —
including two of our own scoring bugs — in the main text, because the gauges that caught them
are, we argue, part of the method.

## 2. Setup

**Agent.** DreamerV3 (PyTorch implementation; 17M-parameter world model: CNN encoder, RSSM,
reward/continuation/decoder heads; 1.8M-parameter actor; imagination horizon 15). Greedy
exploration; all defaults from the reference MiniGrid configuration unless stated.

**Tasks and chains.** MiniGrid, 64×64 RGB observations. Core chain (all of §§3–6):
DoorKey-5x5 → SimpleCrossingS9N1 → LavaGapS5 → MultiRoom-N2-S4 — mixed episode horizons
(~10 to ~80 policy steps) and one lethal task, properties that turn out to matter (§6). The
scale-up chain (§7) appends four audition-gated tasks: LavaCrossingS9N1, DistShift2,
DoorKey-6x6, Unlock (eight tasks, six distinct mechanics, three lethal).

**Protocol.** Sequential phases. During phase *i* the agent acts only in task *i*; a shared
never-clear buffer accumulates all experience (the Continual-Dreamer recipe at our scale).
Every 2,000 environment steps, every learned task is evaluated for 10 real-environment
episodes. A phase advances when all learned tasks score ≥ 0.6 (mean return) for 3 consecutive
evaluation rounds, or at a 150k-step cap. Final retention = last-3-round mean per task; a seed
passes if all tasks finish ≥ 0.6. n=3 seeds everywhere; no per-seed tuning; every experiment's
bars and interpretation matrix committed before launch. Because run-level nondeterminism is
substantial (the same seed can produce qualitatively different trajectories), we report
per-seed values and dispersion rather than pass counts alone.

## 3. Where forgetting lives

**3.1 The phenomenon.** Two tasks, no replay: retention of task A after training B falls from
0.96 to 0.0 / 0.27 / 0.12 across seeds. Four tasks *with* unbounded replay: 0/3 chains pass,
and the casualty is the same task every seed — SimpleCrossing retains 0.35 / 0.51 / 0.25 while
DoorKey (~0.96), LavaGap (~0.94), and MultiRoom (0.79–0.82) survive. Replay protects some
tasks and not others; whatever replay protects, it is not uniformly "the policy."

**3.2 Freeze bracket (interventions).** Freezing the actor's weights at task-A competence
while the world model continues training: retention 0.0 / 0.0 / 0.07 — actor protection
without representation stability is useless; the latent space moves under the frozen head.
Freezing the world model instead: the new task never learns at all (B stuck at 0.0; the actor
learns exclusively in imagination, and a frozen world model only dreams task A) — one seed,
terminated early as a foregone conclusion; we report it as direction-finding only. The bracket
kills both naive protection strategies and sets up the real question: with replay keeping the
world model trained on all tasks, what exactly is still being lost?

**3.3 The world model retains everything we can measure.** We probed each component of the
final chain checkpoint against every earlier phase checkpoint (deterministic posterior-mode
encoding; n=3 seeds; all pre-registered with the prediction that the reward head would show
selective forgetting — a prediction the data refuted):

- *Reward knowledge.* Discrimination D = (mean predicted reward at true success steps) − (at
  zero-reward steps) on each task's held episodes. Retention ratio R = D(final)/D(own-phase)
  for the always-forgotten task T1: **0.99 / 1.06 / 1.01**. The reward head never forgets — on
  one seed it *improves* with continued replay.
- *Off-distribution check.* The same discrimination measured on the degraded policy's own
  final-era episodes — the states a drifted actor actually visits, including outright failures:
  D = 0.87 (vs 0.90 on curated episodes), with **zero** false reward on failure trajectories.
  The signal is intact where re-teaching needs it, not just where replay polishes it.
- *Values.* Critic means on old-task states remain high and rise across phases (0.84 → 0.91).
- *Termination model.* Continuation-head discrimination at true terminal steps: 0.95–1.0 at
  every checkpoint, including immediately after the shortest task phase.
- *Representations, with a caveat that matters.* A frozen-action-margin probe shows old-task
  action preferences under re-encoded latents degrade (margin −0.25 at run end vs +4.5 and
  +2.7 for stabler tasks): the latent space *does* drift. Precisely: **co-trained heads track
  the drift; frozen heads decay under it; and the actor — co-trained but through the RL
  channel — decays too.** "The world model remembers" means replay-maintained knowledge, not a
  frozen latent geometry.

An important scope note: this is *replay-maintained* memory. Without replay, representation
overwrite is total (§3.2). Our claim is regime-specific and does not contradict findings that
model-free fine-tuning irreversibly changes policy representations [Wolczyk et al. 2024] —
there, no mechanism exists that could preserve anything (no world model, no dynamics
objective, no retained data). In the regime the continual-MBRL literature actually operates
in, the premise "the world model is what needs protecting" has it backwards: the world model
is the component that replay already protects.

**3.4 The isolation reference.** For completeness we built the multi-head upper bound made
task-agnostic: frozen per-task actor snapshots plus a nearest-centroid task router over
encoder embeddings (no task labels at evaluation). With correct routing (measured ~1.00
per-task offline), the oracle read passes 5/5 seeds — but the honest routed read passes 3/5,
entirely because the hardest task's *stored* policy delivers only 0.62 ± 0.13 against a 0.6
bar; two seeds flipped verdicts in opposite directions between two reads of the same trained
agent. Storing policies caps retention at snapshot quality and converts the pass/fail question
into a coin flip at the bar. Isolation is a reference point, not a fix. (This experiment also
surfaced an album-ordering bug in our router evaluation whose diagnosis — an offline exact
replication of the corrupted metric — set the pattern for the gauges of §6; the corrected
re-read is the number reported here, with both reads in the artifact.)

## 4. The channel is the failure: recovery by dreams alone

If §3 is right, the actor should be re-teachable from the world model alone. We froze the
entire final world model of each 4-task run (encoder, RSSM, all heads; parameter checksums
asserted unchanged), selected the always-forgotten task's episodes from the buffer by phase
window (selection purity gate ≥ 80% verified by routing), and re-trained only the drifted
live actor on imagined rollouts from those states. Two teachers, same starts, same budget
(20k updates), zero new environment steps, real-environment evaluation every 500 updates:

- **RL-in-imagination** (standard DreamerV3 actor–critic update on the frozen model): **0/3.**
  One seed reaches 0.84 and oscillates back to 0.38; one collapses to 0.0 *and destroys the
  collateral tasks' behavior* (a co-trained task falls to 0.08); one never exceeds 0.57.
- **Dream self-imitation** (roll the current actor with sampling in imagination; grade each
  trajectory with the frozen reward and value heads; behavior-clone the top 25%): **3/3**, at
  2,000 / 3,500 / 7,500 updates — 0.38→0.85, 0.66→0.92, 0.66→0.85 — above the frozen-head
  storage ceiling of §3.4, with collateral tasks intact.

Same model, same knowledge, same dreams. The supervised channel converts the signal into
behavior; the policy-gradient channel thrashes. This is the pre-registered arm-comparison's
strongest row ("only imitation passes → the instability is the RL channel"), and it is why we
frame forgetting in this regime as a channel problem. The world model is a sufficient
behavioral memory at this scale; what was missing was a stable way to read behavior back out
of it.

## 5. Dream rehearsal: prevention during the chain

The recovery result suggests the fix: never let the actor drift in the first place. After each
task's phase ends, keep its buffered episodes as *rehearsal starts*; from then on, after every
2,000 environment steps of new-task training, run 50 dream-self-imitation updates per prior
task — imagine from that task's states with the current (sampling) actor, grade each imagined
trajectory with the live reward/continuation/value heads, and behavior-clone the top 25%.
One live actor throughout. No task labels at training or evaluation, no frozen policies, no
router, no new parameters; the world model and critic train exactly as in the plain recipe.
Rehearsal adds ~15% compute at four tasks (linear in task count).

**Four-task result: 3/3 seeds pass all four tasks** (final retention; plain replay 0/3;
isolation reference 3/5):

| Seed | DoorKey | SimpleCrossing | LavaGap | MultiRoom | total env steps |
|---|---|---|---|---|---|
| 1 | 0.959 | 0.905 | 0.760 | 0.814 | 136k |
| 2 | 0.956 | 0.740 | 0.662 | 0.806 | 244k |
| 3 | 0.958 | 0.826 | 0.727 | 0.799 | 130k |

The historically doomed task (SimpleCrossing) averages **0.824** — above the isolation
ceiling's 0.62 ± 0.13 — and stays above bar in **153/160 (96%)** of all post-acquisition
evaluations. Two of three seeds finish the whole chain *faster* than the isolation
architecture despite the rehearsal overhead, because no skill ever needs relearning:
acquisition of later tasks accelerates (LavaGap learned in 8–10k steps in-chain on every
seed). Retention converges to each task's own competence ceiling, not to a common value —
which is what "not forgetting" should look like.

*(The table above is the initial-scorer run; §6's corrected grader lifts the one unstable
task's stability from 27/19/42% to 95/100/67% with all-task passes preserved on all seeds —
final numbers used in Fig. 2 are the corrected-run values: seed means 0.960/0.868/0.899/0.751,
0.961/0.751/0.944/0.800, 0.964/0.825/0.784/0.768.)*

**Why dream at all?** The buffer already stores the real episodes, actions included; cloning
them directly (CLEAR-style) is the obvious simpler alternative. We pre-registered this arm
adversarially (two of four interpretation rows favor the simpler method; publication of the
outcome in the main body was committed either way), with the strongest fair version: clone only
*competent* episodes (return > 0.05), matched update-for-update to the dream schedule. One
methods note: our first implementation of the cloning arm silently mis-aligned features and
actions by one step (the RSSM's action convention lags the actor's decision), which *destroyed*
skills while cloning their own successes — a flattering artifact for our method, caught by an
isolation gauge (clone a skill's own success; sound code must preserve it) and fixed before any
comparison was read. With the corrected update:

| T1 retention (per seed) | mean | all-four passes |
|---|---|---|
| Dream rehearsal: 0.868 / 0.751 / 0.825 | **0.815** | 3/3 |
| Real-BC (competent-filtered): 0.630 / 0.678 / 0.743 | 0.684 | 3/3 |

Real-episode cloning *works* — it clears every bar and is cheaper per update; we report it as a
viable baseline. But it retains consistently less: paired difference **+0.131**, bootstrap 95%
CI **[0.073, 0.238]** (10k draws), meeting the pre-registered DISTINCT criterion (CI excludes
zero, point ≥ 0.10) — with complete seed separation (every dream seed exceeds every cloning
seed). Imagination contributes beyond the supervised channel; the natural candidates are fresh
on-policy state coverage from the current world model and the gradability of imagined data
(§6). The unfiltered cloning variant was not re-run after the bug fix: it clones a strict
superset of the filtered arm's data plus pre-competence flailing, and cannot plausibly exceed
the filtered arm it would need to beat; we disclose it as unrun.

## 6. The grader is load-bearing

One systematic wart survived the four-task result: the lethal short-episode task (LavaGap)
wobbled after its phase — above bar at the final read on all seeds, but only 27% / 19% / 42%
of its post-acquisition evaluations cleared the bar. Pre-registered probes (predictions
committed, two of three refuted) convicted neither the rehearsal data (the task's library was
the *largest and most competent* of all tasks) nor the world model's physics (termination
discrimination 0.95–1.0 from the earliest checkpoint), but the **scorer**: with ~10-step
episodes and a 15-step imagination horizon, **99.7% of imagined trajectories terminate
in-horizon**, and a naive score (discounted reward plus terminal value bootstrap) keeps
scoring imagination *past the dream's own ending* — post-terminal latents the model never
trained on. Selection was near-random on this task (AUC 0.49–0.56): 10–38% of what rehearsal
cloned were trajectories that walked into lava, ranked there by post-terminal bootstrap noise.
On long-horizon tasks, where dreams rarely terminate in-horizon, the same scorer is clean —
which is exactly why only the short-horizon task wobbled.

Fixing this exposed a second, opposite failure mode, caught **offline before any environment
cost** by a selection gauge we now consider part of the method: generate sibling dreams from
banked starts, label each by its own imagined outcome, and measure each candidate scorer's
selection quality directly (AUC, top-quartile purity) rather than inferring it from returns.
Survival-weighting the rewards (with or without the exact λ-return construction) *inverts*
selection on long-horizon tasks (AUC 0.004): a dream that actually reaches the goal has its
terminal reward and bootstrap suppressed, while a dream that merely wanders keeps the critic's
optimistic value — promises outbid realized successes. A recovery-referee run confirmed the
inversion behaviorally (flat 0.0 where the naive scorer had recovered the skill).

The rule that closes both failure modes is **realized-first grading**: trajectories that
actually achieved reward in imagination (reach-probability-weighted, so nothing counts past
termination) outrank all value-promises; the termination-aware score orders within groups.
This principle is not new on real data — it is exactly the self-imitation advantage gate of
Oh et al. (2018), (R − V(s))₊: imitate only when realized return beats the critic's estimate.
Our contribution is scoped to making it safe for imagination, where the critic's optimism is
unconstrained by any real outcome and where dreams, unlike real episodes, do not stop at their
own endings: the reach-probability weighting, the termination-aware bootstrap, and the offline
gauge. On the gauge, realized-first scores AUC 1.0 / top-quartile purity 1.0 on both task
profiles; in the recovery referee it passes at 2,000 updates (1.75× faster than the naive
scorer); re-running the four-task chain with it lifts the wobbling task's stability to
**95% / 100% / 67%** with every guardrail held (all-four passes on all seeds; hardest-task
mean 0.815). We note one adjacent null in the literature that our target distinction predicts:
reward-*weighted episode selection for world-model training* does not help and can cost
plasticity [Continual-Dreamer, app. D.8; similarly PGR]. Selecting which real episodes train
the world model is a different object from selecting which imagined trajectories the actor
clones; our gauge measures the latter directly.

## 7. Scale-up: eight tasks

Doubling chain length with four audition-gated additional tasks (each verified to be learnable
solo within the phase budget; candidates that could not bootstrap under greedy exploration —
long-horizon sparse tasks — were excluded and are reported as the current acquisition
frontier): DoorKey-5x5 → SimpleCrossing → LavaGap → MultiRoom → LavaCrossing → DistShift2 →
DoorKey-6x6 → Unlock. Same recipe, same constants, one changed variable (task count).

**All three seeds pass all eight tasks** — unanimous, exceeding the pre-registered ≥2/3 bar:

| Seed | DoorKey5 | SimpleX | LavaGap | MultiRoom | LavaCross | DistShift | DoorKey6 | Unlock |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.964 | 0.943 | 0.947 | 0.759 | 0.861 | 0.960 | 0.915 | 0.861 |
| 2 | 0.963 | 0.919 | 0.943 | 0.746 | 0.865 | 0.929 | 0.962 | 0.888 |

By the final phase, rehearsal runs 350 dream-imitation updates per 2,000-step chunk across seven
prior tasks — still one actor, still no task labels, still no added parameters.

Two secondary reads matter more than the pass itself. **First, consistency:** the two seeds agree
within ~0.03 on six of eight tasks (max gap 0.047). The isolation reference of §3.4 delivered
0.62 ± 0.13 on its hardest task at *half* this chain length, with two seeds flipping verdicts
between two reads of the same agent. At 2× the tasks, rehearsal is materially more stable than
isolation was at 1×.

**Second, the predicted scale-failure did not appear.** We pre-registered rehearsal dilution
(early tasks sagging as a fixed per-task budget spreads across more tasks) as the expected
failure mode at increased chain length, with a stability-weighted allocator as its remedy. No
dilution is observed: the incumbent quartet holds at or above its four-task band, so the
allocator is not triggered and we do not build it.

*Observation, explicitly not a result:* the historically-doomed task reads higher at eight tasks
(0.943 / 0.919) than at four (0.868 / 0.751 / 0.825). We decline to interpret this. The gap
between the arm means (0.116) is the same magnitude as the four-task arm's own seed-to-seed
spread (0.117), the comparison is across two different experiments (different downstream tasks,
rehearsal-pass counts, and total budgets) with n=2 versus n=3, and no matched-budget control was
run. It is a difference, not an effect, and it is recorded here only so that a future controlled
test — same chain, varying only the count of downstream tasks — has a reason to exist.

Retention again settles at each task's own ceiling rather than a common value (easy tasks ~0.96;
MultiRoom ~0.75, its band in every chain we have run; LavaCrossing ~0.86).

## 8. Related work

*(as rewritten per the adversarial novelty scan — see draft v0 §8 for the full text with
per-paper greps and quotes; summary here)* Continual-Dreamer, WMAR, and ARROW form the
replay-protected world-model line whose untested premise §3 measures; our plain-replay control
is Continual-Dreamer's recipe at our scale, and ARROW is the closest concurrent competitor on
setting with an orthogonal mechanism (distribution-matched *real* replay). ReGen (June 2026)
is concurrent, structurally the closest to §5's mechanism (world-model pseudo-replay + single
policy trained by behavior cloning), in continual *imitation* learning on a video-diffusion
model with instruction-conditioned replay — not task-agnostic and not RL; we cite it as
convergent evidence for the mechanism family. Self-Imitation Learning (Oh et al. 2018) is
prior art for §6's advantage gate on real data, which we say plainly. Deep Generative
Replay/pseudo-rehearsal is the ancestral idea; ours is imagination-native (no generator is
trained — the world model already in the loop is the generator) and grades rather than
replays. Wolczyk et al. (2024) is the apparent counter-evidence to §3, resolved as a regime
difference (model-free, no retained data — nothing could preserve a representation). C-CHAIN
and the plasticity/churn line are mechanistically adjacent to §4's channel finding but never
measure backward transfer; parameter-isolation methods are §3.4's reference; CLEAR is the
real-data ancestor of the §5 cloning comparison, now distinguished empirically (DISTINCT,
complete seed separation) rather than rhetorically.

## 9. Limitations

MiniGrid scale; a 17M-parameter world model; one task ordering per chain length; return bars
rather than normalized scores (we additionally report retention relative to each task's own
post-acquisition competence). n=3 with real run-level nondeterminism — identical seeds can
diverge qualitatively — so we report dispersion and bar-distance, not only pass counts.
Rehearsal cost grows linearly with task count; the observed acquisition speed-up offsets it at
n=8, but the crossover point is unmeasured. The buffer retains raw experience: dream rehearsal
removes stored *policies*, not stored *data* — replacing rehearsal starts with generated
starts is future work. Tasks that cannot bootstrap under greedy exploration within our budget
mark the acquisition frontier of this study; nothing here addresses exploration. Two scoring
bugs shipped during development and were caught by our own offline gauges before contaminating
results; we report both (§3.4, §6) and draw the methodological conclusion that selection
gauges belong in the method, not the appendix.

## 10. Reproducibility

Every experiment was pre-registered: protocol, bars, and an interpretation matrix committed to
git before launch (commit hashes in the artifact index), with off-machine timestamped bundles.
All refuted hypotheses are reported. Code, run summaries, probe artifacts, per-round curves,
and the full pre-registration trail are released at https://github.com/gurpnijjer/dream-rehearsal. Total compute: a
single NVIDIA GB10 workstation; the complete experimental record of this paper is ~3 weeks of
one GPU.
