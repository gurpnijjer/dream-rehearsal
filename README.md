# The World Model Remembers, the Actor Forgets
### Dream Rehearsal for Continual Model-Based RL

**Gurp Nijjer** — [Quantegra](https://quantegra.ca)

[![arXiv](https://img.shields.io/badge/arXiv-2607.19749-b31b1b.svg)](https://arxiv.org/abs/2607.19749)
[![DOI](https://zenodo.org/badge/1307022956.svg)](https://zenodo.org/badge/latestdoi/1307022956)

**Paper: [arXiv:2607.19749](https://arxiv.org/abs/2607.19749)**

Code, pre-registration trail, and run data for the paper
(*[paper/dream_rehearsal_paper.md](paper/dream_rehearsal_paper.md)*).

## TL;DR

DreamerV3-family agents forget catastrophically across task sequences — but **the world model
isn't what forgets**. Under never-clear replay, representations, reward heads, and critics all
retain old-task knowledge (reward-head retention ≈ 1.0); the actor's behavior collapses anyway.
The failure is the policy-gradient *channel*, not the memory: with a frozen world model and
identical imagined data, RL-in-imagination fails to re-teach a lost skill (0/3 seeds) while
supervised self-imitation on the world model's own *graded dreams* recovers it (3/3, zero
environment steps). Interleaving this **dream rehearsal** during training yields a
task-label-free, parameter-constant continual learner: 3/3 retention on four-task chains and
3/3 on eight-task chains where plain replay passes 0/3 — and it beats matched real-episode
cloning (paired diff +0.13, 95% CI [0.07, 0.24], complete seed separation). The dream *grading*
step is load-bearing; we characterize its failure modes and ship the offline gauge that caught
two scoring bugs before they contaminated results.

## Layout

- `paper/` — the paper (markdown) + figures
- `src/` — orchestrators, probes, gauges, sweep scripts (see Setup)
- `prereg/` — the pre-registration trail: every protocol, bar, and interpretation matrix
  git-committed before its experiment ran, including refuted hypotheses and two caught bugs
- `substrate/` — the MiniGrid environment wrapper for the base repo
- `results/` — per-run summaries and full evaluation traces (`chain_metrics.jsonl`) for every
  run in the paper

## Setup

Experiments run on [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) (PyTorch
DreamerV3). To reproduce:

1. Clone `dreamerv3-torch`; install its requirements plus `minigrid` (gymnasium).
2. Copy `substrate/minigrid_env.py` to `envs/minigrid.py`; register the `minigrid` suite in
   `dreamer.py`'s `make_env` and add the `minigrid` config block (both shown in
   `substrate/SETUP.md`).
3. Drop `src/*` into the repo root.
4. Example — the 4-task dream-rehearsal chain:
   ```
   python orchestrator_chain_nm512.py \
     --tasks minigrid_DoorKey-5x5,minigrid_SimpleCrossingS9N1,minigrid_LavaGapS5,minigrid_MultiRoom-N2-S4 \
     --tunnel_rehearsal --cont_grading --rehearsal_updates 50 \
     --eval_every 2000 --phase_max_steps 150000 --logdir ./tunl4b_s1 --seed 1
   ```
   Hardware: every result in the paper was produced on a single NVIDIA GB10 box.

## Citation

```bibtex
@article{nijjer2026dreamrehearsal,
  title={The World Model Remembers, the Actor Forgets: Dream Rehearsal for Continual Model-Based RL},
  author={Nijjer, Gurp},
  year={2026},
  eprint={2607.19749},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.5281/zenodo.21462836},
  note={https://github.com/gurpnijjer/dream-rehearsal}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
