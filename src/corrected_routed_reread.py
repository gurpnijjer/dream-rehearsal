"""corrected_routed_reread.py — re-execute ONLY the final routed read with corrected albums.

Pre-reg: ROUTER_ALBUM_BUG_2026-07-10.md §corrected read. Per seed: final WM + per-phase frozen
heads from chain_phase{0..3}.pt; albums per the REGISTERED rule (newest 20 eval eps at phase-end
mtime cutoff — the rule the deployed [-20:] bug violated); bandit arms FIXED to each seed's
recorded converged arm (head unless the recorded bandit favored live); commit_k=10; K=3 rounds
x eval_episode_num episodes per task. Rollouts write to fresh reread_eval_* dirs (never the
run's own eval dirs — mtime-based album selection must stay clean for future re-analysis).

Usage:  OMP_NUM_THREADS=3 python -u corrected_routed_reread.py --logdir ~/dv3_logs/comp4_s1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import OrderedDict

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_cand / "dreamer.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

import torch  # noqa: E402
import dreamer as D  # noqa: E402
import tools  # noqa: E402
from composite_posthoc_eval import load_actor_from_ckpt  # noqa: E402
from orchestrator_ab_nm512 import _make_capture_logger, build_config, build_task_envs  # noqa: E402
from orchestrator_chain_nm512 import (  # noqa: E402
    RoutedEvalPolicy, _build_centroids, eval_return_routed)

TASKS = ["minigrid_DoorKey-5x5", "minigrid_SimpleCrossingS9N1",
         "minigrid_LavaGapS5", "minigrid_MultiRoom-N2-S4"]
K = 3
BAR = 0.6


def load_eps(evaldir, mtime_max, last=20):
    files = sorted(pathlib.Path(evaldir).glob("*.npz"), key=lambda f: f.stat().st_mtime)
    files = [f for f in files if f.stat().st_mtime <= mtime_max][-last:]
    eps = []
    for f in files:
        with np.load(f) as z:
            eps.append({k: z[k] for k in z.files})
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    logdir = pathlib.Path(args.logdir).expanduser()

    overrides = dict(task=TASKS[0], size=[64, 64], device=args.device, compile=False,
                     video_pred_log=False, seed=1, steps=int(1e9), eval_episode_num=10,
                     expl_behavior="greedy", expl_until=0, expl_extr_scale=1.0)
    config = build_config(["minigrid"], overrides)
    tools.set_seed_everywhere(config.seed)
    logger = _make_capture_logger(tools, logdir / "reread_logs", 0)

    train_envs, eval_envs0 = build_task_envs(D, config, TASKS[0])
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    eps0 = tools.load_episodes(logdir / f"eval_eps_0_{TASKS[0]}", limit=100)
    agent = D.Dreamer(train_envs[0].observation_space, acts, config, logger,
                      D.make_dataset(eps0, config)).to(config.device)
    agent.requires_grad_(False)
    sd = torch.load(logdir / "chain_phase3.pt", map_location=config.device,
                    weights_only=False)["agent_state_dict"]
    agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    agent.eval()
    for env in train_envs + eval_envs0:
        try:
            env.close()
        except Exception:
            pass

    heads = {j: load_actor_from_ckpt(agent, logdir / f"chain_phase{j}.pt", config.device)
             for j in range(4)}

    # Arms fixed from the recorded bandit (pre-reg: head unless recorded bandit favored live).
    rec = json.loads((logdir / "chain_summary.json").read_text())
    arms = {}
    for j in range(4):
        b = (rec.get("arbitration_bandit") or {}).get(str(j), {})
        h, l = b.get("head"), b.get("live")
        arms[j] = "live" if (h is not None and l is not None and l > h) else "head"
    policy_by_task = {j: (heads[j] if arms[j] == "head" else agent._task_behavior.actor)
                      for j in range(4)}

    # Corrected albums: newest 20 at phase-end cutoff (the registered rule).
    phase_end = {i: (logdir / f"chain_phase{i}.pt").stat().st_mtime for i in range(4)}
    albums = {j: load_eps(logdir / f"eval_eps_{j}_{t}", phase_end[j])
              for j, t in enumerate(TASKS)}
    for j in range(4):
        assert len(albums[j]) == 20, f"album {j}: {len(albums[j])} eps"
    cents, tids = _build_centroids(agent, albums)

    rounds = {j: [] for j in range(4)}
    acc_by = {j: [0, 0] for j in range(4)}
    for _ in range(K):
        for j, task in enumerate(TASKS):
            _, eval_envs = build_task_envs(D, config, task)
            rp = RoutedEvalPolicy(agent, cents, tids, policy_by_task, commit_k=10)
            r = eval_return_routed(tools, rp, eval_envs, OrderedDict(),
                                   logdir / f"reread_eval_{j}", logger,
                                   config.eval_episode_num)
            rounds[j].append(r)
            acc_by[j][0] += sum(1 for pk in rp.episode_picks if pk == j)
            acc_by[j][1] += len(rp.episode_picks)
            for env in eval_envs:
                try:
                    env.close()
                except Exception:
                    pass

    corrected = {TASKS[j]: (float(np.mean([x for x in v if x is not None]))
                            if any(x is not None for x in v) else None)
                 for j, v in rounds.items()}
    verdict = all(v is not None and v >= BAR for v in corrected.values())
    out = {"logdir": str(logdir), "arms": {TASKS[j]: a for j, a in arms.items()},
           "corrected_final_routed": corrected,
           "rounds": {TASKS[j]: v for j, v in rounds.items()},
           "router_acc_by_task": {TASKS[j]: (h / t if t else None)
                                  for j, (h, t) in acc_by.items()},
           "corrected_routed_success": verdict,
           "deployed_final_routed": rec.get("final_retention_routed"),
           "deployed_routed_success": rec.get("success_all_tasks_routed")}
    (logdir / "corrected_reread.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
