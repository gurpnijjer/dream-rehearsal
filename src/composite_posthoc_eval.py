"""composite_posthoc_eval.py — per-task frozen heads through the FINAL chain WM, real-env eval.

Pre-reg: COMPOSITE_POSTHOC_PREREG_2026-07-06.md. For each task j: agent = WM+critic from
chain_phase3.pt with the ACTOR swapped to chain_phase{j}.pt's actor (head_j, frozen at task-j
competence). Real-env eval of task j. The decisive cell is T1: live-actor final was 0.25-0.51;
if the chain preserved T1's representation, head_1 recovers it.

Runs on Spark (needs env rollouts + GPU is free). Reuses eval_return_with_actor (try/finally swap).
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_cand / "dreamer.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from orchestrator_ab_nm512 import (  # noqa: E402
    _make_capture_logger, build_config, build_task_envs, eval_return, eval_return_with_actor)

TASKS = ["minigrid_DoorKey-5x5", "minigrid_SimpleCrossingS9N1",
         "minigrid_LavaGapS5", "minigrid_MultiRoom-N2-S4"]
ACTOR_PREFIX = "_task_behavior.actor."


def load_actor_from_ckpt(agent, ckpt_path, device):
    """Deep-copy the agent's actor and load ONLY the actor weights from ckpt_path into it."""
    import torch
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["agent_state_dict"]
    actor_sd = {k[len(ACTOR_PREFIX):]: v for k, v in sd.items()
                if k.replace("_orig_mod.", "").startswith(ACTOR_PREFIX)}
    assert actor_sd, f"no actor keys under '{ACTOR_PREFIX}' in {ckpt_path}"
    head = copy.deepcopy(agent._task_behavior.actor)
    head.load_state_dict(actor_sd)
    for p in head.parameters():
        p.requires_grad_(False)
    head.eval()
    return head


def run(args):
    from collections import OrderedDict
    import numpy as np
    import torch
    import dreamer as D
    import tools

    logdir = pathlib.Path(args.logdir).expanduser()
    overrides = dict(task=TASKS[0], size=[64, 64], device=args.device, compile=False,
                     video_pred_log=False, seed=1, steps=int(1e9),
                     eval_episode_num=args.eval_episodes,
                     expl_behavior="greedy", expl_until=0, expl_extr_scale=1.0)
    config = build_config(["minigrid"], overrides)
    tools.set_seed_everywhere(config.seed)
    logger = _make_capture_logger(tools, logdir / "posthoc_logs", 0)

    # spaces from task 0 (identical across our tasks); agent built once, weights from phase3
    train_envs, eval_envs0 = build_task_envs(D, config, TASKS[0])
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    obs_space = train_envs[0].observation_space
    # dataset only needed for Dreamer construction; weights come from the checkpoint
    eval0_dir = logdir / "eval_eps_0_minigrid_DoorKey-5x5"
    eps0 = tools.load_episodes(eval0_dir, limit=1000)
    agent = D.Dreamer(obs_space, acts, config, logger, D.make_dataset(eps0, config)).to(config.device)
    agent.requires_grad_(False)
    final_sd = torch.load(logdir / "chain_phase3.pt", map_location=config.device,
                          weights_only=False)["agent_state_dict"]
    agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in final_sd.items()})
    agent.eval()

    heads = {j: load_actor_from_ckpt(agent, logdir / f"chain_phase{j}.pt", config.device)
             for j in range(4)}

    results = {}
    for j, task in enumerate(TASKS):
        _, eval_envs = build_task_envs(D, config, task)
        eval_eps, evaldir = OrderedDict(), logdir / f"posthoc_eval_{j}"
        rets_head, rets_live = [], []
        for _ in range(args.rounds):
            rets_head.append(eval_return_with_actor(
                tools, agent, heads[j], eval_envs, eval_eps, evaldir, logger,
                config.eval_episode_num))
            rets_live.append(eval_return(
                tools, agent, eval_envs, eval_eps, evaldir, logger, config.eval_episode_num))
        mh = float(np.mean([r for r in rets_head if r is not None]))
        ml = float(np.mean([r for r in rets_live if r is not None]))
        results[task] = {"head": round(mh, 3), "live": round(ml, 3)}
        print(f"T{j} {task.replace('minigrid_',''):22s} | head_{j} {mh:.3f} | live {ml:.3f}")
        for env in eval_envs:
            try:
                env.close()
            except Exception:
                pass

    t1 = results[TASKS[1]]
    verdict = ("COMPOSITE VALIDATED post-hoc: T1 head recovers to "
               f"{t1['head']:.2f} (live {t1['live']:.2f})" if t1["head"] >= 0.6 else
               f"T1 rep DEGRADED in-chain: head_1 only {t1['head']:.2f} — rep-side fix needed first")
    print("\n" + verdict)
    (logdir / "composite_posthoc.json").write_text(json.dumps(
        {"results": results, "verdict": verdict}, indent=2))
    for env in train_envs + eval_envs0:
        try:
            env.close()
        except Exception:
            pass


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
