"""dream_recovery.py — can the frozen WM re-teach the drifted actor T1, dreams only?

Pre-reg: DREAM_RECOVERY_PREREG_2026-07-10.md (+ Arm B amendment 0920d182).
Arm A (rl): standard DreamerV3 actor-critic updates on imagined rollouts from T1-phase
  posterior starts. WM entirely frozen; zero new env steps for training.
Arm B (tunnel): imagine with the current (sampling) actor from the same starts; grade each
  imagined trajectory with the frozen reward head + frozen critic bootstrap; keep the top 25%
  per batch; behavior-clone the actor on the kept state->action pairs. No policy gradient,
  no critic update.
PRIMARY (per arm): T1 real-env >= 0.75 for 3 consecutive evals within 20k updates, >=2/3 seeds.
Gates: WM param-sum identical before/after; T1-episode mtime selection must route >=80% to T1
  through corrected albums (else abort).

Usage: OMP_NUM_THREADS=3 python -u dream_recovery.py --logdir ~/dv3_logs/comp4_s1 --arm rl
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
from orchestrator_ab_nm512 import (  # noqa: E402
    _make_capture_logger, build_config, build_task_envs, eval_return)
from orchestrator_chain_nm512 import _build_centroids, _episode_mean_embed  # noqa: E402

TASKS = ["minigrid_DoorKey-5x5", "minigrid_SimpleCrossingS9N1",
         "minigrid_LavaGapS5", "minigrid_MultiRoom-N2-S4"]
T1 = 1
MAX_UPDATES = 20000
EVAL_T1_EVERY = 500
EVAL_ALL_EVERY = 2000
PASS_BAR, PASS_STREAK = 0.75, 3
TOPK_FRAC = 0.25


def load_eps_window(dirpath, mtime_min, mtime_max, min_len=3):
    files = sorted(pathlib.Path(dirpath).glob("*.npz"), key=lambda f: f.stat().st_mtime)
    files = [f for f in files
             if (mtime_min is None or f.stat().st_mtime > mtime_min)
             and (mtime_max is None or f.stat().st_mtime <= mtime_max)]
    eps = OrderedDict()
    for f in files:
        try:
            with np.load(f) as z:
                ep = {k: z[k] for k in z.files}
        except Exception:
            continue
        if len(next(iter(ep.values()))) >= min_len:
            eps[f.stem] = ep
    return eps


def wm_param_sum(wm):
    with torch.no_grad():
        return float(sum(p.double().abs().sum().item() for p in wm.parameters()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--arm", choices=["rl", "tunnel"], required=True)
    p.add_argument("--scorer", choices=["dep", "shift", "lex"], default="dep",
                   help="tunnel dream scorer: dep = banked 3/3 formula; shift = reach-"
                        "probability-weighted; lex = realized-first (achieved dreams outrank "
                        "value promises, shift-ordered within groups)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    logdir = pathlib.Path(args.logdir).expanduser()
    tag = f"dream_{args.arm}" + (f"_{args.scorer}" if args.scorer != "dep" else "")

    overrides = dict(task=TASKS[T1], size=[64, 64], device=args.device, compile=False,
                     video_pred_log=False, seed=1, steps=int(1e9), eval_episode_num=10,
                     expl_behavior="greedy", expl_until=0, expl_extr_scale=1.0)
    config = build_config(["minigrid"], overrides)
    tools.set_seed_everywhere(config.seed)
    logger = _make_capture_logger(tools, logdir / f"{tag}_logs", 0)

    train_envs, _ = build_task_envs(D, config, TASKS[0])
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    obs_space = train_envs[0].observation_space
    for env in train_envs:
        try:
            env.close()
        except Exception:
            pass

    # --- T1-phase training episodes from the shared buffer, by mtime window --------------
    phase_end = {i: (logdir / f"chain_phase{i}.pt").stat().st_mtime for i in range(4)}
    t1_eps = load_eps_window(logdir / "train_eps", phase_end[0], phase_end[1])
    assert len(t1_eps) >= 50, f"only {len(t1_eps)} T1-window episodes"

    # --- Agent from final checkpoint; WM frozen -----------------------------------------
    agent = D.Dreamer(obs_space, acts, config, logger,
                      D.make_dataset(t1_eps, config)).to(config.device)
    sd = torch.load(logdir / "chain_phase3.pt", map_location=config.device,
                    weights_only=False)["agent_state_dict"]
    agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    wm, behavior = agent._wm, agent._task_behavior
    wm.requires_grad_(False)
    wm.eval()
    wm_sum0 = wm_param_sum(wm)

    # --- Gate: selected episodes must be T1 (route through corrected albums) ------------
    albums = {}
    for j, t in enumerate(TASKS):
        files = sorted((logdir / f"eval_eps_{j}_{t}").glob("*.npz"),
                       key=lambda f: f.stat().st_mtime)
        files = [f for f in files if f.stat().st_mtime <= phase_end[j]][-20:]
        albums[j] = []
        for f in files:
            with np.load(f) as z:
                albums[j].append({k: z[k] for k in z.files})
    cents, tids = _build_centroids(agent, albums)
    sample = list(t1_eps.values())[:: max(1, len(t1_eps) // 20)][:20]
    hits = 0
    for ep in sample:
        emb = _episode_mean_embed(agent, ep)
        hits += int(tids[int(np.linalg.norm(cents - emb[None], axis=1).argmin())] == T1)
    frac = hits / len(sample)
    print(f"[gate] T1-window purity: {hits}/{len(sample)} = {frac:.2f}")
    assert frac >= 0.8, f"episode-window gate FAILED ({frac:.2f} < 0.8) — fix selection"

    dataset = D.make_dataset(t1_eps, config)
    reward_fn = lambda f, s, a: wm.heads["reward"](wm.dynamics.get_feat(s)).mode()
    task_envs = {}
    for j, t in enumerate(TASKS):
        _, ev = build_task_envs(D, config, t)
        task_envs[j] = ev

    def encode_starts(batch):
        with torch.no_grad():
            data = wm.preprocess(batch)
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        return {k: v.detach() for k, v in post.items()}

    curve, all_task_checks, streak, passed_at = [], [], 0, None
    log_path = logdir / f"{tag}_curve.jsonl"
    log_path.write_text("")
    for u in range(1, MAX_UPDATES + 1):
        batch = next(dataset)
        start = encode_starts(batch)
        if args.arm == "rl":
            behavior._train(start, reward_fn)
        else:  # tunnel: grade dreams with the frozen heads, clone the best
            with torch.no_grad():
                feats, states, actions = behavior._imagine(
                    start, behavior.actor, config.imag_horizon)
                r = wm.heads["reward"](feats).mode().squeeze(-1)      # [H, B]
                gam = config.discount ** torch.arange(
                    r.shape[0], device=r.device, dtype=r.dtype).unsqueeze(-1)
                v_last = behavior.value(feats[-1]).mode().squeeze(-1)  # [B]
                if args.scorer in ("shift", "lex"):
                    cont = wm.heads["cont"](feats).mean.squeeze(-1)
                    surv = torch.cumprod(cont, dim=0)
                    reach = torch.cat([torch.ones_like(cont[:1]), surv[:-1]], dim=0)
                    score = (gam * reach * r).sum(0) + \
                        (config.discount ** r.shape[0]) * surv[-1] * v_last
                    if args.scorer == "lex":
                        achieved = ((gam * reach * r).sum(0) > 0.3).float()
                        score = achieved * 10.0 + score
                else:
                    score = (gam * r).sum(0) + (config.discount ** r.shape[0]) * v_last
                k = max(1, int(TOPK_FRAC * score.shape[0]))
                top = score.topk(k).indices
                bc_feat = feats[:, top].detach()
                bc_act = actions[:, top].detach()
            with tools.RequiresGrad(behavior.actor):
                dist = behavior.actor(bc_feat)
                bc_loss = -(dist.log_prob(bc_act)).mean()
                behavior._actor_opt(bc_loss, behavior.actor.parameters())

        if u % EVAL_T1_EVERY == 0:
            ret = eval_return(tools, agent, task_envs[T1], OrderedDict(),
                              logdir / f"{tag}_eval", logger, config.eval_episode_num)
            curve.append({"update": u, "t1": ret})
            with log_path.open("a") as f:
                f.write(json.dumps(curve[-1]) + "\n")
            print(f"[{tag} u={u}] T1 {ret}")
            streak = streak + 1 if (ret is not None and ret >= PASS_BAR) else 0
            if streak >= PASS_STREAK and passed_at is None:
                passed_at = u
                print(f"[{tag}] PRIMARY PASS at update {u}")
                break
        if u % EVAL_ALL_EVERY == 0:
            row = {"update": u}
            for j in (0, 2, 3):
                row[f"t{j}"] = eval_return(tools, agent, task_envs[j], OrderedDict(),
                                           logdir / f"{tag}_eval", logger,
                                           config.eval_episode_num)
            all_task_checks.append(row)
            print(f"[{tag} u={u}] collateral {row}")

    assert wm_param_sum(wm) == wm_sum0, "WM PARAMS CHANGED — freeze breached, run invalid"
    vals = [c["t1"] for c in curve if c["t1"] is not None]
    out = {"logdir": str(logdir), "arm": args.arm, "passed_at_update": passed_at,
           "primary_pass": passed_at is not None, "n_updates_run": u,
           "t1_start": vals[0] if vals else None, "t1_best": max(vals) if vals else None,
           "auc": float(np.mean(vals)) if vals else None,
           "curve": curve, "collateral": all_task_checks,
           "n_t1_window_eps": len(t1_eps), "gate_purity": frac}
    (logdir / f"{tag}_result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("curve", "collateral")}))
    for ev in task_envs.values():
        for env in ev:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
