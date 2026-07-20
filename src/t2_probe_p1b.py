"""t2_probe_p1b.py — P1 follow-up: goal-dreams vs death-dreams inside each scorer's top-25%.

Classify each imagined T2 trajectory by its FIRST termination (cont < 0.5 at t*):
  success-dream: max imagined reward over t <= t* is > 0.3 (reached goal before/at ending)
  death-dream:   terminated with max reward < 0.05 (ended with nothing = lava)
Report composition of top-25% under deployed vs cont-aware scoring, plus overall.
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
from orchestrator_ab_nm512 import _make_capture_logger, build_config, build_task_envs  # noqa: E402
from t2_wobble_probes import load_eps_window  # noqa: E402

TASKS = ["minigrid_DoorKey-5x5", "minigrid_SimpleCrossingS9N1",
         "minigrid_LavaGapS5", "minigrid_MultiRoom-N2-S4"]


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
    logger = _make_capture_logger(tools, logdir / "probe_logs", 0)
    train_envs, eval_envs = build_task_envs(D, config, TASKS[0])
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    obs_space = train_envs[0].observation_space
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    phase_end = {i: (logdir / f"chain_phase{i}.pt").stat().st_mtime for i in range(4)}
    lib2 = load_eps_window(logdir / "train_eps", phase_end[1], phase_end[2])

    eps0 = tools.load_episodes(logdir / f"eval_eps_0_{TASKS[0]}", limit=50)
    agent = D.Dreamer(obs_space, acts, config, logger,
                      D.make_dataset(eps0, config)).to(config.device)
    agent.requires_grad_(False)
    sd = torch.load(logdir / "chain_phase3.pt", map_location=config.device,
                    weights_only=False)["agent_state_dict"]
    agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    agent.eval()
    wm, behavior = agent._wm, agent._task_behavior

    ds = D.make_dataset(lib2, config)
    agg = {n: {"success": 0, "death": 0, "alive": 0, "n": 0} for n in
           ("overall", "deployed_top", "cont_top")}
    for _ in range(8):
        batch = next(ds)
        with torch.no_grad():
            data = wm.preprocess(batch)
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
            start = {k: v.detach() for k, v in post.items()}
            feats, _st, _ac = behavior._imagine(start, behavior.actor, config.imag_horizon)
            r = wm.heads["reward"](feats).mode().squeeze(-1)      # [H, B]
            cont = wm.heads["cont"](feats).mean.squeeze(-1)       # [H, B]
            v_last = behavior.value(feats[-1]).mode().squeeze(-1)
            H, B = r.shape
            gam = config.discount ** torch.arange(H, device=r.device,
                                                  dtype=r.dtype).unsqueeze(-1)
            surv = torch.cumprod(cont, dim=0)
            s_dep = (gam * r).sum(0) + (config.discount ** H) * v_last
            s_cont = (gam * surv * r).sum(0) + (config.discount ** H) * surv[-1] * v_last

            term = cont < 0.5                                     # [H, B]
            any_term = term.any(dim=0)
            tstar = torch.where(any_term, term.float().argmax(dim=0),
                                torch.full_like(any_term, H - 1, dtype=torch.long))
            # max reward up to and including t*
            idx = torch.arange(H, device=r.device).unsqueeze(-1).expand(H, B)
            mask = idx <= tstar.unsqueeze(0)
            rmax = (r * mask).max(dim=0).values
            success = any_term & (rmax > 0.3)
            death = any_term & (rmax < 0.05)

            k = max(1, int(0.25 * B))
            picks = {"overall": torch.arange(B, device=r.device),
                     "deployed_top": s_dep.topk(k).indices,
                     "cont_top": s_cont.topk(k).indices}
            for n, ix in picks.items():
                agg[n]["success"] += int(success[ix].sum())
                agg[n]["death"] += int(death[ix].sum())
                agg[n]["alive"] += int((~any_term[ix]).sum())
                agg[n]["n"] += int(ix.numel())

    out = {"logdir": str(logdir)}
    for n, a in agg.items():
        out[n] = {kk: round(a[kk] / max(1, a["n"]), 3) for kk in ("success", "death", "alive")}
        out[n]["n"] = a["n"]
    (logdir / "t2_probe_p1b.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
