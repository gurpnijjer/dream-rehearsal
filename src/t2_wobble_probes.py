"""t2_wobble_probes.py — three offline probes to convict the T2-wobble mechanism.

Pre-reg: T2_WOBBLE_PROBES_PREREG_2026-07-13.md. Runs on a finished tunl4 logdir.
P1 grade audit: deployed vs cont-aware dream scoring — death rate inside each top-25%.
P2 library audit: size + competent fraction of each task's deployed rehearsal library.
P3 cont autopsy: terminal-step discrimination of the cont head, ckpt p2 vs p3, T2 vs T0.

Usage: OMP_NUM_THREADS=3 python -u t2_wobble_probes.py --logdir ~/dv3_logs/tunl4_s1
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
from nm512_margin_probe import _encode_feats  # noqa: E402
from orchestrator_ab_nm512 import _make_capture_logger, build_config, build_task_envs  # noqa: E402

TASKS = ["minigrid_DoorKey-5x5", "minigrid_SimpleCrossingS9N1",
         "minigrid_LavaGapS5", "minigrid_MultiRoom-N2-S4"]
TOPK = 0.25
N_BATCH = 8


def load_eps_window(dirpath, lo, hi, min_len=3):
    files = sorted(pathlib.Path(dirpath).glob("*.npz"), key=lambda f: f.stat().st_mtime)
    files = [f for f in files if (lo is None or f.stat().st_mtime > lo)
             and (hi is None or f.stat().st_mtime <= hi)]
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
    win = {j: (phase_end[j - 1] if j else None, phase_end[j]) for j in range(4)}

    # ---------------- P2: library audit (no GPU) ----------------------------------------
    out = {"logdir": str(logdir), "P2_library": {}}
    libs = {}
    for j in (1, 2):
        eps = load_eps_window(logdir / "train_eps", win[j][0], win[j][1])
        libs[j] = eps
        rets = [float(np.asarray(e["reward"]).sum()) for e in eps.values()]
        comp = [r for r in rets if r > 0.05]
        out["P2_library"][f"T{j}"] = {
            "n_eps": len(eps), "frac_competent": round(len(comp) / max(1, len(rets)), 3),
            "mean_ep_reward": round(float(np.mean(rets)), 3) if rets else None}

    # ---------------- agent from final ckpt ---------------------------------------------
    eps0 = tools.load_episodes(logdir / f"eval_eps_0_{TASKS[0]}", limit=50)
    agent = D.Dreamer(obs_space, acts, config, logger,
                      D.make_dataset(eps0, config)).to(config.device)
    agent.requires_grad_(False)

    def load_ckpt(pn):
        sd = torch.load(logdir / f"chain_phase{pn}.pt", map_location=config.device,
                        weights_only=False)["agent_state_dict"]
        agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        agent.eval()

    # ---------------- P1: grade audit under final ckpt ----------------------------------
    load_ckpt(3)
    wm, behavior = agent._wm, agent._task_behavior
    ds = D.make_dataset(libs[2], config)
    stats = {"deployed": {"death_in_top": 0, "top": 0},
             "cont_aware": {"death_in_top": 0, "top": 0},
             "overall_death_rate": [], "n_traj": 0}
    for _ in range(N_BATCH):
        batch = next(ds)
        with torch.no_grad():
            data = wm.preprocess(batch)
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
            start = {k: v.detach() for k, v in post.items()}
            feats, _st, _ac = behavior._imagine(start, behavior.actor, config.imag_horizon)
            r = wm.heads["reward"](feats).mode().squeeze(-1)          # [H, B]
            cont = wm.heads["cont"](feats).mean.squeeze(-1)           # [H, B] prob continue
            v_last = behavior.value(feats[-1]).mode().squeeze(-1)     # [B]
            H = r.shape[0]
            gam = config.discount ** torch.arange(H, device=r.device,
                                                  dtype=r.dtype).unsqueeze(-1)
            surv = torch.cumprod(cont, dim=0)                         # survival prob to t
            dead = (cont < 0.5).any(dim=0)                            # [B] death event
            s_dep = (gam * r).sum(0) + (config.discount ** H) * v_last
            s_cont = (gam * surv * r).sum(0) + (config.discount ** H) * surv[-1] * v_last
            k = max(1, int(TOPK * s_dep.shape[0]))
            for name, sc in (("deployed", s_dep), ("cont_aware", s_cont)):
                top = sc.topk(k).indices
                stats[name]["death_in_top"] += int(dead[top].sum())
                stats[name]["top"] += k
            stats["overall_death_rate"].append(float(dead.float().mean()))
            stats["n_traj"] += int(dead.shape[0])
    out["P1_grade_audit"] = {
        "overall_imagined_death_rate": round(float(np.mean(stats["overall_death_rate"])), 3),
        "deployed_top25_death_rate": round(stats["deployed"]["death_in_top"]
                                           / max(1, stats["deployed"]["top"]), 3),
        "cont_aware_top25_death_rate": round(stats["cont_aware"]["death_in_top"]
                                             / max(1, stats["cont_aware"]["top"]), 3),
        "n_trajectories": stats["n_traj"]}

    # ---------------- P3: cont-head terminal discrimination, p2 vs p3 -------------------
    out["P3_cont_autopsy"] = {}
    for j in (0, 2):
        eval_eps = load_eps_window(logdir / f"eval_eps_{j}_{TASKS[j]}",
                                   None, phase_end[j])
        eps = list(eval_eps.values())[-20:]
        for pn in (2, 3):
            if pn < j:
                continue
            load_ckpt(pn)
            wm = agent._wm
            term_p, nonterm_p = [], []
            for ep in eps:
                batch = {k: np.asarray(v)[None] for k, v in ep.items() if "log_" not in k}
                feat = _encode_feats(agent, batch)
                with torch.no_grad():
                    c = wm.heads["cont"](feat).mean[0].float().cpu().numpy().squeeze(-1)
                disc = np.asarray(ep["discount"]).squeeze()
                term = disc < 0.5
                if term.any():
                    term_p.extend(c[term].tolist())
                    nonterm_p.extend(c[~term].tolist())
            d = (float(np.mean(nonterm_p)) - float(np.mean(term_p))) if term_p else None
            out["P3_cont_autopsy"][f"T{j}_p{pn}"] = {
                "D_cont": round(d, 3) if d is not None else None,
                "mean_cont_at_terminal": round(float(np.mean(term_p)), 3) if term_p else None,
                "n_terminal_steps": len(term_p)}

    (logdir / "t2_wobble_probes.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
