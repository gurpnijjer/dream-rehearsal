"""reward_head_autopsy.py — does the shared reward head forget T1?

Pre-reg: REWARD_HEAD_AUTOPSY_PREREG_2026-07-10.md. Offline: for each task j, run its 20
competent phase-end eval episodes through every LATER phase checkpoint's WM (deterministic
encode) and measure reward discrimination D(j,p) = mean(pred @ success steps) − mean(pred @
zero steps). Registered read: retention ratio R(j) = D(j,3)/D(j,j). Exploratory: critic mean.

Usage (Spark):  OMP_NUM_THREADS=3 python -u reward_head_autopsy.py --logdir ~/dv3_logs/comp4b_s2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

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
SUCCESS_THR = 0.05


def load_eps(evaldir, mtime_max=None, mtime_min=None, last=20):
    files = sorted(pathlib.Path(evaldir).glob("*.npz"), key=lambda f: f.stat().st_mtime)
    if mtime_max is not None:
        files = [f for f in files if f.stat().st_mtime <= mtime_max]
    if mtime_min is not None:
        files = [f for f in files if f.stat().st_mtime > mtime_min]
    if last is not None:
        files = files[-last:]
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
    logger = _make_capture_logger(tools, logdir / "autopsy_logs", 0)
    train_envs, eval_envs = build_task_envs(D, config, TASKS[0])
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    eps0 = tools.load_episodes(logdir / f"eval_eps_0_{TASKS[0]}", limit=100)
    agent = D.Dreamer(train_envs[0].observation_space, acts, config, logger,
                      D.make_dataset(eps0, config)).to(config.device)
    agent.requires_grad_(False)
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    phase_end = {i: (logdir / f"chain_phase{i}.pt").stat().st_mtime for i in range(4)}
    episodes = {j: load_eps(logdir / f"eval_eps_{j}_{t}", phase_end[j])
                for j, t in enumerate(TASKS)}

    out = {"logdir": str(logdir), "D": {}, "critic_mean": {}, "n_success_eps": {}}
    for pck in range(4):
        sd = torch.load(logdir / f"chain_phase{pck}.pt", map_location=config.device,
                        weights_only=False)["agent_state_dict"]
        agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        agent.eval()
        wm = agent._wm
        for j in range(pck + 1):  # tasks completed by ckpt pck
            succ_preds, zero_preds, crits, n_succ = [], [], [], 0
            for ep in episodes[j]:
                batch = {k: np.asarray(v)[None] for k, v in ep.items() if "log_" not in k}
                feat = _encode_feats(agent, batch)
                with torch.no_grad():
                    pred = wm.heads["reward"](feat).mode()[0].float().cpu().numpy().squeeze(-1)
                    try:
                        crit = agent._task_behavior.value(feat).mode()[0]
                        crits.append(float(crit.float().mean()))
                    except Exception:
                        pass
                actual = np.asarray(ep["reward"]).squeeze()
                succ = actual > SUCCESS_THR
                if succ.any():
                    n_succ += 1
                    succ_preds.extend(pred[succ].tolist())
                zero_preds.extend(pred[~succ].tolist())
            d = (float(np.mean(succ_preds)) - float(np.mean(zero_preds))) if succ_preds else None
            out["D"][f"T{j}_p{pck}"] = round(d, 4) if d is not None else None
            out["critic_mean"][f"T{j}_p{pck}"] = (round(float(np.mean(crits)), 4)
                                                  if crits else None)
            out["n_success_eps"][f"T{j}_p{pck}"] = n_succ

    out["retention_ratio"] = {}
    for j in range(4):
        d_own, d_fin = out["D"].get(f"T{j}_p{j}"), out["D"].get(f"T{j}_p3")
        out["retention_ratio"][f"T{j}"] = (round(d_fin / d_own, 3)
                                           if d_own and d_fin is not None else None)

    # ---- Addendum probe (2026-07-10 audit): degraded-distribution read -----------------
    # The main D is measured on competent-era episodes that replay keeps training on — a
    # selection effect. Here: final-read era episodes (mtime > phase3 ckpt), i.e. states the
    # DRIFTED/routed policy actually visits, scored under the final ckpt (still loaded, p=3).
    # Two reads per task: D on final-era SUCCESS episodes (does the head still detect success
    # reached off the competent path?) and mean predicted reward on FAILED episodes' steps
    # (does it hallucinate reward where there is none?).
    out["final_era"] = {}
    for j in range(4):
        eps = load_eps(logdir / f"eval_eps_{j}_{TASKS[j]}", mtime_min=phase_end[3], last=None)
        succ_preds, zero_preds, fail_preds, n_s, n_f = [], [], [], 0, 0
        for ep in eps:
            batch = {k: np.asarray(v)[None] for k, v in ep.items() if "log_" not in k}
            feat = _encode_feats(agent, batch)
            with torch.no_grad():
                pred = wm.heads["reward"](feat).mode()[0].float().cpu().numpy().squeeze(-1)
            actual = np.asarray(ep["reward"]).squeeze()
            succ = actual > SUCCESS_THR
            if succ.any():
                n_s += 1
                succ_preds.extend(pred[succ].tolist())
                zero_preds.extend(pred[~succ].tolist())
            else:
                n_f += 1
                fail_preds.extend(pred.tolist())
        out["final_era"][f"T{j}"] = {
            "n_success_eps": n_s, "n_fail_eps": n_f,
            "D_success": (round(float(np.mean(succ_preds)) - float(np.mean(zero_preds)), 4)
                          if succ_preds and zero_preds else None),
            "fail_falsepos_mean": (round(float(np.mean(fail_preds)), 4)
                                   if fail_preds else None),
            "fail_falsepos_max": (round(float(np.max(fail_preds)), 4)
                                  if fail_preds else None)}

    (logdir / "reward_head_autopsy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["n_success_eps"]))
    print("\nReward discrimination D(task, ckpt):  rows=task, cols=phase ckpt")
    for j in range(4):
        cells = [f"p{pk}={out['D'].get(f'T{j}_p{pk}')}" for pk in range(j, 4)]
        print(f"  T{j}: " + "  ".join(cells) + f"   -> R(T{j}) = {out['retention_ratio'][f'T{j}']}")
    print("\nCritic mean value (exploratory):")
    for j in range(4):
        cells = [f"p{pk}={out['critic_mean'].get(f'T{j}_p{pk}')}" for pk in range(j, 4)]
        print(f"  T{j}: " + "  ".join(cells))
    print("\nFinal-era (degraded-distribution) read under p3:")
    for j in range(4):
        print(f"  T{j}: {out['final_era'][f'T{j}']}")
    print(f"\nwrote {logdir / 'reward_head_autopsy.json'}")


if __name__ == "__main__":
    main()
