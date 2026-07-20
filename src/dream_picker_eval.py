"""dream_picker_eval.py — offline picker harness: rank scorer variants on banked dream data.

Selector-style evaluation of the tunnel's dream-grader (ESCALATION_LADDER_DESIGN §A), run on
BANKED tunl4 artifacts — measurement only, no arms, no policy updates.

For each start state we imagine K rollouts (sampling actor, final ckpt) and label each by its
own imagined outcome (success: reward>0.3 before first termination; death: terminated with
reward<0.05). Scorer variants ranked by how well they select success-dreams:

  deployed   : sum gamma^t r + gamma^H V(last)                      (tunl4's flawed scorer)
  cont       : survival-weighted rewards + survival-weighted V      (Fix A, running in tunl4b)
  cont+cons  : cont score * consensus weight, where consensus = fraction of the K siblings
               from the same start that share this rollout's outcome label (Selector's
               output_consensus analog)

Metrics per variant: AUC (score vs success label), top-25% composition, and for the best
variant an ABSTENTION curve — keep only top-25% picks whose score-margin over the batch median
exceeds a threshold; report purity vs coverage (the Selector's accuracy-at-coverage analog).

Usage: OMP_NUM_THREADS=3 python -u dream_picker_eval.py --logdir ~/dv3_logs/tunl4_s1 --task 2
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
K_SIB = 4
N_BATCH = 6


def auc(scores, labels):
    s, l = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    if l.all() or (~l).all():
        return None
    pos, neg = s[l], s[~l]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return round(float(wins / (len(pos) * len(neg))), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--task", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    logdir = pathlib.Path(args.logdir).expanduser()
    j = args.task

    overrides = dict(task=TASKS[0], size=[64, 64], device=args.device, compile=False,
                     video_pred_log=False, seed=1, steps=int(1e9), eval_episode_num=10,
                     expl_behavior="greedy", expl_until=0, expl_extr_scale=1.0)
    config = build_config(["minigrid"], overrides)
    tools.set_seed_everywhere(config.seed)
    logger = _make_capture_logger(tools, logdir / "picker_logs", 0)
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
    lib = load_eps_window(logdir / "train_eps",
                          phase_end[j - 1] if j else None, phase_end[j])
    eps0 = tools.load_episodes(logdir / f"eval_eps_0_{TASKS[0]}", limit=50)
    agent = D.Dreamer(obs_space, acts, config, logger,
                      D.make_dataset(eps0, config)).to(config.device)
    agent.requires_grad_(False)
    sd = torch.load(logdir / "chain_phase3.pt", map_location=config.device,
                    weights_only=False)["agent_state_dict"]
    agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    agent.eval()
    wm, behavior = agent._wm, agent._task_behavior

    ds = D.make_dataset(lib, config)
    rows = []  # per rollout: scores + label
    for _ in range(N_BATCH):
        batch = next(ds)
        with torch.no_grad():
            data = wm.preprocess(batch)
            embed = wm.encoder(data)
            post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
            start = {k: v.detach() for k, v in post.items()}
            sib_scores_dep, sib_scores_cont, sib_scores_shift, sib_scores_lam, \
                sib_scores_lex, sib_labels = [], [], [], [], [], []
            for _k in range(K_SIB):
                feats, states, _ac = behavior._imagine(start, behavior.actor, config.imag_horizon)
                r = wm.heads["reward"](feats).mode().squeeze(-1)
                cont = wm.heads["cont"](feats).mean.squeeze(-1)
                v_last = behavior.value(feats[-1]).mode().squeeze(-1)
                H = r.shape[0]
                gam = config.discount ** torch.arange(H, device=r.device,
                                                      dtype=r.dtype).unsqueeze(-1)
                surv = torch.cumprod(cont, dim=0)
                # reach-probability weights: prob of being alive ENTERING step t (shifted by 1)
                reach = torch.cat([torch.ones_like(cont[:1]), surv[:-1]], dim=0)
                s_dep = (gam * r).sum(0) + config.discount ** H * v_last
                s_cont = (gam * surv * r).sum(0) + config.discount ** H * surv[-1] * v_last
                s_shift = (gam * reach * r).sum(0) + config.discount ** H * surv[-1] * v_last
                # lam: DreamerV3's own lambda-return target — the battle-tested trajectory
                # valuation (cont-discounted, realized rewards dominate bootstrap promises)
                rew_full = wm.heads["reward"](feats).mode()
                target, _w, _b = behavior._compute_target(feats, states, rew_full)
                tgt = target if torch.is_tensor(target) else torch.stack(list(target), dim=1)
                s_lam = tgt[0].squeeze(-1)  # lambda-return at the first imagined step, [B]
                # lex: realized-first — dreams that actually scored outrank value promises;
                # shift score orders within each group (Selector principle: truth > confidence)
                achieved = ((gam * reach * r).sum(0) > 0.3).float()
                s_lex = achieved * 10.0 + s_shift
                term = cont < 0.5
                any_term = term.any(dim=0)
                tstar = torch.where(any_term, term.float().argmax(dim=0),
                                    torch.full_like(any_term, H - 1, dtype=torch.long))
                idx = torch.arange(H, device=r.device).unsqueeze(-1).expand_as(r)
                rmax = (r * (idx <= tstar.unsqueeze(0))).max(dim=0).values
                success = (any_term & (rmax > 0.3)) | (~any_term & (rmax > 0.3))
                sib_scores_dep.append(s_dep.cpu().numpy())
                sib_scores_cont.append(s_cont.cpu().numpy())
                sib_scores_shift.append(s_shift.cpu().numpy())
                sib_scores_lam.append(s_lam.float().cpu().numpy())
                sib_scores_lex.append(s_lex.float().cpu().numpy())
                sib_labels.append(success.cpu().numpy())
            sd_ = np.stack(sib_scores_dep)     # [K, B]
            sc_ = np.stack(sib_scores_cont)
            sh_ = np.stack(sib_scores_shift)
            sl_ = np.stack(sib_scores_lam)
            sx_ = np.stack(sib_scores_lex)
            lb_ = np.stack(sib_labels)
            cons = lb_.mean(0, keepdims=True)  # fraction of siblings succeeding per start
            cons_w = np.where(lb_, cons, 1 - cons)  # agreement of THIS rollout with siblings
            for kk in range(K_SIB):
                for bb in range(sd_.shape[1]):
                    rows.append({"dep": float(sd_[kk, bb]), "cont": float(sc_[kk, bb]),
                                 "shift": float(sh_[kk, bb]), "lam": float(sl_[kk, bb]),
                                 "lex": float(sx_[kk, bb]),
                                 "cons": float(sh_[kk, bb]) * float(cons_w[kk, bb]),
                                 "y": bool(lb_[kk, bb])})

    y = [r["y"] for r in rows]
    out = {"logdir": str(logdir), "task": f"T{j}", "n_rollouts": len(rows),
           "base_success_rate": round(float(np.mean(y)), 3), "variants": {}}
    for name in ("dep", "cont", "shift", "lam", "lex", "cons"):
        s = [r[name] for r in rows]
        order = np.argsort(s)[::-1]
        k25 = max(1, len(rows) // 4)
        top = order[:k25]
        out["variants"][name] = {
            "auc": auc(s, y),
            "top25_success": round(float(np.mean([y[i] for i in top])), 3)}
    # abstention curve on the best-AUC variant
    best = max(out["variants"], key=lambda n: out["variants"][n]["auc"] or 0)
    s = np.asarray([r[best] for r in rows])
    med = np.median(s)
    curve = []
    for q in (0.0, 0.25, 0.5, 0.75, 0.9):
        thr = med + q * (s.max() - med)
        keep = s >= thr
        if keep.sum() >= 20:
            curve.append({"margin_q": q, "coverage": round(float(keep.mean()), 3),
                          "purity": round(float(np.mean(np.asarray(y)[keep])), 3)})
    out["abstention_curve_best"] = {"variant": best, "points": curve}
    (logdir / f"dream_picker_eval_T{j}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
