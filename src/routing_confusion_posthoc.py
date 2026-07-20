"""routing_confusion_posthoc.py — why does T1 (SimpleCrossing) misroute at the FINAL read?

Post-hoc, read-only. Loads chain_phase3.pt (final WM/encoder), rebuilds the albums exactly as
the orchestrator's phase-end snapshot rule (last 20 eval eps with mtime <= phase-i ckpt mtime),
then probes routing decisions offline:

  H1 (4th-centroid): in-phase routing picks among centroids {0,1,2}; the T3 centroid only enters
      at the final read. Compare per-task accuracy with 3 vs 4 centroids.
  H2 (statistic mismatch): centroids are FULL-episode mean embeddings but the router commits on
      the first-10-step running mean. Sweep commit k and try statistic-matched centroids
      (built from first-10 means of album episodes).

Probe sets per task: album episodes' own-phase peers (clean, competent) AND the episodes written
AFTER the phase3 ckpt (the exact final-read rollouts the deployed router saw).

Usage (Spark):  OMP_NUM_THREADS=3 python -u routing_confusion_posthoc.py --logdir ~/dv3_logs/comp4b_s2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

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

TASKS = ["minigrid_DoorKey-5x5", "minigrid_SimpleCrossingS9N1",
         "minigrid_LavaGapS5", "minigrid_MultiRoom-N2-S4"]
KS = [1, 3, 5, 10, 15, 20, 30]  # commit-k sweep; "full" added separately


def frame_embeds(agent, ep):
    """Per-frame embeddings [T, E] under the current encoder (matches RoutedEvalPolicy._embed
    frame-for-frame: the encoder is per-frame, no temporal mixing)."""
    wm = agent._wm
    batch = {k: np.asarray(v)[None] for k, v in ep.items() if "log_" not in k}
    with torch.no_grad():
        data = wm.preprocess(batch)
        e = wm.encoder(data)  # [1, T, E]
    return e[0].detach().cpu().numpy()


def load_eps(evaldir, mtime_max=None, mtime_min=None, last=None):
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


def route(mean_emb, cents):
    d = np.linalg.norm(cents - mean_emb[None], axis=1)
    return int(d.argmin()), d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe_eps", type=int, default=40)
    p.add_argument("--album_rule", choices=["mtime", "deployed"], default="mtime",
                   help="mtime = intended rule (newest 20 competent eps at phase end); "
                        "deployed = replicate the orchestrator's album_src[-20:] on the "
                        "newest-first load_episodes OrderedDict, i.e. the OLDEST 20")
    args = p.parse_args()
    logdir = pathlib.Path(args.logdir).expanduser()

    overrides = dict(task=TASKS[0], size=[64, 64], device=args.device, compile=False,
                     video_pred_log=False, seed=1, steps=int(1e9), eval_episode_num=10,
                     expl_behavior="greedy", expl_until=0, expl_extr_scale=1.0)
    config = build_config(["minigrid"], overrides)
    tools.set_seed_everywhere(config.seed)
    logger = _make_capture_logger(tools, logdir / "routing_confusion_logs", 0)

    train_envs, eval_envs = build_task_envs(D, config, TASKS[0])
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    eval0 = logdir / f"eval_eps_0_{TASKS[0]}"
    eps0 = tools.load_episodes(eval0, limit=100)
    agent = D.Dreamer(train_envs[0].observation_space, acts, config, logger,
                      D.make_dataset(eps0, config)).to(config.device)
    agent.requires_grad_(False)
    sd = torch.load(logdir / "chain_phase3.pt", map_location=config.device,
                    weights_only=False)["agent_state_dict"]
    agent.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    agent.eval()
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    phase_end = {i: (logdir / f"chain_phase{i}.pt").stat().st_mtime for i in range(4)}
    evaldirs = {j: logdir / f"eval_eps_{j}_{t}" for j, t in enumerate(TASKS)}

    # Albums: intended rule = last 20 eval eps existing at phase-j end. Deployed rule
    # replicates the album bug: load_episodes is newest-first, so album_src[-20:] takes
    # the OLDEST 20 (early-phase incompetent episodes).
    if args.album_rule == "deployed":
        albums = {}
        for j in range(4):
            disk = tools.load_episodes(evaldirs[j], limit=10 ** 9)  # newest first
            src = [e for e in disk.values() if len(next(iter(e.values()))) >= 3]
            # restrict to files existing at phase-j end, preserving newest-first order
            n_at_end = len(load_eps(evaldirs[j], mtime_max=phase_end[j]))
            albums[j] = src[len(src) - n_at_end:][-20:] if n_at_end else src[-20:]
    else:
        albums = {j: load_eps(evaldirs[j], mtime_max=phase_end[j], last=20) for j in range(4)}
    for j in range(4):
        assert albums[j], f"empty album {j}"
    album_embeds = {j: [frame_embeds(agent, ep) for ep in albums[j]] for j in range(4)}

    # Two centroid recipes: full-episode mean (deployed) vs first-10 running mean (matched).
    cents_full = np.stack([np.mean([e.mean(0) for e in album_embeds[j]], axis=0)
                           for j in range(4)])
    cents_k10 = np.stack([np.mean([e[:10].mean(0) for e in album_embeds[j]], axis=0)
                          for j in range(4)])

    # Probe sets: clean own-phase episodes (excluding album members: take the 40 BEFORE the
    # album's 20) + the post-phase3 final-read rollouts.
    probes = {}
    for j in range(4):
        clean = load_eps(evaldirs[j], mtime_max=phase_end[j], last=20 + args.probe_eps)
        probes[(j, "clean")] = clean[:-20] if len(clean) > 20 else []
        probes[(j, "final")] = load_eps(evaldirs[j], mtime_min=phase_end[3])

    report = {"logdir": str(logdir), "n_probe": {}, "results": {}}
    for (j, kind), eps in sorted(probes.items()):
        if not eps:
            continue
        embeds = [frame_embeds(agent, ep) for ep in eps]
        report["n_probe"][f"T{j}_{kind}"] = len(eps)
        for label, cents, ks in (("full4", cents_full, KS + ["full"]),
                                 ("full3", cents_full[:3], [10]),
                                 ("matched4", cents_k10, [10])):
            if label == "full3" and j == 3:
                continue
            for k in ks:
                picks, margins = [], []
                for e in embeds:
                    m = e.mean(0) if k == "full" else e[:min(k, len(e))].mean(0)
                    pick, d = route(m, cents)
                    picks.append(pick)
                    others = np.delete(d, j) if j < len(d) else d
                    margins.append(float(d[j] - others.min()) if j < len(d) else None)
                acc = float(np.mean([p == j for p in picks]))
                row = {"acc": round(acc, 3),
                       "confusion": dict(Counter(picks)),
                       "median_margin": (round(float(np.median([m for m in margins
                                                                if m is not None])), 3)
                                         if margins and margins[0] is not None else None)}
                report["results"][f"T{j}_{kind}_{label}_k{k}"] = row

    out = logdir / "routing_confusion.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["n_probe"]))
    print(f"\n{'probe':28s} {'set':9s} {'k':>4s}  acc   picks->  margin(d_true - d_bestwrong)")
    for key, row in report["results"].items():
        print(f"{key:42s} acc={row['acc']:.2f}  conf={row['confusion']}  "
              f"marg={row['median_margin']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
