"""Fig 3: eight-task chain retention (scale figure). Reads the released run summaries in
results/tunl8_s{1,2,3}/chain_summary.json — no hardcoded numbers. Same palette as figs 1-2."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
RESULTS = HERE.parent.parent / "results"
BLUE, INK, MUTED, GRID = "#2a78d6", "#333333", "#777777", "#dddddd"
plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "figure.dpi": 300})

SHORT = {"DoorKey-5x5": "DoorKey5", "SimpleCrossingS9N1": "SimpleX", "LavaGapS5": "LavaGap",
         "MultiRoom-N2-S4": "MultiRoom", "LavaCrossingS9N1": "LavaCross",
         "DistShift2": "DistShift", "DoorKey-6x6": "DoorKey6", "Unlock": "Unlock"}

seeds = []
for s in (1, 2, 3):
    d = json.loads((RESULTS / f"tunl8_s{s}" / "chain_summary.json").read_text())
    seeds.append({SHORT[k.replace("minigrid_", "")]: v
                  for k, v in d["final_retention"].items()})
tasks = list(seeds[0])
means = [np.mean([s[t] for s in seeds]) for t in tasks]

fig, ax = plt.subplots(figsize=(6.2, 2.9))
x = np.arange(len(tasks))
ax.bar(x, means, width=0.62, color=BLUE, zorder=3)
for i, t in enumerate(tasks):                       # per-seed points over the bar
    ax.scatter([i] * 3, [s[t] for s in seeds], s=11, color="#333333", alpha=0.75, zorder=4)
    ax.text(i, 0.045, f"{means[i]:.2f}", ha="center", color="white", fontsize=7.5,
            fontweight="bold", zorder=5)
ax.axhline(0.6, ls="--", lw=0.8, color=MUTED, zorder=2)
ax.text(len(tasks) - 0.4, 0.615, "pass bar", color=MUTED, fontsize=7.5, ha="right")
ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=18, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Final retention (real-env return)")
ax.set_title("Eight-task chain: all 3 seeds retain all 8 tasks", fontsize=9.5,
             color=INK, loc="left", pad=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig3_eight_task.{ext}", bbox_inches="tight")
print("wrote fig3_eight_task.png/.pdf from", RESULTS)
