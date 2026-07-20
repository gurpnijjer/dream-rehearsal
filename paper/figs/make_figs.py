"""Paper figures. Palette: dataviz reference (validated): blue #2a78d6 (ours),
aqua #1baf7a (composite), yellow #eda100 (plain/baseline). Light mode, print."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#333333", "#777777", "#dddddd"

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "figure.dpi": 300})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------- Fig 1: recovery race --------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 2.9))
runs = {"tunnel": (BLUE, "Dream self-imitation — 3/3 recovered"),
        "rl": (YELLOW, "RL-in-imagination — 0/3")}
for arm, (color, _label) in runs.items():
    for seed in ("comp4_s1", "comp4b_s2", "comp4_s3"):
        rows = [json.loads(l) for l in (DATA / f"{seed}_{arm}.jsonl").read_text().splitlines()]
        xs = [r["update"] / 1000 for r in rows]
        ys = [r["t1"] for r in rows]
        ax.plot(xs, ys, color=color, lw=1.8, alpha=0.85, zorder=3,
                solid_capstyle="round")
ax.axhline(0.75, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(19.7, 0.77, "pass bar (0.75, held 3 evals)", ha="right", color=MUTED, fontsize=7)
ax.text(5.2, 0.97, "Dream self-imitation (3/3, zero env steps)", color=BLUE, fontsize=8,
        fontweight="bold")
ax.text(11.5, 0.18, "RL-in-imagination (0/3, same dreams)", color="#b07800", fontsize=8,
        fontweight="bold")
ax.set_xlabel("Imagination-only updates (thousands)")
ax.set_ylabel("Lost-task real-env return")
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.02)
style(ax)
fig.tight_layout()
fig.savefig(HERE / "fig1_recovery_race.png", bbox_inches="tight")
fig.savefig(HERE / "fig1_recovery_race.pdf", bbox_inches="tight")

# ---------------- Fig 2: final retention by architecture --------------------------------
tasks = ["DoorKey\n(first task)", "SimpleCrossing\n(hardest)", "LavaGap\n(lethal)",
         "MultiRoom\n(last task)"]
per_seed = {
    "Plain replay (0/3)": (YELLOW, [[0.962, 0.961, 0.962], [0.347, 0.509, 0.247],
                                    [0.945, 0.944, 0.945], [0.786, 0.809, 0.824]]),
    "Frozen heads + router (3/5)": (AQUA, [[0.964, 0.964, 0.961, 0.964, 0.965],
                                           [0.474, 0.676, 0.633, 0.811, 0.494],
                                           [0.877, 0.945, 0.935, 0.924, 0.943],
                                           [0.791, 0.761, 0.682, 0.827, 0.812]]),
    "Dream rehearsal (3/3, ours)": (BLUE, [[0.959, 0.956, 0.958], [0.905, 0.740, 0.826],
                                           [0.760, 0.662, 0.727], [0.814, 0.806, 0.799]]),
}
fig, ax = plt.subplots(figsize=(4.9, 2.9))
w, n = 0.26, len(per_seed)
for i, (name, (color, vals)) in enumerate(per_seed.items()):
    xs = [t + (i - (n - 1) / 2) * w for t in range(len(tasks))]
    means = [sum(v) / len(v) for v in vals]
    ax.bar(xs, means, width=w - 0.03, color=color, zorder=3, label=name)
    for x, v in zip(xs, vals):
        ax.scatter([x] * len(v), v, s=5, color=INK, alpha=0.55, zorder=4, lw=0)
    for x, m in zip(xs, means):
        ax.text(x, 0.03, f"{m:.2f}", ha="center", color="white", fontsize=6.4,
                fontweight="bold", zorder=5)
ax.axhline(0.6, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.text(3.42, 0.615, "pass bar", ha="right", color=MUTED, fontsize=7)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(tasks, fontsize=7.5, color=INK)
ax.set_ylabel("Final retention (real-env return)")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
          fontsize=7, handlelength=1.2, columnspacing=1.0)
style(ax)
fig.tight_layout()
fig.savefig(HERE / "fig2_retention.png", bbox_inches="tight")
fig.savefig(HERE / "fig2_retention.pdf", bbox_inches="tight")
print("wrote", [p.name for p in HERE.glob("fig*.p*")])
