# Run data — every experiment in the paper

Each directory is one run. `chain_summary.json` / `ab_summary.json` hold the final read
(per-task retention, pass/fail, protocol constants as executed); `chain_metrics.jsonl` /
`ab_metrics.jsonl` hold **every evaluation round** — one row per round with the real-environment
return of every learned task at that point, so the full trajectory of every run is inspectable,
not just its endpoint.

| Directory | Paper section | What it is |
|---|---|---|
| `ab_s{1,2,3}` | §4.1 | Two tasks, **no replay** — the raw phenomenon (task A: 0.96 → 0.0/0.27/0.12) |
| `ab_rp*` (not included) | §4.1 | Partial-replay fraction sweep; superseded by the never-clear recipe |
| `ab_frz_s{1,2,3}` | §4.2 | **Freeze bracket, actor frozen** at task-A competence (retention 0.0/0.0/0.07) |
| `ab_frzrep_s1` | §4.2 | **Freeze bracket, world model frozen** — task B never learns; one seed, terminated early, reported as direction-finding only |
| `ab_fl_s{1,2,3}` | §4 | Frozen-head/live-actor comparison runs |
| `chain4_s{1,2,3}` | §4.1, Fig. 2 | **Plain never-clear replay**, four tasks — 0/3 chains pass |
| `comp4_s{1,2,3}`, `comp4b_s{2,4,5}` | §4.4, Fig. 2 | **Isolation reference**: frozen per-task actor snapshots + nearest-centroid router (routed read 3/5). `comp4b_*` are the corrected re-reads after the episode-ordering bug (both reads retained on purpose) |
| `tunl4_s{1,2,3}` | §6.1 | Four-task dream rehearsal, **initial scorer** (the run whose short-horizon task wobbled — 27/19/42% stability) |
| `tunl4b_s{1,2,3}` | §6.1, §7, Fig. 2 | Four-task dream rehearsal, **corrected realized-first scorer** (stability 95/100/67%) |
| `realbcf_s{1,2,3}` | §6.2 | **Real-episode cloning** (competent-filtered), the matched comparison arm |
| `tunl8_s{1,2,3}` | §8 | **Eight-task chains** — all three seeds retain all eight |

Recovery-race curves (§5, Fig. 1) are in [`../paper/figs/data/`](../paper/figs/data/) —
`comp4_s{1,3}_rl.jsonl` / `_tunnel.jsonl` and `comp4b_s2_*`: real-environment return every 500
imagination-only updates, per arm per seed.

Note on what is **not** here: raw episode buffers (hundreds of GB, and reconstructable) and
model checkpoints. Everything needed to verify the numbers in the paper is included.
