"""Seed-averaged learning curves (reward + obstacle-collision rate) for the planar-drone sweeps.

Builds one figure with two panels from the TensorBoard scalars rsl-rl logs once per iteration:
  Train/mean_reward          mean return over the last 100 completed episodes
  Episode/ep_term_obstacle   fraction of the episodes that ended in the iteration by hitting the obstacle

x-axis: cumulative environment steps, (iteration + 1) * steps_per_env * num_envs, because the methods
collect different amounts of experience per iteration (RL-SGF 400 steps/env, PPO and CPO 100). Every
run is cut at the common budget (default 6.144e9 = 3M steps/env * 2048 envs).
Smoothing: each seed separately, before averaging, with a centred moving average spanning ~75M env
steps, rounded to an odd number of iterations (91 for RL-SGF, 367 for PPO/CPO).
Mean +- 95% CI across seeds (t-distribution, seeds are the independent samples): solid line + shaded
band with dashed edges. The label at the end of each line is the final smoothed mean.

Two separate figures: <out>_reward.{png,pdf} and <out>_collision.{png,pdf}.

Usage:  python plot_seeded_curves.py [--out results/seeded_curves] [--exclude RL-SGF:4]
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

NUM_ENVS = 2048
BUDGET = 3_000_000 * NUM_ENVS          # 3M steps per env, reached by every seed of every method
SMOOTH_STEPS = 75_000_000              # smoothing window in env steps
THIN_TO = 3000                         # max points drawn per curve

# key -> (run-dir glob with {seed}, steps per env per iteration, seeds, colour, plot label)
# SafePPO (experiment 15, PPO + the RL-SGF closed-form projection) was trained on another machine; its
# tensorboard logs come from the planar_safeppo_5seeds.zip hand-off unpacked under logs/external/.
METHODS = {
    "SafePPO": ("logs/external/planar_safeppo_5seeds/tensorboard/seed_{seed}",      100, [1, 2, 3, 4, 5], "#e8734a", "RSGF-RL$_1$"),
    "RL-SGF":  ("logs/exp*_t_no_bootstrap_reset_critic_epochs_7500it_seed_{seed}", 400, [1, 2, 3, 4, 5], "#3b7ddd", "RSGF-RL$_2$"),
    "CPO":     ("logs/exp*_cpo_t4_30000it_seed_{seed}",                            100, [1, 2, 3, 4, 5], "#2ab07f", "CPO"),
    "PPO":     ("logs/exp*_pure_ppo_t4_no_r1_30000it_seed_{seed}",                 100, [1, 2, 3, 4, 5], "#f0a500", "PPO"),
}
# tag, y-axis label, suffix, scale, end-label format
METRICS = [("Train/mean_reward", "Reward", "reward", 1.0, "{:.0f}"),
           ("Episode/ep_term_obstacle", "% of episodes", "collision", 100.0, "{:.2f}%")]
DEFAULT_EXCLUDE = ["RL-SGF:4"]          # seed 4 never learned obstacle avoidance (14% collisions)

BG = "#fcfcfd"
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "legend.fontsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c9c9", "axes.linewidth": 1.0,
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "grid.color": "#dddddd", "grid.linewidth": 0.9, "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.labelcolor": "#444444", "legend.frameon": False,
})

# two-sided 97.5% t quantiles for n-1 dof (n = number of seeds); avoids a scipy dependency
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def load_run(run_dir: str, tags: list[str]) -> dict[str, np.ndarray] | None:
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    have = ea.Tags()["scalars"]
    if any(t not in have for t in tags):
        return None
    out = {}
    for t in tags:
        ev = ea.Scalars(t)
        out[t] = np.array([e.value for e in ev], dtype=np.float64)
        out[t + "/step"] = np.array([e.step for e in ev], dtype=np.int64)
    return out


def smooth(y: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average with an odd window; edges use the shrinking symmetric window."""
    if win <= 1:
        return y
    half = win // 2
    c = np.concatenate([[0.0], np.cumsum(y)])
    n = len(y)
    i = np.arange(n)
    lo = np.maximum(i - half, 0)
    hi = np.minimum(i + half + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/seeded_curves", help="output stem; _reward/_collision are appended")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="METHOD:SEED pairs to drop (default: RL-SGF:4); pass --exclude with no values to keep all")
    ap.add_argument("--complete-only", action="store_true", default=True,
                    help="skip seeds that have not reached the budget (default on)")
    ap.add_argument("--allow-partial", dest="complete_only", action="store_false")
    args = ap.parse_args()
    excluded = {tuple(e.split(":")) for e in args.exclude}

    tags = [m[0] for m in METRICS]
    figs, axes = [], []
    for _ in METRICS:
        f, a = plt.subplots(figsize=(9.2, 5.4))
        figs.append(f); axes.append(a)
    end_labels = {i: [] for i in range(len(METRICS))}   # per panel: (x, y, colour, text)
    end_labels_max = {i: [] for i in range(len(METRICS))}
    notes = []

    for name, (pattern, spe, seeds, colour, disp) in METHODS.items():
        iters_budget = BUDGET // (spe * NUM_ENVS)
        win = int(round(SMOOTH_STEPS / (spe * NUM_ENVS)))
        win = win if win % 2 == 1 else win + 1
        runs = {}
        for s in seeds:
            if (name, str(s)) in excluded:
                continue
            hits = sorted(glob.glob(pattern.format(seed=s)))
            if not hits:
                continue
            data = load_run(hits[0], tags)
            if data is None:
                continue
            n_it = len(data[tags[0]])
            if n_it < iters_budget:
                if args.complete_only:
                    notes.append(f"{name} seed {s}: {n_it}/{iters_budget} iterations, skipped")
                    continue
                notes.append(f"{name} seed {s}: partial ({n_it}/{iters_budget} iterations)")
            runs[s] = data
        if not runs:
            notes.append(f"{name}: no complete runs")
            continue
        n_common = min(min(len(d[t]) for t in tags) for d in runs.values())
        n_common = min(n_common, iters_budget)
        x = (np.arange(n_common) + 1) * spe * NUM_ENVS
        used = sorted(runs)
        n = len(used)
        for i, (ax, (tag, label, suffix, scale, fmt)) in enumerate(zip(axes, METRICS)):
            Y = scale * np.stack([smooth(runs[s][tag][:n_common], win) for s in used])   # (n_seeds, n_it)
            mean = Y.mean(0)
            if n >= 2:
                sd = Y.std(0, ddof=1)
                ci = T975.get(n, 1.96) * sd / math.sqrt(n)
            else:
                ci = np.zeros_like(mean)
            if n_common > THIN_TO:
                idx = np.unique(np.round(np.linspace(0, n_common - 1, THIN_TO)).astype(int))
            else:
                idx = np.arange(n_common)
            ax.plot(x[idx], mean[idx], color=colour, lw=3.2, label=f"{disp} (n={n})", solid_capstyle="round")
            if n >= 2:
                ax.fill_between(x[idx], (mean - ci)[idx], (mean + ci)[idx], color=colour, alpha=0.15, lw=0)
                ax.plot(x[idx], (mean - ci)[idx], color=colour, lw=0.8, ls="--", alpha=0.6)
                ax.plot(x[idx], (mean + ci)[idx], color=colour, lw=0.8, ls="--", alpha=0.6)
            ax.plot([x[-1]], [mean[-1]], "o", color=colour, ms=6, zorder=5)
            end_labels[i].append((x[-1], mean[-1], colour, f"{disp} {fmt.format(mean[-1])}"))
            end_labels_max[i].append((name, mean.max(), None, None))
        notes.append(f"{name}: seeds {used}, {spe} steps/env/it, smoothing window {win} it")

    for i, (fig, ax, (tag, label, suffix, scale, fmt)) in enumerate(zip(figs, axes, METRICS)):
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(label)
        ax.set_xlim(0, BUDGET * 1.01)
        ax.grid(True, alpha=1.0)
        ax.set_axisbelow(True)
        if "obstacle" in tag:
            # clip to the means (the n=2 CPO band with t=12.7 would otherwise set the scale)
            ax.set_ylim(0, 1.15 * max(m for _, m, _, _ in end_labels_max[i]))
            ax.legend(loc="upper center")
        else:
            ax.set_ylim(0, 1.08 * max(m for _, m, _, _ in end_labels_max[i]))
            ax.legend(loc="lower right", bbox_to_anchor=(0.72, 0.02))
        # end-of-line labels outside the axes, pushed apart so they never overlap, with leader lines
        labels = sorted(end_labels[i], key=lambda t: t[1])
        y0, y1 = ax.get_ylim()
        min_gap = 0.075 * (y1 - y0)
        ys = [t[1] for t in labels]
        for j in range(1, len(ys)):
            ys[j] = max(ys[j], ys[j - 1] + min_gap)
        overflow = ys[-1] + 0.4 * min_gap - y1          # keep the label stack inside the axes
        if overflow > 0:
            ys = [y - overflow for y in ys]
        x_text = BUDGET * 1.045
        for (xe, ye, colour, text), yt in zip(labels, ys):
            ax.annotate(text, xy=(xe, ye), xytext=(x_text, yt), textcoords="data", color="#333333",
                        fontsize=14, va="center", ha="left", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=colour, lw=1.2, shrinkA=0, shrinkB=4,
                                        connectionstyle="arc3,rad=0"))
        fig.subplots_adjust(left=0.10, right=0.80, bottom=0.14, top=0.96)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        stem = f"{args.out}_{suffix}"
        fig.savefig(stem + ".png", dpi=170)
        fig.savefig(stem + ".pdf")
        print("wrote", stem + ".png / .pdf")
    for n_ in notes:
        print(" ", n_)


if __name__ == "__main__":
    main()
