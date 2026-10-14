#!/usr/bin/env python3
"""Paired seed comparison for two training settings.

Runs setting A and setting B on the SAME set of random seeds, logs each run to
comparisons/<name>/<A|B>/seed_<s>/, then builds 95% t-CI bands (over seeds,
aligned by iteration index) for a chosen list of metrics and plots A vs B.

Example:
    nohup python run_comparison.py \
        --name comp01 --iterations 2000 --seeds 5 \
        --A "algorithm.q=1" --B "algorithm.q=2" \
        --metrics Loss/V0 Loss/V1 Train/mean_reward Loss/u \
        > comparisons/comp01.out 2>&1 &
"""

from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- #
# 1. run the training subprocesses
# ----------------------------------------------------------------------------- #
def launch_runs(root: Path, iterations: int, seeds: list[int],
                settings: dict[str, list[str]], train_script: str,
                config: str) -> None:
    for label, overrides in settings.items():
        for seed in seeds:
            out = root / label / f"seed_{seed}"
            if (out / "DONE").exists():
                print(f"[skip] {label}/seed_{seed} already complete", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, train_script,
                "--config", config,
                "--iterations", str(iterations),
                "--seed", str(seed),
                "--log-dir", str(out),
                "--no-show-dashboard-after",
            ]
            for o in overrides:
                cmd += ["--set", o]

            print(f"[run ] {label}/seed_{seed}: {' '.join(cmd)}", flush=True)
            # inherit stdout/stderr so nohup captures everything in one file
            subprocess.run(cmd, check=True)
            (out / "DONE").touch()


# ----------------------------------------------------------------------------- #
# 2. read TensorBoard event files back
# ----------------------------------------------------------------------------- #
def load_run(run_dir: Path) -> dict[str, np.ndarray]:
    """Return {tag: values aligned by logging order} for one run.

    Uses tbparse. Values are ordered by step; we index by position (iteration
    order), not by step value, so runs of differing step counts still align.
    """
    from tbparse import SummaryReader
    reader = SummaryReader(str(run_dir), pivot=False)
    df = reader.scalars
    if df.empty:
        return {}
    out: dict[str, np.ndarray] = {}
    for tag, sub in df.groupby("tag"):
        sub = sub.sort_values("step")
        out[tag] = sub["value"].to_numpy()
    return out


def collect(root: Path, label: str, seeds: list[int]) -> dict[str, list[np.ndarray]]:
    """{tag: [seed0_series, seed1_series, ...]} for one setting."""
    per_tag: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        run = load_run(root / label / f"seed_{seed}")
        for tag, vals in run.items():
            per_tag.setdefault(tag, []).append(vals)
    return per_tag


# ----------------------------------------------------------------------------- #
# 3. t-CI band across seeds, aligned by iteration index, NaN-safe
# ----------------------------------------------------------------------------- #
def ci_band(series_list: list[np.ndarray], level: float = 0.95):
    """mean, lower, upper, n_valid  -- each length = shortest series.

    Aligns by position, uses Student-t (correct for small n), and treats
    non-finite entries (a NaN'd seed) as missing rather than poisoning the mean.
    """
    from scipy import stats

    if not series_list:
        return None
    L = min(len(s) for s in series_list)
    M = np.stack([np.asarray(s[:L], dtype=float) for s in series_list])  # (n, L)

    valid = np.isfinite(M)
    n = valid.sum(axis=0)                                    # seeds present per iter
    safe = np.where(valid, M, 0.0)
    total = safe.sum(axis=0)
    mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)

    # sample std with ddof=1, over valid entries only
    diff2 = np.where(valid, (M - mean) ** 2, 0.0).sum(axis=0)
    var = np.where(n > 1, diff2 / np.maximum(n - 1, 1), np.nan)
    sem = np.sqrt(var) / np.sqrt(np.maximum(n, 1))

    # per-iteration t quantile (df = n-1); n can vary if seeds NaN out
    tq = np.where(n > 1, stats.t.ppf(0.5 + level / 2, df=np.maximum(n - 1, 1)), np.nan)
    half = tq * sem
    return mean, mean - half, mean + half, n


# ----------------------------------------------------------------------------- #
# 4. plot A vs B for each requested metric
# ----------------------------------------------------------------------------- #
def plot_metrics(root: Path, labels: list[str], seeds: list[int],
                 metrics: list[str], level: float) -> None:
    import matplotlib
    matplotlib.use("Agg")                # headless, safe under nohup
    import matplotlib.pyplot as plt

    collected = {lab: collect(root, lab, seeds) for lab in labels}
    colors = {labels[0]: "tab:blue", labels[1]: "tab:red"}
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        any_data = False
        for lab in labels:
            series = collected[lab].get(metric)
            if not series:
                print(f"[warn] metric '{metric}' missing for setting {lab}", flush=True)
                continue
            band = ci_band(series, level)
            if band is None:
                continue
            mean, lo, hi, n = band
            x = np.arange(len(mean))
            c = colors.get(lab, None)
            ax.plot(x, mean, label=f"{lab} (n\u2264{len(series)})", color=c, lw=1.5)
            ax.fill_between(x, lo, hi, alpha=0.25, color=c, linewidth=0)
            # mark where any seed dropped out (n < max)
            if (n < n.max()).any():
                drop = x[n < n.max()]
                ax.scatter(drop, mean[drop], s=4, color=c, alpha=0.4)
            any_data = True

        if not any_data:
            plt.close(fig)
            continue
        ax.set_title(f"{metric}   ({int(level*100)}% t-CI over {len(seeds)} seeds)")
        ax.set_xlabel("update index")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(alpha=0.3)
        safe_name = metric.replace("/", "__")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=130)
        plt.close(fig)
        print(f"[plot] {plot_dir / (safe_name + '.png')}", flush=True)

    # endpoint table: last-100-update mean +/- t-CI per setting per metric
    from scipy import stats
    lines = []
    for metric in metrics:
        row = [metric]
        for lab in labels:
            series = collected[lab].get(metric)
            if not series:
                row.append("n/a")
                continue
            L = min(len(s) for s in series)
            tail = np.array([np.nanmean(np.asarray(s[:L])[-100:]) for s in series])
            tail = tail[np.isfinite(tail)]
            if len(tail) < 2:
                row.append("n/a")
                continue
            m = tail.mean()
            h = stats.t.ppf(0.5 + level / 2, df=len(tail) - 1) * tail.std(ddof=1) / np.sqrt(len(tail))
            row.append(f"{m:.4f} \u00b1 {h:.4f}")
        lines.append("\t".join(row))
    table = "metric\t" + "\t".join(labels) + "\n" + "\n".join(lines)
    (root / "endpoint_table.tsv").write_text(table)
    print("\n[endpoint: last-100-update mean \u00b1 t-CI]\n" + table, flush=True)


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Paired-seed A/B training comparison.")
    p.add_argument("--name", required=True, help="comparison name -> comparisons/<name>/")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds")
    p.add_argument("--base-seed", type=int, default=0, help="seed for choosing the seed set")
    p.add_argument("--A", nargs="*", default=[], help="setting-A overrides, e.g. algorithm.q=1")
    p.add_argument("--B", nargs="*", default=[], help="setting-B overrides, e.g. algorithm.q=2")
    p.add_argument("--metrics", nargs="*", required=True,
                   help="TensorBoard tags to plot, e.g. Loss/V0 Loss/V1 Train/mean_reward")
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--train-script", default="planar_drone/train.py")
    p.add_argument("--config", default="config_rlsgf.yaml")
    p.add_argument("--plot-only", action="store_true",
                   help="skip training, just regenerate plots from existing logs")
    args = p.parse_args()

    root = Path("comparisons") / args.name
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.base_seed)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(args.seeds)]
    (root / "seeds.json").write_text(json.dumps(seeds))
    settings = {"A": args.A, "B": args.B}
    (root / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"[seeds] {seeds}", flush=True)
    print(f"[A] {args.A}\n[B] {args.B}", flush=True)

    if not args.plot_only:
        launch_runs(root, args.iterations, seeds, settings, args.train_script, args.config)

    plot_metrics(root, ["A", "B"], seeds, args.metrics, args.level)


if __name__ == "__main__":
    main()