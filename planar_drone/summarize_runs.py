"""Summarize training runs from their TensorBoard event files.

Usage:
    python summarize_runs.py --out results/summary.md --window-steps 164000000 RUN [RUN ...]

For each run (a directory name under planar_drone/logs, or a glob) it reports
mean reward, unsafe-step fraction and obstacle-termination rate averaged over the
final `window-steps` environment steps (so runs with different rollout lengths
are compared over the same number of samples), plus the final logged value.
"""
import argparse, glob, os, sys, datetime, json
import numpy as np
from tensorboard.backend.event_processing import event_file_loader

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_ENVS = 2048
TAGS = {"mean_reward": "reward", "frac_steps_unsafe": "unsafe frac", "frac_steps_term_obstacle": "obstacle rate"}


def load_run(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents*")))
    if not files:
        return None
    data = {k: [] for k in TAGS}
    steps = []
    for f in files:
        for ev in event_file_loader.LegacyEventFileLoader(f).Load():
            if not ev.HasField("summary"):
                continue
            for v in ev.summary.value:
                k = v.tag.split("/")[-1]
                if k in data:
                    data[k].append(v.simple_value)
                if k == "mean_reward":
                    steps.append(ev.step)
    # env steps per iteration = rollout length * num_envs (num_envs is fixed at 2048 in train.py)
    spi = 0
    cfg_path = os.path.join(run_dir, "config_used.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            spi = int(json.load(f).get("num_steps_per_env", 0)) * NUM_ENVS
    return {k: np.asarray(v) for k, v in data.items()}, spi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Run summary")
    ap.add_argument("--window-steps", type=float, default=2048 * 400 * 200,
                    help="env steps at the end of training to average over (default: 200 T-iterations)")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    rows = []
    for pat in args.runs:
        dirs = sorted(glob.glob(os.path.join(HERE, "logs", pat)))
        if not dirs:
            rows.append((pat, None)); continue
        for d in dirs:
            rows.append((os.path.basename(d), load_run(d)))

    lines = [f"# {args.title}", "", f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}.", ""]
    if args.notes:
        lines += [args.notes, ""]
    lines += [f"Values are means over the final {args.window_steps:.3g} env steps of each run "
              "(same sample window regardless of rollout length); 'final' is the last logged iteration.", "",
              "| run | iters | steps/iter | reward (mean / final) | unsafe frac (mean / final) | obstacle rate (mean / final) | ~obstacle hits per 819k steps |",
              "|---|---|---|---|---|---|---|"]
    for name, res in rows:
        if res is None:
            lines.append(f"| {name} | (no event file found) | | | | | |"); continue
        data, spi = res
        n_it = len(data["mean_reward"])
        n = max(1, int(round(args.window_steps / spi))) if spi else n_it
        cell = {}
        for k in TAGS:
            v = data[k]
            if len(v) == 0:
                cell[k] = ("n/a", "n/a"); continue
            cell[k] = (v[-n:].mean(), v[-1])
        r, u, o = cell["mean_reward"], cell["frac_steps_unsafe"], cell["frac_steps_term_obstacle"]
        hits = f"{o[0] * 819200:.0f}" if isinstance(o[0], float) else "n/a"
        f2 = lambda x: f"{x:.1f}" if isinstance(x, float) else x
        fe = lambda x: f"{x:.2e}" if isinstance(x, float) else x
        lines.append(f"| {name} | {n_it} | {spi} | {f2(r[0])} / {f2(r[1])} | {fe(u[0])} / {fe(u[1])} | {fe(o[0])} / {fe(o[1])} | {hits} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
