"""Experiment-directory naming: auto-incrementing exp1, exp2_name, ...

  next_exp_dir()   -> the next unused logs/expN[_name] (use when training)
  latest_exp_dir() -> the highest numbered experiment (use when evaluating)

Numbering is "one past the highest existing", so deleting a middle experiment
never causes a name collision.
"""

import re
from pathlib import Path

_EXP_RE = re.compile(r"exp(\d+)(?:_.+)?")


def _exp_numbers(base):
    base = Path(base)
    if not base.exists():
        return []
    nums = []
    for p in base.iterdir():
        if p.is_dir():
            m = _EXP_RE.fullmatch(p.name)
            if m:
                nums.append(int(m.group(1)))
    return nums


def _sanitize_run_name(run_name):
    """Convert a human-readable run name to a filesystem-friendly suffix."""
    if run_name is None:
        return None
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", run_name.strip()).strip("_").lower()
    if not suffix:
        raise ValueError("run_name must contain at least one letter or number")
    return suffix


def next_exp_dir(base="logs", run_name=None):
    """Return the next logs/expN path, optionally suffixed with a run name."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    n = max(_exp_numbers(base), default=0) + 1
    suffix = _sanitize_run_name(run_name)
    name = f"exp{n}_{suffix}" if suffix else f"exp{n}"
    return base / name


def latest_exp_dir(base="logs"):
    """The highest-numbered existing logs/expN."""
    base = Path(base)
    experiments = []
    if base.exists():
        for path in base.iterdir():
            if path.is_dir():
                match = _EXP_RE.fullmatch(path.name)
                if match:
                    experiments.append((int(match.group(1)), path))
    if not experiments:
        raise FileNotFoundError(f"No exp* directories found in {base}/")
    return max(experiments, key=lambda item: item[0])[1]
