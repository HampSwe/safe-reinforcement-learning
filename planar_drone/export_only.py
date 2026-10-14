"""Re-export a completed run's final checkpoint (jit + onnx) without retraining.

Use this after a training run finished but export_policy_to_onnx failed
(e.g. missing onnxscript) so you don't have to redo the whole run.
"""
import argparse
import yaml
import environment
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerReset
from pathlib import Path
import log_handler

STEPS_PER_ENV_EPISODE_LENGTH_MULTIPLIER = 2


def make_env(device, total_iters):
    return environment.VecDrone2D(num_envs=int(2048), device=device, total_iters=total_iters)


def main(exp=None, config="config_rlsgf.yaml", iterations=2000, device="gpu"):
    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / "config" / config

    if device == "gpu":
        device = "cuda:0"

    log_dir = Path(exp) if exp else log_handler.latest_exp_dir(project_dir / "logs")
    print(f"Using experiment dir: {log_dir}")

    env = make_env(device, iterations)

    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg["runner"]

    reset_runner = False
    alg_name = train_cfg["algorithm"]["class_name"]
    if isinstance(alg_name, str) and alg_name.split(":")[-1] == "RLSGF":
        train_cfg["num_steps_per_env"] = STEPS_PER_ENV_EPISODE_LENGTH_MULTIPLIER * env.max_episode_length
        reset_runner = True
    else:
        env.subtract_r1_from_r0 = True

    env.num_steps_per_env = train_cfg["num_steps_per_env"]

    runner_cls = OnPolicyRunnerReset if reset_runner else OnPolicyRunner
    runner = runner_cls(env=env, train_cfg=train_cfg, log_dir=str(log_dir), device=device)

    checkpoints = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {log_dir}")
    ckpt = checkpoints[-1]
    print(f"Loading checkpoint: {ckpt}")
    runner.load(str(ckpt))

    out_dir = log_dir / "exported"
    runner.export_policy_to_jit(str(out_dir), filename="policy.pt")
    runner.export_policy_to_onnx(str(out_dir), filename="policy.onnx")
    print(f"Exported policy.pt and policy.onnx to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-export a completed run's final checkpoint.")
    parser.add_argument("--exp", default=None, help="Experiment dir (default: latest logs/expN).")
    parser.add_argument("--config", default="config_rlsgf.yaml")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    main(exp=args.exp, config=args.config, iterations=args.iterations, device=args.device)
