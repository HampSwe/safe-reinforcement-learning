import yaml
import argparse
import random
import numpy as np
import environment
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerReset
from pathlib import Path
import log_handler
import rsl_rl.algorithms.rl_sgf as rl_sgf
import torch
import json, os

# 500 is minimum. 750 okay. 1000 ideal? Should increase much more, 3000 perhaps, and see if it improves even more (it most likely will)

START_SAFE = True
LEGACY_RLSGF_STEPS_MULTIPLIER = 2

# Example run:
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 python planar_drone/train.py --iterations 300 --no-show-dashboard-after

def make_env(device, total_iters, num_envs=2048):
    #return environment.VecDrone2D(num_envs=4096, device=device)
    #return environment.VecDrone2D(num_envs=int(1500), device=device)
    #return environment.VecDrone2D(num_envs=int(4096*2), device=device)
    # default 2048 (was 3000, then 1500); runner.num_envs in the YAML overrides it (2026-09-22)
    return environment.VecDrone2D(num_envs=int(num_envs), device=device, total_iters=total_iters)

    # if device == "cpu":
    #     return environment.VecDrone2D(num_envs=64, device=device)


def main(config="config_rlsgf.yaml", iterations=2000, show_dashboard_after=True, device="gpu", run_name=None, seed=None):
    config_path = Path(__file__).resolve().parent / "config" / config

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Resolve the device argument to a torch device string.
    if device == "gpu":
        device = "cuda:0"
    # (device == "cpu" stays "cpu"; an explicit "cuda:0" also passes through)

    # 1) Load the YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg["runner"]

    # 2) Create your environment (runner.num_envs overrides the default 2048)
    num_envs = int(train_cfg.pop("num_envs", 2048))
    env = make_env(device, iterations, num_envs)
    T = env.max_episode_length
    print(f"num_envs = {num_envs}")

    # runner.terminate_on_obstacle: false keeps episodes running through obstacle contact (2026-09-19);
    # applies to every algorithm, so it is popped before the RLSGF / PPO branches below.
    env.terminate_on_obstacle = bool(train_cfg.pop("terminate_on_obstacle", True))
    if not env.terminate_on_obstacle:
        print("Obstacle contact does NOT terminate the episode (runner.terminate_on_obstacle: false)")

    # runner.unsafe_cost: r1 value inside the unsafe set (default 1 - beta = 0.95); 2026-09-19.
    unsafe_cost = train_cfg.pop("unsafe_cost", None)
    if unsafe_cost is not None:
        env.unsafe_cost = float(unsafe_cost)
        print(f"r1 inside the unsafe set set to {env.unsafe_cost} (runner.unsafe_cost)")
    # runner.unsafe_speed_ref: charge the unsafe-set cost per unit path length, unsafe_cost * |v| / ref (2026-09-24).
    unsafe_speed_ref = train_cfg.pop("unsafe_speed_ref", None)
    if unsafe_speed_ref is not None:
        env.unsafe_speed_ref = float(unsafe_speed_ref)
        print(f"r1 inside the unsafe set = {env.unsafe_cost} * |v| / {env.unsafe_speed_ref} m/s (runner.unsafe_speed_ref)")

    # runner.r1_mode: "step" (default) or "episode" (cost once per episode at the first unsafe-set entry,
    # credit -collision_budget spread over the episode: V1 <= 0 <=> P(unsafe entry) <= collision_budget);
    # runner.collision_budget defaults to the env beta. 2026-09-20.
    r1_mode = train_cfg.pop("r1_mode", "step")
    if r1_mode not in ("step", "episode"):
        raise ValueError("runner.r1_mode must be 'step' or 'episode'")
    env.r1_mode = r1_mode
    r1_credit = train_cfg.pop("r1_credit", "start")
    if r1_credit not in ("start", "step"):
        raise ValueError(f"r1_credit must be 'start' or 'step', got {r1_credit!r}")
    env.r1_credit = r1_credit
    env.r1_walls = bool(train_cfg.pop("r1_walls", False))
    env.wall_margin = float(train_cfg.pop("wall_margin", 0.5))
    env.r1_floor = bool(train_cfg.pop("r1_floor", False))
    env.floor_margin = float(train_cfg.pop("floor_margin", 1.0))
    # runner.band_unsafe_cost: flat r1 inside the wall/floor bands (default: same as unsafe_cost); 2026-09-25.
    band_unsafe_cost = train_cfg.pop("band_unsafe_cost", None)
    if band_unsafe_cost is not None:
        env.band_unsafe_cost = float(band_unsafe_cost)
        print(f"r1 inside the wall/floor bands set to {env.band_unsafe_cost} (runner.band_unsafe_cost)")
    env.terminate_on_oob = bool(train_cfg.pop("terminate_on_oob", True))
    if not env.terminate_on_oob:
        print("terminate_on_oob: false -- leaving the world box does not end the episode")
    env.terminate_on_crash = bool(train_cfg.pop("terminate_on_crash", True))
    if not env.terminate_on_crash:
        print("terminate_on_crash: false -- floor contact does not end the episode")
    if env.r1_walls:
        print(f"r1_walls: world boundary band of {env.wall_margin} m is part of the unsafe set")
    if env.r1_floor:
        print(f"r1_floor: the band y < {env.floor_margin} m above the floor is part of the unsafe set")
    stagger_initial_episodes = bool(train_cfg.pop("stagger_initial_episodes", False))
    if stagger_initial_episodes:
        env.stagger_episodes()
        print("stagger_initial_episodes: true -- initial episode clocks drawn uniformly in [0, max_episode_length)")
    collision_budget = train_cfg.pop("collision_budget", None)
    if collision_budget is not None:
        env.collision_budget = float(collision_budget)
    if r1_mode == "episode":
        credit_txt = (f"-{env.collision_budget} once at t=1" if env.r1_credit == "start"
                      else f"-{env.collision_budget}/T per step")
        print(f"r1_mode episode: r1 = 1 at first unsafe-set entry, {credit_txt} "
              f"(V1 <= 0 <=> P(unsafe entry) <= {env.collision_budget})")

    alg_name = train_cfg["algorithm"]["class_name"]
    is_rlsgf = isinstance(alg_name, str) and "RLSGF" in alg_name.split(":")[-1]
    reset_runner = False
    if is_rlsgf:
        bootstrap_final_state = train_cfg["algorithm"].get("bootstrap_final_state", False)
        if not isinstance(bootstrap_final_state, bool):
            raise ValueError("algorithm.bootstrap_final_state must be true or false")

        default_steps_multiplier = 1 if bootstrap_final_state else LEGACY_RLSGF_STEPS_MULTIPLIER
        steps_multiplier = train_cfg.get("num_steps_per_env_multiplier", default_steps_multiplier)
        if isinstance(steps_multiplier, bool) or not isinstance(steps_multiplier, (int, float)) or steps_multiplier <= 0:
            raise ValueError("runner.num_steps_per_env_multiplier must be a positive number")
        if float(steps_multiplier * T) != int(steps_multiplier * T):
            raise ValueError(f"runner.num_steps_per_env_multiplier * T must be an integer (got {steps_multiplier} * {T})")
        train_cfg["num_steps_per_env_multiplier"] = steps_multiplier

        reset_runner = train_cfg.get("reset_after_update", not bootstrap_final_state)
        if not isinstance(reset_runner, bool):
            raise ValueError("runner.reset_after_update must be true or false")

        train_cfg["num_steps_per_env"] = int(steps_multiplier * T)
        print(
            "RLSGF rollout: "
            f"{steps_multiplier}*T ({train_cfg['num_steps_per_env']} steps), "
            f"bootstrap_final_state={bootstrap_final_state}, "
            f"reset_after_update={reset_runner}"
        )
    else:
        # With PPO, subtract r1 from r0 (fixed-penalty baseline) unless the config opts out with
        # runner.subtract_r1_from_r0: false (pure PPO on the task reward r0 only; 2026-09-12).
        env.subtract_r1_from_r0 = bool(train_cfg.pop("subtract_r1_from_r0", True))
        print(f"{alg_name}: subtract_r1_from_r0={env.subtract_r1_from_r0}, num_steps_per_env={train_cfg['num_steps_per_env']}")
    
    # The env curriculum advances per "iteration" = global_step / env.num_steps_per_env.
    # Optionally decouple this from the rollout length so that shorter rollouts
    # (e.g. T/4) keep the same curriculum schedule in env-step terms.
    curriculum_steps = train_cfg.pop("curriculum_steps_per_env", None)
    env.num_steps_per_env = int(curriculum_steps) if curriculum_steps else train_cfg["num_steps_per_env"]
    if curriculum_steps:
        print(f"Curriculum iteration size pinned to {env.num_steps_per_env} env steps")

    #log_dir = log_handler.next_exp_dir("logs")        # logs/exp1, logs/exp2, ...
    project_dir = Path(__file__).resolve().parent
    log_dir = log_handler.next_exp_dir(project_dir / "logs", run_name=run_name)
    print(f"Logging to {log_dir}")



    # 3) Build the runner
    if reset_runner:
        runner = OnPolicyRunnerReset(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device,
        )
    else:
        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device,
        )

    if START_SAFE:
        with torch.no_grad():
            linears = [m for m in runner.alg.actor.mlp if isinstance(m, torch.nn.Linear)]
            linears[-1].weight.mul_(0.01)
            linears[-1].bias.zero_()

    if is_rlsgf:
        alg = runner.alg
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config_used.json"), "w") as f:
            json.dump({
                # pulled from the actual algorithm instance — the real values in use
                "alpha": alg.alpha,
                "h_ref": alg.h_ref,
                "h_exp": alg.h_exp,
                "h_schedule": getattr(alg, "h_schedule", None),
                "z": getattr(alg, "z", None),
                "max_theta_step": getattr(alg, "max_theta_step", None),
                "h_cap": alg.h_cap,
                "h_min": alg.h_min,
                "max_theta_step": alg.max_theta_step,
                "weight_decay": alg.weight_decay,
                "gamma": alg.gamma,
                "gamma1": getattr(alg, "gamma1", None),
                "r1_mode": env.r1_mode, "r1_credit": env.r1_credit if env.r1_mode == "episode" else None,
                "r1_walls": env.r1_walls,
                "wall_margin": env.wall_margin if env.r1_walls else None,
                "r1_floor": env.r1_floor,
                "floor_margin": env.floor_margin if env.r1_floor else None,
                "unsafe_cost": env.unsafe_cost, "unsafe_speed_ref": env.unsafe_speed_ref,
                "band_unsafe_cost": env.band_unsafe_cost,
                "terminate_on_oob": env.terminate_on_oob,
                "terminate_on_crash": env.terminate_on_crash,
                "stagger_initial_episodes": stagger_initial_episodes,
                "num_envs": env.num_envs,
                "collision_budget": env.collision_budget if env.r1_mode == "episode" else None,
                "critic_lr": alg.critic_lr,
                "critic_epochs": getattr(alg, "critic_epochs", 1),
                "critic_minibatch": getattr(alg, "critic_minibatch", None),
                "critic_use_epochs": getattr(alg, "critic_use_epochs", False),
                "bootstrap_final_state": getattr(alg, "bootstrap_final_state", False),
                # env / run
                "config": config,
                "run_name": run_name,
                "num_steps_per_env": train_cfg.get("num_steps_per_env"),
                "num_steps_per_env_multiplier": train_cfg.get("num_steps_per_env_multiplier"),
                "reset_after_update": reset_runner,
                "iterations": iterations,
                "seed": seed,
                # env constants (if accessible)
                "max_episode_length": env.max_episode_length,
            }, f, indent=2, default=str)
        print(f"Wrote constants to {log_dir}/config_used.json")

    # 4) Train
    runner.learn(num_learning_iterations=iterations)

    # 5) Export the trained policy for deployment
    runner.export_policy_to_jit(str(log_dir / "exported"), filename="policy.pt")
    runner.export_policy_to_onnx(str(log_dir / "exported"), filename="policy.onnx")

    print(f"Finished. Experiment: {log_dir.name}  (full path: {log_dir})")

    # 6) Optionally visualize the just-exported policy right away
    if show_dashboard_after:
        import run_exported   # imported here so training itself has no pygame dependency
        # Don't pass a device: run_exported picks whatever is available itself.
        run_exported.run(
            log_dir / "exported" / "policy.pt",
            num_envs=64,       # small batch for the live view, not the 4096 used to train
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the planar-drone policy.")
    parser.add_argument(
        "--config", default="config_rlsgf.yaml",
        help="Config filename inside the config/ directory (default: config_rlsgf.yaml).",
    )
    parser.add_argument(
        "--iterations", type=int, default=2000,
        help="Number of training iterations (default: 2000).",
    )
    parser.add_argument(
        "--show-dashboard-after", dest="show_dashboard_after",
        action="store_true", default=True,
        help="Open the exported-policy dashboard when training finishes (default: on).",
    )
    parser.add_argument(
        "--no-show-dashboard-after", dest="show_dashboard_after",
        action="store_false",
        help="Do not open the dashboard when training finishes.",
    )
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="gpu",
        help="Device to train on (default: gpu).",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Optional descriptive suffix for the experiment log directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed Python, NumPy, and PyTorch random number generators.",
    )
    args = parser.parse_args()
    main(
        config=args.config,
        iterations=args.iterations,
        show_dashboard_after=args.show_dashboard_after,
        device=args.device,
        run_name=args.run_name,
        seed=args.seed,
    )
