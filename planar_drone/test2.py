#!/usr/bin/env python3
"""
Diagnostics for VecDrone2D.

Examples
--------
Zero action means hover thrust and zero moment:

    python diagnose_env.py --env-file env.py --device cuda

Run in real time:

    python diagnose_env.py --env-file env.py --device cuda --realtime

Use random actions:

    python diagnose_env.py --env-file env.py --device cuda \
        --action-mode random --realtime

Replace choose_actions() below to test your trained policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose rapid termination/resetting in VecDrone2D."
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="Path to the Python file containing VecDrone2D.",
    )
    parser.add_argument(
        "--class-name",
        default="VecDrone2D",
        help="Environment class name. Default: VecDrone2D",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Use 1 for easy-to-read diagnostics.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum number of control steps.",
    )
    parser.add_argument(
        "--max-resets",
        type=int,
        default=10,
        help="Stop after this many reset events.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("zero", "random", "custom"),
        default="zero",
    )
    parser.add_argument(
        "--random-scale",
        type=float,
        default=1.0,
        help="Random actions are sampled from [-scale, scale].",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep for dt * decimation after each control step.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print a status line every N control steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Keep torch.compile enabled. Disabled by default for diagnostics.",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=0.0,
        help="Initial-state perturbation passed to VecDrone2D.",
    )

    return parser.parse_args()


def load_environment_module(path: Path) -> Any:
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Environment file does not exist: {path}")

    spec = importlib.util.spec_from_file_location("diagnostic_env_module", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def choose_actions(
    env: Any,
    observations: Any,
    mode: str,
    random_scale: float,
) -> torch.Tensor:
    """
    Return raw policy actions in [-1, 1].

    IMPORTANT:
    Replace the custom branch with the same action calculation used by your
    watch script to test the trained policy.
    """

    if mode == "zero":
        # For your environment:
        # action[:, 0] = 0 gives F = hover
        # action[:, 1] = 0 gives M = 0
        return torch.zeros(
            env.num_envs,
            env.num_actions,
            device=env.device,
        )

    if mode == "random":
        actions = (
            2.0
            * torch.rand(
                env.num_envs,
                env.num_actions,
                device=env.device,
            )
            - 1.0
        )
        return actions * random_scale

    if mode == "custom":
        # ---------------------------------------------------------------
        # PUT THE ACTION-CALCULATION FROM YOUR WATCH SCRIPT HERE.
        #
        # Example shape:
        #
        # with torch.inference_mode():
        #     actions = policy(observations["policy"])
        # return actions
        # ---------------------------------------------------------------
        raise RuntimeError(
            "Custom action mode selected, but choose_actions() has not "
            "been connected to your trained policy."
        )

    raise ValueError(f"Unknown action mode: {mode}")


def tensor_list(value: torch.Tensor, digits: int = 5) -> list:
    data = value.detach().cpu().tolist()

    def rounded(x: Any) -> Any:
        if isinstance(x, list):
            return [rounded(v) for v in x]
        if isinstance(x, float):
            return round(x, digits)
        return x

    return rounded(data)


def get_reason_data(
    env: Any,
    module: Any,
    ids: torch.Tensor,
) -> dict[str, Any]:
    """
    Compute termination reasons before reset_idx() destroys terminal states.
    """

    ids = ids.to(env.device)

    state = env.state[ids].detach().clone()
    target = env.target[ids].detach().clone()
    obstacle = env.obstacle[ids].detach().clone()
    episode_length = env.episode_length_buf[ids].detach().clone()

    x, y, theta, vx, vy, omega = state.unbind(dim=1)

    world_x = float(getattr(module, "WORLD_X", 10.0))
    world_y = float(getattr(module, "WORLD_Y", 10.0))

    obstacle_radius_sq = float(
        getattr(
            env,
            "OBSTACLE_RADIUS_SQ",
            getattr(module, "OBSTACLE_RADIUS", 1.0) ** 2,
        )
    )

    obstacle_delta = state[:, :2] - obstacle
    obstacle_distance_sq = (obstacle_delta**2).sum(dim=1)
    obstacle_distance = torch.sqrt(obstacle_distance_sq)

    target_distance = torch.linalg.vector_norm(
        state[:, :2] - target,
        dim=1,
    )

    crashed = y <= 0.0
    left_or_right = x.abs() > world_x
    above_ceiling = y > world_y
    out_of_bounds = left_or_right | above_ceiling
    hit_obstacle = obstacle_distance_sq < obstacle_radius_sq
    timed_out = episode_length >= env.max_episode_length

    nonfinite_state = ~torch.isfinite(state).all(dim=1)
    nonfinite_target = ~torch.isfinite(target).all(dim=1)
    nonfinite_obstacle = ~torch.isfinite(obstacle).all(dim=1)

    excessive_angle = theta.abs() > math.pi / 2
    excessive_speed = torch.sqrt(vx**2 + vy**2) > 20.0
    excessive_omega = omega.abs() > 20.0

    reasons_per_env: list[list[str]] = []

    for local_index in range(ids.numel()):
        reasons: list[str] = []

        if bool(crashed[local_index]):
            reasons.append("GROUND_CRASH")

        if bool(left_or_right[local_index]):
            reasons.append("X_OUT_OF_BOUNDS")

        if bool(above_ceiling[local_index]):
            reasons.append("Y_OUT_OF_BOUNDS")

        if bool(hit_obstacle[local_index]):
            reasons.append("HIT_OBSTACLE")

        if bool(timed_out[local_index]):
            reasons.append("TIMEOUT")

        if bool(nonfinite_state[local_index]):
            reasons.append("NONFINITE_STATE")

        if bool(nonfinite_target[local_index]):
            reasons.append("NONFINITE_TARGET")

        if bool(nonfinite_obstacle[local_index]):
            reasons.append("NONFINITE_OBSTACLE")

        if not reasons:
            reasons.append("RESET_WITHOUT_KNOWN_TERMINATION_REASON")

        reasons_per_env.append(reasons)

    return {
        "ids": ids.detach().clone(),
        "state": state,
        "target": target,
        "obstacle": obstacle,
        "episode_length": episode_length,
        "obstacle_distance": obstacle_distance,
        "target_distance": target_distance,
        "reasons": reasons_per_env,
        "large_angle": excessive_angle,
        "large_speed": excessive_speed,
        "large_omega": excessive_omega,
    }


def inspect_current_state(env: Any, module: Any, label: str) -> None:
    all_ids = torch.arange(env.num_envs, device=env.device)
    data = get_reason_data(env, module, all_ids)

    actual_terminated = env._terminated().detach()

    print(f"\n=== {label} ===")
    print(f"state:              {tensor_list(data['state'])}")
    print(f"target:             {tensor_list(data['target'])}")
    print(f"obstacle:           {tensor_list(data['obstacle'])}")
    print(f"target distance:    {tensor_list(data['target_distance'])}")
    print(f"obstacle distance:  {tensor_list(data['obstacle_distance'])}")
    print(f"episode length:     {tensor_list(data['episode_length'])}")
    print(f"_terminated():      {tensor_list(actual_terminated)}")

    if actual_terminated.any():
        print("WARNING: At least one environment is terminal immediately.")


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    print("=== Configuration ===")
    print(f"environment file:   {args.env_file.expanduser().resolve()}")
    print(f"class:              {args.class_name}")
    print(f"device:             {args.device}")
    print(f"num environments:   {args.num_envs}")
    print(f"action mode:        {args.action_mode}")
    print(f"seed:               {args.seed}")
    print(f"torch compile:      {args.enable_compile}")

    # Disabling compilation produces simpler stack traces and eliminates
    # torch.compile as a possible source of confusing behavior.
    original_torch_compile = getattr(torch, "compile", None)

    if not args.enable_compile and original_torch_compile is not None:
        torch.compile = lambda function, *unused_args, **unused_kwargs: function

    try:
        module = load_environment_module(args.env_file)
        env_class = getattr(module, args.class_name)

        env = env_class(
            num_envs=args.num_envs,
            device=args.device,
            perturb=args.perturb,
        )
    finally:
        if original_torch_compile is not None:
            torch.compile = original_torch_compile

    control_dt = float(env.dt * env.decimation)
    simulated_episode_duration = (
        env.max_episode_length * control_dt
    )

    print("\n=== Environment timing ===")
    print(f"physics dt:                  {env.dt}")
    print(f"decimation:                  {env.decimation}")
    print(f"control dt:                  {control_dt:.6f} seconds")
    print(f"maximum control steps:       {env.max_episode_length}")
    print(
        "maximum simulated duration: "
        f"{simulated_episode_duration:.6f} seconds"
    )

    inspect_current_state(env, module, "STATE DIRECTLY AFTER INITIAL RESET")

    initially_terminal = env._terminated().detach()

    if initially_terminal.any():
        bad_ids = initially_terminal.nonzero(as_tuple=False).flatten()
        print(
            "\nFATAL DIAGNOSTIC: Some environments are terminal immediately "
            f"after reset. IDs: {tensor_list(bad_ids)}"
        )

    # ------------------------------------------------------------------
    # Wrap reset_idx so terminal states can be recorded before reset.
    # ------------------------------------------------------------------

    original_reset_idx = env.reset_idx
    reset_events: list[dict[str, Any]] = []

    diagnostic_context: dict[str, Any] = {
        "global_step": -1,
        "last_action": None,
        "inside_step": False,
    }

    def diagnostic_reset_idx(ids: torch.Tensor) -> None:
        ids = ids.flatten()

        event = get_reason_data(env, module, ids)
        event["global_step"] = diagnostic_context["global_step"]
        event["inside_step"] = diagnostic_context["inside_step"]

        last_action = diagnostic_context["last_action"]

        if last_action is not None:
            event["actions"] = last_action[ids].detach().clone()
        else:
            event["actions"] = None

        reset_events.append(event)

        original_reset_idx(ids)

        reset_state = env.state[ids].detach()

        event["post_reset_state"] = reset_state.clone()
        event["post_reset_terminal"] = env._terminated()[ids].detach().clone()

    env.reset_idx = diagnostic_reset_idx

    observations = env.get_observations()

    reason_counter: Counter[str] = Counter()
    total_done_returns = 0
    total_reset_envs = 0
    completed_episode_lengths: list[int] = []

    start_wall_time = time.perf_counter()
    last_status_time = start_wall_time

    try:
        for step_index in range(args.steps):
            diagnostic_context["global_step"] = step_index

            actions = choose_actions(
                env=env,
                observations=observations,
                mode=args.action_mode,
                random_scale=args.random_scale,
            )

            if actions.shape != (env.num_envs, env.num_actions):
                raise ValueError(
                    "Action shape is incorrect. Expected "
                    f"{(env.num_envs, env.num_actions)}, got "
                    f"{tuple(actions.shape)}."
                )

            if not torch.isfinite(actions).all():
                raise FloatingPointError(
                    f"Nonfinite action at step {step_index}: "
                    f"{tensor_list(actions)}"
                )

            diagnostic_context["last_action"] = actions.detach().clone()

            events_before_step = len(reset_events)
            diagnostic_context["inside_step"] = True

            observations, rewards, dones, extras = env.step(actions)

            diagnostic_context["inside_step"] = False

            if not torch.isfinite(env.state).all():
                print("\nNONFINITE STATE DETECTED AFTER STEP")
                print(f"step:   {step_index}")
                print(f"state:  {tensor_list(env.state)}")
                print(f"action: {tensor_list(actions)}")
                break

            if not torch.isfinite(rewards).all():
                print("\nNONFINITE REWARD DETECTED")
                print(f"step:    {step_index}")
                print(f"rewards: {tensor_list(rewards)}")
                print(f"state:   {tensor_list(env.state)}")
                break

            new_events = reset_events[events_before_step:]

            returned_done_count = int(dones.sum().item())
            total_done_returns += returned_done_count

            reset_count_this_step = sum(
                event["ids"].numel() for event in new_events
            )
            total_reset_envs += reset_count_this_step

            if returned_done_count != reset_count_this_step:
                print("\nWARNING: done/reset mismatch")
                print(f"step:                 {step_index}")
                print(f"returned done count:  {returned_done_count}")
                print(f"captured reset count: {reset_count_this_step}")
                print(f"dones:                {tensor_list(dones)}")

            for event in new_events:
                ids_cpu = event["ids"].detach().cpu().tolist()

                print("\n" + "=" * 72)
                print(
                    f"RESET EVENT AT GLOBAL CONTROL STEP "
                    f"{event['global_step']}"
                )
                print("=" * 72)

                for local_index, env_id in enumerate(ids_cpu):
                    reasons = event["reasons"][local_index]

                    for reason in reasons:
                        reason_counter[reason] += 1

                    episode_length = int(
                        event["episode_length"][local_index].item()
                    )
                    completed_episode_lengths.append(episode_length)

                    state = event["state"][local_index]
                    target = event["target"][local_index]
                    obstacle = event["obstacle"][local_index]

                    simulated_time = episode_length * control_dt

                    print(f"\nenvironment ID:      {env_id}")
                    print(f"reasons:             {', '.join(reasons)}")
                    print(f"episode steps:       {episode_length}")
                    print(
                        f"simulated time:      {simulated_time:.6f} seconds"
                    )
                    print(f"terminal state:      {tensor_list(state)}")
                    print(f"target:              {tensor_list(target)}")
                    print(f"obstacle:            {tensor_list(obstacle)}")
                    print(
                        "target distance:     "
                        f"{event['target_distance'][local_index].item():.6f}"
                    )
                    print(
                        "obstacle distance:   "
                        f"{event['obstacle_distance'][local_index].item():.6f}"
                    )

                    if event["actions"] is not None:
                        print(
                            "last raw action:     "
                            f"{tensor_list(event['actions'][local_index])}"
                        )

                    print(
                        "large angle warning: "
                        f"{bool(event['large_angle'][local_index])}"
                    )
                    print(
                        "large speed warning: "
                        f"{bool(event['large_speed'][local_index])}"
                    )
                    print(
                        "large omega warning: "
                        f"{bool(event['large_omega'][local_index])}"
                    )
                    print(
                        "post-reset state:     "
                        f"{tensor_list(event['post_reset_state'][local_index])}"
                    )
                    print(
                        "terminal after reset: "
                        f"{bool(event['post_reset_terminal'][local_index])}"
                    )

                    if "TIMEOUT" in reasons and len(reasons) == 1:
                        print(
                            "INTERPRETATION: The episode did not crash. "
                            "It reached max_episode_length."
                        )

                    if bool(event["post_reset_terminal"][local_index]):
                        print(
                            "IMPORTANT: This newly generated reset state is "
                            "already terminal."
                        )

            if step_index % args.print_every == 0:
                now = time.perf_counter()
                elapsed = now - start_wall_time
                status_interval = now - last_status_time
                last_status_time = now

                simulated_time_total = (step_index + 1) * control_dt

                print(
                    f"\nstep={step_index:6d} | "
                    f"episode_length={tensor_list(env.episode_length_buf)} | "
                    f"reward={tensor_list(rewards)} | "
                    f"done={tensor_list(dones)} | "
                    f"simulated={simulated_time_total:.3f}s | "
                    f"wall={elapsed:.3f}s | "
                    f"status_interval={status_interval:.3f}s"
                )

                if "r1" in extras:
                    print(f"r1={tensor_list(extras['r1'])}")

                if "time_outs" in extras:
                    print(
                        f"time_outs={tensor_list(extras['time_outs'])}"
                    )

            if args.realtime:
                time.sleep(control_dt)

            if total_reset_envs >= args.max_resets:
                print(
                    f"\nReached --max-resets={args.max_resets}. Stopping."
                )
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception:
        print("\n=== EXCEPTION DURING DIAGNOSTIC RUN ===")
        traceback.print_exc()

        print("\nLast available values:")

        try:
            print(f"global step: {diagnostic_context['global_step']}")
            print(f"state:       {tensor_list(env.state)}")
            print(f"target:      {tensor_list(env.target)}")
            print(f"obstacle:    {tensor_list(env.obstacle)}")
            print(
                f"episode len: {tensor_list(env.episode_length_buf)}"
            )

            if diagnostic_context["last_action"] is not None:
                print(
                    "last action: "
                    f"{tensor_list(diagnostic_context['last_action'])}"
                )
        except Exception:
            pass

    finally:
        end_wall_time = time.perf_counter()
        wall_duration = end_wall_time - start_wall_time

        print("\n" + "#" * 72)
        print("DIAGNOSTIC SUMMARY")
        print("#" * 72)

        print(f"wall-clock duration:       {wall_duration:.6f} seconds")
        print(f"returned done count:       {total_done_returns}")
        print(f"captured reset env count:  {total_reset_envs}")

        print("\nTermination reason counts:")

        if reason_counter:
            for reason, count in reason_counter.most_common():
                print(f"  {reason:40s} {count}")
        else:
            print("  No resets captured.")

        if completed_episode_lengths:
            lengths = torch.tensor(
                completed_episode_lengths,
                dtype=torch.float32,
            )

            print("\nCompleted episode lengths:")
            print(f"  minimum: {int(lengths.min().item())}")
            print(f"  maximum: {int(lengths.max().item())}")
            print(f"  mean:    {lengths.mean().item():.3f}")

        if reason_counter and set(reason_counter) == {"TIMEOUT"}:
            print(
                "\nLIKELY DIAGNOSIS:\n"
                "Every recorded episode ended only because of the time limit. "
                "The viewer is probably running without real-time sleeping, "
                "so six simulated seconds pass almost instantly."
            )

        if reason_counter.get("RESET_WITHOUT_KNOWN_TERMINATION_REASON", 0):
            print(
                "\nWARNING:\n"
                "reset_idx() was called without one of the known termination "
                "conditions. Search the rest of your code for additional calls "
                "to reset_idx()."
            )

        print(
            "\nPaste the complete terminal output back into the chat, "
            "especially the first RESET EVENT and DIAGNOSTIC SUMMARY."
        )


if __name__ == "__main__":
    main()