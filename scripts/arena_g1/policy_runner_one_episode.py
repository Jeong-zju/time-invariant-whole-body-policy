#!/usr/bin/env python3
"""Run exactly one Arena rollout, treating the step budget as a failed timeout."""

from __future__ import annotations

import random

import numpy as np
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.examples.policy_runner_cli import create_policy, setup_policy_argument_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext


def main() -> None:
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli):
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()
        policy, step_budget = create_policy(args_cli)
        from isaaclab_arena.metrics.metrics import compute_metrics

        steps_executed = 0
        terminated_once = False
        truncated_once = False
        for step in tqdm.tqdm(range(step_budget)):
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)
            steps_executed = step + 1
            terminated_once = bool(terminated.any().item())
            truncated_once = bool(truncated.any().item())
            if terminated_once or truncated_once:
                break

        rollout = {
            "steps_executed": steps_executed,
            "step_budget": step_budget,
            "terminated": terminated_once,
            "truncated": truncated_once,
            "budget_exhausted": not (terminated_once or truncated_once),
        }
        metrics = compute_metrics(env)
        print(f"Rollout: {rollout}")
        print(f"Metrics: {metrics}")
        env.close()


if __name__ == "__main__":
    main()
