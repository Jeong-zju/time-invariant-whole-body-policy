"""Run the official GR00T rollout loop with RoboCasa fixed at real 30 Hz."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-client-host", default="127.0.0.1")
    parser.add_argument("--policy-client-port", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=675)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--robocasa-split", default="target")
    args = parser.parse_args()

    from gr00t.eval import rollout_policy as rollout
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.utils.determinism import seed_everything

    def get_robocasa_env_fn_30hz(env_name: str, robocasa_split: str = ""):
        def env_fn():
            kwargs = {"control_freq": 30}
            if env_name.startswith("robocasa365_panda_omron/"):
                import gr00t.eval.sim.robocasa365.gymnasium_groot  # noqa: F401

                if robocasa_split:
                    kwargs["split"] = robocasa_split
            else:
                import robocasa  # noqa: F401
                import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
            env = gym.make(env_name, enable_render=True, **kwargs)
            actual = float(env.unwrapped.control_freq)
            timestep = float(env.unwrapped.control_timestep)
            if abs(actual - 30.0) > 1e-9 or abs(timestep - 1.0 / 30.0) > 1e-9:
                raise AssertionError(f"environment is not 30 Hz: {actual}, {timestep}")
            print({"event": "ENVIRONMENT_30HZ_VERIFIED", "control_hz": actual, "timestep_s": timestep})
            return env

        return env_fn

    rollout.get_robocasa_env_fn = get_robocasa_env_fn_30hz
    effective_seed = seed_everything(args.seed)
    policy = rollout.create_gr00t_sim_policy(
        model_path="",
        embodiment_tag=rollout.get_embodiment_tag_from_env_name(
            "robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env"
        ),
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
    )
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=1)
    wrappers = rollout.WrapperConfigs(
        multistep=rollout.MultiStepConfig(
            contract=contract,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
        video=rollout.VideoConfig(
            video_dir=str(args.video_dir),
            steps_per_render=1,
            max_episode_steps=args.max_episode_steps,
            fps=30,
        ),
    )
    results = rollout.run_rollout_gymnasium_policy(
        env_name="robocasa365_panda_omron/NavigateKitchen_PandaOmron_Env",
        policy=policy,
        wrapper_configs=wrappers,
        n_episodes=1,
        n_envs=1,
        seed=effective_seed,
        robocasa_split=args.robocasa_split,
    )
    print("results: ", results)
    print("success rate: ", float(results[1][0]))


if __name__ == "__main__":
    main()
