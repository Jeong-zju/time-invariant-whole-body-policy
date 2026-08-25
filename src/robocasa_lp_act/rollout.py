"""Roll out a trained ACT or base-only LP-ACT checkpoint in RoboCasa."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .data import load_robocasa_data
from .execution import ControllerCalibration, decode_lp_chunk
from .policy import make_act_policy
from .schema import CAMERA_KEYS, EXECUTION_STEPS
from .stats import load_stats

RAW_CAMERA_NAMES = (
    "robot0_eye_in_hand",
    "robot0_agentview_left",
    "robot0_agentview_right",
)


def lerobot_to_env_action(action: np.ndarray) -> np.ndarray:
    """Map verified LeRobot ordering to RoboCasa controller ordering."""

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (12,):
        raise ValueError(action.shape)
    result = np.empty(12, dtype=np.float64)
    result[0:3] = action[5:8]
    result[3:6] = action[8:11]
    result[6] = action[11]
    result[7:11] = action[0:4]
    result[11] = action[4]
    return np.clip(result, -1.0, 1.0)


def state_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            observation["robot0_base_pos"],
            observation["robot0_base_quat"],
            observation["robot0_base_to_eef_pos"],
            observation["robot0_base_to_eef_quat"],
            observation["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def policy_batch(observation: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(state_from_observation(observation)).unsqueeze(0).to(device)
    }
    for lerobot_key, raw_name in zip(CAMERA_KEYS, RAW_CAMERA_NAMES, strict=True):
        image = np.array(observation[f"{raw_name}_image"], copy=True)
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
        batch[lerobot_key] = tensor.unsqueeze(0).to(device)
    return batch


def load_calibration(path: Path | None) -> ControllerCalibration:
    if path is None:
        return ControllerCalibration()
    value = json.loads(path.read_text())
    return ControllerCalibration(**value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standard", "lp"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-skip", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import robocasa.utils.lerobot_utils as LU
    import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

    device = torch.device(args.device)
    stats = load_stats(args.stats)
    policy = make_act_policy(args.mode, stats, device=args.device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["mode"] != args.mode:
        raise ValueError(f"Checkpoint mode {checkpoint['mode']} does not match {args.mode}")
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    calibration = load_calibration(args.calibration)

    env_meta = LU.get_env_metadata(args.data_root)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=list(RAW_CAMERA_NAMES),
        camera_height=256,
        camera_width=256,
        reward_shaping=False,
    )
    states = LU.get_episode_states(args.data_root, args.episode)
    initial_state = {
        "model": LU.get_episode_model_xml(args.data_root, args.episode),
        "states": states[0],
        "ep_meta": json.dumps(LU.get_episode_meta(args.data_root, args.episode)),
    }
    observation = env.reset_to(initial_state)

    trajectory = load_robocasa_data(args.data_root)
    start, end = trajectory.episode_bounds[args.episode]
    initial_state_error = np.abs(state_from_observation(observation) - trajectory.states[start])
    initial_image_errors = {}
    for lerobot_key, raw_name in zip(CAMERA_KEYS, RAW_CAMERA_NAMES, strict=True):
        # Dataset extraction used this same reset path, so the first image should
        # be pixel-identical or differ only by renderer roundoff.
        initial_image_errors[lerobot_key] = {
            "shape": list(observation[f"{raw_name}_image"].shape),
            "dtype": str(observation[f"{raw_name}_image"].dtype),
        }

    args.video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.video, fps=20 / args.video_skip, codec="libx264", quality=7)
    success = False
    executed_steps = 0
    inference_times = []
    action_clip_values = 0
    max_steps = min(args.max_steps, end - start)
    try:
        while executed_steps < max_steps and not success:
            batch = policy_batch(observation, device)
            before = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
            inference_times.append(time.perf_counter() - before)
            if args.mode == "standard":
                chunk = prediction[:EXECUTION_STEPS]
            else:
                decoded = decode_lp_chunk(prediction, calibration=calibration)
                chunk = decoded.actions
                action_clip_values += int(np.sum(np.abs(decoded.unclipped_base_actions) > 1.0))

            for action in chunk:
                observation, _, _, info = env.step(lerobot_to_env_action(action))
                if executed_steps % args.video_skip == 0:
                    frame = env.render(
                        mode="rgb_array", height=512, width=768, camera_name="robot0_agentview_center"
                    )
                    writer.append_data(frame)
                executed_steps += 1
                success = bool(info["is_success"]["task"])
                if executed_steps >= max_steps or success:
                    break
    finally:
        writer.close()
        env.env.close()

    report = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "episode": args.episode,
        "validation_episode": args.episode % 5 == 0,
        "success": success,
        "executed_steps": executed_steps,
        "max_steps": max_steps,
        "queries": len(inference_times),
        "mean_inference_s": float(np.mean(inference_times)),
        "base_action_clip_values": action_clip_values,
        "initial_state_max_abs_error": float(initial_state_error.max()),
        "initial_image_observations": initial_image_errors,
        "video": str(args.video),
    }
    args.video.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
