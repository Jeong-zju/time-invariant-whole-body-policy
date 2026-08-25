"""Upload the paired RoboCasa rollout artifacts to an existing HF dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-prefix",
        default="robocasa/LineUpCondiments/base_only_lpact_v1/validation_episode_000",
    )
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("Hugging Face token file is empty")
    names = (
        "robocasa_rollout_act_best_ep0_v1.mp4",
        "robocasa_rollout_act_best_ep0_v1.json",
        "robocasa_rollout_lpact_best_ep0_v1.mp4",
        "robocasa_rollout_lpact_best_ep0_v1.json",
    )
    api = HfApi(token=token)
    for name in names:
        local_path = args.output_dir / name
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        result = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{args.repo_prefix}/{name}",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add RoboCasa LineUpCondiments rollout: {name}",
        )
        print(f"uploaded {name}: {result}", flush=True)


if __name__ == "__main__":
    main()
