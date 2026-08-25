"""Train paired RoboCasa baseline ACT or base-only LP-ACT V1."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .data import load_robocasa_data, split_episodes
from .labels import MetricConfig
from .policy import make_act_policy
from .stats import load_stats
from .training_data import LineUpCondimentsDataset


class CachedSamples(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standard", "lp"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiny-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--validate-every", type=int, default=1_000)
    parser.add_argument("--validation-samples", type=int, default=256)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, policy, optimizer, step: int, args: argparse.Namespace, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "mode": args.mode,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "stats": stats,
        },
        path,
    )


@torch.no_grad()
def evaluate(policy, loader: DataLoader, device: torch.device) -> float:
    policy.eval()
    total_error = 0.0
    total_values = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = policy.predict_action_chunk(batch)
        valid = ~batch["action_is_pad"].unsqueeze(-1)
        total_error += float((torch.abs(prediction - batch["action"]) * valid).sum())
        total_values += int(valid.sum()) * prediction.shape[-1]
    policy.train()
    return total_error / max(total_values, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device(args.device)
    stats = load_stats(args.stats)
    metric = MetricConfig.from_dict(json.loads(args.metric_config.read_text()))
    trajectory = load_robocasa_data(args.data_root)
    train_episodes, validation_episodes = split_episodes(trajectory.episode_bounds)

    dataset: Dataset = LineUpCondimentsDataset(
        args.data_root,
        train_episodes,
        args.mode,
        metric=metric,
        trajectory=trajectory,
        frame_cache=args.frame_cache,
    )
    validation_loader = None
    num_workers = args.num_workers
    if args.tiny_samples:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(dataset), size=min(args.tiny_samples, len(dataset)), replace=False).tolist())
        subset = Subset(dataset, indices)
        dataset = CachedSamples([subset[index] for index in range(len(subset))])
        num_workers = 0
    else:
        validation = LineUpCondimentsDataset(
            args.data_root,
            validation_episodes,
            args.mode,
            metric=metric,
            trajectory=trajectory,
            frame_cache=args.frame_cache,
        )
        rng = np.random.default_rng(args.seed + 1)
        indices = sorted(rng.choice(len(validation), size=min(args.validation_samples, len(validation)), replace=False).tolist())
        subset = Subset(validation, indices)
        cached = CachedSamples([subset[index] for index in range(len(subset))])
        validation_loader = DataLoader(cached, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    policy = make_act_policy(args.mode, stats, learning_rate=args.learning_rate, device=args.device).to(device)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.learning_rate, weight_decay=policy.config.optimizer_weight_decay
    )
    start_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint["mode"] != args.mode:
            raise ValueError(f"Checkpoint mode {checkpoint['mode']} does not match {args.mode}")
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        print(json.dumps({"event": "resumed", "step": start_step, "checkpoint": str(args.resume_checkpoint)}), flush=True)
    iterator = iter(loader)
    start_time = time.time()
    initial_l1 = None
    best_validation = float("inf")
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, loss_dict = policy(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        if initial_l1 is None:
            initial_l1 = float(loss_dict["l1_loss"])
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "mode": args.mode,
                        "loss": float(loss.detach()),
                        **loss_dict,
                        "initial_l1_loss": initial_l1,
                        "grad_norm": float(grad_norm),
                        "elapsed_s": time.time() - start_time,
                        "batch_size": args.batch_size,
                        "samples": len(dataset),
                    }
                ),
                flush=True,
            )
        if step % args.checkpoint_every == 0 and step != args.steps:
            save_checkpoint(args.output_dir / f"checkpoint_{step:07d}.pt", policy, optimizer, step, args, stats)
        if validation_loader is not None and (step % args.validate_every == 0 or step == args.steps):
            validation_l1 = evaluate(policy, validation_loader, device)
            print(json.dumps({"step": step, "mode": args.mode, "split": "validation", "l1_loss": validation_l1}), flush=True)
            if validation_l1 < best_validation:
                best_validation = validation_l1
                save_checkpoint(args.output_dir / "checkpoint_best.pt", policy, optimizer, step, args, stats)
    save_checkpoint(args.output_dir / "checkpoint_final.pt", policy, optimizer, args.steps, args, stats)


if __name__ == "__main__":
    main()
