"""Train standard ACT or LP-ACT V1 on turning_on_radio only."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from .labels import MetricConfig
from .parquet_data import load_turning_data, split_episodes
from .policy import make_act_policy
from .stats import load_stats
from .training_data import TurningACTDataset


class CachedSamples(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class EpisodeBlockBatchSampler(Sampler[list[int]]):
    """Shuffle contiguous, full-size blocks without crossing episode boundaries."""

    def __init__(self, episode_lengths: list[int], batch_size: int, seed: int) -> None:
        self.blocks: list[list[int]] = []
        offset = 0
        for length in episode_lengths:
            for start in range(0, length - batch_size + 1, batch_size):
                self.blocks.append(list(range(offset + start, offset + start + batch_size)))
            offset += length
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for index in rng.permutation(len(self.blocks)):
            yield self.blocks[int(index)]

    def __len__(self) -> int:
        return len(self.blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standard", "lp"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiny-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--sampling-mode", choices=("random", "episode-block"), default="random")
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
            "args": vars(args),
            "stats": {key: value.to_dict() for key, value in stats.items()},
        },
        path,
    )


@torch.no_grad()
def evaluate(policy, loader: DataLoader, device: torch.device) -> dict[str, float]:
    policy.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions_hat = policy.predict_action_chunk(batch)
            abs_err = torch.nn.functional.l1_loss(batch["action"], actions_hat, reduction="none")
            valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = valid_mask.sum() * abs_err.shape[-1]
            l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        values = {"loss": float(l1_loss), "l1_loss": float(l1_loss)}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
    policy.train()
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.mode == "lp" and args.metric_config is None:
        raise ValueError("--metric-config is required for LP mode")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device(args.device)
    stats = load_stats(args.stats)
    target_stats = stats["standard_action" if args.mode == "standard" else "lp_action"]
    metric = None
    if args.metric_config is not None:
        with args.metric_config.open() as handle:
            metric = MetricConfig.from_dict(json.load(handle))

    trajectory = load_turning_data(args.data_root)
    train_episodes, validation_episodes = split_episodes(trajectory.episode_bounds)
    dataset: Dataset = TurningACTDataset(
        args.data_root,
        train_episodes,
        args.mode,
        stats["observation.state"],
        target_stats,
        metric,
    )
    if args.tiny_samples:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(dataset), size=min(args.tiny_samples, len(dataset)), replace=False).tolist())
        subset = Subset(dataset, indices)
        dataset = CachedSamples([subset[index] for index in range(len(subset))])
        num_workers = 0
        shuffle = True
        validation_loader = None
    else:
        num_workers = args.num_workers
        shuffle = True
        validation_dataset = TurningACTDataset(
            args.data_root,
            validation_episodes,
            args.mode,
            stats["observation.state"],
            target_stats,
            metric,
            video_backend="pyav",
        )
        validation_rng = np.random.default_rng(args.seed + 1)
        validation_indices = sorted(
            validation_rng.choice(
                len(validation_dataset),
                size=min(args.validation_samples, len(validation_dataset)),
                replace=False,
            ).tolist()
        )
        validation_subset = Subset(validation_dataset, validation_indices)
        cached_validation = CachedSamples(
            [validation_subset[index] for index in range(len(validation_subset))]
        )
        validation_loader = DataLoader(
            cached_validation,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    if args.sampling_mode == "episode-block" and not args.tiny_samples:
        episode_lengths = [trajectory.episode_bounds[episode][1] - trajectory.episode_bounds[episode][0] for episode in train_episodes]
        batch_sampler = EpisodeBlockBatchSampler(episode_lengths, args.batch_size, args.seed)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    policy = make_act_policy(args.mode, learning_rate=args.learning_rate).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.learning_rate, weight_decay=policy.config.optimizer_weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    iterator = iter(loader)
    start_time = time.time()
    initial_l1 = None
    best_validation_l1 = float("inf")
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, loss_dict = policy(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
        if initial_l1 is None:
            initial_l1 = loss_dict["l1_loss"]
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "mode": args.mode,
                "loss": float(loss.detach()),
                **loss_dict,
                "initial_l1_loss": initial_l1,
                "grad_norm": float(grad_norm),
                "elapsed_s": time.time() - start_time,
                "samples": len(dataset),
                "train_episodes": len(train_episodes),
                "validation_episodes": len(validation_episodes),
            }
            print(json.dumps(record), flush=True)
        if step % args.checkpoint_every == 0 and step != args.steps:
            save_checkpoint(args.output_dir / f"checkpoint_{step:07d}.pt", policy, optimizer, step, args, stats)
        if validation_loader is not None and (step % args.validate_every == 0 or step == args.steps):
            validation = evaluate(policy, validation_loader, device)
            record = {
                "step": step,
                "mode": args.mode,
                "split": "validation",
                **validation,
                "elapsed_s": time.time() - start_time,
                "validation_samples": len(validation_loader.dataset),
            }
            print(json.dumps(record), flush=True)
            if validation["l1_loss"] < best_validation_l1:
                best_validation_l1 = validation["l1_loss"]
                save_checkpoint(args.output_dir / "checkpoint_best.pt", policy, optimizer, step, args, stats)
    save_checkpoint(args.output_dir / "checkpoint_final.pt", policy, optimizer, args.steps, args, stats)


if __name__ == "__main__":
    main()
