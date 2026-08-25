"""Four-GPU scratch ACT training for NavigateKitchen."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from .data import load_data, load_split
from .dataset import NavigateACTDataset, NavigateBaseOnlyACTDataset, NavigateGeometricACTDataset
from .policy import make_policy


class CachedSamples(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--tiny-samples", type=int, default=0)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--geometric-path", action="store_true")
    parser.add_argument("--path-cache", type=Path)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite(value: torch.Tensor | float) -> bool:
    return math.isfinite(float(value))


def save_checkpoint(path: Path, model, optimizer, step: int, args: argparse.Namespace, stats: dict, *, include_optimizer: bool) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "stats": stats,
        "protocol": {
            "task": "NavigateKitchen",
            "representation": (
                "fixed_token_common_origin_measured_se2_path"
                if args.geometric_path
                else "native_time_indexed_base_command_3d"
                if args.base_only
                else "native_time_indexed_12d_action_chunk"
            ),
            "learned_action_dim": 3 if (args.base_only or args.geometric_path) else 12,
            "execution_defaults": (
                {
                    "torso": 0.0,
                    "control_mode": 1.0,
                    "end_effector_position": [0.0, 0.0, 0.0],
                    "end_effector_rotation": [0.0, 0.0, 0.0],
                    "gripper_close": -1.0,
                }
                if (args.base_only or args.geometric_path) else None
            ),
            "label_frequency_hz": 20,
            "execution_control_frequency_hz": 30 if args.geometric_path else 20,
            "predicts_time_duration_velocity_rate": False if args.geometric_path else None,
            "scratch": True,
            "chunk_size": 32,
            "n_action_steps": 8,
            "batch_size_per_gpu": args.batch_size_per_gpu,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "global_batch_size": args.batch_size_per_gpu * int(os.environ.get("WORLD_SIZE", "1")),
            "learning_rate": args.learning_rate,
            "optimizer_steps": step,
            "seed": args.seed,
        },
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.base_only and args.geometric_path:
        raise ValueError("--base-only and --geometric-path are mutually exclusive")
    if args.geometric_path and args.path_cache is None:
        raise ValueError("--geometric-path requires --path-cache")
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed_all(args.seed + rank)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()

    stats = json.loads(args.stats.read_text())
    data = load_data(args.data_root)
    splits = load_split(args.split)
    if args.geometric_path:
        dataset: Dataset = NavigateGeometricACTDataset(
            data, splits["train"], args.frame_cache, args.path_cache
        )
    else:
        dataset_class = NavigateBaseOnlyACTDataset if args.base_only else NavigateACTDataset
        dataset = dataset_class(data, splits["train"], args.frame_cache)
    workers = args.num_workers
    if args.tiny_samples:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(dataset), size=min(args.tiny_samples, len(dataset)), replace=False).tolist())
        subset = Subset(dataset, indices)
        dataset = CachedSamples([subset[index] for index in range(len(subset))])
        workers = 0
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
    )
    policy = make_policy(stats, learning_rate=args.learning_rate, device="cuda").to(device)
    wrapped = DistributedDataParallel(policy, device_ids=[local_rank], broadcast_buffers=False) if distributed else policy
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.learning_rate, weight_decay=policy.config.optimizer_weight_decay)
    iterator = iter(loader)
    epoch = 0
    start = time.time()
    initial_l1 = None
    if rank == 0:
        print(json.dumps({
            "event": "TRAINING_STARTED", "scratch": True, "steps": args.steps,
            "world_size": world, "batch_size_per_gpu": args.batch_size_per_gpu,
            "global_batch_size": world * args.batch_size_per_gpu, "samples": len(dataset),
        }), flush=True)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_dict = wrapped(batch)
        if not finite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        if not finite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}: {float(grad_norm)}")
        optimizer.step()
        if initial_l1 is None:
            initial_l1 = float(loss_dict["l1_loss"])
        if rank == 0 and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(json.dumps({
                "step": step, "loss": float(loss.detach()),
                "l1_loss": float(loss_dict["l1_loss"]), "kld_loss": float(loss_dict["kld_loss"]),
                "initial_l1_loss": initial_l1, "grad_norm": float(grad_norm),
                "elapsed_s": time.time() - start, "global_batch_size": world * args.batch_size_per_gpu,
            }), flush=True)
        if rank == 0 and (step % args.checkpoint_every == 0 or step == args.steps):
            save_checkpoint(args.output_dir / f"checkpoint-{step:05d}-model.pt", policy, optimizer, step, args, stats, include_optimizer=False)
            save_checkpoint(args.output_dir / "resume-latest.pt", policy, optimizer, step, args, stats, include_optimizer=True)
            print(json.dumps({"event": "CHECKPOINT_SAVED", "step": step}), flush=True)
    if rank == 0:
        (args.output_dir / "TRAINING_COMPLETE").write_text("\n")
        print(json.dumps({"event": "TRAINING_COMPLETE", "step": args.steps}), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
