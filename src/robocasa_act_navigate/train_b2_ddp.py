"""Four-GPU scratch ACT training with B2 Path-RateFree targets."""

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

from .b2_dataset import NavigateB2Dataset
from .b2_labels import RateFreeMetric
from .data import load_data, load_split
from .policy import make_policy
from .train_ddp import CachedSamples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--tiny-samples", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = args.seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()

    stats = json.loads(args.stats.read_text())
    metric = RateFreeMetric.from_dict(json.loads(args.metric.read_text()))
    data = load_data(args.data_root)
    split = load_split(args.split)
    dataset: Dataset = NavigateB2Dataset(data, split["train"], args.frame_cache, metric)
    workers = args.num_workers
    if args.tiny_samples:
        rng = np.random.default_rng(args.seed)
        selected = sorted(rng.choice(len(dataset), size=min(args.tiny_samples, len(dataset)), replace=False).tolist())
        subset = Subset(dataset, selected)
        dataset = CachedSamples([subset[index] for index in range(len(subset))])
        workers = 0
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, sampler=sampler, shuffle=sampler is None,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0, drop_last=True)
    policy = make_policy(stats, learning_rate=args.learning_rate, device="cuda").to(device)
    wrapped = DistributedDataParallel(policy, device_ids=[local_rank], broadcast_buffers=False) if distributed else policy
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.learning_rate, weight_decay=policy.config.optimizer_weight_decay)
    start_step = 0
    if args.resume_from is not None:
        payload = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
    iterator, epoch = iter(loader), 0
    start = time.time()
    initial_l1 = None
    if rank == 0:
        print(json.dumps({"event": "B2_TRAINING_STARTED", "scratch": start_step == 0,
                          "start_step": start_step, "additional_steps": args.steps,
                          "target_step": start_step + args.steps,
                          "world_size": world, "batch_size_per_gpu": args.batch_size_per_gpu,
                          "global_batch_size": world * args.batch_size_per_gpu, "samples": len(dataset),
                          "predicted_time_duration_velocity_rate": False, "metric": metric.to_dict()}), flush=True)
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None: sampler.set_epoch(epoch)
            iterator = iter(loader); batch = next(iterator)
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, losses = wrapped(batch)
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        if not math.isfinite(float(grad)):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        if initial_l1 is None: initial_l1 = float(losses["l1_loss"])
        if rank == 0 and (local_step == 1 or step % args.log_every == 0 or local_step == args.steps):
            record = {"step": step, "loss": float(loss), "initial_l1_loss": initial_l1,
                      "grad_norm": float(grad), "elapsed_s": time.time() - start,
                      "global_batch_size": world * args.batch_size_per_gpu}
            record.update({name: float(value) for name, value in losses.items()})
            print(json.dumps(record), flush=True)
        if rank == 0 and (step % args.checkpoint_every == 0 or local_step == args.steps):
            protocol = {"task": "NavigateKitchen", "representation": "B2_Path_RateFree_SE2Delta_Stop_WholeBody_V4",
                        "scratch": True, "chunk_size": metric.num_anchors, "batch_size_per_gpu": args.batch_size_per_gpu,
                        "world_size": world, "global_batch_size": world * args.batch_size_per_gpu,
                        "optimizer_steps": step, "training_start_step": start_step,
                        "seed": args.seed, "metric": metric.to_dict(),
                        "predicted_time_duration_velocity_rate": False, "predicted_stop": True,
                        "whole_body_progress_sync": "attained_anchor",
                        "base_parameterization": "body_frame_se2_lie_increments",
                        "geometry_loss": stats.get("training", {}).get("geometry_loss", {})}
            for name, include_optimizer in ((f"checkpoint-{step:05d}-model.pt", False), ("resume-latest.pt", True)):
                payload = {"step": step, "model": policy.state_dict(), "stats": stats, "protocol": protocol}
                if include_optimizer: payload["optimizer"] = optimizer.state_dict()
                path = args.output_dir / name; temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(payload, temporary); temporary.replace(path)
            print(json.dumps({"event": "CHECKPOINT_SAVED", "step": step}), flush=True)
    if rank == 0:
        (args.output_dir / "TRAINING_COMPLETE").write_text("\n")
        print(json.dumps({"event": "TRAINING_COMPLETE", "step": start_step + args.steps}), flush=True)
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
