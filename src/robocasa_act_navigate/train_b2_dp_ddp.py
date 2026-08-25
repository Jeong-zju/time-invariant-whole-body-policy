"""Four-GPU Diffusion Policy training on B2 Path-RateFree whole-body targets."""

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
from .dp_policy import make_dp_policy, temporalize_dp_batch
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
    parser.add_argument("--schedule-total-steps", type=int, default=50_000)
    parser.add_argument("--batch-size-per-gpu", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--tiny-samples", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl", device_id=device)
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
    selected: list[int] | None = None
    if args.tiny_samples:
        rng = np.random.default_rng(args.seed)
        selected = sorted(rng.choice(len(dataset), size=min(args.tiny_samples, len(dataset)), replace=False).tolist())
        subset = Subset(dataset, selected)
        dataset = CachedSamples([subset[index] for index in range(len(subset))])
        workers = 0
        if rank == 0:
            (args.output_dir / "selected-indices.json").write_text(json.dumps(selected) + "\n")
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
    policy = make_dp_policy(stats, learning_rate=args.learning_rate, device="cuda").to(device)
    wrapped = DistributedDataParallel(policy, device_ids=[local_rank], broadcast_buffers=False) if distributed else policy
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.learning_rate,
        betas=policy.config.optimizer_betas,
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )
    scheduler = policy.config.get_scheduler_preset().build(optimizer, num_training_steps=args.schedule_total_steps)
    start_step = 0
    if args.resume_from is not None:
        payload = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
    if start_step + args.steps > args.schedule_total_steps:
        raise ValueError("requested training exceeds the declared cosine schedule")

    iterator, epoch = iter(loader), 0
    started = time.time()
    first_loss = None
    if rank == 0:
        print(json.dumps({
            "event": "B2_DP_TRAINING_STARTED",
            "scratch": start_step == 0,
            "start_step": start_step,
            "additional_steps": args.steps,
            "target_step": start_step + args.steps,
            "schedule_total_steps": args.schedule_total_steps,
            "world_size": world,
            "batch_size_per_gpu": args.batch_size_per_gpu,
            "global_batch_size": world * args.batch_size_per_gpu,
            "samples": len(dataset),
            "parameters": sum(value.numel() for value in policy.parameters()),
            "policy": "DiffusionPolicy",
            "prediction_type": policy.config.prediction_type,
            "predicted_time_duration_velocity_rate": False,
            "metric": metric.to_dict(),
        }), flush=True)

    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        batch = temporalize_dp_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = wrapped(batch)
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f"non-finite DP loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        if not math.isfinite(float(grad)):
            raise FloatingPointError(f"non-finite DP gradient at step {step}")
        optimizer.step()
        scheduler.step()
        if first_loss is None:
            first_loss = float(loss)
        if rank == 0 and (local_step == 1 or step % args.log_every == 0 or local_step == args.steps):
            print(json.dumps({
                "step": step,
                "loss": float(loss),
                "initial_loss": first_loss,
                "grad_norm": float(grad),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - started,
                "global_batch_size": world * args.batch_size_per_gpu,
            }), flush=True)
        should_save = not args.no_save and (
            (args.checkpoint_every > 0 and step % args.checkpoint_every == 0) or local_step == args.steps
        )
        if rank == 0 and should_save:
            protocol = {
                "task": "NavigateKitchen",
                "representation": "B2_Path_RateFree_SE2Delta_Stop_WholeBody_DP_V1",
                "policy": "LeRobot_DiffusionPolicy",
                "scratch_initialization": True,
                "horizon": metric.num_anchors,
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "world_size": world,
                "global_batch_size": world * args.batch_size_per_gpu,
                "optimizer_steps": step,
                "training_start_step": start_step,
                "seed": args.seed,
                "metric": metric.to_dict(),
                "noise_prediction_type": policy.config.prediction_type,
                "num_train_timesteps": policy.config.num_train_timesteps,
                "num_inference_steps": policy.diffusion.num_inference_steps,
                "predicted_time_duration_velocity_rate": False,
                "predicted_stop": True,
                "whole_body_progress_sync": "attained_anchor",
                "base_parameterization": "body_frame_se2_lie_increments",
                "geometry_loss": False,
            }
            model_payload = {"step": step, "model": policy.state_dict(), "stats": stats, "protocol": protocol}
            model_path = args.output_dir / f"checkpoint-{step:05d}-model.pt"
            temporary = model_path.with_suffix(model_path.suffix + ".tmp")
            torch.save(model_payload, temporary); temporary.replace(model_path)
            resume_payload = dict(model_payload)
            resume_payload.update({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
            resume_path = args.output_dir / "resume-latest.pt"
            temporary = resume_path.with_suffix(resume_path.suffix + ".tmp")
            torch.save(resume_payload, temporary); temporary.replace(resume_path)
            print(json.dumps({"event": "B2_DP_CHECKPOINT_SAVED", "step": step}), flush=True)
    if rank == 0:
        (args.output_dir / "TRAINING_COMPLETE").write_text("\n")
        print(json.dumps({"event": "B2_DP_TRAINING_COMPLETE", "step": start_step + args.steps}), flush=True)
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
