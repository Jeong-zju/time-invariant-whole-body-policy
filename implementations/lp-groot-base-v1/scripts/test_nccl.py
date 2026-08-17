"""Minimal two-rank NCCL broadcast smoke test."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    value = torch.ones(25_000_000, dtype=torch.float32, device=device)
    dist.broadcast(value, src=0)
    torch.cuda.synchronize(device)
    print(f"rank={local_rank} value={float(value[0])} bytes={value.nbytes}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
