#!/usr/bin/env python3
"""Small correctness and bandwidth smoke test for a multi-node NCCL world."""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    check = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(check)
    expected = world * (world + 1) / 2
    if check.item() != expected:
        raise RuntimeError(f"all_reduce mismatch: {check.item()} != {expected}")

    size_mb = int(os.environ.get("NCCL_SMOKE_MB", "64"))
    warmup = int(os.environ.get("NCCL_SMOKE_WARMUP", "5"))
    iterations = int(os.environ.get("NCCL_SMOKE_ITERS", "20"))
    tensor = torch.ones(size_mb * 1024 * 1024 // 4, device=f"cuda:{local_rank}")
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - started

    local = torch.tensor([elapsed], dtype=torch.float64, device=f"cuda:{local_rank}")
    dist.reduce(local, dst=0, op=dist.ReduceOp.MAX)
    print(
        f"NCCL_RANK_PASS rank={rank} local_rank={local_rank} "
        f"host={socket.gethostname()} world={world}",
        flush=True,
    )
    if rank == 0:
        slowest = local.item()
        payload_gib = size_mb * iterations / 1024
        print(
            f"NCCL_SMOKE_PASS world={world} payload_mb={size_mb} "
            f"iterations={iterations} slowest_seconds={slowest:.6f} "
            f"payload_gib_per_second={payload_gib / slowest:.3f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
