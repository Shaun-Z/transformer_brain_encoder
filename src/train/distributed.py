from __future__ import annotations

import os

import torch
import torch.distributed as dist


def distributed_requested(explicit_flag: bool) -> bool:
    return explicit_flag or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(explicit_flag: bool) -> tuple[bool, int, int, int]:
    if not distributed_requested(explicit_flag):
        return False, 0, 1, 0

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0
