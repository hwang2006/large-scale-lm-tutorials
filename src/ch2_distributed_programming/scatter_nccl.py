"""
src/scatter_nccl.py
$ srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu14:12345 scatter_nccl.py
"""

import torch
import torch.distributed as dist

import os

dist.init_process_group("nccl")
local_rank = int(os.environ['LOCAL_RANK'])
world_size = dist.get_world_size()
rank = dist.get_rank()
torch.cuda.set_device(local_rank)

inputs = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])

sections = []
N = inputs.size(0)
base = N // world_size
extra = N % world_size

for i in range(world_size):
    sections.append(base + (1 if i < extra else 0))
# sections = [3, 3, 2, 2]
inputs_split = torch.split(inputs, dim=-1, split_size_or_sections=sections)
output = inputs_split[rank].contiguous().to(torch.cuda.current_device())

dist.destroy_process_group()

print(f"rank {rank}: {output}")

