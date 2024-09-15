"""
src/scatter.py

$srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu15:12345 scatter.py
"""

import torch
import torch.distributed as dist

import os

dist.init_process_group("gloo")

local_rank = int(os.environ["LOCAL_RANK"])
# nccl은 scatter를 지원하지 않습니다.
rank = dist.get_rank()

torch.cuda.set_device(local_rank)


output = torch.zeros(1)
print(f"before rank {rank}: {output}\n")

if rank == 0:
    inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
    inputs = torch.split(inputs, dim=0, split_size_or_sections=1)
    #inputs = inputs.split(1, dim=0)
    # (tensor([10]), tensor([20]), tensor([30]), tensor([40]))
    dist.scatter(output, scatter_list=list(inputs), src=0)
else:
    dist.scatter(output, src=0)

print(f"after rank {rank}: {output}\n")
