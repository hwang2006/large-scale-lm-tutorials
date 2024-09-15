"""
src/reduce_sum.py
$srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu15:12345 reduce_sum.py
"""

import torch
import torch.distributed as dist

import os

dist.init_process_group("nccl")
#dist.init_process_group("gloo")
rank = dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])

#torch.cuda.set_device(rank)
torch.cuda.set_device(local_rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.reduce(tensor, op=torch.distributed.ReduceOp.SUM, dst=0)

if rank == 0:
    print(tensor)
