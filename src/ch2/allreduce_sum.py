"""
src/allreduce_sum.py
$ srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu14:12345 allreduce_sum.py
"""

import torch
import torch.distributed as dist

import os

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
rank = dist.get_rank()
torch.cuda.set_device(local_rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

print(f"rank {rank}: {tensor}\n")
