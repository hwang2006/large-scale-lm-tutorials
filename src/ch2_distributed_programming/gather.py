"""
src/gather.py
$ srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu14:12345 gather.py
"""

import torch
import torch.distributed as dist

import os

dist.init_process_group("gloo")
local_rank = int(os.environ["LOCAL_RANK"])
# nccl은 gather를 지원하지 않습니다.
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

input = torch.ones(1) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

if rank == 0:
    #outputs_list = [torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    #outputs_list = [torch.tensor([0.0]) for _ in range(world_size)]
    outputs_list = [torch.zeros(1) for _ in range(world_size)]
    dist.gather(input, gather_list=outputs_list, dst=0)
    print(outputs_list)
else:
    dist.gather(input, dst=0)

