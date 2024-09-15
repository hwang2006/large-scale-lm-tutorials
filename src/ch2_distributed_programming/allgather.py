"""
src/allgather.py
$ srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu14:12345 allgather.py
"""

import torch
import torch.distributed as dist

import os

dist.init_process_group("nccl")
local_rank = int(os.environ['LOCAL_RANK'])
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

input = torch.ones(1).to(torch.cuda.current_device()) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

outputs_list = [
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
]

outputs_list =[torch.zeros(1).to(torch.cuda.current_device()) for _ in range(world_size)]

dist.all_gather(tensor_list=outputs_list, tensor=input)
print(outputs_list)
