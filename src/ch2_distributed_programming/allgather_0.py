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

input = torch.ones(1000,250).to(torch.cuda.current_device()) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

outputs_list = [
    torch.zeros(1000, 250, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1000, 250, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1000, 250, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1000, 250, device=torch.device(torch.cuda.current_device())),
]

outputs_list =[torch.zeros(1000, 250).to(torch.cuda.current_device()) for _ in range(world_size)]

dist.all_gather(tensor_list=outputs_list, tensor=input)
#print(outputs_list)

output = torch.cat((outputs_list[0], outputs_list[1], outputs_list[2], outputs_list[3]), dim=1)
print(output)

