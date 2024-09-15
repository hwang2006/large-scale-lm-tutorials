"""
src/broadcast.py
$salloc --partition=cas_v100_4 -J debug --nodes=2 --time=12:00:00 --gres=gpu:2 --comment=python
salloc: Granted job allocation 299473
salloc: Waiting for resource configuration
salloc: Nodes gpu[15,19] are ready for job

$srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu15:12345 broadcast.py
"""

import torch
import torch.distributed as dist

import os


dist.init_process_group("nccl")
#dist.init_process_group("gloo")
#dist.init_process_group("mpi")
rank = dist.get_rank()
local_rank = int(os.environ['LOCAL_RANK'])

#torch.cuda.set_device(rank)
torch.cuda.set_device(local_rank)
# device를 setting하면 이후에 rank에 맞는 디바이스에 접근 가능합니다.

size = dist.get_world_size()
group = dist.new_group([0,size-1])

if rank == 0:
    tensor = torch.randn(1000, 1000).to(torch.cuda.current_device())
else:
    tensor = torch.zeros(1000, 1000).to(torch.cuda.current_device())

print(f"before rank {rank}: {tensor}\n")
#dist.broadcast(tensor, src=0)
dist.broadcast(tensor, src=0, group=group)
print(f"after rank {rank}: {tensor}\n")
