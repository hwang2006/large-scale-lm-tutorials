"""
src/reduce_scatter.py
$ srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu14:12345 reduce_scatter.pe
"""

import torch
import torch.distributed as dist

import os

dist.init_process_group("nccl")
#dist.init_process_group("mpi")
#dist.init_process_group("gloo")
local_rank = int(os.environ['LOCAL_RANK'])
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

input_list = torch.tensor([1, 10, 100, 1000]).to(torch.cuda.current_device()) * rank
#print(type(input_list)) #<class 'torch.Tensor'>
#print(input_list)
#rank0 => tensor([   0,   0,  0, 0], device='cuda:1')
#rank1 => tensor([   1,   10,  100, 1000], device='cuda:1')
#rank2 => tensor([   2,   20,  200, 2000], device='cuda:1')
#rank3 => tensor([   3,   30,  300, 3000], device='cuda:1')

#print(type(list(input_list))) #<class 'list'>
#print(list(input_list))
# rank0 => [tensor([0]), tensor([0]), tensor([0]), tensor([0])]
# rank1 => [tensor([1]), tensor([10]), tensor([100]), tensor([1000])]
# rank2 => [tensor([2]), tensor([20]), tensor([200]), tensor([2000])]
# rank3 => [tensor([3]), tensor([30]), tensor([300]), tensor([3000])]

#output = torch.tensor([0], device=torch.device(torch.cuda.current_device()),)
#output = torch.zeros(1).to(torch.cuda.current_device())
output = torch.tensor([0]).to(torch.cuda.current_device())

print(f"Before: rank {rank} output: {output}\n")

dist.reduce_scatter(
    output=output,
    input_list=list(input_list),
    op=torch.distributed.ReduceOp.SUM,
)

print(f"After: rank {rank} output: {output}\n")

