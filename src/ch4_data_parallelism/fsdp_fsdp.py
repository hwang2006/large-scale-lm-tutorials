import os
import torch
from torch import nn
import torch.optim as optim

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            *(nn.Linear(10000, 10000) for _ in range(10))
        )

    def forward(self, x):
        return self.linear(x)


model = Layer()

from torch.optim import SGD
from torch.cuda import max_memory_allocated
from torch.cuda.amp import autocast 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel 
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import _module_wrap_policy

import functools

#dist.init_process_group("nccl")
#torch.cuda.set_device(rank)

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    wrap_policy=functools.partial(_module_wrap_policy, module_classes={nn.Linear})
    fsdp_model = FullyShardedDataParallel(model, device_id=rank,auto_wrap_policy=wrap_policy)
    optimizer = SGD(fsdp_model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for _ in range(10):
        optimizer.zero_grad()
        output = fsdp_model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        memory = max_memory_allocated()
        if rank == 0:
            print(f'step memory allocate: {memory / 1e9:.3f}G')
        torch.cuda.reset_max_memory_allocated()

if __name__ == "__main__":
   main()
