#torchrun fsdp_ddp.py
#parameter size:p, gradients:p, optimizer states:p for SGD

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

from torch.optim import SGD, Adam, AdamW
from torch.cuda import max_memory_allocated
from torch.cuda.amp import autocast 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel 

#dist.init_process_group("nccl")
#torch.cuda.set_device(rank)

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    model = DistributedDataParallel(Layer().cuda())
    #optimizer = AdamW(model.parameters(), lr=0.1)
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for _ in range(10):
        with autocast(cache_enabled=False):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        memory = max_memory_allocated()
        if rank == 0:
            print(f'memory allocated: {memory / 1e9:.3f}G')


if __name__ == "__main__":
   main()

