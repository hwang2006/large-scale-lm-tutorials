"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
#from torch.multiprocessing import Process
import torch.multiprocessing as mp


def init_processes(rank, size, backend="gloo"):
#def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)

def run(rank, size):
    init_processes(rank, size)
    #tensor = torch.ones(2,2) * rank
    tensor = torch.ones(2,2)
    group = dist.new_group([_ for _ in range(size)])
    gr1 = dist.new_group([2,3])
    print(f"before rank {rank}: {tensor}\n")
    #dist.broadcast(tensor, src=0)
    #dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    #dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=gr1)
    print(f"after rank {rank}: {tensor}\n")


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
