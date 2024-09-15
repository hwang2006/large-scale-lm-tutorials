"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
#from torch.multiprocessing import Process
import torch.multiprocessing as mp


def run(rank, size):
    """ Distributed function to be implemented later. """
#    pass
    print(f"rank: {rank}, size:{size}")

def init_processes(rank, size, fn, backend="gloo"):
#def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
#    os.environ['MASTER_ADDR'] = "localhost"
#    os.environ['MASTER_PORT'] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")

    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"

    for rank in range(size):
#       p = Process(target=init_processes, args=(rank, size, run))
        p = mp.Process(target=init_processes, args=(rank, size, run))
        p.daemon = False
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

