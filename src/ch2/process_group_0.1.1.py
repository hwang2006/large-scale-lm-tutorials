"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
#from torch.multiprocessing import Process
import torch.multiprocessing as mp


#def run(rank, size):
#    """ Distributed function to be implemented later. """
#    print(f"rank: {rank}, size:{size}")


def run(rank, size):
    """Blocking point-to-point communication."""
    init_processes(rank, size, backend="gloo")

    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        for i in range(size-1):
           dist.send(tensor=tensor, dst=i+1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
#    print('Rank ', rank, ' has data ', tensor[0])
    print(f'Rank {rank} has data {tensor}')


def init_processes(rank, size, backend="gloo"):
#def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    #run(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")

#    os.environ['MASTER_ADDR'] = "localhost"
#    os.environ['MASTER_PORT'] = "29500"

    for rank in range(size):
        # p = Process(target=init_processes, args=(rank, size, run))
        # p = mp.Process(target=init_processes, args=(rank, size))
        p = mp.Process(target=run, args=(rank, size))
        p.daemon = False
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

