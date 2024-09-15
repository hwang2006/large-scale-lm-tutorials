import torch.multiprocessing as mp
import torch.distributed as dist
import os


def worker(rank, world_size):
    # Initialize distributed process group with Gloo backend
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Your training or parallel computation logic here
    # This part would typically involve data loading, model definition,
    # optimizer setup, and training loop with communication across processes.

    print(f"Rank {rank} is ready for training!")

    # Clean up the distributed process group after training
    dist.destroy_process_group()


if __name__ == "__main__":
    # Set environment variables (optional, can be launched with command-line arguments)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 4  # Adjust this based on the number of processes

    # Spawn worker processes
    mp.spawn(
        worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
        daemon=False,
        start_method="spawn",
    )

