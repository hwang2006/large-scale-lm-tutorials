# Distributed Programming
Large-scale 모델은 크기가 크기 때문에 여러대의 GPU에 쪼개서 모델을 올려야 합니다. 그리고 쪼개진 각 모델의 조각들끼리 네트워크로 통신을 하면서 값을 주고 받아야 합니다. 이렇게 커다란 리소스를 여러대의 컴퓨터 혹은 여러대의 장비에 분산시켜서 처리하는 것을 '분산처리'라고 합니다. 이번 세션에서는 PyTorch를 이용한 분산 프로그래밍의 기초에 대해 알아보겠습니다.
  
## 1. Multi-processing with PyTorch
   
분산프로그래밍 튜토리얼에 앞서 PyTorch로 구현된 Multi-processing 애플리케이션에 대한 튜토리얼을 진행합니다. 쓰레드 및 프로세스의 개념 등은 Computer Scienece 전공자라면 운영체제 시간에 배우는 것들이니 생략하도록 하겠습니다. 만약 이러한 개념에 대해 잘 모르신다면, 구글에 검색하시거나 https://www.backblaze.com/blog/whats-the-diff-programs-processes-and-threads/ 와 같은 글을 먼저 읽어보는 것을 추천드립니다.

### Multi-process 통신에 쓰이는 기본 용어
- Node: 일반적으로 컴퓨터라고 생각하시면 됩니다. 노드 3대라고 하면 컴퓨터 3대를 의미합니다.
- Global Rank: 원래는 프로세스의 우선순위를 의미하지만 **ML에서는 GPU의 ID**라고 보시면 됩니다.
- Local Rank: 원래는 한 노드내에서의 프로세스 우선순위를 의미하지만 **ML에서는 노드내의 GPU ID**라고 보시면 됩니다
- World Size: 프로세스의 개수를 의미합니다.
    
![](../images/process_terms.png)
    
### Multi-process Application 실행 방법
PyTorch로 구현된 Multi-process 애플리케이션을 실행시키는 방법은 크게 두가지가 있습니다.
1. 사용자의 코드가 메인프로세스가 되어 특정 함수를 서브프로세스로 분기한다.
2. PyTorch 런처가 메인프로세스가 되어 사용자 코드 전체를 서브프로세스로 분기한다.
    
이 두가지 방법에 대해 모두 알아보겠습니다. 이때, '분기한다.'라는 표현이 나오는데, 이는 한 프로세스가 부모가 되어 여러개의 서브프로세스를 동시에 실행시키는 것을 의미합니다.
   
### 1) 사용자의 코드가 메인프로세스가 되어 특정 함수를 서브프로세스로 분기한다.
이 방식은 사용자의 코드가 메인프로세스가 되며 특정 function을 서브프로세스로써 분기하는 방식입니다.

![](../images/multi_process_1.png)
    
일반적으로 `Spawn`과 `Fork` 등 두가지 방식으로 서브프로세스를 분기 할 수 있습니다.
- `Spawn`
  - 메인프로세스의 자원을 물려주지 않고 필요한 만큼의 자원만 서브프로세스에게 새로 할당.
  - 속도가 느리지만 안전한 방식.
- `Fork`
  - 메인프로세스의 모든 자원을 서브프로세스와 공유하고 프로세스를 시작.
  - 속도가 빠르지만 위험한 방식.
p.s. 실제로는 `Forkserver` 방식도 있지만 자주 사용되지 않는 생소한 방식이기에 생략합니다.

```
"""
src/ch2/multi_process_1.py

참고:
Jupyter notebook은 멀티프로세싱 애플리케이션을 구동하는데에 많은 제약이 있습니다.
따라서 대부분의 경우 이곳에는 코드만 동봉하고 실행은 `src` 폴더에 있는 코드를 동작시키겠습니다.
실제 코드 동작은 `src` 폴더에 있는 코드를 실행시켜주세요.
"""

import torch.multiprocessing as mp
# 일반적으로 mp와 같은 이름을 사용합니다.


# 서브프로세스에서 동시에 실행되는 영역
def fn(rank, param1, param2):
    print(f"{param1} {param2} - rank: {rank}")


# 메인 프로세스
if __name__ == "__main__":
    processes = []
    # 시작 방법 설정
    mp.set_start_method("spawn")

    for rank in range(4):
        process = mp.Process(target=fn, args=(rank, "A0", "B1"))
        # 서브프로세스 생성
        #process.daemon = False
        process.daemon = True
        # False means that child processes will run independently of the main process
        # and will not be terminated when the main process exits.
        # 데몬 여부 (메인프로세스 종료시 함께 종료)
        process.start()
        # 서브프로세스 시작
        processes.append(process)

    for process in processes:
        process.join()  # main process is waitiing for the subprocesses exit.
        # 서브 프로세스 join (=완료되면 종료)

    print("Main Process is done")
```

코드를 실행하기 위해서 모듀을 로드하고 `src/ch2` 디렉토리로 이동하고 콘다 환경을 활성화 시킨다.
```
[glogin01]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.1
[glogin01]$ cd large-scale-lm-tutorials/src/ch2
[glogin01]$ conda activate largc-scale-lm   
(large-scale-lm) [glogin01]$ python multi_process_1.py
A0 B1 - rank: 0
A0 B1 - rank: 2
A0 B1 - rank: 3
A0 B1 - rank: 1
Main Process is done
```
`torch.multiprocessing.spawn` 함수를 이용하면 이 과정을 매우 쉽게 진행 할 수 있습니다.

```
"""
src/ch2/multi_process_2.py
"""

import torch.multiprocessing as mp


# 서브프로세스에서 동시에 실행되는 영역
def fn(rank, param1, param2):
    # rank는 기본적으로 들어옴. param1, param2는 spawn시에 입력됨.
    print(f"{param1} {param2} - rank: {rank}")


# 메인 프로세스
if __name__ == "__main__":
    mp.spawn(
        fn=fn,
        args=("A0", "B1"),
        nprocs=4,  # 만들 프로세스 개수
        join=True,  # 프로세스 join 여부
        daemon=False,  # 데몬 여부
        start_method="spawn",  # 시작 방법 설정
    )
```

```
(large-scale-lm) [glogin01]$ python multi_process_2.py
A0 B1 - rank: 1
A0 B1 - rank: 0
A0 B1 - rank: 3
A0 B1 - rank: 2
```

```
"""
참고: torch/multiprocessing/spawn.py

mp.spawn 함수는 아래와 같이 동작합니다.
"""

def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    _python_version_check()
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass
```

### 2) PyTorch 런처가 부모 프로세스가 되어 사용자 코드 전체를 서브프로세스로 분기한다.
이 방식은 torch에 내장된 멀티프로세싱 런처가 사용자 코드 전체를 서브프로세스로 실행시켜주는 매우 편리한 방식입니다.

`python -m torch.distributed.launch --nproc_per_node=n OOO.py` 또는 `torchrun --nproc_per_node=4 000.py`와 같은 명령어를 사용합니다.
```
"""
src/ch2/multi_process_3.py
"""

# 코드 전체가 서브프로세스가 됩니다.
import os

# RANK, LOCAL_RANK, WORLD_SIZE 등의 변수가 자동으로 설정됩니다.
print(f"hello world, {os.environ['RANK']}")
```

```
# (large-scale-lm) [glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 multi_process_3.py
(large-scale-lm) [glogin01]$ torchrun --nproc_per_node=4 multi_process_3.py
W0908 14:12:43.550000 140009840834368 torch/distributed/run.py:757]
W0908 14:12:43.550000 140009840834368 torch/distributed/run.py:757] *****************************************
W0908 14:12:43.550000 140009840834368 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0908 14:12:43.550000 140009840834368 torch/distributed/run.py:757] *****************************************
hello world, 0
hello world, 3
hello world, 1
hello world, 2
```
 

## 2. Distributed Programming with PyTorch
### Concept of Message Passing
메시지 패싱이란 동일한 주소공간을 공유하지 않는 여러 프로세스들이 데이터를 주고 받을 수 있도록 메시지라는 간접 정보를 주고 받는 것입니다. 예를 들면 Process-1이 특정 태그가 달린 데이터를 메시지 큐에 send하도록, Process-2가 해당 데이터를 receive하도록 코딩해놓으면 두 프로세스가 공유하는 메모리 공간 없이도 데이터를 주고 받을 수 있죠. Large-scale 모델 개발시에 사용되는 분산 통신에는 대부분 이러한 message passing 기법이 사용됩니다.
  
![](../images/message_passing.png)
   
### MPI (Massage Passing Interface)
MPI는 Message Passing에 대한 표준 인터페이스를 의미합니다. MPI에는 Process간의 Message Passing에 사용되는 여러 연산(e.g. broadcast, reduce, scatter, gather, ...)이 정의되어 있으며 대표적으로 OpenMPI라는 오픈소스가 존재합니다.

![](../images/open_mpi.png)
   
### NCCL & GLOO\n",
실제로는 openmpi 보다는 nccl이나 gloo 같은 라이브러리를 사용하게 됩니다.
- NCCL (NVIDIA Collective Communication Library)
  - NVIDIA에서 개발한 GPU 특화 Message Passing 라이브러리 ('nickel'이라고 읽음)\n",
  - NVIDIA GPU에서 사용시, 다른 도구에 비해 월등히 높은 성능을 보여주는 것으로 알려져있습니다.\n",
- GLOO (Facebook's Collective Communication Library)\n",
  - Facebook에서 개발된 Message Passing 라이브러리. \n",
  - `torch`에서는 주로 CPU 분산처리에 사용하라고 추천하고 있습니다.\n",
  
### 백엔드 라이브러리 선택 가이드
openmpi를 써야할 특별한 이유가 있는 것이 아니라면 nccl이나 gloo를 사용하는데, GPU에서 사용시 nccl, CPU에서 사용시 gloo를 사용하시면 됩니다. 더 자세한 정보는  https://pytorch.org/docs/stable/distributed.html 여기를 참고하세요.
  
각 백엔드별로 수행 가능한 연산은 다음과 같습니다.

![](../images/backends.png)
    
### `torch.distributed` 패키지
`gloo`, `nccl`, `openmpi` 등을 직접 사용해보는 것은 분명 좋은 경험이 될 것입니다. 그러나 시간 관계상 이들을 모두 다룰 수는 없고, 이들을 wrapping 하고 있는 `torch.distributed` 패키지를  사용하여 진행하겠습니다. 실제로 활용 단으로 가면 `nccl` 등을 직접 사용하지 않고 대부분의 경우 `torch.distributed` 등의 하이레벨 패키지를 사용하여 프로그래밍 하게 됩니다.
  
### Process Group
많은 프로세스를 관리하는 것은 어려운 일입니다. 따라서 프로세스 그룹을 만들어서 관리를 용이하게 합니다. `init_process_group`를 호출하면 전체 프로세스가 속한 default_pg(process group)가 만들어집니다. 프로세스 그룹을 초기화하는 `init_process_group` 함수는 **반드시 서브프로세스에서 실행**되어야 하며, 만약 추가로 사용자가 원하는 프로세스들만 모아서 그룹을 생성하려면 `new_group`을 호출하면 됩니다.
```
"""
src/ch2/process_group_1.py
"""

import torch.distributed as dist
# 일반적으로 dist와 같은 이름을 사용합니다.

dist.init_process_group(backend="nccl", rank=0, world_size=1)
# 프로세스 그룹 초기화
# 본 예제에서는 가장 자주 사용하는 nccl을 기반으로 진행하겠습니다.
# backend에 'nccl' 대신 'mpi'나 'gloo'를 넣어도 됩니다.

process_group = dist.new_group([0])
# 0번 프로세스가 속한 프로세스 그룹 생성

print(process_group)
```

```
(large-scale-lm) [glogin01]$ python process_group_1.py
Traceback (most recent call last):
  File "/scratch/qualis/git-projects/large-scale-lm-tutorials/src/ch2/process_group_1.py", line 8, in <module>
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 75, in wrapper
    return func(*args, **kwargs)
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 89, in wrapper
    func_return = func(*args, **kwargs)
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 1305, in init_process_group
    store, rank, world_size = next(rendezvous_iterator)
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/rendezvous.py", line 242, in _env_rendezvous_handler
    master_addr = _get_env_or_raise("MASTER_ADDR")
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/rendezvous.py", line 219, in _get_env_or_raise
    raise _env_error(env_var)
ValueError: Error initializing torch.distributed using env:// rendezvous: environment variable MASTER_ADDR expected, but not set
```

위 코드를 실행하면 에러가 발생합니다. 그 이유는 `MASTER_ADDR`, `MASTER_PORT` 등 필요한 변수가 설정되지 않았기 때문입니다. 이 값들을 설정하고 다시 실행시키겠습니다.
```
"""
src/ch2/process_group_2.py
"""

import torch.distributed as dist
import os


# 일반적으로 이 값들도 환경변수로 등록하고 사용합니다.
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# 통신에 필요한 주소를 설정합니다.
os.environ["MASTER_ADDR"] = "localhost"  # 통신할 주소 (보통 localhost를 씁니다.)
os.environ["MASTER_PORT"] = "29500"  # 통신할 포트 (임의의 값을 설정해도 좋습니다.)

dist.init_process_group(backend="nccl", rank=0, world_size=1)
# 프로세스 그룹 초기화

process_group = dist.new_group([0])
# 0번 프로세스가 속한 프로세스 그룹 생성

print(process_group)
```

```
(large-scale-lm) [globin01]$ python process_group_2.py
<torch.distributed.distributed_c10d.ProcessGroup object at 0x7f1ca3ed60f0>
```

위 예제는 프로세스 그룹의 API를 보여주기 위해서 메인프로세스에서 실행하였습니다. (프로세스가 1개 뿐이라 상관없었음) 실제로는 멀티프로세스 작업시 프로세스 그룹 생성 등의 작업은 반드시 서브프로세스에서 실행되어야 합니다.

```
"""
src/process_group_3.py
"""

import torch.multiprocessing as mp
import torch.distributed as dist
import os


# 서브프로세스에서 동시에 실행되는 영역
def fn(rank, world_size):
    # rank는 기본적으로 들어옴. world_size는 입력됨.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.new_group([_ for _ in range(world_size)])
    print(f"{group} - rank: {rank}")


# 메인 프로세스
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "4"

    mp.spawn(
        fn=fn,
        args=(4,),  # world_size 입력
        nprocs=4,  # 만들 프로세스 개수
        join=True,  # 프로세스 join 여부
        daemon=False,  # 데몬 여부
        start_method="spawn",  # 시작 방법 설정
    )
```

```
(large-scale-lm) [glogin01]$ python process_group_3.py
<torch.distributed.distributed_c10d.ProcessGroup object at 0x7f883a7f5870> - rank: 1
<torch.distributed.distributed_c10d.ProcessGroup object at 0x7f1515d05a30> - rank: 2
<torch.distributed.distributed_c10d.ProcessGroup object at 0x7f311f6fb5f0> - rank: 3
<torch.distributed.distributed_c10d.ProcessGroup object at 0x7f1c00707d30> - rank: 0
```

`python -m torch.distributed.launch --nproc_per_node=n OOO.py`를 사용할때는 아래와 같이 처리합니다. `dist.get_rank()`, `dist_get_world_size()`와 같은 함수를 이용하여 `rank`와 `world_size`를 알 수 있습니다.

```
"""
src/ch2/process_group_4.py
"""

import torch.distributed as dist

dist.init_process_group(backend="nccl")
# 프로세스 그룹 초기화

group = dist.new_group([_ for _ in range(dist.get_world_size())])
# 프로세스 그룹 생성

print(f"{group} - rank: {dist.get_rank()}\n")
```
```
# (large-scale-lm) [glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 process_group_4.py
(large-scale-lm) [glogin01]$ torchrun --nproc_per_node=4 process_group_4.py
W0908 14:33:37.263000 140621451208512 torch/distributed/run.py:757]
W0908 14:33:37.263000 140621451208512 torch/distributed/run.py:757] *****************************************
W0908 14:33:37.263000 140621451208512 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0908 14:33:37.263000 140621451208512 torch/distributed/run.py:757] *****************************************
<torch.distributed.distributed_c10d.ProcessGroup object at 0x7fee69a086f0> - rank: 3

<torch.distributed.distributed_c10d.ProcessGroup object at 0x7faeaaa9b8f0> - rank: 1

<torch.distributed.distributed_c10d.ProcessGroup object at 0x7efd8c4e1930> - rank: 0

<torch.distributed.distributed_c10d.ProcessGroup object at 0x7fde9bf62d30> - rank: 2
```

### P2P Communication (Point to point)
![](../images/p2p.png)
P2P (Point to point, 점 대 점) 통신은 특정 프로세스에서 다른 프로세스 데이터를 전송하는 통신이며 `torch.distributed` 패키지의 `send`, `recv` 함수를 활용하여 통신할 수 있습니다.

```
"""
src/ch2/p2p_communication.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
# 현재 nccl은 send, recv를 지원하지 않습니다. (2021/10/21)

if dist.get_rank() == 0:
    tensor = torch.randn(2, 2)
    dist.send(tensor, dst=1)

elif dist.get_rank() == 1:
    tensor = torch.zeros(2, 2)
    print(f"rank 1 before: {tensor}\n")
    dist.recv(tensor, src=0)
    print(f"rank 1 after: {tensor}\n")

else:
    raise RuntimeError("wrong rank")
```

```
#(large-scale-lm) [glogin01]$ python -m torch.distributed.launch --nproc_per_node=2 p2p_communication.py
(large-scale-lm) [glogin01]$ torchrun --nproc_per_node=2 p2p_communication.py
W0908 15:33:07.475000 139958444984128 torch/distributed/run.py:757]
W0908 15:33:07.475000 139958444984128 torch/distributed/run.py:757] *****************************************
W0908 15:33:07.475000 139958444984128 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0908 15:33:07.475000 139958444984128 torch/distributed/run.py:757] *****************************************
rank 1 before: tensor([[0., 0.],
        [0., 0.]])

rank 1 after: tensor([[ 3.4373,  1.1562],
        [ 0.1356, -1.2088]])
```
주의할 것은 이들이 동기적으로 통신한다는 것입니다. 비동기 통신(non-blocking)에는 `isend`, `irecv`를 이용합니다. 이들은 비동기적으로 작동하기 때문에 `wait()` 메서드를 통해 다른 프로세스의 통신이 끝날때 까지 기다리고 난 뒤에 접근해야합니다.


```
"""
src/ch2/p2p_communication_non_blocking.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
# 현재 nccl은 send, recv를 지원하지 않습니다. (2021/10/21)

if dist.get_rank() == 0:
    tensor = torch.randn(2, 2)
    request = dist.isend(tensor, dst=1)
elif dist.get_rank() == 1:
    tensor = torch.zeros(2, 2)
    request = dist.irecv(tensor, src=0)
else:
    raise RuntimeError("wrong rank")

request.wait()

print(f"rank {dist.get_rank()}: {tensor}")
```

```
#(large-scale-lm) [globin01]$ python -m torch.distributed.launch --nproc_per_node=2 p2p_communication_non_blocking.py
(large-scale-lm) [glogin01]$ torchrun --nproc_per_node=2 p2p_communication_non_blocking.py
W0908 15:36:30.701000 139730208278336 torch/distributed/run.py:757]
W0908 15:36:30.701000 139730208278336 torch/distributed/run.py:757] *****************************************
W0908 15:36:30.701000 139730208278336 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0908 15:36:30.701000 139730208278336 torch/distributed/run.py:757] *****************************************
rank 0: tensor([[-1.2224,  0.7753],
        [ 0.9491, -1.0411]])
rank 1: tensor([[-1.2224,  0.7753],
        [ 0.9491, -1.0411]])
```

### Collective Communication
Collective Communication은 집합통신이라는 뜻으로 여러 프로세스가 참여하여 통신하는 것을 의미합니다. 다양한 연산들이 있지만 기본적으로 아래와 같은 4개의 연산(`broadcast`, `scatter`, `gather`, `reduce`)이 기본 세트입니다.
   
![](../images/collective.png)
  
여기에 추가로 `all-reduce`, `all-gather`, `reduce-scatter` 등의 복합 연산과 동기화 연산인 `barrier`까지 총 8개 연산에 대해 알아보겠습니다. 추가로 만약 이러한 연산들을 비동기 모드로 실행하려면 각 연산 수행시 `async_op` 파라미터를 `True`로 설정하면 됩니다.
   
#### 1) Broadcast
Broadcast는 특정 프로세스에 있는 데이터를 그룹내의 모든 프로세스에 복사하는 연산입니다.
 
![](../images/broadcast.png)

```
"""
src/broadcast.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
# device를 setting하면 이후에 rank에 맞는 디바이스에 접근 가능합니다.

if rank == 0:
    tensor = torch.randn(2, 2).to(torch.cuda.current_device())
else:
    tensor = torch.zeros(2, 2).to(torch.cuda.current_device())

print(f"before rank {rank}: {tensor}\n")
dist.broadcast(tensor, src=0)
print(f"after rank {rank}: {tensor}\n")
```

Collective Communication 코드를 실행하기 위해서 뉴론 시스템에서 GPU 4개를 할당받는다
```
# [glogin01]$ salloc --partition=amd_a100nv_8 -J debug --nodes=1 --time=8:00:00 --gres=gpu:4 --comment pytorch
[glogin01]$ salloc --partition=cas_v100_4 -J debug --nodes=2 --time=12:00:00 --gres=gpu:2 --comment pytorch
salloc: Pending job allocation 340482
salloc: job 340482 queued and waiting for resources
salloc: job 340482 has been allocated resources
salloc: Granted job allocation 340482
salloc: Waiting for resource configuration
salloc: Nodes gpu[10,17] are ready for job
```

```
[gpu10]$ pwd
/scratch/qualis/git-projects/large-scale-lm-tutorials/src/ch2
[gpu10]$ conda activate large-scale-lm
(large-scale-lm) [gpu10]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.1
(large-scale-lm) [gpu10]$ srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu10:12345 broadcast.py
W0908 15:57:10.869000 47684370574400 torch/distributed/run.py:757]
W0908 15:57:10.869000 47684370574400 torch/distributed/run.py:757] *****************************************
W0908 15:57:10.869000 47684370574400 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable
for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for
optimal performance in your application as needed.
W0908 15:57:10.869000 47684370574400 torch/distributed/run.py:757] *****************************************
W0908 15:57:11.077000 47206882413632 torch/distributed/run.py:757]
W0908 15:57:11.077000 47206882413632 torch/distributed/run.py:757] *****************************************
W0908 15:57:11.077000 47206882413632 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable
for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for
optimal performance in your application as needed.
W0908 15:57:11.077000 47206882413632 torch/distributed/run.py:757] *****************************************
before rank 1: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:1')

before rank 0: tensor([[-1.1589, -1.0418, -0.2843,  ...,  0.1931, -2.5755, -0.0232],
        [ 1.5140, -0.9782, -0.3955,  ...,  0.2272,  1.4831,  0.7004],
        [ 0.6604,  0.0533, -2.1374,  ..., -0.7254,  0.1704, -2.5956],
        ...,
        [ 0.9011,  1.0888,  0.0632,  ...,  0.8425, -0.6508,  0.4218],
        [ 0.5068, -0.3434, -0.2022,  ..., -0.0231, -0.1183, -1.1983],
        [ 0.3403, -1.5891,  0.4225,  ..., -0.1733,  1.2176,  1.0935]],
       device='cuda:0')

before rank 3: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:1')

before rank 2: tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0')

after rank 0: tensor([[-1.1589, -1.0418, -0.2843,  ...,  0.1931, -2.5755, -0.0232],
        [ 1.5140, -0.9782, -0.3955,  ...,  0.2272,  1.4831,  0.7004],
        [ 0.6604,  0.0533, -2.1374,  ..., -0.7254,  0.1704, -2.5956],
        ...,
        [ 0.9011,  1.0888,  0.0632,  ...,  0.8425, -0.6508,  0.4218],
        [ 0.5068, -0.3434, -0.2022,  ..., -0.0231, -0.1183, -1.1983],
        [ 0.3403, -1.5891,  0.4225,  ..., -0.1733,  1.2176,  1.0935]],
       device='cuda:0')

after rank 1: tensor([[-1.1589, -1.0418, -0.2843,  ...,  0.1931, -2.5755, -0.0232],
        [ 1.5140, -0.9782, -0.3955,  ...,  0.2272,  1.4831,  0.7004],
        [ 0.6604,  0.0533, -2.1374,  ..., -0.7254,  0.1704, -2.5956],
        ...,
        [ 0.9011,  1.0888,  0.0632,  ...,  0.8425, -0.6508,  0.4218],
        [ 0.5068, -0.3434, -0.2022,  ..., -0.0231, -0.1183, -1.1983],
        [ 0.3403, -1.5891,  0.4225,  ..., -0.1733,  1.2176,  1.0935]],
       device='cuda:1')

after rank 2: tensor([[-1.1589, -1.0418, -0.2843,  ...,  0.1931, -2.5755, -0.0232],
        [ 1.5140, -0.9782, -0.3955,  ...,  0.2272,  1.4831,  0.7004],
        [ 0.6604,  0.0533, -2.1374,  ..., -0.7254,  0.1704, -2.5956],
        ...,
        [ 0.9011,  1.0888,  0.0632,  ...,  0.8425, -0.6508,  0.4218],
        [ 0.5068, -0.3434, -0.2022,  ..., -0.0231, -0.1183, -1.1983],
        [ 0.3403, -1.5891,  0.4225,  ..., -0.1733,  1.2176,  1.0935]],
       device='cuda:0')

after rank 3: tensor([[-1.1589, -1.0418, -0.2843,  ...,  0.1931, -2.5755, -0.0232],
        [ 1.5140, -0.9782, -0.3955,  ...,  0.2272,  1.4831,  0.7004],
        [ 0.6604,  0.0533, -2.1374,  ..., -0.7254,  0.1704, -2.5956],
        ...,
        [ 0.9011,  1.0888,  0.0632,  ...,  0.8425, -0.6508,  0.4218],
        [ 0.5068, -0.3434, -0.2022,  ..., -0.0231, -0.1183, -1.1983],
        [ 0.3403, -1.5891,  0.4225,  ..., -0.1733,  1.2176,  1.0935]],
       device='cuda:1')
```
`send`, `recv` 등의 P2P 연산이 지원되지 않을때 않아서 `broadcast`를 P2P 통신 용도로 사용하기도 합니다. src=0, dst=1 일때, `new_group([0, 1])` 그룹을 만들고 `broadcast`를 수행하면 0 -> 1 P2P와 동일합니다.

```
"""
참고: deepspeed/deepspeed/runtime/pipe/p2p.py
"""

def send(tensor, dest_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesnt support async_op true"
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    if async_op:
        global _async
        op = dist.isend(tensor, dest_rank)
        _async.append(op)
    else:

        if can_send_recv():
            return dist.send(tensor, dest_rank)
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            src_rank = _grid.stage_to_global(stage_id=src_stage)
            return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)
```
#### 2) Reduce
Reduce는 각 프로세스가 가진 데이터로 특정 연산을 수행해서 출력을 하나의 디바이스로 모아주는 연산입니다. 연산은 주로 sum, max, min 등이 가능합니다.
![](../images/reduce.png)
```
"""
src/reduce_sum.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.reduce(tensor, op=torch.distributed.ReduceOp.SUM, dst=0)

if rank == 0:
    print(tensor)
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/reduce_sum.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
tensor([[6., 6.],
        [6., 6.]], device='cuda:0')
```

```
"""
src/reduce_max.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.reduce(tensor, op=torch.distributed.ReduceOp.MAX, dst=0)

if rank == 0:
    print(tensor)
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/reduce_max.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
tensor([[3., 3.],
        [3., 3.]], device='cuda:0')
```
#### 3) Scatter
Scatter는 여러개의 element를 쪼개서 각 device에 뿌려주는 연산입니다.
![](../images/scatter.png)
```
"""
src/scatter.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
# nccl은 scatter를 지원하지 않습니다.
rank = dist.get_rank()
torch.cuda.set_device(rank)


output = torch.zeros(1)
print(f"before rank {rank}: {output}\n")

if rank == 0:
    inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
    inputs = torch.split(inputs, dim=0, split_size_or_sections=1)
    # (tensor([10]), tensor([20]), tensor([30]), tensor([40]))
    dist.scatter(output, scatter_list=list(inputs), src=0)
else:
    dist.scatter(output, src=0)

print(f"after rank {rank}: {output}\n")
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/scatter.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
before rank 0: tensor([0.])

before rank 3: tensor([0.])

after rank 3: tensor([40.])

before rank 1: tensor([0.])

before rank 2: tensor([0.])

after rank 0: tensor([10.])
after rank 1: tensor([20.])


after rank 2: tensor([30.])
```
nccl에서는 scatter가 지원되지 않기 때문에 아래와 같은 방법으로 scatter 연산을 수행합니다.
```
"""
src/scatter_nccl.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
inputs = torch.split(tensor=inputs, dim=-1, split_size_or_sections=1)
output = inputs[rank].contiguous().to(torch.cuda.current_device())
print(f"after rank {rank}: {output}\n")
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/scatter_nccl.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system beingoverloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
after rank 2: tensor([30.], device='cuda:2')

after rank 3: tensor([40.], device='cuda:3')

after rank 0: tensor([10.], device='cuda:0')

after rank 1: tensor([20.], device='cuda:1')
```
```
"""
참고: megatron-lm/megatron/mpu/mappings.py
"""

def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output

class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)
```
#### 4) Gather
Gather는 여러 디바이스에 존재하는 텐서를 하나로 모아주는 연산입니다.
![](../images/gather.png)
```
"""
src/gather.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
# nccl은 gather를 지원하지 않습니다.
rank = dist.get_rank()
torch.cuda.set_device(rank)

input = torch.ones(1) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

if rank == 0:
    outputs_list = [torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    dist.gather(input, gather_list=outputs_list, dst=0)
    print(outputs_list)
else:
    dist.gather(input, dst=0)
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/gather.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[tensor([0.]), tensor([1.]), tensor([2.]), tensor([3.])]
```
#### 5) All-reduce
이름 앞에 All- 이 붙은 연산들은 해당 연산을 수행 한뒤, 결과를 모든 디바이스로 broadcast하는 연산입니다. 아래 그림처럼 All-reduce는 reduce를 수행한 뒤, 계산된 결과를 모든 디바이스로 복사합니다.\n",
![](../images/allreduce.png)
```
"""
src/allreduce_sum.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

print(f"rank {rank}: {tensor}\n")
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/allreduce_sum.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
rank 1: tensor([[6., 6.],
        [6., 6.]], device='cuda:1')

rank 2: tensor([[6., 6.],
        [6., 6.]], device='cuda:2')
rank 0: tensor([[6., 6.],
        [6., 6.]], device='cuda:0')


rank 3: tensor([[6., 6.],
        [6., 6.]], device='cuda:3')

```
```
"""
src/allreduce_max.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]

dist.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)

print(f"rank {rank}: {tensor}\n")
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/allreduce_max.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
rank 3: tensor([[3., 3.],
        [3., 3.]], device='cuda:3')

rank 1: tensor([[3., 3.],
        [3., 3.]], device='cuda:1')

rank 2: tensor([[3., 3.],
        [3., 3.]], device='cuda:2')

rank 0: tensor([[3., 3.],
        [3., 3.]], device='cuda:0')
```

#### 6) All-gather
All-gather는 gather를 수행한 뒤, 모아진 결과를 모든 디바이스로 복사합니다.
![](../images/allgather.png)
```
"""
src/allgather.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

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

dist.all_gather(tensor_list=outputs_list, tensor=input)
print(outputs_list)
```
```
[globin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/allgather.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
[tensor([0.], device='cuda:1'), tensor([1.], device='cuda:1'), tensor([2.], device='cuda:1'), tensor([3.], device='cuda:1')]
[tensor([0.], device='cuda:0'), tensor([1.], device='cuda:0'), tensor([2.], device='cuda:0'), tensor([3.], device='cuda:0')]
[tensor([0.], device='cuda:2'), tensor([1.], device='cuda:2'), tensor([2.], device='cuda:2'), tensor([3.], device='cuda:2')]
[tensor([0.], device='cuda:3'), tensor([1.], device='cuda:3'), tensor([2.], device='cuda:3'), tensor([3.], device='cuda:3')]
```
#### 7) Reduce-scatter
Reduce scatter는 Reduce를 수행한 뒤, 결과를 쪼개서 디바이스에 반환합니다.
![](../images/reduce_scatter.png)
```
"""
src/reduce_scatter.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

input_list = torch.tensor([1, 10, 100, 1000]).to(torch.cuda.current_device()) * rank
input_list = torch.split(input_list, dim=0, split_size_or_sections=1)
# rank==0 => [0, 00, 000, 0000]
# rank==1 => [1, 10, 100, 1000]
# rank==2 => [2, 20, 200, 2000]
# rank==3 => [3, 30, 300, 3000]

output = torch.tensor([0], device=torch.device(torch.cuda.current_device()),)

dist.reduce_scatter(
    output=output,
    input_list=list(input_list),
    op=torch.distributed.ReduceOp.SUM,
)

print(f"rank {rank}: {output}\n")
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/reduce_scatter.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
rank 0: tensor([6], device='cuda:0')
rank 2: tensor([600], device='cuda:2')


rank 1: tensor([60], device='cuda:1')

rank 3: tensor([6000], device='cuda:3')
```

#### 8) Barrier
Barrier는 프로세스를 동기화 하기 위해 사용됩니다. 먼저 barrier에 도착한 프로세스는 모든 프로세스가 해당 지점까지 실행되는 것을 기다립니다.
```
"""
src/barrier.py
"""
import time
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()

if rank == 0:
    seconds = 0
    while seconds <= 3:
        time.sleep(1)
        seconds += 1
        print(f"rank 0 - seconds: {seconds}\n")

print(f"rank {rank}: no-barrier\n")
dist.barrier()
print(f"rank {rank}: barrier\n")
```
```
[glogin01]$ python -m torch.distributed.launch --nproc_per_node=4 ../src/barrier.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
rank 2: no-barrier
rank 1: no-barrier
rank 3: no-barrier



rank 0 - seconds: 1

rank 0 - seconds: 2

rank 0 - seconds: 3

rank 0 - seconds: 4

rank 0: no-barrier

rank 0: barrier

rank 1: barrier

rank 3: barrier

rank 2: barrier

```
### 너무 많죠...? 😅
아래 4개의 기본 연산만 잘 기억해둬도 대부분 유추해서 사용할 수 있습니다.
![](../images/collective.png)
4가지 연산을 기반으로 아래의 사항들을 익혀두시면 됩니다.
- `all-reduce`, `all-gather`는 해당 연산을 수행하고 나서 `broadcast` 연산을 수행하는 것이라고 생각하면 됩니다.
- `reduce-scatter`는 말 그대로 `reduce` 연산의 결과를 `scatter (쪼개기)` 처리한다고 생각하면 됩니다.
- `barrier`는 영어 뜻 그대로 벽과 같은 것입니다. 먼저 도착한 프로세스들이 못 지나가게 벽처럼 막아두는 함수입니다.
