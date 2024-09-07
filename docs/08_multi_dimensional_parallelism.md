# Multi-dimensional Parallelism\n"
이번 세션에서는 Multi-dimensional Parallelism을 위해 사용되는 몇가지 개념과 실습을 진행해보도록 하겠습니다.

## 1. Multi-dimensional Parallelism
    
Multi-dimensional Parallelism (다차원 병렬화)는 지금까지 공부했던 다양한 병렬화 기법을 함께 사용하는 것입니다. 예를 들면 GPU가 0번부터 7번까지 8대가 있다면 2대는 Data parallelism, 2대는 Pipeline parallelism, 2대는 Tensor parallelism 등을 적용할 수 있습니다. 이때 몇개 차원으로 병렬화가 적용되는지에 따라 N차원 병렬화라고 불리며 방금의 예시는 (2x2x2)로 3차원 병렬화가 되겠죠.
    
![](../images/parallelism.png)
    
이렇게 다양한 병렬처리 기법을 동시에 적용하는 것은 매우 고난도의 기술이 필요합니다. 이번 세션에서는 이러한 병렬처리 기법을 동시에 다루는 방법에 대해 알아봅시다. 
   
## 2. MPU (Model Parallel Unit)

https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/initialize.py#L57
    
MPU는 Megatron-LM에서 제안된 개념으로 모델 병렬처리와 관련된 다양한 모듈들을 제공합니다. 특히 **MPU는 다차원 병렬화를 매우 손쉽게 수행할 수 있도록 프로세스 그룹을 자동으로 생성하고 관리해주는 기능**을 가지고 있습니다.
    
다차원 병렬화를 위한 프로세스 그룹 예시를 살펴봅시다. 

우리가 16개의 GPU를 가지고 있다고 가정해봅시다. 그리고 (Data:2 x Tensor:2 x Pipeline:4)의 차원으로 모델을 병렬화 한다고 해봅시다. 그러면 Tensor parallelism group은 8개, Data parallelism group은 8개, Pipeline parallelism group 4개가 생성됩니다. 즉, 전체 GPU의 수를 해당 병렬화에 할당된 차원의 수로 나눈 값 만큼의 프로세스가 생성됩니다.

- Tensor parallelism group은 다음과 같이 생성 될 수 있습니다.
  - `[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]`
  - 즉, 0번 gpu는 1번 gpu와 tensor parallel 통신을 수행할 수 있으며 2번 gpu는 3번 gpu와 통신 가능합니다.
   
- Data parallelism group은 다음과 같이 생성 될 수 있습니다.\n",
  - `[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]`\n",
  - 즉, 0번 gpu는 2번 gpu와 data parallel  통신을 수행할 수 있으며 1번 gpu는 3번 gpu와 통신 가능합니다.
   
- Pipeline parallelism group은 다음과 같이 생성 될 수 있습니다.
  - `[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]`
  - Forward 시 0 → 4 → 8 → 12번 방향으로, Backward시 12 → 8 → 4 → 0번 방향으로 통신이 수행됩니다.
   
꽤나 복잡하지만 프로세스가 생성되는 순서를 외우거나 할 필요는 없습니다. MPU 객체가 알아서 이를 처리합니다. 이를 그림으로 나타내면 아래와 같습니다.
   
 ```
               +---------+  +---------+  +---------+  +---------+
       tensor  |   g00   |  |   g04   |  |   g08   |  |   g12   |
 data          +---------+  +---------+  +---------+  +---------+ ===> forward\n",
       tensor  |   g01   |  |   g05   |  |   g09   |  |   g13   |
               +---------+  +---------+  +---------+  +---------+
                pipeline     pipeline     pipeline     pipeline
    
               +---------+  +---------+  +---------+  +---------+
       tensor  |   g02   |  |   g06   |  |   g10   |  |   g14   |
 data          +---------+  +---------+  +---------+  +---------+ ===> forward\n",
       tensor  |   g03   |  |   g07   |  |   g11   |  |   g15   |
               +---------+  +---------+  +---------+  +---------+
                 pipeline     pipeline     pipeline     pipeline
```
    
3개의 차원으로 분할되었으니 아래와 같이 3D와 같은 육면체를 구성할 수도 있습니다.
    
```
                        [g02, g06, g10, g14]
                      /  |              /  |
                     [g00, g04, g08, g12]  |
                     |   |             |   |
        3D parallel  |  [g03, g07, g11, g15]
                     |  /              |  /
                     [g01, g05, g09, g13]
```

MPU를 직접 구현해서 사용해보겠습니다. 이 구현은 Megatron-LM의 코드를 기반으로 제가 변경한 것입니다.
```
import os
import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.function import Function


class MPU(object):
    """
    MPU: Model Parallel Unit

    Notes:
        Let's say we have a total of 16 GPUs denoted g0 ... g15 and we use 2 GPUs to parallelize the model tensor,
        and 4 GPUs to parallelize the model pipeline. The present method will create 8 tensor model-parallel group,
        4 pipeline model parallel groups and 8 data parallel groups as:

        - width: 4 pipeline parallel group
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
        - height: 8 tensor parallel group
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        - depth: 8 data parallel group
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]

                        [g02, g06, g10, g14]
                      /  |              /  |
                     [g00, g04, g08, g12]  |
                     |   |             |   |
        3D parallel  |  [g03, g07, g11, g15]
                     |  /              |  /
                     [g01, g05, g09, g13]

                      +---------+  +---------+  +---------+  +---------+
              tensor  |   g00   |  |   g04   |  |   g08   |  |   g14   |
        data          +---------+  +---------+  +---------+  +---------+ ===> forward
              tensor  |   g01   |  |   g05   |  |   g09   |  |   g13   |
                      +---------+  +---------+  +---------+  +---------+
                        pipeline     pipeline     pipeline     pipeline

                      +---------+  +---------+  +---------+  +---------+
              tensor  |   g02   |  |   g06   |  |   g10   |  |   g12   |
        data          +---------+  +---------+  +---------+  +---------+ ===> forward
              tensor  |   g03   |  |   g07   |  |   g11   |  |   g15   |
                      +---------+  +---------+  +---------+  +---------+
                        pipeline     pipeline     pipeline     pipeline

    References:
        Original MPU implementation of Megatron-LM.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/initialize.py

    """

    _tensor_model_parallel_group = None
    _pipeline_model_parallel_group = None
    _data_parallel_group = None

    _tensor_model_parallel_world_size = None
    _pipeline_model_parallel_world_size = None
    _data_parallel_world_size = None

    _tensor_model_parallel_rank = None
    _pipeline_model_parallel_rank = None
    _pipeline_global_ranks = None

    def __init__(
        self,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        backend: str,
        master_port: int,
    ) -> None:
        """
        Initialize MPU object. All process groups are initialized in this method.

        Args:
            tensor_model_parallel_size (int): tensor model parallel world size
            pipeline_model_parallel_size (int): pipeline model parallel world size
        """

        if not dist.is_initialized():
            self.initialize_distributed(backend, master_port)

        current_rank = dist.get_rank()
        global_world_size = dist.get_world_size()

        assert (
            global_world_size >= tensor_model_parallel_size
        ), "param `tensor_model_parallel_size` must be smaller than global world size."

        assert (
            global_world_size >= pipeline_model_parallel_size
        ), "param `pipeline_model_parallel_size` must be smaller than global world size."

        total_model_parallel_size = (
            tensor_model_parallel_size * pipeline_model_parallel_size
        )

        assert (
            global_world_size % total_model_parallel_size == 0
        ), "global world sizes must be divisible by model parallel world sizes (tp * pp)"

        num_tensor_model_parallel_groups = (
            global_world_size // tensor_model_parallel_size
        )

        num_pipeline_model_parallel_groups = (
            global_world_size // pipeline_model_parallel_size
        )

        # 1. initialize data parallel group
        self._initialize_data_parallel_group(
            current_rank=current_rank,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            num_pipeline_model_parallel_groups=num_pipeline_model_parallel_groups,
        )

        # 2. initialize tensor model parallel group
        self._initialize_tensor_model_parallel_group(
            current_rank=current_rank,
            tensor_model_parallel_size=tensor_model_parallel_size,
            num_tensor_model_parallel_groups=num_tensor_model_parallel_groups,
        )

        # 3. initialize pipeline model parallel group
        self._initialize_pipeline_model_parallel_group(
            current_rank=current_rank,
            global_world_size=global_world_size,
            num_pipeline_model_parallel_groups=num_pipeline_model_parallel_groups,
        )

        # 4. create distributed functions
        functions = self._initialize_functions()
        self._broadcast_fn = functions["broadcast"]
        self._reduce_fn = functions["reduce"]
        self._scatter_fn = functions["scatter"]
        self._gather_fn = functions["gather"]

    def _initialize_data_parallel_group(
        self,
        current_rank: int,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        num_pipeline_model_parallel_groups: int,
    ) -> None:
        """
        Initialize data parallel group

        Args:
            current_rank (int): current rank
            tensor_model_parallel_size (int): tensor model parallel world size
            pipeline_model_parallel_size (int): pipeline model parallel world size
            num_pipeline_model_parallel_groups (int): the number of pipeline model parallel groups
        """
        assert (
            self._data_parallel_group is None
        ), "data parallel group is already initialized."

        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups

            for j in range(tensor_model_parallel_size):
                ranks = list(
                    range(start_rank + j, end_rank, tensor_model_parallel_size)
                )

                group = dist.new_group(ranks)
                if current_rank in ranks:
                    self._data_parallel_group = group

    def _initialize_tensor_model_parallel_group(
        self,
        current_rank: int,
        tensor_model_parallel_size: int,
        num_tensor_model_parallel_groups: int,
    ) -> None:
        """
        Initialize tensor model parallel group

        Args:
            current_rank (int): current rank
            tensor_model_parallel_size (int): tensor model parallel world size
            num_tensor_model_parallel_groups (int): the number of tensor model parallel groups
        """
        assert (
            self._tensor_model_parallel_group is None
        ), "tensor model parallel group is already initialized."

        for i in range(num_tensor_model_parallel_groups):
            start_rank = i * tensor_model_parallel_size
            end_rank = (i + 1) * tensor_model_parallel_size

            ranks = list(range(start_rank, end_rank))
            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._tensor_model_parallel_group = group

    def _initialize_pipeline_model_parallel_group(
        self,
        current_rank: int,
        global_world_size: int,
        num_pipeline_model_parallel_groups: int,
    ) -> None:
        """
        Initialize pipeline model parallel group

        Args:
            current_rank (int): current rank
            global_world_size (int): global world size
            num_pipeline_model_parallel_groups (int): the number of model parallel groups
        """
        assert (
            self._pipeline_model_parallel_group is None
        ), "pipeline model parallel group is already initialized."

        for i in range(num_pipeline_model_parallel_groups):
            ranks = list(
                range(i, global_world_size, num_pipeline_model_parallel_groups)
            )

            group = dist.new_group(ranks)

            if current_rank in ranks:
                self._pipeline_model_parallel_group = group
                self._pipeline_global_ranks = ranks

    def model_parallel_is_initialized(self) -> bool:
        """
        Check if model and data parallel groups are initialized.

        Returns:
            bool: whether MPU is initialized
        """
        if (
            self._tensor_model_parallel_group is None
            or self._pipeline_model_parallel_group is None
            or self._data_parallel_group is None
        ):
            return False
        return True

    def get_model_parallel_group(self):
        """
        Get the tensor model parallel group.

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_group()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            ProcessGroup: tensor model parallel group
        """
        return self.get_tensor_model_parallel_group()

    def get_model_parallel_world_size(self) -> int:
        """
        Get the tensor model parallel world size

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_world_size()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            int: tensor model parallel world size
        """
        return self.get_tensor_model_parallel_world_size()

    def get_model_parallel_rank(self) -> int:
        """
        Get the tensor model parallel rank

        Notes:
            This method existed in the old version of Megatron-LM. It is the same as `get_tensor_model_parallel_rank()`,
            But we must support backward compatibility because this method is invoked by libraries such as DeepSpeed.

        Returns:
            int: tensor model parallel world size
        """
        return self.get_tensor_model_parallel_rank()

    def get_tensor_model_parallel_group(self):
        """
        Get tensor model parallel group

        Returns:
            ProcessGroup: tensor model parallel group
        """

        assert (
            self._tensor_model_parallel_group is not None
        ), "tensor model parallel group is not initialized."

        return self._tensor_model_parallel_group

    def get_pipeline_model_parallel_group(self):
        """
        Get pipeline model parallel group

        Returns:
            ProcessGroup: pipeline model parallel group
        """
        assert (
            self._pipeline_model_parallel_group is not None
        ), "pipeline model parallel group is not initialized."

        return self._pipeline_model_parallel_group

    def get_data_parallel_group(self):
        assert (
            self._data_parallel_group is not None
        ), "data parallel group is not initialized."

        return self._data_parallel_group

    def get_tensor_model_parallel_world_size(self) -> int:
        """
        Get tensor model parallel world size

        Returns:
            int: tensor model parallel world size
        """
        if self._tensor_model_parallel_world_size is not None:
            return self._tensor_model_parallel_world_size

        return dist.get_world_size(self.get_tensor_model_parallel_group())

    def set_tensor_model_parallel_world_size(self, world_size: int) -> None:
        """
        Set tensor model parallel world size

        Args:
            world_size (int): tensor model parallel world size
        """
        self._tensor_model_parallel_world_size = world_size

    def get_pipeline_model_parallel_world_size(self) -> int:
        """
        Get pipeline model parallel world size

        Returns:
            int: pipeline model parallel world size
        """
        if self._pipeline_model_parallel_world_size is not None:
            return self._pipeline_model_parallel_world_size

        return dist.get_world_size(self.get_pipeline_model_parallel_group())

    def set_pipeline_model_parallel_world_size(self, world_size: int) -> None:
        """
        Set pipeline model parallel world size

        Args:
            world_size (int): pipeline model parallel world size
        """
        self._pipeline_model_parallel_world_size = world_size

    def get_tensor_model_parallel_rank(self) -> int:
        """
        Get tensor model parallel rank

        Returns:
            int: tensor model parallel rank
        """
        if self._tensor_model_parallel_rank is not None:
            return self._tensor_model_parallel_rank

        return dist.get_rank(self.get_tensor_model_parallel_group())

    def set_tensor_model_parallel_rank(self, rank: int) -> None:
        """
        Set tensor model parallel rank

        Args:
            rank (int): tensor model parallel rank
        """

        self._tensor_model_parallel_rank = rank

    def get_pipeline_model_parallel_rank(self) -> int:
        """
        Get pipeline model parallel rank

        Returns:
            int: pipeline model parallel rank
        """
        if self._pipeline_model_parallel_rank is not None:
            return self._pipeline_model_parallel_rank

        return dist.get_rank(self.get_pipeline_model_parallel_group())

    def set_pipeline_model_parallel_rank(self, rank: int) -> None:
        """
        Set pipeline model parallel rank

        Args:
            rank (int): pipeline model parallel rank
        """

        self._pipeline_model_parallel_rank = rank

    def is_pipeline_fist_stage(self) -> bool:
        """
        Return `True` if in the first pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is first
        """
        return self.get_pipeline_model_parallel_rank() == 0

    def is_pipeline_last_stage(self) -> bool:
        """
        Return `True` if in the last pipeline model parallel stage, `False` otherwise

        Returns:
            bool: whether current pipeline model parallel stage is last
        """
        return self.get_pipeline_model_parallel_rank() == (
            self.get_pipeline_model_parallel_world_size() - 1
        )

    def get_tensor_model_parallel_src_rank(self) -> int:
        """
        Calculate the global rank corresponding to the first local rank in the tensor model parallel group.

        Returns:
            int: tensor model parallel source rank
        """

        global_rank = dist.get_rank()
        local_world_size = self.get_tensor_model_parallel_world_size()
        return (global_rank // local_world_size) * local_world_size

    def get_pipeline_model_parallel_fist_rank(self):
        """
        Get the first pipeline model parallel rank

        Returns:
            int: the first pipeline model parallel rank
        """
        return self._pipeline_global_ranks[0]

    def get_pipeline_model_parallel_last_rank(self):
        """
        Get the last pipeline model parallel rank

        Returns:
            int: the last pipeline model parallel rank
        """
        return self._pipeline_global_ranks[
            self.get_pipeline_model_parallel_world_size() - 1
        ]

    def get_pipeline_model_parallel_next_rank(self) -> int:
        """
        Get the next pipeline model parallel rank comparison with current stage.

        Returns:
            int: the next pipeline model parallel rank
        """
        assert (
            self._pipeline_global_ranks is not None
        ), "pipeline model parallel group is not initialized."

        rank_in_pipe = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self._pipeline_global_ranks[(rank_in_pipe + 1) % world_size]

    def get_pipeline_model_parallel_prev_rank(self) -> int:
        """
        Get the previous pipeline model parallel rank comparison with current stage.

        Returns:
            int: the previous pipeline model parallel rank
        """
        assert (
            self._pipeline_global_ranks is not None
        ), "pipeline model parallel group is not initialized."

        rank_in_pipe = self.get_pipeline_model_parallel_rank()
        world_size = self.get_pipeline_model_parallel_world_size()
        return self._pipeline_global_ranks[(rank_in_pipe - 1) % world_size]

    def get_data_parallel_world_size(self) -> int:
        """
        Get data parallel world size

        Returns:
            int: data parallel world size
        """

        return dist.get_world_size(self.get_data_parallel_group())

    def get_data_parallel_rank(self) -> int:
        """
        Get data parallel rank

        Returns:
            int: data parallel rank
        """
        return dist.get_rank(self.get_data_parallel_group())

    def destroy_model_parallel(self) -> None:
        """
        Destroy all the model parallel groups
        """

        self._tensor_model_parallel_group = None
        self._pipeline_model_parallel_group = None
        self._data_parallel_group = None

    def _broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the model parallel region.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: broadcast tensor
        """
        return inputs.clone()

    def _reduce(self, inputs: Tensor):
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """
        if self.get_tensor_model_parallel_world_size() == 1:
            return inputs

        dist.all_reduce(inputs, group=self.get_tensor_model_parallel_group())
        return inputs

    def _scatter(self, inputs: Tensor) -> Tensor:
        """
        Split the tensor along its last dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: scattered tensor
        """
        world_size = self.get_tensor_model_parallel_world_size()

        if world_size == 1:
            return inputs

        last_dim = inputs.dim() - 1
        last_dim_size = inputs.size()[last_dim] // world_size

        inputs_list = torch.split(
            tensor=inputs,
            split_size_or_sections=last_dim_size,
            dim=last_dim,
        )

        rank = self.get_tensor_model_parallel_rank()
        outputs = inputs_list[rank].contiguous()
        return outputs

    def _gather(self, inputs: Tensor) -> Tensor:
        """
        Gather tensors and concatenate along the last dimension

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: gathered tensor
        """
        world_size = self.get_tensor_model_parallel_world_size()

        if world_size == 1:
            return inputs

        last_dim = inputs.dim() - 1
        rank = self.get_tensor_model_parallel_rank()

        tensor_list = [torch.empty_like(inputs) for _ in range(world_size)]
        tensor_list[rank] = inputs
        torch.distributed.all_gather(
            tensor_list, inputs, group=self.get_tensor_model_parallel_group()
        )
        outputs = torch.cat(tensor_list, dim=last_dim).contiguous()
        return outputs

    def broadcast(self, inputs: Tensor) -> Tensor:
        """
        Pass the input to the model parallel region.

        Args:
            inputs (Tensor):

        Returns:
            Tensor: broadcast tensor
        """

        if self._enable_grad(inputs):
            outputs = self._broadcast_fn.apply(inputs)
        else:
            outputs = self._broadcast(inputs)
        return outputs

    def reduce(self, inputs: Tensor) -> Tensor:
        """
        All-reduce the input tensor across tensor model parallel group.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: all-reduced tensor
        """

        if self._enable_grad(inputs):
            outputs = self._reduce_fn.apply(inputs)
        else:
            outputs = self._reduce(inputs)
        return outputs

    def scatter(self, inputs: Tensor) -> Tensor:
        """
        Split the tensor along its last dimension and keep the corresponding slice.

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: scattered tensor
        """

        if self._enable_grad(inputs):
            outputs = self._scatter_fn.apply(inputs)
        else:
            outputs = self._scatter(inputs)
        return outputs

    def gather(self, inputs: Tensor) -> Tensor:
        """
        Gather tensors and concatenate along the last dimension

        Args:
            inputs (Tensor): input tensor

        Returns:
            Tensor: gathered tensor
        """

        if self._enable_grad(inputs):
            outputs = self._gather_fn.apply(inputs)
        else:
            outputs = self._gather(inputs)
        return outputs

    @staticmethod
    def _enable_grad(inputs: Tensor) -> bool:
        """
        Check current tensor is enabled to pass gradient.

        Args:
            inputs (Tensor): input tensor

        Returns:
            bool: whether gradient can be passed or not
        """
        return torch.is_grad_enabled() and inputs.requires_grad

    def _initialize_functions(self):
        class Broadcast(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._broadcast(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._reduce(inputs)

        class Reduce(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._reduce(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._broadcast(inputs)

        class Scatter(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._scatter(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._gather(inputs)

        class Gather(Function):
            @staticmethod
            def forward(ctx, inputs):
                return self._gather(inputs)

            @staticmethod
            def backward(ctx, inputs):
                return self._scatter(inputs)

        return {
            "broadcast": Broadcast,
            "reduce": Reduce,
            "scatter": Scatter,
            "gather": Gather,
        }

    @staticmethod
    def initialize_distributed(backend, master_port):
        """Initialize torch.distributed and mpu."""
        if not torch.distributed.is_initialized():
            rank = int(os.getenv("RANK", 0))
            world_size = int(os.getenv("WORLD_SIZE", 1))
            os.environ["MASTER_PORT"] = str(master_port)
            device_count = torch.cuda.device_count()

            if device_count > 0:
                device = rank % device_count
                torch.cuda.set_device(device)

            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
            )
```

```
"""
src/test_mpu.py
"""
import torch
import torch.distributed as dist
from mpu import MPU

mpu = MPU(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    backend="nccl",
    master_port=5678,
)

# 1. MPU는 다음과 같이 프로세스 그룹을 자동으로 생성하고 그들에 접근할 수 있는 메서드를 제공합니다.
print(f"TP group: {mpu.get_tensor_model_parallel_group()}")
print(f"TP wsz: {mpu.get_tensor_model_parallel_world_size()}")
print(f"TP rank: {mpu.get_tensor_model_parallel_rank()}")
dist.barrier()
print("\n")

print(f"PP group: {mpu.get_pipeline_model_parallel_group()}")
print(f"PP wsz: {mpu.get_pipeline_model_parallel_world_size()}")
print(f"PP rank: {mpu.get_pipeline_model_parallel_rank()}")
dist.barrier()
print("\n")

# 2. Data parallel size는 TP와 PP 사이즈에 맞게 알아서 설정됩니다.
# 만약 16대의 GPU에서 TP=4, PP=1을 설정했다면 16 / (4 * 1) = 4로 자동으로 DP size는 4가 됩니다.
print(f"DP group: {mpu.get_data_parallel_group()}")
print(f"DP wsz: {mpu.get_data_parallel_world_size()}")
print(f"DP rank: {mpu.get_data_parallel_rank()}")
dist.barrier()
print("\n")

# 3. MPU는 reduce, scatter, gather, braodcast 등의 연산을 지원합니다.
# 이들은 대부분 Tensor parallel group에서만 사용되기 때문에 Tensor parallel group을 기본으로 설장해뒀습니다.
a = torch.tensor([2, 3, 4, 5]).cuda() * dist.get_rank()
a = mpu.reduce(a)
print(a)
```

```
[glogin01]$ deepspeed --num_gpus=4 ../src/test_mpu.py
[2021-10-28 00:29:04,099] [WARNING] [runner.py:122:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2021-10-28 00:29:04,275] [INFO] [runner.py:360:main] cmd = /usr/bin/python3 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 ../src/test_mpu.py
[2021-10-28 00:29:05,352] [INFO] [launch.py:80:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2021-10-28 00:29:05,352] [INFO] [launch.py:89:main] nnodes=1, num_local_procs=4, node_rank=0
[2021-10-28 00:29:05,352] [INFO] [launch.py:101:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2021-10-28 00:29:05,352] [INFO] [launch.py:102:main] dist_world_size=4
[2021-10-28 00:29:05,352] [INFO] [launch.py:105:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
0: TP group: <torch.distributed.ProcessGroupNCCL object at 0x7f29d30bed50>
3: TP group: <torch.distributed.ProcessGroupNCCL object at 0x7fe4dcd0fd50>
2: TP group: <torch.distributed.ProcessGroupNCCL object at 0x7fafec84ed50>
1: TP group: <torch.distributed.ProcessGroupNCCL object at 0x7fa4dbbeed50>
0: TP wsz: 2
3: TP wsz: 2
2: TP wsz: 2
1: TP wsz: 2
0: TP rank: 0
3: TP rank: 1
2: TP rank: 0
1: TP rank: 1








0: PP group: <torch.distributed.ProcessGroupNCCL object at 0x7f29d30bedc0>
3: PP group: <torch.distributed.ProcessGroupNCCL object at 0x7fe4dcd0fdc0>
2: PP group: <torch.distributed.ProcessGroupNCCL object at 0x7fafec84edc0>
1: PP group: <torch.distributed.ProcessGroupNCCL object at 0x7fa4dbbeedc0>
0: PP wsz: 2
3: PP wsz: 2
2: PP wsz: 2
1: PP wsz: 2
3: PP rank: 1
0: PP rank: 0
2: PP rank: 1
1: PP rank: 0








0: DP group: <torch.distributed.ProcessGroupNCCL object at 0x7f29d30bece0>
1: DP group: <torch.distributed.ProcessGroupNCCL object at 0x7fa4dbbeece0>
2: DP group: <torch.distributed.ProcessGroupNCCL object at 0x7fafec84ece0>
3: DP group: <torch.distributed.ProcessGroupNCCL object at 0x7fe4dcd0fce0>
0: DP wsz: 1
1: DP wsz: 1
2: DP wsz: 1
3: DP wsz: 1
0: DP rank: 0
1: DP rank: 0
2: DP rank: 0
3: DP rank: 0








1: all-reduce => tensor([2, 3, 4, 5], device='cuda:1')
0: all-reduce => tensor([2, 3, 4, 5], device='cuda:0')
3: all-reduce => tensor([10, 15, 20, 25], device='cuda:3')
2: all-reduce => tensor([10, 15, 20, 25], device='cuda:2')
```

## 3. Large-scale 프로젝트 소개\n",
 
현재 거의 모든 빅모델 관련 프로젝트는 Megatron-LM을 기반으로 하고 있습니다. Tensor parallelism 세션에서 이미 Megatron-LM의 실습을 진행했기 때문에 이번 세션에서는 따로 실습을 진행하지는 않겠습니다. 지금부터 소개 드릴 3개의 프로젝트 모두 Megatron-LM의 Fork 레포지토리이기 때문에 사용법이 거의 동일하고 Argument 등만 변경된 것이 대부분입니다. 각각 프로젝트 소개를 읽어보시고 적합하다 싶은 프로젝트의 레포를 클론하셔서 사용하시길 바랍니다.

### 1) Megatron-DeepSpeed
   
- Maintainer: Shaden, Jeff, ... DeepSpeed Team
- [Megatron 최신버전 + DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed): Upstream 유지됨
- [3D parallelism (Megatron 1.1.5) + ZeRO-1](https://github.com/microsoft/DeepSpeedExamples/tree/c1b206c137bb028fc8124fd7d434c2da10efc033/Megatron-LM-v1.1.5-3D_parallelism/megatron): Upstream 유지하지 않음
- [2D parallelism (Megatron 1.1.5) + ZeRO-3](https://github.com/microsoft/DeepSpeedExamples/tree/c1b206c137bb028fc8124fd7d434c2da10efc033/Megatron-LM-v1.1.5-ZeRO3): Upstream 유지하지 않음
    
Megatron-LM에 DeepSpeed ZeRO를 추가한 프로젝트입니다. 최근에 NVIDIA-Microsoft가 함께 개발했다고 알려진 530B의 Megatron-Turing의 코드 베이스로 예상됩니다. DeepSpeed **ZeRO 역시 다차원 병렬화의 대상이며 DP 대신 ZeRO-DP를 적용**할 수 있습니다. 그러나 ZeRO 2,3와 Pipeline 병렬화는 호환되지 않습니다. 따라서 **ZeRO 2,3를 사용하려면 Pipeline 병렬화를 적용하지 않아야** 합니다. 
    
따라서 (ZeRO-2,3 + TP) 혹은 (ZeRO-1 + TP + PP) 중 한가지를 선택을 해야 합니다. 만약 모델의 크기가 극도로 크다면 ZeRO-3 등을 고려해도 좋지만 현재까지는 3D Parallelism + ZeRO-1의 효율이 좋은 것으로 알려져있습니다. 그 이유는 Tensor parallelism이 노드내에서 통신할 때는 빠르지만 노드간 통신으로 가면 급격하게 느려지기 때문입니다. 
    
![](../images/megatron_3d.png)

대부분의 경우 위 그림처럼 노드내에서의 병렬화는 TP를 사용하고 노드간 통신은 PP를 사용하며 추가로 ZeRO-1을 적용해서 사용합니다. 그러나 ZeRO 논문에서 언급된 것 처럼 동일한 리소스를 사용할 때 학습 가능한 모델의 최대 크기는 ZeRO가 훨씬 크기 때문에 현재 리소스와 학습하고자 하는 모델 사이즈, 서버의 구성에 맞춰서 병렬화 전략을 잘 선택하시길 바랍니다.
  
### 2) GPT-NeoX
    
![](../images/eleuther_ai.png)
  
- Maintainer: Stella, ... EleutherAI Team
- https://github.com/EleutherAI/gpt-neox: Upstream 유지하지 않음
- https://github.com/EleutherAI/deeperspeed: Upstream 유지하지 않음
    
GPT-NeoX는 GPT-Neo로 유명한 EleutherAI 팀이 만들고 있는 GPU 버전 코드베이스입니다. GPT-NeoX는 **DeepSpeed Examples에 있는 [3D parallelism (Megatron 1.1.5) + ZeRO-1](https://github.com/microsoft/DeepSpeedExamples/tree/c1b206c137bb028fc8124fd7d434c2da10efc033/Megatron-LM-v1.1.5-3D_parallelism/megatron)에 여러가지 모델링 컴포넌트가 추가되어 있는 프로젝트**입니다. 예를 들어 ScaleNorm, RMSNorm, Rotary Embedding, Alibi Embedding, Shampoo Optimizer, SM3 Optimizer 등이 추가되어 있으며 DeepSpeed Sparse Attention 등을 지원하고 있습니다. 이들은 영어 모델을 개발하고 있으며 GPT-NeoX는 **GPT 모델만 구현되어 있으며** 다른 모델 구조 (Bert, T5, ...) 등은 제거되어 있습니다.

한가지 눈 여겨 볼 것은 현재 Megatron-LM과의 업스트림이 유지되지 않고 있으며 서로 꽤나 많이 달라졌습니다. Megatron-LM도 지속적으로 업데이트가 되고 있기 때문에 **업스트림이 유지되는지 유지되지 않는지도 참고할 사항**입니다. 또한 GPT-NeoX는 DeeperSpeed라는 DeepSpeed 포크 레포지토리를 개발해서 사용하고 있는데, 역시 업스트림 유지가 되고있지 않습니다. 
 
### 3) Big-Science\n",
    
![](../images/big_science.png)
- Maintainer: Stas, Wang, ... Hugging Face Team
- https://github.com/bigscience-workshop/Megatron-DeepSpeed: Upstream 유지됨
   
Big Science Workshop은 Huggingface에서 코드베이스 개발을 리드하고 프랑스 정부 산하의 컴퓨팅 장비를 사용하여 **멀티링구얼 빅모델**을 개발하고자 하는 1년간의 워크숍입니다. [Megatron 최신버전 + DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)의 Fork 레포지토리이며, 가장 최근 개발되기 시작했지만 참여인원이 많아서 가장 빠른속도로 개발되고 있습니다. 이들도 마찬가지로 ScaleNorm, RMSNorm, Rotary Embedding, Alibi Embedding 등의 **최근 성능이 우수하다고 알려진 컴포넌트들을 지속적으로 추가**하고 있으며 **GPT 모델 이외에 Bert, T5, Prefix-LM 등도 운용**하고 있습니다. Big-Science는 Megatron-LM의 업스트림을 유지하고 있다는 특징도 있습니다.
