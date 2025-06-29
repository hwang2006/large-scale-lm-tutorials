# Pipeline Parallelism
이번 세션에서는 파이프라인 병렬화에 대해 알아보겠습니다.
   
## 1. Inter-layer model parallelism
파이프라인 병렬화는 Inter-layer 모델 병렬화를 개선한 것입니다. Inter-layer 모델 병렬화는 아래와 같이 특정 GPU에 특정 레이어들을 할당하는 모델 병렬화 방법이였죠. 아래 그림에서는 GPU1번에 1,2,3번 레이어가 할당되었고, GPU2번에 4,5번 레이어가 할당 되었는데, 이 때 쪼개진 하나의 조각을 `stage(스테이지)`라고 합니다. 아래 예시의 경우 2개의 스테이지로 분할되었습니다.
  
![](../images/inter_layer.png)
   
그러나 이전 레이어의 출력을 다음 레이어의 입력으로 하는 신경망의 특성상 특정 GPU의 연산이 끝나야 다른 GPU가 연산을 시작할 수 있습니다. 즉, 아래의 그림처럼 Inter-layer 모델 병렬화는 동시에 하나의 GPU만 사용할 수 있다는 치명적인 한계를 가지고 있습니다.
    
![](../images/inter_layer_2.png)
![](../images/inter_layer_3.gif)

## 2. GPipe
GPipe는 Google에서 개발된 파이프라인 병렬화 기법으로 Inter Layer 모델 병렬화 시 GPU가 쉬는 시간 (idle time)을 줄이기 위해 등장했으며, mini-batch를 micro-batch로 한번 더 쪼개서 학습 과정을 파이프라이닝 하는 방식으로 동작합니다.
    
![](../images/gpipe_1.png)

![](../images/pipeline_parallelism2.png)
 
### Micro-batch
- Mini-batch는 전체 데이터셋을 n개로 분할한 서브샘플 집합입니다.
- Micro-batch는 Mini-batch를 m개로 한번 더 분할한 서브샘플 집합입니다.
    
![](../images/gpipe_2.png)
 
### Pipelining
GPipe는 미니배치를 마이크로 배치로 쪼개고 연산을 파이프라이닝 합니다. 붉은색 (GPU가 쉬는 부분)을 Bubble time이라고 하는데, Micro batch 사이즈가 커질 수록 Bubble time이 줄어드는 것을 알 수 있습니다.
   
![](../images/gpipe_3.gif)
 
### GPipe with PyTorch
kakaobrain에서 공개한 `torchgpipe`를 사용하면 손쉽게 GPipe를 사용할 수 있습니다. 단, `nn.Sequential`로 래핑된 모델만 사용 가능하며 모든 모듈의 입력과 출력 타입은 `torch.Tensor` 혹은 `Tuple[torch.Tensor]`로 제한됩니다. 따라서 코딩하기가 상당히 까다롭습니다.
```
"""
src/ch5/gpipe.org.py
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchgpipe import GPipe
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as GPT2BlockBase


class GPT2Preprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        position_ids = torch.arange(
            0, input_shape[-1], dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states


class GPT2Block(GPT2BlockBase):
    def forward(self, hidden_states):
        hidden_states = super(GPT2Block, self).forward(
            hidden_states=hidden_states,
        )
        return hidden_states[0]


class GPT2Postprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


def create_model_from_pretrained(model_name):
    pretrained = GPT2LMHeadModel.from_pretrained(model_name)
    preprocess = GPT2Preprocessing(pretrained.config)
    preprocess.wte.weight = pretrained.transformer.wte.weight
    preprocess.wpe.weight = pretrained.transformer.wpe.weight

    blocks = pretrained.transformer.h
    for i, block in enumerate(blocks):
        block.__class__ = GPT2Block

    postprocess = GPT2Postprocessing(pretrained.config)
    postprocess.ln_f.weight = pretrained.transformer.ln_f.weight
    postprocess.ln_f.bias = pretrained.transformer.ln_f.bias
    postprocess.lm_head.weight.data = pretrained.lm_head.weight.data.clone()

    return nn.Sequential(preprocess, *blocks, postprocess)


if __name__ == "__main__":
    world_size = 4

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = create_model_from_pretrained(model_name="gpt2")
    model = GPipe(
        model,
        balance=[4, 3, 3, 4],
        devices=[0, 1, 2, 3],
        chunks=world_size,
    )

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets]
    data_loader = DataLoader(datasets, batch_size=8, num_workers=8)

    optimizer = Adam(model.parameters(), lr=3e-5)
    loss_fn = nn.CrossEntropyLoss()

    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        tokens = tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        input_ids = tokens.input_ids.to(0)
        labels = tokens.input_ids.to(world_size - 1)

        lm_logits = model(input_ids)
        shift_logits = lm_logits[..., :-1, :].contiguous() # 마지막 토큰 제외, 마지막에서 첫번째 까지
        shift_labels = labels[..., 1:].contiguous() # 첫 토큰 제외, 두번째 토큰부터 마지막 토큰까지
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"step: {i}, loss: {loss}")
        if i == 300:
            break
```
```
(large-scale-lm) [gpu05]$ pip install torchgpipe
```
```
# (large-scale-lm) [gpu05]$ python -m torch.distributed.launch --nproc_per_node=4 gpipe.org.py
# (large-scale-lm) [gpu05]$ torchrun --nproc_per_node=4 gpipe.org.py
# (large-scale-lm) [gpu05]$ srun torchrun --nproc_per_node=4 gpipe.org.py
(large-scale-lm) [gpu05]$ python gpipe.org.py
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torchgpipe/stream.py:99: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  tensor = tensor.new_empty([0]).set_(tensor.storage())
step: 0, loss: 6.082091331481934
step: 10, loss: 3.249594211578369
step: 20, loss: 2.804326295852661
step: 30, loss: 2.5457987785339355
step: 40, loss: 2.852287530899048
step: 50, loss: 2.3473057746887207
step: 60, loss: 2.5315937995910645
step: 70, loss: 2.2457172870635986
step: 80, loss: 2.4787349700927734
step: 90, loss: 2.935778856277466
step: 100, loss: 2.8190064430236816
step: 110, loss: 2.4800403118133545
step: 120, loss: 2.951277494430542
step: 130, loss: 2.404855728149414
step: 140, loss: 2.9455151557922363
step: 150, loss: 3.9279768466949463
step: 160, loss: 3.024629592895508
step: 170, loss: 3.0173544883728027
step: 180, loss: 1.6787645816802979
step: 190, loss: 3.5459213256835938
step: 200, loss: 3.663504123687744
step: 210, loss: 3.522679328918457
step: 220, loss: 2.9959702491760254
step: 230, loss: 3.183525562286377
step: 240, loss: 2.5458672046661377
step: 250, loss: 3.148806571960449
step: 260, loss: 3.4613168239593506
step: 270, loss: 3.1852593421936035
step: 280, loss: 2.616974115371704
step: 290, loss: 2.1446080207824707
step: 300, loss: 3.436387300491333
```

```
"""
src/ch5/gpipe.py
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchgpipe import GPipe
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as GPT2BlockBase


class GPT2Preprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        position_ids = torch.arange(
            0, input_shape[-1], dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states


class GPT2Block(GPT2BlockBase):
    def forward(self, hidden_states):
        hidden_states = super(GPT2Block, self).forward(
            hidden_states=hidden_states,
        )
        return hidden_states[0]


class GPT2Postprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


def create_model_from_pretrained(model_name):
    pretrained = GPT2LMHeadModel.from_pretrained(model_name)
    preprocess = GPT2Preprocessing(pretrained.config)
    preprocess.wte.weight = pretrained.transformer.wte.weight
    preprocess.wpe.weight = pretrained.transformer.wpe.weight

    blocks = pretrained.transformer.h
    for i, block in enumerate(blocks):
        block.__class__ = GPT2Block

    postprocess = GPT2Postprocessing(pretrained.config)
    postprocess.ln_f.weight = pretrained.transformer.ln_f.weight
    postprocess.ln_f.bias = pretrained.transformer.ln_f.bias
    postprocess.lm_head.weight.data = pretrained.lm_head.weight.data.clone()

    return nn.Sequential(preprocess, *blocks, postprocess)


if __name__ == "__main__":
    world_size = 4

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = create_model_from_pretrained(model_name="gpt2")
    model = GPipe(
        model,
        #balance=[4,3,3,4],
        balance=[3,5,5,1],
        devices=[0, 1, 2, 3],
        chunks=world_size,
    )

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets]
    #data_loader = DataLoader(datasets, batch_size=8, num_workers=8)
    data_loader = DataLoader(datasets, batch_size=24, num_workers=8)

    optimizer = Adam(model.parameters(), lr=3e-5)
    loss_fn = nn.CrossEntropyLoss()

    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        tokens = tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        input_ids = tokens.input_ids.to(0)
        labels = tokens.input_ids.to(world_size - 1)

        lm_logits = model(input_ids)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"step: {i}, loss: {loss}")
        if i == 300:
             break
```
```
(large-scale-lm) [gpu05]$ python gpipe.py
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torchgpipe/stream.py:99: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  tensor = tensor.new_empty([0]).set_(tensor.storage())
step: 0, loss: 7.221696376800537
step: 10, loss: 2.02380633354187
step: 20, loss: 2.257195472717285
step: 30, loss: 2.9732449054718018
step: 40, loss: 3.133763551712036
step: 50, loss: 3.2365360260009766
step: 60, loss: 1.7914276123046875
step: 70, loss: 3.4165256023406982
step: 80, loss: 2.8382716178894043
step: 90, loss: 2.248094081878662
step: 100, loss: 2.6100006103515625
step: 110, loss: 1.4518338441848755
step: 120, loss: 2.2637827396392822
step: 130, loss: 2.457871913909912
step: 140, loss: 2.3267903327941895
step: 150, loss: 2.572256326675415
step: 160, loss: 2.491018056869507
step: 170, loss: 1.8535436391830444
step: 180, loss: 0.9895756244659424
step: 190, loss: 1.5809266567230225
step: 200, loss: 2.164494276046753
step: 210, loss: 2.1609110832214355
step: 220, loss: 2.5104143619537354
step: 230, loss: 2.2100632190704346
step: 240, loss: 2.479640483856201
step: 250, loss: 2.0741147994995117
step: 260, loss: 2.7896156311035156
step: 270, loss: 3.0486862659454346
step: 280, loss: 1.6412955522537231
step: 290, loss: 2.4288651943206787
step: 300, loss: 2.189744234085083
```

## 3. 1F1B Pipelining (PipeDream)
Microsoft에서 공개한 `PipeDream`은 `GPipe`와는 약간 다른 방식의 파이프라이닝을 수행합니다. 흔히 이 방법을 1F1B라고 부르는데, 모든 Forward가 끝나고 나서 Backward를 수행하는 GPipe와 달리 `PipeDream`은 Forward와 Backward를 번갈아가면서 수행합니다.
   
![](../images/1f1b.png)

1F1B Pipelining에는 다음과 같은 두가지 챌린지가 존재합니다.
1. Weight version managing
2. Work partitioning
 
### 1) Weight version managinig
GPipe의 경우 하나의 weight 버전만 운용하지만 주기적으로 Pipeline flush가 일어납니다. **Pipeline flush란 계산된 Gradient를 통해 파라미터를 업데이트(`optimizer.step()`) 하는 과정**입니다. 이러한 flush 과정 중에는 어떠한 **forward, backward 연산도 하지 않기** 때문에 처리 효율이 떨어집니다.

![](../images/pipeline_flush.png)
   
PipeDream은 이러한 flush 없이 계속해서 파라미터를 업데이트 해나갑니다. 따라서 forward와 backward가 모두 쉬는 시간이 사라집니다. 그러나 이를 위해서는 여러 버전의 파라미터 상태를 지속적으로 관리해야 합니다. 만약 최신버전의 파라미터만 저장하고 있으면 이전 layer의 출력이 다음 layer로 전송될 때, 다음 layer 부분이 업데이트 될 수도 있기 때문이죠.

![](../images/1f1b.gif)

이러한 문제를 막기 위해 여러 버전의 weight를 저장하여 관리하는데 그러면 weight를 저장하면 메모리 공간을 많이 차지하게 됩니다. 따라서 이 부분에서 트레이드 오프가 발생합니다.
- GPipe: 메모리 효율적, 프로세싱 비효율적
- PipeDream: 메모리 비효율적, 프로세싱 효율적
    
### 2) Work Partitioning
두번쨰 문제는 뉴럴넷을 어떻게 쪼갤건지에 대한 문제입니다. 단순히 Layer별로 동일한 수의 레이어를 갖게끔 하는 것이 항상 최고의 솔루션이라고 할 수는 없겠죠. 우리에게 가장 중요한 것은 idle time을 최소화을 최소화 하는 것입니다. 그러기 위해서는 각 파티션의 running time이 비슷해야겠죠. 그 이외에도 추가로 parameter size, activation memory 등을 고려해야 합니다.

![](../images/pipe_dream.png)
    
PipeDream은 Profiling과 Optimizing을 통해 최적의 Partioning 전략을 찾아냅니다.
  
## 4. Variations of 1F1B Pipelining

PipeDream의 1F1B 파이프라이닝을 개선한 두가지 버전의 파이프라인 전략을 소개합니다.   
### 1) PipeDream 2BW (2-buffered weight update)
PipeDream 2BW는 PipeDream의 메모리 비효율성을 개선하기 위해 등장했습니다. 핵심 아이디어는 파이프라이닝 중에 Gradient Accumulation을 수행하는 것입니다. 여러개의 Gradient들을 모아두다가 한번에 업데이트를 수행하는 방식으로 메모리 비효율성 문제를 해결했죠. 2BW는 이전과 달리 단 두개의 weight version만 유지하면 됩니다.
   
![](../images/pipe_dream_2bw.png)
  
### 2) PipeDream Flush
PipeDream Flush는 1F1B와 Pipeline Flush를 결합한 파이프라이닝 방법입니다. 이 파이프라이닝 방법은 Flush가 일어나기 때문에 GPIpe와 비교하여 idle time은 비슷하나, forward-backward 과정에서 유지해야 하는 **activation memory가 줄어듭니다.** PipeDream Flush는 Flush가 일어나기 때문에 여러버전의 파라미터를 관리할 필요가 없습니다. 따라서 단일 가중치만 유지하면 되기 때문에 PipeDream 2BW보다도 더 메모리 효율적입니다. (지금까지 소개드린 기법들 중 가장 메모리 효율적입니다.)
    
![](../images/pipe_dream_flush.png)

![](../images/pipe_dream_flush_2.png)
 
### 잠깐... 근데 Activation Memory가 뭐야?
대부분의 Layer들은 Backward를 호출하기 전에 Forward에서 나온 출력값들을 저장하고 있습니다. 이는 `torch.autograd.Function`을 사용해보신 분들은 잘 아실텐데요. `ctx`변수에 forward 레이어의 출력값들을 저장해둡니다.
```
"""
참고: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
"""

import torch


class ReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # input 값을 저장하고 있음.
        
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```
 
이는 미분값(Gradient)을 계산할때 Forward 과정에서 사용했던 값들이 필요하기 때문입니다. 다음 예시를 봅시다.
  
![](../images/max_pooling.png)
    
위는 Max Pooling 연산과 그에 대한 Gradient를 계산한 것입니다. Backward를 수행할때는 [[0.8, 1.2], [0.9, 0.5]]와 같은 (2, 2) 텐서가 입력으로 들어옵니다. 이 값을 가지고 오른쪽의 Gradient Matrix를 찾아내야 하는데 반드시 Forward에서 받았던 (4, 4)의 텐서가 필요합니다. 따라서 이 텐서를 메모리에 저장하고 있는 것이죠. 이렇게 Backward를 수행하기 위해 Forward 당시에 쓰였던 텐서들을 저장해두기 위해 필요한 메모리를 Activation Memory라고 합니다."
 
이제 Activation Memory가 뭔지 알았으니, PipeDream을 실습해볼까요? **PipeDream Flush는 MS의 분산처리 라이브러리 DeepSpeed에 구현되어 있습니다.** (참고: https://github.com/microsoft/DeepSpeed/issues/1110) 따라서 DeepSpeed를 사용해봅시다.

### DeepSpeed 명령어 사용법
아 참, 그 전에 `deepspeed`가 제공하는 매우 편리한 기능을 먼저 알아보고 가겠습니다. 기존에는 분산처리를 위해 `python -m torch.distributed.launch --nproc_per_node=n OOO.py`를 사용했으나 너무 길어서 불편했죠. DeepSpeed는 `deepspeed` 혹은 `ds`와 같은 명령어를 제공하고 있습니다. 
- `ds --num_gpus=n OOO.py`
- `deepspeed --num_gpus=n OOO.py`
  
위와 같은 명령어를 입력하면 `torch.distributed.launch`와 동일하게 작동합니다. 이제부터는 모든 분산처리 프로그램에 `deepspeed`의 명령어를 사용하도록 하겠습니다. (솔직히 `torch.distributed.launch`는 너무 길어요 😭)
```
"""
src/ch5/pipe_dream.org.py
"""
import deepspeed
import torch
import torch.nn as nn
from datasets import load_dataset
from deepspeed import PipelineModule
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as GPT2BlockBase
import torch.distributed as dist


class GPT2Preprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        position_ids = torch.arange(
            0, input_shape[-1], dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states


class GPT2Block(GPT2BlockBase):
    def forward(self, hidden_states):
        hidden_states = super(GPT2Block, self).forward(
            hidden_states=hidden_states,
        )
        return hidden_states[0]


class GPT2Postprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


def create_model_from_pretrained(model_name):
    pretrained = GPT2LMHeadModel.from_pretrained(model_name)
    preprocess = GPT2Preprocessing(pretrained.config)
    preprocess.wte.weight = pretrained.transformer.wte.weight
    preprocess.wpe.weight = pretrained.transformer.wpe.weight

    blocks = pretrained.transformer.h
    for i, block in enumerate(blocks):
        block.__class__ = GPT2Block

    postprocess = GPT2Postprocessing(pretrained.config)
    postprocess.ln_f.weight = pretrained.transformer.ln_f.weight
    postprocess.ln_f.bias = pretrained.transformer.ln_f.bias
    postprocess.lm_head.weight.data = pretrained.lm_head.weight.data.clone()

    return nn.Sequential(preprocess, *blocks, postprocess)


def collate_fn(batch):
    batch_encoding = tokenizer.pad(
        {"input_ids": batch}, padding="max_length", max_length=1024
    )
    return batch_encoding.input_ids


def batch_fn(data):
    input_ids = data
    labels = data
    return input_ids, labels


def loss_fn(logits, labels):
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    return nn.CrossEntropyLoss()(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )


if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size, rank = dist.get_world_size(), dist.get_rank()
    batch_size, train_steps = 16, 300
    train_samples = batch_size * train_steps

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = PipelineModule(
        create_model_from_pretrained(model_name="gpt2"),
        loss_fn=loss_fn,
        num_stages=world_size,
        partition_method="type:GPT2Block"
        # partition_method를 통해 병렬화 하고 싶은 레이어를 고를 수 있습니다.
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=Adam(model.parameters(), lr=3e-5),
        config={
            "train_batch_size": batch_size,
            "steps_per_print": 9999999,
            # turn off: https://github.com/microsoft/DeepSpeed/issues/1119
        },
    )
    engine.set_batch_fn(batch_fn)

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for i, sample in enumerate(datasets) if i < train_samples]
    datasets = [
        tokenizer(data, return_tensors="pt", max_length=1024).input_ids[0]
        for data in tqdm(datasets)
    ]
    data_loader = iter(
        DataLoader(
            sorted(datasets, key=len, reverse=True),
            # uniform length batching
            # https://mccormickml.com/2020/07/29/smart-batching-tutorial/
            batch_size=batch_size,
            #num_workers=8,
            num_workers=0,
            collate_fn=collate_fn,
            shuffle=False,
        )
    )

    for i in range(train_steps):
        loss = engine.train_batch(data_loader)

        if i % 10 == 0 and rank == 0:
            print(f"step: {i}, loss: {loss}")
```

```
(large-scale-lm) [gpu05]$ pip install deepspeed==0.15.1
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting deepspeed
  Downloading deepspeed-0.15.1.tar.gz (1.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 16.5 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting hjson (from deepspeed)
  Downloading hjson-3.1.0-py3-none-any.whl.metadata (2.6 kB)
Collecting ninja (from deepspeed)
  Downloading ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl.metadata (5.3 kB)
Requirement already satisfied: numpy in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from deepspeed) (2.0.1)
Requirement already satisfied: packaging>=20.0 in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from deepspeed) (24.1)
Collecting psutil (from deepspeed)
  Downloading psutil-6.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)
Collecting py-cpuinfo (from deepspeed)
  Downloading py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
Collecting pydantic>=2.0.0 (from deepspeed)
  Downloading pydantic-2.9.0-py3-none-any.whl.metadata (146 kB)
Requirement already satisfied: torch in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from deepspeed) (2.3.0)
Requirement already satisfied: tqdm in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from deepspeed) (4.66.5)
Collecting nvidia-ml-py (from deepspeed)
  Downloading nvidia_ml_py-12.560.30-py3-none-any.whl.metadata (8.6 kB)
Collecting annotated-types>=0.4.0 (from pydantic>=2.0.0->deepspeed)
  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.23.2 (from pydantic>=2.0.0->deepspeed)
  Downloading pydantic_core-2.23.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
Requirement already satisfied: typing-extensions>=4.6.1 in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from pydantic>=2.0.0->deepspeed) (4.11.0)
Requirement already satisfied: tzdata in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from pydantic>=2.0.0->deepspeed) (2024.1)
Requirement already satisfied: filelock in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from torch->deepspeed) (3.13.1)
Requirement already satisfied: sympy in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from torch->deepspeed) (1.13.2)
Requirement already satisfied: networkx in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from torch->deepspeed) (3.2.1)
Requirement already satisfied: jinja2 in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from torch->deepspeed) (3.1.4)
Requirement already satisfied: fsspec in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from torch->deepspeed) (2024.6.1)
Requirement already satisfied: MarkupSafe>=2.0 in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from jinja2->torch->deepspeed) (2.1.3)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages (from sympy->torch->deepspeed) (1.3.0)
Downloading pydantic-2.9.0-py3-none-any.whl (434 kB)
Downloading pydantic_core-2.23.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 27.5 MB/s eta 0:00:00
Downloading hjson-3.1.0-py3-none-any.whl (54 kB)
Downloading ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (307 kB)
Downloading nvidia_ml_py-12.560.30-py3-none-any.whl (40 kB)
Downloading psutil-6.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (290 kB)
Downloading py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
Building wheels for collected packages: deepspeed
  Building wheel for deepspeed (setup.py) ... done
  Created wheel for deepspeed: filename=deepspeed-0.15.1-py3-none-any.whl size=1483871 sha256=8df093efa22a297d30e79d70db35ce6ff73e58b19dac1ff3015eaf3a9a2cd60b
  Stored in directory: /tmp/pip-ephem-wheel-cache-c_ezqmsr/wheels/da/cb/14/9cbba50c73df044eb32a7ca29e34844c5f8959e12d22ae8b60
Successfully built deepspeed
Installing collected packages: py-cpuinfo, nvidia-ml-py, ninja, hjson, pydantic-core, psutil, annotated-types, pydantic, deepspeed
Successfully installed annotated-types-0.7.0 deepspeed-0.15.1 hjson-3.1.0 ninja-1.11.1.1 nvidia-ml-py-12.560.30 psutil-6.0.0 py-cpuinfo-9.0.0 pydantic-2.9.0 pydantic-core-2.23.2
```
```
(large-scale-lm) [gpu05]$ ds --num_gpus=4 pipe_dream.org.py
[2024-09-08 23:02:41,542] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-08 23:02:45,778] [WARNING] [runner.py:212:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
Detected VISIBLE_DEVICES=0,1,2,3 but ignoring it because one or several of --include/--exclude/--num_gpus/--num_nodes cl args were used. If you want to use CUDA_VISIBLE_DEVICES don't pass any of these arguments to deepspeed.
[2024-09-08 23:02:45,778] [INFO] [runner.py:585:main] cmd = /scratch/qualis/miniconda3/envs/large-scale-lm/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None pipe_dream.org.py
[2024-09-08 23:02:47,226] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-08 23:02:49,632] [INFO] [launch.py:139:main] 0 NCCL_HOME=/scratch/qualis/nccl_2.11.4-1+cuda11.4_x86_64
[2024-09-08 23:02:49,632] [INFO] [launch.py:146:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2024-09-08 23:02:49,632] [INFO] [launch.py:152:main] nnodes=1, num_local_procs=4, node_rank=0
[2024-09-08 23:02:49,632] [INFO] [launch.py:163:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2024-09-08 23:02:49,632] [INFO] [launch.py:164:main] dist_world_size=4
[2024-09-08 23:02:49,632] [INFO] [launch.py:168:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
[2024-09-08 23:02:49,633] [INFO] [launch.py:256:main] process 51266 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'pipe_dream.org.py', '--local_rank=0']
[2024-09-08 23:02:49,634] [INFO] [launch.py:256:main] process 51267 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'pipe_dream.org.py', '--local_rank=1']
[2024-09-08 23:02:49,635] [INFO] [launch.py:256:main] process 51268 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'pipe_dream.org.py', '--local_rank=2']
[2024-09-08 23:02:49,636] [INFO] [launch.py:256:main] process 51269 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'pipe_dream.org.py', '--local_rank=3']
[2024-09-08 23:02:51,219] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-08 23:02:51,257] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-08 23:02:51,266] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-08 23:02:51,269] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-08 23:02:56,095] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-08 23:02:56,095] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-08 23:02:56,095] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-08 23:02:56,095] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[2024-09-08 23:02:56,095] [INFO] [comm.py:652:init_distributed] cdb=None
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-08 23:02:58,121] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 1
SEED_LAYERS=False BASE_SEED=1234 SEED_FN=None
Using topology: {ProcessCoord(pipe=0, data=0): 0, ProcessCoord(pipe=1, data=0): 1, ProcessCoord(pipe=2, data=0): 2, ProcessCoord(pipe=3, data=0): 3}
[2024-09-08 23:02:58,452] [INFO] [module.py:396:_partition_layers] Partitioning pipeline stages with method type:GPT2Block
stage=0 layers=4
     0: GPT2Preprocessing
     1: GPT2Block
     2: GPT2Block
     3: GPT2Block
stage=1 layers=3
     4: GPT2Block
     5: GPT2Block
     6: GPT2Block
stage=2 layers=3
     7: GPT2Block
     8: GPT2Block
     9: GPT2Block
stage=3 layers=4
    10: GPT2Block
    11: GPT2Block
    12: GPT2Block
    13: GPT2Postprocessing
  loss: loss_fn
[2024-09-08 23:02:58,513] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-09-08 23:02:58,539] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 1
[2024-09-08 23:02:58,753] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-08 23:02:58,753] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 1
[2024-09-08 23:02:58,757] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-09-08 23:02:58,911] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 1
[2024-09-08 23:02:58,980] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2024-09-08 23:02:58,981] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2024-09-08 23:02:58,981] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2024-09-08 23:02:58,982] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = Adam
[2024-09-08 23:02:58,983] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = Adam
[2024-09-08 23:02:58,983] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using configured LR scheduler = None
[2024-09-08 23:02:58,983] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2024-09-08 23:02:58,983] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[3e-05], mom=[(0.9, 0.999)]
[2024-09-08 23:02:58,984] [INFO] [config.py:999:print] DeepSpeedEngine configuration:
[2024-09-08 23:02:58,985] [INFO] [config.py:1003:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2024-09-08 23:02:58,985] [INFO] [config.py:1003:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True, 'use_gds': False}
[2024-09-08 23:02:58,985] [INFO] [config.py:1003:print]   amp_enabled .................. False
[2024-09-08 23:02:58,985] [INFO] [config.py:1003:print]   amp_params ................... False
[2024-09-08 23:02:58,986] [INFO] [config.py:1003:print]   autotuning_config ............ {
    "enabled": false,
    "start_step": null,
    "end_step": null,
    "metric_path": null,
    "arg_mappings": null,
    "metric": "throughput",
    "model_info": null,
    "results_dir": "autotuning_results",
    "exps_dir": "autotuning_exps",
    "overwrite": true,
    "fast": true,
    "start_profile_step": 3,
    "end_profile_step": 5,
    "tuner_type": "gridsearch",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "model_info_path": null,
    "mp_size": 1,
    "max_train_batch_size": null,
    "min_train_batch_size": 1,
    "max_train_micro_batch_size_per_gpu": 1.024000e+03,
    "min_train_micro_batch_size_per_gpu": 1,
    "num_tuning_micro_batch_sizes": 3
}
[2024-09-08 23:02:58,986] [INFO] [config.py:1003:print]   bfloat16_enabled ............. False
[2024-09-08 23:02:58,986] [INFO] [config.py:1003:print]   bfloat16_immediate_grad_update  False
[2024-09-08 23:02:58,986] [INFO] [config.py:1003:print]   checkpoint_parallel_write_pipeline  False
[2024-09-08 23:02:58,986] [INFO] [config.py:1003:print]   checkpoint_tag_validation_enabled  True
[2024-09-08 23:02:58,986] [INFO] [config.py:1003:print]   checkpoint_tag_validation_fail  False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x2af14644d480>
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   communication_data_type ...... None
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   curriculum_enabled_legacy .... False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   curriculum_params_legacy ..... False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   data_efficiency_enabled ...... False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   dataloader_drop_last ......... False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   disable_allgather ............ False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   dump_state ................... False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   dynamic_loss_scale_args ...... None
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   eigenvalue_enabled ........... False
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   eigenvalue_gas_boundary_resolution  1
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   eigenvalue_layer_num ......... 0
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   eigenvalue_max_iter .......... 100
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   eigenvalue_stability ......... 1e-06
[2024-09-08 23:02:58,987] [INFO] [config.py:1003:print]   eigenvalue_tol ............... 0.01
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   eigenvalue_verbose ........... False
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   elasticity_enabled ........... False
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   fp16_auto_cast ............... None
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   fp16_enabled ................. False
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   fp16_master_weights_and_gradients  False
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   global_rank .................. 0
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   grad_accum_dtype ............. None
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   gradient_accumulation_steps .. 1
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   gradient_clipping ............ 0.0
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   gradient_predivide_factor .... 1.0
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   graph_harvesting ............. False
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   initial_dynamic_scale ........ 65536
[2024-09-08 23:02:58,988] [INFO] [config.py:1003:print]   load_universal_checkpoint .... False
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   loss_scale ................... 0
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   memory_breakdown ............. False
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   mics_hierarchial_params_gather  False
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   mics_shard_size .............. -1
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') comet=CometConfig(enabled=False, samples_log_interval=100, project=None, workspace=None, api_key=None, experiment_name=None, experiment_key=None, online=None, mode=None) wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName')
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   optimizer_legacy_fusion ...... False
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   optimizer_name ............... None
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   optimizer_params ............. None
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   pld_enabled .................. False
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   pld_params ................... False
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   prescale_gradients ........... False
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   scheduler_name ............... None
[2024-09-08 23:02:58,989] [INFO] [config.py:1003:print]   scheduler_params ............. None
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   seq_parallel_communication_data_type  torch.float32
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   sparse_attention ............. None
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   sparse_gradients_enabled ..... False
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   steps_per_print .............. 9999999
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   timers_config ................ enabled=True synchronized=True
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   train_batch_size ............. 16
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   train_micro_batch_size_per_gpu  16
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   use_data_before_expert_parallel_  False
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   use_node_local_storage ....... False
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   wall_clock_breakdown ......... False
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   weight_quantization_config ... None
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   world_size ................... 1
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   zero_allow_untested_optimizer  False
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   zero_config .................. stage=0 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500000000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1000000000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50000000 param_persistence_threshold=100000 model_persistence_threshold=9223372036854775807 max_live_parameters=1000000000 max_reuse_distance=1000000000 gather_16bit_weights_on_model_save=False use_all_reduce_for_fetch_params=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   zero_enabled ................. False
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   zero_force_ds_cpu_optimizer .. True
[2024-09-08 23:02:58,990] [INFO] [config.py:1003:print]   zero_optimization_stage ...... 0
[2024-09-08 23:02:58,991] [INFO] [config.py:989:print_user_config]   json = {
    "train_batch_size": 16,
    "steps_per_print": 9.999999e+06
}
[2024-09-08 23:02:58,991] [INFO] [engine.py:105:__init__] CONFIG: micro_batches=1 micro_batch_size=16
[2024-09-08 23:02:58,991] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-09-08 23:02:59,124] [INFO] [engine.py:146:__init__] is_pipe_partitioned= False is_grad_partitioned= False
[2024-09-08 23:02:59,384] [INFO] [engine.py:165:__init__] RANK=0 STAGE=0 LAYERS=4 [0, 4) STAGE_PARAMS=60647424 (60.647M) TOTAL_PARAMS=163037184 (163.037M) UNIQUE_PARAMS=163037184 (163.037M)
[2024-09-08 23:02:59,384] [INFO] [engine.py:165:__init__] RANK=3 STAGE=3 LAYERS=4 [10, 14) STAGE_PARAMS=59862528 (59.863M) TOTAL_PARAMS=163037184 (163.037M) UNIQUE_PARAMS=163037184 (163.037M)
[2024-09-08 23:02:59,384] [INFO] [engine.py:165:__init__] RANK=1 STAGE=1 LAYERS=3 [4, 7) STAGE_PARAMS=21263616 (21.264M) TOTAL_PARAMS=163037184 (163.037M) UNIQUE_PARAMS=163037184 (163.037M)
[2024-09-08 23:02:59,384] [INFO] [engine.py:165:__init__] RANK=2 STAGE=2 LAYERS=3 [7, 10) STAGE_PARAMS=21263616 (21.264M) TOTAL_PARAMS=163037184 (163.037M) UNIQUE_PARAMS=163037184 (163.037M)
  0%|                                                                        | 0/4800 [00:00<?, ?it/s]Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
  0%|                                                                        | 0/4800 [00:00<?, ?it/s]Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
  0%|                                                                        | 0/4800 [00:00<?, ?it/s]Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
  0%|                                                                        | 0/4800 [00:00<?, ?it/s]Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
100%|███████████████████████████████████████████████████████████| 4800/4800 [00:03<00:00, 1214.26it/s]
100%|███████████████████████████████████████████████████████████| 4800/4800 [00:04<00:00, 1181.26it/s]
100%|███████████████████████████████████████████████████████████| 4800/4800 [00:04<00:00, 1145.26it/s]
100%|███████████████████████████████████████████████████████████| 4800/4800 [00:04<00:00, 1133.00it/s]
step: 0, loss: 6.572026252746582
step: 10, loss: 1.0279344320297241
step: 20, loss: 1.028913974761963
step: 30, loss: 0.7047958970069885
step: 40, loss: 0.7454834580421448
step: 50, loss: 0.812176525592804
step: 60, loss: 0.8095153570175171
step: 70, loss: 0.7496554255485535
step: 80, loss: 0.6204543113708496
step: 90, loss: 0.5449276566505432
step: 100, loss: 0.6550153493881226
step: 110, loss: 0.5415732264518738
step: 120, loss: 0.6244872808456421
step: 130, loss: 0.6143221855163574
step: 140, loss: 0.5201988816261292
step: 150, loss: 0.46794116497039795
step: 160, loss: 0.46409428119659424
step: 170, loss: 0.5390290021896362
step: 180, loss: 0.48016053438186646
step: 190, loss: 0.4120233952999115
step: 200, loss: 0.4341399371623993
step: 210, loss: 0.39358749985694885
step: 220, loss: 0.36393406987190247
step: 230, loss: 0.3397323787212372
step: 240, loss: 0.30079731345176697
step: 250, loss: 0.2613028287887573
step: 260, loss: 0.29585179686546326
step: 270, loss: 0.2654482126235962
step: 280, loss: 0.2011309415102005
step: 290, loss: 0.18337950110435486
[2024-09-08 23:09:58,105] [INFO] [launch.py:351:main] Process 51269 exits successfully.
[2024-09-08 23:09:58,105] [INFO] [launch.py:351:main] Process 51266 exits successfully.
[2024-09-08 23:09:59,106] [INFO] [launch.py:351:main] Process 51267 exits successfully.
[2024-09-08 23:09:59,107] [INFO] [launch.py:351:main] Process 51268 exits successfully.
```

## 5. Interleaved Scheduling
이전에는 하나의 스테이지(연속된 레이어 집합)를 순차적으로 계산해서 결과 값을 출력했습니다. 예를 들면 8개의 레이어가 있고 2개의 디바이스가 주어졌다고 가정한다면, 일반적으로 1번 device에 1-4번 레이어, 2번 device에 5-8번 레이어에 할당되겠죠. 그러면 1번 device는 1~4번 레이어를 순차적으로 진행하여 출력했습니다. (GPipe, 1F1B 모두 이렇게 동작함)

![](../images/interleaved_1.png)
   
그러나 **Interleaved Scheduling은 Bubble time을 극도로 줄이기 위해 하나의 스테이지를 중첩해서 진행**합니다. 예를 들면  1번 device가 1-4번 레이어에 할당 되었다면, 1-2번 레이어의 동시에 3-4번 레이어를 동시에 수행합니다. 이렇게 되면 Bubble time은 줄어들지만 통신비용이 커지기 때문에 잘 조절할 필요가 있습니다. (트레이드 오프 존재)
