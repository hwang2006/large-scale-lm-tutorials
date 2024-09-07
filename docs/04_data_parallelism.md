# Data Parallelism
이번 세션에는 데이터 병렬화 기법에 대해 알아보겠습니다.
    
## 1. `torch.nn.DataParallel`
가장 먼저 우리에게 친숙한 `torch.nn.DataParallel`의 동작 방식에 대해 알아봅시다. `torch.nn.DataParallel`은 single-node & multi-GPU에서 동작하는 multi-thread 모듈입니다.

### 1) Forward Pass
1. 입력된 mini-batch를 **Scatter**하여 각 디바이스로 전송.
2. GPU-1에 올라와 있는 모델의 파라미터를 GPU-2,3,4로 **Broadcast**.
3. 각 디바이스로 복제된 모델로 **Forward**하여 Logits을 계산 함.
4. 계산된 Logits을 **Gather**하여 GPU-1에 모음.
5. Logits으로부터 **Loss**를 계산함. (with loss reduction)
   
![](../images/dp_forward.png)
    
코드로 나타내면 아래와 같습니다.
```
import torch.nn as nn


def data_parallel(module, inputs, labels, device_ids, output_device):
    inputs = nn.parallel.scatter(inputs, device_ids)
    # 입력 데이터를 device_ids들에 Scatter함

    replicas = nn.parallel.replicate(module, device_ids)
    # 모델을 device_ids들에 복제함.
   
    logit = nn.parallel.parallel_apply(replicas, inputs)
    # 각 device에 복제된 모델이 각 device의 데이터를 Forward함.

    logits = nn.parallel.gather(outputs, output_device)
    # 모델의 logit을 output_device(하나의 device)로 모음
    
    return logits
```

### 2) Backward Pass
1. 계산된 Loss를 각 디바이스에 **Scatter**함.
2. 전달받은 Loss를 이용해서 각 디바이스에서 **Backward**를 수행하여 Gradients 계산.
3. 계산된 모든 Gradient를 GPU-1로 **Reduce**하여 GPU-1에 전부 더함.
4. 더해진 Gradients를 이용하여 GPU-1에 있는 모델을 업데이트.
![](../images/dp_backward.png)
   
#### 혹시나 모르시는 분들을 위해...
- `loss.backward()`: 기울기를 미분해서 Gradient를 계산
- `optimizer.step()`: 계산된 Gradient를 이용해서 파라미터를 업데이트
- Computation cost는 `backward()` > `step()`.

![](../images/backward_step.png)
```
"""
src/data_parallel.py
"""

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# 1. create dataset
datasets = load_dataset("multi_nli").data["train"]
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]
data_loader = DataLoader(datasets, batch_size=128, num_workers=4)

# 2. create model and tokenizer
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()

# 3. make data parallel module
# device_ids: 사용할 디바이스 리스트 / output_device: 출력값을 모을 디바이스
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)

# 4. create optimizer and loss fn
optimizer = Adam(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss(reduction="mean")

# 5. start training
for i, data in enumerate(data_loader):
    optimizer.zero_grad()
    tokens = tokenizer(
        data["premise"],
        data["hypothesis"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    logits = model(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        return_dict=False,
    )[0]

    loss = loss_fn(logits, data["labels"].cuda())
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"step:{i}, loss:{loss}")

    if i == 300:
        break
```

```
[glogin01]$ python ../src/data_parallel.py
Using custom data configuration default
Reusing dataset multi_nli
(/home/ubuntu/.cache/huggingface/datasets/multi_nli/default/0.0.0/591f72eb6263d1ab527561777936b199b714cda156d35716881158a2bd144f39)
100%|█████████████████████████████████████████████| 3/3 [00:00<00:00, 58.31it/s]
Some weights of the model checkpoint at bert-base-cased were not used when initializing
BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.bias',
'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight',
'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.weight',
'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on
another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a
BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that
you expect to be exactly identical (initializing a BertForSequenceClassification model frm
BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased
and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
step:0, loss:1.1612184047698975
step:10, loss:1.1026676893234253
step:20, loss:1.0577733516693115
step:30, loss:0.9685771465301514
step:40, loss:0.8478926420211792
step:50, loss:0.8693557977676392
step:60, loss:0.7827763557434082
step:70, loss:0.7895966172218323
step:80, loss:0.7631332278251648
step:90, loss:0.6766361594200134
step:100, loss:0.6931278109550476
step:110, loss:0.7477961778640747
step:120, loss:0.7386300563812256
step:130, loss:0.7414667010307312
step:140, loss:0.7170238494873047
step:150, loss:0.7286601066589355
step:160, loss:0.7063153982162476
step:170, loss:0.6415464282035828
step:180, loss:0.7068504095077515
step:190, loss:0.593433678150177
step:200, loss:0.6224725246429443
step:210, loss:0.7025654315948486
step:220, loss:0.5605336427688599
step:230, loss:0.578403890132904
step:240, loss:0.7344318628311157
step:250, loss:0.5977576971054077
step:260, loss:0.6717301607131958
step:270, loss:0.7103744745254517
step:280, loss:0.6679482460021973
step:290, loss:0.635512113571167
step:300, loss:0.45178914070129395
```
![](../images/dp_training.png)
Multi-GPU에서 학습이 잘 되는군요. 그런데 문제는 0번 GPU에 Logits이 쏠리다보니 GPU 메모리 불균형 문제가 일어납니다. 이러한 문제는 0번 device로 Logits이 아닌 Loss를 Gather하는 방식으로 변경하면 어느정도 완화시킬 수 있습니다. Logits에 비해 Loss는 Scalar이기 때문에 크기가 훨씬 작기 때문이죠. 이 작업은 [당근마켓 블로그](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)에 소개되었던 [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)의 `DataParallelCriterion`과 동일합니다. 블로그에 꽤나 복잡하게 설명되어 있는데, 복잡한 방법 대신 간단하게 **forward 함수를 오버라이드 하는 것** 만으로 동일 기능을 쉽게 구현 할 수 있습니다.
    
![](../images/dp_forward_2.png)
    
핵심은 Loss Computation과 Loss가 reduction을 multi-thread 안에서 작동 시키는 것입니다. 모델의 forward 함수는 multi-thread에서 작동되고 있기 때문에 Loss Computation 부분을 forward 함수 안에 넣으면 매우 쉽게 구현할 수 있겠죠.
    
한가지 특이한 점은 이렇게 구현하면 Loss의 reduction이 2번 일어나게 되는데요. multi-thread에서 batch_size//4개에서 4개로 reduction 되는 과정(그림에서 4번)이 한번 일어나고, 각 디바이스에서 출력된 4개의 Loss를 1개로 Reduction 하는 과정(그림에서 5번)이 다시 일어나게 됩니다. 그렇다고 하더라도 Loss computation 부분을 병렬화 시킬 수 있고, 0번 GPU에 가해지는 메모리 부담이 적기 때문에 훨씬 효율적이죠.
```
"""
src/custom_data_parallel.py
"""

from torch import nn


# logits을 출력하는 일반적인 모델
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 3)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs


# forward pass에서 loss를 출력하는 parallel 모델
class ParallelLossModel(Model):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, labels):
        logits = super(ParallelLossModel, self).forward(inputs)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
        return loss
```
운이 좋게도 우리가 자주 사용하는 Huggingface Transformers 모델들은 forward pass에서 곧 바로 Loss를 구하는 기능을 내장하고 있습니다. 따라서 이러한 과정 없이 transformers의 기능을 이용하여 진행하겠습니다. 아래의 코드는 Transformers 모델의 `labels`인자에 라벨을 입력하여 Loss를 바로 출력합니다.
```
"""
src/efficient_data_parallel.py
"""

# 1 ~ 4까지 생략...

# 5. start training
for i, data in enumerate(data_loader):
    optimizer.zero_grad()
    tokens = tokenizer(
        data["premise"],
        data["hypothesis"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    loss = model(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=data["labels"],
    ).loss
    
    loss = loss.mean()
    # (4,) -> (1,)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"step:{i}, loss:{loss}")

    if i == 300:
        break
```

```
[glogin01]$ python ../src/efficient_data_parallel.py
Using custom data configuration default
Reusing dataset multi_nli
(/home/ubuntu/.cache/huggingface/datasets/multi_nli/default/0.0.0/591f72eb6263d1ab527561777936b199b714cda156d35716881158a2bd144f39)
100%|████████████████████████████████████████████| 3/3 [00:00<00:00, 199.34it/s]
Some weights of the model checkpoint at bert-base-cased were not used when initializing
BertForSequenceClassification: ['cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight',
'cls.predictions.decoder.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias',
'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on
another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a
BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that
you expect to be exactly identical (initializing a BertForSequenceClassification model from a
BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased
and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/home/ubuntu/kevin/kevin_env/lib/python3.8/site-packages/torch/nn/parallel/_functions.py:64: UserWarning: Was
asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
step:0, loss:1.186471700668335
step:10, loss:1.1163532733917236
step:20, loss:1.091385841369629
step:30, loss:1.0980195999145508
step:40, loss:1.0779412984848022
step:50, loss:1.053116798400879
step:60, loss:0.9878815412521362
step:70, loss:0.9763977527618408
step:80, loss:0.8458528518676758
step:90, loss:0.8098542094230652
step:100, loss:0.7924742698669434
step:110, loss:0.8259536027908325
step:120, loss:0.8083906173706055
step:130, loss:0.7789419889450073
step:140, loss:0.7848180532455444
step:150, loss:0.7716841697692871
step:160, loss:0.7316021919250488
step:170, loss:0.6465802192687988
step:180, loss:0.7471408843994141
step:190, loss:0.5954912900924683
step:200, loss:0.6941753029823303
step:210, loss:0.7786209583282471
step:220, loss:0.6332131028175354
step:230, loss:0.6579948663711548
step:240, loss:0.7271711230278015
step:250, loss:0.5837332010269165
step:260, loss:0.6737046241760254
step:270, loss:0.6502429246902466
step:280, loss:0.6647026538848877
step:290, loss:0.6707975268363953
step:300, loss:0.47382402420043945
```

## 2. `torch.nn.DataParallel`의 문제점
### 1) 멀티쓰레드 모듈이기 때문에 Python에서 비효율적임.
Python은 GIL (Global Interpreter Lock)에 의해 하나의 프로세스에서 동시에 여러개의 쓰레드가 작동 할 수 없습니다. 따라서 근본적으로 멀티 쓰레드가 아닌 **멀티 프로세스 프로그램**으로 만들어서 여러개의 프로세스를 동시에 실행하게 해야합니다.
 
### 2) 하나의 모델에서 업데이트 된 모델이 다른 device로 매 스텝마다 복제되어야 함.
현재의 방식은 각 디바이스에서 계산된 Gradient를 하나의 디바이스로 모아서(Gather) 업데이트 하는 방식이기 때문에 업데이트된 모델을 매번 다른 디바이스들로 복제(Broadcast)해야 하는데, 이 과정이 꽤나 비쌉니다. 그러나 Gradient를 Gather하지 않고 각 디바이스에서 자체적으로 `step()`을 수행한다면 모델을 매번 복제하지 않아도 되겠죠. 어떻게 이 것을 구현 할 수 있을까요?

### Solution? ➝ All-reduce!! 🤔
![](../images/allreduce.png)

정답은 앞서 배웠던 All-reduce 연산입니다. 각 디바이스에서 계산된 Gradients를 모두 더해서 모든 디바이스에 균일하게 뿌려준다면 각 디바이스에서 자체적으로 `step()`을 수행 할 수 있습니다. 그러면 매번 모델을 특정 디바이스로부터 복제해 올 필요가 없겠죠. 따라서 All-reduce를 활용하는 방식으로 기존 방식을 개선해야 합니다.

### 그러나...  🤔
그러나 All-reduce는 매우 비용이 높은 연산에 속합니다. 왜 그럴까요? All-reduce의 세부 구현을 살펴봅시다.
    
### Reduce + Broadcast 구현 방식
![](../images/allreduce_1.png)


### All to All 구현 방식
![](../images/allreduce_2.png)
 
## 3. `torch.nn.parallel.DistributedDataParallel` (이하 DDP)
### Ring All-reduce 💍
Ring All-reduce는 2017년에 바이두의 연구진이 개발한 새로운 연산입니다. 기존의 방식들에 비해 월등히 효율적인 성능을 보여줬기 때문에 DDP 개발의 핵심이 되었죠.
- https://github.com/baidu-research/baidu-allreduce
![](../images/ring_allreduce.gif)
![](../images/ring_allreduce.png)

### DDP란?
DDP는 기존 DataParallel의 문제를 개선하기 위해 등장한 데이터 병렬처리 모듈이며 single/multi-node & multi-GPU에서 동작하는 multi-process 모듈입니다. All-reduce를 활용하게 되면서 마스터 프로세스의 개념이 없어졌기 때문에 학습 과정이 매우 심플하게 변합니다.
  
![](../images/ddp.png)

```
"""
src/ddp.py
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# 1. initialize process group
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()

# 2. create dataset
datasets = load_dataset("multi_nli").data["train"]
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]

# 3. create DistributedSampler
# DistributedSampler는 데이터를 쪼개서 다른 프로세스로 전송하기 위한 모듈입니다.
sampler = DistributedSampler(
    datasets,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)
data_loader = DataLoader(
    datasets,
    batch_size=32,
    num_workers=4,
    sampler=sampler,
    shuffle=False,
    pin_memory=True,
)


# 4. create model and tokenizer
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()
# 5. make distributed data parallel module
model = DistributedDataParallel(model, device_ids=[device], output_device=device)

# 5. create optimizer
optimizer = Adam(model.parameters(), lr=3e-5)

# 6. start training
for i, data in enumerate(data_loader):
    optimizer.zero_grad()
    tokens = tokenizer(
        data["premise"],
        data["hypothesis"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    loss = model(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=data["labels"],
    ).loss

    loss.backward()
    optimizer.step()

    if i % 10 == 0 and rank == 0:
        print(f"step:{i}, loss:{loss}")

    if i == 300:
        break
```

멀티프로세스 애플리케이션이기 때문에 `torch.distributed.launch`를 사용합니다.
```
[glogin01]$ python -m  torch.distributed.launch --nproc_per_node=4 ../src/ddp.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
Using custom data configuration default
Using custom data configuration default
Reusing dataset multi_nli (/home/ubuntu/.cache/huggingface/datasets/multi_nli/default/0.0.0/591f72eb6263d1ab527561777936b199b714cda156d35716881158a2bd144f39)
Reusing dataset multi_nli (/home/ubuntu/.cache/huggingface/datasets/multi_nli/default/0.0.0/591f72eb6263d1ab527561777936b199b714cda156d35716881158a2bd144f39)
100%|████████████████████████████████████████████| 3/3 [00:00<00:00, 181.01it/s]
100%|████████████████████████████████████████████| 3/3 [00:00<00:00, 149.46it/s]
Using custom data configuration default
Reusing dataset multi_nli (/home/ubuntu/.cache/huggingface/datasets/multi_nli/default/0.0.0/591f72eb6263d1ab527561777936b199b714cda156d35716881158a2bd144f39)
100%|████████████████████████████████████████████| 3/3 [00:00<00:00, 229.28it/s]
Using custom data configuration default
Reusing dataset multi_nli (/home/ubuntu/.cache/huggingface/datasets/multi_nli/default/0.0.0/591f72eb6263d1ab527561777936b199b714cda156d35716881158a2bd144f39)
100%|████████████████████████████████████████████| 3/3 [00:00<00:00, 361.84it/s]
Some weights of the model checkpoint at bert-base-cased were not used when initializing
BertForSequenceClassification: ['cls.predictions.decoder.weight', 'cls.predictions.transform.dense.weight',
'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.seq_relationship.weight',
'cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on
another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a
BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that
you expect to be exactly identical (initializing a BertForSequenceClassification model from a
BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased
and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of the model checkpoint at bert-base-cased were not used when initializing
BertForSequenceClassification: ['cls.predictions.transform.dense.weight', 'cls.predictions.bias',
'cls.seq_relationship.bias',
'cls.predictions.transform.LayerNorm.bias','cls.predictions.transform.LayerNorm.weight',
'cls.predictions.transform.dense.bias', 'cls.seq_relationship.weight', 'cls.predictions.decoder.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on
another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a
BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that
you expect to be exactly identical (initializing a BertForSequenceClassification model from a
BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased
and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of the model checkpoint at bert-base-cased were not used when initializing
BertForSequenceClassification: ['cls.predictions.transform.dense.weight', 'cls.predictions.bias',
'cls.seq_relationship.weight', 'cls.predictions.decoder.weight', 'cls.seq_relationship.bias',
'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', '
cls.predictions.transform.LayerNorm.bias']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on
another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a
BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that
you expect to be exactly identical (initializing a BertForSequenceClassification model from a
BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased
and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of the model checkpoint at bert-base-cased were not used when initializing
BertForSequenceClassification: ['cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias',
'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias',
'cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.decoder.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on
another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a
BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that
you expect to be exactly identical (initializing a BertForSequenceClassification model from a
BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased
and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
step:0, loss:1.1451387405395508
step:10, loss:1.0912988185882568
step:20, loss:1.0485237836837769
step:30, loss:0.9971571564674377
step:40, loss:0.9472718238830566
step:50, loss:1.0532103776931763
step:60, loss:0.6478840112686157
step:70, loss:0.9035330414772034
step:80, loss:0.8176743388175964
step:90, loss:1.058182716369629
step:100, loss:0.7739772796630859
step:110, loss:0.6652507185935974
step:120, loss:0.7778272032737732
step:130, loss:0.827933669090271
step:140, loss:0.6303764581680298
step:150, loss:0.5062040090560913
step:160, loss:0.8570529222488403
step:170, loss:0.6550942063331604
step:180, loss:0.6157522797584534
step:190, loss:0.7612558007240295
step:200, loss:0.7380551099777222
step:210, loss:0.7818665504455566
step:220, loss:0.9607051610946655
step:230, loss:0.8241059184074402
step:240, loss:0.5454672574996948
step:250, loss:0.4731343686580658
step:260, loss:0.8883727788925171
step:270, loss:0.4605785310268402
step:280, loss:0.7553415298461914
step:290, loss:0.8398311138153076
step:300, loss:0.45668572187423706
```

### 그런데 잠깐, All-reduce를 언제 수행하는게 좋을까요?
- All-reduce를 `backward()`연산과 함께 하는게 좋을까요?
- 아니면 `backward()`가 모두 끝나고 `step()` 시작 전에 하는게 좋을까요?
   
![](../images/ddp_analysis_1.png)
 
### 결과적으로 `backward()`와 `all-reduce`를 중첩시키는 것이 좋습니다.
결과적으로 `backward()`와 `all-reduce`를 중첩시키는 것이 가장 효율적인 방식입니다. `all_reduce`는 네트워크 통신, `backward()`, `step()` 등은 GPU 연산이기 때문에 동시에 처리할 수 있죠. 이들을 중첩시키면 즉, computation과 communication이 최대한으로 overlap 되기 때문에 연산 효율이 크게 증가합니다.

![](../images/ddp_analysis_2.png)
   
분석 결과 `backward()`와 `step()`을 비교해보면 `backward()`가 훨씬 무거운 연산이였습니다.
![](../images/ddp_analysis_3.png)
당연히 더 무거운 연산을 중첩시킬 수록 전체 학습 과정을 수행하는 시간이 짧아집니다. 분석 결과 `backward()`가 끝날때 까지 기다리는 것 보다 `all-reduce`를 함께 수행하는 것이 훨씬 빨랐습니다.\n",
    
![](../images/ddp_analysis_4.png)
  
### 이 때, 생길 수 있는 궁금증들...
- Q1: `backward()` 연산 중에 Gradient가 모두 계산되지 않았는데 어떻게 `all-reduce`를 수행합니까?
  - A1: `backward()`는 뒤쪽 레이어부터 순차적으로 이루어지기 때문에 계산이 끝난 레이어 먼저 전송하면 됩니다.
    
- Q2: 그렇다면 언제마다 `all-reduce`를 수행하나요? 레이어마다 이루어지나요?
  - A2: 아닙니다. Gradient Bucketing을 수행합니다. Bucket이 가득차면 All-reduce를 수행합니다.
   
### Gradient Bucekting
Gradient Bucekting는 Gradient를 일정한 사이즈의 bucket에 저장해두고 가득차면 다른 프로세스로 전송하는 방식입니다. 가장 먼저 `backward()` 연산 도중 뒤쪽부터 계산된 Gradient들을 차례대로 bucket에 저장하다가 bucket의 용량이 가득차면 All-reduce를 수행해서 각 device에 Gradient의 합을 전달합니다. 그림 때문에 헷갈릴 수도 있는데, bucket에 저장되는 것은 모델의 파라미터가 아닌 해당 레이어에서 출력된 Gradient입니다. 모든 bucket은 일정한 사이즈를 가지고 있으며 `bucket_size_mb` 인자를 통해 mega-byte 단위로 용량을 설정 할 수 있습니다.
   
![](../images/ddp_analysis_5.png)
