# Data Parallelism
이번 세션에서는 다양한 데이터 병렬화 기법에 대해 알아보겠습니다.

**Contents**
* [DP in PyTorch (torch.nn.DataParallel)](#data-paralell-in-pytorch)
* [DDP in PyTorch (torch.nn.parallel.distributeddataparallel)](#distributed-data-parallel-in-pytorch)
* [Distributed Training with Horovod](#distributed-training-on-a-supercomputer-with-horovod)
* [Distributed Training with Pytorch Lightning](#distributed-training-on-a-supercomputer-with-pytorch-lightning)
    
## Data Paralell in PyTorch
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
src/ch4/data_parallel.org.py
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

Data Parallel 코드를 실행하기 위해서 뉴론 시스템에서 노드 1개와 GPU 4개를 할당받는다. `src/ch4` 디렉토리로 이동하고 모듈을 로드한다.
```
[glogin01]$ salloc --partition=cas_v100nv_8 -J debug --nodes=1 --time=8:00:00 --gres=gpu:4 --comment pytorch
...
salloc: Nodes gpu[05] are ready for job
[gpu05] $ pwd
/scratch/qualis/git-projects/large-scale-lm-tutorials/src/ch4
[gpu05]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.1
```
```
[gup05]$ conda activate large-scale-lm
(large-scale-lm) [gpu05]$ pip install transformers datasets evaluate scikit-learn
(large-scale-lm) [gpu05]$ python data_parallel.org.py
Downloading readme: 100%|████████████████████████████████████████| 8.89k/8.89k [00:00<00:00, 21.9kB/s]
Downloading data: 100%|████████████████████████████████████████████| 214M/214M [00:22<00:00, 9.63MB/s]
Downloading data: 100%|███████████████████████████████████████████| 4.94M/4.94M [00:09<00:00, 522kB/s]
Downloading data: 100%|██████████████████████████████████████████| 5.10M/5.10M [00:02<00:00, 1.95MB/s]
Generating train split: 100%|███████████████████████| 392702/392702 [00:04<00:00, 94180.67 examples/s]
Generating validation_matched split: 100%|█████████████| 9815/9815 [00:00<00:00, 113395.79 examples/s]
Generating validation_mismatched split: 100%|███████████| 9832/9832 [00:00<00:00, 11346.93 examples/s]
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
model.safetensors: 100%|███████████████████████████████████████████| 436M/436M [00:07<00:00, 62.1MB/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
step:0, loss:1.1370294094085693
step:10, loss:1.1108639240264893
step:20, loss:1.0596747398376465
step:30, loss:0.9757345914840698
step:40, loss:0.8766409158706665
step:50, loss:0.8808916211128235
step:60, loss:0.8087615966796875
step:70, loss:0.8037520051002502
step:80, loss:0.7576047778129578
step:90, loss:0.6615070104598999
step:100, loss:0.7510014176368713
step:110, loss:0.7559493780136108
step:120, loss:0.7757337689399719
step:130, loss:0.7405450940132141
step:140, loss:0.7066846489906311
step:150, loss:0.716975212097168
step:160, loss:0.6581017971038818
step:170, loss:0.649760901927948
step:180, loss:0.7065140008926392
step:190, loss:0.5565707683563232
step:200, loss:0.6459232568740845
step:210, loss:0.689096212387085
step:220, loss:0.5366467833518982
step:230, loss:0.6110734939575195
step:240, loss:0.7274714112281799
step:250, loss:0.5718880891799927
step:260, loss:0.7024426460266113
step:270, loss:0.6618905663490295
step:280, loss:0.6490603685379028
step:290, loss:0.6166819334030151
step:300, loss:0.44881874322891235
```

```
"""
src/ch4/data_parallel.py
"""
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification
from transformers import AdamW


from transformers import get_scheduler

from tqdm.auto import tqdm

import evaluate

batch_size = 32
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

raw_datasets = load_dataset("multi_nli", split="train")

train_test_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

tokenized_datasets = train_test_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#train_datasets = tokenized_datasets["train"].shuffle(seed=12).select(range(20000))
train_datasets = tokenized_datasets["train"].shuffle(seed=12).select(range(353431))
#valid_datasets = tokenized_datasets["test"].shuffle(seed=12).select(range(2000))
valid_datasets = tokenized_datasets["test"].shuffle(seed=12).select(range(39271))



train_dataloader = DataLoader(
    #tokenized_datasets["train"], shuffle=True, batch_size=32, =data_collator
    train_datasets, shuffle=True, batch_size=batch_size*device_count, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    #tokenized_datasets["test"], batch_size=64, collate_fn=data_collator
    valid_datasets, batch_size=batch_size*device_count, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
#num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# data parallel
#model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)
model = nn.DataParallel(model)

progress_bar = tqdm(range(num_training_steps))

loss_fn = nn.CrossEntropyLoss(reduction="mean")

model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        #loss = outputs.loss
        # print(f"loss: {loss}") #loss: tensor([1.1256, 1.0732, 1.2382, 1.2967], device='cuda:0',grad_fn=<GatherBackward>)
        #loss = torch.mean(loss)
        #print(f"after torch mean loss: {loss}") # loss: 1.1834330558776855
        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"])


        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        #steps = (epoch+1)*i
        #if steps % 100 == 0:
        #   print(f"step:{steps}, loss:{loss}")



metric = evaluate.load("accuracy")

num_eval_steps = len(eval_dataloader)
progress_bar = tqdm(range(num_eval_steps))

model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)

evaluation = metric.compute()
print(f"\nMetric: {evaluation}")

```
```
(large-scale-lm) [gpu05]$ python data_parallel.py
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
  0%|                                                                        | 0/2762 [00:00<?, ?it/s]/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
100%|█████████████████████████████████████████████████████████████| 2762/2762 [11:50<00:00,  3.89it/s]
  0%|                                                                         | 0/307 [00:00<?, ?it/s]
Metric: {'accuracy': 0.8307657049731354}████████████████████████████| 307/307 [00:37<00:00,  9.14it/s]
100%|███████████████████████████████████████████████████████████████| 307/307 [00:37<00:00,  8.15it/s]
```
![](../images/dp_training.png)
Multi-GPU에서 학습이 잘 되는군요. 그런데 문제는 0번 GPU에 Logits이 쏠리다보니 GPU 메모리 불균형 문제가 일어납니다. 이러한 문제는 0번 device로 Logits이 아닌 Loss를 Gather하는 방식으로 변경하면 어느정도 완화시킬 수 있습니다. Logits에 비해 Loss는 Scalar이기 때문에 크기가 훨씬 작기 때문이죠. 이 작업은 [당근마켓 블로그](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)에 소개되었던 [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)의 `DataParallelCriterion`과 동일합니다. 블로그에 꽤나 복잡하게 설명되어 있는데, 복잡한 방법 대신 간단하게 **forward 함수를 오버라이드 하는 것** 만으로 동일 기능을 쉽게 구현 할 수 있습니다.
    
![](../images/dp_forward_2.png)
    
핵심은 Loss Computation과 Loss가 reduction을 multi-thread 안에서 작동 시키는 것입니다. 모델의 forward 함수는 multi-thread에서 작동되고 있기 때문에 Loss Computation 부분을 forward 함수 안에 넣으면 매우 쉽게 구현할 수 있겠죠.
    
한가지 특이한 점은 이렇게 구현하면 Loss의 reduction이 2번 일어나게 되는데요. multi-thread에서 batch_size//4개에서 4개로 reduction 되는 과정(그림에서 4번)이 한번 일어나고, 각 디바이스에서 출력된 4개의 Loss를 1개로 Reduction 하는 과정(그림에서 5번)이 다시 일어나게 됩니다. 그렇다고 하더라도 Loss computation 부분을 병렬화 시킬 수 있고, 0번 GPU에 가해지는 메모리 부담이 적기 때문에 훨씬 효율적이죠.
```
"""
src/ch4/custom_data_parallel.py
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
src/efficient_data_parallel.org.py
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
(large-scale-lm) [gpu05]$ python efficient_data_parallel.org.py
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
step:0, loss:1.1640931367874146
step:10, loss:1.1262718439102173
step:20, loss:1.105604887008667
step:30, loss:1.0968422889709473
step:40, loss:1.0840587615966797
step:50, loss:1.0564978122711182
step:60, loss:1.0217713117599487
step:70, loss:0.9702370166778564
step:80, loss:0.8745643496513367
step:90, loss:0.7839782238006592
step:100, loss:0.7462543845176697
step:110, loss:0.9073987007141113
step:120, loss:0.8226974010467529
step:130, loss:0.8410908579826355
step:140, loss:0.8973464965820312
step:150, loss:0.7872884273529053
step:160, loss:0.8032517433166504
step:170, loss:0.7142447233200073
step:180, loss:0.8116591572761536
step:190, loss:0.6684067249298096
step:200, loss:0.7727738618850708
step:210, loss:0.866787314414978
step:220, loss:0.6384953856468201
step:230, loss:0.7246848344802856
step:240, loss:0.8128511309623718
step:250, loss:0.5956273078918457
step:260, loss:0.7926182746887207
step:270, loss:0.7693862318992615
step:280, loss:0.7714772820472717
step:290, loss:0.7309739589691162
step:300, loss:0.5764233469963074
```

## `torch.nn.DataParallel`의 문제점
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
 
## Distributed Data Parallel in PyTorch
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
src/ch4/ddp.org.py
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
(large-scale-lm) [gpu05]$ python -m  torch.distributed.launch --nproc_per_node=4 ddp.org.py
(large-scale-lm) [gpu05]$ torchrun --nproc_per_node=4 ddp.org.py
(large-scale-lm) [gpu05]$ srun torchrun --nproc_per_node=4 ddp.org.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being
overloaded, please further tune the variable for optimal performance in your application as needed. 
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

```
"""
src/ch4/ddp.py
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoModelForSequenceClassification
from transformers import AdamW


from transformers import get_scheduler

from tqdm.auto import tqdm

import evaluate


# initialize process group
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()


batch_size = 32
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

raw_datasets = load_dataset("multi_nli", split="train")

train_test_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True)

tokenized_datasets = train_test_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#train_datasets = tokenized_datasets["train"].shuffle(seed=12).select(range(20000))
train_datasets = tokenized_datasets["train"].shuffle(seed=12).select(range(353431))
#valid_datasets = tokenized_datasets["test"].shuffle(seed=12).select(range(2000))
valid_datasets = tokenized_datasets["test"].shuffle(seed=12).select(range(39271))

train_dataloader = DataLoader(
    #tokenized_datasets["train"], shuffle=True, batch_size=32, =data_collator
    train_datasets, shuffle=True, batch_size=batch_size*device_count, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    #tokenized_datasets["test"], batch_size=64, collate_fn=data_collator
    valid_datasets, batch_size=batch_size*device_count, collate_fn=data_collator
)

# create DistributedSampler
train_sampler = DistributedSampler(
    #datasets,
    train_datasets,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)
train_dataloader = DataLoader(
    #datasets,
        train_datasets,
    batch_size=32,
    num_workers=8,
    sampler=train_sampler,
    shuffle=False,
    pin_memory=True,
    collate_fn=data_collator,
)

eval_sampler = DistributedSampler(
    #datasets,
    valid_datasets,
    num_replicas=world_size,
    rank=rank,
    shuffle=False,
)
eval_dataloader = DataLoader(
    #datasets,
    valid_datasets,
    batch_size=32,
    num_workers=8,
    sampler=eval_sampler,
    shuffle=False,
    pin_memory=True,
    collate_fn=data_collator,
)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

optimizer = AdamW(model.parameters(), lr=5e-5)

#num_epochs = 5
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# distributed data parallel
model = model.to(device)
#model = DistributedDataParallel(model, device_ids=[device])
model = DistributedDataParallel(model)

progress_bar = tqdm(range(num_training_steps))

loss_fn = nn.CrossEntropyLoss(reduction="mean")

model.train()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        # print(f"loss: {loss}") #loss: tensor([1.1256, 1.0732, 1.2382, 1.2967], device='cuda:0',grad_fn=<GatherBackward>)
        #loss = torch.mean(loss)
        #print(f"after torch mean loss: {loss}") # loss: 1.1834330558776855
        #logits = outputs.logits
        #loss = loss_fn(logits, batch["labels"])


        loss.backward()

        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        #steps = (epoch+1)*i
        #if steps % 100 == 0:
        #   print(f"step:{steps}, loss:{loss}")



metric = evaluate.load("accuracy")

num_eval_steps = len(eval_dataloader)
progress_bar = tqdm(range(num_eval_steps))

model.eval()

for batch in eval_dataloader:
    #batch = {k: v.to(device) for k, v in batch.items()}
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    progress_bar.update(1)

evaluation = metric.compute()
#print(type(evaluation)) # class dict
#print(evaluation["accuracy"])
print(f"\nMetric: {evaluation}")

evaluation_tensor = torch.tensor(evaluation["accuracy"]).to(device)

dist.reduce(evaluation_tensor, op=torch.distributed.ReduceOp.AVG, dst=0)

if rank == 0:
    print(f"\nAverge Accuracy: {evaluation_tensor.item():.3f}")
```

```
(large-scale-lm) [gpu05]$ torchrun --nproc_per_node=4 ddp.py
W0908 21:52:13.856000 47481656617152 torch/distributed/run.py:757]
W0908 21:52:13.856000 47481656617152 torch/distributed/run.py:757] *****************************************
W0908 21:52:13.856000 47481656617152 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0908 21:52:13.856000 47481656617152 torch/distributed/run.py:757] *****************************************
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
Map:   0%|                                                           | 0/39271 [00:00<?, ? examples/s]/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
Map:  20%|█████████▎                                    | 8000/39271 [00:01<00:04, 6525.43 examples/s]/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
Map: 100%|█████████████████████████████████████████████| 39271/39271 [00:05<00:00, 6732.23 examples/s]
Map: 100%|█████████████████████████████████████████████| 39271/39271 [00:05<00:00, 6585.32 examples/s]
Map: 100%|█████████████████████████████████████████████| 39271/39271 [00:05<00:00, 6578.37 examples/s]
Map:  81%|████████████████████████████████████▋        | 32000/39271 [00:04<00:00, 7292.94 examples/s]Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
Map:  84%|█████████████████████████████████████▊       | 33000/39271 [00:04<00:00, 7603.37 examples/s]Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
Map:  87%|██████████████████████████████████████▉      | 34000/39271 [00:05<00:00, 7669.44 examples/s]Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
Map: 100%|█████████████████████████████████████████████| 39271/39271 [00:05<00:00, 6713.38 examples/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
  0%|                                                                        | 0/2762 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
100%|█████████████████████████████████████████████████████████████| 2762/2762 [09:48<00:00,  4.69it/s]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
100%|█████████████████████████████████████████████████████████████| 2762/2762 [09:48<00:00,  4.69it/s]
100%|█████████████████████████████████████████████████████████████| 2762/2762 [09:48<00:00,  4.69it/s]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
100%|█████████████████████████████████████████████████████████████| 2762/2762 [09:48<00:00,  4.69it/s]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

Metric: {'accuracy': 0.8206355673253208}█████████████████████       | 273/307 [00:15<00:01, 19.72it/s]
 92%|█████████████████████████████████████████████████████████▊     | 282/307 [00:15<00:01, 18.73it/s]
Metric: {'accuracy': 0.8306172336524751}████████████████████████▋   | 291/307 [00:16<00:00, 21.02it/s]
 98%|█████████████████████████████████████████████████████████████▌ | 300/307 [00:16<00:00, 20.47it/s]
Metric: {'accuracy': 0.8219596659197392}
100%|███████████████████████████████████████████████████████████████| 307/307 [00:16<00:00, 18.25it/s]
100%|███████████████████████████████████████████████████████████████| 307/307 [00:16<00:00, 18.25it/s]

Metric: {'accuracy': 0.8224689346099002}███████████████████████████▊| 306/307 [00:17<00:00, 22.01it/s]

Averge Accuracy: 0.824
100%|███████████████████████████████████████████████████████████████| 307/307 [00:17<00:00, 17.48it/s]
100%|███████████████████████████████████████████████████████████████| 307/307 [00:17<00:00, 17.50it/s]
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


## Distributed Training on a Supercomputer with Horovod
슈퍼컴퓨터에서 Horovod를 사용한 Distributed Data Parallel 실습하기는 아래 깃허브 사이트에 자세히 기술되었습니다. 
- https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod


## Distributed Training on a Supercomputer with Pytorch Lightning
슈퍼컴퓨터에서 Pytorch Lightning을 사용한 Distributed Data Parallel 실습하기는 아래 깃허브 사이트에 자세히 기술되었습니다. 
- https://github.com/hwang2006/distributed-training-on-supercomputer-with-pytorch-lightning


