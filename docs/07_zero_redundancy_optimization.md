# Zero Redundancy Optimization (ZeRO)

이번 세션에는 Microsoft의 뉴럴넷 학습 최적화 솔루션인 ZeRO에 대해서 알아보도록 하겠습니다.
 
## 1. Mixed Precision
최신 GPU들이 Lower precision에 대한 계산을 지원하면서 현대의 뉴럴넷 학습은 대부분 FP16(half)과 FP32(single)을 함께 사용하는 Mixed precision 방식을 사용합니다. V100 기준으로 FP32에서 속도가 14TFLOPS 정도라면, FP16에서는 100TFLOPS의 속도로 모델을 학습할 수 있습니다. 또한 FP16을 사용하면 모델의 사이즈가 줄기 때문에 학습 뿐만 아니라 배포시에도 장점이 있죠.
 
![](../images/mixed_precision_1.png)
    
### 그런데 왜 Mixed?
그런데 여기에서 의문이 듭니다. FP16으로만 모델을 학습시키면 되지, 굳이 FP32와 FP16을 같이 쓸 필요가 있을까요? 결과부터 말하자면 FP16만으로 학습시 Loss가 심하게 발산하여 학습이 거의 불가능합니다. Gradient를 FP16로 유지하면 대부분의 소수점을 버리는 것이기 때문에 정밀한 학습이 불가능해집니다. 따라서 속도가 빠른 FP16과 정확도가 높은 FP32를 모두 사용해서 두 방식의 장점만을 취하려고 하는 것이죠. 

![](../images/ddp_analysis_3.png)
    
Computation cost가 큰 Forward와 Backward는 FP16 모델로 하고, 계산된 Gradient를 정밀도가 높은 FP32 모델에 복사해서 weight를 업데이트 합니다. 그런데 여기서 궁금한 점이 생깁니다. FP16의 Gradient를 FP32에 적용하려면 어떻게 해야할까요? 연구진이 실험한 결과, FP16으로 계산된 Loss를 Backward 하면 크기가 크기가 작았던 일부 값들(그림에서 왼쪽)은 계산이 되면서 0으로 변해버렸다고 합니다.
    
![](../images/mixed_precision_4.png)
 
### Loss Scaling
이러한 문제를 어떻게 해결할 수 있을까요? 매우 심플한 아이디어로, Loss Gradient에 매우 큰 값을 곱해줘서 분포를 오른쪽으로 밀어주면 됩니다. 이러한 기술의 이름을 Loss scaling이라고 합니다. FP16의 Loss에 매우 큰 값을 곱하면, FP32에 적용 했을 때 사라져 버릴 수 있는 값들도 잘 살려낼 수 있죠.
    
![](../images/mixed_precision_5.png)

```
"""
참고: apex/apex/amp/opt.py
"""

import contextlib

@contextlib.contextmanager
def scale_loss(self, loss):
    if not self._amp_handle.is_active():
        yield loss
        return

    # When there are multiple losses per-optimizer, we need
    # to save out current grad accumulation, since we won't be
    # able to unscale this particulare loss once the grads are
    # all mixed together.
    cached_grads = []
    if self._loss_idx > 0:
        for p in master_params(self._optimizer):
            if p.grad is not None:
                cached_grads.append(p.grad.data.detach().clone())
            else:
                cached_grads.append(None)
        self._optimizer.zero_grad()

    loss_scale = self._cur_loss_scaler().loss_scale()
    yield loss * loss_scale
```
```
"""
참고: apex/tests/L0/run_amp/test_fused_sgd.py
"""

with amp.scale_loss(loss0, optimizer, loss_id=loss_ids[0]) as scaled_loss:
    scaled_loss.backward()
    if i == inject_inf and which_backward == 0:
        if inject_inf_loc == "fp32":
            model0.weight0.grad[0] = float('inf')
        elif inject_inf_loc == "fp16":
            model0.weight1.grad[0] = float('inf')
```

실제로 아래 그림처럼 Loss에 큰 값을 곱해주면 발산하지 않고 학습이 잘 되었다고 합니다. 회색 그래프는 scaling을 하지 않았을때, 녹색은 scaling 했을때의 성능입니다. 놀랍게도 FP32와 성능이 거의 흡사하죠.

![](../images/mixed_precision_2.png)

이러한 이유로 FP16과 FP32를 함께 사용하는 Mixed precision은 현대 뉴럴넷 학습에 거의 필수가 되었습니다. FP16 정도의 저장 용량으로 FP32의 커버리지를 커버하는 bfloat16 (Google TPU) 방식이 지금보다 더 다양한 GPU에서 지원되고 대중화 되기 전까지는 FP16 + 32의 Mixed precision training은 뉴럴넷 학습에 필수적으로 쓰이는 기술일 것입니다.
   
### Mixed Precision의 동작방식
  
다음은 Mixed Precision의 동작 방식을 나타낸 그림입니다.  코드와 수식을 이용해 진행 과정을 자세히 살펴봅시다.
   
![](../images/mixed_precision_33.png)

### 0) 모델과 옵티마이저 생성

먼저 주피터를 설치하자. 
```
(large-scale-lm) [gpu05]$ conda install jupyter chardet cchardet -y; python -m ipykernel install --user --name large-scale-lm
```

이제 주피터 서버를 뉴론에 실행하고 노트북에서 브라우저로 연결해서 주피터를 실행하자.  

- to create a batch script for launching a jupyter notebook server: 
```
(large-scale-lm) [gpu05]$ pwd
/scratch/qualis/large-scale-lm-tutorials/src
(large-scale-lm) [gpu05]$ cat jupyter_run.sh
#!/bin/bash
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
#SBATCH --time=12:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load gcc/10.2.0 cuda/11.6

echo "execute jupyter"
source ~/.bashrc
conda activate large-scale-lm
#conda activate tf-nlp
cd /scratch/qualis/large-scale-lm/src  # the root/work directory of Jupyter lab/notebook
#cd /scratch/$USER  # the root/work directory of Jupyter lab/notebook
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --no-browser --NotebookApp.token=${USER} #jupyter token: your account ID
echo "end of the job"
```
- to launch a jupyter notebook server 
```
[glogin01]$ sbatch jupyter_run.sh
Submitted batch job XXXXXX
```
- to check if the jupyter notebook server is up and running
```
(large-scale-lm) [gpu05]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
[glogin01]$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://gpu##:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
(large-scale-lm) [gpu05]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```
6. open a new SSH client (e.g., Putty, MobaXterm, PowerShell, Command Prompt, etc) on your PC or laptop and log in to the Neuron system just by copying and pasting the port_forwarding_command:

![20240123_102609](https://github.com/hwang2006/Generative-AI-with-LLMs/assets/84169368/1f5dd57f-9872-491b-8dd4-0aa99b867789)

7. open a web browser on your PC or laptop to access the jupyter server
```
URL Address: localhost:8888
Password or token: $USER    # your account name on Neuron
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 


이제 주피터가 띄웠으니 `ch7/mixed_precision.ipynb` 불러서 mixed precision 하나씩 수행합니다.
먼저 2개의 레이어를 가진 뉴럴넷을 정의합니다.
```
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(512, 512, bias=False)
        self.w2 = nn.Linear(512, 1, bias=False)
    
    def forward(self, x):
        z1 = self.w1(x)
        z2 = self.w2(z1)
        return z2
```

학습할 뉴럴넷과, 옵티마이저 생성합니다.
```
from torch.optim import SGD

fp32_model= Net().to("cuda")
optimizer = SGD(fp32_model.parameters(), lr=1e-2)
```
```
f"GPU = {torch.cuda.memory_allocated(0) / (1024 ** 2)} MB"
```
`'GPU = 1.001953125 MB'`


### 1)  Float2Half
    
이 과정은 단순히 `0.524796132`와 같은 파라미터를 `0.5247`과 같이 잘라내는 작업입니다.
   
보시다시피 용량도 FP32 모델의 절반정도 사이즈를 가집니다. (1.0 MB + 0.5 MB)
```
fp16_model = Net().half().to("cuda")
fp16_model.load_state_dict(fp32_model.state_dict())
```
`<All keys matched successfully>`

```
f"GPU = {torch.cuda.memory_allocated(0) / (1024 ** 2)} MB"
```
`'GPU = 1.5029296875 MB'`

### 2) Forward
    
fp16으로 복사된 모델을 이용하여 forward pass를 수행합니다.

$z_1 = w_1 \\cdot x \\;$ (FWD: layer1)
    
$z_2 = w_2 \\cdot z_1 \\;$ (FWD: layer2)

```
import torch

# example input sizes
batch_size, hidden_size = 4, 512

# create dummy data (bsz=4, hid=512)
x = torch.randn(batch_size,hidden_size, dtype=torch.half, device="cuda") 

# do forward
z2 = fp16_model(x)

# check dtypr of output logits
f"logits type = {z2.dtype}"
```
`'logits type = torch.float16'`

계산된 FP16의 출력값을 이용하여 Loss를 계산합니다.
    
$L = \\frac{(y - z_2)^2}{2} \\;$ (Loss computation)

```
# craete dummy data (bsz=4)
y = torch.tensor([[1.9], [9.5], [0.9], [1.2]], dtype=torch.half, device="cuda")

# compute mean square error loss
L = torch.nn.functional.mse_loss(z2, y)

# check dtype of loss
f"loss type = {L.dtype}"
```
`'loss type = torch.float16'`


### 3) Backward

이제 $w_n := w_n - lr \\cdot \\frac{dL}{dw_n}$와 같은 Gradient Descent Rule로 모델의 파라미터를 업데이트 해야 합니다.
   
따라서 $\\frac{dL}{dw_1}$과 $\\frac{dL}{dw_2}$와 같은 Gradient를 구해야 하는데요. 이들은 대략 아래와 같습니다. (chain rule에 의해서 원하는 결과를 얻을 수 있습니다.)
    
$\\frac{dL}{dw_2} = \\frac{dL}{dz_2} \\cdot \\frac{dz_2}{dw_2}$

$\\frac{dL}{dw_1} = \\frac{dL}{dz_2} \\cdot \\frac{dz_2}{dz_1} \\cdot \\frac{dz_1}{dw_1}$

구체적으로는 아래와 같습니다.
 
$\\frac{dL}{dz_2} =  y - z_2 \\;$ (BWD-activation: layer2)
   
$\\frac{dz_2}{dw_2} = z_1 \\;$ (BWD-weight: layer2)
    
$\\frac{dz_2}{dz_1} = w_2 \\;$ (BWD-activation: layer1)
    
$\\frac{dz_1}{dw_1} = x \\;$ (BWD-weight: layer1)
   
$\\frac{dL}{dw_2} = (y - z_2) \\cdot z_1$
  
$\\frac{dL}{dw_1} = (y - z_2) \\cdot w_2 \\cdot x$ 

```
# loss scaling
L *= 1024

# do backward
L.backward()
```

### 4) Update Weight
 
마지막으로 파라미터를 업데이트하기 위해 `optimizer.step()`를 수행합니다.
   
$w_1 := w_1 - lr \\cdot \\frac{dL}{dw_1} \\;$ (Weight Update)
   
$w_2 := w_2 - lr \\cdot \\frac{dL}{dw_2} \\;$ (Weight Update)

```
print(f'before: {fp32_model.w1.weight}\n')
optimizer.step()
print(f'after: {fp32_model.w1.weight}\n')
```
```
before: Parameter containing:
tensor([[-0.0283, -0.0093,  0.0111,  ...,  0.0222,  0.0309, -0.0075],
        [ 0.0058,  0.0112,  0.0040,  ...,  0.0198,  0.0111,  0.0109],
        [-0.0315, -0.0137,  0.0191,  ..., -0.0410,  0.0124, -0.0407],
        ...,
        [ 0.0271, -0.0025, -0.0171,  ...,  0.0044,  0.0420,  0.0172],
        [ 0.0396,  0.0372, -0.0292,  ...,  0.0309, -0.0390,  0.0402],
        [-0.0374,  0.0217,  0.0248,  ..., -0.0145, -0.0097, -0.0148]],
       device='cuda:0', requires_grad=True)

after: Parameter containing:
tensor([[-0.0283, -0.0093,  0.0111,  ...,  0.0222,  0.0309, -0.0075],
        [ 0.0058,  0.0112,  0.0040,  ...,  0.0198,  0.0111,  0.0109],
        [-0.0315, -0.0137,  0.0191,  ..., -0.0410,  0.0124, -0.0407],
        ...,
        [ 0.0271, -0.0025, -0.0171,  ...,  0.0044,  0.0420,  0.0172],
        [ 0.0396,  0.0372, -0.0292,  ...,  0.0309, -0.0390,  0.0402],
        [-0.0374,  0.0217,  0.0248,  ..., -0.0145, -0.0097, -0.0148]],
       device='cuda:0', requires_grad=True)
```

생각해보면, FP32 모델은 forward & backward를 수행한적이 없었죠. 따라서 gradient 텐서를 갖고있지 않습니다. 그래서 `optimizer.step()`을 수행 해도 값이 변하지 않았습니다. 따라서 `optimizer.step()`을 수행하기 전에, `backward()`를 거친 FP16모델의 gradient를 복사해야 합니다.
    
참고로 PyTorch는 파라미터(`nn.Parameter`) 중 `requires_grad=True`로 설정된 파라미터들은 모두 `grad`라는 애트리뷰트를 가지고 있습니다. 모델이 출력한 텐서의 `backward`가 호출되면 graph를 타고 뒤로 돌아오면서 미분 계산을 수행하고 결과 값을 `grad`라는 공간에 저장합니다. `grad`는 해당 텐서와 동일한 사이즈이기 때문에 모델의 용량이 10GB라면 gradient도 10GB 만큼 필요합니다. 우리가 인퍼런스 할 때 보다 학습할때 메모리가 훨씬 많이 필요한 이유 중 하나입니다. 따라서 학습에 사용될 텐서가 아니라면 반드시 `requires_grad`를 `False`로 설정해야 불필요한 메모리 소모를 막을 수 있습니다.

```
# copy gradient to FP32 model
fp32_model.w1.weight.grad = fp16_model.w1.weight.grad.float()
fp32_model.w2.weight.grad = fp16_model.w2.weight.grad.float()
```
```
print(f'before: {fp32_model.w1.weight}\n')
optimizer.step()
print(f'after: {fp32_model.w1.weight}\n')
```
```
before: Parameter containing:
tensor([[-0.0283, -0.0093,  0.0111,  ...,  0.0222,  0.0309, -0.0075],
        [ 0.0058,  0.0112,  0.0040,  ...,  0.0198,  0.0111,  0.0109],
        [-0.0315, -0.0137,  0.0191,  ..., -0.0410,  0.0124, -0.0407],
        ...,
        [ 0.0271, -0.0025, -0.0171,  ...,  0.0044,  0.0420,  0.0172],
        [ 0.0396,  0.0372, -0.0292,  ...,  0.0309, -0.0390,  0.0402],
        [-0.0374,  0.0217,  0.0248,  ..., -0.0145, -0.0097, -0.0148]],
       device='cuda:0', requires_grad=True)

after: Parameter containing:
tensor([[-0.6399, -1.1324, -2.4314,  ..., -3.1628,  1.4747, -1.5225],
        [-0.5933, -1.0895, -2.3885,  ..., -3.1002,  1.4261, -1.4741],
        [ 0.3779,  0.7382,  1.6541,  ...,  2.0902, -0.9539,  0.9737],
        ...,
        [-0.0872, -0.2125, -0.4740,  ..., -0.5913,  0.3122, -0.2662],
        [ 0.0162, -0.0058, -0.1226,  ..., -0.0910,  0.0162, -0.0178],
        [ 0.1802,  0.4217,  0.8942,  ...,  1.1192, -0.5238,  0.5246]],
       device='cuda:0', requires_grad=True)
```

### Pytorch에서 Mixed precision training 수행하기
   
Pytorch에서는 다음과 같이 손쉽게 Mixed precision training을 수행할 수 있습니다.
```
# 참고: https://pytorch.org/docs/stable/notes/amp_examples.html#automatic-mixed-precision-examples
# Creates model and optimizer in default precision
import torch

model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = torch.cuda.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

```
  
### Dynamic Loss Scaling
    
Loss Scaling은 Mixed Precision 학습을 매우 효과적으로 만들어줬습니다. 그러나 scale 수치를 몇으로 설정하는 것이 가장 좋을지 알기가 매우 어렵습니다. 따라서 몇몇 오픈소스에는 이러한 문제를 해결하기 위해 Dynamic Loss Scaling 기법을 제안합니다. 이는 NVIDIA의 `amp`나 MS의 `deepspeed`에도 구현되어 있습니다. 
   
Dynamic Loss Scaling의 아이디어는 매우 간단합니다. **목표는 Gradient의 소수점들이 Overflow 되지 않는 선에서 scale값을 최대로 유지하는 것**입니다. Gradient 값을 키우면 키울수록 좋지만 너무 커지면 Overflow가 발생하기 때문에 Overflow가 되지 않는 선에서 최대로 키워주는 것이죠. 
    
따라서 학습 초반에 매우 큰 값을 scale 값으로 설정합니다. `deepspeed`의 경우 기본 값이 $2^{32}$로 설정되어 있습니다. 이 값으로 Loss를 backward 해보고 만약 Gradient가 Overflow 되었다면 scale 값을 2배 줄입니다. 이 과정을 여러번 반복하면서 Overflow가 발생하지 않는 최대의 scale값을 찾아내는 것이 바로 Dynamic Loss Scaling입니다.
  
  
### AMP (Apex Mixed Precision)
  
`apex`는 NVIDIA에서 개발한 라이브러리로, Mixed Precision 라이브러리 중에서 가장 유명한 인지도를 가지고 있습니다. 요즘에는 `torch`자체에 mixed precision 기능이 내장되기도 하고 DeepSpeed, Pytorch-Lightning 등의 도구가 많이 나오게 돼서 `apex`를 예전만큼은 자주 사용하지 않지만 그래도 여전히 많이 사용되고 있는 라이브러리입니다. 사용법은 아래와 같이 매우 간단합니다.
```
import torch
from apex import amp


# Declare model and optimizer as usual, with default (FP32) precision
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Allow Amp to perform casts as required by the opt_level
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

# loss.backward() becomes:
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```
위 코드를 보면 `opt_level`이라는 파라미터가 보입니다. `apex`에는 mixed precision의 level을 설정할 수 있는 기능이 있는데 이를 알아두면 추후에 `apex`를 사용할 일이 생길때 매우 유용할 것입니다. (참고로 알파벳 O + 숫자 0,1,2,3입니다.)
    
![](../images/apex.png)
    
- `O0`: FP32 학습
- `O1`: FP16을 잘 지원하는 Tensor Core 연산들은 FP16 / 나머지는 FP32
- `O2`: Normalization의 weight를 제외한 모든 파라미터를 FP16으로 설정
- `O3`: FP16 학습
 
## 2. Zero Redundancy Optimization
    
FP16과 FP32를 함께 사용하게 됨으로써 학습 속도는 매우 빨라지게 되었지만 단점이 생겼습니다. 바로 메모리인데요. FP32의 master weight과 FP16 파라미터, Gradient를 모두 GPU에 올려둔 상태이기 때문에 메모리가 기존보다 더 많이 필요해집니다. 
   
![](../images/zero_1.png)
   
그리고 모델 파라미터가 FP16로 존재한다고 해도, Optimization은 FP32에서 일어나기 때문에 AdaGrad, Adam 등의 Adaptive optimizer 들이 필요로 하는 Variance 및 Momentum과 같은 텐서들은 여전히 FP32로 보관되어야 합니다.

![](../images/adam.png)

```
"""
참고: pytorch/torch/optim/adam.py 
"""

@torch.no_grad()
def step(self, closure=None):
    """Performs a single optimization step.

    Args:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization                
                # 모든 파라미터에 대해서 동일 사이즈로 `exp_avg`와 `exp_avg_sq`로 가지고 있음
                # 이 때문에 Adam 기반의 optimizer를 사용하면 모델 2개에 해당하는 GPU 메모리가 더 필요해짐. 
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

        F.adam(params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'])
    return loss
```

지금까지 FP16 parameter, gradient, FP32 parameter, gradient, momentum, variance 등 우리가 모델을 학습 할 때 메모리에 할당되는 텐서들의 종류에 대해서 조사했습니다. 놀라운 것은 진짜 모델이 차지하는 영역은 얼마 안된다는 것이죠. 이렇게 학습시에는 모델 외에도 **부가적으로 어마어마한 양의 텐서가 GPU 메모리에 할당됩니다.**
  
![](../images/memory.png)

추가로 **Data 텐서**와 **Activation 텐서**도 메모리에 할당됩니다. Data 텐서는 모델에 입력되기 전의 토큰 상태의 텐서를 의미하며, Activation 텐서는 Forward & Bacward 과정에서 연산되는 Hidden states 등의 텐서를 의미합니다. 추가로 분산처리를 수행하면 **통신 중에 텐서들을 담아둘 Bucket 공간** 등도 필요합니다. 버킷에 대해서는 이미 Data Parallelism 세션에서 Gradient Bucketing 등으로 다루었던 적이 있죠. 따라서 **모델과 데이터만 병렬화 할 것이 아니라 이러한 Optimizer States(분산, 모멘텀), Data & Activation Memory 등도 관리할 필요**가 있습니다. 

Zero Redundancy Optimization (이하 ZeRO)는 이러한 부분들을 매우 효율적으로 관리 할 수 있도록 도와주는 **메모리 최적화 기술의 집합체**입니다. 크게 **ZeRO-DP** (ZeRO Data Parallelism)과 **ZeRO-R** (ZeRO Residual States) 등의 솔루션이 존재합니다. 이제부터 차근 차근 알아봅시다.

## 3. ZeRO Data Parallelism
   
가장 먼저 메모리 상태를 조사해보면, 위 그림에서 왼편 (FP16, 32, model & optimizer & gradient)가 가장 큰 공간을 차지합니다. 따라서 이들을 효율적으로 쪼개서 관리해야 합니다. ZeRO-DP는 Data Parallel과 함께 이러한 텐서들을 디바이스마다 쪼개서 관리 할 수 있도록 도와줍니다.
  
![](../images/zero_2.png)
  
ZeRO-DP는 4개의 stage로 나누어서 제공되고 있으며 `DeepSpeed` 라이브러리를 통해 선택적으로 적용 할 수 있습니다.
   
- **Stage 0**: 
  - No Partitioning
  - ZeRO-DP를 적용하지 않습니다.
- **Stage 1**: 
  - Optimizer States Partitioning
  - Optimizer Stages(모멘텀, 분산) 텐서를 여러 GPU로 분할합니다.
  - 메모리 소비량 4배 감소 
  - 기존과 비슷한 양의 Communication Cost
- **Stage 2**: 
  - Stage 1 + Gradient partitioning
  - Gradient(기울기) 텐서를 여러 GPU로 분할합니다.
  - 메모리 소비량 2배 더 감소
  - 기존과 비슷한 양의 Communication Cost
- **Stage 3**: 
  - Parameter partitioning
  - Parameter(모델) 텐서를 여러 GPU로 분할합니다.
  - 메모리 소비량 분할 수준에 따라 선형적 감소
  - 기존보다 1.5배 많은 Communication Cost
    
ZeRO-DP의 동작은 매우 복잡하기 때문에 영상으로 확인하겠습니다.
    
https://www.microsoft.com/en-us/research/uploads/prod/2020/02/Turing-Animation.mp4?_=1"
```
from IPython.display import HTML

HTML("""
<div align="middle">
<video width="80%" controls>
      <source src="../images/zero_video.mp4" type="video/mp4">
</video></div>""")
```

![](../images/zero_3.png)
결론적으로 ZeRO-DP를 적용하면 기존보다 훨씬 큰 모델을 작은 GPU에서 학습시킬 수 있습니다. 바로 실습해봅시다. 먼저 configuration 파일을 만듭니다. 저는 learning rate scheduler, fp16, zero optimization (stage 1) 등을 활성화 시켰습니다. 이외에도 deepspeed configuration에는 매우 다양한 옵션들이 있습니다. 더 많은 옵션들은 https://www.deepspeed.ai/docs/config-json 여기에서 확인하세요.

```
"""
src/ch7/zero_dp_config.org.json
"""
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 300,
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 30
    }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 1
  },
  "zero_allow_untested_optimizer": true,
  "wall_clock_breakdown": false,
  "steps_per_print": 9999999999
}
```


그리고 다음과 같은 코드를 작성합니다. argument parser의 옵션으로 `--local_rank`와 `--deepspeed_config`가 반드시 필요하며, 이 중 `--local_rank`는 스크립트 실행시에 자동으로 입력됩니다.

```
"""
src/ch7/zero_args.org.py
"""
from argparse import ArgumentParser
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import deepspeed
import torch.distributed as dist

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

parser = ArgumentParser()
parser.add_argument(
    "--deepspeed_config", default="../src/zero_dp_config.json", type=str
)
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=3e-7)

engine, optimizer, _, scheduler = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets]
data_loader = DataLoader(datasets, batch_size=8, num_workers=8)

for i, data in enumerate(data_loader):
    tokens = tokenizer(
        data,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    )

    loss = engine(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=tokens.input_ids.cuda(),
    ).loss

    engine.backward(loss)
    engine.step()

    if i % 10 == 0 and dist.get_rank() == 0:
        print(f"step:{i}, loss:{loss}")

    if i >= 300:
        break
```

```
(large-scale-lm) [gpu05]$ cd ch7; pwd
/scratch/qualis/git-projects/large-scale-lm-tutorials/src/ch7
(large-scale-lm) [gpu05]$ deepspeed --num_gpus=4 zero_args.org.py --deepspeed_config=zero_dp_config.org.json
[2024-09-15 00:13:05,386] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:13:07,115] [WARNING] [runner.py:212:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
Detected VISIBLE_DEVICES=0,1,2,3 but ignoring it because one or several of --include/--exclude/--num_gpus/--num_nodes cl args were used. If you want to use CUDA_VISIBLE_DEVICES don't pass any of these arguments to deepspeed.
[2024-09-15 00:13:07,116] [INFO] [runner.py:585:main] cmd = /scratch/qualis/miniconda3/envs/large-scale-lm/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None zero_args.org.py --deepspeed_config=zero_dp_config.org.json
[2024-09-15 00:13:08,380] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:13:10,097] [INFO] [launch.py:139:main] 0 NCCL_HOME=/scratch/qualis/nccl_2.11.4-1+cuda11.4_x86_64
[2024-09-15 00:13:10,097] [INFO] [launch.py:146:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2024-09-15 00:13:10,097] [INFO] [launch.py:152:main] nnodes=1, num_local_procs=4, node_rank=0
[2024-09-15 00:13:10,097] [INFO] [launch.py:163:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2024-09-15 00:13:10,097] [INFO] [launch.py:164:main] dist_world_size=4
[2024-09-15 00:13:10,097] [INFO] [launch.py:168:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
[2024-09-15 00:13:10,098] [INFO] [launch.py:256:main] process 23917 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_args.org.py', '--local_rank=0', '--deepspeed_config=zero_dp_config.org.json']
[2024-09-15 00:13:10,099] [INFO] [launch.py:256:main] process 23918 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_args.org.py', '--local_rank=1', '--deepspeed_config=zero_dp_config.org.json']
[2024-09-15 00:13:10,099] [INFO] [launch.py:256:main] process 23919 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_args.org.py', '--local_rank=2', '--deepspeed_config=zero_dp_config.org.json']
[2024-09-15 00:13:10,100] [INFO] [launch.py:256:main] process 23920 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_args.org.py', '--local_rank=3', '--deepspeed_config=zero_dp_config.org.json']
[2024-09-15 00:13:13,191] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:13:13,215] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:13:13,250] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:13:13,250] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:13:15,786] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:13:15,786] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-15 00:13:15,786] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:13:15,926] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:13:15,927] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-15 00:13:15,931] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:13:15,976] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:13:15,976] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-15 00:13:15,978] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:13:16,013] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:13:16,013] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-15 00:13:16,016] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
[2024-09-15 00:13:16,026] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
[2024-09-15 00:13:20,450] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2024-09-15 00:13:20,451] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2024-09-15 00:13:20,451] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2024-09-15 00:13:20,453] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = Adam
[2024-09-15 00:13:20,453] [INFO] [utils.py:59:is_zero_supported_optimizer] Checking ZeRO support for optimizer=Adam type=<class 'torch.optim.adam.Adam'>
[2024-09-15 00:13:20,453] [INFO] [logging.py:96:log_dist] [Rank 0] Creating torch.float16 ZeRO stage 1 optimizer
[2024-09-15 00:13:20,453] [INFO] [stage_1_and_2.py:148:__init__] Reduce bucket size 500000000
[2024-09-15 00:13:20,453] [INFO] [stage_1_and_2.py:149:__init__] Allgather bucket size 500000000
[2024-09-15 00:13:20,454] [INFO] [stage_1_and_2.py:150:__init__] CPU Offload: False
[2024-09-15 00:13:20,454] [INFO] [stage_1_and_2.py:151:__init__] Round robin gradient partitioning: False
[2024-09-15 00:13:21,213] [INFO] [utils.py:781:see_memory_usage] Before initializing optimizer states
[2024-09-15 00:13:21,214] [INFO] [utils.py:782:see_memory_usage] MA 0.36 GB         Max_MA 0.42 GB         CA 0.42 GB         Max_CA 0 GB
[2024-09-15 00:13:21,214] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 81.76 GB, percent = 8.1%
[2024-09-15 00:13:21,307] [INFO] [utils.py:781:see_memory_usage] After initializing optimizer states
[2024-09-15 00:13:21,307] [INFO] [utils.py:782:see_memory_usage] MA 0.36 GB         Max_MA 0.48 GB         CA 0.54 GB         Max_CA 1 GB
[2024-09-15 00:13:21,308] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 81.76 GB, percent = 8.1%
[2024-09-15 00:13:21,308] [INFO] [stage_1_and_2.py:543:__init__] optimizer state initialized
[2024-09-15 00:13:21,396] [INFO] [utils.py:781:see_memory_usage] After initializing ZeRO optimizer
[2024-09-15 00:13:21,396] [INFO] [utils.py:782:see_memory_usage] MA 0.36 GB         Max_MA 0.36 GB         CA 0.54 GB         Max_CA 1 GB
[2024-09-15 00:13:21,397] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 81.76 GB, percent = 8.1%
[2024-09-15 00:13:21,397] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer
[2024-09-15 00:13:21,397] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using configured LR scheduler = WarmupDecayLR
[2024-09-15 00:13:21,397] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = <deepspeed.runtime.lr_schedules.WarmupDecayLR object at 0x2acdfaf0a380>
[2024-09-15 00:13:21,397] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[3e-05], mom=[(0.9, 0.999)]
[2024-09-15 00:13:21,398] [INFO] [config.py:999:print] DeepSpeedEngine configuration:
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True, 'use_gds': False}
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   amp_enabled .................. False
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   amp_params ................... False
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   autotuning_config ............ {
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
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   bfloat16_enabled ............. False
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   bfloat16_immediate_grad_update  False
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   checkpoint_parallel_write_pipeline  False
[2024-09-15 00:13:21,398] [INFO] [config.py:1003:print]   checkpoint_tag_validation_enabled  True
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   checkpoint_tag_validation_fail  False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x2acdfaf0b250>
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   communication_data_type ...... None
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   curriculum_enabled_legacy .... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   curriculum_params_legacy ..... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   data_efficiency_enabled ...... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   dataloader_drop_last ......... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   disable_allgather ............ False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   dump_state ................... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   dynamic_loss_scale_args ...... {'init_scale': 4294967296, 'scale_window': 1000, 'delayed_shift': 2, 'consecutive_hysteresis': False, 'min_scale': 1}
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_enabled ........... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_gas_boundary_resolution  1
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_layer_num ......... 0
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_max_iter .......... 100
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_stability ......... 1e-06
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_tol ............... 0.01
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   eigenvalue_verbose ........... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   elasticity_enabled ........... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   fp16_auto_cast ............... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   fp16_enabled ................. True
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   fp16_master_weights_and_gradients  False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   global_rank .................. 0
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   grad_accum_dtype ............. None
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   gradient_accumulation_steps .. 1
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   gradient_clipping ............ 0.0
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   gradient_predivide_factor .... 1.0
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   graph_harvesting ............. False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   initial_dynamic_scale ........ 4294967296
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   load_universal_checkpoint .... False
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   loss_scale ................... 0
[2024-09-15 00:13:21,399] [INFO] [config.py:1003:print]   memory_breakdown ............. False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   mics_hierarchial_params_gather  False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   mics_shard_size .............. -1
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') comet=CometConfig(enabled=False, samples_log_interval=100, project=None, workspace=None, api_key=None, experiment_name=None, experiment_key=None, online=None, mode=None) wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName')
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   optimizer_legacy_fusion ...... False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   optimizer_name ............... None
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   optimizer_params ............. None
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   pld_enabled .................. False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   pld_params ................... False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   prescale_gradients ........... False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   scheduler_name ............... WarmupDecayLR
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   scheduler_params ............. {'total_num_steps': 300, 'warmup_min_lr': 0, 'warmup_max_lr': 3e-05, 'warmup_num_steps': 30}
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   seq_parallel_communication_data_type  torch.float32
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   sparse_attention ............. None
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   sparse_gradients_enabled ..... False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   steps_per_print .............. 9999999999
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   timers_config ................ enabled=True synchronized=True
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   train_batch_size ............. 16
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   train_micro_batch_size_per_gpu  4
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   use_data_before_expert_parallel_  False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   use_node_local_storage ....... False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   wall_clock_breakdown ......... False
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   weight_quantization_config ... None
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   world_size ................... 4
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   zero_allow_untested_optimizer  True
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   zero_config .................. stage=1 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500000000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1000000000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50000000 param_persistence_threshold=100000 model_persistence_threshold=9223372036854775807 max_live_parameters=1000000000 max_reuse_distance=1000000000 gather_16bit_weights_on_model_save=False use_all_reduce_for_fetch_params=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   zero_enabled ................. True
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   zero_force_ds_cpu_optimizer .. True
[2024-09-15 00:13:21,400] [INFO] [config.py:1003:print]   zero_optimization_stage ...... 1
[2024-09-15 00:13:21,400] [INFO] [config.py:989:print_user_config]   json = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": 300,
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-05,
            "warmup_num_steps": 30
        }
    },
    "fp16": {
        "enabled": true,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 1
    },
    "zero_allow_untested_optimizer": true,
    "wall_clock_breakdown": false,
    "steps_per_print": 1.000000e+10
}
[2024-09-15 00:13:28,261] [INFO] [loss_scaler.py:190:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4294967296, but hysteresis is 2. Reducing hysteresis to 1
step:0, loss:5.453125
[2024-09-15 00:13:28,307] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4294967296, reducing to 2147483648
[2024-09-15 00:13:28,340] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2147483648, reducing to 1073741824
[2024-09-15 00:13:28,383] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1073741824, reducing to 536870912
[2024-09-15 00:13:28,421] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 536870912, reducing to 268435456
[2024-09-15 00:13:28,461] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 268435456, reducing to 134217728
[2024-09-15 00:13:28,502] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 134217728, reducing to 67108864
[2024-09-15 00:13:28,541] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 67108864, reducing to 33554432
[2024-09-15 00:13:28,574] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 33554432, reducing to 16777216
[2024-09-15 00:13:28,623] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16777216, reducing to 8388608
[2024-09-15 00:13:28,665] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8388608, reducing to 4194304
step:10, loss:3.650390625
[2024-09-15 00:13:28,701] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4194304, reducing to 2097152
[2024-09-15 00:13:28,745] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2097152, reducing to 1048576
[2024-09-15 00:13:28,788] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1048576, reducing to 524288
[2024-09-15 00:13:28,825] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 524288, reducing to 262144
[2024-09-15 00:13:28,867] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072
[2024-09-15 00:13:28,899] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 131072, reducing to 65536
[2024-09-15 00:13:28,934] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 65536, reducing to 32768
[2024-09-15 00:13:28,981] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 32768, reducing to 16384
[2024-09-15 00:13:29,027] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16384, reducing to 8192
[2024-09-15 00:13:29,056] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8192, reducing to 4096
step:20, loss:3.546875
[2024-09-15 00:13:29,166] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4096, reducing to 2048
[2024-09-15 00:13:29,199] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2048, reducing to 1024
step:30, loss:3.71875
step:40, loss:2.87890625
step:50, loss:2.400390625
step:60, loss:2.5234375
step:70, loss:2.2578125
step:80, loss:2.5
step:90, loss:2.9375
step:100, loss:2.7890625
step:110, loss:2.484375
step:120, loss:2.955078125
step:130, loss:2.361328125
step:140, loss:2.92578125
step:150, loss:3.85546875
step:160, loss:3.044921875
step:170, loss:3.0546875
step:180, loss:1.65625
step:190, loss:3.5078125
step:200, loss:3.712890625
step:210, loss:3.556640625
step:220, loss:2.98046875
step:230, loss:3.25
step:240, loss:2.560546875
step:250, loss:3.19921875
step:260, loss:3.560546875
step:270, loss:3.240234375
step:280, loss:2.615234375
step:290, loss:2.2265625
step:300, loss:3.48828125
[2024-09-15 00:13:42,132] [INFO] [launch.py:351:main] Process 23917 exits successfully.
[2024-09-15 00:13:42,132] [INFO] [launch.py:351:main] Process 23919 exits successfully.
[2024-09-15 00:13:42,132] [INFO] [launch.py:351:main] Process 23920 exits successfully.
[2024-09-15 00:13:42,132] [INFO] [launch.py:351:main] Process 23918 exits successfully.
```

혹은 configuration을 `deepspeed.initialize()`에 직접 넣을 수도 있습니다.
```
"""
src/ch7/zero_config.org.py
"""
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import deepspeed
import torch.distributed as dist

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=3e-7)

engine, optimizer, _, scheduler = deepspeed.initialize(
    optimizer=optimizer,
    model=model,
    config={
        "train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": 300,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 30,
            },
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 32,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
        "steps_per_print": 9999999999,
    },
)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets]
data_loader = DataLoader(datasets, batch_size=8, num_workers=8)

for i, data in enumerate(data_loader):
    tokens = tokenizer(
        data,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    )

    loss = engine(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=tokens.input_ids.cuda(),
    ).loss

    engine.backward(loss)
    engine.step()

    if i % 10 == 0 and dist.get_rank() == 0:
        print(f"step:{i}, loss:{loss}")

    if i >= 300:
        break
```

```
[glogin01]$ deepspeed --num_gpus=4 zero_config.org.py
[2024-09-15 00:25:52,507] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:25:54,343] [WARNING] [runner.py:212:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
Detected VISIBLE_DEVICES=0,1,2,3 but ignoring it because one or several of --include/--exclude/--num_gpus/--num_nodes cl args were used. If you want to use CUDA_VISIBLE_DEVICES don't pass any of these arguments to deepspeed.
[2024-09-15 00:25:54,343] [INFO] [runner.py:585:main] cmd = /scratch/qualis/miniconda3/envs/large-scale-lm/bin/python -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 --enable_each_rank_log=None zero_config.org.py
[2024-09-15 00:25:55,571] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:25:57,292] [INFO] [launch.py:139:main] 0 NCCL_HOME=/scratch/qualis/nccl_2.11.4-1+cuda11.4_x86_64
[2024-09-15 00:25:57,292] [INFO] [launch.py:146:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2024-09-15 00:25:57,292] [INFO] [launch.py:152:main] nnodes=1, num_local_procs=4, node_rank=0
[2024-09-15 00:25:57,292] [INFO] [launch.py:163:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2024-09-15 00:25:57,292] [INFO] [launch.py:164:main] dist_world_size=4
[2024-09-15 00:25:57,292] [INFO] [launch.py:168:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
[2024-09-15 00:25:57,293] [INFO] [launch.py:256:main] process 32067 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_config.org.py', '--local_rank=0']
[2024-09-15 00:25:57,294] [INFO] [launch.py:256:main] process 32068 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_config.org.py', '--local_rank=1']
[2024-09-15 00:25:57,294] [INFO] [launch.py:256:main] process 32069 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_config.org.py', '--local_rank=2']
[2024-09-15 00:25:57,295] [INFO] [launch.py:256:main] process 32070 spawned with command: ['/scratch/qualis/miniconda3/envs/large-scale-lm/bin/python', '-u', 'zero_config.org.py', '--local_rank=3']
[2024-09-15 00:26:00,343] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:26:00,480] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:26:00,513] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-09-15 00:26:00,538] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:26:02,793] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:26:02,793] [INFO] [comm.py:652:init_distributed] cdb=None
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:26:03,237] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:26:03,237] [INFO] [comm.py:652:init_distributed] cdb=None
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:26:03,441] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:26:03,441] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-15 00:26:03,442] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
[2024-09-15 00:26:03,492] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.15.1, git-hash=unknown, git-branch=unknown
[2024-09-15 00:26:03,492] [INFO] [comm.py:652:init_distributed] cdb=None
[2024-09-15 00:26:03,494] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
[2024-09-15 00:26:03,796] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
[2024-09-15 00:26:04,240] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
[2024-09-15 00:26:04,243] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 4
[2024-09-15 00:26:07,009] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2024-09-15 00:26:07,010] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2024-09-15 00:26:07,010] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2024-09-15 00:26:07,012] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = Adam
[2024-09-15 00:26:07,013] [INFO] [utils.py:59:is_zero_supported_optimizer] Checking ZeRO support for optimizer=Adam type=<class 'torch.optim.adam.Adam'>
[2024-09-15 00:26:07,013] [INFO] [logging.py:96:log_dist] [Rank 0] Creating torch.float16 ZeRO stage 1 optimizer
[2024-09-15 00:26:07,013] [INFO] [stage_1_and_2.py:148:__init__] Reduce bucket size 500000000
[2024-09-15 00:26:07,013] [INFO] [stage_1_and_2.py:149:__init__] Allgather bucket size 500000000
[2024-09-15 00:26:07,013] [INFO] [stage_1_and_2.py:150:__init__] CPU Offload: False
[2024-09-15 00:26:07,013] [INFO] [stage_1_and_2.py:151:__init__] Round robin gradient partitioning: False
[2024-09-15 00:26:07,560] [INFO] [utils.py:781:see_memory_usage] Before initializing optimizer states
[2024-09-15 00:26:07,561] [INFO] [utils.py:782:see_memory_usage] MA 0.36 GB         Max_MA 0.42 GB         CA 0.42 GB         Max_CA 0 GB
[2024-09-15 00:26:07,561] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 82.11 GB, percent = 8.2%
[2024-09-15 00:26:07,736] [INFO] [utils.py:781:see_memory_usage] After initializing optimizer states
[2024-09-15 00:26:07,737] [INFO] [utils.py:782:see_memory_usage] MA 0.36 GB         Max_MA 0.48 GB         CA 0.54 GB         Max_CA 1 GB
[2024-09-15 00:26:07,737] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 82.1 GB, percent = 8.2%
[2024-09-15 00:26:07,737] [INFO] [stage_1_and_2.py:543:__init__] optimizer state initialized
[2024-09-15 00:26:07,899] [INFO] [utils.py:781:see_memory_usage] After initializing ZeRO optimizer
[2024-09-15 00:26:07,899] [INFO] [utils.py:782:see_memory_usage] MA 0.36 GB         Max_MA 0.36 GB         CA 0.54 GB         Max_CA 1 GB
[2024-09-15 00:26:07,900] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 82.1 GB, percent = 8.2%
[2024-09-15 00:26:07,900] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer
[2024-09-15 00:26:07,901] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using configured LR scheduler = WarmupDecayLR
[2024-09-15 00:26:07,901] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = <deepspeed.runtime.lr_schedules.WarmupDecayLR object at 0x2b42dfc4b280>
[2024-09-15 00:26:07,901] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[3e-05], mom=[(0.9, 0.999)]
[2024-09-15 00:26:07,901] [INFO] [config.py:999:print] DeepSpeedEngine configuration:
[2024-09-15 00:26:07,901] [INFO] [config.py:1003:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True, 'use_gds': False}
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   amp_enabled .................. False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   amp_params ................... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   autotuning_config ............ {
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
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   bfloat16_enabled ............. False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   bfloat16_immediate_grad_update  False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   checkpoint_parallel_write_pipeline  False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   checkpoint_tag_validation_enabled  True
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   checkpoint_tag_validation_fail  False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x2b42e0c2ca60>
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   communication_data_type ...... None
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   curriculum_enabled_legacy .... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   curriculum_params_legacy ..... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   data_efficiency_enabled ...... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   dataloader_drop_last ......... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   disable_allgather ............ False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   dump_state ................... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   dynamic_loss_scale_args ...... {'init_scale': 4294967296, 'scale_window': 1000, 'delayed_shift': 2, 'consecutive_hysteresis': False, 'min_scale': 1}
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_enabled ........... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_gas_boundary_resolution  1
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_layer_num ......... 0
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_max_iter .......... 100
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_stability ......... 1e-06
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_tol ............... 0.01
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   eigenvalue_verbose ........... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   elasticity_enabled ........... False
[2024-09-15 00:26:07,902] [INFO] [config.py:1003:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   fp16_auto_cast ............... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   fp16_enabled ................. True
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   fp16_master_weights_and_gradients  False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   global_rank .................. 0
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   grad_accum_dtype ............. None
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   gradient_accumulation_steps .. 1
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   gradient_clipping ............ 0.0
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   gradient_predivide_factor .... 1.0
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   graph_harvesting ............. False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   initial_dynamic_scale ........ 4294967296
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   load_universal_checkpoint .... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   loss_scale ................... 0
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   memory_breakdown ............. False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   mics_hierarchial_params_gather  False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   mics_shard_size .............. -1
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') comet=CometConfig(enabled=False, samples_log_interval=100, project=None, workspace=None, api_key=None, experiment_name=None, experiment_key=None, online=None, mode=None) wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName')
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   optimizer_legacy_fusion ...... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   optimizer_name ............... None
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   optimizer_params ............. None
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   pld_enabled .................. False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   pld_params ................... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   prescale_gradients ........... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   scheduler_name ............... WarmupDecayLR
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   scheduler_params ............. {'total_num_steps': 300, 'warmup_min_lr': 0, 'warmup_max_lr': 3e-05, 'warmup_num_steps': 30}
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   seq_parallel_communication_data_type  torch.float32
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   sparse_attention ............. None
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   sparse_gradients_enabled ..... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   steps_per_print .............. 9999999999
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   timers_config ................ enabled=True synchronized=True
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   train_batch_size ............. 16
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   train_micro_batch_size_per_gpu  4
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   use_data_before_expert_parallel_  False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   use_node_local_storage ....... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   wall_clock_breakdown ......... False
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   weight_quantization_config ... None
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   world_size ................... 4
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   zero_allow_untested_optimizer  True
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   zero_config .................. stage=1 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500000000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1000000000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50000000 param_persistence_threshold=100000 model_persistence_threshold=9223372036854775807 max_live_parameters=1000000000 max_reuse_distance=1000000000 gather_16bit_weights_on_model_save=False use_all_reduce_for_fetch_params=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   zero_enabled ................. True
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   zero_force_ds_cpu_optimizer .. True
[2024-09-15 00:26:07,903] [INFO] [config.py:1003:print]   zero_optimization_stage ...... 1
[2024-09-15 00:26:07,903] [INFO] [config.py:989:print_user_config]   json = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": 300,
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-05,
            "warmup_num_steps": 30
        }
    },
    "fp16": {
        "enabled": true,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": true,
        "allgather_bucket_size": 5.000000e+08,
        "overlap_comm": false,
        "reduce_scatter": true,
        "reduce_bucket_size": 5.000000e+08,
        "contiguous_gradients": true
    },
    "zero_allow_untested_optimizer": true,
    "wall_clock_breakdown": false,
    "steps_per_print": 1.000000e+10
}
[2024-09-15 00:26:14,626] [INFO] [loss_scaler.py:190:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4294967296, but hysteresis is 2. Reducing hysteresis to 1
step:0, loss:5.453125
[2024-09-15 00:26:14,672] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4294967296, reducing to 2147483648
[2024-09-15 00:26:14,705] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2147483648, reducing to 1073741824
[2024-09-15 00:26:14,749] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1073741824, reducing to 536870912
[2024-09-15 00:26:14,789] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 536870912, reducing to 268435456
[2024-09-15 00:26:14,831] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 268435456, reducing to 134217728
[2024-09-15 00:26:14,874] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 134217728, reducing to 67108864
[2024-09-15 00:26:14,912] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 67108864, reducing to 33554432
[2024-09-15 00:26:14,946] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 33554432, reducing to 16777216
[2024-09-15 00:26:14,995] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16777216, reducing to 8388608
[2024-09-15 00:26:15,036] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8388608, reducing to 4194304
step:10, loss:3.650390625
[2024-09-15 00:26:15,072] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4194304, reducing to 2097152
[2024-09-15 00:26:15,116] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2097152, reducing to 1048576
[2024-09-15 00:26:15,159] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1048576, reducing to 524288
[2024-09-15 00:26:15,196] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 524288, reducing to 262144
[2024-09-15 00:26:15,237] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072
[2024-09-15 00:26:15,269] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 131072, reducing to 65536
[2024-09-15 00:26:15,304] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 65536, reducing to 32768
[2024-09-15 00:26:15,351] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 32768, reducing to 16384
[2024-09-15 00:26:15,397] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16384, reducing to 8192
[2024-09-15 00:26:15,426] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8192, reducing to 4096
step:20, loss:3.546875
[2024-09-15 00:26:15,535] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4096, reducing to 2048
[2024-09-15 00:26:15,569] [INFO] [loss_scaler.py:183:update_scale] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2048, reducing to 1024
step:30, loss:3.71875
step:40, loss:2.87890625
step:50, loss:2.400390625
step:60, loss:2.525390625
step:70, loss:2.259765625
step:80, loss:2.5
step:90, loss:2.94140625
step:100, loss:2.7890625
step:110, loss:2.484375
step:120, loss:2.955078125
step:130, loss:2.359375
step:140, loss:2.92578125
step:150, loss:3.853515625
step:160, loss:3.044921875
step:170, loss:3.056640625
step:180, loss:1.6572265625
step:190, loss:3.505859375
step:200, loss:3.71484375
step:210, loss:3.556640625
step:220, loss:2.98046875
step:230, loss:3.25
step:240, loss:2.5625
step:250, loss:3.19921875
step:260, loss:3.560546875
step:270, loss:3.23828125
step:280, loss:2.6171875
step:290, loss:2.2265625
step:300, loss:3.48828125
[2024-09-15 00:26:28,325] [INFO] [launch.py:351:main] Process 32069 exits successfully.
[2024-09-15 00:26:28,325] [INFO] [launch.py:351:main] Process 32070 exits successfully.
[2024-09-15 00:26:28,325] [INFO] [launch.py:351:main] Process 32067 exits successfully.
[2024-09-15 00:26:28,325] [INFO] [launch.py:351:main] Process 32068 exits successfully.
```
 
## 4. Activation Checkpointing
    
FP 16과 32의 model, gradient, optimizer state 이외에 또 하나의 큰 메모리 영역은 Activation Memory 영역입니다. Activation은 model weight에 곱해지는 입력텐서들을 의미하는데요. 만약 $y = w_1 \\cdot (w_2 \\cdot x)$와 같은 뉴럴넷이 있다면, $w_1$과 곱해지는 $x$와 $w_2$와 곱해지는 $w_2 \\cdot x$ 등의 텐서들이 Activation Memory에 해당합니다. 
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

![](../images/max_pooling.png)
    
이전에 Pipeline parallelism 세션에서 Backward 패스시에 Forward 때 사용했던 Activation 텐서를 저장한다고 말씀드린적 있습니다. 위와 같이 Maxpooling 레이어의 미분계수를 구하려면 pooling된 값들의 원래 위치가 필요하므로 반드시 Forward 때 입력되었던 텐서가 필요합니다. 또한 위의 `ReLU` 구현을 보면 `ctx.save_for_backward`를 통해 `input` 텐서를 저장하고 있는 것을 볼 수 있습니다.
   
![](../images/checkpoint_full_act.gif)
    
**즉, Backward 단계를 수행하기 위해 Forward 단계의 입력들을 저장해야 합니다.** 위는 그것을 영상으로 보여줍니다.  그러나 이렇게 모든 곳에서 Activation을 저장하고 있으면 메모리 소비량이 매우 커집니다.
    
![](../images/checkpoint_no_act.gif)
    
따라서 Activation 텐서를 저장하지 않는다면 메모리 소비량을 훨씬 아낄 수 있습니다. 그러나 Activation 텐서를 저장하지 않으면, 위와 같이 Backward 시점에 Forward를 한번 더 해서 Activation 텐서를 구해야합니다. Activation Checkpointing은 두가지 장점을 결합한 방식으로 중간 중간마다 Activation을 저장해둡니다.
   
![](../images/checkpoint_act.gif)
    
위와 같이 중간 중간에만 저장을 하게 되면 매번 Forwad를 처음부터 하지 않고 중간부터 수행하게 하여 연산 시간을 아낄 수 있고, 거의 대부분의 Activation을 제거함으로써 메모리 소비량을 크게 줄일 수 있습니다. 이렇게 **Activation를 중간 중간마다 저장** 해놓고 Forward가 필요하면 체크포인트 된 곳부터 Forward를 수행해나가게끔 하는 기법을 Activation Checkpointing이라고 합니다. 파이토치에는 이미 checkpointing 기능이 내장되어 있습니다. pytorch를 이용해서 실습해봅시다.

```
"""
src/checkpointing.py
"""
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import BertTokenizer, BertLayer, BertConfig

config = BertConfig.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer("Hello I am Kevin", return_tensors="pt")

embedding = nn.Embedding(tokenizer.vocab_size, config.hidden_size)
layers = nn.ModuleList([BertLayer(config) for _ in range(6)])

hidden_states = embedding(tokens.input_ids)
attention_mask = tokens.attention_mask

for i, layer_module in enumerate(layers):
    layer_outputs = checkpoint(
        layer_module,
        hidden_states,
        attention_mask,
    )

    hidden_states = layer_outputs[0]

print(f"output: {hidden_states}")
```
output: tensor([[[-1.4583,  0.6489, -1.3392,  ..., -0.6039,  0.2626,  0.6193],
         [ 1.2056, -1.7527,  1.4104,  ..., -0.1405, -0.9028, -1.6564],
         [-1.8641,  0.6331, -0.3740,  ..., -0.1908, -0.4829, -0.6025],
         [ 0.8196,  1.9792,  0.1852,  ...,  0.8961,  0.6273, -1.2254],
         [ 0.7911, -0.3338, -0.7460,  ...,  0.6872, -1.0973,  1.7147],
         [-1.4739, -1.9196, -0.4886,  ..., -1.6297, -0.0368,  1.1412]]],
       grad_fn=<CheckpointFunctionBackward>)

 
사용법은 위 예제처럼 기존에 **`module(a, b, c)`와 같이 사용하던 것을 `checkpoint(module, a, b, c)`와 같이 변경**하기만 하면 끝입니다. 
   
또한 우리가 자주 사용하는 Hugging Face `transformers`에도 거의 대부분 모델에 이러한 Activation Checkpointing 기능이 탑재되어 있습니다. **단순히 `model.gradient_checkpointing_enable()`와 `model.gradient_checkpointing_disable()`으로 켜고 끌 수 있습니다.** 정말 쉽죠?


## 5. ZeRO-R
   
ZeRO-R은 Activation Memory, Communication Bucket 등의 영역을 고도로 최적화 하기 위한 기술들의 집합입니다.
   
![](../images/zero_r_1.png)
    
이전 챕터에서 알아본 ZeRO-DP를 통해 Model state memory (FP16 & 32 Parameters, Gradient, Optimizer States)를 효율적으로 개선했습니다. ZeRO-R에서는 다음과 같은 세가지 솔루션을 제안합니다.
- **Activation Memory Partitioning**
- **Constant Size Buffer**
- **Memory Defragmentation**
  
각각에 대해 알아보겠습니다.
 
### 1) Activation Memory Partitioning
   
![](../images/zero_r_2.png)
    
Activation Checkpointing이 메모리 효율성과 속도 향상에 도움이 될 수도 있지만, 큰 모델을 학습할 때는 상당한 메모리 문제를 야기할 수 있습니다. 특히 모델 병렬화와 결합될 경우 Forward가 계산되고나서 여기저기에 Activation Tensor의 사본들이 많이 생겨나게 됩니다. **ZeRO-R은 이러한 Activation Tensor들을 All-gather하여 그 중 필요한 것들만 추려서 GPU로 Partitioning합니다.** 또한 너무 커다란 Activation 들은 속도를 약간 희생하더라도 CPU RAM에 Checkpointing 시켜서 GPU 메모리를 절약합니다.

### 2) Constant Memory Buffer
    
Constant Memory Buffer는 All-reduce, All-gather 등에 사용되는 **버킷의 사이즈를 Constant하게 유지하는 기법**을 의미합니다. 일반적으로 모델이 커질수록 통신에 사용하는 Bucket도 함께 커지는 것이 좋습니다. 그러나 모델의 크기가 매우 커지면 Buffer의 사이즈가 너무 커져서 GPU의 상당부분을 차지하게 되는 경우도 있습니다. 따라서 **Bucket 사이즈의 최대값을 제한하여 고정된 값보다 더 크게는 할당되지 않도록** 합니다. Bucket의 사이즈가 일정 수준 이상으로 커지면 더 키우지 않고 유지만해도 충분히 좋은 효율성을 얻을 수 있습니다.
  
### 3) Memory Defragmentation (Contiguous Checkpointing)
    
![](../images/zero_r_3.jpeg)
    
모델을 학습하다보면 텐서들이 많이 생겨나고 제거됨에 따라 **GPU 메모리의 단편화가 매우 자주 발생**합니다. 때로는 GPU 내의 용량이 충분히 많지만 공간이 단편화 되어있어서 Contiguous한 텐서를 올리지 못하는 문제가 발생할 수 있습니다. 따라서 ZeRO-R은 **빈 공간에 Activation, Gradient 등을 담을 수 있는 빈 메모리 공간을 미리 만들어두고** 비슷한 사이즈의 텐서들이 생성되면 **해당 공간으로 옮겨서 단편화**를 최대한 방지합니다.
  
ZeRO-DP와 마찬가지로 간단한 Configuration만 작성하면 됩니다. 
- **Constant Buffer Size** 
  - `allgather_bucket_size`와 `reduce_bucket_size`를 통해 버킷 사이즈의 최대값을 결정하였음.
- **Activation Memory** 
  - `partition_activations`을 통해 activation 메모리의 GPU 간에 분할함.
  - `cpu_checkpointing`을 통해 매우 큰 activation 텐서는 CPU로 오프로드
- **Memory Defragmentation**:
  - `contiguous_memory_optimization`를 통해 메모리 단편화 완화.
   
이러한 기법들 외에도 수 많은 기법이 존재합니다. 더 다양한 옵션들에 대해 자세히 알고 싶으시면 논문과 도큐먼트를 참고하세요.
```
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 300,
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 30
    }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 1,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4
  },
  "zero_allow_untested_optimizer": true,
  "wall_clock_breakdown": false,
  "steps_per_print": 9999999999
}
```
```
[glogin01]$ deepspeed --num_gpus=4 ../src/zero_args.py --deepspeed_config=../src/zero_r_config.json
[2021-10-27 22:30:25,615] [WARNING] [runner.py:122:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2021-10-27 22:30:25,791] [INFO] [runner.py:360:main] cmd = /usr/bin/python3 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 ../src/zero_args.py --deepspeed_config=../src/zero_r_config.json
[2021-10-27 22:30:26,909] [INFO] [launch.py:80:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2021-10-27 22:30:26,909] [INFO] [launch.py:89:main] nnodes=1, num_local_procs=4, node_rank=0
[2021-10-27 22:30:26,909] [INFO] [launch.py:101:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2021-10-27 22:30:26,909] [INFO] [launch.py:102:main] dist_world_size=4
[2021-10-27 22:30:26,910] [INFO] [launch.py:105:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
[2021-10-27 22:30:32,066] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 22:30:32,069] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 22:30:32,126] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 22:30:32,129] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 22:30:32,144] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 22:30:32,148] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 22:30:32,153] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 22:30:32,156] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 22:30:37,512] [INFO] [logging.py:68:log_dist] [Rank 0] initializing deepspeed groups
[2021-10-27 22:30:37,512] [INFO] [logging.py:68:log_dist] [Rank 0] initializing deepspeed model parallel group with size 1
[2021-10-27 22:30:37,517] [INFO] [logging.py:68:log_dist] [Rank 0] initializing deepspeed expert parallel group with size 1
[2021-10-27 22:30:37,517] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert data parallel process group with ranks: [0, 1, 2, 3]
[2021-10-27 22:30:37,517] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [0]
[2021-10-27 22:30:37,518] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [1]
[2021-10-27 22:30:37,518] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [2]
[2021-10-27 22:30:37,518] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [3]
[2021-10-27 22:30:37,781] [INFO] [engine.py:205:__init__] DeepSpeed Flops Profiler Enabled: False
[2021-10-27 22:30:37,781] [INFO] [engine.py:849:_configure_optimizer] Removing param_group that has no 'params' in the client Optimizer
[2021-10-27 22:30:37,781] [INFO] [engine.py:854:_configure_optimizer] Using client Optimizer as basic optimizer
[2021-10-27 22:30:37,786] [INFO] [engine.py:871:_configure_optimizer] DeepSpeed Basic Optimizer = Adam
[2021-10-27 22:30:37,786] [INFO] [utils.py:44:is_zero_supported_optimizer] Checking ZeRO support for optimizer=Adam type=<class 'torch.optim.adam.Adam'>
[2021-10-27 22:30:37,786] [INFO] [logging.py:68:log_dist] [Rank 0] Creating fp16 ZeRO stage 1 optimizer
[2021-10-27 22:30:37,786] [INFO] [stage2.py:111:__init__] Reduce bucket size 500000000.0
[2021-10-27 22:30:37,786] [INFO] [stage2.py:112:__init__] Allgather bucket size 500000000.0
[2021-10-27 22:30:37,786] [INFO] [stage2.py:113:__init__] CPU Offload: False
[2021-10-27 22:30:37,786] [INFO] [stage2.py:114:__init__] Round robin gradient partitioning: False
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Emitting ninja build file /home/ubuntu/.cache/torch_extensions/utils/build.ninja...
Building extension module utils...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module utils...
Time to load utils op: 0.34302282333374023 seconds
Loading extension module utils...
Time to load utils op: 0.40213775634765625 seconds
Loading extension module utils...
Time to load utils op: 0.4021179676055908 seconds
Loading extension module utils...
Time to load utils op: 0.4021291732788086 seconds
Rank: 0 partition count [4] and sizes[(31109952, False)] 
Rank: 3 partition count [4] and sizes[(31109952, False)] 
Rank: 1 partition count [4] and sizes[(31109952, False)] 
Rank: 2 partition count [4] and sizes[(31109952, False)] 
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Loading extension module utils...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.00044083595275878906 seconds
Time to load utils op: 0.0004825592041015625 seconds
Time to load utils op: 0.00045371055603027344 seconds
[2021-10-27 22:30:39,142] [INFO] [utils.py:806:see_memory_usage] Before initializing optimizer states
[2021-10-27 22:30:39,142] [INFO] [utils.py:811:see_memory_usage] MA 0.36 GB         Max_MA 0.42 GB         CA 0.61 GB         Max_CA 1 GB 
[2021-10-27 22:30:39,143] [INFO] [utils.py:816:see_memory_usage] CPU Virtual Memory:  used = 16.61 GB, percent = 6.9%
[2021-10-27 22:30:39,182] [INFO] [utils.py:806:see_memory_usage] After initializing optimizer states
[2021-10-27 22:30:39,183] [INFO] [utils.py:811:see_memory_usage] MA 0.59 GB         Max_MA 1.06 GB         CA 1.31 GB         Max_CA 1 GB 
[2021-10-27 22:30:39,183] [INFO] [utils.py:816:see_memory_usage] CPU Virtual Memory:  used = 16.61 GB, percent = 6.9%
[2021-10-27 22:30:39,183] [INFO] [stage2.py:474:__init__] optimizer state initialized
[2021-10-27 22:30:39,219] [INFO] [utils.py:806:see_memory_usage] After initializing ZeRO optimizer
[2021-10-27 22:30:39,220] [INFO] [utils.py:811:see_memory_usage] MA 0.59 GB         Max_MA 0.59 GB         CA 1.31 GB         Max_CA 1 GB 
[2021-10-27 22:30:39,220] [INFO] [utils.py:816:see_memory_usage] CPU Virtual Memory:  used = 16.61 GB, percent = 6.9%
[2021-10-27 22:30:39,221] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed Final Optimizer = Adam
[2021-10-27 22:30:39,221] [INFO] [engine.py:587:_configure_lr_scheduler] DeepSpeed using configured LR scheduler = WarmupDecayLR
[2021-10-27 22:30:39,221] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed LR Scheduler = <deepspeed.runtime.lr_schedules.WarmupDecayLR object at 0x7f48927a1518>
[2021-10-27 22:30:39,221] [INFO] [logging.py:68:log_dist] [Rank 0] step=0, skipped=0, lr=[3e-05], mom=[(0.9, 0.999)]
[2021-10-27 22:30:39,221] [INFO] [config.py:940:print] DeepSpeedEngine configuration:
[2021-10-27 22:30:39,221] [INFO] [config.py:944:print]   activation_checkpointing_config  {
    "partition_activations": true, 
    "contiguous_memory_optimization": true, 
    "cpu_checkpointing": true, 
    "number_checkpoints": 4, 
    "synchronize_checkpoint_boundary": false, 
    "profile": false
}
[2021-10-27 22:30:39,221] [INFO] [config.py:944:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2021-10-27 22:30:39,221] [INFO] [config.py:944:print]   allreduce_always_fp32 ........ False
[2021-10-27 22:30:39,221] [INFO] [config.py:944:print]   amp_enabled .................. False
[2021-10-27 22:30:39,221] [INFO] [config.py:944:print]   amp_params ................... False
[2021-10-27 22:30:39,221] [INFO] [config.py:944:print]   checkpoint_tag_validation_enabled  True
[2021-10-27 22:30:39,221] [INFO] [config.py:944:print]   checkpoint_tag_validation_fail  False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   curriculum_enabled ........... False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   curriculum_params ............ False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   dataloader_drop_last ......... False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   disable_allgather ............ False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   dump_state ................... False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   dynamic_loss_scale_args ...... {'init_scale': 4294967296, 'scale_window': 1000, 'delayed_shift': 2, 'min_scale': 1}
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_enabled ........... False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_gas_boundary_resolution  1
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_layer_num ......... 0
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_max_iter .......... 100
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_stability ......... 1e-06
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_tol ............... 0.01
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   eigenvalue_verbose ........... False
[2021-10-27 22:30:39,222] [INFO] [config.py:944:print]   elasticity_enabled ........... False
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   flops_profiler_config ........ {
    "enabled": false, 
    "profile_step": 1, 
    "module_depth": -1, 
    "top_modules": 1, 
    "detailed": true, 
    "output_file": null
}
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   fp16_enabled ................. True
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   fp16_master_weights_and_gradients  False
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   fp16_mixed_quantize .......... False
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   global_rank .................. 0
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   gradient_accumulation_steps .. 1
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   gradient_clipping ............ 0.0
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   gradient_predivide_factor .... 1.0
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   initial_dynamic_scale ........ 4294967296
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   loss_scale ................... 0
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   memory_breakdown ............. False
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   optimizer_legacy_fusion ...... False
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   optimizer_name ............... None
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   optimizer_params ............. None
[2021-10-27 22:30:39,224] [INFO] [config.py:944:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0}
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   pld_enabled .................. False
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   pld_params ................... False
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   prescale_gradients ........... False
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_change_rate ......... 0.001
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_groups .............. 1
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_offset .............. 1000
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_period .............. 1000
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_rounding ............ 0
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_start_bits .......... 16
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_target_bits ......... 8
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_training_enabled .... False
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_type ................ 0
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   quantize_verbose ............. False
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   scheduler_name ............... WarmupDecayLR
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   scheduler_params ............. {'total_num_steps': 300, 'warmup_min_lr': 0, 'warmup_max_lr': 3e-05, 'warmup_num_steps': 30}
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   sparse_attention ............. None
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   sparse_gradients_enabled ..... False
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   steps_per_print .............. 9999999999
[2021-10-27 22:30:39,225] [INFO] [config.py:944:print]   tensorboard_enabled .......... False
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   tensorboard_job_name ......... DeepSpeedJobName
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   tensorboard_output_path ...... 
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   train_batch_size ............. 16
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   train_micro_batch_size_per_gpu  4
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   use_quantizer_kernel ......... False
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   wall_clock_breakdown ......... False
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   world_size ................... 4
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   zero_allow_untested_optimizer  True
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   zero_config .................. {
    "stage": 1, 
    "contiguous_gradients": true, 
    "reduce_scatter": true, 
    "reduce_bucket_size": 5.000000e+08, 
    "allgather_partitions": true, 
    "allgather_bucket_size": 5.000000e+08, 
    "overlap_comm": false, 
    "load_from_fp32_weights": true, 
    "elastic_checkpoint": true, 
    "offload_param": null, 
    "offload_optimizer": null, 
    "sub_group_size": 1.000000e+09, 
    "prefetch_bucket_size": 5.000000e+07, 
    "param_persistence_threshold": 1.000000e+05, 
    "max_live_parameters": 1.000000e+09, 
    "max_reuse_distance": 1.000000e+09, 
    "gather_fp16_weights_on_model_save": false, 
    "ignore_unused_parameters": true, 
    "round_robin_gradients": false, 
    "legacy_stage1": false
}
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   zero_enabled ................. True
[2021-10-27 22:30:39,226] [INFO] [config.py:944:print]   zero_optimization_stage ...... 1
[2021-10-27 22:30:39,227] [INFO] [config.py:952:print]   json = {
    "train_batch_size": 16, 
    "gradient_accumulation_steps": 1, 
    "scheduler": {
        "type": "WarmupDecayLR", 
        "params": {
            "total_num_steps": 300, 
            "warmup_min_lr": 0, 
            "warmup_max_lr": 3e-05, 
            "warmup_num_steps": 30
        }
    }, 
    "fp16": {
        "enabled": true, 
        "initial_scale_power": 32, 
        "loss_scale_window": 1000, 
        "hysteresis": 2, 
        "min_loss_scale": 1
    }, 
    "zero_optimization": {
        "stage": 1, 
        "allgather_bucket_size": 5.000000e+08, 
        "reduce_bucket_size": 5.000000e+08
    }, 
    "activation_checkpointing": {
        "partition_activations": true, 
        "cpu_checkpointing": true, 
        "contiguous_memory_optimization": true, 
        "number_checkpoints": 4
    }, 
    "zero_allow_untested_optimizer": true, 
    "wall_clock_breakdown": false, 
    "steps_per_print": 1.000000e+10
}
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.0004620552062988281 seconds
Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 565.38it/s]
Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 609.24it/s]
Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 550.14it/s]
Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 549.78it/s]
step:0, loss:5.453125
step:10, loss:3.6484375
step:20, loss:3.546875
step:30, loss:3.76953125
step:40, loss:2.880859375
step:50, loss:2.408203125
step:60, loss:2.5234375
step:70, loss:2.265625
step:80, loss:2.505859375
step:90, loss:2.939453125
step:100, loss:2.791015625
step:110, loss:2.48828125
step:120, loss:2.95703125
step:130, loss:2.361328125
step:140, loss:2.92578125
step:150, loss:3.8515625
step:160, loss:3.044921875
step:170, loss:3.052734375
step:180, loss:1.65625
step:190, loss:3.509765625
step:200, loss:3.716796875
step:210, loss:3.560546875
step:220, loss:2.98046875
step:230, loss:3.251953125
step:240, loss:2.564453125
step:250, loss:3.19921875
step:260, loss:3.564453125
step:270, loss:3.23828125
step:280, loss:2.615234375
step:290, loss:2.23046875
step:300, loss:3.48828125
```

## 6. ZeRO Offload
  
이전챕터에서 봤던 `Activation Memory Partitioning` 기술은 너무 큰 Activation 텐서를 CPU로 내리는 기능을 포함하였습니다. ZeRO-R의 후속작인 Zero Offload는 **Model의 일부분을 CPU RAM으로 오프로드 시키는 방법**을 통해 GPU의 용량 한계를 깨부술 수 있었습니다. ZeRO Offload의 핵심 아이디어는 다음과 같습니다.
    
![](../images/zero_off_1.png)
   
#### GPU-side
- GPU에 FP16 Parameter & Gradients가 상주한다.
- GPU에서 그들을 이용해 Forward & Backward를 수행한다. (무거운 연산이기 때문)
    
#### CPU-side
- CPU에 FP32 Paramter & Gradient & Optimizer States가 상주한다.
- CPU에서 Weight Update를 수행한다. (가벼운 연산이기 때문)
- 특히 CPU에서 매우 빠르게 작동할 수 있는 CPU Adam 옵티마이저를 구현했다.
  
일반적으로 CPU의 처리속도는 GPU의 처리속도에 비해 수십배는 느립니다. 따라서 아주 큰 Computation은 반드시 GPU에서 수행해야 합니다. 이러한 이유로 Forward & Backward 연산은 GPU에서 수행합니다. 생각해보면 GPU의 대부분을 FP32의 Parameter & Gradient & Optimizer States가 차지하는데, 정작 그들이 수행하는 연산은 Computation Cost가 적은 Weight Update 파트입니다.

![](../images/ddp_analysis_3.png)
    
따라서 FP32 부분을 모두 CPU로 내려버리면 GPU는 정말 GPU 연산이 반드시 필요한 FP16만 남기 때문에 매우 널널한 상태가 됩니다. 
    
### DPU: Delayed Paramter Update
   
![](../images/zero_off_2.png)
    
GPU에서 Forward & Backward이 모두 완료되고나서 CPU로 보내기 시작하면 통신하는 시간동안 GPU가 기다려야 합니다. ZeRO Offload는 Delayed Paramter Update(DPU)라는 기법을 도입했는데, 이는 DDP의 Gradient Bucketing이 그랬던 것과 비슷하게 **Communication과 Computation을 오버랩해서 전체 처리 시간을 단축**시키는 전략입니다.
   
![](../images/zero_off_3.png)
    
실험 결과 DPU를 적용해도 성능에는 문제가 없었으며, 속도를 다소 개선 할 수 있었다고 합니다.
 
### ZeRO Offload + ZeRO DP
    
![](../images/zero_off_4.png)
    
ZeRO Offload 기술은 ZeRO DP와 결합할 수 있습니다. 만약 ZeRO-DP를 적용한 상태로 Optimizer States와 Gradient를 CPU로 Offload하면 위와 같은 형태를 띄게 됩니다. 참고로 ZeRO DP와 Offload 간의 결합은 stage 2부터 가능하며, 파라미터까지 Offload 시키려면 ZeRO stage를 3로 설정해야 합니다.
    
- **ZeRO stage 2**: Optimizer States Offload
- **ZeRO stage 3**: Optimizer States + Parameter Offlaod
 
### CPU Adam
   
현재의 Adam Optimizer는 GPU에서 최적화 되어있기 때문에 CPU에서 동작시키면 다소 느린 것이 사실입니다. 다양한 최적화 기법들을 적용하여 CPU에서 매우 빠르게 동작하는 Adam Optimizer를 제공합니다. CPU Adam의 구현은 머신러닝이나 분산처리 분야가 아닌 거의 컴퓨터 아키텍처나 운영체제에 가까운 영역이라서 본 자료에서 자세히 다루지 않겠습니다. 더 자세한 내용은 논문을 참고해주세요. (사실 저도 이 부분은 자세 안보고 넘어가서 잘 모릅니다.. 이 부분 깊게 공부하신 분 계시면 이슈로 알려주세요.)

![](../images/cpu_adam.png)

ZeRO Offload를 실습해봅시다. 마찬가지로 먼저 Configuration을 변경합니다. Optimizer와 Parameter를 모두 Offload 하기 위해서 ZeRO stage를 3으로 설정하였으며 `offload_param`와 `offload_optimizer`을 추가하였습니다.

```
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 300,
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 30
    }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4
  },
  "zero_allow_untested_optimizer": true,
  "wall_clock_breakdown": false,
  "steps_per_print": 9999999999
}
```

```
[glogin01]$ deepspeed --num_gpus=4 ../src/zero_args.py --deepspeed_config=../src/zero_off_config.json
[2021-10-27 23:25:24,828] [WARNING] [runner.py:122:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
[2021-10-27 23:25:25,004] [INFO] [runner.py:360:main] cmd = /usr/bin/python3 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMSwgMiwgM119 --master_addr=127.0.0.1 --master_port=29500 ../src/zero_args.py --deepspeed_config=../src/zero_off_config.json
[2021-10-27 23:25:26,109] [INFO] [launch.py:80:main] WORLD INFO DICT: {'localhost': [0, 1, 2, 3]}
[2021-10-27 23:25:26,109] [INFO] [launch.py:89:main] nnodes=1, num_local_procs=4, node_rank=0
[2021-10-27 23:25:26,109] [INFO] [launch.py:101:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1, 2, 3]})
[2021-10-27 23:25:26,109] [INFO] [launch.py:102:main] dist_world_size=4
[2021-10-27 23:25:26,109] [INFO] [launch.py:105:main] Setting CUDA_VISIBLE_DEVICES=0,1,2,3
[2021-10-27 23:25:31,292] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 23:25:31,295] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 23:25:31,337] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 23:25:31,340] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 23:25:31,355] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 23:25:31,358] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 23:25:31,366] [INFO] [logging.py:68:log_dist] [Rank -1] DeepSpeed info: version=0.5.4, git-hash=unknown, git-branch=unknown
[2021-10-27 23:25:31,369] [INFO] [distributed.py:47:init_distributed] Initializing torch distributed with backend: nccl
[2021-10-27 23:25:36,773] [INFO] [logging.py:68:log_dist] [Rank 0] initializing deepspeed groups
[2021-10-27 23:25:36,774] [INFO] [logging.py:68:log_dist] [Rank 0] initializing deepspeed model parallel group with size 1
[2021-10-27 23:25:36,779] [INFO] [logging.py:68:log_dist] [Rank 0] initializing deepspeed expert parallel group with size 1
[2021-10-27 23:25:36,780] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert data parallel process group with ranks: [0, 1, 2, 3]
[2021-10-27 23:25:36,780] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [0]
[2021-10-27 23:25:36,780] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [1]
[2021-10-27 23:25:36,780] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [2]
[2021-10-27 23:25:36,780] [INFO] [logging.py:68:log_dist] [Rank 0] creating expert parallel process group with ranks: [3]
[2021-10-27 23:25:37,092] [INFO] [engine.py:205:__init__] DeepSpeed Flops Profiler Enabled: False
[2021-10-27 23:25:37,092] [INFO] [engine.py:849:_configure_optimizer] Removing param_group that has no 'params' in the client Optimizer
[2021-10-27 23:25:37,092] [INFO] [engine.py:854:_configure_optimizer] Using client Optimizer as basic optimizer
[2021-10-27 23:25:37,097] [INFO] [engine.py:871:_configure_optimizer] DeepSpeed Basic Optimizer = Adam
[2021-10-27 23:25:37,097] [INFO] [utils.py:44:is_zero_supported_optimizer] Checking ZeRO support for optimizer=Adam type=<class 'torch.optim.adam.Adam'>
[2021-10-27 23:25:37,097] [INFO] [logging.py:68:log_dist] [Rank 0] Creating fp16 ZeRO stage 3 optimizer
Initializing ZeRO Stage 3
[2021-10-27 23:25:37,101] [INFO] [stage3.py:638:__init__] Reduce bucket size 500000000.0
[2021-10-27 23:25:37,101] [INFO] [stage3.py:639:__init__] Allgather bucket size 50000000
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
Emitting ninja build file /home/ubuntu/.cache/torch_extensions/utils/build.ninja...
Building extension module utils...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module utils...
Time to load utils op: 0.3441762924194336 seconds
Loading extension module utils...
Time to load utils op: 0.4021937847137451 seconds
Loading extension module utils...
Time to load utils op: 0.40210652351379395 seconds
Loading extension module utils...
Time to load utils op: 0.40212202072143555 seconds
[2021-10-27 23:25:38,717] [INFO] [stage3.py:831:__init__] optimizer state initialized
[2021-10-27 23:25:38,942] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed Final Optimizer = Adam
[2021-10-27 23:25:38,942] [INFO] [engine.py:587:_configure_lr_scheduler] DeepSpeed using configured LR scheduler = WarmupDecayLR
[2021-10-27 23:25:38,942] [INFO] [logging.py:68:log_dist] [Rank 0] DeepSpeed LR Scheduler = <deepspeed.runtime.lr_schedules.WarmupDecayLR object at 0x7f62c83abcf8>
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
[2021-10-27 23:25:38,943] [INFO] [logging.py:68:log_dist] [Rank 0] step=0, skipped=0, lr=[3e-05], mom=[(0.9, 0.999)]
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
[2021-10-27 23:25:38,943] [INFO] [config.py:940:print] DeepSpeedEngine configuration:
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.00048089027404785156 seconds
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   activation_checkpointing_config  {
    "partition_activations": true, 
    "contiguous_memory_optimization": true, 
    "cpu_checkpointing": true, 
    "number_checkpoints": 4, 
    "synchronize_checkpoint_boundary": false, 
    "profile": false
}
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
Time to load utils op: 0.0004799365997314453 seconds
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   allreduce_always_fp32 ........ False
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   amp_enabled .................. False
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   amp_params ................... False
Time to load utils op: 0.00047278404235839844 seconds
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   checkpoint_tag_validation_enabled  True
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   checkpoint_tag_validation_fail  False
[2021-10-27 23:25:38,943] [INFO] [config.py:944:print]   curriculum_enabled ........... False
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   curriculum_params ............ False
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   dataloader_drop_last ......... False
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   disable_allgather ............ False
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   dump_state ................... False
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   dynamic_loss_scale_args ...... {'init_scale': 4294967296, 'scale_window': 1000, 'delayed_shift': 2, 'min_scale': 1}
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_enabled ........... False
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_gas_boundary_resolution  1
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_layer_num ......... 0
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_max_iter .......... 100
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_stability ......... 1e-06
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_tol ............... 0.01
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   eigenvalue_verbose ........... False
[2021-10-27 23:25:38,944] [INFO] [config.py:944:print]   elasticity_enabled ........... False
[2021-10-27 23:25:38,946] [INFO] [config.py:944:print]   flops_profiler_config ........ {
    "enabled": false, 
    "profile_step": 1, 
    "module_depth": -1, 
    "top_modules": 1, 
    "detailed": true, 
    "output_file": null
}
[2021-10-27 23:25:38,946] [INFO] [config.py:944:print]   fp16_enabled ................. True
[2021-10-27 23:25:38,946] [INFO] [config.py:944:print]   fp16_master_weights_and_gradients  False
[2021-10-27 23:25:38,946] [INFO] [config.py:944:print]   fp16_mixed_quantize .......... False
[2021-10-27 23:25:38,946] [INFO] [config.py:944:print]   global_rank .................. 0
[2021-10-27 23:25:38,946] [INFO] [config.py:944:print]   gradient_accumulation_steps .. 1
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   gradient_clipping ............ 0.0
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   gradient_predivide_factor .... 1.0
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   initial_dynamic_scale ........ 4294967296
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   loss_scale ................... 0
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   memory_breakdown ............. False
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   optimizer_legacy_fusion ...... False
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   optimizer_name ............... None
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   optimizer_params ............. None
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0}
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   pld_enabled .................. False
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   pld_params ................... False
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   prescale_gradients ........... False
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   quantize_change_rate ......... 0.001
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   quantize_groups .............. 1
[2021-10-27 23:25:38,947] [INFO] [config.py:944:print]   quantize_offset .............. 1000
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   quantize_period .............. 1000
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   quantize_rounding ............ 0
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   quantize_start_bits .......... 16
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   quantize_target_bits ......... 8
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   quantize_training_enabled .... False
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   quantize_type ................ 0
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   quantize_verbose ............. False
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   scheduler_name ............... WarmupDecayLR
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   scheduler_params ............. {'total_num_steps': 300, 'warmup_min_lr': 0, 'warmup_max_lr': 3e-05, 'warmup_num_steps': 30}
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   sparse_attention ............. None
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   sparse_gradients_enabled ..... False
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   steps_per_print .............. 9999999999
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   tensorboard_enabled .......... False
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   tensorboard_job_name ......... DeepSpeedJobName
[2021-10-27 23:25:38,948] [INFO] [config.py:944:print]   tensorboard_output_path ...... 
[2021-10-27 23:25:38,949] [INFO] [config.py:944:print]   train_batch_size ............. 16
[2021-10-27 23:25:38,949] [INFO] [config.py:944:print]   train_micro_batch_size_per_gpu  4
[2021-10-27 23:25:38,949] [INFO] [config.py:944:print]   use_quantizer_kernel ......... False
[2021-10-27 23:25:38,949] [INFO] [config.py:944:print]   wall_clock_breakdown ......... False
[2021-10-27 23:25:38,949] [INFO] [config.py:944:print]   world_size ................... 4
[2021-10-27 23:25:38,949] [INFO] [config.py:944:print]   zero_allow_untested_optimizer  True
[2021-10-27 23:25:38,950] [INFO] [config.py:944:print]   zero_config .................. {
    "stage": 3, 
    "contiguous_gradients": true, 
    "reduce_scatter": true, 
    "reduce_bucket_size": 5.000000e+08, 
    "allgather_partitions": true, 
    "allgather_bucket_size": 5.000000e+08, 
    "overlap_comm": true, 
    "load_from_fp32_weights": true, 
    "elastic_checkpoint": true, 
    "offload_param": {
        "device": "cpu", 
        "nvme_path": null, 
        "buffer_count": 5, 
        "buffer_size": 1.000000e+08, 
        "max_in_cpu": 1.000000e+09, 
        "pin_memory": true
    }, 
    "offload_optimizer": {
        "device": "cpu", 
        "nvme_path": null, 
        "buffer_count": 4, 
        "pin_memory": true, 
        "pipeline_read": false, 
        "pipeline_write": false, 
        "fast_init": false, 
        "pipeline": false
    }, 
    "sub_group_size": 1.000000e+09, 
    "prefetch_bucket_size": 5.000000e+07, 
    "param_persistence_threshold": 1.000000e+05, 
    "max_live_parameters": 1.000000e+09, 
    "max_reuse_distance": 1.000000e+09, 
    "gather_fp16_weights_on_model_save": false, 
    "ignore_unused_parameters": true, 
    "round_robin_gradients": false, 
    "legacy_stage1": false
}
[2021-10-27 23:25:38,950] [INFO] [config.py:944:print]   zero_enabled ................. True
[2021-10-27 23:25:38,950] [INFO] [config.py:944:print]   zero_optimization_stage ...... 3
[2021-10-27 23:25:38,950] [INFO] [config.py:952:print]   json = {
    "train_batch_size": 16, 
    "gradient_accumulation_steps": 1, 
    "scheduler": {
        "type": "WarmupDecayLR", 
        "params": {
            "total_num_steps": 300, 
            "warmup_min_lr": 0, 
            "warmup_max_lr": 3e-05, 
            "warmup_num_steps": 30
        }
    }, 
    "fp16": {
        "enabled": true, 
        "initial_scale_power": 32, 
        "loss_scale_window": 1000, 
        "hysteresis": 2, 
        "min_loss_scale": 1
    }, 
    "zero_optimization": {
        "stage": 3, 
        "allgather_bucket_size": 5.000000e+08, 
        "reduce_bucket_size": 5.000000e+08, 
        "offload_param": {
            "device": "cpu", 
            "pin_memory": true
        }, 
        "offload_optimizer": {
            "device": "cpu", 
            "pin_memory": true
        }
    }, 
    "activation_checkpointing": {
        "partition_activations": true, 
        "cpu_checkpointing": true, 
        "contiguous_memory_optimization": true, 
        "number_checkpoints": 4
    }, 
    "zero_allow_untested_optimizer": true, 
    "wall_clock_breakdown": false, 
    "steps_per_print": 1.000000e+10
}
Using /home/ubuntu/.cache/torch_extensions as PyTorch extensions root...
No modifications detected for re-loaded extension module utils, skipping build step...
Loading extension module utils...
Time to load utils op: 0.0004980564117431641 seconds
Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
  0%|                                                     | 0/2 [00:00<?, ?it/s]Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 613.34it/s]
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 589.50it/s]
Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
  0%|                                                     | 0/2 [00:00<?, ?it/s]Reusing dataset squad (/home/ubuntu/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453)
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 599.19it/s]
100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 618.08it/s]
[2021-10-27 23:25:40,395] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4294967296, reducing to 4294967296
step:0, loss:5.453125
[2021-10-27 23:25:40,917] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4294967296, reducing to 2147483648.0
[2021-10-27 23:25:41,416] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2147483648.0, reducing to 1073741824.0
[2021-10-27 23:25:41,931] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1073741824.0, reducing to 536870912.0
[2021-10-27 23:25:42,436] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 536870912.0, reducing to 268435456.0
[2021-10-27 23:25:42,943] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 268435456.0, reducing to 134217728.0
[2021-10-27 23:25:43,450] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 134217728.0, reducing to 67108864.0
[2021-10-27 23:25:43,955] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 67108864.0, reducing to 33554432.0
[2021-10-27 23:25:44,455] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 33554432.0, reducing to 16777216.0
[2021-10-27 23:25:44,980] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16777216.0, reducing to 8388608.0
[2021-10-27 23:25:45,489] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8388608.0, reducing to 4194304.0
step:10, loss:3.6484375
[2021-10-27 23:25:45,991] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4194304.0, reducing to 2097152.0
[2021-10-27 23:25:46,500] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2097152.0, reducing to 1048576.0
[2021-10-27 23:25:47,008] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1048576.0, reducing to 524288.0
[2021-10-27 23:25:47,509] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 524288.0, reducing to 262144.0
[2021-10-27 23:25:48,017] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144.0, reducing to 131072.0
[2021-10-27 23:25:48,516] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 131072.0, reducing to 65536.0
[2021-10-27 23:25:49,018] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 65536.0, reducing to 32768.0
[2021-10-27 23:25:49,533] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 32768.0, reducing to 16384.0
[2021-10-27 23:25:50,043] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 16384.0, reducing to 8192.0
[2021-10-27 23:25:50,535] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 8192.0, reducing to 4096.0
step:20, loss:3.546875
[2021-10-27 23:25:52,296] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4096.0, reducing to 2048.0
[2021-10-27 23:25:52,792] [INFO] [stage3.py:2736:_overflow_clean_up] [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 2048.0, reducing to 1024.0
step:30, loss:3.76953125
step:40, loss:2.880859375
step:50, loss:2.408203125
step:60, loss:2.5234375
step:70, loss:2.263671875
step:80, loss:2.5078125
step:90, loss:2.939453125
step:100, loss:2.7890625
step:110, loss:2.490234375
step:120, loss:2.95703125
step:130, loss:2.359375
step:140, loss:2.923828125
step:150, loss:3.8515625
step:160, loss:3.04296875
step:170, loss:3.056640625
step:180, loss:1.65625
step:190, loss:3.509765625
step:200, loss:3.71484375
step:210, loss:3.55859375
step:220, loss:2.978515625
step:230, loss:3.251953125
step:240, loss:2.560546875
step:250, loss:3.201171875
step:260, loss:3.564453125
step:270, loss:3.23828125
step:280, loss:2.6171875
step:290, loss:2.232421875
step:300, loss:3.486328125
```

## 7. ZeRO Infinity
   
ZeRO Infinity는 NVMe(SSD) 메모리에 파라미터를 저장하는 방식을 채택하였습니다. NVMe는 CPU 메모리보다도 훨씬 커다란 메모리 용량을 가지기 때문에 메모리의 한계를 한번 더 돌파 하였다고 평가됩니다. ZeRO Infinity 알고리즘 역시 매우 복잡하기 영샹으로 확인하겠습니다. https://www.microsoft.com/en-us/research/uploads/prod/2021/04/1400x788_deepspeed_nologo-1.mp4"
```
from IPython.display import HTML

HTML("""
<div align="middle">
<video width="80%" controls>
      <source src="../images/zero_infinity.mp4" type="video/mp4">
</video></div>""")
```

### ZeRO Infinity의 핵심 아이디어
    
ZeRO Infinity는 ZeRO Offload의 확장판입니다. 기존에 ZeRO Offload는 CPU RAM과 GPU VRAM를 다음과 같이 운용하였습니다.   
- **GPU**: FP16 parameter & gradient 상주, **Forward & Backward 수행**.
- **CPU**: FP32 parameter & gradient & optimizer 상주, **Weight Update 수행**.
   
ZeRO Infinity는 NVMe가 추가되어 세개의 디바이스를 운용합니다. 활용법은 아래와 같습니다.
- **NVMe**: 기본적으로 모든 파라미터는 사용되지 않을때 NVMe에 상주.
- **GPU**: FP16 parameter & gradient가 Forward & Backward를 수행해야 할 때 NVMe에서 필요한 부분만 GPU로 업로드.
- **CPU**: FP32 parameter & gradient, optimizer가 Weight Update를 수행해야 할 때 NVMe에서 필요한 부분만 CPU로 업로드.
    
즉, 기본적으로 모든 텐서를 NVMe로 내리고 있다가, 그들이 필요할때만 CPU & GPU 등의 연산 장비로 올리는 방식을 사용합니다.
    
![](../images/zero_infinity.png)
    
ZeRO Infinity는 거의 모든 텐서를 NVMe에 내려놓고 있기 때문에 실제로 CPU와 GPU는 거의 텅텅 빈 상태가 됩니다. 따라서 위 그림 처럼 기존의 기법들로는 아예 학습이 불가능 하던 수준의 모델도 학습할 수 있습니다. 또한 실험 결과에 의하면 ZeRO Offload와 비교했을때 속도가 더 빨랐다고 합니다.
  
ZeRO Infinity는 NVMe와 연결된 디바이스가 필요하기 때문에 본 자료에서 실습을 하지는 않겠습니다. NVMe가 탑재된 디바이스라면 아래와 같이 `offload_param`과 `offload_optimizer`의 동작 디바이스를 `nvme`로 변경하고 `nvme_path`를 알맞게 설정해주시면 됩니다.

```
"offload_param": {
    "device": "nvme", 
    "nvme_path": "/local_nvme",
    "pin_memory": true
}, 
"offload_optimizer": {
    "device": "nvme", 
    "nvme_path": "/local_nvme",
    "pin_memory": true
}
```
