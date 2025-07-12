# Large-scale language modeling tutorials with PyTorch

![](images/megatron_3d.png)

This repository provides a comprehensive, in-depth guide for large-scale distributed training of Large Language Models (LLMs) on supercomputers managed with SLURM. It briefly covers some basics of collective communication in messege passing including broadcast, gather, scatter and all-gather operations, then delving into data parallelism techniques such as Data Parallelism (DP) and Distributed Data Parallelism (DDP) in PyTorch, and model parallelism techniques including Tensor Parallelism, Pipeline Parallelism, and 3D Parallelism, with hands-on PyTorch code examples. It also covers how to set up and leverage distributed training tools like Megatron-LM and DeepSpeed to efficiently run the PyTorch codes using multiple GPUs on a supercomputer.

본 튜토리얼 사이트는 TUNiB의 고현웅님의 [large-scale language modeling tutorials with PyTorch](https://github.com/tunib-ai/large-scale-lm-tutorials) 사이트를 기반으로 작성한 사이트 입니다. 이 자료는 대규모 언어모델 개발에 필요한 여러가지 기술들을 소개드리기 위해 마련하였으며 기본적으로 PyTorch와 Transformer 언어모델에 대한 지식이 있다고 가정하고 만들었습니다. 내용 중 틀린부분이 있거나 궁금하신 부분이 있으시면 이슈나 메일로 문의 주시면 감사하겠습니다. 

- 목차의 대분류는 '세션', 소분류는 '챕터'라고 명명하였습니다.
- 모든 소스코드 및 노트북 파일은 [Github](https://github.com/hwang2006/large-scale-lm-tutorials)에 공개되어 있습니다. <br>
<!-- - Github에서 열람하시는 것보다 [NBViewer](https://nbviewer.org/github/hwang2006/large-scale-lm-tutorials/tree/main/notebooks/) 로 열람하시는 것을 추천드립니다. -->

## Contents

1. [Motivation](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/01_motivation.md)
2. [Distributed Programming](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/02_distributed_programming.md)
3. [Overview of Parallelism](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/03_overview_of_parallelism.md)
4. [Data Parallelism](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/04_data_parallelism.md)
5. [Pipeline Parallelism](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/05_pipeline_parallelism.md)
6. [Tensor Parallelism](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/06_tensor_parallelism.md)
7. [Zero Redundancy Optimization](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/07_zero_redundancy_optimization.md)
8. [Multi-dimensional Parallelism](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/08_multi_dimensional_parallelism.md)
9. [Additional Techniques](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/09_additional_techiques.md)

## Environments
### KISTI Neuron GPU Cluster
Neuron is a [KISTI GPU cluster system](https://docs-ksc.gitbook.io/neuron-user-guide) consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

<!---
### Local Environments
- Linux Ubuntu 18.04 LTS
- 4 * A100 GPU
- Python 3.7
- pytorch==1.9.0+cu111

### Docker Environments
- `docker pull pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel`
- 원활한 실습을 위해 `--shm-size`를 키우거나 `--ipc=host` 옵션을 설정해주세요.
-->

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
no change     /scratch/qualis/miniconda3/etc/profile.d/conda.csh
modified      /home01/qualis/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Miniconda3!
```

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 24.3.0
```
## Creating a Conda Virtual Environment
You need to create a virtual envrionment for a large-scale language model tutorial.

1. Create a conda virtual environment with a python version 3.10
```
[glogin01]$ conda create -n large-scale-lm python=3.12 -y
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3/envs/large-scale-lm

  added / updated specs:
    - python=3.10
.
.
.
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate large-scale-lm
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

2. Install PyTorch, jupyter, transformers and datasets
```
[glogin01]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.1
[glogin01]$ conda activate large-scale-lm
(large-scale-lm) [glogin01]$ pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
(large-scale-lm) [glogin01]$ pip install jupyter transformers datasets torchgpipe deepspeed nltk pybind11
```

## 주피터 띄워서 실습하기(Hands-on Practices with Jupyter on a Supercomputer)
참고로 터미날 상에서 모든 실습을 할 수도 있지만 주피터를 띄워서 본 레포지터리에 있는 코드들을 Hands-on 실습을 할 수도 있습니다. [7. Zero Redundancy Optimization](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/07_zero_redundancy_optimization.md) 첫부분인 **Mixed Precision** 실습 부분에 뉴론시스템 컴퓨터 노드에 주피터 서버를 실행하고 각자의 PC 또는 노트북에서 브라우저를 띄워서 뉴론 주피터 서버에 연결하는 방법이 설명되어 있습니다.  

## LICENSE

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


