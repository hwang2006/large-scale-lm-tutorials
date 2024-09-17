# Large-scale language modeling tutorials with PyTorch

![](images/megatron_3d.png)

This repository provides a comprehensive guide to large-scale distributed training on SLURM-enabled supercomputers. It delves into data parallelism techniques such as Data Parallelism (DP) and Distributed Data Parallelism (DDP) in PyTorch, and model parallelism techniques including Tensor Parallelism, Pipeline Parallelism, and 3D Parallelism, with hands-on PyTorch code examples. It also covers how to set up and leverage distributed training tools like Megatron-LM and DeepSpeed to efficiently run the PyTorch codes using multiple GPUs on a supercomputer.

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
[glogin01]$ conda create -n large-scale-lm python=3.10 -y
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

2. Install PyTorch
```
[glogin01]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.1
[glogin01]$ conda activate large-scale-lm
(large-scale-lm) [glogin01]$ conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
Channels:
 - pytorch
 - nvidia
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3/envs/test

  added / updated specs:
    - pytorch-cuda=12.1
    - pytorch==2.3.0
    - torchaudio==2.3.0
    - torchvision==0.18.0


The following NEW packages will be INSTALLED:

  blas               pkgs/main/linux-64::blas-1.0-mkl
  brotli-python      pkgs/main/linux-64::brotli-python-1.0.9-py310h6a678d5_8
  certifi            pkgs/main/linux-64::certifi-2024.8.30-py310h06a4308_0
  charset-normalizer pkgs/main/noarch::charset-normalizer-3.3.2-pyhd3eb1b0_0
  cuda-cudart        nvidia/linux-64::cuda-cudart-12.1.105-0
  cuda-cupti         nvidia/linux-64::cuda-cupti-12.1.105-0
  cuda-libraries     nvidia/linux-64::cuda-libraries-12.1.0-0
  cuda-nvrtc         nvidia/linux-64::cuda-nvrtc-12.1.105-0
  cuda-nvtx          nvidia/linux-64::cuda-nvtx-12.1.105-0
  cuda-opencl        nvidia/linux-64::cuda-opencl-12.6.68-0
  cuda-runtime       nvidia/linux-64::cuda-runtime-12.1.0-0
  cuda-version       nvidia/noarch::cuda-version-12.6-3
  ffmpeg             pytorch/linux-64::ffmpeg-4.3-hf484d3e_0
  filelock           pkgs/main/linux-64::filelock-3.13.1-py310h06a4308_0
  freetype           pkgs/main/linux-64::freetype-2.12.1-h4a9f257_0
  gmp                pkgs/main/linux-64::gmp-6.2.1-h295c915_3
  gmpy2              pkgs/main/linux-64::gmpy2-2.1.2-py310heeb90bb_0
  gnutls             pkgs/main/linux-64::gnutls-3.6.15-he1e5248_0
  idna               pkgs/main/linux-64::idna-3.7-py310h06a4308_0
  intel-openmp       pkgs/main/linux-64::intel-openmp-2023.1.0-hdb19cb5_46306
  jinja2             pkgs/main/linux-64::jinja2-3.1.4-py310h06a4308_0
  jpeg               pkgs/main/linux-64::jpeg-9e-h5eee18b_3
  lame               pkgs/main/linux-64::lame-3.100-h7b6447c_0
  lcms2              pkgs/main/linux-64::lcms2-2.12-h3be6417_0
  lerc               pkgs/main/linux-64::lerc-3.0-h295c915_0
  libcublas          nvidia/linux-64::libcublas-12.1.0.26-0
  libcufft           nvidia/linux-64::libcufft-11.0.2.4-0
  libcufile          nvidia/linux-64::libcufile-1.11.1.6-0
  libcurand          nvidia/linux-64::libcurand-10.3.7.68-0
  libcusolver        nvidia/linux-64::libcusolver-11.4.4.55-0
  libcusparse        nvidia/linux-64::libcusparse-12.0.2.55-0
  libdeflate         pkgs/main/linux-64::libdeflate-1.17-h5eee18b_1
  libiconv           pkgs/main/linux-64::libiconv-1.16-h5eee18b_3
  libidn2            pkgs/main/linux-64::libidn2-2.3.4-h5eee18b_0
  libjpeg-turbo      pytorch/linux-64::libjpeg-turbo-2.0.0-h9bf148f_0
  libnpp             nvidia/linux-64::libnpp-12.0.2.50-0
  libnvjitlink       nvidia/linux-64::libnvjitlink-12.1.105-0
  libnvjpeg          nvidia/linux-64::libnvjpeg-12.1.1.14-0
  libpng             pkgs/main/linux-64::libpng-1.6.39-h5eee18b_0
  libtasn1           pkgs/main/linux-64::libtasn1-4.19.0-h5eee18b_0
  libtiff            pkgs/main/linux-64::libtiff-4.5.1-h6a678d5_0
  libunistring       pkgs/main/linux-64::libunistring-0.9.10-h27cfd23_0
  libwebp-base       pkgs/main/linux-64::libwebp-base-1.3.2-h5eee18b_0
  llvm-openmp        pkgs/main/linux-64::llvm-openmp-14.0.6-h9e868ea_0
  lz4-c              pkgs/main/linux-64::lz4-c-1.9.4-h6a678d5_1
  markupsafe         pkgs/main/linux-64::markupsafe-2.1.3-py310h5eee18b_0
  mkl                pkgs/main/linux-64::mkl-2023.1.0-h213fc3f_46344
  mkl-service        pkgs/main/linux-64::mkl-service-2.4.0-py310h5eee18b_1
  mkl_fft            pkgs/main/linux-64::mkl_fft-1.3.10-py310h5eee18b_0
  mkl_random         pkgs/main/linux-64::mkl_random-1.2.7-py310h1128e8f_0
  mpc                pkgs/main/linux-64::mpc-1.1.0-h10f8cd9_1
  mpfr               pkgs/main/linux-64::mpfr-4.0.2-hb69a4c5_1
  mpmath             pkgs/main/linux-64::mpmath-1.3.0-py310h06a4308_0
  nettle             pkgs/main/linux-64::nettle-3.7.3-hbbd107a_1
  networkx           pkgs/main/linux-64::networkx-3.2.1-py310h06a4308_0
  numpy              pkgs/main/linux-64::numpy-2.0.1-py310h5f9d8c6_1
  numpy-base         pkgs/main/linux-64::numpy-base-2.0.1-py310hb5e798b_1
  openh264           pkgs/main/linux-64::openh264-2.1.1-h4ff587b_0
  openjpeg           pkgs/main/linux-64::openjpeg-2.5.2-he7f1fd0_0
  pillow             pkgs/main/linux-64::pillow-10.4.0-py310h5eee18b_0
  pysocks            pkgs/main/linux-64::pysocks-1.7.1-py310h06a4308_0
  pytorch            pytorch/linux-64::pytorch-2.3.0-py3.10_cuda12.1_cudnn8.9.2_0
  pytorch-cuda       pytorch/linux-64::pytorch-cuda-12.1-ha16c6d3_5
  pytorch-mutex      pytorch/noarch::pytorch-mutex-1.0-cuda
  pyyaml             pkgs/main/linux-64::pyyaml-6.0.1-py310h5eee18b_0
  requests           pkgs/main/linux-64::requests-2.32.3-py310h06a4308_0
  sympy              pkgs/main/linux-64::sympy-1.13.2-py310h06a4308_0
  tbb                pkgs/main/linux-64::tbb-2021.8.0-hdb19cb5_0
  torchaudio         pytorch/linux-64::torchaudio-2.3.0-py310_cu121
  torchtriton        pytorch/linux-64::torchtriton-2.3.0-py310
  torchvision        pytorch/linux-64::torchvision-0.18.0-py310_cu121
  typing_extensions  pkgs/main/linux-64::typing_extensions-4.11.0-py310h06a4308_0
  urllib3            pkgs/main/linux-64::urllib3-2.2.2-py310h06a4308_0
  yaml               pkgs/main/linux-64::yaml-0.2.5-h7b6447c_0
  zstd               pkgs/main/linux-64::zstd-1.5.5-hc292b87_2



Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

## 주피터 띄워서 실습하기 
참고로 터미날 상에서 모든 실습을 할 수도 있지만 주피터를 뜨위서 본 레포지터리에 있는 코드들을 Hands-on 실습을 할 수도 있습니다. [7. Zero Redundancy Optimization](https://github.com/hwang2006/large-scale-lm-tutorials/blob/main/docs/07_zero_redundancy_optimization.md) Mixed Precision 실습 부분에 뉴론시스템 컴퓨터 노드에 주피터 서버를 실행하고 각자의 PC 또는 노트북에서 브라우저를 띄워서 뉴론 주피터 서버에 연결하는 방법이 설명되어 있습니다.  

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


