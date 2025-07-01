# Tensor Parallelism
이번 세션에서는 Tensor parallelism에 대해서 알아보겠습니다.
  
## 1. Intra-layer model parallelism
Tensor Parallelism은 Intra-layer 모델 병렬화 방식으로 **레이어 내부에서 텐서 단위로 모델을 쪼갭니다.** Inter-layer 모델 병렬화는 상식적으로 이해가 가지만, Intra-layer 병렬화의 경우는 처음 보시는 분들은 어떻게 이 것이 가능한지 궁금하실거에요.

![](../images/intra_layer.png)

우리가 흔히 사용하는 내적 연산은 연산하고자 하는 행렬을 쪼개서 병렬적으로 수행하고 결과를 더하거나 이어붙여도 최종 출력값이 변하지 않는 성질이 있습니다. 이러한 내적 연산의 성질을 이용하여 모델을 병렬화 하는것을 Tensor 병렬화라고 합니다. 용어가 다소 헷갈릴 수 있는데 Intra-layer는 레이어 단위에서 일어나지 않는 모든 병렬화를 의미하기 때문에 더 큰 범주이고, Tensor 병렬화는 Intra-layer 병렬화의 구현하는 방법 중 한가지 입니다.
 
## 2. Megatron-LM
Megatron-LM은 NVIDA에서 공개한 Intra-layer 모델 병렬화 구현체로, 현재 Large-scale 모델 개발에 있어서 가장 중요한 프로젝트 중 하나입니다.

![](../images/megatron_lm.jpeg)
  
### Column & Row parallelism
다음은 Megatron-LM에서 사용되는 column parallelism과 row parallelism을 그림으로 나타낸 것입니다.
- Column parallelism은 **모델의 파라미터(A)를 수직방향으로 분할(A1, A2)하는 방법**입니다.
- Row parallelism은 **모델의 파라미터(A)를 수평방향으로 분할(A1, A2)하는 방법**입니다.
    
![](../images/intra_layer_2.png)

직접 코딩해서 결과를 확인해봅시다. 가장 먼저 텐서 X와 텐서 A의 행렬곱 결과는 다음과 같습니다.
```
"""
src/ch6/non_parallelism.py
"""

import torch

X = torch.tensor(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
)

A = torch.tensor(
    [
        [10, 14],
        [11, 15],
        [12, 16],
        [13, 17],        
    ]
)

Y = X @ A

print(Y)
```
```
(large-scale-lm) [glogin01]$ python non_parallelism.py
tensor([[ 74,  98],
        [258, 346]])
```

column parallelism은 모델의 파라미터(A)를 수직방향으로 자른 뒤 연산후 연산 결과를 concat하는 방식입니다. 그림에서와 같이 X는 복제하고 텐서 A를 수직방향으로 분할한 뒤 연산 후 concat 해보겠습니다.
```
"""
src/ch6/column_parallelism.py
"""

import torch

X = torch.tensor(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
)

A1 = torch.tensor(
    [
        [10],
        [11],
        [12],
        [13],        
    ]
)

A2 = torch.tensor(
    [
        [14],
        [15],
        [16],
        [17],        
    ]
)

Y1 = X @ A1
Y2 = X @ A2

print(Y1)
print(Y2)

Y = torch.cat([Y1, Y2], dim=1)
print(Y)
```
```
(large-scale-lm) [glogin01]$ python column_parallelism.py
tensor([[ 74],
        [258]])
tensor([[ 98],
        [346]])
tensor([[ 74,  98],
        [258, 346]])
```

병렬화 전 후의 연산 결과가 동일한 것을 확인 할 수 있습니다. 
그 다음으로 row parallelism를 알아봅시다. row parallelism은 모델의 파라미터(A)를 수평방향으로 분할 한 뒤 연산 결과를 더하는 방식입니다. 그림과 같이 X와 Y 모두를 분할한 뒤 연산 후 결과 값을 더해보겠습니다.
```
"""
src/ch6/row_parallelism.py
"""

import torch

X1 = torch.tensor(
    [
        [0, 1],
        [4, 5],
    ]
)

X2 = torch.tensor(
    [
        [2, 3],
        [6, 7],
    ]
)

A1 = torch.tensor(
    [
        [10, 14],
        [11, 15],      
    ]
)

A2 = torch.tensor(
    [
        [12, 16],
        [13, 17],        
    ]
)

Y1 = X1 @ A1
Y2 = X2 @ A2

print(Y1)
print(Y2)

Y = Y1 + Y2

print(Y)
```
```
(large-scale-lm) [glogin01]$ python row_parallelism.py
tensor([[ 11,  15],
        [ 95, 131]])
tensor([[ 63,  83],
        [163, 215]])
tensor([[ 74,  98],
        [258, 346]])
```
연산 결과가 동일한 것을 확인할 수 있습니다.
    
### Column parallelism: $(D, D) → (D, \\frac{D}{n}) \\times n$
앞선 예시에서 본 것 처럼, Column Parallelism은 **입력텐서(X)를 복사**하고, 모델의 파라미터(A)를 **수직방향으로 분할(A1, A2)하여 내적** 후 concat하는 연산입니다.
    
![](../images/column_parallel.png)
    
Megatron-LM에서는 **분할된 파라미터 (A1, A2)를 서로 다른 디바이스에 올려서 모델을 병렬화** 합니다. 이에 따라 행렬 곱 연산도 여러개의 GPU에서 동시에 일어나게 되고, 이를 처리하기 위해 분산 프로그래밍이 필요합니다. Column Parallelism을 위해서는 Broadcast와 All-gather 연산을 사용합니다
- 서로 다른 GPU에 동일한 입력을 전송하기 위해 **Broadcast** 연산를 사용합니다.
- 행렬 곱 연산 결과를 모으기 위해 **All-gather** 연산을 사용합니다.

```
"""
참고: ColumnParallelLinear in megatron-lm/megatron/mpu/layers.py
"""

def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None

    # Set up backprop all-reduce.
    input_parallel = copy_to_tensor_model_parallel_region(input_)

    # Matrix multiply.
    output_parallel = F.linear(input_parallel, self.weight, bias)

    if self.gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```


Row Parallelism은 **입력텐서(X)를 분할**하고, 모델의 파라미터(A)를 **수평방향으로 분할(A1, A2)하여 내적** 후 더하는 연산입니다.
![](../images/row_parallelism.png)
마찬가지로 Row Parallelism을 여러 GPU에서 실행하기 위해서는 분산 프로그래밍이 필요합니다. Row Parallelism을 위해서는 Scatter와 All-reduce을 사용합니다.
- 서로 다른 GPU에 입력을 분할하여 전송하기 위해 **Scatter** 연산를 사용합니다.
- 행렬 곱 연산 결과를 더하기 위해서 **All-reduce** 연산을 사용합니다.

```
"""
참고: RowParallelLinear in megatron-lm/megatron/mpu/layers.py
"""

def forward(self, input_):
    # Set up backprop all-reduce.
    if self.input_is_parallel:
        input_parallel = input_
    else:
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    
    # Matrix multiply.
    output_parallel = F.linear(input_parallel, self.weight)
    
    # All-reduce across all the partitions.
    output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    
    if not self.skip_bias_add:
        output = output_ + self.bias if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias
```

### Transformer Block
이제 Column, Row parallelism에 대해 이해했으니 본격적으로 어떻게 Transformer를 병렬화 할지 살펴봅시다. 우리가 흔히 아는 Transformer Block은 다음과 같이 구성되어 있습니다. Megatron-LM은 여기에서 파라미터의 크기가 매우 적은 Layer Norm 레이어는 파라미터를 모든 디바이스로 복제하고, Layer Norm 레이어를 제외한 다른 레이어들(Attention, MLP)은 위와 같이 Column, Row parallelism을 통해 병렬처리를 수행합니다.
![](../images/megatron_block.png)

### MLP Layer
가장 먼저 MLP 레이어에 대해 알아보겠습니다. MLP 레이어는 `Linear1` → `GeLU` → `Linear2` → `Dropout`순으로 진행됩니다.
![](../images/megatron_mlp.png)
```
"""
참고 transformers/models/gpt_neo/modeling_gpt_neo.py
"""

import torch.nn as nn


class GPTNeoMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
```
여기에서 **첫번째 Linear는 Coulmn Parallelism**을, **두번째 Linear는 Row Parallelism**을 적용합니다.
![](../images/megatron_mlp_2.png)
MLP 레이어에서 Column-Row 순으로 병렬화를 적용하는 이유는 두가지가 있습니다.
- 첫번째 이유는 **`All-gather` 연산과 `Scatter` 연산을 생략** 할 수 있기 때문입니다.
![](../images/megatron_mlp_3.png)
왼쪽 녹색 영역의 연산 결과는 입 력데이터 X와 각 디바이스로 병렬화된 W를 내적한 것입니다. 그리고 나서 붉은색 영역에서 이 결과값을 `All-gather`해서 이어붙인 다음에 다시 `Scatter`하여 쪼개죠. 여기에서 흥미로운 사실은 이어 붙인 텐서를 다시 쪼갰기 때문에 이는 이어붙이기 전과 동일하다는 것입니다.  따라서 오른쪽의 녹색 영역과 왼쪽의 녹색영역 값은 동일하죠. 결과적으로 붉은색 영역 (`All-gather`-`Scatter`)을 생략할 수 있고, 속도 면에서 큰 이득을 가져올 수 있습니다. 
    
이는 Column-Row 순으로 병렬화 할때만 나타나는 독특한 현상으로, 만약 Column-Column, Row-Column, Row-Row와 같이 병렬화 한다면 두 Linear 레이어 사이에서 발생하는 통신을 생략할 수 없게 됩니다.
![](../images/megatron_mlp_4.png)
`All-gather`와 `Scatter`를 생략하는 기법은 Megatron-LM에 `input_is_parallel`와 `gather_output`라는 파라미터로 구현되어있습니다.
```
"""
참고: ColumnParallelLinear in megatron-lm/megatron/mpu/layers.py
"""

def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None

    # Set up backprop all-reduce.
    input_parallel = copy_to_tensor_model_parallel_region(input_)

    # Matrix multiply.
    output_parallel = F.linear(input_parallel, self.weight, bias)

    # gather_output을 False로 설정하여 output을 병렬화된 채로 출력합니다.
    if self.gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


"""
참고: RowParallelLinear in megatron-lm/megatron/mpu/layers.py
"""

def forward(self, input_):
    # Set up backprop all-reduce.

    # input_is_parallel True로 설정하여 input을 병렬화된 채로 입력받습니다.
    if self.input_is_parallel:
        input_parallel = input_
    else:
        input_parallel = scatter_to_tensor_model_parallel_region(input_)
    
    # Matrix multiply.
    output_parallel = F.linear(input_parallel, self.weight)
    
    # All-reduce across all the partitions.
    output_ = reduce_from_tensor_model_parallel_region(output_parallel)
    
    if not self.skip_bias_add:
        output = output_ + self.bias if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias
    return output, output_bias
```

- Column-Row 방식으로 병렬화하는 2번째 이유는 `Scatter`와 `All-gather`를 생략하려면 **GeLU 연산**이 병렬화된 채로 수행되어야 하기 때문입니다.
![](../images/megatron_mlp_5.png)
위 그림은 `Scatter`와 `All-gather`를 생략하지 않는 상황에서 GeLU 연산을 두 Linear 레이어 사이에 삽입한 것입니다. 만약 여기에서 두 연산을 생략하도록 구현하면 아래와 같이 GeLU 연산은 반드시 각각의 디바이스에서 이루어져야 합니다.
   
![](../images/megatron_mlp_6.png)
   
그러나 이렇게 GeLU 연산을 서로 다른 디바이스에서 하도록 병렬화 시키려면 반드시 병렬적으로 계산된 GeLU의 출력은 병렬화 되지 않은 상태에서 계산된 GeLU의 출력과 동일해야겠죠. 즉 다음과 같은 공식이 성립해야 합니다. ($\\circledcirc$ 기호는 concatenation을 의미합니다.)
    
$$Row Paralleism: GeLU(XW1 + XW2) = GeLU(XW1) + GeLU(XW2)$$

$$Column Paralleism: GeLU(XW1 \\circledcirc XW2) = GeLU(XW1) \\circledcirc GeLU(XW2)$$
   
문제는 위와 같은 공식이 Column Parallelism에서만 성립하고, **Row Parallelism 에서는 성립하지 않는다는 것**입니다.
  
$$Row Paralleism: GeLU(XW1 + XW2) \\neq GeLU(XW1) + GeLU(XW2)$$
 
이를 코드로 구현해서 확인해봅시다."
```
"""
src/ch6/megatron_mlp_gelu.py
"""

import torch
from torch.nn.functional import gelu


w = torch.randn(6, 6)
x = torch.randn(6, 6)


class RowParallelLinear(torch.nn.Module):
    def __init__(self):
        super(RowParallelLinear, self).__init__()
        chunked = torch.chunk(w, 2, dim=0)

        # row parallelized parameters
        self.w1 = chunked[0]  # [3, 6]
        self.w2 = chunked[1]  # [3, 6]

    def forward(self, x):
        # GeLU(X1A1 + X2A2) != GeLU(X1A1) + GeLU(X2A2)
        x1, x2 = torch.chunk(x, 2, dim=1)

        # parallel output
        y1 = gelu(x1 @ self.w1) + gelu(x2 @ self.w2)

        # non-parallel output
        y2 = gelu(x1 @ self.w1 + x2 @ self.w2)

        return torch.all(y1 == y2)


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self):
        super(ColumnParallelLinear, self).__init__()
        chunked = torch.chunk(w, 2, dim=1)

        # column parallelized parameters
        self.w1 = chunked[0]  # [6, 3]
        self.w2 = chunked[1]  # [6, 3]

    def forward(self, x):
        # GeLU(X1A1 cat X2A2) == GeLU(X1A1) cat GeLU(X2A2)

        # parallel output
        y1 = torch.cat([gelu(x @ self.w1), gelu(x @ self.w2)], dim=1)

        # non-parallel output
        y2 = gelu(torch.cat([(x @ self.w1), (x @ self.w2)], dim=1))

        return torch.all(y1 == y2)


# Row Parallelism
print("Is GeLU in RowParallelLinear same with non-parallel = ", end="")
print(RowParallelLinear()(x).item())

# Column Parallelism
print("Is GeLU in ColumnParallelLinear same with non-parallel = ", end="")
print(ColumnParallelLinear()(x).item())
```
```
(large-scale-lm) [glogin01]$ ls
./                     Megatron-LM/                      parallelformers_inference.py
../                    megatron_mlp_gelu.py              row_parallelism.py
column_parallelism.py  non_parallelism.py
megatron_datasets.py   parallelformers_inference.org.py

(large-scale-lm) [glogin01]$ python megatron_mlp_gelu.py
Is GeLU in RowParallelLinear same with non-parallel = False
Is GeLU in ColumnParallelLinear same with non-parallel = True
```

따라서 GeLU 연산을 병렬화 시키려면 반드시 GeLU 이전의 Linear 레이어는 Column 방향으로 병렬화 되어있어야 합니다. 따라서 Column-Row 순서로 병렬화 하는 것이 가장 효율적인 방식이죠."

### Multi-head Attention Layer
  
다음으로 Multi-head Attention 레이어에 대해 알아보겠습니다. Multi-head Attention 레이어는 `Linear1` → `Split heads` → `ScaleDotProductAttention` → `Concat(Merge) heads` → `Linear2` → `Dropout` 순으로 진행됩니다.
    
![](../images/multi_head_attention.png)

```
"""
참고 transformers/models/gpt_neo/modeling_gpt_neo.py
"""

class GPTNeoSelfAttention(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 1. linear projection
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 2. split heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 3. scale dot product attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 4. concat (merge) heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        # 5. linear projection
        attn_output = self.out_proj(attn_output)
        
        # 6. dropout
        attn_output = self.resid_dropout(attn_output)

        return outputs
```

![](../images/megatron_attention.jpeg)
   
Megatron-LM은 Attention 레이어의 Q, K, V Linear projection과 Output projection 부분을 병렬화 합니다. 마찬가지로 Q, K, V Linear projection 부분은 Column parallelism, Output projection 부분은 Row parallelism으로 처리하여 **Column-Row의 패턴을 만듭니다.** 이를 통해 Attention 레이어에서도 MLP 레이어와 마찬가지로 `Scatter`, `All-gather` 연산을 생략 할 수 있습니다.

### Vocab Parallel Embedding
Megatron LM은 Word embedding 레이어도 역시 병렬화 합니다. 독특한 점은 Vocab size dimension을 기준으로 병렬화 한다는 점입니다. 예를 들어 Vocab size가 50000인 Word embedding matrix가 있다고 가정하면 이 matrix의 사이즈는 (50000, embedding_dim)인 됩니다. Megatron-LM은 여기에서 Vocab size dimension을 기준으로 matrix를 병렬화 합니다. 이러한 독특한 병렬화 기법을 **Vocab Parallel Embedding**이라고 합니다. 
  
![](../images/vpe_1.png)
 
위 그림은 병렬화를 하지 않은 상태에서의 Word embedding을 나타냅니다. 길이가 6인 시퀀스가 입력되면 [6, embedding_dim]의 사이즈를 갖는 입력 텐서를 만듭니다.

![](../images/vpe_2.png)
    
위 그림은 Vocab parallel embedding의 작동 방식을 나타냅니다. 기존의 임베딩 매트릭스를 절반으로 쪼개서 0번부터 24999번 토큰까지 담당하는 임베딩 매트릭스와 25000번부터 50000번 토큰까지 담당하는 임베딩 매트릭스로 분할합니다. 그리고 데이터가 들어오면 **해당 매트릭스가 커버하는 범위를 넘어서는 토큰은 마스킹**하여 처리합니다. 이후에 **마스킹 처리된 부분의 벡터는 전부 0으로 초기화** 한 뒤, 두 매트릭스를 **더하면 모든 단어의 벡터를 갖고 있는 완벽한 입력 텐서**가 됩니다.

```
"""
참고: VocabParallelEmbedding in megatron-lm/megatron/mpu/layers.py
"""

def forward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)

        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

    else:
        masked_input = input_
        # Get the embeddings.
    
    output_parallel = F.embedding(masked_input, self.weight,
                                  self.padding_idx, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq,
                                  self.sparse)

    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel[input_mask, :] = 0.0
    
    # Reduce across all the model parallel GPUs.
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output
```

그런데 여기에서 문제가 하나 발생합니다. Tensor parallelism은 반드시 짝수개의 GPU로 병렬화 되어야 하는데 52527은 짝수가 아니기 때문에 2로 나눌 수가 없습니다. 이를 위해 Word embedding matrix에 사용하지 않는 토큰을 추가하여 vocab size를 짝수로 만듭니다. 이를 `padded vocab size`라고 하며 Megatron-LM에서는 `make-vocab-size-divisible-by`이라는 argument로 vocab size를 조절할 수 있습니다. (vocab size가 설정한 값의 배수가 되도록 만듭니다.)

결론적으로 Megatron-LM은 Vocab Parallel Embedding을 적용하여 메모리 효율성을 더욱 높힐 수 있습니다.
 
### Vocab Parallel Cross Entropy
GPT2의 Causal Language Modeling이나 BERT의 Masked Language Modeling 같은 태스크는 최종 출력으로 자연어 토큰을 생성합니다. 따라서 마지막 Transformer 레이어를 거친 이후에 모델의 출력 사이즈는 (bsz, length, vocab_size)로 확장됩니다. (classification이나 tagging 같은 태스크는 해당하지 이에 않습니다.)
![](../images/lm_head.png)

이 때, 만약 입력과 출력 임베딩을 묶는다면(weight tying) Language Modeling Head (이하 LM Head)에 사용되는 Linear 레이어의 파라미터를 새로 초기화 시키는 대신 word embedding matrix를 사용하게 됩니다. 현재 공개된 Bert, GPT2, GPTNeo 등의 대부분 모델들의 출력 임베딩(LM Head)은 입력 임베딩과 묶여있습니다.
```
"""
참고 transformers/models/gpt_neo/modeling_gpt_neo.py
"""

class GPTNeoForCausalLM(GPTNeoPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head\.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]
    # 3. 그렇기 때문에 `lm_head.weight` 파라미터는 load 및 save하지 않습니다.
    # 굳이 동일한 텐서를 두번 저장하거나 로드 할 필요 없기 때문이죠.

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 1. 언뜻 보면 nn.Linear 레이어의 파라미터를 새로 할당해서 사용하는 것 처럼 보입니다.

        self.init_weights()
        # 2. 그러나 이 메서드를 호출하면서 입력과 출력 임베딩(lm head)을 묶게 됩니다. 
        # 이 때 word embeddig matrix의 weight를 nn.Linear 레이어의 weight로 복사하게 됩니다.
        # 복사는 deep-copy가 아닌 shallow-copy를 수행합니다. (reference가 아닌 value만 공유)
        # 따라서 `lm_head.weight`은 word embedding과 동일한 주소 공간에 있는 하나의 텐서입니다.

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
```
```
"""
참고 transformers/modeling_utils.py
"""

def init_weights(self):
    """
    If needed prunes and maybe initializes weights.
    """
    # Prune heads if needed
    if self.config.pruned_heads:
        self.prune_heads(self.config.pruned_heads)

    if _init_weights:
        # Initialize weights
        self.apply(self._init_weights)

        # weight tying을 지원하는 모델은 이 메서드가 호출됨과 동시에
        # 입력 임베딩과 출력 임베딩(= lm head)가 묶이게 됩니다.
        self.tie_weights()


def tie_weights(self):
    """
    Tie the weights between the input embeddings and the output embeddings.
    If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
    the weights instead.
    """
    output_embeddings = self.get_output_embeddings()
    if output_embeddings is not None and self.config.tie_word_embeddings:
        self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
        # 이 메서드가 호출되면서 output 임베딩(lm head)이 input 임베딩과 묶이게 됩니다.

    if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
        if hasattr(self, self.base_model_prefix):
            self = getattr(self, self.base_model_prefix)
        self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

    for module in self.modules():
        if hasattr(module, "_tie_weights"):
            module._tie_weights()
```

   
그러나 여기서 문제가 생깁니다. 일반적으로 LM Head로 부터 출력된 Logits과 Target 데이터 사이의 Loss를 계산할 때는 다음과 같은 과정이 일어납니다.
   
![](../images/vpce_1.png)

그러나 Megatron-LM은 Vocab Parallel Embedding을 사용하기 때문에 Embedding 레이어가 여러 디바이스를 걸쳐 분할되어 있습니다. 때문에 weight tying을 하게 된다면 **출력 임베딩(LM Head) 역시 여러 디바이스로 분할**되게 됩니다. 따라서 모델에서 출력되는 Logits의 사이즈는 vocab size를 분할한 사이즈가 됩니다. 

![](../images/vpce_2.png)
  
위 그림처럼 vocab size가 50,000이라면 원래는 (bsz, length, 50000)의 텐서가 출력되어야 하지만 위의 예시처럼 2개의 디바이스로 분할되어 있다면 (bsz, length, 25000)의 사이즈를 갖는 2개의 logits이 나오게 되며, 각 디바이스의 logits은 서로 다른 값을 갖게 될 것입니다. **이 것을 Parallel LM Logits이라고 부릅니다.** 이렇게 되면 target sentence와의 loss를 어떻게 계산해야 할까요? Traget 데이터에는 0번 부터 49999번째 토큰까지 모두 존재하는데 비해 logits의 사이즈는 그 절반밖에 되지 않으니까요.
    
![](../images/vpce_3.png)
  
이 경우 **기존의 cross entropy가 아닌 vocab parallel cross entropy라고 불리는 특별한 loss 함수를 사용**해야 합니다. Vocab parallel corss entropy loss의 연산은 위와 같이 진행됩니다. 계산된 Logit에서 해당 디바이스가 커버 할 수 있는 부분만 남기고 Masking하여 Loss를 계산합니다. 그리고 계산된 Loss들을 All-reduce 해서 최종 Loss를 계산합니다.

```
"""
참고: _VocabParallelCrossEntropy in megatron-lm/megatron/mpu/cross_entropy.py
"""

@staticmethod
def forward(ctx, vocab_parallel_logits, target):
    # Maximum value along vocab dimension across all GPUs.
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    torch.distributed.all_reduce(logits_max,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=get_tensor_model_parallel_group())

    # Subtract the maximum value.
    vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

    # Get the partition's vocab indecies
    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    vocab_start_index, vocab_end_index = get_vocab_range(
        partition_vocab_size, rank, world_size)

    # Create a mask of valid vocab ids (1 means it needs to be masked).
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target[target_mask] = 0

    # Get predicted-logits = logits[target].
    # For Simplicity, we convert logits to a 2-D tensor with size
    # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)
    predicted_logits[target_mask] = 0.0
    
    # All reduce is needed to get the chunks from other GPUs.
    torch.distributed.all_reduce(predicted_logits,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=get_tensor_model_parallel_group())

    # Sum of exponential of logits along vocab dimension across all GPUs.
    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)
    torch.distributed.all_reduce(sum_exp_logits,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=get_tensor_model_parallel_group())

    # Loss = log(sum(exp(logits))) - predicted-logit.
    loss = torch.log(sum_exp_logits) - predicted_logits

    # Store softmax, target-mask and masked-target for backward pass.
    exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
    ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

    return loss
```

### Megatron-LM으로 모델 학습해보기
Megatron-LM을 사용해서 모델을 학습해보도록 하겠습니다. Megaton-LM은 Hugging Face `transformers`와 같이 코드레벨로 사용하는 프레임워크가 아니라 이미 잘 짜여진 코드를 활용하여 모델을 만드는 데에 쓰입니다. 따라서 레포를 클론한 뒤에 진행하도록 하겠습니다.
<!--
```
# git과 wget이 설치되어있지 않다면 아래 명령어를 통해 설치합니다.
[glogin01]$ apt update && apt install git wget -y
Ign:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
Hit:2 http://archive.ubuntu.com/ubuntu bionic InRelease             
Ign:3 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
Hit:4 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
Get:5 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]m
Hit:6 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
Get:7 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]      
Get:10 http://archive.ubuntu.com/ubuntu bionic-backports InRelease [74.6 kB]   
Fetched 252 kB in 2s (126 kB/s)[0m 
Reading package lists... Done
Building dependency tree       
Reading state information... Done
54 packages can be upgraded. Run 'apt list --upgradable' to see them.
Reading package lists... Done
Building dependency tree       
Reading state information... Done
git is already the newest version (1:2.17.1-1ubuntu0.9).
wget is already the newest version (1.19.4-1ubuntu2.2).
0 upgraded, 0 newly installed, 0 to remove and 54 not upgraded.
```
-->
```
# Megatron-LM을 clone 합니다.
[glogin01]$ pwd
/scratch/qualis/large-scale-lm-tutorials/src/ch6_tensor_parallelism
[glogin01]$ git clone https://github.com/NVIDIA/Megatron-LM
Cloning into 'Megatron-LM'...
remote: Enumerating objects: 6281, done.
remote: Counting objects: 100% (2295/2295), done.
remote: Compressing objects: 100% (689/689), done.
remote: Total 6281 (delta 1667), reused 2188 (delta 1602), pack-reused 3986
Receiving objects: 100% (6281/6281), 2.29 MiB | 6.63 MiB/s, done.
Resolving deltas: 100% (4648/4648), done.
```
```
[glogin01]$ cd Megatron-LM

```

이제 필요한 몇가지 패키지를 설치해보도록 하겠습니다. Megatron-LM에는 `nltk`로 데이터를 문장단위로 분할해서 전처리 하는 기능이 있습니다. 저는 지금 이 기능을 사용하진 않을것이지만 설치되어 있지 않으면 에러가 발생하기 때문에 `nltk`를 설치하겠습니다.

```
[glogin01]$ pip install nltk
Requirement already satisfied: nltk in /opt/conda/lib/python3.7/site-packages (3.6.5)
Requirement already satisfied: tqdm in /opt/conda/lib/python3.7/site-packages (from nltk) (4.62.3)
Requirement already satisfied: click in /opt/conda/lib/python3.7/site-packages (from nltk) (8.0.3)
Requirement already satisfied: joblib in /opt/conda/lib/python3.7/site-packages (from nltk) (1.1.0)
Requirement already satisfied: regex>=2021.8.3 in /opt/conda/lib/python3.7/site-packages (from nltk) (2021.10.23)
Requirement already satisfied: importlib-metadata in /opt/conda/lib/python3.7/site-packages (from click->nltk) (4.8.1)
Requirement already satisfied: typing-extensions>=3.6.4 in /opt/conda/lib/python3.7/site-packages (from importlib-metadata->click->nltk) (3.7.4.3)
Requirement already satisfied: zipp>=0.5 in /opt/conda/lib/python3.7/site-packages (from importlib-metadata->click->nltk) (3.6.0)
WARNING: Running pip as root will break packages and permissions. You should install packages reliably by using venv: https://pip.pypa.io/warnings/venv
```

```
[glogin01]$ pip install pybind11
Requirement already satisfied: pybind11 in /opt/conda/lib/python3.7/site-packages (2.8.0)
WARNING: Running pip as root will break packages and permissions. You should install packages reliably by using venv: https://pip.pypa.io/warnings/venv
```

```
[glogin01]$ git clone https://github.com/NVIDIA/apex
[glogin01]$ cd apex
[glogin01]$ pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
.
.
.
RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED at "/opt/conda/conda-bld/pytorch_1720538459595/work/aten/src/ATen/cuda/CUDAContext.cpp":49, please report a bug to PyTorch. device=, num_gpus=
.
.
.
```
MIG (Multi-Instance GPU)에서 빌딩하면 위와 같이 런타임 에러가 발생합니다. 뉴론 시스템 로그인 노드 1번과 3번은 MIG 설정된 노드입니다. 노드 2번(neuron02.ksc.re.kr)으로 직접 로그인 한 후에 Apex를 다시 빌딩합니다.  
```
[globin02]$ cd apex
[glogin02]$ pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
[glogin02]$ cd ..
[glogin02]$ pwd
/scratch/qualis/large-scale-lm-tutorials/src/ch6_tensor_parallelism
```
Apex를 성공적으로 빌당되면 로그인 노드 1 - 3번 중에 아무 노드에서 계속해서 튜토리얼을 진행해도 됩니다.  
이제 데이터셋을 만들어보도록 하겠습니다. Megatron-LM으로 모델을 Pre-training을 할 때는 `{\"text\": \"샘플\"}`과 같은 json 구조가 여러라인으로 구성된 간단한 구조의 jsonl 파일을 만들면 되고, Fine-tuning의 경우는 해당 태스크에 맞게 데이터셋을 구성해야 합니다. 본 튜토리얼에서는 Pre-training만 다루고 있기 때문에 Fine-tuning이 필요하시면 Megatron-LM 깃헙 레포를 참고해주세요.
```
"""
src/ch6_tensor_parallelism/megatron_datasets.py
"""

import json
import os
from datasets import load_dataset

train_samples, min_length = 10000, 512
filename = "megatron_datasets.jsonl"
curr_num_datasets = 0

if os.path.exists(filename):
    os.remove(filename)

datasets = load_dataset("wikitext", "wikitext-103-raw-v1")
datasets = datasets.data["train"]["text"]
dataset_fp_write = open(filename, mode="w", encoding="utf-8")

for sample in datasets:
    sample = sample.as_py()

    if len(sample) >= min_length:
        line = json.dumps(
            {"text": sample},
            ensure_ascii=False,
        )

        dataset_fp_write.write(line + "\n")
        curr_num_datasets += 1

        # 튜토리얼이기 때문에 적은 양의 데이터만 만들겠습니다.
        if curr_num_datasets >= train_samples:
            break

dataset_fp_read = open(filename, mode="r", encoding="utf-8")
dataset_read = dataset_fp_read.read().splitlines()[:3]

# 데이터의 구조를 확인합니다.
for sample in dataset_read:
    print(sample, end="\n\n")
```

```
[glogin01]$ cd Megatron-LM/
[glogin01]$ python ../megatron_datasets.py
Downloading readme: 100%|██████████████████████████████████████████████| 10.5k/10.5k [00:00<00:00, 25.5kB/s]
{"text": " Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the \" Nameless \" , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit \" Calamaty Raven \" . \n"}

{"text": " The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . \n"}

{"text": " It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 . \n"}

```

Tokenization에 사용할 Vocab을 다운로드 받습니다.
```
[glogin01]$ wget https://huggingface.co/gpt2/raw/main/vocab.json
[glogin01]$ wget https://huggingface.co/gpt2/raw/main/merges.txt
--2021-10-24 01:46:38--  https://huggingface.co/gpt2/raw/main/vocab.json
Resolving huggingface.co (huggingface.co)... 54.84.200.39, 107.23.77.87, 2600:1f18:147f:e850:859c:8fe1:ce4c:b9ed, ...
Connecting to huggingface.co (huggingface.co)|54.84.200.39|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1042301 (1018K) [application/json]
Saving to: ‘vocab.json’

vocab.json          100%[===================>]   1018K   769KB/s    in 1.3s    

2021-10-24 01:46:40 (769 KB/s) - ‘vocab.json’ saved [1042301/1042301]

--2021-10-24 01:46:40--  https://huggingface.co/gpt2/raw/main/merges.txt
Resolving huggingface.co (huggingface.co)... 54.84.200.39, 107.23.77.87, 2600:1f18:147f:e800:db5d:bef3:91a:a059, ...
Connecting to huggingface.co (huggingface.co)|54.84.200.39|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 456318 (446K) [text/plain]
Saving to: ‘merges.txt’

merges.txt          100%[===================>] 445.62K   332KB/s    in 1.3s    

2021-10-24 01:46:43 (332 KB/s) - ‘merges.txt’ saved [456318/456318]
```
```
[glogin01]$ ls
LICENSE    megatron/                pretrain_ict.py  tools/
README.md  megatron_datasets.jsonl  pretrain_t5.py   vocab.json
apex/      merges.txt               pretrain_vit.py
examples/  pretrain_bert.py         tasks/
images/    pretrain_gpt.py          tests/
```
이제 Dataset을 전처리합니다. 여기서 수행하는 전처리는 Tokenization과 Binarization을 함께 수행합니다. Megatron-LM의 전처리 코드는 Fairseq의 Indexed dataset의 코드를 카피해서 사용하고 있습니다. Fairseq의 데이터셋 전처리에 사용되는 방식은 크게 `lazy`, `cached`, `mmap` 등 크게 3가지가 존재하는데, 전처리 방식들에 대해 간략하게 설명하고 진행하겠습니다.

#### 1) Lazy
`lazy`는 필요한 데이터를 매 스텝마다 디스크에서 메모리로 불러옵니다. 즉, `Dataset` 클래스에서 `__getitem__()`이 호출 될 때마다 지정된 주소에 접근하여 데이터를 메모리로 로드하는 방식입니다. 그러나 매 스텝마다 File Buffer를 통해 디스크와의 I/O를 수행하기 때문에 처리 속도가 다소 느릴 수 있습니다.
```
"""
참고: fairseq/fairseq/data/indexed_dataset.py 
주석은 제가 직접 추가하였습니다.
"""


from typing import Union
import numpy as np


def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
    if not self.data_file:
        # 파일 버퍼 로드
        self.read_data(self.path)

    if isinstance(idx, int):
        # 인덱스 유효성 체크
        self.check_index(idx)

        # 로드할 텐서 사이즈 계산
        tensor_size = self.sizes[self.dim_offsets[idx] : self.dim_offsets[idx + 1]]

        # 텐서를 담을 빈 메모리 공간 할당
        array = np.empty(tensor_size, dtype=self.dtype)

        # offset을 기반으로 읽어올 파일의 디스크 주소를 지정
        self.data_file.seek(self.data_offsets[idx] * self.element_size)

        # 디스크로부터 메모리로 데이터 로드 (파일 I/O)
        self.data_file.readinto(array)
        return array
```

#### 2) Cached
`cached`는 모든 데이터를 학습 이전에 prefetch 하여 인메모리에 올려두고 접근하는 방식입니다. 학습 중에 데이터 로딩을 위해 디스크에 접근하지 않기 때문에 속도가 다른 방식들 보다는 빠른 편이지만 메모리의 크기에는 한계가 존재하므로 데이터셋의 용량이 매우 큰 경우에는 사용하기 어렵습니다.
```
"""
참고: fairseq/fairseq/data/indexed_dataset.py
주석은 제가 직접 추가하였습니다.
"""


from typing import List


def prefetch(self, indices: List[int]) -> None:
    if all(i in self.cache_index for i in indices):
        # 이미 모든 데이터가 캐싱되었다면 메서드 종료
        return

    if not self.data_file:
        # 파일버퍼가 로드되지 않았다면 파일버퍼를 로드
        self.read_data(self.path)

    # 연속된 전체 메모리 사이즈를 계산하기 위해서 indices를 정렬
    indices = sorted(set(indices))

    total_size = 0
    for i in indices:
        total_size += self.data_offsets[i + 1] - self.data_offsets[i]

    # 캐시로 사용할 전체 메모리 공간 할당
    self.cache = np.empty(
        total_size,
        dtype=self.dtype,
    )

    self.cache_index.clear()
    ptx = 0

    for i in indices:
        # 전체 어레이 사이즈를 저장
        self.cache_index[i] = ptx

        # offset으로부터 데이터 사이즈를 계산해서 현재 샘플이 저장될 메모리 공간을 변수에 할당
        size = self.data_offsets[i + 1] - self.data_offsets[i]
        array = self.cache[ptx : ptx + size]

        # offset을 기반으로 읽어올 파일의 디스크 주소를 지정
        self.data_file.seek(self.data_offsets[i] * self.element_size)

        # 현재의 샘플을 할당된 메모리에 씀
        self.data_file.readinto(array)
        ptx += size

    if self.data_file:
        # 파일버퍼의 데이터를 모두 불러왔으니 버퍼를 닫고 참조를 해제
        self.data_file.close()
        self.data_file = None
```
```
"""
참고: fairseq/fairseq/data/indexed_dataset.py
주석은 제가 직접 추가하였습니다.
"""

def __getitem__(self, idx: Union[int, tuple]) -> Union[np.ndarray, List]:
    if isinstance(idx, int):
        # 인덱스 유효성 검사
        self.check_index(idx)

        # 텐서 사이즈 계산
        tensor_size = self.sizes[self.dim_offsets[idx] : self.dim_offsets[idx + 1]]

        # 메모리 공간 할당
        array = np.empty(tensor_size, dtype=self.dtype)

        # 프리패치된 데이터를 로드 (파일 I/O가 일어나지 않음)
        ptx = self.cache_index[idx]

        # 캐시에 프리패치된 데이터를 메모리 공간에 복사
        np.copyto(array, self.cache[ptx : ptx + array.size])
        return array

    elif isinstance(idx, slice):
        return [self[i] for i in range(*idx.indices(len(self)))]
```

#### 3) Mmap
`mmap`은 `lazy`와 동일하게 매 스텝마다 필요한 만큼의 데이터를 메모리로 로드하지만 File Buffer 대신 Memory Map을 사용하는 방식입니다. Memory Map은 File Buffer와 달리 현재 프로세스에게 할당된 가상메모리에 파일의 주소를 맵핑시키기 때문에 데이터가 마치 메모리 상에 존재하는 것 처럼 작업할 수 있습니다. 디스크와의 직접적인 I/O를 수행하지 않으며 페이지(4KB) 단위로 데이터를 로드 할 수 있고 실제로 메모리에서 모든 작업이 일어나기 때문에 File Buffer에 비해 처리 속도가 비교적 빠른 편입니다. 
```
"""
참고: fairseq/fairseq/data/indexed_dataset.py
주석은 제가 직접 추가하였습니다.
"""

def __init__(self, path: str):
    with open(path, "rb") as stream:
        # 1. 매직 스트링 로드
        # 매직스트링은 현재 저장된 데이터 구조가 어떤 형식인지 구분하기 위한 것.
        # lazy인지 mmap인지 등등... (cached는 lazy와 같은 값을 가짐)
        magic_test = stream.read(9)
        assert magic_test == self._HDR_MAGIC, (
            "Index file doesn't match expected format. "
            "Please check your configuration file."
        )
        
        # 2. 버전 로드 (little endian unsigned long long)
        # 코드 보니까 버전은 무조건 1로 쓰던데 별 의미 없는 변수인듯?
        # b'\x01\x00\x00\x00\x00\x00\x00\x00'
        version = struct.unpack("<Q", stream.read(8))
        assert (1,) == version

        # 3. 데이터 타입 로드 (little endian unsigned char)
        (dtype_code,) = struct.unpack("<B", stream.read(1))
        self._dtype = _code_to_dtype[dtype_code]
        self._dtype_size = self._dtype().itemsize

        # 4. 데이터셋의 전체 길이 로드 (little endian unsigned long long)
        self._len = struct.unpack("<Q", stream.read(8))[0]

        # 5. 전체 샘플의 개수 로드 (little endian unsigned long long)
        self._doc_count = struct.unpack("<Q", stream.read(8))[0]
        offset = stream.tell()

    # 6. 캐시 warmup 수행 
    _warmup_mmap_file(path)

    # 7. 메모리맵 어레이 생성
    self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
    self._bin_buffer = memoryview(self._bin_buffer_mmap)

    # 8. 샘플들의 사이즈가 담긴 데이터를 메모리맵 어레이로 로드
    self._sizes = np.frombuffer(
        self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
    )

    # 9. 데이터 포인터(위치) 값들을 메모리맵 어레이로 로드
    self._pointers = np.frombuffer(
        self._bin_buffer,
        dtype=np.int64,
        count=self._len,
        offset=offset + self._sizes.nbytes,
    )

    # 10. 데이터 인덱스들을 메모리맵 어레이로 로드
    self._doc_idx = np.frombuffer(
        self._bin_buffer,
        dtype=np.int64,
        count=self._doc_count,
        offset=offset + self._sizes.nbytes + self._pointers.nbytes,
    )
```

```
"""
참고: fairseq/fairseq/data/indexed_dataset.py
주석은 제가 직접 추가하였습니다.
"""

from typing import Union

def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
    if not self.data_file:
        # 인덱스 파일이 로드되지 않았다면 로드
        self.read_data(self.path)

    if isinstance(idx, int):
        # 인덱스 유효성 검사
        self.check_index(idx)

        # 텐서 사이즈 계산
        tensor_size = self.sizes[self.dim_offsets[idx] : self.dim_offsets[idx + 1]]

        # 메모리 공간 할당
        array = np.empty(tensor_size, dtype=self.dtype)

        # offset을 기반으로 읽어올 데이터의 가상메모리 주소를 지정
        self.data_file.seek(self.data_offsets[idx] * self.element_size)

        # 메모리로 데이터 로드
        self.data_file.readinto(array)
        return array

    elif isinstance(idx, slice):
        start, stop, step = idx.indices(len(self))
        if step != 1:
            # 슬라이스로 입력시 반드시 반드시 연속되어야 함
            raise ValueError("Slices into indexed_dataset must be contiguous")

        # 텐서의 사이즈들이 담긴 리스트와 전체 합을 계산
        sizes = self.sizes[self.dim_offsets[start] : self.dim_offsets[stop]]
        total_size = sum(sizes)

        # 필요한 만큼의 메모리 공간 할당
        array = np.empty(total_size, dtype=self.dtype)

        # offset을 기반으로 읽어올 데이터의 가상메모리 주소를 지정
        self.data_file.seek(self.data_offsets[start] * self.element_size)
        self.data_file.readinto(array)

        # 텐서 사이즈를 기반으로 여러개의 샘플로 분할 
        offsets = list(accumulate(sizes))
        sentences = np.split(array, offsets[:-1])
        return sentences
```

이제 데이터셋 전처리를 수행합니다. 저는 `mmap` 방식을 사용하여 전처리 하도록 하겠습니다. 

이 때, `append-eod`라는 옵션이 보입니다. Megatron-LM은 패딩을 만들지 않기 위해 Pre-train 시에 모든 데이터를 연결해서 학습합니다. 예를 들어, `{\"text\": \"I am a boy.\"}`\"라는 샘플과 `{\"text\": \"You are so lucky\"}`라는 샘플이 있으면 Pre-train 할 때는 `input = \"I am a boy. You are so lucky ...\"`과 같이 모든 샘플을 연결합니다. 그리고나서 사용자가 설정한 길이(e.g. 2048)로 데이터를 잘라서 학습합니다. 

그러나 이렇게 모든 샘플을 하나의 문자열로 연결해버리면 샘플과 샘플사이에 구분이 없어지기 때문에 문제가 될 수 있는데요. `append-eod` 옵션을 추가하면 샘플들 사이에 `end of document`로써 토큰을 추가하여 샘플과 샘플을 구분합니다. GPT2의 경우, `eod` 토큰은 `eos`토큰으로 설정되어 있습니다. 

```
(large-scale-lm) [glogin01]$ python tools/preprocess_data.py  --input megatron_datasets.jsonl  --output-prefix my-gpt2    --vocab-file vocab.json    --tokenizer-type GPT2BPETokenizer   --merge-file merges.txt   --append-eod --workers 8
Opening megatron_datasets.jsonl
Time to startup: 0.2418653964996338
Processed 1000 documents (2414.283880556293 docs/s, 2.1787396095175984 MB/s).
Processed 2000 documents (3086.184695575906 docs/s, 2.699775874171669 MB/s).
Processed 3000 documents (3535.2736126489676 docs/s, 3.1018583425089634 MB/s).
Processed 4000 documents (3640.7253406802256 docs/s, 3.190650072794039 MB/s).
Processed 5000 documents (3853.9685188326316 docs/s, 3.3612373264254836 MB/s).
Processed 6000 documents (3870.7082725855316 docs/s, 3.3536631789648315 MB/s).
Processed 7000 documents (4040.228723286409 docs/s, 3.482047137539327 MB/s).
Processed 8000 documents (4217.510323365528 docs/s, 3.638737196582043 MB/s).
Processed 9000 documents (4320.326483412309 docs/s, 3.731670701586145 MB/s).
Processed 10000 documents (4462.9152662988345 docs/s, 3.8410912984141734 MB/s).
```

데이터셋 전처리가 완료되었습니다. 데이터를 확인해봅시다
- my-gpt2_text_document.bin
- my-gpt2_text_document.idx
와 같은 파일들이 생겼습니다. `idx`파일은 데이터의 위치 등의 메타데이터가 저장되어 있으며, `bin` 파일에는 실제로 Tokenized 된 데이터가 저장되어 있습니다.
```
[glogin01]$ ls
./                  examples/       MANIFEST.in                pretrain_mamba.py            pytest.ini
../                 .flake8         megatron/                  pretrain_retro.py            README.md
CHANGELOG.md        .git/           megatron_datasets.jsonl    pretrain_t5.py               setup.py
CODEOWNERS          .github/        merges.txt                 pretrain_vision_classify.py  tasks/
CONTRIBUTING.md     .gitignore      my-gpt2_text_document.bin  pretrain_vision_dino.py      tests/
.coveragerc         .gitlab/        my-gpt2_text_document.idx  pretrain_vision_inpaint.py   tools/
Dockerfile.ci       .gitlab-ci.yml  pretrain_bert.py           pretrain_vlm.py              vocab.json
Dockerfile.linting  images/         pretrain_gpt.py            .pylintrc
docs/               LICENSE         pretrain_ict.py            pyproject.toml
```

모델을 시작하기 전에 Mix Precesion 지원을 위해서 [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)을 설치해야 한다. 참고로, 뉴론에 `cuDNN 8.x`버전 설치되어 있으므로 8.x버전과 호환하는 과거 TransformerEngine 버전(e.g., v1.10) 설치해야 한다. 최신 TransformerEngine로 빌드할 경우에 cuDNN 9.x 버전에만 있는 `$CUDA_HOME/inlcude/cudnn_graph.h`파일을 찾을 수 없다는 빌드 에러가 발생한다. 

```
#(large-scale-lm) [glogin01]$ pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.10 -v
(large-scale-lm) [glogin01]$ pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.10
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting git+https://github.com/NVIDIA/TransformerEngine.git@stable
  Cloning https://github.com/NVIDIA/TransformerEngine.git (to revision stable) to /tmp/pip-req-build-qfbxqfc2
  Running command git clone --quiet https://github.com/NVIDIA/TransformerEngine.git /tmp/pip-req-build-qfbxqfc2
  Running command git checkout -b stable --track origin/stable
  Switched to a new branch 'stable'
  Branch stable set up to track remote branch stable from origin.
  Resolved https://github.com/NVIDIA/TransformerEngine.git to commit 08a85d3b2657f1d4e0b478f6682c17fe6bba8b05
  Running command git submodule update --init --recursive -q
  Preparing metadata (setup.py) ... done
Requirement already satisfied: pydantic in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from transformer_engine==1.10.0+08a85d3) (2.9.1)
Requirement already satisfied: packaging in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from transformer_engine==1.10.0+08a85d3) (24.1)
Requirement already satisfied: importlib-metadata>=1.0 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from transformer_engine==1.10.0+08a85d3) (8.4.0)
Requirement already satisfied: flash-attn!=2.0.9,!=2.1.0,<=2.5.8,>=2.0.6 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from transformer_engine==1.10.0+08a85d3) (2.4.2)
Requirement already satisfied: torch in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from transformer_engine==1.10.0+08a85d3) (2.4.0)
Requirement already satisfied: einops in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from flash-attn!=2.0.9,!=2.1.0,<=2.5.8,>=2.0.6->transformer_engine==1.10.0+08a85d3) (0.8.0)
Requirement already satisfied: ninja in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from flash-attn!=2.0.9,!=2.1.0,<=2.5.8,>=2.0.6->transformer_engine==1.10.0+08a85d3) (1.11.1.1)
Requirement already satisfied: zipp>=0.5 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from importlib-metadata>=1.0->transformer_engine==1.10.0+08a85d3) (3.20.1)
Requirement already satisfied: annotated-types>=0.6.0 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from pydantic->transformer_engine==1.10.0+08a85d3) (0.7.0)
Requirement already satisfied: pydantic-core==2.23.3 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from pydantic->transformer_engine==1.10.0+08a85d3) (2.23.3)
Requirement already satisfied: typing-extensions>=4.6.1 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from pydantic->transformer_engine==1.10.0+08a85d3) (4.11.0)
Requirement already satisfied: filelock in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from torch->transformer_engine==1.10.0+08a85d3) (3.13.1)
Requirement already satisfied: sympy in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from torch->transformer_engine==1.10.0+08a85d3) (1.13.2)
Requirement already satisfied: networkx in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from torch->transformer_engine==1.10.0+08a85d3) (3.2.1)
Requirement already satisfied: jinja2 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from torch->transformer_engine==1.10.0+08a85d3) (3.1.4)
Requirement already satisfied: fsspec in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from torch->transformer_engine==1.10.0+08a85d3) (2024.6.1)
Requirement already satisfied: MarkupSafe>=2.0 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from jinja2->torch->transformer_engine==1.10.0+08a85d3) (2.1.3)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /scratch/qualis/miniconda3/envs/megatron/lib/python3.10/site-packages (from sympy->torch->transformer_engine==1.10.0+08a85d3) (1.3.0)
Building wheels for collected packages: transformer_engine
  Building wheel for transformer_engine (setup.py) ... done
  Created wheel for transformer_engine: filename=transformer_engine-1.10.0+08a85d3-cp310-cp310-linux_x86_64.whl size=120308837 sha256=31b1990d90b87e4e4cbf3f762b66b8b1cf4f2f3b148a293045936b3a60f7b9d1
  Stored in directory: /tmp/pip-ephem-wheel-cache-0yc2vyhu/wheels/ab/50/b6/2909ca23194f6269dc6544353545c76efae2d0cc25c692828c
Successfully built transformer_engine
Installing collected packages: transformer_engine
Successfully installed transformer_engine-1.10.0+08a85d3
```

먼저 뉴론 시스템에서 노드 1개와 GPU 4개를 할당받는다. src/ch6 디렉토리로 이동하고 모듈을 로드한다.

```
[glogin01]$ salloc --partition=cas_v100nv_8 -J debug --nodes=1 --time=8:00:00 --gres=gpu:4 --comment pytorch
...
salloc: Nodes gpu[05] are ready for job
[gpu05]$ pwd
/scratch/qualis/git-projects/large-scale-lm-tutorials/src/ch6_tensor_parallelism/Megatron-LM
[gpu05]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.1
[gpu05]$ conda activate large-scale-lm
```

이제 모델 학습을 시작해보겠습니다. 
```
# 일단 Tensor parallelism만 사용해보도록 하겠습니다.
# Data parallelism과 Pipeline parallelism은 Multi-dimensional Parallelism 세션에서 사용해봅시다. :)
# 학습은 200 스텝만 시키도록 하겠습니다. 실제 학습할 땐 더 많은 숫자로 설정해주세요.


(large-scale-lm) [gpu05]$ CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4  pretrain_gpt.py     --tensor-model-parallel-size 4     --pipeline-model-parallel-size 1         --num-layers 24     --hidden-size 1024     --num-attention-heads 16     --seq-length 1024     --max-position-embeddings 1024     --micro-batch-size 4     --global-batch-size 16     --lr 0.00015     --train-iters 200     --lr-decay-iters 320000     --lr-decay-style cosine     --min-lr 1.0e-5     --weight-decay 1e-2     --lr-warmup-fraction .01     --clip-grad 1.0     --fp16 --data-path  my-gpt2_text_document     --vocab-file vocab.json     --merge-file merges.txt     --split 949,50,1 --log-interval 10     --save-interval 50    --eval-interval 100     --eval-iters 10 --distributed-backend nccl  --save checkpoints/gpt2_345m_dist_mp     --load  checkpoints/gpt2_345m_dist_mp --attention-softmax-in-fp32 --sequence-parallel
W0914 19:43:58.713000 47642202304448 torch/distributed/run.py:757]
W0914 19:43:58.713000 47642202304448 torch/distributed/run.py:757] *****************************************
W0914 19:43:58.713000 47642202304448 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0914 19:43:58.713000 47642202304448 torch/distributed/run.py:757] *****************************************
using world size: 4, data-parallel size: 1, context-parallel size: 1, tensor-model-parallel size: 4, encoder-tensor-model-parallel size: 0, pipeline-model-parallel size: 1, encoder-pipeline-model-parallel size: 0
WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False since non-interleaved schedule does not support overlapping p2p communication and aligned param AG
WARNING: Setting args.check_for_nan_in_loss_and_grad to False since dynamic loss scaling is being used
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  add_qkv_bias .................................... False
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  align_grad_reduce ............................... True
  align_param_gather .............................. False
  app_tag_run_name ................................ None
  app_tag_run_version ............................. 0.0.0
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... False
  apply_residual_connection_post_layernorm ........ False
  apply_rope_fusion ............................... True
  async_save ...................................... None
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... True
  auto_detect_ckpt_format ......................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  bias_swiglu_fusion .............................. True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  calculate_per_token_loss ........................ False
  check_for_nan_in_loss_and_grad .................. False
  check_weight_hash_across_dp_replicas_interval ... None
  ckpt_assume_constant_structure .................. False
  ckpt_convert_format ............................. None
  ckpt_convert_save ............................... None
  ckpt_convert_update_legacy_dist_opt_format ...... False
  ckpt_format ..................................... torch_dist
  ckpt_fully_parallel_load ........................ False
  ckpt_fully_parallel_save ........................ True
  ckpt_fully_parallel_save_deprecated ............. False
  ckpt_step ....................................... None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  clone_scatter_output_in_embedding ............... True
  config_logger_dir ...............................
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  context_parallel_size ........................... 1
  create_attention_mask_in_dataloader ............. True
  cross_entropy_loss_fusion ....................... False
  data_cache_path ................................. None
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... ['my-gpt2_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  ddp_average_in_collective ....................... False
  ddp_bucket_size ................................. None
  decoder_first_pipeline_num_layers ............... None
  decoder_last_pipeline_num_layers ................ None
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  decoupled_lr .................................... None
  decoupled_min_lr ................................ None
  decrease_batch_size_if_needed ................... False
  defer_embedding_wgrad_compute ................... False
  deprecated_use_mcore_models ..................... False
  deterministic_mode .............................. False
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  disable_straggler_on_startup .................... False
  dist_ckpt_format_deprecated ..................... None
  dist_ckpt_strictness ............................ assume_ok_unexpected
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  enable_ft_package ............................... False
  enable_one_logger ............................... True
  encoder_num_layers .............................. 24
  encoder_pipeline_model_parallel_size ............ 0
  encoder_seq_length .............................. 1024
  encoder_tensor_model_parallel_size .............. 0
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  expert_model_parallel_size ...................... 1
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8 ............................................. None
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_param_gather ................................ False
  fp8_wgrad ....................................... True
  global_batch_size ............................... 16
  gradient_accumulation_fusion .................... True
  group_query_attention ........................... False
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hybrid_attention_ratio .......................... 0.0
  hybrid_mlp_ratio ................................ 0.0
  hybrid_override_pattern ......................... None
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  lazy_mpu_init ................................... None
  load ............................................ checkpoints/gpt2_345m_dist_mp
  local_rank ...................................... 0
  log_interval .................................... 10
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_progress .................................... False
  log_straggler ................................... False
  log_throughput .................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  logging_level ................................... None
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_init .................................. 0.0
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  lr_wsd_decay_iters .............................. None
  lr_wsd_decay_samples ............................ None
  lr_wsd_decay_style .............................. exponential
  make_vocab_size_divisible_by .................... 128
  manual_gc ....................................... False
  manual_gc_eval .................................. True
  manual_gc_interval .............................. 0
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... merges.txt
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_bin_files .................................. True
  mock_data ....................................... False
  moe_aux_loss_coeff .............................. 0.0
  moe_expert_capacity_factor ...................... None
  moe_extended_tp ................................. False
  moe_grouped_gemm ................................ False
  moe_input_jitter_eps ............................ None
  moe_layer_recompute ............................. False
  moe_pad_expert_input_to_capacity ................ False
  moe_per_layer_logging ........................... False
  moe_router_load_balancing_type .................. aux_loss
  moe_router_pre_softmax .......................... False
  moe_router_topk ................................. 2
  moe_shared_expert_intermediate_size ............. None
  moe_shared_expert_overlap ....................... False
  moe_token_dispatcher_type ....................... allgather
  moe_token_drop_policy ........................... probs
  moe_use_upcycling ............................... False
  moe_z_loss_coeff ................................ None
  nccl_communicator_config_path ................... None
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  non_persistent_ckpt_type ........................ None
  non_persistent_global_ckpt_dir .................. None
  non_persistent_local_ckpt_algo .................. fully_parallel
  non_persistent_local_ckpt_dir ................... None
  non_persistent_save_interval .................... None
  norm_epsilon .................................... 1e-05
  normalization ................................... LayerNorm
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_dataset_builder_threads ..................... 1
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_query_groups ................................ 1
  num_workers ..................................... 2
  one_logger_async ................................ False
  one_logger_project .............................. megatron-lm
  one_logger_run_name ............................. None
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_grad_reduce ............................. False
  overlap_p2p_comm ................................ False
  overlap_param_gather ............................ False
  overlap_param_gather_with_optimizer_step ........ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  position_embedding_type ......................... learned_absolute
  pretrained_checkpoint ........................... None
  profile ......................................... False
  profile_ranks ................................... [0]
  profile_step_end ................................ 12
  profile_step_start .............................. 10
  qk_layernorm .................................... False
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ None
  renormalize_blend_weights ....................... False
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_attention_gate ............................ 1
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_project_dir ............................... None
  retro_verify_neighbor_count ..................... True
  rotary_base ..................................... 10000
  rotary_interleaved .............................. False
  rotary_percent .................................. 1.0
  rotary_seq_len_interpolation_factor ............. None
  s3_cache_path ................................... None
  sample_rate ..................................... 1.0
  save ............................................ checkpoints/gpt2_345m_dist_mp
  save_interval ................................... 50
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... True
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  skip_train ...................................... False
  skipped_train_samples ........................... 0
  spec ............................................ None
  split ........................................... 949,50,1
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  straggler_ctrlr_port ............................ 65535
  straggler_minmax_count .......................... 1
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  tensor_model_parallel_size ...................... 4
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  test_mode ....................................... False
  tiktoken_num_special_tokens ..................... 1000
  tiktoken_pattern ................................ None
  tiktoken_special_tokens ......................... None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  tp_comm_bulk_dgrad .............................. True
  tp_comm_bulk_wgrad .............................. True
  tp_comm_overlap ................................. False
  tp_comm_overlap_ag .............................. True
  tp_comm_overlap_cfg ............................. None
  tp_comm_overlap_rs .............................. True
  tp_comm_overlap_rs_dgrad ........................ False
  tp_comm_split_ag ................................ True
  tp_comm_split_rs ................................ True
  train_data_path ................................. None
  train_iters ..................................... 200
  train_samples ................................... None
  train_sync_interval ............................. None
  transformer_impl ................................ transformer_engine
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_cpu_initialization .......................... None
  use_dist_ckpt ................................... True
  use_dist_ckpt_deprecated ........................ False
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_legacy_models ............................... False
  use_one_sent_docs ............................... False
  use_pytorch_profiler ............................ False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. False
  use_tp_pp_dp_mapping ............................ False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... vocab.json
  vocab_size ...................................... None
  wandb_exp_name ..................................
  wandb_project ...................................
  wandb_save_dir ..................................
  weight_decay .................................... 0.01
  weight_decay_incr_style ......................... constant
  wgrad_deferral_limit ............................ 0
  world_size ...................................... 4
  yaml_cfg ........................................ None
-------------------- end of arguments ---------------------
INFO:megatron.core.num_microbatches_calculator:setting number of microbatches to constant 4
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 431 dummy tokens (new size: 50688)
> initializing torch distributed ...
WARNING: one_logger package is required to enable e2e metrics tracking. please go to https://confluence.nvidia.com/display/MLWFO/Package+Repositories for details to install it
> initialized tensor model parallel with size 4
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory `/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/datasets'
make: Nothing to be done for `default'.
make: Leaving directory `/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/datasets'
>>> done with dataset index builder. Compilation time: 0.044 seconds
> compiling and loading fused kernels ...
>>> done with compiling and loading fused kernels. Compilation time: 1.855 seconds
[rank1]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
[rank3]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
[rank2]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
[rank0]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
time to initialize megatron (seconds): 4.541
[after megatron is initialized] datetime: 2024-09-14 19:44:09
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (2, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (3, 0): 89714688
INFO:megatron.core.distributed.distributed_data_parallel:Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=False, overlap_grad_reduce=False, overlap_param_gather=False, align_param_gather=False, use_distributed_optimizer=False, check_for_nan_in_grad=False, bucket_size=None, average_in_collective=False, fp8_param_gather=False)
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 1
Params for bucket 1 (89714688 elements):
        module.decoder.layers.14.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.6.mlp.linear_fc1.weight
        module.decoder.layers.5.mlp.linear_fc1.weight
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.0.mlp.linear_fc2.weight
        module.decoder.layers.12.mlp.linear_fc1.layer_norm_weight
        module.decoder.final_layernorm.weight
        module.decoder.layers.13.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.4.mlp.linear_fc1.bias
        module.decoder.layers.4.mlp.linear_fc1.weight
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.9.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.0.self_attention.linear_qkv.bias
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.layers.6.mlp.linear_fc2.bias
        module.decoder.layers.5.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.3.mlp.linear_fc1.weight
        module.decoder.layers.2.mlp.linear_fc2.bias
        module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.3.self_attention.linear_proj.bias
        module.decoder.layers.18.mlp.linear_fc2.weight
        module.decoder.layers.23.mlp.linear_fc2.weight
        module.decoder.layers.22.mlp.linear_fc2.weight
        module.decoder.layers.21.mlp.linear_fc2.weight
        module.decoder.layers.20.mlp.linear_fc2.weight
        module.decoder.layers.19.mlp.linear_fc2.weight
        module.decoder.layers.12.mlp.linear_fc2.weight
        module.decoder.layers.11.mlp.linear_fc2.weight
        module.decoder.layers.5.self_attention.linear_qkv.bias
        module.decoder.layers.8.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.1.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.8.mlp.linear_fc1.bias
        module.decoder.layers.3.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.9.mlp.linear_fc2.weight
        module.decoder.layers.8.mlp.linear_fc2.weight
        module.decoder.layers.7.mlp.linear_fc2.weight
        module.decoder.layers.6.self_attention.linear_qkv.weight
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.5.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.0.mlp.linear_fc1.weight
        module.decoder.layers.13.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.5.self_attention.linear_proj.bias
        module.decoder.layers.4.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.3.self_attention.linear_qkv.bias
        module.decoder.layers.1.mlp.linear_fc2.bias
        module.decoder.layers.1.mlp.linear_fc1.bias
        module.decoder.layers.3.mlp.linear_fc2.weight
        module.decoder.layers.1.self_attention.linear_qkv.bias
        module.embedding.position_embeddings.weight
        module.decoder.layers.0.self_attention.linear_qkv.weight
        module.decoder.layers.13.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.4.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.10.self_attention.linear_qkv.bias
        module.decoder.layers.11.mlp.linear_fc1.weight
        module.decoder.layers.17.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.16.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.15.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.14.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.10.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.5.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.2.self_attention.linear_qkv.bias
        module.decoder.final_layernorm.bias
        module.decoder.layers.6.self_attention.linear_proj.weight
        module.decoder.layers.5.self_attention.linear_qkv.weight
        module.decoder.layers.4.self_attention.linear_proj.bias
        module.decoder.layers.3.mlp.linear_fc1.bias
        module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.embedding.word_embeddings.weight
        module.decoder.layers.9.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.1.self_attention.linear_proj.weight
        module.decoder.layers.9.mlp.linear_fc1.bias
        module.decoder.layers.2.self_attention.linear_qkv.weight
        module.decoder.layers.2.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.6.mlp.linear_fc2.weight
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.self_attention.linear_proj.weight
        module.decoder.layers.0.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.14.self_attention.linear_proj.weight
        module.decoder.layers.13.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.3.self_attention.linear_proj.weight
        module.decoder.layers.2.mlp.linear_fc1.weight
        module.decoder.layers.2.mlp.linear_fc1.bias
        module.decoder.layers.1.mlp.linear_fc1.weight
        module.decoder.layers.1.mlp.linear_fc2.weight
        module.decoder.layers.19.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.23.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.22.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.21.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.18.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.20.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.17.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.16.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.15.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.13.mlp.linear_fc1.bias
        module.decoder.layers.6.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.17.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.16.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.15.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.14.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.14.self_attention.linear_qkv.weight
        module.decoder.layers.13.self_attention.linear_proj.bias
        module.decoder.layers.10.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.9.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.5.mlp.linear_fc2.weight
        module.decoder.layers.4.self_attention.linear_qkv.weight
        module.decoder.layers.4.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.self_attention.linear_proj.bias
        module.decoder.layers.10.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.11.mlp.linear_fc1.bias
        module.decoder.layers.10.mlp.linear_fc1.weight
        module.decoder.layers.4.mlp.linear_fc2.weight
        module.decoder.layers.23.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.22.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.21.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.20.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.19.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.18.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.13.self_attention.linear_qkv.bias
        module.decoder.layers.12.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.11.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.4.mlp.linear_fc2.bias
        module.decoder.layers.2.mlp.linear_fc2.weight
        module.decoder.layers.11.self_attention.linear_proj.bias
        module.decoder.layers.6.self_attention.linear_proj.bias
        module.decoder.layers.23.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.22.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.21.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.20.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.19.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.18.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.17.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.16.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.15.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.13.mlp.linear_fc1.weight
        module.decoder.layers.7.self_attention.linear_qkv.bias
        module.decoder.layers.17.mlp.linear_fc1.bias
        module.decoder.layers.16.mlp.linear_fc1.bias
        module.decoder.layers.15.mlp.linear_fc1.bias
        module.decoder.layers.14.mlp.linear_fc1.bias
        module.decoder.layers.13.self_attention.linear_proj.weight
        module.decoder.layers.10.mlp.linear_fc1.bias
        module.decoder.layers.9.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.8.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.7.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.6.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.17.self_attention.linear_proj.bias
        module.decoder.layers.16.self_attention.linear_proj.bias
        module.decoder.layers.15.self_attention.linear_proj.bias
        module.decoder.layers.13.mlp.linear_fc2.bias
        module.decoder.layers.10.self_attention.linear_proj.bias
        module.decoder.layers.4.self_attention.linear_proj.weight
        module.decoder.layers.3.mlp.linear_fc2.bias
        module.decoder.layers.6.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.6.mlp.linear_fc1.bias
        module.decoder.layers.14.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.5.mlp.linear_fc2.bias
        module.decoder.layers.4.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.23.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.22.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.21.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.20.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.19.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.18.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.13.self_attention.linear_qkv.weight
        module.decoder.layers.12.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.11.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.11.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.10.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.0.mlp.linear_fc2.bias
        module.decoder.layers.12.self_attention.linear_proj.bias
        module.decoder.layers.5.self_attention.linear_proj.weight
        module.decoder.layers.23.mlp.linear_fc1.bias
        module.decoder.layers.22.mlp.linear_fc1.bias
        module.decoder.layers.21.mlp.linear_fc1.bias
        module.decoder.layers.20.mlp.linear_fc1.bias
        module.decoder.layers.19.mlp.linear_fc1.bias
        module.decoder.layers.18.mlp.linear_fc1.bias
        module.decoder.layers.17.self_attention.linear_qkv.bias
        module.decoder.layers.16.self_attention.linear_qkv.bias
        module.decoder.layers.15.self_attention.linear_qkv.bias
        module.decoder.layers.12.mlp.linear_fc1.bias
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.8.self_attention.linear_qkv.bias
        module.decoder.layers.23.self_attention.linear_proj.bias
        module.decoder.layers.22.self_attention.linear_proj.bias
        module.decoder.layers.21.self_attention.linear_proj.bias
        module.decoder.layers.20.self_attention.linear_proj.bias
        module.decoder.layers.19.self_attention.linear_proj.bias
        module.decoder.layers.18.self_attention.linear_proj.bias
        module.decoder.layers.17.mlp.linear_fc1.weight
        module.decoder.layers.16.mlp.linear_fc1.weight
        module.decoder.layers.15.mlp.linear_fc1.weight
        module.decoder.layers.14.mlp.linear_fc1.weight
        module.decoder.layers.5.mlp.linear_fc1.bias
        module.decoder.layers.17.self_attention.linear_proj.weight
        module.decoder.layers.16.self_attention.linear_proj.weight
        module.decoder.layers.15.self_attention.linear_proj.weight
        module.decoder.layers.14.self_attention.linear_qkv.bias
        module.decoder.layers.10.self_attention.linear_proj.weight
        module.decoder.layers.9.self_attention.linear_proj.bias
        module.decoder.layers.8.self_attention.linear_proj.bias
        module.decoder.layers.7.self_attention.linear_proj.bias
        module.decoder.layers.3.self_attention.linear_qkv.weight
        module.decoder.layers.2.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.0.mlp.linear_fc1.bias
        module.decoder.layers.7.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.3.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.7.mlp.linear_fc1.weight
        module.decoder.layers.17.mlp.linear_fc2.bias
        module.decoder.layers.16.mlp.linear_fc2.bias
        module.decoder.layers.15.mlp.linear_fc2.bias
        module.decoder.layers.14.mlp.linear_fc2.bias
        module.decoder.layers.14.self_attention.linear_proj.bias
        module.decoder.layers.10.mlp.linear_fc2.bias
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.1.self_attention.linear_proj.bias
        module.decoder.layers.23.self_attention.linear_qkv.bias
        module.decoder.layers.22.self_attention.linear_qkv.bias
        module.decoder.layers.21.self_attention.linear_qkv.bias
        module.decoder.layers.20.self_attention.linear_qkv.bias
        module.decoder.layers.19.self_attention.linear_qkv.bias
        module.decoder.layers.18.self_attention.linear_qkv.bias
        module.decoder.layers.13.mlp.linear_fc2.weight
        module.decoder.layers.12.self_attention.linear_qkv.bias
        module.decoder.layers.11.self_attention.linear_qkv.bias
        module.decoder.layers.12.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.6.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.7.self_attention.linear_proj.weight
        module.decoder.layers.23.mlp.linear_fc1.weight
        module.decoder.layers.22.mlp.linear_fc1.weight
        module.decoder.layers.21.mlp.linear_fc1.weight
        module.decoder.layers.20.mlp.linear_fc1.weight
        module.decoder.layers.19.mlp.linear_fc1.weight
        module.decoder.layers.18.mlp.linear_fc1.weight
        module.decoder.layers.17.self_attention.linear_qkv.weight
        module.decoder.layers.16.self_attention.linear_qkv.weight
        module.decoder.layers.15.self_attention.linear_qkv.weight
        module.decoder.layers.12.mlp.linear_fc1.weight
        module.decoder.layers.9.self_attention.linear_qkv.bias
        module.decoder.layers.18.self_attention.linear_proj.weight
        module.decoder.layers.20.self_attention.linear_proj.weight
        module.decoder.layers.23.self_attention.linear_proj.weight
        module.decoder.layers.22.self_attention.linear_proj.weight
        module.decoder.layers.21.self_attention.linear_proj.weight
        module.decoder.layers.19.self_attention.linear_proj.weight
        module.decoder.layers.12.self_attention.linear_proj.weight
        module.decoder.layers.11.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.11.self_attention.linear_proj.weight
        module.decoder.layers.9.mlp.linear_fc1.weight
        module.decoder.layers.23.mlp.linear_fc2.bias
        module.decoder.layers.22.mlp.linear_fc2.bias
        module.decoder.layers.21.mlp.linear_fc2.bias
        module.decoder.layers.20.mlp.linear_fc2.bias
        module.decoder.layers.19.mlp.linear_fc2.bias
        module.decoder.layers.18.mlp.linear_fc2.bias
        module.decoder.layers.12.mlp.linear_fc2.bias
        module.decoder.layers.11.mlp.linear_fc2.bias
        module.decoder.layers.9.self_attention.linear_proj.weight
        module.decoder.layers.8.self_attention.linear_proj.weight
        module.decoder.layers.8.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.8.mlp.linear_fc1.weight
        module.decoder.layers.9.mlp.linear_fc2.bias
        module.decoder.layers.8.mlp.linear_fc2.bias
        module.decoder.layers.7.mlp.linear_fc2.bias
        module.decoder.layers.18.self_attention.linear_qkv.weight
        module.decoder.layers.23.self_attention.linear_qkv.weight
        module.decoder.layers.22.self_attention.linear_qkv.weight
        module.decoder.layers.21.self_attention.linear_qkv.weight
        module.decoder.layers.20.self_attention.linear_qkv.weight
        module.decoder.layers.19.self_attention.linear_qkv.weight
        module.decoder.layers.12.self_attention.linear_qkv.weight
        module.decoder.layers.11.self_attention.linear_qkv.weight
        module.decoder.layers.5.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.7.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.7.mlp.linear_fc1.bias
        module.decoder.layers.4.self_attention.linear_qkv.bias
        module.decoder.layers.17.mlp.linear_fc2.weight
        module.decoder.layers.16.mlp.linear_fc2.weight
        module.decoder.layers.15.mlp.linear_fc2.weight
        module.decoder.layers.14.mlp.linear_fc2.weight
        module.decoder.layers.10.mlp.linear_fc2.weight
        module.decoder.layers.9.self_attention.linear_qkv.weight
        module.decoder.layers.8.self_attention.linear_qkv.weight
        module.decoder.layers.7.self_attention.linear_qkv.weight
        module.decoder.layers.6.self_attention.linear_qkv.bias
        module.decoder.layers.0.self_attention.linear_proj.weight
        module.decoder.layers.0.self_attention.linear_proj.bias
        module.decoder.layers.10.self_attention.linear_qkv.weight
INFO:megatron.core.optimizer:Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=0.00015, min_lr=1e-05, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.01, fp16=True, bf16=False, params_dtype=torch.float16, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=False, overlap_param_gather_with_optimizer_step=False, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=True, timers=<megatron.core.timers.Timers object at 0x2b229c6a7dc0>, config_logger_dir='')
INFO:megatron.core.optimizer_param_scheduler:> learning rate decay style: cosine
WARNING: could not find the metadata file checkpoints/gpt2_345m_dist_mp/latest_checkpointed_iteration.txt
    will not load any checkpoints and will start from random
(min, max) time across ranks (ms):
    load-checkpoint ................................: (0.83, 0.85)
[after model, optimizer, and learning rate scheduler are built] datetime: 2024-09-14 19:44:09
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      3200
    validation: 480
    test:       160
INFO:megatron.core.datasets.blended_megatron_dataset_config:Let split_matrix = [(0, 0.949), (0.949, 0.999), (0.999, 1.0)]
> building train, validation, and test datasets for GPT ...
INFO:megatron.core.datasets.blended_megatron_dataset_builder:Building dataset splits with cls=GPTDataset, sizes=(3200, 480, 160), and config=GPTDatasetConfig(random_seed=1234, sequence_length=1024, blend=(['my-gpt2_text_document'], None), blend_per_split=[None, None, None], renormalize_blend_weights=False, split='949,50,1', split_matrix=[(0, 0.949), (0.949, 0.999), (0.999, 1.0)], num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True, mock=False, tokenizer=<megatron.training.tokenizer.tokenizer._GPT2BPETokenizer object at 0x2b229c6a74f0>, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask=True, drop_last_partial_validation_sequence=True, add_extra_token_to_sequence=True, s3_cache_path=None)
INFO:megatron.core.datasets.indexed_dataset:Load the _IndexReader from my-gpt2_text_document.idx
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence lengths
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence pointers
INFO:megatron.core.datasets.indexed_dataset:    Extract the document indices
INFO:megatron.core.datasets.indexed_dataset:> total number of sequences: 10000
INFO:megatron.core.datasets.indexed_dataset:> total number of documents: 10000
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset train indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from a375193ee7bd0ffbfa0d131aff630f11-GPTDataset-train-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from a375193ee7bd0ffbfa0d131aff630f11-GPTDataset-train-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from a375193ee7bd0ffbfa0d131aff630f11-GPTDataset-train-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 3570
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset valid indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from e9d835bb86bcaca4cd08b4977794f406-GPTDataset-valid-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from e9d835bb86bcaca4cd08b4977794f406-GPTDataset-valid-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from e9d835bb86bcaca4cd08b4977794f406-GPTDataset-valid-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 530
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset test indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from fceb23beb87bf7aa15e27415203c9636-GPTDataset-test-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from fceb23beb87bf7aa15e27415203c9636-GPTDataset-test-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from fceb23beb87bf7aa15e27415203c9636-GPTDataset-test-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 161
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2024-09-14 19:44:09
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (129.73, 136.12)
    train/valid/test-data-iterators-setup ..........: (7.84, 159.33)
training ...
[before the start of training step] datetime: 2024-09-14 19:44:10
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/tensor_parallel/layers.py:609: UserWarning: async_grad_allreduce is deprecated, not in use anymore and will be fully removed with 0.10.0. Please use allreduce_dgrad instead.
  warnings.warn(
 [2024-09-14 19:44:19] iteration       10/     200 | consumed samples:          160 | elapsed time per iteration (ms): 995.3 | learning rate: 0.000000E+00 | global batch size:    16 | loss scale: 8388608.0 | number of skipped iterations:  10 | number of nan iterations:   0 |
 [2024-09-14 19:44:24] iteration       20/     200 | consumed samples:          320 | elapsed time per iteration (ms): 457.6 | learning rate: 2.343750E-07 | global batch size:    16 | lm loss: 1.105231E+01 | loss scale: 262144.0 | grad norm: 23.668 | number of skipped iterations:   5 | number of nan iterations:   0 |
Number of parameters in transformer layers in billions:  0.30
Number of parameters in embedding layers in billions: 0.05
Total number of parameters in billions: 0.35
Number of parameters in most loaded shard in billions: 0.0885
Theoretical memory footprints: weight and optimizer=1519.18 MB
[Rank 2] (after 20 iterations) memory (MB) | allocated: 1761.685546875 | max allocated: 2589.13525390625 | reserved: 2608.0 | max reserved: 2608.0
[Rank 3] (after 20 iterations) memory (MB) | allocated: 1761.685546875 | max allocated: 2589.13525390625 | reserved: 2608.0 | max reserved: 2608.0
[Rank 1] (after 20 iterations) memory (MB) | allocated: 1761.685546875 | max allocated: 2589.13525390625 | reserved: 2608.0 | max reserved: 2608.0
[Rank 0] (after 20 iterations) memory (MB) | allocated: 1761.685546875 | max allocated: 2589.13525390625 | reserved: 2608.0 | max reserved: 2608.0
 [2024-09-14 19:44:29] iteration       30/     200 | consumed samples:          480 | elapsed time per iteration (ms): 464.2 | learning rate: 7.031250E-07 | global batch size:    16 | lm loss: 1.077634E+01 | loss scale: 262144.0 | grad norm: 17.701 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-09-14 19:44:33] iteration       40/     200 | consumed samples:          640 | elapsed time per iteration (ms): 437.8 | learning rate: 1.171875E-06 | global batch size:    16 | lm loss: 1.010798E+01 | loss scale: 262144.0 | grad norm: 8.279 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-09-14 19:44:37] iteration       50/     200 | consumed samples:          800 | elapsed time per iteration (ms): 436.9 | learning rate: 1.640625E-06 | global batch size:    16 | lm loss: 9.606478E+00 | loss scale: 262144.0 | grad norm: 4.420 | number of skipped iterations:   0 | number of nan iterations:   0 |
saving checkpoint at iteration      50 to checkpoints/gpt2_345m_dist_mp in torch_dist format
[rank2]: Traceback (most recent call last):
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/pretrain_gpt.py", line 264, in <module>
[rank2]:     pretrain(
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 355, in pretrain
[rank2]:     iteration, num_floating_point_operations_so_far = train(
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1368, in train
[rank2]:     save_checkpoint_and_time(iteration, model, optimizer,
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1072, in save_checkpoint_and_time
[rank2]:     save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 401, in save_checkpoint
[rank2]:     state_dict = generate_state_dict(
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 613, in generate_state_dict
[rank2]:     state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/optimizer/optimizer.py", line 654, in sharded_state_dict
[rank2]:     optim_state_to_sharding_state(
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 120, in optim_state_to_sharding_state
[rank2]:     sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
[rank2]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 83, in make_sharded_optimizer_tensor
[rank2]:     tuple(optim_param.shape) == model_param.local_shape
[rank2]: AttributeError: 'NoneType' object has no attribute 'shape'
[rank3]: Traceback (most recent call last):
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/pretrain_gpt.py", line 264, in <module>
[rank3]:     pretrain(
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 355, in pretrain
[rank3]:     iteration, num_floating_point_operations_so_far = train(
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1368, in train
[rank3]:     save_checkpoint_and_time(iteration, model, optimizer,
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1072, in save_checkpoint_and_time
[rank3]:     save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 401, in save_checkpoint
[rank3]:     state_dict = generate_state_dict(
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 613, in generate_state_dict
[rank3]:     state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/optimizer/optimizer.py", line 654, in sharded_state_dict
[rank3]:     optim_state_to_sharding_state(
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 120, in optim_state_to_sharding_state
[rank3]:     sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
[rank3]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 83, in make_sharded_optimizer_tensor
[rank3]:     tuple(optim_param.shape) == model_param.local_shape
[rank3]: AttributeError: 'NoneType' object has no attribute 'shape'
[rank0]: Traceback (most recent call last):
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/pretrain_gpt.py", line 264, in <module>
[rank0]:     pretrain(
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 355, in pretrain
[rank0]:     iteration, num_floating_point_operations_so_far = train(
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1368, in train
[rank0]:     save_checkpoint_and_time(iteration, model, optimizer,
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1072, in save_checkpoint_and_time
[rank0]:     save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 401, in save_checkpoint
[rank0]:     state_dict = generate_state_dict(
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 613, in generate_state_dict
[rank0]:     state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/optimizer/optimizer.py", line 654, in sharded_state_dict
[rank0]:     optim_state_to_sharding_state(
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 120, in optim_state_to_sharding_state
[rank0]:     sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
[rank0]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 83, in make_sharded_optimizer_tensor
[rank0]:     tuple(optim_param.shape) == model_param.local_shape
[rank0]: AttributeError: 'NoneType' object has no attribute 'shape'
[rank1]: Traceback (most recent call last):
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/pretrain_gpt.py", line 264, in <module>
[rank1]:     pretrain(
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 355, in pretrain
[rank1]:     iteration, num_floating_point_operations_so_far = train(
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1368, in train
[rank1]:     save_checkpoint_and_time(iteration, model, optimizer,
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/training.py", line 1072, in save_checkpoint_and_time
[rank1]:     save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 401, in save_checkpoint
[rank1]:     state_dict = generate_state_dict(
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/training/checkpointing.py", line 613, in generate_state_dict
[rank1]:     state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/optimizer/optimizer.py", line 654, in sharded_state_dict
[rank1]:     optim_state_to_sharding_state(
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 120, in optim_state_to_sharding_state
[rank1]:     sharded_state[param_id][state_key] = make_sharded_optimizer_tensor(
[rank1]:   File "/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py", line 83, in make_sharded_optimizer_tensor
[rank1]:     tuple(optim_param.shape) == model_param.local_shape
[rank1]: AttributeError: 'NoneType' object has no attribute 'shape'
W0914 19:44:43.761000 47642202304448 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 11158 closing signal SIGTERM
E0914 19:44:44.276000 47642202304448 torch/distributed/elastic/multiprocessing/api.py:826] failed (exitcode: 1) local_rank: 1 (pid: 11159) of binary: /scratch/qualis/miniconda3/envs/large-scale-lm/bin/python
Traceback (most recent call last):
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.3.0', 'console_scripts', 'torchrun')())
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/run.py", line 879, in main
    run(args)
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/run.py", line 870, in run
    elastic_launch(
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 263, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
pretrain_gpt.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2024-09-14_19:44:43
  host      : gpu32.eth
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 11160)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2024-09-14_19:44:43
  host      : gpu32.eth
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 11161)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-09-14_19:44:43
  host      : gpu32.eth
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 11159)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```

모델을 저장할 때에 (위의 경우에 50스탭 후에) 에러가 발생합니다. 예전 Megatron-LM 브랜치(`core_r0.5.0`)를 체크아웃해서 다시 실행해 보도록 하겠습니다.
#### 해당 에러 관련해서 Megatron-LM GitHub 이슈(https://github.com/NVIDIA/Megatron-LM/issues/1134) 를 생성했습니다. 임시 해결 방안으로 체크포인터 포맷을 `torch`로 지정하고 (--ckpt-format torch) 실행하면 에러 없이 잘 실행되는 것이 확인할 수 있습니다. 참고로 디폴트 체크포인터 포맷은 `torch_dist`입니다. 또한 아래와 같이 old Megatron-LM Branch 코드를 git checkout하고 실행할 수도 있습니다.    
 
```
(large-scale-lm) [gpu05]$ pwd
/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM
(large-scale-lm) [gpu05]$ git checkout core_r0.5.0
Branch core_r0.5.0 set up to track remote branch core_r0.5.0 from origin.
Switched to a new branch 'core_r0.5.0'
(large-scale-lm) [gpu05]$ git branch 
* core_r0.5.0
  main
(large-scale-lm) [gpu05]$ ls
./               .gitignore                 my-gpt2_text_document.bin.bak  pretrain_vision_inpaint.py
../              .gitlab-ci.yml             my-gpt2_text_document.idx      pyproject.toml
CODEOWNERS       images/                    my-gpt2_text_document.idx.bak  README.md
CONTRIBUTING.md  jet-tests.yml              pretrain_bert.py               report_theoretical_memory.py
.coveragerc      LICENSE                    pretrain_gpt.py                setup.py
Dockerfile.ci    MANIFEST.in                pretrain_ict.py                tasks/
docs/            megatron/                  pretrain_retro.py              tests/
examples/        megatron_datasets.jsonl    pretrain_t5.py                 tools/
.git/            merges.txt                 pretrain_vision_classify.py    vocab.json
.github/         my-gpt2_text_document.bin  pretrain_vision_dino.py
(large-scale-lm) [gpu05]$ CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4  pretrain_gpt.py     --tensor-model-parallel-size 4     --pipeline-model-parallel-size 1         --num-layers 24     --hidden-size 1024     --num-attention-heads 16     --seq-length 1024     --max-position-embeddings 1024     --micro-batch-size 4     --global-batch-size 16     --lr 0.00015     --train-iters 200     --lr-decay-iters 320000     --lr-decay-style cosine     --min-lr 1.0e-5     --weight-decay 1e-2     --lr-warmup-fraction .01     --clip-grad 1.0     --fp16 --data-path  my-gpt2_text_document     --vocab-file vocab.json     --merge-file merges.txt     --split 949,50,1 --log-interval 10     --save-interval 50    --eval-interval 100     --eval-iters 10 --distributed-backend nccl  --save checkpoints/gpt2_345m_dist_mp     --load  checkpoints/gpt2_345m_dist_mp --attention-softmax-in-fp32 --sequence-parallel
W0914 19:37:31.953000 47780312846272 torch/distributed/run.py:757]
W0914 19:37:31.953000 47780312846272 torch/distributed/run.py:757] *****************************************
W0914 19:37:31.953000 47780312846272 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0914 19:37:31.953000 47780312846272 torch/distributed/run.py:757] *****************************************
Zarr-based strategies will not be registered because of missing packages
using world size: 4, data-parallel size: 1, context-parallel size: 1 tensor-model-parallel size: 4, pipeline-model-parallel size: 1
WARNING: Setting args.overlap_p2p_comm to False since non-interleaved schedule does not support overlapping p2p communication
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  add_qkv_bias .................................... False
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... False
  apply_residual_connection_post_layernorm ........ False
  apply_rope_fusion ............................... True
  async_tensor_model_parallel_allreduce ........... False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... True
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  bias_swiglu_fusion .............................. True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  check_for_nan_in_loss_and_grad .................. True
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  clone_scatter_output_in_embedding ............... True
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  context_parallel_size ........................... 1
  data_cache_path ................................. None
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 1
  data_path ....................................... ['my-gpt2_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  delay_grad_reduce ............................... True
  delay_param_gather .............................. False
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  enable_one_logger ............................... False
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 1024
  end_weight_decay ................................ 0.01
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  expert_model_parallel_size ...................... 1
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8 ............................................. None
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 16
  gradient_accumulation_fusion .................... True
  group_query_attention ........................... False
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  lazy_mpu_init ................................... None
  load ............................................ checkpoints/gpt2_345m_dist_mp
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_progress .................................... False
  log_throughput .................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 320000
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_init .................................. 0.0
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  manual_gc ....................................... False
  manual_gc_eval .................................. True
  manual_gc_interval .............................. 0
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... merges.txt
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mock_data ....................................... False
  moe_aux_loss_coeff .............................. 0.0
  moe_grouped_gemm ................................ False
  moe_input_jitter_eps ............................ None
  moe_router_load_balancing_type .................. aux_loss
  moe_router_topk ................................. 2
  moe_token_dropping .............................. False
  moe_z_loss_coeff ................................ None
  nccl_communicator_config_path ................... None
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  norm_epsilon .................................... 1e-05
  normalization ................................... LayerNorm
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_query_groups ................................ 1
  num_workers ..................................... 2
  one_logger_entity ............................... hwinf_dcm
  one_logger_project .............................. e2e-tracking
  one_logger_run_name ............................. None
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_grad_reduce ............................. False
  overlap_p2p_comm ................................ False
  overlap_param_gather ............................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  position_embedding_type ......................... learned_absolute
  profile ......................................... False
  profile_ranks ................................... [0]
  profile_step_end ................................ 12
  profile_step_start .............................. 10
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_num_layers ............................ None
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_attention_gate ............................ 1
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_return_doc_ids ............................ False
  retro_verify_neighbor_count ..................... True
  retro_workdir ................................... None
  rotary_interleaved .............................. False
  rotary_percent .................................. 1.0
  rotary_seq_len_interpolation_factor ............. None
  sample_rate ..................................... 1.0
  save ............................................ checkpoints/gpt2_345m_dist_mp
  save_interval ................................... 50
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sequence_parallel ............................... True
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  skip_train ...................................... False
  spec ............................................ None
  split ........................................... 949,50,1
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.01
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  tensor_model_parallel_size ...................... 4
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  tp_comm_bulk_dgrad .............................. True
  tp_comm_bulk_wgrad .............................. True
  tp_comm_overlap ................................. False
  tp_comm_overlap_cfg ............................. None
  tp_comm_split_ag ................................ True
  tp_comm_split_rs ................................ True
  train_data_path ................................. None
  train_iters ..................................... 200
  train_samples ................................... None
  transformer_impl ................................ local
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_cpu_initialization .......................... None
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_mcore_models ................................ False
  use_one_sent_docs ............................... False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. False
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... vocab.json
  vocab_size ...................................... None
  wandb_exp_name ..................................
  wandb_project ...................................
  wandb_save_dir ..................................
  weight_decay .................................... 0.01
  weight_decay_incr_style ......................... constant
  world_size ...................................... 4
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 4
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 431 dummy tokens (new size: 50688)
> initializing torch distributed ...
> initialized tensor model parallel with size 4
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory `/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/datasets'
make: Nothing to be done for `default'.
make: Leaving directory `/scratch/qualis/large-scale-lm-tutorials/src/Megatron-LM/megatron/core/datasets'
>>> done with dataset index builder. Compilation time: 0.048 seconds
> compiling and loading fused kernels ...
>>> done with compiling and loading fused kernels. Compilation time: 1.908 seconds
[rank2]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
[rank1]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
[rank0]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
[rank3]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
time to initialize megatron (seconds): 4.743
[after megatron is initialized] datetime: 2024-09-14 19:37:42
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (3, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (2, 0): 89714688
INFO:megatron.core.distributed.grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 1
INFO:megatron.core.distributed.grad_buffer:Params for bucket 1 (89714688 elements):
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.embedding.word_embeddings.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.embedding.position_embeddings.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.final_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.4.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.final_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.20.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.input_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.17.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.15.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.9.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.1.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.3.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.2.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.11.self_attention.query_key_value.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.10.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.5.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.6.self_attention.query_key_value.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.22.mlp.dense_h_to_4h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.post_attention_norm.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.8.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.23.self_attention.dense.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.16.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.14.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.7.input_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.12.post_attention_norm.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.0.mlp.dense_4h_to_h.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.21.self_attention.dense.weight
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.19.mlp.dense_4h_to_h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.18.mlp.dense_h_to_4h.bias
INFO:megatron.core.distributed.grad_buffer:    module.language_model.encoder.layers.13.input_norm.bias
> learning rate decay style: cosine
WARNING: could not find the metadata file checkpoints/gpt2_345m_dist_mp/latest_checkpointed_iteration.txt
    will not load any checkpoints and will start from random
(min, max) time across ranks (ms):
    load-checkpoint ................................: (0.42, 0.44)
[after model, optimizer, and learning rate scheduler are built] datetime: 2024-09-14 19:37:43
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      3200
    validation: 480
    test:       160
INFO:megatron.core.datasets.blended_megatron_dataset_config:mock = False
INFO:megatron.core.datasets.blended_megatron_dataset_config:Let split_matrix = [(0, 0.949), (0.949, 0.999), (0.999, 1.0)]
> building train, validation, and test datasets for GPT ...
INFO:megatron.core.datasets.indexed_dataset:Load the _IndexReader from my-gpt2_text_document.idx
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence lengths
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence pointers
INFO:megatron.core.datasets.indexed_dataset:    Extract the document indices
INFO:megatron.core.datasets.indexed_dataset:> total number of sequences: 10000
INFO:megatron.core.datasets.indexed_dataset:> total number of documents: 10000
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset train indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from a375193ee7bd0ffbfa0d131aff630f11-GPTDataset-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from a375193ee7bd0ffbfa0d131aff630f11-GPTDataset-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from a375193ee7bd0ffbfa0d131aff630f11-GPTDataset-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 3570
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 2
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset valid indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from e9d835bb86bcaca4cd08b4977794f406-GPTDataset-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from e9d835bb86bcaca4cd08b4977794f406-GPTDataset-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from e9d835bb86bcaca4cd08b4977794f406-GPTDataset-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 530
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 6
INFO:megatron.core.datasets.gpt_dataset:Load the GPTDataset test indices
INFO:megatron.core.datasets.gpt_dataset:        Load the document index from fceb23beb87bf7aa15e27415203c9636-GPTDataset-document_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the sample index from fceb23beb87bf7aa15e27415203c9636-GPTDataset-sample_index.npy
INFO:megatron.core.datasets.gpt_dataset:        Load the shuffle index from fceb23beb87bf7aa15e27415203c9636-GPTDataset-shuffle_index.npy
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 161
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 106
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2024-09-14 19:37:43
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (108.83, 126.89)
    train/valid/test-data-iterators-setup ..........: (6.98, 158.54)
training ...
[before the start of training step] datetime: 2024-09-14 19:37:43
 iteration       10/     200 | consumed samples:          160 | elapsed time per iteration (ms): 1011.8 | learning rate: 0.000E+00 | global batch size:    16 | loss scale: 8388608.0 | number of skipped iterations:  10 | number of nan iterations:   0 |
 iteration       20/     200 | consumed samples:          320 | elapsed time per iteration (ms): 388.3 | learning rate: 2.344E-07 | global batch size:    16 | lm loss: 1.096171E+01 | loss scale: 262144.0 | grad norm: 25.472 | number of skipped iterations:   5 | number of nan iterations:   0 |
[Rank 2] (after 20 iterations) memory (MB) | allocated: 1786.68505859375 | max allocated: 4673.77880859375 | reserved: 4746.0 | max reserved: 4746.0[Rank 1] (after 20 iterations) memory (MB) | allocated: 1786.68505859375 | max allocated: 4673.77880859375 | reserved: 4746.0 | max reserved: 4746.0

[Rank 3] (after 20 iterations) memory (MB) | allocated: 1786.68505859375 | max allocated: 4673.77880859375 | reserved: 4746.0 | max reserved: 4746.0[Rank 0] (after 20 iterations) memory (MB) | allocated: 1786.68505859375 | max allocated: 4673.77880859375 | reserved: 4746.0 | max reserved: 4746.0

 iteration       30/     200 | consumed samples:          480 | elapsed time per iteration (ms): 385.1 | learning rate: 7.031E-07 | global batch size:    16 | lm loss: 1.067098E+01 | loss scale: 262144.0 | grad norm: 17.569 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration       40/     200 | consumed samples:          640 | elapsed time per iteration (ms): 392.6 | learning rate: 1.172E-06 | global batch size:    16 | lm loss: 1.001153E+01 | loss scale: 262144.0 | grad norm: 7.634 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration       50/     200 | consumed samples:          800 | elapsed time per iteration (ms): 384.9 | learning rate: 1.641E-06 | global batch size:    16 | lm loss: 9.559246E+00 | loss scale: 262144.0 | grad norm: 4.010 | number of skipped iterations:   0 | number of nan iterations:   0 |
saving checkpoint at iteration      50 to checkpoints/gpt2_345m_dist_mp
  successfully saved checkpoint at iteration      50 to checkpoints/gpt2_345m_dist_mp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (1796.45, 1796.45)
 iteration       60/     200 | consumed samples:          960 | elapsed time per iteration (ms): 385.6 | learning rate: 2.109E-06 | global batch size:    16 | lm loss: 9.356075E+00 | loss scale: 262144.0 | grad norm: 2.978 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration       70/     200 | consumed samples:         1120 | elapsed time per iteration (ms): 384.4 | learning rate: 2.578E-06 | global batch size:    16 | lm loss: 9.246494E+00 | loss scale: 262144.0 | grad norm: 2.805 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration       80/     200 | consumed samples:         1280 | elapsed time per iteration (ms): 385.1 | learning rate: 3.047E-06 | global batch size:    16 | lm loss: 9.104178E+00 | loss scale: 262144.0 | grad norm: 3.359 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration       90/     200 | consumed samples:         1440 | elapsed time per iteration (ms): 385.6 | learning rate: 3.516E-06 | global batch size:    16 | lm loss: 8.897788E+00 | loss scale: 262144.0 | grad norm: 3.078 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      100/     200 | consumed samples:         1600 | elapsed time per iteration (ms): 395.7 | learning rate: 3.984E-06 | global batch size:    16 | lm loss: 8.766370E+00 | loss scale: 262144.0 | grad norm: 2.635 | number of skipped iterations:   0 | number of nan iterations:   0 |
(min, max) time across ranks (ms):
    evaluate .......................................: (1970.23, 1970.28)
-----------------------------------------------------------------------------------------------
 validation loss at iteration 100 | lm loss value: 8.687662E+00 | lm loss PPL: 5.929304E+03 |
-----------------------------------------------------------------------------------------------
saving checkpoint at iteration     100 to checkpoints/gpt2_345m_dist_mp
  successfully saved checkpoint at iteration     100 to checkpoints/gpt2_345m_dist_mp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (1616.24, 1616.25)
 iteration      110/     200 | consumed samples:         1760 | elapsed time per iteration (ms): 385.2 | learning rate: 4.453E-06 | global batch size:    16 | lm loss: 8.635515E+00 | loss scale: 262144.0 | grad norm: 2.624 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      120/     200 | consumed samples:         1920 | elapsed time per iteration (ms): 384.3 | learning rate: 4.922E-06 | global batch size:    16 | lm loss: 8.545827E+00 | loss scale: 262144.0 | grad norm: 2.270 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      130/     200 | consumed samples:         2080 | elapsed time per iteration (ms): 383.3 | learning rate: 5.391E-06 | global batch size:    16 | lm loss: 8.472692E+00 | loss scale: 262144.0 | grad norm: 2.226 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      140/     200 | consumed samples:         2240 | elapsed time per iteration (ms): 384.5 | learning rate: 5.859E-06 | global batch size:    16 | lm loss: 8.391251E+00 | loss scale: 262144.0 | grad norm: 2.068 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      150/     200 | consumed samples:         2400 | elapsed time per iteration (ms): 382.2 | learning rate: 6.328E-06 | global batch size:    16 | lm loss: 8.315047E+00 | loss scale: 262144.0 | grad norm: 2.387 | number of skipped iterations:   0 | number of nan iterations:   0 |
saving checkpoint at iteration     150 to checkpoints/gpt2_345m_dist_mp
  successfully saved checkpoint at iteration     150 to checkpoints/gpt2_345m_dist_mp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (1490.23, 1490.23)
 iteration      160/     200 | consumed samples:         2560 | elapsed time per iteration (ms): 382.8 | learning rate: 6.797E-06 | global batch size:    16 | lm loss: 8.226217E+00 | loss scale: 262144.0 | grad norm: 2.053 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      170/     200 | consumed samples:         2720 | elapsed time per iteration (ms): 383.6 | learning rate: 7.266E-06 | global batch size:    16 | lm loss: 8.159902E+00 | loss scale: 262144.0 | grad norm: 2.645 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      180/     200 | consumed samples:         2880 | elapsed time per iteration (ms): 389.7 | learning rate: 7.734E-06 | global batch size:    16 | lm loss: 8.071524E+00 | loss scale: 262144.0 | grad norm: 2.367 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      190/     200 | consumed samples:         3040 | elapsed time per iteration (ms): 386.0 | learning rate: 8.203E-06 | global batch size:    16 | lm loss: 8.010191E+00 | loss scale: 262144.0 | grad norm: 2.396 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration      200/     200 | consumed samples:         3200 | elapsed time per iteration (ms): 384.4 | learning rate: 8.672E-06 | global batch size:    16 | lm loss: 7.913759E+00 | loss scale: 262144.0 | grad norm: 1.925 | number of skipped iterations:   0 | number of nan iterations:   0 |
(min, max) time across ranks (ms):
    evaluate .......................................: (1455.35, 1455.46)
-----------------------------------------------------------------------------------------------
 validation loss at iteration 200 | lm loss value: 7.900960E+00 | lm loss PPL: 2.699873E+03 |
-----------------------------------------------------------------------------------------------
saving checkpoint at iteration     200 to checkpoints/gpt2_345m_dist_mp
  successfully saved checkpoint at iteration     200 to checkpoints/gpt2_345m_dist_mp
(min, max) time across ranks (ms):
    save-checkpoint ................................: (1592.99, 1593.01)
[after training is done] datetime: 2024-09-14 19:39:16
saving checkpoint at iteration     200 to checkpoints/gpt2_345m_dist_mp
  successfully saved checkpoint at iteration     200 to checkpoints/gpt2_345m_dist_mp
Evaluating on 160 samples
Evaluating iter 1/10
Evaluating iter 2/10
Evaluating iter 3/10
Evaluating iter 4/10
Evaluating iter 5/10
Evaluating iter 6/10
Evaluating iter 7/10
Evaluating iter 8/10
Evaluating iter 9/10
Evaluating iter 10/10
(min, max) time across ranks (ms):
    evaluate .......................................: (1460.59, 1460.86)
-----------------------------------------------------------------------------------------------------------------
 validation loss at iteration 200 on validation set | lm loss value: 7.900509E+00 | lm loss PPL: 2.698655E+03 |
-----------------------------------------------------------------------------------------------------------------
Evaluating on 160 samples
Evaluating iter 1/10
Evaluating iter 2/10
Evaluating iter 3/10
Evaluating iter 4/10
Evaluating iter 5/10
Evaluating iter 6/10
Evaluating iter 7/10
Evaluating iter 8/10
Evaluating iter 9/10
Evaluating iter 10/10
(min, max) time across ranks (ms):
    evaluate .......................................: (1454.27, 1454.35)
-----------------------------------------------------------------------------------------------------------
 validation loss at iteration 200 on test set | lm loss value: 7.693538E+00 | lm loss PPL: 2.194124E+03 |
-----------------------------------------------------------------------------------------------------------
```
#### Can we do some experiments with 3D Parallelism, including Data Parallel (DP), using the `pretrain_gpt.py` script?

- Yes, it appears that you can indeed experiment with 3D Parallelism (Tensor, Pipeline, Data Parallel), using the pretrain_gpt.py script.
- For example, given the parameters in the command line like `--nproc_per_node 4  pretrain_gpt.py  --tensor-model-parallel-size 2  --pipeline-model-parallel-size 1`, it is observed that the DP degree is indeed automatically set to 2 based on the following rule:
  - `Total processes = Tensor Parallel size × Pipeline Parallel size × Data Parallel size`
```
(large-scale-lm) [gpu05]$ CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 4  pretrain_gpt.py     --tensor-model-parallel-size 2     --pipeline-model-parallel-size 1         --num-layers 24     --hidden-size 1024     --num-attention-heads 16     --seq-length 1024     --max-position-embeddings 1024     --micro-batch-size 4     --global-batch-size 16     --lr 0.00015     --train-iters 200     --lr-decay-iters 320000     --lr-decay-style cosine     --min-lr 1.0e-5     --weight-decay 1e-2     --lr-warmup-fraction .01     --clip-grad 1.0     --fp16 --data-path  my-gpt2_text_document     --vocab-file vocab.json     --merge-file merges.txt     --split 949,50,1 --log-interval 10     --save-interval 50    --eval-interval 100     --eval-iters 10 --distributed-backend nccl  --save checkpoints/gpt2_345m_dist_mp     --load  checkpoints/gpt2_345m_dist_mp --attention-softmax-in-fp32 --sequence-parallel --ckpt-format torch
.
.
.
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  .
  .
  .
  data_parallel_size .............................. 2
  .
  .
  pipeline_model_parallel_size .................... 1
  .
  .
  tensor_model_parallel_size ...................... 2
  .
  .
```

#### Note that if you are allocated 3 nodes with 2 GPUs each, instead of 1 node with 4 GPUs, then the pretrain_gpt.py command would look as follows:
```
(large-scale-lm) [gpu05]$ CUDA_DEVICE_MAX_CONNECTIONS=1 srun torchrun --nnodes=3 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu30:12345  pretrain_gpt.py     --tensor-model-parallel-size 2     --pipeline-model-parallel-size 3         --num-layers 24     --hidden-size 1024     --num-attention-heads 16     --seq-length 1024     --max-position-embeddings 1024     --micro-batch-size 4     --global-batch-size 16     --lr 0.00015     --train-iters 200     --lr-decay-iters 320000     --lr-decay-style cosine     --min-lr 1.0e-5     --weight-decay 1e-2     --lr-warmup-fraction .01     --clip-grad 1.0     --fp16 --data-path  my-gpt2_text_document     --vocab-file vocab.json     --merge-file merges.txt     --split 949,50,1 --log-interval 10     --save-interval 50    --eval-interval 100     --eval-iters 10 --distributed-backend nccl  --save checkpoints/gpt2_345m_dist_mp     --load  checkpoints/gpt2_345m_dist_mp --attention-softmax-in-fp32 --sequence-parallel --ckpt-format torch
```

#### Note that if you want to use the ZERO1(?) distributed optimizer, you can add --use-distributed-optimizer to the command line as follows: 

```
CUDA_DEVICE_MAX_CONNECTIONS=1 srun torchrun --nnodes=3 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint gpu30:12345  pretrain_gpt.py     --tensor-model-parallel-size 2     --pipeline-model-parallel-size 3         --num-layers 24     --hidden-size 1024     --num-attention-heads 16     --seq-length 1024     --max-position-embeddings 1024     --micro-batch-size 4     --global-batch-size 16     --lr 0.00015     --train-iters 200     --lr-decay-iters 320000     --lr-decay-style cosine     --min-lr 1.0e-5     --weight-decay 1e-2     --lr-warmup-fraction .01     --clip-grad 1.0     --fp16 --data-path  my-gpt2_text_document     --vocab-file vocab.json     --merge-file merges.txt     --split 949,50,1 --log-interval 10     --save-interval 50    --eval-interval 100     --eval-iters 10 --distributed-backend nccl  --save checkpoints/gpt2_345m_dist_mp     --load  checkpoints/gpt2_345m_dist_mp --attention-softmax-in-fp32 --sequence-parallel --ckpt-format torch  --use-distributed-optimizer
```

#### Please refer to some GitHub issues related to pipeline parallelism combined with ZeRO 1/2/3.
- https://github.com/NVIDIA/Megatron-LM/issues/589
- https://github.com/microsoft/DeepSpeed/issues/1110




```
(large-scale-lm) [gpu05]$ cd ..
/scratch/qualis/git-projects/large-scale-lm-tutorials/src
```


## 3. Parallelformers
![](../images/parallelformers.png)
지금까지 Megatron-LM으로 모델을 학습해봤습니다. Megatron-LM은 훌륭한 Tensor Parallelism 기능을 보유하고 있지만, 기존에 우리가 자주 쓰던 Hugging Face `transformers`로 학습된 모델을 병렬화 할 수는 없었습니다. 이러한 문제를 해결하기 위해 TUNiB은 2021년 `parallelformers`라는 오픈소스를 공개했습니다. `parallelformers`는 코드 한 두줄로 Hugging Face `transformers`로 학습된 거의 대부분의 모델에 Tensor Parallelism을 적용하여 인퍼런스 할 수 있는 도구 입니다.
`parallelformers`를 설치해봅시다.
```
(large-scale-lm) [gpu05]$ pip install parallelformers
Collecting parallelformers
  Downloading parallelformers-1.0.1-py3-none-any.whl (110 kB)
     |████████████████████████████████| 110 kB 24.0 MB/s eta 0:00:01
Requirement already satisfied: transformers>=4.2 in /opt/conda/lib/python3.7/site-packages (from parallelformers) (4.11.3)
Requirement already satisfied: torch in /opt/conda/lib/python3.7/site-packages (from parallelformers) (1.9.0)
Collecting dacite
  Downloading dacite-1.6.0-py3-none-any.whl (12 kB)
Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (0.10.3)
Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (2021.10.23)
Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (1.20.2)
Requirement already satisfied: requests in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (2.24.0)
Requirement already satisfied: tqdm>=4.27 in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (4.62.3)
Requirement already satisfied: filelock in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (3.0.12)
Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (21.0)
Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (5.4.1)
Requirement already satisfied: sacremoses in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (0.0.46)
Requirement already satisfied: huggingface-hub>=0.0.17 in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (0.0.19)
Requirement already satisfied: importlib-metadata in /opt/conda/lib/python3.7/site-packages (from transformers>=4.2->parallelformers) (4.8.1)
Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.7/site-packages (from huggingface-hub>=0.0.17->transformers>=4.2->parallelformers) (3.7.4.3)
Requirement already satisfied: pyparsing>=2.0.2 in /opt/conda/lib/python3.7/site-packages (from packaging>=20.0->transformers>=4.2->parallelformers) (3.0.0)
Requirement already satisfied: zipp>=0.5 in /opt/conda/lib/python3.7/site-packages (from importlib-metadata->transformers>=4.2->parallelformers) (3.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests->transformers>=4.2->parallelformers) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests->transformers>=4.2->parallelformers) (2021.5.30)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests->transformers>=4.2->parallelformers) (1.25.11)
Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests->transformers>=4.2->parallelformers) (2.10)
Requirement already satisfied: joblib in /opt/conda/lib/python3.7/site-packages (from sacremoses->transformers>=4.2->parallelformers) (1.1.0)
Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from sacremoses->transformers>=4.2->parallelformers) (1.16.0)
Requirement already satisfied: click in /opt/conda/lib/python3.7/site-packages (from sacremoses->transformers>=4.2->parallelformers) (8.0.3)
Installing collected packages: dacite, parallelformers
Successfully installed dacite-1.6.0 parallelformers-1.0.1
WARNING: Running pip as root will break packages and permissions. You should install packages reliably by using venv: https://pip.pypa.io/warnings/venv
```
parallelformers는 아래 코드와 같이 parallelize 함수를 이용하여 기존 모델을 병렬화 할 수 있으며, num_gpus와 fp16 등의 몇가지 옵션을 추가로 제공합니다.
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from parallelformers import parallelize

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    parallelize(model, num_gpus=4, fp16=True, verbose="simple")

    inputs = tokenizer(
        "Parallelformers is",
        return_tensors="pt",
    ).to("cuda")

    model = model.to("cuda")

    outputs = model.generate(
        **inputs,
        num_beams=5,
        no_repeat_ngram_size=4,
        max_length=15,
    )

    print(f"\nOutput: {tokenizer.batch_decode(outputs)[0]}")
```

주의: `parallelformers`는 프로세스간 데이터 통신을 위해 공유메모리를 사용합니다. 따라서 **docker와 같이 제한된 리소스만 허용되는 환경에서 사용할 때는 반드시 shared memory 사이즈를 키워줘야 합니다.**

`docker run ... --shm_size=?gb` 옵션을 통해 공유메모리 사이즈를 키우거나 `docker run ... --ipc=host` 옵션을 통해 공유메모리 제한을 해제할 수 있습니다. docker에서 발생하는 거의 모든 문제는 공유메모리의 제한 때문에 일어나는 것으로 확인 되었으며 더 큰 모델을 사용하려면 더 큰 사이즈의 shared memory 할당이 요구됩니다.

```
(large-scale-lm) [gpu05]$ python ch6/parallelformers_inference.py
/scratch/qualis/miniconda3/envs/large-scale-lm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
GPU 1 alloc: 1527883776
GPU 1 cached: 1904214016

GPU 2 alloc: 1527883776
GPU 2 cached: 1904214016

GPU 0 alloc: 1527883776
GPU 0 cached: 1904214016

GPU 3 alloc: 1527883776
GPU 3 cached: 1904214016

Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

Output: Parallelformers is an open-source library for parallel programming in Haskell
```


### Parallelformers의 동작 원리
 
![](../images/tensor_replace.png)

그렇다면 `parallelformers`는 어떻게 모델링 코드의 변화 없이 Tensor parallelism을 수행 할 수 있을까요? 정답은 `Tensor Replacement` 메커니즘에 있습니다. `parallelformers`는 기존 모델의 파라미터를 전부 추출한 뒤, Megatron-LM과 동일한 방식으로 텐서를 쪼개고 쪼개진 텐서로 원래 모델에 존재하던 파라미터를 교체함으로써 모델의 구조 변화 없이 병렬화를 수행할 수 있었습니다. 이를 통해 약 70여가지의 모델을 병렬화 할 수 있었습니다. 이외에도 몇가지 메커니즘이 도입되었지만 텐서 병렬화와 관계 있는 내용은 아니기 때문에 생략하도록 하겠습니다. 만약 더 자세한 내용이 궁금하시다면 다음 주소를 참고해주세요.
- 한국어: https://tunib.notion.site/TECH-2021-07-26-Parallelformers-_-0dcceeaddc5247429745ba36c6549fe5
- English: https://tunib.notion.site/TECH-2021-07-26-Parallelformers-Journey-to-deploying-big-models_TUNiB-32b19a599c38497abaad2a98727f6dc8
