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
src/non_parallelism.py
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
[glogin01]$ python ../src/non_parallelism.py
tensor([[ 74,  98],
        [258, 346]])
```

column parallelism은 모델의 파라미터(A)를 수직방향으로 자른 뒤 연산후 연산 결과를 concat하는 방식입니다. 그림에서와 같이 X는 복제하고 텐서 A를 수직방향으로 분할한 뒤 연산 후 concat 해보겠습니다.
```
"""
src/column_parallelism.py
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
[glogin01]$ python ../src/column_parallelism.py
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
src/row_parallelism.py
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
[glogin01]$ python ../src/row_parallelism.py
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
![](../images/row_parallelism.png)\n",
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
src/megatron_mlp_gelu.py
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

![](../images/megatron_attention.jpeg)\n",
   
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
```
# Megatron-LM을 clone 합니다.
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
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM
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
[glogin01]$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
[glogin01]$ cd ..
```

이제 데이터셋을 만들어보도록 하겠습니다. Megatron-LM으로 모델을 Pre-training을 할 때는 `{\"text\": \"샘플\"}`과 같은 json 구조가 여러라인으로 구성된 간단한 구조의 jsonl 파일을 만들면 되고, Fine-tuning의 경우는 해당 태스크에 맞게 데이터셋을 구성해야 합니다. 본 튜토리얼에서는 Pre-training만 다루고 있기 때문에 Fine-tuning이 필요하시면 Megatron-LM 깃헙 레포를 참고해주세요.
```
"""
src/megatron_datasets.py
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
[glogin01]$ python ../../src/megatron_datasets.py
Reusing dataset wikitext (/root/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/1.0.0/aa5e094000ec7afeb74c3be92c88313cd6f132d564c7effd961c10fd47c76f20)
100%|████████████████████████████████████████████| 3/3 [00:00<00:00, 369.35it/s]
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
!python tools/preprocess_data.py \
       --input megatron_datasets.jsonl \
       --output-prefix my-gpt2 \
       --vocab vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file merges.txt \
       --append-eod
Opening megatron_datasets.jsonl
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
> building GPT2BPETokenizer tokenizer ...
Vocab size: 50257
Output prefix: my-gpt2
Time to startup: 0.10664248466491699
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
Processed 100 documents (341.3833406586251 docs/s, 0.3231593169572366 MB/s).
Processed 200 documents (444.607170228531 docs/s, 0.40740195023601483 MB/s).
Processed 300 documents (507.4913064836572 docs/s, 0.4595645619121138 MB/s).
Processed 400 documents (548.2460078910855 docs/s, 0.500913350338969 MB/s).
Processed 500 documents (599.4126924512631 docs/s, 0.5385256945623461 MB/s).
Processed 600 documents (632.3274718382903 docs/s, 0.567304677135345 MB/s).
Processed 700 documents (656.5582187492173 docs/s, 0.5942094322137902 MB/s).
Processed 800 documents (679.473341028289 docs/s, 0.6114298442783954 MB/s).
Processed 900 documents (694.3271845460217 docs/s, 0.62567246230091 MB/s).
Processed 1000 documents (711.744028291217 docs/s, 0.6423042951843672 MB/s).
Processed 1100 documents (729.9288697633211 docs/s, 0.6542551575749905 MB/s).
Processed 1200 documents (747.7031942801465 docs/s, 0.6652560847870334 MB/s).
Processed 1300 documents (764.753374323305 docs/s, 0.674766482549341 MB/s).
Processed 1400 documents (779.1577277991438 docs/s, 0.6864365578362863 MB/s).
Processed 1500 documents (790.4512469780717 docs/s, 0.6969633845696908 MB/s).
Processed 1600 documents (801.193106685724 docs/s, 0.7046774423849909 MB/s).
Processed 1700 documents (806.756631686137 docs/s, 0.7161883857098408 MB/s).
Processed 1800 documents (816.3964718380284 docs/s, 0.7252975026731121 MB/s).
Processed 1900 documents (831.3950475039509 docs/s, 0.7346526580053557 MB/s).
Processed 2000 documents (847.4097235137235 docs/s, 0.7413089470505299 MB/s).
Processed 2100 documents (860.2405635449649 docs/s, 0.7517782182921232 MB/s).
Processed 2200 documents (865.0414363514374 docs/s, 0.7580306631156802 MB/s).
Processed 2300 documents (873.5639245027145 docs/s, 0.7657585442996709 MB/s).
Processed 2400 documents (882.2530951029984 docs/s, 0.7761644859970351 MB/s).
Processed 2500 documents (891.107956554136 docs/s, 0.7840518788650122 MB/s).
Processed 2600 documents (897.586649345728 docs/s, 0.7875825232354006 MB/s).
Processed 2700 documents (905.6928118212825 docs/s, 0.7926364268970261 MB/s).
Processed 2800 documents (913.7765768432254 docs/s, 0.8018945842245223 MB/s).
Processed 2900 documents (921.524677708312 docs/s, 0.8078279296759729 MB/s).
Processed 3000 documents (927.0432557886642 docs/s, 0.8133901847218254 MB/s).
Processed 3100 documents (933.5282763595969 docs/s, 0.8191782812561477 MB/s).
Processed 3200 documents (939.0246851114703 docs/s, 0.8237833190795807 MB/s).
Processed 3300 documents (944.3738598478612 docs/s, 0.8299497689998975 MB/s).
Processed 3400 documents (949.8985798514814 docs/s, 0.8350220705057206 MB/s).
Processed 3500 documents (954.9541380070546 docs/s, 0.8388551190620309 MB/s).
Processed 3600 documents (961.0590894824982 docs/s, 0.8433869951112278 MB/s).
Processed 3700 documents (966.5140585535103 docs/s, 0.8475759648916847 MB/s).
Processed 3800 documents (970.2837410964693 docs/s, 0.8515277975713496 MB/s).
Processed 3900 documents (972.3412659421506 docs/s, 0.8543419280083547 MB/s).
Processed 4000 documents (977.5591959913655 docs/s, 0.8567109649824823 MB/s).
Processed 4100 documents (982.1164774000912 docs/s, 0.8622407256560518 MB/s).
Processed 4200 documents (988.3455181621163 docs/s, 0.8663715108177805 MB/s).
Processed 4300 documents (994.3620512933041 docs/s, 0.8692481274017909 MB/s).
Processed 4400 documents (1000.1573051427661 docs/s, 0.873466585121912 MB/s).
Processed 4500 documents (1005.4718143362228 docs/s, 0.8773737720380832 MB/s).
Processed 4600 documents (1011.1278894667505 docs/s, 0.8824701168076114 MB/s).
Processed 4700 documents (1015.2351380478192 docs/s, 0.8859525079616727 MB/s).
Processed 4800 documents (1019.9879167915307 docs/s, 0.8900802431435392 MB/s).
Processed 4900 documents (1025.282676115802 docs/s, 0.8932444033863115 MB/s).
Processed 5000 documents (1028.8323471665524 docs/s, 0.8972957799296275 MB/s).
Processed 5100 documents (1034.2091617577757 docs/s, 0.9009465518440909 MB/s).
Processed 5200 documents (1038.183730940605 docs/s, 0.9030075814649406 MB/s).
Processed 5300 documents (1041.6586495443958 docs/s, 0.9053369784979305 MB/s).
Processed 5400 documents (1044.9539766559153 docs/s, 0.9071277474789929 MB/s).
Processed 5500 documents (1046.3650306901518 docs/s, 0.9081639586462895 MB/s).
Processed 5600 documents (1048.7924539068933 docs/s, 0.9100967579157073 MB/s).
Processed 5700 documents (1052.2622870902087 docs/s, 0.9127531725504879 MB/s).
Processed 5800 documents (1054.5030800058553 docs/s, 0.9157142263287714 MB/s).
Processed 5900 documents (1059.7019272813493 docs/s, 0.9178686686803795 MB/s).
Processed 6000 documents (1059.6815063398376 docs/s, 0.9181303779496783 MB/s).
Processed 6100 documents (1063.2851114676184 docs/s, 0.9198782219223729 MB/s).
Processed 6200 documents (1066.3967801931324 docs/s, 0.9227200915033248 MB/s).
Processed 6300 documents (1069.4742049968804 docs/s, 0.9247615387280403 MB/s).
Processed 6400 documents (1071.4283102781826 docs/s, 0.9253310711247418 MB/s).
Processed 6500 documents (1076.4308423075795 docs/s, 0.927823003864186 MB/s).
Processed 6600 documents (1080.6832634984266 docs/s, 0.9302894964375688 MB/s).
Processed 6700 documents (1084.397451300038 docs/s, 0.9318886244572675 MB/s).
Processed 6800 documents (1086.1487193706437 docs/s, 0.9348595253581037 MB/s).
Processed 6900 documents (1087.0879823630048 docs/s, 0.9361755003019657 MB/s).
Processed 7000 documents (1089.6505729269404 docs/s, 0.9391088768093419 MB/s).
Processed 7100 documents (1093.8213019164195 docs/s, 0.9425288656669819 MB/s).
Processed 7200 documents (1095.2519889281548 docs/s, 0.9441749502405283 MB/s).
Processed 7300 documents (1096.064652681706 docs/s, 0.9454755215221007 MB/s).
Processed 7400 documents (1099.8706995407156 docs/s, 0.9476494730035326 MB/s).
Processed 7500 documents (1102.1014294958984 docs/s, 0.949519284921672 MB/s).
Processed 7600 documents (1105.5783850300058 docs/s, 0.9512555401678863 MB/s).
Processed 7700 documents (1108.5371715132862 docs/s, 0.953457808444335 MB/s).
Processed 7800 documents (1110.6911912777482 docs/s, 0.9559848048283311 MB/s).
Processed 7900 documents (1111.308633853119 docs/s, 0.9587627880560639 MB/s).
Processed 8000 documents (1112.5253845256852 docs/s, 0.9598524220291513 MB/s).
Processed 8100 documents (1107.9573503503889 docs/s, 0.9570326111709778 MB/s).
Processed 8200 documents (1101.5736838840019 docs/s, 0.9513888668691332 MB/s).
Processed 8300 documents (1104.7169150313523 docs/s, 0.9545691909319388 MB/s).
Processed 8400 documents (1105.1751785362064 docs/s, 0.955716897713522 MB/s).
Processed 8500 documents (1107.2811136051005 docs/s, 0.9568425534848921 MB/s).
Processed 8600 documents (1111.0993018845525 docs/s, 0.9593719961195478 MB/s).
Processed 8700 documents (1113.3003045048206 docs/s, 0.961563294988248 MB/s).
Processed 8800 documents (1116.428968728601 docs/s, 0.9632621638462902 MB/s).
Processed 8900 documents (1118.174477463872 docs/s, 0.9645029463232418 MB/s).
Processed 9000 documents (1119.366311295253 docs/s, 0.9668497240291574 MB/s).
Processed 9100 documents (1121.6658086995078 docs/s, 0.9681949708107495 MB/s).
Processed 9200 documents (1124.1491570526603 docs/s, 0.9693218680263417 MB/s).
Processed 9300 documents (1126.638738761637 docs/s, 0.9728182383237711 MB/s).
Processed 9400 documents (1128.1680778666728 docs/s, 0.9741808572135104 MB/s).
Processed 9500 documents (1130.7785091089281 docs/s, 0.9758539826709395 MB/s).
Processed 9600 documents (1133.1699075004083 docs/s, 0.9774758616877791 MB/s).
Processed 9700 documents (1135.8416609360074 docs/s, 0.9789269570433953 MB/s).
Processed 9800 documents (1137.7800453559933 docs/s, 0.9799055810547384 MB/s).
Processed 9900 documents (1139.50573789508 docs/s, 0.9805926001463556 MB/s).
Processed 10000 documents (1141.5376306433773 docs/s, 0.9824856620038298 MB/s).
```

데이터셋 전처리가 완료되었습니다. 데이터를 확인해봅시다
- my-gpt2_text_document.bin
- my-gpt2_text_document.idx
와 같은 파일들이 생겼습니다. `idx`파일은 데이터의 위치 등의 메타데이터가 저장되어 있으며, `bin` 파일에는 실제로 Tokenized 된 데이터가 저장되어 있습니다.
```
[glogin01]$ ls
LICENSE    megatron/                  pretrain_bert.py  tasks/
README.md  megatron_datasets.jsonl    pretrain_gpt.py   tests/
apex/      merges.txt                 pretrain_ict.py   tools/
examples/  my-gpt2_text_document.bin  pretrain_t5.py    vocab.json
images/    my-gpt2_text_document.idx  pretrain_vit.py
```

이제 모델 학습을 시작해보겠습니다.
```
# 일단 Tensor parallelism만 사용해보도록 하겠습니다.
# Data parallelism과 Pipeline parallelism은 Multi-dimensional Parallelism 세션에서 사용해봅시다. :)
# 학습은 1000 스텝만 시키도록 하겠습니다. 실제 학습할 땐 더 많은 숫자로 설정해주세요.

!python -m torch.distributed.launch \
                  --nproc_per_node "4" \
                  --nnodes "1" \
                  --node_rank "0" \
                  --master_addr "localhost" \
                  --master_port "6000" \
                  ./pretrain_gpt.py \
                  --num-layers "24" \
                  --hidden-size "1024" \
                  --num-attention-heads "16" \
                  --seq-length "1024" \
                  --max-position-embeddings "1024" \
                  --micro-batch-size "4" \
                  --global-batch-size "8" \
                  --lr "0.00015" \
                  --train-iters "1000" \
                  --lr-decay-iters "300" \
                  --lr-decay-style cosine \
                  --vocab-file "vocab.json" \
                  --merge-file "merges.txt" \
                  --lr-warmup-fraction ".01" \
                  --fp16 \
                  --log-interval "10" \
                  --save-interval "500" \
                  --eval-interval "100" \
                  --eval-iters 10 \
                  --activations-checkpoint-method "uniform" \
                  --save "checkpoints/gpt2_345m" \
                  --load "checkpoints/gpt2_345m" \
                  --data-path "my-gpt2_text_document" \
                  --tensor-model-parallel-size "4" \
                  --pipeline-model-parallel-size "1" \
                  --DDP-impl "torch"

# Megatron-LM에는 위에 설정한 옵션 이외에도 굉장히 많은 옵션들이 있습니다.
# 모든 옵션을 설명하기는 어려우니 아래 주소를 참고해주세요.
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py

/opt/conda/lib/python3.7/site-packages/torch/distributed/launch.py:164: DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead
  "The module torch.distributed.launch is deprecated "
The module torch.distributed.launch is deprecated and going to be removed in future.Migrate to torch.distributed.run
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
WARNING:torch.distributed.run:--use_env is deprecated and will be removed in future releases.
 Please read local_rank from `os.environ('LOCAL_RANK')` instead.
INFO:torch.distributed.launcher.api:Starting elastic_operator with launch configs:
  entrypoint       : ./pretrain_gpt.py
  min_nodes        : 1
  max_nodes        : 1
  nproc_per_node   : 4
  run_id           : none
  rdzv_backend     : static
  rdzv_endpoint    : localhost:6000
  rdzv_configs     : {'rank': 0, 'timeout': 900}
  max_restarts     : 3
  monitor_interval : 5
  log_dir          : None
  metrics_cfg      : {}

INFO:torch.distributed.elastic.agent.server.local_elastic_agent:log directory set to: /tmp/torchelastic_2zzg5_sz/none_mjqneoph
INFO:torch.distributed.elastic.agent.server.api:[default] starting workers for entrypoint: python
INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous'ing worker group
/opt/conda/lib/python3.7/site-packages/torch/distributed/elastic/utils/store.py:53: FutureWarning: This is an experimental API and will be changed in future.
  "This is an experimental API and will be changed in future.", FutureWarning
INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous complete for workers. Result:
  restart_count=0
  master_addr=localhost
  master_port=6000
  group_rank=0
  group_world_size=1
  local_ranks=[0, 1, 2, 3]
  role_ranks=[0, 1, 2, 3]
  global_ranks=[0, 1, 2, 3]
  role_world_sizes=[4, 4, 4, 4]
  global_world_sizes=[4, 4, 4, 4]

INFO:torch.distributed.elastic.agent.server.api:[default] Starting worker group
INFO:torch.distributed.elastic.multiprocessing:Setting worker0 reply file to: /tmp/torchelastic_2zzg5_sz/none_mjqneoph/attempt_0/0/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker1 reply file to: /tmp/torchelastic_2zzg5_sz/none_mjqneoph/attempt_0/1/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker2 reply file to: /tmp/torchelastic_2zzg5_sz/none_mjqneoph/attempt_0/2/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker3 reply file to: /tmp/torchelastic_2zzg5_sz/none_mjqneoph/attempt_0/3/error.json
using world size: 4, data-parallel-size: 1, tensor-model-parallel size: 4, pipeline-model-parallel size: 1 
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  activations_checkpoint_method ................... uniform
  activations_checkpoint_num_layers ............... 1
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... infer
  data_parallel_size .............................. 1
  data_path ....................................... ['my-gpt2_text_document']
  dataloader_type ................................. single
  DDP_impl ........................................ torch
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  encoder_seq_length .............................. 1024
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 8
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ checkpoints/gpt2_345m
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.00015
  lr_decay_iters .................................. 300
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 1024
  merge_file ...................................... merges.txt
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 0.0
  mmap_warmup ..................................... False
  no_async_tensor_model_parallel_allreduce ........ False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ checkpoints/gpt2_345m
  save_interval ................................... 500
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 1024
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 969, 30, 1
  tensor_model_parallel_size ...................... 4
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. GPT2BPETokenizer
  train_iters ..................................... 1000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_local_ddp ............. False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... vocab.json
  weight_decay .................................... 0.01
  world_size ...................................... 4
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 2
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 431 dummy tokens (new size: 50688)
> initializing torch distributed ...
> initializing tensor model parallel with size 4
> initializing pipeline model parallel with size 1
[W ProcessGroupNCCL.cpp:1569] Rank 3 using best-guess GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
[W ProcessGroupNCCL.cpp:1569] Rank 1 using best-guess GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
> setting random seeds to 1234 ...
> initializing model parallel cuda seeds on global rank 0, model parallel rank 0, and data parallel rank 0 with model parallel seed: 3952 and data parallel seed: 1234
[W ProcessGroupNCCL.cpp:1569] Rank 2 using best-guess GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
> compiling dataset index builder ...
make: Entering directory '/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/data'
g++ -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color -I/opt/conda/include/python3.7m -I/opt/conda/lib/python3.7/site-packages/pybind11/include helpers.cpp -o helpers.cpython-37m-x86_64-linux-gnu.so
make: Leaving directory '/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/data'
>>> done with dataset index builder. Compilation time: 4.847 seconds
> compiling and loading fused kernels ...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_upper_triang_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF scaled_upper_triang_masked_softmax.o.d -DTORCH_EXTENSION_NAME=scaled_upper_triang_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/lib/python3.7/site-packages/torch/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/TH -isystem /opt/conda/lib/python3.7/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /opt/conda/include/python3.7m -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++14 -O3 -c /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/scaled_upper_triang_masked_softmax.cpp -o scaled_upper_triang_masked_softmax.o 
[2/3] /usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=scaled_upper_triang_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/lib/python3.7/site-packages/torch/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/TH -isystem /opt/conda/lib/python3.7/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /opt/conda/include/python3.7m -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -std=c++14 -c /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/scaled_upper_triang_masked_softmax_cuda.cu -o scaled_upper_triang_masked_softmax_cuda.cuda.o 
[3/3] c++ scaled_upper_triang_masked_softmax.o scaled_upper_triang_masked_softmax_cuda.cuda.o -shared -L/opt/conda/lib/python3.7/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda_cu -ltorch_cuda_cpp -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o scaled_upper_triang_masked_softmax_cuda.so
Loading extension module scaled_upper_triang_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module scaled_masked_softmax_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF scaled_masked_softmax.o.d -DTORCH_EXTENSION_NAME=scaled_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/lib/python3.7/site-packages/torch/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/TH -isystem /opt/conda/lib/python3.7/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /opt/conda/include/python3.7m -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++14 -O3 -c /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/scaled_masked_softmax.cpp -o scaled_masked_softmax.o 
[2/3] /usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=scaled_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/lib/python3.7/site-packages/torch/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/TH -isystem /opt/conda/lib/python3.7/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /opt/conda/include/python3.7m -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 -std=c++14 -c /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/scaled_masked_softmax_cuda.cu -o scaled_masked_softmax_cuda.cuda.o 
[3/3] c++ scaled_masked_softmax.o scaled_masked_softmax_cuda.cuda.o -shared -L/opt/conda/lib/python3.7/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda_cu -ltorch_cuda_cpp -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o scaled_masked_softmax_cuda.so
Loading extension module scaled_masked_softmax_cuda...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/build/build.ninja...
Building extension module fused_mix_prec_layer_norm_cuda...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF layer_norm_cuda.o.d -DTORCH_EXTENSION_NAME=fused_mix_prec_layer_norm_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/lib/python3.7/site-packages/torch/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/TH -isystem /opt/conda/lib/python3.7/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /opt/conda/include/python3.7m -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++14 -O3 -c /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda.cpp -o layer_norm_cuda.o 
[2/3] /usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=fused_mix_prec_layer_norm_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/lib/python3.7/site-packages/torch/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/lib/python3.7/site-packages/torch/include/TH -isystem /opt/conda/lib/python3.7/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /opt/conda/include/python3.7m -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_80,code=compute_80 -gencode=arch=compute_80,code=sm_80 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -maxrregcount=50 -gencode arch=compute_80,code=sm_80 -std=c++14 -c /home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu -o layer_norm_cuda_kernel.cuda.o 
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In function ‘void cuda_layer_norm(at::Tensor*, at::Tensor*, at::Tensor*, at::Tensor*, int, int, c10::IntList, at::Tensor*, at::Tensor*, double)’:
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:224: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:247: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:272: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:296: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                        ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:359: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:414: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                              ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:119: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:142: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                              ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:167: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:191: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                               ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:258: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                  ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:317: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                             ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:131: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                   ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:154: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                          ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:179: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                   ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:203: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                           ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:274: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                  ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:337: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                 ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:151: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:174: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                              ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:199: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:227: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                   ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:294: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                      ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:353: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                 ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:166: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                      ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:189: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                             ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:214: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                      ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:246: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                      ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:317: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                             ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:703:380: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                            ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In function ‘void cuda_layer_norm_gradient(at::Tensor*, at::Tensor*, at::Tensor*, at::Tensor*, int, int, c10::IntList, at::Tensor*, at::Tensor*, double, at::Tensor*, at::Tensor*, at::Tensor*)’:
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:224: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:247: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:272: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:333: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                             ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:389: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                     ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:438: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:489: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:550: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:120: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                        ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:143: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                               ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:168: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                        ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:233: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                         ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:293: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                     ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:342: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                      ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:397: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                             ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:462: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:132: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                    ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:155: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                           ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:180: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                    ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:249: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                         ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:313: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                         ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:362: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                          ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:421: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:490: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:152: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                        ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:175: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                               ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:200: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                        ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:265: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                         ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:325: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                     ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:378: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                          ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:433: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                 ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:498: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:167: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:190: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                              ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:215: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                       ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:284: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                            ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:348: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                            ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:405: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                     ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:464: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:533: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = float; U = float; V = float]’:
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:562:   required from here
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:779:97: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = float; U = float; V = c10::Half]’:
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:474:   required from here
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:779:97: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = float; U = float; V = c10::BFloat16]’:
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:502:   required from here
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:779:97: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = c10::Half; U = float; V = c10::Half]’:
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:510:   required from here
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:779:97: warning: ‘T* at::Tensor::data() const [with T = c10::Half]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu: In instantiation of ‘void HostLayerNormGradient(const V*, const U*, const U*, at::Tensor*, int, int, const V*, const V*, double, T*, V*, V*) [with T = c10::BFloat16; U = float; V = c10::BFloat16]’:
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:811:545:   required from here
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:748:106: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
                                                                                                          ^                                                                                                                                                     
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:761:102: warning: ‘T* at::Tensor::data() const [with T = float]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
       cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
                                                                                                      ^                                                                                                                        
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
/home/ubuntu/kevin/jupyter/notebooks/Megatron-LM/megatron/fused_kernels/layer_norm_cuda_kernel.cu:779:97: warning: ‘T* at::Tensor::data() const [with T = c10::BFloat16]’ is deprecated: Tensor.data<T>() is deprecated. Please use Tensor.data_ptr<T>() instead. [-Wdeprecated-declarations]
     cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
                                                                                                 ^                                                                                         
/opt/conda/lib/python3.7/site-packages/torch/include/ATen/core/TensorBody.h:501:1: note: declared here
   T * data() const {
 ^ ~~
[3/3] c++ layer_norm_cuda.o layer_norm_cuda_kernel.cuda.o -shared -L/opt/conda/lib/python3.7/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda_cu -ltorch_cuda_cpp -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o fused_mix_prec_layer_norm_cuda.so
Loading extension module fused_mix_prec_layer_norm_cuda...
[W ProcessGroupNCCL.cpp:1569] Rank 0 using best-guess GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
>>> done with compiling and loading fused kernels. Compilation time: 213.651 seconds
time to initialize megatron (seconds): 219.000
[after megatron is initialized] datetime: 2021-10-24 01:50:51 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (2, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (3, 0): 89714688
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 89714688
> learning rate decay style: cosine
WARNING: could not find the metadata file checkpoints/gpt2_345m/latest_checkpointed_iteration.txt 
    will not load any checkpoints and will start from random
time (ms) | load-checkpoint: 0.33
[after model, optimizer, and learning rate scheduler are built] datetime: 2021-10-24 01:50:52 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      8000
    validation: 880
    test:       80
> building train, validation, and test datasets for GPT ...
 > building dataset index ...
    reading sizes...
    reading pointers...
    reading document index...
    creating numpy buffer of mmap...
    creating memory view of numpy buffer...
 > finished creating indexed dataset in 0.002126 seconds
    number of documents: 10000
 > dataset split:
    train:
     document indices in [0, 9690) total of 9690 documents
    validation:
     document indices in [9690, 9990) total of 300 documents
    test:
     document indices in [9990, 10000) total of 10 documents
 > WARNING: could not find index map files, building the indices on rank 0 ...
 > last epoch number of samples (722) is smaller than 80% of number of samples per epoch (1819), setting separate_last_epoch to True
 > elasped time to build and save doc-idx mapping (seconds): 0.005900
    using:
     number of documents:       9690
     number of epochs:          5
     sequence length:           1024
     total number of samples:   9097
 > elasped time to build and save sample-idx mapping (seconds): 0.002049
 > building shuffle index with split [0, 7278) and [7278, 9097) ...
 > elasped time to build and save shuffle-idx mapping (seconds): 0.001257
 > loading doc-idx mapping from my-gpt2_text_document_train_indexmap_8000ns_1024sl_1234s_doc_idx.npy
 > loading sample-idx mapping from my-gpt2_text_document_train_indexmap_8000ns_1024sl_1234s_sample_idx.npy
 > loading shuffle-idx mapping from my-gpt2_text_document_train_indexmap_8000ns_1024sl_1234s_shuffle_idx.npy
    loaded indexed file in 0.048 seconds
    total number of samples: 9098
    total number of epochs: 5
 > WARNING: could not find index map files, building the indices on rank 0 ...
 > last epoch number of samples (16) is smaller than 80% of number of samples per epoch (54), setting separate_last_epoch to True
 > elasped time to build and save doc-idx mapping (seconds): 0.000634
    using:
     number of documents:       300
     number of epochs:          17
     sequence length:           1024
     total number of samples:   918
 > elasped time to build and save sample-idx mapping (seconds): 0.000432
 > building shuffle index with split [0, 864) and [864, 918) ...
 > elasped time to build and save shuffle-idx mapping (seconds): 0.000260
 > loading doc-idx mapping from my-gpt2_text_document_valid_indexmap_880ns_1024sl_1234s_doc_idx.npy
 > loading sample-idx mapping from my-gpt2_text_document_valid_indexmap_880ns_1024sl_1234s_sample_idx.npy
 > loading shuffle-idx mapping from my-gpt2_text_document_valid_indexmap_880ns_1024sl_1234s_shuffle_idx.npy
    loaded indexed file in 0.001 seconds
    total number of samples: 919
    total number of epochs: 17
 > WARNING: could not find index map files, building the indices on rank 0 ...
 > last epoch number of samples (1) is larger than 80% of number of samples per epoch (1), setting separate_last_epoch to False
 > elasped time to build and save doc-idx mapping (seconds): 0.000269
    using:
     number of documents:       10
     number of epochs:          53
     sequence length:           1024
     total number of samples:   80
 > elasped time to build and save sample-idx mapping (seconds): 0.000271
 > building shuffle index with split [0, 80) and [80, 80) ...
 > elasped time to build and save shuffle-idx mapping (seconds): 0.000210
 > loading doc-idx mapping from my-gpt2_text_document_test_indexmap_80ns_1024sl_1234s_doc_idx.npy
 > loading sample-idx mapping from my-gpt2_text_document_test_indexmap_80ns_1024sl_1234s_sample_idx.npy
 > loading shuffle-idx mapping from my-gpt2_text_document_test_indexmap_80ns_1024sl_1234s_shuffle_idx.npy
    loaded indexed file in 0.001 seconds
    total number of samples: 81
    total number of epochs: 53
> finished creating GPT datasets ...
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
time (ms) | model-and-optimizer-setup: 760.83 | train/valid/test-data-iterators-setup: 3023.08
[after dataloaders are built] datetime: 2021-10-24 01:50:56 
done with setup ...
training ...
[before the start of training step] datetime: 2021-10-24 01:50:56 
 iteration       10/    1000 | consumed samples:           80 | elapsed time per iteration (ms): 1210.0 | learning rate: 0.000E+00 | global batch size:     8 | loss scale: 8388608.0 | number of skipped iterations:  10 | number of nan iterations:   0 |
time (ms) | forward-compute: 762.12 | backward-compute: 240.12 | backward-embedding-all-reduce: 0.05 | optimizer-copy-to-main-grad: 6.75 | optimizer-unscale-and-check-inf: 198.62 | optimizer: 205.56 | batch-generator: 3.78
 iteration       20/    1000 | consumed samples:          160 | elapsed time per iteration (ms): 223.5 | learning rate: 1.500E-04 | global batch size:     8 | lm loss: 1.032014E+01 | loss scale: 131072.0 | number of skipped iterations:   6 | number of nan iterations:   0 |
[Rank 1] (after 20 iterations) memory (MB) | allocated: 1735.81884765625 | max allocated: 2072.75390625 | reserved: 2394.0 | max reserved: 2394.0
[Rank 2] (after 20 iterations) memory (MB) | allocated: 1728.82275390625 | max allocated: 2071.75634765625 | reserved: 2394.0 | max reserved: 2394.0[Rank 0] (after 20 iterations) memory (MB) | allocated: 1734.69384765625 | max allocated: 2070.740234375 | reserved: 2390.0 | max reserved: 2390.0

[Rank 3] (after 20 iterations) memory (MB) | allocated: 1728.1044921875 | max allocated: 2071.29248046875 | reserved: 2340.0 | max reserved: 2340.0
time (ms) | forward-compute: 67.56 | backward-compute: 134.86 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.44 | optimizer-unscale-and-check-inf: 1.96 | optimizer-clip-main-grad: 1.22 | optimizer-copy-main-to-model-params: 1.02 | optimizer: 18.85 | batch-generator: 1.87
 iteration       30/    1000 | consumed samples:          240 | elapsed time per iteration (ms): 217.2 | learning rate: 1.496E-04 | global batch size:     8 | lm loss: 8.906515E+00 | loss scale: 65536.0 | grad norm: 1.458 | number of skipped iterations:   1 | number of nan iterations:   0 |
time (ms) | forward-compute: 66.82 | backward-compute: 133.33 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.07 | optimizer-unscale-and-check-inf: 2.00 | optimizer-clip-main-grad: 2.97 | optimizer-copy-main-to-model-params: 2.27 | optimizer: 14.86 | batch-generator: 2.12
 iteration       40/    1000 | consumed samples:          320 | elapsed time per iteration (ms): 217.4 | learning rate: 1.483E-04 | global batch size:     8 | lm loss: 7.911682E+00 | loss scale: 65536.0 | grad norm: 0.737 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 66.62 | backward-compute: 133.69 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.25 | optimizer-unscale-and-check-inf: 1.86 | optimizer-clip-main-grad: 2.64 | optimizer-copy-main-to-model-params: 2.51 | optimizer: 15.06 | batch-generator: 2.56
 iteration       50/    1000 | consumed samples:          400 | elapsed time per iteration (ms): 207.5 | learning rate: 1.463E-04 | global batch size:     8 | lm loss: 7.568282E+00 | loss scale: 65536.0 | grad norm: 22.903 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 61.79 | backward-compute: 129.01 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.11 | optimizer-unscale-and-check-inf: 2.00 | optimizer-clip-main-grad: 2.39 | optimizer-copy-main-to-model-params: 2.47 | optimizer: 14.76 | batch-generator: 1.75
 iteration       60/    1000 | consumed samples:          480 | elapsed time per iteration (ms): 217.9 | learning rate: 1.434E-04 | global batch size:     8 | lm loss: 7.328167E+00 | loss scale: 65536.0 | grad norm: 1.077 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.53 | backward-compute: 135.97 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.20 | optimizer-unscale-and-check-inf: 2.04 | optimizer-clip-main-grad: 2.71 | optimizer-copy-main-to-model-params: 2.61 | optimizer: 15.39 | batch-generator: 2.10
 iteration       70/    1000 | consumed samples:          560 | elapsed time per iteration (ms): 214.9 | learning rate: 1.398E-04 | global batch size:     8 | lm loss: 7.168855E+00 | loss scale: 65536.0 | grad norm: 0.704 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 61.95 | backward-compute: 134.92 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.15 | optimizer-unscale-and-check-inf: 2.74 | optimizer-clip-main-grad: 2.76 | optimizer-copy-main-to-model-params: 2.52 | optimizer: 16.01 | batch-generator: 1.82
 iteration       80/    1000 | consumed samples:          640 | elapsed time per iteration (ms): 210.3 | learning rate: 1.354E-04 | global batch size:     8 | lm loss: 7.056370E+00 | loss scale: 65536.0 | grad norm: 0.623 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 62.49 | backward-compute: 130.66 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.08 | optimizer-unscale-and-check-inf: 2.25 | optimizer-clip-main-grad: 2.45 | optimizer-copy-main-to-model-params: 2.50 | optimizer: 15.09 | batch-generator: 1.76
 iteration       90/    1000 | consumed samples:          720 | elapsed time per iteration (ms): 211.4 | learning rate: 1.304E-04 | global batch size:     8 | lm loss: 7.006987E+00 | loss scale: 65536.0 | grad norm: 0.671 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 63.06 | backward-compute: 131.17 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.17 | optimizer-unscale-and-check-inf: 2.16 | optimizer-clip-main-grad: 2.40 | optimizer-copy-main-to-model-params: 2.53 | optimizer: 15.09 | batch-generator: 1.74
 iteration      100/    1000 | consumed samples:          800 | elapsed time per iteration (ms): 214.3 | learning rate: 1.247E-04 | global batch size:     8 | lm loss: 6.978156E+00 | loss scale: 65536.0 | grad norm: 0.562 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.87 | backward-compute: 131.98 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.17 | optimizer-unscale-and-check-inf: 2.32 | optimizer-clip-main-grad: 2.44 | optimizer-copy-main-to-model-params: 2.52 | optimizer: 15.29 | batch-generator: 1.81
-----------------------------------------------------------------------------------------------
 validation loss at iteration 100 | lm loss value: 7.086492E+00 | lm loss PPL: 1.195706E+03 | 
-----------------------------------------------------------------------------------------------
 iteration      110/    1000 | consumed samples:          880 | elapsed time per iteration (ms): 281.4 | learning rate: 1.185E-04 | global batch size:     8 | lm loss: 6.893764E+00 | loss scale: 65536.0 | grad norm: 0.518 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 124.13 | backward-compute: 139.23 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.45 | optimizer-unscale-and-check-inf: 2.08 | optimizer-clip-main-grad: 2.51 | optimizer-copy-main-to-model-params: 2.51 | optimizer: 15.41 | batch-generator: 5.25
 iteration      120/    1000 | consumed samples:          960 | elapsed time per iteration (ms): 221.7 | learning rate: 1.118E-04 | global batch size:     8 | lm loss: 6.834893E+00 | loss scale: 65536.0 | grad norm: 0.615 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 68.97 | backward-compute: 134.54 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.76 | optimizer-unscale-and-check-inf: 3.53 | optimizer-clip-main-grad: 2.75 | optimizer-copy-main-to-model-params: 2.63 | optimizer: 16.50 | batch-generator: 2.29
 iteration      130/    1000 | consumed samples:         1040 | elapsed time per iteration (ms): 227.8 | learning rate: 1.047E-04 | global batch size:     8 | lm loss: 6.838863E+00 | loss scale: 65536.0 | grad norm: 0.607 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 65.06 | backward-compute: 146.34 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.44 | optimizer-unscale-and-check-inf: 2.93 | optimizer-clip-main-grad: 2.40 | optimizer-copy-main-to-model-params: 2.41 | optimizer: 14.87 | batch-generator: 2.45
 iteration      140/    1000 | consumed samples:         1120 | elapsed time per iteration (ms): 228.0 | learning rate: 9.727E-05 | global batch size:     8 | lm loss: 6.824157E+00 | loss scale: 65536.0 | grad norm: 0.715 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.07 | backward-compute: 146.24 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.07 | optimizer-unscale-and-check-inf: 2.36 | optimizer-clip-main-grad: 3.20 | optimizer-copy-main-to-model-params: 2.45 | optimizer: 15.85 | batch-generator: 1.77
 iteration      150/    1000 | consumed samples:         1200 | elapsed time per iteration (ms): 228.7 | learning rate: 8.958E-05 | global batch size:     8 | lm loss: 6.753051E+00 | loss scale: 65536.0 | grad norm: 0.732 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 68.35 | backward-compute: 142.91 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.34 | optimizer-unscale-and-check-inf: 2.12 | optimizer-clip-main-grad: 2.61 | optimizer-copy-main-to-model-params: 2.61 | optimizer: 15.49 | batch-generator: 1.82
 iteration      160/    1000 | consumed samples:         1280 | elapsed time per iteration (ms): 224.1 | learning rate: 8.173E-05 | global batch size:     8 | lm loss: 6.697730E+00 | loss scale: 65536.0 | grad norm: 0.740 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 68.91 | backward-compute: 136.67 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.68 | optimizer-unscale-and-check-inf: 2.34 | optimizer-clip-main-grad: 2.88 | optimizer-copy-main-to-model-params: 2.65 | optimizer: 16.45 | batch-generator: 2.46
 iteration      170/    1000 | consumed samples:         1360 | elapsed time per iteration (ms): 210.2 | learning rate: 7.381E-05 | global batch size:     8 | lm loss: 6.682823E+00 | loss scale: 65536.0 | grad norm: 0.645 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 65.78 | backward-compute: 127.74 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.89 | optimizer-unscale-and-check-inf: 2.28 | optimizer-clip-main-grad: 2.51 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 14.90 | batch-generator: 1.43
 iteration      180/    1000 | consumed samples:         1440 | elapsed time per iteration (ms): 225.2 | learning rate: 6.590E-05 | global batch size:     8 | lm loss: 6.639360E+00 | loss scale: 65536.0 | grad norm: 0.676 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 63.28 | backward-compute: 146.22 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.45 | optimizer-unscale-and-check-inf: 2.01 | optimizer-clip-main-grad: 2.35 | optimizer-copy-main-to-model-params: 2.43 | optimizer: 14.04 | batch-generator: 1.64
 iteration      190/    1000 | consumed samples:         1520 | elapsed time per iteration (ms): 213.4 | learning rate: 5.809E-05 | global batch size:     8 | lm loss: 6.666003E+00 | loss scale: 65536.0 | grad norm: 0.697 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 62.21 | backward-compute: 135.18 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.62 | optimizer-unscale-and-check-inf: 2.06 | optimizer-clip-main-grad: 2.37 | optimizer-copy-main-to-model-params: 2.49 | optimizer: 14.32 | batch-generator: 1.87
 iteration      200/    1000 | consumed samples:         1600 | elapsed time per iteration (ms): 230.0 | learning rate: 5.047E-05 | global batch size:     8 | lm loss: 6.610323E+00 | loss scale: 65536.0 | grad norm: 0.747 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 74.85 | backward-compute: 137.16 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.46 | optimizer-unscale-and-check-inf: 2.66 | optimizer-clip-main-grad: 2.66 | optimizer-copy-main-to-model-params: 2.44 | optimizer: 15.05 | batch-generator: 2.15
-----------------------------------------------------------------------------------------------
 validation loss at iteration 200 | lm loss value: 6.859974E+00 | lm loss PPL: 9.533422E+02 | 
-----------------------------------------------------------------------------------------------
 iteration      210/    1000 | consumed samples:         1680 | elapsed time per iteration (ms): 310.6 | learning rate: 4.312E-05 | global batch size:     8 | lm loss: 6.620341E+00 | loss scale: 65536.0 | grad norm: 0.699 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 151.66 | backward-compute: 138.52 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.81 | optimizer-unscale-and-check-inf: 4.17 | optimizer-clip-main-grad: 3.51 | optimizer-copy-main-to-model-params: 2.73 | optimizer: 18.15 | batch-generator: 3.99
 iteration      220/    1000 | consumed samples:         1760 | elapsed time per iteration (ms): 228.1 | learning rate: 3.613E-05 | global batch size:     8 | lm loss: 6.600768E+00 | loss scale: 65536.0 | grad norm: 0.754 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 77.23 | backward-compute: 132.63 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.11 | optimizer-unscale-and-check-inf: 4.54 | optimizer-clip-main-grad: 2.83 | optimizer-copy-main-to-model-params: 2.48 | optimizer: 16.73 | batch-generator: 3.24
 iteration      230/    1000 | consumed samples:         1840 | elapsed time per iteration (ms): 221.8 | learning rate: 2.958E-05 | global batch size:     8 | lm loss: 6.569440E+00 | loss scale: 65536.0 | grad norm: 0.718 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 69.80 | backward-compute: 135.45 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.42 | optimizer-unscale-and-check-inf: 2.80 | optimizer-clip-main-grad: 2.48 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 14.97 | batch-generator: 2.90
 iteration      240/    1000 | consumed samples:         1920 | elapsed time per iteration (ms): 231.9 | learning rate: 2.353E-05 | global batch size:     8 | lm loss: 6.547186E+00 | loss scale: 65536.0 | grad norm: 0.693 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 72.26 | backward-compute: 142.22 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.48 | optimizer-unscale-and-check-inf: 3.20 | optimizer-clip-main-grad: 2.70 | optimizer-copy-main-to-model-params: 2.57 | optimizer: 15.80 | batch-generator: 3.49
 iteration      250/    1000 | consumed samples:         2000 | elapsed time per iteration (ms): 238.5 | learning rate: 1.806E-05 | global batch size:     8 | lm loss: 6.529446E+00 | loss scale: 65536.0 | grad norm: 0.639 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 75.63 | backward-compute: 144.86 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.25 | optimizer-unscale-and-check-inf: 3.83 | optimizer-clip-main-grad: 3.11 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 16.44 | batch-generator: 2.98
 iteration      260/    1000 | consumed samples:         2080 | elapsed time per iteration (ms): 239.5 | learning rate: 1.322E-05 | global batch size:     8 | lm loss: 6.534592E+00 | loss scale: 65536.0 | grad norm: 0.637 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 79.20 | backward-compute: 142.68 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.32 | optimizer-unscale-and-check-inf: 3.56 | optimizer-clip-main-grad: 2.91 | optimizer-copy-main-to-model-params: 2.42 | optimizer: 16.00 | batch-generator: 3.47
 iteration      270/    1000 | consumed samples:         2160 | elapsed time per iteration (ms): 234.9 | learning rate: 9.079E-06 | global batch size:     8 | lm loss: 6.515455E+00 | loss scale: 65536.0 | grad norm: 0.597 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 69.58 | backward-compute: 148.37 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.22 | optimizer-unscale-and-check-inf: 3.18 | optimizer-clip-main-grad: 2.56 | optimizer-copy-main-to-model-params: 2.50 | optimizer: 15.25 | batch-generator: 2.36
 iteration      280/    1000 | consumed samples:         2240 | elapsed time per iteration (ms): 216.3 | learning rate: 5.671E-06 | global batch size:     8 | lm loss: 6.529461E+00 | loss scale: 65536.0 | grad norm: 0.583 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.92 | backward-compute: 132.46 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 3.71 | optimizer-unscale-and-check-inf: 3.36 | optimizer-clip-main-grad: 2.55 | optimizer-copy-main-to-model-params: 2.35 | optimizer: 14.64 | batch-generator: 2.19
 iteration      290/    1000 | consumed samples:         2320 | elapsed time per iteration (ms): 207.9 | learning rate: 3.038E-06 | global batch size:     8 | lm loss: 6.513412E+00 | loss scale: 65536.0 | grad norm: 0.618 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.06 | backward-compute: 127.69 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 3.65 | optimizer-unscale-and-check-inf: 3.66 | optimizer-clip-main-grad: 2.44 | optimizer-copy-main-to-model-params: 2.40 | optimizer: 14.82 | batch-generator: 2.86
 iteration      300/    1000 | consumed samples:         2400 | elapsed time per iteration (ms): 215.7 | learning rate: 1.209E-06 | global batch size:     8 | lm loss: 6.521043E+00 | loss scale: 65536.0 | grad norm: 0.642 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 65.02 | backward-compute: 134.84 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 4.03 | optimizer-unscale-and-check-inf: 2.93 | optimizer-clip-main-grad: 2.40 | optimizer-copy-main-to-model-params: 2.37 | optimizer: 14.42 | batch-generator: 2.00
-----------------------------------------------------------------------------------------------
 validation loss at iteration 300 | lm loss value: 6.749321E+00 | lm loss PPL: 8.534790E+02 | 
-----------------------------------------------------------------------------------------------
 iteration      310/    1000 | consumed samples:         2480 | elapsed time per iteration (ms): 279.7 | learning rate: 2.055E-07 | global batch size:     8 | lm loss: 6.526622E+00 | loss scale: 65536.0 | grad norm: 0.633 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 125.71 | backward-compute: 136.51 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.30 | optimizer-unscale-and-check-inf: 2.96 | optimizer-clip-main-grad: 2.82 | optimizer-copy-main-to-model-params: 2.53 | optimizer: 15.45 | batch-generator: 4.78
 iteration      320/    1000 | consumed samples:         2560 | elapsed time per iteration (ms): 220.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.520487E+00 | loss scale: 65536.0 | grad norm: 0.600 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 68.29 | backward-compute: 135.67 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.19 | optimizer-unscale-and-check-inf: 2.77 | optimizer-clip-main-grad: 2.60 | optimizer-copy-main-to-model-params: 2.41 | optimizer: 14.74 | batch-generator: 2.25
 iteration      330/    1000 | consumed samples:         2640 | elapsed time per iteration (ms): 222.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.492335E+00 | loss scale: 65536.0 | grad norm: 0.644 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.09 | backward-compute: 138.64 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.27 | optimizer-unscale-and-check-inf: 3.01 | optimizer-clip-main-grad: 2.46 | optimizer-copy-main-to-model-params: 2.43 | optimizer: 14.97 | batch-generator: 2.09
 iteration      340/    1000 | consumed samples:         2720 | elapsed time per iteration (ms): 211.9 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.539624E+00 | loss scale: 65536.0 | grad norm: 0.597 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 65.04 | backward-compute: 130.94 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.01 | optimizer-unscale-and-check-inf: 2.98 | optimizer-clip-main-grad: 2.35 | optimizer-copy-main-to-model-params: 2.40 | optimizer: 14.49 | batch-generator: 2.48
 iteration      350/    1000 | consumed samples:         2800 | elapsed time per iteration (ms): 226.3 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.521507E+00 | loss scale: 65536.0 | grad norm: 0.657 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 78.07 | backward-compute: 130.17 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 3.99 | optimizer-unscale-and-check-inf: 4.61 | optimizer-clip-main-grad: 2.88 | optimizer-copy-main-to-model-params: 2.42 | optimizer: 16.64 | batch-generator: 4.07
 iteration      360/    1000 | consumed samples:         2880 | elapsed time per iteration (ms): 217.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.495512E+00 | loss scale: 65536.0 | grad norm: 0.615 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 66.52 | backward-compute: 134.08 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 3.94 | optimizer-unscale-and-check-inf: 3.63 | optimizer-clip-main-grad: 2.44 | optimizer-copy-main-to-model-params: 2.38 | optimizer: 15.10 | batch-generator: 3.51
 iteration      370/    1000 | consumed samples:         2960 | elapsed time per iteration (ms): 225.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.504739E+00 | loss scale: 65536.0 | grad norm: 0.596 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.82 | backward-compute: 140.81 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 3.87 | optimizer-unscale-and-check-inf: 3.75 | optimizer-clip-main-grad: 2.45 | optimizer-copy-main-to-model-params: 2.37 | optimizer: 15.16 | batch-generator: 3.64
 iteration      380/    1000 | consumed samples:         3040 | elapsed time per iteration (ms): 213.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.548125E+00 | loss scale: 65536.0 | grad norm: 0.618 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 65.14 | backward-compute: 132.85 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 3.86 | optimizer-unscale-and-check-inf: 3.05 | optimizer-clip-main-grad: 2.35 | optimizer-copy-main-to-model-params: 2.37 | optimizer: 14.36 | batch-generator: 2.85
 iteration      390/    1000 | consumed samples:         3120 | elapsed time per iteration (ms): 218.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.529753E+00 | loss scale: 65536.0 | grad norm: 0.588 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 63.89 | backward-compute: 138.93 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.09 | optimizer-unscale-and-check-inf: 2.31 | optimizer-clip-main-grad: 2.35 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 13.97 | batch-generator: 2.09
 iteration      400/    1000 | consumed samples:         3200 | elapsed time per iteration (ms): 222.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.502573E+00 | loss scale: 65536.0 | grad norm: 0.583 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 72.92 | backward-compute: 132.56 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 3.93 | optimizer-unscale-and-check-inf: 3.37 | optimizer-clip-main-grad: 2.87 | optimizer-copy-main-to-model-params: 2.38 | optimizer: 15.28 | batch-generator: 3.32
-----------------------------------------------------------------------------------------------
 validation loss at iteration 400 | lm loss value: 6.788503E+00 | lm loss PPL: 8.875836E+02 | 
-----------------------------------------------------------------------------------------------
 iteration      410/    1000 | consumed samples:         3280 | elapsed time per iteration (ms): 288.0 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.540978E+00 | loss scale: 65536.0 | grad norm: 0.642 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 133.56 | backward-compute: 136.65 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.93 | optimizer-unscale-and-check-inf: 2.58 | optimizer-clip-main-grad: 2.55 | optimizer-copy-main-to-model-params: 2.70 | optimizer: 15.68 | batch-generator: 3.58
 iteration      420/    1000 | consumed samples:         3360 | elapsed time per iteration (ms): 217.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.535336E+00 | loss scale: 65536.0 | grad norm: 0.578 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 69.30 | backward-compute: 130.61 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.37 | optimizer-unscale-and-check-inf: 2.22 | optimizer-clip-main-grad: 2.53 | optimizer-copy-main-to-model-params: 2.73 | optimizer: 15.65 | batch-generator: 2.09
 iteration      430/    1000 | consumed samples:         3440 | elapsed time per iteration (ms): 207.8 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.531789E+00 | loss scale: 65536.0 | grad norm: 0.585 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 61.52 | backward-compute: 130.11 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 5.04 | optimizer-unscale-and-check-inf: 1.98 | optimizer-clip-main-grad: 2.34 | optimizer-copy-main-to-model-params: 2.43 | optimizer: 14.43 | batch-generator: 1.86
 iteration      440/    1000 | consumed samples:         3520 | elapsed time per iteration (ms): 208.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.468220E+00 | loss scale: 65536.0 | grad norm: 0.582 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 62.16 | backward-compute: 129.98 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.99 | optimizer-unscale-and-check-inf: 1.94 | optimizer-clip-main-grad: 2.39 | optimizer-copy-main-to-model-params: 2.54 | optimizer: 14.53 | batch-generator: 1.61
 iteration      450/    1000 | consumed samples:         3600 | elapsed time per iteration (ms): 208.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.536092E+00 | loss scale: 65536.0 | grad norm: 0.608 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 62.16 | backward-compute: 130.02 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.07 | optimizer-unscale-and-check-inf: 1.97 | optimizer-clip-main-grad: 2.44 | optimizer-copy-main-to-model-params: 2.44 | optimizer: 14.59 | batch-generator: 2.09
 iteration      460/    1000 | consumed samples:         3680 | elapsed time per iteration (ms): 213.4 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.524807E+00 | loss scale: 65536.0 | grad norm: 0.649 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.32 | backward-compute: 129.27 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.12 | optimizer-unscale-and-check-inf: 2.01 | optimizer-clip-main-grad: 2.58 | optimizer-copy-main-to-model-params: 2.47 | optimizer: 14.92 | batch-generator: 2.22
 iteration      470/    1000 | consumed samples:         3760 | elapsed time per iteration (ms): 213.3 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.509077E+00 | loss scale: 65536.0 | grad norm: 0.617 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.78 | backward-compute: 132.19 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.05 | optimizer-unscale-and-check-inf: 1.92 | optimizer-clip-main-grad: 2.40 | optimizer-copy-main-to-model-params: 2.45 | optimizer: 14.50 | batch-generator: 1.71
 iteration      480/    1000 | consumed samples:         3840 | elapsed time per iteration (ms): 215.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.516943E+00 | loss scale: 65536.0 | grad norm: 0.596 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 65.34 | backward-compute: 134.10 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 5.00 | optimizer-unscale-and-check-inf: 1.79 | optimizer-clip-main-grad: 2.41 | optimizer-copy-main-to-model-params: 2.50 | optimizer: 14.37 | batch-generator: 2.12
 iteration      490/    1000 | consumed samples:         3920 | elapsed time per iteration (ms): 223.8 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.487343E+00 | loss scale: 65536.0 | grad norm: 0.622 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 72.52 | backward-compute: 134.40 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.87 | optimizer-unscale-and-check-inf: 2.37 | optimizer-clip-main-grad: 2.91 | optimizer-copy-main-to-model-params: 2.43 | optimizer: 15.21 | batch-generator: 2.55
 iteration      500/    1000 | consumed samples:         4000 | elapsed time per iteration (ms): 225.4 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.547549E+00 | loss scale: 65536.0 | grad norm: 0.610 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 68.22 | backward-compute: 139.69 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.70 | optimizer-unscale-and-check-inf: 1.67 | optimizer-clip-main-grad: 2.61 | optimizer-copy-main-to-model-params: 2.62 | optimizer: 15.40 | batch-generator: 3.66
-----------------------------------------------------------------------------------------------
 validation loss at iteration 500 | lm loss value: 6.763964E+00 | lm loss PPL: 8.660686E+02 | 
-----------------------------------------------------------------------------------------------
saving checkpoint at iteration     500 to checkpoints/gpt2_345m
  successfully saved checkpoint at iteration     500 to checkpoints/gpt2_345m
time (ms) | save-checkpoint: 5743.17
 iteration      510/    1000 | consumed samples:         4080 | elapsed time per iteration (ms): 872.0 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.522501E+00 | loss scale: 65536.0 | grad norm: 0.619 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 132.46 | backward-compute: 147.89 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 5.05 | optimizer-unscale-and-check-inf: 2.08 | optimizer-clip-main-grad: 2.55 | optimizer-copy-main-to-model-params: 2.57 | optimizer: 14.99 | batch-generator: 4.20
 iteration      520/    1000 | consumed samples:         4160 | elapsed time per iteration (ms): 222.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.509373E+00 | loss scale: 65536.0 | grad norm: 0.664 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.61 | backward-compute: 139.86 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 4.67 | optimizer-unscale-and-check-inf: 3.71 | optimizer-clip-main-grad: 2.63 | optimizer-copy-main-to-model-params: 2.51 | optimizer: 16.13 | batch-generator: 2.87
 iteration      530/    1000 | consumed samples:         4240 | elapsed time per iteration (ms): 243.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.504818E+00 | loss scale: 65536.0 | grad norm: 0.617 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 79.36 | backward-compute: 145.10 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.11 | optimizer-unscale-and-check-inf: 3.53 | optimizer-clip-main-grad: 3.05 | optimizer-copy-main-to-model-params: 2.56 | optimizer: 17.03 | batch-generator: 4.34
 iteration      540/    1000 | consumed samples:         4320 | elapsed time per iteration (ms): 230.0 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.529472E+00 | loss scale: 65536.0 | grad norm: 0.642 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 72.33 | backward-compute: 138.56 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.36 | optimizer-unscale-and-check-inf: 3.62 | optimizer-clip-main-grad: 2.93 | optimizer-copy-main-to-model-params: 2.61 | optimizer: 17.25 | batch-generator: 3.33
 iteration      550/    1000 | consumed samples:         4400 | elapsed time per iteration (ms): 227.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.471246E+00 | loss scale: 65536.0 | grad norm: 0.593 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 73.41 | backward-compute: 135.56 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.16 | optimizer-unscale-and-check-inf: 2.67 | optimizer-clip-main-grad: 2.74 | optimizer-copy-main-to-model-params: 2.80 | optimizer: 16.22 | batch-generator: 2.79
 iteration      560/    1000 | consumed samples:         4480 | elapsed time per iteration (ms): 234.8 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.502017E+00 | loss scale: 65536.0 | grad norm: 0.615 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 76.29 | backward-compute: 139.64 | backward-embedding-all-reduce: 0.06 | optimizer-copy-to-main-grad: 5.08 | optimizer-unscale-and-check-inf: 2.71 | optimizer-clip-main-grad: 3.00 | optimizer-copy-main-to-model-params: 2.78 | optimizer: 16.54 | batch-generator: 2.92
 iteration      570/    1000 | consumed samples:         4560 | elapsed time per iteration (ms): 239.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.512360E+00 | loss scale: 65536.0 | grad norm: 0.631 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 79.99 | backward-compute: 140.13 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.87 | optimizer-unscale-and-check-inf: 2.30 | optimizer-clip-main-grad: 2.97 | optimizer-copy-main-to-model-params: 2.90 | optimizer: 16.94 | batch-generator: 3.02
 iteration      580/    1000 | consumed samples:         4640 | elapsed time per iteration (ms): 255.9 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.503098E+00 | loss scale: 65536.0 | grad norm: 0.578 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 84.41 | backward-compute: 150.92 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.79 | optimizer-unscale-and-check-inf: 3.68 | optimizer-clip-main-grad: 2.82 | optimizer-copy-main-to-model-params: 3.01 | optimizer: 18.21 | batch-generator: 4.64
 iteration      590/    1000 | consumed samples:         4720 | elapsed time per iteration (ms): 233.6 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.527862E+00 | loss scale: 65536.0 | grad norm: 0.585 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 76.05 | backward-compute: 139.84 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 5.02 | optimizer-unscale-and-check-inf: 3.01 | optimizer-clip-main-grad: 2.96 | optimizer-copy-main-to-model-params: 2.43 | optimizer: 16.05 | batch-generator: 3.35
 iteration      600/    1000 | consumed samples:         4800 | elapsed time per iteration (ms): 232.4 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.491470E+00 | loss scale: 65536.0 | grad norm: 0.648 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 71.36 | backward-compute: 142.82 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.41 | optimizer-unscale-and-check-inf: 2.49 | optimizer-clip-main-grad: 2.72 | optimizer-copy-main-to-model-params: 2.67 | optimizer: 16.18 | batch-generator: 2.25
-----------------------------------------------------------------------------------------------
 validation loss at iteration 600 | lm loss value: 6.787935E+00 | lm loss PPL: 8.870797E+02 | 
-----------------------------------------------------------------------------------------------
 iteration      610/    1000 | consumed samples:         4880 | elapsed time per iteration (ms): 306.0 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.484891E+00 | loss scale: 65536.0 | grad norm: 0.628 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 145.15 | backward-compute: 140.91 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.14 | optimizer-unscale-and-check-inf: 4.35 | optimizer-clip-main-grad: 2.46 | optimizer-copy-main-to-model-params: 2.58 | optimizer: 17.29 | batch-generator: 6.33
 iteration      620/    1000 | consumed samples:         4960 | elapsed time per iteration (ms): 228.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.514090E+00 | loss scale: 65536.0 | grad norm: 0.603 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 72.59 | backward-compute: 137.39 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.36 | optimizer-unscale-and-check-inf: 3.08 | optimizer-clip-main-grad: 2.82 | optimizer-copy-main-to-model-params: 2.60 | optimizer: 16.68 | batch-generator: 2.49
 iteration      630/    1000 | consumed samples:         5040 | elapsed time per iteration (ms): 218.8 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.501205E+00 | loss scale: 65536.0 | grad norm: 0.582 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 68.57 | backward-compute: 133.76 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.46 | optimizer-unscale-and-check-inf: 2.80 | optimizer-clip-main-grad: 2.44 | optimizer-copy-main-to-model-params: 2.47 | optimizer: 14.96 | batch-generator: 2.93
 iteration      640/    1000 | consumed samples:         5120 | elapsed time per iteration (ms): 244.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.509297E+00 | loss scale: 65536.0 | grad norm: 0.598 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 74.58 | backward-compute: 152.13 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.08 | optimizer-unscale-and-check-inf: 2.98 | optimizer-clip-main-grad: 2.76 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 16.02 | batch-generator: 2.56
 iteration      650/    1000 | consumed samples:         5200 | elapsed time per iteration (ms): 223.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.522154E+00 | loss scale: 65536.0 | grad norm: 0.595 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 62.54 | backward-compute: 144.62 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.14 | optimizer-unscale-and-check-inf: 1.83 | optimizer-clip-main-grad: 2.35 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 14.53 | batch-generator: 2.13
 iteration      660/    1000 | consumed samples:         5280 | elapsed time per iteration (ms): 220.9 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.519691E+00 | loss scale: 65536.0 | grad norm: 0.620 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 66.70 | backward-compute: 137.31 | backward-embedding-all-reduce: 0.05 | optimizer-copy-to-main-grad: 4.85 | optimizer-unscale-and-check-inf: 2.51 | optimizer-clip-main-grad: 2.56 | optimizer-copy-main-to-model-params: 2.47 | optimizer: 15.14 | batch-generator: 2.77
 iteration      670/    1000 | consumed samples:         5360 | elapsed time per iteration (ms): 230.0 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.529554E+00 | loss scale: 65536.0 | grad norm: 0.636 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 71.73 | backward-compute: 140.32 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.81 | optimizer-unscale-and-check-inf: 3.13 | optimizer-clip-main-grad: 2.89 | optimizer-copy-main-to-model-params: 2.54 | optimizer: 16.10 | batch-generator: 3.92
 iteration      680/    1000 | consumed samples:         5440 | elapsed time per iteration (ms): 231.4 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.507520E+00 | loss scale: 65536.0 | grad norm: 0.593 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 71.19 | backward-compute: 142.42 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.93 | optimizer-unscale-and-check-inf: 3.04 | optimizer-clip-main-grad: 2.63 | optimizer-copy-main-to-model-params: 2.49 | optimizer: 15.90 | batch-generator: 2.32
 iteration      690/    1000 | consumed samples:         5520 | elapsed time per iteration (ms): 237.4 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.479575E+00 | loss scale: 65536.0 | grad norm: 0.600 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 72.01 | backward-compute: 146.95 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.21 | optimizer-unscale-and-check-inf: 2.95 | optimizer-clip-main-grad: 2.78 | optimizer-copy-main-to-model-params: 2.71 | optimizer: 16.49 | batch-generator: 2.37
 iteration      700/    1000 | consumed samples:         5600 | elapsed time per iteration (ms): 217.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.457946E+00 | loss scale: 65536.0 | grad norm: 0.596 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 66.75 | backward-compute: 133.90 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.88 | optimizer-unscale-and-check-inf: 2.18 | optimizer-clip-main-grad: 2.36 | optimizer-copy-main-to-model-params: 2.48 | optimizer: 14.67 | batch-generator: 2.02
-----------------------------------------------------------------------------------------------
 validation loss at iteration 700 | lm loss value: 6.769063E+00 | lm loss PPL: 8.704963E+02 | 
-----------------------------------------------------------------------------------------------
 iteration      710/    1000 | consumed samples:         5680 | elapsed time per iteration (ms): 289.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.499170E+00 | loss scale: 65536.0 | grad norm: 0.693 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 127.53 | backward-compute: 143.53 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.98 | optimizer-unscale-and-check-inf: 2.88 | optimizer-clip-main-grad: 2.61 | optimizer-copy-main-to-model-params: 2.48 | optimizer: 15.73 | batch-generator: 4.16
 iteration      720/    1000 | consumed samples:         5760 | elapsed time per iteration (ms): 235.4 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.530855E+00 | loss scale: 65536.0 | grad norm: 0.622 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 79.19 | backward-compute: 136.64 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.36 | optimizer-unscale-and-check-inf: 3.69 | optimizer-clip-main-grad: 3.03 | optimizer-copy-main-to-model-params: 2.65 | optimizer: 17.58 | batch-generator: 3.56
 iteration      730/    1000 | consumed samples:         5840 | elapsed time per iteration (ms): 221.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.553014E+00 | loss scale: 65536.0 | grad norm: 0.649 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.43 | backward-compute: 136.61 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.03 | optimizer-unscale-and-check-inf: 3.75 | optimizer-clip-main-grad: 2.46 | optimizer-copy-main-to-model-params: 2.43 | optimizer: 15.46 | batch-generator: 2.40
 iteration      740/    1000 | consumed samples:         5920 | elapsed time per iteration (ms): 232.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.515872E+00 | loss scale: 65536.0 | grad norm: 0.578 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 77.36 | backward-compute: 136.51 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.37 | optimizer-unscale-and-check-inf: 4.17 | optimizer-clip-main-grad: 2.76 | optimizer-copy-main-to-model-params: 2.44 | optimizer: 16.52 | batch-generator: 3.30
 iteration      750/    1000 | consumed samples:         6000 | elapsed time per iteration (ms): 231.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.508788E+00 | loss scale: 65536.0 | grad norm: 0.601 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 69.48 | backward-compute: 143.51 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.92 | optimizer-unscale-and-check-inf: 3.33 | optimizer-clip-main-grad: 2.90 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 16.34 | batch-generator: 2.74
 iteration      760/    1000 | consumed samples:         6080 | elapsed time per iteration (ms): 211.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.548122E+00 | loss scale: 65536.0 | grad norm: 0.584 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 66.67 | backward-compute: 127.41 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.90 | optimizer-unscale-and-check-inf: 2.57 | optimizer-clip-main-grad: 2.71 | optimizer-copy-main-to-model-params: 2.44 | optimizer: 15.35 | batch-generator: 2.79
 iteration      770/    1000 | consumed samples:         6160 | elapsed time per iteration (ms): 231.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.528199E+00 | loss scale: 65536.0 | grad norm: 0.604 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.55 | backward-compute: 146.65 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.08 | optimizer-unscale-and-check-inf: 2.22 | optimizer-clip-main-grad: 2.69 | optimizer-copy-main-to-model-params: 2.74 | optimizer: 15.58 | batch-generator: 1.94
 iteration      780/    1000 | consumed samples:         6240 | elapsed time per iteration (ms): 232.3 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.535383E+00 | loss scale: 65536.0 | grad norm: 0.599 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 74.10 | backward-compute: 140.54 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.81 | optimizer-unscale-and-check-inf: 3.13 | optimizer-clip-main-grad: 2.76 | optimizer-copy-main-to-model-params: 2.47 | optimizer: 15.95 | batch-generator: 2.40
 iteration      790/    1000 | consumed samples:         6320 | elapsed time per iteration (ms): 228.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.488659E+00 | loss scale: 65536.0 | grad norm: 0.581 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 72.56 | backward-compute: 136.73 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.85 | optimizer-unscale-and-check-inf: 4.19 | optimizer-clip-main-grad: 2.89 | optimizer-copy-main-to-model-params: 2.45 | optimizer: 17.12 | batch-generator: 3.51
 iteration      800/    1000 | consumed samples:         6400 | elapsed time per iteration (ms): 222.5 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.524655E+00 | loss scale: 65536.0 | grad norm: 0.582 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.41 | backward-compute: 141.33 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.97 | optimizer-unscale-and-check-inf: 2.37 | optimizer-clip-main-grad: 2.41 | optimizer-copy-main-to-model-params: 2.48 | optimizer: 15.00 | batch-generator: 3.35
-----------------------------------------------------------------------------------------------
 validation loss at iteration 800 | lm loss value: 6.752643E+00 | lm loss PPL: 8.563191E+02 | 
-----------------------------------------------------------------------------------------------
 iteration      810/    1000 | consumed samples:         6480 | elapsed time per iteration (ms): 288.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.534463E+00 | loss scale: 65536.0 | grad norm: 0.630 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 131.82 | backward-compute: 137.34 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.26 | optimizer-unscale-and-check-inf: 2.99 | optimizer-clip-main-grad: 2.86 | optimizer-copy-main-to-model-params: 2.54 | optimizer: 16.52 | batch-generator: 5.32
 iteration      820/    1000 | consumed samples:         6560 | elapsed time per iteration (ms): 226.2 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.547289E+00 | loss scale: 65536.0 | grad norm: 0.576 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 71.85 | backward-compute: 137.76 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.01 | optimizer-unscale-and-check-inf: 2.10 | optimizer-clip-main-grad: 2.41 | optimizer-copy-main-to-model-params: 2.45 | optimizer: 14.76 | batch-generator: 4.93
 iteration      830/    1000 | consumed samples:         6640 | elapsed time per iteration (ms): 221.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.519830E+00 | loss scale: 65536.0 | grad norm: 0.588 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 68.31 | backward-compute: 135.82 | backward-embedding-all-reduce: 0.04 | optimizer-copy-to-main-grad: 5.01 | optimizer-unscale-and-check-inf: 2.53 | optimizer-clip-main-grad: 2.82 | optimizer-copy-main-to-model-params: 2.49 | optimizer: 15.66 | batch-generator: 3.74
 iteration      840/    1000 | consumed samples:         6720 | elapsed time per iteration (ms): 229.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.521774E+00 | loss scale: 65536.0 | grad norm: 0.620 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 73.88 | backward-compute: 138.38 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.04 | optimizer-unscale-and-check-inf: 1.74 | optimizer-clip-main-grad: 2.51 | optimizer-copy-main-to-model-params: 2.65 | optimizer: 14.88 | batch-generator: 3.08
 iteration      850/    1000 | consumed samples:         6800 | elapsed time per iteration (ms): 212.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.523515E+00 | loss scale: 65536.0 | grad norm: 0.624 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 63.57 | backward-compute: 133.21 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.99 | optimizer-unscale-and-check-inf: 1.64 | optimizer-clip-main-grad: 2.33 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 14.17 | batch-generator: 2.17
 iteration      860/    1000 | consumed samples:         6880 | elapsed time per iteration (ms): 213.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.476500E+00 | loss scale: 65536.0 | grad norm: 0.606 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 62.64 | backward-compute: 135.12 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.88 | optimizer-unscale-and-check-inf: 1.72 | optimizer-clip-main-grad: 2.31 | optimizer-copy-main-to-model-params: 2.47 | optimizer: 14.15 | batch-generator: 1.77
 iteration      870/    1000 | consumed samples:         6960 | elapsed time per iteration (ms): 221.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.514040E+00 | loss scale: 65536.0 | grad norm: 0.609 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 70.14 | backward-compute: 134.20 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.05 | optimizer-unscale-and-check-inf: 2.49 | optimizer-clip-main-grad: 2.75 | optimizer-copy-main-to-model-params: 2.49 | optimizer: 15.55 | batch-generator: 2.58
 iteration      880/    1000 | consumed samples:         7040 | elapsed time per iteration (ms): 220.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.503362E+00 | loss scale: 65536.0 | grad norm: 0.589 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 63.77 | backward-compute: 140.37 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 4.76 | optimizer-unscale-and-check-inf: 2.48 | optimizer-clip-main-grad: 2.52 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 14.90 | batch-generator: 2.01
 iteration      890/    1000 | consumed samples:         7120 | elapsed time per iteration (ms): 237.8 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.472827E+00 | loss scale: 65536.0 | grad norm: 0.639 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 75.39 | backward-compute: 144.88 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.76 | optimizer-unscale-and-check-inf: 2.77 | optimizer-clip-main-grad: 2.54 | optimizer-copy-main-to-model-params: 2.65 | optimizer: 15.65 | batch-generator: 2.50
 iteration      900/    1000 | consumed samples:         7200 | elapsed time per iteration (ms): 213.9 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.486347E+00 | loss scale: 65536.0 | grad norm: 0.596 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 62.05 | backward-compute: 134.82 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.22 | optimizer-unscale-and-check-inf: 3.44 | optimizer-clip-main-grad: 2.64 | optimizer-copy-main-to-model-params: 2.42 | optimizer: 15.52 | batch-generator: 3.00
-----------------------------------------------------------------------------------------------
 validation loss at iteration 900 | lm loss value: 6.769065E+00 | lm loss PPL: 8.704979E+02 | 
-----------------------------------------------------------------------------------------------
 iteration      910/    1000 | consumed samples:         7280 | elapsed time per iteration (ms): 288.3 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.529728E+00 | loss scale: 65536.0 | grad norm: 0.592 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 127.65 | backward-compute: 142.65 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.26 | optimizer-unscale-and-check-inf: 3.65 | optimizer-clip-main-grad: 2.79 | optimizer-copy-main-to-model-params: 2.41 | optimizer: 15.94 | batch-generator: 4.78
 iteration      920/    1000 | consumed samples:         7360 | elapsed time per iteration (ms): 218.8 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.488005E+00 | loss scale: 65536.0 | grad norm: 0.588 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.98 | backward-compute: 137.06 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.29 | optimizer-unscale-and-check-inf: 3.20 | optimizer-clip-main-grad: 2.47 | optimizer-copy-main-to-model-params: 2.41 | optimizer: 15.15 | batch-generator: 2.80
 iteration      930/    1000 | consumed samples:         7440 | elapsed time per iteration (ms): 228.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.510791E+00 | loss scale: 65536.0 | grad norm: 0.619 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.60 | backward-compute: 146.57 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.81 | optimizer-unscale-and-check-inf: 2.80 | optimizer-clip-main-grad: 2.48 | optimizer-copy-main-to-model-params: 2.44 | optimizer: 15.31 | batch-generator: 1.84
 iteration      940/    1000 | consumed samples:         7520 | elapsed time per iteration (ms): 211.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.539231E+00 | loss scale: 65536.0 | grad norm: 0.575 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 63.55 | backward-compute: 131.36 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 5.09 | optimizer-unscale-and-check-inf: 1.70 | optimizer-clip-main-grad: 2.52 | optimizer-copy-main-to-model-params: 2.45 | optimizer: 14.50 | batch-generator: 2.44
 iteration      950/    1000 | consumed samples:         7600 | elapsed time per iteration (ms): 214.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.469627E+00 | loss scale: 65536.0 | grad norm: 0.587 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 64.68 | backward-compute: 133.72 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.48 | optimizer-unscale-and-check-inf: 1.96 | optimizer-clip-main-grad: 2.44 | optimizer-copy-main-to-model-params: 2.44 | optimizer: 14.09 | batch-generator: 2.40
 iteration      960/    1000 | consumed samples:         7680 | elapsed time per iteration (ms): 223.7 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.537416E+00 | loss scale: 65536.0 | grad norm: 0.625 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 71.94 | backward-compute: 134.60 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.16 | optimizer-unscale-and-check-inf: 3.62 | optimizer-clip-main-grad: 2.73 | optimizer-copy-main-to-model-params: 2.40 | optimizer: 15.65 | batch-generator: 3.01
 iteration      970/    1000 | consumed samples:         7760 | elapsed time per iteration (ms): 224.0 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.480746E+00 | loss scale: 65536.0 | grad norm: 0.613 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 71.85 | backward-compute: 134.52 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.36 | optimizer-unscale-and-check-inf: 3.78 | optimizer-clip-main-grad: 2.71 | optimizer-copy-main-to-model-params: 2.45 | optimizer: 16.08 | batch-generator: 2.95
 iteration      980/    1000 | consumed samples:         7840 | elapsed time per iteration (ms): 217.1 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.523880E+00 | loss scale: 65536.0 | grad norm: 0.609 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.95 | backward-compute: 132.26 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.17 | optimizer-unscale-and-check-inf: 3.51 | optimizer-clip-main-grad: 2.59 | optimizer-copy-main-to-model-params: 2.46 | optimizer: 15.48 | batch-generator: 3.00
 iteration      990/    1000 | consumed samples:         7920 | elapsed time per iteration (ms): 217.0 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.502033E+00 | loss scale: 65536.0 | grad norm: 0.624 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 65.62 | backward-compute: 134.21 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.11 | optimizer-unscale-and-check-inf: 3.42 | optimizer-clip-main-grad: 3.05 | optimizer-copy-main-to-model-params: 2.38 | optimizer: 15.70 | batch-generator: 2.25
 iteration     1000/    1000 | consumed samples:         8000 | elapsed time per iteration (ms): 230.9 | learning rate: 0.000E+00 | global batch size:     8 | lm loss: 6.476170E+00 | loss scale: 65536.0 | grad norm: 0.584 | number of skipped iterations:   0 | number of nan iterations:   0 |
time (ms) | forward-compute: 67.75 | backward-compute: 145.95 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 4.10 | optimizer-unscale-and-check-inf: 4.00 | optimizer-clip-main-grad: 2.56 | optimizer-copy-main-to-model-params: 2.45 | optimizer: 15.83 | batch-generator: 3.14
------------------------------------------------------------------------------------------------
 validation loss at iteration 1000 | lm loss value: 6.758039E+00 | lm loss PPL: 8.609518E+02 | 
------------------------------------------------------------------------------------------------
saving checkpoint at iteration    1000 to checkpoints/gpt2_345m
  successfully saved checkpoint at iteration    1000 to checkpoints/gpt2_345m
time (ms) | save-checkpoint: 71189.35
[after training is done] datetime: 2021-10-24 01:56:13 
saving checkpoint at iteration    1000 to checkpoints/gpt2_345m
------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for val data | lm loss value: 6.744719E+00 | lm loss PPL: 8.495604E+02 | 
------------------------------------------------------------------------------------------------------------------
  successfully saved checkpoint at iteration    1000 to checkpoints/gpt2_345m
Evaluating iter 10/10
-------------------------------------------------------------------------------------------------------------------
 validation loss at the end of training for test data | lm loss value: 6.614303E+00 | lm loss PPL: 7.456849E+02 | 
-------------------------------------------------------------------------------------------------------------------
INFO:torch.distributed.elastic.agent.server.api:[default] worker group successfully finished. Waiting 300 seconds for other agents to finish.
INFO:torch.distributed.elastic.agent.server.api:Local worker group finished (SUCCEEDED). Waiting 300 seconds for other agents to finish
/opt/conda/lib/python3.7/site-packages/torch/distributed/elastic/utils/store.py:71: FutureWarning: This is an experimental API and will be changed in future.
  "This is an experimental API and will be changed in future.", FutureWarning
INFO:torch.distributed.elastic.agent.server.api:Done waiting for other agents. Elapsed: 0.0009801387786865234 seconds
{"name": "torchelastic.worker.status.SUCCEEDED", "source": "WORKER", "timestamp": 0, "metadata": {"run_id": "none", "global_rank": 0, "group_rank": 0, "worker_id": "18862", "role": "default", "hostname": "8632db213dbf", "state": "SUCCEEDED", "total_run_time": 570, "rdzv_backend": "static", "raw_error": null, "metadata": "{\"group_world_size\": 1, \"entry_point\": \"python\", \"local_rank\": [0], \"role_rank\": [0], \"role_world_size\": [4]}", "agent_restarts": 0}}
{"name": "torchelastic.worker.status.SUCCEEDED", "source": "WORKER", "timestamp": 0, "metadata": {"run_id": "none", "global_rank": 1, "group_rank": 0, "worker_id": "18863", "role": "default", "hostname": "8632db213dbf", "state": "SUCCEEDED", "total_run_time": 570, "rdzv_backend": "static", "raw_error": null, "metadata": "{\"group_world_size\": 1, \"entry_point\": \"python\", \"local_rank\": [1], \"role_rank\": [1], \"role_world_size\": [4]}", "agent_restarts": 0}}
{"name": "torchelastic.worker.status.SUCCEEDED", "source": "WORKER", "timestamp": 0, "metadata": {"run_id": "none", "global_rank": 2, "group_rank": 0, "worker_id": "18864", "role": "default", "hostname": "8632db213dbf", "state": "SUCCEEDED", "total_run_time": 570, "rdzv_backend": "static", "raw_error": null, "metadata": "{\"group_world_size\": 1, \"entry_point\": \"python\", \"local_rank\": [2], \"role_rank\": [2], \"role_world_size\": [4]}", "agent_restarts": 0}}
{"name": "torchelastic.worker.status.SUCCEEDED", "source": "WORKER", "timestamp": 0, "metadata": {"run_id": "none", "global_rank": 3, "group_rank": 0, "worker_id": "18868", "role": "default", "hostname": "8632db213dbf", "state": "SUCCEEDED", "total_run_time": 570, "rdzv_backend": "static", "raw_error": null, "metadata": "{\"group_world_size\": 1, \"entry_point\": \"python\", \"local_rank\": [3], \"role_rank\": [3], \"role_world_size\": [4]}", "agent_restarts": 0}}
{"name": "torchelastic.worker.status.SUCCEEDED", "source": "AGENT", "timestamp": 0, "metadata": {"run_id": "none", "global_rank": null, "group_rank": 0, "worker_id": null, "role": "default", "hostname": "8632db213dbf", "state": "SUCCEEDED", "total_run_time": 570, "rdzv_backend": "static", "raw_error": null, "metadata": "{\"group_world_size\": 1, \"entry_point\": \"python\"}", "agent_restarts": 0}}
```
```
[glogin01]$ cd ..
/home/ubuntu/kevin/jupyter/notebooks
```

## 3. Parallelformers\n",
![](../images/parallelformers.png)
지금까지 Megatron-LM으로 모델을 학습해봤습니다. Megatron-LM은 훌륭한 Tensor Parallelism 기능을 보유하고 있지만, 기존에 우리가 자주 쓰던 Hugging Face `transformers`로 학습된 모델을 병렬화 할 수는 없었습니다. 이러한 문제를 해결하기 위해 TUNiB은 2021년 `parallelformers`라는 오픈소스를 공개했습니다. `parallelformers`는 코드 한 두줄로 Hugging Face `transformers`로 학습된 거의 대부분의 모델에 Tensor Parallelism을 적용하여 인퍼런스 할 수 있는 도구 입니다.
`parallelformers`를 설치해봅시다.
```
[glogin01]$ pip install parallelformers
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
    )

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
[glogin01]$ python ../src/parallelformers_inference.py
Downloading: 100%|██████████████████████████| 1.42k/1.42k [00:00<00:00, 864kB/s]
Downloading: 100%|█████████████████████████| 9.94G/9.94G [04:34<00:00, 38.8MB/s]
Downloading: 100%|██████████████████████████████| 200/200 [00:00<00:00, 241kB/s]
Downloading: 100%|████████████████████████████| 779k/779k [00:01<00:00, 721kB/s]
Downloading: 100%|████████████████████████████| 446k/446k [00:01<00:00, 413kB/s]
Downloading: 100%|███████████████████████████| 90.0/90.0 [00:00<00:00, 66.3kB/s]
GPU 2 alloc: 1662117888
GPU 2 cached: 2051014656

GPU 1 alloc: 1662117888
GPU 1 cached: 2051014656

GPU 3 alloc: 1662117888
GPU 3 cached: 2051014656

GPU 0 alloc: 1662117888
GPU 0 cached: 2051014656

Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

Output: Parallelformers is an open-source library for parallel programming in Haskell
```

### Parallelformers의 동작 원리
 
![](../images/tensor_replace.png)

그렇다면 `parallelformers`는 어떻게 모델링 코드의 변화 없이 Tensor parallelism을 수행 할 수 있을까요? 정답은 `Tensor Replacement` 메커니즘에 있습니다. `parallelformers`는 기존 모델의 파라미터를 전부 추출한 뒤, Megatron-LM과 동일한 방식으로 텐서를 쪼개고 쪼개진 텐서로 원래 모델에 존재하던 파라미터를 교체함으로써 모델의 구조 변화 없이 병렬화를 수행할 수 있었습니다. 이를 통해 약 70여가지의 모델을 병렬화 할 수 있었습니다. 이외에도 몇가지 메커니즘이 도입되었지만 텐서 병렬화와 관계 있는 내용은 아니기 때문에 생략하도록 하겠습니다. 만약 더 자세한 내용이 궁금하시다면 다음 주소를 참고해주세요.
- 한국어: https://tunib.notion.site/TECH-2021-07-26-Parallelformers-_-0dcceeaddc5247429745ba36c6549fe5
- English: https://tunib.notion.site/TECH-2021-07-26-Parallelformers-Journey-to-deploying-big-models_TUNiB-32b19a599c38497abaad2a98727f6dc8
