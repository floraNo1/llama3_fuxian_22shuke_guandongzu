```
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt
```

设置分词器模型文件的路径

```
tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
```

定义一组特殊的token

```
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
```

加载BPE分词器的可合并排名，决定如何合并token

tiktoken.Encoding(分词器)，配置包括：

​	name:分词器模型文件名称；

​	pat_str:匹配文本中不同模式的正则表达式，以便分词器可以正确将文本分割成token。分词器的核心部分，决定如何将文本拆分成更小的单元。

`(?i:'s|'t|'re|'ve|'m|'ll|'d)`：

​	`(?i:)`：忽略大小写匹配。

- `|'s|'t|'re|'ve|'m|'ll|'d`：匹配常见的英语缩写（如's, 't, 're等）。

`[^\r\n\p{L}\p{N}]?\p{L}+`：

- `[^\r\n\p{L}\p{N}]?`：匹配一个非换行符、非字母和非数字的字符（如果存在的话）。
- `\p{L}+`：匹配一个或多个字母。

`\p{N}{1,3}`：

- `\p{N}{1,3}`：匹配1到3个连续的数字。

`?[^\s\p{L}\p{N}]+[\r\n]*`：

- `?`：匹配一个空格（如果存在的话）。
- `[^\s\p{L}\p{N}]+`：匹配一个或多个非空格、非字母和非数字的字符。
- `[\r\n]*`：匹配零个或多个换行符。

`\s*[\r\n]+`：

- `\s*`：匹配零个或多个空白字符。
- `[\r\n]+`：匹配一个或多个换行符。

`\s+(?!\S)`：

- `\s+`：匹配一个或多个空白字符。
- `(?!\S)`：负向前瞻，确保后面不是非空白字符。

`\s+`：

- `\s+`：匹配一个或多个空白字符。

  ​	mergeable_ranks:可合并排名

  ​	special_tokens：特殊token及其对应的索引

```
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)
```

###### 以上主要是初始化BPE分词器，测试编码和解码功能。

从文件中加载模型，打印模型中前20个键，以json形式缩进4个空格。

model.key()返回模型中所有参数的键。

模型的前20个键代表了模型中的参数和层的名称，用于索引模型内部参数的字典键。每个键对应于模型中的一个特定的权重或偏置项。

```
model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
print(json.dumps(list(model.keys())[:20], indent=4))
```

**`tok_embeddings.weight`**:

- 词嵌入矩阵的权重，用于将输入词转换为嵌入向量。

**`layers.0.attention.wq.weight`**:

- 第一层中的注意力机制的查询（query）权重。

**`layers.0.attention.wk.weight`**:

- 第一层中的注意力机制的键（key）权重。

**`layers.0.attention.wv.weight`**:

- 第一层中的注意力机制的值（value）权重。

**`layers.0.attention.wo.weight`**:

- 第一层中的注意力机制输出（output）权重。

**`layers.0.feed_forward.w1.weight`**:

- 第一层中的前馈神经网络（feed-forward network, FFN）的第一个全连接层的权重。

**`layers.0.feed_forward.w3.weight`**:

- 第一层中的前馈神经网络的第三个全连接层的权重。

**`layers.0.feed_forward.w2.weight`**:

- 第一层中的前馈神经网络的第二个全连接层的权重。

**`layers.0.attention_norm.weight`**:

- 第一层中的注意力机制的层规范化（Layer Normalization）权重。

**`layers.0.ffn_norm.weight`**:

- 第一层中的前馈神经网络的层规范化权重。

**`layers.1.attention.wq.weight`**:

- 第二层中的注意力机制的查询（query）权重。

**`layers.1.attention.wk.weight`**:

- 第二层中的注意力机制的键（key）权重。

**`layers.1.attention.wv.weight`**:

- 第二层中的注意力机制的值（value）权重。

**`layers.1.attention.wo.weight`**:

- 第二层中的注意力机制输出（output）权重。

**`layers.1.feed_forward.w1.weight`**:

- 第二层中的前馈神经网络的第一个全连接层的权重。

**`layers.1.feed_forward.w3.weight`**:

- 第二层中的前馈神经网络的第三个全连接层的权重。

**`layers.1.feed_forward.w2.weight`**:

- 第二层中的前馈神经网络的第二个全连接层的权重。

**`layers.1.attention_norm.weight`**:

- 第二层中的注意力机制的层规范化权重。

**`layers.1.ffn_norm.weight`**:

- 第二层中的前馈神经网络的层规范化权重。

**`layers.2.attention.wq.weight`**:

- 第三层中的注意力机制的查询（query）权重。



读取json配置文件并将其内容加载到变量‘config'中

```
with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)
config
```

```
{
    'dim': 4096,
    'n_layers': 32,
    'n_heads': 32,
    'n_kv_heads': 8,
    'vocab_size': 128256,
    'multiple_of': 1024,
    'ffn_dim_multiplier': 1.3,
    'norm_eps': 1e-05,
    'rope_theta': 500000.0
}
```

**`dim`: 4096**

- 模型的隐藏层维度，决定了每一层中的特征数量。

**`n_layers`: 32**

- 模型的层数，表示Transformer块的数量。

**`n_heads`: 32**

- 多头注意力机制的头数，每层有32个注意力头。

**`n_kv_heads`: 8**

- 键（Key）和值（Value）注意力头的数量。

**`vocab_size`: 128256**

- 词汇表的大小，即模型可以处理的不同词的数量。

**`multiple_of`: 1024**

- 前馈网络（FFN）的维度必须是此值的倍数，确保计算上的一致性和效率。

**`ffn_dim_multiplier`: 1.3**

- 前馈网络的维度乘数，用于确定前馈网络中的隐藏层大小。

**`norm_eps`: 1e-05**

- 层规范化中的epsilon值，用于数值稳定性，防止除零错误。

**`rope_theta`: 500000.0**

- 相对位置编码中的一个参数，用于调整位置编码的尺度。

```
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])
```



如何将一个提示文本（‘prompt’）编码为tokens

```
prompt = "the answer to the ultimate question of life, the universe, and everything is "
#定义一个文本提示，将特殊token与编码后的tokens合并
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
#将tokens转换为pytorch张量
tokens = torch.tensor(tokens)
#将每个token解码为对应的文本，token.item()将张量中的token值转换为python整数
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)
```



加载预训练模型的词嵌入权重，并使用这些权重初始化一个嵌入层。将tokens转换为嵌入向量。

```
embedding_layer = torch.nn.Embedding(vocab_size, dim)
# 将预训练的词嵌入权重复制到嵌入层
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
# 将tokens转换为未归一化的嵌入向量，并转换为bfloat16
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
token_embeddings_unnormalized.shape
```



```
# def rms_norm(tensor, norm_weights):
#     rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5
#     return tensor * (norm_weights / rms)
def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights
```

对张量进行归一化处理。

tensor形状[batch_size, embedding_dim]

norm_weights形状[embedding_dim]



```
# 应用RMS归一化
token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])
token_embeddings.shape
```



# 每个头单独设置指每个注意力头有自己的查询，值和键向量

### 多头注意力机制中的查询、键、值

假设一个注意力层有h个头，每个头的维度为d_k，

查询向量：Q=XW_Q

键向量：K=XW_K

值向量：V=XW_V



```
# 获取第0层的查询权重
q_layer0 = model["layers.0.attention.wq.weight"]
# 计算每个头的维度
head_dim = q_layer0.shape[0] // n_heads
# 将查询权重重塑为 [n_heads, head_dim, dim] 形状
q_layer0 = q_layer0.view(n_heads, head_dim, dim)
q_layer0.shape
```

```
# 从已经重塑的查询权重矩阵中提取第一个注意力头的查询权重
q_layer0_head0 = q_layer0[0]
q_layer0_head0.shape
```

```
# 计算每个token的查询向量
q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)
q_per_token.shape
```

`torch.matmul` 进行矩阵乘法，将 `token_embeddings` 与 `q_layer0_head0.T`（转置后的查询权重）相乘。

```
# 将查询向量转换为浮点型，并拆分成对
q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
q_per_token_split_into_pairs.shape
```

`view(q_per_token.shape[0], -1, 2)` 将 `q_per_token` 的形状从 `[17, 128]` 转换为 `[17, 64, 2]`。

这里，`q_per_token.shape[0]` 保持批量大小不变，`-1` 表示自动计算，`2` 表示每个查询向量被拆分成对。

为旋转位置嵌入（RoPE）做准备

```
# 生成一个从0到63的整数序列，然后除以64，得到从0到1分成64部分的张量
zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64
zero_to_one_split_into_64_parts
```

```
# 计算频率张量
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
```

该张量将用于RoPE操作。

**定义 `rope_theta`**：

- `rope_theta` 是一个用于计算频率的常数。
- 在这个例子中，我们使用 `rope_theta = 10000.0` 作为示例值。

**计算频率张量**：

- `freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)` 使用 `rope_theta` 和 `zero_to_one_split_into_64_parts` 计算频率张量。
- `zero_to_one_split_into_64_parts` 是一个从0到1分成64部分的张量。
- `rope_theta ** zero_to_one_split_into_64_parts` 计算每个部分的幂值，然后取其倒数得到频率。

```
# 生成每个token的频率张量
freqs_for_each_token = torch.outer(torch.arange(17), freqs)

# 生成频率复数向量
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

# 打印频率复数向量的形状
print(freqs_cis.shape)

# 查看freqs_cis的第三行
value = freqs_cis[3]

# 绘制第三行的复数向量
plt.figure()
for i, element in enumerate(value[:17]):
    plt.plot([0, element.real], [0, element.imag], color='blue', linewidth=1)
    plt.annotate(f"{i}", xy=(element.real, element.imag), color='red')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Plot of one row of freqs_cis')
plt.show()
```

```
# 将查询向量转换为复数形式
q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
# 使用频率复数向量对查询向量进行旋转
q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis
# 将旋转后的复数查询向量转换回实数对
q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)
# 打印转换回实数对后的查询向量的形状
print(q_per_token_split_into_pairs_rotated.shape)
# 将查询向量的形状还原为原始形状
q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
```

# RoPE（Rotary Position Embedding) 旋转位置嵌入

**改进相对位置编码**：传统的 Transformer 使用绝对位置编码，而 RoPE 提供了一种将位置编码到查询（query）和键（key）向量中的方法，使得模型能够更好地处理相对位置。

**旋转操作**：RoPE 通过旋转操作将位置信息编码到查询和键向量中，从而增强了模型对序列中相对位置信息的建模能力。

### RoPE 的计算步骤

**生成频率张量**：计算每个位置的频率。

**生成频率复数向量**：将频率转换为复数形式。

**将查询向量转换为复数**：将查询向量的实部和虚部转换为复数。

**旋转查询向量**：通过与频率复数向量相乘，实现对查询向量的旋转。

**转换回实数对**：将旋转后的复数查询向量转换回实数对，并还原其形状。

### 频率复数向量的作用

1. **引入位置信息**：
   - 频率复数向量用于将位置信息编码到查询和键向量中，使得模型能够区分序列中不同位置的元素。
   - 这种方法通过在复数域中旋转向量，实现了相对位置编码。
2. **旋转操作**：
   - 频率复数向量定义了每个位置的旋转角度。通过将查询和键向量转换为复数，然后与频率复数向量相乘，实现了向量的旋转。
   - 这种旋转操作增强了模型对序列中相对位置关系的建模能力。

### 查询向量的作用

查询向量在多头自注意力机制中用于计算注意力权重，从而决定模型在处理序列数据时对哪些位置的关注度更高。RoPE通过旋转操作将位置信息编码到查询向量中，使得模型能够更好地捕捉序列中元素的相对位置关系。



# 键向量

```
# 1. 提取并重塑第0层的键权重
k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)
# 2. 提取第一个注意力头的键权重
k_layer0_head0 = k_layer0[0]
# 3. 计算每个token的键向量
k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)
# 4. 将键向量拆分成对
k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
# 5. 将键向量转换为复数形式
k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
# 6. 旋转键向量
k_per_token_as_complex_numbers_rotated = k_per_token_as_complex_numbers * freqs_cis
# 7. 将旋转后的复数键向量转换回实数对
k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers_rotated)
# 8. 将键向量的形状还原为原始形状
k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
```

计算查询向量与键向量的点击，并对结果进行缩放以获得自注意力机制的评分矩阵。

```
# 计算注意力评分矩阵，并进行缩放
qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (head_dim ** 0.5)
```

这个评分矩阵表示每个token与其他token之间的相关性，并且通过缩放因子缓解了梯度爆炸问题。



# 对未来token的qk评分屏蔽

在自注意力机制中，为了保证模型在训练和推理阶段只能看到过去的信息，我们需要屏蔽未来的token。这种屏蔽机制确保模型不会泄露未来的信息，从而使得模型只能基于已经看到的token进行预测。

### 屏蔽机制的实现步骤

1. **生成上三角矩阵**：创建一个用于屏蔽未来token的上三角矩阵。
2. **应用屏蔽矩阵**：将上三角矩阵应用到注意力评分矩阵上，将未来的token评分设为负无穷大（或零）。

```
# 定义显示热图的函数
def display_qk_heatmap(qk_per_token):
	# 创建一个图和轴
    _, ax = plt.subplots()
    # 显示QK评分矩阵的热图
    im = ax.imshow(qk_per_token.to(float).detach(), cmap='viridis')
    # 设置轴的刻度和标签
    ax.set_xticks(range(len(prompt_split_as_tokens)))
    ax.set_yticks(range(len(prompt_split_as_tokens)))
    ax.set_xticklabels(prompt_split_as_tokens)
    ax.set_yticklabels(prompt_split_as_tokens)
    ax.figure.colorbar(im, ax=ax)
    
display_qk_heatmap(qk_per_token)
```

```
# 生成上三角矩阵
# 生成一个形状为 (num_tokens, num_tokens) 的矩阵，初始化为负无穷大
mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
# 将矩阵的上三角部分（不包括对角线）设为负无穷大
mask = torch.triu(mask, diagonal=1)

# 将屏蔽矩阵应用到QK评分矩阵上并生成热图显示应用屏蔽后的QK评分矩阵
qk_per_token_after_masking = qk_per_token + mask
display_qk_heatmap(qk_per_token_after_masking)

# 对处理后的QK评分矩阵进行softmax操作【归一化的注意力权重，这些权重表示每个 token 对其他 token 的相对重要性】
qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
display_qk_heatmap(qk_per_token_after_masking_after_softmax)
```



在自注意力机制中，注意力权重（通过Softmax获得的分数）用于加权value矩阵，以计算每个token的注意力输出。为了节省计算资源，value矩阵的权重在多个注意力头之间共享。

### 具体步骤

1. **定义value权重矩阵**：定义一个在多个注意力头之间共享的value权重矩阵。
2. **计算value向量**：使用计算得到的注意力权重和value矩阵计算每个token的注意力输出。

```
# 从模型中提取value权重矩阵
v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)
# 提取第一个注意力头的value权重
v_layer0_head0 = v_layer0[0]
```

# value vectors

```
# 计算每个 token 的 value 向量，大小为 [17x128]
v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)
```

# attention

```
# 使用归一化后的注意力权重矩阵和 value 矩阵计算每个 token 的最终注意力输出
qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
```

# multi head attention

```
# 存储注意力输出的列表
qkv_attention_store = []
# 遍历每个注意力头
for head in range(n_heads):
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head//4] # key weights are shared across 4 heads	# key 权重在每4个头之间共享
    v_layer0_head = v_layer0[head//4] # value weights are shared across 4 heads	# value 权重在每4个头之间共享
    # 计算query，key和value向量
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)
	# 对 query 和 key 向量进行旋转位置编码
    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
    k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
	# 计算 QK 评分矩阵并应用屏蔽和 softmax
    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
    mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)

len(qkv_attention_store)
```

# weight matrix

```
# 提取第0层注意力模块的输出权重（wo）
w_layer0 = model["layers.0.attention.wo.weight"]

# 将多个注意力头的输出堆叠在一起
stacked_qkv_attention = torch.stack(qkv_attention_store, dim=0).permute(1, 0, 2).contiguous().view(len(tokens), -1)

# 计算嵌入向量的线性变换
embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)

# 将线性变换的结果应用到原始嵌入向量上
embedding_after_edit = token_embeddings_unnormalized + embedding_delta

# 对结果进行归一化处理
embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])

# 打印最终嵌入向量的形状
print("最终嵌入向量的形状:", embedding_after_edit_normalized.shape)
```

# 加载前馈神经网络权重并实现

SwiGLU

```
# 从模型中加载前馈网络的权重
w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"]
w3 = model["layers.0.feed_forward.w3.weight"]
# 使用SwiGLU前馈网络计算输出
output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
output_after_feedforward.shape
```



# Llama3

**嵌入层（Embedding Layer）**：

- 将输入的token序列转换为嵌入向量。
- 通常包含词汇表嵌入和位置嵌入。

**多头自注意力层（Multi-Head Self-Attention Layer）**：

- 包含多个注意力头，每个头独立计算注意力并生成输出。
- 注意力头的输出通常会进行线性变换并连接在一起。

**前馈网络层（Feed-Forward Network Layer）**：

- 通常由两个线性变换和一个非线性激活函数（如ReLU或GELU）组成。
- 在LLaMA3中使用的是SwiGLU前馈网络。

**归一化层（Normalization Layer）**：

- 在每个注意力层和前馈网络层后面进行归一化处理，通常使用的是RMSNorm或LayerNorm。

**输出层（Output Layer）**：

- 将最终的嵌入向量映射回词汇表的维度，以预测下一个token。

