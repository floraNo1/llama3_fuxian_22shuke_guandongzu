基于PaddlePaddle复现Llama3



准备工作：两种实现tokenizer的方式

```
# 使用llama官方tokenizer
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import json
import paddle
import numpy as np
import matplotlib.pyplot as plt
import pickle
from paddlenlp.transformers import AutoModelForCausalLM
```

```
# 定义分词器
tokenizer_path = "Meta-Llama-3-8B-Instruct-torch/tokenizer.model"
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
# 加载合并标词等级
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
# 创建编码器
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)
# Test
tokenizer.decode(tokenizer.encode("hello world!"))
```


```
#Andrej Karpathy https://github.com/karpathy/minbpe
from pathlib import Path
import torch
import tiktoken
import json
import matplotlib.pyplot as plt
from tiktoken.load import load_tiktoken_bpe
from tokenier.tokenier import GPT4Tokenizer 

tokenizer = GPT4Tokenizer()


# 编码和解码示例
input_text = "how's the whether ?"
encoded_input = tokenizer.encode(input_text)
decoded_output = tokenizer.decode(encoded_input)

print(f"Encoded input: {encoded_input}")
print(f"Decoded output: {decoded_output}")
```


```
# 加载预训练模型并转换格式为PaddlePaddle格式
AutoModelForCausalLM.from_pretrained("Meta-Llama-3-8B-Instruct", convert_from_torch=True, dtype="float16")
print(json.dumps(list(model.keys())[:20], indent=4))
```

准备工作：加载模型配置并提取相关参数

```
# 加载模型配置并提取相关参数
with open("Meta-Llama-3-8B-Instruct/params.json", "r") as f:
    config = json.load(f)
print(config)

dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
# 将rope_theta转换为PaddlePaddle张量
rope_theta = paddle.to_tensor(np.array(config["rope_theta"]))
```



## 嵌入向量化

```
# 定义提示词
prompt = "the answer to the ultimate question of life, the universe, and everything is "
# 编码提示词
tokens = [128000] + tokenizer.encode(prompt)

# 将 tokens 转换为 PaddlePaddle 张量
tokens = paddle.to_tensor(tokens, dtype='int64')

# 将 tokens 转换回 prompt 字符串
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens.numpy()]

```

```
# 定义嵌入层
embedding_layer = paddle.nn.Embedding(vocab_size, dim)

# 将模型的权重复制到嵌入层
embedding_layer.weight.set_value(paddle.to_tensor(model["tok_embeddings.weight"].numpy()))

# 获取未归一化的 token 嵌入
token_embeddings_unnormalized = embedding_layer(tokens).astype('bfloat16')

# 打印检查嵌入的形状
#print(token_embeddings_unnormalized.shape)
```



## RMSNorm

```
# 正则化层实现(Root Mean Square Layer Normalization)
class RMSNorm(paddle.nn.Layer):
    def __init__(self, d, eps=1e-8):
    # 调用父类'paddle.nn.Layer'的初始化方法
        super(RMSNorm, self).__init__()
        # eps：一个很小的常数，防止计算均方根时分母为零
        self.eps = eps
        # 创建可训练参数，'scale'为可缩放训练参数
        self.scale = self.create_parameter(
            shape=[d],
            default_initializer=paddle.nn.initializer.Constant(1.0)
        )

    def forward(self, x):
        norm_x = x * paddle.rsqrt(paddle.mean(x**2, axis=-1, keepdim=True) + self.eps)
        return self.scale * norm_x
        
# 假设d是embedding的维度
d = dim  # 使用之前定义的dim

# 初始化RMSNorm
rms_norm = RMSNorm(d)

# 将模型的权重复制到RMSNorm层
rms_norm.scale.set_value(paddle.to_tensor(model["layers.0.attention_norm.weight"].numpy()))

# 应用RMSNorm
token_embeddings = rms_norm(token_embeddings_unnormalized)

# 打印检查嵌入的形状
#print(token_embeddings.shape)
```





##  这部分放到2.5代码部分

```
# 获取权重并进行形状转换
q_layer0 = model["layers.0.attention.wq.weight"]

# 确定每个头的维度
head_dim = q_layer0.shape[0] // n_heads

# 重新调整权重的形状
q_layer0 = q_layer0.reshape([n_heads, head_dim, dim])

# 打印检查形状
# print(q_layer0.shape)
```

```
# 获取第一个头部的权重
q_layer0_head0 = q_layer0[0]

# 打印第一个头部的权重形状
#print(q_layer0_head0.shape)
```

```
# 计算 q_per_token（查询向量与输入标记嵌入的点积）
q_per_token = paddle.matmul(token_embeddings, q_layer0_head0.T)
#print(q_per_token.shape)
```

```
# 将 q_per_token 切分成对
q_per_token_split_into_pairs = q_per_token.astype('float32').reshape([q_per_token.shape[0], -1, 2])
#print(q_per_token_split_into_pairs.shape)
```

```
# 生成频率向量
zero_to_one_split_into_64_parts = paddle.to_tensor(np.arange(64) / 64, dtype='float32')
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
freqs_for_each_token = paddle.outer(paddle.arange(17), freqs)
freqs_cis = paddle.to_complex(freqs_for_each_token)
#print(freqs_cis.shape)
```

```
# 将 q_per_token 视为复数
q_per_token_as_complex_numbers = paddle.view_as_complex(q_per_token_split_into_pairs)
#print(q_per_token_as_complex_numbers.shape)

# 旋转 q_per_token
q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis
#print(q_per_token_as_complex_numbers_rotated.shape)

# 将旋转后的 q_per_token 切分成对
q_per_token_split_into_pairs_rotated = paddle.view_as_real(q_per_token_as_complex_numbers_rotated)
#print(q_per_token_split_into_pairs_rotated.shape)

# 还原 q_per_token
q_per_token_rotated = q_per_token_split_into_pairs_rotated.reshape(q_per_token.shape)
#print(q_per_token_rotated.shape)
```


```
# 处理 k 层权重
k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.reshape([n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim])
#print(k_layer0.shape)
```

```
# 获取第一个头部的 k 层权重
k_layer0_head0 = k_layer0[0]
#print(k_layer0_head0.shape)
```

```
# 计算 k_per_token
k_per_token = paddle.matmul(token_embeddings, k_layer0_head0.T)
#print(k_per_token.shape)
```

```
# 将 k_per_token 切分成对
k_per_token_split_into_pairs = k_per_token.astype('float32').reshape([k_per_token.shape[0], -1, 2])
#print(k_per_token_split_into_pairs.shape)
```

```
# 将 k_per_token 视为复数
k_per_token_as_complex_numbers = paddle.view_as_complex(k_per_token_split_into_pairs)
#print(k_per_token_as_complex_numbers.shape)
```

```
# 旋转 k_per_token
k_per_token_split_into_pairs_rotated = paddle.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
#print(k_per_token_split_into_pairs_rotated.shape)
```

```
# 还原 k_per_token
k_per_token_rotated = k_per_token_split_into_pairs_rotated.reshape(k_per_token.shape)
#print(k_per_token_rotated.shape)
```

```
# 计算 qk_per_token
qk_per_token = paddle.matmul(q_per_token_rotated, k_per_token_rotated.T) / (head_dim)**0.5
print(qk_per_token.shape)
```

```
# 可视化 qk_per_token
def display_qk_heatmap(qk_per_token):
    _, ax = plt.subplots()
    im = ax.imshow(qk_per_token.astype('float32').detach().numpy(), cmap='viridis')
    ax.set_xticks(range(len(prompt_split_as_tokens)))
    ax.set_yticks(range(len(prompt_split_as_tokens)))
    ax.set_xticklabels(prompt_split_as_tokens)
    ax.set_yticklabels(prompt_split_as_tokens)
    ax.figure.colorbar(im, ax=ax)

display_qk_heatmap(qk_per_token)
```

```
# 遮罩矩阵
mask = paddle.full((len(tokens), len(tokens)), float("-inf"), dtype=qk_per_token.dtype)
mask = paddle.triu(mask, diagonal=1)
#print(mask)

qk_per_token_after_masking = qk_per_token + mask
display_qk_heatmap(qk_per_token_after_masking)
```

```
# Softmax 
qk_per_token_after_masking_after_softmax = paddle.nn.functional.softmax(qk_per_token_after_masking, axis=1).astype('bfloat16')
display_qk_heatmap(qk_per_token_after_masking_after_softmax)
```

```
# 处理 v 层权重
v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.reshape([n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim])
#print(v_layer0.shape)
```

```
# 获取第一个头部的 v 层权重
v_layer0_head0 = v_layer0[0]
#print(v_layer0_head0.shape)
```

```
# 计算 v_per_token
v_per_token = paddle.matmul(token_embeddings, v_layer0_head0.T)
#print(v_per_token.shape)
```

```
# 计算 qkv_attention
qkv_attention = paddle.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
#print(qkv_attention.shape)
```

```
# multi-head attention
qkv_attention_store = []
for head in range(n_heads):
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head // 4]  # key weights are shared across 4 heads
    v_layer0_head = v_layer0[head // 4]  # value weights are shared across 4 heads

    q_per_token = paddle.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = paddle.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = paddle.matmul(token_embeddings, v_layer0_head.T)

    q_per_token_split_into_pairs = q_per_token.astype('float32').reshape([q_per_token.shape[0], -1, 2])
    q_per_token_as_complex_numbers = paddle.view_as_complex(q_per_token_split_into_pairs)
    q_per_token_split_into_pairs_rotated = paddle.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.reshape(q_per_token.shape)

    k_per_token_split_into_pairs = k_per_token.astype('float32').reshape([k_per_token.shape[0], -1, 2])
    k_per_token_as_complex_numbers = paddle.view_as_complex(k_per_token_split_into_pairs)
    k_per_token_split_into_pairs_rotated = paddle.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.reshape(k_per_token.shape)

    qk_per_token = paddle.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128)**0.5
    mask = paddle.full((len(tokens), len(tokens)), float("-inf"), dtype=qk_per_token.dtype)
    mask = paddle.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = paddle.nn.functional.softmax(qk_per_token_after_masking, axis=1).astype('bfloat16')
    qkv_attention = paddle.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)
# 拼接 qkv_attention
stacked_qkv_attention = paddle.concat(qkv_attention_store, axis=-1)
print(stacked_qkv_attention.shape)
```

```
# 处理输出层权重
w_layer0 = model["layers.0.attention.wo.weight"]
#print(w_layer0.shape)
```

```
# 计算 embedding_delta
embedding_delta = paddle.matmul(stacked_qkv_attention, w_layer0.T)
print(embedding_delta.shape)
```

```
# 更新嵌入
embedding_after_edit = token_embeddings_unnormalized + embedding_delta
print(embedding_after_edit.shape)
```

```
# 归一化
embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])
print(embedding_after_edit_normalized.shape)
```





这部分放到3.3代码部分


```
# 处理前向传播层权重

w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"]
w3 = model["layers.0.feed_forward.w3.weight"]

# 计算 output_after_feedforward
output_after_feedforward = paddle.matmul(paddle.nn.functional.silu(paddle.matmul(embedding_after_edit_normalized, w1, transpose_y=True)) * 
    paddle.matmul(embedding_after_edit_normalized, w3, transpose_y=True),
    w2, transpose_y=True
)

# 获取输出的形状
output_after_feedforward_shape = output_after_feedforward.shape
```

```
layer_0_embedding = embedding_after_edit+output_after_feedforward
#layer_0_embedding.shape

```



这部分放最后  四的代码部分

```
final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.reshape([n_heads, q_layer.shape[0] // n_heads, dim])
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.reshape([n_kv_heads, k_layer.shape[0] // n_kv_heads, dim])
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.reshape([n_kv_heads, v_layer.shape[0] // n_kv_heads, dim])
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head // 4]
        v_layer_head = v_layer[head // 4]
        q_per_token = paddle.matmul(layer_embedding_norm, q_layer_head, transpose_y=True)
        k_per_token = paddle.matmul(layer_embedding_norm, k_layer_head, transpose_y=True)
        v_per_token = paddle.matmul(layer_embedding_norm, v_layer_head, transpose_y=True)
        q_per_token_split_into_pairs = q_per_token.astype('float32').reshape([q_per_token.shape[0], -1, 2])
        q_per_token_as_complex_numbers = paddle.to_tensor(q_per_token_split_into_pairs, dtype='complex64')
        q_per_token_split_into_pairs_rotated = paddle.to_tensor(q_per_token_as_complex_numbers * freqs_cis, dtype='float32')
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.reshape(q_per_token.shape)
        k_per_token_split_into_pairs = k_per_token.astype('float32').reshape([k_per_token.shape[0], -1, 2])
        k_per_token_as_complex_numbers = paddle.to_tensor(k_per_token_split_into_pairs, dtype='complex64')
        k_per_token_split_into_pairs_rotated = paddle.to_tensor(k_per_token_as_complex_numbers * freqs_cis, dtype='float32')
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.reshape(k_per_token.shape)
        qk_per_token = paddle.matmul(q_per_token_rotated, k_per_token_rotated, transpose_y=True) / (128)**0.5
        mask = paddle.full([len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)], float("-inf"))
        mask = paddle.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = paddle.nn.functional.softmax(qk_per_token_after_masking, axis=1).astype('bfloat16')
        qkv_attention = paddle.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = paddle.concat(qkv_attention_store, axis=-1)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = paddle.matmul(stacked_qkv_attention, w_layer, transpose_y=True)
    embedding_after_edit = final_embedding + embedding_delta
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    output_after_feedforward = paddle.matmul(
        paddle.nn.functional.silu(paddle.matmul(embedding_after_edit_normalized, w1, transpose_y=True)) * 
        paddle.matmul(embedding_after_edit_normalized, w3, transpose_y=True),
        w2, transpose_y=True
    )
    final_embedding = embedding_after_edit + output_after_feedforward

```
```
d = dim  # 使用之前定义的 dim

# 初始化 RMSNorm
rms_norm = RMSNorm(d)
rms_norm.scale.set_value(paddle.to_tensor(model["layers.0.attention_norm.weight"].numpy()))

final_embedding = rms_norm(final_embedding)
final_embedding_shape = final_embedding.shape


```

```
#检查output权重形状
model["output.weight"].shape
```


```
# 计算 logits
logits = paddle.matmul(final_embedding[-1], model["output.weight"], transpose_y=True)

#logits_shape = logits.shape

```

```
next_token = paddle.argmax(logits, axis=-1)
```

```
#直接decode检查是否对应
tokenizer.decode([next_token.item()])
```
```
# PaddlePaddle提供了许多内置工具，如本次复现过程中使用的PaddleNLP，便于工作进行
```

