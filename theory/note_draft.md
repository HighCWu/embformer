# Embformer: Embedding weight-only Transformer

[code](https://github.com/highcwu/embformer)

## 001. Why could we replace linear layers with attention

```python
# %% [markdown]
# # 001. Why could we replace linear layers with attention layers?

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# %%
# Compute a random linear layer result
x = torch.randn(4, 8)
fc_weight = torch.randn(32, 8)
y1 = x @ fc_weight.T

# %%
# Compute `v` of the attention layer inputs
x_abs = x.abs()
x_sum = x_abs.sum(dim=-1, keepdim=True)
attn_weights = (x_abs / x_sum)[:,None]
v = fc_weight.T[None] * (x_sum[...,None] * x.sign()[...,None]) # (4, 8, 32) (bsz, seq_len, out_dim)
_y = (attn_weights @ v)[:,0]

assert torch.allclose(_y, y1)


# Recovered a random `logits` (result of `q @ k^T`)
logits = torch.log(attn_weights)
logits = logits - torch.randn(1)

## Verify that softmax is approximate to the original logits
softmax_result = torch.softmax(logits, dim=-1)
# print("Recovered logits:\n", logits)
# print("Re-softmax result:\n", softmax_result)
# print("Original p:\n", _attn_weights)

assert torch.allclose(softmax_result, attn_weights)

# %%
# Recovered a random `q` and `k` of the attention layer inputs

import math

d_k = 8

s = logits * math.sqrt(d_k)

## Use random factorization: s = q @ k^T
## Choose random k, then solve for q = s @ inv(k^T)
k = torch.randn(4, 8, d_k)
k_inv = torch.linalg.inv(k.transpose(-2, -1))
q = s @ k_inv
logits_rec = q @ k.transpose(-2, -1) / math.sqrt(d_k)

assert torch.allclose(logits_rec, logits, atol=1e-3)

# %%
# Check whether the output result of attention is consistent with the output result of linear

y2 = F.scaled_dot_product_attention(q, k, v)[:,0]

assert torch.allclose(y2, y1, atol=1e-3)

# %% [markdown]
# In fact, the process of backtracking `q`, `k`, and `logits` is not strictly necessary.
# 
# After eliminating the effect of `sign bits` and `softmax`, 
# we can approximately regard the `input` to a linear layer as the attention layer's computed `attn_weights`,
# and the linear layer's weight matrix `fc_weight(out_features, in_features)` as the attention layer's value matrix `v`,
# where the `sequence length` equals `in_features`.

```
---

## 002. Why could we use embedding layers everywhere?

According to the official explanation of **DeepEmbed** in RWKV8 \[1], a trainable high-dimensional vector is learned for each token in the vocabulary within the FFN of every model layer. These can be conceptualized as layer-wise embedding modules. These vectors are trainable during training, but during inference, they can be stored in RAM or SSD. Only a minimal subset of parameters needs to be preloaded per token, which drastically reduces VRAM usage.

During inference, the model can prefetch the embedding vector for a given token index at each layer in advance. This vector is then used to perform *channelwise scaling* of the FFN outputs.

These token-specific embedding vectors form a large yet sparse knowledge base that significantly enhances the model’s ability to store and retrieve world knowledge. Although this adds to the total parameter count, these vectors do not consume GPU memory during inference. Furthermore, during training, Tensor Parallelism (TP) can be used to avoid the bandwidth cost of gradient synchronization in Data Parallelism (DP), and they can also be offloaded to RAM or SSD.

In edge inference scenarios, these vectors can be stored in system memory or loaded on-demand via mechanisms like `mmap`. Each token only incurs an additional memory access cost of several dozen kilobytes, making the method highly suitable for deployment on edge devices.

A concurrent approach, **Per-Layer Embeddings (PLE)** from Google’s Gemma 3n \[2], proposes a similar idea.

My own proposal takes this concept further. In replacing linear layers entirely with attention mechanisms, I propose geting the *top-layer query* and all layers' *key* and *value* vectors directly from trainable embedding layers. This idea is inspired by the findings of Wu and Tu \[3], which show that the top-layer KV representations carry the most information, making it reasonable to focus primarily on them. Since the top-layer KV representations are closely related to the original input embeddings, it becomes feasible to directly substitute them with embedding outputs.

However, recognizing that removing linear layers severely limits model expressive capability, I retain the use of separate KVs across layers (as in traditional Transformers), but with their values sourced entirely from different embedding layer outputs.

To further test the viability of using pure embedding layers, I completely removed all linear layers from the FFN, using DeepEmbed-style channelwise scaling on attention outputs instead. Since removing linear layers eliminates inter-head communication, I introduced a *channel mixing mechanism* within the FFN to allow partial information exchange across attention heads.

---

\[1] DeepEmbed: Peng Bo, [https://x.com/BlinkDL\_AI/status/1926941496684519805](https://x.com/BlinkDL_AI/status/1926941496684519805) and [https://www.rwkv.cn/news/read?id=20250527](https://www.rwkv.cn/news/read?id=20250527)

\[2] PLE: Google, [https://developers.googleblog.com/en/introducing-gemma-3n/](https://developers.googleblog.com/en/introducing-gemma-3n/)

\[3] Layer-Condensed KV Cache for Efficient Inference of Large Language Models: Haoyi Wu, Kewei Tu, [https://arxiv.org/abs/2405.10637](https://arxiv.org/abs/2405.10637)

---

## 003. Rapid Experiment

I used the Qwen3 model-related code from the [transformers](https://github.com/huggingface/transformers) library as the base and removed all linear layers. Instead, the query values from the top-level attention, all key and value values from attention layers, and the deepembed values from the FFN layers are directly obtained from the embedding layer. The `lm_head` weight parameters are also directly tied to the input embedding layer, resulting in a model where only the embedding layer weights are trainable.

I used [MiniMind](https://github.com/jingyaogong/minimind) as the training framework. It not only provides training code but also a complete dataset needed for training. An important point is that its vocab size is very small—only 6400. If we used a commonly seen large vocab size, the memory consumption of the embedding layers would be significantly larger. This is a key area for future optimization. However, the optimization direction is not to use a smaller vocab size, but to explore how to offload embedding layers during training. This is because my method essentially replaces computation with lookup, and apart from the attention layers that involve cross-token computation, most other values can be retrieved quickly.

I trained a 16-layer model with a hidden size of 768. The training results aligned with my expectations—it performed notably worse than a traditional transformer model with the same configuration, but could still perform inference to some extent. The reason is clear: during inference, only a small portion of the model’s weights are used, since all weights come from the embedding layer.

## 004. Conclusion

The experimental results demonstrate that it's feasible to use the embedding layer in place of linear layers to achieve transformer capabilities. However, under the same parameter count, the Embformer model performs significantly worse in training compared to traditional transformer models. The main reason is that only a small fraction of the model's weights are used during each inference. By increasing the depth or width of the Embformer, better performance can be achieved. Although deepening or widening the model has minimal impact on the output speed of embedding layers (which simply query by token ID), matrix multiplications within the attention layers become the computational bottleneck and are heavily affected. Therefore, one future optimization focus should be improving the efficiency of attention operations. Another focus is how to reduce memory usage during training when increasing vocab size.
