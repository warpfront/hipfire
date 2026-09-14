# 大语言模型推理引擎的核心：深入解析 KV Cache 的工程实践

对于从事 LLM 推理系统开发的工程师而言，**KV Cache（Key-Value Cache）** 是理解性能瓶颈、优化显存管理和设计高吞吐服务架构的基石。它不仅仅是一个“缓存”，更是决定推理延迟（Latency）和吞吐量（Throughput）的核心数据结构。

本文将系统性地拆解 KV Cache 的技术细节，从原理到工程落地，明确区分**已知事实**、**经验判断**与**常见误区**。

---

## 1. 为什么需要缓存 K/V？：Prefill 与 Decode 的本质差异

在 Transformer 架构中，自注意力机制（Self-Attention）的计算复杂度为 $O(N^2)$，其中 $N$ 是序列长度。如果不加优化，每生成一个新 Token，都需要重新计算整个序列的 Attention。

### 1.1 数学推导
对于第 $t$ 个时间步，输入向量 $x_t$ 经过线性变换得到 Query ($Q_t$)、Key ($K_t$) 和 Value ($V_t$)。
注意力输出 $O_t$ 的计算公式为：
$$
O_t = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}
$$
其中 $K_{1:t}$ 和 $V_{1:t}$ 是前 $t$ 个 Token 的 Key 和 Value 矩阵。

**关键洞察**：
- **Prefill 阶段（预填充）**：处理用户输入的 Prompt（长度 $L_{prompt}$）。此时 $Q, K, V$ 都是批量矩阵，计算是 **Compute-Bound**（计算受限）。GPU 的算力被充分利用，主要瓶颈是 FLOPs。
- **Decode 阶段（解码）**：逐 Token 生成。此时 $Q_t$ 是单个向量，但 $K_{1:t}$ 和 $V_{1:t}$ 是累积的历史矩阵。由于 $Q_t$ 维度低，矩阵乘法退化为向量-矩阵乘法，GPU 利用率极低，主要瓶颈是 **Memory-Bandwidth Bound**（显存带宽受限）。

### 1.2 缓存的作用
如果每次 Decode 都重新计算 $K_{1:t}$ 和 $V_{1:t}$，计算量将呈二次方增长。
**KV Cache 的核心价值**：将历史 Token 的 $K$ 和 $V$ 存储在显存中。在 Decode 阶段，只需计算当前 Token 的 $Q_t, K_t, V_t$，并将 $K_t, V_t$ 追加到缓存中，然后仅用 $Q_t$ 与缓存中的 $K_{1:t-1}$ 做点积。
- **已知事实**：KV Cache 将 Decode 阶段的计算复杂度从 $O(N^2)$ 降低到 $O(N)$（相对于当前步长），但引入了巨大的显存开销。

---

## 2. 显存占用估算：工程中的“第一性原理”

在部署模型前，必须精确估算 KV Cache 的显存占用，否则会导致 OOM（Out of Memory）。

### 2.1 基础公式
假设：
- $L$：最大上下文长度（Max Context Length）
- $H$：隐藏层维度（Hidden Size）
- $N_{heads}$：注意力头数量
- $N_{layers}$：Transformer 层数
- $B$：Batch Size（并发请求数）
- $D$：数据类型字节数（FP16=2, FP8=1, INT8=1）

每个 Token 在每个头、每层的 KV 占用为：
$$
\text{Bytes per Token} = 2 \times N_{layers} \times N_{heads} \times H_{head} \times D
$$
其中 $H_{head} = H / N_{heads}$，所以简化为：
$$
\text{Bytes per Token} = 2 \times N_{layers} \times H \times D
$$
> **注意**：这里的 $2$ 代表 Key 和 Value 各一份。

总显存占用：
$$
\text{Total KV Cache} = B \times L \times \text{Bytes per Token}
$$

### 2.2 具体算例
以 **Llama-2-70B** 为例：
- $N_{layers} = 80$
- $H = 8192$
- $D = 2$ (FP16)
- 假设最大上下文 $L = 4096$
- 并发数 $B = 1$

$$
\text{Bytes per Token} = 2 \times 80 \times 8192 \times 2 = 2,621,440 \text{ Bytes} \approx 2.5 \text{ MB}
$$
$$
\text{Total KV Cache} = 1 \times 4096 \times 2.5 \text{ MB} \approx 10 \text{ GB}
$$

**经验判断**：
- 对于 70B 模型，仅 KV Cache 就可能需要 10GB+ 显存。如果并发数 $B=10$，则需要 100GB 显存。这解释了为什么大模型推理通常需要多卡或量化。
- **常见误区**：认为模型权重占显存大头。实际上，在长上下文或高并发场景下，**KV Cache 的显存占用往往超过模型权重本身**。例如，Llama-2-70B 权重约 140GB（FP16），但 10 个并发、4K 上下文的 KV Cache 就达 100GB。

---

## 3. 架构优化：MQA 与 GQA

为了减少 KV Cache 的显存占用，业界引入了多头查询注意力（Multi-Query Attention, MQA）和分组查询注意力（Grouped-Query Attention, GQA）。

### 3.1 原理
- **MHA（标准）**：每个头有独立的 $K, V$。$N_{heads}$ 个头对应 $N_{heads}$ 组 $K, V$。
- **MQA**：所有头共享**一组** $K, V$。$N_{heads}$ 个头对应 **1** 组 $K, V$。
- **GQA**：折中方案。将 $N_{heads}$ 个头分成 $G$ 组，每组共享一组 $K, V$。$N_{heads}$ 个头对应 $G$ 组 $K, V$。

### 3.2 显存节省比例
$$
\text{KV Cache Size}_{MQA} = \frac{1}{N_{heads}} \times \text{KV Cache Size}_{MHA}
$$
$$
\text{KV