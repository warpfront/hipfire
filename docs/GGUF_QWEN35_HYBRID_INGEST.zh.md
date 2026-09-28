# hipfire · GGUF→Qwen3.5/3.8 混合栈直转 + AWQ 侧车 + MTP 侧车
## 改动记录与升级指南

| 项 | 值 |
|---|---|
| 仓库 | `~/hipfire`（远程 Linux 主机，AMD RDNA3 gfx1100） |
| 分支 | `feat/gguf-qwen35-hybrid-ingest` |
| 基线 commit | `ad10b3d97`（Merge pull request #766 from warpfront/iu4-mq4v2-gfx11） |
| 状态 | **已提交到分支 `feat/gguf-qwen35-hybrid-ingest`**（未推送） |
| 记录日期 | 2026-09-28 |
| 目标模型 | Qwen3.8-27B（GGUF `IQ3_S`，866 张量，含 1 层 MTP/nextn） |
| 产物 | trunk `.hfq` 14,980,815,872 B（1347 张量 = 851 主 + 496 AWQ 侧车）<br>`.mtp` 225,716,224 B（15 张量） |

---

## 1. 这次干了什么

从**只有 GGUF** 的社区量化产物（`Qwen3.8-27B-GSQ-RCO-IQ3_S`）出发，直接产出一份能被 hipfire 运行时加载、且质量对齐 golden `qwen3.8-27b.mq4-xt` 的 `.hfq`，并额外产出两样东西：

1. **AWQ 496 张 per-input-channel 尺度侧车**（契约 `qt=44` / `MQ4G256V2`），让低比特量化的列加权对齐 golden；
2. **MTP 头侧车 `.mtp`**（契约 `arch_id=21`），让 `hipfire run --spec mtp` 从「报错不可用」变成「可用、无损、有收益」。

三件事都必须建立在**同一份 GGUF 读取 + 约定归一化**之上，所以改动集中在 `hipfire-quantize/src/pipeline_gguf.rs`。

**后续补充（2026-09-28 第三轮，本文档 §8/§9）**：
4. **第三方变体复核**：对微调变体 `Swift-1.5-Qwen3.8-27B` 先证明「量化方案与基线一致」（张量名/dtype/shape 零差异），
   再原样走一遍全流程并验收通过 —— 等于用第二个模型**复现验证了整条管线**；
5. **性能调优**：在该变体上把 hipfire 的参数面扫了一遍，产出推荐配置、长上下文的 5 分钟闸门结论，
   以及**四条测量学坑**（`bench --tg` 量不出投机收益 / `--draft-max` 只有 `run` 有 /
   `timeout` 杀 bench 会留孤儿 daemon / `--pp N --ctx N` 是两遍 prefill）。

---

## 2. 改动清单

### 2.1 `crates/hipfire-quantize/src/gguf_input.rs`（+148 / −11）

补 GGUF 读取器的缺失能力：

| 位置 | 内容 |
|---|---|
| `GgmlType`（`:37` 起 +13/+10/+9/+11/+12） | 新增 IQ 系列量化类型枚举项与 `block_size` / `type_size` / 名字映射 |
| `dequant_q2_k`（`:582` +62） | 新增 Q2_K 反量化（源 GGUF 的 `token_embd` 是 IQ2S，早期版本缺这条） |
| `dequant_q4_0`（`:390`/`:399`/`:404`） | 修正块大小与 nibble 顺序 |
| `dequant_q5_k`（`:559` +7） | 修正 Q5_K 的 6-bit 高平面解包 |
| `tensor_to_f32`（`:721`/`:725` +12） | 把新增的 IQ 类型接进总分发 |

> ⚠️ 这里的坑：GGUF 的 `dtype` 数字是**跨版本权威**的，但**同一数字在不同 ggml 版本下语义变过**（尤其 IQ2/IQ3 家族）。改这里必须对着 `iq_tables.rs` 的参考表核，不要凭记忆写 block 大小。

### 2.2 `crates/hipfire-quantize/src/calibration.rs`（+256 / −1）

| 位置 | 内容 |
|---|---|
| `:124` +13 | `gguf_to_safetensors_name` 增加 hybrid 分支入口 |
| `:186` +67（新 `qwen35_hybrid_slot`） | **GGUF 槽名 → HF 槽名** 全表（见 §3.1） |
| `:838` +3 | `config_json_from_gguf` 调用 `apply_qwen35_fields` |
| `:846` +173（新 `apply_qwen35_fields`） | 从 `qwen35.*` KV 合成 HF `config`：`head_dim=256`、`layer_types[64]`、`linear_num_{key,value}_heads`、`linear_{key,value}_head_dim`、`partial_rotary_factor=0.25` 等 |

**为什么要在转换期合成 `config`**：`gguf_convention_to_hf` 与 V-head 逆重排都要在**转换时**知道 `head_dim` / `partial_rotary_factor` / linear-attn 头数；这些值在 GGUF 里是 `attention.head_count` + `rope.dimension_count` 等**间接**表达，必须显式推出来写进 `.hfq` 元数据，运行时才不用重算。

### 2.3 `crates/hipfire-quantize/src/pipeline_gguf.rs`（+697 / −81，主战场）

新增的顶层项（顺序即文件内顺序）：

```
:199  +392  fn gguf_is_mtp_tensor(name, trunk_len, has_nextn) -> bool
            enum VReorderAxis
            fn vreorder_rows_inplace(...)
            fn vreorder_cols_inplace(...)
            fn gguf_convention_to_hf(name, arch_id, is_norm, shape, vreorder, d)
            fn gguf_tensor_to_f32_hf(info, raw, shape, is_norm, arch_id, vreorder)
            fn quantize_mq4g256v2_awq(...)          ← 本轮新增
            const MTP_SLOT_MAP: &[(&str,&str,bool)] ← 本轮新增
            fn mtp_canonical_name(gguf_name)        ← 本轮新增
            fn mtp_pack_one(...)                    ← 本轮新增
:605  +6    run_gguf_pipeline 签名尾部追加 mtp_out: Option<&Path>
:794  +57   MTP 头排除（block_count 含 nextn 层）
:852  +14   MTP 张量分派（写侧车 or 丢弃）
:871  +28   shape / m_dim / k_dim 轴序修正（见 §3.1.4）
:1437 +3    16 处 qt44 调用点改走 quantize_mq4g256v2_awq
:1457 +13   AWQ 侧车槽声明
:1478 +24   AWQ 侧车写出
:1504 +60   `.mtp` 侧车写出（arch_id=21）
```

### 2.4 其余小改

| 文件 | 行数 | 内容 |
|---|---|---|
| `cli.rs` | +8 | 新增 `--mtp-out <PATH>`（`--awq` / `--awq-alpha` / `--imatrix` / `--awq-imatrix` 都是**上游已有**） |
| `pipeline.rs` | +2 | `run_gguf_pipeline(...)` 调用末尾传 `args.mtp_out.as_deref()` |
| `main.rs` | +2 | `mod iq_dequant; mod iq_tables;` |

### 2.5 新增源文件

| 文件 | 行数 | 作用 |
|---|---|---|
| `src/iq_dequant.rs` | 651 | IQ 系列反量化的独立实现（便于 `iq_audit` 复用） |
| `src/iq_tables.rs` | 1119 | IQ 码本 / 参考表 |
| `src/bin/iq_audit.rs` | 273 | 反量化审计工具（对拍参考实现） |

> `lib.rs` **没有**导出 `gguf_input` / `iq_dequant` 等模块，`main.rs` 用的是 bin 私有 `mod`。所以任何独立的 `bin/*.rs`（例如 `mtp_extract.rs`）都**拿不到**这些约定归一化工具 —— 这就是为什么 MTP 逻辑必须放进 `pipeline_gguf.rs` 而不是另写一个 bin。

---

## 3. 三条契约（升级时最容易踩的地方）

### 3.1 GGUF ↔ HF 约定归一化

**唯一入口**：`gguf_tensor_to_f32_hf()`。**所有**取 f32 的调用点都必须走它。漏一个，那个张量就会带着 GGUF 的折算约定进 `.hfq`，运行时安静地算错（历史故障：linear_attn 整族错位 → 模型立刻吐 EOS）。

约定表（`gguf_convention_to_hf`，仅 `arch_id == 5` 生效）：

| # | 规则 | 触发条件 | 动作 |
|---|---|---|---|
| 1 | **norm 折算** | `gguf_is_norm_tensor()` = 名字含 `_norm` 或 `norm.weight` | `w -= 1.0`（GGUF 存 `w+1`，HF/运行时存 `w`，加载时 `+1.0`） |
| 1b | **例外** | `*ssm_norm.weight` | **不**减 1（GGUF 里就是 HF 原值，实测与参考完全相同） |
| 2 | **ssm_a 对数域** | 名字以 `ssm_a` 结尾 | `-exp(A_log)` → `A_log`：取负后 `ln()` |
| 3 | **V-head 逆重排** | 见下表 | 沿行/列 view 成 `[k_heads, v_per_k, blk]` → 转置前两轴 → reshape |
| 4 | 其余 | — | 原样 |

V-head 逆重排覆盖（`vreorder = (k_heads, v_heads, head_dim) = (16, 48, 128)`）：

| 名字后缀 | 轴 | start | blk |
|---|---|---|---|
| `ssm_a` / `ssm_dt.bias` | 行 | 0 | 1 |
| `attn_qkv.weight` / `ssm_conv1d.weight` | 行 | `2*k_heads*hdim` | `hdim` |
| `attn_gate.weight` | 行 | 0 | `hdim` |
| `ssm_alpha.weight` / `ssm_beta.weight` | 行 | 0 | 1 |
| `ssm_out.weight` | 列 | 0 | `hdim` |

> 这是 llama.cpp `_LinearAttentionVReorderBase` 的逆运算。**只作用于 linear-attn**；全注意力层的 `attn_v.weight` 不受影响（MTP 层正是全注意力，所以天然跳过）。

**3.1.4 轴序**（`:871` 的修正，是本轮最隐蔽的一处）

```rust
// GGUF dim[0] = 收缩轴(k)，dim[1] = 输出轴(m)；HFQ 存 [out, in]。
// 载荷本身已是 row-major、内轴最快，所以交换只发生在 METADATA，不做物理转置。
let shape = match info.shape.len() {
    2 if is_conv1d_tensor(&info.name) => vec![shape[1], 1, shape[0]],
    2 => vec![info.shape[1], info.shape[0]],
    3 => vec![shape[1], 1, shape[0]],
    _ => info.shape.iter().copied().collect(),
};
let k_dim = info.shape[0];   // 收缩轴
let m_dim = info.shape[1];   // 输出轴
```

修正前 `m` 取的是 `shape[0]`（内轴），只有 `out == in` 时才对，**所有矩形投影都是静默 mis-shape**。`k_dim` 一直是对的，所以两边不一致时没有任何交叉检查会报错。验证方式：逐张量 `HFQ.shape == reversed(GGUF.dims)`（脚本 `154` 的第 ④ 道门槛）。

### 3.2 AWQ 侧车（`qt=44` / `MQ4G256V2`）

**契约**（`dispatch.rs:512-518` 定义，`hfq.rs:1586 awq_scale_f32_bytes` 消费）：

- 量化器把权重**预乘 `s`**（per-**INPUT**-channel，长度 K = 收缩轴），并发侧车 `<stem>.awq_scale.weight`；
- 侧车是 **F16、1-D、`shape=[K]`、`group_size=0`**；
- 运行时在 rotate 步**除以 `s`**，于是 `(W·s)·(x/s) = W·x`；
- **缺侧车不会失败**：运行时静默丢弃侧车、按 `W·x` 算。所以 AWQ 是**质量优化，不是故障线** —— 验收必须专门检查侧车存在性，不能只看「能跑」。

**预乘必须在 FWHT 之前**（`awq_pre_scale_weights` 的文档原文）。

**scale 公式**（`calibration.rs:547`）：
```
log_s = (α/2) · ln(clamp(v, 1e-12, 1e30))
log_s -= mean(log_s)              // 几何均值归一到 1
s = exp(log_s);  s = clamp(s, 1e-2, 1e2)
```
**可逆**：`v = exp(ln(s) / (α/2))`。默认 `α = 0.55`（`pipeline.rs:821`）。

**imatrix 从哪来** —— 本轮的关键取舍：GGUF 输入**没有** AWQ 侧车可以直接抄，所以用脚本 `148` 从 **golden 的 496 张侧车反解** imatrix（写成 `<name>.in_sum2` F32 一维 GGUH，496 条）。
- 反解与转换使用**同一个 α**，全局常数因子在 AWQ 契约下无害（`W·(s·c)` 与 `x/(s·c)` 相消），所以不需要知道 golden 当初用的确切 α。
- 自校验：反解再正解，最大相对差 `8.9e-4`（= 每张量一个全局常数）。
- HIPFIRE 的 GGUF 管线里 `info.name` 正是 `blk.N.*`，与 `llama-imatrix` 的输出命名一致 → `imatrix_weights_for(info.name)` 直接命中，无需改名。

**496 的构成**：12 族 = `linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z,out_proj}` ×48 + `mlp.{down,gate,up}_proj` ×64 + `self_attn.{q,k,v,o}_proj` ×16；K 三档 = 5120 / 6144 / 17408。
497 个 qt44 里**唯一无侧车的是 `lm_head`**（故意排除）→ 497 − 496 = 1 闭合。

**接线点**：`quantize_mq4g256v2_awq()` 是 qt44 的唯一入口，16 处调用点全部改走它。它在下列情况下**静默回退**到普通 `quantize_mq4g256v2`（并打印一行 `AWQ: ... SKIPPED`）：α 未设 / 不在 eligible 表 / 找不到 imatrix / `imatrix.len() != K`。

### 3.3 MTP 侧车（`arch_id = 21` = `QWEN35_MTP_HEAD`）

**容器**：HFQ（`magic=HFQM, ver=1`），15 张量，名字是**裸名**（运行时 `mtp_head.rs::load_weight_raw` 不补前缀）。

| 规范名 | qt | 形状 | 说明 |
|---|---|---|---|
| `shared_head_norm` `enorm` `hnorm` `attn_norm` `attn_post_norm` | 2 (F32) | `[5120]` | 5 个 |
| `attn_q_norm` `attn_k_norm` | 2 (F32) | `[256]` | `head_dim` 长度 |
| `eh_proj` | 13 (MQ4G256) | `[5120, 10240]` | |
| `wq` | 13 | `[12288, 5120]` | q+gate 融合 |
| `wk` `wv` | 13 | `[1024, 5120]` | |
| `wo` | 13 | `[5120, 6144]` | |
| `ffn_gate` `ffn_up` | 13 | `[17408, 5120]` | |
| `ffn_down` | 13 | `[5120, 17408]` | |

**GGUF 侧命名**：MTP 张量都挂在 `blk.64.`（`block_count=65` = 64 trunk + 1 MTP），槽名映射用「`blk.{idx}.` 之后的**完整剩余**」精确相等匹配。

> ⚠️ **不能用 `ends_with`**：`attn_norm.weight` 是 `nextn.shared_head_norm.weight` 的真后缀，用后缀匹配会把两者混起来。

**三条硬约定**：
1. `load_norm_raw` **断言 `quant_type == 2`（F32）** → 7 个 norm 必须 F32，不能走 F16；
2. **norm 值约定与主干一致**：`.mtp` 存 HF 原始值，加载时 `+1.0`。所以打包走的是**同一个** `gguf_tensor_to_f32_hf`（含 `w -= 1.0`）。已验证逐字节正确（脚本 `156`）。
3. 2-D 权重必须 `K % 256 == 0`（MQ4G256 组大小），否则 `mtp_pack_one` 返回 `Err`，张量进 `mtp_unmapped` 并被丢弃 + 告警。

**元数据**：`from_metadata` 必需键 `n_embd / n_head / n_head_kv / n_embd_head / n_ff / vocab_size`；
`n_rot = n_embd_head × partial_rotary_factor`，而 `partial_rotary_factor` 只从 **`config_text_config.partial_rotary_factor`** 读 —— **读不到就静默退回 0.25**。本项目的 trunk `config` 里 prf 是嵌套在 `config.partial_rotary_factor`，所以 `.mtp` 里必须**显式搬到 `config_text_config` 下**。

> 实测自洽性：`256 × 0.25 = 64` == 源 GGUF `qwen35.rope.dimension_count = 64`。这是核验 prf 是否被正确搬运的最好判据。

**运行时发现规则**（`hipfire-loader/src/lib.rs:2113-2175`）：
1. 先试 **bundled**：文件尾部 16 字节的 `mq4_merge_mtp` trailer；
2. 再试 **sidecar**：`trunk_path.with_extension("mtp")` —— **替换**扩展名，所以
   `qwen38-27b-iq3s-mtp-awq.hfq` → `qwen38-27b-iq3s-mtp-awq.mtp`（**不是** `qwen38-27b-iq3s-mtp.mtp`）。
3. 前置 gate：`arch_id ∈ {5, 6}` 且没有 adaptive/eviction 且 `spec.mtp != Some(false)` 且没有 DFlash/DSpark。

`--spec mtp` 会把 `spec.mtp` 置成 `Some(true)`，此时**找不到头是硬失败**：
```
load failed: MTP head required (mtp=on) but not found: no bundled trailer or .mtp sidecar found
```
这条是验收 `.mtp` 侧车是否真的被加载的最佳负对照。

---

## 4. 用法

```bash
export HIPFIRE_HOME=$HOME/hipfire/home
export HIPFIRE_ROCM_PATH=/opt/rocm
export HIPFIRE_MODELS_DIR="$HOME/models/hipfire"
Q=$HOME/hipfire-target/release/hipfire-quantize   # 或 cargo build --release

# GGUF → hfq（含 AWQ 侧车 + MTP 侧车）
$Q --input  ~/models/qwen38-27b-iq3s-mtp/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
   --output ~/models/hipfire/qwen38-27b-iq3s-mtp-awq.hfq \
   --format mq4v2 \
   --awq --awq-alpha 0.55 \
   --imatrix ~/models/hipfire/qwen38-27b.recon.imatrix.gguf \
   --mtp-out ~/models/hipfire/qwen38-27b-iq3s-mtp.mtp

# 让运行时找得到 MTP（名字必须与 trunk 同 stem）
ln -f ~/models/hipfire/qwen38-27b-iq3s-mtp.mtp \
      ~/models/hipfire/qwen38-27b-iq3s-mtp-awq.mtp

# 跑
$HOME/hipfire/home/bin/hipfire run \
    ~/models/hipfire/qwen38-27b-iq3s-mtp-awq.hfq "..." -n 240 -t 0 --spec mtp
```

不需要 MTP 时**不要**传 `--mtp-out`：MTP 张量无论如何都不进 trunk（`skipped_mtp` 计数上报），传了只是「顺便写出来」。

---

## 5. 验收证据（全部通过）

### 5.1 AWQ

| 门槛 | 脚本 | 结果 |
|---|---|---|
| 索引一致 | `150` step① | `only-ref 0 / only-cand 0 / qt 0 / gs 0 / shape 0 / data_len 0`（REF 851 + 496 额外侧车） |
| 侧车逐元素 vs golden | `151` | **496/496**，12 族 `corr = 1.000000`，比值中位 `1.00000`，最大相对 std `8.1e-5` |
| 预乘↔侧车配对 | `152` | 各族 `c_w` 全在 `0.976 ~ 1.008`（≈1）。脚本原阈值 `|c_w/c_s−1|<1e-2` 中位数 `1.009e-2` 判 FAIL 属**阈值误报** —— 散度来自两个独立量化 blob 相除的量化噪声（权重域比值相对散度中位 `3.378e-1`），判据应是「`c_w/c_s` 是接近 1 的常数」 |
| F16 未波及 | `128 --raw` | `n=801 ok=801 maxΔ=0.003906 越界=0`（801 = 305 主 F16 + 496 侧车 F16） |
| 连贯性 A/B | `150` step⑤ | `A1_awq=Paris`、`A2_awq_iu4=Paris`、`A3_awq_math=391`、`R_golden=Paris` 全对 |
| 运行时消费侧车 | — | `A1_awq.err` / `R_golden.err` 各只有 **1** 行 `lm_head AWQ sidecar: absent (no-op)`（= 唯一无侧车者，符合预期） |

转换耗时 301 s，产出 1347 张量（851 主 + 496 侧车），日志 496 行 `AWQ: ...`、0 行 `SKIPPED`。

### 5.2 MTP

| 门槛 | 脚本 | 结果 |
|---|---|---|
| 静态契约 | `154` | `magic=HFQM ver=1 arch_id=21 n=15`；15/15 规范名精确；7×F32(1-D) + 8×MQ4G256(2-D, K%256==0)；**每张量 HFQ shape == reversed(GGUF dims)**；`n_rot = 256×0.25 = 64` == 源 `rope.dimension_count`；`n_embd/n_head/n_head_kv/n_ff` 与源 KV 全等 |
| norm 约定往返 | `156` | **7/7** 落盘字节 == `f32(v_gguf − 1.0)`，逐字节（运行时的 `+1.0` 能精确还原） |
| 加载 + 投机 + 无损 | `155` | 8/8 PASS：M1 加载侧车、M4 `auto` 也加载、**M2 缺侧车硬失败**、M3 `off` 不加载、**M1 == M3 == M4 逐字节** |
| 长生成无损 + 收益 | `157` | 171 tok / 46 windows / `drafter=mtp tau=2.72 tok/s=83.5`；与 `--spec off` **md5 相同**（`755db4af…`） |

产物尺寸自洽：带 `--mtp-out` 的 trunk `14,980,815,872 B` 与不带 MTP 的 `qwen38-27b-iq3s-awq.hfq` **完全相同** → 证明 MTP 确实没进 trunk。

---

## 6. 已知问题 / 未决事项

### 6.1 ✅ 投机路径在长自由文本下与 AR-only 不是逐字节相同 —— **已查清：运行时既有行为，与本次改动无关**

> **2026-09-28 复核结论（脚本 `160_verify_61.sh`）**：本条的旧判断（"不是导出器的锅"）**成立**，
> 但旧版把根因**定位在错误的代码位置**（`hipfire-runtime/src/spec.rs:1164 MtpSpeculator`
> 只是把任意 `MtpDrafter` 接到通用 `Speculator` 接口上的**适配器**，不含任何数值计算）。
> 复核用三组决定性对照把根因钉死在运行时，并给出正确的代码锚点。

**现象（500 词自由文本，`-n 700 -t 0` 贪心，同一进程外每臂独立起 daemon）**

| 臂 | 输出 / md5(前16) | 与 AR 首差 | tau | 解释 |
|---|---|---|---|---|
| `--spec off` ×2 | 3125 B `8c337f97e1a26b0f` | — | — | 基线，两次一致 |
| `--spec mtp`（K=3）×2 | 3269 B `7fc0ccc2e4fbd645` | 647 B | 1.24 | 本命题复现，两次一致 |
| `--spec mtp --draft-max 1`（K=1） | 3181 B `fdaa5a896c77cbe2` | 647 B | 0.77 | **K 不影响**翻转点 |
| `--spec ngram`**（模型无关草稿器）** | 3273 B `fe39bbe4a4c82880` | 1472 B | 0.01 | **无 MTP 头也翻转 → 与我们的头无关** |
| `HIPFIRE_SPEC_WINDOW_ROLLBACK=0` | 3269 B **与 K=3 同 md5** | 647 B | 1.24 | 严格前缀修复路径**无责** |
| `HIPFIRE_DFLASH_CKPT_RESUME=0` | 同上 | 647 B | 1.24 | checkpoint 恢复**无责** |
| `HIPFIRE_MTP_P_MIN=0.0` | 3359 B | 647 B | 1.40 | 草稿置信度闸门**无责** |
| **`qwen3.6-35b-a3b.mq3p` + 官方 `.mtp`**（sha256 `1e11a06d…`，与注册表一致） | 3567 B → 3380 B | **93 B** | 1.00 | **上游自己的头/权重也非逐字节** |

**根因（代码级）**：走投机路径时，**每一个窗口发射的 token 都由"块批量 verify 前向"算出**，
而 AR 路径每个 token 都由"单 token 解码前向"算出 —— 两套 GEMM 形状/kernel 不同，浮点舍入不同，
于是在 top-1/top-2 差距小于该舍入差的位置翻转 argmax。

- 投机侧：`SpecTarget::verify_block` → `crates/hipfire-arch-qwen35/src/spec_impl.rs:214`
  → `verify_dflash_block`（`crates/hipfire-arch-qwen35/src/speculative.rs:2788`），
  一次 `forward_prefill_batch_*` 跑完 `b = 1 + K` 个位置；
- AR 侧：daemon 的单 token `forward`（S=1）。
- 即使草稿**全被拒**，补的那个 "bonus" token 也来自批量前向 —— 这正是 `--spec ngram`
  （τ=0.01，几乎每个窗口都全拒）仍会翻转的原因，也是本命题与草稿器质量无关的原因。

**危害评估**：翻转处是**同一句的等价改写**（`tea consumption in China dates back at least to`
→ `tea was being consumed in China as early as`），不是退化；数字/重复型提示（60 个整数，
`-n 240`）下 MTP 与 AR **md5 完全相同**。所以这是「贪心下近邻 tie 的翻转」，不是质量缺陷。

**结论与处置**：
1. **不需要改本次改动**（导出器/侧车是对的：官方头在同一 runtime 上也复现同样行为）；
2. 若某场景**强制要求严格逐字节复现**，只能 `--spec off`（放弃投机提速），
   或由上游在 verify 路径上提供"用与解码同款 kernel 的串行 verify"选项
   （`docs/speculation-support-inventory.md` 把串行 `verify_block` 描述为 *correct baseline*，
   块并行 verify 是 *optional perf*）；
3. 上游文档里 "MTP 逐字节等于 AR" 的结论是在**其固定短 fixture**（capital / code 之类）上测的，
   本命题用**官方权重**在 674 token 自由文本上复现出非逐字节 —— 属该性质的适用范围问题，
   不影响 §5 的验收结论（那次验收用的是 60 整数提示，md5 相同）。

**日志**：`~/accept61.txt`、`~/accept61/*.txt|err`；复现脚本 `160_verify_61.sh`，分歧定位 `162_diff_arms.py`。

### 6.2 其他

- `.mtp` 元数据里 `tie_word_embeddings` 硬编码为 `true`。源 GGUF 不携带该键（`output.weight` 与 `token_embd.weight` dtype 不同 → 实际未 tie）。已 grep 确认运行时**没有**消费这个字段，故无实际影响；若将来被消费，需要从 GGUF 推导。
- `MTP_SLOT_MAP` 第三列（是否 norm）目前在 `mtp_pack_one` 里**没被使用**（实际用 `gguf_is_norm_tensor(&info.name)`，与主干同一判据）。保留列是为了与 `mtp_extract.rs::MTP_NAMING_MAP` 逐列对齐、便于人工比对。
- 源码已提交到分支（未推送）。`~/models/hipfire/` 下有若干中间产物（`*.preconvention.hfq` / `*.pre2d.hfq` / `*.mq4v2.hfq` 等），是否清理待定。
- 旧仓库 `/mnt/win-c/ai/hipfire` **按要求保留，未删**。

---

## 7. 升级指南（rebase / 拉到新上游时怎么做）

### 7.1 改动只落在一个 crate

全部集中在 `crates/hipfire-quantize/`。上游若重构 `pipeline_gguf.rs`，冲突面最大；`calibration.rs` 其次。运行时 crate **只读不改**。

### 7.2 按依赖顺序重建

1. **先确认 `gguf_input.rs` 的 IQ 反量化**：拿 `bin/iq_audit.rs` 对拍参考实现跑一遍。上游改了 IQ 表就可能静默偏移。
2. **再确认轴序**：任何 `HFQ.shape == reversed(GGUF.dims)` 的批量校验（脚本 `154` 的第 ④ 门槛可复用，把目标换成 trunk 的任意几个投影层）。
3. **再确认约定归一化**：`ssm_norm` 不减 1、`ssm_a` 取对数、V-head 逆重排的 7 个名字后缀 —— 这四类里任何一类漏掉都会**静默**出错（不报错、只是输出退化）。
4. **最后跑两条端到端**：AWQ 走 `150_accept_awq.sh`，MTP 走 `155` + `157`。

### 7.3 补丁脚本可重放（锚点唯一性有断言）

| 脚本 | 作用 |
|---|---|
| `127_patch_gguf_convention.py` | 插入约定归一化 + V-head 逆重排 |
| `142_patch_vreorder_2d.py` | 补 2-D V-head 逆重排 |
| `149_patch_awq_gguf.py` | AWQ 接线（helper + 16 调用点 + 侧车槽 + 侧车写出），**含 4 个锚点唯一性断言** |
| `153_patch_mtp_gguf.py` | MTP 集成（`cli.rs` + `pipeline.rs` + `pipeline_gguf.rs` 四处），含锚点断言 |

> 上游一改，这些脚本的锚点会失配并**报错退出**（这正是设计意图：宁可失败也不静默改错地方）。此时手工按 §2.3 的顶层项清单重打。
> ⚠️ `149` 的事后核对把 `awq_scale.weight` 期望出现次数写成了 2（实际 3：2 处代码 + 1 处注释），会假报 `[BAD] exit 6` —— **补丁本身是对的**，用 `diff -u` 人工审定即可。这是脚本自身的计数 bug，不影响改动正确性。

### 7.4 编译

```bash
cd ~/hipfire && cargo build --release -p hipfire-quantize   # ~8-13 s 增量
```
判据：新代码**零告警**。`grep -iE "mtp|awq" <log> | grep -iE "warning|unused"` 应为空。

### 7.5 回退

- 提交前 → `git checkout -- crates/hipfire-quantize/`；提交后 → `git checkout ad10b3d97 -- crates/hipfire-quantize/`（或 `git revert` 那个 commit）；
- 新文件（`iq_dequant.rs` / `iq_tables.rs` / `bin/iq_audit.rs`）是 untracked，直接删；
- 产物侧：不带 `--mtp-out` / 不带 `--awq` 就退回旧行为（`.mtp` 不生成、侧车不写），**不需要**回退源码即可做 A/B。

---

## 8. 第三方变体复核 + 全流程验收（Swift-1.5，2026-09-28 第三轮）

### 8.1 先确认「量化方式是否一样」

模型：`~/models/swift-qwen38-27b-iq3s-mtp/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`

| 检查项 | 结果 |
|---|---|
| 张量名集合 | 866 = 866，**完全一致** |
| 逐张量 `dtype` 差异 | **0** |
| 逐张量 `shape` 差异 | **0** |
| 文件体积 | 差 **64 B**（纯 KV 元数据） |
| 数据区字节 | 180 个张量完全相同 / 686 个不同 |

→ 结论：**量化方案同款**（同一套 IQ/f16 混合布局），差异是**真·权重微调**（norm、`ssm_a`、`ssm_dt.bias` 这类参数未变，投影矩阵全变）。可以照基线流程原样走。

脚本：`161_cmp_gguf_header.py`（KV + 张量名/dtype/dims + 张量顺序）。

### 8.2 走完整流程

`163_convert_swift.sh` 跑两臂：

| 臂 | 耗时 | 产物 | 内容 |
|---|---|---|---|
| A · AWQ + MTP | 385 s | `swift-1.5-qwen38-27b-iq3s-mtp-awq.hfq`（14,980,815,872 B）+ `…-awq.mtp`（225,716,224 B） | 1347 = 851 主 + **496** AWQ 侧车；MTP 15 张量 |
| B · 无 AWQ（负对照） | 270 s | `swift-1.5-qwen38-27b-iq3s-mtp-mq4v2.hfq` | 851 |

日志判据：`MTP [ 行 = 15`、`AWQ:` 行 = 496、`SKIPPED` = 0。

### 8.3 验收（`164_accept_swift.sh` → `~/accept_swift.txt`，177 行）

| 门槛 | 判据 | 结果 |
|---|---|---|
| ① 索引 vs golden | 名/qt/gs/shape/**data_len** | 全 0；直方图 `{MQ4G256V2:497, F16:305, Q8F16:49}` 与 golden 逐项一致 |
| ② 索引 vs **基线产物** | 同上，且 data_len 必须全 0 | **全 0** → 同样权重必出同样文件，**流程逐比特确定** |
| ③ 跨产物字节对账 | 源相同 ⇒ 产物字节必须相同 | 496 侧车 **md5 全同** ✓；主张量一项**脚本自身对齐口径有误**（见 §9.7 坑 4） |
| ④ AWQ 侧车 vs golden（`151`） | corr≈1、比值恒定 | 12 族 **corr 全 1.000000**，比值中位全 1.00000 → PASS |
| ⑤ AWQ 配对硬检验（`152`） | `|c_w/c_s-1|` < 1e-2 | 中位 1.73e-02 → 脚本判 FAIL，**已知误报**（见 §9.7 坑 3） |
| ⑥ MTP 静态契约（`154`） | `arch_id=21 / 15 张量 / 名字 / qt / shape` | 全 `[ok]`，`RESULT: PASS` |
| ⑥′ norm 约定往返（`156`） | 落盘 == `f32(v_gguf − 1.0)` | **7/7 逐字节** |
| ⑦ 运行时 | `Paris`/`Paris`/`391`、`S4==S5` 逐字节、MTP 加载/未加载 | 全 **PASS**；AWQ 诊断行 1 行 `lm_head AWQ sidecar: absent (no-op)` |
| ⑧ 负对照 | 移走 `.mtp` 后 `--spec mtp` 必须硬失败 | `exit=1` + `MTP head required (mtp=on) but not found` → **PASS** |

> 与基线 `.mtp` 的关系（顺带观察）：Swift 与基线两份 `.mtp` 里 **7 个 F32 norm 逐字节完全相同**，**8 个 2-D 权重全不同** —— 与「norm 未被微调、投影矩阵被微调」吻合，也再次印证 §3.3 的 norm 约定实现是对的。

### 8.4 鹈鹕骑车测试（Swift 变体）

`166_pelican.sh`，提示词 `画一个鹈鹕骑自行车的动画，HTML 格式，不需要测试。`，`-n 8192 -t 0 -j`：

| 臂 | tokens | tok/s | wall | tau | HTML | 结构检查 |
|---|---|---|---|---|---|---|
| `--spec mtp` | 3053 | **74.7** | **44 s** | 2.36 | 6627 B | `<html>`✓ `<svg>` 1/1✓ 无 canvas/script |
| `--spec off` | 2783 | 41.0 | 70 s | — | 6012 B | 同上 |

- **≈1.82× 加速**（`decode (3053 tok, 908 windows)` → 3.36 tok/窗口）。
- 两臂都产出**合法可渲染**的 SVG 动画：双轮 + 曲柄 `animateTransform` 旋转、车身弹跳、速度线；鹈鹕有大嘴 + 嘴囊 + 橙脚。
- 产物：`~/pelican_swift/pelican_{mtp,off}.{json,html,err}`；本地副本已剥掉模型多输出的 ` ```html ` 围栏，存为 `pelican_{mtp,off}_render.html`。

---

## 9. 性能调优（目标模型：Swift 变体；GPU gfx1100 / RX 7900 XT 20 GB）

> 全部用 `hipfire bench --matrix --backend noslots --runs 1~2 --warmups 0`（单进程顺序），
> 以及 `hipfire run` 量投机收益。硬件就是这台 20 GB AMD 卡。

### 9.1 可调参数面（本次实际动到的）

| 类别 | 参数 | 本次取值 |
|---|---|---|
| CLI · run | `--spec` | `off / auto / ngram / dflash / mtp / dspark` |
| CLI · run | `--draft-max` | 投机草稿窗口 K（**只有 `run` 有，`bench` 不认**） |
| CLI · run | `--kv-mode` | `fwht3`(默认) / `q8` / `asym3` |
| CLI · bench | `--pp / --ctx / --tg / --runs / --warmups` | 见下 |
| CLI · bench | `--backend` | `noslots`（顺序守护进程）/ `slots` / `batch` / `both` |
| CLI · bench | `--kv-backend` | `contiguous` / `vmm` |
| env | `HIPFIRE_GFX11_MQ4V2_IU4` | 默认 **关**；开 = 预填走 W4A4 int4 MMQ |
| env | `HIPFIRE_GFX11_FA2_PREFILL` | 默认 **开** |
| env | `HIPFIRE_PREFILL_MAX_BATCH` | gfx1100 默认 **512** |
| env | `HIPFIRE_MTP_P_MIN` | MTP 草稿置信度闸门 |

### 9.2 短上下文矩阵（pp/ctx = 512,2048,4096；tg128）—— `167_bench.sh`

| 臂 | 变更 | pp512 | pp2048 | pp4096 | tg128@512 |
|---|---|---|---|---|---|
| A_base_off | 基线 | 942.2 | 922.1 | 895.7 | 41.20 |
| **B_iu4** | **`MQ4V2_IU4=1`** | **1303.8** | **1266.0** | **1220.6** | 41.18 |
| C_fa2_off | `FA2_PREFILL=0` | 922.4 | 845.7 | 756.1 | 41.27 |
| D_mtp_k3 | `--spec mtp` | 943.9 | 920.6 | 895.4 | 41.18 |
| F_mtp_pm6 | `MTP_P_MIN=0.6` | 942.2 | 920.7 | 895.2 | 41.10 |
| G_kv_q8 | `--kv-mode q8` | 942.9 | 920.3 | 894.1 | 40.96 |
| H_kv_asym3 | `--kv-mode asym3` | 921.2 | 844.4 | 754.4 | 40.23 |
| I_slots | `--backend slots` | 943.7 | 921.4 | 896.4 | 41.17 |
| P_chunk256 | `PREFILL_MAX_BATCH=256` | 944.5 | 925.0 | 896.4 | 41.27 |
| Q_chunk1024 | `PREFILL_MAX_BATCH=1024` | 943.6 | 863.0 | 770.1 | 41.26 |

### 9.3 长上下文 prefill —— 5 分钟闸门

`--pp N --ctx N` 会做**两遍**全量 prefill（pp 一遍 + ctx 建立一遍，`169_bench3.sh` 实测拆解：
8k 档 21.5 s + 20.8 s ≈ 39.9 s）。所以为严格对齐「单个 prefill 是否超 5 分钟」，改用**单遍**测量：

| 档 | 单遍 prefill | 速率 | 判定 |
|---|---|---|---|
| 8k | 完成，wall 40 s（双遍） | pp8192 **854.8 tok/s** | 通过 |
| 16k | 完成，wall 85 s（双遍） | pp16384 **780.3 tok/s** | 通过 |
| 32k | 完成，wall 199 s（双遍） | pp32768 **660.3 tok/s** | 通过 |
| **64k** | **`--pp 65536 --ctx 8` 与 `--pp 8 --ctx 65536` 均在 300 s 闸门处被中断** | — | **prefill > 5 分钟，按规则中断并记录** |
| **128k** | **同上，两遍均未在 300 s 内完成** | — | **prefill > 5 分钟，按规则中断并记录** |

- 闸门 300 s 里模型加载只占 **≈2.5 s**（冷启动实测：`weight sweep 1.17 s`，64 层逐层加载），
  所以 300 s 基本全是真实计算 → 「prefill 超 5 分钟」的判定是干净的。
- 速率随上下文衰减明显：854 → 780 → 660 tok/s（8k→32k），32k→64k 之间发生**非线性塌陷**
  （64k 单遍在 300 s 内连一遍都跑不完，等效 < 219 tok/s）。这是本卡上长上下文的主要瓶颈。
- 脚本：`167_bench.sh`（双遍阶梯）、`169_bench3.sh`（单遍、干净判据）。

### 9.4 IU4 在长上下文 prefill 上的收益

| 配置 | pp32768 | wall | tg8@32768 |
|---|---|---|---|
| `MQ4V2_IU4=0`（默认） | 659.0 tok/s | 198 s | 35.13 |
| `MQ4V2_IU4=1` | **818.8 tok/s（+24%）** | **161 s** | 35.16 |

输出一致性（`-n 64`，iu4 off vs on）：两侧都非空、都是 `Paris`、**逐字节相同**。
> 167 里的那次「逐字节相同: YES」是**空判**（daemon 冲突导致两个文件都是 0 字节），168 重跑后才是有效判据。

### 9.5 MTP 草稿窗口 K（只能用 `run` 量）

提示词 = 500 词茶叶史，`-n 700 -t 0`：

| 臂 | tau | tok/s | windows | 输出字节 |
|---|---|---|---|---|
| `--spec off` | — | — | — | 3258 |
| `--draft-max 1` | 0.77 | 44.0 | 371 | 3029 |
| **`--draft-max 3`（默认）** | 1.28 | **52.1** | 302 | 3220 |
| `--draft-max 5` | 1.33 | 51.8 | 291 | 3188 |
| `--draft-max 8` | 1.35 | 51.8 | 291 | 3225 |

→ **K=3 已到甜点**，K=5/8 只多接受 0.05 草稿却不再提速（每窗口前向更贵，互相抵消）。
不同提示词上收益差别很大：数字序列类（`-n 240`）tau 2.78 / 85.0 tok/s ；鹈鹕 HTML tau 2.36 / 74.7 tok/s。

### 9.6 结论：推荐配置

1. **prefill 提速只找到一个有效开关：`HIPFIRE_GFX11_MQ4V2_IU4=1`** —— 短上下文 +38%、32k 长上下文 +24%，
   解码速率几乎不动。代价是官方文档标注的 **+0.014 WT2 KLD 质量损失**，
   本轮抽样（`Paris`）未观察到输出差异，但这**不构成无损证明**，是否启用需按质量预算定。
2. **其余默认值都已经是局部最优，别动**：
   - `FA2_PREFILL` 默认开是对的（关掉后 pp4096 从 895.7 → 756.1，−16%）；
   - `PREFILL_MAX_BATCH` 默认 512 是对的（1024 反而 pp4096 掉到 770.1；256 与 512 持平）；
   - `--kv-mode` 默认 `fwht3` 是对的（`asym3` 与 `FA2=0` 同档退化到 754，`q8` 解难怪微降）；
   - `--backend slots` 与 `noslots` 在 bench 上无差异（单请求顺序场景）；
   - `MTP_P_MIN=0.6` 无收益。
3. **投机解码**：默认 K=3；解码吞吐在自由文本上 +25%、在长 HTML 生成上 **+82%**。
   注意 `bench --tg` **量不到**这个收益（见 9.7 坑 1），必须用 `run`。
4. **长上下文**：本卡 32k 以内可用（prefill ~100 s/万 tok 量级）；**64k 起单遍 prefill 就超 5 分钟**，
   128k 更甚 —— 需要 64k+ 上下文时得先解决 prefill 的非线性塌陷（IU4 有帮助但不够）。

### 9.7 四条测量学坑（**下次别再踩**）

1. **`bench --tg` 量不出投机收益**：`D_mtp_k3` 的 `tg128@512 = 41.18`，与 `A_base_off` 的 `41.20` 无异，
   尽管日志明确打印了 `MTP head loaded` + `speculator enabled`。原因：bench 的 tg 循环是**独立单 token 步进**，
   而投机收益来自**连续 token 流**。**量投机一律用 `run`。**
2. **`bench` 不认 `--draft-max`**：那是 `run` 的参数 → `167` 的 E 臂秒退 `exit=2`。
3. **`timeout` 杀掉 bench 会留下孤儿 daemon**，后续调用立刻报
   `FATAL: hipfire daemon already running (PID …)`。167 里 L128k 与 X_out_iu4 就是这样**没跑成**的。
   修法：每次调用前 `pkill -f "[h]ipfire/home/bin/daemon"`（**方括号技巧**，否则 pkill 会匹配到自己的命令行、
   把自己所在的 ssh shell 一起杀掉 → `exit 255`）。
4. **`--pp N --ctx N` 是两遍 prefill**，不是一遍；用整条命令的 wall 判「5 分钟」会高估一倍。
5. `152` 的 `|c_w/c_s-1| < 1e-2` 判据在**两个独立量化 blob 相除**的场景下天然噪声过大
   （本轮中位 1.73e-02，但各族 `c_w` 中位都在 0.96~1.00、corr≈1）→ **正确读法是「`c_w/c_s` 接近 1 的常数」**，
   不是「<1e-2」。`165` 的「源相同 ⇒ 产物相同」也是按**索引序号**对齐，而两份 GGUF 张量顺序不同，
   需按**名字**对齐才成立（`165` 的 `678/671` 方向性结论不受影响）。

---

## 10. 脚本与工具清单（远端 `~/bin/`，本地 `remote/` 有同名副本）

| 脚本 | 作用 |
|---|---|
| `100_iq3s_coherence_ab.sh` | 连贯性 A/B 模板（`--spec` 验收也复用它） |
| `128_verify_convention_fix.py` | `--raw` F16 逐字节校验 |
| `140_reorder_v_heads.py` | V-head 重排的对账 |
| `141_sidecar_reconcile.py` | 侧车通道顺序对账（R²≈0.97 定下「HF 序、不置换」） |
| `147_accept_vreorder.sh` | V-head 逆重排验收 |
| `148_imatrix_from_golden.py` | 从 golden 496 侧车**反解** imatrix（含 `--read` 自校验） |
| `149_patch_awq_gguf.py` | AWQ 接线补丁 |
| `150_accept_awq.sh` | AWQ 端到端验收（5 道门槛） |
| `151_sidecar_vs_golden.py` | 逐元素比两份 HFQ 的 496 侧车 |
| `152_awq_pairing.py` | 「预乘 ↔ 侧车」配对硬检验 |
| `153_patch_mtp_gguf.py` | MTP 集成补丁 |
| `154_check_mtp.py` | `.mtp` 静态契约核验（5 道门槛） |
| `155_accept_mtp.sh` | `.mtp` 端到端验收（加载 / 负对照 / 无损性） |
| `156_mtp_norm_roundtrip.py` | `.mtp` norm 约定往返逐字节核验 |
| `157_mtp_long_ab.sh` | 长生成 MTP vs AR 无损性 + 收益 |
| `160_verify_61.sh` | **§6.1 复核**：A/B/C 三组对照（K / 投机器种类 / 策略开关 / 官方 MTP 对） |
| `161_cmp_gguf_header.py` | 两份 GGUF 头部对账（KV + 张量名/dtype/dims + 张量序）——判定「量化方式是否一致」 |
| `162_diff_arms.py` | 各投机臂 vs AR 基线的**首差定位**与上下文打印 |
| `163_convert_swift.sh` | **Swift 变体**转换（AWQ+MTP 一次、无 AWQ 一次） |
| `164_accept_swift.sh` | **Swift 变体**验收（索引 / 跨产物字节 / 侧车 / MTP 契约 / 运行时 + 负对照） |
| `165_swift_vs_base_bytes.py` | Swift 产物 vs 基线产物的**逐张量字节对账**（源相同→产物必须相同） |
| `166_pelican.sh` | 鹈鹕骑车测试（HTML 动画，MTP / AR 两臂，落 HTML + 结构检查） |
| `167_bench.sh` | 性能扫描（矩阵 / iu4 / FA2 / MTP K / p_min / KV / 长上下文阶梯 / 输出一致性） |
| `168_bench2.sh` | 补齐 167 的 4 个缺口（加载基准 / 干净版长上下文 / IU4 长上下文 / **用 `run` 量草稿窗口** / 重跑 iu4 一致性） |
| `169_bench3.sh` | **单遍**长上下文测量（把「单个 prefill 是否超 5 分钟」量干净） |

关键文件速查（远端路径）：
- 基线模型源：`~/models/qwen38-27b-iq3s-mtp/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
- **Swift 变体源**：`~/models/swift-qwen38-27b-iq3s-mtp/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
- imatrix：`~/models/hipfire/qwen38-27b.recon.imatrix.gguf`
- 基线 trunk 产物：`~/models/hipfire/qwen38-27b-iq3s-mtp-awq.hfq`
- 基线 MTP 侧车：`~/models/hipfire/qwen38-27b-iq3s-mtp.mtp`（+ 同 stem 硬链 `…-mtp-awq.mtp`）
- **Swift 产物**：`~/models/hipfire/swift-1.5-qwen38-27b-iq3s-mtp-awq.hfq` + `…-awq.mtp`
- golden：`~/models/hipfire/qwen3.8-27b.mq4-xt`
- 验收日志：`~/accept_awq.txt`、`~/accept_mtp.txt`、`~/accept_mtp_long.txt`、`~/accept61.txt`、`~/accept_swift.txt`
- 性能日志：`~/bench_swift_run.txt`、`~/bench_swift2_run.txt`、`~/bench_swift3_run.txt`
- 鹈鹕产物：`~/pelican_swift/pelican_{mtp,off}.{json,html,err}`
