# Qwen3.8-27B MQV2 — gfx1100 product-tier throughput refresh

**Date:** 2026-08-26  
**Lifecycle:** `historical`  
**Disposition:** **measured product-card refresh evidence** for the exact fixture below. This is not a performance floor, admission gate, current-default baseline, or transferable result. Newest file != current baseline.

## Scope

Refresh the missing MQ5V2 and MQ6V2 XT/Base/Pro rows in the fixed-shape Radeon RX 7900 XTX product table after the gfx1100 multi-wave prefill route was extended beyond MQ4V2. The pre-run expectation was at least 800 PP512 tok/s for every row; the measured Pro rows did not meet that expectation and are retained without rounding or substitution.

## Fixture

| field | value |
|---|---|
| Host | `hipx` |
| Device | HIP device `0`, Radeon RX 7900 XTX, `gfx1100`, BDF `0000:66:00.0` |
| HIP / ROCm | 7.14 |
| Runtime revision | `363e93d771ff` |
| `hipfire` md5 | `fc4464ddc77d39a45e54d7aa5dd1dced` |
| `daemon` md5 | `3d75caedd7229bbe02d4ba7144c8dbb5` |
| Speculation | off |
| Backend / workload | `noslots` / `stateless` |
| KV | Q8 / VMM |
| Shapes | PP512; TG128 from initial context 128 |
| Sampling | 3 warmups, 5 measured samples, one guarded fresh process per product |
| Prompt md5 | N/A — synthetic fixed-shape matrix protocol |

Command shape for every accepted row:

```bash
HIP_VISIBLE_DEVICES=0 hipfire bench MODEL \
  --matrix --pp 512 --ctx 128 --tg 128 \
  --runs 5 --warmups 3 --json \
  --spec off --backend noslots --workload stateless \
  --kv-mode q8 --kv-backend vmm
```

Each command ran through the BDF-specific GPU guard. The first MQ5-XT attempt stalled while the device was blocked and produced no samples; it was terminated with `gpukill` and is excluded. After the device was unblocked, the accepted MQ5-XT retry and all five remaining rows exited successfully through the guard.

## Artifact identity and medians

| Product | measured artifact | artifact md5 | PP512 tok/s | TG128@128 tok/s |
|---|---|---|---:|---:|
| mq5-xt | `qwen3.8-27b.mq5v2.xt.hfq` | `e7c17382812f33df90f5138eb7cf9973` | **814.90** | **40.32** |
| mq5 | `qwen3.8-27b.mq5v2.base.hfq` | `cff5d051ce8a2a44a5eba6b9b43d595d` | **803.30** | **38.95** |
| mq5-pro | `qwen3.8-27b.mq5v2.pro.hfq` | `50119af7b6974320e574774b663d2f6d` | **767.50** | **37.92** |
| mq6-xt | `qwen3.8-27b.mq6v2.xt.hfq` | `de6eb059a10577b80f0a162e5c89249e` | **803.90** | **27.34** |
| mq6 | `qwen3.8-27b.mq6v2.base.hfq` | `ae18a5d9e7926474064586387126aa5c` | **797.00** | **26.88** |
| mq6-pro | `qwen3.8-27b.mq6v2.pro.hfq` | `aac76dcc771b424d605e813b0b4aa1c1` | **759.60** | **27.75** |

Three of six rows are below the stated 800 tok/s expectation if the threshold is interpreted strictly: MQ5-Pro, MQ6 Base, and MQ6-Pro are below 800, while MQ5-XT, MQ5 Base, and MQ6-XT are above it. The table reports the measurements as observed.

## Raw five-sample arrays

### PP512 tok/s

| Product | samples | median |
|---|---|---:|
| mq5-xt | 814.0, 814.5, 816.6, 814.9, 815.3 | 814.90 |
| mq5 | 803.8, 803.3, 803.3, 804.7, 801.6 | 803.30 |
| mq5-pro | 766.5, 768.6, 769.4, 767.5, 765.3 | 767.50 |
| mq6-xt | 800.3, 803.9, 803.2, 804.3, 804.2 | 803.90 |
| mq6 | 794.9, 796.4, 798.0, 798.1, 797.0 | 797.00 |
| mq6-pro | 755.8, 759.6, 759.6, 758.9, 760.2 | 759.60 |

### TG128@128 tok/s

| Product | samples | median |
|---|---|---:|
| mq5-xt | 40.310208, 40.361449, 40.318858, 40.266205, 40.325442 | 40.318858 |
| mq5 | 38.950993, 38.976836, 38.946206, 38.946526, 38.965484 | 38.950993 |
| mq5-pro | 37.898342, 37.918452, 37.918984, 37.851069, 37.941624 | 37.918452 |
| mq6-xt | 27.337083, 27.336227, 27.351454, 27.318038, 27.344321 | 27.337083 |
| mq6 | 26.876205, 26.864568, 26.880447, 26.878828, 26.856155 | 26.876205 |
| mq6-pro | 27.735741, 27.775141, 27.750269, 27.746947, 27.752171 | 27.750269 |

## Machine-local raw evidence

Raw combined stdout/JSON logs are retained on `hipx` under:

`/home/kaden/qcal/modelcard-refresh/20260826-gfx1100-mw-refresh/`

Accepted log files are `mq5-xt-retry2.log`, `mq5-base.log`, `mq5-pro.log`, `mq6-xt.log`, `mq6-base.log`, and `mq6-pro.log`. These host-local paths are discovery pointers; the fixture identity, medians, and full measured arrays are preserved above as the durable in-tree record.
