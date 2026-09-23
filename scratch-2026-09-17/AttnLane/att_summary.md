# ATT attribution: incumbent packet kernel

## Provenance

- Source revision: `c7bf7bb006bdcc768ae5a59d2c0ced7dd583a6aa`
- Source: `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`
- Source SHA-256: `bc7b00da474fd0ef0562a1d350078b11576f458790231f428e61be768295f3c2`
- Captured code object: `/home/kaden/.hipfire-homes/ab4/.hipfire_kernels/gfx1201/attention_fp8_e4m3_fa2_gqa_packet_gfx1201.fcab03e8e867d730.hsaco`
- Code-object SHA-256: `0ed79d32134add3e2217a472e2c2a69b4303d3f3450dac4e715c462ab6d8812c`
- Captured symbol: `attention_fp8_e4m3_fa2_gqa_packet_gfx1201`
- ATT dispatch: agent 2553, dispatch 788, code-object id 13
- Decoded CSV: `att/stats_ui_output_agent_2553_dispatch_788.csv`
- Decoded CSV SHA-256: `296b9b0f2a319bb8d8e4096163f021148314510825740411d68bba6ca50c7088`
- Compiler: HIP 7.15.26333, AMD clang 23.0.0git, LLVM `8f497e0992fb7513f7f78a6f6b6f1056c375e961`
- GPU ordinal: 4
- Workload: qwen3.8:27b-mq4-xt, prefill 8192, fp8 KV, graph disabled
- Capture command: `rocprofv3 --att --att-target-cu 0 --kernel-include-regex 'attention_fp8_e4m3_fa2_gqa_packet' -d scratch-2026-09-17/AttnLane/att -o att -- target/release/examples/profile_prefill_qwen35 /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt --prefill 8192 --kv-mode fp8`
- Environment: `HOME=/home/kaden/.hipfire-homes/ab4 ROCR_VISIBLE_DEVICES=4 HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab4/.hipfire_kernels HIPFIRE_GRAPH=0 HIPFIRE_LLOYD_GFX12=1`

## Hot-loop boundary

The attributed steady packet-tile loop is the contiguous ISA interval `[0x9788, 0xb058]` (decimal `[38792, 45144]`). It begins at the tile-latch wait/barrier and ends at the branch returning from the fourth softmax/PV recurrence. The interval contains 1,020 emitted instruction rows. Dynamic counts validate the KT64 boundary: the active interior path records 6,222 tile-wave iterations, the four-subtile recurrence records 24,798 executions (the difference from exactly four per tile is the masked tail), and WMMA issue records 793,536 executions, or 127.54 per observed tile-wave.

## Required mutually exclusive attribution

The denominator is the sum of all instruction `Latency` and `Idle` samples in the loop: 119,719,185 samples. Classification order is waits, barriers, WMMA, other VALU, LDS issue, global-memory issue, then SALU/control/fence; therefore every emitted instruction belongs to exactly one row. Idle is reported separately.

| Class | Samples | Share |
|---|---:|---:|
| Pure global-load wait | 18,482,962 | 15.44% |
| Pure LDS wait | 10,732,564 | 8.96% |
| Mixed global-load + LDS wait | 16,513,091 | 13.79% |
| Barrier signal/wait | 9,305,460 | 7.77% |
| WMMA issue | 7,841,242 | 6.55% |
| Non-WMMA VALU | 15,661,447 | 13.08% |
| LDS instruction issue | 2,128,127 | 1.78% |
| Global-memory instruction issue | 1,068,712 | 0.89% |
| SALU/control/fence | 2,430,347 | 2.03% |
| Idle | 35,555,233 | 29.70% |
| **Total** | **119,719,185** | **100.00%** |

The loop contains 51,765,712 explicit stall samples: 61.51% of instruction latency and 43.24% of the latency-plus-idle denominator.

## Top stall instructions

| PC | Instruction | Stall samples | Stall share | Denominator share |
|---|---|---:|---:|---:|
| `0xab8c` | `s_wait_loadcnt_dscnt 0x301` | 16,134,268 | 31.17% | 13.48% |
| `0x9878` | `s_wait_loadcnt 0x2` | 2,010,757 | 3.88% | 1.68% |
| `0x9c04` | `s_wait_loadcnt 0x0` | 1,880,653 | 3.63% | 1.57% |

The dominant mixed wait is inside the QK/softmax recurrence, not the K/V fill barrier. The remaining leading pure-load waits are in the K/V tile fill. The two explicit barrier-wait PCs contribute little direct `Stall`, but their execution latency remains visible in the barrier attribution row.

## Normalized dynamic instruction footprint

This is a census of emitted instruction hits from the captured binary and the stated loop interval, normalized by 6,222 active tile-wave iterations. Each incumbent wave owns 16 query-head rows, and each tile spans 64 keys.

| Measure | Value |
|---|---:|
| Emitted instruction hits | 14,703,242 |
| Instructions / tile-wave | 2,363.105 |
| Instructions / query-head row / KT64 | 147.694 |
| Instructions / query-head row / key | 2.30772 |
| WMMA / tile-wave | 127.537 |
| Global-memory issues / tile-wave | 82.646 |
| LDS issues / tile-wave | 201.696 |

The small fractional departures from the static full-tile values reflect masked edge paths present in the real producer-input capture. No synthetic steady-state weighting was substituted.

## Compiler resource receipt

A fresh source-to-ISA compile is in `baseline-resources.log`; the assembly is `baseline-packet.s` (SHA-256 `e31ce165a8f7c935576f667b05defb99f0263bcaa911fa772f29f5247ac323c0`). The direct and partial packet entries each compile with 237 live VGPRs, zero scratch, zero SGPR spills, zero VGPR spills, and compiler-reported occupancy of six waves/SIMD. The captured executable metadata rounds the direct entry to 240 allocated VGPRs. Dynamic LDS at launch is 49,408 bytes.
