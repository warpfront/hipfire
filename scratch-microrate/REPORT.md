# gfx1201 WMMA micro-rate measurement

## Verdict

**No: iu8 is not rate-equal to fp8 at every measured occupancy.** At 3 blocks/CU, iu8 was **1.468% faster** on the mean paired comparison (median +1.355%, all 8 pairs positive), narrowly outside the 1.3% drift envelope. At 8 blocks/CU the two are rate-equal within that envelope (iu8 +0.813% mean); 1 block/CU is inconclusive because the ordering effect is larger than the mean difference.

## Results

Card-C was selected only through UUID `GPU-085289909a86cc63`. Each row is the mean of 8 fresh-process results in ABBA order (forward, reverse, reverse, forward, repeated). A separate process was discarded as a warm-up. Within each process, every cell had one untimed launch followed by 7 individually event-timed launches; the process result is their median. `spread` is `(max - min) / mean` across the 8 process medians. Ratios are the mean of the 8 within-process ratios against f16 at the same occupancy.

The issue column is the achieved aggregate cadence per CU, derived from the emitted instruction shape: 8,192 operations for f16/iu8/fp8 and 16,384 for iu4. It is throughput cadence, not single-instruction latency.

| Dtype | Blocks/CU | Achieved | f16 ratio | WMMA cadence (ns/instruction/CU) | n processes | Spread |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 1 | 169.753 TFLOPS | 1.000x | 1.5443 | 8 | 3.026% |
| iu8 | 1 | 340.073 TOPS | 2.004x | 0.7708 | 8 | 7.376% |
| fp8 E4M3 | 1 | 338.577 TFLOPS | 1.995x | 0.7743 | 8 | 3.022% |
| iu4 | 1 | 685.791 TOPS | 4.040x | 0.7645 | 8 | 4.606% |
| f16 | 3 | 179.426 TFLOPS | 1.000x | 1.4610 | 8 | 3.079% |
| iu8 | 3 | 348.490 TOPS | 1.943x | 0.7522 | 8 | 4.715% |
| fp8 E4M3 | 3 | 343.426 TFLOPS | 1.914x | 0.7633 | 8 | 2.102% |
| iu4 | 3 | **695.871 TOPS** | 3.879x | 0.7534 | 8 | 2.884% |
| f16 | 8 | 192.243 TFLOPS | 1.000x | 1.3636 | 8 | 4.483% |
| iu8 | 8 | 365.628 TOPS | 1.903x | 0.7170 | 8 | 2.462% |
| fp8 E4M3 | 8 | 362.714 TFLOPS | 1.888x | 0.7227 | 8 | 4.288% |
| iu4 | 8 | 734.014 TOPS | 3.821x | 0.7143 | 8 | 5.116% |

### Paired iu8 versus fp8

Positive means iu8 was faster. The range is across the 8 fresh processes.

| Blocks/CU | Mean delta | Median delta | Pair range | Interpretation |
|---:|---:|---:|---:|---|
| 1 | +0.443% | -0.198% | -2.817% to +5.428% | Inconclusive: the forward/reverse order effect dominates. |
| 3 | **+1.468%** | **+1.355%** | +0.035% to +2.783% | Not equal by the stated 1.3% envelope; iu8 is faster. |
| 8 | +0.813% | +0.579% | -0.411% to +2.206% | Rate-equal within the stated envelope. |

The two four-process ABBA blocks gave iu8-minus-fp8 means of +1.623% and +1.313% at 3 blocks/CU, and +0.653% and +0.973% at 8 blocks/CU. At 1 block/CU they changed sign (+1.683%, -0.797%), which is why that occupancy is not called either way.

## iu4 anchor

The matched 3-block/CU row is **695.871 TOPS**, just **+0.113%** from the prior 695.085-TOPS anchor, so the operation count and launch geometry pass the requested sanity check. The 8-block/CU row reached 734.014 TOPS versus the prior 690.040 TOPS (+6.373%); its 5.116% fresh-process spread and the known occupancy ordering inversion make that row less stable, but it does not disturb the matched 3-block anchor.

## Loop and ISA proof

Source: `wmma_rate.hip`. Saved compiler ISA: `wmma_rate-hip-amdgcn-amd-amdhsa-gfx1201.s`.

Every kernel has 8 independent register-resident accumulator chains. One loop trip emits 8 consecutive WMMA instructions, so a chain's recurrence distance is 8 WMMAs. The only other steady-loop instructions are scalar decrement, scalar delay, compare, and branch: 4 scalar control instructions per 8 WMMAs. There are no global or LDS operations in any timed loop. Each wave executes 10,000 trips, or 80,000 WMMAs. Global traffic occurs only after the loop to make the accumulators observable.

The emitted steady loops quote as follows (the six same-opcode lines between the shown first/last lines are consecutive in the saved ISA):

```text
# f16, ISA lines 64-75
v_wmma_f32_16x16x16_f16 v[1:8], v[65:68], v[69:72], v[1:8]
v_wmma_f32_16x16x16_f16 v[9:16], v[65:68], v[69:72], v[9:16]
... 6 more consecutive v_wmma_f32_16x16x16_f16 ...
s_add_co_i32 s2, s2, -1
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_lg_u32 s2, 0
s_cbranch_scc1 .LBB0_2

# iu8, ISA lines 277-288
v_wmma_i32_16x16x16_iu8 v[41:48], v[65:66], v[67:68], v[41:48] neg_lo:[1,1,0]
v_wmma_i32_16x16x16_iu8 v[57:64], v[65:66], v[67:68], v[57:64] neg_lo:[1,1,0]
... 6 more consecutive v_wmma_i32_16x16x16_iu8 ...
s_add_co_i32 s2, s2, -1
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_lg_u32 s2, 0
s_cbranch_scc1 .LBB1_2

# fp8 E4M3, ISA lines 586-597
v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[67:68], v[49:56]
... 6 more consecutive v_wmma_f32_16x16x16_fp8_fp8 ...
s_add_co_i32 s2, s2, -1
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_lg_u32 s2, 0
s_cbranch_scc1 .LBB2_2

# iu4, ISA lines 902-913
v_wmma_i32_16x16x32_iu4 v[1:8], v[65:66], v[67:68], v[1:8] neg_lo:[1,1,0]
v_wmma_i32_16x16x32_iu4 v[9:16], v[65:66], v[67:68], v[9:16] neg_lo:[1,1,0]
... 6 more consecutive v_wmma_i32_16x16x32_iu4 ...
s_add_co_i32 s2, s2, -1
s_delay_alu instid0(SALU_CYCLE_1)
s_cmp_lg_u32 s2, 0
s_cbranch_scc1 .LBB3_2
```

All four intended opcodes are present. No cell is an opcode-absent failure.

## Reproduction

Build:

```sh
/opt/rocm/core/bin/hipcc --offload-arch=gfx1201 -O3 -save-temps wmma_rate.hip -o wmma_rate
```

Run under the card-C environment, alternating the optional order flag in `F R R F F R R F` order after one discarded process:

```sh
HOME=/home/kaden/.hipfire-homes/ab2 \
ROCR_VISIBLE_DEVICES=GPU-085289909a86cc63 \
HIP_VISIBLE_DEVICES=GPU-085289909a86cc63 \
HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels \
HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models \
HIPFIRE_GRAPH=1 \
HIPFIRE_DAEMON_BIN=/home/kaden/ClaudeCode/warpfront/wt-microrate/target/release/daemon \
./wmma_rate [--reverse]
```

The 8 raw process logs are `final-rep{1..8}-*.log`; `final-warm.log` is the discarded warm-up process.
