# iu4port baselines (pre-change, kernel @44146fc77: 16-row x 256-col, 4 waves, W-only LDS tile)
# Session: 2026-09-18, ordinal 1 (ROCR_VISIBLE_DEVICES=1, HOME=ab1)
# Fixture: qwen3.8-27b.mq4-xt sha256 verified (check_fixture.sh --sha OK)
# Oracle binary: target/release/examples/tmp_iu4_gfx12_oracle (HEAD build)
# Correctness: 9/9 PASS, cpu_bitwise=OK mism=0, repeat_identical=OK mism=0 (see baseline-oracle.log)

## TIME=1 medians, N=512 (isolated GEMM us/call)
| case | M | K | N | mode | med us/call | GFLOP | TOPS (% of 539.7 K32 peak) |
|---|---|---|---|---|---|---|---|
| gate | 17408 | 5120 | 512 | set | 1120.6 | 91.27 | 81.4 (15.1%) |
| gate | 17408 | 5120 | 512 | add | 1266.2 | 91.27 | 72.1 (13.4%) |
| down (residual) | 5120 | 17408 | 512 | add | 1239.1 | 91.27 | 73.7 (13.6%) |
| qkvza-synth | 16480 | 5120 | 512 | set | 1014.3 | 86.40 | 85.2 (15.8%) |
| qkv-synth | 14336 | 5120 | 512 | set | 924.2 | 75.16 | 81.3 (15.1%) |

## Reference points (other sessions, labels only)
- iu4 v1 NB4 (21630fba1 msg): gate 1150/1193, down 1112, qkvza 1064, qkv 931
- fp8 s2bt8 hist (V/oracle-n512-v2/results.json): gate_up 1804.7, qkv 770.1, qkvza 825.7, residual 676.8
- Admission for the port (assignment): <= 0.5x same-session fp8 s2bt8 per row.
