# OddFp8 evidence index (NEVER git-add scratch-2026-09-17/)

## Predicates changed (all: drop N%64 admission; kernels mask partial N tiles
## natively — clamped loads, `oc < N` guarded stores, single TU
## kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip)

- crates/rdna-compute/src/dispatch.rs: fp8_stream_active: `batch >= 64 &&
  batch % 64 == 0` → `batch >= 64` (producer grid is [batch], row-parallel)
- crates/rdna-compute/src/gemm.rs:
  - gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8{,_prepared} entries: guard → N>0
  - gemm_qkv_hfq4g256_wmma_gfx12_mq4v2_fp8{,_prepared} entries: guard → N>0
  - gemm_gate_up..._bt12_prepared{,_lloyd}: deleted %64 reject
  - gemm_qkvza/qkv..._prepared_lloyd: deleted %64 reject
  - gemm_hfq4g256_residual..._fp8 entry: guard → M>0,N>0; intercept dropped %64
  - qkvza/qkv/gate_up router intercepts: dropped `batch_size % 64 == 0`
- docs: dispatch.rs + prefill.rs + v2-selection comments (N%64 → native masking)

## prof1813 top-6 (rocprofv3 kernel-trace, GRAPH=0, CHUNK=8192, IU4=0)
BEFORE (base c7bf7bb00): gate_up_bt12 33.6 / residual_bt12 24.8 /
  qkvza_bt12 22.3 / qkv_fp8_gfx1201_bt8 6.7 / gdn_q8_fast 4.5 /
  fused_silu_mul_mq_rotate_awq 2.0 (+convert_f32_to_f16 1.8)
  copies: 184 / 0.5ms
AFTER: gate_up_fp8_v2 33.5 / residual_fp8_v2 24.3 / qkvza_fp8_v2 11.8 /
  gdn_q8_fast 9.9 / silu_awq 4.2 (parity, iu4-only fusion) / qkv_fp8_v2 3.6
  (+pack 2.9, fused_fp8_producer 1.9, FA_packet 1.6)
  copies: 184 / 0.5ms (temp-pad revision had 2456 / 277ms — deleted)

## matrix (daemon, GRAPH=1, IU4=0, runs=3 warmups=2, medians)
pp277 1414 (base 800) / pp512 1885 (1869) / pp1813 1845 (884) /
pp4096 1909 (1908) / pp5909 1843 (872) / pp8192 1874 (1874)
pp1813: 2.1% under pp512, 3.3% under pp4096 (avg-neighbour 2.7%)
pp5909: 3.5% under pp4096, 1.7% under pp8192 (avg-neighbour 2.6%)

## KLD fp8v2 (ab1, GRAPH=0, IU4=0)
c1:   mine F8592E39 (KLD 0.030084) vs ref D1078C0F (0.029194) — odd N=1023
  moved F16-production → E4M3-fp8 (precision change, cf. iu4 re-pin 8c1265e50)
  fused-odd == monolithic-odd bit-exact (stream-off run identical)
c24:  mine FD44FE56 (KLD 0.049236) vs ref F4AF3430 (0.046960) — chunk24 N=1000
  (15x64+40) same precision move; chunks 1-23 (all 1024) identical paths
c23 (even-only 23x1024): base-vs-mine md5 PENDING — must match bit-exactly

## ttft PENDING (before/after, ttft_5900, runs 8 warmups 2 json)
## iu4 sanity PENDING (flag unset, pp5909/8192 — must equal base rows)
