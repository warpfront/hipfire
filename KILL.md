# Producer body optimization — KILL

Card-C (`GPU-085289909a86cc63`), gfx1201, symmetric mq4 XT artifact, RTN producer build (`IU4_A4_CANDIDATES=1`). Baseline source is `96ec01887`.

## Decision

KILL. None of the evaluated producer-body changes met the shipment gate. The only byte-exact reduction change lowered the traced producer sum by 0.081%, left it at 29.508 us/token rather than the 28 us/token target, had no reproducible TTFT benefit, and produced a -1.03% pp8192 row in the first paired matrix run. No code change is retained.

## In-daemon ATT: `fused_silu_mul_mq_rotate_awq_i4_gfx12`

`rocprofv3 --att --att-target-cu 0 --kernel-include-regex ...` captured dispatch 503 on CU0. The decoded trace contains 3,147 dynamic wave hits. Strict wave time is latency plus idle; stall is reported separately.

| Phase / PC range | Latency | Idle | Strict wave time | Share | Stall | Stall share |
|---|---:|---:|---:|---:|---:|---:|
| setup + gate/up + SiLU + AWQ, 6400–9319 | 92,110,193 | 16,541,842 | 108,652,035 | 78.62% | 90,719,219 | 86.65% |
| FWHT, 9320–10423 | 11,373,869 | 2,237,101 | 13,610,970 | 9.85% | 10,857,761 | 10.37% |
| nullable F32 store, 10424–10471 | 0 | 0 | 0 | 0% | 0 | 0% |
| RTN quant/header/pack, 10472–13128 | 4,503,944 | 11,424,121 | 15,928,065 | 11.53% | 3,122,411 | 2.98% |
| total | 107,988,006 | 30,203,064 | 138,191,070 | 100% | 104,699,391 | 100% |

The dominant single instruction is the source-load `s_wait_loadcnt 0x3` at PC 6640: 61.666M latency and 61.663M stall. The FWHT's late load wait contributes about 10.1M. The nullable F32 store is dead on the hot IU4 route.

At I=17,408, gate+up traffic is 139,264 B/layer/token and 8,912,896 B/token across 64 layers. The packed `block_i4_128` output is 9,792 B/layer/token and 626,688 B/token across 64 layers. The compiler already emits ten `global_load_b128` instructions for gate/up, AWQ and signs in the static SiLU body.

## Lever results

| Lever | ISA/resource delta | Real-input byte identity | In-daemon result | Decision |
|---|---|---|---|---|
| (a) FWHT `v_permlane16/32` + DPP, no LDS | SiLU: static vector+DS 883→886, `ds_swizzle` 40→0, +16 DPP/+16 permlane16/+8 permlanex16, SGPR/VGPR 18/55 unchanged. RMS: 805→811 and SGPR 34→37. GDN: 690→694. | PASS: SiLU 1,253,376 block B; RMS 655,360 F32 + 368,640 block B; GDN 786,432 F32 + 442,368 block B. | rocprof producer time regressed: SiLU +1.915%, RMS +1.443%, GDN +1.738%. Paired TTFT median was -0.228% throughput-equivalent, with outliers. | KILL: LDS swizzle is faster here. |
| (b) one row/wave vs WG occupancy | SiLU is already one wave/WG (`block=32`, 55 VGPR, 16-wave/SIMD occupancy). RMS needs 8 waves/token for its byte-stable reduction tree. GDN needs two waves for the two-head normalize/handoff. No legal byte-exact occupancy increase exists without changing those ownership trees. | N/A: incumbent geometry retained. | 0 delta (structural no-op). | KILL: already at the useful geometry. |
| (c) explicit `float4`/b128 gate+up loads | `global_load_b128` stayed 10→10; SGPR/VGPR and occupancy stayed 18/55 and 16; static instructions 1152→1153 and one VOPD pair was lost. | N/A: rejected before launch at the ISA gate. | N/A: compiler already vectorizes the scalar source. | KILL: source spelling cannot improve the generated loads. |
| (d) RMS second-X-read fusion via K-sized LDS transpose | Static instructions 1085→1094; global b128 loads 16→12; VGPR 88→89; dynamic LDS 1,024→21,504 B; +1 `ds_store_b32`, +4 `ds_load_b128`. LDS caps residency at three 256-thread WGs/CU instead of the small-LDS path. | PASS: 368,640 RMS block bytes, MD5 `80764c1c021f99767e18f87826574640` for both arms. | RMS 7.2031→8.6643 us/token (+20.29%); producer sum 29.5076→30.9682 us/token (+4.95%). | KILL: saved global reads do not repay the LDS occupancy loss. |
| (e) RTN single-wave reductions, no candidate loop | Baseline already compiles only `d=amax/7`; candidate search code is removed by `IU4_A4_CANDIDATES=1`. A byte-exact DPP cut for XOR offsets 2/1 changed static instructions 1152→1145, `ds_bpermute` 36→28 and `s_wait_dscnt` 55→47, adding 8 DPP; 55 VGPR, 0 LDS/spills, occupancy unchanged. A full permlane reduction was rejected because it changed output bytes. | PASS for exact DPP cut on real N=128 producer inputs: RMS `80764c1c021f99767e18f87826574640`, GDN `9bf0e610114a1a24efbd93886dffbffa`, rotate `fbc43bedee0c2cf955cc050a1d0c5979`, SiLU `2d292fa46c911cdf3c1f0331414557c0`; each candidate MD5 matched its incumbent. | Fresh pp8192 rocprof: SiLU 16.4985→16.5186, RMS 7.2373→7.2031, GDN 5.7957→5.7859, sum 29.5315→29.5076 us/token (-0.081%). | KILL: below noise and far short of 28 us/token. |

## Paired daemon gates for the byte-exact DPP reduction

Matrix used `--matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`. Pair order was B→C then C→B.

| Pair | Arm | pp512 tok/s | pp8192 tok/s | tg128@128 tok/s |
|---|---|---:|---:|---:|
| 1 | baseline | 3378.6 | 3611.0 | 36.4919 |
| 1 | candidate | 3361.8 (-0.497%) | 3573.8 (-1.030%) | 36.4403 (-0.141%) |
| 2 | candidate | 3341.6 (+0.186%) | 3562.3 (+0.208%) | 36.3943 (+0.115%) |
| 2 | baseline | 3335.4 | 3554.9 | 36.3527 |

TTFT used `benchmarks/prompts/ttft_5900.txt`, 5,909 tokens, `--runs 8 --warmups 2 --json`. Pair order was B→C then C→B.

| Pair | Arm | TTFT median ms | Derived tok/s | Candidate delta |
|---|---|---:|---:|---:|
| 1 | baseline | 1688.395 | 3499.78 | — |
| 1 | candidate | 1694.962 | 3486.22 | -0.388% tok/s |
| 2 | candidate | 1698.869 | 3478.20 | +0.144% tok/s |
| 2 | baseline | 1701.309 | 3473.21 | — |

The two-pair TTFT result is directionally inconsistent and averages to approximately -0.12% candidate throughput, versus the required +1.5%. The first pp8192 matrix row also exceeds the allowed 1% regression. KLD and battery gates were not run because this branch fails performance gates and ships no implementation.
