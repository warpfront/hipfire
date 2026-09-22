# gfx11 Q-resident wide-row attention experiment

## Verdict

**KILL.** The structural gfx1201 ideas, isolated from the rejected integer arithmetic, lost decisively against the production gfx11 attention kernel:

| GPU | Incumbent median | Wide median | Candidate / incumbent throughput |
|---|---:|---:|---:|
| gfx1100 | 4.080410 ms | 8.359451 ms | **0.4881x** |
| gfx1151 | 7.941058 ms | 23.061447 ms | **0.3443x** |

This is not an end-to-end tuning gap. The candidate missed the 1.25x standalone gate on both GPUs, so no TTFT, graph, battery, or c24 campaign was run. The typed knob remains default-off and the shipped route is unchanged.

These candidates failed for measured and inspectable reasons; this is not a claim that the ISA makes every possible structural attention kernel impossible. Together with the earlier in-place iu8 result (0.532x), however, the two separable halves of the gfx1201 attention design have now both lost on gfx11: the arithmetic half because RDNA3 iu8 WMMA runs at f16 rate while Q8_0 adds eight scale folds, and this structural half because compressed-LDS fragment construction plus the register-mandated D split adds substantially more arithmetic than the saved traffic can repay.

Two initial explanations are explicitly ruled out. First, this is **not** a register-occupancy-knee failure: the candidate compiled to 193 VGPR / 32 SGPR with zero spills and the HIP API measured one 16-wave block/CU, exactly at the latency-hiding knee. The similarity between the gfx1151 0.344x result and the earlier eight-wave RB64 failure band is coincidental. Second, the capacity premise was real but **non-binding**: compressed LDS does fit more keys, but production had already selected KT32 over a measured KT64, deliberately declining the larger f16-plane capacity.

The generalizable mechanism is plane-sharing amortization. The incumbent pays code-times-scale conversion once while constructing a shared f16 LDS plane, then every query consuming that plane reuses the result. Fragment-time dequantization repeats the conversion for every consuming wave and duplicate lane half: roughly 6x more K and V conversions per query/subtile here. The zero-spill D128 shape additionally raises WMMA work 1.5x by repeating QK and softmax. Native fp8 on gfx1201 does not pay this conversion penalty at fragment build, which is why the same structural layout can win there without contradicting this gfx11 result.

## Corrected production baseline

The experiment was rebased on production commit `7a34a572a`. The initial brief described two 32 KiB f16 planes. That does **not** describe the shipped kernel at this commit:

- It is KT32, 128 threads / four wave32s, with three compute waves and one helper wave.
- K and V are two **16 KiB** f16 planes: `32 keys * 256 dims * 2 B = 16,384 B` each, or 32,768 B total dynamic LDS.
- Its F4b launcher first preconverts f32 Q into a Gpu-owned f16 scratch buffer. The attention body loads f16 Q once per KT32 tile and reuses it across that tile's two 16-key subtiles.
- The source records that KT64 (two 32 KiB planes, 65,536 B total) was measured and removed in favor of KT32 on both gfx1100 and gfx1151.

Compiled metadata and the HIP occupancy API report the incumbent body at 224 VGPR, 28 SGPR, zero VGPR/SGPR spills, zero private bytes, 32,768 B dynamic LDS, two blocks/CU, and eight resident waves/CU. The preconvert helper is a separate small kernel.

### Does LDS capacity bind the incumbent?

No, not in the performance sense hypothesized by the brief. Capacity changes the feasible tile size, but the production evidence already selected the smaller KT32 tile over the available KT64 tile. More capacity per tile was not profitable.

The byte budgets make the distinction explicit:

- Materialized f16 K+V costs `2 * 256 * 2 = 1,024 B/key`. The production 32 KiB budget therefore holds exactly 32 keys.
- Raw Q8_0 K+V costs `2 * (256 code bytes + 8 f16 scales) = 544 B/key`. The same 32 KiB holds 60 keys, or 48 after aligning to 16-key WMMA subtiles. A 64 KiB budget holds 120, or 112 aligned keys.
- The candidate stores full K plus only the current D128 half of V: `272 + 136 = 408 B/key`. KT128 therefore consumes exactly `128 * 408 = 52,224 B`.

At S8283, production KT32 makes 259 outer tiles. A raw full-K/full-V layout in the same 32 KiB would reduce that to 173; the tested D128 candidate makes 65 KT128 tiles in each of two D passes, or 130 K scans. There is real byte headroom, but it is only useful if fragment construction is cheap enough. It was not.

At B512/S8283, source-level address accounting gives:

- production logical Q8 K+V reads: 1,153,523,712 B;
- candidate logical K-twice-plus-V-once reads: 324,428,544 B (3.56x less);
- production f16-Q body reads: 1,629,487,104 B across 259 KT32 tiles, in addition to the one-time F4b preconvert;
- candidate Q read: one 12,582,912 B f32 pass into resident f16 registers.

Despite those traffic reductions, runtime doubled on gfx1100 and nearly tripled on gfx1151. LDS capacity and external traffic are therefore not the limiting axis for this production kernel.

## Tested candidate

The candidate is deliberately free of the rejected arithmetic path:

- 512 threads / 16 wave32s / 256 query-head rows per workgroup;
- grid `[ceil(6*batch/256), 4, 1]` for H24/KV4/D256;
- Q loaded once from f32, rounded to f16, and retained as sixteen lane-half fragments;
- KT128 raw Q8_0 codes and f16 scales in LDS;
- code-times-scale conversion while f16 fragments are assembled;
- ordinary `v_wmma_f32_16x16x16_f16` for both QK and PV; emitted ISA contains no integer WMMA;
- two D128 passes, required to keep the live output accumulator below the 256-VGPR limit;
- online softmax and output accumulation in f32; probabilities rounded to f16 as in the incumbent.

Compiled metadata is identical on the two target builds: 193 VGPR, 32 SGPR, zero VGPR spills, zero SGPR spills, zero private bytes. The HIP runtime reports 52,224 B dynamic LDS, one block/CU, and **16 resident waves/CU**.

The gfx1151 result is numerically close to the 0.35--0.44x collapse previously seen when GEMM register blocking fell to eight waves, but that is not the failure here. This candidate reaches the 16-wave latency-hiding knee, has no spills, and uses fewer VGPR than the incumbent. It is not an RB64-style occupancy collapse.

The register limit matters indirectly: the full D256 output accumulator does not fit, so the viable zero-spill shape must use two D128 passes. For every 16-key subtile, the incumbent performs 16 QK plus 16 PV WMMAs (32 total). The split candidate performs `(16 QK + 8 PV) * 2 = 48`, so it executes 1.5x as many WMMAs and repeats score construction and online softmax.

Raw-code fragment construction also gives up the incumbent's dequantization amortization. The incumbent converts each K/V plane element once per workgroup and shares it across 48 query-head rows. The candidate rebuilds scaled f16 operands independently in all 16 compute waves, and gfx11's lane-pair basis duplicates identical operands in both lane halves. Per query row and 16-key subtile, that is about 512 K conversions and 512 V conversions versus about 85.3 of each amortized by the incumbent: roughly 6x more scale/code conversions per row. The larger tile saves global/LDS capacity traffic but moves too much unpack-and-scale work into the hottest fragment path.

## Correctness

A deterministic odd-tail check used B17/S83, consecutive causal positions, random f32 Q, random signed Q8_0 K/V codes, and varying f16 scales. It compared both kernels with a scalar f32 oracle and compared their output bits directly. Results were identical on both GPUs and in all six fresh processes:

- incumbent versus f32 oracle: NRMSE `2.502722127e-4`, max absolute error `5.349051207e-6`;
- wide versus f32 oracle: NRMSE `2.502722127e-4`, max absolute error `5.349051207e-6`;
- wide versus incumbent: NRMSE `0`, max absolute error `0`, **0 bit mismatches**.

The candidate therefore carries no numerical penalty; performance alone kills it.

## Timing evidence

ROCm 7.15.26333, B512/S8283, three warmups, eight timed iterations, ABBA order, three fresh processes per GPU. Each process asserted the selected architecture. Cards were empty before and after each leased window except the zero-VRAM `gpusentry` process.

### gfx1100

| Process | A1 incumbent | B1 wide | B2 wide | A2 incumbent |
|---|---:|---:|---:|---:|
| 1 | 4.234224 | 8.541289 | 8.150050 | 3.926595 |
| 2 | 4.326156 | 8.563074 | 8.152956 | 3.873358 |
| 3 | 4.265117 | 8.567734 | 8.177613 | 3.873399 |

Median of the six samples: incumbent 4.0804095 ms, candidate 8.359451 ms, ratio 0.488119x.

### gfx1151

| Process | A1 incumbent | B1 wide | B2 wide | A2 incumbent |
|---|---:|---:|---:|---:|
| 1 | 7.979589 | 22.999133 | 23.013088 | 7.915157 |
| 2 | 7.951625 | 23.080531 | 23.063460 | 7.909678 |
| 3 | 7.936562 | 23.059433 | 23.093913 | 7.945554 |

Median of the six samples: incumbent 7.941058 ms, candidate 23.0614465 ms, ratio 0.344343x.

## Loader, addressing, and lane audit

This audit was completed before accepting the ratios:

- Q/out use `query * 6144 + head * 256`; `row = row_base + 16*wave + lane%16`, `query = row/6`, and `head = 6*kv_h + row%6`. Across grid Y=4 and the `6*batch` row interval this covers every query head exactly once.
- Q8 cache addressing is `token * 1088 + (kv_h*8 + block) * 34`, matching four KV heads, eight 32-dim blocks/head, and 34 bytes/block.
- K stages blocks 0..7. V pass 0 stages blocks 0..3 and pass 1 stages blocks 4..7. Tail keys are zero-filled under `gk < seq_len`.
- Per-workgroup causal bounds cover every query touched by the row interval. Per-row masking uses both `gk < seq_len` and `gk <= positions[query]`.
- A post-mailbox barrier prevents the first K-scale fill from racing another wave's bounds read; this was found and fixed during the source audit before measurement.
- At B512, `6*512 = 3072` is divisible by 256: all 48 workgroups are full and every one of the 16 waves owns 16 valid query-head rows. There is no inactive compute-wave explanation for the measured loss.
- The 512 staging threads are also fully used: K has 1,024 scale records (two per thread), while each D128 V pass has 512 (one per thread). The B17 correctness tail is intentionally partial; invalid rows are qok-masked while all threads still participate in staging/barriers.
- Compiled metadata reports no spills/private segment. Disassembly contains only f16 WMMA opcodes, and Q global loads precede the D-pass/key loops, consistent with actual Q residency.

## Artifacts

- Candidate kernel: `kernels/src/attention_q8_0_fa2_gqa_wide.gfx11.hip`
- Default-off typed knob: `kernel.gfx11_attn_wide` / `HIPFIRE_GFX11_ATTN_WIDE`
- Direct correctness/timing harness: `scratch-gfx11attn/src/main.rs`
- HIP resource/occupancy probe: `scratch-gfx11attn/resource_probe.hip`
