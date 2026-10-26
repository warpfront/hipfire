# Flash-Next NPU trunk: CPU preparation

The Qwen4 `npu_trunk` module owns CPU-only model preparation for hipfire 0.4.2;
`peacemaker-npu` owns the numerical reference, panel layout and deployment builder.
Neither preparation path opens hardware or schedules prefill.

Qwen4 reads `fusions.npu_trunk` (env `HIPFIRE_QWEN4_NPU_TRUNK`) with
`hipfire_runtime::coproc::Mode::from_config(SWITCH, Mode::Auto)`: `off`, `auto` (default), `on`, or `emulate`. Mode, containment, admission and pair geometry are shared in `hipfire_runtime::coproc`.
At load, auto resolves silently to off on every arch except gfx1151.
The qwen3.8-27b switch is `fusions.qwen35_npu_trunk` (env `HIPFIRE_QWEN35_NPU_TRUNK`, default `off`); any other value is refused at load until its prefill driver lands.

## Serving status (0.4.2)

This guide is the chronological design and bring-up record of the route. This
section summarizes what ships; the root `CHANGELOG.md` entry for the 0.4.2
Flash-Next NPU trunk is the authority for measured numbers, and
[`docs/env-vars.md`](../env-vars.md) and [`docs/CONFIG.md`](../CONFIG.md) own the
knob tables. Sections marked **historical** record design-time simulator work
(including int8/I32 comparisons) that the BF16-direct route superseded; they are
kept for provenance and are not current recommendations.

* **Route.** Flash-Next (Qwen4) family B, K=2560, stock MQ6 shapes, single
  device without EP. The GDN (`GdnIn`) and QSA (`QsaIn`) trunk **input
  projections** run on the XDNA2 NPU in a two-bank, layer-major pipeline against
  the GPU. Mixer cores, output projections, HC, MoE and decode stay on the GPU.
* **Mode.** `HIPFIRE_QWEN4_NPU_TRUNK` / `fusions.npu_trunk` is
  `off`, `auto`, `on` or `emulate`; the default is `auto`. Auto arms on gfx1151
  only, with an open NPU and a matching keyed calibration; every other arch is
  effectively `off` with no NPU I/O or allocation. Any auto load refusal logs one
  reason and runs GPU-only. `off` is the kill switch.
* **Confirmed hardware.** Strix Halo (Ryzen AI Max+ 395) and the Ryzen AI Max+
  PRO 495 / Radeon 8065S: the latter is not a Strix Halo part but enumerates as
  gfx1151 on the same PCI ID (`1002:1586`) and carries the same XDNA2 NPU
  (fw 1.1.2.65). On a 192 GB PRO 495 host the built-in calibration applied with
  the 5% kernel-release margin, `on` armed (`runtime=open`), pp16384 measured
  1980.9 -> 2209.9 tok/s over `off` (+11.6%) with `qwen3.8:flash-next`, and a
  greedy 400-token generation was byte-identical to `emulate`. Numbers are
  STAPM-throttled chassis numbers; see issue #966 for the full report.
* **Calibration.** The compiled-in measured Halo records are the normal path: they
  bind on arch, build, model index, epilogue and trunk route only, and apply on any
  Halo kernel, NPU firmware or GPU PCI address at margin 0.0 (see below).
  `fusions.npu_trunk_calibration` / `HIPFIRE_QWEN4_NPU_TRUNK_CAL` (default
  `<hipfire home>/npu-trunk/calibration.json`) is an optional strict override: a
  file there wins, must match every key field, and is produced by
  `hipfire calibrate-npu-trunk`. No file is required. The schema is **4**,
  BF16-direct only: there is no int32 contract
  (`glue.fold_ms_per_col_8192` is gone and `epilogue` must be `bf16`), and a
  schema-3 record is rejected.
* **Auto planning (0.4.3).** Under `auto` each request picks its NPU family
  subset and pair plan at request start from rate scales learned live in this
  process: a GPU-window scale and an NPU-chain scale, EWMA 0.25 against the
  built-in model, clamped 0.5-2.0, frozen once settled, reopened after 3
  consistent deviations over 5%. The first long request of a process runs the
  built-in prior. All four families are packed at load under `auto`; per-request
  subsets run contiguous panel ranges of the packed chains. Forced modes (`on`,
  `emulate`) plan from the static calibration only, so `on` == `emulate` stays
  comparable. A guard sends the rest of a request GPU-only when, after a pair, the
  GPU's measured waits on the NPU exceed what pairing saves it (one `npu-trunk: guard: ...`
  line per request). Output numerics differ only
  between the NPU and GPU projection routes, both within the accepted KLD cap.
  Details: Live rate scales (auto), below. Developer variable
  `HIPFIRE_QWEN4_NPU_TRUNK_FAMILIES` forces the per-request subset.
* **Trace.** `HIPFIRE_QWEN4_NPU_TRUNK_TRACE` is unset by default; set it to a
  path to append per-(layer, bank) JSONL. Tracing changes timing; retain an
  untraced run for primary correctness and performance evidence.
* **Pairs and chains.** Natural 8192+B pairs, the planner's rebalanced last pair
  (two banks of about equal rows) and one interior session-snapshot row per pair
  execute. NPU chains are row-adaptive in 256-row waves: a bank below 8192 rows
  runs the G80 program of its own wave count, built on first use. At most four
  wave counts are admitted per load; a pair that would need a fifth runs
  GPU-native.
* **Not shipped.** The K = 6144 out-projection program exists (see below) but is
  **not wired** into the route. QSA `o_proj` (`qsa.o`) and GDN `out_proj`
  (`gdn.out`) stay on the GPU, and the planner's out families are
  `selectable: false`.

Measured TTFT, exactness and quality results for this route are recorded, with
their fixtures and methods, in the 0.4.2 Flash-Next entry of the root
[`CHANGELOG.md`](../../CHANGELOG.md); they are not repeated here. Other figures
in this guide are design-time estimates or earlier-round measurements and are
labeled as such.

## Shapes and masks

| Shape | Fused family order and output columns | Full padded columns / panels |
| --- | --- | --- |
| `GdnIn` | qkv 10240, z 6144, a 48, b 48 | 16640 / 13 |
| `QsaIn` | q 12288, k 512, v 512, idx 640 | 14080 / 11 |

`FamilyMask` compacts selected families in this order. `ColumnMap::parts`
records each retained family's fused start and width. Padding to 1280-column
G80 panels is zero; padding has no family and its weight scale is zero. An
empty mask produces no panels. Matrix references are supplied in fused family
order; unselected families may be omitted. Retained matrices must have their
family's exact row count and K=2560.

## Numerics and packing

`pack_matrix` accepts MQ6G256V2 rows: each 256-value group is 200 bytes, with
little-endian f16 s0/z0/s1/z1 followed by 192 payload bytes. Decoding uses
separate f32 multiply and add. Row scales are `((absmax as f64)/127.0) as f32`;
W8 codes use f32 division, ties-to-even rounding and clamping to [-127,127].
Zero-scale rows become zero codes. Direct `pack_matrix` calls panic on invalid
length or a K not divisible by 256; shape/trunk packers return descriptive
errors after validating the matrix metadata.

`pack_shape` transposes W8 into panels and invokes `ArrayDesign::pack_b`
directly. `PackedB::ws` contains the padded fused scales.
`pack_trunk` preserves layer order and stores per-layer shape/panels and scales.

The only contract is `EpilogueContract::Bf16Direct` (`as_str` "bf16_direct"; the `I32Fold` contract and its
GPU fold were removed: no int32 partial, combine or fold exists in the product route). The ordinary pack
functions use it: `Bf16Direct` rounds each raw weight scale upward with `peacemaker-npu`'s public `grid15_up`,
quantizes with that exact grid scale, and returns padded `(Sm,Se)` sidecars in `PackedB::scales` as well
as embedding them in each packed panel; `PackedB::ws` holds the grid scale as f32.

## Programs

`build_programs(contract, rows_max)` selects one G80 direct-BF16/Fast
design per shape, via `peacemaker-npu::kernels::gemm_g80::build_panel_program`.
M is rounded up to 256, N=1280, K=2560; `rows_max` must be 1..=65536.
`build_programs_for_families` uses the same compacted maps as weight packing.
Each shape record reports panel count, padded width, per-submit A/B/C argument
sizes and per-submit host bytes read/written. The designs include PDI and TXN
bytes; panels reuse their shape's design.

`build_rowmajor_programs_for_families(rows_max, mask)` builds BF16-direct designs
with col0=0 and the selected map's padded fused pitch. qkv-only GdnIn is exactly
8 panels / pitch 10240. Zero-panel shapes are never submitted or given GPU views.
Arg2 declares the compact fused arena; each retained panel command advances its
C base by `2*panel*1280` bytes.

## CPU checks

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
nice -n 19 cargo test --release -p hipfire-arch-qwen4 -j 12 --test npu_trunk_packing --test npu_trunk_programs --test npu_trunk_simulation
```

The test suite includes independent MQ6/W8 oracles, a copied lab CPU twin,
column-map/mask arithmetic and a functional G80 simulator test. The ignored
`model_sanity` test uses a CPU HFQM reader and prints weight-scale statistics
and pack time for layer 0 GDN qkv and layer 3 QSA q from the pinned
`qwen3.8-flash-next-gptq3.mq4` artifact (recorded MD5 prefix `be007fc3`).
Set `HIPFIRE_NPU_TRUNK_TEST_MODEL` to override its default model path and run
`cargo test --release -p hipfire-arch-qwen4 --test npu_trunk_model_sanity -- --ignored --nocapture`.

## G80 direct BF16 (design record, simulator only; historical)

> **Historical.** This section through "Fused row-major BF16 drain" is the
> design-time simulator record. Throughput figures are estimates scaled from a
> measured int8 wave, and the int8/I32 comparisons describe alternatives that
> BF16-direct superseded (schema 4 has no int32 contract). Hardware results for
> the shipped route are in the root changelog and the serving status above.

The epilogue uses no floating operations: 16x16 multiplies construct
`J=I*Sm*Dm` in acc64 by Horner limbs, then an IEF15-derived highest-bit ladder,
sticky-preserving normalization and one `conv_even` SRS emit 32 BF16 values
at a time, descending into C's upper half `[0x5000,0xa000)`.
IEEE ties, carries, gradual underflow, signed underflow and infinity are
included; an exact zero product is positive zero.

Existing A/B DMA rings receive larger chunks, with no new channels or BDs:

* A half: 4096 code bytes + Dm[128] u16 + De[128] i16 = **4608 B**.
  Both E/O halves repeat full token scales for every K64 chunk.
* B tile: 5120 code bytes + Sm[96] u16 + Se[96] i16 = **5504 B**.
  The last 16 feature entries are zero padding for aligned loads.
* Core A slots: `0xa000/0xb200`; B: `0xc400/0xd980`; epilogue scratch
  `[0xef00,0x10000)`. The last input sidecars are captured before returning
  the input locks, so next-wave lookahead cannot overwrite them.
* Memtile C: one whole 20480-B stage per vertical pair at `0x4000/0x9000`.
  Finite row0-drain hands EMPTY to row1, and row1-drain returns it to row0;
  no row-order race or lookahead credit. A slots `0xe000/0x10000`, B
  starts `0x12000`, resident end `0x7d800` at K2560. The low lean-ring
  scratch remains untouched. C tasks are finite for any M-wave parity.
* 18 memtile BDs on injecting columns, 14 otherwise; 5 core BDs, unchanged
  channel counts and bank assignments. B is read once per panel, not per wave.

For W M256 waves and kc K64 chunks, shim A is `4*W*kc*4608`,
B is `8*kc*2*5504`, C is `8*W*4*20480` bytes. At 8192x1280x2560:
read **27,115,520 B**, write **20,971,520 B**, total **48,087,040 B**,
**0.895690918 B/kop** (declared shim bytes, not physical DRAM measurements).
Repeating sidecars is the explicit traffic tradeoff for retaining the existing
input ring machinery; pair-stage serialization makes the resident panel fit.

`npu-gemm M 1280 K --variant G80 --epi bf16` supplies deterministic grid
scales and compares checked outputs to `peacemaker-npu::kernels::w8a8_epi::bf16_direct`.
At design time this arithmetic, DMA layout and the simulator timings carried no
silicon claim; the shipped route has since been validated on hardware (see
Serving status). The legacy int8/I32 PDI/TXN hashes for M256/M8192, N1280,
K2560, Fast/Slow remain pinned in `peacemaker-npu`'s `golden_bytes.rs`
(`g80_legacy_bytes_match_golden`) as records of the pre-BF16 images; no trunk
serving path uses them.

At one M256 wave, K2560 (functional simulator, Fast), the executed PDI/TXN
models **113,393 / 140,888 / 209,386 cycles** for int8 / I32 / BF16.
Scaling measured int8 **33.1 TOPS** by those cycle ratios gives **33.1 /
26.6404 / 17.9253 TOPS**, respectively: estimates, not hardware measurements.
The BF16 arithmetic group issues in **223 cycles per32 outputs**. This
integer path meets the host-byte target but predicts a substantial throughput
loss; it does not establish the hardware throughput promotion gate.

Two-wave simulator totals at K2560 are **171,948 / 226,923 / 357,502**
ticks (int8/I32: M512; BF16: M300 padded to the same M512 deployment).
Subtracting the one-wave totals removes startup/tail: incremental wave
costs are **58,555 / 86,035 / 148,116** ticks. Scaling 33.1 TOPS by these
incremental ratios estimates **33.1 / 22.5277 / 13.0855 TOPS**.
These derived warm-wave numbers, not the optimistic cold-submit ratios,
are the relevant throughput warning for a resident multi-wave trunk.

### Fused row-major BF16 drain

`gemm_g80::build_panel_program_rowmajor(rows_max, pitch, col0)` writes
`C[t][col0+n]` at byte `2*(t*pitch+col0+n)`; arg2 is
`ceil(rows_max/256)*256*pitch*2` bytes. Pitch and col0 must be even.
The complete padded panel is written (zero padded tokens), but no other
columns or rows beyond the padded extent are touched. Native-tile BF16
M256/M8192 PDI/TXN hashes are pinned alongside the unchanged int8/I32 pins.

No retile or additional DMA transfer: the memtile MM2S drain uses four
inner-first `(step,wrap)` dimensions in 32-bit words:
`(1,4), (32,10), (4,8), (320,0)`. These read the native 8x8 blocks in
row order. One shim BD per channel/wave scatters both 128x80 parity tiles
with `(1,40), (pitch/2,128), (40,0)`; transfer length is 10240 words.
The outer +40-word step returns to row zero at the adjacent 80-column tile.
Fields are `Bd::memtile_words` words 2..5 (17-bit step-minus-one,
10-bit wraps) and `Bd::shim_words` words 3..5 (20-bit step-minus-one,
10-bit wraps), transcribed from aie-rt
[`xaie2pgbl_params.h`](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/global/xaie2pgbl_params.h)
and [`xaie_dma_aieml.c`](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/dma/xaie_dma_aieml.c).
Unused outer wrap is zero; length terminates the outer dimension.

At pitches 14080/16640, a 256-row wave step exceeds the shim's 20-bit
stride/iteration field. The drain instead batches **K=7 waves**, with absolute
DDR addresses for every wave. AIE2P has **16 shared shim BDs per tile**, not
16 per channel (`Bd::address`, four-bit NEXT_BD in `Bd::shim_words`, and the
[vendor register fields](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/global/xaie2pgbl_params.h)).
B=0 and A=2 remain live. C uses seven BDs per channel:
S2MM0 `[1,4,6,7,8,9,10]`, S2MM1 `[3,5,11,12,13,14,15]`.
All 16 BDs are used on A-injecting columns (15 otherwise). Two BDs per wave
leave room for at most seven complete waves without borrowing A/B descriptors
or changing stream bytes/order.

Each channel gets one finite seven-wave chain/task and one completion token.
Both C SYNCs precede every refresh. The first batch of every full or lean
body writes its C descriptors; later batches refresh **DDR patches only**.
Iteration wrap=1 keeps `current=0`. A partial final batch cuts its last
links with two tail-word writes per column; the next body's initial writes
restore the chain. Ring POLL/SENT/DONE temporarily borrow 4/5/6 outside the
completed GEMM body, so initial C writes are never omitted by same-design
lean bodies. A/B still queue once per submit; the API and panel base offsets
are unchanged. No extra data transfer, channel, stage or memtile/core BD.

For W waves and H=ceil(W/7), C drain TXN operations are
`16*min(W,7) + 16*W + 18*H + 16*(W>7 && W%7!=0)`,
versus the previous `82*W` and native-tile `50` (descriptor writes,
patches, task writes, two aggregate SYNCs). At W=32 this is **730 vs 2624**
C ops (72.2% fewer); SYNCs/tasks fall from 32 to 5 pairs (84.4% fewer)
and descriptor block writes from 1024 to 112 (89.1% fewer).
At W=256: **4890 vs 20992** C ops, 37 vs 256 SYNC/task pairs.
TXN bytes/counts change; the PDI, native BF16 TXN and int8/I32 golden bytes
remain unchanged. Design-time expectation: firmware overhead is substantially
reduced, but recovery of the then-measured 9% throughput loss was a silicon gate, not a claim.
Firmware command overhead is not covered by the phase-sum cycle estimate.

Build one design with col0=0 per pitch/row bound. All 11 QsaIn / 13 GdnIn
panels can reuse its PDI and program; advance arg2's base by `2*panel*1280`
through `append_run_body` or `append_lean_run_body`. Retain the full fused
arena behind the offset, and enforce `panel_col0+1280 <= pitch`.
Alternatively the explicit col0 builder changes DDR patch offsets only.
Shim traffic remains A `4*W*kc*4608`, B `8*kc*2*5504`, C
`8*W*4*20480` bytes: the phase-sum estimate is unchanged from native tiles.
Simulator coverage includes 1, 2, K, K+1 and 256 waves at both fused pitches,
full/lean reuse, adjacent panels sharing an offset arena, padded zeros,
outside-window poison and exact shim bytes. Tests compare TXN op counts with
the previous per-wave drain and native tiles; functional ticks are modelled
timings, not firmware latency or silicon measurements.

Design-time silicon commands (not run by that change; repeat at pitch 14080):

```sh
# Every live value, padded zero and outside-window poison checked.
target/release/npu-gemm 8192 1280 2560 --variant G80 --epi bf16 \
  --rowmajor-pitch 16640 --rowmajor-col0 1280 --iters 3 --reuse-ctx
# Same exact first/final-poison checks, with busy-region TOPS over 15 seconds.
target/release/npu-gemm 8192 1280 2560 --variant G80 --epi bf16 \
  --rowmajor-pitch 16640 --rowmajor-col0 1280 --loop 15
target/release/npu-gemm 8192 1280 2560 --variant G80 --epi bf16 --loop 15
```

Historical Halo measurement of the superseded per-wave row-major drain: 15.05
TOPS versus 16.48 TOPS native. The batched drain was validated afterwards on
silicon as part of the shipped route (root changelog); these figures are not
current performance claims.

## G80 N512 out-projection program (K = 6144; built, not wired into serving)

Status (0.4.2): the program, its packing, a second-PDI relay hardware context
and a transport oracle exist, but no serving route requests them. QSA `o_proj`
(`qsa.o`) and GDN `out_proj` (`gdn.out`) are **not shipped** on the NPU and stay
on the GPU. Silicon results and the measured, unshipped QSA out-projection
wiring attempt are in the root changelog. The design description and
Gate K0 below are historical simulator records.

`peacemaker-npu::kernels::gemm_g80_n512` (`design_g80_n512(m, 512, 6144, Epilogue::Bf16, Fast)`,
`build_panel_program_n512(rows_max, pitch, col0, 6144)`) is the program of the QSA `o_proj` / GDN
`out_proj` shapes (`Shape::{QsaO, GdnOut}`, N = 2560 = five 512-column panels, one pass each,
K = 6144). Numerics: integer-exact accumulation on the array over the whole K, one RNE to BF16
(`bf16_direct`); no int32 partial, combine or fold anywhere. It reuses the silicon-proven V8 geometry
(512 x 512 wave, 128x64 pair cores, V8 switch circuits unchanged) with the B panel resident in the
memtiles, a new 64-column BF16 epilogue (`g80_bf16::NARROW`, the 128x80 map is byte-identical) and
`gemm_core::pair_chunk` as the K loop. Core `(col, rho)` computes A block `2*(rho/2) + col%2`,
B half `rho%2` of block `col/2`; every column injects one A half (4096 codes + `Dm[128]`/`De[128]`
= 4608 B per K64 chunk) and holds one B half (4096 codes + `Sm[64]`/`Se[64]` = 4352 B per chunk).

Memtile (per column): C stage 4 x 16,384 B at `0x4000`, A ping/pong 2 x 4608 B at `0x14000`, B
`kc*4352` B at `0x16400` (417,792 B at kc = 96, ends `0x7c400` of `0x80000`, `MAX_KC` = 99).
17 memtile BDs (15 at kc <= 64) and 14 locks per column: C producers `0/24/2/26` and drains `6->7 / 32->33` (finite,
`repeat = W`), B fill `4,5` / guarded read `30,31` (finite groups of at most 64 chunks: the iteration
wrap field is 6 bits) and the lock-free whole-panel replay `36` (`repeat = W-1`), A rings `28<->29`,
`14<->15`. A memtile lock holds at most 63 and the fill can run a whole panel ahead of the stalled
reader, so the fill acquires a credit lock (id 6, initial `min(kc,63)`) that the guarded reader
returns. Core image: lower/upper programs 9,776 B each (limit 16 KiB), PDI 340,944 B, full TXN
46,448 B (M512) / 63,984 B (row-major M8192), lean TXN 10,608 B (M512). Row-major drain: one shim BD
per channel and wave with `(1,32),(pitch/2,128),(32,0)` over 8192 words, memtile dims
`(1,4),(32,8),(4,8),(256,0)`; the existing 7-wave batching applies. PDI/TXN/program hashes are pinned
in `golden_bytes.rs` (`g80_n512_out_bytes_match_golden`); the 128x80 pins are unchanged.

Host layout: arg0 = A, 8 segments of `W*kc*4608` B (`W` 512-row waves; 56,623,104 B at 8192 rows),
arg1 = B, 8 segments of `kc*4352` B (3,342,336 B), arg2 = C row-major `rows*pitch*2` B (pitch 2560,
panel `p` at `col0 = 512*p`, `append_run_body` / `append_lean_run_body` with `arena_base[2] =
2*512*p`) or native 16,384 B tiles. `pack_a_bf16` / `pack_b_bf16` dispatch on the design.

### qwen3.8-27b dense in-projections: K = 5120 on N512 (HIPF-161, kernel/program/transport only)

The same N512 program at `k = 5120` (kc 80, two B groups of 40, credit 63; B panel 348,160 B per column) holds the
27B in-projections, whose every NPU-eligible projection has K = 5120: GDN qkv 10240 + z 6144 + a 48 + b 48 = 16480
columns (33 panels, pitch 16896) and FA gated q 12288 + k 1024 + v 1024 = 14336 (28 panels). One BF16 rounding after
the whole K, as above; the N1280 core (`MAX_KC` 40) cannot hold K5120 without an int32 partial combine.

- GPU producer: `Gpu::npu_trunk_pack_a_bf16` with `NpuBf16Layout { k: 5120, n512: true, .. }` writes the 8-segment
  512-row-wave A stream (`rows * 5760` B) and the compact `Dm/De` vector (`npu_trunk_pack_a_bf16_k5120`, same grid15-up
  quant DAG as K2560). `n512: false` writes the N1280 4-segment stream the emulate GEMM addresses (`npu_trunk_i8_gemm_pm`
  is K-generic); consume and the emulate GEMM refuse `n512` layouts (the NPU writes row-major C itself).
- Transport: `railgun::npu::trunk::Shape::{DenseGdnIn, DenseFaIn}` (`Shape::k()` = 5120, `is_n512()`); the N512 A arena is
  sized for the largest K of the open N512 shapes, B panels from `k`. The relay keys adaptive contexts by `(k, padded rows)`.
- Oracles: `npu_trunk_glue_check` (pack vs the CPU N512 stream, emulate vs `bf16_direct` at both shapes),
  `g80_n512_k5120_*` and the ignored `g80_n512_real_27b_tile` (real qwen3.8-27b tiles) simulator tests,
  `npu-trunk-transport --shape dense_gdn_in --shape dense_fa_in [--corun]` on silicon.

Measured on gfx1151 (8192 rows, exact every round, fclk not pinned): GDN 60.5 ms busy p50 = 22.9 TOPS (wall 20.7),
FA 51.3 ms = 23.4 TOPS (wall 21.2); against 16 GiB/round of GPU copies GDN 17.8 / 15.1 and FA 19.5 / 16.6 TOPS.

### Gate K0 (design round verdict section 2), simulator ticks, never silicon (historical)

`cargo test --release -p peacemaker-npu --test g80 g80_n512 -- --nocapture` (golden bytes: complete padded
arena equal to the `bf16_direct` twin over the limb x sign x tie fixture at K = 6144, 1 / 2 / 16
waves, padded rows zero, outside-window poison untouched, five adjacent panels in one buffer, lean
and forced-requeue bodies, kc 1..99). Measured with `g80_n512_gate_k0_*`:

| | value |
|---|---|
| ticks, 1 / 2 / 16 waves (512 rows each) | 186,175 / 362,484 / 2,830,810 |
| incremental ticks per wave | **176,309** (K loop 1,152 ticks/chunk = A half 4608 B at 4 B/cycle, per-wave epilogue + drain 65,717) |
| int8 baseline wave (K2560, N1280) | 58,555 ticks = 33.1 TOPS |
| simulated rate alone (scaled through the same incremental-tick ratio) | **21.11 TOPS** (gate >= 16) |
| bytes per 8192-row fragment | A 56,623,104 + B 3,342,336 + C 8,388,608 = 68,354,048 = **1.3262 B/kop** (gate <= 1.4) |
| 128x64 against 128x80 K-loop MAC efficiency | 88.9 % vs 91.6 %: loses 3.0 % (gate 15 %) |

K0 passes; Option 2 (K-split with on-array partial reload) is not needed. The rate is an estimate
scaled from the measured int8 wave exactly as the K2560 BF16 numbers above, not a silicon result;
the K loop is bound by the 1,152-cycle A half stream, not by the MAC schedule.

Relay and transport: `railgun::npu::trunk` runs one hardware context per distinct PDI (input shapes
and out shapes have different PDIs, both out shapes share one); the single relay thread submits
serially, B residency, arenas and the device-side rejoin are unchanged. Out shapes use per-bank
`a_out` (`rows*6912` B = 56,623,104 B at 8192) and BF16 out arenas (`rows*2560*2` B = 41,943,040 B)
registered like the input C arenas. `npu-trunk-transport --shape gdn_out|qsa_o` is the 20-round
exact oracle (whole-K int32 sum + `bf16_direct`, both banks); `--corun` queues 16 GiB of GPU copies
against every chain; `--coexist` measures second-context creation / `config_cu` / context-switch
cost. Halo (H3, <= 90 s window):

```sh
# K = 6144 program, 8192 rows, exact vs the twin, rate alone (CPU reference runs before the device opens)
target/release/npu-gemm 8192 512 6144 --variant G80 --epi bf16 --n512 --rowmajor-pitch 2560 --rowmajor-col0 512 --iters 3 --reuse-ctx
target/release/npu-gemm 8192 512 6144 --variant G80 --epi bf16 --n512 --rowmajor-pitch 2560 --rowmajor-col0 512 --loop 15
# out-shape transport oracle: 5-panel chains, 2 banks, 20 exact rounds, forced-watchdog negative; then co-run
target/release/npu-trunk-transport --shape gdn_out --rounds 20
target/release/npu-trunk-transport --shape gdn_out --rounds 6 --corun
# second HwCtx coexistence, config_cu and context-switch cost (kill: no coexistence, or > 2 ms)
target/release/npu-trunk-transport --coexist --reps 10
```

## GPU Bf16Direct glue

`rdna_compute::npu_trunk` exposes an explicit `NpuBf16Layout` (padded rows,
K64-compatible K, panel count) and native `NpuTrunkOutput` ranges. S1 maps bind
destinations with `ColumnMap::gpu_output`; no fused output tensor is needed.
The producer writes every packed A byte, including zero padding, directly to a
device tensor or a device-accessible host-registered arena (`unsafe ..._ptr`).
Compact scale sidecars are four bytes per token: LE u16 mantissa then LE i16
exponent. Weight sidecars use the same encoding, indexed by fused column.

On gfx1151 at K2560, A-pack uses eight wave32 rows per workgroup, 128-bit
activation loads retained in registers, an integer half-magnitude wave max,
and one LDS barrier to coalesce native 8x8 byte tiles into 128-bit stores.
The scale and RNE quantizer DAG is unchanged. Other K/architectures retain the
scalar producer, also exposed as a lab-only old/new oracle. At 8192 rows the
single-read traffic is 65.57 MB including replicated scales and sidecars;
180 GB/s implies **0.364 ms** before arithmetic/LDS/launch overhead: a design-time
bandwidth estimate. The measured pack time after this coalescing is in the root
changelog.

`npu_trunk_emulate_bf16` is the `emulate` GPU twin: it reads exactly that packed A
and row-major S1 W8 codes, uses a caller-owned int32 `[rows,n]` accumulator (lab
scratch private to the twin; no int32 value is an NPU output or a serving
contract), then applies the lab's all-integer
single-rounding BF16 epilogue into native outputs. The consumer reads panel-major
G80 BF16 tiles without rounding; F32 stores are exact `bits << 16`. The producer
requires finite F16 inputs, as does the screened lab contract. Geometry may be
larger than the live token count; it must equal the submitted NPU program.

The signed-IU8 WMMA GEMM and BF16/F32 scatter are certified PM gfx1151 kernels
(`peacemaker-isa emit --kernel npu_trunk`, committed image and M7 receipt).
Activation grid `ceilf` and epilogue unsigned CLZ currently use hipcc, with the
two missing ISA-table operations logged in the PeaceMaker gap ledger. This section
is the primitive API layer: it has no scheduler, forward hook or NPU device open
of its own; the serving route above drives it.

Halo GPU byte oracle (do not hide the GPU for this command):

```sh
nice -n 19 cargo run --release -p rdna-compute --features lab -j 12 \
  --example npu_trunk_glue_check
```

It compares packed bytes/compact scales to the CPU quantizer and G80 packer,
every native output to exact CPU integer dots plus `bf16_direct`, and synthetic
tile scatter including nonfinite/signed-zero bit patterns. Full GDN/QSA maps,
both output dtypes, 512/8192/ragged-4647 tokens and poison tails are checked.
Repeated deterministic input/weight rows cache CPU dots, never GPU answers.
Typed upload shapes count elements (including guard elements), not bytes.
The oracle checks shape/dtype byte counts against every allocation and before
every download, and emits a phase label before launches and comparisons.
It additionally compares old/new complete A arenas and compact sidecars, including
padding and poison guards, at 1/37/512/4647/8151/8192 rows. Zero, maximum finite
F16, RNE half-integer ties and grid-scale boundaries are included even at one row.
It prints old/new pack GPU-event timings and bandwidth for each row count, plus
full-width consumer and integer GEMM timings; any mismatch exits 1. CPU layout/map tests:
`cargo test --release -p hipfire-arch-qwen4 --lib npu_trunk::gpu::tests -j 12`.

## Generic co-processor stage (`hipfire_runtime::coproc`, 0.4.4)

The route's topology-free parts are shared runtime code; the Qwen4 module keeps
only Flash-Next's shape description, weight packing, programs and the hooks of
one (layer, bank) visit.

* **Op-level stage map** (`coproc::stage::StageMap`). Per layer kind (0 = GDN,
  1 = attention) the layer count, input-projection K and out family; the input
  projection families the NPU may own, in fused order; and the ownership groups
  (families that move as a unit, one contiguous fused-column range each). Mixer
  core, recurrent/KV state, residual, FFN and head always stay on the GPU
  (`StageMap::owner`). Flash-Next's map is `npu_trunk::FLASH_NEXT`: 36 GDN and 12
  QSA layers, K = 2560, groups `qkv`, `zab`, `q`, `kvidx`.
* **Ownership table** (`FamilySet`). The EP-like table of the families the NPU
  owns, whole groups only, chosen by the planner per load (`select_families`) or
  per request (`choose_families`).
* **Numerics contract** (`EpilogueContract::Bf16Direct`, calibration spelling
  `bf16`). Every NPU-owned output is exact integer accumulation over the whole K
  with grid15-up scale sidecars and one BF16 rounding; no int32 partial, combine or
  fold exists on either device.
* **Calibration and planner** (`coproc::{calibration, plan, rates}`). The schema-4
  record below is validated against a map (group names, kind names, out
  families); the planner, crossover, selection, rate estimator and the
  max-window budget (`admit_wave_budget`, at most four row-adaptive programs per
  load) take the map. The arithmetic and the goldens are unchanged.
* **Pair schedule** (`coproc::schedule`). A chunk pair is two micro-batches walked
  layer-major: `Prepare(0,A)`, `Init(B)`, `Prepare(0,B)`, then per layer
  `Run(l,A)`, `Prepare(l+1,A)`, `Run(l,B)`, `Prepare(l+1,B)`. `pair_edges` lists the
  dependencies the order respects (activation pack to the NPU and its result
  back, A-arena reuse after retire, residual rows, layer state A before B); the
  Flash-Next pair executor iterates `pair_schedule`. Placing `Init(B)` after the
  first kick is a latency choice (the pipeline fill), not a dependency.
* **Model check.** At open, `StageMap::check_model` compares each kind's layer count
  with the loaded model's mixers and `in_k` with its hidden size. A mismatch leaves
  the route off with that reason (`on`: load error), so a map can never price
  another model's plans.

Limits of the stage as it stands (a third kind or a fifth group is a schema change):

* Exactly two layer kinds (`KINDS = 2`) and four ownership groups (`GROUPS = 4`);
  the planner enumerates the 15 nonempty group subsets.
* Kind 1 is the attention mixer: the `qsa_*` GPU terms, `qsa_order` and the
  attention out-family order stall price it; kind 0 takes `gdn_core_per_layer_ms`.
* A pair's fill is the first bank-A chain of kind 0 when kind 0 moves anything,
  else of kind 1 (the layer stack is assumed to start with kind 0).
* A map with more natural families than four groups must merge them; the dense
  27B test map owns FA `k` and `v` as one `kv` group.

## Stored calibration and deterministic planning (0.4.2)

`Calibration::{load,store,validate,key_matches}` uses schema version **4** JSON
(4 removed `glue.fold_ms_per_col_8192` and the I32 contract: `npu.epilogue` and
`key.epilogue` must be `bf16`, BF16-direct only).
Measured fields are mandatory; only an omitted `key.build` defaults to the
release contract. Unknown fields/versions and invalid costs are errors.
Runtime `CalKey` derives:

* `device.arch`: the selected GPU's architecture; `device.pci_bdf`: the selected
  GPU's PCI address (HIP's PCI bus ID, lowercase; not an NPU slot);
  `device.npu_fw_version`: trimmed
  `/sys/class/accel/accel0/device/fw_version` (the amdxdna device railgun opens).
* `build`: `hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1`, shared by serving and
  `calibrate-npu-trunk`; no git SHA and no binary hashing.
* `kernel.uname_release`: trimmed `uname -r`. The recorded PerfOpt arming
  signal is `AMD-Vi: PerfOpt armed` in privileged dmesg, not a readable
  sysfs/proc ABI. Schema 3 and later (including 4) therefore drop `perfopt_armed`, rather than inventing
  a boolean from a kernel name. Recollect if boot arming changes under the same
  release: this key cannot distinguish that case.
* `model_trunk_index_sha256`: `model_trunk_index_sha256(&HfqFile)`, an index-only
  digest of what the NPU trunk depends on, equal for a monolithic MQ4 and its
  split MQ4 XTS base. Preimage (little-endian): domain
  `hipfire-npu-trunk-index-v1\0`; `u32` arch id; `u64` length + canonical
  metadata JSON (effective metadata without the top-level `hipfire_ple_rows`
  marker, compact, object keys sorted at every depth); `u64` tensor count, then
  per tensor sorted by name: `u32` name length + name, `u8` quant type,
  `u32` dim count + `u32` dims, `u64` data size. Only the 128 PLE shard tensors
  `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_N.weight`
  (canonical decimal `N`) are excluded; offsets and tensor payloads never enter.
  Both the verified local `qwen3.8-flash-next-gptq3.mq4` and its split XTS base
  digest to `098bd75fd3a6a5174d06033f2b3ec7febeb7777e7eba4696b64cecca5e3418b6`.
  CPU proof (`cargo test -p hipfire-arch-qwen4 npu_trunk::calibration`): the always-on
  `split_and_monolithic_resolve_the_same_calibration_admission_and_pairs` builds a
  monolithic and a split HFQ fixture and runs the shared `bind_builtin`,
  `auto_admission`, `select_families`, `crossover`, `plan_executable` and
  `executable_plan_for` for both; family mask, eligibility table and the executable
  16K/64K pair ranges must equal the built-in record's. With
  `NPU_TRUNK_TEST_MODELS=<monolithic>:<split base>` the same comparison runs on the
  real artifacts (it requires one of each layout; unset skips).
  The trunk-overlay and expert-hot-set bindings keep the raw
  `HfqFile::index_sha256` over bytes `[0,data_offset)`. No full-file hash is
  computed at load.
* `epilogue`: serving uses `bf16`; `trunk_route`: `stock` or
  `overlay:<marker>` from the **successful overlay outcome**, not its presence.

Mismatch reasons name the first differing field in this order. Loading stored
calibration never measures or recalibrates. The two `g80bf16-{stock,overlay}-v2.json`
lab files and corresponding fixtures retain their historical filenames but now
carry schema 4 identity; measured cost inputs and Python/Rust cost goldens are unchanged.

`plan(&Calibration, FamilyMask, rows) -> Plan` and
`crossover(&Calibration, FamilyMask) -> Option<usize>` port the reference
`npu-trunk/crossover_model.py`, SHA256
`294bf437f58c8ac1568c5531a99034b6360bf0b8b1df9a4a2daf06ccf76da0ce`.
The planner uses an 8192-row chunk DP, GPU-only/pair choices, 64-aligned
tail rebalance, even re-banking with boundary tax, and a taxed single-chunk
split. `Step::Gpu { chunk }` and `Step::Pipe { a, b }` carry exact half-open
prompt row ranges, including every pair of a re-banked plan. GPU wins ties;
existing earlier pairing wins ties against rebalance/re-bank. `min_bank_rows`
has the reference's placement (rebalance, single split, and re-bank eligibility).
GPU costs distinguish 36 GDN and 12 QSA layers and prefix-dependent attention;
NPU time rounds fused columns up to `panel_cols`. Contention is a burst term
`(alpha_overlap - 1) * min(G_other, N_this)` plus, when `ms_per_gb` is set,
`ms_per_gb * bytes / 1e9` over the NPU DRAM bytes of the overlapped chains
(whole-prefill slowdown is not modelled). The per-layer stall is
`max(0, N_chain - G_other)` per bank: with the measured balanced pairs (K1: GDN chain
58.3 vs window 61.5 ms) it is zero, so the stall is the pipeline fill alone (steady
state ~1 ms/pair, GPU wait 40 ms = A0 fill) and it only bites unbalanced pairs.
The planner takes a `Moved` set (`Moved::from(FamilyMask)` for the in-proj families;
`pair_time_moved` / `bank_segments_moved` / `plan_moved` also take `gdn.out`/`qsa.o`,
which no `FamilyMask` bit reaches yet, so `select_families` never selects them).
A moved `qsa.o` prices the order `C_A C_B P_A P_B`: each chain stalls by
`max(0, chain - the GPU window it overlaps)` with the measured Core/post windows.
GPU segments include pack once per moved layer shape and consume on each
selected family's live (not panel-padded) columns. `consume_mode=retile`
charges `rows/8192 * cols * consume_ms_per_col_8192[out_dtype]` per layer;
`direct` charges zero. Consume is GPU work, not NPU time or fill.

`margin` is a fraction in `[0,1)`. Every candidate must cost strictly less
than `(1-margin)` times the GPU-only cost of the rows it replaces, including
tax for splits/re-banking. At nonzero margin every re-bank pair must also pass
its local margin test. Margin zero preserves the Python reference. The crossover
is a **diagnostic**, not a hard-coded activation threshold: it scans
1024..=49152 by 64 and finds the first point whose entire remaining scan
suffix beats GPU-only. Route choice always comes from the prompt's plan.
No live clock, throughput measurement, or previous request time is consulted.

The measured families are grouped as `qkv`, `zab`, `q`, `kvidx`. Whole `zab`
and `kvidx` groups map to the existing family mask; a partially selected group
has no calibrated attribution and stays GPU-only. Calibration column counts
are used verbatim (the legacy lab examples differ from current packing widths).
Never apply a legacy example key to a production model.

`select_families(&Calibration) -> (FamilyMask, SelectionReport)` evaluates all
15 nonempty subsets with `plan` at 16384, 32768, and 65536 rows. The winner
minimizes mean `T_pipe/T_gpu`; exact ties prefer fewer groups, then the sorted
family-name sequence lexicographically. That winner must strictly beat GPU-only
by `margin` at 16384, otherwise the mask is empty (no runner-up substitution).
The report includes all subset times/ratios and the pre-gate winner for logging.
A zero-cost GPU baseline has ratio 1 and cannot pass the strict gate.
Python prints the same table with
`python3 crossover_model.py cal/g80bf16-overlay-v2.json select`;
an optional final `retile|direct` argument overrides consume mode without
changing the stored record.

`load_log_line` takes a `LoadLogKey { current, mode, families }`, crossover,
and plans at 16384/65536. It checks the key, hashes canonical serialized
calibration JSON, and logs the load-line fields (mode, families, key/hash, crossover, predicted times), with times as
pipe/GPU milliseconds. `request_log_line(rows, &plan)` prints e.g.
`npu-trunk: rows=16384 plan=[pipe 0..8192+8192..16384] predicted=-14.7%`.
The key/hash plus stored calibration identify every deterministic model input.

### Assembling measurements

```sh
hipfire calibrate-npu-trunk --from measurements.json --out cal.json
```

This command assembles and validates an already measured record; it does **not**
perform measurements, probe hardware, fit contention, or default measured costs.
The measurements schema is the same as the schema-version-4 output record:

* `schema_version: 4`, `margin` and the current `key` described above.
  The collector supplies device/firmware, kernel release, model **index** digest,
  epilogue and trunk route. Omitted `key.build` uses the stable release/NPU
  contract. The CLI checks `uname -r` and refuses a kernel mismatch. Serving
  compares **all** key fields of a calibration file to its runtime identity.
* `gpu.trunk_in_ms_per_8192`: required `qkv`, `zab`, `q`, `kvidx` millisecond
  totals across the respective 36/12 layers from kernel-trace attribution.
  `gpu.uniform_per_layer_ms`, `gdn_core_per_layer_ms`, `qsa_per_layer_ms`,
  `qsa_growth_per_layer_ms_per_8192_prefix`, `fixed_per_chunk_ms`,
  `boundary_tax_ms`, and `min_bank_rows` are required.
* `families`: those four names, each with `layer` (`gdn` for qkv/zab,
  `qsa` for q/kvidx), positive fused `cols`, and `out_dtype` (`bf16` for qkv,
  `f32` for the other families in today's serving route).
* `glue.pack_ms_8192`, `sync_ms`: measured pack/sync costs (schema 4 has no
  fold term: every contract is BF16-direct).
  `glue.consume_mode` is `retile|direct`; the required
  `consume_ms_per_col_8192` object contains finite nonnegative `bf16` and `f32`
  costs. Direct mode keeps those measurements but charges zero consume.
* `npu.tops_corun`, `bytes_per_kop`, `epilogue`, `panel_cols`: measured
  co-running rate, measured or declared traffic, chosen output contract, padding.
* `contention.alpha_overlap` (>=1) and `model`: fitted/measured overlap
  slowdown and the provenance/fit description. The planner consumes the alpha;
  it does not reinterpret the descriptive model string. Optional
  `contention.ms_per_gb` (planner refit, schema 4): contention per NPU DRAM GB of the overlapped family
  chains (`bytes_per_kop * kops` per chain, plus the out family's own bytes/kop).
* Optional planner-refit fields (schema 4; records without them load unchanged): `glue.visit_ms_8192`
  (GPU overhead per 8192 rows of a bank visit that moves families),
  `npu.tops_alone` (in-proj alone rate the co-run rate is measured against),
  `families.gdn.out` / `families.qsa.o` (an `out` block: `k`, `panel_cols`,
  simulator `tops_alone`, `bytes_per_kop`, `gpu_ms_per_8192` of the GPU o-proj GEMM it
  removes; their co-run rate is `tops_alone * npu.tops_corun / npu.tops_alone`), a
  per-family `selectable` flag (default true; the two out families ship `false`), and
  `qsa_order` (measured QSA Core/post windows and in-proj chain for the order
  `C_A C_B P_A P_B`; required with `qsa.o`).

Collect once per exact device/build/kernel/PerfOpt/model/epilogue/trunk-route tuple.
The collection procedure lives in the maintainers' NPU lab workspace, not in this repository:
`ptrace.sh` attributes GPU kernel costs by layer/family; `contend.sh` measures
co-run throughput and the contention fit; `tax.sh` measures the extra boundary
tax; `g80bf16.sh` measures the direct-BF16 NPU rate/traffic. Measure GPU glue
separately and record costs on the same tuple. Preserve those source results
alongside the record. Hardware measurement is a separately scheduled gate;
the CLI performs only offline assembly and never claims to have measured.

### Offline golden parity

`tests/fixtures/npu_trunk` preserves the three BF16 legacy calibration JSONs
(`g80bf16-{est-fast,est-slow,measured}.json`) and the stock/overlay v2 records;
the I32 records (`g80i32-est`, `v8i32-measured`) are gone with the I32 contract.
`golden.json` captures the actual Python CLI
stdout and full-precision imported-model times/plans for both family sets,
at 8192, 9319, 12288, 16384, 24576, 32768, 57544, 65536 and each crossover:
54 exact-range vectors, 6 crossovers, relative time tolerance 1e-6.
Tests adapt legacy fixture keys/version/margin and Python's absent-panel default
(`panel_cols=1` for V8), not measured costs.
`golden-v2.json` adds all-subset selection and half-bank consume vectors for
stock/overlay × retile/direct, nonzero margins and a zero-cost GPU
baseline, plus 40 exact-range plans. `golden-exec.json` pins the executable
projection (`executable_plan`: natural and rebalanced pairs) and the prefill group
schedule (`schedule`: banks and the in-bank snapshot cut, cache off / on / on
without sub-launches) for 8000..65536 rows on the stock/overlay records.
`golden-v5.json` (fixture `g80bf16-overlay-v5.json`, a schema-4 record; `v5` names the planner refit) pins the refit model: the 63-subset
selection table (the 15 in-proj subsets compared in Rust, out-family subsets `selectable: false`),
per-set bank segments (in-proj/out chains, NPU bytes), pair times (stall, contention, fill)
and plans including the out families. The same
tests run in the library and the `npu_trunk_plan` integration target (the
schedule golden runs in the library, where the scheduler is visible).
**All goldens are generated** by the lab's `gen_goldens.py` (maintainers' NPU lab
workspace, not in this repository) into
`crates/hipfire-arch-qwen4/tests/fixtures/npu_trunk` from `crossover_model.py`
and the fixture calibrations (a pure function; no hardware), never edited by hand.
The migrated lab `g80bf16-{stock,overlay}-v2.json` records preserve measured
costs and add Halo j110 consume rates: BF16 `2.85/16480`, F32 `4.60/16480`
ms per column per 8192 rows. Overlay changes only trunk identity and zab
GPU cost to 154 ms/chunk (`228 - 74` for the measured A4 tier); the NPU
W8 stock-MQ6 numerical contract is unchanged.
Run offline:

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
nice -n 19 cargo test -p hipfire-arch-qwen4 --lib npu_trunk -j 12
```

## Calibrated serving route (Linux gfx1151, family B)

`HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` is resolved at load.
The default is `auto`, effective only on gfx1151 (Strix Halo). Other arches
resolve auto silently to off before index hashing, calibration/device I/O,
packing, NPU allocations, trace opens or new logging. Explicit on/emulate
refuse non-gfx1151. `off` / `fusions.npu_trunk = off` is the kill switch.

`fusions.npu_trunk_calibration` (alias `npu_trunk_calibration`, env
`HIPFIRE_QWEN4_NPU_TRUNK_CAL`) is an optional strict override; when empty it names
`<hipfire home>/npu-trunk/calibration.json`. A file at that path takes precedence.
Only an absent file falls back to the compiled-in measured Halo schema-4 records,
the normal path: `halo-xdna2-overlay-v4.json` (`overlay:gfx1151-zab-v1`) or
`halo-xdna2-stock-v2.json` (`stock`). The stock lab record is schema-compatible,
keyed stock, and measured for `hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1`.
Invalid/unreadable files never fall back. Successful arming logs
`cal_source=file|builtin`. A file must match every key field. The built-in
record binds on the hard fields only (`device.arch`, `build`, `model_trunk_index_sha256`,
`epilogue`, `trunk_route`; `Calibration::bind_builtin`): it takes this host's
`device.pci_bdf` (the GPU's PCI address, not an NPU slot), `device.npu_fw_version`
and `kernel.uname_release`, keeps `margin` 0.0, and when any of them differed the
load logs one line
`npu-trunk: built-in Halo calibration (measured on kernel <k>, NPU firmware <fw>, GPU <bdf>) applies by arch/model/route; this host kernel <k>, NPU firmware <fw>, GPU <bdf>`.
The speed it predicts for another kernel/firmware is corrected by the live rate
scales below, not by a margin. The reference key is
gfx1151, the reference unit's GPU PCI address (`device.pci_bdf` in the built-in record), firmware `1.1.2.65`, uname `7.3.0-rc7-perfopt+`,
model index `098bd75fd3a6a5174d06033f2b3ec7febeb7777e7eba4696b64cecca5e3418b6`,
BF16 and the matching stock/overlay route; the builtin does not admit another
arch, build, model, epilogue or route.

Admission runs in this order, before any stock MQ6 download/packing or NPU allocation:

1. Flash-Next family-B K=2560, stock MQ6 shapes for every trunk family,
   single-device/no EP, full scratch, 8192-row maximum chunk.
2. The exact `/dev/accel/accel0` node railgun opens must be an amdxdna PCI
   device with readable, nonempty `fw_version`; no runtime open yet.
3. Load file-or-builtin calibration and match device, firmware, build contract,
   uname, model index sha, epilogue and anticipated trunk route; select families.
4. Download stock MQ6 before the overlay swap, pack only admitted families,
   recheck the successfully applied route and actual scratch plan, then open
   the runtime and allocate the second bank. Any auto preparation/open failure
   frees partial resources and runs GPU-only. Armed logs are emitted only after success.

Every auto refusal on gfx1151 logs exactly one line and never fails a load. A model that
does not qualify (step 1) logs `npu-trunk: off (<reason>)`. A qualifying model whose
NPU path fails (steps 2–4: firmware, calibration, preparation, runtime open) logs
`WARN npu-trunk: off (<reason>); npu_trunk=auto requested the NPU but the trunk did not arm, ...`.
`npu_trunk=on` turns either refusal into a load error.

Faults. After admission an NPU runtime fault (chain watchdog, submit error, ERT state other than COMPLETED, relay exit)
never publishes a partial NPU result and never hangs the process: the pair executor commits position, PLE, QSA and
session boundaries only on success, and every fault aborts the runtime, which releases every GPU wait on the NPU at
once. Under `auto` the route then latches off for the rest of the process, logs one
`WARN npu-trunk: NPU pair fault, the route is off for the rest of this process and prefill runs GPU-only (<reason>); host kernel=.. iommu=..`
line, and the faulted request restarts GPU-only from its session start (the committed snapshot, or row 0; the live
continuation was consumed by the attempt), logging `npu-trunk: the faulted request restarts GPU-only from row N of M`.
From that start its output is byte-identical to `npu_trunk=off`: the latched route plans no pair, so scheduling is
exactly the `off` one. Later requests in the process are GPU-only, never touch the runtime, and add no sync or I/O.
`on` fails the request closed and stays failed (sticky), but still aborts the runtime so it never hangs; `emulate` has
no runtime and is unchanged. A forced advance in the middle of a turn (`spec_advance` without reset) cannot be
replayed, so it fails closed; the route stays off for later requests. Relay teardown is bounded at 3 s: a relay that
does not exit is detached and its banks are quarantined. `HIPFIRE_QWEN4_NPU_TRUNK_FAULT_CHAIN=<n>` injects a fault on
the nth relay chain for testing. The armed load line ends with
`host=[kernel=.. iommu=gpu:..,npu:.. amdxdna=.. npu_fw=..]`, the same facts the fault line carries. A GPU HIP error
inside a pair (the GPU context is sticky, so a rerun could only fail) also aborts the runtime and latches the route off,
but it logs no WARN and is not rerun: the request fails with its original error, as without the NPU.

The first default-on measurement used the Halo default serve config with the zab
overlay, three fresh processes per arm, before rebalanced pairs and row-adaptive
chains (fixture-bound; superseded by the figures in Serving status): p16k
7,878 → 7,173 ms (−8.9%), p56k 30,052 → 27,867 ms (−7.3%). Greedy `on` and
`emulate` bytes match; flag-off `serve_harness` battery and chain match beta.
Quality accepted: S11 j103 BF16-direct family B, ΔKLD WT2 −0.0005 / code24
+0.0052 (cap +0.015).

| Mode | Calibration unavailable/invalid/key mismatch | Match: all selected | Match: subset/none |
| --- | --- | --- | --- |
| `off` | No I/O | GPU-only | GPU-only |
| `auto` | Route off, one reason line | Pack ALL; each request picks its subset from the live scales | Same (the static selection is only the admission check); none stays native |
| `on` | `calibration=none (forced)`, ALL, natural 8192+B pairs | Use selected families; planned pairs from the first row (no crossover arming) | Use subset; none stays native |
| `emulate` | Same as `on`, GPU twin | Same as `on`, GPU twin | Same as `on`, GPU twin |

An auto key mismatch (any field for a file; a hard field for the built-in) logs exactly
`npu-trunk: off (calibration key mismatch: <field>)`.
Matching records log `load_log_line` with selection, key/hash, crossover and
predicted 16K/64K times. Forced modes pack, build programs and C arenas for that selected mask and plan
every request with it (or with `HIPFIRE_QWEN4_NPU_TRUNK_FAMILIES`); `auto` packs ALL (the static selection
must be nonempty to admit) and picks a subset per request. Without a match, forced modes use ALL.
Auto derives n* from the request's family subset and the estimator snapshot (cached until either changes). Each request's uncached prompt segment runs
`plan(cal, families, rows)` (auto: the request's family subset planned from the estimator snapshot, see
Live rate scales below; forced modes: the static calibration) and logs `request_log_line` once. Natural planned
8192+B pairs, with B >= `max(512, cal.gpu.min_bank_rows)`, and the planner's
`pipe-rebalanced` last pair (two banks <= 8192 rows, 64-aligned split, each >=
`max(512, min_bank_rows)`, ending the request) execute above n*. Split and
rebanked ranges are projected to ordinary GPU chunks, never force-paired. The
logged executable projection reflects those restrictions; a second snapshot row
inside a pair may additionally shorten a candidate group. Rebalanced pairs and
in-bank snapshot cuts run in the default profile too: partial Core windows route as
their bank (`with_bank_routing`), and banks need at least 512 rows.

### Live rate scales (auto)

Under `auto` the matched calibration (built-in unless a file overrides it) is the
*prior*; one estimator per process learns two rate scales against it and the
request planner reads its snapshot. Forced modes (`on`, `emulate`) never use the
estimator: they plan from the static calibration only, so `on` and `emulate`
stay the bit-exact comparable pair.

* **Samples.** After each executed pair the executor compares the measured NPU
  chain time (relay `Completion` statistics) and the GPU window hiding those
  chains (pack end to wait begin of each chain, minus the NPU wait of the bank visit
  inside it, so a slow NPU never reads as a slow GPU) against `pair_model` for the
  same pair. The ratio measured/model of each
  is clamped to 0.5-2.0 and folded into an EWMA (alpha 0.25; the first valid
  sample initializes it). Samples whose modelled time is under 50 ms or whose
  numbers are not finite and positive are ignored.
* **Scales.** The GPU-window scale multiplies the prior's `gpu.*` ms terms, glue
  pack/visit/consume terms and out-family GPU times; the NPU-chain scale divides
  `npu.tops_corun`/`tops_alone` and multiplies `glue.sync_ms`
  (`scaled_calibration`). Nothing else changes; scales 1.0/1.0 are the prior.
* **Settle then freeze.** The snapshot takes the EWMA values when either differs
  from it by more than 1%. After 6 valid samples with the last 4 each moving both
  EWMAs by less than 1%, the snapshot freezes (the EWMAs keep updating). A frozen
  snapshot reopens, taking the EWMA values, after 3 consecutive samples whose raw
  ratio deviates from the frozen scale by more than 5% in the same direction.
* **First request.** The first long request of a process runs the built-in prior
  (scales 1.0/1.0); later requests use what was learned.
* **Per-request families.** All four families (qkv, zab, q, kvidx) are packed at
  load under `auto`. At request start the planner picks the fastest family subset
  for the request's rows from the snapshot (`choose_families`), then plans the
  pairs with it. Both use `plan_executable`: the planner without its single-chunk split
  and re-banked options, which the executor cannot run (under learned scales the
  re-banked plan can win the model by a fraction of a percent and would otherwise
  be projected to GPU chunks). A subset runs a contiguous panel range of the packed chains
  (`ColumnMap::panel_range`), so no repacking happens per request; the rest of the
  fused projection runs on the GPU. Developer variable
  `HIPFIRE_QWEN4_NPU_TRUNK_FAMILIES` (comma list of `qkv,zab,q,kvidx`, or `all`)
  forces the per-request subset in every mode.
* **Guard.** After each pair, if the GPU's measured waits on the NPU (summed
  `wait_gpu` time over the pair's chains) exceed what pairing saves it (`pair_model`
  `savings_ms` under the current scales: the moved families' GPU projection time minus
  the pair's GPU glue and DRAM contention), the pair lost to GPU-only and the rest of
  that request runs GPU-only; one `npu-trunk: guard: ...` line per request. The rule
  compares a measurement with a prefix-independent model term, so the model's error
  at depth (its QSA growth term) cannot trip it; the planner's identity
  `t_pipe = gpu_only - savings + stall + fill` ties it to the plan.
* **Counters.** `HIPFIRE_QWEN4_NPU_TRUNK_SAMPLE_LOG=1` (developer) prints one
  `npu-trunk: pair-sample ...` line per executed pair: measured NPU chain time, GPU
  window and GPU wait on the NPU, next to the model the request was planned with, the
  prior and the scales after the sample. It reads the same always-on events and adds
  no synchronization.
* **Numerics.** Output differs only between the NPU and GPU projection routes
  (BF16-direct W8 rows on the NPU), both within the accepted KLD cap; `on` equals
  `emulate` bit-exactly.

For the historical stock-v2/overlay-v2 planner records, executable pair starts/lengths are:
8000: none; 9000: `(0,9000)` (4544+4456); 9319: `(0,9319)` (4672+4647);
16343: `(0,16343)`; 20000: `(0,16384)`; 57544:
`(0,16384),(16384,16384),(32768,16384),(49152,8392)` (4224+4168 tail);
65536: four `16384` pairs. n* is 6848 stock / 8256 overlay. The Python mapping is
`crossover_model.py executable_plan`/`schedule`; Rust is checked against
`golden-exec.json` generated by `gen_goldens.py` (see Offline golden parity).

### Host memlock limit

The armed trunk pins **1,053,843,456 B (1005 MiB)** against `RLIMIT_MEMLOCK`. That is
two banks of userptr arenas, each 8192×2880 B of A, the GDN-in and QSA-in C arenas
(272,629,760 + 230,686,720 B) and three 4 KiB placeholder pages. The driver pins them with
`pin_user_pages(FOLL_LONGTERM)` and charges them to the process's pinned memory (`VmPin`,
`amdxdna_ubuf.c:127-133`). Nothing else in the NPU path counts against the limit. The
heap, command and B-panel BOs are driver-pinned `VM_PFNMAP` mappings, which railgun maps
without `MAP_LOCKED`. Before 0.4.2 those flags made the kernel test 3.1 GiB of mappings
against the limit, although it never locked them (`VmLck` stayed 0).

At open, the trunk computes the exact requirement from its bank layout, `VmPin` + arena
bytes. If the soft limit is short and the hard limit allows, it raises the soft limit to
exactly that figure. Otherwise it fails closed with a message naming the bytes and the fix:
a `WARN npu-trunk: off (...)` line under `auto`, a load error under `on`. The 8 MiB
default that ssh sessions and many desktop logins get is therefore too small. Grant the
hipfire process a **bounded** limit, the requirement × 1.25 rounded up to the MiB:
1,318,060,032 B (1257 MiB). Do not use `infinity`.

```ini
# systemd unit or `systemctl edit` drop-in for the service that runs `hipfire serve`
[Service]
LimitMEMLOCK=1318060032
```

```text
# /etc/security/limits.d/90-hipfire-npu.conf (pam_limits logins; value in KiB)
<user>  soft  memlock  1287168
<user>  hard  memlock  1287168
```

An interactive shell whose hard limit already allows it can run `ulimit -l 1287168` before
`hipfire serve`. Tailscale SSH skips pam_limits and always starts at 8 MiB, so use the
systemd drop-in or `prlimit` for those sessions. Driver citations are in the comment above
`railgun::npu::Device::map_heap`. Arenas that need no memlock (driver/TTM allocations
shared over dma-buf) are planned for 0.4.3.

### Checking a host: `hipfire doctor --npu`

`hipfire doctor --npu` checks a Strix Halo host before anyone arms the route on it. It first runs GPU-free checks (a gfx1151 GPU, the `amdxdna` XDNA2 NPU and its `accel0` node and firmware, the memlock limit above, a ROCm runtime, the daemon, a local Flash-Next model that a calibration record admits, and no running `hipfire serve`) and names a tier: `reference` for a host that matches a validated profile, `compatible-unvalidated` for any other host that passes, `blocked` when a check fails. After confirmation it runs one daemon with `npu_trunk=auto` that alternates the GPU-only route and `auto` on cold 16K and 32K-row prompts (about 6–7 minutes in all), then compares `on` and `emulate` on a short greedy generation, which must be identical. It ends with a recommendation. `--apply` only ever sets `npu_trunk off`, on a measured loss or an exactness mismatch. `--json` prints a report with no hostname, user, path or PCI address, and `--soak` runs a long stability soak instead. The full reference (checks, profiles, how the verdicts are computed, the report fields and the soak files) is in [CLI.md](../CLI.md#hipfire-doctor---npu).

### Diagnostic GPU-only route

The A/B above needs a GPU-only arm on the same loaded daemon, so the daemon accepts a control message:

```json
{"type":"npu_trunk_route","route":"gpu"}
```

`route` is `"gpu"` or `"auto"`; the daemon echoes `{"type":"npu_trunk_route","route":"gpu"}` (or `"auto"`) and answers any other value with a validation error. It is a diagnostic control, not a config key or an environment variable, it sets a process-wide switch and touches neither the model nor the GPU, and it exists on Linux only (other platforms answer with an `unsupported` error). The switch is read once, at the next request's begin, and cannot change a request in flight.

A request that begins with the route on `gpu` takes the same path as a tripped fault latch (`route_off`): it runs exactly the `npu_trunk=off` schedule and plans no NPU pair, so it feeds no sample to the rate estimator. It leaves the fault latch and the rate estimator untouched, so switching back to `auto` finds both as they were; a tripped latch stays tripped. Each such request logs one line:

```text
npu-trunk: rows=<n> route=gpu (diagnostic)
```

`hipfire-daemon --version` prints `hipfire-daemon <version> <commit>` (the full build commit) and exits. It answers before any config, protocol or GPU work, so it is GPU-free. `hipfire doctor --npu` reads it to name the kernel set (`jit:<commit12>`) when the install holds no kernel-pack manifest.

### Installing and testing auto on a Strix Halo host (maintainer hardware gate)

After CPU acceptance, install the **same** release CLI/daemon on the Halo host.
The built-in calibration covers the reference model on any Halo kernel/firmware/GPU
address (it is keyed by arch, build, model, epilogue and route), so no calibration file is
needed there. A file at `<hipfire home>/npu-trunk/calibration.json` is needed
only to override it, and must be a schema-4 BF16-direct record whose key matches
this device exactly (schema-3 and I32 records are refused). These commands are
for the maintainer running the gate, not the CPU worker:

```sh
GATE=<gate directory on the Halo host>
ssh <halo-host> "mkdir -p $GATE/bin"
scp target/release/hipfire target/release/hipfire-daemon <halo-host>:$GATE/bin/
# Optional override only (schema 4, key must match this device):
# scp <schema-4 calibration.json> <halo-host>:<hipfire home>/npu-trunk/calibration.json
```

On the Halo host, `uname -r`, the GPU PCI address and
`cat /sys/class/accel/accel0/device/fw_version` may differ from the built-in
reference unit's (the load logs one `npu-trunk: built-in Halo calibration ...`
line saying so). An override file must match all three exactly (the reference
values are in the built-in record). Do not rewrite a measured device/kernel key
to hide a mismatch. The default route uses the
verified overlay-bound model; for the stock route instead set
`HIPFIRE_QWEN4_TRUNK_OVERLAY=0` (the built-in stock record then applies). Do
**not** inherit the CPU gate's visibility masks (`-1`); use the Halo device
selection. Keep the default session cache.

```sh
export HIPFIRE_DAEMON_BIN="$GATE/bin/hipfire-daemon"
export MODEL=<pinned Flash-Next MQ6 family-B overlay-bound artifact>
export PROMPT=<prompt file with at least 16K tokens>
# ROCR_VISIBLE_DEVICES: ordinal of the gfx1151 GPU (require gfx1151)
unset HIP_VISIBLE_DEVICES
for mode in auto off; do
  HIPFIRE_QWEN4_NPU_TRUNK="$mode" "$GATE/bin/hipfire" \
    bench "$MODEL" --ttft --prompt-file "$PROMPT" --runs 3
done
```

The changelog measurements used cold, nonce-prefixed requests to fresh serve
processes (3 per arm) through a lab serve harness that is not in this tree; the
serve flow in the Halo hardware gate below exercises the same HTTP route.

Require load `mode=auto`, matching `cal_key`, all families and (for the reference
overlay record, fixture-bound) n*=8256, not an off/mismatch reason. Require the
actual prompt row count and one request plan line; 16343 rows must select one
8192+8151 pair. Repeat fresh processes for `auto` and `off` at 16K/57K, and a
greedy prefix-sharing second request through the serve flow below with the same
model and default cache. Confirm sub-8256-row prompts (8000) stay GPU-only; 9319
must plan one rebalanced 4672+4647 pair and 57556 must plan 4224+4180 as its
final pair, each with `on == emulate` greedy output including a prefix hit at the
8192 snapshot taken inside bank B's Core sub-launch (C1 gate). The previously
documented snapshot and quality/TTFT gates still apply; CPU acceptance is not a
silicon claim.


The route requires Flash-Next K=2560 and stock MQ6G256V2 for all eight
input families. After metadata/device/calibration admission, `attach_forward`
downloads stock MQ6 and packs auto's selected S1 `Bf16Direct` panels **before**
applying the pending trunk overlay or developer requantization. Only packed
panels cross the overlay swap under auto; raw bytes are released. Forced
on/emulate retain stock host rows until the successfully applied route selects
their calibration, then pack panels (and the exact row-major W8/Sm/Se twin
for emulation). No extra stock GPU copies or second download are retained.

Both arms still load the beta `<model>.trunk-overlay`. Covered z/a/b use W8 from
the **stock MQ6**, never the replacement QT44 bytes. The existing load line adds
`trunk_route=overlay:gfx1151-zab-v1|stock` (only successfully applied overlays
count) and `zab_source=stock-MQ6-DtoH-before-overlay|stock-MQ6-DtoH`, alongside
`download_ms`, `pack_ms`, resident weight bytes and runtime status.
`off` does no download, packing, allocation or new logging; its existing mode
read moves before overlay application, whose payloads and native route are
unchanged. `on` and admitted `auto` open two 8192-row row-major G80 programs (smaller banks add row-adaptive ones, below),
resident 13/11-panel chains, and two banks each with registered A plus separate
GDN/QSA C arenas. Hardware open errors fail forced on loads; auto frees partial
resources, logs one refusal and remains GPU-only.

Both `on` and `emulate` use the same deterministic, segment-local pair plan.
Only paired banks use BF16-direct numerics: `on`/`auto` consume NPU results, while
`emulate` computes them with the exact IU8 twin on the GPU inside the same pair executor.
Unpaired chunks/tails, rows below 512, decode and uncovered paths retain the
incumbent GPU projections in **both** modes: z/a/b take the same A4/IU4 overlay
route as `off` wherever that native route applies. No paired input projection
is computed twice: `project_trunk_masked` starts covered families as completed,
then runs only uncovered families through the existing MQ6/A4/IU4 dispatch
and native scratch, in each bank's GPU fragment before Core. The debug assertion
is per family, in addition to the Core/OutProj phase assertion.
Shapes with no selected families keep Whole/native execution in both banks.
Under `on` the GPU packs A, then the mixer core directly reads read-only BF16
`Bf16Arena` views of fused C; **there is no consume/retile pass**. Explicit
`GdnTrunkArenas`/`QsaTrunkArenas` arguments select arena launchers only for
`Core`; omitted views retain the incumbent native scratch path. `emulate`
runs the same layer-major pair executor: its `prepare` runs the IU8 twin
projection into scratch where `on` packs and kicks, and its Core reads that
scratch. Bank B therefore owns private copies of the in-projection outputs
(`NpuBankKind::Emulated`; the load log's `bank_b_bytes` shows the extra bytes versus `on`). Out-projections and
GPU recurrence/attention/HC/MoE remain native.

Each arena family is `Option<Bf16Arena>`: GDN's fused convolution/parameter
consumer independently reads qkv from its arena or native BF16/F32 projection,
and a/b from their arenas or native F32 scratch. A zero pitch in the opt-in
kernel ABI denotes contiguous F32 scratch, with the incumbent parameter-core
BF16 rounding. Z selects the existing
native gate/chunk kernel when absent. QSA selects native norm/RoPE, index append
and V cache append per absent family. Emulate scatters only covered twin outputs
to native scratch, runs the identical uncovered projection dispatcher, and
executes the incumbent core; arena readers preserve the same BF16 numeric values.
Calibration selection still groups zab and kvidx because costs are measured
per group; dispatch and column maps support independent family views.

### Bank state and ordering

`on` requires the full 8192-row scratch plan and single-device placement.
Bank B is a load-time view of bank A's layer-local buffers, with only the
four fields that cross a bank switch allocated privately. Bank A's ownership/capacities
and all `off` allocations remain unchanged.

| Bank-B fields | Storage | Why |
| --- | --- | --- |
| `streams` | Private | HC residual rows persist through every layer and final capture/head; B must not overwrite A's rows. |
| `hc_mixed` | Private | Prepared mixer input survives the other bank's prepare/fragment. Uncovered native projections also read it in Core. |
| `hc_gates` | Private | The attention HC write's gates, produced by prepare's paired HC read (or by the bank's previous fragment's MoE HC row fold), survive the other bank's prepare/fragment until this bank's OutProj, whose `hcw` epilogue (or pregated BF16-stream write) consumes them. |
| `ple_rows` | Private | Initialized token-dependent PLE input survives until its layer's prepare. |
| `token_ids`, `embedding_rot`, `embeddings`, `ple_staged` | Shared | Each bank's initialization consumes these completely before the next initialization on the same stream. MTP uploads its own prompt IDs. |
| `hc_normalized`, `hc_low`, `hc_up`, `rotation`, `gdn_bf16` | Shared | Each prepare/fragment/head consumes its own values before returning; no value is handed across their boundaries. |
| `projection`, `projection2`, `gdn_a`, `gdn_b`, `gdn_gate`, `gdn_beta`, `gdn_recurrent_output`, `gdn_z`, `gdn_output` | Shared | Native uncovered projections, GDN Core and OutProj finish every use inside G(l,bank). |
| `qsa_index`, `qsa_qgate`, `qsa_k`, `qsa_v`, `qsa_output`, `qsa_selected`, `attention_output` | Shared | QSA transform/cache/index/attention and the attention HC write finish inside the fragment. |
| `ple_key`, `ple_value`, `ple_query`, `ple_gated`, `ple_normed`, `ple_output` | Shared | The PLE prepare consumes these before the bank switch. `ple_normed` is retained for ordinary verify rollback, not prompt pairs; pair snapshots copy fixed convolution state instead. |
| `router_logits`, `moe_x_rot`, `moe_gate_up`, `moe_gate`, `moe_up`, `moe_hidden`, `ffn_output`, `moe_shared_output`, `moe_gate_batch`, `moe_up_batch`, `moe_rot_batch`, `moe_topk_indices`, `moe_topk_weights`, `moe_down_expanded`, `moe_scalar`, `moe_expert_token_counts`, `moe_expert_offsets`, `moe_sorted_slot_index`, `moe_expert_tile_ids`, `moe_inverse_perm`, `moe_y_gate_up_grouped`, `moe_y_down_grouped` | Shared | Routing, shared/routed expert work, permutation/gather/combine and the FFN HC write finish before the next fragment. |
| `logits` | Shared | Head scratch is reused only after the preceding bank's head has consumed it. Requested logits/capture rows live in separate caller-owned outputs. |

The same-stream order is the sharing contract, not host launch completion.
`pack(l+1,A)` is queued before `G(l,B)`; its source `hc_mixed` is private.
Rotation's temporary F16 input is consumed by that pack before any subsequent
GPU rotation overwrites it. NPU DMA reads only its separately registered,
bank-private A arena, never the shared GPU layer intermediates.
Capture/MTP copies each bank's private `streams` into its own output row range
after both trunks finish. GDN recurrence/convolution, QSA KV/index caches and
PLE convolution state remain shared model state: B observes A's ordered updates.
NPU-pair execution uses mapped experts, not donor-layer staging.
The private NPU owner closes the runtime and frees bank B before bank A;
borrowed fields are never freed by B. None of railgun's internal HIP Rc
resources escapes the exclusive owner.

### Load footprint

Only `emulate` uploads the row-major W8 twin and its weight Sm/Se scales,
or allocates its packed-A and int32 accumulator (`sums`) buffers: lab scratch
private to the GPU twin (no int32 partial exists in the physical route). Physical `on`/admitted `auto`
retain NPU panels instead; tails, unpaired chunks and uncovered projections
use the native GPU route. Two compact activation sidecar buffers (32 KiB
each at 8192 rows) remain required by the GPU A-pack producer in both modes;
they are not the twin's weight scales.
The load log reports `resident_bytes=npu_panels_bytes+gpu_twin_bytes` for
resident weights, plus `bank_b_bytes` for the new private scratch allocation.

Allocation-plan comparison at 8192 rows (36 GDN, 12 QSA), all families: physical
`on`/`auto` against an `emulate`-style layout (GPU twin, unshared bank B). The twin
rows, including the int32 accumulator, exist only in that layout; `on`/`auto`
allocate none of them:

| Removed allocation | Bytes |
| --- | ---: |
| Bank-B layer-local scratch (`Full` plan 8,957,998,980 minus private 503,447,552) | 8,454,551,428 |
| GPU W8 twin (`(36*16640+12*14080)*2560`) | 1,966,080,000 |
| Twin weight Sm/Se (`(36*16640+12*14080)*4`) | 3,072,000 |
| Twin int32 accumulator, `emulate` only (`8192*16640*4`) | 545,259,520 |
| Twin packed A (`4*32*40*4608`) | 23,592,960 |
| **Total** | **10,992,555,908 (10.237615 GiB)** |

The twin-only subtotal is 2,538,004,480 bytes (2.363701 GiB); bank-B sharing
saves 7.873915 GiB and leaves 480.1 MiB private. Subset panel/twin savings are
additional and mask-dependent; the log reflects the actual selected mask.
These are allocation-plan bytes, not a GTT residency measurement (the changelog's measured peak GTT delta is +0.8 GiB at 56K).
CPU tests `npu_bank_sharing_covers_every_field_and_never_crosses_a_bank_switch`
and `npu_private_bank_bytes_are_pinned_at_full_and_partial_rows` enumerate all
58 fields, check produced-value lifetimes against the actual A/B event order,
and pin private bytes for 8192, 8151 and 512 rows.

Sharing is unsafe if a future change hands another layer-local value across
a prepare/mixer boundary, stages projections before the other bank's fragment, adds a
cross-bank stream, or retains a layer-local value for later capture/rollback.
Such a change must update the exhaustive lifetime table and private set.
The pair executor carries each bank's HC fusion pairings across its prepare,
Core and OutProj windows (`HcFusionCursor`), so the fused HC kernels the
GPU-only route takes (paired read gates, `hcw` out-projection epilogue, MoE
HC row fold) also run under the NPU route; only `hc_gates` crosses the switch.
Partial B changes its view metadata only; A's tensor extents and capacities
stay intact. Numerical operations/order and external capture offsets are
unchanged, but full-model `on`/`emulate` byte parity and GTT savings still
require the Halo hardware gate.

With an 8192-row chunk cap, a natural pair is 8192+B whenever at least 8192+512
rows remain: bank A has 8192 live rows and B has `min(remaining-8192,8192)`. The
planner's rebalanced last pair (two banks of about equal rows, 64-row aligned
split, each at least `max(512, min_bank_rows)`, e.g. 9,319 as 4672+4647 or the
57K tail 8404 as 4224+4180) runs the same executor: both banks bind their live
rows (`bind_bank_rows`; bank A's full extents are restored after the pair) and
bank B's state view advances by A's rows. `emulate` runs the same banks. A shorter unpaired chunk remains sequential. For
example, 16200 rows are one 8192+8008 pair, 20000 are a full pair plus 3616
sequential rows, and 40000 are two full pairs plus 7232 sequential rows. For
each layer:

```text
prepare(0,A), pack/kick(0,A); prepare(0,B), pack/kick(0,B)
wait(l,A), Core(l,A), retire A, Out/HC/MoE(l,A)
prepare(l+1,A), pack/kick(l+1,A)
wait(l,B), Core(l,B), retire B, Out/HC/MoE(l,B)
prepare(l+1,B), pack/kick(l+1,B)
```

`prepare` includes PLE when present and the attention HC read. `G` is mixer
core/out, attention HC write and MLP HC/MoE/write. The existing typed ops and
output/HC/MoE fusions execute those fragments; the numerical twin uses the
same fragment boundaries. Retirement is recorded after the complete Core,
which encloses every C reader; output projection and the rest of `G` read only
native scratch. `prepare_bank` queues a wait on that retirement event before
packing new A on the same stream. `kick` records its system-release A-ready
event after that wait and pack; the relay queries this event to completion
before submitting the next chain. Thus
`old C reads -> retired -> prepare wait -> pack -> ready -> next NPU C writes`,
even when the next layer changes shape. No arena is overwritten early.
`wait_gpu` now enqueues the per-bank monotonic GEQ terminal-signal wait and
system acquire event immediately, without a host-completion gate; retirement
also records its event without waiting on the host. The following complete `G`
fragment is therefore issued ahead of NPU completion. The relay remains serial.
Bounded host checks at bank reuse protect the ready-event generation, but depend
only on previously issued A production/NPU work, never an unqueued fragment or
the just-enqueued `G`. In `prepare/kick(l+1,A)` the checked chain is NPU(l,A),
whose terminal signal must precede GPU G(l,A): this dependency resolves by
G(l,A)'s start, leaving the host free to enqueue wait(l,B)+G(l,B) while G(l,A)
runs. Retirement itself is a GPU stream dependency, not a host wait for G(l,A).

Every kick receives a terminal signal, including relay errors, a 300 ms
chain-only watchdog (5x the measured ~60 ms chain), relay unwind and shutdown.
The registered signal is terminal rather than success-only; failures are sticky
and stored before releasing the GPU. Retire checks known errors, and pair-end
waits/checks finish before heads/capture or state commits. Failed pair results
are discarded and the request errors; late success cannot clear a failure.
Shutdown releases pending waits before joining the relay, then checks retirement
before unregistering/freeing (or quarantines still-live operands).

The exact live readers (pitch/col0 in BF16 elements) are:

| Arena/window | Core reader |
| --- | --- |
| GDN 16640 / qkv 0..10240, a 16384..16432, b 16432..16480 | `gated_delta_conv_params_batched_arena`, or its `qknorm` variant |
| GDN 16640 / z 10240..16384 | chunk `gated_delta_step_gate_wmma[_qknormed]_arena` (retaining normal/inline-Q8 selection), or non-chunk `gated_delta_gate_batched[_rotate]_arena` |
| QSA 14080 / q 0..12288, k 12288..12800, idx 13312..13952 | out-of-place `indexed_attention_norm_rope_batch_arena` for each window |
| QSA 14080 / idx key tail 13824..13952 | `indexed_attention_index_key_append_batch_arena` to F32 index scratch and raw-key cache |
| QSA 14080 / v 12800..13312 | `indexed_attention_cache_append_batch_arena(key=None)`; K comes from normalized F32 scratch |

No reader includes GDN padding 16480..16640 or QSA padding 13952..14080.
Attention/index selection read transformed F32 scratch, never raw C.

The NPU programs are row-adaptive (0.4.2). A bank of fewer than 8192 rows runs
the G80 program of its own wave count: the cores' wave loop count is part of the
PDI, so the relay builds that PDI/TXN on its thread the first time the count is
kicked, gives it its own hardware context and caches it per (class, padded rows)
(`railgun::npu::trunk::TxnBuilder`, `chain_rows`). The producer packs the A
stream for the same padded rows (whole 256-row waves) and explicitly zeros every
padded row; kick records the live count, and all arena readers use only the
live rows. Without this (before the row-adaptive programs; Halo measurements) a 4,672-row bank paid the full 8192-row chain (55.7 ms
for a GDN layer against a ≈ 35 ms GPU window), which made the rebalanced 9.3K
pair 13 % slower than `off` (Halo H2b); with it the chain scales with rows
(34 ms at 4,672, Halo H4). The relay holds at most `MAX_ADAPTIVE_CONTEXTS` (4)
row-adaptive contexts per load; the request plan admits the wave counts of its
pairs in first-use order (`admitted_plan`, every mode, so `on` and `emulate`
agree) and a pair that would need a fifth count runs GPU-native instead.
All GPU layer operations, router views, PLE staging, capture/MTP rows and
head work use the bank's real row count. Bank B's output offsets remain
8192, not its own row count.
Before building a partial B program, row-checked bank-B extents are rebound
from the existing `Full` plan for live rows (including grouped MoE, router,
PLE staging/output). Four dedicated native padding buffers—HcMixed, GdnZ,
GdnA and GdnB—retain their original max-chunk logical capacities, exactly as
sequential `off`/`emulate` prefill. Mixer operations still pass live rows:
8192/512 take dense IU4 when admitted; 8151 takes the same padded ragged IU4
route (or the same native fallback when ragged is disabled) in both modes.
The pair and sequential forward share the native prefill-scope gate and
restore it on success/error. Padded input clearing touches private HcMixed;
shared z/a/b outputs are produced and consumed only inside one GPU fragment,
never across prepare. Allocation/ownership stay unchanged, with no extra
allocation/copy. CPU coverage pins decisions and extents at 8192, 8151 and 512.

Bank A starts at the request's current position; B starts at position+8192.
Each bank-B QSA view is advanced using bank A's exact `next_lengths()` tuple
(full/raw/pooled/selected/position), rather than independently guessed offsets.
GPU `G(l,A)` precedes `G(l,B)`, so B sees A's recurrent updates and appended
KV. PLE row hashing advances history through A before hashing B. Only after
both banks finish do the head, requested wide-hidden/MTP capture, QSA metadata
and global position/history publish. A boundary at A's end is retained through
fixed-state staging without rewinding the live pair-end state.

The serving entry points are `Qwen4MtpDrafter::mtp_prefill` (default native
MTP), `SpecTarget::spec_advance` (generic speculative target prefill), and
`Qwen4Bundle::prefill_final` (AR, including `--spec off`). The native MTP and
generic target loops now offer up to two chunks to the target instead of
pre-splitting every request at 8192. CLI TTFT bench and fresh singleton HTTP
chat requests both dispatch through these same generate routes.

`on` and admitted `auto` double `spec_hidden` to 16384 rows; verify windows and MTP append
scratch keep their original capacities. MTP head append still runs separate
A/B subchunks with the original fill-row boundaries and prompt-ID uploads:
bank B reads hidden rows starting at its bank offset in every mode.
Pending hidden names the last row of B.
Only the final bank of a prompt-ending pair runs the target pick; other pairs
capture silently. Forced modes log their first scheduled pair once, with the request's
family groups: `npu-trunk: pair rows=8192+8151 position=0 families=qkv,zab,q,kvidx` for a
16343-row fresh prompt, and `rows=4672+4647` for a 9,319-row one (the rebalanced pair). With a matched
calibration `on` and `emulate` run the same executable plan as `auto` (natural and
rebalanced pairs), armed from the first row instead of the crossover n*, so
`emulate` exercises exactly the pairs `on` runs; without a matched calibration
they keep the natural 8192+B pairing.

A pair may carry **one interior session snapshot row**, anywhere in the group
(C1, 0.4.2). At the A/B split bank A runs whole and its fixed state is staged
after it, as before. Inside a bank (`coproc::pair::CutSite::Bank`), that bank's Core runs as
two launches `[0,r)` and `[r,rows)` (`execute_*_core_rows`, `CoreWindow`):
GDN: each launch ends with the ordinary chunk-end Q8 round trip (frame
`position + lo + rows - 1`) and the conv ring cursor `(start + lo) % 3`; QSA:
the second launch starts from `next_lengths_after(r)`; the PLE grouped depthwise
step of that bank splits at the same row. The staged copies (GDN recurrent +
conv state, QSA selected indices, PLE conv state) are taken after the first
launch and never after the second; the layer's input projections, output
projection, HC and MoE stay bank-wide. A second interior snapshot row ends the
group at the first boundary as an ordinary sequential bank. `emulate` runs the
very same `pair_scoped` (same banks, same Core windows, same staging and
publishing: `launch_geometry` is the single source), with its GPU twin
projection in place of pack + kick, so `on == emulate` holds by construction;
the Halo comparison must retain the default cache and exercise a later prefix
hit at 8192 after a rebalanced pair. Partial windows launch under the routing
of a canonical whole chunk (`with_bank_routing`), so no invariant-profile
setting is needed and banks must have at least 512 rows.
See the staging layout and cost details below.
Generic resumed target advances whose pending snapshot cursor is already behind
their position remain native until that snapshot is reported by its owner.

### Overlap trace

With `HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk.jsonl`, successful pairs
append one JSON object per `(layer,bank)` with `position`, real live `rows`, `seq`,
`pack_issue_ns`, `kick_ns`, `npu_submit_ns`, `npu_done_ns`,
`gpu_wait_enqueue_ns`, `consume_issue_ns`, `mixer_start_ns`, `mixer_end_ns`
and `anchor_uncertainty_ns`. Host/relay timestamps use Linux CLOCK_MONOTONIC.
Fine fields `prep_begin_ns`, `prep_end_ns`, `pack_end_ns`, `wait_begin_ns`,
`wait_end_ns`, and `signal_write_ns` separate GPU HC/PLE/rotation preparation,
packing, the device wait plus system acquire, and relay terminal publication.
`prep_end_ns` follows rotation and precedes packing; `pack_end_ns` follows the
pack kernel. `signal_write_ns` is sampled after the terminal store and its 64-B
CLFLUSH. All GPU timestamps are after-pair HIP event resolutions, not CPU issue
times. The same-stream pipeline and its bank/snapshot ordering are unchanged.
`consume_issue_ns` is retained for trace compatibility and is always zero.
The mixer interval covers the complete `G` fragment, including HC/MoE.
HIP timing events are resolved only after the pair, relative to a monotonic
anchor calibrated after initialization and **before** pipeline work; its host
bracket width is reported as uncertainty. There is no per-layer GPU sync.
`gpu_wait_enqueue_ns` records the actual enqueue-ahead call and should precede
`npu_done_ns` while work overlaps. Relay submit/done timestamps are harvested at
the next bank reuse, after `prepare_bank` has already waited for that chain,
and at the final pair check. The trace lookup adds no new completion wait at
bank reuse. Extra timing events and anchor/final GPU synchronization can still
change scheduling. Keep tracing unset for
primary correctness and performance runs; use a separate fresh traced run to
inspect completed chains and overlap. A traced pass alone does not establish
untraced handoff correctness. The transport gate additionally
asserts enqueue-before-done on every exact round and prints NPU-done→GPU-resume
event latency with clock-anchor uncertainty, plus a forced-watchdog release test.
No trace records are emitted for GPU-only tails or `emulate`.

### Halo hardware gate (maintainer-run)

No full-model hardware parity or speed claim is established by CPU builds.
Use the same MQ6 family-B model, prompt and greedy settings in fresh daemons;
confirm the tokenized prompt contains at least 8704 rows (a file named `p16k`
is not itself evidence of its token count). Keep chunk admission at 8192.
Do not inherit the build-only `ROCR_VISIBLE_DEVICES=-1`/`HIP_VISIBLE_DEVICES=-1`.

Run transport before either serving arm, with devices visible:

```sh
target/release/npu-trunk-transport --pci <gfx1151 GPU PCI bus ID>
# Exercise the production cross-crate handoff, with hip-bridge loaded first:
target/release/npu-trunk-transport --pci <gfx1151 GPU PCI bus ID> --gpu-runtime serving
# The same producer/consumer streams under GPU contention:
target/release/npu-trunk-transport --pci <gfx1151 GPU PCI bus ID> --gpu-runtime serving --corun
```

The default `--gpu-runtime relay` retains the standalone transport baseline.
`--gpu-runtime serving` uses hip-bridge's allocations, copy kernels, streams and
events for the GPU producers and consumers, including the co-run stream and
forced-watchdog release control. The NPU relay still owns the registered arenas
and its handoff events, just as it does during serving. Before submitting work,
the gate requires both live runtimes to resolve the same `hipEventRecord`
address; library paths are logged for diagnosis. A divergent preload fails at
`stage=GPU-runtime-identity` rather than exercising unordered foreign handles.
Both modes retain the full-C CPU oracle, alias/padding checks, enqueue-ahead
assertion and all three negative controls. This transport result is hardware
handoff evidence; the fresh-daemon model parity below remains a separate gate.

```sh
export HIPFIRE_DAEMON_BIN="$PWD/target/release/daemon"
export HIPFIRE_PREFILL_CHUNK_ROWS=8192
unset HIPFIRE_SESSION_CACHE_BYTES # retain the product-default 8 GiB prefix cache
unset HIPFIRE_QWEN4_NPU_TRUNK_TRACE # primary parity runs omit trace instrumentation
# MODEL must name the pinned MQ6 family-B artifact, not an MQ4 artifact.
HIPFIRE_QWEN4_NPU_TRUNK=on \
  target/release/hipfire serve "$MODEL" 127.0.0.1 11435 --max-seq 32768
```

For a fresh, non-streaming singleton chat request against that server:

```sh
jq -n --arg model "$MODEL" --rawfile prompt p16k \
  '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:64,stream:false}' \
  | curl -sS http://127.0.0.1:11435/v1/chat/completions \
      -H 'Content-Type: application/json' --data-binary @-
```
Save the response, then send a second independent request with the same p16k
prefix and a different suffix (not the first response's generated continuation).
Require a reported prefix-cache hit with reused rows > 0. Save that response too.
Stop the `on` server, then start a fresh `emulate` server with the same settings
and send both identical requests in the same order:

```sh
HIPFIRE_QWEN4_NPU_TRUNK=emulate \
  target/release/hipfire serve "$MODEL" 127.0.0.1 11435 --max-seq 32768
```

Compare generated token IDs/content byte-for-byte separately for the cold request
and cache-hit request (not the nondeterministic response ID/timestamp/timings).
`on` must equal `emulate` on both. Inspect the `on` trace for paired A/B banks
despite the default cache in a separate fresh diagnostic run with
`HIPFIRE_QWEN4_NPU_TRUNK_TRACE` set. For the untraced runs, retain the first
scheduled-pair log, `mode=on`/`runtime=open` receipt and successful nonempty
response together; the scheduled-pair line alone precedes execution and does
not prove completion. Separately compare `off` to unchanged beta, not to
BF16-direct `on`; repeat the cold/cache-hit gate with native MTP and `--spec off`.


Native MTP is enabled by default; repeat with `--spec off` for the AR route.
In a separate terminal, then repeat in fresh processes with `emulate` and `off`:

```sh
HIPFIRE_QWEN4_NPU_TRUNK=on \
  target/release/hipfire bench "$MODEL" --ttft --prompt-file p16k --runs 3

for mode in on emulate off; do
  HIPFIRE_LOCAL=1 HIPFIRE_QWEN4_NPU_TRUNK="$mode" \
    target/release/hipfire run "$MODEL" --temp 0 --repeat-penalty 1 \
    --no-stream --max-seq 32768 --max-tokens 64 "$(cat p16k)" \
    > "/tmp/npu-trunk-$mode.txt"
done
cmp /tmp/npu-trunk-on.txt /tmp/npu-trunk-emulate.txt
# Compare off to the unchanged beta binary's output on the same model/settings,
# not to emulate (emulate intentionally changes the trunk numerical contract).
```

Continuous-batch/lane `prefill_wave` is not paired; its tiles retain the native
GPU route in both arms. A pair carries at most one interior session snapshot row
(see above); a second interior boundary ends the group at the first and runs as
an ordinary sequential bank. Message-end cadence and cache scope are unchanged.
EP and compact scratch are outside the route (auto refuses them at admission).
CPU scheduling/capture/retirement tests live under
`gpu_forward::npu_forward::tests`; existing MTP tests cover head append IDs
and pick boundaries. Full-model greedy comparison and trace overlap
inspection remain the maintainer-run silicon acceptance gate.

### Default-cache snapshot staging

`state.rs::session_layout/session_parts_at` defines the trunk snapshot:
each GDN layer's recurrent live slot and F32 convolution history; each QSA
layer's complete selection buffer and append-only K/V, raw-index and completed
pooled-index prefixes; optional PLE convolution history; and hyper feedback.
Host metadata is model ID, absolute position, optional two-token PLE context,
and six words per QSA layer (full/raw/pooled/partial/selected lengths, position).
`mtp_gpu.rs::session_parts` adds the head's selection buffer, device selected
length, wide hidden, four append-only streams, model ID, lengths, position,
step index, and draft policy's full-vocabulary hold/margin bits. Request-local
pending scratch is not a session-cache part.

For `on`, `session_pair.rs` allocates one reusable fixed-state mirror before
the first boundary-bearing pair, only with an attached cache. In
`npu_trunk/forward.rs`, GDN recurrent/conv and QSA selection are copied by a
stream-ordered copy kernel immediately after each A layer and before its B
layer. PLE is copied immediately after A's PLE operation, before B's preparation
can overwrite that shared history; unchanged hyper feedback is copied before
the pipeline starts. No staging copy synchronizes the host, including on the
default HIP stream. The ordinary metadata serializer describes A's QSA marks
and token history without rewinding the live B owners. Debug builds reconstruct
the layout through the ordinary restore parser and compare fixed sizes, stream
row widths/counts on the CPU, without draining the stream.

The cache does **not** store QSA KV by lengths alone: `session_cache.rs` copies
append-only row deltas above the parent and pins that parent. A staged snapshot
therefore supplies A's lengths over the still-valid live prefixes, plus staged
fixed bytes. Publication uses the unchanged `SessionCache::at_boundary`:
after the pair and A's MTP head append, before B's head append; AR publishes
inside `pair()` at the snapshot row. `emulate` runs the same pair executor and
publishes the same way. The next B-end snapshot is ordinary in all modes.

Additional device memory is the sum of trunk fixed-part byte sizes, independent
of prompt length; no KV mirror and no extra MTP mirror are allocated. At pinned
Flash-Next geometry with F32 GDN this is **118,177,936 B (112.70 MiB)**:
36 recurrent/conv pairs, 12 selections of 2051 i32s, nine 10240-wide PLE history
rows, and 10240 feedback floats. Other GDN storage formats use their actual
recurrent byte sizes. Host staging retains one metadata blob and 48 row counts.
Each boundary-bearing pair adds that many logical copy bytes, or about
236.36 MB of read+write traffic, in 50 copy-kernel launches (one per GDN/QSA
layer plus PLE and feedback). A hypothetical effective 100–200 GB/s gives
**2.36–1.18 ms plus launch/serialization cost**; this is a bandwidth estimate,
not a measured Halo timing. Cache publication's ordinary fixed/row copies and
allocation costs remain in addition, just as for sequential capture.

CPU acceptance uses GPUs hidden, `nice -n 19`, the worktree's target directory
and `-j 12`: build `hipfire-daemon`, `hipfire-cli`, and `qwen4_kld --features lab`;
run `cargo test -p hipfire-arch-qwen4 --lib` and
`cargo test -p hipfire-runtime --lib session_cache`. These checks do not certify
snapshot bytes on silicon; the cold/default-cache and branched cache-hit
`on`/`emulate` Halo gate above remains required.


## Direct BF16 arena consumer interfaces

`rdna_compute::tensor_ops::Bf16Arena { tensor, row_pitch_elems, col0_elems }`
is an immutable BF16 view; pitch and column offset are **elements**, not bytes.
The tensor can be a borrowed GPU-visible host-registered buffer and must remain
alive through stream completion. `ColumnMap` is authoritative for the selected
mask. The ALL baseline has GDN pitch 16640, qkv at 0, z at 10240,
**a at 16384, b at 16432**; QSA pitch 14080, qgate at 0, k at 12288, v at 12800,
index at 13312. Subsets compact these offsets and pad only the fused total.
Qgate is 24 repetitions of
`q[256] | gate[256]`, not a global q/gate split. Index is `query[512] | key[128]`.

All functions below take `gpu` first and return `HipResult<()>`. `p`, `conv`,
`params`, `step`, and `gate` are the corresponding incumbent parameter structs;
all non-projection arguments retain their meaning.

| Variant signature after `gpu` | Projection/output convention | Implementation |
| --- | --- | --- |
| `gated_delta_conv_batched_arena(p, qkv)` | `p.input` ignored; output and history unchanged, including aliasing | hipcc compile-time twin |
| `gated_delta_conv_params_batched_arena(conv, params, qkv, a, b)` | Input/a/b fields ignored; fused convolution and parameter blocks | hipcc compile-time twin |
| `gated_delta_conv_params_qknorm_batched_arena(conv, params, step, keep_qk_conv, qkv, a, b)` | Same fusion; QN/KN go to the incumbent shared scratch | hipcc compile-time twin |
| `gated_delta_conv_params_batched_mixed(conv, params, qkv?, a?, b?)` | Each absent source reads native scratch; present sources use immutable BF16 views | hipcc opt-in twin |
| `gated_delta_conv_params_qknorm_batched_mixed(conv, params, step, keep_qk_conv, qkv?, a?, b?)` | Same mixed sources and QN/KN scratch contract | hipcc opt-in twin |
| `gated_delta_params_batched_arena(p, a, b)` | `p.a/p.b` ignored; gate/beta remain contiguous F32 | hipcc |
| `gated_delta_gate_arena(p, z)` | `p.z` ignored; single-row, read-only recurrent input | hipcc |
| `gated_delta_gate_batched_arena(p, z)` | `p.z` ignored; rounded recurrent input updated as before | hipcc |
| `gated_delta_gate_batched_rotate_arena(p, rotated, z)` | Non-chunk prefill: gate plus all FWHT rotation outputs; same prerotation marker | hipcc |
| `gated_delta_step_gate_wmma_arena(step, gate, z)` | `gate.z` ignored; normal state/inline-Q8 selector retained | hipcc compile-time twin |
| `gated_delta_step_gate_wmma_qknormed_arena(step, gate, z)` | Same, with convolution-provided QN/KN | hipcc compile-time twin |
| `gated_delta_step_gate_q8_wmma_arena(step, gate, z)` | Explicit inline-Q8 arm, independent of inline selector | hipcc compile-time twin |
| `indexed_attention_norm_rope_batch_arena(p, src)` | Immutable BF16 input; `p.values` is out-of-place F32 output, at `p.row_stride/p.head_stride` | hipcc |
| `indexed_attention_cache_append_batch_arena(p, key: Option<Bf16Arena>, value)` | `Some(key)` reads BF16; `None` reads normalized contiguous F32 `p.key`; `p.value` ignored | PM for F32 caches; hipcc for Q8/FP8 |
| `indexed_attention_index_key_append_batch_arena(p, index)` | `p.index_rows` is F32 output: query preserved, key tail written; raw keys may be BF16 or F32 | PM |

The out-of-place norm copies untouched gates and index-key tails as it writes
the normalized heads. Thus **attention reads its transformed F32 qgate output**
and **index selection reads its transformed F32 index query**; neither reads a
raw trunk projection, so neither needs an additional arena kernel. Cache append
after K normalization uses `key=None` plus the original BF16 V view. Index append
writes only the output tail and raw-key cache, never the source arena.

`npu_trunk_copy_projection_arena(gpu, src, dst, rows, cols, dst_pitch,
dst_col0, position_start)` is the PM row-copy primitive used for cache stores
and the lab's exact old-path materialization. Its three certified gfx1151
exports cover BF16→F32 (`bits << 16`), BF16→BF16 and F32→F32. The committed
`npu_trunk_arena_pm_gfx1151.{s,co,hxaco,proof.json,m7.json}` carry M7 byte-exact
lifts, empty obligations and zero ambiguous delays; the embedded image is
SHA256-pinned. Hipcc exceptions have one entry per kernel in the PeaceMaker
gap ledger.

### Halo consumer oracle (maintainer-run)

```sh
HIPFIRE_QWEN4_GDN_PREFILL=wy HIPFIRE_QWEN4_F16_WMMA=1 \
nice -n 19 cargo run --release -p rdna-compute --features lab \
  --example npu_trunk_consumers_check -j 12
```

Run with GPUs visible on Halo, not in the CPU build environment. For 8192 and
4647 rows at each family's pitch, this compares all output/state bytes:
convolution history (including in-place aliasing), normalized QN/KN scratch,
recurrent state (including inline-Q8 codes/scales and diagnostic F32 capture),
non-chunk fused-gate FWHT rotation outputs,
QSA gates/index tails, complete K/V caches and raw index keys with untouched
prefix/suffix guards. It verifies the materializer independently against CPU
bit widening, prints old timings including materialization versus direct reads,
checks immutable arena bytes, and exits 1 on any mismatch. GPU parity/timings
are maintainer-run silicon evidence, not implied by CPU build/test success.

The mixed oracle additionally uses a compact qkv-only pitch of 10240 with
native F32 z/a/b carrying nonzero low mantissa bits. It compares fused
convolution/parameters and both Q/K-normalization keep modes, including
history and QN/KN scratch; native F32 projections retain the incumbent core's
BF16 rounding at parameter consumption, not at projection storage.

Replay follows the incumbent position contracts: cache/raw-key stores and
RoPE positions are bound to replay position; compact F32 scratch-tail writes
and the lab copy primitive's concrete destination offset remain position-independent.
