# G80 implementation and small-M expert-wave feasibility

> **Lifecycle: historical paper/simulator feasibility record (2026-10-02/03); not a current
> admission.** The G80 int8 and (simulator-only) int32 epilogues here are lab designs. The
> int32-partials contract with a GPU fold was dropped: the 0.4.2 Flash-Next gfx1151 NPU trunk
> uses the G80 direct BF16 program (`Epilogue::Bf16`, input projections only, calibration
> schema 4, BF16-direct only, no I32 contract; out-projection program exists but is not
> wired). See [`npu-trunk.md`](npu-trunk.md) for the current route and schema. All ticks,
> TOPS and byte figures below are dated simulator or paper-model values (or estimates
> explicitly labelled as such), not current performance. `report.md:N`, `J:N` and `F:N`
> citations are line pointers into the NPU workspace's copies at the time of writing and
> are not stable against this repository; treat them as section pointers.

G80 is an explicit opt-in Rust-emitted int8 GEMM for XDNA2: `gemm_g80::design_g80(m,n,k,epi,ctl)` and
`npu-gemm --variant G80`. It uses 32 cores, 128x80 core tiles and a 256x1280 array wave. N must equal 1280;
K must be a positive multiple of 64 no greater than 2560; positive M is padded to 256-row waves (at most 256 waves).
The default epilogue is the int8 floor/shift/saturate SRS (shift 0..31); an opt-in simulator-only int32 epilogue
(`Epilogue::I32`, `npu-gemm --variant G80 --epi i32`, no `--shift`) is described in "I32 epilogue (simulator-only)".
Both use the fixed core layout 0.
Fast, slow control, serial, nocompute and repeat probes use the existing pair-lock protocol.
Nocompute/repeat are timing probes, not ordinary GEMM results.

The full TXN resets channels, rewrites every G80 memtile descriptor (including iteration state), initializes locks,
and starts the cores last. Lean submits are valid only immediately after a completed full/lean submit of the same
design; the finite B tasks are requeued each time. The first 16 KiB of each memtile, memtile BDs 10/11 and shim
BDs 4/5/6 are reserved for the persistent ring. Both C-token SYNCs retire before the next expert can overwrite B.
`npu-experts --design G80 --proj gate_up --m 160 --experts 3 --insts-only` exercises the offline grouped launcher;
its one-wave expert interface accepts M 1..256. Default `--design V9` behavior is unchanged; G80 down is rejected.

No hardware was accessed. Functional simulator ticks below are not silicon cycles or performance measurements.
The retained paper analysis uses source arithmetic and offline Rust/Bun experiments, including the standalone
emitter (`cand.rs`) and its captured `out_full.txt`, both workspace-local and not in this repository.
The +3.5% normalized bank-model regression remains a risk, not a veto; the measured-baseline extrapolation still
fails the 50-TOPS budget (§7). Smaller TM32/TM64 waves remain paper-only candidates, not implemented replacements.

Source abbreviations: **C** = `crates/peacemaker-npu/src/kernels/gemm_core.rs`; **G** = `crates/peacemaker-npu/src/kernels/gemm_array.rs`;
**D** = `crates/peacemaker-npu/src/dma.rs`; **R** = `crates/peacemaker-npu/src/route.rs`; **J** =
`docs/npu/levers-joint.md`; **F** = `docs/npu/levers-fable.md`.
All byte counts use signed int8 inputs and int8 SRS output, not the GPU's int4 storage or BF16 epilogue.
MiB/KiB mean powers of two. Geometry comes from J:121-145 and F:58-63, and expert dimensions from `report.md:79-80`.

## Gate 1 — core capacity, reverse SRS and bank pressure: capacity passes; bank risk

Per core, `128*80*4 = 40960` B C, `2*(128/2)*64 = 8192` B A, and `2*64*80 = 10240` B B
sum to **59392 B = 58 KiB**, below `0x10000` (C:77-82,108-110,1102-1109; J:141).
A separate `128*80 = 10240` B Cout would make 68 KiB and fail.

Two explicit disjoint maps (half-open offsets in own memory):

| region | simple map used by emitted proof | minimum shared-pressure map |
|---|---|---|
| C i32 | `[0x0000,0xa000)` | `[0x3d00,0xdd00)` |
| A ping | `[0xa000,0xb000)` | `[0xdd00,0xed00)` |
| A pong | `[0xb000,0xc000)` | `[0xed00,0xfd00)` |
| B ping | `[0xc000,0xd400)` | `[0x0000,0x1400)` |
| B pong | `[0xd400,0xe800)` | `[0x1400,0x2800)` |
| Cout, aliased | `[0x7800,0xa000)` | `[0xb500,0xdd00)` |

For reverse block `j=159..0`, load the full source `[256*j,256*(j+1))`, then store to
`[30720+64*j,30720+64*(j+1))`, relative to C. The destination minus unread-prefix end is
`30720-192*j >= 192` for every actual j. Thus no unread lower block is overwritten; overlap with
the current block is safe because it was loaded in full. DMA subsequently walks the Cout buffer forwards:
the bytes for logical block j remain at `Cout+64*j`, so reverse execution does **not** reverse packed-C order
(J:141; C:106,145-148,785-819). The emitted-address proof appears under gate 5.
A standalone Rust byte-buffer proof also checked all 160 blocks and 10240 lanes at shifts 0/1/7/15/31,
with i32 extrema, negative floor cases and block-varying nonsaturated values; ascending packed output was exact.
This is an arithmetic/retirement proof, not simulator execution of `VST.SRS`.

The physical-bank model is C:37-70,1116-1223: eight single-port banks, four 16-KiB double-bank windows,
`bank = 2*(off>>14)+((off>>4)&1)`. A 512-bit access reserves both banks at nominal memory stage 5.
In the simple map C uses windows 0,1 and the lower half of 2; A uses the upper half of 2; B uses 3.
In the minimum-pressure map C spans all four windows, A shares window 3 with C, and B shares window 0 with C.
Hence bank isolation of the baseline layout is lost even though address ranges are disjoint.

The adapted Rust trace uses 40 groups, `group%5`, B stride 640, C row stride 2560, unchanged MAC/A-load times
(C:392-471), E/O peer views (C:709-714), and the exact metrics in C:1197-1218.
First it reproduced every baseline table cell at C:58-63 for all four layouts and both slots.
Then it exhaustively searched 64-B-aligned disjoint active `(A,B,C)` triples, with
C bases `0..24576`, A bases `0..61440`, B bases `0..60416`, all stepped by 64 (alignment rule C:1108).
Compute-only pressure depends on the active slot only; allowing just one A/B slot is a relaxation of the full map.
Subset pruning is sound because per-cycle maxima and pair counts cannot decrease when requests are added.
The relaxed shared<=1000 search evaluated **13291336 triples**; all excluded triples have subset pressure above
1000, and every C base passed the independent C-load/store bound. Thus the observed minimum 968 is global
for this fixed schedule and contiguous layout, not merely a coarse-grid near miss.

| accumulate-chunk metric | layout 0 / 1024 VMACs | simple map / 1280 | minimum map / 1280 |
|---|---:|---:|---:|
| peer A pairs | 1024 | 1280 | 1280 |
| B/C-load pairs | 0 | 0 | 28 |
| C load/store pairs | 944 | 1128 | 1056 |
| split-port pressure slots | 512 | 640 | 647 |
| shared-port pressure slots | 748 | 979 | 968 |

The nonworse shared bound is `748*1280/1024 = 935`; even the minimum has `968/935-1 = 3.5294%` more
normalized model pressure. A separate exhaustive full-map 1-KiB-grid search covered 55440 placements and found
no nonworse map. Single-PADDS C-skip sensitivity also had minimum 968.
These are **model reservation counts, not elapsed cycles**. The minimum is not claimed over different schedules
or noncontiguous C layouts. The revised project decision supersedes F:63's strict model veto: lever 2 removed
1024 peer-A collisions and 512 split-port slots yet the measured V9 slope went 1323→1345
(`report.md:514-532`). No model slot is credited as a cycle saving or added as a measured stall.
Evidence: `levers-paper/g80_bank_model_all.txt:1-124,155-175` and `g80_bank_model_skip1_min.txt:74-95`.

## Gate 2 — next-wave Cout retirement: passes with early C_EMPTY; exposed cost remains

The existing int8 program acquires C_EMPTY only at conversion (C:849-895), which is unsafe when Cout aliases C.
The proposed program instead acquires it **before chunk 0** of every wave, retaining ownership through
conversion, and releases C_FULL only after SRS stores retire (C:791-845). The core MM2S acquires C_FULL,
reads the complete 10240-B Cout, then releases C_EMPTY (C:923-924).
This creates the required happens-before edge from the last old-Cout DMA read to every next-wave C store.
The initial empty credit of 1 permits wave 0 (C:99,910). The emitted candidate has this early acquire.

Under the source's assumed 4 B/core-lane-cycle, the exposed drain is `10240/4 = 2560` cycles/wave,
or `2560/40 = 64` cycles/chunk at kc=40 (J:142; C:66).
This is an estimate, not a hard physical lane limit; J:74 and `report.md:508-512` warn against that inference.
There are 15 inter-wave waits plus the final output drain for 16 M-waves. A conservative budget below reserves
16 such drains and separately charges the final memtile pair drain, deliberately not assuming perfect overlap.

## Gate 3 — memtile, descriptor, lock and channel budget: paper passes

At kc=40, two resident B streams each occupy `40*64*80 = 204800` B = 200 KiB.
Double C is `2*4*128*80 = 81920` B = 80 KiB; reserve 16 KiB for A staging
(J:144; G:75-79,99-102 for the mechanisms). Operand/staging reservation totals **496 KiB**.
The remaining 16 KiB is reserved at the start for ring scratch; the complete reserved map fits exactly 512 KiB at kc40.
DMA own-view addresses add `0x80000`; lock ids add 64 (D:149-150).

| region | exact half-open offsets |
|---|---|
| ring reservation | `[0,0x4000)`; includes scratch `0x1000` and `0x3000` |
| C slot 0 rows 0/1/2/3 | `[0x4000,0x6800)`, `[0x6800,0x9000)`, `[0x9000,0xb800)`, `[0xb800,0xe000)` |
| C slot 1 rows 0/1/2/3 | `[0xe000,0x10800)`, `[0x10800,0x13000)`, `[0x13000,0x15800)`, `[0x15800,0x18000)` |
| A ping reservation / live bytes | `[0x18000,0x1a000)` / first `0x1000` |
| A pong reservation / live bytes | `[0x1a000,0x1c000)` / first `0x1000` |
| B parity 0 | `[0x1c000,0x4e000)` |
| B parity 1 | `[0x4e000,0x80000)` |

Only columns 0..3 inject A, but the conservative table reserves A space in every column.
Core (c,r) owns M tile `r/2`, N tile `2*c+r%2`; columns 0/1/2/3 inject the four A halves into rows
0/1/2/3 respectively. A half payload is 4096 B per chunk (G:43-49; changed mapping J:125).

Every descriptor id below is unique globally within one memtile. Even channels use ids<24; odd use ids>=24
(D:230-239). The plan uses **26/48 BDs**, six S2MM and five MM2S channels; ids<48 and channels<6
(D:151-168,189-194,231-238).

| function | channel | BD ids / next chain | locks |
|---|---|---|---|
| C producers row 0 | S2MM0 | `0↔1` (slots 0/1) | empty/full `8/9`, `16/17` |
| C producers row 1 | S2MM1 | `24↔25` | `10/11`, `18/19` |
| C producers row 2 | S2MM2 | `2↔3` | `12/13`, `20/21` |
| C producers row 3 | S2MM3 | `26↔27` | `14/15`, `22/23` |
| alternating B fill | S2MM4 | `4→5→end`, task repeat 40 | BD4 releases ready0 id4; BD5 releases ready1 id6 |
| A fill, injecting columns only | S2MM5 | `28↔29` | A ping `0/1`, pong `2/3` |
| C drain rows 0/1 | MM2S0 | `6→7→8→9→6`: slot0 rows0/1, slot1 rows0/1 | corresponding row FULL acquire, EMPTY release |
| C drain rows 2/3 | MM2S3 | `32→33→34→35→32`: slot0 rows2/3, slot1 rows2/3 | corresponding row FULL/EMPTY |
| B parity0 guarded / immutable replay | MM2S1 | finite `30`, then finite `31` | BD30 acquires ready0 id4; BD31 no locks |
| B parity1 guarded / immutable replay | MM2S5 | finite `42`, then finite `43` | BD42 acquires ready1 id6; BD43 no locks |
| A replay, injecting columns only | MM2S2 | `14↔15` | FULL acquire, EMPTY release |

C producer BDs acquire EMPTY -1 before writing and release FULL +1 afterwards; drain BDs do the inverse after
their last read. Each C/A EMPTY is initially 1, every FULL initially 0. B ready ids 4 and 6 begin at 0;
ids 5/7 are unused. Thus **22 active locks** in the reserved namespace 0..23, below 64 (D:261-267).
Core A/B/C/peer locks remain C:94-105,910: the two cores still have independent FULL and peer acknowledgement.

Host B is column-major, then chunk-major `[parity0 5120 B, parity1 5120 B]`.
Fill BD4/5 each has length **1280 words**, base B0/B1, iteration `step=1280, wrap=40, current=0`.
BD4 links to BD5; BD5 has no next. One finite task repeats that chain 40 times
(task-chain contract D:219-242; per-BD iteration advances G:103-108).
No dimension wrap of 1280 is needed: every transfer is plain 1-D; the legal 6-bit iteration wrap is 40
(D:96-98,151-168). This avoids filling all parity0 chunks before the first parity1 chunk.

Each guarded reader is a finite 1280-word BD with the same iteration, repeated 40 times.
Each subsequent whole-stream BD is length `40*1280 = 51200` words, no lock, repeated `MW-1=15` times;
omit that task for MW=1. The whole-stream task follows the guarded task on the **same channel**.
Consequently reader s starts chunk j only after fills s,0..j completed, and unguarded traversal starts only
after that reader consumed all 40 ready credits. B is immutable afterwards. Credits are separate:
`ready_s = completed_fills_s - started_guarded_reads_s`; no shared `FULL+=2` pool exists
(J:145,193; G:176-201). Maximum ready count 40 is below the lock maximum 63.
Each B MM2S queues 2 tasks, B S2MM 1 task, within the four-task queue; repeat 40/15 is within 256
(G:195-199; D:236). Iteration reset and lock reinitialisation on every submit are mandatory
(G:170-174,184-185,203-218), including both fill BDs and both guarded readers.

An abstract asymmetric-consumer readiness model exercised kc=1/2/40, waves=1/2/3/16 and three seeds
(36 cases): no read of an unfilled chunk, no premature immutable replay, final ready credits zero.
This is **not** a full DMA simulator gate. The specific two-BD finite-chain repeat/iteration interaction
and context reuse must still be exercised in `peacemaker-npu-sim` before any real design is accepted.

Shim contracts at gate_up: B MM2S0 in each of 8 columns reads 409600 B; A MM2S1 in columns 0..3 reads
`16*40*4096 = 2621440` B each, and is absent in columns 4..7. Each C S2MM0/1 reads two row tiles per wave:
`16*2*10240 = 327680` B/channel. Totals are exactly **A 10 MiB, B 3.125 MiB, C 5 MiB**
(J:125-135; G:80-83 for shim/task mechanisms). Token waits cover both C channels across all columns.
No AxQOS or shim BD word-5 attribute change is proposed.

## Gate 4 — routes and switch ports: paper graph passes

Use V8's shim and C-return mechanisms, without its neighbour-column B crossing
(G:59-74); port capacities are R:22-27. All listed mappings are slave→master:

* Shim: MM2S0→North0 (interleaved B), MM2S1→North1 in columns 0..3 (A);
  North0→S2MM0 and North1→S2MM1 for C. The established shim NOC mux is R:112-137.
* Memtile: South0→Dma4 (B fill), South1→Dma5 in columns 0..3 (A fill);
  Dma1→North4 (parity0 B), Dma5→North3 (parity1 B), Dma2→North5 in columns 0..3 (A).
  North(r)→Dma(r) for all four C producers, Dma0→South0 and Dma3→South1 for the two C drains.
* B parity0 travels on lane4: every row below row2 passes South4→North4;
  rows0/2 also receive South4→Dma1. Parity1 travels on lane3:
  rows0/1/2 pass South3→North3, rows1/3 receive South3→Dma1.
  Each core's DMA1 master has exactly one driver, determined by its parity.
* Column r injects A on North5 up to core row r, with South5→North5 below that row.
  At row r, columns x<r select East0, x=r South5, x>r West0; that slave drives Dma0,
  West0 for x≤r,x>0, and East0 for x≥r,x<7. Thus the same half reaches all columns.
  This is one horizontal A lane per row, not V8's per-column-parity A selection.
* C from core row r uses Dma0→South(r); lower rows forward North(s)→South(s) for s>r.
  North3 **slave** for descending C and North3 **master** for ascending B are distinct ports;
  their lane numbers are preserved rather than remapped.

A Bun graph check built all **326 circuits**, verified every port against R:22-27,
found **326 unique master drivers**, followed four A streams to all eight columns,
both B streams to exactly rows0/2 or1/3, and all C producers back to their memtile row.
Multicast is one slave driving several masters, which is legal (R:56-63).
This proves topology and capacity only, not backpressure timing.


## Geometry under test

| item | baseline V9 pair (128x64) | TN80 (128x80) |
|---|---|---|
| tile | M128 x N64, 16x8 blocks of 8x8 | M128 x N80, 16x10 blocks |
| groups / chunk (2 mb x 2 nb each, 32 cycles) | 32 (`wavefront` `0..32`, `gemm_core.rs:416`) | 40 = 8 M-pairs x 5 N-pairs, `group % 5` |
| B k-stride / C mb-row | 512 / 2048 B | 640 / 2560 B |
| wavefront bundles | 1059 (`vec![..; 1059]`, `gemm_core.rs:414`) | 1315 = 40·32 + 35 |
| VMAC (VMUL on k=0 of chunk 0) / VLDB per chunk | 1024 / 1024 | 1280 / 1280 — exactly one VLDB (`x11`) per VMAC, unchanged (`gemm_core.rs:421,438-446`) |
| VLDA x (A) per chunk | 512 | 640 |
| data map (own tile) | A 4 KiB halves, B 4 KiB slots, C 32 KiB | C `0x0000..0xa000` (40 KiB), A `[0xa000,0xb000]`, B `[0xc000,0xd400]` (5 KiB slots), Cout `0x7800..0xa000` (10 KiB, inside C), top `0xe800` < 64 KiB |

Pair (E/O) addressing is mandatory for TN80, not an option: A-reset uses modifier `-512` only (the `group%5==4` A step is
absent, `gemm_core.rs:453-458`). Without E/O pair addressing the A cursor would also need `+512`, giving 9 distinct modifier
values (7 B steps + `-512` + `+512`) for 8 `m` registers (3-bit `mod` field, `PADDx_pstm_nrm`).

### Derived modifier table (8/8 used, no spare)
B steps from the time-sorted B offsets `k·640 + (group%5)·128 + (dm%2)·64` (formula `gemm_core.rs:421`, generalised):
`{-3072, -2560, -2432, 640, 3072, 3200, 3712}` (7 values; `640` is also the default step of the last load, as `512` is
in the baseline `map_or(512, ..)`, line 441). Plus A reset `-512`. C skip = modifier `3072` shared with a B step, followed
by `PADDS imm -512` (net +2560 = one 10-block mb row; same construction as baseline `2560` + imm `-512` = 2048,
`gemm_core.rs:462`; imm field is signed 4-bit x64, range -512..+448, `gen.rs` `PADDS_pstm_nrm_imm` `c10s_step64`).
Table emitted by the candidate: `m0..m7 = [-3072, -2560, -2432, -512, 640, 3072, 3200, 3712]` = the 8 modifiers of the task
(A-reset idx 3, C-skip idx 5). Baseline table is `MODIFIERS` (`gemm_core.rs:397`).

## Gate 5a — core program size ≤ 16384 B: **PASS (executed)**

Candidate = `program_pair(kc, tiles, role, Epilogue::Int8, ..)` adapted to TN80, finished through `isa::Program::finish`.

| program | bytes | `isa::rules::check` | ≤ 16384 |
|---|---:|---|---|
| baseline Lower/Upper Fast I32 (rlib `program_pair`) | 14224 | ok | yes |
| baseline Lower/Upper Fast Int8 (rlib `program_pair`) | 14896 | ok | yes |
| baseline re-emitted by my parametrised copy (np=4, legacy plan) | 14224 / 14896 | ok | **byte-identical to rlib for Lower, Upper x I32, Int8** |
| TN80 naive (original-style peel 170 + body 160 x6 + tail 185 = 515 bundles/chunk) | 17968 | ok | **no** (+1584 B) |
| **TN80 tight (prefix 37 + body 160 x7 + suffix 158 = 355 bundles/chunk)** | **12848** | **ok** | **yes, slack 3536 B** |

Size is independent of `kc`/`tiles` (`program_pair(1,1)` and `(40,256)` both 12848 B, they differ only by MOVXM immediates).

Per-section sizes of the tight TN80 program (bytes; from emitter program counters; lower role, Int8 epilogue):

| section | bytes | note |
|---|---:|---|
| prologue: 5 scalar MOVXM, 8 modifier MOVXM, 3 `MOVX cr`, `MOVXM s0`, `ACQ C_EMPTY`+3 NOP, `MOVXM r5` | 112 | |
| `pair_acquire` (ping/pong, 4 locks each, 4 MOVXM pointers) | 160 | |
| chunk 0 (overwrite): 2 MOVXM p4/p5 + MOVXM r12 | 12 | |
| chunk 0 prefix, cycles `[0,37)`, plus alignment | 612 | 37 bundles x 16 B + 20 B |
| chunk 0 loop body, cycles `[37,197)` (executed 7x) | 2560 | 160 bundles; `ADD r12` at body 146, `JNZ r12` at body 154, delay slots body 155..159 |
| chunk 0 suffix, cycles `[1157,1315)` | 2528 | 158 bundles |
| `pair_release` | 128 | |
| `pair_acquire` (accumulate loop head) | 176 | |
| chunk N (accumulate): 4 MOVXM p4..p7 + r12 | 24 | |
| chunk N prefix / body / suffix | 600 / 2560 / 2528 | prefix 592 B + 8 B alignment |
| `pair_release` | 128 | |
| epilogue: 3 MOVXM + pad, 16 peeled cycles, 16-cycle JNZ body x39, 8-bundle drain (+6 NOP) | 684 | 160 blocks = 640 loads |
| `rel C_FULL`, `ADD r3`, `JNZ r3`+5 delay, `DONE`, final 16 B pad | 30 + 6 | |
| **total** | **12848** | |

Static-loop proof (what makes the 355-bundle chunk legal), all asserted/executed in the candidate:

* The 1315-cycle schedule is 160-periodic: for every start `s` in `37..=159` the 7 windows `[s+160i, s+160(i+1))`, i=0..6, are
  byte-identical (both overwrite and accumulate wavefronts). `s=37` is the earliest (matches the assignment's "periodic from
  cycle 37"), giving prefix 37 + 7·160 + suffix 158. **8 windows is impossible** (no `s` has 8 equal windows): the last window
  would lack the next period's prologue loads (up to 12 cycles early) and 37 + 8·160 = 1317 > 1315.
* Peel contents (cycles `[0,37)`) are the prologue loads and the missing "period −1" spill (C-skip `PADDS` of the previous
  period land up to 25 cycles into the next one: `store_end+1 = g·32+row·16+41`, `gemm_core.rs:461-462`), so the first valid
  window start is 10 + 25 + 2 = 37.
* Difference from the original `chunk` (`gemm_core.rs:449`: pointer-step loop `for group in 0..31`, last group skipped):
  the candidate emits the pointer ops of the **last group too** (`0..40`) so period 7 equals the others. They run after the
  final use of each pointer (A: after the last A load; C: after the last store of that pointer), so they cannot change any
  address of this chunk; pointers are re-`MOVXM`ed at the next chunk. The one C-skip pair of group 39 / row 1
  (`39·32+57 = 1305` → index 1315) falls past the end of the schedule and is dropped (asserted: only the final group may drop).
  No cycles are added: 1315 bundles, same as the unrolled count.
* Loop mechanics identical to baseline (`gemm_core.rs:491-513`): no hardware ZOL (`ls/le/lc` never written,
  `Rule::ZeroOverheadLoop`), `ADD r12,-1` in a free ALU slot, `JNZ r12` in a free LNG slot at body index `160-1-5 = 154`
  so its five delay slots are body bundles 155..159 (VLIW compute, no lock/branch/DONE: `Rule::DelaySlot`), loop head 16-byte
  aligned by NOP padding in `Assembler::bind`/`chunk` (`Rule::BranchTarget`). All 123 starts `s=37..159` were checked as legal
  (bundle format exists for the JNZ bundle and ALU is free); `s=37` is used. Counter `r12 = 7`.
* `isa::rules` passed on the real `Program::finish` (all rules: branch targets, no ZOL register writes, lock spacing ≥ 4
  bundles — `C_EMPTY` acquire precedes `A_FULL` by 4 bundles, delay slots).
* `isa::sched::check_decoded_program` on the TN80 image: 1032 static bundles, **0 hazards**, 1824 timing observations
  (baseline same call: 1160 bundles, 0 hazards, 2190 observations).

Analytic cross-check of 12848: `112+160+12+612+2560+2528+128+176+24+600+2560+2528+128+684+36 = 12848`; lower bound for any
two-chunk emission (355·16·2 + control) is 11360 + control, upper bound without loop compression 2·1315·16 = 42080 ≫ 16384 —
so only the loop form passes, and the original-style 515-bundle form (17968 B) does not.

## Gate 5b — address-sequence proof of the emitted program: **PASS (executed)**

A tracer executes the *emitted, compressed* program (MOVXM/ADD/XOR/JNZ/JZ, five delay slots, pointer `PADD*`,
post-modify of every VLDA/VLDB/VST) with kc=2 (chunk 0 + one accumulate chunk, ping then pong), records every memory
access, and compares each pointer's full address sequence with the semantic address of `A(mb,k)`, `B(k,nb)`, `C(mb,nb,q)`:

* VLDA x p0/p1 (E/O halves): 640/640 accesses exact, both roles (Lower p1 via North view, Upper p0 via South view).
* VLDB p2/p3: 1280/1280 exact (time-sorted, dm0/dm2 interleaved), both B slots (`0xc000`, `0xd400`).
* VST bm p4/p5: 640/640 exact; VLDA bm (accumulate) p6/p7: 320/320 exact.
* Same tracer on the baseline 128x64 emitter: exact (control).
* Epilogue: VLDA bm p6 640/640, VST.SRS.4x p4 160/160 exact, descending blocks 159→0 and quarters 3,2,1,0.

Not covered: numeric GEMM correctness (no simulator run on the new geometry), lane order of dm vs bm quarters (already an
open simulator assumption, `gemm_core.rs:24-26`), lock-protocol liveness, DMA BDs.

### Reverse epilogue with in-place Cout (C0 / Cout `0x7800`)
`COUT_ADDR=0x7800` lies inside the i32 C (`0x0000..0xa000`): Cout block `b` (64 B) lands in C block `t = 120 + ⌊b/4⌋`.
Processing blocks 159→0 makes `t ≥ b` for every `b ≤ 159` (`t > b` for b<159; for `b=159`, `t=159` and the 64 B store hits
quarter 3 of its own block, which is loaded first because quarters are loaded 3,2,1,0). Executed check over all 160 stores x 640
loads: 160 overlapping (store, load) pairs, **none** with the store issued at/before the overlapping load, minimum margin
store−load = 10 issue cycles (≥ 7 required: `EPI_STORE_DELAY`, `gemm_core.rs:789`). VLDA bm/VST.SRS use post-modify imm raw field 15 = `-64 B` (signed 4-bit x 64, `c10s_step64` in `gen.rs`; the executed tracer
confirms the decoded step is exactly -64 per access, so no pointer-fix instructions are needed between blocks; bm quarter
destination stays `4*dm+q`) in lockstep, ptr starts `p6 = c + 256·159 + 192`, `p4 = cout + 64·159`.
`C_EMPTY` therefore must be acquired **before the first chunk of each wave** (the VMUL chunk 0 overwrites C, which the Cout DMA of
the previous wave is still reading); in the candidate `ACQ C_EMPTY` is at wave start (before `mov r5`, `program_pair`), `REL
C_FULL` after the epilogue drain, the initial lock vector `initial_locks()[8]=1` still supplies the first wave.

## Gate 5c — issue-cycle estimate (locks never blocking; not elapsed silicon cycles)

Interpreter validated on the baseline: reproduces `PAIR_CHUNK_CYCLES[0] = [1131, 1123]` (`gemm_core.rs:702`) and
`PAIR_EPILOGUE_CYCLES[0] = 530` (`:705`) exactly.

| | baseline 128x64 | TN80 128x80 | ratio |
|---|---:|---:|---:|
| wavefront bundles | 1059 | 1315 | 1.242 |
| later chunk, ping / pong | 1131 / 1123 | **1387 / 1379** | 1.226 / 1.228 |
| epilogue (Int8) | 530 | 663 (= 160·4+7 + setup/pad) | 1.251 |
| wave (tile) kc=40 | 45630 | 56003 | 1.227 |
| wave kc=20 / kc=10 | 23090 / 11820 | 28343 / 14513 | 1.227 / 1.228 |
| VMACs/chunk / issue duty at mean later-chunk cycles | 1024 / 90.86 % (=1024/1127) | 1280 / 92.55 % (=1280/1383) | |
| issue cycles per output element (kc=40) | 5.570 | 5.469 | −1.8 % |

So per-chunk fixed control is unchanged (1387−1315 = 72 ping, 64 pong, same as baseline), and the issue-bound gain over the
baseline is only ≈1.8 % per output element (control amortisation). Assigned budget: measured stall-inflated compute slope
1323–1337 cycles/chunk scaled x1.25 → **1654–1671 cycles/chunk**; this document does not claim that the cycle gate passes —
it only shows the size gate is not the binding one. Not modelled: DMA, bank arbitration (gate-1 +3.5 % risk), phase drift.
One TN80-specific serialisation to include in the cycle budget (model estimate from the `gemm_core.rs:66` 4 B/cycle/channel
assumption, unmeasured): the in-place Cout makes chunk 0 of wave n+1 wait for the 10240 B Cout DMA of wave n (≈2560 cycles
per wave, ≈4.6 % of the 55 640 kc=40 chunk issue cycles), whereas the baseline Int8 core only waits at the epilogue.

## Legality limits (what bounds the design)

| limit | value | status |
|---|---|---|
| program memory | ≤ 16384 B | 12848 B, slack 3536 B (tight); naive 17968 B fails |
| `m` modifier registers | 8 | 8/8 used (7 B steps + `-512`; `3072` shared by B and C skip); E/O pair addressing required |
| `PADDS` imm | signed 4 bit x 64 (-512..448) | C skip uses `-512` (at the lower edge) |
| loop mechanics | no ZOL; JNZ at body 154, delay slots 155..159 | 123/123 starts legal; needs free ALU + LNG slot |
| branch targets | 16 B aligned (NOP padded) | asserted in `isa::rules` |
| lock spacing | ≥ 4 bundles | passes (`C_EMPTY` early lock keeps 4-bundle gap to `A_FULL`) |
| data memory | 64 KiB | top of use `0xe800` |
| Cout in place in C | reverse blocks, `t ≥ b` | executed check, margin ≥ 10 cycles |
| `MAX_KC` | 160 (`gemm_core.rs:76`) | unchanged: the bound abs(c) <= 128*128*10240 < 2^31 is per output element and independent of N |

## Not proven / open

* Numeric correctness on the simulator for TN80 (not run: would need harness/geometry changes, outside paper scope).
* Silicon behaviour (no hardware).
* DMA channel execution, same-context reset and numeric whole-array correctness remain untested.
* Gate 7 below evaluates the cycle budget; the size/rules proof alone does not authorize Phase 2.
* Source citations used: `crates/peacemaker-npu/src/kernels/gemm_core.rs` — Assembler `242-333`, `wavefront 413-471`, `chunk 483-517`,
  `epilogue_wavefront/epilogue 801-847`, `program_pair 855-896`, `PAIR_CHUNK_CYCLES 702`, `PAIR_EPILOGUE_CYCLES 705`,
  `MODIFIERS 397`; `crates/peacemaker-npu/src/isa.rs:224-234` (`Program::try_finish/finish`); `crates/peacemaker-npu/src/isa/rules.rs` (rules table,
  `check`); `report.md:514-535,614-616` (lever 2 stall attribution).

## Gate 6 — two-stream prefix startup: paper passes; latency is estimated

A whole-B barrier costs `3276800 / 55.4e9 = 59.148 µs`, or 106466 cycles at 1.8 GHz:
11.02% of the `26843545600/50e12 = 536.871 µs` useful-operation budget (J:51,125-145;
`report.md:329-331`). Do not put such a barrier into the candidate.
Gate 3 instead publishes parity0 and parity1 independently after each 5120-B fill.
With the alternating host/BD layout, neither stream waits for the other stream's whole segment.
Both core rows of each parity receive one multicast replay; a single producer's ready credit is consumed
by its one MM2S stream, not independently stolen by the two receiving cores.

First prefixes across all columns total `8*2*5120 = 81920` B; first A halves add `4*4096 = 16384` B.
At the measured read plateau their aggregate service estimate is `98304/55.4e9 = 1.7744 µs`.
Adding one 5120-B core transfer at the source-model 4 B/cycle gives a conservative scheduling allowance
of `1.7744 + 1280/1800 = 2.4856 µs = 4474 cycles`.
This is **an estimate**, not a demonstrated startup upper bound: arbitration, packet latency and backpressure
are not modelled, and transfer stages can overlap. It explains the removal of the whole-B barrier but does
not assert an achieved latency. The readiness/immutability proof is independent of these rates.

## Gate 7 — 50-TOPS cycle budget: fails the requested measured-slope extrapolation

Useful work is `2*4096*1280*2560 = 26843545600` ops; at 50 TOPS and 1.8 GHz the budget is
**966367.6416 cycles**, across `16*40=640` per-core chunk steps, hence
**1509.94944 cycles/chunk** (J:51,125-135).
The measured repeat-slope range is **1323–1337 cycles per 1024 VMACs**
(`report.md:527-528,542-543,613-616`). Scaling by `1280/1024 = 1.25` gives
**1653.75–1671.25 cycles for the larger compute region**, already over budget by
143.80–161.30 cycles even with zero epilogue, drain or startup cost.
This scaling is the requested **estimate**, not a measured G80 timing law.

The slope is a repeated compute-region measurement, not the full DMA/control loop.
Existing later compute issue is 1067 cycles (C:11-13), while the mean whole-chunk issue is
`(1131+1123)/2 = 1127` (C:699-705): reserve **60 control cycles/chunk**.
The emitted wider program retains that gap: mean 1383, compute issue `1067+256 = 1323`.

| budget contribution | cycles/chunk | basis |
|---|---:|---|
| measured-slope-scaled compute | 1653.75–1671.25 | measured baseline, unmeasured extrapolation |
| input/phase/loop control | 60 | baseline issue difference, retained in emitted candidate |
| reverse SRS | 16.575 | executed issue count `663/40` |
| exposed Cout drain allowance | 64 | model `2560/40`, gate 2 |
| initial prefixes / first transfer allowance | 6.991 | estimate `4474/640`, gate 6 |
| final memtile joined-C drain | 8 | two 10240-B tiles/channel, `20480/4/640` |
| **array-path estimate** | **1809.32–1826.82** | conservative sum, no bank-slot timing conversion |

The estimated array path is **643.31–649.53 µs**. The measured old per-submit intercept is approximately
**215 µs** (`report.md:499-502`), separate from chunk work: adding it yields
**858.31–864.53 µs**, or another 604.6875 equivalent cycles/chunk.
Lean/grouped launch may amortize that intercept; it cannot be silently removed from wall-time estimates.
These sums are not rigorous bounds: the drain allowance intentionally does not assume useful overlap,
and the baseline slope need not scale linearly to a different geometry.

For contrast, the *issue-only* emitted count plus the same drain/startup allowances is
`16*56003 + 40960 + 5120 + 4474 = 946602` cycles, about 525.89 µs, nominally within the
50-TOPS budget. This optimistic result excludes exactly the baseline's measured issue-to-silicon gap.
It is not evidence of 50 TOPS. The bank model is also demonstrably not a valid elapsed-cycle predictor
(`report.md:518,527-532`); neither its +3.5% pressure nor hypothetical collision removal repairs this estimate.

DDR alone would allow a shorter path: reads `(10+3.125) MiB/55.4 GB/s = 248.42 µs` and
5 MiB writes at 54.2 GB/s = 96.73 µs. These are directional necessary conditions only.
The equal-volume mixed test does not establish a universal combined-bandwidth ceiling (J:22,47,74).
Verdict: **not supported as a 50-TOPS candidate by the requested measured-cycle budget**;
capacity/ISA feasibility is not killed, and no silicon verdict is made.

## Shipped-shape applicability, utilisation and bytes/op

Shapes and row distribution are now required by the 2026-10-02 23:44 UTC shipped-shape entry of `report.md` (routed-row-spread directive).
Expert gate_up is N1280/K2560, down N2560/K640 (`report.md:79-80`).
The average row count is `8192*10/512 = 160` (J:242 and `report.md:573`).
The following arithmetic must not be read as an implemented constructor.

For the fixed G80 wave let `P=ceil(M/256)*256`, `Q=ceil(N/1280)*1280`, `NW=Q/1280`.
A hypothetical N-wave-outer extension reads `A=P*K*NW`, `B=Q*K`, writes `C=P*Q`.
For V9 replace wave dimensions by 512/512 (G:4-6,43-56,87-96).
Bytes/op means `(A+B+C)/(2*M*N*K)` including padded traffic; row utilisation is M/P.
The hypothetical down extension is shown only for comparison: the assignment remains gate_up-only,
and J:147 recommends TN64 down. It needs new N-wave/residency scheduling, not the fixed N1280 constructor.

| expert M | row utilisation G80 / V9 | GU bytes/op G80 / V9 | down bytes/op hypothetical G80 / V9 |
|---:|---|---|---|
| 64 | 25% / 12.5% | 0.01015625 / 0.020625 | 0.0125 / 0.021875 |
| 128 | 50% / 25% | 0.005078125 / 0.0103125 | 0.00625 / 0.0109375 |
| 160 | 62.5% / 31.25% | 0.0040625 / 0.00825 | 0.005 / 0.00875 |
| 256 | 100% / 50% | 0.0025390625 / 0.00515625 | 0.003125 / 0.00546875 |
| 512 | 100% / 100% | 0.0015625 / 0.002578125 | 0.0021484375 / 0.002734375 |

All GU rows satisfy the 400-KiB-per-column B capacity gate and N1280 geometry, but no numeric G80
simulator run exists. All down rows have N divisible by1280 and K640 capacity headroom, but are outside
the gate_up-only design and have no schedule/cycle acceptance. V9 GU additionally wastes N padding
`1280/1536 = 83.333%`; multiply this by its row utilisation to obtain actual arithmetic utilisation.
For example, M160 V9's total is 26.042%, versus G80's 62.5%; those are MAC ceilings, not throughput.

Dense dimensions are the Halo roofline trace `tops.md:26-32` (also tabulated in `dense27b.md` §1);
M8192 is the widened one-chunk acceptance in `report.md:574`.
The requested z+β+α N6400 is retained below: the cited GPU source row is N6240, so 6400 is the requested
test/padded width, not a claim that the source kernel's logical width changed.
Even **one** resident 1280-column N-wave requires B `2*K*80` bytes/column.
That is already too large for every dense row; retaining several N-waves would only increase it.

| dense shape M×N×K | N multiple of1280? | B/N-wave KiB per column | row utilisation G80 / V9 | compulsory int8 DDR bytes/op | resident G80 applies? |
|---|---|---:|---|---:|---|
| qkv 8192×10240×5120 | yes | 800 | 100% / 100% | 0.000207520 | no, B alone >512 KiB |
| z+β+α 8192×6400×5120 | yes | 800 | 100% / 100% | 0.000236816 | no |
| full-attn q 8192×12288×5120 | no | 800 | 100% / 100% | 0.000199382 | no; N padding also needed |
| full-attn k/v 8192×1024×5120 | no | 800 | 100% / 100% | 0.000646973 | no; N padding also needed |
| o 8192×5120×6144 | yes | 960 | 100% / 100% | 0.000240072 | no |
| gate_up 8192×34816×5120 | no | 800 | 100% / 100% | 0.000173053 | no; N padding also needed |
| down 8192×5120×17408 | yes | 2720 | 100% / 100% | 0.000187414 | no |

The compulsory byte ratio is `(M*K+K*N+M*N)/(2*M*N*K) = 1/(2*N)+1/(2*M)+1/(2*K)`;
it is an all-once, no-padding **lower bound**, not achievable DDR traffic for a nonexistent dense G80 kernel.
Dense B streaming or K-splitting would be a different design; intermediate int32 lifetime and rounding
would need a new proof. None is silently substituted here. Full-model GPU epilogue parity is also outside this int8 contract.

## Additional paper gate — 64-/128-row waves for small routed experts

The follow-up addition asks whether reducing the 512-row expert floor is more useful than G80 at M4096.
Evaluate 32 cores with the same `(c,r)→(r/2,2c+r%2)` mapping but TM32 or TM64, TN80.
Wave rows are `2*TM = 64/128`; N width remains1280. GU K2560 still needs 40 chunks.
This is **not** obtained by giving the existing TM128 program fewer `tiles`: its 32-group wavefront,
fixed EPI_BLOCKS128, fixed C row strides and BD lengths are compile-time geometry
(C:77-82,397,413-471,483-517,784,855-926). Both candidates require a new parameterised core emission.
The pair lock protocol, E/O neighbour views, route topology and descriptor mechanisms are reusable.

### Core maps, banks and memtile capacity

Half-open offsets; all regions are 64-B aligned. Cout is **separate**, removing G80's next-wave alias wait.

| core region | TM32/TN80 | TM64/TN80 |
|---|---|---|
| A ping | `[0,0x400)` (1 KiB) | `[0,0x800)` (2 KiB) |
| A pong | `[0x2000,0x2400)` | `[0x2000,0x2800)` |
| Cout | `[0x2800,0x3200)` (2.5 KiB) | `[0x2800,0x3c00)` (5 KiB) |
| B ping | `[0x4000,0x5400)` (5 KiB) | same |
| B pong | `[0x6000,0x7400)` | same |
| C i32 | `[0x8000,0xa800)` (10 KiB) | `[0x8000,0xd000)` (20 KiB) |
| live total | 24.5 KiB | 39 KiB |

Sizes are `2*(TM/2)*64 + 2*64*80 + TM*80*4 + TM*80`
(C:77-82,108-110 adapted to TM). A/Cout use double bank0, B uses1, C uses2 (TM32) or2/3 (TM64);
compute A/B/C double banks are disjoint. Cout DMA may still contend with A in bank0, and peer-A remains.
No smaller-geometry bank trace or silicon contention measurement was run.

| memtile region | TM32 (64-row wave) | TM64 (128-row wave) |
|---|---|---|
| C slot0, row r=0..3 | `[r*0xa00,(r+1)*0xa00)` | `[r*0x1400,(r+1)*0x1400)` |
| C slot1, row r | `[0x2800+r*0xa00,0x2800+(r+1)*0xa00)` | `[0x5000+r*0x1400,0x5000+(r+1)*0x1400)` |
| A reservation / ping,pong bases | `[0x5000,0x6000)` / `0x5000,0x5800` | `[0xa000,0xc000)` / `0xa000,0xb000` |
| B parity0 | `[0x6000,0x38000)` | `[0xc000,0x3e000)` |
| B parity1 | `[0x38000,0x6a000)` | `[0x3e000,0x70000)` |
| total / free | 424 KiB / 88 KiB | 448 KiB / 64 KiB |

Per-column B remains400 KiB. C double buffers shrink to20/40 KiB; A reserves4/8 KiB.
Both fit512 KiB. The resulting capacity ceilings are
`kc <= floor((512-24)/10)=48` (TM32) and `floor((512-48)/10)=46` (TM64), so kc40 fits.
The four A-half injections remain columns0..3; payloads become1024/2048 B per chunk.
Gate3's **26-BD,22-active-lock** allocation and channels apply unchanged, with shorter A/C lengths:
core A/B/C lengths are respectively256/1280/640 words (TM32) or512/1280/1280 words (TM64).
Memtile C producers/drains shorten similarly, and A ring slots use the listed bases.
The same alternating B fill, two independent ready locks and two B replay channels are necessary.
Routes are identical to gate4; shrinking payloads does not change master ownership or port counts.
Thus capacity/BD/lock/channel/route **paper feasibility passes**, but smaller core code and DMA execution
have not been emitted or simulator-verified.

### Utilisation, DDR bytes and estimated expert time

Let `W=ceil(M/(2*TM))`, `P=2*TM*W`. GU reads A=P*2560, B=3276800 B once, writes C=P*1280.
The B-only read floor is **59.148 µs** at55.4 GB/s (not57 µs at this exact measured rate);
at M160 this alone limits useful int8 throughput to **17.728 TOPS**, irrespective of MAC utilisation
(`report.md:79-80,329-331`; J:242).
The directional necessary DDR time is
`max((A+B)/55.4e9, C/54.2e9)`, not `(A+B+C)/55.4e9`: the latter would assume a mixed envelope not measured.

Compute-region estimates use `(1323..1337)*(VMACs/1024)`, plus `4*(TM*80/64)+18` epilogue cycles/wave
(baseline shift-independent epilogue structure C:798-845; these smaller epilogues are **estimates**).
VMACs/chunk are `TM*80*64/512 = 320/640`.
Holding 43 compute-setup/tail cycles (35 wavefront tail plus8 setup) and60 outer control cycles instead of
scaling them gives `(slope-43)*(VMACs/1024)+43+60`: approximately503–508 cycles/chunk (TM32)
or903–912 (TM64), still below the modelled B-feed service time. The proportional estimates in the table
omit this fixed control allowance and are deliberately optimistic; neither is a measured new schedule.

B replay remains5120 B/chunk. The old 4-B/cycle model gives1280cycles/chunk and
**28.444 µs per40-chunk wave**, regardless of TM; a4.8-B/cycle sensitivity gives23.704 µs.
The measured lower bound is **at least**4.8 B/cycle in the no-compute TN64 context
(`report.md:508-512`), so4 B/cycle is a pessimistic service model, not a physical hard limit or a justified veto.
After the guarded first M-wave, later M-waves replay all B again from the memtile.

| candidate / M | W | row utilisation | A / C KiB (B=3200 KiB in every row) | DDR directional time µs | compute+epi estimate µs | B replay service @4 / @4.8 µs | serial-first-pass pipeline estimate @4 µs |
|---|---:|---:|---|---:|---|---|---:|
| 64-row /64 | 1 | 100% | 160 /80 | 62.11 | 9.3–9.4 | 28.44 /23.70 | 63.6 |
| 64-row /128 | 2 | 100% | 320 /160 | 65.06 | 18.6–18.8 | 56.89 /47.41 | 92.1 |
| 64-row /160 | 3 | 83.33% | 480 /240 | 68.02 | 27.9–28.2 | 85.33 /71.11 | 120.5 |
| 64-row /256 | 4 | 100% | 640 /320 | 70.98 | 37.1–37.5 | 113.78 /94.81 | 149.0 |
| 128-row /64 | 1 | 50% | 320 /160 | 65.06 | 18.6–18.8 | 28.44 /23.70 | 68.1 |
| 128-row /128 | 1 | 100% | 320 /160 | 65.06 | 18.6–18.8 | 28.44 /23.70 | 68.1 |
| 128-row /160 | 2 | 62.5% | 640 /320 | 70.98 | 37.1–37.5 | 56.89 /47.41 | 96.5 |
| 128-row /256 | 2 | 100% | 640 /320 | 70.98 | 37.1–37.5 | 56.89 /47.41 | 96.5 |

The pipeline column is a scheduling **heuristic**, not an upper/lower bound:
`(B+A_first_wave)/55.4e9 + (W-1)*28.444µs + C_last_wave/54.2e9`.
It captures the unavoidable transition from a fill-paced first pass to later immutable replays,
but omits the final input-transfer/compute/epilogue tail, arbitration and mixed-traffic changes.
It also assumes resident B is not overwritten by another expert. Separate experts still fetch distinct B.
The 64-row candidate maximises row utilisation but pays more immutable B replays at M160/256;
the 128-row candidate is the better paper default for the stated average distribution.
At4.8 B/cycle the M160 pipeline sensitivities are approximately111.0µs and91.8µs respectively.
This is a priority judgment, not a proof that TM32 cannot ever win at M64 or at a faster replay rate.

**Grouped V9 comparison:** for all M≤512, V9 pads M512,N1536, runs3 N-waves, and reads
`A=512*2560*3 = 3932160` B, `B=1536*2560 = 3932160` B, writes786432 B
(G:43-56,87-96). Its read-only necessary time is **141.96µs**, write14.51µs;
scaled compute with3 epilogues is89.1–90.0µs, so compute alone is not its expert-time estimate.
Observed V9 M512 was361.9µs/submit (`report.md:585`); subtracting the approximate old215µs intercept
gives146.9µs, consistent with that read floor. The assignment's rough110µs floor would read A once,
but V9 reads it three times; it is not the V9 byte contract.

For E experts sharing a lean launch, add **(40..70)/E µs per expert** to both candidates and V9.
The40–70µs interval is the assignment's new planning input, **not measured here**, and grouping does not erase B DDR traffic.
At E4 this is10–17.5µs: the TM64/M160 pipeline estimate becomes106.5–114.0µs versus approximately
156.9–164.4µs for grouped V9 using its measured-array146.9µs estimate.
These are scheduling estimates, not benchmark improvements; a fixed group E4 is an example, not the launch ABI.
At M64, TM32 saves only about4.5µs of first-pass traffic versus TM64 while requiring another core geometry.

### Small-M down, retaining TN64

Down remains N2560,K640 (not N640,K1280), as required by `report.md:79-80` and J:147.
With the same vertical-pair mapping but TN64, a reduced-M wave spans1024 N columns;
there are3 N-waves, padded N3072. This is a changed geometry, not stock V9.
Hold A for all3 N-waves, as V10 does, and retain the complete B segment for all M-waves
(G:128-165). The following maps prove capacity; numeric/route execution and latency remain untested.

| down region | TM32/TN64 | TM64/TN64 |
|---|---|---|
| core A0/A1 | `[0,0x400)`, `[0x2000,0x2400)` | `[0,0x800)`, `[0x2000,0x2800)` |
| core B0/B1 | `[0x4000,0x5000)`, `[0x6000,0x7000)` | same |
| core C / Cout | `[0x8000,0xa000)`, `[0x2800,0x3000)` | `[0x8000,0xc000)`, `[0x2800,0x3800)` |
| core live total | 20 KiB | 32 KiB |
| memtile C, two slots × four rows | `[0,0x4000)` | `[0,0x8000)` |
| memtile whole-A slots0/1 | `[0x4000,0x6800)`, `[0x6800,0x9000)` | `[0x8000,0xd000)`, `[0xd000,0x12000)` |
| memtile B parity0/1, each3×10 chunks | `[0x9000,0x27000)`, `[0x27000,0x45000)` | `[0x12000,0x30000)`, `[0x30000,0x4e000)` |
| memtile total | 276 KiB | 312 KiB |

C slot/row offset is `slot*(4*TM*64)+row*TM*64`. Core A-half and C-output BD lengths shrink to
256/512 and512/1024 words; B is1024words.
B fill/guard iterations are `step=1024, wrap=30` (3 N-waves×10 chunks) with separate ready credits,
then whole-stream replay length30720 words. All are within D:96-98,134-168,236.
Use gate3's B/C descriptor/channel/lock plan, but replace the two A replay BDs by even-bank
`14→15→16→14`: each starts at A slot0 with iteration stride one whole-A slot and wrap2,
first acquires A_FULL, last returns A_EMPTY, all three advance in lockstep.
Fill BDs28/29 use shared ordered A_EMPTY/FULL ids0/1, initial2/0, as G:152-164;
the A slot is not released until all3 N-waves consumed it. This makes **27 BDs** with unchanged
six S2MM/five MM2S channels, and fewer than24 active lock ids. Core peer locks remain unchanged.
The same gate4 topology applies, with TN64 B lengths and a repeated N-wave host/output order.

Padded B is `3072*640 = 1966080` B (1.875MiB), versus useful B1638400B (1.5625MiB);
its read floor is35.49µs versus29.57µs. Thus even these small-M candidates pay N-padding.
Each N-wave replays `10*4096/4/1800 = 5.689µs` in the old service model.

| down candidate /M | row utilisation | total core waves | A /C KiB, B1920KiB | DDR directional estimate µs | B replay @4 µs | pipeline heuristic µs |
|---|---:|---:|---|---:|---:|---:|
| 64-row /64 | 100% | 3 | 40 /192 | 36.23 | 17.07 | 39.9 |
| 64-row /128 | 100% | 6 | 80 /384 | 36.97 | 34.13 | 56.9 |
| 64-row /160 | 83.33% | 9 | 120 /576 | 37.71 | 51.20 | 74.0 |
| 64-row /256 | 100% | 12 | 160 /768 | 38.45 | 68.27 | 91.1 |
| 128-row /64 | 50% | 3 | 80 /384 | 36.97 | 17.07 | 44.2 |
| 128-row /128 | 100% | 3 | 80 /384 | 36.97 | 17.07 | 44.2 |
| 128-row /160 | 62.5% | 6 | 160 /768 | 38.45 | 34.13 | 61.3 |
| 128-row /256 | 100% | 6 | 160 /768 | 38.45 | 34.13 | 61.3 |

Here the pipeline heuristic is first-M-wave `(B+A_first)/55.4e9`, then remaining M-waves'
three N-wave replays, then one final-M-wave C write at54.2GB/s.
Per-wave compute-region estimates are about1.92–1.94µs (TM32) or3.83–3.87µs (TM64), plus
epilogues/control not included in those two numbers; the DDR/feed paths dominate.
Stock V9 small-M down reads3,276,800B and writes1,310,720B (G:87-96); directional time59.15/24.18µs.
Its measured M160 time was280.5–299.2µs (`report.md:583`), or approximately65.5–84.2µs
after the old215µs intercept. Therefore the TM64/TN64 pipeline estimate61.3µs leaves a modest,
unproven margin, not a guaranteed shipped win. Add the same lean-TXN/E allowance to both.

Small-M verdict: **128-row GU paper capacity/route feasible and worth prioritising for simulator proof**;
64-row GU is a possible M64 specialization but not the default at M160/256.
TN64 128-row down is also paper-capacity feasible; its N-padding and fixed overhead make performance risk
more material. No new numeric GEMM, ABI, grouped-expert refill protocol or smaller-core rule gate is proven.
In particular, overlapping the next expert's B fill with old expert replays would require independent
retirement/EMPTY proofs; grouping a launch by itself does not authorize overwriting the400-KiB resident segment.

## I32 epilogue (simulator-only; historical, superseded by the BF16-direct trunk)

> The int32-C / GPU-fold trunk contract this section was a candidate for was dropped
> before 0.4.2 (calibration schema 4 is BF16-direct only). `Epilogue::I32` remains in
> the emitter as a lab epilogue only. The "accepted design", gate and B/kop figures
> below are historical model values, not a product contract or an admission.

`gemm_g80::design_g80(m,n,k,Epilogue::I32,ctl)` and `npu-gemm M 1280 K --variant G80 --epi i32` (`--shift` rejected,
V9/V10 still int8-only) store the exact int32 C instead of SRS int8 tiles. This is a simulator-level design: no
hardware was accessed and no silicon validation or silicon rate is claimed. The int8 default is unchanged.

Design contract (the emitted source is authoritative where it differs):

- There is no SRS and no `Cout`. The core's C accumulator `[0, 0xa000)` is drained as two 20480 B stripes on BD4 then
  BD5 then BD4 again. The core acquires `C_FULL` only for the first stripe and returns `C_EMPTY` only for the second.
- The memtile holds a single slot per row at `0x4000 + r*0x5000`, ending at `0x18000`.
- Finite producer chains (S2MM per core row, two BDs, stripe 0 then stripe 1) use BD IDs `[0,1]`, `[24,25]`, `[2,3]`,
  `[26,27]` for rows 0..3 with task `repeat = mw`. Finite drains (MM2S0 rows 0/1, MM2S3 rows 2/3) use BDs `6..9` /
  `32..35`: row 0 stripe 0 and 1, then row 1 stripe 0 and 1, also `repeat = mw`.
- Locks are `8 + 2r` empty and `9 + 2r` full (range `8..15`); the bank split is even `< 24`, odd `>= 24`. Memtile BDs
  10/11 and shim BDs 4/5/6 are untouched. Because the finite C tasks end idle and balanced, a lean TXN needs no int8
  odd-wave C repair; it requeues the four producers and two drains (`repeat = mw`) on every submit.
- Host C layout (little-endian int32, 4 B per output): column, then stream (row pair), then wave, then row parity,
  then tile `(mb, nb)` of 8x8. `ArrayDesign::unpack_out` returns row-major `m*n` and drops M padding; B is read once
  per submit.

`npu-gemm` compares every element of `unpack_out` with `ArrayDesign::reference` (the plain int32 GEMM for `I32`) exactly
as for V8.

ESTIMATE (a model, never a measurement; the 4 B/cycle stream rate is an [INFERENCE], and no silicon guarantee is made):
`gemm_g80::estimate` has no epilogue body and no overlap credit for I32. One 20480 B slot per row serialises every
stripe hand-off, each phase `P = 20480 / 4 = 5120` cycles, and six serial phases (`30720` cycles per wave) are exposed
before the core regains `C_EMPTY`, plus one final `5120`-cycle drain phase at the end. The int8 path instead pays the
`663`-cycle epilogue plus the `2560`-cycle `Cout` drain (`3223` per wave) and overlaps the memtile-to-shim transfer.
The I32 model is therefore `27497` cycles per wave longer than int8. At K = 2560 that lowers modelled throughput by
roughly 25-27% versus int8; this cost is not hidden, and the six-phase count is an upper-end estimate of the drain.

Observed simulator functional ticks (Fast control; not silicon cycles, not a performance measurement):

| shape | waves | int8 ticks | I32 ticks | I32 - int8 per wave |
|---|---:|---:|---:|---:|
| 64x1280x2560 | 1 | 113393 | 140888 | 27495 |
| 512x1280x2560 | 2 | 171948 (85974/wave) | 226923 (113461.5/wave) | 27487.5 |
| 512x1280x64 | 2 | 18685 | 73154 | 27234.5 |

The per-wave deltas (27234.5..27495) agree with the modelled `27497` to within about 1%. K = 2560 fits the I32 design
with no fallback.

This simple contiguous stripe layout was the accepted design at the time of writing. The then-next hardware
promotion gate (never exercised; superseded with the I32 contract) was bit-exact output, at most 1.26 B/kop, and at least 20 TOPS in co-run. The earlier kill
criterion of a drain penalty above 10% is dropped for the bytes-first pipeline.

Declared shim accounting for the single-panel trunk shape M8192 N1280 K2560 (formula from the design's byte counts only;
not a physical DRAM measurement and not hardware evidence): reads `24248320` B (A `20971520` + B `3276800`, B once per
submit), writes `41943040` B (I32 C, 4 B per output), useful ops `2*8192*1280*2560 = 53687091200`. That is
`66191360 / 53687091.2 kop = 1.23291015625` B/kop (about 1.233), the figure the 1.26 B/kop gate above is measured against.

Exercised (CPU simulator only, no hardware): `ROCR_VISIBLE_DEVICES=-1 nice -n19 cargo test -p peacemaker-npu --release --test g80
--test golden_bytes --test lean --test ring -- --nocapture` passed: g80 84 tests (95.56 s), golden_bytes 1 (0.72 s),
lean 27 (122.04 s), ring 15 (13.99 s). `ROCR_VISIBLE_DEVICES=-1 nice -n19 cargo build -p npu-tools --release --bin npu-gemm`
passed (0.04 s). The all-bin build was blocked by the separately in-flight FP probe's missing module registration,
not by G80; no all-bin build pass is claimed.

## Verification and final disposition

The retained Phase1 offline proofs cover bank-model searches, reverse-SRS overlap/floor/saturation, legal routes,
independent B readiness credits and the tight emitted program. Production Fast programs are 12848 B; slow-control
programs are 13808 B. Both fit the 16 KiB core instruction memory and pass the real `Program::finish` rule gate.

An actual throwaway Rust simulator executable exercised G80 `200x1280x192`, then three lean submits with new
operands: all 256000 output elements were CPU-exact on each submit, and packed inputs were unchanged.
Each submit consumed 294912 DDR-read bytes and 327680 DDR-write bytes in 18681 functional ticks.
The image was 441056 B; full/lean TXNs were 49424 B/22576 B. The smoke source and executable were removed afterward.

Offline CLI smoke exercised G80 M64/M160 groups and the M256 ring group without opening a device:
E3 gate_up used 76176 B instructions, or 77412 B with the ring (two patch sites), with a 441056 B PDI.
Per expert, packed A/B/C were 655360 B/3276800 B/327680 B. Default and explicit V9 offline output matched exactly.
G80 down, M257, invalid design, wrong N and `--b-prefix` were rejected with exit 2; that smoke predates the I32 epilogue
(its then-rejected i32 request is superseded by the section "I32 epilogue (simulator-only)").

Compatibility gates passed: golden_bytes 1, levers3 10, lean 27, ring 11 and experts 14 tests.
`cargo test --release -p peacemaker-npu -- --skip vendor_decode_oracle_roundtrip` passed 105 tests
(the known missing-LLVM oracle was skipped); `cargo build --release -p npu-gemm` passed.

`cargo test --release -p peacemaker-npu-sim --test g80 -- --nocapture` passed all 51 tests in 44.54 s.
Every requested shape was CPU-exact with unchanged inputs and exact per-channel shim byte totals.
The suite also covers padded M200/M513, signed extrema with floor/shift/saturation, three changing seeds at odd/even
`kc*waves`, full/lean interleaves, forced ring-channel repair, and equal retained state snapshots versus fresh full runs.
Real `experts::grouped` E3 mixed-M launches and late-producer ring launches match their all-full/eager oracles.
Test-only reversed BD4/BD5 fill payloads plus reversed host B halves prove odd-ready-first as well as the default
even-ready-first order under actual DMA semantics, including kc40 and multiple M waves.
Negative runtime cases omit the finite fill BD4, guard BD30/42 or whole replay BD31/43 requeues:
each produces a bounded C-SYNC timeout. Omitting the forced col0 S2MM4 reset after a real grouped-ring poll also
produces a C-SYNC timeout, proving that the POLL self-loop cannot simply be left running.

| shape | Fast functional ticks | Slow functional ticks | DDR read / write B | full / lean TXN B |
|---|---:|---:|---:|---:|
| 256x1280x64 | 13565 | 13643 | 98304 / 327680 | 49424 / 22576 |
| 256x1280x128 | 16113 | 16199 | 196608 / 327680 | 49424 / 13392 |
| 160x1280x2560 | 113393 | 113479 | 3932160 / 327680 | 49424 / 13392 |
| 64x1280x2560 | 113393 | 113479 | 3932160 / 327680 | 49424 / 13392 |
| 512x1280x640 | 53658 | 54583 | 1146880 / 655360 | 49808 / 9168 |
| 4096x1280x2560 | 991718 | 1040389 | 13762560 / 5242880 | 49808 / 9168 |

Fast/Slow PDI sizes are 441056 B/471776 B. At the full4096 gate_up shape, each column reads 409600 B from B MM2S0;
columns0..3 additionally read 2621440 B from A MM2S1; each column writes 327680 B through each of C S2MM0/S2MM1.
Total packed A/B/C are 10 MiB/3.125 MiB/5 MiB. At K2560, the E3 mixed-M group is 76176 B with lean tails versus
148240 B all-full, and both execute 340179 functional ticks. These byte counts matter to launch interception;
functional ticks do not model the silicon firmware/TXN intercept, memory-bank contention or elapsed hardware time.

No silicon speedup is claimed. Dense resident G80 remains capacity-inapplicable at the larger K values;
the 50-TOPS extrapolation fails, and TM64 small-M is still a separate unimplemented paper candidate.
