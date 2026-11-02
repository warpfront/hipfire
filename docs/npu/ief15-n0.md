# IEF15 N0: NPU pair core for the mixed contract (dense Qwen3.8-27B)

> **Lifecycle: historical dark research record (2026-10-03); the contract it implements is
> not a shipped path.** The IEF15 mixed-contract co-op was never wired into any serving route
> (see [`fold-contract.md`](fold-contract.md): full IEF15 killed, mixed plan unimplemented).
> The kernel exists only as dark `peacemaker-npu` code and a lab `npu-gemm --fold-contract ief15`
> probe. It is unrelated to the 0.4.2 Flash-Next NPU trunk, which is BF16-direct only
> (calibration schema 4, input projections only, out-projection unwired; no I32/IEF15
> contract); see [`npu-trunk.md`](npu-trunk.md). Text written before §8 ("nothing below is a
> hardware result", "declared window") describes the state before the window; §8 holds the
> dated fixture-bound silicon results, which are not current performance or an admission.

Workspace branch `npu/ief15-n0` (not in this repository), 2026-10-03. Design under review when written; nothing in §§1–7 is a hardware result until the
declared silicon window (§7, results in §8). Sources: `docs/npu/fold-contract.md` §3/§8, CPU oracle `tools/npu/fold-model/src/lib.rs::{grid,
round_scaled, fixed}`, llvm-aie intrinsic headers `lib/clang/22/include/aie2p/*.h` of the oracle wheel (semantic source
for every new instruction), `crates/peacemaker-npu/src/isa/gen.rs` (encodings), `crates/peacemaker-npu/src/isa/sched.rs` (itineraries).

Code: `crates/peacemaker-npu/src/kernels/iu4_ief15.rs` (oracle, encoders, packing, array design), `iu4_ief15_core.rs` (core program),
`crates/peacemaker-npu/tests/iu4_ief15.rs`, `crates/peacemaker-npu/tests/ief15_ops.rs`, `npu-gemm --fold-contract ief15`.

## 1. Arithmetic contract (frozen, fold-contract §3)
Per output `(t, r)`: `I = Σ_e S[r,e]·D[t,e]·C[t,r,e]` exact (i64, |I| < 2^51 for E ≤ 136), `C[t,r,e] = Σ_{j<128} (u[r,128e+j]-8)·q[t,128e+j]`
∈ [−7168, 8192]; `Y = round_scaled(I, a_r + b_t)` (one RNE to binary32; integer 0 → +0; negative underflow → −0; overflow → ±inf).
S: u16 `[feature][K/128]` ≤ 32767, a: i16 `[feature]`; D: u16 `[K/128][token]` ≤ 32767, b: i16 `[token]`. Encoders use
`grid(x, 15)` per weight row (all epochs) / per token (all epochs). Weights: QT44 `iu4::Weights`; activations: A4 72 B blocks `iu4::Acts`.

## 2. Array geometry (V8 pair topology, switch circuits identical to V8 `pair_column_circuits`)
* Wave = 256 tokens × 256 features. `MW = ceil(M/256)`, `NW = ceil(N/256)`, `waves = MW·NW ≤ 256`, wave order `w = mw·NW + nw`.
* Core (column c, core row rho = 2q+h): tokens `mw·256 + (2q + c%2)·64 ..+64` (A block `blk = 2q + c%2`), features
  `nw·256 + (c/2)·64 + h·32 ..+32` (B block `c/2`, half `h`). The vertical pair (rows 2q, 2q+1) shares A block `blk`:
  lower core (h=0, `PairRole::Lower`) receives the **E** half (8-token blocks `mb` even), upper (h=1) the **O** half (`mb` odd).
  Each core reads the other half through the North/South neighbour view (V8 protocol `PEER_FULL/PEER_EMPTY`).
* Epoch e = one K128 block. E = K/128 (K multiple of 128, 1 ≤ E ≤ 136).
* Core-local row order = `mb` 0..8 (8-token blocks of the 64-token A block, token `blk·64 + mb·8 + i`); even `mb` come from
  the E half, odd from the O half (own or peer depending on the role). The core processes rows as E/O pairs.

### 2.1 A half chunk (one per core per epoch, 2176 B = 544 words)
* bytes `[0, 2048)`: int4 codes, `[mbl 0..4][kb 0..16]` blocks of 32 B; block = 8 tokens × 8 k, element (i, kk) index
  `x = i·8 + kk` in byte `x/2`, low nibble if x even. Token `blk·64 + (2·mbl + h)·8 + i`, k = `128e + 8kb + kk`.
  Value = the A4 two's-complement nibble, i.e. block row i = A4 `qs[kb·4 .. kb·4+4]` of that token/epoch copied verbatim.
* bytes `[2048, 2112)`: D (u16 LE) for the 32 half tokens in order (mbl, i).
* bytes `[2112, 2176)`: b (i16 LE) for the same 32 tokens (identical in every epoch).
* Padding tokens: all zero.

### 2.2 B half chunk (one per core per epoch, 2176 B)
* bytes `[0, 2048)`: int4, `[kb 0..16][nb 0..4]` blocks of 32 B; block = 8 k × 8 features, element (kk, j) index `x = kk·8 + j`,
  byte `x/2`, low nibble if x even. Feature `nw·256 + (c/2)·64 + h·32 + nb·8 + j`, k = `128e + 8kb + kk`. Nibble = `u ^ 8`
  (two's complement of `u − 8`).
* `[2048, 2112)`: S (u16) for the 32 features (order nb, j); `[2112, 2176)`: a (i16), same order. Padding features zero.

### 2.3 Host arguments (DDR-patched like V8)
* arg0 A: 8 injecting-column segments (V8: column c injects half `h = (c/2)%2` of A block `2·(c/4) + c%2`), each
  `(wave w, epoch e, 2176 B)`; A depends on mw only and is repeated for every nw.
* arg1 B: 8 column segments, column `c = 2j + h` = B block j half h, each `(wave, epoch, 2176 B)` (repeated for every mw).
* arg2 C: the V8 int8 layout byte-for-byte (`Geometry::pair_c_offset`, tile 8192 B): per column the S2MM0 stream
  (core rows 0, 1) then S2MM1 (rows 2, 3), `(wave, rho%2)` tiles. Tile = f32 LE, `[mb 0..8][nb 0..4]` blocks of 256 B,
  block element (i, j) at `(i·8 + j)·4` = Y[blk·64 + mb·8 + i][feature nb·8+j]. Padding outputs are written +0 and discarded.

## 3. Core data memory (own view offsets; core address = 0x70000 + off)
| region | offset | bytes |
|---|---|---|
| A slot 0 / 1 | 0x0000 / 0x0880 | 2176 each |
| B slot 0 / 1 | 0x1100 / 0x1980 | 2176 each |
| C16 scratch (int16 C of one epoch) | 0x2200 | 4096 |
| Dpat / Spat / ab copy / Bpat / Apat / consts | 0x3200.. 0x4000 | core-owned |
| I (i64 state, raw accumulator round trip) | 0x4000 | 16384 |
| Cout (f32 tile, drained by core MM2S0 BD4) | 0x8000 | 8192 |

Locks / BDs: exactly V8 pair numbering (`gemm_core::{A_EMPTY,A_FULL,B_EMPTY,B_FULL,C_EMPTY,C_FULL,PEER_FULL,PEER_EMPTY}`,
`A_BD`, `B_BD`, `C_BD`, channels, `initial_locks_pair`). Core BDs: S2MM0 A ring BD0 (slot 0) ↔ BD1 (slot 1), 544 words,
acquire A_EMPTY[s] −1 / release A_FULL[s] +1; S2MM1 B ring BD2/BD3 same with B locks; MM2S0 BD4 Cout 2048 words,
acquire C_FULL −1 / release C_EMPTY +1, self-cyclic. Constants exported by `iu4_ief15_core.rs`.

Memtile: the V8 memtile map, BD ids, locks and chains unchanged except lengths: A rings 544 words, B rings 544 words, C
2048 words (= V8 int8 COUT). Shim BDs: A `waves·E·2176` B per column, B `waves·E·2176`, C as V8 int8.
Lean TXN: the V8 ring parity plan with `J = E·waves`, `W = waves`.

## 4. New core instructions and the modelled semantics (simulator contract)
Lane conventions: x register = 64 B; 16-bit lane i = bytes 2i..2i+1 LE, 32-bit lane i = bytes 4i..4i+3. y_k = (x_{2k}, x_{2k+1}).
Accumulator dm_k = 256 B; acc32 lane i = bytes 4i..; acc64 lane i = bytes 8i..8i+7 LE; `cml_k` = bytes 0..127 of dm_k,
`cmh_k` = bytes 128..255; bm quarter q = bytes 64q..64q+63 (existing). Raw VLDA/VST bm round trips are layout-agnostic.
Masks (`eRS16`, r16..r31): bit i ↔ lane i.

| instruction | semantics |
|---|---|
| `VMUL/VMAC_vmul_cm_core_X_X`, `VADDMAC_vmac_cm2_add_reg_vmul_cm_core_X_X` with conf `amode=1,bmode=3,variant=2` (`aie2p_compute_control`) | 32 lanes acc64: `p_i = A(s1)[i]·B(s2)[i]`, A/B 16-bit lanes, signed iff conf bit 9 (s1) / bit 8 (s2). VMUL: `dst = p` (zero_acc irrelevant). VMAC: `dst = sh(acc1) + p`. VADDMAC (dst = acc1 field): `dst = sh(acc1) + acc2 + p`. `sh(x) = zero_acc(bit0) ? 0 : (shift16(bit10) ? x<<16 : x)`, wrapping i64. sub bits (11..13) set → Unsupported. Existing int8 matrix mode (conf `amode=0,bmode=1,variant=0`) unchanged. |
| `VNEG` conf amode=1 | `dst = −acc1` per acc64 lane (amode=0: acc32 lanes). |
| `VSRS_4x_mv_x_srs_dm_srsSign1` | crSRSMode=1: 32 acc64 lanes → 32 int16 lanes; crSRSMode=0: 64 acc32 → 64 int8. |
| `VSRS_2x_mv_x_srs_cm_srsSign1` | crSRSMode=1: 16 acc64 (cm half) → 16 int32; crSRSMode=0: 32 acc32 → 32 int16. |
| SRS lane rule | `v >> s` (s = s-register `su`, 0..63) with crRnd rounding exactly as `srs_lane` (floor 0, … conv_even 12), then crSat to the output width: 0 wrap (low bits), 1 saturate signed, 3 symmetric. |
| `VUPS_4x_mv_ups_x2d_upsSign{0,1}` | crUPSMode=1: 32 16-bit lanes → 32 acc64 lanes, `(signed? sext : zext)(v) << s`. |
| `VUPS_2x_mv_ups_x2c_upsSign{0,1}` | crUPSMode=1: 16 32-bit lanes → 16 acc64 (cm). crUPSMode=0: 32 16-bit → 32 acc32. |
| `VLDB_UNPACK_dmx_ldb_unpack_pstm_nrm_imm_unpackSign1` | load 64 B at ptr (64-B aligned), crUnpackSize=0: nibble n (byte n/2, low nibble for even n) sign-extended → byte n of y_dst (bytes 0..63 = x_{2k}, 64..127 = x_{2k+1}); ptr += imm. |
| `VUNPACK_mv_unpack_x_unpackSign1` | same unpack from x src to y dst (LDB slot). |
| `VLDB_dmx_ldb_x_pstm_nrm_imm`, `VST_dmx_sts_x_pstm_nrm_imm`, `VLDA_dmx_lda_x_pstm_nrm_imm` | 64 B x load/store, post-increment imm (`c10s_step64`). |
| `VSHUFFLE_vec_shuffle_x` (mode in r reg) | modes `T16_2x32_lo/hi` (18/19), `T32_2x16_lo/hi` (16/17), `T64_2x8_lo/hi` (14/15), `T128_2x4_lo/hi` (12/13): element width W ∈ {16,32,64,128} bits, interleave `s1[0], s2[0], s1[1], s2[1], …`, lo = first 64 B, hi = second 64 B. |
| `VLT/VGE_{16,32}_vaddSign{0,1}`, `VEQZ_{16,32}` | mask bit i = predicate on lane i (sign0 unsigned, sign1 signed). |
| `VSEL_{16,32}` | `d[i] = sel bit i ? s2[i] : s1[i]`. |
| `VADD/VSUB_{16,32}`, `VBAND`, `VBOR`, `VMAX_LT/VMIN_GE_{16,32}_vaddSign1`, `VBCST_{16,32}` | wrapping lane add/sub; bitwise; signed max/min (cmp r16 = lane-wise s1 < s2 / s1 ≥ s2); broadcast scalar low bits. |
| `MOVX_mvx_cr_imm` crUPSMode / crUnpackSize | new control registers (reset 0). |
| scalar `AND`, `OR`, `LSHL`, `ADD_alu_r_rr`, `MOV_OR`, `EQZ`, `NEZ` | 32-bit scalar ALU. |

## 5. Core program per wave
1. Wave start: zero I (16 KiB of x stores).
2. Per epoch: pair acquire (V8 9-step protocol), build the D pattern (each D_t repeated over the 8 lanes of its block row,
   `T16/T32/T64` interleave shuffles) and the S pattern (8 features repeated 4x, `T128` interleaves), copy b/a.
3. Phase G: 8 core rows (mb) x 64 int8 VMACs (8x8x8, conf 0x308) into four of the five dm registers (rotating), A via
   `VLDA x` + `VUNPACK`, B via `VLDB.UNPACK`; each finished block → `VSRS_2x` (acc32 → int16, shift 0, wrap; C fits int16)
   → C16 scratch.
4. Pair release (A/B slots free; the DMA refills while the fold runs).
5. Phase F, per 32-lane half block: `P = Dpat·Spat` (elem 16x16, unsigned), `Plo = wrap16(P)`, `Phi = P >> 16` (`VSRS_4x`,
   64-bit mode, floor), `T = Phi·C16`, `T = (T << 16) + I + Plo·C16` (VADDMAC shift16), `I := T` (raw bm round trip).
6. Wave end: C_EMPTY, final pack (§6) → Cout, C_FULL.

## 6. Final pack (one RNE per output, integer only)
Per 32 lanes, with `X = clamp(a_r + b_t, ±300)` (host rejects |a|, |b| > 16383 so the 16-bit sum cannot wrap; outside
±300 every result is already 0 / ±inf):
1. limbs `l_k = wrap16(floor(I / 2^16k))`, sign from `l3`, magnitude limbs `m_k` from `VNEG(I)` selected by sign.
2. `L = floor(log2 m)`: highest nonzero limb, then a 4-step shift/compare ladder (8, 4, 2, 1).
3. `sh = min(max(L − 23, −149 − X), 55)`; class `S ∈ {−9, 7, 23, 39, 55}` by `sh` range, `k = S − sh ∈ [0, 15]`, `p = 2^k`.
4. Horner over class-selected 16-bit limbs (normal `(m3,m2,m1,m0)`; `sh ≤ −9`: `(0,m0,0,0)`; `sh ≥ 40`: round-to-odd
   `(0,m3,m2,m1|sticky)`): `acc = m·p` exactly (< 2^63 for every class).
5. `q = SRS_conv_even(acc, 7 | 23 | 39)` (one rounding), `bits = q + (sh + X + 149) << 23` clamped to `0x7F800000`,
   `q == 0 → 0`, sign OR. Equals `round_scaled` including ties, subnormals, carry into the exponent, overflow and −0.

## 7. Estimates (ESTIMATE, not measured) and the silicon plan
* Core, simulator issue cycles (not hardware): 1000 cycles/epoch (G 549, two-epoch fold F2 ~270 per epoch, patterns 63,
  locks/control ~120) + ~13.6k per wave (zero I, X patterns, final pack 64 x 199) → issue-bound ESTIMATE 27.4 TOPS at
  down (E=136), 22.5 at gate_up (E=40), 1.8 GHz;

* DDR at the ~55 GB/s plateau: int4 stream 1.95 B/kop × 1.0625 (sidecars) + f32 out (0.11 at K=17408, 0.39 at K=5120)
  → down ~25 TOPS, gate_up ~22 TOPS cap.
* One declared window, in order: (1) single-core semantics probe of every §4 instruction on edge operands, compared bit for
  bit with the simulator; stop at the first mismatch. (2) Only if (1) passes: the IEF15 kernel, down first, exact on first
  and final submits, then rates and switch cost. (3) V9 health check.
* Step (1) is `crates/peacemaker-npu/src/kernels/iu4_ief15_probe.rs` (`CASES`: one row per instruction/operand set with its output byte range;
  straight-line single-core program, every dependent pair >= 8 bundles apart, 12.5 KB). Expected bytes come from the exact
  PDI + TXN in `peacemaker_npu_sim::config::Config` (`crates/peacemaker-npu/tests/ief15_probe.rs` also checks every case against an independent
  lane model). Window command: `npu-gemm --ief15-probe --dump <prefix>` (one fresh-context submit; per-case MATCH/MISMATCH
  table, exit 1 on any mismatch or a non-completed wait; `--expected-only` runs the simulator side only).

## 8. Window 1 results (2026-10-03, Halo test host, `tools/npu/npu-ief15-w1.sh` @ a2bf975, npu-gemm md5 ce658da6b8044adc390c7ef9f342e208)

* Probe: 437/437 MATCH (all families: VSRS 132, VMAC 44, VADDMAC 36, VUPS 30, VSHUFFLE/VGE/VLT/VMAX/VMIN 16 each, VMUL 15,
  VLDB 13, VUNPACK 10, VADD/VSUB/VEQZ 8, ADD/AND/OR/LSHL/VBCST/VSEL 6, EQZ/NEZ 4, VBAND/VBOR 3, VNEG 2, VST 1);
  `probe.got` md5 = `probe.expected` md5 = 6357278936927f8b8a57c68f4bde43f1.
* All kernel runs byte-exact vs the CPU IEF15 oracle on first and final submits.
* down 8192x1024x17408: `--iters 3` 21.48 TOPS busy / 3.63 wall; `--loop 2` 144 submits 22.43 busy / 20.08 wall.
* gate_up 8192x2048x5120: `--iters 3` 17.08 TOPS busy / 2.33 wall (below the 19 TOPS bar).
* Switch (down <-> gate_up): alternating adds +1020.6 us/submit on down (14333.2 vs 13312.6) and +1198.0 us on gate_up
  (11201.9 vs 10003.9).
* V9 health: PASS, 954 submits, 25.76 TOPS busy, exact. No wedge sign; kernel log empty.
