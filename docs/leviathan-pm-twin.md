# Leviathan PeaceMaker twin (Route A)

`lane_leviathan_verify` (`kernels/src/lane_leviathan.hip`) has committed PeaceMaker twins for
`gfx1200` and `gfx1201`, covering modes 0 (full row), 1 (select) and 2 (merged). The three modes
are `mode` kernarg branches of one symbol, so sealing the symbol seals all of them. The kernarg
ABI is unchanged: seven explicit args `region, lanes_off, rows_off, ptab, mode, cand_local,
cand_peer` at offsets 0, 8, 12, 16, 24, 32, 40. Dispatch is not changed by these artifacts.

## Row-batched draft top-8 (gfx1201)

`topk8_rows_mq6g256v2_pm_gfx1201.hxaco` seals both
`topk8_partial_f32_rows` and `topk8_rescore_mq6g256v2_k2560_rows` in one module.
The anchor is `topk8_rows_mq6g256v2_hipcc_gfx1201.co`; the manifest records the
exact concatenation of `topk8_select.hip`, `qwen4_gemv_mq6g256v2.hip` and
`topk8_rows.gfx1201.hip`, with no `HIPFIRE_TOPK8_RESCORE` define. Both explicit
kernarg layouts are 40 bytes; row strides are in floats (partial input and
re-score input) or ints (re-score output), and each partial slot is 8192 bytes.
`hip_twin_topk8_rows` checks source provenance, full device-ELF identity, both
entry ABIs and wrong-arch/wrong-symbol rejection using committed files only.
The HIP and PM wrapper hashes can differ: the complete device ELF is identical.

The committed module's complete gfx1201 device ELF is 25,480 bytes, SHA256
`037150d701462a89cafdb0173eddbb86f8d0dff7df4d0b5f27da135abfc04d72`
in both the HIP anchor and PM twin. Wrapper SHA256s are
`22c1e30c1cd2e2c6c7f677ebaae32dff94caa8885ba609ed663b862ddad35eeb`
(HIP) and `c0da81e80943325ff3ded561fcdb94137eebd46136dfcc73a090b592a4a0135a`
(PM); the HIP host triple has a trailing `-`, unlike PM's empty host entry.
This is an envelope spelling difference, not an ELF or kernel-byte mismatch.

Regenerate explicitly:

```sh
scripts/pm-leviathan-twin.py --family topk8-rows --arch gfx1201 \
  --build-wrapper /home/kaden/qcal/locks/build-slot.sh
```

The wrapper is a local build-serialization policy, not an artifact dependency;
substitute your cargo wrapper or omit it on another machine. No local toolchain
is needed to consume the committed twins. Dispatch stays on HIP and the existing
exact-gfx1201 default remains on. `HIPFIRE_QWEN4_DRAFT_TOPK8_ROWS=0` kills row
batching and restores the per-row launch pairs.

## What Route A proves

The twin is not a rewrite. `hip_twin` (`crates/peacemaker-isa/src/hip_twin.rs`) lifts the hipcc
device ELF, re-emits it with `emit::checked`, and requires the emitted bytes to equal the input
ELF byte for byte. The ELF is then wrapped by `native::bundle` and sha-pinned. This is a stronger
claim than a numerical or output comparison: the twin's machine code, descriptor and metadata
are the compiler's. Metadata kernel bodies pass checked lift/re-emission and static hazard
checks; out-of-line executable helper functions also pass typed instruction decode/re-encode,
with their extents, instruction counts and hashes recorded separately. Helper checks are not
a claim of interprocedural CFG or emulator certification. The full ELF equality preserves the
call relationships, helper bytes, descriptor and metadata.

## Artifacts (`kernels/`)

| File | Content |
|---|---|
| `lane_leviathan_hipcc_<arch>.co` | hipcc anchor (HIP offload bundle from `--genco`) |
| `lane_leviathan_pm_<arch>.hxaco` | sealed twin |
| `lane_leviathan_pm_<arch>.hxaco.sha256` | sha256 of the `.hxaco` |
| `lane_leviathan_pm_<arch>.twin.json` | manifest: source sha256, compiler provenance, flags, seal report (incl. ABI), math evidence |

Anchors are built with the runtime's `core_hipcc_flags`: `--genco --offload-arch=<arch> -O3
--no-offload-compress -fuse-cuid=none -mcode-object-version=6`.

## Regenerating

Explicit invocation only; tests never compile or run it.

```
scripts/pm-leviathan-twin.py --arch all [--rocm-path DIR] [--build-wrapper 'CMD ...'] [--seal-bin PATH]
```

* `ROCM_PATH` (or `--rocm-path`) locates `bin/hipcc`.
* Without `--seal-bin` the script runs `cargo run --release -p peacemaker-isa --features lab --example hip_twin`;
  `--build-wrapper` prefixes that cargo command (for example a build-slot script). Default: none.
* The manifest records `hipcc --version` with host-path lines removed, portable flags and relative
  source path. Regenerate whenever `lane_leviathan.hip` changes: the test
  `manifest_records_source_and_artifact_hashes` fails on a stale source hash.

## Math evidence

The manifest's `math_evidence` is produced by the script, not asserted by hand: it recompiles with
the same core flags plus `-save-temps=obj`, disassembles the pre-link device bitcode and records
the callees found in the `expf` and `powf` wrapper bodies (`_ZL4expff`, `_ZL4powfff`) with their
extracted text. Read the generated field for the actual callee of each function; do not assume
both reach `__ocml_*`. Math paths must be obtained before sealing: if the bitcode,
`llvm-dis` or either wrapper is unavailable, generation fails rather than writing
an unproven receipt. The sealed bytes are exactly the compiler's output.

The observed paths for both committed targets are `expf` → `llvm.exp.f32` and
`powf` → `__ocml_pow_f32`. The linked OCML exponential wrapper itself uses the LLVM
exponential intrinsic. The final code contains range reduction, `v_exp_f32`,
`v_ldexp_f32` and range selects; replacing that with a bare exponential instruction
would not reproduce these bytes. `llv_build_p`, which contains the probability
exponentials, is inlined into the out-of-line `llv_tail`; the helper checks cover
that complete linked function.

## Tests

`crates/peacemaker-isa/tests/hip_twin_leviathan.rs` is CPU-only and consumes committed files via
`include_bytes!`. It checks manifest hashes against source and artifacts, anchor vs twin ELF and
`.text` hashes, `verify_hxaco` and `seal` with `abi_ref` roundtrips, the exact seven-arg ABI, and
rejection of truncated or corrupted bundles, wrong arch and wrong symbol. Compiler version and
paths are never asserted.

The ignored `hipfire-runtime` target `x1_lane_draw_shard_gpu` additionally loads both
committed objects directly (no source compilation or product-dispatch override):

```
cargo test --release -p peacemaker-isa --test hip_twin_leviathan -- --nocapture
cargo test --release -p hipfire-runtime --test x1_lane_draw_shard_gpu -- --ignored leviathan_twin --nocapture
```

Run the GPU command through the site's GPU admission wrapper. The exact device
architecture selects its object; unsupported devices fail rather than silently skip.
The oracle checks raw mode-0 outputs/p tables, both mode-1 candidate tables, and raw
mode-2 outputs/p tables against hipcc, plus mode 1+2 against mode 0 within each image.
It reuses the X1 matrix over V = 248,320 / 5,000 / 4,098 (360 lanes total), including
temperatures, top-k 0/20/64, top-p/min-p, penalty signs, blocked tokens, LDS-overflow
mass ties and ties at the 124,160 shard boundary, with additional signed-zero,
NaN-payload and infinity inputs.

## Proof boundary: full M7 certification remains OPEN

The twin-equivalence claim is full device-ELF identity, typed helper replay and
hazard replay, exercised by dispatching the committed PM image against hipcc.
It is not a claim that PeaceMaker's stronger analyses certify the original LLVM
kernel. The explicit diagnostic still fails, with no findings suppressed:

```
cargo test --release -p peacemaker-isa --features toolchain --test hip_twin_leviathan -- --ignored m7_reports_open_findings --nocapture
```

Both targets report the following OPEN obligations:

| Class / rule | Count |
|---|---:|
| Definedness / definedness-entry | 4 |
| SrcReadTiming(DsStore) / wait-war-ds-store | 160 |
| SrcReadTiming(VmemStore) / wait-war-vmem-store | 13 |
| Unknown / scratch-in-use | 1 |

The scratch allocation is **272 bytes per work-item**, with bounds not certified.
The four path-definedness sites are `0x6790` (`v7`, `v8`), `0x6b8c` (`s24`),
`0x6bac` (`s24`) and `0x6bc4` (`s22`). The diagnostic also records 51 ambiguous
delay windows and 333 unknown LDS addresses. These are not counted as passes.
Each `.twin.json` carries the complete summary bound to its exact image hash.
The normal CPU gate audits that honest record; the ignored diagnostic above is
intentionally separate and reports its refusal for both architectures.

Hardware receipt **j8420**, gfx1201, is 2/2 tests and 360 lanes byte-identical,
including all three modes; the fabric marked it unconfounded. No gfx1200 hardware
run is claimed. Its checked machine-code and helper hashes equal gfx1201's, while
its ELF machine identifier and bundle target remain independently gfx1200.

Gfx1200 encoding support does not inherit gfx1201's numerical qualification.
The IR emulator admits gfx1200 at construction (the ABI initializer is keyed on
the gfx12 family) but refuses to run it unless `gfx1200_as_gfx1201()` is called
explicitly; that opt-in is shared-ISA reasoning only, with numerics taken from
gfx1201 and no gfx1200 hardware evidence. Helper calls (`s_getpc_b64`,
`s_swappc_b64`, `s_setpc_b64`) execute in the emulator on gfx1201; the M7
all-path obligations remain OPEN.

## F4 `gemm_mq6sym_i8_set_*_x4_gfx1201` (same Route A, proof only)

`scripts/pm-gemm-mq6sym-i8-twin.py` seals the four multi-part F32 narrow entries
`gemm_mq6sym_i8_set_{m64n16,m32n32,m64n48,m64n16y}_x4_gfx1201` from one hipcc object built from the
runtime source text `GEMM_MQ6SYM_I8_GFX1201_SRC` (the `HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE`
define, `block_i8_128_quant.hip`, then `gemm_mq6sym_i8.gfx1201.hip`) with the same core flags.
Route A keeps the whole device ELF, so the image also contains the module's other entries;
only the four x4 symbols are sealed with `--abi-ref` against the anchor (96-byte explicit
kernarg, no hidden args). Dispatch is unchanged: the registry HIP source stays the runtime path.

| File (`kernels/`) | Content |
|---|---|
| `gemm_mq6sym_i8_hipcc_gfx1201.co` | hipcc anchor |
| `gemm_mq6sym_i8_pm_gfx1201.hxaco` / `.hxaco.sha256` | sealed twin |
| `gemm_mq6sym_i8_pm_gfx1201.twin.json` | source, prelude and combined-source sha256, flags, one seal report per entry |

Test: `crates/peacemaker-isa/tests/hip_twin_gemm_mq6sym_i8.rs` (CPU only; stale source or prelude
hash fails it; rerun the script after editing either file).

Hardware receipt **j8542**, gfx1201 card B: the `qwen4_mq6_x4_sym oracle` directly loaded the
committed PM bundle and compared it with hipcc x4 and per-member launches: **88 cells,
42,654,976 words, zero mismatches or tail violations**, all four tiles covered, negative controls
passed. Ten raw-scale edge cells include signed zero, infinities, NaN payloads, subnormal and
maximum-finite scales, in addition to finite random inputs, part permutations and missing slots.
The fabric marked the run unconfounded. This hardware equality does not discharge the OPEN
strict-ledger obligations below.

Twin-build lane trace **j8543** (EP2, eight lanes, MTP k3, 16 tokens): flag on launched 120
`m32n32_x4` and 96 `m64n16_x4` kernels; flag off launched none. All 32 verify dumps
(301,000,215 bytes) and lane/reference/window ID records were equal across arms. Both independent
lane-vs-singleton self-checks exited 1 at the same K8/prompt-5 case; this is not a passing
lane-vs-singleton claim. The trace wrapper exited 0 and the fabric marked it unconfounded.
The identical K8 failure is the pre-existing all-rows lane-isolation defect (also observed at
K4/K8 on base during MtpIsoShep's release triage), not an F4 regression.

### F4 proof boundary

The `hip_twin` seal reports 0 blocking hazards for the four sealed x4 entries.
Separately, the strict CFG ledger reports 36 UNPROVEN target findings
(18 original LLVM symbols x 2 bundle copies: the `.co` hipcc image and the `.hxaco` twin), all DS
store source VGPR writes without a prior DS-counter wait on a CFG path; the pool bodies hosting
them add 36 `Candidate` body pins. Each is allowlisted exactly (instruction, label, need) with no
suppression and no category change. They are original LLVM output, not changes introduced by the
twin, and none is claimed to be a race. The checked full-ELF byte identity of anchor and twin does
not discharge the ledger's strict WAR path, and hardware source-read timing and path feasibility
remain unproven. No stronger analysis proof is claimed.

## `lwmma_gemm_bf16_t2_m{1,2,3,4}_n1_multi` (same Route A, proof only)

The four multi-part lane-WMMA entries of `kernels/src/qwen4_lane_wmma.gfx1201.hip`
(`HIPFIRE_QWEN4_LWMMA_MULTI_PART`) are sealed by `scripts/pm-leviathan-twin.py --family lwmma-multi
--arch gfx1201` into one bundle. The seal covers the whole device ELF (all 67 kernels, the plain
lane-WMMA entries included), which is byte-identical to the hipcc anchor; there is no math
evidence from this script. Dispatch stays the registry HIP source.

| File (`kernels/`) | sha256 | Content |
|---|---|---|
| `qwen4_lane_wmma_multi_hipcc_gfx1201.co` | `4a4d20cc17d64366173e21fddaab1b093589bc8371f814a9b59036b7c0945d0f` | hipcc anchor (1,371,600 bytes) |
| `qwen4_lane_wmma_multi_pm_gfx1201.hxaco` / `.hxaco.sha256` | `eb900a45d084886ab86c1a96b66efd309956644808d3ce5ac0a547d495c87679` | sealed twin (1,371,600 bytes) |
| `qwen4_lane_wmma_multi_pm_gfx1201.twin.json` | | source sha256 `d059713f780edcf9122d16d0c6722ef63ec5abc4531975ca545c8e4ecd43e2dd`, flags, one seal report per symbol |

Device ELF (1,367,504 bytes) sha256 `c3f9b5264627e746df6d6ed05ac41d44fe39dc38a00308b41a1338a8ccb47452`,
recorded by every seal report. Per-entry `.text` sha256 (manifest `seal.reports`):

| Symbol | `.text` bytes | `.text` sha256 | VGPRs | LDS |
|---|---:|---|---:|---:|
| `lwmma_gemm_bf16_t2_m1_n1_multi` | 6,784 | `1a1f81c4a3dc27022667c38348cc99d8695f3cfd457c730696b143817987bbeb` | 104 | 4,100 |
| `lwmma_gemm_bf16_t2_m2_n1_multi` | 11,892 | `3c1b98d0caa588d3f76b7db25c895a40b8ed7c29eca762356976df495a7ef579` | 104 | 8,196 |
| `lwmma_gemm_bf16_t2_m3_n1_multi` | 16,504 | `2bb99d87e61142682adcdc49a271d314ccdd33b021fe8d19b69ca92abae35f80` | 144 | 12,292 |
| `lwmma_gemm_bf16_t2_m4_n1_multi` | 20,824 | `b70fbd6fef1cea6969b554735c9a77699d12bdf379100bbcb539901c9733a34a` | 200 | 16,388 |

Test: `crates/peacemaker-isa/tests/hip_twin_lwmma_multi.rs` (CPU only; a stale source hash fails it).

### Proof boundary (ledger / railgun rows: OPEN)

The `hip_twin` seal reports 0 blocking hazards, 0 advisory findings and 0 scratch for the four sealed
entries. Separately, the strict CFG ledger reports 134 UNPROVEN target findings across the whole ELF
(67 kernels x 2 bundle copies: the `.co` hipcc image and the `.hxaco` twin; `ledger_cfg_pins`
`allowlist.rs`, targets `qwen4_lane_wmma_multi_{hipcc,pm}_gfx1201.*#<symbol>`), each allowlisted
exactly (instruction, label, need) with no suppression:

| Class | Findings | Status |
|---|---:|---|
| DS store source VGPR written without a prior DS-counter wait on a CFG path (`LWMMA_MULTI_DS_WHY`) | 90 | OPEN / UNPROVEN |
| VMEM store source VGPR written without a prior store-counter wait on a CFG path (`LWMMA_MULTI_ST_WHY`) | 40 | OPEN / UNPROVEN |
| `Km` WAW on `s[2:3]` from a wave32 `v_cmp_*_e64` (`LWMMA_MULTI_KM_WHY`; `lwmma_gemm_w8_t{1,2}_m4_n1`, both copies) | 4 | MODEL LIMITATION |

The four `_multi` entries themselves carry 8 of the 134 rows (m1, m2: `Store` WAR; m3, m4: `Ds` WAR; each in
both copies). The pool bodies hosting these findings add 134 `Candidate` body pins (templates 359-498).
All are original LLVM output, not changes introduced by the twin, and none is claimed
to be a race. Checked full-ELF byte identity of anchor and twin does not discharge the ledger's
strict WAR paths, and hardware source-read timing and path feasibility remain unproven. No stronger
analysis proof is claimed, and no railgun certificate is claimed for these entries.

## X8 wide MoE glue and rot planar gate/up (Route A, proof only)

Two gfx1201 families of `scripts/pm-leviathan-twin.py` seal the X8 entries of `HIPFIRE_QWEN4_MOE_GLUE_FOLD`
(`--family moe-route-scatter-wide`, `--family qwen4-moe-ep-a8-rot`, both `--arch gfx1201`). Each compiles the exact
concatenated runtime module source (the manifest's `module_source` records the define, every part's path, bytes
and sha256, and the combined sha256: `MOE_ROUTE_SCATTER_WIDE_SRC`, `QWEN4_MOE_EP_A8_ROT_SRC`) with the core hipcc
flags and seals the whole device ELF; the extracted device ELF (every byte of the amdgcn bundle entry) is byte-identical to the hipcc anchor's (`lift_identity`: `emit::checked ok; emitted
bytes == input ELF`). The `.co`/`.hxaco` files are clang offload bundles (4,096-byte header + device ELF; the two containers have the same size but different sha256 because the bundle header bytes are normalized differently by the seal, so identity is claimed for the extracted device ELF only, not for the container); the
per-symbol `elf_sha256` below is the extracted device ELF, shared by both symbols of a module. No math evidence.
Dispatch stays the registry HIP source.

| Module (`kernels/` prefix) | Symbols | Container sha256 (`.co` anchor / `.hxaco`) | Device ELF sha256 |
|---|---|---|---|
| `moe_route_scatter_wide_{hipcc,pm}_gfx1201` | `moe_scatter_fused_top10_pmask`, `moe_route_scatter_top10_wide_f32` | `ae68aaa75dea39da94758853e32ac283c7e6113476d57a22976813e6008f396a` / `897d2559b4a962d7b35f579d72bc22d243f21cce17b91b27e2f526f18536fbf1` (57,664 bytes each) | `66b336f9ebdcc4095726d8ed28808a7a083eccdb2aeb5fe43ba91fb8482624e1` |
| `qwen4_moe_ep_a8_rot_{hipcc,pm}_gfx1201` | `qwen4_moe_ep_a8_gate_up_rot_planar_nt1`, `_nt2` | `d2682b9183f134b45c171d0d11dfcb857f6b593730e58c36143dc2462a670a1d` / `64690989fb57520200cc3f85e6f8042141aa3ef811d27fe97943b470d6daf3cf` (147,328 bytes each) | `fa83561ed67bb97b17d873d3f3b939a66b58b32f275f067883edb00f6d3c28d2` |

Per-entry `.text` sha256, VGPRs, LDS (manifest `seal.reports`; 0 blocking hazards, 0 scratch):

| Symbol | `.text` sha256 | VGPRs | LDS |
|---|---|---:|---:|
| `moe_scatter_fused_top10_pmask` | `728141e7f45843ee165a80360998a0dbfb25679e5bfc90eeb7c7dce0577ed944` | 24 | 128 |
| `moe_route_scatter_top10_wide_f32` | `6e3a9766c1cc5ceea036fc246939c79d2761783a10d195336d8c2d898028eb0d` | 80 | 16,516 |
| `qwen4_moe_ep_a8_gate_up_rot_planar_nt1` | `98bc68b6037a2c15de5fa99b7662ed2aa2e8e9a1ca0601449e85ac76438cf183` | 168 | 4,352 |
| `qwen4_moe_ep_a8_gate_up_rot_planar_nt2` | `50d597e93acbcf24162bc01689464949c2f98a4b57c03ac1aac36fdbcd72a0cc` | 248 | 8,704 |

The rot module also defines the eight legacy A8 entries; lifting the whole ELF needed `v_cmp_eq_u16_e32`
(VOPC `0x3a`, true16) in the gfx12 decode table (used by `qwen4_moe_ep_a8_down_nt2`/`down_planar`). Test:
`crates/peacemaker-isa/tests/hip_twin_x8.rs` (CPU only; a stale source part hash fails it). Ledger and railgun
rows for these entries are OPEN; no stronger proof is claimed.

ST1 re-seals the rot module after issuing activation loads before weight loads in `EP8R_LOAD`.
The cold-weight story C8 measurements (j8799/j8800) improve by 3.66%; this is a schedule-only
change, not a stronger ledger proof or a change to HIP dispatch.

## gfx942 rows16: typed CDNA Route A

The same anchor/seal/provenance pattern is used for
`gemm_bf16_mfma_gfx942_rows16` and `gemm_bf16_mfma_gfx942_rows16_x2`
from `kernels/src/gemm_bf16_mfma.gfx942.hip`. Production dispatch stays HIP.
This bounded CDNA route reuses `peacemaker-cdna`'s typed codec and the shared
ELF envelope codec; it does not alias gfx942 to an RDNA target. Every kernel
in the object is decoded and re-encoded before full device-ELF identity is
required. The selected entries additionally pass the existing CDNA
software wait-state checker and typed CFG memory-ledger replay, without
inserting NOPs or changing instructions. Unknown encodings, helper bodies,
indirect branches, GDS and unsupported memory effects fail closed.

The four artifacts under `kernels/` are
`gemm_bf16_mfma_gfx942_rows16_hipcc_gfx942.co`,
`gemm_bf16_mfma_gfx942_rows16_pm_gfx942.hxaco`, its `.hxaco.sha256`,
and `.twin.json`. The manifest records source hash, portable core flags,
compiler provenance and each symbol's seal report. Both entries have nine
pointers followed by eight i32 arguments (104 explicit kernarg bytes).
`hip_twin_gfx942_rows16` checks source staleness, artifact pins, ELF/text
identity, ABI and checked evidence without invoking a compiler.

Regenerate explicitly with `scripts/pm-gfx942-rows16-twin.py`; its options
mirror the Leviathan generator. The lab example `gemm_bf16_rows16_oracle`
loads the committed anchor and PM image directly, and compares their
outputs with the shipped tiled MFMA kernel over rows, groups, dtypes,
tails, random/edge/NaN/Inf inputs and poisoned output guards.
Its full hardware receipt is bound to the image hashes in `twin.json`.
No broader CDNA IR/rewrite, helper or emulator qualification is claimed.

## P3 BF16 prefill tiles (Route A, proof only)

`scripts/pm-leviathan-twin.py --family lwmma-tile --arch gfx1201` compiles
`kernels/src/qwen4_lwmma_tile.gfx1201.hip` with the portable core flags and seals
all 10 selectable entries. The committed artifacts are
`qwen4_lwmma_tile_hipcc_gfx1201.co`, `qwen4_lwmma_tile_pm_gfx1201.hxaco`,
its `.hxaco.sha256`, and `.twin.json`, under `kernels/`.
The manifest pins source, compiler provenance, container hashes and each
entry's device-ELF and text hashes. Identity means every byte of the extracted
device ELF, not necessarily the enclosing offload-bundle header.

The selected one-term tiles are `b32x32w2x1g4`, `b32x64w2x2`, `b64x64w4x1`.
The two-term tiles are `b16x32w1x1g4`, `b32x64w2x2`, `b64x64w4x1`,
`b64x64w4x2`, `b32x128w2x2`, `b64x128w4x2`, `b64x128w2x4`.
Their nine explicit arguments are `W, X, Y, M, K, rows, split_k, scale, hc_act`
(48 bytes). The ABI family is `qwen4_lwmma_tile.hipcc.gfx12.bf16_tile`.
`hip_twin_lwmma_tile` checks committed inputs only: source staleness, portable
recipe, per-entry resealing, complete ELF identity, ABI and rejection of a
different architecture or absent symbol.

Dispatch stays on HIP. Ledger and railgun proof status remains OPEN: a
Route-A byte-identity seal does not discharge strict CFG wait findings,
state-space refusals, path feasibility or hardware store-source timing.
Exact exceptions belong to `ledger_cfg_pins/allowlist.rs`; dispatcher
verification remains fatal and no railgun certificate is claimed.

The shared extracted device-ELF sha256 is
`cabbcdc60def8995550c0cddbe70ae85e0aa0f5ccea6c2ba2efdded73b089dd0`.
The 232,168-byte anchor container sha256 is
`aac57cde433f4e8d9fb2be78d65d1a35b4119e3aff5bfbbf543eea9b8c37c747`;
the same-size sealed container sha256 is
`6160b53cb59c46e8a90ac068cdaf75d2c3bb2daddef085e02cbbfc37923ca309`.
Every selected entry's seal records zero scratch and zero blocking/advisory hazards.

The exact ledger rows cover 20 standalone findings and 20 hosted-body
findings (10 entries in each image): 18 Store-source WAR findings are
OPEN / UNPROVEN; the two `t2_b16x32w1x1g4` Km WAW findings are a wave32
scalar-destination-pair MODEL LIMITATION. The new templates are 505–524;
later pre-existing positional template ids move by 20 without changing
their findings. No tile STATE-CAP refusal was observed in the scan.

## GGUF-native decode GEMV shim (gfx1201, Route A, proof only)

`scripts/pm-leviathan-twin.py --family ggml-mmvq --arch gfx1201` compiles
`kernels/src/ggml_mmvq_shim.hip` with the portable core flags plus the recipe of
`scripts/build-ggml-mmvq-shim.sh` (`-ffp-contract=fast-honor-pragmas -DGGML_USE_HIP
-Ithird_party/ggml/{include,src,src/ggml-cuda}`) and requires the result to equal the
shipped runtime object `kernels/ggml_mmvq_shim_gfx1201.co` byte for byte: that object
is the anchor, so no second copy is committed. It seals all five exports,
`ggml_q8_1_quantize_gfx1201` and `ggml_mmvq_{iq4_xs,q4_k,q6_k,q8_0}_gfx1201`, into
`kernels/ggml_mmvq_shim_pm_gfx1201.hxaco` (+ `.hxaco.sha256`, `.twin.json`). The manifest
pins the source, the `third_party/ggml/PIN` commit file, the flags and every export's seal
report. `hip_twin_ggml_mmvq` checks committed files only: provenance, that the twin covers
every kernel of the anchor, per-export reseal, full device-ELF identity, explicit kernargs
(producer 5 args / segment 288; mmvq 4 args / segment 28 / 896 B LDS), and rejection of a
different architecture or an absent symbol.

The shared extracted device-ELF sha256 is
`e59bc73ff31d37aa3833cda34f83593ff99c944f32f1cf9456ef9e5736d02548` (14,968 bytes); the
19,064-byte anchor container sha256 is
`8b50bc03ef6f719478cba6d22aa5f6c734339dab9907d6bceaf217c9492e1a4c` and the same-size
sealed container sha256 is
`d126becf2f5f668cb5a9a42c54b9f0ec392d691a7f7a7d845b78c0a75cf610e7`.

Dispatch stays on HIP (`S1_IMAGE`). Ledger status is OPEN and is pinned negatively by
`hip_twin_ggml_mmvq::ledger_verdict_is_pinned_open_for_q4_k_only`: four exports verify and
`ggml_mmvq_q4_k_gfx1201` reports one `d16_hi` load use, identically in anchor and twin.
The ledger allowlist mirrors exactly that anchor finding and its four multi-axis
region-conversion refusals for the twin. These are existing classes, not discharged
obligations. Stable pool keys require no edits to any previously committed row.

## gfx942 FP8 grouped MoE: typed CDNA Route A

The same pattern seals the production object of
`kernels/src/gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip`: its three kernels
`gemm_fp8b128_moe_grouped_bf16mfma`, `gemm_fp8b128_moe_grouped_bf16mfma_xbf16`
(64-byte kernarg, `W, expert_stride, tile_desc, sorted_rows, X, Y, R, K,
x_row_div, n_exp`) and `gemm_fp8b128_moe_tile_desc_pad16` (20 bytes). The
diagnostic `_reference` entries exist only under `FP8MOE_TEST_REFERENCE`, which
the production build never defines, so they are in neither committed image.
Production dispatch stays HIP. The typed lift covers the object's FP8 FNUZ pair
convert (`v_cvt_pk_f32_fp8`, e32 and SDWA), LDS stage/read (`ds_write_b128`,
`ds_read2_b64`, `ds_bpermute_b32`), BF16 MFMA, packed F32 multiply and 64-bit
VALU address math; the object contains no atomic. Every kernel is decoded and
re-encoded to the identical device ELF, and each sealed entry passes the CDNA
hazard checker and the typed CFG wait ledger with no finding.

Artifacts under `kernels/`: `gemm_fp8b128_moe_grouped_bf16mfma_hipcc_gfx942.co`,
`gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.hxaco`, its `.hxaco.sha256`, and
`.twin.json`. Regenerate explicitly with `scripts/pm-gfx942-moe-grouped-twin.py`
(options as the rows16 script). `hip_twin_gfx942_moe_grouped` checks source
staleness, artifact pins, ELF/text identity, ABI, typed instruction classes and
checked evidence without invoking a compiler; `ledger_cfg_pins` and
`peacemaker-lift`'s `cdna_lift` certify both images. A hardware byte-oracle
receipt is not yet recorded: `twin.json` carries a `gpu_oracle` entry only after
an MI300X run bound to both image hashes, and the test fails a committed
receipt that is partial or stale.

## Twin identity vs. exploration diagnostics

A twin's identity is what its seal claims about the bytes: device-ELF, `.text` and
bundle hashes, `lift_identity`, ABI, the hazard checker result (`findings`, `blocking`,
`advisory`, `bypasses`, `hazard.*`, `unreachable_instructions`) and the structural counts
of the typed CFG wait ledger (`ledger.verifier`, `labels`, `branches`, `barriers`, `waitcnt*`,
`memory.*`). All of these are compared exactly when a fresh seal is checked against the
recorded `seal.reports`.

The gfx942 ledger replay additionally records what the verifier explored while proving
those claims: `hazards.ledger.steps` (instruction visits) and `hazards.ledger.label_states`
(distinct label/in-flight-state pairs). These are *diagnostics*, listed in
`hip_twin::LEDGER_DIAGNOSTIC_FIELDS`. They depend on the verifier's search strategy, so
they may change when the replay is made cheaper or more precise without any change to
the bytes or the findings; they are not part of a twin's identity and are never pinned.

- `hip_twin::pinned_report(&report)` returns the report with those two counters removed
  from `hazards.ledger`. Tests compare `pinned_report(fresh)` with `pinned_report(recorded)`,
  never the raw reports. Only the two known counters, and only when they are integers, are
  removed; everything else, including any unknown or malformed field, still has to match.
- `hip_twin::ledger_diagnostics(&report)` reads the counters for informational output only
  (`cdna_rip942_corpus` prints them when they move from the recorded values).
- Committed `kernels/*.twin.json` files are unchanged. Their legacy reports keep the
  counters in place and normalize exactly like fresh reports; a regenerated manifest
  may carry different counters with identical pinned claims.
- A change in findings, hazards, bytes or any structural count is a changed claim and fails
  the same checks even when the counters moved too (`diagnostics_tests` in `hip_twin.rs`,
  `exploration_counters_are_diagnostics_but_hazards_findings_and_claims_stay_pinned` in
  `hip_twin_gfx942_glue_elem`).
