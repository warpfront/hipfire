# Leviathan PeaceMaker twin (Route A)

`lane_leviathan_verify` (`kernels/src/lane_leviathan.hip`) has committed PeaceMaker twins for
`gfx1200` and `gfx1201`, covering modes 0 (full row), 1 (select) and 2 (merged). The three modes
are `mode` kernarg branches of one symbol, so sealing the symbol seals all of them. The kernarg
ABI is unchanged: seven explicit args `region, lanes_off, rows_off, ptab, mode, cand_local,
cand_peer` at offsets 0, 8, 12, 16, 24, 32, 40. Dispatch is not changed by these artifacts.

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
