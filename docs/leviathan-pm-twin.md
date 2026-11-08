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
