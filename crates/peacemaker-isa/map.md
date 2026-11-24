# peacemaker-isa — map

> **Status:** `production` / `research` / `legacy` — pick exactly one
> (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** see the layering table in
> [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) — do not restate it here.

## Purpose

<!-- hand-written: what this crate owns and why it exists. Prefer pointing at
     the `//!` docs in `src/lib.rs` over copying prose. -->

## Gotchas

<!-- hand-written: what not to do here; traps a reader will hit. -->
- `examples/cdna_mix_hw_oracle.rs` compares actual gfx942 `v_fma_mix_f32` instructions against `peacemaker_cdna::sim::exec_one`, including raw NaN/sign bits and explicit MODE readback. Run it through the MI300X ledger wrapper; `--dry-run` is interpreter determinism evidence only, never a hardware receipt.
- `examples/cdna_nor_hw_oracle.rs` is a narrow gfx942 integer-SALU hardware/interpreter oracle: 4,096 edge/random pairs, separate and both aliased destinations, alternating incoming SCC, two output poisons and allocation guards. Its 24,576 exact comparisons and 12 detected corruption controls are bound by `tests/cdna_nor_receipt.rs`; run only through the MI300X ledger wrapper.


- Wave-private LDS: `lds::Lds::add_wave_private` declares a `stride x waves` window (`LdsSlot::wave_private`, recorded in `BuilderProof::lds_slots`; `lds::private_windows` / `PrivateWindow` read it back). Its slot steps (`private_store/load/publish/retire`) are reachable only through the typed core's `Backend` (`author.rs`), refuse barrier transitions and shared store/load, and publish with no instruction. `pm_check::lds_wave_private` proves from the object that every lane of every wave stays in its own slice; the Builder's assembly carries one `; pm.wave_private symbol= base= stride= waves=` marker line per wave-private slot (`emit::kernel_assembly`, before `.protected`; `lds::private_marker` writes it, `lds::private_markers` reads it per symbol, so no author can omit it), and `toolchain::certify` derives the windows from those markers: whenever any exist it runs `lds_wave_private` with them (refusing a non-M7 symbol), and a contract's `IsaShapeContract::wave_private_windows` must be empty or equal to the derived set as a set, else it refuses. Kernels without wave-private slots emit byte-identical assembly.

- Fresh-hipcc SiLU oracle fixtures are keyed by the shared ROCm resolver's
  clang major (23 or 24); other majors fail rather than reuse a foreign golden.
  Keep the original clang-23 region templates: the native builders still use
  those DAGs. To regenerate an oracle fixture, follow the hipcc compilation,
  unbundling and disassembly procedure in `tests/gfx11_builder.rs` or
  `tests/iu4_gemm_shape.rs`, then capture `peacemaker-isa region-import --arch
  gfx1100|gfx1201 --disassembly FILE` stdout in the corresponding
  `kernels/*.silu.clang24.region.s` file (the comparison against clang 23 is
  expected to fail). Include the ROCm/HIP/clang provenance as a `;` comment.
- `native::VERSION` describes the lld 24 layout verified by `native_identity`;
  it is not the ELF `.comment` stamp. `NATIVE_COMMENT` keeps version 0.4.1.
  Re-emission via `peacemaker-isa emit ... --bundle FILE` therefore preserves the
  committed bundle bytes. Most committed bundles use `--host-target
  host-x86_64-unknown-linux-gnu-`; the separate A4-fusion and gfx1151 MQ6 bundles
  use the default spelling without the trailing hyphen. The `_b1_control` object intentionally
  preserves an older instruction schedule, not the current builder's module.
- Every builder module (`emit::assembly`, kept once by the `iu4_gemm::module` / `fp8_gemm::module`
  combiners) ends with LLVM's gfx10+ AMDHSA `s_code_end` guard, `emit::CODE_END_TAIL`: the final
  `.text` aligned to 128 bytes, then 96 dwords of `0xbf9f0000` (`.p2alignl 7, 3214868480` +
  `.fill 96, 4, 3214868480`, from `AMDGPUTargetAsmStreamer::EmitCodeEnd`; the same for gfx1100,
  gfx1151 and gfx1201). It sits after `.end_amdgpu_metadata`, so a module takes the first kernel's
  copy and drops the rest; `native::assemble` lays it out after the last kernel and no symbol size or
  descriptor includes it. The typed `link_encoded` paths (gfx942 CDNA, ACO gfx1201) do not add it.
  Committed `.hxaco`/`.s` products predate it until re-emitted.
- `kernels/mq4_verify.rs` emits byte-identical twins of the hipcc
  `_vt{4,8}w{4,8}_k32` instantiations of
  `kernels/src/gemm_mq4g256v2_wmma_gfx12_vt_core.hip`. A change to either side
  must be mirrored in the other and the committed bundle re-emitted; the
  `cb_verify_gemm_oracle` `prodcross` test compares the two families byte for
  byte.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
