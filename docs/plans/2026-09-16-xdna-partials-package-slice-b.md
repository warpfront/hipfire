# XDNA partials repair — slice B: package only executed, parity-verified artifacts

## Ownership and concurrency

Composer owns hipx `/home/kaden/npu-screen/part-h/package_sidecar.py`, staging under `/home/kaden/npu-screen/sidecar/`, and append-only §7 status in `/home/kaden/ClaudeCode/warpfront/wt-lloyd/docs/plans/2026-09-16-halo-npu-max.md` after coordinating with Main. No hipfire production edits, no hipfire-beta, no commits. Skip formatters, linters, and project-wide builds/tests. Main dispatches this independently while Slice A compiles/validates; package script preparation needs no NPU and cannot mutate the generator, harness, or profile outputs. Final package creation waits for Slice A's accepted results; reviewer owns final veto.

## Frozen inputs and public schema

Source profiles are `part-h/cfg-ph-wait-M1024`, `cfg-ph-wait-M2048`, `cfg-ph-wait-M4096`, not old cfg-ph-M* or JIT cache. Each must have design.xclbin, insts.bin, and a results.jsonl record naming exactly those executed artifacts and their SHA256, `validation=PASS`, K5120/N512, 10 warmups, 50 real NPU timing samples, and no missing samples. Require the independent changed-input/sentinel replay PASS from Slice A before release. If any result is absent or hash differs, refuse to publish; do not silently select an old PASS or an artifact with the same basename.

The current parser is `/home/kaden/ClaudeCode/warpfront/wt-lloyd/crates/hipfire-registry/src/xdna.rs`, symbols `XdnaArgLayout`, `XdnaProfile`, `XdnaManifest::validate`, `XdnaProfile::validate`, `XdnaSidecarDescriptor::load_verified`, `read_stored_entries`. Read their actual bodies. It denies unknown fields and accepts precisely:

- root: version, model_sha256, quant, arch, npu, toolchain, profiles;
- profile: tensor_role, M, K, N, row_count, tile_m, tile_k, tile_n, cols, k_mt, pdi, insts, arg_layout, sha256_pdi, sha256_insts;
- arg_layout entry: name, offset, size.

Bind `version=1`, model SHA256 `9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`, quant `mq4g256v2`, arch `gfx1151`, npu `npu5`, tensor_role `gate_up`; M=row_count in {1024,2048,4096}; K5120,N512; tile_m128/tile_k64/tile_n64, cols8,k_mt512. Keep toolchain provenance map factual (mlir_aie f50bef713297b7c5287a3ebbce964944354be11f; Peano22.0.0.2026090701 as recorded, verify against build provenance if discrepant).

## Artifact and layout interface

Manifest arg offsets are payload-relative address slots from generated kernels_main.json, not offsets into concatenated tensors: A20, B28, P36. Preserve `size` as total buffer allocation bytes (existing artifact contract); parser currently checks nonzero and offset+size overflow, not overlap. Do not conflate these sizes with the 8-byte address slot width.

Raw A is row-major int8 `[M,K]` weight codes 0…15. Raw B is row-major int8 `[K,N]` activation codes −8…7. P is little-endian int32, contiguous shape:

`[pair=M/1024][kg=K/512][col=8][rb=2][h4=4][r=512][c=64]`.

Logical mapping: `row=pair*1024+rb*512+r`, `half=kg*4+h4`, `column=col*64+c`; value is sum over exactly K indices `[half*128,(half+1)*128)`.

Since frozen v1 has no layout metadata object, encode the output convention directly in its allowed name field: `P:i32le[pair=M/1024][kg=K/512][col=8][rb=2][h4=4][r=512][c=64]`. A/B names remain A/B. Confirm Main has not frozen an incompatible literal-P name with the integration consumer before publication; never add an unknown JSON field. This is a format identifier, not a second payload or alias. CPU decoder is reshape into the seven dimensions, transpose(0,3,5,1,4,2,6), reshape(M,K/128,N). The GPU consumer reads this layout directly; no device-side reorder is claimed.

Byte contracts (calculated, not timing results):

| M | A bytes | B bytes | P bytes | dense i32 bytes | extra DDR writes vs dense | logical integer operations |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 5242880 | 2621440 | 83886080 | 2097152 | 81788928 | 5368709120 |
| 2048 | 10485760 | 2621440 | 167772160 | 4194304 | 163577856 | 10737418240 |
| 4096 | 20971520 | 2621440 | 335544320 | 8388608 | 327155712 | 21474836480 |

## Implementation steps

1. Update existing package_sidecar.py to consume cfg-ph-wait-M* and verify the exact executed-hash/timing/parity records before extraction/publication. Coordinate record field names directly with Slice A; they own those records, not this script. Do not alter measured records or round numbers before calculating statistics.
2. Extract `PDI:RAW` from each exact executed xclbin using xclbinutil into that profile's main.pdi (or staging if Slice A still owns profile contents); hash those extracted bytes and the actual executed insts.bin. Record source xclbin SHA beside the zip as provenance, not a new manifest field.
3. Produce exactly seven stored ZIP entries: manifest.json and `profiles/gate_up_M{M}/{main.pdi,insts.bin}` for each M. No payload compression, unknown JSON fields, stale entries, or `.xclbin` substitution. Target is `/home/kaden/npu-screen/sidecar/qwen3.8-27b.mq4-xt.xdna.zip`. Build to a temporary sibling, verify, then atomically publish so no failed build replaces a previously accepted artifact.
4. Open the resulting archive and verify stored method0 for every entry, exact entry set, manifest identity/geometry, payload SHA256, nonempty instruction words (size divisible4), and expected argument byte sizes/layout. Print full manifest and archive SHA256/size.
5. Invoke actual `XdnaSidecarDescriptor::load_verified` on the copied ZIP in a throwaway external Rust smoke executable using this module plus serde/serde_json/sha2/thiserror, or an existing scoped registry harness supplied by Main. Prefer reusing already-built dependencies. No project-wide compile/test. Python schema imitation alone is NOT proof of actual parser acceptance. Also check `admit_for_arch("gfx1151")` succeeds and gfx1201/gfx1100 reject; no NPU opens are needed for this CPU-only parser gate.
6. Append a SHORT measured status subsection to §7 of the existing halo plan only after all results are present. State it is int8→i32 K128 partials, not the future f32 hardware fold described elsewhere in §7. Include per-profile NPU median/p90 and end-to-end µs, TOPS, extra DDR bytes, layout mapping, exact ZIP/hash, n=50, parity and changed-input replay proof, plus root cause: wrong decoder order + stale JIT artifact + premature live-BD reuse. Do not rewrite older historical measurements as if they were new evidence.

## Lifecycle and release invariants

Unvalidated AOT files are candidate-owned and never published. Validation consumes immutable candidate hashes. Extraction derives only from validated xclbin bytes. Packaging owns a private staging archive until parser/hashes pass; atomic publication transfers a complete manifest+three payload pairs as one unit. Future runtime admission remains default-off and exact-gfx1151/npu5/model-bound. A zip passing load_verified is format/hash acceptance only: no GPU epilogue parity, raw-ioctl execution, model KLD, net overlap speedup, graph capture, retained replay, decode, down-projection, gfx1201, or gfx1100 behavior is certified by this slice.

## Claim-scoped acceptance and abandon criterion

Provide exactly 50 device NPU samples for each shape, exact integer parity and changed-input proof, actual parser acceptance, seven stored entries, manifest print, and final artifact path/hash. Compute TOPS as `2*M*K*N/(median_us*1e6)`; report end-to-end separately so input synchronization/copy costs are not hidden. Partial output volume is 40× dense i32 and its GPU read/fold cost remains unmeasured. Do not advertise a speedup from the earlier 15.33 dense TOPS. If measured complete copy/fold/contention service cannot beat GPU-alone beyond drift, integration must choose zero NPU rows. No optional kernel/layout optimization is a deliverable unless measured; do not hold this correctness artifact for unbounded tuning.

Final user-facing report must be 10 lines: three profile timing/parity lines, layout, exact root cause, raw sample scope, DDR overhead, ZIP path/hash, parser result, and scope/non-performance caveat. Missing gates must remain explicitly unaccepted, not described as done.
