# kernels/rust — Rust-native kernels (experiment)

Rust (`amdgcn-amd-amdhsa`) ports of HIP kernels, to compare rustc vs
hipcc codegen through the same dispatcher.

Crates: `vl_yuv_preprocess` (VCN YUV→RGB + patches), `gemv_mq4g256v2`
(HFQ4-G256 GEMV, generic path: runtime K, `y[row] = acc`).

Build: `./build.sh <crate> <gfxNNNN> [--fp-contract]` →
`target/<gfx>/<crate>[-fpc].elf`. `--fp-contract` passes
`-fp-contract=fast` (matches hipcc default; plain build matches
hipcc `-ffp-contract=off`).
Verify: `llvm-readelf -n` (kernargs) + `llvm-objdump -d` (no `s_trap`,
ends `s_endpgm`); parity A/B runs in saddle-lab.
Workgroup shapes are fixed kernel contracts (16×16 rgb, 256 patch,
32-thread GEMV wave).
