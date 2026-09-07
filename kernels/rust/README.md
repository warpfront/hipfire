# kernels/rust — Rust-native VCN preprocess kernels (experiment)

Rust (`amdgcn-amd-amdhsa`) port of `kernels/src/vl_yuv_preprocess.hip`,
to compare rustc vs hipcc codegen through the same dispatcher.

Build: `./build.sh <gfxNNNN> [--fp-contract]` → `target/<gfx>/*.elf`.
Verify: `llvm-readelf -n` (kernargs 64/40) + `llvm-objdump -d`
(no `s_trap`/`ds_`, ends `s_endpgm`); parity A/B runs in saddle-lab.
Workgroup shape (16×16 rgb, 256 patch) is a fixed kernel contract.
