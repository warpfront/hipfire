gfx1100 hipcc fixture for `tests/x1_gfx1100_repass.rs` (PM-KRSI-1100 lift-and-repass).

`gemm_mqv2_wmma_gfx1100_mw_lds.hsaco` is the production gfx1100 object of module
`gemm_mqv2_wmma_gfx1100_mw_lds` (`rdna-compute` `GEMM_MQV2_WMMA_GFX11_MW_LDS_SRC`, the `.hip` beside it),
built by `rdna-compute/examples/lift_probe_compile.rs` through `KernelCompiler::compile_to_paths`:
`hipcc --genco --offload-arch=gfx1100 -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6
--rocm-path=/opt/rocm/core --hip-path=/opt/rocm/core -I/opt/rocm/core/include` (HIP 7.16.26385).
sha256 c81a1c46757a4c9c27ae696a750534d4d70e301d4716be45d5b8491f3b7c27da.
