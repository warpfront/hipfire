# hipfire resolves HIPFIRE_ROCM_PATH as one authoritative root that must hold
# HIP headers, libamdhip64, libhsa-runtime64 and bin/hipcc; its kernel compiles
# pass --rocm-path=<root>, so clang also looks for amdgcn/bitcode there. nixpkgs
# splits these between clr, rocm-runtime and rocm-device-libs: join them.
{ symlinkJoin, rocmPackages }:

symlinkJoin {
  name = "hipfire-rocm-root";
  paths = [ rocmPackages.clr rocmPackages.rocm-runtime rocmPackages.rocm-device-libs ];
}
