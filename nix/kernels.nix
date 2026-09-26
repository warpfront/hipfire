{ lib
, rustPlatform
, rocmPackages
, src ? lib.cleanSource ./..
, cargoLockFile ? ../Cargo.lock
, gpuTargets ? [ "gfx1201" "gfx1100" "gfx1151" "gfx906" "gfx942" ]
}:

let
  cargoToml = builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"));
in
rustPlatform.buildRustPackage {
  pname = "hipfire-kernels";
  version = cargoToml.workspace.package.version or cargoToml.package.version;

  inherit src;
  cargoLock.lockFile = cargoLockFile;
  doCheck = false;
  dontCargoInstall = true;

  nativeBuildInputs = [ rocmPackages.clr rocmPackages.llvm.clang ];

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    export HIPFIRE_ROCM_PATH=${rocmPackages.clr}
    export HIPFIRE_HIPCC_EXTRA_FLAGS=--rocm-device-lib-path=${rocmPackages.rocm-device-libs}/amdgcn/bitcode
    bash scripts/compile-kernels.sh ${lib.concatStringsSep " " gpuTargets}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/kernels/compiled"
    for arch in ${lib.concatStringsSep " " gpuTargets}; do
      test -f "kernels/compiled/$arch/rmsnorm.index.json"
      cp -r "kernels/compiled/$arch" "$out/kernels/compiled/"
    done
    runHook postInstall
  '';

  meta = with lib; {
    description = "Pre-compiled GPU kernels for hipfire";
    license = [ licenses.asl20 licenses.mit ];
    platforms = [ "x86_64-linux" ];
  };
}
