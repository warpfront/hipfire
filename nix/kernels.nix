let
  # rdna-compute kernel_registry::SUPPORTED_ARCHES. Other targets (gfx1200,
  # gfx1030, ...) have no registry to package; the daemon JIT-compiles them.
  registryArchs = [ "gfx1201" "gfx1100" "gfx1151" "gfx906" "gfx942" ];
in
{ lib
, rustPlatform
, symlinkJoin
, rocmPackages
, src ? lib.cleanSource ./..
, cargoLockFile ? ../Cargo.lock
, gpuTargets ? registryArchs
}:

let
  cargoToml = builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"));
  packagedTargets = lib.concatStringsSep " " (lib.filter (arch: lib.elem arch registryArchs) gpuTargets);
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
    export HIPFIRE_ROCM_PATH=${import ./rocm-root.nix { inherit symlinkJoin rocmPackages; }}
    # No arguments would make the script package every registry arch.
    ${lib.optionalString (packagedTargets != "") "bash scripts/compile-kernels.sh ${packagedTargets}"}
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/kernels/compiled"
    for arch in ${packagedTargets}; do
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
