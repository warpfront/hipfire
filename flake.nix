{
  description = "hipfire — LLM inference for AMD RDNA GPUs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    let lib = nixpkgs.lib; in
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        hipfire = pkgs.callPackage ./nix/package.nix {
          rocmSupport = true;
          src = lib.cleanSource ./.;
          cargoLockFile = ./Cargo.lock;
          kernels = hipfire-kernels;
        };

        # The package and standalone kernel output share the same source and
        # exact-source registry. Unsupported GPUs still use hipcc JIT.
        hipfire-kernels = pkgs.callPackage ./nix/kernels.nix {
          src = lib.cleanSource ./.;
          cargoLockFile = ./Cargo.lock;
        };
      in
      {
        packages = {
          default = hipfire;
          inherit hipfire hipfire-kernels;
        };

        devShells.default = pkgs.callPackage ./nix/dev-shell.nix {
          rust-bin = pkgs.rust-bin;
          rocmSupport = true;
        };
      }
    ) // {
      nixosModules.default = import ./nix/module.nix;

      # Inject rocmPackages from this flake's pinned nixpkgs (unstable,
      # ROCm 7.x). nixos-25.11 ships rocm 6.4.3 whose libamdhip64 segfaults
      # on gfx1151 (Strix Halo) during weight upload. The override flows to
      # both hipfire/hipfire-kernels (callPackage with `final`) and to the
      # NixOS module's `pkgs.rocmPackages` LD_LIBRARY_PATH.
      overlays.default = final: prev: {
        rocmPackages = (import nixpkgs {
          inherit (final) system;
          config = { rocmSupport = true; allowUnfree = true; };
        }).rocmPackages;
        hipfire = final.callPackage ./nix/package.nix {
          rocmSupport = true;
          src = lib.cleanSource ./.;
          cargoLockFile = ./Cargo.lock;
          kernels = final.hipfire-kernels;
        };
        hipfire-kernels = final.callPackage ./nix/kernels.nix {
          src = lib.cleanSource ./.;
          cargoLockFile = ./Cargo.lock;
        };
      };
    };
}
