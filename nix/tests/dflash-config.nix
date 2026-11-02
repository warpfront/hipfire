# Evaluation check for services.hipfire.dflashMode.
#
# An untouched option must write no `speculation.dflash` (the registry's
# per-model policy then applies); explicit "auto" / "off" / "on", and the
# legacy extraSettings.dflash_mode alias, must be written as globals.
#
# Only the tiny config.toml derivations are built: hipfire and hipfire-kernels
# are replaced by empty stub derivations through an overlay.
{ pkgs, lib, nixosSystem, system }:

let
  stub = name:
    let d = pkgs.runCommand name { } "mkdir -p $out/bin";
    in d // { override = _: d; };

  setupScript = settings:
    (nixosSystem {
      inherit system;
      modules = [
        ../module.nix
        {
          nixpkgs.overlays = [
            (_final: _prev: {
              hipfire = stub "hipfire-stub";
              hipfire-kernels = stub "hipfire-kernels-stub";
            })
          ];
          services.hipfire = {
            enable = true;
            gpuTargets = [ "gfx1100" ];
          } // settings;
        }
      ];
    }).config.systemd.services.hipfire-setup.script;
in
pkgs.runCommand "hipfire-dflash-mode-config" {
  # Each script string carries a context dependency on its generated
  # config.toml, which is built before the check runs.
  unset = setupScript { };
  explicitAuto = setupScript { dflashMode = "auto"; };
  explicitOff = setupScript { dflashMode = "off"; };
  explicitOn = setupScript { dflashMode = "on"; };
  extraOn = setupScript { extraSettings.dflash_mode = "on"; };
} ''
  config_of() {
    printf '%s\n' "$1" | grep -o '/nix/store/[^ ]*hipfire-config\.toml' | head -n1
  }
  expect_absent() {
    f=$(config_of "$2")
    if grep -q 'speculation\.dflash' "$f"; then
      echo "FAIL ($1): speculation.dflash written:"; cat "$f"; exit 1
    fi
  }
  expect_value() {
    f=$(config_of "$3")
    if ! grep -Eq "^\"speculation\\.dflash\" *= *\"$2\"\$" "$f"; then
      echo "FAIL ($1): expected speculation.dflash = \"$2\":"; cat "$f"; exit 1
    fi
  }
  expect_absent unset "$unset"
  expect_value explicit-auto auto "$explicitAuto"
  expect_value explicit-off off "$explicitOff"
  expect_value explicit-on on "$explicitOn"
  expect_value extraSettings-dflash_mode on "$extraOn"
  touch $out
''
