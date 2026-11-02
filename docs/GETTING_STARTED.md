# Getting started

Audience: first install on an AMD GPU host. Goal: install → verify → pull a model → run or chat.

## Prerequisites

- **Linux:** AMD GPU with `/dev/kfd` plus a ROCm HIP stack. A release-tag
  install on an admitted GPU (gfx1201, gfx1100, gfx1151, gfx906, gfx942) can
  use the tag's prebuilt [kernel pack](#prebuilt-kernel-packs) for the kernel
  registry, and installs with only the HIP runtime (`lib/libamdhip64.so`,
  `libhsa-runtime64.so`) and `bin/rocm_agent_enumerator`. Kernels outside the
  registry still JIT on first use. On gfx1201 the registry covers Qwen3.8 H2
  AR, MTP and DFlash `hipfire run` without a device compiler; other models and
  routes can still JIT, so running models generally needs hipcc. The selected
  ROCm root should therefore also provide `include/hip/hip_runtime.h` and
  `bin/hipcc`. Install a supported AMD ROCm HIP runtime, development headers, and device
  compiler via
  [AMD's live install selector](https://rocm.docs.amd.com/en/latest/install/rocm.html)
  (choose packages for your GPU, OS, and ROCm version — package names drift;
  the selector is authoritative).
- **Supported ROCm range:** Linux with **ROCm 6 or newer** (project baseline from
  [README.md](../README.md)). **ROCm 6.4+** for RDNA4 (`gfx1200`/`gfx1201`);
  **ROCm 7.2+** for Strix Halo / gfx115x. hipfire's path resolver does not
  hardcode a required release — install a supported stack for your GPU.
- **Windows (untested on GPU hardware):** nothing in hipfire has been run on a
  Windows GPU yet, on either route; see [Windows status](#windows--status).
  - **WSL2 with ROCm 7.2.1 and AMD's ROCDXG library:** hipfire is the ordinary
    Linux build inside the distro; see [Windows — WSL2](#windows--wsl2) for the
    driver, ROCm and `librocdxg` steps and the supported GPU list.
  - **Native:** [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) (`hipcc` + `amdhip64.dll`); see [the limits](#windows--native).
- Disk space for models under `~/.hipfire/models/` (a few GB for small tags; tens of GB for 27B+).

Live model tags, VRAM floors, and formats: [MODELS.md](MODELS.md). Full env list: [env-vars.md](env-vars.md).

For a non-default or side-by-side install, pin one coherent SDK root before
starting hipfire:

```bash
export HIPFIRE_ROCM_PATH=/absolute/path/to/rocm
# If HIPFIRE_ROCM_PATH is unset, ROCM_PATH then HIP_PATH are accepted:
# export ROCM_PATH=/absolute/path/to/rocm
# export HIP_PATH=/absolute/path/to/rocm   # or .../hip (normalized to the parent)
```

Priority is `HIPFIRE_ROCM_PATH` > `ROCM_PATH` > `HIP_PATH`. An explicit override
is authoritative: once a root is selected, HIP/HSA libraries, headers, and
`hipcc` stay in that root family — hipfire will not fall back to another install
or a bare soname. If several complete roots are equally eligible (for example
multiple `/opt/rocm-*` with no active `/opt/rocm`), discovery refuses to guess;
set `HIPFIRE_ROCM_PATH` to one absolute root.

## Install

### Linux — master or beta in one command

The revision selector controls the managed source checkout under
`~/.hipfire/src`; no installer editing is needed:

```bash
# Current master:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh | bash

# Integration/testing branch:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh \
  | bash -s -- --branch beta
```

Branch installs remain on that branch when `hipfire update` is run without a
selector. The equivalent generic form is `--ref beta`; it auto-detects a
branch, tag, or commit. `HIPFIRE_INSTALL_REF=beta` is available for automation.

Both `master` and `beta` are mutable. For a reproducible install, pin and
inspect the installer itself, then ask it to install the same tag or commit:

```bash
PIN=v0.4.5
curl -fsSL "https://raw.githubusercontent.com/warpfront/hipfire/${PIN}/scripts/install.sh" \
  -o /tmp/hipfire-install.sh
sha256sum /tmp/hipfire-install.sh
less /tmp/hipfire-install.sh
bash /tmp/hipfire-install.sh --ref "$PIN"
```

Use `--tag v0.4.5` when the kind is known, or `--commit <full-sha>` for an
immutable commit. Fetching a pinned script but omitting the selector installs
`master`, so keep the two pins together.

The installer detects GPU arch and ROCm and builds the daemon and native CLI.
With `--tag` (or a tag-named `--ref`) on an admitted GPU it then installs the
tag's [prebuilt kernel pack](#prebuilt-kernel-packs); otherwise, or when the
pack cannot be used, it packages exact-source registry kernels with
`hipfire-kernel-pack`'s shared compiler. Either way the result is
`~/.hipfire/bin/kernels/compiled/<arch>/`, where each `.hsaco` requires a
matching `.index.json`; re-run the installer after upgrading an older install.
The bin directory can be added to `PATH`; reload the shell afterward.

GPUs not in the admitted registry remain JIT-only and need hipcc at runtime;
they cannot run with `HIPFIRE_NO_DEVICE_COMPILER=1`.

Other `install.sh` options (`bash install.sh --help` lists them all; the ones
marked "setup" are forwarded to `hipfire setup`):

| Option | Effect |
|---|---|
| `--branch NAME`, `--ref REF`, `--tag TAG`, `--commit SHA` | Source revision; at most one. `HIPFIRE_INSTALL_REF` is a generic `--ref` for automation |
| `--compile-kernels`, `--kernel-pack-url URL` | Skip or relocate the [prebuilt kernel pack](#prebuilt-kernel-packs) (setup) |
| `--rocm-root PATH`, `--hipcc PATH`, `--strict-rocm` | Pin the ROCm root, the device compiler, or require the compiler under the selected root (setup) |
| `--gpu-arch ARCH` | Override the detected GPU architecture (setup) |
| `--profile auto\|hip\|redline` | Replay profile; `auto` leaves `replay.backend=auto` (setup) |
| `--yes`, `-y`, `--non-interactive` | Do not prompt, for example for the source channel |

### Prebuilt kernel packs

Each release tag carries one pack per admitted architecture, built from the
tag's exact commit by `scripts/build-kernel-pack.sh` (hipcc, no GPU):

| Release asset | Contents |
|---|---|
| `hipfire-kernels-<tag>-<arch>.tar.gz` | `manifest.json` and `<arch>/` with every registry module's `.hsaco`, `.hash` and `.index.json` |
| `hipfire-kernels-<tag>-<arch>.tar.gz.sha256` | `sha256sum` line for the tarball |
| `hipfire-kernels-<tag>-<arch>.manifest.json` | copy of the manifest: commit, ROCm and HIP version, compiler identity, code-object version, admitted ROCm range |

`install.sh --tag <tag>` (also `--ref <tag>` and `hipfire update --tag <tag>`)
downloads the pack for the detected arch and installs it only when all of
these hold; otherwise it says why and compiles locally with hipcc:

- the tarball matches the published SHA-256, and holds only flat regular files;
- the manifest names the same tag, arch and commit as the checkout, and the
  runtime's kernel cache ABI;
- the local ROCm version (`<root>/.info/version`) is inside the manifest's
  range — by default the build's `major.minor` up to the next major release;
- a local hipcc, if any, is the same build that compiled the pack (a different
  one would make the runtime reject and recompile every kernel), and the
  active hipfire config selects the same kernel flags;
- every registry module's index matches the checked-out sources, recipe and
  object bytes, and the directory holds nothing else.

Pass `--compile-kernels` to skip the pack. `--kernel-pack-url` points at another
directory or URL holding the assets (`https://…`, `file:///…`, or a plain
path), for mirrors and offline installs:

```bash
bash install.sh --tag v0.4.5 --kernel-pack-url file:///srv/hipfire-packs/v0.4.5
```

`hipfire kernel-pack install --tag <tag> --arch <arch> --source ~/.hipfire/src`
runs the same download and checks on an existing install, for example to
restore kernels on a machine without hipcc.

#### Hosts outside the pack's ROCm range (Ubuntu amdrocm 10.0)

Ubuntu's `amdrocm` 10.0 packages are the common host a pack's `[10.1, 11.0)`
range excludes. Nothing about the model runtime differs on such a host: the
same kernels compile locally through its own hipcc.

- On `beta` and from 0.4.4, `install.sh --tag <tag>` (and
  `hipfire update --tag <tag>`) handles this on its own. When the pack is
  skipped for the range, setup selects the umbrella `/opt/rocm` root
  automatically (the amdrocm packaging installs it alongside versioned
  children like `/opt/rocm/core-10.0` and a compiler-only `/usr`), finds
  `bin/hipcc` there, and compiles.
- On v0.4.3.x the automatic selection is absent (#967), and the amdrocm layout
  makes the roots ambiguous enough that naming them explicitly is the reliable
  path: run the installer once to fetch the source (it stops at the pack
  check), then complete with setup directly:

```bash
curl -fsSL https://hipfire.dev/install.sh | bash -s -- --tag v0.4.3.1 --yes   # stops at the pack check
export PATH="$HOME/.cargo/bin:$PATH"                                          # cargo is required
hipfire setup --source ~/.hipfire/src \
  --rocm-root /opt/rocm --hipcc /opt/rocm/bin/hipcc \
  --yes --compile-kernels
```

`~/.cargo/bin` must be on PATH for non-interactive shells (`rustup` installs
it; the bootstrap checks cargo up front, a direct `hipfire setup` reaches the
daemon build without checking). Verified on a live amdrocm `10.0.0-4` host
(gfx1151, issues #967 and #966): every kernel compiles through the host hipcc
in a few minutes, the runtime verifies all indexes at load, and the Flash-Next
NPU trunk JIT-compiles its object when the certified pack does not match the
local toolchain.

Maintainers: the tag-triggered `release-kernel-packs` workflow
(`.github/workflows/release.yml`) builds every arch in a ROCm dev image and
attaches the assets to the release. The default image is
`docker.io/rocm/dev-ubuntu-26.04` (image tag `10.1.0-full`, pinned by the digest
`sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce`; the
full reference is `<image>:<tag>@<digest>`)
(ROCm 10.1.0 packages `10.1.0-3`, HIP 7.16.26385, clang 24), so its packs admit
ROCm [10.1, 11.0). A pack only installs where the local hipcc, if any, is the
build that compiled it: a host whose hipcc differs (for example ROCm 10.0)
compiles locally, while a host with no compiler accepts an admitted pack.
Repository variables choose another image
(`HIPFIRE_ROCM_IMAGE`) and, optionally, a wider admitted range
(`HIPFIRE_PACK_ROCM_MIN` and `HIPFIRE_PACK_ROCM_MAX_EXCLUSIVE`, set both). To
publish from a local toolchain instead:

```bash
scripts/build-kernel-pack.sh --tag v0.4.5 --out dist            # every admitted arch
scripts/build-kernel-pack.sh --tag v0.4.5 --out dist gfx1201    # or a subset
gh release upload v0.4.5 dist/hipfire-kernels-v0.4.5-*
```

The script compiles `git archive` of the tag's commit with a fresh `HOME` and
no `HIPFIRE_*` feature overrides, verifies every index against that toolchain,
and writes byte-reproducible tarballs. `--rocm-min X.Y --rocm-max-exclusive X.Y`
declares a range other than the default; it must contain the build's ROCm.

### Windows — status

> **Native Windows and WSL2 are both untested on GPU hardware.** The only
> Windows CI job is `Windows MSVC (no GPU)` on a hosted runner without
> a GPU, and the hardware gate runs on Linux only. Neither route is certified
> until a community test report lands; the notes below describe what the code
> does, not what has been observed on a card. To help, see
> [Help test Windows](#help-test-windows).

### Windows — WSL2

Inside the distro hipfire is the Linux build (Linux installer, `hipfire update`,
file locks, signals);
ROCm reaches the GPU through AMD's ROCDXG library and Microsoft's DXCore
device `/dev/dxg`, while the Windows Adrenalin driver owns the card. AMD's
current method (ROCm 7.2.1, ROCDXG; the older `amdgpu-install --usecase=wsl`
/ roc4wsl packaging stops at ROCm 7.2) is documented in
[WSL How-to — Use ROCm on Radeon](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html),
[WSL How-to — Use ROCm on Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html)
and the [librocdxg Quickstart](https://github.com/ROCm/librocdxg#quickstart):

1. On Windows 11, install **AMD Software: Adrenalin Edition 26.2.2 for
   WSL2** ([release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html)).
   It is the first driver with Ryzen Strix / Strix Halo support on WSL; the
   Radeon [WSL support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
   lists 26.1.1 as the minimum for discrete cards.
2. Install WSL2 with **Ubuntu 24.04 or 22.04**
   ([Microsoft: install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)).
3. In the distro, install the **ROCm 7.2.1** packages with the
   [ROCm Linux quick start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).
   Install ROCm userspace only; there is no Linux kernel driver to install
   under WSL.
4. Install **librocdxg** (1.2.x for ROCm 7.2.x): the prebuilt
   `rocdxg-roct_<version>_amd64.deb` from the
   [librocdxg releases](https://github.com/ROCm/librocdxg/releases)
   (`sudo dpkg -i rocdxg-roct_<version>_amd64.deb`), or build it from source
   as the Quickstart describes. Its source now lives in
   [rocm-systems](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime/libhsakmt/src/dxg).
5. ROCm releases before 7.13 (so 7.2.1) need DXG detection switched on. Add it
   to your shell profile so the hipfire daemon inherits it:

   ```bash
   echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc && source ~/.bashrc
   ```
6. Check that `rocminfo` lists your GPU as an agent (`Name: gfx1201`, …),
   then run the [Linux installer](#linux--master-or-beta-in-one-command)
   inside the distro.

Supported GPUs for ROCm 7.2.x on WSL (librocdxg 1.2.0 matrix; hipfire's tuned
targets are gfx1201, gfx1100 and gfx1151):

| arch | products |
|---|---|
| gfx1201 | Radeon AI PRO R9700, RX 9070, RX 9070 XT, RX 9070 GRE |
| gfx1200 | RX 9060, RX 9060 XT |
| gfx1100 | RX 7900 XTX, RX 7900 XT, RX 7900 GRE, PRO W7900 (incl. Dual Slot), PRO W7800 (incl. 48GB) |
| gfx1101 | RX 7800 XT, PRO W7700 |
| gfx1151 | Ryzen AI Max+ 395, Ryzen AI Max 390, Ryzen AI Max 385 (Strix Halo) |
| gfx1150 | Ryzen AI 9 HX 375, Ryzen AI 9 HX 370, Ryzen AI 9 365 |

What differs from native Linux until each piece is certified on WSL hardware:

- The daemon identifies cards from HIP's UUID and PCI address (there is no
  KFD topology), so `hardware.devices` is refused; select cards with
  `HIP_VISIBLE_DEVICES`. AMD does not support multi-GPU under WSL.
- The retained Redline PM4 default falls back to the HIP graph with one
  `[redline] retained default refused` log line (historically measured on Linux
  about 4–6% slower decode on the routes that default to PM4), and an explicit `replay.backend=redline`
  is refused. `HIPFIRE_UNSAFE_WSL_REDLINE=1` lifts this; it is **unsafe
  until certified**.
- The KV cache uses the legacy backend; an explicit `kv_backend=vmm` is
  refused. `HIPFIRE_UNSAFE_WSL_VMM_KV=1` lifts this; it is **unsafe until
  certified** (WDDM VA growth may alias earlier KV pages, as it does on native
  Windows).
- Host RAM guards see the WSL VM's memory. On Strix Halo the GPU pool is also
  bounded by the `.wslconfig` `memory=` setting
  ([ROCm#6022](https://github.com/ROCm/ROCm/issues/6022)); raise it for large
  models. `rocm-smi`/`amd-smi` are limited and the ROCm profiler and debugger
  are unsupported under WSL.

### Windows — native

```powershell
# Current master:
iex (irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1)

# Integration/testing branch:
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1))) `
  -Branch beta
```

For a reviewed, pinned installation of the latest release (recommended for
testers: a release install uses the release's prebuilt kernel pack instead of a
long local kernel compile):

```powershell
$Pin = (Invoke-RestMethod https://api.github.com/repos/warpfront/hipfire/releases/latest).tag_name
$Pin
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1" `
  -OutFile "$env:TEMP\hipfire-install.ps1"
Get-FileHash "$env:TEMP\hipfire-install.ps1" -Algorithm SHA256
notepad "$env:TEMP\hipfire-install.ps1"
& "$env:TEMP\hipfire-install.ps1" -Tag $Pin
```

The installer script is downloaded from `master` rather than from the tag:
`scripts/install.ps1` in every release from `v0.4.1` through `v0.4.3.1` starts with a doubled UTF-8
byte-order mark, which prevents `-Tag`, `-Ref`, `-Branch`, `-Commit`,
`-CompileKernels` and `-KernelPackUrl` from binding (#979). Releases up to
`v0.4.3.1` also predate Windows kernel-pack acceptance and the HIP DLL
selection below, so on Windows they compile kernels locally.
PowerShell also accepts `-Branch beta` and `-Commit <sha>`.
On Windows, `hipfire update` reports the installed version and exits non-zero
without changing files: Windows updates require re-running `scripts/install.ps1`
after stopping the engine. For current master:

```powershell
iex (irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1)
```

Use the reviewed, pinned command above when selecting a specific revision.

Builds `daemon.exe` and the native CLI from the same selected source revision
under `~\.hipfire\src`. A release install (`-Tag vX.Y.Z`, a tag-named `-Ref`, or
a branch/ref whose checked-out HEAD has a release tag pointing at it) on an
admitted GPU runs `hipfire kernel-pack install` for that release's
[prebuilt kernel pack](#prebuilt-kernel-packs) (`-KernelPackUrl` overrides the
download location, `-CompileKernels` skips the pack and forces a local compile).
The pack objects are HSA code objects built on Linux; whether the Windows HIP
runtime loads them is unverified until a test report lands. Otherwise, or when
the pack cannot be used, hipcc from the HIP SDK packages exact registry sources:
the installer runs `daemon.exe --precompile` to produce indexed packages in
`~\.hipfire\bin\kernels\compiled\<arch>\`. Re-run `install.ps1` after
upgrading: copying bare `.hsaco` files is insufficient.

Unsupported GPU architectures likewise require hipcc JIT at runtime; the
installer does not copy bare checkout objects into the installed cache.

Native Windows has not been run on GPU hardware (see [Windows status](#windows--status)).
The limits below are properties of the code, not observations from a card:

- **No Redline PM4.** The Windows HIP SDK ships no ROCr (`libhsa-runtime64`),
  so decode runs on the HIP graph. On Linux the HIP graph has been measured
  about 4–6% slower than the PM4 default (one gfx1201 Qwen3.8 27B tg128 at 8K
  measurement, ≈38.2 instead of 40.3 tok/s; a Linux number, not a current
  baseline, and not measured on Windows).
  An explicit `replay.backend=redline` is refused.
- **Legacy KV only.** HIP VMM growth is not certified on Windows; automatic
  KV selects legacy and `kv_backend=vmm` is refused.
- **No tensor or pipeline parallel**: the HIP SDK has no RCCL. One GPU per
  daemon; `hardware.devices` is unavailable, so select the card with
  `HIP_VISIBLE_DEVICES`.
- **HIP runtime DLL selection.** hipfire loads `amdhip64*.dll` by absolute path
  from the selected HIP SDK root (`HIP_PATH`, `HIPFIRE_ROCM_PATH`, `ROCM_PATH`,
  or `Program Files\AMD\ROCm`), searching the DLL's own directory first for its
  dependencies, so the older copy the AMD driver installs in `System32` cannot
  shadow the SDK's. With no SDK root it tries hipfire's exe directory, then
  `~\.hipfire\runtime`, then the standard DLL search. The daemon logs the
  resolved DLL path and the HIP runtime/driver versions at start, and
  `hipfire diag --json` reports them under `live_gpu`
  (`hip_runtime_library`, `hip_runtime_version_raw`, `hip_driver_version_raw`).
- **Model file mapping.** On unified-memory GPUs (Strix Halo) the model file
  mapping is released after the headers are read, as on Linux; discrete GPUs
  keep it on both OSes.
- GPU locks are `LockFileEx` files under `%ProgramData%\hipfire\locks`
  (or `HIPFIRE_LOCK_DIR`); a second daemon on the same card reports the
  holder PID, as on Linux.
- hipfire uses `HIPFIRE_HOME` when set; otherwise `%USERPROFILE%\.hipfire`,
  then `%HOME%\.hipfire`. Windows requires an absolute home path and reports an
  actionable error rather than creating a home in the current directory.
- `hipfire stop` authenticates to a separate loopback management listener and
  waits for graceful shutdown. Older `serve.pid` records lack the management
  credential: stop those servers with Ctrl-C, then restart with the updated CLI.
  `stop --force` / `--all` never kill processes whose install ownership cannot be
  proven; Windows cannot prove ownership from another process's environment.
- `hipfire update` does not replace a running Windows executable; re-run
  `scripts/install.ps1` as described above.

#### CI developer artifact

The required `Windows MSVC (no GPU)` job in
[CI](../.github/workflows/ci.yml) builds the native CLI and daemon, tests the
config/registry/client libraries, checks that `scripts/*.ps1` parse and that
`install.ps1`'s parameters bind, checks `hipfire --version` and parseable
`hipfire diag --json` output without an AMD HIP runtime, and lists the DLLs the
packaged binaries import against the allowlist in `scripts/check-pe-imports.ps1`.
It is promoted to a
required check after its first green run on `beta`; adding the job is not
evidence of a passing Windows run or GPU certification.

After successful packaging, that run's `hipfire-windows-msvc` artifact contains
`hipfire-<version>-x86_64-pc-windows-msvc.zip` and its `.zip.sha256` checksum.
The ZIP includes `hipfire.exe`, `daemon.exe`, licenses/NOTICE,
`README-windows.txt`, and a provenance/checksum `manifest.json`. To package
already-built release binaries locally, use
`.\scripts\package-windows.ps1` (defaults: `target\release` to `dist`).
This is an **unsigned developer artifact**, not an installer or signed update:
no HIP runtime, driver, or kernel packs are bundled. A user-installed AMD HIP
runtime is required; runtime tuple certification and signing ownership remain
pending M0. Checksums detect corruption, not publisher authenticity. These
artifacts do not change the installation instructions above.

The Windows build does not imply support for Unix-only PLE conversion,
hot-row/direct-read or hot-pack I/O, cache-residency probing, or prewarm paths;
those I/O entry points report unsupported on Windows. POSIX-FD VMM sharing,
Redline, and VMM KV are not native Windows capabilities.

### Help test Windows

No maintainer has a Windows GPU machine, so native Windows (and WSL2) stay
unverified until someone runs them on a card. A one-script test kit and an issue
form exist for that.

Native Windows:

1. Install a release as shown [above](#windows--native) (the pinned `-Tag`
   install).
2. Run the kit (the installer clones the source into `~\.hipfire\src`):

   ```powershell
   powershell -ExecutionPolicy Bypass -File $env:USERPROFILE\.hipfire\src\scripts\windows-check.ps1
   ```

   `-ExecutionPolicy Bypass` on the command line applies to that one process
   only; it does not change the machine or user execution policy.
3. File the result with the
   [Windows test report form](https://github.com/warpfront/hipfire/issues/new?template=windows-test-report.yml),
   including the generated `report.txt` and `report.json`.

[`scripts/windows-check.ps1`](../scripts/windows-check.ps1) collects the OS,
driver, HIP SDK, `amdhip64.dll` resolution and GPU architecture; pulls the
smallest Qwen3.5 text model; runs a short greedy generation and a short
`hipfire bench`; and, with `-Include27B` when VRAM is at least 24 GB, also tries
a 27B model. It writes `report.json` and `report.txt` and does not change system
settings. Parameters: `-OutDir`, `-SkipModel`, `-Include27B`, `-HipfireExe`
(see the parameter block at the top of the script). The model download uses
disk space and network under `~\.hipfire\models`.

WSL2: the kit is a PowerShell script for the native install. For a WSL2 report,
use the same issue form and include `hipfire diag --json` and `rocminfo` output
from inside the distro.

### Source checkout

```bash
git clone https://github.com/warpfront/hipfire
cd hipfire
cargo build --release -p hipfire-daemon
cargo build --release -p hipfire-cli
cargo build --release -p hipfire-quantize
# optional TUI:
cargo build --release -p hipfire-tui
./scripts/install.sh   # from a checkout: local mode wires CLI + PATH
```

Other packaging: [NIXOS.md](NIXOS.md), [CONTAINER.md](CONTAINER.md).

## Uninstall a managed Linux install

The default uninstall removes the installed binaries, kernels, clean managed
source checkout, runtime PID/log files, and the PATH entry created by the
installer. It preserves downloaded models and settings under `~/.hipfire`:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh | bash
```

Preview without changing anything:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh \
  | bash -s -- --dry-run
```

Use `--purge` only when models, configuration, and every other file under
`~/.hipfire` should also be deleted. The script asks for an explicit
confirmation; automation can add `--yes`. It does not remove ROCm, Rust, or
other shared system dependencies.

## Verify

```bash
hipfire --version
hipfire version
hipfire diag
hipfire doctor
```

`--version` prints the release, source commit, and source ref in one line.
`hipfire version` additionally reports whether the managed checkout still
matches the binary and hashes the installed daemon; add `--json` for support
reports. `diag` reports GPU arch, VRAM, HIP/ROCm, kernel locations, model dir,
and config overrides.

On Linux, run `hipfire doctor` before the first model load: it checks the amdgpu
driver, `/dev/kfd` access, the ROCm install and your GPU's device libraries, any
`HSA_OVERRIDE_GFX_VERSION`, the kernel pack, and that the CLI and the daemon are
one build, then prints one `fix:` line per problem
([details](CLI.md#hipfire-doctor-runtime-preflight)).

- Before pulling a large model, run `hipfire fit <tag> --ctx N`: it reports
  whether the model loads at that context on this machine (no GPU work, no
  daemon), what the load would choose, and what to change when it does not
  ([details](CLI.md#hipfire-fit)).

### Update or switch channels on Linux

```bash
hipfire update                  # advance the currently selected branch
hipfire update @beta            # auto-detect branch/tag/commit
hipfire update --branch master
hipfire update --tag v0.4.5
hipfire update --commit <sha>
```

Tags and commits are detached, immutable pins; a later selector is required to
move away from them. Before switching, the updater retains the previous commit
under `refs/hipfire/backups/` and stashes dirty files with a recoverable
`hipfire-update-*` label.

## First inference

```bash
hipfire pull qwen3.5:4b
hipfire run  qwen3.5:4b "Explain FFT in one line"
```

Which tag fits your card: [PICK-A-MODEL.md](PICK-A-MODEL.md).

- `pull` downloads the registry artifact into `~/.hipfire/models/` (and published sidecars when the registry entry lists them).
- `run` accepts a registry tag, alias, or path. **Recognized registry tags that are not local yet are auto-pulled** (can be multi-GB). Unresolved tags, aliases, or paths error with a `pull` / `list --remote` hint — they do not download.
- Cold start loads weights and may JIT kernels; later calls are faster if a daemon is already up.

Interactive multi-turn:

```bash
hipfire chat qwen3.5:4b
```

See [CHAT.md](CHAT.md).

## Keep a daemon warm

```bash
hipfire serve qwen3.5:4b -d    # background; OpenAI-compatible HTTP
hipfire run qwen3.5:4b "..."   # reuses serve when healthy
hipfire stop                   # graceful stop of the tracked daemon
```

Defaults (overridable in config): bind **`127.0.0.1:11435`** (loopback only), pre-warm **`default_model`** (`qwen3.5:9b` unless you set another). HTTP surface: [SERVE.md](SERVE.md). Subcommand flags: [CLI.md](CLI.md).

`hipfire serve --ui` also serves the embedded chat UI at `/ui` on the same listener (`--open` launches it in the default browser and implies `--ui`).

> **Key optional, TLS absent:** the serve HTTP API has **no TLS**, and bearer authentication is **optional** (`serve.api_key_file` / `--api-key-file`; see [SERVE.md](SERVE.md#security-bearer-authentication-and-https-proxies)). The default bind `127.0.0.1` accepts connections from this machine only. `hipfire config set host 0.0.0.0` (or `hipfire serve 0.0.0.0 11435`) listens on all interfaces and exposes inference to any reachable network (including chat-spawned serves); without a key anyone who can reach it can call every endpoint, so a key is **recommended** on any non-loopback bind. Expose beyond localhost only on a trusted/firewalled network **or** behind a TLS-terminating reverse proxy you control — never publish the raw port to the internet.

Force a one-shot daemon and skip HTTP:

```bash
HIPFIRE_LOCAL=1 hipfire run qwen3.5:4b "..."
```

## Light configuration

```bash
hipfire config                                      # global TUI → ~/.hipfire/config.toml
hipfire config qwen3.5:9b                           # resolved per-model policy
hipfire config qwen3.5:9b set generation.temperature 0.7
hipfire config qwen3.5:9b list                      # overlay + provenance
```

Defaults that matter on day one (from the native schema; full table in [CONFIG.md](CONFIG.md)). **Sampling send path:** `run` / `serve` transmit explicit request/CLI values, per-model TOML overlays, or the complete registry `recommended_settings` recipe (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty`, plus fallback `system_prompt`). Otherwise sampling fields are omitted for daemon/HFQ/arch fallback. Bare global sampling values alone are **not** effective `run`/`serve` defaults. **Chat** is the exception — it uses a global config snapshot for the session ([CHAT.md](CHAT.md)).

| Key | Default | Note |
|---|---|---|
| `temperature` | `0.3` | Stored global default only for run/serve send (see above); Chat session seed |
| `max_tokens` | `4096` | Per-request generation cap for `run` / API fallback |
| `kv_cache` | `auto` | Resolves via a registry non-q8 `default_kv_mode`, else the architecture default: Qwen native `fp8` on eligible exact gfx1201, `q8` on gfx1100 / gfx1151 and elsewhere. `fwht3` is an optional headroom mode |
| `dflash_mode` | **`off`** (schema) | DFlash is opt-in; pulling a draft does not enable it. Exception: Qwen3.8-27B tiers with a registry-declared draft (`qwen3.8:27b`, `-mq3*`, `-mq4*`) resolve `auto` through the registry layer, so the pulled draft is used there; if the draft file is absent the load keeps MTP `auto` (AR for this Qwen35 producer: it declares no measured AR floor, so only `--spec mtp` arms its head). `hipfire config set dflash_mode off` is the kill switch ([CONFIG.md](CONFIG.md#qwen38-27b-paired-dflash-default)). |
| `speculation` | `auto` | Mechanism selector; DFlash stays off when `dflash_mode=off`. Native **MTP** under `auto` arms only a producer that declares a measured AR floor (today only Flash-Next / Qwen4); Qwen3.5-class and DeepSeek V4 heads stay AR until `--spec mtp` opts in. An eligible **DSpark** sidecar may still activate under `auto`. Use `speculation=off` to force plain AR. |
| `thinking` | `on` | Reasoning models may emit `<think>`; display strip is CLI/API-side |
| `host` / `port` | `127.0.0.1` / `11435` | Serve bind (no TLS; key optional, recommended off loopback) |

Enable draft-model speculation only when you intend to:

```bash
hipfire pull qwen3.5:9b
hipfire pull qwen3.5:9b-draft
hipfire config set dflash_mode auto    # or on / per-model
```

`auto` uses a pulled draft when present; `on` fails the load without it. Qwen3.8-27B paired tiers already resolve `auto` from the registry, so no `config set` is needed there.
`developer.dflash_draft` / `HIPFIRE_DFLASH_DRAFT` override the registry sidecar.
Several tags can share one draft file — `hipfire rm` keeps that sidecar while
another installed target still declares it (see [CLI.md](CLI.md)).

## Long context (optional)

Long context needs no extra setup beyond `max_seq` and VRAM; for KV headroom
pick a compact `kv_cache` mode (see [CONFIG.md](CONFIG.md)). CASK/TriAttention
eviction is deprecated and will be removed in 0.5.0 — off by default, not
supported, and not a recommended route to long context.

**YaRN context extension (since 0.4.2, Qwen3.8-Flash-Next only).** The engine can
apply YaRN RoPE scaling at load through `attention.yarn_factor` (env
`HIPFIRE_YARN_FACTOR`): unset/`null` keeps the checkpoint's own
`rope_parameters`, a value above 1 extends the admitted context to
`factor × original` positions, and `1` disables a checkpoint's YaRN. It is
static scaling, so it also changes outputs at short context: set it only for
long-context loads. Other families ignore it.

```bash
hipfire config qwen3.8:flash-next set attention.yarn_factor 4
```

Admitted `max_seq` and measured reach: [CONFIG.md](CONFIG.md#attention).

### Measured capacity (Qwen3.5/3.6 35B-A3B-class, 24GB)

On 24GB GPUs, Q8 KV is about 10,880 B/token across 10 full-attention layers,
plus O(N) flash partials (~2,064 B/token), ~25 MiB DeltaNet state, and multi-GiB
fixed HIP/graph overhead. **50K Q8 is tight but physically feasible**; **200K Q8
is not a 24GB-class configuration** — it needs >32GB-class VRAM or compressed
KV. Some historical 131K sidecar benches clamped physical capacity to ~2432
tokens and should not be read as full-context Q8 support. Long-context decode
slowdown is expected O(N) full-attention bandwidth, not by itself an admission
regression.

## Flash-Next multi-device and NPU defaults

These behaviors shipped in 0.4.2 and are in every later release.

- **Strix Halo NPU trunk (gfx1151).** `fusions.npu_trunk` /
  `HIPFIRE_QWEN4_NPU_TRUNK` (`off|auto|on|emulate`) defaults to `auto`. Auto
  arms only with an open XDNA2 NPU and a matching calibration; otherwise it
  logs one reason and runs GPU-only. Every other architecture resolves auto
  silently to off. `off` is the kill switch; timings are fixture-bound and live
  in the CHANGELOG, not here. Keys: [CONFIG.md](CONFIG.md).
- **Qwen3.8-Flash-Next PLE residency.** The artifact embeds its PLE rows as
  Q8 (54.4 GB); `hipfire ple derive --format q4 <model>` writes a `.ple-q4`
  sidecar beside it that the load picks up automatically, dropping those rows
  to 28.8 GB. Measured on a gfx1151 host (Ryzen AI Max+ PRO 495, 192 GB):
  25.6 GB less residency, the load admits at 154.7 GB free, and generation
  stays coherent - but numerics change (a greedy paragraph differs from the
  embedded-q8 run), so treat the sidecar as a capacity lever for 128 GB-class
  hosts, not a free default.
- **EP owner-routed prefill and expert staging.** With `topology.ep` 2..=8,
  `HIPFIRE_EP_OWNER_ROUTED_PREFILL` defaults on when every admission condition
  holds (exact gfx1201 on the symmetric IU4 route, exact gfx1100 on the F16
  WMMA grouped route); `HIPFIRE_EP_EXPERT_STAGE` defaults on only on gfx1201.
  The exact value `0` disables each. Single-card runs are unchanged. Setup and
  gates: [multi-gpu.md](multi-gpu.md); variable semantics:
  [env-vars.md](env-vars.md).

## If something fails

| Symptom | What to try |
|---|---|
| `hipfire: command not found` | Reload shell; ensure install dir is on `PATH` |
| HIP / `/dev/kfd` / arch errors | `hipfire diag`; match ROCm/HIP version to arch (above) |
| Model not found | `hipfire list` / `hipfire list -r`; `hipfire pull <tag>` |
| Port in use / stale serve | `hipfire ps`; `hipfire stop --force`; check `~/.hipfire/serve.pid` and (detached only) `serve.log` |
| Draft pulled but no speedup | Expected for most models: the schema `dflash_mode` default is **off** (Qwen3.8-27B paired tiers resolve `auto` from the registry; check `hipfire config <tag>` and that the draft file is present; with the draft absent the load runs AR, because native MTP `auto` arms only Flash-Next, and `--spec mtp` is the opt-in for the 27B MTP head) |
| Truncated answers, or empty `content` from an OpenAI client, on thinking models | `max_tokens` covers the `<think>` span too: raise `max_tokens` / `thinking_budget`, or send `"chat_template_kwargs": {"enable_thinking": false}` (see [SERVE.md](SERVE.md)); [MODELS.md](MODELS.md) thinking section |

```bash
hipfire diag
tail -f ~/.hipfire/serve.log
```

## What to read next

| Doc | When |
|---|---|
| [PICK-A-MODEL.md](PICK-A-MODEL.md) | Which tag and speculation setting for your VRAM |
| [CLI.md](CLI.md) | Every subcommand and flag surface |
| [CHAT.md](CHAT.md) | Interactive chat, thinking display, daemon attach |
| [SERVE.md](SERVE.md) | OpenAI-compatible HTTP |
| [MODELS.md](MODELS.md) | Tags, VRAM, BYO quantize, thinking/templates |
| [multi-gpu.md](multi-gpu.md) | Tensor / expert parallel setup and the Flash-Next EP2 route |
| [env-vars.md](env-vars.md) | Environment variables (developer/diagnostic levers) |
| [CONFIG.md](CONFIG.md) | All config keys |
| [QUANTIZE.md](QUANTIZE.md) | `hipfire quantize` operator guide |
| [INDEX.md](INDEX.md) | Ownership map for the rest of `docs/` |
