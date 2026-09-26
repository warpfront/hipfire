﻿# hipfire installer for Windows — detects GPU, builds binaries + indexed kernels.
# Usage: irm https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.ps1 | iex
param(
    [string]$Ref,
    [string]$Branch,
    [string]$Tag,
    [string]$Commit
)

$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 turns native stderr into a terminating error under
# EAP=Stop, and git prints its normal progress (fetch etc.) on stderr. Scope
# the relaxation to git invocations via this wrapper instead of weakening the
# whole script; real failures are still caught by explicit $LASTEXITCODE
# checks and try/catch at every call site.
function Invoke-Git {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & git @args
    } finally {
        $ErrorActionPreference = $prev
    }
}

# ─── Paths ───────────────────────────────────────────────
$HipfireDir  = "$env:USERPROFILE\.hipfire"
$BinDir      = "$HipfireDir\bin"
$RuntimeDir  = "$HipfireDir\runtime"
$ModelsDir   = "$HipfireDir\models"
$SrcDir      = "$HipfireDir\src"

# ─── Constants ───────────────────────────────────────────
$GithubRepo = "warpfront/hipfire"
$InstallRef = if ($env:HIPFIRE_INSTALL_REF) { $env:HIPFIRE_INSTALL_REF } else { "master" }
$InstallRefKind = if ($env:HIPFIRE_INSTALL_REF) { "auto" } else { "branch" }
$Selectors = @(
    if ($PSBoundParameters.ContainsKey("Ref")) { @{ Value = $Ref; Kind = "auto" } }
    if ($PSBoundParameters.ContainsKey("Branch")) { @{ Value = $Branch; Kind = "branch" } }
    if ($PSBoundParameters.ContainsKey("Tag")) { @{ Value = $Tag; Kind = "tag" } }
    if ($PSBoundParameters.ContainsKey("Commit")) { @{ Value = $Commit; Kind = "commit" } }
)
if ($Selectors.Count -gt 1) {
    throw "Choose only one -Ref, -Branch, -Tag, or -Commit."
}
if ($Selectors.Count -eq 1) {
    $InstallRef = [string]$Selectors[0].Value
    $InstallRefKind = [string]$Selectors[0].Kind
}
$InstallRef = $InstallRef.Trim().TrimStart("@")
if ($InstallRef.StartsWith("refs/heads/")) {
    $InstallRef = $InstallRef.Substring(11)
    $InstallRefKind = "branch"
} elseif ($InstallRef.StartsWith("refs/tags/")) {
    $InstallRef = $InstallRef.Substring(10)
    $InstallRefKind = "tag"
} elseif ($InstallRef.StartsWith("origin/")) {
    $InstallRef = $InstallRef.Substring(7)
    if ($InstallRefKind -eq "auto") { $InstallRefKind = "branch" }
}
if (
    [string]::IsNullOrWhiteSpace($InstallRef) -or
    $InstallRef -match '^[./-]|[./]$|\.\.|@\{|//|[\s\\:\?\*\[\]\^~]'
) {
    throw "Unsafe or invalid git revision '$InstallRef'."
}
if ($InstallRefKind -eq "commit" -and $InstallRef -notmatch '^[0-9a-fA-F]{7,40}$') {
    throw "-Commit requires a 7-40 character hexadecimal git commit."
}

function Test-RemoteRef([string]$Repo, [string]$RemoteRef) {
    Invoke-Git -C $Repo ls-remote --exit-code origin $RemoteRef 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}

function Checkout-InstallRef([string]$Repo) {
    $kind = $script:InstallRefKind
    if ($kind -eq "auto") {
        if (Test-RemoteRef $Repo "refs/heads/$script:InstallRef") {
            $kind = "branch"
        } elseif (Test-RemoteRef $Repo "refs/tags/$script:InstallRef") {
            $kind = "tag"
        } else {
            $kind = "commit"
        }
    }
    switch ($kind) {
        "branch" {
            if (-not (Test-RemoteRef $Repo "refs/heads/$script:InstallRef")) {
                throw "Origin has no branch '$script:InstallRef'."
            }
            Invoke-Git -C $Repo fetch --depth 1 origin "+refs/heads/$script:InstallRef`:refs/remotes/origin/$script:InstallRef"
            if ($LASTEXITCODE -ne 0) { throw "git fetch branch failed." }
            Invoke-Git -C $Repo checkout -B $script:InstallRef "refs/remotes/origin/$script:InstallRef"
            if ($LASTEXITCODE -ne 0) { throw "git checkout branch failed." }
        }
        "tag" {
            if (-not (Test-RemoteRef $Repo "refs/tags/$script:InstallRef")) {
                throw "Origin has no tag '$script:InstallRef'."
            }
            Invoke-Git -C $Repo fetch --depth 1 origin "refs/tags/$script:InstallRef"
            if ($LASTEXITCODE -ne 0) { throw "git fetch tag failed." }
            Invoke-Git -C $Repo checkout --detach "FETCH_HEAD^{commit}"
            if ($LASTEXITCODE -ne 0) { throw "git checkout tag failed." }
        }
        "commit" {
            Invoke-Git -C $Repo fetch --depth 1 origin $script:InstallRef
            if ($LASTEXITCODE -ne 0) { throw "git fetch commit failed." }
            Invoke-Git -C $Repo checkout --detach "FETCH_HEAD^{commit}"
            if ($LASTEXITCODE -ne 0) { throw "git checkout commit failed." }
        }
        default { throw "Unsupported revision kind '$kind'." }
    }
    $script:InstallRefKind = $kind
}

Write-Host "=== hipfire installer ===" -ForegroundColor Cyan
Write-Host "Requested source: $InstallRefKind '$InstallRef'"
Write-Host ""

# ─── GPU Detection ───────────────────────────────────────
Write-Host "Checking for AMD GPU..." -ForegroundColor Cyan

$GpuArch = "unknown"
try {
    $VideoControllers = Get-CimInstance Win32_VideoController -ErrorAction Stop
    # Prefer the most capable AMD adapter: on APU + discrete-GPU systems the
    # integrated controller enumerates first, and its name often matches no
    # arch regex below. Largest reported memory is a reliable discrete signal
    # across RX / AI PRO / embedded naming.
    $AmdGpu = $VideoControllers |
        Where-Object { $_.Name -match "AMD|Radeon" } |
        Sort-Object { if ($null -eq $_.AdapterRAM) { [uint64]0 } else { [uint64]$_.AdapterRAM } } -Descending |
        Select-Object -First 1
    if ($AmdGpu) {
        $GpuName = $AmdGpu.Name
        Write-Host "  Found: $GpuName"

        # Map GPU name to arch
        if ($GpuName -match "5700|RX 5[0-9]{3}") {
            $GpuArch = "gfx1010"
        } elseif ($GpuName -match "6[89]00|6[79]50|6[89]50|RX 6[0-9]{3}") {
            $GpuArch = "gfx1030"
        } elseif ($GpuName -match "7900|7800|7700|7600|RX 7[0-9]{3}") {
            $GpuArch = "gfx1100"
        } elseif ($GpuName -match "9070") {
            $GpuArch = "gfx1201"
        } elseif ($GpuName -match "9060|RX 9[0-9]{3}") {
            $GpuArch = "gfx1200"
        }
    } else {
        Write-Host "  WARNING: No AMD/Radeon GPU found in Win32_VideoController." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  WARNING: Could not query GPU information: $_" -ForegroundColor Yellow
}

if ($GpuArch -eq "unknown") {
    Write-Host "  WARNING: Could not detect GPU architecture." -ForegroundColor Yellow
    Write-Host "  Supported: gfx1010 (RX 5700), gfx1030 (RX 6800), gfx1100 (RX 7900), gfx1200 (RX 9060), gfx1201 (RX 9070)"
    $GpuArch = Read-Host "  Enter your GPU arch [or Enter to skip]"
    if ([string]::IsNullOrWhiteSpace($GpuArch)) { $GpuArch = "unknown" }
}
Write-Host "  GPU arch: $GpuArch" -ForegroundColor Green

# ─── Create directories ──────────────────────────────────
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $BinDir    | Out-Null
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir  | Out-Null
Write-Host "  $BinDir" -ForegroundColor Green
Write-Host "  $RuntimeDir" -ForegroundColor Green
Write-Host "  $ModelsDir" -ForegroundColor Green

# ─── HIP DLL (amdhip64.dll) ──────────────────────────────
Write-Host ""
Write-Host "Checking HIP runtime (amdhip64.dll)..." -ForegroundColor Cyan

$HipDllFound = $false
$HipDllDest  = "$RuntimeDir\amdhip64.dll"

# Check RuntimeDir first (idempotent re-runs)
if (Test-Path $HipDllDest) {
    Write-Host "  amdhip64.dll: found in RuntimeDir ✓" -ForegroundColor Green
    $HipDllFound = $true
}

# Check %HIP_PATH%\bin (unversioned and versioned)
if (-not $HipDllFound -and $env:HIP_PATH) {
    foreach ($dllName in @("amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll")) {
        $candidate = Join-Path $env:HIP_PATH "bin\$dllName"
        if (Test-Path $candidate) {
            Write-Host "  ${dllName}: found at $candidate ✓" -ForegroundColor Green
            Copy-Item $candidate $HipDllDest -Force
            $HipDllFound = $true
            break
        }
    }
}

# Check standard ROCm install locations (unversioned and versioned)
if (-not $HipDllFound) {
    foreach ($dllName in @("amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll")) {
        # Check versioned ROCm dirs (e.g. C:\Program Files\AMD\ROCm\7.1\bin\)
        $rocmBase = "C:\Program Files\AMD\ROCm"
        if (Test-Path $rocmBase) {
            foreach ($verDir in (Get-ChildItem $rocmBase -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending)) {
                $candidate = Join-Path $verDir.FullName "bin\$dllName"
                if (Test-Path $candidate) {
                    Write-Host "  ${dllName}: found at $candidate ✓" -ForegroundColor Green
                    Copy-Item $candidate $HipDllDest -Force
                    $HipDllFound = $true
                    break
                }
            }
            if ($HipDllFound) { break }
        }
        # Also check flat layout
        $candidate = "C:\Program Files\AMD\ROCm\bin\$dllName"
        if (Test-Path $candidate) {
            Write-Host "  ${dllName}: found at $candidate ✓" -ForegroundColor Green
            Copy-Item $candidate $HipDllDest -Force
            $HipDllFound = $true
            break
        }
    }
}

# Attempt download from GitHub release
if (-not $HipDllFound) {
    Write-Host "  amdhip64.dll: not found locally. Downloading from GitHub release..." -ForegroundColor Yellow
    $DllUrl = "https://github.com/$GithubRepo/releases/download/hip-runtime/amdhip64.dll"
    try {
        Invoke-WebRequest -Uri $DllUrl -OutFile $HipDllDest -UseBasicParsing
        Write-Host "  amdhip64.dll: downloaded ✓" -ForegroundColor Green
        $HipDllFound = $true
    } catch {
        Write-Host "  amdhip64.dll: download failed: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "  hipfire needs amdhip64.dll to run. Install ROCm for Windows manually:" -ForegroundColor Yellow
        Write-Host "    https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html"
        Write-Host "  Or place amdhip64.dll in: $RuntimeDir"
        Write-Host ""
        $reply = Read-Host "  Continue without HIP runtime? [y/N]"
        if ($reply -notmatch "^[Yy]$") {
            Write-Host "Exiting. Re-run after installing ROCm." -ForegroundColor Red
            exit 1
        }
    }
}

# Ensure runtime dir is in PATH for this session so daemon can find the DLL
if ($HipDllFound) {
    $env:PATH = "$RuntimeDir;$env:PATH"
}

# ─── HIP version vs GPU arch check ──────────────────────
if ($HipDllFound -and $GpuArch -ne "unknown") {
    # Try to get HIP version from the DLL or hipconfig
    $HipVer = ""
    $hipconfig = "$env:HIP_PATH\bin\hipconfig.exe"
    if (-not (Test-Path $hipconfig)) { $hipconfig = "C:\Program Files\AMD\ROCm\bin\hipconfig.exe" }
    if (Test-Path $hipconfig) {
        try { $HipVer = (& $hipconfig --version 2>$null) -replace '[^\d.]','' | Select-Object -First 1 } catch {}
    }
    # Fallback: check DLL file version
    if (-not $HipVer) {
        try {
            $dllPath = if (Test-Path $HipDllDest) { $HipDllDest } else { $candidate }
            $ver = (Get-Item $dllPath).VersionInfo.ProductVersion
            if ($ver) { $HipVer = $ver }
        } catch {}
    }

    if ($HipVer) {
        $parts = $HipVer.Split(".")
        $major = [int]$parts[0]
        $minor = if ($parts.Length -gt 1) { [int]$parts[1] } else { 0 }
        Write-Host "  HIP version: $major.$minor" -ForegroundColor Green

        # Minimum versions per arch
        $minMajor = 5; $minMinor = 0
        switch ($GpuArch) {
            { $_ -in "gfx1200","gfx1201" } { $minMajor = 6; $minMinor = 4 }
            { $_ -in "gfx1100","gfx1101" } { $minMajor = 5; $minMinor = 5 }
        }

        if ($major -lt $minMajor -or ($major -eq $minMajor -and $minor -lt $minMinor)) {
            Write-Host ""
            Write-Host "  WARNING: HIP $major.$minor is too old for $GpuArch (needs $minMajor.$minMinor+)" -ForegroundColor Red
            Write-Host "  Kernels may fail to load. Update AMD HIP SDK:" -ForegroundColor Yellow
            Write-Host "    https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html" -ForegroundColor Yellow
            Write-Host ""
            $reply = Read-Host "  Continue anyway? [y/N]"
            if ($reply -notmatch "^[Yy]$") { exit 1 }
        }
    }
}

# ─── Clone / update repo ─────────────────────────────────
Write-Host ""
Write-Host "Setting up hipfire source..." -ForegroundColor Cyan

if (-not (Test-Path "$SrcDir\.git")) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "  ERROR: git is required. Install from https://git-scm.com and re-run." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Initializing https://github.com/$GithubRepo.git ..."
    try {
        if ((Test-Path $SrcDir) -and (Get-ChildItem $SrcDir -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
            throw "$SrcDir exists but is not a git checkout; move it aside and retry."
        }
        New-Item -ItemType Directory -Force -Path $SrcDir | Out-Null
        Invoke-Git -C $SrcDir init --quiet
        if ($LASTEXITCODE -ne 0) { throw "git init failed." }
        Invoke-Git -C $SrcDir remote add origin "https://github.com/$GithubRepo.git"
        if ($LASTEXITCODE -ne 0) { throw "git remote add failed." }
        Checkout-InstallRef $SrcDir
        Write-Host "  Checked out $InstallRefKind '$InstallRef' ✓" -ForegroundColor Green
    } catch {
        Write-Host "  Checkout failed: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  Existing clone found at $SrcDir"
    $PreviousCommit = (Invoke-Git -C $SrcDir rev-parse --verify HEAD 2>$null | Out-String).Trim()
    if ($PreviousCommit) {
        $stamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH-mm-ssZ")
        $shortCommit = $PreviousCommit.Substring(0, [Math]::Min(12, $PreviousCommit.Length))
        $backupRef = "refs/hipfire/backups/pre-install-$stamp-$shortCommit"
        Invoke-Git -C $SrcDir update-ref $backupRef $PreviousCommit
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Previous source retained at $backupRef" -ForegroundColor Yellow
        }
    }
    $status = Invoke-Git -C $SrcDir status --porcelain 2>&1 | Out-String
    $ProceedWithUpdate = $true
    if ($status.Trim()) {
        Write-Host "  WARNING: local modifications detected." -ForegroundColor Yellow
        $reply = Read-Host "  Stash local changes and update? [y/N]"
        if ($reply -match "^[Yy]$") {
            $stamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH-mm-ssZ")
            $stashMessage = "hipfire-install-$stamp"
            Invoke-Git -C $SrcDir stash push --include-untracked -m $stashMessage
            if ($LASTEXITCODE -ne 0) {
                throw "git stash failed; source checkout was not changed."
            }
            Write-Host "  Recover later with: git -C $SrcDir stash pop" -ForegroundColor Yellow
        } else {
            Write-Host "  Keeping existing checkout." -ForegroundColor Yellow
            $ProceedWithUpdate = $false
        }
    }
    if ($ProceedWithUpdate) {
        Write-Host "  Updating to $InstallRefKind '$InstallRef'..."
        try {
            $env:GIT_TERMINAL_PROMPT = "0"
            Checkout-InstallRef $SrcDir
            Write-Host "  Checked out $InstallRefKind '$InstallRef' ✓" -ForegroundColor Green
        } catch {
            Write-Host "  Update failed (non-fatal). Using existing checkout." -ForegroundColor Yellow
        }
    }
}

$RepoDir = $SrcDir
$ResolvedCommit = (Invoke-Git -C $RepoDir rev-parse --verify HEAD 2>$null | Out-String).Trim()
$ResolvedRef = (Invoke-Git -C $RepoDir describe --tags --exact-match HEAD 2>$null | Out-String).Trim()
if (-not $ResolvedRef) {
    $ResolvedRef = (Invoke-Git -C $RepoDir symbolic-ref --short HEAD 2>$null | Out-String).Trim()
}
if (-not $ResolvedRef) { $ResolvedRef = "detached" }
Write-Host "Source resolved: $ResolvedRef @ $ResolvedCommit" -ForegroundColor Green

# ─── Build / install binaries ────────────────────────────
Write-Host ""
Write-Host "Installing hipfire binaries..." -ForegroundColor Cyan

# Compile the daemon from this checkout: a release binary built at a different
# revision may embed a different kernel registry and reject its sidecar indexes.
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $RustupExe = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $RustupExe -UseBasicParsing
    & $RustupExe -y --default-toolchain stable
    if ($LASTEXITCODE -ne 0) { throw "Rust installation failed" }
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
}
Push-Location $RepoDir
try {
    cargo build --release -p hipfire-daemon
    if ($LASTEXITCODE -ne 0) { throw "daemon source build failed" }
} finally {
    Pop-Location
}
$Meta = cargo metadata --format-version 1 --manifest-path "$RepoDir\Cargo.toml" 2>$null | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $Meta.target_directory) { throw "could not resolve Cargo target directory" }
$TargetDir = $Meta.target_directory
$BuiltExe = "$TargetDir\release\daemon.exe"
if (-not (Test-Path $BuiltExe)) { throw "daemon source build did not produce $BuiltExe" }
Copy-Item $BuiltExe "$BinDir\daemon.exe" -Force
Write-Host "  daemon.exe built from selected revision ✓" -ForegroundColor Green

# Build the native CLI from the same checkout and revision as the daemon.
$CliExe = "$TargetDir\release\hipfire.exe"
Push-Location $RepoDir
try {
    cargo build --release -p hipfire-cli
    if ($LASTEXITCODE -ne 0) { throw "native CLI build failed" }
} finally {
    Pop-Location
}
if (-not (Test-Path $CliExe)) {
    Write-Host "  Native CLI binary not found at $CliExe" -ForegroundColor Red
    exit 1
}
Copy-Item $CliExe "$BinDir\hipfire.exe" -Force

# Copy optional helper binaries if present
foreach ($exe in @("infer.exe", "infer_hfq.exe")) {
    $src = "$TargetDir\release\examples\$exe"
    if (Test-Path $src) { Copy-Item $src "$BinDir\$exe" -Force }
}

# Terminal UI (workspace bin — lives under target\release\, not examples\).
# Build on demand if the pre-built tree didn't include it. The TUI is
# OPTIONAL — a failed build (or absent cargo) must NOT abort the install,
# but it must NOT be swallowed silently either: warn clearly so the user
# knows the terminal UI won't be available and how to get it.
$TuiExe = "$TargetDir\release\hipfire-tui.exe"
if (-not (Test-Path $TuiExe)) {
    if (Get-Command cargo -ErrorAction SilentlyContinue) {
        Push-Location $RepoDir
        try { cargo build --release -p hipfire-tui } catch {} finally { Pop-Location }
    }
}
if (Test-Path $TuiExe) {
    Copy-Item $TuiExe "$BinDir\hipfire-tui.exe" -Force
    Write-Host "  hipfire-tui (terminal UI) installed ✓" -ForegroundColor Green
} else {
    # Windows remedy: `hipfire update` is NOT Windows-aware (its GPU-arch +
    # git-reset path is Linux/sysfs-only), so recommend the direct cargo build
    # as the PRIMARY fix on Windows, then re-running this installer to copy it.
    Write-Warning "hipfire-tui (terminal UI) was not built — it will be unavailable. To get it: install Rust, then ``cargo build --release -p hipfire-tui`` and re-run scripts\install.ps1 (copies it into ~/.hipfire/bin/)."
}

# CPU quantizer (optional, but build it on demand so `hipfire quantize` works
# without a second manual installation step).
$QuantExe = "$TargetDir\release\hipfire-quantize.exe"
if (-not (Test-Path $QuantExe) -and (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Push-Location $RepoDir
    try { cargo build --release -p hipfire-quantize } catch {} finally { Pop-Location }
}
if (Test-Path $QuantExe) {
    Copy-Item $QuantExe "$BinDir\hipfire-quantize.exe" -Force
} else {
    Write-Warning "hipfire-quantize was not built; the quantize subcommand will be unavailable."
}

# ─── Native CLI ──────────────────────────────────────────
Write-Host ""
Write-Host "Installing CLI..." -ForegroundColor Cyan
$LegacyCliDir = "$HipfireDir\cli"
if (Test-Path $LegacyCliDir) { Remove-Item $LegacyCliDir -Recurse -Force }
$LegacyWrapper = "$BinDir\hipfire.cmd"
if (Test-Path $LegacyWrapper) { Remove-Item $LegacyWrapper -Force }
Write-Host "  Native CLI: $BinDir\hipfire.exe ✓" -ForegroundColor Green

# ─── Indexed kernel package ──────────────────────────────
# The daemon resolves the active GPU architecture and builds the exact Rust
# registry using the same pack_to implementation as hipfire-kernel-pack.
# Never seed unindexed .hsaco files from an unrelated source checkout.
$DaemonExe = "$BinDir\daemon.exe"
if (Test-Path $DaemonExe) {
    Write-Host ""
    Write-Host "Packaging indexed GPU kernels from the exact-source registry..." -ForegroundColor Cyan
    $hipccAvailable = $false
    if ($env:HIPFIRE_HIPCC -and (Test-Path $env:HIPFIRE_HIPCC)) { $hipccAvailable = $true }
    elseif ($env:HIP_PATH -and (Test-Path (Join-Path $env:HIP_PATH "bin\hipcc.bat"))) { $hipccAvailable = $true }
    elseif ($env:HIP_PATH -and (Test-Path (Join-Path $env:HIP_PATH "bin\hipcc.exe"))) { $hipccAvailable = $true }
    elseif (Get-Command hipcc -ErrorAction SilentlyContinue) { $hipccAvailable = $true }
    elseif (Test-Path "C:\Program Files\AMD\ROCm") {
        $rocmHipcc = Get-ChildItem "C:\Program Files\AMD\ROCm" -Recurse -Filter "hipcc.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($rocmHipcc) { $hipccAvailable = $true }
    }

    if (-not $hipccAvailable) {
        throw "hipcc is required to build indexed kernel packages. Install the AMD HIP SDK and re-run this installer; unindexed .hsaco files cannot load compiler-free."
    }
    if ($GpuArch -in @("gfx1201", "gfx1100", "gfx1151", "gfx906", "gfx942")) {
        & $DaemonExe --precompile
        if ($LASTEXITCODE -ne 0) {
            throw "Indexed kernel packaging failed (exit $LASTEXITCODE). No compiler-free install was produced."
        }
        Write-Host "  Indexed kernel package complete ✓" -ForegroundColor Green
    } else {
        Write-Warning "No indexed kernel registry for $GpuArch; this GPU requires hipcc JIT. Compiler-free operation is unavailable."
    }
}

# ─── Config ──────────────────────────────────────────────
$ConfigFile = "$HipfireDir\config.toml"
$LegacyConfigFile = "$HipfireDir\config.json"
if (-not (Test-Path $ConfigFile) -and -not (Test-Path $LegacyConfigFile)) {
    [System.IO.File]::WriteAllText($ConfigFile, "schema_version = 1`n")
    Write-Host ""
    Write-Host "Config written: $ConfigFile" -ForegroundColor Green
}

# ─── PATH ────────────────────────────────────────────────
Write-Host ""
$NoPath = $args -contains "--no-path"
$CurrentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($null -eq $CurrentUserPath) { $CurrentUserPath = "" }

if ($NoPath) {
    Write-Host "Skipping PATH modification (--no-path)" -ForegroundColor Yellow
    Write-Host "  Add manually to user PATH: $BinDir" -ForegroundColor Yellow
} elseif ($CurrentUserPath -notlike "*$BinDir*") {
    Write-Host "hipfire bin dir is not in your user PATH." -ForegroundColor Yellow
    Write-Host "  $BinDir"
    $reply = Read-Host "Add to user PATH permanently? [Y/n]"
    if ($reply -notmatch "^[Nn]$") {
        $NewPath = "$BinDir;$CurrentUserPath"
        # Safety: warn if PATH would exceed Windows limit (2047 chars)
        if ($NewPath.Length -gt 2040) {
            Write-Host "  WARNING: User PATH would be $($NewPath.Length) chars (limit ~2047)." -ForegroundColor Red
            Write-Host "  Skipping to avoid PATH truncation. Add manually:" -ForegroundColor Red
            Write-Host "    $BinDir" -ForegroundColor Yellow
        } else {
            [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
            $env:PATH = "$BinDir;$env:PATH"
            Write-Host "  PATH updated ✓ (restart your shell to apply)" -ForegroundColor Green
        }
    } else {
        Write-Host "  Add manually to user PATH: $BinDir" -ForegroundColor Yellow
    }
} else {
    Write-Host "hipfire already in PATH ✓" -ForegroundColor Green
}

# ─── Quick start ─────────────────────────────────────────
Write-Host ""
Write-Host "=== hipfire installed ===" -ForegroundColor Cyan
if (Test-Path "$BinDir\hipfire.exe") {
    & "$BinDir\hipfire.exe" --version
    Write-Host "  source: $ResolvedRef @ $ResolvedCommit" -ForegroundColor Green
}
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Green
Write-Host "  hipfire list                        # see local models"
Write-Host "  hipfire run <model.hfq> `"Hello`"    # generate text"
Write-Host "  hipfire serve                       # start OpenAI-compatible API"
Write-Host "  hipfire version                     # verify binary/source identity"
Write-Host ""
Write-Host "Models go in $ModelsDir"
Write-Host ""
