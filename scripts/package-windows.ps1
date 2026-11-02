<#
.SYNOPSIS
    Package already-built Windows release binaries into an unsigned developer ZIP.

.DESCRIPTION
    Writes dist/hipfire-<version>-x86_64-pc-windows-msvc.zip and a .zip.sha256
    next to it. Nothing is built, downloaded or signed. The version comes from
    the built hipfire.exe --version, and packaging fails if the commit that
    binary reports differs from git HEAD (stale binaries).

    ZIP root, in ordinal order: LICENSE, LICENSE-APACHE, LICENSE-MIT, NOTICE,
    README-windows.txt, daemon.exe, hipfire.exe, manifest.json. daemon.exe is the
    daemon crate's [[bin]] name and the name the CLI looks for.

    manifest.json lists a SHA-256 for every file except itself (it cannot hold
    its own hash). The ZIP is not byte-reproducible (timestamps vary).

.PARAMETER ReleaseDir
    Directory with hipfire.exe and daemon.exe (default target/release).

.PARAMETER OutputDir
    Directory for the ZIP and .sha256 (default dist). Relative paths in both
    parameters are resolved against the repository root.
#>
[CmdletBinding()]
param(
    [string]$ReleaseDir = 'target/release',
    [string]$OutputDir = 'dist'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression

$Triple = 'x86_64-pc-windows-msvc'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($ReleaseDir)) { $ReleaseDir = Join-Path $RepoRoot $ReleaseDir }
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $RepoRoot $OutputDir }

# Payload name -> source path. Binaries come from the build output, licenses from the repo root.
$sources = @{
    'hipfire.exe' = Join-Path $ReleaseDir 'hipfire.exe'
    'daemon.exe'  = Join-Path $ReleaseDir 'daemon.exe'
}
foreach ($name in 'LICENSE', 'LICENSE-APACHE', 'LICENSE-MIT', 'NOTICE') {
    $sources[$name] = Join-Path $RepoRoot $name
}
foreach ($name in $sources.Keys) {
    if (-not (Test-Path -LiteralPath $sources[$name] -PathType Leaf)) {
        throw "Missing $($sources[$name]). Build first: cargo build --release -p hipfire-cli -p hipfire-daemon"
    }
}

# Build identity: the binary's own version output, git HEAD, rustc on PATH.
$versionLine = @(& $sources['hipfire.exe'] --version)[0]
if ($LASTEXITCODE -ne 0) { throw "hipfire.exe --version failed (exit $LASTEXITCODE)" }
if ("$versionLine" -notmatch '^hipfire\s+(?<ver>\d+\.\d+\.\d+[0-9A-Za-z.+-]*)\s+\((?<sha>[0-9a-f]{7,64});') {
    throw "Cannot parse version and commit from 'hipfire.exe --version': $versionLine"
}
$version = $Matches['ver']
$binarySha = $Matches['sha']

$gitCommit = (& git -C $RepoRoot rev-parse HEAD | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "git rev-parse HEAD failed (exit $LASTEXITCODE)" }
if (-not $gitCommit.StartsWith($binarySha)) {
    throw "Stale binaries: hipfire.exe reports commit $binarySha but git HEAD is $gitCommit. Rebuild before packaging."
}

$rustc = (& rustc --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "rustc --version failed (exit $LASTEXITCODE)" }

$zipName = "hipfire-$version-$Triple.zip"
$zipPath = Join-Path $OutputDir $zipName
$utf8 = New-Object System.Text.UTF8Encoding($false)
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ('hipfire-package-' + [guid]::NewGuid().ToString('N'))
$ok = $false

try {
    New-Item -ItemType Directory -Path $stage, $OutputDir -Force | Out-Null
    foreach ($name in $sources.Keys) { Copy-Item -LiteralPath $sources[$name] -Destination (Join-Path $stage $name) }

    $readme = @(
        "hipfire $version for Windows ($Triple)",
        "",
        "UNSIGNED DEVELOPER ARTIFACT. This package is not code-signed, so Windows",
        "SmartScreen or antivirus software may warn when you run it.",
        "",
        "Contents: hipfire.exe (CLI), daemon.exe (inference daemon), license files,",
        "and manifest.json (version, commit, toolchain, per-file SHA-256).",
        "",
        "Not bundled:",
        "  - The AMD HIP runtime. A user-installed AMD HIP runtime is required;",
        "    runtime tuple certification is pending (0.4.3 M0).",
        "  - Kernel packs.",
        "",
        "Checksums: manifest.json hashes every file except itself (a file cannot",
        "contain its own hash). The .zip.sha256 file covers the whole ZIP. Both only",
        "detect corruption; a checksum is not proof of publisher authenticity.",
        "build_time_utc and rustc_version describe the packaging environment at the",
        "time this ZIP was assembled."
    ) -join "`r`n"
    [System.IO.File]::WriteAllText((Join-Path $stage 'README-windows.txt'), $readme + "`r`n", $utf8)

    [string[]]$payload = @($sources.Keys) + 'README-windows.txt'
    [Array]::Sort($payload, [StringComparer]::Ordinal)
    $files = foreach ($name in $payload) {
        $item = Get-Item -LiteralPath (Join-Path $stage $name)
        [ordered]@{
            path   = $name
            size   = $item.Length
            sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $manifest = [ordered]@{
        version                    = $version
        git_commit                 = $gitCommit
        rustc_version              = $rustc
        target_triple              = $Triple
        build_time_utc             = [DateTime]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", [Globalization.CultureInfo]::InvariantCulture)
        hip_runtime                = [ordered]@{ bundled = $false; note = 'user-installed AMD HIP runtime required; runtime tuple certification pending (0.4.3 M0)' }
        kernel_packs               = [ordered]@{ bundled = $false }
        signed                     = $false
        files_excluded_from_hashes = @('manifest.json')
        files                      = @($files)
    }
    [System.IO.File]::WriteAllText((Join-Path $stage 'manifest.json'), (($manifest | ConvertTo-Json -Depth 6) + "`n"), $utf8)

    [string[]]$entries = $payload + 'manifest.json'
    [Array]::Sort($entries, [StringComparer]::Ordinal)
    $zipStream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::Create)
    try {
        $zip = New-Object System.IO.Compression.ZipArchive($zipStream, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($name in $entries) {
                $entry = $zip.CreateEntry($name, [System.IO.Compression.CompressionLevel]::Optimal)
                $out = $entry.Open()
                try {
                    $in = [System.IO.File]::OpenRead((Join-Path $stage $name))
                    try { $in.CopyTo($out) } finally { $in.Dispose() }
                } finally { $out.Dispose() }
            }
        } finally { $zip.Dispose() }
    } finally { $zipStream.Dispose() }

    $zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    [System.IO.File]::WriteAllText("$zipPath.sha256", "$zipHash  $zipName`n", $utf8)
    $ok = $true
    Write-Host "Created $zipPath"
    Write-Host "Created $zipPath.sha256 ($zipHash)"
}
finally {
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
    if (-not $ok -and (Test-Path -LiteralPath $zipPath)) { Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue }
}
