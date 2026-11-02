# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
    Fail if a Windows PE binary imports a DLL outside an explicit allowlist.

.DESCRIPTION
    hipfire loads the HIP runtime at run time (LoadLibrary via libloading) and
    never links it, so hipfire.exe and daemon.exe must start on a machine that
    has no AMD software at all. This script lists the DLLs named in each file's
    import table and delay-import table (llvm-readobj --coff-imports) and exits
    1 if any of them is not on the allowlist below. Every imported DLL is
    printed, so the CI log documents the import surface of each build.

    Allowlist (matched case-insensitively):

      * Windows system DLLs, by exact name (see $SystemDlls): present on every
        supported Windows install, so a dependency on them cannot break startup.
      * api-ms-win-*.dll: Windows API-set contracts (UCRT and core API sets),
        resolved by the loader on every supported Windows.
      * vcruntime140*.dll, msvcp140*.dll: the MSVC C/C++ runtime. Rust's default
        x86_64-pc-windows-msvc target links it dynamically. The Visual Studio
        Build Tools that scripts/install.ps1's source build requires install it,
        and the redistributable carries it; it is not part of a bare Windows.
      * amdhip64*.dll, amd_comgr*.dll, hiprtc*.dll: the HIP SDK runtime set.
        Allowed but NOT expected, because hipfire loads HIP lazily at run time.
        If one shows up as a static import, startup now needs the HIP SDK on
        PATH and the CI log shows it here.

    llvm-readobj comes from the Rust `llvm-tools` component
    (<sysroot>/lib/rustlib/<host>/bin) unless -LlvmReadobj is given.

.PARAMETER Path
    One or more .exe/.dll files to check.

.PARAMETER LlvmReadobj
    Explicit path to llvm-readobj(.exe). Default: the Rust toolchain's copy,
    then llvm-readobj on PATH.

.PARAMETER SelfTest
    Run the parser and allowlist against canned llvm-readobj output and exit
    0 on success, 1 on failure. Needs neither llvm-readobj nor -Path.

.EXAMPLE
    ./scripts/check-pe-imports.ps1 -Path target/release/hipfire.exe, target/release/daemon.exe
#>
[CmdletBinding()]
param(
    [string[]]$Path,
    [string]$LlvmReadobj,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Stop'

# Windows system DLLs the binaries may import (exact names, lower case).
$SystemDlls = @(
    'kernel32.dll', 'kernelbase.dll', 'ntdll.dll',        # process, memory, threads, loader
    'advapi32.dll', 'secur32.dll', 'crypt32.dll',         # registry, tokens, TLS/cert store
    'bcrypt.dll', 'bcryptprimitives.dll', 'ncrypt.dll',   # CNG: RNG and crypto (std, rustls/ring)
    'ws2_32.dll', 'mswsock.dll', 'iphlpapi.dll',          # sockets, network adapters
    'userenv.dll', 'shell32.dll', 'shlwapi.dll',          # profile dirs, known folders, path helpers
    'ole32.dll', 'oleaut32.dll', 'rpcrt4.dll',            # COM and RPC (known folders, WMI-style queries)
    'user32.dll', 'version.dll', 'normaliz.dll',          # window/console helpers, file versions, IDN
    'dbghelp.dll', 'psapi.dll', 'powrprof.dll',           # backtraces, process info, power state
    'dxgi.dll', 'd3d12.dll', 'dxcore.dll',                # adapter and memory-budget queries
    'setupapi.dll', 'cfgmgr32.dll', 'propsys.dll',        # device enumeration, property store
    'wtsapi32.dll', 'winmm.dll'                           # session info, timers
)

# Wildcard patterns (PowerShell -like, case-insensitive).
$AllowedPatterns = @(
    'api-ms-win-*.dll',   # API-set contracts
    'vcruntime140*.dll',  # MSVC runtime
    'msvcp140*.dll',      # MSVC C++ runtime
    'amdhip64*.dll',      # HIP SDK (allowed, not expected)
    'amd_comgr*.dll',
    'hiprtc*.dll'
)

function Test-DllAllowed {
    param([Parameter(Mandatory)][string]$Name)
    $n = $Name.ToLowerInvariant()
    if ($SystemDlls -contains $n) { return $true }
    foreach ($pattern in $AllowedPatterns) {
        if ($n -like $pattern) { return $true }
    }
    return $false
}

# Returns the DLL names from `llvm-readobj --coff-imports` output, in order,
# for both "Import {" and "DelayImport {" blocks. The first "Name:" line of
# each block is the DLL; per-symbol lines use "Symbol:" and are ignored.
function ConvertFrom-CoffImports {
    param([Parameter(Mandatory)][AllowEmptyString()][string[]]$Lines)
    $dlls = New-Object System.Collections.Generic.List[string]
    $inBlock = $false
    $haveName = $false
    foreach ($line in $Lines) {
        if ($line -match '^(Import|DelayImport)\s*\{\s*$') {
            $inBlock = $true
            $haveName = $false
        } elseif ($line -match '^\}\s*$') {
            $inBlock = $false
        } elseif ($inBlock -and -not $haveName -and $line -match '^\s+Name:\s*(\S+)\s*$') {
            $dlls.Add($Matches[1])
            $haveName = $true
        }
    }
    return , $dlls.ToArray()
}

function Find-LlvmReadobj {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path -LiteralPath $Explicit -PathType Leaf)) {
            throw "llvm-readobj not found at -LlvmReadobj path: $Explicit"
        }
        return (Resolve-Path -LiteralPath $Explicit).Path
    }
    $rustc = Get-Command rustc -ErrorAction SilentlyContinue
    if ($rustc) {
        $sysroot = (& rustc --print sysroot | Out-String).Trim()
        $hostLine = (& rustc -vV) | Where-Object { $_ -like 'host:*' } | Select-Object -First 1
        if ($sysroot -and $hostLine) {
            $hostTriple = ($hostLine -replace '^host:\s*', '').Trim()
            foreach ($name in 'llvm-readobj.exe', 'llvm-readobj') {
                $candidate = Join-Path $sysroot (Join-Path 'lib' (Join-Path 'rustlib' (Join-Path $hostTriple (Join-Path 'bin' $name))))
                if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
            }
        }
    }
    $onPath = Get-Command llvm-readobj -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw 'llvm-readobj not found: install the Rust llvm-tools component (rustup component add llvm-tools) or pass -LlvmReadobj.'
}

function Invoke-SelfTest {
    $canned = @'
File: hipfire.exe
Format: COFF-x86-64
Arch: x86_64
AddressSize: 64bit
Import {
  Name: KERNEL32.dll
  ImportLookupTableRVA: 0x1000
  ImportAddressTableRVA: 0x2000
  Symbol: GetLastError (0)
  Symbol: Name: (1)
}
Import {
  Name: api-ms-win-crt-runtime-l1-1-0.dll
  ImportLookupTableRVA: 0x1100
  Symbol: abort (0)
}
Import {
  Name: VCRUNTIME140.dll
  Symbol: memcpy (0)
}
DelayImport {
  Name: amdhip64_6.dll
  Attributes: 0x1
  Symbol: hipInit (0)
}
'@ -split "\r?\n"
    $failures = New-Object System.Collections.Generic.List[string]
    $got = ConvertFrom-CoffImports -Lines $canned
    $want = 'KERNEL32.dll', 'api-ms-win-crt-runtime-l1-1-0.dll', 'VCRUNTIME140.dll', 'amdhip64_6.dll'
    if (($got -join '|') -ne ($want -join '|')) {
        $failures.Add("parser: expected [$($want -join ', ')], got [$($got -join ', ')]")
    }
    $empty = ConvertFrom-CoffImports -Lines @('File: x.exe', 'Format: COFF-x86-64')
    if ($empty.Count -ne 0) { $failures.Add('parser: no import blocks must yield no DLLs') }
    foreach ($ok in 'KERNEL32.dll', 'ntdll.dll', 'API-MS-WIN-CORE-SYNCH-L1-2-0.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll', 'amdhip64_6.dll', 'amd_comgr0602.dll', 'hiprtc0602.dll') {
        if (-not (Test-DllAllowed $ok)) { $failures.Add("allowlist: $ok must be allowed") }
    }
    foreach ($bad in 'libamdhip64.so', 'msvcr120.dll', 'ucrtbase_clr0400.dll', 'vcomp140.dll', 'libcurl.dll', 'kernel32.dll.evil', 'amdhip.dll') {
        if (Test-DllAllowed $bad) { $failures.Add("allowlist: $bad must be rejected") }
    }
    if ($failures.Count -gt 0) {
        $failures | ForEach-Object { Write-Host "SELFTEST FAIL: $_" }
        return 1
    }
    Write-Host 'SELFTEST OK: import parser and allowlist behave as specified'
    return 0
}

if ($SelfTest) { exit (Invoke-SelfTest) }

if (-not $Path -or $Path.Count -eq 0) {
    throw 'Specify -Path <exe/dll...> (or -SelfTest).'
}

$readobj = Find-LlvmReadobj -Explicit $LlvmReadobj
Write-Host "llvm-readobj: $readobj"

$offenders = New-Object System.Collections.Generic.List[string]
$checked = 0
foreach ($file in $Path) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "File not found: $file" }
    $full = (Resolve-Path -LiteralPath $file).Path
    $output = & $readobj --coff-imports $full
    if ($LASTEXITCODE -ne 0) { throw "llvm-readobj failed (exit $LASTEXITCODE) on $full" }
    $dlls = ConvertFrom-CoffImports -Lines @($output)
    if ($dlls.Count -eq 0) {
        # Every Rust binary imports kernel32; an empty result means the parser
        # or llvm-readobj output format is not what this script expects.
        throw "No imports parsed from $full; refusing to report a clean result."
    }
    Write-Host "== $full ($($dlls.Count) imported DLLs)"
    foreach ($dll in ($dlls | Sort-Object -Unique)) {
        if (Test-DllAllowed $dll) {
            Write-Host "   ok       $dll"
        } else {
            Write-Host "   DISALLOWED $dll"
            $offenders.Add("$([System.IO.Path]::GetFileName($full)): $dll")
        }
    }
    $checked++
}

if ($offenders.Count -gt 0) {
    Write-Host "FAIL: $($offenders.Count) disallowed import(s) in $checked file(s):"
    $offenders | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "PASS: $checked file(s) import only allowlisted DLLs"
exit 0
