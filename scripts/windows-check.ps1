# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
    One-shot Windows validation kit for hipfire. Run it after scripts\install.ps1
    and attach the two files it writes (report.json and report.txt) to a GitHub
    issue.

.DESCRIPTION
    What it does:
      1. Collects system facts: Windows version, CPU, RAM, GPU adapters, AMD
         driver version, HIP SDK version, every amdhip64*.dll it can find (to
         show whether the driver's older copy in System32 shadows the SDK's),
         hipInfo output, hipfire --version / version --json / diag --json, the
         installed kernel pack manifest, and `hipfire doctor` where that
         command exists.
      2. Unless -SkipModel is given, runs a short smoke test with the smallest
         Qwen3.5 text model: `hipfire pull`, one greedy generation, one short
         `hipfire bench`. Each step has a timeout; a hung step is killed.
      3. With -Include27B (opt-in), and only when at least ~23 GiB of VRAM is
         detected, repeats the pull / generate / bench steps for the 27B tag
         qwen3.8:27b-mq4-xts.
      4. Writes report.json and report.txt and prints where they are.

    What it does NOT do: change any system setting. No registry writes, no
    PATH or environment persistence, no driver or service changes, no
    execution-policy change, no administrator rights needed. It only reads
    system state, runs hipfire / hipInfo as child processes, and writes files
    under the output folder (plus whatever `hipfire pull` downloads into
    %USERPROFILE%\.hipfire\models).

    DOWNLOAD AND DISK USE: unless -SkipModel is given, the script runs
    `hipfire pull`, which downloads a model. Smoke model qwen3.5:0.8b is
    0.55 GB. With -Include27B, qwen3.8:27b-mq4-xts is 14.99 GB plus about
    1.2 GB of sidecars (vision tower 0.93 GB, MTP head 0.23 GB; the draft
    sidecar is skipped with --no-draft). The report folder itself is small
    (a few hundred KB to a few MB).

    PRIVACY: the Windows user name, the user profile path and the computer name
    are replaced in every file the script writes (<user>, %USERPROFILE%,
    <host>). Everything else (GPU, driver, versions, hipfire output) is
    included as-is. Read the files before attaching them if in doubt.

    Model choice: qwen3.5:0.8b is the smallest Qwen3.5 text tag in the
    registry (size_gb 0.55: registry/models.json:4-7, docs/MODELS.md:49; the
    next smallest text tag, qwen3.5:0.8b-mq6, is 0.67 GB, docs/MODELS.md:50;
    the smaller-looking qwen3.5:9b-draft is a draft sidecar, not a text
    model, docs/MODELS.md:288). The 27B tag qwen3.8:27b-mq4-xts is
    registry/models.json:936 (size_gb 14.99) and docs/MODELS.md:99.

    Works on Windows PowerShell 5.1 and PowerShell 7. ASCII only.

.PARAMETER OutDir
    Folder for report.json, report.txt and the per-step stdout/stderr files.
    Default: %USERPROFILE%\hipfire-windows-report-<yyyyMMdd-HHmmss>

.PARAMETER SkipModel
    Diagnostics only: do not pull a model and do not run generation or bench.

.PARAMETER Include27B
    Also test qwen3.8:27b-mq4-xts, but only if the detected VRAM is at least
    ~23 GiB (a 24 GB card reports slightly less than 24 GiB). Otherwise the
    reason it was skipped is printed and recorded.

.PARAMETER HipfireExe
    Path or command name of hipfire. Default: `hipfire` on PATH, else
    %HIPFIRE_HOME%\bin\hipfire.exe, else %USERPROFILE%\.hipfire\bin\hipfire.exe.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows-check.ps1

.EXAMPLE
    .\scripts\windows-check.ps1 -SkipModel
#>
[CmdletBinding()]
param(
    [string]$OutDir = '',
    [switch]$SkipModel,
    [switch]$Include27B,
    [string]$HipfireExe = ''
)

$ErrorActionPreference = 'Continue'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
$script:ScriptVersion = '1.0.0'
$script:SchemaVersion = 1
$script:IssueUrl = 'https://github.com/warpfront/hipfire/issues/new?template=windows-test-report.yml'

# Smoke model: smallest Qwen3.5 text tag (see header). 27B tag: opt-in.
$script:SmokeTag = 'qwen3.5:0.8b'
$script:SmokeSizeGb = 0.55
$script:Tag27B = 'qwen3.8:27b-mq4-xts'
$script:Size27BGb = 16.2        # 14.99 model + 0.93 vision + 0.23 MTP head
# A 24 GB card reports a little under 24 GiB, so gate on 23 GiB (in MiB).
$script:Min27BVramMb = 23 * 1024
$script:SmokePrompt = 'Count from 1 to 10.'
$script:SmokeMaxTokens = 64

$script:Errors = New-Object System.Collections.ArrayList
$script:Steps = New-Object System.Collections.ArrayList
$script:StepSeq = 0
$script:OutDirPath = ''

# ---------------------------------------------------------------------------
# Helpers: redaction, file output, errors
# ---------------------------------------------------------------------------

# Replace the user's name / profile path / computer name with placeholders.
function Protect-Text {
    param([string]$Text)
    if ([string]::IsNullOrEmpty($Text)) { return $Text }
    $opts = [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    $out = $Text
    $profilePath = $env:USERPROFILE
    if ($profilePath -and $profilePath.Length -ge 4) {
        # Plain, forward-slash and JSON-escaped (doubled backslash) spellings.
        $variants = @($profilePath.Replace('\', '\\'), $profilePath, $profilePath.Replace('\', '/'))
        foreach ($v in $variants) {
            $out = [regex]::Replace($out, [regex]::Escape($v), '%USERPROFILE%', $opts)
        }
    }
    if ($env:USERNAME -and $env:USERNAME.Length -ge 3) {
        $pattern = '(?<![A-Za-z0-9])' + [regex]::Escape($env:USERNAME) + '(?![A-Za-z0-9])'
        $out = [regex]::Replace($out, $pattern, '<user>', $opts)
    }
    if ($env:COMPUTERNAME -and $env:COMPUTERNAME.Length -ge 3) {
        $pattern = '(?<![A-Za-z0-9])' + [regex]::Escape($env:COMPUTERNAME) + '(?![A-Za-z0-9])'
        $out = [regex]::Replace($out, $pattern, '<host>', $opts)
    }
    return $out
}

# Write text as UTF-8 without a byte order mark (identical on 5.1 and 7).
function Write-Utf8File {
    param([string]$Path, [string]$Text)
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Read-Utf8File {
    param([string]$Path)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return '' }
    $text = [System.IO.File]::ReadAllText($Path, (New-Object System.Text.UTF8Encoding($false)))
    if ($null -eq $text) { return '' }
    return $text
}

# Redact a captured output file in place.
function Protect-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Write-Utf8File -Path $Path -Text (Protect-Text (Read-Utf8File -Path $Path))
}

function Add-ReportError {
    param([string]$Section, [string]$Message)
    [void]$script:Errors.Add([ordered]@{
        section = $Section
        message = (Protect-Text $Message)
    })
    Write-Host ("  [error] {0}: {1}" -f $Section, (Protect-Text $Message)) -ForegroundColor Yellow
}

# Run one collection section; any exception becomes a report error, not an abort.
function Invoke-Section {
    param([string]$Name, [scriptblock]$Body)
    try {
        return (& $Body)
    } catch {
        Add-ReportError -Section $Name -Message $_.Exception.Message
        return $null
    }
}

# Walk nested properties of a parsed-JSON object; $null when anything is missing.
function Get-Prop {
    param($Obj, [string[]]$Path)
    $cur = $Obj
    foreach ($key in $Path) {
        if ($null -eq $cur) { return $null }
        $prop = $cur.PSObject.Properties[$key]
        if ($null -eq $prop) { return $null }
        $cur = $prop.Value
    }
    return $cur
}

# ConvertFrom-Json that returns $null instead of throwing, and tolerates log
# lines around a JSON object.
function ConvertFrom-JsonSafe {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return $null }
    try { return ($Text | ConvertFrom-Json) } catch { }
    $first = $Text.IndexOf('{')
    $last = $Text.LastIndexOf('}')
    if ($first -ge 0 -and $last -gt $first) {
        try { return ($Text.Substring($first, $last - $first + 1) | ConvertFrom-Json) } catch { }
    }
    return $null
}

function Format-Date {
    param($Value)
    if ($null -eq $Value) { return $null }
    try { return ([datetime]$Value).ToString('yyyy-MM-dd') } catch { return [string]$Value }
}

# ---------------------------------------------------------------------------
# Helpers: running child processes with a timeout
# ---------------------------------------------------------------------------

# Quote an argument list into one command-line string (Windows rules), because
# Start-Process on 5.1 joins an array without quoting.
function ConvertTo-ArgString {
    param([string[]]$ArgList)
    $parts = New-Object System.Collections.ArrayList
    foreach ($a in $ArgList) {
        if ($null -eq $a -or $a.Length -eq 0) {
            [void]$parts.Add('""')
        } elseif ($a -notmatch '[\s"]') {
            [void]$parts.Add($a)
        } else {
            $s = $a -replace '(\\*)"', '$1$1\"'
            $s = $s -replace '(\\+)$', '$1$1'
            [void]$parts.Add('"' + $s + '"')
        }
    }
    return ($parts -join ' ')
}

# Kill a process and its children (hipfire spawns a daemon child process).
function Stop-ProcessTree {
    param([int]$ProcessId)
    $taskkill = $null
    if ($env:SystemRoot) { $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe' }
    if ($taskkill -and (Test-Path -LiteralPath $taskkill)) {
        & $taskkill /PID $ProcessId /T /F *> $null
    } else {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

# Run FilePath with stdout/stderr redirected to files under the output folder,
# wait at most TimeoutSec, kill on timeout. Always returns a step record.
function Invoke-Step {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$ArgumentList = @(),
        [int]$TimeoutSec = 300
    )
    $script:StepSeq++
    $prefix = '{0:D2}-{1}' -f $script:StepSeq, $Name
    $stdoutName = $prefix + '.stdout.txt'
    $stderrName = $prefix + '.stderr.txt'
    $stdoutPath = Join-Path $script:OutDirPath $stdoutName
    $stderrPath = Join-Path $script:OutDirPath $stderrName
    $argString = ConvertTo-ArgString -ArgList $ArgumentList

    $step = [ordered]@{
        name        = $Name
        command     = (Protect-Text (("{0} {1}" -f $FilePath, $argString).Trim()))
        exit_code   = $null
        timed_out   = $false
        seconds     = $null
        stdout_file = $null
        stderr_file = $null
        error       = $null
    }

    Write-Host ("  running: {0}" -f $step.command)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $FilePath) { throw 'no executable path' }
        $startArgs = @{
            FilePath               = $FilePath
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError  = $stderrPath
            NoNewWindow            = $true
            PassThru               = $true
        }
        if ($argString) { $startArgs['ArgumentList'] = $argString }
        $proc = Start-Process @startArgs
        # Touching Handle keeps ExitCode readable on Windows PowerShell 5.1.
        $null = $proc.Handle
        if ($proc.WaitForExit($TimeoutSec * 1000)) {
            $proc.WaitForExit()   # flush redirected streams
            $step.exit_code = $proc.ExitCode
        } else {
            $step.timed_out = $true
            $step.error = ('timed out after {0} s; process tree killed' -f $TimeoutSec)
            Stop-ProcessTree -ProcessId $proc.Id
            [void]$proc.WaitForExit(15000)
        }
    } catch {
        $step.error = Protect-Text $_.Exception.Message
    }
    $sw.Stop()
    $step.seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)

    foreach ($pair in @(@($stdoutPath, $stdoutName, 'stdout_file'), @($stderrPath, $stderrName, 'stderr_file'))) {
        if (Test-Path -LiteralPath $pair[0]) {
            try { Protect-File -Path $pair[0] } catch { }
            $step[$pair[2]] = $pair[1]
        }
    }
    [void]$script:Steps.Add($step)
    $status = 'ok'
    if ($step.timed_out -or $step.error -or ($null -ne $step.exit_code -and $step.exit_code -ne 0)) { $status = 'FAILED' }
    Write-Host ("    -> {0}, exit {1}, {2} s" -f $status, $step.exit_code, $step.seconds)
    return $step
}

function Get-StepText {
    param($Step, [string]$Stream)
    if ($null -eq $Step) { return '' }
    $file = $Step[$Stream + '_file']
    if (-not $file) { return '' }
    return (Read-Utf8File -Path (Join-Path $script:OutDirPath $file))
}

function Test-StepOk {
    param($Step)
    if ($null -eq $Step) { return $false }
    return ((-not $Step.timed_out) -and (-not $Step.error) -and ($Step.exit_code -eq 0))
}

# ---------------------------------------------------------------------------
# Collectors (read-only)
# ---------------------------------------------------------------------------

function Get-HipfireHome {
    if ($env:HIPFIRE_HOME) { return $env:HIPFIRE_HOME }
    return (Join-Path $env:USERPROFILE '.hipfire')
}

function Resolve-HipfireExe {
    param([string]$Requested)
    if ($Requested) {
        if (Test-Path -LiteralPath $Requested -PathType Leaf) {
            return (Resolve-Path -LiteralPath $Requested).Path
        }
        $named = Get-Command $Requested -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($named) { return $named.Source }
        return $null
    }
    $cmd = Get-Command 'hipfire' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path (Get-HipfireHome) 'bin\hipfire.exe'
    if (Test-Path -LiteralPath $fallback -PathType Leaf) { return $fallback }
    $fallback2 = Join-Path $env:USERPROFILE '.hipfire\bin\hipfire.exe'
    if (Test-Path -LiteralPath $fallback2 -PathType Leaf) { return $fallback2 }
    return $null
}

function Get-OsInfo {
    $info = [ordered]@{}
    $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
    $info.caption = $os.Caption
    $info.version = $os.Version
    $info.build_number = $os.BuildNumber
    $info.architecture = $os.OSArchitecture
    $info.environment_osversion = [string][Environment]::OSVersion.VersionString
    $info.is_64bit_os = [Environment]::Is64BitOperatingSystem
    $info.powershell_version = [string]$PSVersionTable.PSVersion
    $info.powershell_edition = [string]$PSVersionTable.PSEdition
    # Read-only registry lookup for the feature-update name and patch level.
    try {
        $cv = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction Stop
        $info.display_version = $cv.DisplayVersion
        $info.ubr = $cv.UBR
    } catch {
        $info.display_version = $null
        $info.ubr = $null
    }
    return $info
}

function Get-CpuInfo {
    $list = @()
    foreach ($c in @(Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop)) {
        $list += [ordered]@{
            name               = ([string]$c.Name).Trim()
            cores              = $c.NumberOfCores
            logical_processors = $c.NumberOfLogicalProcessors
            max_clock_mhz      = $c.MaxClockSpeed
        }
    }
    return ,$list
}

function Get-RamInfo {
    $cs = Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction Stop
    $bytes = [double]$cs.TotalPhysicalMemory
    return [ordered]@{
        total_bytes = $cs.TotalPhysicalMemory
        total_gib   = [math]::Round($bytes / 1GB, 1)
    }
}

# Win32_VideoController.AdapterRAM is a 32-bit field and caps at 4 GB, so it
# is reported but not trusted; see Get-DisplayClassMemory for the real size.
function Get-GpuAdapters {
    $list = @()
    foreach ($g in @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop)) {
        $list += [ordered]@{
            name                  = $g.Name
            driver_version        = $g.DriverVersion
            driver_date           = (Format-Date $g.DriverDate)
            adapter_ram_bytes_32  = $g.AdapterRAM
            video_processor       = $g.VideoProcessor
            pnp_device_id         = $g.PNPDeviceID
            status                = $g.Status
        }
    }
    return ,$list
}

# READ-ONLY look at the display class registry key for the 64-bit memory size.
function Get-DisplayClassMemory {
    $classKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11cd-be10-08002be10318}'
    $list = @()
    foreach ($k in @(Get-ChildItem -LiteralPath $classKey -ErrorAction Stop)) {
        if ($k.PSChildName -notmatch '^\d{4}$') { continue }
        $p = Get-ItemProperty -LiteralPath $k.PSPath -ErrorAction SilentlyContinue
        if ($null -eq $p) { continue }
        $desc = $p.DriverDesc
        if (-not $desc) { continue }
        $raw = $p.'HardwareInformation.qwMemorySize'
        $bytes = $null
        if ($raw -is [byte[]]) {
            if ($raw.Length -ge 8) { $bytes = [BitConverter]::ToUInt64($raw, 0) }
        } elseif ($null -ne $raw) {
            try { $bytes = [uint64]$raw } catch { $bytes = $null }
        }
        $list += [ordered]@{
            key             = $k.PSChildName
            driver_desc     = $desc
            driver_version  = $p.DriverVersion
            provider        = $p.ProviderName
            memory_bytes    = $bytes
            memory_mib      = $(if ($null -ne $bytes) { [math]::Round([double]$bytes / 1MB, 0) } else { $null })
        }
    }
    return ,$list
}

function Get-EnvVars {
    $list = [ordered]@{}
    foreach ($e in @(Get-ChildItem -Path Env: | Sort-Object Name)) {
        if ($e.Name -match '^((HIP|HIPFIRE|ROCM|HSA|ROCR|AMD|GPU)_|CUDA_VISIBLE_DEVICES$)') {
            if ($e.Name -match '(TOKEN|KEY|SECRET|PASSWORD)') {
                $list[$e.Name] = '<redacted>'
            } else {
                $list[$e.Name] = (Protect-Text $e.Value)
            }
        }
    }
    return $list
}

function Read-VersionFile {
    param([string]$Dir)
    $f = Join-Path $Dir '.info\version'
    if (Test-Path -LiteralPath $f -PathType Leaf) {
        return ((Get-Content -LiteralPath $f -TotalCount 1) -join '').Trim()
    }
    return $null
}

function Get-HipSdkInfo {
    $info = [ordered]@{}
    $info.hip_path = (Protect-Text $env:HIP_PATH)
    $info.hip_path_version_file = $null
    if ($env:HIP_PATH) { $info.hip_path_version_file = Read-VersionFile -Dir $env:HIP_PATH }
    $programFiles = $env:ProgramFiles
    if (-not $programFiles) { $programFiles = 'C:\Program Files' }
    $root = Join-Path $programFiles 'AMD\ROCm'
    $info.rocm_root = (Protect-Text $root)
    $info.rocm_root_exists = (Test-Path -LiteralPath $root -PathType Container)
    $folders = @()
    if ($info.rocm_root_exists) {
        foreach ($d in @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue)) {
            $folders += [ordered]@{ name = $d.Name; version_file = (Read-VersionFile -Dir $d.FullName) }
        }
    }
    $info.rocm_folders = $folders
    return $info
}

function Get-FileEvidence {
    param([string]$Label, [System.IO.FileInfo]$File)
    $sha = $null
    try { $sha = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256 -ErrorAction Stop).Hash } catch { }
    $vi = $File.VersionInfo
    return [ordered]@{
        location        = $Label
        path            = (Protect-Text $File.FullName)
        file_version    = $vi.FileVersion
        product_version = $vi.ProductVersion
        company         = $vi.CompanyName
        size_bytes      = $File.Length
        last_write_utc  = $File.LastWriteTimeUtc.ToString('yyyy-MM-dd HH:mm:ss')
        sha256          = $sha
    }
}

# Every amdhip64*.dll in the places the loader or hipfire could pick one up.
# A stale copy in System32 shadowing the one hipfire ships is the thing to spot.
function Get-HipRuntimeDlls {
    param([string]$ExePath)
    $home1 = Get-HipfireHome
    $dirs = New-Object System.Collections.ArrayList
    if ($env:HIP_PATH) { [void]$dirs.Add(@('HIP_PATH\bin', (Join-Path $env:HIP_PATH 'bin'))) }
    if ($env:SystemRoot) { [void]$dirs.Add(@('System32', (Join-Path $env:SystemRoot 'System32'))) }
    if ($ExePath) { [void]$dirs.Add(@('hipfire exe dir', (Split-Path -Parent $ExePath))) }
    [void]$dirs.Add(@('hipfire bin', (Join-Path $home1 'bin')))
    [void]$dirs.Add(@('hipfire runtime', (Join-Path $home1 'runtime')))

    $seen = @{}
    $found = @()
    foreach ($d in $dirs) {
        $dir = $d[1]
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { continue }
        foreach ($f in @(Get-ChildItem -LiteralPath $dir -Filter 'amdhip64*.dll' -File -ErrorAction SilentlyContinue)) {
            $key = $f.FullName.ToLowerInvariant()
            if ($seen.ContainsKey($key)) { continue }
            $seen[$key] = $true
            $found += (Get-FileEvidence -Label $d[0] -File $f)
        }
    }

    # Which PATH entries hold an amdhip64*.dll, in PATH order.
    $pathHits = @()
    $idx = 0
    foreach ($entry in @($env:PATH -split ';')) {
        $idx++
        if (-not $entry) { continue }
        if (-not (Test-Path -LiteralPath $entry -PathType Container -ErrorAction SilentlyContinue)) { continue }
        $names = @(Get-ChildItem -LiteralPath $entry -Filter 'amdhip64*.dll' -File -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
        if ($names.Count -gt 0) {
            $pathHits += [ordered]@{ path_index = $idx; directory = (Protect-Text $entry); files = $names }
        }
    }
    return [ordered]@{ dlls = $found; path_entries_with_amdhip64 = $pathHits }
}

function Get-HipInfo {
    $exe = $null
    if ($env:HIP_PATH) { $exe = Join-Path $env:HIP_PATH 'bin\hipInfo.exe' }
    if (-not $exe -or -not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        return [ordered]@{ present = $false; note = 'hipInfo.exe not found under HIP_PATH\bin'; step = $null; summary_lines = @() }
    }
    $step = Invoke-Step -Name 'hipinfo' -FilePath $exe -TimeoutSec 60
    $lines = @()
    foreach ($l in @((Get-StepText -Step $step -Stream 'stdout') -split "`r?`n")) {
        if ($l -match '^\s*(name|gcnArchName|totalGlobalMem|pciBusID|pciDeviceID)\s*:') { $lines += $l.Trim() }
    }
    return [ordered]@{ present = $true; step = $step; summary_lines = $lines }
}

function Get-KernelPackManifests {
    param([string]$ExePath)
    # The installer writes <compiled>\<arch>.manifest.json next to the <arch>
    # directory (rdna-compute kernel_pack.rs installed_manifest_path).
    $dirs = New-Object System.Collections.ArrayList
    [void]$dirs.Add((Join-Path (Get-HipfireHome) 'bin\kernels\compiled'))
    if ($ExePath) { [void]$dirs.Add((Join-Path (Split-Path -Parent $ExePath) 'kernels\compiled')) }
    $seen = @{}
    $list = @()
    foreach ($dir in $dirs) {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { continue }
        foreach ($f in @(Get-ChildItem -LiteralPath $dir -Filter '*manifest*.json' -File -ErrorAction SilentlyContinue)) {
            $key = $f.FullName.ToLowerInvariant()
            if ($seen.ContainsKey($key)) { continue }
            $seen[$key] = $true
            $parsed = $null
            try { $parsed = (Read-Utf8File -Path $f.FullName) | ConvertFrom-Json } catch { }
            $list += [ordered]@{ path = (Protect-Text $f.FullName); manifest = $parsed }
        }
        $archDirs = @(Get-ChildItem -LiteralPath $dir -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
        if ($archDirs.Count -gt 0) {
            $list += [ordered]@{ path = (Protect-Text $dir); arch_directories = $archDirs }
        }
    }
    return ,$list
}

function Get-FreeDiskGb {
    param([string]$Path)
    $root = [System.IO.Path]::GetPathRoot($Path)
    if (-not $root) { return $null }
    $drive = New-Object System.IO.DriveInfo($root)
    return [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
}

# ---------------------------------------------------------------------------
# Model smoke test
# ---------------------------------------------------------------------------

# pull -> greedy run -> short bench for one tag. Returns a result record.
function Invoke-ModelTest {
    param(
        [string]$Exe,
        [string]$Tag,
        [int]$PullTimeoutSec,
        [int]$RunTimeoutSec,
        [int]$BenchTimeoutSec,
        [int]$BenchRuns
    )
    $res = [ordered]@{
        tag             = $Tag
        skipped         = $false
        skip_reason     = $null
        pull            = $null
        run             = $null
        bench           = $null
        run_result      = $null
        bench_result    = $null
    }

    # hipfire pull: crates/hipfire-cli/src/main.rs PullArgs (--yes, --no-draft).
    $res.pull = Invoke-Step -Name ('pull-' + ($Tag -replace '[^A-Za-z0-9]+', '_')) -FilePath $Exe `
        -ArgumentList @('pull', $Tag, '--yes', '--no-draft') -TimeoutSec $PullTimeoutSec
    if (-not (Test-StepOk $res.pull)) {
        $res.skipped = $true
        $res.skip_reason = 'pull failed; run and bench not attempted'
        return $res
    }

    # hipfire run: RunArgs --temp 0 (greedy), --max-tokens, --json.
    $res.run = Invoke-Step -Name ('run-' + ($Tag -replace '[^A-Za-z0-9]+', '_')) -FilePath $Exe `
        -ArgumentList @('run', $Tag, '--temp', '0', '--max-tokens', [string]$script:SmokeMaxTokens, '--json', $script:SmokePrompt) `
        -TimeoutSec $RunTimeoutSec
    $json = ConvertFrom-JsonSafe -Text (Get-StepText -Step $res.run -Stream 'stdout')
    if ($null -ne $json) {
        $content = [string](Get-Prop $json @('content'))
        # Heuristic only: the digits 1..10 appear in order.
        $ordered = $true
        $pos = 0
        for ($n = 1; $n -le 10; $n++) {
            $m = [regex]::Match($content.Substring($pos), ('(?<!\d)' + $n + '(?!\d)'))
            if (-not $m.Success) { $ordered = $false; break }
            $pos += $m.Index + $m.Length
        }
        $res.run_result = [ordered]@{
            content            = (Protect-Text $content)
            tokens             = (Get-Prop $json @('tokens'))
            tok_s              = (Get-Prop $json @('tok_s'))
            finish_reason      = (Get-Prop $json @('finish_reason'))
            digits_1_to_10_in_order_heuristic = $ordered
        }
    }

    # hipfire bench: BenchArgs --runs, --warmups, --max-tokens, --json.
    $res.bench = Invoke-Step -Name ('bench-' + ($Tag -replace '[^A-Za-z0-9]+', '_')) -FilePath $Exe `
        -ArgumentList @('bench', $Tag, '--runs', [string]$BenchRuns, '--warmups', '1', '--max-tokens', [string]$script:SmokeMaxTokens, '--json') `
        -TimeoutSec $BenchTimeoutSec
    $bj = ConvertFrom-JsonSafe -Text (Get-StepText -Step $res.bench -Stream 'stdout')
    if ($null -ne $bj) {
        $res.bench_result = [ordered]@{
            protocol           = (Get-Prop $bj @('protocol'))
            runs               = (Get-Prop $bj @('runs'))
            max_tokens         = (Get-Prop $bj @('max_tokens'))
            prompt_tokens      = (Get-Prop $bj @('prompt_tokens'))
            decode_tok_s_median  = (Get-Prop $bj @('decode_tok_s', 'median'))
            prefill_tok_s_median = (Get-Prop $bj @('prefill_tok_s', 'median'))
            wall_tok_s_median    = (Get-Prop $bj @('wall_tok_s', 'median'))
            ttft_ms_median       = (Get-Prop $bj @('ttft_ms', 'median'))
            warnings             = @(Get-Prop $bj @('warnings') | Where-Object { $null -ne $_ })
        }
    }
    return $res
}

# ---------------------------------------------------------------------------
# Report text
# ---------------------------------------------------------------------------

function Format-StepLine {
    param($Step)
    if ($null -eq $Step) { return '  (not run)' }
    $status = 'ok'
    if ($Step.timed_out) { $status = 'TIMEOUT' }
    elseif ($Step.error) { $status = 'ERROR' }
    elseif ($null -ne $Step.exit_code -and $Step.exit_code -ne 0) { $status = 'FAILED' }
    $line = '  [{0}] {1}  exit={2}  {3}s' -f $status, $Step.name, $Step.exit_code, $Step.seconds
    if ($Step.error) { $line += '  (' + $Step.error + ')' }
    return $line
}

function Format-ModelSection {
    param($Label, $Result)
    $L = New-Object System.Collections.ArrayList
    [void]$L.Add('')
    [void]$L.Add($Label)
    if ($null -eq $Result) { [void]$L.Add('  not run'); return $L }
    if ($Result.Contains('skipped_before_start') -and $Result.skipped_before_start) {
        [void]$L.Add('  skipped: ' + $Result.skip_reason)
        return $L
    }
    [void]$L.Add((Format-StepLine $Result.pull))
    if ($Result.skipped) { [void]$L.Add('  ' + $Result.skip_reason); return $L }
    [void]$L.Add((Format-StepLine $Result.run))
    $rr = $Result.run_result
    if ($null -ne $rr) {
        [void]$L.Add(('    tokens={0}  tok/s={1}  finish={2}  1..10-in-order heuristic={3}' -f $rr.tokens, $rr.tok_s, $rr.finish_reason, $rr.digits_1_to_10_in_order_heuristic))
        $c = [string]$rr.content
        if ($c.Length -gt 600) { $c = $c.Substring(0, 600) + ' ...' }
        [void]$L.Add('    output (judge for yourself whether it is sensible text):')
        foreach ($cl in @($c -split "`r?`n")) { [void]$L.Add('      | ' + $cl) }
    } else {
        [void]$L.Add('    (no JSON result parsed from hipfire run; see the stdout/stderr files)')
    }
    [void]$L.Add((Format-StepLine $Result.bench))
    $br = $Result.bench_result
    if ($null -ne $br) {
        [void]$L.Add(('    decode tok/s (median)={0}  prefill tok/s (median)={1}  ttft ms={2}  prompt_tokens={3}' -f $br.decode_tok_s_median, $br.prefill_tok_s_median, $br.ttft_ms_median, $br.prompt_tokens))
    } else {
        [void]$L.Add('    (no JSON result parsed from hipfire bench; see the stdout/stderr files)')
    }
    return $L
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function Main {
    if ($env:OS -ne 'Windows_NT') {
        Write-Host 'Warning: this does not look like Windows; most checks will report errors.' -ForegroundColor Yellow
    }

    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    if (-not $OutDir) { $OutDir = Join-Path $env:USERPROFILE ('hipfire-windows-report-' + $stamp) }
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    $script:OutDirPath = (Resolve-Path -LiteralPath $OutDir).Path

    Write-Host ''
    Write-Host ('hipfire Windows check v{0}' -f $script:ScriptVersion) -ForegroundColor Cyan
    Write-Host ('Report folder: {0}' -f $script:OutDirPath)
    Write-Host 'This script only READS system state. It changes no settings, needs no admin rights,'
    Write-Host 'and writes only under the report folder (plus models downloaded by hipfire).'
    if ($SkipModel) {
        Write-Host 'Mode: -SkipModel, diagnostics only, nothing will be downloaded.'
    } else {
        Write-Host ('NOTE: this will download a model with "hipfire pull": {0} is {1} GB.' -f $script:SmokeTag, $script:SmokeSizeGb)
        if ($Include27B) {
            Write-Host ('      -Include27B: {0} is about {1} GB more (only if >= 23 GiB VRAM is detected).' -f $script:Tag27B, $script:Size27BGb)
        }
        Write-Host '      Use -SkipModel to skip downloads and generation. Ctrl+C within 5 seconds cancels.'
        Start-Sleep -Seconds 5
    }
    Write-Host 'The Windows user name, profile path and computer name are replaced in all output files.'
    Write-Host ''

    $report = [ordered]@{
        schema_version = $script:SchemaVersion
        script_version = $script:ScriptVersion
        timestamp      = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        redaction      = 'user name -> <user>, user profile path -> %USERPROFILE%, computer name -> <host>'
        parameters     = [ordered]@{
            skip_model  = [bool]$SkipModel
            include_27b = [bool]$Include27B
            out_dir     = (Protect-Text $script:OutDirPath)
        }
    }

    # --- System ---
    Write-Host '[1/6] System, GPU and HIP SDK facts' -ForegroundColor Cyan
    $report.os                   = Invoke-Section 'os' { Get-OsInfo }
    $report.cpu                  = @(Invoke-Section 'cpu' { Get-CpuInfo } | Where-Object { $null -ne $_ })
    $report.ram                  = Invoke-Section 'ram' { Get-RamInfo }
    $report.gpu_adapters         = @(Invoke-Section 'gpu_adapters' { Get-GpuAdapters } | Where-Object { $null -ne $_ })
    $report.gpu_memory_registry  = @(Invoke-Section 'gpu_memory_registry' { Get-DisplayClassMemory } | Where-Object { $null -ne $_ })
    $report.environment          = Invoke-Section 'environment' { Get-EnvVars }
    $report.hip_sdk              = Invoke-Section 'hip_sdk' { Get-HipSdkInfo }
    $report.hipinfo              = Invoke-Section 'hipinfo' { Get-HipInfo }

    # --- hipfire ---
    Write-Host '[2/6] Locating hipfire' -ForegroundColor Cyan
    $exe = Invoke-Section 'hipfire_exe' { Resolve-HipfireExe -Requested $HipfireExe }
    if ($exe) {
        Write-Host ('  hipfire: {0}' -f (Protect-Text $exe))
    } else {
        Add-ReportError -Section 'hipfire_exe' -Message 'hipfire was not found (not on PATH, not in the .hipfire\bin folder). Run scripts\install.ps1 first or pass -HipfireExe.'
    }
    $report.hipfire = [ordered]@{ exe = (Protect-Text $exe) }

    $report.hip_runtime_dlls     = Invoke-Section 'hip_runtime_dlls' { Get-HipRuntimeDlls -ExePath $exe }
    $report.kernel_pack          = @(Invoke-Section 'kernel_pack' { Get-KernelPackManifests -ExePath $exe } | Where-Object { $null -ne $_ })

    Write-Host '[3/6] hipfire version, diag, doctor' -ForegroundColor Cyan
    $diagSummary = $null
    if ($exe) {
        # `hipfire --version` (clap version flag) and `hipfire version --json`
        # (Commands::Version, crates/hipfire-cli/src/main.rs).
        $report.hipfire.version_step      = Invoke-Step -Name 'version-flag' -FilePath $exe -ArgumentList @('--version') -TimeoutSec 60
        $report.hipfire.version_step_json = Invoke-Step -Name 'version-json' -FilePath $exe -ArgumentList @('version', '--json') -TimeoutSec 120
        $report.hipfire.version_text = (Protect-Text ((Get-StepText -Step $report.hipfire.version_step -Stream 'stdout').Trim()))
        $report.hipfire.version      = ConvertFrom-JsonSafe -Text (Get-StepText -Step $report.hipfire.version_step_json -Stream 'stdout')

        # `hipfire diag --json` (Commands::Diag + OutputArgs --json). This
        # starts the daemon, so it exercises HIP loading on the GPU.
        $diagStep = Invoke-Step -Name 'diag-json' -FilePath $exe -ArgumentList @('diag', '--json') -TimeoutSec 180
        $diag = ConvertFrom-JsonSafe -Text (Get-StepText -Step $diagStep -Stream 'stdout')
        $live = Get-Prop $diag @('live_gpu')
        $diagSummary = [ordered]@{
            step                     = $diagStep
            parsed                   = ($null -ne $diag)
            platform                 = (Get-Prop $diag @('platform'))
            live_gpu_present         = ($null -ne $live)
            live_gpu_arch            = (Get-Prop $live @('arch'))
            live_gpu_hip_version     = (Get-Prop $live @('hip_version'))
            live_gpu_vram_total_mb   = (Get-Prop $live @('vram_total_mb'))
            live_gpu_vram_free_mb    = (Get-Prop $live @('vram_free_mb'))
            hip_runtime_library      = (Protect-Text ([string](Get-Prop $live @('hip_runtime_library'))))
            hip_runtime_version_raw  = (Get-Prop $live @('hip_runtime_version_raw'))
            hip_driver_version_raw   = (Get-Prop $live @('hip_driver_version_raw'))
            gpu_memory_budget        = (Get-Prop $diag @('gpu_memory_budget'))
            rocm                     = (Get-Prop $diag @('rocm'))
            warnings                 = @(Get-Prop $diag @('warnings') | Where-Object { $null -ne $_ })
        }
        if (-not $diagSummary.hip_runtime_library) { $diagSummary.hip_runtime_library = $null }
        if ($null -eq $diag) {
            Add-ReportError -Section 'diag' -Message 'hipfire diag --json produced no parseable JSON; see the diag-json stdout/stderr files.'
        } elseif ($null -eq $live) {
            Add-ReportError -Section 'diag' -Message 'diag has no live_gpu: the daemon did not answer, so HIP/GPU init likely failed (see the diag-json files).'
        }

        # `hipfire doctor` exists only where the build registers it (the
        # Commands::Doctor variant is cfg(target_os = "linux") in
        # crates/hipfire-cli/src/main.rs). Ask --help first, then use its
        # real --json flag (doctor/mod.rs DoctorArgs) only if it is listed.
        $helpStep = Invoke-Step -Name 'help' -FilePath $exe -ArgumentList @('--help') -TimeoutSec 60
        $helpText = Get-StepText -Step $helpStep -Stream 'stdout'
        $hasDoctor = ($helpText -match '(?m)^\s+doctor\s')
        $doctor = [ordered]@{ available = $hasDoctor; step = $null; parsed = $null; note = $null }
        if ($hasDoctor) {
            $doctor.step = Invoke-Step -Name 'doctor-json' -FilePath $exe -ArgumentList @('doctor', '--json') -TimeoutSec 180
            $doctor.parsed = ConvertFrom-JsonSafe -Text (Get-StepText -Step $doctor.step -Stream 'stdout')
        } else {
            $doctor.note = 'this hipfire build does not list a doctor command (Linux-only in the sources this kit was written against)'
            Write-Host '  doctor: not available in this build, skipped'
        }
        $report.hipfire.doctor = $doctor
    } else {
        $report.hipfire.note = 'hipfire not found; version/diag/doctor/model steps skipped'
    }
    $report.diag = $diagSummary

    # --- VRAM detection (for the 27B gate) ---
    $vramMb = $null
    $vramSource = $null
    if ($diagSummary -and $diagSummary.live_gpu_vram_total_mb) {
        $vramMb = [double]$diagSummary.live_gpu_vram_total_mb
        $vramSource = 'hipfire diag live_gpu.vram_total_mb (HIP)'
    }
    if ($null -eq $vramMb -and $diagSummary -and $diagSummary.gpu_memory_budget) {
        foreach ($a in @(Get-Prop $diagSummary.gpu_memory_budget @('adapters'))) {
            if ($null -eq $a -or -not (Get-Prop $a @('amd'))) { continue }
            $mb = [double](Get-Prop $a @('dedicated_video_memory_bytes')) / 1MB
            if ($null -eq $vramMb -or $mb -gt $vramMb) { $vramMb = $mb; $vramSource = 'hipfire diag gpu_memory_budget (DXGI dedicated)' }
        }
    }
    if ($null -eq $vramMb) {
        foreach ($m in @($report.gpu_memory_registry)) {
            if ($null -eq $m -or $null -eq $m.memory_mib) { continue }
            if ($m.driver_desc -notmatch 'AMD|Radeon') { continue }
            if ($null -eq $vramMb -or [double]$m.memory_mib -gt $vramMb) { $vramMb = [double]$m.memory_mib; $vramSource = 'display class registry qwMemorySize' }
        }
    }
    $report.detected_vram = [ordered]@{ mib = $vramMb; source = $vramSource }

    # --- Model tests ---
    $report.model_tests = [ordered]@{ skipped_all = [bool]$SkipModel; smoke = $null; large_27b = $null }
    if ($SkipModel) {
        Write-Host '[4/6] Model smoke test: skipped (-SkipModel)' -ForegroundColor Cyan
        Write-Host '[5/6] 27B test: skipped (-SkipModel)' -ForegroundColor Cyan
    } elseif (-not $exe) {
        Write-Host '[4/6] Model smoke test: skipped (hipfire not found)' -ForegroundColor Yellow
        Write-Host '[5/6] 27B test: skipped (hipfire not found)' -ForegroundColor Yellow
    } else {
        Write-Host ('[4/6] Model smoke test: {0}' -f $script:SmokeTag) -ForegroundColor Cyan
        $free = Invoke-Section 'disk_free' { Get-FreeDiskGb -Path (Get-HipfireHome) }
        $report.disk_free_gb_on_hipfire_drive = $free
        if ($null -ne $free -and $free -lt 2) {
            $report.model_tests.smoke = [ordered]@{ tag = $script:SmokeTag; skipped = $true; skipped_before_start = $true; skip_reason = ('only {0} GB free on the hipfire drive; need about 2 GB' -f $free) }
            Write-Host ('  skipped: only {0} GB free' -f $free) -ForegroundColor Yellow
        } else {
            $report.model_tests.smoke = Invoke-ModelTest -Exe $exe -Tag $script:SmokeTag `
                -PullTimeoutSec 1800 -RunTimeoutSec 600 -BenchTimeoutSec 900 -BenchRuns 3
        }

        Write-Host ('[5/6] 27B test: {0}' -f $script:Tag27B) -ForegroundColor Cyan
        if (-not $Include27B) {
            Write-Host '  skipped: not requested (pass -Include27B on a card with >= 24 GB)'
            $report.model_tests.large_27b = [ordered]@{ tag = $script:Tag27B; skipped = $true; skipped_before_start = $true; skip_reason = 'not requested (-Include27B not given)' }
        } elseif ($null -eq $vramMb) {
            $reason = 'VRAM could not be detected (no live_gpu in diag, no DXGI budget, no registry size)'
            Write-Host ('  skipped: {0}' -f $reason) -ForegroundColor Yellow
            $report.model_tests.large_27b = [ordered]@{ tag = $script:Tag27B; skipped = $true; skipped_before_start = $true; skip_reason = $reason }
        } elseif ($vramMb -lt $script:Min27BVramMb) {
            $reason = ('detected VRAM is {0:N0} MiB (< {1} MiB needed for the 27B test)' -f $vramMb, $script:Min27BVramMb)
            Write-Host ('  skipped: {0}' -f $reason) -ForegroundColor Yellow
            $report.model_tests.large_27b = [ordered]@{ tag = $script:Tag27B; skipped = $true; skipped_before_start = $true; skip_reason = $reason }
        } elseif ($null -ne $free -and $free -lt 20) {
            $reason = ('only {0} GB free on the hipfire drive; need about 20 GB' -f $free)
            Write-Host ('  skipped: {0}' -f $reason) -ForegroundColor Yellow
            $report.model_tests.large_27b = [ordered]@{ tag = $script:Tag27B; skipped = $true; skipped_before_start = $true; skip_reason = $reason }
        } else {
            $report.model_tests.large_27b = Invoke-ModelTest -Exe $exe -Tag $script:Tag27B `
                -PullTimeoutSec 5400 -RunTimeoutSec 1200 -BenchTimeoutSec 1800 -BenchRuns 2
        }
    }

    # --- Write the report ---
    Write-Host '[6/6] Writing report' -ForegroundColor Cyan
    $report.steps  = @($script:Steps)
    $report.errors = @($script:Errors)

    $json = $report | ConvertTo-Json -Depth 12
    Write-Utf8File -Path (Join-Path $script:OutDirPath 'report.json') -Text (Protect-Text $json)

    $L = New-Object System.Collections.ArrayList
    [void]$L.Add(('hipfire Windows check report (script {0}, schema {1})' -f $script:ScriptVersion, $script:SchemaVersion))
    [void]$L.Add(('generated (UTC): {0}' -f $report.timestamp))
    [void]$L.Add('redaction: user name -> <user>, profile path -> %USERPROFILE%, computer name -> <host>')
    [void]$L.Add('')
    [void]$L.Add('SYSTEM')
    if ($report.os) {
        [void]$L.Add(('  Windows: {0} version {1} build {2}.{3} ({4}) display version {5}' -f $report.os.caption, $report.os.version, $report.os.build_number, $report.os.ubr, $report.os.architecture, $report.os.display_version))
        [void]$L.Add(('  PowerShell: {0} {1}' -f $report.os.powershell_edition, $report.os.powershell_version))
    }
    foreach ($c in @($report.cpu)) { if ($c) { [void]$L.Add(('  CPU: {0} ({1} cores / {2} threads)' -f $c.name, $c.cores, $c.logical_processors)) } }
    if ($report.ram) { [void]$L.Add(('  RAM: {0} GiB' -f $report.ram.total_gib)) }
    [void]$L.Add('')
    [void]$L.Add('GPU ADAPTERS (Win32_VideoController; AdapterRAM caps at 4 GB so use the registry size)')
    foreach ($g in @($report.gpu_adapters)) {
        if ($g) { [void]$L.Add(('  {0}  driver {1} ({2})  AdapterRAM(32-bit)={3}' -f $g.name, $g.driver_version, $g.driver_date, $g.adapter_ram_bytes_32)) }
    }
    foreach ($m in @($report.gpu_memory_registry)) {
        if ($m) { [void]$L.Add(('  registry: {0}  memory={1} MiB  driver {2}' -f $m.driver_desc, $m.memory_mib, $m.driver_version)) }
    }
    if ($null -ne $vramMb) { [void]$L.Add(('  detected VRAM: {0:N0} MiB (source: {1})' -f $vramMb, $vramSource)) }
    [void]$L.Add('')
    [void]$L.Add('HIP SDK')
    if ($report.hip_sdk) {
        [void]$L.Add(('  HIP_PATH: {0}  (.info\version: {1})' -f $report.hip_sdk.hip_path, $report.hip_sdk.hip_path_version_file))
        foreach ($f in @($report.hip_sdk.rocm_folders)) { if ($f) { [void]$L.Add(('  ROCm folder: {0}  (.info\version: {1})' -f $f.name, $f.version_file)) } }
    }
    if ($report.hipinfo -and $report.hipinfo.present) {
        foreach ($l in @($report.hipinfo.summary_lines)) { [void]$L.Add('  hipInfo: ' + $l) }
    } else {
        [void]$L.Add('  hipInfo.exe: not found')
    }
    [void]$L.Add('')
    [void]$L.Add('amdhip64*.dll FOUND (look for an old System32 copy shadowing the hipfire one)')
    if ($report.hip_runtime_dlls) {
        foreach ($d in @($report.hip_runtime_dlls.dlls)) {
            if ($d) { [void]$L.Add(('  [{0}] {1}  FileVersion={2}  size={3}  {4}' -f $d.location, $d.path, $d.file_version, $d.size_bytes, $d.last_write_utc)) }
        }
        foreach ($p in @($report.hip_runtime_dlls.path_entries_with_amdhip64)) {
            if ($p) { [void]$L.Add(('  PATH entry #{0} has {1}: {2}' -f $p.path_index, ($p.files -join ', '), $p.directory)) }
        }
    }
    [void]$L.Add('')
    [void]$L.Add('HIPFIRE')
    [void]$L.Add(('  exe: {0}' -f $report.hipfire.exe))
    [void]$L.Add(('  --version: {0}' -f $report.hipfire.version_text))
    if ($diagSummary) {
        [void]$L.Add(('  diag: parsed={0}  live_gpu={1}  arch={2}  hip_version={3}  vram_total_mb={4}' -f $diagSummary.parsed, $diagSummary.live_gpu_present, $diagSummary.live_gpu_arch, $diagSummary.live_gpu_hip_version, $diagSummary.live_gpu_vram_total_mb))
        [void]$L.Add(('  diag: hip_runtime_library={0}' -f $diagSummary.hip_runtime_library))
        [void]$L.Add(('  diag: hip_runtime_version_raw={0}  hip_driver_version_raw={1}' -f $diagSummary.hip_runtime_version_raw, $diagSummary.hip_driver_version_raw))
    }
    if ($report.hipfire.doctor) {
        if ($report.hipfire.doctor.available) { [void]$L.Add((Format-StepLine $report.hipfire.doctor.step)) }
        else { [void]$L.Add('  doctor: ' + $report.hipfire.doctor.note) }
    }
    foreach ($k in @($report.kernel_pack)) {
        if ($k -and $k.manifest) {
            [void]$L.Add(('  kernel pack manifest: arch={0} tag={1} modules={2} rocm_version={3} rocm_min={4} rocm_max_exclusive={5}' -f $k.manifest.arch, $k.manifest.tag, $k.manifest.modules, $k.manifest.rocm_version, $k.manifest.rocm_min, $k.manifest.rocm_max_exclusive))
        }
    }
    if (@($report.kernel_pack).Count -eq 0) { [void]$L.Add('  kernel pack manifest: none found (kernels would be compiled with hipcc)') }
    [void]$L.Add('')
    [void]$L.Add('ENVIRONMENT (HIP*/ROCM*/HSA*/AMD* only)')
    if ($report.environment) { foreach ($key in $report.environment.Keys) { [void]$L.Add(('  {0}={1}' -f $key, $report.environment[$key])) } }

    $smokeLines = Format-ModelSection -Label ('MODEL SMOKE TEST ({0})' -f $script:SmokeTag) -Result $report.model_tests.smoke
    foreach ($x in $smokeLines) { [void]$L.Add($x) }
    $bigLines = Format-ModelSection -Label ('27B TEST ({0})' -f $script:Tag27B) -Result $report.model_tests.large_27b
    foreach ($x in $bigLines) { [void]$L.Add($x) }
    if ($SkipModel) { [void]$L.Add(''); [void]$L.Add('(model tests skipped: -SkipModel)') }

    [void]$L.Add('')
    [void]$L.Add('ALL STEPS')
    foreach ($s in @($script:Steps)) { [void]$L.Add((Format-StepLine $s)) }
    [void]$L.Add('')
    [void]$L.Add(('ERRORS ({0})' -f $script:Errors.Count))
    foreach ($e in @($script:Errors)) { [void]$L.Add(('  {0}: {1}' -f $e.section, $e.message)) }

    Write-Utf8File -Path (Join-Path $script:OutDirPath 'report.txt') -Text (Protect-Text (($L -join "`r`n") + "`r`n"))

    Write-Host ''
    Write-Host 'Done.' -ForegroundColor Green
    Write-Host ('Report folder: {0}' -f $script:OutDirPath) -ForegroundColor Green
    Write-Host '  report.json  (machine-readable)'
    Write-Host '  report.txt   (human summary)'
    Write-Host 'Your user name, profile path and computer name were replaced by <user>, %USERPROFILE% and <host>.'
    Write-Host 'Please look through the files before sharing them.'
    Write-Host ''
    Write-Host ('Attach report.json and report.txt to a new issue: {0}' -f $script:IssueUrl) -ForegroundColor Cyan
}

# Run only when executed, not when dot-sourced (lets the helpers be tested alone).
if ($MyInvocation.InvocationName -ne '.') {
    Main
}
