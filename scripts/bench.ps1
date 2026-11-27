# An unattended, repeatable measurement of the title in the world, for A/B runs of switches
# (one batch of changes, then one bench run each way).
#   scripts\bench.ps1 -Exe <path of your copy's executable> -Label base
#   scripts\bench.ps1 -Exe <...> -Label residency -More @{VICTORY_RESIDENCY='1'}
# What it does: starts the title with play.ps1's settings (its NVIDIA profile on an NVIDIA card) and the
# diagnostics build, presses Enter once the log shows the start screen (frames of few draws), waits for
# the world (a frame line of -WorldDraws draws or more), waits -CinematicSeconds through the opening
# cinematic with no input, measures -MeasureSeconds, then ends the run it started and writes a summary.
# Output: %LOCALAPPDATA%\victory\bench\<date>-<label>\ : log.txt (the run's log), settings.txt,
# summary.txt (frame time median and p95 of the log's frame lines in the window, fps, preparing and GPU
# ms, draws, the process's working set and private bytes, mean and peak).
# -GpuTiming    each list timed on the card, slow ones logged with their commands (VICTORY_GPU_TIMING=1
#               with the full device-removed data, VICTORY_D3D12_DRED=1: slower frames, for where the time goes)
# -Shots        pictures of what reaches the screen, half size, in frames\ (VICTORY_CAPTURE: the frames
#               whose number is a power of two)
# -Build        diag (the default: the scripted Enter and the counters exist only there) or release
# -Safe, -RayTracing, -SoftwareRT, -ConstantAmbient, -Nvidia, -CardAwake, -Fps, -More: as play.ps1's.
# The scripted Enter goes through a file the build reads (VICTORY_KEYS_FILE, gameinput.rs).
# After Belobug PR #139 (scripts/bench.ps1, by Zetsukae) and fxceless PR #617.
param(
    [Parameter(Mandatory)][string]$Exe,
    [string]$Label = 'run',
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [int]$CinematicSeconds = 150,
    [int]$MeasureSeconds = 90,
    [int]$TimeoutSeconds = 1500,
    [int]$TitleDraws = 30,
    [int]$WorldDraws = 300,
    [switch]$GpuTiming,
    [switch]$Shots,
    [switch]$Safe,
    [switch]$RayTracing,
    [switch]$SoftwareRT,
    [switch]$ConstantAmbient,
    [switch]$Nvidia,
    [switch]$CardAwake,
    [ValidateRange(0, 1000)][int]$Fps,
    [hashtable]$More = @{}
)
$ErrorActionPreference = 'Stop'
if (Get-Process victory -ErrorAction SilentlyContinue) { Write-Host 'victory.exe is already running: close it first.'; exit 1 }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path $env:LOCALAPPDATA "victory\bench\$stamp-$Label"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out 'log.txt'
$keys = Join-Path $out 'keys.txt'
[IO.File]::WriteAllText($keys, '')

# The bench's own settings, under what -More asks for.
$extra = @{ VICTORY_LOG = 'info'; VICTORY_KEYS_FILE = $keys }
if ($GpuTiming) { $extra.VICTORY_GPU_TIMING = '1'; $extra.VICTORY_D3D12_DRED = '1' }
if ($Shots) {
    $frames = Join-Path $out 'frames'
    New-Item -ItemType Directory -Force $frames | Out-Null
    $extra.VICTORY_CAPTURE = $frames
}
foreach ($k in $More.Keys) { $extra[$k] = $More[$k] }
$playArgs = @{ Exe = $Exe; Build = $Build; More = $extra; Print = $true }
foreach ($name in 'Safe', 'RayTracing', 'SoftwareRT', 'ConstantAmbient', 'Nvidia', 'CardAwake') {
    if ((Get-Variable $name -ValueOnly).IsPresent) { $playArgs[$name] = $true }
}
if ($PSBoundParameters.ContainsKey('Fps')) { $playArgs.Fps = $Fps }
$run = & (Join-Path $PSScriptRoot 'play.ps1') @playArgs
if (-not $run) { exit 1 }

# As play.ps1: other settings left over in this window go; where your files are stays.
$paths = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR'
Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $paths } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
foreach ($k in $run.Settings.Keys) { Set-Item "Env:$k" $run.Settings[$k] }
$run.Settings.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" } |
    Set-Content -Encoding utf8 (Join-Path $out 'settings.txt')

# One frame line of the log: fps, ms, preparing, copies, textures, GPU, copying back, draws.
$frameRe = '([\d.]+) fps, ([\d.]+) ms/frame \(([\d.]+) preparing: copies ([\d.]+) ms [\d.]+ MB, textures ([\d.]+) ms [\d.]+ MB; ([\d.]+) on the GPU; ([\d.]+) copying back [\d.]+ MB\), (\d+) draws/frame'
$lines = New-Object System.Collections.Generic.List[object]
$read = 0
function Read-Frames {
    # New frame lines since the last call, stamped with the time they were seen.
    if (-not (Test-Path -LiteralPath $log)) { return }
    $fs = [IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
    try {
        [void]$fs.Seek($script:read, 'Begin')
        $text = (New-Object IO.StreamReader($fs)).ReadToEnd()
    } finally { $fs.Close() }
    $end = $text.LastIndexOf("`n")
    if ($end -lt 0) { return }
    $script:read += [Text.Encoding]::UTF8.GetByteCount($text.Substring(0, $end + 1))
    $now = Get-Date
    foreach ($l in $text.Substring(0, $end) -split "`n") {
        if ($l -match $frameRe) {
            $script:lines.Add([pscustomobject]@{
                Time = $now; Fps = [double]$Matches[1]; Ms = [double]$Matches[2]; Prep = [double]$Matches[3]
                Copies = [double]$Matches[4]; Textures = [double]$Matches[5]; Gpu = [double]$Matches[6]
                Back = [double]$Matches[7]; Draws = [int]$Matches[8]
            })
        }
    }
}
function Write-Status([string]$text) { Write-Host ('[{0:HH:mm:ss}] {1}' -f (Get-Date), $text) }
function Send-Enter { [IO.File]::AppendAllText($keys, "Enter`r`n") }

$started = Get-Date
$p = Start-Process -FilePath $run.Victory -ArgumentList @('run', "`"$($run.Exe)`"", '--execute') `
    -WorkingDirectory (Join-Path $PSScriptRoot '..') -RedirectStandardError $log `
    -RedirectStandardOutput (Join-Path $out 'stdout.txt') -PassThru
Write-Status "started $($run.Victory) ($Build), results in $out"
$phase = 'start screen'; $pressed = $null; $world = $null; $measureFrom = $null
$memory = New-Object System.Collections.Generic.List[object]
try {
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 2
        if (((Get-Date) - $started).TotalSeconds -gt $TimeoutSeconds) { Write-Status 'timed out'; break }
        Read-Frames
        $recent = @($lines | Select-Object -Last 2)
        $light = $recent.Count -eq 2 -and @($recent | Where-Object { $_.Draws -le $TitleDraws }).Count -eq 2
        switch ($phase) {
            'start screen' {
                if ($light) { Send-Enter; $pressed = Get-Date; $phase = 'loading'; Write-Status 'start screen: Enter pressed' }
            }
            'loading' {
                $last = $lines | Select-Object -Last 1
                if ($last -and $last.Time -gt $pressed -and $last.Draws -ge $WorldDraws) {
                    $world = Get-Date; $phase = 'cinematic'; Write-Status "the world is drawn: waiting $CinematicSeconds s"
                } elseif (((Get-Date) - $pressed).TotalSeconds -gt 45 -and $light -and $recent[0].Time -gt $pressed) {
                    Send-Enter; $pressed = Get-Date; Write-Status 'still on a light screen: Enter again'
                }
            }
            'cinematic' {
                if (((Get-Date) - $world).TotalSeconds -ge $CinematicSeconds) {
                    $measureFrom = Get-Date; $phase = 'measuring'; Write-Status "measuring $MeasureSeconds s"
                }
            }
            'measuring' {
                $p.Refresh()
                $memory.Add([pscustomobject]@{ Set = $p.WorkingSet64 / 1MB; Private = $p.PrivateMemorySize64 / 1MB })
                if (((Get-Date) - $measureFrom).TotalSeconds -ge $MeasureSeconds) { $phase = 'done' }
            }
        }
        if ($phase -eq 'done') { break }
    }
} finally {
    # Only the run this script started.
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; $p.WaitForExit(10000) | Out-Null }
}
Read-Frames

function Get-Quantile([double[]]$values, [double]$q) {
    $sorted = @($values | Sort-Object)
    if ($sorted.Count -eq 0) { return [double]::NaN }
    $sorted[[Math]::Min($sorted.Count - 1, [Math]::Floor($q * $sorted.Count))]
}
$summary = New-Object System.Collections.Generic.List[string]
$summary.Add("label: $Label   build: $Build   phase reached: $phase   exit code: $(if ($p.HasExited) { $p.ExitCode } else { '-' })")
# The first 5 s of the window left out: a line may count frames from before it.
$window = @(if ($measureFrom) { $lines | Where-Object { $_.Time -gt $measureFrom.AddSeconds(5) } })
$summary.Add("frame lines in the window: $($window.Count)")
if ($window.Count -gt 0) {
    $mean = { param($name) ($window | Measure-Object -Property $name -Average).Average }
    $ms = [double[]]@($window | ForEach-Object { $_.Ms })
    $summary.Add(('ms/frame: median {0:N1}, p95 {1:N1}, worst {2:N1}; fps mean {3:N2}' -f
        (Get-Quantile $ms 0.5), (Get-Quantile $ms 0.95), ($ms | Measure-Object -Maximum).Maximum, (& $mean Fps)))
    $summary.Add(('mean ms/frame: preparing {0:N1} (copies {1:N1}, textures {2:N1}), on the GPU {3:N1}, copying back {4:N1}' -f
        (& $mean Prep), (& $mean Copies), (& $mean Textures), (& $mean Gpu), (& $mean Back)))
    $summary.Add(('draws/frame: median {0:N0}' -f (Get-Quantile ([double[]]@($window | ForEach-Object { $_.Draws })) 0.5)))
}
if ($memory.Count -gt 0) {
    $set = $memory | Measure-Object -Property Set -Average -Maximum
    $private = $memory | Measure-Object -Property Private -Average -Maximum
    $summary.Add(('RAM: working set mean {0:N0} MB, peak {1:N0} MB; private bytes mean {2:N0} MB, peak {3:N0} MB' -f
        $set.Average, $set.Maximum, $private.Average, $private.Maximum))
}
if ($More.Count -gt 0) { $summary.Add('-More: ' + (($More.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' ')) }
$summary | Set-Content -Encoding utf8 (Join-Path $out 'summary.txt')
$summary | ForEach-Object { Write-Host $_ }
Write-Host "Results in $out"
