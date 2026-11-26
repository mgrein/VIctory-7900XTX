# Starts the title with the settings this fork is played with (README.md, "How to run it").
#   scripts\play.ps1 -Exe <path of your own copy's executable>
# -Safe         without the newer, faster ways (the first thing to try when something is wrong)
# -RayTracing   the title's own ray-traced indirect light: lit interiors, about a third slower
# -Build        target\diag (the default, the build all measurements are of) or target\release (faster to play)
# -More         other settings: -More @{VICTORY_PANEL='1'}
# -Nvidia       the NVIDIA profile even where no NVIDIA card is found (it is taken by itself on one; README.md, "NVIDIA")
# -CardAwake    on NVIDIA, keep the card awake between submissions all the same (VICTORY_CARD_AWAKE=1, as on AMD)
# -Fps <n>      n frames a second: our frame clock and the title's own frame rate setting (VICTORY_FRAME_INTERVAL_MS=1000/n; 0: no pacing wait)
# -SmoothMotion start a copy named victory_sm.exe, so that a Smooth Motion profile set for it in the NVIDIA App applies
# Run it from a normal window, not as administrator: the driver then keeps its shader cache elsewhere.
param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$Safe,
    [switch]$RayTracing,
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [hashtable]$More = @{},
    [switch]$Nvidia,
    [switch]$CardAwake,
    [ValidateRange(0, 1000)][int]$Fps,
    [switch]$SmoothMotion
)
$settings = @{
    VICTORY_TEXTURE_NO_WAIT = '1'; VICTORY_RELAY = '2'; VICTORY_WIDE_BACK = '1'; VICTORY_CARD_REHASH_RUNS = '1'
    VICTORY_JOIN_COPIES = '1'; VICTORY_INDIRECT_WINDOW = '1'; VICTORY_RELAY_COPY = '1'; VICTORY_CARD_STEPS = '2'
    VICTORY_INDIRECT_BATCH = '2'; VICTORY_MAP_THREADS = '1'; VICTORY_TITLE_PRIORITIES = '0'; VICTORY_CARD_AWAKE = '1'
    VICTORY_SHOW_AFTER = '1'; VICTORY_AMBIENT = '0.8'; VICTORY_D3D12_DRED = 'light'
}
if ($Safe) {
    $settings.VICTORY_UNDER_WIDE = '0'; $settings.VICTORY_LAND_ECHO = '0'
} else {
    $settings.VICTORY_THIN = '1'; $settings.VICTORY_JOIN_PARTS = '8'; $settings.VICTORY_UPLOAD_AHEAD = '1'
}
if ($RayTracing) { $settings.VICTORY_BVH_BUILDER = '1' }

# The NVIDIA profile: on an NVIDIA card (or with -Nvidia).
if (-not $Nvidia) {
    try {
        $cards = Get-CimInstance Win32_VideoController -ErrorAction Stop
        $Nvidia = [bool]($cards | Where-Object { $_.Name -match 'NVIDIA' -or $_.AdapterCompatibility -match 'NVIDIA' })
    } catch { }
}
if ($Nvidia) {
    Write-Host 'NVIDIA profile (README.md, "NVIDIA").'
    # Only the faulting address of a lost device: no breadcrumbs, which cost on every list.
    $settings.VICTORY_D3D12_DRED = 'faults'
    # No book of the card's memory: it reads descriptors as AMD's and costs memory.
    $settings.VICTORY_BOOK = '0'
    # A spinning 64-thread compute queue that keeps the card awake contends with NVIDIA's time slicing.
    if (-not $CardAwake) { $settings.Remove('VICTORY_CARD_AWAKE') }
    # Switches to try, one at a time (-More @{NAME='value'}); each is read only by builds that have it:
    #   VICTORY_INDIRECT_WIDE=1   indirect dispatches up to 2^20 groups in x on any card (cut to 65,535 on NVIDIA otherwise)
    #   VICTORY_JOIN_BARRIERS=1   barrier calls held and joined into fewer
    #   VICTORY_UP_QUEUE=1        copies in the card's memory brought up on a copy queue
    #   VICTORY_WINDOWS_BAR_MB=n  up to n MB of windows outside buffers in the card's memory, filled through the bus
    #   VICTORY_REFLEX=1          NVIDIA Reflex low-latency mode (VICTORY_REFLEX_BOOST=1: with Low Latency Boost)
    #   VICTORY_FRAMES_AHEAD=n    at most n frames ahead of what the card finished (0: no limit)
    #   VICTORY_PACE_AFTER=1      the frame pacing sleep after a present instead of before the next flip
    #   VICTORY_BVH_NO_HIT=1      ray walks find no hit while the structure builder is not running whole
    #   VICTORY_POOL_INPLACE=1    the page pool's windows mapped in place
}
if ($PSBoundParameters.ContainsKey('Fps')) {
    # One frame-rate switch (upstream's): it paces our frame clock and the loader gives the title the same rate.
    $settings.VICTORY_FRAME_INTERVAL_MS = if ($Fps -eq 0) { '0' } else { (1000.0 / $Fps).ToString('0.###', [Globalization.CultureInfo]::InvariantCulture) }
}
foreach ($k in $More.Keys) { $settings[$k] = $More[$k] }

$victory = Join-Path $PSScriptRoot "..\target\$Build\victory.exe"
if (-not (Test-Path $victory)) { Write-Host "No $victory. Build it first (README.md)."; exit 1 }
if (-not (Test-Path $Exe)) { Write-Host "No $Exe."; exit 1 }
if ($SmoothMotion) {
    # (After trustno1-coder a1849fac, scripts/play-nvidia.ps1.) A copy under the name the NVIDIA App's
    # Smooth Motion profile is set for, refreshed when the build is newer.
    $sm = Join-Path (Split-Path $victory) 'victory_sm.exe'
    if (-not (Test-Path $sm) -or (Get-Item $sm).LastWriteTime -lt (Get-Item $victory).LastWriteTime) {
        Copy-Item $victory $sm -Force
    }
    $victory = $sm
    Write-Host "Starting ${sm}. Smooth Motion adds about a frame of latency; VICTORY_REFLEX=1 (-More) goes with it, in builds that have it."
}
# (Other settings left over in this window go; where your files are stays.)
$paths = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR'
$inherited = @(Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $paths })
foreach ($entry in $inherited) { Remove-Item "Env:$($entry.Name)" }
foreach ($k in $settings.Keys) { Set-Item "Env:$k" $settings[$k] }
$discarded = $inherited | Where-Object {
    $_.Name -notin $More.Keys -and -not $_.Value.Equals([Environment]::GetEnvironmentVariable($_.Name))
} | Sort-Object Name
if ($discarded) {
    Write-Host "Discarded inherited values for $($discarded.Name -join ', '). Use -More to pass them explicitly."
}
$Exe = (Resolve-Path $Exe).Path
Push-Location (Join-Path $PSScriptRoot '..')   # victory.toml is looked for here
try { & $victory run $Exe --execute } finally { Pop-Location }
