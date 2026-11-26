# VIctory on an RX 7900 XTX

A fork of [VIctory](https://github.com/Marvisak/VIctory), a compatibility layer
that runs an Xbox Series title natively on Windows from files the user supplies.
This one is about **speed on AMD RDNA 3**: the world at 30 frames a second on
one PC, without giving up the picture. It is not there yet.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

## Where it stands (2026-11-25)

Tested on one machine only: Windows 11, Intel i9-12900K, 64 GB RAM, Radeon
RX 7900 XTX, 1440p, the Series X profile.

| | |
|---|---|
| Menus, videos, loading screens | 30 fps |
| Prologue, indoors | 50-75 ms a frame (13-20 fps) |
| First mission with the boat | 46-53 ms (19-22 fps) |
| Street chase with many cars and people | 100-165 ms (6-10 fps), single frames far longer |
| Outdoors on foot, the camp | 60-90 ms |
| Flying | 42-57 ms |
| Longest session so far | 41 minutes, no crash |
| Memory | 40-46 GB of private memory: **64 GB of RAM recommended** |

It started at 280 ms a frame in the same scenes two days earlier.

What is known to be wrong:

- Trees, plants and far buildings flicker outdoors in some places.
- Rooms with little direct light are too dark without `-RayTracing`.
- Textures of what just came into view appear late while the title streams.
- The first run with a new driver or a new build stutters while the driver makes
  pipelines (`victory warm` makes them ahead).
- Recording a frame in the world makes the title's own watchdog end it.

## How to run it

You need Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with shader
model 6.6 and mesh shaders, Rust (the toolchain is pinned), the MSVC build
tools, and your own dump of the title.

```powershell
git clone https://github.com/avatarmanz/VIctory-7900XTX
cd VIctory-7900XTX
copy victory.example.toml victory.toml      # then set your paths, outside this folder
cargo build --locked --profile diag -p victory --features diagnostics
scripts\play.ps1 -Exe <path of your copy's executable>
```

`scripts\play.ps1 -Safe` leaves out the newest, faster ways; try it first when
something is wrong. `-RayTracing` turns the title's ray-traced indirect light on.

The faster build to play with is the plain release one, without the
diagnostics feature: `cargo build --locked --release -p victory`, then
`scripts\play.ps1 -Build release -Exe ...`. Both builds use thin LTO, one
codegen unit and `target-cpu=x86-64-v3` (`Cargo.toml`, `.cargo/config.toml`),
so a build takes longer than it did.

Before the first run, ahead of time (both can be stopped and go on later):

```powershell
target\diag\victory.exe shaders <the shader archives>   # every shader translated, minutes
target\diag\victory.exe warm --used                     # pipelines earlier runs drew with, for a new driver or build
```

Commands, debugging switches and upstream's own description:
[docs/README-fxceless.md](docs/README-fxceless.md). Every `VICTORY_...` switch is
described in a comment where it is read.

## NVIDIA

Measured on an RTX 4070 12 GB, 32 GB RAM, 12 threads, Series X profile, the
same scripted path into the world each time, warm driver cache, 8-10 minutes
per run (median of the last 15 five-second lines):

| | before (main 85d5641) | now |
|---|---|---|
| World frame time | 160 ms (6.2 fps) | 79 ms (12.7 fps) |
| Draws dropped a frame | ~850 | ~350 (4 with ray tracing on) |
| Peak private memory | 29.5 GB | 26-27 GB |
| Ray tracing (`-RayTracing`) | builder off: 5 of 52 programs refused | builder runs whole (53 pipelines), 8.4 fps, no device loss in 10 min |
| Device losses in 10-minute runs | none | none |

What made the difference on NVIDIA (each behind a `VICTORY_...` switch
described where it is read):

- **The ray tracing structure builder translates for 32-lane waves**: a
  first-lane read of a value the same in every lane of the console wave takes
  the host wave's own lane (`VICTORY_WAVE_SAME`), the box rehydration kernels'
  short last groups keep only their started lanes in EXEC (`VICTORY_WAVE_LIVE`),
  and the builder never runs on an approximate translation.
- **Translator fixes for 32-lane hosts** from upstream and the forks: wave64
  `ds_bpermute` scoped to its 32-lane half, LDS bounds, the constant table
  after the instructions, divergent barriers in one-wave groups that keep
  their fence but stop waiting (`VICTORY_GROUP_SYNC` auto), `s_pack_*`.
- **The capped view heap** gives its upper 606,784 slots to dynamic tables
  (from DexNom #68), and indirect dispatches past 65,535 groups run as asked.
- **Memory**: streamed BC textures without their largest levels as the card
  fills (`VICTORY_TEXTURE_MAX`), finer windows of title memory (256 KiB with
  tails), fewer pooled host lists, a governor that lets go of what is kept
  for reuse when commit headroom is short (`VICTORY_RAM_TRIM_MB`).
- **Latency**: at most two frames ahead of what the card finished
  (`VICTORY_FRAMES_AHEAD`), a tearing-capable swap chain for VRR, input polled
  every 4 ms on a high-priority thread, a frame-wait race fixed; optional
  NVIDIA Reflex (`VICTORY_REFLEX=1`) and pacing after the present
  (`VICTORY_PACE_AFTER=1`).
- **Frame time**: thin LTO with x86-64-v3, thread pools sized to the CPU, small
  title buffers' copies in the card's memory, the submission's list resets on
  a thread of ours.

Known: with ray tracing on, the card's 11 GB fill and about 7 GB of the PC's
memory is used as the card's, so 32 GB is tight. Whether interiors are lit
correctly with the builder running was not checked by eye.

`scripts\play.ps1` takes an NVIDIA profile by itself when it finds an NVIDIA
card (or with `-Nvidia`): device-removed data keeps only the faulting address
(`VICTORY_D3D12_DRED=faults`, not `light`), no book of the card's memory
(`VICTORY_BOOK=0`: it reads descriptors as AMD's), and the card is not kept
awake by a spinning compute queue (`VICTORY_CARD_AWAKE` left unset, which
contends with NVIDIA's time slicing; `-CardAwake` sets it all the same). AMD
cards get the same settings as before. More:

- `-Fps <n>` sets `VICTORY_FPS` (the title's frame pacing at n frames a
  second; 0: no pacing sleep), in builds that read it.
- `-SmoothMotion` starts a copy named `victory_sm.exe` (refreshed when the
  build is newer), so a Smooth Motion profile set for it in the NVIDIA App
  applies every time (after trustno1-coder's `play-nvidia.ps1`). It adds
  about a frame of latency; `VICTORY_REFLEX=1` goes with it.
- Switches to try one at a time with `-More @{NAME='value'}` are listed,
  commented, in the script: `VICTORY_INDIRECT_WIDE`, `VICTORY_JOIN_BARRIERS`,
  `VICTORY_UP_QUEUE`, `VICTORY_WINDOWS_BAR_MB`, `VICTORY_REFLEX`,
  `VICTORY_FRAMES_AHEAD`, `VICTORY_PACE_AFTER`, `VICTORY_BVH_NO_HIT`,
  `VICTORY_POOL_INPLACE`.

Defaults that follow the machine rather than the script, on any card: on a
budget of 12 GiB or less (an RTX 4070's is about 11) every title buffer from
64 KiB has its copy in the card's memory, up to 1280 MiB of them
(`VICTORY_CARD_BUFFERS_MIN_KB`, `VICTORY_CARD_BUFFERS_MB`); the pools of
threads of ours scale with the logical processors (on 12: four list threads,
three for the title's bytes under flights, three for descriptor records,
three for views; as before from 24 on); with 12 logical processors or fewer
the submitting thread has a core to itself (`VICTORY_LANE_CORE=0`: not); the
book of the card's memory is kept only on AMD cards (`VICTORY_BOOK=1`: on
any).

## What is different here

In short; the reasons and the measurements are in [docs/findings.md](docs/findings.md).

- **List calls are written down, not carried out, on the title's threads**
  (`VICTORY_THIN`): up to eight threads of ours translate them.
- **Submissions are prepared by a thread of ours and joined**
  (`VICTORY_ASYNC_SUBMIT`, `VICTORY_JOIN_PARTS`), with what can be spread over
  threads spread: refreshes, compares (AVX2), descriptor records, landings.
- **What shaders wrote comes back beside the write guards**, by the dwords that
  changed, on a thread of its own (`VICTORY_RELAY`, `VICTORY_LAND`).
- **Buffer fills that need bytes still on the card are done on the card**
  (`VICTORY_CARD_STEPS`) instead of waiting for it.
- **The title's writes are caught more cheaply**: one view opened at a fault,
  blocks that rest, places in the title's code that look before they write.
- **Textures go up on a copy queue of their own** and a slot shows one only after
  its upload ran, which also ended a device loss that came every 15 minutes or so.
- **Boats float**: textures that shaders write and the title's CPU reads go back
  to title memory (from fxceless and Geckomacabre, on by default here).
- **Frame record and replay** with per-pass times and shader variants compared
  turn by turn ([docs/replay.md](docs/replay.md)).

## Credits

This is a tree of other people's work with more on top. The history is kept, and
commits taken from elsewhere name the commit and its author.

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors: the loader, the runtime services, the Direct3D 12 back end and
  the shader recompiler.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**: the fork
  this one started from, and the people whose pull requests it carries.
- **[DexNom/VIctory](https://github.com/DexNom/VIctory)**: ray tracing fixes,
  cube gradients, sparse textures.
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)**:
  the ambient light idea, pipeline workers, and work by romainhedouin.
- **[FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY)**.
- **[myokex/victory-amd](https://github.com/myokex/victory-amd)**: the guard over
  every view of a page (with Spexivus, joroyt and tightfall), and Joshua Odom's
  reads into title memory.
- **Geckomacabre** and **cxy0072019-sketch**: the texture read-back.
- **[XCL](https://github.com/ivall/xcl)** (GPL-2.0-only) was read for how the same
  title behaves under another layer; no code of it is here.
- This fork's changes were made with [Claude Code](https://claude.com/claude-code).

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE); the one file that is not
covered is named in [docs/asset-policy.md](docs/asset-policy.md).
