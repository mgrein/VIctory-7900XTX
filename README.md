# VIctory on an RX 7900 XTX

A fork of [VIctory](https://github.com/Marvisak/VIctory), a compatibility layer
that runs an Xbox Series title natively on Windows from files the user supplies.

## The goal

**The Series X version at 4K output (drawn at 1440p) with ray tracing on,
looking the way it does on the Xbox**, on an AMD RX 7900 XTX first.

In this order:

1. **A correct picture.** Textures and models reach full detail and stay there
   (no pop-in or pop-out, no flickering trees), and ray-traced lighting is right:
   not too dark, not too bright, no flicker.
2. **Then the frame rate:** 30 fps or more with ray tracing on, then as high a
   frame rate and as low frame times as the PC allows.
3. **Then smaller PCs:** 32 GB (later 16 GB) of RAM and smaller cards.

The way there is **porting, not emulating**: patching the title's renderer so it
behaves the way a PC renderer does, rather than having the layer watch and copy
the console's shared memory every frame. Shaders are translated ahead of time
into a versioned store that is kept between builds. The Series S profile and
ray tracing off stay available.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

## Where it stands (2026-11-26)

Tested on one machine only: Windows 11, Intel i9-12900K, 64 GB RAM, Radeon
RX 7900 XTX, renderer patches 1, 2, 5, 6 and 7 on. Runs of 270 s in a city
street: standing still, then walking and turning the camera.

| Version | Standing | Walking | RAM |
|---|---|---|---|
| **Series X, 4K drawn at 1440p, ray tracing on (the goal)** | 99 ms (10 fps) | ~110 ms (9 fps) | 45 GB |
| Series X, 4K drawn at 1440p, ray tracing off | 49 ms (20 fps) | ~62 ms (16 fps) | 45 GB |
| Series X, 1080p output, ray tracing off | 40 ms (25 fps) | ~43 ms (23 fps) | 42 GB |
| Series S, ray tracing off | 43 ms (23 fps) | ~64 ms (16 fps) | 37 GB |

Known problems with ray tracing on, being worked on now: lighting flickers and
swings between too dark and too bright even when nothing moves; dark rooms and
backgrounds; models and textures stay at their lowest detail for a long time
while the title streams (worst in the intro); trees flicker.

Earlier measurements with ray tracing off:

| | |
|---|---|
| City street, Series S profile, renderer patches 1, 2, 5, 6 and 7 on | about 45 ms a frame (about 22 fps, 44.9 ms in run merge-5) |
| Same, patches 1, 2, 5 and 6 | 45.6 ms, p95 47 ms |
| Same, patches 1, 2 and 5 | 47.3 ms (about 21 fps), p95 49 ms |
| Series X profile, patches 1+2 | about 53 ms (about 19 fps) |
| Same scene before the patches | 57-75 ms (13-17 fps) |
| Shaders the title uses that translate | all of them (25,790 of 25,985; the rest are unused geometry shaders) |
| Memory | about 30 GB of private memory, 10.5 GB of VRAM: **64 GB of RAM recommended** for now |

It started at 280 ms a frame in the same scenes three days earlier. What sets
the frame now is the one thread that submits the title's work to the card
(about 40 ms); the card itself needs about 36 ms and now waits about 7 ms a frame with nothing queued.

### Renderer patches (off by default while they are tested)

| Switch | What it does |
|---|---|
| `VICTORY_PORT_CONSTANTS=1` | The title's 174 MB constants heap is read by the card where the title writes it: no uploads, no fills |
| `VICTORY_PORT_DESCRIPTORS=1` | The title's descriptor-table stores are patched (byte-checked) to report the slots they write, instead of a scan of ~600,000 slots a frame |
| `VICTORY_PORT_CBV=1` | Translated shaders read constant buffers in place as raw bytes (~22,000 fills a frame to none; -4.7 ms) |
| `VICTORY_PORT_GUARD=1` | The write guards' lock is shared by faults and refreshes, with a claim per 64 KiB block; the submission's planning runs beside it (title threads' waits 29 ms to ~2 ms a frame; -2 ms) |
| `VICTORY_PORT_JOURNAL=1` | The heaps' records are asked for at the title's submit, and the journal's notes are split into 64 parts with locks of their own (the submit thread's waits ~5 ms + 1 ms to ~0.3 ms; -1 to -3 ms) |

### Other switches

| Switch | What it does |
|---|---|
| `--console series-s` | The Series S profile: 1080p output, about 5 ms faster, a softer picture |
| `VICTORY_OUTPUT=1920x1080` | The title renders to that output size itself (Series X profile; about 16 ms less card time) |
| `VICTORY_FRAME_INTERVAL_MS=16.667` | Lets the title go above 30 fps (menus and videos reach 60; the game's clock stays right) |
| `VICTORY_AMBIENT_SETTLE` | How fast light settles after walking indoors (on by default, `250,3`; `0` turns it off) |
| `VICTORY_TEXTURE_STORE=1` | Converted textures kept between runs (about 8 ms a frame for now: off) |

What is known to be wrong:

- Trees, signs and far objects flicker in the distance in some places.
- Lighting is too bright or too dark after walking indoors, until the title has
  run for a while; rooms with little direct light are too dark without `-RayTracing`.
- Textures of what just came into view appear late while the title streams.
- The first run of a new build stutters while the driver makes pipelines
  (`victory warm` makes them ahead).
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

On a hybrid-graphics PC, the runtime and offline pipeline checks prefer a
high-performance adapter with Shader Model 6.6. The runtime logs its adapter
name, PCI vendor and device IDs, and dedicated VRAM. Set
`VICTORY_D3D12_ADAPTER` to a zero-based DXGI index or part of the adapter's
name to select it explicitly. With the launcher, pass
`-More @{VICTORY_D3D12_ADAPTER='NVIDIA'}`.

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
| World frame time | 160 ms (6.2 fps) | 74 ms (13.6 fps) |
| Draws dropped a frame | ~850 | ~350 (4 with ray tracing on) |
| Peak private memory | 29.5 GB | 25-26 GB |
| Ray tracing (`-RayTracing`) | builder off: 5 of 52 programs refused | builder runs whole (53 pipelines), 8.0-8.5 fps, 28-29 GB peak, no device loss in 10 min |
| Device losses in 10-minute runs | none | none |
| Heavier area (later save, ~4,000 draws), `play.ps1` NVIDIA profile | — | 67 ms (14.9 fps) |
| Same area, ray tracing: emulated walk / DXR 1.1 ray queries | — | 153 ms (6.5 fps) / 108 ms (9.3 fps) |
| Render queue (`VICTORY_REFLEX=1`, driver-measured) | not available | 0.2 ms |

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
- **The threads that translate lists** keep the views they make in a CPU-only
  heap of their own and copy them when the same view comes again
  (`VICTORY_VIEW_COPIES`): on NVIDIA making a view costs far more than copying
  one. The constant buffers copied into a list's scratch memory for each draw
  have their views kept with the scratch chunk, by offset and size, for as
  long as the chunk lives, on whichever thread (`VICTORY_CHUNK_VIEWS`).
  Copies of views' own bytes are pieces of shared 32 MiB buffers
  (`VICTORY_EXACT_ARENA`), not a committed buffer each: on a host without raw
  twins (NVIDIA's capped heap) there are thousands a frame. Diagnostics
  counters reach one thread-local value per call.
- **Frame time**: thin LTO with x86-64-v3 (also for a windows-gnu toolchain), thread pools sized to the CPU, small
  title buffers' copies in the card's memory, the submission's list resets on
  a thread of ours.

**Ray tracing on the card's own units** (from maren-halv/VIctory-DXR,
`VICTORY_NATIVE_RT=1`, what `play.ps1 -RayTracing` uses on NVIDIA): 47 of the
title's 64 ray-walking programs become DXR 1.1 ray queries against
acceleration structures the host builds; the other 17 keep the emulated
walk. **Renderer ports 1 and 5** (`VICTORY_PORT_CONSTANTS`, `VICTORY_PORT_CBV`,
as DoubleGhosty/cool_fox's launcher) are on in the NVIDIA profile.

Known: the frame is still bound by the processor (the card about 35-50 % busy
without ray tracing): the four threads that translate list calls and the
submitting thread. Areas with 6,000-8,000 draws a frame still run at 3-5 fps.
With ray tracing on, the card's 11 GB fill and about 7 GB of the PC's
memory is used as the card's, so 32 GB is tight. Whether interiors are lit
correctly with the builder running was not checked by eye.

`scripts\play.ps1` takes an NVIDIA profile by itself when it finds an NVIDIA
card (or with `-Nvidia`): device-removed data keeps only the faulting address
(`VICTORY_D3D12_DRED=faults`, not `light`), no book of the card's memory
(`VICTORY_BOOK=0`: it reads descriptors as AMD's), and the card is not kept
awake by a spinning compute queue (`VICTORY_CARD_AWAKE` left unset, which
contends with NVIDIA's time slicing; `-CardAwake` sets it all the same). AMD
cards get the same settings as before. More:

- `-Fps <n>` sets `VICTORY_FRAME_INTERVAL_MS` to 1000/n (0 for 0: no pacing
  wait): our frame clock ticks at n a second and the loader gives the title
  the same frame rate. `VICTORY_FPS=n` is still read by the frame clock alone
  where `VICTORY_FRAME_INTERVAL_MS` is not set.
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
