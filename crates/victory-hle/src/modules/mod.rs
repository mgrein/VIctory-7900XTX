//! Implementations, one module per area of the console API.
//!
//! Adding an API: write an `extern "system"` function named exactly like
//! the import, then list it with `hle_export!` in the module's `module()`.
//! Run `victory census` over your titles to see what is still missing.
//!
//! Planned areas (driven by census results, not by completeness):
//! `file` (package paths), `thread` (TLS callbacks for title threads),
//! `xaudio2`, `xstore` stubs.

#[cfg(windows)]
pub mod apu;
pub mod audio;
#[cfg(windows)]
pub mod bvh_builder;
pub mod d3d12x;
pub mod display;
pub mod dsp;
#[cfg(windows)]
pub mod dstorage;
#[cfg(windows)]
pub mod fidelityfx;
#[cfg(windows)]
pub mod files;
pub mod gameinput;
pub mod kernel;
#[cfg(windows)]
pub mod lifecycle;
#[cfg(windows)]
pub mod media;
#[cfg(windows)]
pub mod net;
pub mod rpcrt4;
pub mod speech;
pub mod storage;
pub mod xasync;
pub mod xg;
pub mod xgame;
pub mod xgameruntime;
pub mod xgamesave;
pub mod xgameui;
#[cfg(windows)]
pub mod xmem;
pub mod xnetworking;
pub mod xpackage;
pub mod xstore;
pub mod xsystem;
pub mod xuser;

use crate::HleModule;

pub(crate) fn all() -> Vec<HleModule> {
    let mut all = vec![
        kernel::module(),
        xgameruntime::module(),
        gameinput::module(),
        d3d12x::module(),
        audio::module(),
        dsp::module(),
        xg::module(),
    ];
    #[cfg(windows)]
    all.push(apu::module());
    // `XboxBvhBuilder.dll` is served (and the title's own forwarded to) only
    // with native ray tracing asked for (`VICTORY_NATIVE_RT=1`): otherwise
    // the title's DLL loads as before.
    #[cfg(windows)]
    if victory_gpu::dxr::requested() {
        all.push(bvh_builder::module());
    }
    #[cfg(windows)]
    all.push(media::module());
    #[cfg(windows)]
    all.push(dstorage::module());
    #[cfg(windows)]
    all.push(files::module());
    #[cfg(windows)]
    all.push(lifecycle::module());
    #[cfg(windows)]
    all.push(net::module());
    all.push(rpcrt4::module());
    #[cfg(windows)]
    all.push(xmem::module());
    #[cfg(windows)]
    all.push(xsystem::module());
    // `amd_fidelityfx_loader_xbox.dll` is served by the host's PC upscaler
    // only with an FSR profile asked for (`fidelityfx::use_host_dir`, before
    // this first runs): otherwise the title's own DLLs load as before.
    #[cfg(windows)]
    if fidelityfx::serving() {
        all.push(fidelityfx::module());
    }
    all
}
