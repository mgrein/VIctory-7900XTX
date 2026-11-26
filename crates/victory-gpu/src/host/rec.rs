//! A host list's D3D12 list, with barrier calls joined
//! (`VICTORY_JOIN_BARRIERS=1`; off by default; A/B-able with
//! `VICTORY_AB=<s>,VICTORY_JOIN_BARRIERS`).
//!
//! The state changes the host asks for (`crate::list::ensure_state`, the
//! aliasing before a texture's first upload) and the title's own barriers
//! wait here instead of going to the list at once; they go, in the order
//! they were asked for, in one call, at the next use of the list for
//! anything else. Every use goes through [`Rec`]'s `Deref`, so nothing is
//! recorded before a barrier asked for earlier. In a replay of capture
//! replay5 joining the calls that had no work between them saved 2.4 ms of
//! 79.5 (2,490 calls joined, 1,619 of them a texture's state change after
//! its upload followed by the next texture's aliasing).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

/// Whether barriers are held now (`VICTORY_JOIN_BARRIERS=1`, or its phase
/// of `VICTORY_AB`).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_BARRIERS") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_BARRIERS").is_some_and(|v| v == "1"))
}

/// Barriers held. Their resources are borrowed: the list's states hold the
/// textures (`HostList::transition`), and the title's barriers name what
/// the title keeps alive until the list has run.
struct Held(Vec<D3D12_RESOURCE_BARRIER>);

// SAFETY: borrowed interface pointers of free-threaded objects.
unsafe impl Send for Held {}

/// Since the last frame-rate line: barrier calls asked for while held, and
/// calls made of them.
pub(crate) static STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) struct Rec {
    list: ID3D12GraphicsCommandList,
    any: AtomicBool,
    held: Mutex<Held>,
    /// Whether work that can write buffers (a dispatch, a draw whose
    /// shaders store, a copy into a title buffer, our own clamp pass that
    /// ends with no wait, a title barrier) may have been recorded since the
    /// last barrier on every unordered access, ours or the title's
    /// ([`Rec::barriers`]). Starts set; read by the argument clamp under
    /// `VICTORY_CLAMP_BEFORE_UAV=0` (`crate::indirect`), which leaves its
    /// own leading wait out when nothing is left to wait for.
    uav_work: AtomicBool,
    /// Calls recorded since the last reset (each use through `Deref`, each
    /// barrier call asked for), and the most of any recording of the
    /// list's life ([`Rec::reset_count`]): a command allocator keeps the
    /// memory of its biggest recording across resets, so a pooled list
    /// that once recorded a big frame holds it for the run
    /// (`crate::list::recycle`, `VICTORY_LIST_FAT`).
    recorded: AtomicU32,
    most: AtomicU32,
}

impl Rec {
    pub(crate) fn new(list: ID3D12GraphicsCommandList) -> Self {
        Self {
            list,
            any: AtomicBool::new(false),
            held: Mutex::new(Held(Vec::new())),
            uav_work: AtomicBool::new(true),
            recorded: AtomicU32::new(0),
            most: AtomicU32::new(0),
        }
    }

    /// Whether work that may write buffers was recorded since the last
    /// barrier on every unordered access ([`Rec::uav_work`]'s field).
    pub(crate) fn uav_work(&self) -> bool {
        self.uav_work.load(Relaxed)
    }

    /// Work that may write buffers was recorded (or, `false`, a barrier on
    /// every unordered access was).
    pub(crate) fn note_uav_work(&self, recorded: bool) {
        self.uav_work.store(recorded, Relaxed);
    }

    /// The most calls any recording of the list made, this one included.
    pub(crate) fn recorded_most(&self) -> u32 {
        self.most.load(Relaxed).max(self.recorded.load(Relaxed))
    }

    /// At the list's reset: this recording's calls count toward the most
    /// ([`Rec::recorded_most`]), and the next recording counts from 0.
    pub(crate) fn reset_count(&self) {
        let n = self.recorded.swap(0, Relaxed);
        self.most.fetch_max(n, Relaxed);
    }

    /// Records `barriers` now, or holds them for the next use of the list.
    ///
    /// # Safety
    /// As `ResourceBarrier`'s: an open list, live resources.
    pub(crate) unsafe fn barriers(&self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        if barriers.is_empty() {
            return;
        }
        self.recorded.fetch_add(1, Relaxed);
        // A barrier on every unordered access orders all shader writes
        // before what follows (held ones go before the next use too).
        if barriers.iter().any(every_uav) {
            self.uav_work.store(false, Relaxed);
        }
        if !on() {
            // (What was held while it was on goes first.)
            if self.any.load(Relaxed) {
                self.flush();
            }
            // SAFETY: the caller's.
            unsafe { self.list.ResourceBarrier(barriers) };
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        // A resource whose state changes again: what is held goes first, so
        // that no call holds two changes of one resource.
        if barriers.iter().filter_map(named).any(|r| held.0.iter().filter_map(named).any(|h| h == r)) {
            // SAFETY: as in `flush`.
            unsafe { self.list.ResourceBarrier(&held.0) };
            held.0.clear();
            STATS[1].fetch_add(1, Relaxed);
        }
        // SAFETY: a plain copy of the descriptions; nothing is released
        // through them (their pointers are borrowed, `ManuallyDrop`).
        held.0.extend(barriers.iter().map(|b| unsafe { std::ptr::read(b) }));
        STATS[0].fetch_add(1, Relaxed);
        self.any.store(true, Relaxed);
    }

    /// What is held goes to the list, in one call.
    fn flush(&self) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        self.any.store(false, Relaxed);
        if held.0.is_empty() {
            return;
        }
        // SAFETY: the barriers were asked for on this open list, in order;
        // their resources are alive (`Held`).
        unsafe { self.list.ResourceBarrier(&held.0) };
        held.0.clear();
        STATS[1].fetch_add(1, Relaxed);
    }
}

/// Whether `b` is an unordered access barrier on every resource.
fn every_uav(b: &D3D12_RESOURCE_BARRIER) -> bool {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && unsafe { b.Anonymous.UAV.pResource.is_none() }
}

/// The resource a state change or aliasing names (its interface pointer).
fn named(b: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    // SAFETY: the union's member is the one `Type` says; a pointer read.
    unsafe {
        let p: Option<&ID3D12Resource> = match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => b.Anonymous.Transition.pResource.as_ref(),
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => b.Anonymous.Aliasing.pResourceAfter.as_ref(),
            _ => None,
        };
        p.map(|r| windows::core::Interface::as_raw(r) as usize)
    }
}

impl std::ops::Deref for Rec {
    type Target = ID3D12GraphicsCommandList;

    /// The list, with what is held recorded first.
    fn deref(&self) -> &ID3D12GraphicsCommandList {
        self.recorded.fetch_add(1, Relaxed);
        if self.any.load(Relaxed) {
            self.flush();
        }
        &self.list
    }
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    if asked > 0.0 {
        tracing::info!(
            "barrier calls per frame (VICTORY_JOIN_BARRIERS): {asked:.0} asked for, {made:.0} made of them"
        );
    }
}
