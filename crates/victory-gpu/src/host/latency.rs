//! Input latency on the presenting side: how far the title's thread runs
//! ahead of the card (`VICTORY_FRAMES_AHEAD`), the swap chain's flags
//! (tearing, a frame-latency waitable object) and sync interval, and the
//! throttle of failed swap chain creation. The title's thread comes here
//! once a frame, at the end of its present ([`title_presented`]); the host's
//! `Present` asks [`present_args`] and reports to [`presented`].

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{HRESULT, Interface};

use super::Host;

/// The host if it is made already (never made here).
fn made() -> Option<&'static Host> {
    super::HOST.get()?.as_ref()
}

/// The adapter vendor id of NVIDIA's cards.
pub(crate) const NVIDIA: u32 = 0x10DE;

/// The host adapter's vendor id (0 while there is no host or it is not
/// known).
pub(crate) fn vendor() -> u32 {
    static VENDOR: OnceLock<u32> = OnceLock::new();
    if let Some(&v) = VENDOR.get() {
        return v;
    }
    let Some(host) = made() else { return 0 };
    *VENDOR.get_or_init(|| {
        // SAFETY: a plain query of the host's adapter.
        host.adapter.as_ref().and_then(|a| unsafe { a.GetDesc1() }.ok()).map_or(0, |d| d.VendorId)
    })
}

fn number(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// `VICTORY_FRAMES_AHEAD=n` (read here once): after its present of frame F
/// the title's thread waits (at most 250 ms) until the card has run the
/// present of frame F-n, so it runs at most n frames ahead of the card and
/// what it reads (input) is shown sooner. Default 2 on NVIDIA (vendor
/// 0x10DE), 0 (off) elsewhere.
fn frames_ahead() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_FRAMES_AHEAD").unwrap_or(if vendor() == NVIDIA { 2 } else { 0 }))
}

/// Per present (by frame mod 16): the frame and the host fence value its
/// copy flight signals (0: none, it was waited for).
static PRESENT_FENCE: [(AtomicU64, AtomicU64); 16] = [const { (AtomicU64::new(0), AtomicU64::new(0)) }; 16];

/// The host's present of `frame` flies as host fence value `fence`.
pub(super) fn note_flight(frame: u64, fence: u64) {
    let (at, value) = &PRESENT_FENCE[(frame % 16) as usize];
    at.store(u64::MAX, Ordering::Release);
    value.store(fence, Ordering::Release);
    at.store(frame, Ordering::Release);
}

/// The fence value of `frame`'s present; 0 if none is noted (done).
fn flight_of(frame: u64) -> u64 {
    let (at, value) = &PRESENT_FENCE[(frame % 16) as usize];
    if at.load(Ordering::Acquire) != frame {
        return 0;
    }
    let v = value.load(Ordering::Acquire);
    if at.load(Ordering::Acquire) == frame { v } else { 0 }
}

impl Host {
    /// Waits until the host fence reaches `value`, at most `timeout`;
    /// whether it did. At once if it is there (or `value` is 0).
    pub(crate) fn wait_fence(&self, value: u64, timeout: Duration) -> bool {
        thread_local! {
            // SAFETY: plain event creation.
            static EVENT: Option<HANDLE> = unsafe { CreateEventW(None, false, false, None) }.ok();
        }
        // SAFETY: our fence; the event is this thread's.
        let done = || value == 0 || unsafe { self.fence.GetCompletedValue() } >= value;
        if done() {
            return true;
        }
        let Some(Some(event)) = EVENT.try_with(|e| *e).ok() else {
            return false;
        };
        let until = Instant::now() + timeout;
        // (An event left signalled by an earlier wait that timed out wakes
        // this one early: the value is looked at again.)
        loop {
            // SAFETY: as above.
            if unsafe { self.fence.SetEventOnCompletion(value, event) }.is_err() {
                return done();
            }
            let left = until.saturating_duration_since(Instant::now());
            // SAFETY: our event.
            let woke = unsafe { WaitForSingleObject(event, left.as_millis() as u32) } == WAIT_OBJECT_0;
            if done() {
                return true;
            }
            if !woke || left.is_zero() {
                return false;
            }
        }
    }
}

/// Since the last line: presents, those that waited for the card, the
/// microseconds they waited, those that timed out; the waits on the swap
/// chain's waitable object and their microseconds.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// On the title's thread, at the end of its present of `frame` (once the
/// present was given to the lane or shown), before any pacing: waits for
/// the card to have run frame `frame - VICTORY_FRAMES_AHEAD`, then for the
/// swap chain's waitable object (`VICTORY_FRAME_LATENCY`).
pub(crate) fn title_presented(frame: u64) {
    let Some(host) = made() else { return };
    STATS[0].fetch_add(1, Ordering::Relaxed);
    let ahead = frames_ahead();
    if ahead > 0 && frame > ahead {
        let value = flight_of(frame - ahead);
        // SAFETY: a plain query.
        if value > 0 && unsafe { host.fence.GetCompletedValue() } < value {
            let began = Instant::now();
            let met = host.wait_fence(value, Duration::from_millis(250));
            STATS[1].fetch_add(1, Ordering::Relaxed);
            STATS[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            if !met {
                STATS[3].fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    waitable_wait();
    line();
}

/// Every 5 s: what the title's thread waited at its presents.
fn line() {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let now = super::now_ms();
    let last = LOGGED.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 5_000
        || LOGGED.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err()
    {
        return;
    }
    let [frames, waits, us, late, chain, chain_us] = [0, 1, 2, 3, 4, 5].map(|i| STATS[i].swap(0, Ordering::Relaxed));
    if frames == 0 || last == 0 {
        return;
    }
    let per = |us: u64| us as f64 / 1000.0 / frames as f64;
    tracing::info!(
        "frames ahead (VICTORY_FRAMES_AHEAD={}): waited {:.1} ms/frame for the card, at {waits} of {frames} presents \
         ({late} gave up after 250 ms); swap chain latency waits {chain}, {:.1} ms/frame",
        frames_ahead(),
        per(us),
        per(chain_us)
    );
}

/// `VICTORY_TEARING` (read here once): unless `=0`, a swap chain that may
/// tear (shown at once with sync interval 0, needed for variable refresh in
/// a window) where the system supports it.
fn tearing_wanted() -> bool {
    std::env::var("VICTORY_TEARING").map_or(true, |v| v.trim() != "0")
}

/// `VICTORY_FRAME_LATENCY=n` (1 to 16, read here once; off by default): a
/// swap chain with a frame-latency waitable object and at most n frames
/// queued for the display; the title's thread waits on it (at most 100 ms)
/// once for each present issued.
fn frame_latency() -> Option<u32> {
    static N: OnceLock<Option<u32>> = OnceLock::new();
    *N.get_or_init(|| number("VICTORY_FRAME_LATENCY").filter(|n| (1..=16).contains(n)).map(|n| n as u32))
}

/// `VICTORY_PRESENT_SYNC=n` (0 to 4, read here once; 0 by default): the
/// sync interval of the host's `Present` (from fxceless PR #1074,
/// superboynick). With 0 and a swap chain that may tear, presents tear.
fn sync_interval() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| match number("VICTORY_PRESENT_SYNC") {
        None => 0,
        Some(n @ 0..=4) => n as u32,
        Some(n) => {
            tracing::warn!(n, "VICTORY_PRESENT_SYNC is 0 to 4; 0 is used");
            0
        }
    })
}

const TEARING: u32 = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32;
const WAITABLE: u32 = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0 as u32;

/// The flags a swap chain is made with (from Marvisak PR #250, wbkepler:
/// the tearing part).
pub(super) fn swap_chain_flags(factory: &IDXGIFactory2) -> u32 {
    let mut flags = 0;
    if tearing_wanted() && tearing_supported(factory) {
        flags |= TEARING;
    }
    if frame_latency().is_some() {
        flags |= WAITABLE;
    }
    flags
}

/// The flags without tearing, if they had it: a swap chain is made again
/// without it when it could not be made with.
pub(super) fn without_tearing(flags: u32) -> Option<u32> {
    (flags & TEARING != 0).then_some(flags & !TEARING)
}

fn tearing_supported(factory: &IDXGIFactory2) -> bool {
    let Ok(factory) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut allow = windows::core::BOOL(0);
    // SAFETY: a BOOL-sized out value, as this feature asks.
    unsafe {
        factory
            .CheckFeatureSupport(
                DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                (&raw mut allow).cast(),
                std::mem::size_of::<windows::core::BOOL>() as u32,
            )
            .is_ok()
            && allow.as_bool()
    }
}

/// The swap chain's waitable object (0: none), and the presents issued
/// with it and the title's waits on it.
static WAITABLE_OBJECT: AtomicUsize = AtomicUsize::new(0);
static ISSUED: AtomicU64 = AtomicU64::new(0);
static WAITED: AtomicU64 = AtomicU64::new(0);

/// A swap chain was made with `flags`: its frame latency is set and its
/// waitable object kept, and the configuration logged.
pub(super) fn swap_chain_made(_host: &Host, swap_chain: &IDXGISwapChain3, flags: u32, size: (u32, u32), format: i32) {
    let latency = frame_latency().filter(|_| flags & WAITABLE != 0);
    if let Some(n) = latency {
        // SAFETY: a swap chain made with the waitable flag.
        unsafe {
            if let Err(e) = swap_chain.SetMaximumFrameLatency(n) {
                tracing::warn!(error = %e, n, "the swap chain's frame latency not set");
            }
            let object = swap_chain.GetFrameLatencyWaitableObject();
            if !object.is_invalid() {
                WAITABLE_OBJECT.store(object.0 as usize, Ordering::Release);
            }
        }
    }
    tracing::info!(
        width = size.0,
        height = size.1,
        format,
        tearing = flags & TEARING != 0,
        frame_latency = latency.unwrap_or(0),
        sync = sync_interval(),
        "presenting to a window"
    );
}

/// The sync interval and flags of a `Present` on a swap chain made with
/// `flags`.
pub(super) fn present_args(flags: u32) -> (u32, DXGI_PRESENT) {
    let sync = sync_interval();
    if sync == 0 && flags & TEARING != 0 {
        (0, DXGI_PRESENT_ALLOW_TEARING)
    } else {
        (sync, DXGI_PRESENT(0))
    }
}

/// A `Present` returned `hr`: anything but S_OK is logged once per code;
/// one that showed counts for the waitable object.
pub(super) fn presented(hr: HRESULT) {
    if hr.0 == 0 {
        if WAITABLE_OBJECT.load(Ordering::Relaxed) != 0 {
            ISSUED.fetch_add(1, Ordering::Release);
        }
        return;
    }
    static SEEN: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&hr.0) {
        seen.push(hr.0);
        tracing::warn!(hr = format_args!("{:#010x}", hr.0 as u32), "the host's Present did not return S_OK (logged once per code)");
    }
}

/// The title's thread waits on the swap chain's waitable object, at most
/// 100 ms, once for each present issued (never for one not issued yet).
fn waitable_wait() {
    let object = WAITABLE_OBJECT.load(Ordering::Acquire);
    if object == 0 || WAITED.load(Ordering::Relaxed) >= ISSUED.load(Ordering::Acquire) {
        return;
    }
    WAITED.fetch_add(1, Ordering::Relaxed);
    let began = Instant::now();
    // SAFETY: the swap chain's waitable object; the swap chain is never
    // released.
    unsafe { WaitForSingleObject(HANDLE(object as *mut _), 100) };
    STATS[4].fetch_add(1, Ordering::Relaxed);
    STATS[5].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// When the swap chain may next be made or resized after a failure (ms of
/// [`super::now_ms`]): at most a try a second while the desktop cannot be
/// presented to (from Marvisak PR #1626, AwsAli05).
static NEXT_TRY: AtomicU64 = AtomicU64::new(0);

/// Whether a swap chain creation or resize may be tried now.
pub(super) fn may_try() -> bool {
    super::now_ms() >= NEXT_TRY.load(Ordering::Relaxed)
}

/// A creation or resize was tried: on failure the next try waits a second.
pub(super) fn tried(ok: bool) {
    NEXT_TRY.store(if ok { 0 } else { super::now_ms() + 1_000 }, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_fences_are_found_by_frame() {
        note_flight(3, 70);
        assert_eq!(flight_of(3), 70);
        // Another frame in the same slot replaces it; a missing one is done.
        note_flight(19, 90);
        assert_eq!(flight_of(3), 0);
        assert_eq!(flight_of(19), 90);
        assert_eq!(flight_of(4), 0);
    }

    #[test]
    fn tearing_is_dropped_on_retry() {
        assert_eq!(without_tearing(TEARING | WAITABLE), Some(WAITABLE));
        assert_eq!(without_tearing(WAITABLE), None);
    }

    #[test]
    fn failed_creation_is_tried_once_a_second() {
        tried(false);
        assert!(!may_try());
        tried(true);
        assert!(may_try());
    }
}
