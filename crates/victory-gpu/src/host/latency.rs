//! Input latency on the presenting side: how far the title's thread runs
//! ahead of the card (`VICTORY_FRAMES_AHEAD`). The title's thread comes here
//! once a frame, at the end of its present ([`title_presented`]).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

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
/// microseconds they waited, those that timed out.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// On the title's thread, at the end of its present of `frame` (once the
/// present was given to the lane or shown), before any pacing: waits for
/// the card to have run frame `frame - VICTORY_FRAMES_AHEAD`.
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
    let [frames, waits, us, late] = [0, 1, 2, 3].map(|i| STATS[i].swap(0, Ordering::Relaxed));
    if frames == 0 || last == 0 {
        return;
    }
    let per = |us: u64| us as f64 / 1000.0 / frames as f64;
    tracing::info!(
        "frames ahead (VICTORY_FRAMES_AHEAD={}): waited {:.1} ms/frame for the card, at {waits} of {frames} presents \
         ({late} gave up after 250 ms)",
        frames_ahead(),
        per(us)
    );
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
}
