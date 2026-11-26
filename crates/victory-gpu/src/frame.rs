//! Frame pacing: the title sets a frame interval and waits for each frame
//! to begin. Ours is a clock started by the first interval set, ticking
//! once per interval; the token is the frame number. Each event kind
//! (origin, vblank, flip) happens once per tick, so two waits for the same
//! kind never end on the same tick: the title polls the origin
//! with "token on time-out" (starting its next frame at once) and is paced
//! by a blocking wait for the flip before each present.
//!
//! shortcut: event offsets are ignored (every event is at the tick); tie
//! the clock to presentation if a title paces by them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::object::{HRESULT, S_FALSE, S_OK, This};

/// What a wait returns when its time is up first.
const TIMEOUT: HRESULT = S_FALSE;
/// The frame-origin event: the one that starts frames and makes tokens.
const ORIGIN: u32 = 0;
/// Make the frame's token even if the wait times out (the title starts the
/// frame early).
const TOKEN_ON_TIMEOUT: u32 = 2;
const SIXTIETH: Duration = Duration::from_micros(16_667);
/// The flip event: the title waits for it (blocking) before each present.
const FLIP: u32 = 2;

struct Clock {
    start: Instant,
    interval: Duration,
    /// The tick number at `start` (ticks go on across interval changes).
    first: u64,
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static FRAME: AtomicU64 = AtomicU64::new(0);

fn clock() -> std::sync::MutexGuard<'static, Option<Clock>> {
    CLOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method: a frame every `length` microseconds. The period is how
/// many intervals a frame may take to reach the display (pipeline depth),
/// not part of the rate.
pub(crate) extern "system" fn set_interval(
    _: This,
    _output: This,
    length: u32,
    _period: u32,
    _flags: u32,
) -> HRESULT {
    let interval = match (fps(), length) {
        (Some(n), _) if n > 0 => Duration::from_nanos(1_000_000_000 / u64::from(n)),
        (_, 0) => SIXTIETH,
        (_, us) => Duration::from_micros(u64::from(us)),
    };
    tracing::info!(
        ?interval,
        title_us = length,
        overridden = fps().is_some_and(|n| n > 0),
        uncapped = uncapped(),
        "frame interval"
    );
    // A new interval starts from now, after the last tick handed out.
    let first = LAST
        .iter()
        .map(|t| t.load(Ordering::Relaxed))
        .max()
        .unwrap_or(0);
    *clock() = Some(Clock {
        start: Instant::now(),
        interval,
        first,
    });
    S_OK
}

/// `VICTORY_FPS=n` (read here once; unset: the title's interval): a frame
/// interval of 1/n s whatever the title sets; `VICTORY_FPS=0`: no pacing
/// sleep at all, every wait ends at once on the next tick (ticks and tokens
/// still count one by one). The frame-pacing part of Belobug fa074817 and
/// 2b6adbb7 (beloooo) and of fxceless PR #1193 (mokillos), without a
/// default. Tests keep the title's interval.
fn fps() -> Option<u32> {
    if cfg!(test) {
        return None;
    }
    static FPS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *FPS.get_or_init(|| std::env::var("VICTORY_FPS").ok()?.trim().parse().ok())
}

/// `VICTORY_FPS=0`: waits do not sleep.
fn uncapped() -> bool {
    fps() == Some(0)
}

/// Device method: events are scheduled at every tick.
pub(crate) extern "system" fn schedule(
    _: This,
    _kind: u32,
    _offset: u32,
    _list: *const u8,
    _flags: u32,
) -> HRESULT {
    S_OK
}

/// Frames begun so far (frame waits that ended).
pub(crate) fn current() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// For tests of what is done once a frame.
#[cfg(test)]
pub(crate) fn test_advance() {
    FRAME.fetch_add(1, Ordering::Relaxed);
}

/// Per event kind (origin, vblank, flip and later kinds): the last tick a
/// wait for it ended on.
static LAST: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn last(kind: u32) -> &'static AtomicU64 {
    &LAST[(kind as usize).min(LAST.len() - 1)]
}

/// The next tick after now and after the last one a wait for `kind` took
/// (an event happens once per tick: a wait woken a little early must not
/// let the next one end on the same tick), and how long until it. A wait
/// that comes within `limit` takes the tick here, under the clock's lock,
/// so two threads waiting for one kind never take the same tick (from
/// DexNom 810f4422, Manuel Pérez).
///
/// A blocking wait for an event that has happened since the last wait for
/// it took its tick (a frame took longer than a tick) ends at once, on the
/// tick that passed: sleeping on to the next one cost 17 ms of every
/// 730 ms frame (run 58: "2.0 frame waits sleeping 16.8 ms"). Polls (a
/// time-out shorter than a tick) answer as before.
fn next_tick(kind: u32, limit: Duration) -> (u64, Duration) {
    let mut clock = clock();
    let c = clock.get_or_insert_with(|| Clock {
        start: Instant::now(),
        interval: match fps() {
            Some(n) if n > 0 => Duration::from_nanos(1_000_000_000 / u64::from(n)),
            _ => SIXTIETH,
        },
        first: 0,
    });
    let last = last(kind);
    if uncapped() {
        // No sleep: the next tick, at once.
        return (last.fetch_add(1, Ordering::Relaxed) + 1, Duration::ZERO);
    }
    let elapsed = c.start.elapsed().as_nanos();
    let interval = c.interval.as_nanos().max(1);
    let taken = last.load(Ordering::Relaxed);
    let mut tick = (c.first + (elapsed / interval + 1) as u64).max(taken + 1);
    let at = u128::from(tick - c.first) * interval;
    let mut wait = Duration::from_nanos(at.saturating_sub(elapsed) as u64);
    if !late_waits_sleep() && limit >= wait && tick > taken + 1 {
        tick -= 1;
        wait = Duration::ZERO;
    }
    if wait <= limit {
        last.fetch_max(tick, Ordering::Relaxed);
    }
    (tick, wait)
}

/// `VICTORY_LATE_WAITS=1`: a blocking wait for an event that has happened
/// since the last wait for it sleeps to the next tick, as before (read here,
/// once).
fn late_waits_sleep() -> bool {
    static SLEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SLEEP.get_or_init(|| std::env::var_os("VICTORY_LATE_WAITS").is_some_and(|v| v == "1"))
}

/// Device method: blocks until the next tick (or the time-out); an origin
/// wait then hands out the new frame's token. Other events keep the token
/// the title passes. Ticks are absolute (start plus whole intervals), so
/// lateness never accumulates.
pub(crate) extern "system" fn wait(
    _: This,
    kind: u32,
    timeout_ms: u32,
    _list: *const u8,
    flags: u32,
    token: *mut u64,
) -> HRESULT {
    let origin = kind == ORIGIN;
    let limit = Duration::from_millis(u64::from(timeout_ms));
    // Paced after the present already ([`after_present`]): the flip wait
    // ends at once and takes no tick (that present took it).
    if kind == FLIP && PACED.load(Ordering::Relaxed) && limit >= interval() && PACED.swap(false, Ordering::Relaxed) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(kind, timeout_ms, flags, frame = current(), "frame wait paced after the present");
        return S_OK;
    }
    // The tick is only logged (diagnostics); taking it is what counts (the
    // binding from DexNom 882664e1, Manuel Pérez).
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (tick, wait) = next_tick(kind, limit);
    let _called = crate::timeline::span("the title's frame wait");
    let hr = if wait > limit {
        std::thread::sleep(limit);
        TIMEOUT
    } else {
        std::thread::sleep(wait);
        S_OK
    };
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        use crate::copystat::{C, add};
        add(C::PaceWaits, 1);
        add(C::PaceUs, wait.min(limit).as_micros() as u64);
    }
    // A frame begun on a time-out takes no tick: its flip still does.
    if origin && (hr == S_OK || flags & TOKEN_ON_TIMEOUT != 0) {
        // Tokens count frames (consecutive), not ticks.
        let frame = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
        if !token.is_null() {
            // SAFETY: checked non-null; the title passes its token.
            unsafe { token.write(frame) };
        }
    }
    #[cfg(feature = "diagnostics")]
    tracing::debug!(
        kind,
        timeout_ms,
        flags,
        hr,
        tick,
        wait_us = wait.as_micros() as u64,
        frame = current(),
        "frame wait"
    );
    hr
}

/// The clock's interval (that of a 60 Hz display before the title sets one).
fn interval() -> Duration {
    clock().as_ref().map_or(SIXTIETH, |c| c.interval)
}

/// `VICTORY_PACE_AFTER=1` (read here once; off by default): the title is
/// paced after its present instead of before it ([`after_present`]), so the
/// frame is shown as soon as it is made and the next one starts (reading
/// input) after the pacing sleep. Subsumes fxceless PR #457 (GamingRipperDoc).
fn pace_after() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PACE_AFTER").is_some_and(|v| v == "1"))
}

/// A present paced the flip ([`after_present`]) and no blocking flip wait
/// has ended on it yet.
static PACED: AtomicBool = AtomicBool::new(false);

/// The last thing the title's present does (`VICTORY_PACE_AFTER=1`): sleeps
/// to the next flip tick as the flip wait would (taking it, with the late
/// rule), so the title's next blocking flip wait ends at once. Tokens and
/// ticks count as before.
pub(crate) fn after_present() {
    if pace_after() {
        pace_flip();
    }
}

fn pace_flip() {
    let (_, wait) = next_tick(FLIP, Duration::MAX);
    if !wait.is_zero() {
        let _sleeping = crate::timeline::span("the title paced after its present");
        std::thread::sleep(wait);
    }
    PACED.store(true, Ordering::Relaxed);
}

/// Device method: no statistics are kept, so none are returned.
pub(crate) extern "system" fn statistics(
    _: This,
    _token: u64,
    _kinds: u32,
    count: *mut u32,
    _out: *mut u8,
) -> HRESULT {
    if !count.is_null() {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
    }
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clock is global: tests of it take turns.
    static TURN: Mutex<()> = Mutex::new(());

    fn turn() -> std::sync::MutexGuard<'static, ()> {
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn waits_end_on_ticks_and_count_frames() {
        let _turn = turn();
        let null = std::ptr::null();
        assert_eq!(set_interval(null, null, 2_000, 1, 0), S_OK);
        let mut first = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut first), S_OK);
        let before = Instant::now();
        let mut second = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut second), S_OK);
        assert!(second > first);
        // At the tick, not at the time-out (a busy PC may wake it late).
        assert!(before.elapsed() <= Duration::from_millis(500));
        assert_eq!(wait(null, 0, 0, std::ptr::null(), 0, &mut second), TIMEOUT);
        // Tokens count frames, a frame started early on a time-out too.
        let mut early = 0;
        let flags = TOKEN_ON_TIMEOUT;
        assert_eq!(
            wait(null, 0, 0, std::ptr::null(), flags, &mut early),
            TIMEOUT
        );
        let mut next = 0;
        assert_eq!(wait(null, 0, 1_000, std::ptr::null(), 0, &mut next), S_OK);
        assert!(early == second + 1 && next == early + 1);
        assert!(LAST[0].load(Ordering::Relaxed) >= 2);

        // The title's loop (the clock is global, so in the same test): an
        // origin poll that times out (and starts the frame), then a
        // blocking flip wait. Each flip takes its own tick.
        // (The ticks are far apart, and it may take a few tries: a busy PC
        // wakes a wait tens of milliseconds late at times.)
        assert_eq!(set_interval(null, null, 60_000, 3, 0), S_OK);
        let mut token = 0;
        let mut four = || -> Vec<u64> {
            (0..4)
                .map(|_| {
                    let _ = wait(null, 0, 0, std::ptr::null(), TOKEN_ON_TIMEOUT, &mut token);
                    assert_eq!(
                        wait(null, FLIP, u32::MAX, std::ptr::null(), 0, &mut token),
                        S_OK
                    );
                    LAST[2].load(Ordering::Relaxed)
                })
                .collect()
        };
        let mut flips = four();
        for _ in 0..10 {
            if flips.windows(2).all(|w| w[1] == w[0] + 1) {
                break;
            }
            flips = four();
        }
        assert!(flips.windows(2).all(|w| w[1] == w[0] + 1), "{flips:?}");
        // Even when a flip wait begins on the tick another just ended on.
        let (tick, _) = next_tick(FLIP, Duration::ZERO);
        assert!(tick > flips[3]);
    }

    /// Two waits for one kind that begin together (two threads) take two
    /// consecutive ticks: each takes its tick before it sleeps. A time-out
    /// takes none. Marking the tick after the sleep gave both the same one.
    #[test]
    fn waits_begun_together_take_consecutive_ticks() {
        let _turn = turn();
        const VBLANK: u32 = 1;
        let null = std::ptr::null();
        assert_eq!(set_interval(null, null, 20_000, 1, 0), S_OK);
        let before = last(VBLANK).load(Ordering::Relaxed);
        let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
        let both: Vec<_> = (0..2)
            .map(|_| {
                let gate = gate.clone();
                std::thread::spawn(move || {
                    gate.wait();
                    let mut token = 0;
                    assert_eq!(wait(std::ptr::null(), VBLANK, 1_000, std::ptr::null(), 0, &mut token), S_OK);
                })
            })
            .collect();
        for t in both {
            t.join().unwrap();
        }
        let after = last(VBLANK).load(Ordering::Relaxed);
        assert!(after >= before + 2, "{before} then {after}");
        let long = Duration::from_secs(1);
        let (first, _) = next_tick(VBLANK, long);
        let (second, _) = next_tick(VBLANK, long);
        assert_eq!(second, first + 1);
        next_tick(VBLANK, Duration::ZERO);
        assert_eq!(last(VBLANK).load(Ordering::Relaxed), second);
    }

    /// A present that paced the flip makes the next blocking flip wait end
    /// at once without a tick; the one after it waits again. A poll does not
    /// use it up.
    #[test]
    fn a_present_side_pace_ends_the_next_flip_wait_at_once() {
        let _turn = turn();
        let null = std::ptr::null();
        assert_eq!(set_interval(null, null, 40_000, 1, 0), S_OK);
        let mut token = 0;
        pace_flip();
        let paced = last(FLIP).load(Ordering::Relaxed);
        assert!(PACED.load(Ordering::Relaxed));
        let _ = wait(null, FLIP, 0, std::ptr::null(), 0, &mut token);
        assert!(PACED.load(Ordering::Relaxed), "a poll took it");
        // (The poll may have taken a tick, as polls do.)
        let before = last(FLIP).load(Ordering::Relaxed);
        let began = Instant::now();
        assert_eq!(wait(null, FLIP, u32::MAX, std::ptr::null(), 0, &mut token), S_OK);
        assert!(began.elapsed() < Duration::from_millis(30), "{:?}", began.elapsed());
        assert_eq!(last(FLIP).load(Ordering::Relaxed), before);
        assert!(paced >= 1 && !PACED.load(Ordering::Relaxed));
        assert_eq!(wait(null, FLIP, u32::MAX, std::ptr::null(), 0, &mut token), S_OK);
        assert!(last(FLIP).load(Ordering::Relaxed) > before);
    }
}
