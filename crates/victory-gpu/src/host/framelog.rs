//! A frame that took long, explained: when a present comes more than
//! `VICTORY_LONG_FRAME_MS` (1000 if unset; `0` off) after the one before,
//! one line says what the counters of the panel ([`super::panel`]) grew by
//! over that frame (the submitting thread's work and waits for the card,
//! the card idle, uploads, textures, copy-back, draws, the process's
//! processor time) with the page faults and the objects waiting to be let
//! go. On by default in diagnostics builds; in others only with the
//! variable set. The frame-rate line resets some of those counters: a long
//! frame across it reads what came after the reset alone.
//!
//! After fxceless/victory-fork's frame log (`framelog.rs`, 39731795 and
//! a2537066, GPL-3.0-or-later), much reduced: long frames only.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::panel::Counters;

/// How long a frame must take to be explained, `None` when off.
fn threshold() -> Option<Duration> {
    static AT: OnceLock<Option<Duration>> = OnceLock::new();
    *AT.get_or_init(|| {
        let ms = match std::env::var("VICTORY_LONG_FRAME_MS") {
            Ok(v) => v.trim().parse::<u64>().unwrap_or(1000),
            Err(_) if cfg!(feature = "diagnostics") => 1000,
            Err(_) => 0,
        };
        (ms > 0).then(|| Duration::from_millis(ms))
    })
}

/// The line for a frame of `ms` milliseconds, `c` what the counters grew by
/// over it.
fn text(ms: f64, c: &[f64], faults: u32, waiting: usize, frame: u64) -> String {
    format!(
        "long frame {frame}: {ms:.0} ms; the submitting thread worked {:.0} ms and waited for the card {:.0} ms; \
         the card had nothing of ours {:.0} ms ({:.0} ms of it behind us); {:.1} MB sent up, textures {:.1} MB, \
         copy-back {:.0} ms ({:.1} MB); {:.0} draws, {:.0} dropped; recording {:.0} ms; processor time {:.0} ms \
         ({:.1} cores); {faults} page faults; {waiting} objects waiting to be let go",
        c[0] / 1e3,
        c[3] / 1e3,
        c[1] / 1e3,
        c[2] / 1e3,
        c[7] / 1e6,
        c[4] / 1e6,
        c[5] / 1e3,
        c[6] / 1e6,
        c[10],
        c[8],
        c[9] / 1e6,
        c[11] / 1e3,
        c[11] / 1e3 / ms.max(0.001),
    )
}

/// What the last present saw.
struct Last {
    at: Instant,
    counters: Counters,
    faults: u32,
}

/// A frame was presented.
pub(super) fn presented() {
    let Some(threshold) = threshold() else { return };
    static LAST: Mutex<Option<Last>> = Mutex::new(None);
    let Ok(mut last) = LAST.try_lock() else { return };
    let now = Instant::now();
    let counters = Counters::now();
    let faults = crate::memstat::process().2;
    if let Some(was) = last.as_ref()
        && now - was.at >= threshold
    {
        let c = counters.since(was.counters, 1.0);
        let line = text(
            (now - was.at).as_secs_f64() * 1e3,
            &c,
            faults.wrapping_sub(was.faults),
            super::retired_waiting(),
            crate::frame::current(),
        );
        tracing::warn!("{line}");
    }
    *last = Some(Last { at: now, counters, faults });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_frame_is_said_in_milliseconds() {
        let mut c = [0.0; super::super::panel::COUNTERS];
        (c[0], c[3], c[7], c[10], c[11]) = (800_000.0, 150_000.0, 3_000_000.0, 1200.0, 2_400_000.0);
        let line = text(1200.0, &c, 42, 7, 99);
        assert!(line.starts_with("long frame 99: 1200 ms; the submitting thread worked 800 ms and waited for the card 150 ms"), "{line}");
        assert!(line.contains("3.0 MB sent up"), "{line}");
        assert!(line.contains("processor time 2400 ms (2.0 cores); 42 page faults; 7 objects"), "{line}");
    }
}
