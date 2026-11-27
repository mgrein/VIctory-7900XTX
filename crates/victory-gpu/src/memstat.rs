//! A short memory summary line, in every build (`VICTORY_MEMSTAT=1`; off by
//! default; one line every `VICTORY_MEMSTAT_SECONDS`, 10 if unset), so a
//! change to RAM or VRAM can be read off a release run as well: the
//! process's private bytes and working set, the page faults since the last
//! line, the card's memory and the PC's memory shared with it against their
//! budgets, then what the layer's own counters say holds the private bytes,
//! and what none of them accounts for (the driver's own growth, the
//! title's, our heap). The diagnostics builds' memory lines
//! (`crate::copystat`) say far more, and overlap; this one adds only
//! disjoint owners, so that the remainder means something.
//!
//! Only counters that exist anyway are read, at the line, on the presenting
//! thread: nothing is added to a hot path.
//!
//! After Belobug PR #270 (`memstat.rs`, by Adika713) and VIctory-DXR
//! 443fd629 (`memory_summary_line`, by Maren Halvorsen), GPL-3.0-or-later;
//! the remainder after XCL PR #2697's idea, written anew.

use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The process's memory, from Windows: working set, private (committed)
/// bytes, and page faults so far. Zeros if the query fails.
pub(crate) fn process() -> (u64, u64, u32) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return (0, 0, 0);
    }
    (c.WorkingSetSize as u64, c.PagefileUsage as u64, c.PageFaultCount)
}

/// Seconds between lines, `None` when off.
fn every() -> Option<Duration> {
    static EVERY: OnceLock<Option<Duration>> = OnceLock::new();
    *EVERY.get_or_init(|| {
        if !crate::switch::on("VICTORY_MEMSTAT") {
            return None;
        }
        let seconds = std::env::var("VICTORY_MEMSTAT_SECONDS")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|s| *s >= 1.0)
            .unwrap_or(10.0);
        Some(Duration::from_secs_f64(seconds))
    })
}

/// Page faults since the last count (the counter is 32 bits and wraps).
fn fault_delta(before: u32, now: u32) -> u32 {
    now.wrapping_sub(before)
}

/// The private bytes that the owners listed do not account for (never
/// below zero: the owners overlap a little, as the driver rounds up).
fn unattributed(private: u64, owners: &[(&str, u64)]) -> u64 {
    private.saturating_sub(owners.iter().map(|(_, b)| *b).sum())
}

/// The line's text. `faults`: page faults since the last line, `seconds`:
/// since it.
fn text(
    (working, private): (u64, u64),
    faults: u32,
    seconds: f64,
    card: [u64; 4],
    owners: &[(&str, u64)],
    others: &[(&str, String)],
) -> String {
    use std::fmt::Write;
    let mb = |b: u64| b >> 20;
    let mut line = format!(
        "memory summary: private {} MB, working set {} MB, page faults {:.0}/s; card {} of {} MB, shared {} of {} MB; private bytes held by",
        mb(private),
        mb(working),
        f64::from(faults) / seconds.max(0.001),
        mb(card[1]),
        mb(card[0]),
        mb(card[3]),
        mb(card[2]),
    );
    for (name, bytes) in owners {
        let _ = write!(line, " {name} {} MB,", mb(*bytes));
    }
    let _ = write!(line, " none of these (driver, title, our heap) {} MB", mb(unattributed(private, owners)));
    for (name, value) in others {
        let _ = write!(line, "; {name} {value}");
    }
    line
}

/// What the last line saw.
struct Last {
    at: Instant,
    faults: u32,
}

/// A frame was presented on `host`: says the line when it is due.
pub(crate) fn presented(host: &crate::host::Host) {
    let Some(every) = every() else { return };
    static LAST: Mutex<Option<Last>> = Mutex::new(None);
    let now = Instant::now();
    let Ok(mut last) = LAST.try_lock() else { return };
    let (working, private, faults) = process();
    let Some(was) = last.as_ref() else {
        *last = Some(Last { at: now, faults });
        return;
    };
    if now - was.at < every {
        return;
    }
    let seconds = (now - was.at).as_secs_f64();
    let delta = fault_delta(was.faults, faults);
    *last = Some(Last { at: now, faults });
    drop(last);

    let card = host.video_memory().unwrap_or_default();
    let alive: Vec<u64> = crate::host::ALIVE.iter().map(|a| a.load(Relaxed)).collect();
    let kept = crate::pipeline::kept();
    // Disjoint owners of private bytes: list scratch chunks (spare ones
    // among them), the lists' other upload buffers, the shader table's
    // containers and translations.
    let owners = [
        ("list scratch", alive[2]),
        ("other list uploads", alive[7]),
        ("shader containers", kept[1]),
        ("translations", kept[3]),
    ];
    let waiting = crate::host::retired_waiting();
    let textures = [0, 1].map(|i| crate::resource::HOST_TEXTURES[i].load(Relaxed));
    let lists: Vec<u64> = crate::host::LISTS.iter().map(|l| l.load(Relaxed)).collect();
    let others = [
        ("host lists alive", alive[0].to_string()),
        ("never reset", lists[3].to_string()),
        ("spare scratch MB", (alive[4] >> 20).to_string()),
        ("objects waiting to be let go", waiting.to_string()),
        ("host textures", format!("{} ({} MB, on the card)", textures[0], textures[1] >> 20)),
        ("pipeline library file MB", (kept[4] >> 20).to_string()),
    ];
    tracing::info!("{}", text((working, private), delta, seconds, card, &owners, &others));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faults_wrap_and_the_remainder_never_goes_below_zero() {
        assert_eq!(fault_delta(10, 25), 15);
        assert_eq!(fault_delta(u32::MAX - 1, 3), 5);
        assert_eq!(unattributed(100, &[("a", 30), ("b", 20)]), 50);
        assert_eq!(unattributed(10, &[("a", 30)]), 0);
    }

    #[test]
    fn the_line_names_every_owner_and_the_rest() {
        let line = text(
            (512 << 20, 1024 << 20),
            50,
            10.0,
            [8192 << 20, 4096 << 20, 16384 << 20, 100 << 20],
            &[("list scratch", 200 << 20), ("translations", 24 << 20)],
            &[("host lists alive", "7".to_string())],
        );
        assert!(line.starts_with("memory summary: private 1024 MB, working set 512 MB, page faults 5/s"), "{line}");
        assert!(line.contains("card 4096 of 8192 MB, shared 100 of 16384 MB"), "{line}");
        assert!(line.contains("list scratch 200 MB, translations 24 MB, none of these (driver, title, our heap) 800 MB"), "{line}");
        assert!(line.ends_with("; host lists alive 7"), "{line}");
    }

    #[test]
    fn the_process_is_measured() {
        let (working, private, faults) = process();
        assert!(working > 0 && private > 0 && faults > 0);
    }
}
