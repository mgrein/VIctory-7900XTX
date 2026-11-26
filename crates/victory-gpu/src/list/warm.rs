//! A thread that keeps the pool of host lists, and the kept preludes,
//! filled ahead of need (`VICTORY_LIST_WARM`, on by default; `=0`: lists
//! and preludes are made where the pool runs short, as before).
//!
//! Why: a submission takes a host list from the pool for each title list it
//! takes the recording of (`detach`), and when the pool was empty it made
//! one there, on the submitting thread, which bounds the frame: 24 lists a
//! frame and 6 ms in run x14, with a prelude for most of them under
//! `Host::submit`; 8 % and 5 % of the submitting thread in profile x28. The
//! driver's work of making them is the same here, on a thread of its own.
//!
//! How many: the pool is kept at a target that grows by twice the lists
//! the pool lacked since the last look, up to half the pool's room, and
//! goes down by an eighth after [`SHRINK_MS`] without a lack, to
//! [`FLOOR`]; the same for the kept preludes. The thread looks every
//! [`LOOK_MS`] and when a taker finds the pool under half its target. It
//! makes nothing while the PC is short of memory (`crate::host::ram_pressure`),
//! and the sweep keeps the target's worth of lists however long they wait
//! (`super::idle_aged_out`). A list made here is closed (or open, with
//! `VICTORY_LIST_POOL_CLOSED=0`) as the pool's lists are, and has never run.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, OnceLock};

/// The least and the first target, how long without a lack before it goes
/// down, and how often the thread looks.
const FLOOR: usize = 8;
const START: usize = 16;
const SHRINK_MS: u64 = 5_000;
const LOOK_MS: u64 = 200;

/// The targets for lists and for preludes (0 until the thread runs).
static TARGET: AtomicUsize = AtomicUsize::new(0);
static PRELUDE_TARGET: AtomicUsize = AtomicUsize::new(0);
/// Lists and preludes asked of an empty pool since the thread last looked.
static MISSES: AtomicU64 = AtomicU64::new(0);
static PRELUDE_MISSES: AtomicU64 = AtomicU64::new(0);
/// Set to have the thread look now.
static WAKE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// Whether the thread runs ([`self`]).
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIST_WARM").is_none_or(|v| v != "0"))
}

/// The lists the pool is kept at now (0: none), for the sweep's floor.
pub(super) fn target() -> usize {
    TARGET.load(Relaxed)
}

/// A list was taken from the pool, `left` remain, or (`missed`) the pool
/// had none.
pub(super) fn taken(left: usize, missed: bool) {
    if !on() || !super::pools() {
        return;
    }
    started();
    if missed {
        MISSES.fetch_add(1, Relaxed);
    }
    if missed || left < TARGET.load(Relaxed) / 2 {
        wake();
    }
}

/// A prelude was taken from those kept, `left` remain, or (`missed`) none
/// was kept (`crate::host`'s `spare_prelude`).
pub(crate) fn prelude_taken(left: usize, missed: bool) {
    if !on() {
        return;
    }
    started();
    if missed {
        PRELUDE_MISSES.fetch_add(1, Relaxed);
    }
    if missed || left < PRELUDE_TARGET.load(Relaxed) / 2 {
        wake();
    }
}

fn wake() {
    let (flag, cv) = &WAKE;
    *flag.lock().unwrap_or_else(|e| e.into_inner()) = true;
    cv.notify_one();
}

/// Starts the thread, once.
fn started() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-list-warm".into()).spawn(run) {
            tracing::warn!(error = %e, "no thread to make host lists ahead of need");
        }
    });
}

/// The next target from `target`, `misses` (asks of an empty pool since the
/// last look) and `still_ms` since the target last moved, at most `most`.
fn next_target(target: usize, misses: u64, still_ms: u64, most: usize) -> usize {
    let most = most.max(FLOOR);
    if misses > 0 {
        (target as u64).saturating_add(misses.saturating_mul(2)).min(most as u64) as usize
    } else if still_ms >= SHRINK_MS && target > FLOOR {
        target.saturating_sub((target / 8).max(1)).max(FLOOR)
    } else {
        target.min(most)
    }
}

fn run() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let most = super::idle_most().0 / 2;
    let prelude_most = crate::host::spare_preludes_most() / 2;
    let mut targets = [START.min(most.max(FLOOR)), START.min(prelude_most.max(FLOOR))];
    let mut moved = [crate::host::now_ms(); 2];
    TARGET.store(targets[0], Relaxed);
    PRELUDE_TARGET.store(targets[1], Relaxed);
    tracing::info!(lists = targets[0], preludes = targets[1], "host lists are made ahead of need (VICTORY_LIST_WARM)");
    loop {
        {
            let (flag, cv) = &WAKE;
            let woken = flag.lock().unwrap_or_else(|e| e.into_inner());
            let (mut woken, _) = cv
                .wait_timeout_while(woken, std::time::Duration::from_millis(LOOK_MS), |w| !*w)
                .unwrap_or_else(|e| e.into_inner());
            *woken = false;
        }
        let now = crate::host::now_ms();
        let misses = [MISSES.swap(0, Relaxed), PRELUDE_MISSES.swap(0, Relaxed)];
        for (i, most) in [most, prelude_most].into_iter().enumerate() {
            let next = next_target(targets[i], misses[i], now.saturating_sub(moved[i]), most);
            if next != targets[i] || misses[i] > 0 {
                moved[i] = now;
            }
            targets[i] = next;
        }
        TARGET.store(targets[0], Relaxed);
        PRELUDE_TARGET.store(targets[1], Relaxed);
        if crate::host::ram_pressure() {
            continue;
        }
        fill_lists(host, targets[0]);
        while crate::host::spare_preludes() < targets[1] {
            if !crate::host::warm_prelude() {
                break;
            }
        }
    }
}

/// Makes host lists until the pool holds `target` (or its room is full).
fn fill_lists(host: &crate::host::Host, target: usize) {
    let cap = super::idle_most().0;
    loop {
        let have = super::IDLE.lock().unwrap_or_else(|e| e.into_inner()).0.len();
        if have >= target.min(cap) {
            return;
        }
        let Ok(list) = host.new_list() else {
            return;
        };
        // As the pool's lists wait.
        if super::pool_closed() {
            list.close();
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ListsAhead, 1);
        let now = crate::host::now_ms();
        let mut idle = super::IDLE.lock().unwrap_or_else(|e| e.into_inner());
        if idle.0.len() >= cap {
            // (Filled meanwhile by lists that ran: this one goes.)
            drop(idle);
            crate::host::retire(list);
            return;
        }
        idle.0.push(super::Idle(list, 0, now));
    }
}

#[cfg(test)]
mod tests {
    use super::{FLOOR, SHRINK_MS, next_target};

    #[test]
    fn the_target_grows_with_lacks_and_shrinks_slowly() {
        // Lacks: twice as many more, up to the most.
        assert_eq!(next_target(16, 5, 0, 80), 26);
        assert_eq!(next_target(70, 30, 0, 80), 80);
        // Calm but not long enough: as it is.
        assert_eq!(next_target(40, 0, SHRINK_MS - 1, 80), 40);
        // Calm long enough: an eighth less, never under the floor.
        assert_eq!(next_target(40, 0, SHRINK_MS, 80), 35);
        assert_eq!(next_target(FLOOR + 1, 0, SHRINK_MS, 80), FLOOR);
        assert_eq!(next_target(FLOOR, 0, SHRINK_MS, 80), FLOOR);
        // A smaller room than the target: the room.
        assert_eq!(next_target(40, 0, 0, 20), 20);
    }

    /// Lists made ahead wait in the pool as its lists do: closed, so the
    /// title's reset opens them.
    #[test]
    fn lists_made_ahead_wait_as_the_pools_lists_do() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        super::fill_lists(host, 2);
        let taken = super::super::IDLE.lock().unwrap_or_else(|e| e.into_inner()).0.pop();
        let super::super::Idle(list, _, _) = taken.expect("a list in the pool");
        if super::super::pool_closed() {
            list.reset();
        }
        list.close();
        host.execute(&[&list]);
    }
}
