//! Which processors our own threads run on, where a processor has two kinds
//! of cores. `cpu_sets` and `keep_on_fast_cores` are fxceless/victory-fork's
//! (`framelog.rs`, commit 0b0074e0, GPL-3.0-or-later, as this).

/// The system's processor sets of group 0: each one's id, its efficiency
/// class (higher is faster) and its core (which two of them may share).
fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.EfficiencyClass, set.CoreIndex));
        }
        at += info.Size as usize;
    }
    sets
}

/// The sets of the fastest class, if there are slower ones too.
fn fast_of(sets: &[(u32, u8, u8)]) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() { Vec::new() } else { fast }
}

/// The sets of one fast core of the last, and all the others, for
/// [`lane_core`]; `None` with fewer than six fast cores.
fn one_core(sets: &[(u32, u8, u8)]) -> Option<(Vec<u32>, Vec<u32>)> {
    let best = sets.iter().map(|c| c.1).max()?;
    let mut cores: Vec<u8> = sets.iter().filter(|c| c.1 == best).map(|c| c.2).collect();
    cores.sort_unstable();
    cores.dedup();
    let core = *cores.last().filter(|_| cores.len() >= 6)?;
    let of = |own: bool| sets.iter().filter(|c| (c.2 == core) == own).map(|c| c.0).collect();
    Some((of(true), of(false)))
}

/// Whether the submitting thread has a fast core to itself, as far as this
/// process goes (`VICTORY_LANE_CORE=1`; `0`: not): nothing else of it runs on
/// the core's two threads. It was the one a frame waited for (94 % of it in a
/// street), and shared its core's units with whatever ran beside it.
///
/// Unset, it is on where the system has 12 logical processors or fewer (a
/// six-core desktop part: there the thread that prepares submissions is what
/// a frame waits for, and every other thread of ours shares its core), off
/// with more. Either way a core is kept only while a submitting thread of
/// ours (`VICTORY_ASYNC_SUBMIT`) has asked for one ([`settle`]).
fn lane_core() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_LANE_CORE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| lane_core_default(std::env::var_os("VICTORY_LANE_CORE").as_deref(), logical()))
}

/// [`lane_core`] from the switch's value, if set, and the logical processors.
fn lane_core_default(switch: Option<&std::ffi::OsStr>, logical: usize) -> bool {
    match switch {
        Some(v) => v == "1",
        None => logical <= 12,
    }
}

/// The logical processors the system gives this process; 24 (the machine
/// the pools' sizes were first chosen on) when that cannot be told.
pub(crate) fn logical() -> usize {
    std::thread::available_parallelism().map_or(24, |n| n.get())
}

/// Threads for a pool of ours: the logical processors over `per`, from `least`
/// to `most`. On 24 logical processors the pools are as they were before they
/// scaled (list threads 8, the title's bytes under flights 6, descriptor
/// records 6, views 4); on a six-core part with 12 they leave room for the
/// title's own threads and the submitting thread.
pub(crate) fn pool(per: usize, least: usize, most: usize) -> usize {
    pool_of(logical(), per, least, most)
}

fn pool_of(logical: usize, per: usize, least: usize, most: usize) -> usize {
    (logical / per.max(1)).clamp(least, most.max(least))
}

/// Threads that asked: a handle of each, and whether it is the submitting one.
static ASKED: std::sync::Mutex<Vec<(usize, bool)>> = std::sync::Mutex::new(Vec::new());

/// What the threads of the process, and those that asked, are told now
/// ([`lane_core`], which may have changed).
pub(crate) fn settle() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessDefaultCpuSets, SetThreadSelectedCpuSets};
    if stage() == 0 {
        return;
    }
    let sets = cpu_sets();
    let fast = fast_of(&sets);
    let asked = ASKED.lock().unwrap_or_else(|e| e.into_inner());
    // A core is kept only for a submitting thread that asked: without one
    // nobody would run on it.
    let lane_asked = asked.iter().any(|&(_, lane)| lane);
    let one = one_core(&sets).filter(|_| lane_core() && lane_asked);
    let others: &[u32] = one.as_ref().map_or(&[], |o| &o.1);
    let null = std::ptr::null::<u32>();
    // SAFETY: our own process and the system's own set ids (none: all).
    unsafe {
        let at = if others.is_empty() { null } else { others.as_ptr() };
        SetProcessDefaultCpuSets(GetCurrentProcess(), at, others.len() as u32);
    }
    let helpers: Vec<u32> = match &one {
        Some((own, _)) => fast.iter().copied().filter(|id| !own.contains(id)).collect(),
        None => fast.clone(),
    };
    for &(thread, lane) in asked.iter() {
        let sets = match &one {
            Some((own, _)) if lane => own,
            _ => &helpers,
        };
        // (None: the thread's own selection cleared, the process' default
        // applies: where all cores are alike, every one but the lane's.)
        let at = if sets.is_empty() { null } else { sets.as_ptr() };
        // SAFETY: a handle kept open, and the system's own set ids.
        unsafe { SetThreadSelectedCpuSets(thread as _, at, sets.len() as u32) };
    }
    drop(asked);
    tracing::info!(on = one.is_some(), core = ?one.as_ref().map(|o| &o.0), "the submitting thread has a core of its own (VICTORY_LANE_CORE)");
}

/// `VICTORY_FAST_CORES`: 0: no thread asks; 1: the threads a frame waits for
/// one by one; 2 (the default): the threads they share their work with too;
/// 3: those that work beside the rest as well (lists translated, descriptors
/// made): more threads by then than the fast cores run at a time, while the
/// slow ones stood idle.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| std::env::var("VICTORY_FAST_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. That fork's submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.) The title's threads are not asked for.
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

fn ask(what: &str, from: u32) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThreadId, OpenThread, THREAD_SET_LIMITED_INFORMATION,
    };
    if stage() < from {
        return;
    }
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    let (fast, of) = FAST.get_or_init(|| {
        let sets = cpu_sets();
        (fast_of(&sets), sets.len())
    });
    // All alike (or nothing known): nothing to ask for, but for the
    // submitting thread, which may have a core to itself all the same
    // ([`lane_core`]; `one_core` takes all the cores when they are alike).
    let lane = what == "victory-submit";
    if fast.is_empty() && !(lane && *of > 0) {
        return;
    }
    // SAFETY: our own thread; the handle stays open.
    let thread = unsafe { OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, GetCurrentThreadId()) };
    if thread.is_null() {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
        return;
    }
    ASKED.lock().unwrap_or_else(|e| e.into_inner()).push((thread as usize, lane));
    settle();
    if !fast.is_empty() {
        tracing::info!(what, fast = fast.len(), of, "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_of(&[(256, 1, 0), (257, 1, 0), (258, 0, 2)]), [256, 257]);
        assert!(fast_of(&[(256, 0, 0), (257, 0, 1)]).is_empty());
        assert!(one_core(&[(256, 1, 0), (257, 1, 0), (258, 0, 2)]).is_none());
        let eight: Vec<(u32, u8, u8)> = (0..16).map(|n| (n, 1, n as u8 / 2 * 2)).chain((16..24).map(|n| (n, 0, n as u8))).collect();
        let (own, others) = one_core(&eight).unwrap();
        assert_eq!((own, others.len()), (vec![14, 15], 22));
        assert!(fast_of(&[]).is_empty());
        // Six cores of one class, two threads each: no fast ones, but the
        // last core is the submitting thread's and the other ten the rest's.
        let six: Vec<(u32, u8, u8)> = (0..12).map(|n| (256 + n, 0, n as u8 / 2 * 2)).collect();
        assert!(fast_of(&six).is_empty());
        let (own, others) = one_core(&six).unwrap();
        assert_eq!((own, others.len()), (vec![266, 267], 10));
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test")).join().unwrap();
    }

    #[test]
    fn pools_scale_with_the_logical_processors() {
        // List threads: n / 3 from 2 to 8.
        assert_eq!(pool_of(24, 3, 2, 8), 8);
        assert_eq!(pool_of(32, 3, 2, 8), 8);
        assert_eq!(pool_of(12, 3, 2, 8), 4);
        assert_eq!(pool_of(8, 3, 2, 8), 2);
        assert_eq!(pool_of(4, 3, 2, 8), 2);
        // The title's bytes under flights and descriptor records: n / 4 from 2 to 6.
        assert_eq!(pool_of(24, 4, 2, 6), 6);
        assert_eq!(pool_of(16, 4, 2, 6), 4);
        assert_eq!(pool_of(12, 4, 2, 6), 3);
        assert_eq!(pool_of(6, 4, 2, 6), 2);
        assert_eq!(pool_of(1, 4, 2, 6), 2);
        // Views: n / 4 from 2 to 4.
        assert_eq!(pool_of(24, 4, 2, 4), 4);
        assert_eq!(pool_of(12, 4, 2, 4), 3);
        // Nothing divides by 0 or clamps upside down.
        assert_eq!(pool_of(12, 0, 2, 8), 8);
        assert_eq!(pool_of(12, 4, 3, 1), 3);
        assert!(pool(3, 2, 8) >= 2);
    }

    #[test]
    fn the_lane_core_is_on_by_default_up_to_twelve_logical_processors() {
        use std::ffi::OsStr;
        assert!(lane_core_default(None, 12));
        assert!(lane_core_default(None, 8));
        assert!(!lane_core_default(None, 16));
        assert!(!lane_core_default(None, 24));
        assert!(lane_core_default(Some(OsStr::new("1")), 24));
        assert!(!lane_core_default(Some(OsStr::new("0")), 12));
    }
}
