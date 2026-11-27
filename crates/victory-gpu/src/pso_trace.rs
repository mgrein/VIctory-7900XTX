//! Diagnostics: the debug layer's messages tied to the pipeline call that
//! drew them (`VICTORY_D3D12_PSO_TRACE=1`, diagnostics builds only; off by
//! default). It turns the debug layer on (as `VICTORY_D3D12_DEBUG=1` does)
//! and registers a callback on the device's `ID3D12InfoQueue1`: the layer
//! calls it on the thread that made the call, so a message that comes while
//! that thread is inside a pipeline library load or a pipeline creation
//! ([`scope`]) is logged with the pipeline's library name and the phase;
//! others are logged as unassigned. Each kind of line has a quota. A
//! library miss is normal (logged at debug level). Same-thread timing is an
//! association, not proof.
//!
//! `VICTORY_D3D12_PSO_ONLY=<name>[,<name>..]`: only the pipelines whose
//! library name contains one of these.
//!
//! The callback calls nothing of D3D12's, touches only statics and its
//! thread's local, and never unwinds. Debug-layer timing is not a
//! performance baseline.
//!
//! After DexNom/VIctory PR #59 (`pso_trace.rs`, by RoachxD),
//! GPL-3.0-or-later.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, PCSTR};

/// Whether it is asked for (and possible: diagnostics builds).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    cfg!(feature = "diagnostics") && *ON.get_or_init(|| crate::switch::on("VICTORY_D3D12_PSO_TRACE"))
}

/// The callback is registered: the info queue's own lines are left to it.
static REGISTERED: AtomicBool = AtomicBool::new(false);

pub(crate) fn registered() -> bool {
    REGISTERED.load(Relaxed)
}

/// A call in progress on this thread.
#[derive(Clone, Copy)]
struct Attempt {
    id: u64,
    /// The pipeline's library name, kept by [`names`].
    name: &'static str,
    phase: &'static str,
    selected: bool,
}

thread_local! {
    static CURRENT: Cell<Option<Attempt>> = const { Cell::new(None) };
}

static NEXT: AtomicU64 = AtomicU64::new(1);
static SCOPED: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static UNSCOPED: AtomicU64 = AtomicU64::new(0);

/// Puts back what was current when dropped.
pub(crate) struct Scope(Option<Option<Attempt>>);

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(before) = self.0 {
            let _ = CURRENT.try_with(|c| c.set(before));
        }
    }
}

/// Whether the filter lets `name` through.
fn selected(name: &str, filter: Option<&str>) -> bool {
    filter.is_none_or(|f| f.split(',').map(str::trim).any(|part| !part.is_empty() && name.contains(part)))
}

fn filter() -> Option<&'static str> {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER.get_or_init(|| std::env::var("VICTORY_D3D12_PSO_ONLY").ok()).as_deref()
}

/// Whether a line of a kind with that quota may still be said (the last
/// one says the rest are left out).
fn permit(counter: &AtomicU64, limit: u64, kind: &str) -> bool {
    let n = counter.fetch_add(1, Relaxed);
    if n == limit {
        tracing::warn!(kind, limit, "pipeline trace: further lines of this kind left out");
    }
    n < limit
}

/// The name, kept for the run (pipelines are few thousand; only when on).
fn names(name: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static NAMES: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let names = names.get_or_insert_with(HashSet::new);
    if let Some(kept) = names.get(name) {
        return kept;
    }
    let kept: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(kept);
    kept
}

/// This thread is about to make a pipeline call (`phase`: "load" or
/// "create") for the pipeline of library name `name`, until the guard is
/// dropped. Nothing when off.
pub(crate) fn scope(name: &str, phase: &'static str) -> Scope {
    if !enabled() || !registered() {
        return Scope(None);
    }
    let attempt = Attempt { id: NEXT.fetch_add(1, Relaxed), name: names(name), phase, selected: selected(name, filter()) };
    Scope(CURRENT.try_with(|c| c.replace(Some(attempt))).ok())
}

/// The library's validation message id for a pipeline not in the library
/// (a miss, which is normal).
const LIBRARY_MISS: i32 = 971;

unsafe extern "system" fn callback(
    category: D3D12_MESSAGE_CATEGORY,
    severity: D3D12_MESSAGE_SEVERITY,
    id: D3D12_MESSAGE_ID,
    description: PCSTR,
    _context: *mut core::ffi::c_void,
) {
    let _ = std::panic::catch_unwind(|| {
        // Corruption, error and warning (lower is graver).
        if severity.0 > D3D12_MESSAGE_SEVERITY_WARNING.0 {
            return;
        }
        let attempt = CURRENT.try_with(Cell::get).ok().flatten();
        if attempt.is_some_and(|a| !a.selected) {
            return;
        }
        let miss = attempt.is_some_and(|a| a.phase == "load") && id.0 == LIBRARY_MISS;
        let allowed = match (miss, attempt.is_some()) {
            (true, _) => permit(&MISSES, 128, "library misses"),
            (false, true) => permit(&SCOPED, 512, "messages of pipeline calls"),
            (false, false) => permit(&UNSCOPED, 128, "other messages"),
        };
        if !allowed {
            return;
        }
        let bytes = if description.is_null() {
            &[][..]
        } else {
            // SAFETY: the layer passes a string ended by 0, alive during the call.
            unsafe { std::ffi::CStr::from_ptr(description.0.cast()).to_bytes() }
        };
        let text = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]);
        match attempt {
            Some(a) if miss => {
                tracing::debug!(attempt = a.id, pipeline = a.name, phase = a.phase, id = id.0, "pipeline trace, library miss: {text}")
            }
            Some(a) => tracing::warn!(
                attempt = a.id,
                pipeline = a.name,
                phase = a.phase,
                category = category.0,
                severity = severity.0,
                id = id.0,
                "pipeline trace, same thread: {text}"
            ),
            None => tracing::warn!(category = category.0, severity = severity.0, id = id.0, "pipeline trace, unassigned: {text}"),
        }
    });
}

/// Registers the callback on `device` (whose debug layer is on), once.
pub(crate) fn register(device: &ID3D12Device) {
    if !enabled() || registered() {
        return;
    }
    let queue = match device.cast::<ID3D12InfoQueue1>() {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(error = %e, "pipeline trace: no ID3D12InfoQueue1 (Graphics Tools missing or an old system): nothing traced");
            return;
        }
    };
    let mut cookie = 0;
    // SAFETY: a plain function that uses statics only, no context; the
    // device lives as long as the process (the host's).
    match unsafe { queue.RegisterMessageCallback(Some(callback), D3D12_MESSAGE_CALLBACK_IGNORE_FILTERS, std::ptr::null_mut(), &mut cookie) } {
        Ok(()) => {
            REGISTERED.store(true, Relaxed);
            tracing::info!("pipeline trace on: the debug layer's messages are tied to the pipeline call on their thread (VICTORY_D3D12_PSO_TRACE)");
        }
        Err(e) => tracing::warn!(error = %e, "pipeline trace: the callback was refused"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotas_filters_and_nested_scopes() {
        let n = AtomicU64::new(0);
        assert!(permit(&n, 1, "test"));
        assert!(!permit(&n, 1, "test"));
        assert!(selected("compute-abc", None));
        assert!(selected("compute-abc", Some("xyz, abc")));
        assert!(!selected("compute-abc", Some("xyz")));
        assert!(!selected("compute-abc", Some(" , ")));
        assert!(std::ptr::eq(names("p1"), names("p1")));
        let outer = Attempt { id: 1, name: "a", phase: "load", selected: true };
        let _outer = Scope(Some(CURRENT.with(|c| c.replace(Some(outer)))));
        {
            let _inner = Scope(Some(CURRENT.with(|c| c.replace(Some(Attempt { id: 2, ..outer })))));
            assert_eq!(CURRENT.with(Cell::get).unwrap().id, 2);
        }
        assert_eq!(CURRENT.with(Cell::get).unwrap().id, 1);
        assert!(std::thread::spawn(|| CURRENT.with(Cell::get).is_none()).join().unwrap());
        // Off (no switch in tests): no scope is taken.
        assert!(scope("x", "create").0.is_none());
    }
}
