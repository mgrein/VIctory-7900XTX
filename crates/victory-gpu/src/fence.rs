//! Fences: a 64-bit value and the events waiting for it. Nothing runs on a
//! GPU yet, so only signals (the title's, later our queues') move one.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 5] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    // ID3D12Fence1.
    guid(
        0x433685fe,
        0xe22b,
        0x4ca0,
        [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

/// A fence that a wait or signal held back for later (in a queue's line,
/// behind flights in the air) refers to: a reference of ours, so the title
/// releasing its last one meanwhile does not free it under us. Null stands
/// for none (a placeholder) and holds nothing.
pub(crate) struct FenceRef(usize);

impl FenceRef {
    /// A placeholder that refers to no fence.
    pub(crate) const NONE: Self = Self(0);

    /// Takes a reference on `this` (a fence of ours).
    pub(crate) fn retain(this: This) -> Self {
        if !this.is_null() {
            object::add_ref(this);
        }
        Self(this as usize)
    }

    pub(crate) fn as_this(&self) -> This {
        self.0 as This
    }
}

impl Clone for FenceRef {
    fn clone(&self) -> Self {
        Self::retain(self.as_this())
    }
}

impl PartialEq for FenceRef {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl std::fmt::Debug for FenceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "fence {:#x}", self.0)
    }
}

impl Drop for FenceRef {
    fn drop(&mut self) {
        if self.0 != 0 {
            release(self.as_this());
        }
    }
}

// SAFETY: a fence is ours and free-threaded (an atomic value, a locked list
// of waiters); the reference count is atomic.
unsafe impl Send for FenceRef {}
unsafe impl Sync for FenceRef {}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

/// Whether a value we store where the title may be waiting on its address
/// (a fence's value, read at the fence's GPU address; what a list writes
/// when it has run) wakes the threads in `WaitOnAddress` there
/// (`VICTORY_ADDRESS_WAKE=0`: not, as before). Such a wait ended only at its
/// timeout: nothing on a PC wakes it when our store changes the value. A
/// wake nobody waits for is a look at an empty table in user mode
/// (cool_fox e05d4633, codec_d).
pub(crate) fn address_wake() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ADDRESS_WAKE").is_none_or(|v| v != "0"))
}

/// Wakes the threads waiting on the address `at` ([`address_wake`]).
pub(crate) fn wake_address(at: *const c_void) {
    #[cfg(windows)]
    if address_wake() && !at.is_null() {
        // SAFETY: the call only looks the address up among the waited-for
        // ones; it does not read through it.
        unsafe { windows_sys::Win32::System::Threading::WakeByAddressAll(at) };
    }
    #[cfg(not(windows))]
    let _ = at;
}

/// For how many milliseconds at most a wait for a fence with no event (the
/// title's "block until reached") waits for another thread to reach the
/// value before it fails as before (`VICTORY_FENCE_BLOCK_MS`, unset or 0:
/// it fails at once). The wait sleeps on the value's address in steps of
/// 10 ms at most and lets queues run what they hold between them
/// (cool_fox e05d4633, codec_d).
fn block_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| std::env::var("VICTORY_FENCE_BLOCK_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
}

/// Waits up to [`block_ms`] for `fence` to reach `value`: whether it did.
#[cfg(windows)]
fn block_until(fence: &Fence, value: u64) -> bool {
    let until = std::time::Instant::now() + std::time::Duration::from_millis(block_ms());
    loop {
        let now = fence.value.load(Ordering::Acquire);
        if now >= value {
            return true;
        }
        let left = until.saturating_duration_since(std::time::Instant::now()).as_millis() as u32;
        if left == 0 {
            return false;
        }
        // SAFETY: the fence's value and a copy of it, eight bytes each;
        // woken by [`set`] or at the timeout.
        unsafe {
            windows_sys::Win32::System::Threading::WaitOnAddress(
                fence.value.as_ptr().cast(),
                (&raw const now).cast(),
                8,
                left.min(10),
            );
        }
        crate::queue::release_held(false);
    }
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
/// The title's `GetCompletedValue`.
extern "system" fn completed_called(this: This) -> u64 {
    let value = completed(this);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::polled(this as usize, value);
    value
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::told("an event asked for", this as usize, value);
    if event.is_null() {
        // With submissions sent to the card and not waited for
        // (`crate::host::late`) the signal may be behind them: they are
        // settled, which sets it.
        #[cfg(windows)]
        if crate::host::late() {
            drop(waiters);
            crate::queue::before_sync();
            crate::host::settle_for_fence();
            if fence.value.load(Ordering::Acquire) >= value || (block_ms() > 0 && block_until(fence, value)) {
                return S_OK;
            }
            tracing::warn!(value, "blocking fence wait not implemented");
            return E_NOTIMPL;
        }
        #[cfg(windows)]
        if block_ms() > 0 {
            drop(waiters);
            if block_until(fence, value) {
                return S_OK;
            }
            tracing::warn!(value, "blocking fence wait not implemented");
            return E_NOTIMPL;
        }
        // shortcut: a null event means "block until reached"; nothing can
        // reach it while no GPU work runs, so fail instead of hanging.
        tracing::warn!(value, "blocking fence wait not implemented");
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}
/// Moves the fence to `value` and sets the events that waited for it.
pub(crate) fn set(this: This, value: u64) {
    let fence = fence(this);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::told("a fence set", this as usize, value);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    drop(waiters);
    // A title thread asleep on the value's address sees it now.
    wake_address(fence.value.as_ptr().cast());
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let _called = crate::timeline::span("a fence set");
    set(this, value);
    // A queue may have waited for it: what it held runs now.
    #[cfg(windows)]
    crate::queue::release_held(false);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed_called as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// A thread asleep on the value's address (as a title that reads the
    /// fence at its GPU address waits) is woken by the set, well before its
    /// own timeout.
    #[cfg(windows)]
    #[test]
    fn a_set_wakes_a_thread_waiting_on_the_value_s_address() {
        use windows_sys::Win32::System::Threading::WaitOnAddress;
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let at = fence(f).value.as_ptr() as usize;
        let waiter = std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let was = 1u64;
            loop {
                // SAFETY: the fence outlives the thread (joined below).
                let now = unsafe { (at as *const u64).read_volatile() };
                if now != was {
                    return (now, started.elapsed());
                }
                // SAFETY: eight readable bytes at both addresses.
                unsafe { WaitOnAddress(at as *const c_void, (&raw const was).cast(), 8, 20_000) };
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        set(f, 7);
        let (seen, took) = waiter.join().unwrap();
        assert_eq!(seen, 7);
        assert!(took < std::time::Duration::from_secs(10), "woken only at the timeout: {took:?}");
    }

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 9, std::ptr::null_mut()), E_NOTIMPL);
        // The title reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    #[test]
    fn held_reference_keeps_a_released_fence() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 2, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let held = FenceRef::retain(f);
        let copy = held.clone();
        assert_eq!(copy, held);
        // The title lets its own reference go: ours remain.
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 2);
        set(held.as_this(), 7);
        assert_eq!(completed(copy.as_this()), 7);
        drop(copy);
        assert_eq!(completed(held.as_this()), 7);
        // The last one, ours, frees it; a placeholder holds nothing.
        drop(held);
        drop(FenceRef::NONE);
        assert!(FenceRef::NONE.as_this().is_null());
    }

    #[test]
    fn fence1_interface_is_created_and_queried() {
        const FENCE1_IID: Iid = guid(
            0x433685fe,
            0xe22b,
            0x4ca0,
            [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
        );
        type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
        type Flags = extern "system" fn(This) -> u32;
        type Release = extern "system" fn(This) -> u32;

        // Direct creation as ID3D12Fence1, and GetCreationFlags at slot 0x58.
        let mut f1 = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 3, 0x5, &FENCE1_IID, &mut f1), S_OK);
        assert!(!f1.is_null());
        assert_eq!(slot::<Flags>(f1, 0x58)(f1), 0x5);
        assert_eq!(slot::<Release>(f1, 0x10)(f1), 0);

        // QueryInterface from ID3D12Fence to ID3D12Fence1 returns the same object.
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), 0, 0x2, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        let mut queried = std::ptr::null();
        assert_eq!(slot::<Query>(f, 0x00)(f, &FENCE1_IID, &mut queried), S_OK);
        assert_eq!(queried, f);
        assert_eq!(slot::<Flags>(queried, 0x58)(queried), 0x2);
        assert_eq!(slot::<Release>(queried, 0x10)(queried), 1);
        assert_eq!(slot::<Release>(f, 0x10)(f), 0);
    }
}
