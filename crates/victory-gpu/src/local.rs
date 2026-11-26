//! Thread-local values that cost one system TLS read per access.
//!
//! The standard library this layer is built with (x86_64-pc-windows-gnu)
//! keeps `thread_local!` values under system TLS keys, and before every
//! access of one it calls `std::sys::thread_local::guard::windows::enable`:
//! a locked add and a locked subtract on one counter shared by the whole
//! process, a fence and a `FlsSetValue`. That is so even for a
//! `const`-initialised `Cell` that needs no destructor (the key's own box
//! does). Measured here: 16 ns an access on one thread, 105 ns with four
//! threads at it and 267 ns with eight, against 4-7 ns for the system calls
//! alone. In a sampled profile of the four recording threads (some 40,000
//! list calls a frame, several thread-local accesses each, and two more per
//! allocation in diagnostics builds for the counting allocator) it was 14%
//! of their time, under that function's name.
//!
//! [`Local`] keeps its value in a fiber-local slot of its own: one
//! `FlsGetValue` an access, the value made at a thread's first access and
//! dropped when the thread ends, as with `thread_local!`. [`Word`] keeps a
//! word in a plain TLS slot, with nothing allocated (for the allocator's own
//! counters). Nothing about them can be switched off: they hold the same
//! values `thread_local!` held, only cheaper to reach.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use windows_sys::Win32::System::Threading::{
    FLS_OUT_OF_INDEXES, FlsAlloc, FlsFree, FlsGetValue, FlsSetValue, TLS_OUT_OF_INDEXES, TlsAlloc, TlsFree,
    TlsGetValue, TlsSetValue,
};

/// A value of each thread's own, made by `init` at the thread's first
/// access (`thread_local!`'s `LocalKey`, without its cost here).
pub(crate) struct Local<T: 'static> {
    /// The fiber-local slot's index plus one (0: none asked for yet).
    index: AtomicU32,
    init: fn() -> T,
}

/// A thread's value, in a line of the processor's memory of its own: the
/// values of different threads would otherwise be neighbours on the heap,
/// and their writes would have the threads wait for each other's lines.
#[repr(C, align(64))]
struct Held<T> {
    index: u32,
    value: T,
}

/// Drops a thread's value as the thread ends (the slot's callback).
unsafe extern "system" fn free<T>(value: *const c_void) {
    if value.is_null() {
        return;
    }
    let held = value.cast_mut().cast::<Held<T>>();
    // SAFETY: the slot holds only boxes `Local::first` made; cleared before
    // the value goes, so that a destructor run after this one which reaches
    // the value makes a new one instead of reading freed memory.
    unsafe {
        FlsSetValue((*held).index, std::ptr::null());
        drop(Box::from_raw(held));
    }
}

impl<T: 'static> Local<T> {
    pub(crate) const fn new(init: fn() -> T) -> Self {
        Self { index: AtomicU32::new(0), init }
    }

    /// Calls `f` with this thread's value.
    #[inline]
    pub(crate) fn with<R>(&'static self, f: impl FnOnce(&T) -> R) -> R {
        let index = self.index.load(Ordering::Acquire);
        let held = if index == 0 {
            std::ptr::null_mut()
        } else {
            // SAFETY: an index this value was given; the slot holds a box of
            // ours or nothing.
            unsafe { FlsGetValue(index - 1) }.cast::<Held<T>>()
        };
        let value = if held.is_null() { self.first() } else { held.cast_const() };
        // SAFETY: the box lives until the thread ends, which is not before
        // `f` returns; it is reached by this thread alone.
        f(unsafe { &(*value).value })
    }

    /// The thread's first access: its value is made.
    #[cold]
    #[inline(never)]
    fn first(&'static self) -> *const Held<T> {
        let index = self.slot();
        let held = Box::into_raw(Box::new(Held { index, value: (self.init)() }));
        // SAFETY: our slot; the box is freed by the slot's callback.
        if unsafe { FlsSetValue(index, held.cast_const().cast()) } == 0 {
            // SAFETY: never handed out.
            drop(unsafe { Box::from_raw(held) });
            panic!("a fiber-local slot could not be set ({})", std::io::Error::last_os_error());
        }
        held
    }

    /// The slot's index, asked for at the first access of any thread.
    fn slot(&'static self) -> u32 {
        let index = self.index.load(Ordering::Acquire);
        if index != 0 {
            return index - 1;
        }
        // SAFETY: `free::<T>` frees exactly what this value's slot holds.
        let new = unsafe { FlsAlloc(Some(free::<T>)) };
        assert!(new != FLS_OUT_OF_INDEXES, "no fiber-local slot left ({})", std::io::Error::last_os_error());
        match self.index.compare_exchange(0, new + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => new,
            Err(won) => {
                // SAFETY: ours, and no thread put anything in it.
                unsafe { FlsFree(new) };
                won - 1
            }
        }
    }
}

impl<T: Copy + 'static> Local<Cell<T>> {
    #[inline]
    pub(crate) fn get(&'static self) -> T {
        self.with(Cell::get)
    }

    #[inline]
    pub(crate) fn set(&'static self, value: T) {
        self.with(|c| c.set(value));
    }

    #[inline]
    pub(crate) fn replace(&'static self, value: T) -> T {
        self.with(|c| c.replace(value))
    }
}

impl<T: 'static> Local<RefCell<T>> {
    #[inline]
    pub(crate) fn with_borrow_mut<R>(&'static self, f: impl FnOnce(&mut T) -> R) -> R {
        self.with(|c| f(&mut c.borrow_mut()))
    }
}

/// A word of each thread's own, 0 at first, in a plain TLS slot: nothing is
/// allocated to reach it, so the global allocator may use it.
pub(crate) struct Word {
    /// The TLS slot's index plus one (0: none asked for yet).
    index: AtomicU32,
}

impl Word {
    pub(crate) const fn new() -> Self {
        Self { index: AtomicU32::new(0) }
    }

    #[inline]
    pub(crate) fn get(&'static self) -> usize {
        let index = self.index.load(Ordering::Acquire);
        if index == 0 {
            return 0;
        }
        // SAFETY: an index this word was given.
        unsafe { TlsGetValue(index - 1) as usize }
    }

    #[inline]
    pub(crate) fn set(&'static self, value: usize) {
        // SAFETY: our slot; a word, not a pointer anyone follows.
        unsafe { TlsSetValue(self.slot(), value as *const c_void) };
    }

    fn slot(&'static self) -> u32 {
        let index = self.index.load(Ordering::Acquire);
        if index != 0 {
            return index - 1;
        }
        // SAFETY: plain calls.
        let new = unsafe { TlsAlloc() };
        assert!(new != TLS_OUT_OF_INDEXES, "no TLS slot left");
        match self.index.compare_exchange(0, new + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => new,
            Err(won) => {
                // SAFETY: ours, and no thread put anything in it.
                unsafe { TlsFree(new) };
                won - 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn each_thread_has_its_own_value_made_at_its_first_access() {
        static X: Local<Cell<u64>> = Local::new(|| Cell::new(7));
        assert_eq!(X.get(), 7);
        X.set(9);
        assert_eq!(X.replace(10), 9);
        std::thread::spawn(|| {
            assert_eq!(X.get(), 7);
            X.set(1);
        })
        .join()
        .unwrap();
        assert_eq!(X.get(), 10);
    }

    #[test]
    fn values_are_dropped_when_their_thread_ends() {
        static DROPPED: AtomicUsize = AtomicUsize::new(0);
        struct Counted(Vec<u8>);
        impl Drop for Counted {
            fn drop(&mut self) {
                DROPPED.fetch_add(self.0.len(), Ordering::SeqCst);
            }
        }
        static X: Local<RefCell<Counted>> = Local::new(|| RefCell::new(Counted(Vec::new())));
        for _ in 0..4 {
            std::thread::spawn(|| X.with_borrow_mut(|c| c.0.extend_from_slice(&[1, 2, 3])))
                .join()
                .unwrap();
        }
        assert_eq!(DROPPED.load(Ordering::SeqCst), 12);
    }

    #[test]
    fn words_start_at_zero_on_every_thread() {
        static W: Word = Word::new();
        assert_eq!(W.get(), 0);
        W.set(5);
        assert_eq!(W.get(), 5);
        std::thread::spawn(|| {
            assert_eq!(W.get(), 0);
            W.set(3);
            assert_eq!(W.get(), 3);
        })
        .join()
        .unwrap();
        assert_eq!(W.get(), 5);
    }
}
