//! A queue's error record (slots 0x48 and 0x50): the first failed request
//! with its file, offset, size and cancellation tag, plus how many failed,
//! kept until the title retrieves it, apart from the status entries; and an
//! event that is set when a request fails and reset when the record is
//! retrieved. The request's name stays empty: its string is the title's,
//! which may be released by the time the read fails.
//!
//! The record's layout (0x290 bytes, the offset at 0x230) is the 2020 one
//! with a console padding word after the count; no title we run has been
//! seen calling either slot, so it is not confirmed, and the slots stay
//! unimplemented unless `VICTORY_DSTORAGE_ERROR_RECORD=1`.

use super::{HRESULT, MEMORY_SOURCE, QueueRef, Request, walls_files};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::sync::{Mutex, OnceLock};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Threading::{CreateEventW, ResetEvent, SetEvent};

/// `VICTORY_DSTORAGE_ERROR_RECORD=1`: queues keep an error record and
/// answer slots 0x48 and 0x50; unset or 0: both slots are unimplemented.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DSTORAGE_ERROR_RECORD").is_some_and(|v| v == "1"))
}

#[repr(C)]
pub(super) struct Record {
    count: u32,
    reserved: u32,
    hr: HRESULT,
    /// 0: a read request; `u32::MAX`: no failure.
    command: u32,
    options: u64,
    destination: u64,
    destination_size: u32,
    file: [u16; 260],
    reserved_file: u32,
    offset: u64,
    memory_source: u64,
    source_size: u32,
    reserved_source: u32,
    tag: u64,
    name: [u8; 64],
}

const _: () = assert!(size_of::<Record>() == 0x290 && std::mem::offset_of!(Record, offset) == 0x230);

impl Record {
    fn empty() -> Record {
        // SAFETY: all fields are integers or arrays of integers.
        Record { command: u32::MAX, ..unsafe { std::mem::zeroed() } }
    }

    /// Counts a failure; the first one is described.
    fn note(&mut self, r: &Request, hr: HRESULT) {
        self.count = self.count.saturating_add(1);
        if self.count != 1 {
            return;
        }
        self.command = 0;
        self.hr = hr;
        self.options = r.options;
        self.destination = r.destination as u64;
        self.destination_size = r.destination_size;
        self.source_size = r.source_size;
        self.tag = r.tag;
        if r.options & MEMORY_SOURCE != 0 {
            self.memory_source = r.source as u64;
            return;
        }
        self.offset = r.offset;
        // The latest file at that address (the map keeps the newest open):
        // an older one there was released.
        if let Some(path) = walls_files().get(&(r.source as usize)) {
            for (out, unit) in self.file[..259].iter_mut().zip(path.encode_utf16()) {
                *out = unit;
            }
        }
    }
}

/// A queue's record and its event.
pub(super) struct Errors {
    record: Mutex<Record>,
    event: OwnedHandle,
}

impl Errors {
    pub(super) fn new() -> Option<Errors> {
        // SAFETY: an unnamed manual-reset event, initially clear.
        let h = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        if h.is_null() {
            return None;
        }
        // SAFETY: a handle we just created and nobody else owns.
        Some(Errors { record: Mutex::new(Record::empty()), event: unsafe { OwnedHandle::from_raw_handle(h) } })
    }

    pub(super) fn note(&self, r: &Request, hr: HRESULT) {
        let mut record = self.record.lock().unwrap_or_else(|e| e.into_inner());
        record.note(r, hr);
        // SAFETY: our event, open while `self` is.
        unsafe { SetEvent(self.event.as_raw_handle()) };
    }

    fn take(&self) -> Record {
        let mut record = self.record.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: as above.
        unsafe { ResetEvent(self.event.as_raw_handle()) };
        std::mem::replace(&mut *record, Record::empty())
    }
}

/// Slot 0x48: the queue's error event, as a duplicate handle of its own, so
/// a title that closes what it was given does not close the queue's.
pub(super) extern "system" fn event(queue: QueueRef) -> HANDLE {
    use std::os::windows::io::IntoRawHandle;
    // SAFETY: a queue we created and the title still holds.
    unsafe { &*queue }
        .state
        .errors
        .as_ref()
        .and_then(|e| e.event.try_clone().ok())
        .map_or(std::ptr::null_mut(), IntoRawHandle::into_raw_handle)
}

/// Slot 0x50: writes the record into the title's and clears ours.
pub(super) extern "system" fn retrieve(queue: QueueRef, out: *mut Record) {
    // SAFETY: as above.
    if !out.is_null()
        && let Some(errors) = &unsafe { &*queue }.state.errors
    {
        // SAFETY: the title passes room for the record.
        unsafe { out.write_unaligned(errors.take()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    #[test]
    fn the_first_failure_is_kept_and_retrieving_clears_it_and_the_event() {
        let errors = Errors::new().unwrap();
        let mut r = Request {
            options: 1,
            destination: std::ptr::null_mut(),
            page_offset: 0,
            destination_size: 256,
            source: std::ptr::null(),
            offset: 0x1_2345_6789,
            source_size: 100,
            intermediate_size: 0,
            tag: 37,
            name: c"chunk".as_ptr(),
        };
        // SAFETY: our event.
        let signalled = || unsafe { WaitForSingleObject(errors.event.as_raw_handle(), 0) } == 0;
        assert!(!signalled());
        errors.note(&r, -17);
        r.options = MEMORY_SOURCE;
        r.tag = 99;
        errors.note(&r, -23);
        assert!(signalled());
        let rec = errors.take();
        assert_eq!((rec.count, rec.hr, rec.command, rec.tag, rec.offset, rec.source_size), (2, -17, 0, 37, r.offset, 100));
        assert_eq!(rec.name, [0; 64]);
        assert!(!signalled());
        assert_eq!(errors.take().command, u32::MAX);
        r.source = std::ptr::from_ref(&r).cast();
        errors.note(&r, -23);
        let rec = errors.take();
        assert_eq!((rec.memory_source, rec.offset, rec.tag), (r.source as u64, 0, 99));
    }
}
