//! The graphics driver's warnings and errors, seen as the system records
//! them.
//!
//! A card is rarely lost out of nothing: in fxceless's runs every device
//! loss on NVIDIA had entries of the driver in the system's event log
//! (provider `nvlddmkm`, event 153, "Error occurred on GPUID") 2.7 to 55
//! seconds before the device was removed, and nothing in the layer saw them.
//! Here the layer subscribes to that log, and each entry of a display
//! driver, or of the system's `Display` source (4101, "display driver
//! stopped responding and has recovered"), at levels 1 to 3 (critical,
//! error, warning) is written to our log when the system records it, with
//! its provider, event number and level, so it can be laid beside what the
//! layer did in that moment. Information entries are left out: the
//! `Display` source writes 4125, the Auto HDR notice, when a game's window
//! first shows.
//!
//! Only logged: nothing of the layer changes. On by default;
//! `VICTORY_DRIVER_WATCH=0` subscribes to nothing.
//!
//! Ported from fxceless/victory-fork a2537066 (by fxceless) and ca8ce8a3
//! (levels 1-3, by Carver), GPL-3.0-or-later; the entry's details are ours.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows_sys::Win32::System::EventLog::{
    EVT_HANDLE, EVT_SUBSCRIBE_NOTIFY_ACTION, EvtRender, EvtRenderEventXml, EvtSubscribe, EvtSubscribeActionDeliver,
    EvtSubscribeToFutureEvents,
};

/// Entries seen since the start.
static SEEN: AtomicU64 = AtomicU64::new(0);

/// Entries of a display driver seen since the start.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn seen() -> u64 {
    SEEN.load(Relaxed)
}

/// The entries [`start`] subscribes to: the display drivers' and the
/// `Display` source's, at levels 1-3 (critical, error, warning).
const QUERY: &str = "*[System[Provider[@Name='nvlddmkm' or @Name='amdkmdag' or @Name='amdwddmg' or @Name='igfx' \
                     or @Name='igfxn' or @Name='Display'] and (Level=1 or Level=2 or Level=3)]]";

/// The value of the attribute `name` of the first element `element` in
/// `xml`, or the element's text when `name` is empty.
fn field<'a>(xml: &'a str, element: &str, name: &str) -> Option<&'a str> {
    let open = format!("<{element}");
    let at = xml.find(&open)? + open.len();
    let rest = &xml[at..];
    let end = rest.find('>')?;
    if name.is_empty() {
        let text = &rest[end + 1..];
        return Some(text[..text.find('<')?].trim());
    }
    let head = &rest[..end];
    let key = format!("{name}='");
    let (key, quote) = match head.find(&key) {
        Some(_) => (key, '\''),
        None => (format!("{name}=\""), '"'),
    };
    let from = head.find(&key)? + key.len();
    let value = &head[from..];
    Some(&value[..value.find(quote)?])
}

/// What the entry is, from its XML: provider, event number, level.
fn described(xml: &str) -> String {
    let level = match field(xml, "Level", "") {
        Some("1") => "critical",
        Some("2") => "error",
        Some("3") => "warning",
        _ => "entry",
    };
    format!(
        "{} {level} {}",
        field(xml, "Provider", "Name").unwrap_or("?"),
        field(xml, "EventID", "").unwrap_or("?"),
    )
}

/// The entry's XML, if the system gives it.
fn xml(event: EVT_HANDLE) -> Option<String> {
    let mut buffer = vec![0u16; 4096];
    let (mut used, mut count) = (0u32, 0u32);
    // SAFETY: a delivered event and a buffer of the size given.
    let ok = unsafe {
        EvtRender(0, event, EvtRenderEventXml, (buffer.len() * 2) as u32, buffer.as_mut_ptr().cast(), &mut used, &mut count)
    } != 0;
    if !ok {
        return None;
    }
    let chars = (used as usize / 2).min(buffer.len());
    Some(String::from_utf16_lossy(&buffer[..chars]).trim_end_matches('\0').to_owned())
}

/// An entry was recorded.
fn note(what: &str) {
    let n = SEEN.fetch_add(1, Relaxed) + 1;
    tracing::error!(n, frame = crate::frame::current(), "the graphics driver wrote to the system's event log: {what}");
}

unsafe extern "system" fn on_event(action: EVT_SUBSCRIBE_NOTIFY_ACTION, _context: *const core::ffi::c_void, event: EVT_HANDLE) -> u32 {
    if action == EvtSubscribeActionDeliver {
        note(&xml(event).map(|x| described(&x)).unwrap_or_else(|| "an entry (unreadable)".to_owned()));
    }
    0
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Subscribes to the display drivers' entries in the system's event log,
/// once. Entries from before the call are not delivered.
pub(crate) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("VICTORY_DRIVER_WATCH").is_some_and(|v| v == "0") {
            return;
        }
        let (channel, query) = (wide("System"), wide(QUERY));
        // SAFETY: valid strings for the call's length; the callback touches
        // only statics and the log; the subscription is kept for the life
        // of the process.
        let subscription = unsafe {
            EvtSubscribe(
                0,
                std::ptr::null_mut(),
                channel.as_ptr(),
                query.as_ptr(),
                0,
                std::ptr::null(),
                Some(on_event),
                EvtSubscribeToFutureEvents,
            )
        };
        if subscription == 0 {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "no subscription to the system's event log: the graphics driver's errors are not seen as they happen"
            );
        } else {
            tracing::info!(
                "the graphics driver's warnings and errors in the system's event log are logged as they are recorded \
                 (VICTORY_DRIVER_WATCH=0: not watched)"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_is_described_and_counted() {
        let xml = "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System>\
                   <Provider Name='nvlddmkm'/><EventID Qualifiers='49322'>153</EventID><Version>0</Version>\
                   <Level>2</Level></System></Event>";
        assert_eq!(described(xml), "nvlddmkm error 153");
        let other = "<Event><System><Provider Name=\"Display\" Guid=\"{x}\"/><EventID>4101</EventID><Level>3</Level></System></Event>";
        assert_eq!(described(other), "Display warning 4101");
        assert_eq!(described("<Event/>"), "? entry ?");
        let before = seen();
        note("a test entry");
        assert_eq!(seen(), before + 1);
    }

    /// The system accepts the channel and the query (a query it cannot read
    /// is refused at once).
    #[test]
    fn the_system_takes_the_subscription() {
        let (channel, query) = (wide("System"), wide(QUERY));
        // SAFETY: as in `start`.
        let subscription = unsafe {
            EvtSubscribe(0, std::ptr::null_mut(), channel.as_ptr(), query.as_ptr(), 0, std::ptr::null(), Some(on_event), EvtSubscribeToFutureEvents)
        };
        assert_ne!(subscription, 0, "{}", std::io::Error::last_os_error());
        // SAFETY: our own subscription.
        unsafe { windows_sys::Win32::System::EventLog::EvtClose(subscription) };
    }
}
