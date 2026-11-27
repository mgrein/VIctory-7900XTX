//! A DualShock 4 or DualSense read as a HID device, for players without
//! Steam Input or DS4Windows (Windows gives these pads to XInput only
//! through such software). A thread of its own finds the pad and reads its
//! input reports; the host's first pad slot uses the latest one while no
//! XInput pad answers there. Buttons, sticks and triggers; no rumble or
//! light bar. Our own code from the public report layout; the idea from
//! XCL 0db1c5a1 and DexNom PR #79.
//!
//! `VICTORY_PLAYSTATION_PAD=0` turns it off (on by default: it only acts
//! with no XInput pad in the first slot, and only on Sony's pads).

use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

const SONY: u16 = 0x054C;
/// DualShock 4 (both models, and Sony's wireless adapter), DualSense and
/// DualSense Edge.
const DUALSHOCK4: [u16; 3] = [0x05C4, 0x09CC, 0x0BA0];
const DUALSENSE: [u16; 2] = [0x0CE6, 0x0DF2];

/// A report older than this is a pad that went away (both pads report a few
/// hundred times a second while on).
const STALE: Duration = Duration::from_millis(500);
/// Looking for a pad again this often while there is none.
const RESCAN: Duration = Duration::from_secs(2);

fn enabled() -> bool {
    std::env::var_os("VICTORY_PLAYSTATION_PAD").is_none_or(|v| v != "0")
}

/// The latest state and when it came.
static LATEST: Mutex<Option<(PadState, Instant)>> = Mutex::new(None);

/// The pad's state, `None` without a Sony pad (or switched off). The first
/// call starts the reading thread.
pub(super) fn pad() -> Option<PadState> {
    static STARTED: LazyLock<bool> = LazyLock::new(|| {
        enabled()
            && std::thread::Builder::new()
                .name("victory-playstation-pad".into())
                .spawn(reader)
                .is_ok()
    });
    if !*STARTED {
        return None;
    }
    let latest = *LATEST.lock().unwrap_or_else(PoisonError::into_inner);
    latest.filter(|(_, at)| at.elapsed() < STALE).map(|(pad, _)| pad)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    DualShock4,
    DualSense,
}

fn kind(vendor: u16, product: u16) -> Option<Kind> {
    if vendor != SONY {
        None
    } else if DUALSHOCK4.contains(&product) {
        Some(Kind::DualShock4)
    } else if DUALSENSE.contains(&product) {
        Some(Kind::DualSense)
    } else {
        None
    }
}

/// One input report as a console pad, `None` for a report of another kind.
///
/// Layouts (byte 0 is the report id):
/// - DualShock 4, report 0x01 (USB, and Bluetooth before full reports are
///   asked for): sticks LX LY RX RY at 1-4, hat and face buttons at 5,
///   shoulders/options/sticks at 6, PS and touch pad at 7, triggers at 8-9.
///   Report 0x11 (Bluetooth, full) is the same two bytes further on.
/// - DualSense, report 0x01 over USB (64 bytes): sticks at 1-4, triggers at
///   5-6, a counter at 7, then the buttons at 8-10. Over Bluetooth the short
///   report 0x01 has the DualShock 4 layout; report 0x31 (full) is the USB
///   layout one byte further on.
fn decode(kind: Kind, report: &[u8]) -> Option<PadState> {
    // (offset of LX, the triggers' offset, the buttons' offset)
    let (sticks, triggers, buttons) = match (kind, *report.first()?, report.len()) {
        (Kind::DualShock4, 0x01, 10..) => (1, 8, 5),
        (Kind::DualShock4, 0x11, 12..) => (3, 10, 7),
        (Kind::DualSense, 0x01, 64..) => (1, 5, 8),
        (Kind::DualSense, 0x01, 10..) => (1, 8, 5),
        (Kind::DualSense, 0x31, 12..) => (2, 6, 9),
        _ => return None,
    };
    let x = |b: u8| ((f32::from(b) - 128.0) / 127.0).clamp(-1.0, 1.0);
    let y = |b: u8| ((128.0 - f32::from(b)) / 127.0).clamp(-1.0, 1.0);
    let (face, shoulders) = (report[buttons], report[buttons + 1]);
    let hat = match face & 0x0F {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => DOWN | RIGHT,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => UP | LEFT,
        _ => 0,
    };
    let bits = [
        (face, 0x10, X),     // square
        (face, 0x20, A),     // cross
        (face, 0x40, B),     // circle
        (face, 0x80, Y),     // triangle
        (shoulders, 0x01, LEFT_SHOULDER),
        (shoulders, 0x02, RIGHT_SHOULDER),
        (shoulders, 0x10, VIEW), // share / create
        (shoulders, 0x20, MENU), // options
        (shoulders, 0x40, LEFT_STICK),
        (shoulders, 0x80, RIGHT_STICK),
    ];
    Some(PadState {
        buttons: bits
            .iter()
            .filter(|&&(byte, mask, _)| byte & mask != 0)
            .fold(hat, |all, &(_, _, bit)| all | bit),
        left_trigger: f32::from(report[triggers]) / 255.0,
        right_trigger: f32::from(report[triggers + 1]) / 255.0,
        left_x: x(report[sticks]),
        left_y: y(report[sticks + 1]),
        right_x: x(report[sticks + 2]),
        right_y: y(report[sticks + 3]),
    })
}

/// Finds a pad, reads it until it goes, and looks again.
#[cfg(windows)]
fn reader() {
    loop {
        match host::open() {
            Some((handle, kind)) => {
                tracing::info!(?kind, "PlayStation pad connected (HID)");
                host::read(&handle, kind);
                *LATEST.lock().unwrap_or_else(PoisonError::into_inner) = None;
                tracing::info!(?kind, "PlayStation pad gone");
            }
            None => std::thread::sleep(RESCAN),
        }
    }
}

#[cfg(not(windows))]
fn reader() {}

#[cfg(windows)]
mod host {
    use std::time::Instant;

    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, SP_DEVICE_INTERFACE_DATA,
        SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInterfaces,
        SetupDiGetClassDevsW, SetupDiGetDeviceInterfaceDetailW,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDD_ATTRIBUTES, HidD_GetAttributes, HidD_GetHidGuid,
    };
    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
    };
    use windows_sys::core::GUID;

    use super::{Kind, LATEST, PoisonError, decode, kind};

    pub(super) struct Handle(HANDLE);

    impl Drop for Handle {
        fn drop(&mut self) {
            // SAFETY: our handle, closed once.
            unsafe { CloseHandle(self.0) };
        }
    }

    fn create(path: &[u16], access: u32) -> Option<Handle> {
        // SAFETY: a NUL-terminated path; no security attributes.
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                access,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        (handle != INVALID_HANDLE_VALUE && !handle.is_null()).then_some(Handle(handle))
    }

    /// The first present Sony pad, opened for reading.
    pub(super) fn open() -> Option<(Handle, Kind)> {
        let mut guid = GUID::from_u128(0);
        // SAFETY: room for the GUID.
        unsafe { HidD_GetHidGuid(&mut guid) };
        // SAFETY: plain query; freed below.
        let set = unsafe {
            SetupDiGetClassDevsW(
                &guid,
                std::ptr::null(),
                std::ptr::null_mut(),
                DIGCF_PRESENT | DIGCF_DEVICEINTERFACE,
            )
        };
        if set == -1 {
            return None;
        }
        let mut found = None;
        for index in 0.. {
            let mut interface = SP_DEVICE_INTERFACE_DATA {
                cbSize: size_of::<SP_DEVICE_INTERFACE_DATA>() as u32,
                ..Default::default()
            };
            // SAFETY: a valid set and interface record.
            if unsafe {
                SetupDiEnumDeviceInterfaces(set, std::ptr::null(), &guid, index, &mut interface)
            } == 0
            {
                break;
            }
            // The detail record: a u32 size, then the path. Asked into a
            // buffer of u32s (aligned), the size field set as for one
            // record (8 on 64-bit: the size and one padded character).
            let mut buffer = [0u32; 1024];
            buffer[0] = if cfg!(target_pointer_width = "64") { 8 } else { 6 };
            // SAFETY: the buffer is as large as we say and starts with the
            // record's size.
            if unsafe {
                SetupDiGetDeviceInterfaceDetailW(
                    set,
                    &interface,
                    buffer.as_mut_ptr().cast(),
                    (buffer.len() * 4) as u32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            } == 0
            {
                continue;
            }
            // SAFETY: the path follows the size, NUL-terminated, in the
            // buffer.
            let path: &[u16] = unsafe {
                std::slice::from_raw_parts(buffer.as_ptr().add(1).cast(), (buffer.len() - 1) * 2)
            };
            let Some(end) = path.iter().position(|&c| c == 0) else {
                continue;
            };
            let path = &path[..=end];
            // No access: enough to ask who it is, even for devices the
            // system holds (keyboards, mice).
            let Some(probe) = create(path, 0) else {
                continue;
            };
            let mut attributes = HIDD_ATTRIBUTES {
                Size: size_of::<HIDD_ATTRIBUTES>() as u32,
                VendorID: 0,
                ProductID: 0,
                VersionNumber: 0,
            };
            // SAFETY: an open HID handle and a sized record.
            if !unsafe { HidD_GetAttributes(probe.0, &mut attributes) } {
                continue;
            }
            drop(probe);
            let Some(kind) = kind(attributes.VendorID, attributes.ProductID) else {
                continue;
            };
            // A pad has several collections; the one that reads is ours.
            if let Some(handle) = create(path, GENERIC_READ) {
                found = Some((handle, kind));
                break;
            }
        }
        // SAFETY: the set from above.
        unsafe { SetupDiDestroyDeviceInfoList(set) };
        found
    }

    /// Reads input reports until the pad goes.
    pub(super) fn read(handle: &Handle, kind: Kind) {
        let mut report = [0u8; 1024];
        loop {
            let mut got = 0;
            // SAFETY: a synchronous read into our buffer.
            let ok = unsafe {
                ReadFile(
                    handle.0,
                    report.as_mut_ptr(),
                    report.len() as u32,
                    &mut got,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                return;
            }
            if let Some(pad) = decode(kind, &report[..got as usize]) {
                *LATEST.lock().unwrap_or_else(PoisonError::into_inner) =
                    Some((pad, Instant::now()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's HID devices can be walked (finds a pad only with one
    /// plugged in; what matters is that the walk itself holds up).
    #[cfg(windows)]
    #[test]
    fn hid_devices_can_be_walked() {
        if let Some((_, kind)) = host::open() {
            eprintln!("found a {kind:?}");
        }
    }

    #[test]
    fn sony_pads_only() {
        assert_eq!(kind(SONY, 0x09CC), Some(Kind::DualShock4));
        assert_eq!(kind(SONY, 0x0CE6), Some(Kind::DualSense));
        assert_eq!(kind(SONY, 0x0268), None); // DualShock 3
        assert_eq!(kind(0x045E, 0x0CE6), None);
    }

    #[test]
    fn reports_decode_to_console_pads() {
        // DualShock 4 over USB: left stick right and up, cross and the hat
        // down-left, L1 and options, triggers half and full.
        let mut ds4 = [0u8; 64];
        ds4[..10].copy_from_slice(&[0x01, 255, 0, 128, 128, 0x25, 0x21, 0, 128, 255]);
        let pad = decode(Kind::DualShock4, &ds4).unwrap();
        assert_eq!(pad.buttons, A | DOWN | LEFT | LEFT_SHOULDER | MENU);
        assert_eq!((pad.left_x, pad.left_y), (1.0, 1.0));
        assert!(pad.right_x.abs() < 0.01 && pad.right_y.abs() < 0.01);
        assert!((pad.left_trigger - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(pad.right_trigger, 1.0);

        // The same over Bluetooth's full report: two bytes further on.
        let mut bt = [0u8; 78];
        bt[0] = 0x11;
        bt[3..12].copy_from_slice(&ds4[1..10]);
        assert_eq!(decode(Kind::DualShock4, &bt), Some(pad));

        // DualSense over USB: triggers before the buttons; hat idle (8),
        // triangle, R3, create.
        let mut ds = [0u8; 64];
        ds[..11].copy_from_slice(&[0x01, 128, 128, 0, 255, 0, 51, 7, 0x88, 0x90, 0]);
        let pad = decode(Kind::DualSense, &ds).unwrap();
        assert_eq!(pad.buttons, Y | RIGHT_STICK | VIEW);
        assert_eq!((pad.right_x, pad.right_y), (-1.0, -1.0));
        assert_eq!(pad.left_trigger, 0.0);
        assert!((pad.right_trigger - 0.2).abs() < 1e-6);

        // Its short Bluetooth report has the DualShock 4 layout.
        let short = [0x01, 128, 128, 128, 128, 0x18, 0x02, 0, 0, 0];
        let pad = decode(Kind::DualSense, &short).unwrap();
        assert_eq!(pad.buttons, X | RIGHT_SHOULDER);

        // Other reports (output, feature, unknown) are not input.
        assert_eq!(decode(Kind::DualSense, &[0x02; 48]), None);
        assert_eq!(decode(Kind::DualShock4, &[0x01; 4]), None);
        assert_eq!(decode(Kind::DualShock4, &[]), None);
    }
}
