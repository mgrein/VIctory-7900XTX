//! NVIDIA Reflex through NVAPI, opt-in (`VICTORY_REFLEX=1`, read here once;
//! `VICTORY_REFLEX_BOOST=1` also asks for Low Latency Boost: the card's top
//! clocks even when it idles part of the frame). Only on an NVIDIA adapter
//! (vendor 0x10DE).
//!
//! NVAPI ships with NVIDIA's driver (`nvapi64.dll` in System32): it is
//! loaded from there, never downloaded. Its functions are found through
//! `nvapi_QueryInterface` by their ids. The ids, the structures and their
//! versions are written by hand from NVIDIA's MIT-licensed NVAPI headers
//! (`nvapi_interface.h` and `nvapi.h` of github.com/NVIDIA/nvapi, R575+;
//! dxvk-nvapi uses the same): `MAKE_NVAPI_VERSION(type, ver)` is
//! `sizeof(type) | ver << 16`. Any failure leaves Reflex off, with one info
//! line.
//!
//! What is marked, all with the title's frame numbers (its present count):
//! the title's thread sleeps in `NvAPI_D3D_Sleep` after each present and
//! marks the next frame's simulation start; its present marks the
//! simulation's end; the lane marks a frame's first and last submissions
//! (render submit); the host's `Present` is marked around. No input-sample
//! marker (the input module has no GPU hooks). Every 5 s the driver's
//! averages are logged.
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{Interface, s, w};

// Function ids (nvapi_interface.h).
const ID_INITIALIZE: u32 = 0x0150_E828;
const ID_SET_SLEEP_MODE: u32 = 0xAC1C_A9E0;
const ID_SLEEP: u32 = 0x852C_D1D2;
const ID_SET_LATENCY_MARKER: u32 = 0xD998_4C05;
const ID_GET_LATENCY: u32 = 0x1A58_7F9C;

/// `NV_LATENCY_MARKER_TYPE` (nvapi.h).
#[derive(Clone, Copy)]
#[repr(u32)]
pub(crate) enum Marker {
    SimulationStart = 0,
    SimulationEnd = 1,
    RenderSubmitStart = 2,
    RenderSubmitEnd = 3,
    PresentStart = 4,
    PresentEnd = 5,
}

/// `MAKE_NVAPI_VERSION(T, ver)`.
const fn version<T>(ver: u32) -> u32 {
    std::mem::size_of::<T>() as u32 | ver << 16
}

/// `NV_SET_SLEEP_MODE_PARAMS_V1` (44 bytes; `NvBool` is a byte).
#[repr(C)]
struct SleepModeParams {
    version: u32,
    bLowLatencyMode: u8,
    bLowLatencyBoost: u8,
    minimumIntervalUs: u32,
    bUseMarkersToOptimize: u8,
    bUseMinQueueTime: u8,
    rsvd: [u8; 30],
}

/// `NV_LATENCY_MARKER_PARAMS_V1` (88 bytes).
#[repr(C)]
struct LatencyMarkerParams {
    version: u32,
    frameID: u64,
    markerType: u32,
    rsvd0: u64,
    rsvd: [u8; 56],
}

/// `NV_LATENCY_RESULT_PARAMS_V1::FrameReport` (240 bytes), times in
/// microseconds.
#[repr(C)]
#[derive(Clone, Copy)]
struct FrameReport {
    frameID: u64,
    inputSampleTime: u64,
    simStartTime: u64,
    simEndTime: u64,
    renderSubmitStartTime: u64,
    renderSubmitEndTime: u64,
    presentStartTime: u64,
    presentEndTime: u64,
    driverStartTime: u64,
    driverEndTime: u64,
    osRenderQueueStartTime: u64,
    osRenderQueueEndTime: u64,
    gpuRenderStartTime: u64,
    gpuRenderEndTime: u64,
    gpuActiveRenderTimeUs: u32,
    gpuFrameTimeUs: u32,
    cameraConstructedTime: u64,
    crossAdapterCopyTimeUs: u32,
    aiFrameTimeUs: u32,
    rsvd: [u8; 104],
}

/// `NV_LATENCY_RESULT_PARAMS_V1` (15400 bytes): the last 64 frames.
#[repr(C)]
struct LatencyResultParams {
    version: u32,
    frameReport: [FrameReport; 64],
    rsvd: [u8; 32],
}

type QueryInterface = unsafe extern "C" fn(u32) -> *mut c_void;
type Initialize = unsafe extern "C" fn() -> i32;
type SetSleepMode = unsafe extern "C" fn(*mut c_void, *mut SleepModeParams) -> i32;
type Sleep = unsafe extern "C" fn(*mut c_void) -> i32;
type SetLatencyMarker = unsafe extern "C" fn(*mut c_void, *mut LatencyMarkerParams) -> i32;
type GetLatency = unsafe extern "C" fn(*mut c_void, *mut LatencyResultParams) -> i32;

struct Api {
    /// The host's device (it lives as long as the process).
    device: usize,
    sleep: Sleep,
    marker: SetLatencyMarker,
    latency: GetLatency,
}

static API: OnceLock<Option<Api>> = OnceLock::new();

/// `VICTORY_REFLEX=1` (read here once): Reflex on an NVIDIA adapter.
pub(crate) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_REFLEX").is_ok_and(|v| v.trim() == "1"))
}

/// `VICTORY_REFLEX_BOOST=1` (read here once): with Reflex, Low Latency
/// Boost too.
fn boost() -> bool {
    std::env::var("VICTORY_REFLEX_BOOST").is_ok_and(|v| v.trim() == "1")
}

/// Whether Reflex is on (set up and working).
pub(crate) fn on() -> bool {
    matches!(API.get(), Some(Some(_)))
}

/// Sets Reflex up on `device` once, after the swap chain is made; false
/// (and one info line) if anything is missing.
pub(crate) fn init(device: &ID3D12Device) -> bool {
    API.get_or_init(|| match load(device) {
        Ok(api) => {
            tracing::info!(boost = boost(), "NVIDIA Reflex on (VICTORY_REFLEX=1)");
            Some(api)
        }
        Err(why) => {
            tracing::info!(why, "NVIDIA Reflex left off");
            None
        }
    })
    .is_some()
}

fn load(device: &ID3D12Device) -> Result<Api, &'static str> {
    // SAFETY: the driver's own library, from the system directory; the
    // functions are cast to the signatures nvapi.h gives them.
    unsafe {
        let library = LoadLibraryW(w!("nvapi64.dll")).map_err(|_| "nvapi64.dll not found")?;
        let query: QueryInterface = std::mem::transmute(
            GetProcAddress(library, s!("nvapi_QueryInterface")).ok_or("no nvapi_QueryInterface")?,
        );
        let find = |id: u32| {
            let f = query(id);
            (!f.is_null()).then_some(f)
        };
        let initialize: Initialize = std::mem::transmute(find(ID_INITIALIZE).ok_or("no NvAPI_Initialize")?);
        let set_sleep_mode: SetSleepMode =
            std::mem::transmute(find(ID_SET_SLEEP_MODE).ok_or("no NvAPI_D3D_SetSleepMode")?);
        let sleep: Sleep = std::mem::transmute(find(ID_SLEEP).ok_or("no NvAPI_D3D_Sleep")?);
        let marker: SetLatencyMarker =
            std::mem::transmute(find(ID_SET_LATENCY_MARKER).ok_or("no NvAPI_D3D_SetLatencyMarker")?);
        let latency: GetLatency = std::mem::transmute(find(ID_GET_LATENCY).ok_or("no NvAPI_D3D_GetLatency")?);
        if initialize() != 0 {
            return Err("NvAPI_Initialize failed");
        }
        let raw = device.as_raw();
        let mut mode = SleepModeParams {
            version: version::<SleepModeParams>(1),
            bLowLatencyMode: 1,
            bLowLatencyBoost: u8::from(boost()),
            minimumIntervalUs: 0,
            bUseMarkersToOptimize: 1,
            bUseMinQueueTime: 0,
            rsvd: [0; 30],
        };
        if set_sleep_mode(raw, &mut mode) != 0 {
            return Err("NvAPI_D3D_SetSleepMode failed");
        }
        Ok(Api {
            device: raw as usize,
            sleep,
            marker,
            latency,
        })
    }
}

/// Marks `marker` of frame `frame`, if Reflex is on.
pub(crate) fn mark(marker: Marker, frame: u64) {
    let Some(Some(api)) = API.get() else { return };
    let mut params = LatencyMarkerParams {
        version: version::<LatencyMarkerParams>(1),
        frameID: frame,
        markerType: marker as u32,
        rsvd0: 0,
        rsvd: [0; 56],
    };
    // SAFETY: a function NVAPI gave for this device and a filled structure.
    unsafe { (api.marker)(api.device as *mut c_void, &mut params) };
}

/// On the title's thread after its present of `frame`: the driver's
/// sleep, then the next frame's simulation begins. Every 5 s its averages
/// are logged.
pub(crate) fn sleep_then_start(frame: u64) {
    let Some(Some(api)) = API.get() else { return };
    // SAFETY: a function NVAPI gave for this device.
    unsafe { (api.sleep)(api.device as *mut c_void) };
    mark(Marker::SimulationStart, frame + 1);
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let now = super::now_ms();
    let last = LOGGED.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= 5_000 && LOGGED.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
        log_latency(api);
    }
}

fn log_latency(api: &Api) {
    // SAFETY: plain data; all zero is a valid value.
    let mut results: Box<LatencyResultParams> = unsafe { Box::new(std::mem::zeroed()) };
    results.version = version::<LatencyResultParams>(1);
    // SAFETY: a function NVAPI gave for this device and a structure of the
    // version it is told.
    if unsafe { (api.latency)(api.device as *mut c_void, &mut *results) } != 0 {
        return;
    }
    let (n, sim_to_gpu, queue, gpu) = averages(&results.frameReport);
    if n > 0 {
        tracing::info!(
            "NVIDIA Reflex over the last {n} frames: simulation start to the card's end of rendering {sim_to_gpu:.1} ms, \
             in the system's render queue {queue:.1} ms, the card's frame {gpu:.1} ms"
        );
    }
}

/// Of the frames reported: how many, and the mean simulation start to the
/// card's end, time in the render queue and the card's frame, in ms.
fn averages(reports: &[FrameReport]) -> (usize, f64, f64, f64) {
    let real: Vec<&FrameReport> = reports
        .iter()
        .filter(|r| r.frameID != 0 && r.simStartTime != 0 && r.gpuRenderEndTime >= r.simStartTime)
        .collect();
    let n = real.len();
    if n == 0 {
        return (0, 0.0, 0.0, 0.0);
    }
    let mean = |f: &dyn Fn(&FrameReport) -> u64| real.iter().map(|r| f(r) as f64).sum::<f64>() / n as f64 / 1000.0;
    (
        n,
        mean(&|r| r.gpuRenderEndTime - r.simStartTime),
        mean(&|r| r.osRenderQueueEndTime.saturating_sub(r.osRenderQueueStartTime)),
        mean(&|r| u64::from(r.gpuFrameTimeUs)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizes and versions NVAPI checks: those of nvapi.h's structures.
    #[test]
    fn structures_have_nvapi_sizes_and_versions() {
        assert_eq!(std::mem::size_of::<SleepModeParams>(), 44);
        assert_eq!(std::mem::offset_of!(SleepModeParams, minimumIntervalUs), 8);
        assert_eq!(std::mem::offset_of!(SleepModeParams, bUseMarkersToOptimize), 12);
        assert_eq!(version::<SleepModeParams>(1), 0x1_002C);
        assert_eq!(std::mem::size_of::<LatencyMarkerParams>(), 88);
        assert_eq!(std::mem::offset_of!(LatencyMarkerParams, frameID), 8);
        assert_eq!(std::mem::offset_of!(LatencyMarkerParams, markerType), 16);
        assert_eq!(version::<LatencyMarkerParams>(1), 0x1_0058);
        assert_eq!(std::mem::size_of::<FrameReport>(), 240);
        assert_eq!(std::mem::offset_of!(FrameReport, gpuActiveRenderTimeUs), 112);
        assert_eq!(std::mem::size_of::<LatencyResultParams>(), 15400);
        assert_eq!(std::mem::offset_of!(LatencyResultParams, frameReport), 8);
        assert_eq!(version::<LatencyResultParams>(1), 0x1_3C28);
    }

    #[test]
    fn averages_skip_empty_reports() {
        // SAFETY: plain data.
        let mut reports: [FrameReport; 3] = unsafe { std::mem::zeroed() };
        reports[0] = FrameReport {
            frameID: 7,
            simStartTime: 1_000,
            gpuRenderEndTime: 41_000,
            osRenderQueueStartTime: 10_000,
            osRenderQueueEndTime: 12_000,
            gpuFrameTimeUs: 30_000,
            ..reports[2]
        };
        let (n, sim, queue, gpu) = averages(&reports);
        assert_eq!(n, 1);
        assert!((sim - 40.0).abs() < 1e-9 && (queue - 2.0).abs() < 1e-9 && (gpu - 30.0).abs() < 1e-9);
    }
}
