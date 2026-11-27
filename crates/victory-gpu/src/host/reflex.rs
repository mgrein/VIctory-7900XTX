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
//!
//! Also here: the driver profile that sizes NVIDIA's shader-code heap
//! (`victory nv-profile`) and the count of live pipelines' driver code
//! (`VICTORY_PSO_CODE_SIZE=1`); see their section below.
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

/// `nvapi_QueryInterface` of the driver's `nvapi64.dll`, once `NvAPI_Initialize`
/// succeeded (loaded and initialised once per process).
fn query() -> Result<QueryInterface, &'static str> {
    static QUERY: OnceLock<Result<usize, &'static str>> = OnceLock::new();
    let query = (*QUERY.get_or_init(|| {
        // SAFETY: the driver's own library, from the system directory; the
        // functions are cast to the signatures nvapi.h gives them.
        unsafe {
            let library = LoadLibraryW(w!("nvapi64.dll")).map_err(|_| "nvapi64.dll not found")?;
            let query: QueryInterface = std::mem::transmute(
                GetProcAddress(library, s!("nvapi_QueryInterface")).ok_or("no nvapi_QueryInterface")?,
            );
            let f = query(ID_INITIALIZE);
            if f.is_null() {
                return Err("no NvAPI_Initialize");
            }
            let initialize: Initialize = std::mem::transmute(f);
            if initialize() != 0 {
                return Err("NvAPI_Initialize failed");
            }
            Ok(query as usize)
        }
    }))?;
    // SAFETY: the pointer stored above, of this type.
    Ok(unsafe { std::mem::transmute::<usize, QueryInterface>(query) })
}

/// The NVAPI function with id `id`, else `missing`.
fn find(id: u32, missing: &'static str) -> Result<*mut c_void, &'static str> {
    let query = query()?;
    // SAFETY: NVAPI's lookup by id; null when the driver lacks it.
    let f = unsafe { query(id) };
    if f.is_null() { Err(missing) } else { Ok(f) }
}

fn load(device: &ID3D12Device) -> Result<Api, &'static str> {
    // SAFETY: the functions are cast to the signatures nvapi.h gives them.
    unsafe {
        let set_sleep_mode: SetSleepMode = std::mem::transmute(find(ID_SET_SLEEP_MODE, "no NvAPI_D3D_SetSleepMode")?);
        let sleep: Sleep = std::mem::transmute(find(ID_SLEEP, "no NvAPI_D3D_Sleep")?);
        let marker: SetLatencyMarker =
            std::mem::transmute(find(ID_SET_LATENCY_MARKER, "no NvAPI_D3D_SetLatencyMarker")?);
        let latency: GetLatency = std::mem::transmute(find(ID_GET_LATENCY, "no NvAPI_D3D_GetLatency")?);
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

// The driver's heap for the machine code of pipelines in use.
//
// NVIDIA's D3D12 driver keeps the compiled code of the pipelines a submission
// uses in one heap of 512 MB by default; a submission needing more removes the
// device (0x887a0005, a DRED page fault at VA 0x0 with no allocation). Measured
// and described in xcl-dev/xcl issue #2403; driver setting 0x001ABC0F of a
// profile raises it (its value in bytes: NVIDIA's own profiles for large
// games hold 1 or 2 GiB, as Profile Inspector shows), the approach of
// xcl-dev/xcl PR #2590. Written here from NVIDIA's MIT-licensed NVAPI headers
// (`nvapi.h`, `nvapi_interface.h`, `nvapi_lite_common.h`), not from that code.
//
// `victory nv-profile install` puts the executable into a driver profile of
// its own, "VIctory shader heap (<exe>)", holding that one setting; the
// driver applies it to later runs by the executable's name. Writing needs an
// elevated prompt once (`scripts\play.ps1 -NvProfile` asks for it); reading
// does not. A profile of that name is only changed or deleted while it holds
// nothing else (see `owned`); an executable some other profile already
// claims is left alone.
// (The driver's own database already gives the name victory.exe another
// game's predefined profile, so `play.ps1 -NvProfile` uses a copy named
// victory_heap.exe.)

const ID_DRS_CREATE_SESSION: u32 = 0x0694_D52E;
const ID_DRS_DESTROY_SESSION: u32 = 0xDAD9_CFF8;
const ID_DRS_LOAD_SETTINGS: u32 = 0x375D_BD6B;
const ID_DRS_SAVE_SETTINGS: u32 = 0xFCBC_7E14;
const ID_DRS_CREATE_PROFILE: u32 = 0xCC17_6068;
const ID_DRS_DELETE_PROFILE: u32 = 0x1709_3206;
const ID_DRS_GET_CURRENT_GLOBAL_PROFILE: u32 = 0x617B_FF9F;
const ID_DRS_GET_PROFILE_INFO: u32 = 0x61CD_6FD6;
const ID_DRS_FIND_PROFILE_BY_NAME: u32 = 0x7E4A_9A0B;
const ID_DRS_CREATE_APPLICATION: u32 = 0x4347_A9DE;
const ID_DRS_FIND_APPLICATION_BY_NAME: u32 = 0xEEE5_66B2;
const ID_DRS_SET_SETTING: u32 = 0x577D_D202;
const ID_DRS_GET_SETTING: u32 = 0x73BF_8338;

/// The driver setting that sizes the shader-code heap, in bytes.
const SHADER_HEAP_SETTING: u32 = 0x001A_BC0F;
/// What the driver uses without the setting.
pub const SHADER_HEAP_DEFAULT_MB: u32 = 512;
/// The sizes `install` takes: 4096 MB in bytes does not fit the setting's u32.
pub const SHADER_HEAP_MB: std::ops::RangeInclusive<u32> = 512..=4095;

// `NvAPI_Status` values (nvapi_lite_common.h).
const NVAPI_INVALID_USER_PRIVILEGE: i32 = -137;
const NVAPI_SETTING_NOT_FOUND: i32 = -160;
const NVAPI_PROFILE_NOT_FOUND: i32 = -163;
const NVAPI_EXECUTABLE_NOT_FOUND: i32 = -166;
const NVAPI_EXECUTABLE_ALREADY_IN_USE: i32 = -167;

/// `NVDRS_DWORD_TYPE`, `NVDRS_CURRENT_PROFILE_LOCATION`.
const DRS_DWORD: u32 = 0;
const DRS_IN_PROFILE: u32 = 0;

/// `NvAPI_UnicodeString`: UTF-16, NUL-terminated.
type Wide = [u16; 2048];

fn wide(text: &str) -> Box<Wide> {
    let mut w = Box::new([0u16; 2048]);
    for (slot, unit) in w.iter_mut().take(2047).zip(text.encode_utf16()) {
        *slot = unit;
    }
    w
}

fn narrow(w: &Wide) -> String {
    let end = w.iter().position(|&u| u == 0).unwrap_or(w.len());
    String::from_utf16_lossy(&w[..end])
}

/// `NVDRS_PROFILE_V1` (4116 bytes; `NVDRS_GPU_SUPPORT` is a u32 of bits).
#[repr(C)]
struct DrsProfile {
    version: u32,
    profileName: Wide,
    gpuSupport: u32,
    isPredefined: u32,
    numOfApps: u32,
    numOfSettings: u32,
}

/// `NVDRS_APPLICATION_V4` (20492 bytes; `isMetro`, `isCommandLine` and the
/// reserved bits are one u32).
#[repr(C)]
struct DrsApplication {
    version: u32,
    isPredefined: u32,
    appName: Wide,
    userFriendlyName: Wide,
    launcher: Wide,
    fileInFolder: Wide,
    flags: u32,
    commandLine: Wide,
}

/// `NVDRS_SETTING_V1` (12320 bytes, packed to 4): the two value unions are
/// 4100 bytes each (`NVDRS_BINARY_SETTING`, their largest member); a DWORD
/// value is their first 4 bytes.
#[repr(C)]
struct DrsSetting {
    version: u32,
    settingName: Wide,
    settingId: u32,
    settingType: u32,
    settingLocation: u32,
    isCurrentPredefined: u32,
    isPredefinedValid: u32,
    predefined: [u8; 4100],
    current: [u8; 4100],
}

/// A zeroed structure of NVAPI's with its version set (they are too big for
/// the stack). Only for the structures above, which start with their u32
/// version and are plain integers and arrays.
fn zeroed<T>(ver: u32) -> Box<T> {
    // SAFETY: plain integers and arrays: all zero is a valid value.
    let mut b: Box<T> = unsafe { Box::new(std::mem::zeroed()) };
    // SAFETY: the structure starts with its u32 version.
    unsafe { *(std::ptr::from_mut(&mut *b).cast::<u32>()) = version::<T>(ver) };
    b
}

type Handle = *mut c_void;
type DrsCreateSession = unsafe extern "C" fn(*mut Handle) -> i32;
type DrsSession = unsafe extern "C" fn(Handle) -> i32;
type DrsCreateProfile = unsafe extern "C" fn(Handle, *mut DrsProfile, *mut Handle) -> i32;
type DrsDeleteProfile = unsafe extern "C" fn(Handle, Handle) -> i32;
type DrsGlobalProfile = unsafe extern "C" fn(Handle, *mut Handle) -> i32;
type DrsGetProfileInfo = unsafe extern "C" fn(Handle, Handle, *mut DrsProfile) -> i32;
type DrsFindProfile = unsafe extern "C" fn(Handle, *const u16, *mut Handle) -> i32;
type DrsCreateApplication = unsafe extern "C" fn(Handle, Handle, *mut DrsApplication) -> i32;
type DrsFindApplication = unsafe extern "C" fn(Handle, *const u16, *mut Handle, *mut DrsApplication) -> i32;
type DrsSetSetting = unsafe extern "C" fn(Handle, Handle, *mut DrsSetting) -> i32;
type DrsGetSetting = unsafe extern "C" fn(Handle, Handle, u32, *mut DrsSetting) -> i32;

/// A loaded DRS session (`NvAPI_DRS_LoadSettings` done), destroyed when dropped.
struct Drs {
    session: Handle,
    destroy: DrsSession,
    save: DrsSession,
    create_profile: DrsCreateProfile,
    delete_profile: DrsDeleteProfile,
    global_profile: DrsGlobalProfile,
    profile_info: DrsGetProfileInfo,
    find_profile: DrsFindProfile,
    create_app: DrsCreateApplication,
    find_app: DrsFindApplication,
    set_setting: DrsSetSetting,
    get_setting: DrsGetSetting,
}

impl Drop for Drs {
    fn drop(&mut self) {
        // SAFETY: the session this made.
        unsafe { (self.destroy)(self.session) };
    }
}

fn status(what: &str, code: i32) -> String {
    if code == NVAPI_INVALID_USER_PRIVILEGE {
        format!("{what}: NVAPI wants administrator rights ({code}); run it once from an elevated prompt")
    } else {
        format!("{what} failed (NvAPI_Status {code})")
    }
}

/// What a profile holds, for `owned`.
#[derive(Clone, Copy, Debug)]
struct Holds {
    predefined: bool,
    apps: u32,
    settings: u32,
    /// The executable is one of its applications.
    exe: bool,
    /// The shader-heap setting is one of its own settings.
    heap: bool,
}

/// Whether a profile could only have been left by `install`: one a user
/// made (not the driver's), with no application but the executable and no
/// setting but the shader heap's. Anything else is somebody else's and is
/// not touched.
fn owned(h: Holds) -> bool {
    !h.predefined && h.apps <= u32::from(h.exe) && h.settings <= u32::from(h.heap)
}

/// The name of the profile `install` makes for `exe`.
fn profile_name(exe: &str) -> String {
    format!("VIctory shader heap ({exe})")
}

impl Drs {
    fn open() -> Result<Drs, String> {
        let f = |id, name| find(id, name).map_err(str::to_owned);
        // SAFETY: the functions are cast to the signatures nvapi.h gives them.
        unsafe {
            let create: DrsCreateSession = std::mem::transmute(f(ID_DRS_CREATE_SESSION, "no NvAPI_DRS_CreateSession")?);
            let load: DrsSession = std::mem::transmute(f(ID_DRS_LOAD_SETTINGS, "no NvAPI_DRS_LoadSettings")?);
            let destroy: DrsSession = std::mem::transmute(f(ID_DRS_DESTROY_SESSION, "no NvAPI_DRS_DestroySession")?);
            let save = std::mem::transmute(f(ID_DRS_SAVE_SETTINGS, "no NvAPI_DRS_SaveSettings")?);
            let create_profile = std::mem::transmute(f(ID_DRS_CREATE_PROFILE, "no NvAPI_DRS_CreateProfile")?);
            let delete_profile = std::mem::transmute(f(ID_DRS_DELETE_PROFILE, "no NvAPI_DRS_DeleteProfile")?);
            let global_profile =
                std::mem::transmute(f(ID_DRS_GET_CURRENT_GLOBAL_PROFILE, "no NvAPI_DRS_GetCurrentGlobalProfile")?);
            let profile_info = std::mem::transmute(f(ID_DRS_GET_PROFILE_INFO, "no NvAPI_DRS_GetProfileInfo")?);
            let find_profile = std::mem::transmute(f(ID_DRS_FIND_PROFILE_BY_NAME, "no NvAPI_DRS_FindProfileByName")?);
            let create_app = std::mem::transmute(f(ID_DRS_CREATE_APPLICATION, "no NvAPI_DRS_CreateApplication")?);
            let find_app =
                std::mem::transmute(f(ID_DRS_FIND_APPLICATION_BY_NAME, "no NvAPI_DRS_FindApplicationByName")?);
            let set_setting = std::mem::transmute(f(ID_DRS_SET_SETTING, "no NvAPI_DRS_SetSetting")?);
            let get_setting = std::mem::transmute(f(ID_DRS_GET_SETTING, "no NvAPI_DRS_GetSetting")?);
            let mut session = std::ptr::null_mut();
            let code = create(&mut session);
            if code != 0 {
                return Err(status("NvAPI_DRS_CreateSession", code));
            }
            let drs = Drs {
                session,
                destroy,
                save,
                create_profile,
                delete_profile,
                global_profile,
                profile_info,
                find_profile,
                create_app,
                find_app,
                set_setting,
                get_setting,
            };
            let code = load(drs.session);
            if code != 0 {
                return Err(status("NvAPI_DRS_LoadSettings", code));
            }
            Ok(drs)
        }
    }

    /// The profile the driver applies to `exe`, if one names it.
    fn app_profile(&self, exe: &str) -> Result<Option<Handle>, String> {
        let name = wide(exe);
        let mut app = zeroed::<DrsApplication>(4);
        let mut profile = std::ptr::null_mut();
        // SAFETY: a session of this, a NUL-terminated name and a structure of
        // the version it is told.
        let code = unsafe { (self.find_app)(self.session, name.as_ptr(), &mut profile, &mut *app) };
        match code {
            0 => Ok(Some(profile)),
            NVAPI_EXECUTABLE_NOT_FOUND | NVAPI_PROFILE_NOT_FOUND => Ok(None),
            _ => Err(status("NvAPI_DRS_FindApplicationByName", code)),
        }
    }

    fn named(&self, name: &str) -> Result<Option<Handle>, String> {
        let name = wide(name);
        let mut profile = std::ptr::null_mut();
        // SAFETY: a session of this and a NUL-terminated name.
        let code = unsafe { (self.find_profile)(self.session, name.as_ptr(), &mut profile) };
        match code {
            0 => Ok(Some(profile)),
            NVAPI_PROFILE_NOT_FOUND => Ok(None),
            _ => Err(status("NvAPI_DRS_FindProfileByName", code)),
        }
    }

    fn info(&self, profile: Handle) -> Result<Box<DrsProfile>, String> {
        let mut info = zeroed::<DrsProfile>(1);
        // SAFETY: a session and profile of this, a structure of the version
        // it is told.
        let code = unsafe { (self.profile_info)(self.session, profile, &mut *info) };
        if code != 0 {
            return Err(status("NvAPI_DRS_GetProfileInfo", code));
        }
        Ok(info)
    }

    /// The shader-heap setting as `profile` resolves it: its value and
    /// whether the profile itself holds it (else it comes from the global
    /// or base profile); None if nothing sets it.
    fn heap(&self, profile: Handle) -> Result<Option<(u32, bool)>, String> {
        let mut setting = zeroed::<DrsSetting>(1);
        // SAFETY: a session and profile of this, a structure of the version
        // it is told.
        let code = unsafe { (self.get_setting)(self.session, profile, SHADER_HEAP_SETTING, &mut *setting) };
        match code {
            0 => {
                let [a, b, c, d, ..] = setting.current;
                Ok(Some((u32::from_le_bytes([a, b, c, d]), setting.settingLocation == DRS_IN_PROFILE)))
            }
            NVAPI_SETTING_NOT_FOUND => Ok(None),
            _ => Err(status("NvAPI_DRS_GetSetting", code)),
        }
    }

    /// What `profile` holds, with `exe` as the executable.
    fn holds(&self, profile: Handle, exe: &str) -> Result<Holds, String> {
        let info = self.info(profile)?;
        let exe_here = match self.app_profile(exe)? {
            Some(p) => narrow(&self.info(p)?.profileName) == narrow(&info.profileName),
            None => false,
        };
        Ok(Holds {
            predefined: info.isPredefined != 0,
            apps: info.numOfApps,
            settings: info.numOfSettings,
            exe: exe_here,
            heap: self.heap(profile)?.is_some_and(|(_, own)| own),
        })
    }

    fn save(&self) -> Result<(), String> {
        // SAFETY: a session of this.
        let code = unsafe { (self.save)(self.session) };
        if code != 0 {
            return Err(status("NvAPI_DRS_SaveSettings", code));
        }
        Ok(())
    }
}

/// The shader-code heap the driver gives an executable.
#[derive(Debug)]
pub struct ShaderHeap {
    /// The profile the driver applies to it, if any names it.
    pub profile: Option<String>,
    /// The setting's value in bytes and where it comes from ("profile",
    /// "global"); None: the driver's default.
    pub bytes: Option<(u32, &'static str)>,
}

impl ShaderHeap {
    /// The heap in MB.
    pub fn mb(&self) -> u32 {
        self.bytes.map_or(SHADER_HEAP_DEFAULT_MB, |(b, _)| b >> 20)
    }
}

/// What the driver will give `exe` (a file name such as `victory.exe`);
/// needs no elevation.
pub fn shader_heap(exe: &str) -> Result<ShaderHeap, String> {
    let drs = Drs::open()?;
    if let Some(profile) = drs.app_profile(exe)? {
        let name = narrow(&drs.info(profile)?.profileName);
        let bytes = drs.heap(profile)?.map(|(v, own)| (v, if own { "profile" } else { "global" }));
        return Ok(ShaderHeap { profile: Some(name), bytes });
    }
    let mut global = std::ptr::null_mut();
    // SAFETY: a session of this.
    let code = unsafe { (drs.global_profile)(drs.session, &mut global) };
    let bytes = if code == 0 {
        drs.heap(global)?.map(|(v, _)| (v, "global"))
    } else {
        None
    };
    Ok(ShaderHeap { profile: None, bytes })
}

/// Puts `exe` into a profile of its own with a shader-code heap of `mb` MB
/// and saves it (needs elevation). Refused if another profile claims `exe`
/// or a profile of our name holds anything else.
pub fn install_shader_heap(exe: &str, mb: u32) -> Result<String, String> {
    if !SHADER_HEAP_MB.contains(&mb) {
        return Err(format!(
            "{mb} MB: the heap takes {} to {} MB",
            SHADER_HEAP_MB.start(),
            SHADER_HEAP_MB.end()
        ));
    }
    let drs = Drs::open()?;
    let ours = profile_name(exe);
    if let Some(p) = drs.app_profile(exe)? {
        let name = narrow(&drs.info(p)?.profileName);
        if name != ours {
            return Err(format!(
                "{exe} already belongs to the driver profile \"{name}\", which this did not make: left alone \
                 (set 0x001ABC0F there with NVIDIA Profile Inspector, or give a copy of the executable another name: the driver matches profiles by name)"
            ));
        }
    }
    let profile = match drs.named(&ours)? {
        Some(p) => {
            let holds = drs.holds(p, exe)?;
            if !owned(holds) {
                return Err(format!(
                    "a profile named \"{ours}\" holds more than this puts there ({holds:?}): left alone"
                ));
            }
            p
        }
        None => {
            let mut info = zeroed::<DrsProfile>(1);
            info.profileName = *wide(&ours);
            let mut p = std::ptr::null_mut();
            // SAFETY: a session of this, a filled structure of the version it
            // is told.
            let code = unsafe { (drs.create_profile)(drs.session, &mut *info, &mut p) };
            if code != 0 {
                return Err(status("NvAPI_DRS_CreateProfile", code));
            }
            p
        }
    };
    if drs.app_profile(exe)?.is_none() {
        let mut app = zeroed::<DrsApplication>(4);
        app.appName = *wide(exe);
        // SAFETY: a session and profile of this, a filled structure of the
        // version it is told.
        let code = unsafe { (drs.create_app)(drs.session, profile, &mut *app) };
        if code == NVAPI_EXECUTABLE_ALREADY_IN_USE {
            return Err(format!("{exe} already belongs to another driver profile: left alone"));
        }
        if code != 0 {
            return Err(status("NvAPI_DRS_CreateApplication", code));
        }
    }
    let mut setting = zeroed::<DrsSetting>(1);
    setting.settingId = SHADER_HEAP_SETTING;
    setting.settingType = DRS_DWORD;
    setting.current[..4].copy_from_slice(&(mb << 20).to_le_bytes());
    // SAFETY: a session and profile of this, a filled structure of the
    // version it is told.
    let code = unsafe { (drs.set_setting)(drs.session, profile, &mut *setting) };
    if code != 0 {
        return Err(status("NvAPI_DRS_SetSetting", code));
    }
    drs.save()?;
    Ok(format!(
        "{exe}: the driver profile \"{ours}\" gives it a {mb} MB shader-code heap from its next start"
    ))
}

/// Deletes the profile `install` made for `exe` (needs elevation); false if
/// there was none. Refused if a profile of that name holds anything else.
pub fn remove_shader_heap(exe: &str) -> Result<bool, String> {
    let drs = Drs::open()?;
    let ours = profile_name(exe);
    let Some(p) = drs.named(&ours)? else { return Ok(false) };
    let holds = drs.holds(p, exe)?;
    if !owned(holds) {
        return Err(format!(
            "the profile \"{ours}\" holds more than this put there ({holds:?}): left alone"
        ));
    }
    // SAFETY: a session and profile of this.
    let code = unsafe { (drs.delete_profile)(drs.session, p) };
    if code != 0 {
        return Err(status("NvAPI_DRS_DeleteProfile", code));
    }
    drs.save()?;
    Ok(true)
}

// The code of live title pipelines: `VICTORY_PSO_CODE_SIZE=1` (read here once;
// default off: each new host pipeline's driver blob is fetched once with
// `GetCachedBlob`, a copy of up to a few MB). Logs the total of the live host
// pipelines' blobs at every 64 MB step up, and warns once it nears the
// shader-code heap the driver gives this executable (read from its profile;
// 512 MB without one). Blobs run about a ninth over the heap's use (xcl-dev/xcl
// issue #2403), so this errs on the early side; it counts pipelines from their
// creation, while the driver counts them from their first use.

fn code_size_wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PSO_CODE_SIZE").is_ok_and(|v| v.trim() == "1"))
}

static LIVE_CODE: AtomicU64 = AtomicU64::new(0);
static LIVE_PIPELINES: AtomicU64 = AtomicU64::new(0);

/// A host pipeline's share of the live code total, given back when dropped.
pub(crate) struct CodeBytes(u64);

impl Drop for CodeBytes {
    fn drop(&mut self) {
        if self.0 != 0 {
            LIVE_CODE.fetch_sub(self.0, Ordering::Relaxed);
            LIVE_PIPELINES.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// The heap this executable gets, in MB, read once.
fn heap_mb() -> u32 {
    static MB: OnceLock<u32> = OnceLock::new();
    *MB.get_or_init(|| {
        let exe = std::env::current_exe()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "victory.exe".to_owned());
        match shader_heap(&exe) {
            Ok(heap) => {
                tracing::info!(profile = ?heap.profile, "the driver's shader-code heap for {exe}: {} MB", heap.mb());
                heap.mb()
            }
            Err(why) => {
                tracing::info!(why, "the driver's shader-code heap not read; {SHADER_HEAP_DEFAULT_MB} MB assumed");
                SHADER_HEAP_DEFAULT_MB
            }
        }
    })
}

/// Counts `pso`'s driver code in the live total (with `VICTORY_PSO_CODE_SIZE=1`).
pub(crate) fn code_bytes(pso: &windows::Win32::Graphics::Direct3D12::ID3D12PipelineState) -> CodeBytes {
    if !code_size_wanted() {
        return CodeBytes(0);
    }
    // SAFETY: a live pipeline; the blob is released when dropped.
    let size = unsafe { pso.GetCachedBlob() }.map_or(0, |b| unsafe { b.GetBufferSize() }) as u64;
    if size == 0 {
        return CodeBytes(0);
    }
    let n = LIVE_PIPELINES.fetch_add(1, Ordering::Relaxed) + 1;
    let before = LIVE_CODE.fetch_add(size, Ordering::Relaxed);
    let live = before + size;
    let step = 64 << 20;
    if live / step > before / step {
        let heap = u64::from(heap_mb()) << 20;
        let (mb, heap_mb) = (live >> 20, heap >> 20);
        if live * 10 >= heap * 9 {
            tracing::warn!(
                "live pipelines' driver code: {mb} MB in {n} pipelines, near the driver's {heap_mb} MB shader-code \
                 heap (a lost device past it; `victory nv-profile install` raises it)"
            );
        } else {
            tracing::info!("live pipelines' driver code: {mb} MB in {n} pipelines (shader-code heap {heap_mb} MB)");
        }
    }
    CodeBytes(size)
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

    /// The DRS structures' sizes and versions: those of nvapi.h's
    /// `NVDRS_PROFILE_VER`, `NVDRS_APPLICATION_VER_V4`, `NVDRS_SETTING_VER1`.
    #[test]
    fn drs_structures_have_nvapi_sizes_and_versions() {
        assert_eq!(std::mem::size_of::<DrsProfile>(), 4116);
        assert_eq!(std::mem::offset_of!(DrsProfile, isPredefined), 4104);
        assert_eq!(version::<DrsProfile>(1), 0x1_1014);
        assert_eq!(std::mem::size_of::<DrsApplication>(), 20492);
        assert_eq!(std::mem::offset_of!(DrsApplication, flags), 16392);
        assert_eq!(version::<DrsApplication>(4), 0x4_500C);
        assert_eq!(std::mem::size_of::<DrsSetting>(), 12320);
        assert_eq!(std::mem::offset_of!(DrsSetting, settingId), 4100);
        assert_eq!(std::mem::offset_of!(DrsSetting, settingLocation), 4108);
        assert_eq!(std::mem::offset_of!(DrsSetting, current), 8220);
        assert_eq!(version::<DrsSetting>(1), 0x1_3020);
        assert_eq!(zeroed::<DrsSetting>(1).version, 0x1_3020);
        assert_eq!(narrow(&wide("victory.exe")), "victory.exe");
        assert_eq!(*SHADER_HEAP_MB.end() << 20, 0xFFF0_0000);
    }

    /// Only a profile `install` could have left is changed or deleted.
    #[test]
    fn only_a_profile_install_could_leave_is_touched() {
        let ours = Holds {
            predefined: false,
            apps: 1,
            settings: 1,
            exe: true,
            heap: true,
        };
        assert!(owned(ours));
        assert!(owned(Holds {
            apps: 0,
            settings: 0,
            exe: false,
            heap: false,
            ..ours
        }));
        assert!(!owned(Holds { predefined: true, ..ours }));
        assert!(!owned(Holds { apps: 2, ..ours }));
        assert!(!owned(Holds { exe: false, ..ours }));
        assert!(!owned(Holds { settings: 2, ..ours }));
        assert!(!owned(Holds { heap: false, ..ours }));
        assert_eq!(profile_name("victory.exe"), "VIctory shader heap (victory.exe)");
    }

    /// Reads (never writes) what the driver gives an executable: on an
    /// NVIDIA machine it answers, elsewhere it says why not.
    #[test]
    fn the_shader_heap_reads_without_elevation() {
        match shader_heap("victory.exe") {
            Ok(heap) => assert!(heap.mb() >= 1, "{heap:?}"),
            Err(why) => assert!(why.contains("nvapi") || why.contains("NvAPI"), "{why}"),
        }
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
