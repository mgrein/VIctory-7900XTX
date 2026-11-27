//! Host work a host library records into a title's command list in place of
//! a console one (the host's FidelityFX upscaler, `victory-hle`'s
//! `fidelityfx`): the host device, a title texture's host copy in the state
//! the library is told, and the host list itself, at the title list's
//! current position.
//!
//! The library sees D3D12 objects as raw COM pointers, so its crate needs no
//! D3D12 bindings of its own; [`HostTexture`] keeps each one alive for the
//! call, and the host list keeps it until it has run.
//!
//! (After Belobug cd8dfc06 and VIctory-DXR 443fd629, both from DexNom
//! ab34a8b2; Marvisak 0b35b7fe first served FidelityFX on the host.)

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::Interface;

use crate::object::This;

/// The host D3D12 device (`ID3D12Device*`), created on first use; `None`
/// without a host GPU. The device lives for the process.
pub fn host_device() -> Option<*mut c_void> {
    crate::host::host().map(|h| h.device.as_raw())
}

/// A title texture's host copy, held for as long as this lives.
pub struct HostTexture(ID3D12Resource);

impl HostTexture {
    /// The `ID3D12Resource*`.
    pub fn raw(&self) -> *mut c_void {
        self.0.as_raw()
    }

    /// Width and height of the host copy.
    pub fn size(&self) -> (u64, u32) {
        // SAFETY: a live resource.
        let desc = unsafe { self.0.GetDesc() };
        (desc.Width, desc.Height)
    }

    /// Its `DXGI_FORMAT`.
    pub fn format(&self) -> i32 {
        // SAFETY: a live resource.
        unsafe { self.0.GetDesc() }.Format.0
    }
}

/// The host list of one synchronous recording call ([`record`]).
pub struct Recording<'a>(&'a crate::host::HostList);

impl Recording<'_> {
    /// Moves a title texture's host copy to `state` (a host resource state)
    /// in the list; returns it and the state it is in then (a texture that
    /// cannot be in `state` is left in the nearest legal one). The list holds
    /// the copy through submission.
    pub fn texture(&self, resource: This, state: u32) -> Option<(HostTexture, u32)> {
        let texture = crate::resource::host_resource(resource)?.clone();
        crate::list::ensure_state(self.0, resource, state);
        let state = crate::host::legal_state(&texture, state);
        Some((HostTexture(texture), state))
    }

    /// Makes a texture the library reads ready as a compute shader's read
    /// would: an alias copied from its source, contents uploaded from title
    /// memory, a depth twin brought up to date.
    pub fn prepare_input(&self, resource: This) -> Option<()> {
        let texture = crate::resource::host_resource(resource)?;
        let aliased = crate::resource::alias_of(resource)
            .is_some_and(|source| crate::texture::alias(self.0, resource, source));
        if !aliased && crate::texture::title_written(texture) {
            crate::texture::upload_from(self.0, resource, 0);
        }
        self.0.twin_use(resource, false, false);
        Some(())
    }

    /// The host list (`ID3D12GraphicsCommandList*`), barriers asked for so
    /// far recorded on it. What the library records counts as work that
    /// writes through unordered access views.
    pub fn commands(&self) -> *mut c_void {
        self.0.list.note_uav_work(true);
        self.0.list.as_raw()
    }
}

/// The library recorded writes of `resource`'s host copy: its contents are
/// the host's from now on, not title memory's.
pub fn output_written(resource: This) {
    crate::texture::host_written(resource);
}

/// What the title submitted so far has run on the host GPU: before a library
/// frees GPU resources of its own that submitted lists may still use.
pub fn finish_submitted() {
    crate::queue::before_sync();
    crate::host::settle_for_fence();
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording) at its current position; `None` without one. What `f` binds
/// (descriptor heaps, root signatures, pipeline) is bound again before the
/// list's next draw or dispatch.
pub fn record<R>(list: This, f: impl FnOnce(&Recording<'_>) -> R) -> Option<R> {
    let list = crate::list::thin::for_external(list)?;
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        out = Some(f(&Recording(host)));
        state.external_compute();
    });
    out
}
