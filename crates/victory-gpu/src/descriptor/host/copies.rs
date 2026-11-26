//! Views made once and copied (`VICTORY_VIEW_COPIES`).
//!
//! On NVIDIA's driver, making a view (`CreateShaderResourceView` and its
//! kin) costs microseconds: the run of 2026-11-26 (profile x22) spent a fifth
//! to a third of each list-translating thread's time making views, nearly
//! all of them the same views of the same host resources as in the frame
//! before, made again for every recording of a list (the descriptors a list
//! makes it keeps for that recording only: `ListViews`). Copying a view's
//! descriptor (`CopyDescriptorsSimple`) is a copy of 32 bytes.
//!
//! So a thread that translates lists keeps the views it made, in a heap of
//! its own the card does not see, by the resource and the whole description
//! the host was given; a view asked for again is copied from there. A view
//! is a pure function of its resource and description while the resource
//! lives: the copy is the view the host would make. Each kept view holds its
//! resource, so its address cannot be another resource's while the view is
//! kept; a view nobody asked for in [`KEEP_FRAMES`] frames is let go, and so
//! is every one of a thread that has had nothing to translate for
//! [`IDLE_MS`] ([`idle`]). A resource only a kept view still held is retired
//! as the slots' resources are (`crate::host::retire_resource`), never freed
//! on a translating thread.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::host;

/// Views a thread keeps at most: a heavy area makes some 12,000 distinct
/// views a frame on each of four translating threads, and keeps them for
/// [`KEEP_FRAMES`] turns, in a heap the card does not see (32 bytes a view
/// on NVIDIA). With 16,384 the slots ran out within a turn.
const SLOTS: u32 = 65536;

/// Turns of [`turn`] a kept view nobody asks for stays.
const KEEP_FRAMES: u64 = 2;

/// Frames begun, and half seconds: views age with frames, and with time
/// where no frame begins (a loading screen).
fn turn() -> u64 {
    crate::frame::current() + crate::host::coarse_ms() / 500
}

/// Milliseconds a translating thread waits for work with views kept before
/// it lets them go ([`idle`]).
pub(crate) const IDLE_MS: u64 = 250;

/// Whether threads that translate lists copy the views they made before
/// (`VICTORY_VIEW_COPIES=0`: every view is made anew, as before).
pub(crate) fn on() -> bool {
    crate::toggled!("VICTORY_VIEW_COPIES")
}

/// Since the last frame line: views copied from one kept, views made and
/// kept, views let go, views made in place (no slot was free), and of the
/// views made and kept, constant buffer views and other buffer views (the
/// rest are of textures, or null ones).
pub(crate) static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// What a view is made of: the kind of view and its dimension, the resource
/// (none: 0), and the description's fields that count for that dimension.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Key {
    resource: usize,
    words: [u32; 10],
}

impl std::hash::Hash for Key {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.resource);
        for pair in self.words.chunks_exact(2) {
            state.write_u64(u64::from(pair[0]) | u64::from(pair[1]) << 32);
        }
    }
}

/// What a view made and kept was ([`Key::kind`]).
enum Made {
    Constant,
    Buffer,
    Other,
}

impl Key {
    fn kind(&self) -> Made {
        let w = self.words[0];
        if w & 0xff00_0000 == CBV {
            Made::Constant
        } else if w & OWN == 0 && w & 0xffff == 1 {
            // (Buffer is dimension 1 of shader resource and unordered access
            // views both.)
            Made::Buffer
        } else {
            Made::Other
        }
    }
}

const SRV: u32 = 1 << 24;
const UAV: u32 = 2 << 24;
const CBV: u32 = 3 << 24;
/// A view with no description: the resource's own.
const OWN: u32 = 1 << 23;

fn raw(resource: Option<&ID3D12Resource>) -> usize {
    resource.map_or(0, |r| r.as_raw() as usize)
}

fn split(n: u64) -> [u32; 2] {
    [n as u32, (n >> 32) as u32]
}

/// The key of a shader resource view of `resource` (none: a null view)
/// described by `desc` (none: the resource's own); `None` for a dimension
/// not kept.
pub(crate) fn srv_key(resource: Option<&ID3D12Resource>, desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>) -> Option<Key> {
    let Some(d) = desc else {
        return resource.map(|r| Key { resource: raw(Some(r)), words: [SRV | OWN, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
    };
    let mut words = [0u32; 10];
    words[0] = SRV | (d.ViewDimension.0 as u32 & 0xffff);
    words[1] = d.Format.0 as u32;
    words[2] = d.Shader4ComponentMapping;
    // SAFETY (each arm): the union member its dimension says; only its own
    // fields are read.
    let mut put = |fields: &[u32]| words[3..3 + fields.len()].copy_from_slice(fields);
    unsafe {
        match d.ViewDimension {
            D3D12_SRV_DIMENSION_BUFFER => {
                let b = d.Anonymous.Buffer;
                let [lo, hi] = split(b.FirstElement);
                put(&[lo, hi, b.NumElements, b.StructureByteStride, b.Flags.0 as u32])
            }
            D3D12_SRV_DIMENSION_TEXTURE1D => {
                let t = d.Anonymous.Texture1D;
                put(&[t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp.to_bits()])
            }
            D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
                let t = d.Anonymous.Texture1DArray;
                put(&[t.MostDetailedMip, t.MipLevels, t.FirstArraySlice, t.ArraySize, t.ResourceMinLODClamp.to_bits()])
            }
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let t = d.Anonymous.Texture2D;
                put(&[t.MostDetailedMip, t.MipLevels, t.PlaneSlice, t.ResourceMinLODClamp.to_bits()])
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let t = d.Anonymous.Texture2DArray;
                put(&[t.MostDetailedMip, t.MipLevels, t.FirstArraySlice, t.ArraySize, t.PlaneSlice, t.ResourceMinLODClamp.to_bits()])
            }
            D3D12_SRV_DIMENSION_TEXTURE2DMS => put(&[]),
            D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
                let t = d.Anonymous.Texture2DMSArray;
                put(&[t.FirstArraySlice, t.ArraySize])
            }
            D3D12_SRV_DIMENSION_TEXTURE3D => {
                let t = d.Anonymous.Texture3D;
                put(&[t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp.to_bits()])
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                let t = d.Anonymous.TextureCube;
                put(&[t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp.to_bits()])
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let t = d.Anonymous.TextureCubeArray;
                put(&[t.MostDetailedMip, t.MipLevels, t.First2DArrayFace, t.NumCubes, t.ResourceMinLODClamp.to_bits()])
            }
            _ => return None,
        }
    }
    Some(Key { resource: raw(resource), words })
}

/// The key of an unordered access view (no counter resource), as
/// [`srv_key`].
pub(crate) fn uav_key(resource: Option<&ID3D12Resource>, desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>) -> Option<Key> {
    let Some(d) = desc else {
        return resource.map(|r| Key { resource: raw(Some(r)), words: [UAV | OWN, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
    };
    let mut words = [0u32; 10];
    words[0] = UAV | (d.ViewDimension.0 as u32 & 0xffff);
    words[1] = d.Format.0 as u32;
    // SAFETY (each arm): the union member its dimension says; only its own
    // fields are read.
    let mut put = |fields: &[u32]| words[2..2 + fields.len()].copy_from_slice(fields);
    unsafe {
        match d.ViewDimension {
            D3D12_UAV_DIMENSION_BUFFER => {
                let b = d.Anonymous.Buffer;
                let ([lo, hi], [clo, chi]) = (split(b.FirstElement), split(b.CounterOffsetInBytes));
                put(&[lo, hi, b.NumElements, b.StructureByteStride, clo, chi, b.Flags.0 as u32])
            }
            D3D12_UAV_DIMENSION_TEXTURE1D => put(&[d.Anonymous.Texture1D.MipSlice]),
            D3D12_UAV_DIMENSION_TEXTURE1DARRAY => {
                let t = d.Anonymous.Texture1DArray;
                put(&[t.MipSlice, t.FirstArraySlice, t.ArraySize])
            }
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                let t = d.Anonymous.Texture2D;
                put(&[t.MipSlice, t.PlaneSlice])
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                let t = d.Anonymous.Texture2DArray;
                put(&[t.MipSlice, t.FirstArraySlice, t.ArraySize, t.PlaneSlice])
            }
            D3D12_UAV_DIMENSION_TEXTURE2DMS => put(&[]),
            D3D12_UAV_DIMENSION_TEXTURE2DMSARRAY => {
                let t = d.Anonymous.Texture2DMSArray;
                put(&[t.FirstArraySlice, t.ArraySize])
            }
            D3D12_UAV_DIMENSION_TEXTURE3D => {
                let t = d.Anonymous.Texture3D;
                put(&[t.MipSlice, t.FirstWSlice, t.WSize])
            }
            _ => return None,
        }
    }
    Some(Key { resource: raw(resource), words })
}

/// The key of a constant buffer view of `owner`'s memory.
pub(crate) fn cbv_key(owner: Option<&ID3D12Resource>, desc: &D3D12_CONSTANT_BUFFER_VIEW_DESC) -> Key {
    let [lo, hi] = split(desc.BufferLocation);
    Key { resource: raw(owner), words: [CBV, lo, hi, desc.SizeInBytes, 0, 0, 0, 0, 0, 0] }
}

struct Kept {
    slot: u32,
    /// The turn ([`turn`]) it was last asked for in.
    used: u64,
    /// Held while the view is kept: its address is no other resource's.
    resource: Option<ID3D12Resource>,
}

struct Copies {
    _heap: ID3D12DescriptorHeap,
    start: usize,
    step: usize,
    free: Vec<u32>,
    next: u32,
    frame: u64,
    /// The turn in which every slot was found taken by a view asked for in
    /// it: until the next, a view not kept is made in place with no look
    /// for a slot to free.
    full: Option<u64>,
    kept: std::collections::HashMap<Key, Kept, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
}

impl Copies {
    fn new() -> Option<Copies> {
        let host = host()?;
        // SAFETY: plain object creation and queries.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: SLOTS,
                    ..Default::default()
                })
                .inspect_err(|e| tracing::warn!(error = %e, "no heap to keep views in: they are made each time"))
                .ok()?;
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            Some(Copies {
                _heap: heap,
                start,
                step,
                free: Vec::new(),
                next: 0,
                frame: turn(),
                full: None,
                kept: Default::default(),
            })
        }
    }

    fn at(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.start + slot as usize * self.step }
    }

    /// Lets go of the views not asked for since turn `since`.
    fn let_go_before(&mut self, since: u64) {
        let Copies { kept, free, .. } = self;
        let mut gone = 0;
        kept.retain(|_, k| {
            if k.used >= since {
                return true;
            }
            free.push(k.slot);
            if let Some(resource) = k.resource.take() {
                let_go(resource);
            }
            gone += 1;
            false
        });
        COUNTS[2].fetch_add(gone, Relaxed);
    }

    /// Once a turn: the views nobody asked for in [`KEEP_FRAMES`] turns go.
    fn age(&mut self) {
        let now = turn();
        if now != self.frame {
            self.frame = now;
            self.let_go_before(now.saturating_sub(KEEP_FRAMES));
        }
    }

    fn take_slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        if self.next < SLOTS {
            self.next += 1;
            return Some(self.next - 1);
        }
        // Full: the views not asked for this turn go. If none did, none will
        // before the next turn: the pass over every view kept is not made
        // again for each view asked for (2% of the translating threads' time
        // in a heavy area, 2026-11-26, at some 20 us a pass).
        if self.full == Some(self.frame) {
            return None;
        }
        self.let_go_before(self.frame);
        let slot = self.free.pop();
        if slot.is_none() {
            self.full = Some(self.frame);
        }
        slot
    }
}

/// Lets go of a kept view's resource: retired if nothing else holds it (it
/// is freed where retired objects are), else just let go.
fn let_go(resource: ID3D12Resource) {
    // SAFETY: plain reference counting of a live resource; the reference
    // taken is given back at once.
    let count = unsafe {
        let unknown = &Interface::vtable(&resource).base__.base__.base__.base__;
        let count = (unknown.AddRef)(resource.as_raw());
        (unknown.Release)(resource.as_raw());
        count
    };
    if count <= 2 {
        crate::host::retire_resource(resource);
    }
}

thread_local! {
    /// Whether this thread keeps views ([`here`]).
    static HERE: Cell<bool> = const { Cell::new(false) };
    static COPIES: RefCell<Option<Copies>> = const { RefCell::new(None) };
}

/// This thread (one that translates lists) keeps the views it makes.
pub(crate) fn here() {
    HERE.set(true);
}

/// Whether this thread keeps views now.
pub(crate) fn kept_here() -> bool {
    HERE.get() && on()
}

/// Makes the view `key` names at `at` (a slot the card does not see): copied
/// from this thread's kept one, or made with `make` in a slot of the
/// thread's, kept, and copied. False: this thread keeps no views (or has no
/// room), and nothing was made: the caller makes it in place.
pub(crate) fn through(
    key: Option<Key>,
    resource: Option<&ID3D12Resource>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    make: impl FnOnce(D3D12_CPU_DESCRIPTOR_HANDLE),
) -> bool {
    let Some(key) = key else { return false };
    if !kept_here() {
        return false;
    }
    let Some(host) = host() else { return false };
    COPIES.with(|copies| {
        let Ok(mut copies) = copies.try_borrow_mut() else { return false };
        if copies.is_none() {
            *copies = Copies::new();
            if copies.is_none() {
                HERE.set(false);
                return false;
            }
        }
        let Some(c) = copies.as_mut() else { return false };
        c.age();
        let frame = c.frame;
        let from = match c.kept.get_mut(&key) {
            Some(k) => {
                k.used = frame;
                COUNTS[0].fetch_add(1, Relaxed);
                let slot = k.slot;
                c.at(slot)
            }
            None => {
                let Some(slot) = c.take_slot() else {
                    COUNTS[3].fetch_add(1, Relaxed);
                    return false;
                };
                let to = c.at(slot);
                make(to);
                c.kept.insert(key, Kept { slot, used: frame, resource: resource.cloned() });
                COUNTS[1].fetch_add(1, Relaxed);
                match key.kind() {
                    Made::Constant => COUNTS[4].fetch_add(1, Relaxed),
                    Made::Buffer => COUNTS[5].fetch_add(1, Relaxed),
                    Made::Other => 0,
                };
                to
            }
        };
        // SAFETY: a slot of the caller's and one of this thread's heap, both
        // of views and not seen by the card.
        unsafe { host.device.CopyDescriptorsSimple(1, at, from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
        true
    })
}

/// The thread has had nothing to translate for [`IDLE_MS`]: it lets go of
/// every view it kept (and of the resources they held).
pub(crate) fn idle() {
    COPIES.with(|copies| {
        if let Ok(mut copies) = copies.try_borrow_mut()
            && let Some(c) = copies.as_mut()
        {
            c.let_go_before(u64::MAX);
        }
    });
}

/// Views this thread keeps now.
pub(crate) fn kept() -> usize {
    COPIES.with(|copies| copies.try_borrow().ok().and_then(|c| c.as_ref().map(|c| c.kept.len())).unwrap_or(0))
}

/// The frame line's words on kept views; nothing if no thread kept any.
pub(crate) fn line(frames: f64) {
    let [copied, made, gone, in_place, constant, buffer] = [0, 1, 2, 3, 4, 5].map(|i| COUNTS[i].swap(0, Relaxed));
    if copied + made + in_place == 0 {
        return;
    }
    tracing::info!(
        "views per frame on the threads that translate lists (VICTORY_VIEW_COPIES): {:.0} copied from one kept, {:.0} made \
         and kept ({:.0} constant buffer views, {:.0} other buffer views), {:.0} let go, {:.1} made in place (no room)",
        copied as f64 / frames,
        made as f64 / frames,
        constant as f64 / frames,
        buffer as f64 / frames,
        gone as f64 / frames,
        in_place as f64 / frames
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_UNKNOWN};

    fn buffer_srv(first: u64, count: u32, stride: u32, raw: bool) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: if raw {
                DXGI_FORMAT_R32_TYPELESS
            } else if stride > 0 {
                DXGI_FORMAT_UNKNOWN
            } else {
                DXGI_FORMAT_R32G32B32A32_FLOAT
            },
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: first,
                    NumElements: count,
                    StructureByteStride: stride,
                    Flags: if raw { D3D12_BUFFER_SRV_FLAG_RAW } else { D3D12_BUFFER_SRV_FLAG_NONE },
                },
            },
        }
    }

    /// Keys tell views apart by what counts for their dimension, and not by
    /// bytes the dimension does not use.
    #[test]
    fn keys_tell_views_apart_by_the_fields_that_count() {
        let a = buffer_srv(8, 100, 0, true);
        assert_eq!(srv_key(None, Some(&a)), srv_key(None, Some(&a)));
        for other in [buffer_srv(9, 100, 0, true), buffer_srv(8, 101, 0, true), buffer_srv(8, 100, 16, false), buffer_srv(8, 100, 0, false)] {
            assert_ne!(srv_key(None, Some(&a)), srv_key(None, Some(&other)));
        }
        let mut t = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        // Bytes of the union past the 2-D member (the buffer member's flags):
        // not the view's.
        t.Anonymous.Buffer = D3D12_BUFFER_SRV { FirstElement: 0, NumElements: 0, StructureByteStride: 0, Flags: D3D12_BUFFER_SRV_FLAG_RAW };
        t.Anonymous.Texture2D = D3D12_TEX2D_SRV { MostDetailedMip: 1, MipLevels: 3, PlaneSlice: 0, ResourceMinLODClamp: 0.0 };
        let mut u = t;
        u.Anonymous.Buffer.Flags = D3D12_BUFFER_SRV_FLAGS(0xbeef);
        assert_eq!(srv_key(None, Some(&t)), srv_key(None, Some(&u)));
        u.Anonymous.Texture2D.MipLevels = 2;
        assert_ne!(srv_key(None, Some(&t)), srv_key(None, Some(&u)));
        // Kinds of view do not meet.
        let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: 0, SizeInBytes: 0 };
        assert_ne!(Some(cbv_key(None, &cbv)), srv_key(None, Some(&D3D12_SHADER_RESOURCE_VIEW_DESC::default())));
        assert_eq!(srv_key(None, None), None, "no resource and no description is no view");
    }

    fn refs(resource: &ID3D12Resource) -> u32 {
        // SAFETY: plain reference counting of a live resource.
        unsafe {
            let unknown = &Interface::vtable(resource).base__.base__.base__.base__;
            (unknown.AddRef)(resource.as_raw());
            (unknown.Release)(resource.as_raw())
        }
    }

    /// On the card: a view this thread kept and copied is, byte for byte, the
    /// view the host makes, the first time (made and kept) and after (copied);
    /// a kept view holds its resource, and lets it go when the thread idles.
    #[test]
    fn a_copied_view_is_the_view_the_host_makes() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        // SAFETY: plain heap creation and queries.
        let (heap, start, step) = unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 4,
                    ..Default::default()
                })
                .unwrap();
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            (heap, start, step)
        };
        let at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + i * step };
        let bytes = |i: usize| {
            let mut b = vec![0u8; step];
            assert!(crate::host::read_into(at(i).ptr as *const u8, &mut b));
            b
        };
        let buffer = host.uav_buffer(1 << 16).expect("a buffer");
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let raw_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { FirstElement: 4, NumElements: 64, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() },
            },
        };
        let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: gpu + 256, SizeInBytes: 512 };
        let (raw, structured, typed, null) =
            (buffer_srv(8, 100, 0, true), buffer_srv(2, 10, 48, false), buffer_srv(1, 33, 0, false), buffer_srv(0, 1, 0, true));
        let make = |what: &str, to: D3D12_CPU_DESCRIPTOR_HANDLE| match what {
            "raw" => super::super::create_srv(Some(&buffer), Some(&raw), to),
            "structured" => super::super::create_srv(Some(&buffer), Some(&structured), to),
            "typed" => super::super::create_srv(Some(&buffer), Some(&typed), to),
            "unordered" => super::super::create_uav(Some(&buffer), Some(&raw_uav), to),
            "constant" => super::super::create_cbv(Some(&buffer), &cbv, to),
            _ => super::super::create_srv(None, Some(&null), to),
        };
        for what in ["raw", "structured", "typed", "unordered", "constant", "null"] {
            // Made in place, as before.
            assert!(!kept_here());
            make(what, at(0));
            here();
            if !kept_here() {
                return; // VICTORY_VIEW_COPIES=0
            }
            let before = kept();
            make(what, at(1));
            assert_eq!(kept(), before + 1, "{what}: kept");
            make(what, at(2));
            assert_eq!(kept(), before + 1, "{what}: copied, not made again");
            assert_eq!(bytes(0), bytes(1), "{what}: made and kept");
            assert_eq!(bytes(0), bytes(2), "{what}: copied");
            HERE.set(false);
        }
        assert_eq!(refs(&buffer), 6, "each of the five views kept of it holds the buffer");
        idle();
        assert_eq!(kept(), 0);
        assert_eq!(refs(&buffer), 1, "let go when the thread idles");
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        drop(heap);
    }

    /// Views nobody asked for in two turns go; those asked for stay.
    #[test]
    fn kept_views_age_out() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let buffer = host.uav_buffer(1 << 12).expect("a buffer");
        let Some(mut c) = Copies::new() else { return };
        let key = |n| srv_key(Some(&buffer), Some(&buffer_srv(n, 4, 0, true))).unwrap();
        for n in 0..3 {
            let slot = c.take_slot().unwrap();
            c.kept.insert(key(n), Kept { slot, used: c.frame, resource: Some(buffer.clone()) });
        }
        assert_eq!(refs(&buffer), 4);
        c.frame += KEEP_FRAMES + 1;
        c.kept.get_mut(&key(0)).unwrap().used = c.frame;
        c.let_go_before(c.frame - KEEP_FRAMES);
        assert_eq!(c.kept.len(), 1, "the one asked for stays");
        assert_eq!(refs(&buffer), 2);
        assert_eq!(c.free.len(), 2, "their slots are free again");
    }

    /// With every slot taken by a view asked for this turn, a view asked for
    /// finds no slot after one pass over the views kept, not one pass each;
    /// in the next turn the views not asked for again go as before. Prints
    /// what each costs.
    #[test]
    fn a_full_turn_looks_for_room_once() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let buffer = host.uav_buffer(1 << 12).expect("a buffer");
        let Some(mut c) = Copies::new() else { return };
        let key = |n: u64| cbv_key(Some(&buffer), &D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: n * 256, SizeInBytes: 256 });
        let mut n = 0u64;
        while let Some(slot) = c.take_slot() {
            c.kept.insert(key(n), Kept { slot, used: c.frame, resource: None });
            n += 1;
        }
        assert_eq!(n, u64::from(SLOTS));
        assert_eq!(c.full, Some(c.frame));
        // What the pass costs, as each view asked for made it before.
        let started = std::time::Instant::now();
        for _ in 0..100 {
            c.let_go_before(c.frame);
        }
        let pass = started.elapsed().as_secs_f64() * 1e9 / 100.0;
        let started = std::time::Instant::now();
        for _ in 0..100_000 {
            assert!(c.take_slot().is_none());
        }
        let now = started.elapsed().as_secs_f64() * 1e9 / 100_000.0;
        eprintln!("a view asked for with every slot taken this turn: {now:.1} ns, {pass:.0} ns with a pass over the {n} views kept");
        // The next turn looks again: the views not asked for in it go.
        c.frame += 1;
        c.kept.get_mut(&key(0)).unwrap().used = c.frame;
        assert!(c.take_slot().is_some(), "the views not asked for this turn went");
        assert_eq!(c.kept.len(), 1);
    }
}
