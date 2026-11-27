//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; unset or `1`: the four water formats found by
//! the upstream trace; `0`: none), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") | Err(_) => DEFAULT.to_owned(),
            Ok(spec) => spec.to_owned(),
        };
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else { return false };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self { copy, layout, ticket })
    }

    pub(crate) fn owner(&self) -> usize { self.ticket.owner }
}

/// Megabytes of read-back buffers kept idle for the next list that copies a
/// texture back (`VICTORY_READBACK_POOL_MB`, default 64; 0: none are kept,
/// every record makes a committed buffer of its own and frees it, as
/// before: 14,200 allocations in about a minute were measured).
///
/// A buffer comes back only once every submission that was running or
/// queued when its list was reset has finished (`crate::host::retire`), so
/// the card is never still writing one that another list is given, and a
/// list is given only a buffer of exactly the bytes it asked for: what the
/// card writes and what [`land`] reads are the same with and without the
/// pool. (After cool_fox PR #27, rafaconvaitis, and fxceless PR #1359;
/// written anew for our tree.)
fn pool_bytes() -> u64 {
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("VICTORY_READBACK_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(64)
            << 20
    })
}

/// Idle things by their size in bytes, at most `limit` bytes of them.
struct Idle<T> {
    by_size: std::collections::HashMap<u64, Vec<T>>,
    bytes: u64,
}

impl<T> Idle<T> {
    fn take(&mut self, size: u64) -> Option<T> {
        let item = self.by_size.get_mut(&size)?.pop()?;
        self.bytes -= size;
        Some(item)
    }

    /// Keeps `item` if there is room under `limit`; gives it back if not.
    fn give(&mut self, size: u64, item: T, limit: u64) -> Option<T> {
        if self.bytes.saturating_add(size) > limit {
            return Some(item);
        }
        self.by_size.entry(size).or_default().push(item);
        self.bytes += size;
        None
    }
}

/// An idle read-back buffer.
struct Buffer(ID3D12Resource);
// SAFETY: a D3D12 resource (free-threaded), handed from the thread that
// drops it to the one that takes it.
unsafe impl Send for Buffer {}

static IDLE: Mutex<Option<Idle<Buffer>>> = Mutex::new(None);

/// Read-back buffers made, taken from the pool, and dropped because the
/// pool was full, since the start.
pub(crate) static POOL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A read-back buffer of `size` bytes: an idle one of that size, or new.
fn buffer(host: &crate::host::Host, size: u64) -> Option<ID3D12Resource> {
    if pool_bytes() != 0 {
        let idle = IDLE.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|idle| idle.take(size));
        if let Some(buffer) = idle {
            POOL[1].fetch_add(1, Ordering::Relaxed);
            return Some(buffer.0);
        }
    }
    POOL[0].fetch_add(1, Ordering::Relaxed);
    host.read_back_buffer(size)
}

/// A read-back buffer on its way back to the pool: retired with the list
/// that held it, it becomes idle when it is dropped, which is once no
/// submission that could still write it runs.
struct Returning(Option<ID3D12Resource>, u64);

impl Drop for Returning {
    fn drop(&mut self) {
        let Some(buffer) = self.0.take() else { return };
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        let idle = idle.get_or_insert_with(|| Idle { by_size: Default::default(), bytes: 0 });
        if let Some(over) = idle.give(self.1, Buffer(buffer), pool_bytes()) {
            POOL[2].fetch_add(1, Ordering::Relaxed);
            drop(over);
        }
    }
}

impl Drop for Back {
    fn drop(&mut self) {
        if pool_bytes() == 0 {
            return;
        }
        if let Some(buffer) = self.buffer.take() {
            crate::host::retire(Returning(Some(buffer), self.size));
        }
    }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    /// (Taken only when it is dropped: back to the pool.)
    buffer: Option<ID3D12Resource>,
    /// The buffer's bytes.
    size: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = buffer(host, total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back { buffer: Some(buffer), size: total, footprints, rows, mips, layout: target.layout, ticket: target.ticket })
}

/// Once the list has run, tile its texels back into the same title allocation.
pub(crate) fn land(back: &Back) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
        }
        return;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Keep title bytes the subresources do not cover.
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return;
    }
    let Some(buffer) = back.buffer.as_ref() else { return };
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: the subresource's part of the mapped buffer.
        let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len) };
        if let Err(e) = back.layout.tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
            // SAFETY: mapped above.
            unsafe { buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { buffer.Unmap(0, None) };
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The idle set gives back only a thing of exactly the size asked for,
    /// each one once, and keeps no more bytes than its limit.
    #[test]
    fn idle_read_back_buffers_are_reused_by_exact_size_within_the_limit() {
        let mut idle = Idle::<u32> { by_size: Default::default(), bytes: 0 };
        assert_eq!(idle.take(4096), None, "nothing idle yet");
        assert_eq!(idle.give(4096, 1, 10_000), None);
        assert_eq!(idle.give(4096, 2, 10_000), None);
        assert_eq!(idle.give(4096, 3, 10_000), Some(3), "over the limit: not kept");
        assert_eq!(idle.give(1024, 4, 10_000), None, "a smaller one still fits");
        assert_eq!(idle.bytes, 2 * 4096 + 1024);
        assert_eq!(idle.take(2048), None, "no other size will do");
        assert_eq!(idle.take(4096), Some(2));
        assert_eq!(idle.take(4096), Some(1));
        assert_eq!(idle.take(4096), None, "each one once");
        assert_eq!(idle.take(1024), Some(4));
        assert_eq!(idle.bytes, 0);
        assert_eq!(idle.give(1, 5, 0), Some(5), "a limit of nothing keeps nothing");
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }
}
