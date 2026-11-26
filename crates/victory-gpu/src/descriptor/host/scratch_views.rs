//! Views of lists' scratch chunks kept with the chunk
//! (`VICTORY_CHUNK_VIEWS`).
//!
//! A constant buffer a table names is copied into the list's scratch memory
//! for each draw (`crate::draw::State::tables`), and its slot made a
//! constant buffer view of the copy and, for programs that read it as raw
//! bytes, a raw view of the same bytes: two views made a draw, at a new
//! place in the scratch memory each time. In a heavy area (2026-11-26) they
//! were most of the 20,000 views the translating threads made a frame, and
//! making views was 14% of those threads' time; the views each thread keeps
//! ([`super::copies`]) for two turns, by the address they view, seldom met
//! them again: a chunk goes back to a pool when its list is reset and is
//! taken by whichever list next needs one, on whichever thread.
//!
//! But the chunks are few and long-lived, and a list fills its chunk from
//! the start, in the same sizes as the last time it was recorded: the same
//! views of the same chunk come back, a frame or many later, on any thread.
//! So each chunk keeps the views made of its memory, by their offset and
//! size, in heaps of its own the card does not see, for as long as the
//! chunk lives; a view asked for again is copied from there. A view is a
//! pure function of its buffer and description: the copy is the view the
//! host would make. The chunk's heaps go with the chunk (the views in them
//! are only ever copied, and their copies in a table's slots hold the
//! chunk's buffer as before).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::host;

/// Whether lists' scratch chunks keep the views made of them
/// (`VICTORY_CHUNK_VIEWS=0`: made for each draw, as before).
pub(crate) fn on() -> bool {
    crate::toggled!("VICTORY_CHUNK_VIEWS")
}

/// Views in a heap of a chunk's.
const PAGE: u32 = 1024;
/// Views a chunk keeps at most: past them, it lets all go and starts over.
/// (256 KB of the PC's memory a chunk of 4 MB at most, 32 bytes a view on
/// NVIDIA: a pool of 128 MB of chunks keeps 8 MB of views.)
const MOST: u32 = 8 * PAGE;

/// Since the last frame line: views copied from one a chunk kept, and views
/// made and kept.
pub(crate) static COUNTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A view of a chunk's memory: a constant buffer view (or a raw one) of
/// `size` bytes from byte `offset`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    offset: u32,
    size: u32,
    raw: bool,
}

#[derive(Default)]
struct Kept {
    heaps: Vec<ID3D12DescriptorHeap>,
    starts: Vec<usize>,
    step: usize,
    next: u32,
    made: std::collections::HashMap<Key, u32, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
}

/// The views made of one scratch chunk.
#[derive(Default)]
pub(crate) struct ChunkViews(Mutex<Kept>);

// SAFETY: descriptor heaps are free-threaded, and only touched under the
// lock.
unsafe impl Send for ChunkViews {}
unsafe impl Sync for ChunkViews {}

impl Kept {
    fn at(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.starts[(slot / PAGE) as usize] + (slot % PAGE) as usize * self.step }
    }

    /// A slot for a new view: the next one, in a new heap if need be; all
    /// let go once [`MOST`] are kept.
    fn take(&mut self) -> Option<u32> {
        if self.next == MOST {
            self.made.clear();
            self.next = 0;
        }
        if self.next / PAGE == self.heaps.len() as u32 {
            let host = host()?;
            // SAFETY: plain object creation and queries.
            unsafe {
                let heap: ID3D12DescriptorHeap = host
                    .device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: PAGE,
                        ..Default::default()
                    })
                    .ok()?;
                self.step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
                self.starts.push(heap.GetCPUDescriptorHandleForHeapStart().ptr);
                self.heaps.push(heap);
            }
        }
        self.next += 1;
        Some(self.next - 1)
    }
}

impl ChunkViews {
    /// Makes at `at` (a slot the card does not see) the view (`raw`: the raw
    /// one) of `size` bytes from byte `offset` of the chunk: copied from the
    /// one the chunk kept, or made with `make` in a slot of the chunk's,
    /// kept, and copied. False: no room, nothing made; the caller makes it.
    pub(crate) fn through(&self, raw: bool, offset: u64, size: u32, at: D3D12_CPU_DESCRIPTOR_HANDLE, make: impl FnOnce(D3D12_CPU_DESCRIPTOR_HANDLE)) -> bool {
        let Ok(offset) = u32::try_from(offset) else { return false };
        let Some(host) = host() else { return false };
        let key = Key { offset, size, raw };
        let mut kept = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let from = match kept.made.get(&key) {
            Some(&slot) => {
                COUNTS[0].fetch_add(1, Relaxed);
                kept.at(slot)
            }
            None => {
                let Some(slot) = kept.take() else { return false };
                let to = kept.at(slot);
                make(to);
                kept.made.insert(key, slot);
                COUNTS[1].fetch_add(1, Relaxed);
                to
            }
        };
        // SAFETY: a slot of the caller's and one of the chunk's heaps, both
        // of views and not seen by the card.
        unsafe { host.device.CopyDescriptorsSimple(1, at, from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
        true
    }

    /// Views the chunk keeps now.
    #[cfg(test)]
    pub(crate) fn kept(&self) -> usize {
        self.0.lock().unwrap().made.len()
    }
}

/// The frame line's words on views of scratch chunks; nothing if none were
/// asked for.
pub(crate) fn line(frames: f64) {
    let [copied, made] = [0, 1].map(|i| COUNTS[i].swap(0, Relaxed));
    if copied + made == 0 {
        return;
    }
    tracing::info!(
        "views of lists' scratch chunks per frame (VICTORY_CHUNK_VIEWS): {:.0} copied from one the chunk kept, {:.0} made and kept",
        copied as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;

    /// On the card: a view a chunk kept and copied is, byte for byte, the
    /// view the host makes, the first time (made and kept) and after
    /// (copied), of each offset and size; and a table's slot made of
    /// scratch memory keeps its views with the chunk.
    #[test]
    fn a_view_a_chunk_kept_is_the_view_the_host_makes() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        // SAFETY: plain heap creation and queries.
        let (heap, start, step) = unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 3,
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
        let (buffer, _) = host.upload_buffer(1 << 20).expect("an upload buffer");
        // SAFETY: a live buffer.
        let base = unsafe { buffer.GetGPUVirtualAddress() };
        let views = ChunkViews::default();
        let cases = [(0u64, 256u32), (256, 512), (768, 256), (0, 512), (4096, 65536)];
        for round in 0..2 {
            for (n, &(offset, size)) in cases.iter().enumerate() {
                let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: base + offset, SizeInBytes: size };
                let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: offset / 4,
                            NumElements: size / 4,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                // SAFETY (each): a slot of a heap of views; the ranges lie in
                // the live buffer.
                let make_cbv = |to| unsafe { host.device.CreateConstantBufferView(Some(&cbv), to) };
                let make_raw = |to| unsafe { host.device.CreateShaderResourceView(&buffer, Some(&raw), to) };
                for (is_raw, make) in [(false, &make_cbv as &dyn Fn(D3D12_CPU_DESCRIPTOR_HANDLE)), (true, &make_raw)] {
                    make(at(0));
                    assert!(views.through(is_raw, offset, size, at(1), make));
                    assert_eq!(bytes(0), bytes(1), "round {round}, view {n}, raw {is_raw}");
                }
            }
            assert_eq!(views.kept(), 2 * cases.len(), "round {round}: made once, then copied");
        }

        // A table's slot of a list's scratch memory.
        let Some(table) = super::super::HostHeap::new(false, 4) else { return };
        let views = std::sync::Arc::new(ChunkViews::default());
        for _ in 0..3 {
            table.constant_at(1, &buffer, base + 512, 300, Some(&(views.clone(), 512)));
        }
        assert!((1..=2).contains(&views.kept()), "the view and its raw twin, if the heap has twins: kept once");
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        drop(heap);
    }
}
