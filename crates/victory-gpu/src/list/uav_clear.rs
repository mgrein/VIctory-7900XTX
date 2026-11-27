//! List methods 53 and 54 (0x1a8, 0x1b0): `ClearUnorderedAccessViewUint`
//! and `...Float` `(gpu handle, cpu handle, resource, values, count,
//! rects)`, as on PC. They were missing (the table's logging stand-in), so
//! every clear a title made of a UAV was dropped.
//!
//! Both handles name view records in title memory. D3D12 wants a descriptor
//! in a heap the card does not see (read at the call) and one in the bound
//! shader-visible heap (read when the clear runs): the view is made in this
//! thread's staging slot and in a dynamic slot of the bound view heap, which
//! the recording holds until it has run ([`crate::draw::State::clear_slot`]).
//! A clear whose handles do not name the same view of `resource`, or whose
//! view the host cannot make, is skipped (as all were before). Lists whose
//! calls are written down (`VICTORY_THIN=1`) still skip them.
//!
//! `VICTORY_UAV_CLEAR=0`: the methods stay missing, as before.

use super::{Rect, This};
use crate::descriptor::View;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the clears are recorded (`VICTORY_UAV_CLEAR`, on unless `0`).
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UAV_CLEAR").is_none_or(|v| v != "0"))
}

/// `table` (of either layout: the clears come before the slots newer
/// titles move) with the two clears in place.
pub(super) const fn with_clears(mut table: [*const (); 160]) -> [*const (); 160] {
    table[53] = clear::<false> as *const ();
    table[54] = clear::<true> as *const ();
    table
}

/// Clears recorded and skipped, for the log.
static DONE: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);

extern "system" fn clear<const FLOAT: bool>(
    this: This,
    gpu: u64,
    cpu: u64,
    resource: This,
    values: *const [u32; 4],
    count: u32,
    rects: *const Rect,
) {
    let recorded = record::<FLOAT>(this, gpu, cpu, resource, values, count, rects);
    let n = if recorded { &DONE } else { &SKIPPED }.fetch_add(1, Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::info!(
            recorded = DONE.load(Relaxed),
            skipped = SKIPPED.load(Relaxed),
            float = FLOAT,
            "UAV clears"
        );
    }
}

/// Records the clear; false if it was skipped.
fn record<const FLOAT: bool>(
    this: This,
    gpu: u64,
    cpu: u64,
    resource: This,
    values: *const [u32; 4],
    count: u32,
    rects: *const Rect,
) -> bool {
    // SAFETY: the title passes four values (null is tolerated).
    let Some(values) = (unsafe { values.as_ref() }) else {
        return false;
    };
    if resource.is_null() || (count != 0 && rects.is_null()) || crate::host::host().is_none() {
        return false;
    }
    let mut words = [0u8; 12];
    if !crate::host::read_into(cpu as *const u8, &mut words) {
        return false;
    }
    let Some(view) = crate::descriptor::view(words.as_ptr() as u64) else {
        return false;
    };
    if !views_resource(&view, resource) {
        return false;
    }
    let buffer = crate::descriptor::host::buffer_copy(&view);
    // A float clear of a buffer: the host's view of it may be raw (where
    // D3D12 has no float clear), and the conversion the console would make
    // is not known here.
    if FLOAT && buffer.is_some() {
        return false;
    }
    let mut recorded = false;
    super::with_state(this, |list, state| {
        let Some((gpu, cpu, backing)) = state.clear_slot(list, gpu, &view) else {
            return;
        };
        if let Some((shadow, address, len)) = &buffer {
            if shadow.buffer != backing {
                return;
            }
            shadow.refresh(list, *address, *len);
            shadow.gpu_writes(list, *address, *len);
        } else if !list.twin_use(resource, true, true) {
            if crate::resource::host_resource(resource) != Some(&backing) {
                return;
            }
            const UNORDERED_ACCESS: u32 = 0x8;
            super::ensure_state(list, resource, UNORDERED_ACCESS);
        }
        list.keep(&backing);
        let rects: &[windows::Win32::Foundation::RECT] = if count == 0 {
            &[]
        } else {
            // SAFETY: the title passes `count` rectangles, laid out as the
            // host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        };
        // SAFETY: an open list; both descriptors are of a view of `backing`,
        // the shader-visible one in the bound heap and held by the recording.
        unsafe {
            if FLOAT {
                list.list
                    .ClearUnorderedAccessViewFloat(gpu, cpu, &backing, &values.map(f32::from_bits), rects);
            } else {
                list.list.ClearUnorderedAccessViewUint(gpu, cpu, &backing, values, rects);
            }
        }
        recorded = true;
    });
    recorded
}

/// Whether `view` is an unordered access view of `resource`: one made on
/// it, or one of placed memory that is the whole of buffer `resource`.
fn views_resource(view: &View, resource: This) -> bool {
    match view {
        View::Unordered(r, _, _) => *r == resource as usize,
        View::Placed(true, ..) => {
            let extent = crate::descriptor::host::buffer_extent(view);
            extent.is_some()
                && extent == crate::resource::buffer_bytes(resource).map(|(p, n)| (p as u64, n))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::*;

    #[test]
    fn both_clears_take_their_slots_in_both_layouts() {
        for newer in [false, true] {
            let old = super::super::list_table(newer);
            let new = with_clears(old);
            assert_eq!(new[53], clear::<false> as *const ());
            assert_eq!(new[54], clear::<true> as *const ());
            assert_eq!(new.iter().zip(old).filter(|(a, b)| **a != *b).count(), 2);
        }
    }

    /// A UAV clear of a whole texture, then of one texel, reaches the host
    /// texture (read back from the host GPU).
    #[test]
    fn clear_values_reach_the_selected_texels() {
        use crate::object::S_OK;
        let Some(host) = crate::host::host() else {
            eprintln!("no D3D12 device");
            return;
        };
        let null = std::ptr::null();
        let mut a = null;
        assert_eq!(super::super::create_allocator(null, 0, &super::super::ALLOCATOR_IIDS[0], &mut a), S_OK);
        let mut l = null;
        assert_eq!(
            super::super::create_list(null, 0, 0, a, null, &super::super::LIST_IIDS[0], &mut l),
            S_OK
        );
        let make_heap = |visible: bool| {
            // Room after it for the newer description's placement (zero).
            let desc = [0u32, 8, u32::from(visible), 0, 0, 0, 0, 0, 0, 0];
            let mut h = null;
            assert_eq!(
                crate::descriptor::create_heap(null, desc.as_ptr().cast(), &crate::resource::PAGEABLE_IIDS[0], &mut h),
                S_OK
            );
            h
        };
        let (visible, staging) = (make_heap(true), make_heap(false));
        // R32_UINT and R32_FLOAT, 2x1, unordered access allowed.
        for (i, format, bits, float) in [(0, 42, 0xdead_beef_u32, false), (1, 41, 0.75f32.to_bits(), true)] {
            let image = crate::resource::test_texture(2, 1, format, 4);
            let slot = |heap: This| crate::descriptor::heap(heap).handle(i).unwrap();
            let (gpu, cpu) = (slot(visible), slot(staging));
            for handle in [gpu, cpu] {
                crate::descriptor::create_uav(null, image, null, std::ptr::null(), handle);
            }
            super::super::with_state(l, |_, state| state.set_heaps(&[visible]));
            let rect = Rect { left: 1, top: 0, right: 2, bottom: 1 };
            let f = if float { clear::<true> } else { clear::<false> };
            let recorded = DONE.load(Relaxed);
            f(l, gpu, cpu, image, &[0; 4], 0, null.cast());
            f(l, gpu, cpu, image, &[bits; 4], 1, &rect);
            assert_eq!(DONE.load(Relaxed), recorded + 2, "format {format}");
            let list = super::super::host_list(l).unwrap();
            let texels = read_back(host, list, image);
            assert_eq!(texels[..2], [0, bits], "format {format}");
            assert_eq!(super::super::reset(l, a, null), S_OK);
        }
    }

    /// The first row's first four texels of 32-bit texture `image`: copied
    /// into a readback buffer at the end of `list`, which is closed and run.
    fn read_back(host: &crate::host::Host, list: &crate::host::HostList, image: This) -> [u32; 4] {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        const COPY_SOURCE: u32 = 0x800;
        super::super::ensure_state(list, image, COPY_SOURCE);
        let texture = crate::resource::host_resource(image).unwrap().clone();
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        // SAFETY: a live texture.
        let format = unsafe { texture.GetDesc() }.Format;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(readback.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: 2, Height: 1, Depth: 1, RowPitch: 256 },
                },
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::ManuallyDrop::new(Some(texture.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // SAFETY: an open list, live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        drop(std::mem::ManuallyDrop::into_inner(to.pResource));
        drop(std::mem::ManuallyDrop::into_inner(from.pResource));
        list.close();
        host.execute(&[list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 256 bytes.
        unsafe {
            readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut mapped)).unwrap();
            let texels = mapped.cast::<[u32; 4]>().read();
            readback.Unmap(0, None);
            texels
        }
    }
}
