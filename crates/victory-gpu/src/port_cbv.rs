//! Renderer port #5: the title's constant buffers read by the card where the
//! title has them (`VICTORY_PORT_CBV=1`, off by default; read in [`on`]).
//!
//! What the console allows: a constant buffer anywhere on 16 bytes. What
//! D3D12 asks: a constant buffer view on 256. So the layer made every
//! constant buffer a draw's tables name afresh at each submission: a copy of
//! its bytes (out of title memory, on the CPU) into upload memory of its own
//! on 256, and a view of that ([`crate::draw`]'s tables, `defer_constants`):
//! ~23,000 a frame of the city, ~9 MB, and as many steps of the submitting
//! thread. With renderer port 1 (`VICTORY_PORT_CONSTANTS=1`) the memory they
//! are copied out of is the card's to read in place already.
//!
//! What is done here instead: shaders are translated to read constant buffers
//! as raw bytes (`victory_shader::emit::constants_raw`): a table's through
//! the raw twin of its slot, a root one as a root SRV. A raw view starts on 16
//! bytes, a root SRV on 4. So a constant buffer of a table in title memory in
//! place is a raw view of the title's bytes, made in the twin of its slot,
//! and nothing is copied or done at submission; a root one is bound at its
//! address as root SRVs are. Constant buffers anywhere else are copied as
//! before (the twin of their slot is a raw view of the copy, as it was).
//! The twin of a title slot that holds a constant buffer view is a raw view
//! of its bytes too (it was a constant buffer view: a raw read through one
//! on AMD has no end).
//!
//! On a host without raw twins (`victory_shader::emit::raw_views` 0: the
//! view heap is capped, NVIDIA) there is no twin to read: tables' constant
//! buffers were read as constant buffers and copied as before, and only root
//! ones read raw. With `VICTORY_PORT_CBV_SLOT=1` (off by default; [`in_slot`])
//! translated shaders read them as raw bytes through the slot itself, and every
//! slot that shows a constant buffer is a raw view of its bytes instead of a
//! constant buffer view: the title's in place where they start on 16 bytes,
//! the copy (on 256) otherwise. A raw view cannot start off 16 bytes
//! (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`), so those stay copied.
//!
//! What it assumes: what the card reads of a constant buffer at a draw is
//! what the title's CPU wrote before it submitted, as for every other title
//! buffer in place (renderer port 1): before, the copy at submission was
//! taken then whatever the title wrote after. The title writes its constants
//! into a ring it waits on its fences for.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether renderer port #5 is on: the shaders' side
/// (`victory_shader::emit::constants_raw`) and the host's views follow the
/// same switch.
pub(crate) fn on() -> bool {
    victory_shader::emit::constants_raw()
}

/// Whether a table's constant buffer in title memory in place is given to the
/// card where it is (on with the port; `VICTORY_PORT_CBV_PLACE=0`: copied as
/// before, and read through the raw twin all the same). With
/// `VICTORY_AB=<s>,VICTORY_PORT_CBV_PLACE` (diagnostics builds) on and off
/// while the title runs: both sides measured in one run.
pub(crate) fn in_place_on() -> bool {
    on() && crate::toggled!("VICTORY_PORT_CBV_PLACE")
}

/// Whether translated shaders read constant buffers through their slots on a
/// host without raw twins (`VICTORY_PORT_CBV_SLOT=1` with the port; off by
/// default; `victory_shader::emit::constants_in_slot`): then the host makes
/// every slot that shows a constant buffer a raw view of its bytes.
pub(crate) fn in_slot() -> bool {
    victory_shader::emit::constants_in_slot()
}

/// Why a table's constant buffer in title memory in place was copied all the
/// same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// It does not start on 16 bytes: no raw view can.
    Unaligned,
    /// The slot has no raw twin (a capped host heap) and translated shaders
    /// do not read through slots (`VICTORY_PORT_CBV_SLOT` not 1).
    NoTwin,
    /// Not in the buffer, or nothing of it is, or no host.
    Other,
}

/// Since the last line: tables' constant buffers given in place, their bytes;
/// copied as before because not in place, not on 16 bytes, no raw twin, for
/// another reason; title slots' constant buffers with a raw view (a twin, or
/// the slot itself), and those with none because not on 16 bytes.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A table's constant buffer of `size` bytes was given in place.
pub(crate) fn in_place(size: u32) {
    STATS[0].fetch_add(1, Relaxed);
    STATS[1].fetch_add(u64::from(size), Relaxed);
}

/// A table's constant buffer was copied as before: not in place (`None`), or
/// in place but refused for `Some(why)`.
pub(crate) fn copied(why: Option<Refused>) {
    let at = match why {
        None => 2,
        Some(Refused::Unaligned) => 3,
        Some(Refused::NoTwin) => 4,
        Some(Refused::Other) => 5,
    };
    STATS[at].fetch_add(1, Relaxed);
}

/// A title slot's constant buffer got a raw view (its twin, or the slot).
pub(crate) fn twinned() {
    STATS[6].fetch_add(1, Relaxed);
}

/// A title slot's constant buffer got no raw view in the slot itself
/// ([`in_slot`]): not on 16 bytes (the slot reads nothing, as a constant
/// buffer view not on 256 bytes read nothing before).
pub(crate) fn slot_refused() {
    STATS[7].fetch_add(1, Relaxed);
}

/// Whether a constant buffer at host address `gpu` can be a raw view of
/// those bytes: on 16 bytes.
pub(crate) fn raw_ok(gpu: u64) -> bool {
    gpu.is_multiple_of(16)
}

/// For the frame-rate line.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64 / frames.max(1e-9)).collect();
    tracing::info!(
        "renderer port 5 (VICTORY_PORT_CBV=1{}) per frame: {:.0} tables' constant buffers read by the card in place ({:.1} KB not copied, \
         no step at submission); copied as before: {:.0} not in place, {:.0} not on 16 bytes, {:.0} without a raw twin (capped host heap; \
         VICTORY_PORT_CBV_SLOT=1 reads through the slot), {:.0} otherwise; {:.0} title slots' constant buffers with a raw view, {:.0} not \
         (not on 16 bytes)",
        if in_slot() { ", VICTORY_PORT_CBV_SLOT=1: through slots" } else { "" },
        s[0],
        s[1] / 1024.0,
        s[2],
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_views_need_sixteen_bytes() {
        assert!(raw_ok(0x3500_0010));
        assert!(raw_ok(0x3500_0100));
        assert!(!raw_ok(0x3500_0008));
        assert!(!raw_ok(0x3500_0004));
    }
}
