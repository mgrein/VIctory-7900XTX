//! Pieces of a few big shared buffers for the small host copies of single
//! buffer views' own bytes (`crate::shadow::exact`).
//!
//! Each such copy was a committed buffer of its own (`Host::shared_buffer`),
//! made on a recording thread and let go soon after: on a host with no raw
//! twins (an RTX 4070, whose view heap is capped) every structured view off
//! 16 bytes asks for one, ~390 were made a frame, and creating and
//! destroying their allocations was ~23% of the recording threads' time
//! (12.8 fps went to 3.3). Here a copy is a piece of a chunk of
//! [`CHUNK`] bytes made once: taking and giving back a piece is a few
//! tree operations under a lock, and the kernel is not asked.
//!
//! A piece is given back when the copy that holds it is dropped, through
//! `crate::host::retire`, so no submission begun, or list recorded and not
//! yet sent, that may still read it sees another copy's bytes there
//! ([`Piece`]'s drop is what gives it back). The chunks are buffers like
//! any shared buffer (the common state, unordered access allowed, mapped
//! write-back): nothing transitions them, so a piece needs no state of its
//! own.
//!
//! `VICTORY_EXACT_ARENA=0` gives every such copy a buffer of its own again
//! ([`mode`]).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// The bytes of a chunk.
pub(crate) const CHUNK: u64 = 32 << 20;

/// Pieces begin and end on this many bytes.
pub(crate) const LINE: u64 = 256;

/// The largest copy given a piece; a bigger one gets a buffer of its own.
pub(crate) const MOST: u64 = 1 << 20;

/// The largest alignment a piece is placed on ([`line_for`]); a view that
/// needs more gets a buffer of its own.
pub(crate) const MOST_ALIGN: u64 = 64 << 10;

/// Where copies of views' own bytes are made (`VICTORY_EXACT_ARENA`, read
/// once): 0, each in a buffer of its own, as before; 1 (the default), those
/// first asked for by shader resource views in pieces of shared chunks, and
/// those first asked for by unordered access views (shaders write them) in
/// buffers of their own; 2, every one in pieces.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_EXACT_ARENA").ok().as_deref().map(str::trim) {
        Some("0") => 0,
        Some("2") => 2,
        _ => 1,
    })
}

/// The alignment a piece for a view whose own first byte must lie on a
/// multiple of `align` is placed on: a multiple of both that and [`LINE`].
pub(crate) fn line_for(align: u64) -> u64 {
    let align = align.max(1);
    let (mut a, mut b) = (LINE, align);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    LINE / a * align
}

/// The free ranges of a chunk of `size` bytes: by offset (to join
/// neighbours when one is given back) and by length (the shortest that
/// holds what is asked is taken).
#[derive(Debug)]
pub(crate) struct Ranges {
    by_at: BTreeMap<u64, u64>,
    by_len: BTreeSet<(u64, u64)>,
    used: u64,
    size: u64,
}

impl Ranges {
    /// All of `[0, size)` free.
    pub(crate) fn new(size: u64) -> Ranges {
        let mut ranges = Ranges { by_at: BTreeMap::new(), by_len: BTreeSet::new(), used: 0, size };
        if size > 0 {
            ranges.free(0, size);
        }
        ranges
    }

    fn free(&mut self, at: u64, len: u64) {
        self.by_at.insert(at, len);
        self.by_len.insert((len, at));
    }

    fn unfree(&mut self, at: u64, len: u64) {
        self.by_at.remove(&at);
        self.by_len.remove(&(len, at));
    }

    /// Takes `len` bytes at an offset that is a multiple of `align`; `None`
    /// if no free range holds them. Of the free ranges long enough, the
    /// first few by length are looked at (an alignment past their start
    /// may leave one short).
    pub(crate) fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        const LOOKS: usize = 64;
        let (len, align) = (len.max(1), align.max(1));
        let (at, free, aligned) = self
            .by_len
            .range((len, 0)..)
            .take(LOOKS)
            .find_map(|&(free, at)| {
                let aligned = at.checked_next_multiple_of(align)?;
                (aligned - at + len <= free).then_some((at, free, aligned))
            })?;
        self.unfree(at, free);
        if aligned > at {
            self.free(at, aligned - at);
        }
        let end = aligned + len;
        if end < at + free {
            self.free(end, at + free - end);
        }
        self.used += len;
        Some(aligned)
    }

    /// Gives back `[at, at + len)`, taken before, joined with the free
    /// ranges beside it.
    pub(crate) fn give(&mut self, at: u64, len: u64) {
        let len = len.max(1);
        debug_assert!(at + len <= self.size, "a range past the chunk's end");
        let (mut from, mut to) = (at, at + len);
        if let Some((&before, &n)) = self.by_at.range(..at).next_back() {
            debug_assert!(before + n <= at, "a range given back twice");
            if before + n == at {
                self.unfree(before, n);
                from = before;
            }
        }
        if let Some(n) = self.by_at.get(&to).copied() {
            self.unfree(to, n);
            to += n;
        }
        debug_assert!(self.by_at.range(from..to).next().is_none(), "a range given back twice");
        self.free(from, to - from);
        self.used = self.used.saturating_sub(len);
    }

    /// The bytes taken and not given back.
    pub(crate) fn used(&self) -> u64 {
        self.used
    }

    /// The free ranges, by offset (tests).
    #[cfg(test)]
    fn free_ranges(&self) -> Vec<(u64, u64)> {
        self.by_at.iter().map(|(&a, &n)| (a, n)).collect()
    }
}

/// A chunk: a shared buffer, its mapping and its GPU address, and its free
/// ranges.
struct Chunk {
    id: u64,
    buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
    ranges: Ranges,
}

#[derive(Default)]
struct Arena {
    chunks: Vec<Chunk>,
    next: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is an address.
unsafe impl Send for Arena {}

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new(), next: 1 });

/// Pieces handed out since the start, and taken back.
static PIECES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A piece of a chunk, held by the copy made in it: `len` bytes from `at`.
/// Dropping it gives the bytes back (the copy hands it to
/// `crate::host::retire` when it is dropped itself).
#[derive(Debug)]
pub(crate) struct Piece {
    chunk: u64,
    pub(crate) at: u64,
    pub(crate) len: u64,
}

impl Drop for Piece {
    fn drop(&mut self) {
        PIECES[1].fetch_add(1, Ordering::Relaxed);
        let gone = {
            let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
            let Some(n) = arena.chunks.iter().position(|c| c.id == self.chunk) else {
                return;
            };
            arena.chunks[n].ranges.give(self.at, self.len);
            // An empty chunk goes while another empty one is kept for what
            // comes next.
            let empty = |c: &Chunk| c.ranges.used() == 0;
            (empty(&arena.chunks[n]) && arena.chunks.iter().filter(|c| empty(c)).count() > 1)
                .then(|| arena.chunks.swap_remove(n))
        };
        // (Outside the lock: the driver's work, if this was its last
        // reference. A list may still hold the buffer; it lives until then.)
        drop(gone);
    }
}

/// What a copy is made in: the piece, the chunk's buffer, the piece's
/// first byte in the chunk's mapping, and its GPU address.
pub(crate) struct Taken {
    pub(crate) piece: Piece,
    pub(crate) buffer: ID3D12Resource,
    pub(crate) data: *mut u8,
    pub(crate) gpu: u64,
}

/// A piece of `len` bytes (rounded up to [`LINE`]) that begins on a multiple
/// of `align` (one of [`LINE`]), from a chunk that has room, or a new one;
/// `None` if `len` is past [`MOST`] or `align` past [`MOST_ALIGN`], or no
/// chunk can be made.
pub(crate) fn take(len: u64, align: u64) -> Option<Taken> {
    let len = len.max(1).next_multiple_of(LINE);
    if len > MOST || align > MOST_ALIGN || !align.is_multiple_of(LINE) {
        return None;
    }
    let mut arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    // The fullest chunks first: the emptier ones may empty and go.
    let mut order: Vec<usize> = (0..arena.chunks.len()).collect();
    order.sort_unstable_by_key(|&n| std::cmp::Reverse(arena.chunks[n].ranges.used()));
    let found = order.into_iter().find_map(|n| arena.chunks[n].ranges.take(len, align).map(|at| (n, at)));
    let (n, at) = match found {
        Some(found) => found,
        None => {
            let (buffer, data) = crate::host::host()?.shared_buffer(CHUNK)?;
            let id = arena.next;
            arena.next += 1;
            crate::host::name(&buffer, || format!("pieces for copies of views' own bytes {id}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            arena.chunks.push(Chunk { id, buffer, data: data as usize, gpu, ranges: Ranges::new(CHUNK) });
            let n = arena.chunks.len() - 1;
            let at = arena.chunks[n].ranges.take(len, align)?;
            (n, at)
        }
    };
    let chunk = &arena.chunks[n];
    PIECES[0].fetch_add(1, Ordering::Relaxed);
    Some(Taken {
        piece: Piece { chunk: chunk.id, at, len },
        buffer: chunk.buffer.clone(),
        data: (chunk.data + at as usize) as *mut u8,
        gpu: chunk.gpu + at,
    })
}

/// For the frame line: the chunks, the bytes of pieces in use, and the
/// pieces held now.
pub(crate) fn stats() -> [u64; 3] {
    let arena = ARENA.lock().unwrap_or_else(|e| e.into_inner());
    let used = arena.chunks.iter().map(|c| c.ranges.used()).sum();
    let held = PIECES[0].load(Ordering::Relaxed).saturating_sub(PIECES[1].load(Ordering::Relaxed));
    [arena.chunks.len() as u64, used, held]
}

#[cfg(test)]
mod tests {
    use super::{LINE, Ranges, line_for};

    /// No two taken ranges share a byte, and the free ones and the taken
    /// ones cover the chunk exactly.
    fn check(ranges: &Ranges, taken: &[(u64, u64)]) {
        let mut all: Vec<(u64, u64)> = taken.iter().copied().chain(ranges.free_ranges()).collect();
        all.sort_unstable();
        let mut at = 0;
        for (a, n) in all {
            assert_eq!(a, at, "a gap or an overlap at {a:#x}: {taken:?} / {:?}", ranges.free_ranges());
            at = a + n;
        }
        assert_eq!(at, ranges.size);
        assert_eq!(ranges.used(), taken.iter().map(|t| t.1).sum::<u64>());
        // Free ranges are joined: no two touch.
        let free = ranges.free_ranges();
        assert!(free.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{free:?}");
    }

    #[test]
    fn pieces_lie_on_their_alignment_and_never_overlap() {
        let mut r = Ranges::new(1 << 20);
        let mut taken = Vec::new();
        for (i, len) in [256u64, 512, 768, 4096, 256, 1280].into_iter().cycle().take(60).enumerate() {
            let align = if i % 3 == 0 { line_for(12) } else { LINE };
            let at = r.take(len, align).expect("room");
            assert_eq!(at % align, 0, "piece {i} at {at:#x}");
            taken.push((at, len));
            check(&r, &taken);
        }
        // 768 for a view of 12-byte elements: a multiple of both.
        assert_eq!(line_for(12), 768);
        assert_eq!(line_for(16), LINE);
        assert_eq!(line_for(1), LINE);
        assert_eq!(line_for(456), 256 * 57);
    }

    #[test]
    fn pieces_given_back_are_taken_again() {
        let mut r = Ranges::new(64 * LINE);
        let a: Vec<u64> = (0..64).map(|_| r.take(LINE, LINE).expect("room")).collect();
        assert_eq!(r.take(LINE, LINE), None, "full");
        assert_eq!(r.used(), 64 * LINE);
        // Every other one back: 32 holes of one line, none of two.
        for &at in a.iter().step_by(2) {
            r.give(at, LINE);
        }
        assert_eq!(r.take(2 * LINE, LINE), None, "no two lines together");
        let again = r.take(LINE, LINE).expect("a hole");
        assert!(a.iter().step_by(2).any(|&at| at == again), "a hole is reused");
        // The rest back: one free range, the whole chunk.
        for &at in a.iter().skip(1).step_by(2) {
            r.give(at, LINE);
        }
        r.give(again, LINE);
        assert_eq!(r.free_ranges(), vec![(0, 64 * LINE)]);
        assert_eq!(r.used(), 0);
        assert_eq!(r.take(64 * LINE, LINE), Some(0));
    }

    /// Pieces taken and given back in a shuffled order keep the books right.
    #[test]
    fn many_pieces_in_any_order() {
        let mut r = Ranges::new(4 << 20);
        let mut taken: Vec<(u64, u64)> = Vec::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..4000 {
            if taken.len() > 40 && next() % 3 == 0 {
                let (at, len) = taken.swap_remove((next() % taken.len() as u64) as usize);
                r.give(at, len);
            } else {
                let len = (1 + next() % 32) * LINE;
                let align = if next() % 4 == 0 { line_for(12) } else { LINE };
                if let Some(at) = r.take(len, align) {
                    assert_eq!(at % align, 0);
                    taken.push((at, len));
                }
            }
        }
        check(&r, &taken);
        for (at, len) in taken.drain(..) {
            r.give(at, len);
        }
        assert_eq!(r.free_ranges(), vec![(0, 4 << 20)]);
    }
}
