//! Instructions the console's CPU (Zen 2) has and an Intel PC's lacks, replaced in the title's
//! code the first time they fault, so that they never fault there again: a fault is a trip into
//! Windows and back through every vectored handler, and the title runs some of them thousands of
//! times a second (wait loops, `clzero` loops through a frame).
//!
//! Only replacing is done here. What a fault does when nothing is replaced (the first time, an
//! instruction in a form not replaced, or with the switches off) is VIctory's own emulation in
//! `crash.rs` (`emulate_amd_only`), which the vectored handler calls next.
//!
//! This fork's own code. (The emulation and decoder this module had before were ported from XCL,
//! GPL-2.0-only, which cannot be combined with this GPL-3.0-or-later program; fxceless PR #1057
//! took them out, and so does this.)

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

/// What is at a faulting address, as far as replacing goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Site {
    /// `monitorx` (`0f 01 fa`).
    Monitorx,
    /// `mwaitx` (`0f 01 fb`).
    Mwaitx,
    /// `clzero; add rax, 0x40; cmp rax, reg; jb` (12 bytes from the fault): the `cmp`'s bytes.
    ClzeroLoop([u8; 3]),
    /// `mov rax, a; add a, 0x40; clzero; cmp a, b; jb` (15 bytes, the fault 7 into them): the
    /// `cmp`'s bytes.
    ClzeroLoop15([u8; 3]),
    /// A lone `clzero`.
    Clzero,
    /// `clwb` of this many bytes.
    Clwb(usize),
    /// `rdpru` (`0f 01 fd`).
    Rdpru,
    /// Something else.
    Other,
}

/// What `code` (the bytes from the fault to the end of its page, at most 32) and the 7 bytes
/// before it (`None` where they lie in another page) are.
fn site(before: Option<[u8; 7]>, code: &[u8]) -> Site {
    match code {
        [0x0f, 0x01, 0xfa, ..] => Site::Monitorx,
        [0x0f, 0x01, 0xfb, ..] => Site::Mwaitx,
        [0x0f, 0x01, 0xfd, ..] => Site::Rdpru,
        [0x0f, 0x01, 0xfc, rest @ ..] => {
            // clzero; add rax, 0x40 (48 83 c0 40); cmp rax, reg (REX.W 39, rm rax, reg not
            // rax); jb back to the clzero (72 f4).
            if let [0x48, 0x83, 0xc0, 0x40, r @ (0x48 | 0x4c), 0x39, m, 0x72, 0xf4, ..] = *rest
                && m & 0xc7 == 0xc0
                && !(r == 0x48 && m == 0xc0)
            {
                return Site::ClzeroLoop([r, 0x39, m]);
            }
            // mov rax, a (REX.W 89, rm rax); add a, 0x40 (REX.W 83 /0 a, 0x40); clzero;
            // cmp a, b (REX.W 39, rm a); jb back to the mov (72 f1). `a` and `b` not rax.
            if let Some([mr @ (0x48 | 0x4c), 0x89, mm, ar @ (0x48 | 0x49), 0x83, am, 0x40]) = before
                && let [cr @ 0x48..=0x4f, 0x39, cm, 0x72, 0xf1, ..] = *rest
            {
                let a = (mm >> 3) & 7 | (mr & 4) << 1;
                let added = am & 7 | (ar & 1) << 3;
                let compared = cm & 7 | (cr & 1) << 3;
                let b = (cm >> 3) & 7 | (cr & 4) << 1;
                if mm & 0xc7 == 0xc0 && am & 0xf8 == 0xc0 && cm >> 6 == 3 && a == added && a == compared && a != 0 && b != 0 {
                    return Site::ClzeroLoop15([cr, 0x39, cm]);
                }
            }
            Site::Clzero
        }
        _ => crate::crash::clwb_len(code).map_or(Site::Other, Site::Clwb),
    }
}

/// Whether `monitorx` and `mwaitx` are replaced in the title's code the first time they fault
/// (`monitorx` by a 3-byte no-op, `mwaitx` by `pause` and a no-op), so that they never fault
/// again. Emulated, each pass of such a wait loop is two trips into Windows and back, through
/// every vectored handler, and no lighter a wait for it: the emulation does no more than yield
/// either, and the loop looks at its value again after it. On by default;
/// `VICTORY_PATCH_WAITS=0` emulates each time, as before.
fn patches_waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_WAITS").is_none_or(|v| v != "0"))
}

/// What replaces the 3 bytes of a wait instruction.
fn wait_patch(what: Site) -> Option<[u8; 3]> {
    match what {
        Site::Monitorx => Some([0x0f, 0x1f, 0x00]),
        Site::Mwaitx => Some([0xf3, 0x90, 0x90]),
        _ => None,
    }
}

/// Whether a `clzero` loop ([`Site::ClzeroLoop`]) is replaced the first time it faults
/// by a jump to the same loop with plain stores ([`clzero_stub`]). Run cpu-5 (the dorm): 25,000
/// exceptions a second, 2,100 a frame, at five such loops of one function of the title, for
/// some kilobytes each; ten times what the wait instructions raised. On by default;
/// `VICTORY_PATCH_CLZERO=0` emulates each time, as before (and the 15-byte form too).
fn patches_clzero() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_CLZERO").is_none_or(|v| v != "0"))
}

/// Whether the 15-byte `clzero` loop ([`Site::ClzeroLoop15`]) is replaced too, by a jump from
/// its `clzero` to a plain-store line and its own compare and branch ([`clzero_stub15`]).
/// Emulated, it costs an exception every time the loop is entered (the emulation does the whole
/// loop in one). On by default (with `VICTORY_PATCH_CLZERO`); `VICTORY_CLZERO_LOOP15=0` leaves
/// it emulated.
fn patches_clzero15() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| patches_clzero() && std::env::var_os("VICTORY_CLZERO_LOOP15").is_none_or(|v| v != "0"))
}

/// Whether a lone `clwb` is replaced by no-ops of its length the first time it faults: the
/// PC's memory is coherent, it has nothing to do. On by default; `VICTORY_PATCH_CLWB=0` leaves
/// it emulated.
fn patches_clwb() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_CLWB").is_none_or(|v| v != "0"))
}

/// `jmp rel32` at `at` to `to`, if it reaches.
pub(crate) fn jump(at: usize, to: usize) -> Option<[u8; 5]> {
    let rel = i32::try_from(to as i64 - (at as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// Zeroes the 64-byte line `rax` is in with plain stores; every register comes out as it went
/// in, the flags do not (the caller compares after). (`rcx` is the scratch register, saved on
/// the stack and back: Windows x64 code keeps nothing below `rsp`.)
fn zero_line() -> Vec<u8> {
    let mut code = vec![
        0x51, // push rcx
        0x48, 0x89, 0xc1, // mov rcx, rax
        0x48, 0x83, 0xe1, 0xc0, // and rcx, -64
    ];
    for k in 0..8 {
        code.extend([0x48, 0xc7, 0x41, k * 8, 0, 0, 0, 0]); // mov qword [rcx + 8k], 0
    }
    code.push(0x59); // pop rcx
    code
}

/// The `clzero` loop of 12 bytes at `at` with plain stores, as code at `stub`: every register
/// and the flags come out as the loop leaves them. `cmp` is its `cmp rax, reg` (3 bytes, the
/// register not `rax`), `None` for another form.
fn clzero_stub(at: usize, stub: usize, cmp: [u8; 3]) -> Option<Vec<u8>> {
    if !matches!(cmp[0], 0x48 | 0x4c) || cmp[1] != 0x39 || cmp[2] & 0xc7 != 0xc0 || (cmp[0] == 0x48 && cmp[2] == 0xc0) {
        return None;
    }
    let mut code = zero_line();
    code.extend([0x48, 0x83, 0xc0, 0x40]); // add rax, 0x40
    code.extend(cmp);
    let back = -(code.len() as i64 + 2);
    code.extend([0x72, i8::try_from(back).ok()? as u8]); // jb the start
    let end = stub + code.len();
    code.extend(jump(end, at + 12)?);
    Some(code)
}

/// What the `clzero` at `at` of a 15-byte loop jumps to, as code at `stub`: the line, the loop's
/// own `cmp a, b`, `jb` back to the loop's `mov` 7 bytes before `at`, else on after the loop's
/// `jb` (8 bytes after `at`). The `mov` and `add` stay in the title's code and run as before.
fn clzero_stub15(at: usize, stub: usize, cmp: [u8; 3]) -> Option<Vec<u8>> {
    let mut code = zero_line();
    code.extend(cmp);
    // jb rel32 to the loop's top.
    let from = stub + code.len() + 6;
    let rel = i32::try_from((at as i64 - 7) - from as i64).ok()?;
    code.extend([0x0f, 0x82]);
    code.extend(rel.to_le_bytes());
    let end = stub + code.len();
    code.extend(jump(end, at + 8)?);
    Some(code)
}

/// Room for [`clzero_stub`]s near the title's code: where it is and the bytes used.
static STUBS: std::sync::Mutex<Option<(usize, usize)>> = std::sync::Mutex::new(None);
pub(crate) const STUBS_LEN: usize = 64 << 10;
/// Where [`STUBS`] is (0: not made yet), read without its lock by [`is_replacement`].
static STUBS_AT: AtomicUsize = AtomicUsize::new(0);

/// Memory for code within reach of a `jmp rel32` at `at`.
pub(crate) fn near(at: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
    let at = at & !0xffff;
    (0x10_0000..0x7000_0000).step_by(0x10_0000).flat_map(|d| [at.checked_sub(d), at.checked_add(d)]).flatten().find_map(|place| {
        // SAFETY: reserves free memory at that address or fails.
        let got = unsafe { VirtualAlloc(place as *const _, STUBS_LEN, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) } as usize;
        (got != 0).then_some(got)
    })
}

/// Replaces the 5 bytes at `at`, which are `was`, by a jump to `body` placed near it: whether
/// it did.
///
/// # Safety
/// `at` is in the title's code, which stays mapped; `body` does what the code it jumps over did,
/// and no thread is ever inside those 5 bytes but at their first.
unsafe fn patch_to_stub(at: usize, was: [u8; 5], body: impl FnOnce(usize) -> Option<Vec<u8>>) -> bool {
    let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
    if stubs.is_none() {
        *stubs = near(at).map(|place| (place, 0));
        if let Some((place, _)) = *stubs {
            STUBS_AT.store(place, Relaxed);
        }
    }
    let Some((place, used)) = stubs.as_mut() else {
        return false;
    };
    let stub = *place + *used;
    let (Some(body), Some(jump)) = (body(stub), jump(at, stub)) else {
        return false;
    };
    if *used + body.len() > STUBS_LEN || (at & 15) + 5 > 16 {
        return false;
    }
    // SAFETY: inside the memory made for it, which nothing runs yet; then the title's code.
    unsafe {
        std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
        let done = patch(at, &was, &jump);
        // A stub nothing jumps to (another thread replaced the code first, with its own stub)
        // gives its room back; the lock is held, so no other stub was placed after it.
        if done {
            *used += body.len().next_multiple_of(16);
        }
        done
    }
}

/// The aligned 16 bytes `word` around address `at` with the bytes at `at` replaced; `None`
/// where they do not all lie in those 16. (16 and not 8: the title's busiest one lies over the
/// end of its 8 and took 28,000 exceptions a second, run cpu-6.)
fn with_patch(word: u128, at: usize, patch: &[u8]) -> Option<u128> {
    let offset = at & 15;
    if offset + patch.len() > 16 {
        return None;
    }
    let mut bytes = word.to_le_bytes();
    bytes[offset..offset + patch.len()].copy_from_slice(patch);
    Some(u128::from_le_bytes(bytes))
}

/// One thread at a time changes code: a second one would take the first one's loosened
/// protection for the page's own.
static PATCHING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Replaces the bytes at `at`, which are `was`, by `patch` (as long), in one store that other
/// threads see whole: whether they are replaced now (not where they cross 16 aligned bytes, or
/// the page cannot be written).
///
/// # Safety
/// `at` is in code that stays mapped, and `patch` does for it what `was` does.
unsafe fn patch(at: usize, was: &[u8], patch: &[u8]) -> bool {
    use std::sync::atomic::Ordering::SeqCst;
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let n = patch.len();
    if was.len() != n || (at & 15) + n > 16 || !std::is_x86_feature_detected!("cmpxchg16b") {
        return false;
    }
    let _one = PATCHING.lock().unwrap_or_else(|e| e.into_inner());
    let word = (at & !15) as *mut u128;
    let here = (at & 15)..(at & 15) + n;
    let mut old = 0u32;
    // SAFETY: the aligned 16 bytes around the instruction lie in one page of mapped code, made
    // writable for the store and put back; the processor has the instruction (asked above).
    unsafe {
        if VirtualProtect(word.cast(), 16, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        // What is there (zeros are replaced by zeros).
        let mut now = std::arch::x86_64::cmpxchg16b(word, 0, 0, SeqCst, SeqCst);
        let done = loop {
            let bytes = &now.to_le_bytes()[here.clone()];
            if bytes == patch {
                // (Another thread's fault at the same place: done already.)
                break true;
            }
            let Some(new) = (bytes == was).then(|| with_patch(now, at, patch)).flatten() else {
                break false;
            };
            let there = std::arch::x86_64::cmpxchg16b(word, now, new, SeqCst, SeqCst);
            if there == now {
                break true;
            }
            now = there;
        };
        VirtualProtect(word.cast(), 16, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), word.cast(), 16);
        done
    }
}

/// Whether the code at `rip` is one of the replacements made here: `0f 1f 00` (for `monitorx`),
/// `pause; nop` (for `mwaitx`), a run of the no-ops a `clwb` becomes, or a jump into the stubs.
/// A thread that faulted on the instruction before another thread replaced it finds this there
/// in its handler, and goes on into the replacement instead of being left to an emulation that
/// knows only the console's instructions (the illegal instruction would kill the title).
/// Replacements are never undone, so once this is true it stays true.
///
/// # Safety
/// `rip` is where the faulting thread is, in code that stays mapped.
pub(crate) unsafe fn is_replacement(rip: usize) -> bool {
    let room = 0x1000 - (rip & 0xfff);
    let mut code = [0u8; 5];
    let n = room.min(code.len());
    // SAFETY: inside the page of the faulting instruction.
    unsafe { std::ptr::copy_nonoverlapping(rip as *const u8, code.as_mut_ptr(), n) };
    replacement(&code[..n], rip, STUBS_AT.load(Relaxed))
}

/// [`is_replacement`] for `code` (the bytes at `at`, up to 5) and stubs at `stubs` (0: none).
fn replacement(code: &[u8], at: usize, stubs: usize) -> bool {
    match *code {
        [0x0f, 0x1f, 0x00, ..] | [0xf3, 0x90, 0x90, ..] | [0x90, 0x90, 0x90, 0x90, ..] => true,
        [0xe9, a, b, c, d] if stubs != 0 => {
            let to = (at as i64 + 5 + i64::from(i32::from_le_bytes([a, b, c, d]))) as usize;
            (stubs..stubs + STUBS_LEN).contains(&to)
        }
        _ => false,
    }
}

/// Places replaced so far, of each kind (wait, `clzero` loop, 15-byte `clzero` loop, `clwb`).
static REPLACED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Replaces the instruction at `rip`, where an illegal-instruction exception was just raised,
/// when it is one replaced here: whether the code at `rip` is the replacement now, so the thread
/// goes on there as it is (it runs the replacement, which does the instruction's job). `false`
/// leaves it to the emulation.
///
/// # Safety
/// `rip` is where the faulting thread is, in code that stays mapped.
pub(crate) unsafe fn replace_at_fault(rip: usize) -> bool {
    // Only bytes inside the faulting instruction's page are read (the CPU fetched from it); the
    // next page, or the one before, may not be mapped.
    let room = 0x1000 - (rip & 0xfff);
    let mut code = [0u8; 32];
    let n = room.min(code.len());
    // SAFETY: inside the page of the faulting instruction.
    unsafe { std::ptr::copy_nonoverlapping(rip as *const u8, code.as_mut_ptr(), n) };
    let before = (rip & 0xfff >= 7).then(|| {
        // SAFETY: as above.
        unsafe { ((rip - 7) as *const [u8; 7]).read_unaligned() }
    });
    let what = site(before, &code[..n]);
    #[cfg(feature = "diagnostics")]
    sites::count(rip, what);
    // SAFETY: (each) the title's code, where the instruction just faulted, and a replacement
    // that does its job.
    let (kind, done) = unsafe {
        match what {
            Site::Monitorx | Site::Mwaitx if patches_waits() => {
                // Both replacements are waits that wait for nothing, as the emulation is.
                (0, wait_patch(what).is_some_and(|new| patch(rip, &code[..3], &new)))
            }
            Site::ClzeroLoop(cmp) if patches_clzero() => {
                // No thread is ever inside the loop's other instructions (the emulation does
                // the whole loop in one), so the five bytes of the jump may take the place of
                // two of them.
                let was = [code[0], code[1], code[2], code[3], code[4]];
                (1, patch_to_stub(rip, was, |stub| clzero_stub(rip, stub, cmp)))
            }
            Site::ClzeroLoop15(cmp) if patches_clzero15() => {
                // The jump takes the place of the `clzero` and the `cmp`'s first two bytes; no
                // thread is ever at the `cmp` (the emulation does the whole loop in one, and
                // leaves past its `jb`).
                let was = [code[0], code[1], code[2], code[3], code[4]];
                (2, patch_to_stub(rip, was, |stub| clzero_stub15(rip, stub, cmp)))
            }
            Site::Clwb(len) if patches_clwb() => (3, patch(rip, &code[..len], &[0x90; 16][..len])),
            // (Something else: perhaps another thread's replacement, made since this fault.)
            _ => return is_replacement(rip),
        }
    };
    if done {
        REPLACED[kind].fetch_add(1, Relaxed);
        #[cfg(feature = "diagnostics")]
        sites::patched();
        let all: u64 = REPLACED.iter().map(|r| r.load(Relaxed)).sum();
        if all.is_power_of_two() {
            tracing::info!(
                waits = REPLACED[0].load(Relaxed),
                clzero_loops = REPLACED[1].load(Relaxed),
                clzero_loops15 = REPLACED[2].load(Relaxed),
                clwb = REPLACED[3].load(Relaxed),
                "console CPU instructions replaced in the title's code so far"
            );
        }
    }
    // Not replaced by this thread: by another one meanwhile (with another stub, say)?
    // SAFETY: as above.
    done || unsafe { is_replacement(rip) }
}

/// Diagnostics: how often each place in the title's code raises one of these exceptions, the
/// busiest logged every five seconds.
#[cfg(feature = "diagnostics")]
mod sites {
    use super::{AtomicU64, Relaxed, Site};
    use std::sync::atomic::AtomicUsize;

    /// Address, what is there (an index into [`NAMES`]), exceptions since the last line.
    static SITES: [(AtomicUsize, AtomicUsize, AtomicU64); 512] =
        [const { (AtomicUsize::new(0), AtomicUsize::new(0), AtomicU64::new(0)) }; 512];
    const NAMES: [&str; 8] = ["other", "clwb", "monitorx", "mwaitx", "clzero", "clzero loop", "rdpru", "clzero loop 15"];
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    static PATCHED: AtomicU64 = AtomicU64::new(0);

    pub(super) fn patched() {
        PATCHED.fetch_add(1, Relaxed);
    }

    pub(super) fn count(at: usize, what: Site) {
        let kind = match what {
            Site::Clwb(_) => 1,
            Site::Monitorx => 2,
            Site::Mwaitx => 3,
            Site::Clzero => 4,
            Site::ClzeroLoop(_) => 5,
            Site::Rdpru => 6,
            Site::ClzeroLoop15(_) => 7,
            Site::Other => 0,
        };
        let mut i = (at ^ at >> 9) % SITES.len();
        for _ in 0..8 {
            let s = &SITES[i];
            let there = s.0.compare_exchange(0, at, Relaxed, Relaxed).unwrap_or_else(|there| there);
            if there == 0 || there == at {
                s.1.store(kind, Relaxed);
                s.2.fetch_add(1, Relaxed);
                break;
            }
            i = (i + 1) % SITES.len();
        }
        // Now and then: is a line due? Never waited for.
        if let Ok(mut last) = LAST.try_lock() {
            let since = *last.get_or_insert_with(std::time::Instant::now);
            if since.elapsed().as_secs() >= 5 {
                let s = since.elapsed().as_secs_f64();
                *last = Some(std::time::Instant::now());
                drop(last);
                line(s);
            }
        }
    }

    fn line(seconds: f64) {
        let mut all: Vec<(u64, usize, usize)> =
            SITES.iter().map(|s| (s.2.swap(0, Relaxed), s.0.load(Relaxed), s.1.load(Relaxed))).filter(|s| s.0 > 0).collect();
        all.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
        let mut kinds = [0u64; NAMES.len()];
        for s in &all {
            kinds[s.2] += s.0;
        }
        let kinds: Vec<String> = NAMES
            .iter()
            .zip(kinds)
            .filter(|(_, n)| *n > 0)
            .map(|(name, n)| format!("{name} {:.0}", n as f64 / seconds))
            .collect();
        let top: Vec<String> =
            all.iter().take(8).map(|s| format!("{:#x} {} {:.0}", s.1, NAMES[s.2], s.0 as f64 / seconds)).collect();
        tracing::info!(
            "console CPU instructions that fault, a second: {}; places replaced so far: {}; the busiest places: {}",
            kinds.join(", "),
            PATCHED.load(Relaxed),
            top.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
    };

    #[test]
    fn sites_are_told_apart() {
        assert_eq!(site(None, &[0x0f, 0x01, 0xfa]), Site::Monitorx);
        assert_eq!(site(None, &[0x0f, 0x01, 0xfb, 0xc3]), Site::Mwaitx);
        assert_eq!(site(None, &[0x0f, 0x01, 0xfd]), Site::Rdpru);
        assert_eq!(site(None, &[0x0f, 0x01, 0xfc, 0xc3]), Site::Clzero);
        let lp = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4];
        assert_eq!(site(None, &lp), Site::ClzeroLoop([0x4c, 0x39, 0xd0]));
        // Against rax itself, or a cut loop: one line.
        assert_eq!(site(None, &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xc0, 0x72, 0xf4]), Site::Clzero);
        assert_eq!(site(None, &lp[..11]), Site::Clzero);
        // mov rax, rdx; add rdx, 0x40 | clzero; cmp rdx, r8; jb
        let before = [0x48, 0x89, 0xd0, 0x48, 0x83, 0xc2, 0x40];
        let rest = [0x0f, 0x01, 0xfc, 0x4c, 0x39, 0xc2, 0x72, 0xf1];
        assert_eq!(site(Some(before), &rest), Site::ClzeroLoop15([0x4c, 0x39, 0xc2]));
        assert_eq!(site(None, &rest), Site::Clzero, "the bytes before in another page");
        // The add on another register than the mov's.
        assert_eq!(site(Some([0x48, 0x89, 0xd0, 0x48, 0x83, 0xc1, 0x40]), &rest), Site::Clzero);
        // clwb [rax]; clwb [r9 + 8]
        assert_eq!(site(None, &[0x66, 0x0f, 0xae, 0x30]), Site::Clwb(4));
        assert_eq!(site(None, &[0x66, 0x41, 0x0f, 0xae, 0x71, 0x08]), Site::Clwb(6));
        assert_eq!(site(None, &[0x90]), Site::Other);
    }

    #[test]
    fn replacements_are_known_again() {
        assert!(replacement(&[0x0f, 0x1f, 0x00, 0xc3], 0x1000, 0));
        assert!(replacement(&[0xf3, 0x90, 0x90], 0x1000, 0));
        assert!(replacement(&[0x90, 0x90, 0x90, 0x90, 0xc3], 0x1000, 0));
        assert!(!replacement(&[0x90, 0x90, 0x90, 0xc3], 0x1000, 0), "shorter than any clwb");
        assert!(!replacement(&[0x0f, 0x01, 0xfb], 0x1000, 0));
        assert!(!replacement(&[0x90], 0x1000, 0), "cut by the page's end");
        // jmp from 0x10_0000 to 0x20_0000, the stubs there or elsewhere.
        let j = jump(0x10_0000, 0x20_0000).unwrap();
        assert!(replacement(&j, 0x10_0000, 0x20_0000));
        assert!(replacement(&j, 0x10_0000, 0x20_0000 - STUBS_LEN + 1));
        assert!(!replacement(&j, 0x10_0000, 0x20_0001));
        assert!(!replacement(&j, 0x10_0000, 0x20_0000 - STUBS_LEN));
        assert!(!replacement(&j, 0x10_0000, 0), "no stubs yet");
    }

    #[test]
    fn wait_instructions_are_replaced_inside_sixteen_aligned_bytes() {
        let word = u128::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 0x0f, 0x01, 0xfb, 11, 12, 13, 14, 15, 16]);
        let new = with_patch(word, 0x1007, &[0xf3, 0x90, 0x90]).unwrap();
        assert_eq!(new.to_le_bytes(), [1, 2, 3, 4, 5, 6, 7, 0xf3, 0x90, 0x90, 11, 12, 13, 14, 15, 16]);
        assert!(with_patch(word, 0x100d, &[0; 3]).is_some());
        assert!(with_patch(word, 0x100e, &[0; 3]).is_none(), "crosses into the next sixteen");
    }

    /// `monitorx; mwaitx; clwb [rax]; ret` in a page of its own: each is replaced at its first
    /// fault, and the page is protected as before.
    #[test]
    fn waits_and_clwb_fault_once_and_the_page_keeps_its_protection() {
        // SAFETY: memory of our own, read, written and run within its page.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!page.is_null());
            let code = [0x0f, 0x01, 0xfa, 0x0f, 0x01, 0xfb, 0x66, 0x0f, 0xae, 0x30, 0xc3];
            // The first over the end of its aligned eight bytes, as the title's is; and one
            // over the end of its sixteen (at 46), which is left alone.
            std::ptr::copy_nonoverlapping(code.as_ptr(), page.add(6), code.len());
            std::ptr::copy_nonoverlapping(code.as_ptr(), page.add(46), code.len());
            let mut old = 0;
            assert_ne!(VirtualProtect(page.cast(), 4096, PAGE_EXECUTE_READ, &mut old), 0);
            let at = page as usize;
            for rip in [at + 6, at + 9, at + 12] {
                assert_eq!(replace_at_fault(rip), patches_waits() || rip == at + 12 && patches_clwb());
            }
            assert!(!replace_at_fault(at + 46), "crosses its sixteen bytes");
            if patches_waits() && patches_clwb() {
                let now = std::slice::from_raw_parts(page.add(6), 11);
                assert_eq!(now, [0x0f, 0x1f, 0x00, 0xf3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0xc3]);
                // They run now.
                let run: extern "system" fn() = std::mem::transmute(page.add(6));
                run();
                for rip in [at + 6, at + 9, at + 12] {
                    assert!(replace_at_fault(rip), "a thread that faulted before the replacement");
                }
                // A second thread that faulted at the same place meanwhile.
                assert!(patch(at + 6, &[0x0f, 0x01, 0xfa], &[0x0f, 0x1f, 0x00]));
            }
            assert_eq!(std::slice::from_raw_parts(page.add(46), 3), [0x0f, 0x01, 0xfa]);
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(page.cast(), &mut m, size_of_val(&m));
            assert_eq!(m.Protect, PAGE_EXECUTE_READ);
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }
    }

    /// `clzero` loops run as plain stores: the lines they name are zero and no other byte,
    /// and the pointer, the scratch register and the flags come out as from the loop.
    #[test]
    fn a_clzero_loop_becomes_plain_stores() {
        // Against rdx (rm rax, reg rdx) and against rcx, the scratch register.
        for (cmp, by) in [([0x48, 0x39, 0xd0], 2), ([0x48, 0x39, 0xc8], 1), ([0x4c, 0x39, 0xc0], 8)] {
            // SAFETY: memory of our own, written and run inside it.
            unsafe {
                let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                assert_ne!(page, 0);
                // A function of (start, end): the start in rax, the end in its register, the
                // loop at 16, and what rax became.
                let mut head = vec![0x48, 0x89, 0xc8]; // mov rax, rcx
                match by {
                    1 => head.extend([0x48, 0x89, 0xd1]), // mov rcx, rdx
                    8 => head.extend([0x49, 0x89, 0xd0]), // mov r8, rdx
                    _ => {}
                }
                head.resize(16, 0x90);
                let mut code = head;
                code.extend([0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40]);
                code.extend(cmp);
                code.extend([0x72, 0xf4]);
                code.push(0xc3);
                std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len());
                let mut block = vec![0xaau8; 64 * 8 + 64];
                let base = block.as_mut_ptr() as usize;
                let first = base.next_multiple_of(64);
                assert_eq!(replace_at_fault(page + 16), patches_clzero());
                if patches_clzero() {
                    assert_eq!(*((page + 16) as *const u8), 0xe9, "a jump");
                    // A thread that read the loop before the jump was written: its own stub is
                    // not used (and gives its room back), and it goes on into the jump there.
                    let was = [0x0f, 0x01, 0xfc, 0x48, 0x83];
                    assert!(!patch_to_stub(page + 16, was, |stub| clzero_stub(page + 16, stub, cmp)));
                    assert!(is_replacement(page + 16));
                    assert!(replace_at_fault(page + 16), "a second fault there finds the jump");
                    let run: extern "system" fn(usize, usize) -> usize = std::mem::transmute(page);
                    // From inside a line to inside the third: whole lines, one past the end.
                    assert_eq!(run(first + 8, first + 130), first + 8 + 128);
                    let at = first - base;
                    assert!(block[at..at + 128].iter().all(|&b| b == 0));
                    assert!(block[..at].iter().chain(&block[at + 128..]).all(|&b| b == 0xaa));
                    // An end at the start: one line.
                    block.fill(0xaa);
                    assert_eq!(run(first, first), first + 64);
                    assert!(block[at..at + 64].iter().all(|&b| b == 0) && block[at + 64] == 0xaa);
                }
                VirtualFree(page as *mut _, 0, MEM_RELEASE);
            }
        }
        assert!(clzero_stub(0x1000, 0x2000, [0x48, 0x39, 0xc0]).is_none(), "against rax itself");
        assert!(jump(0x1000, 0x9000_0000_0000).is_none(), "out of reach");
    }

    /// The 15-byte loop, `mov rax, a; add a, 0x40; clzero; cmp a, b; jb`, runs as plain stores
    /// after its first fault, with `a` and the flags as the loop leaves them.
    #[test]
    fn a_fifteen_byte_clzero_loop_becomes_plain_stores() {
        // SAFETY: memory of our own, written and run inside it.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
            assert_ne!(page, 0);
            // fn(start: rcx, end: rdx) -> a: a = rcx in r9, the loop at 9 (clzero at 16), then
            // mov rax, r9; ret.
            let mut code = vec![0x49, 0x89, 0xc9]; // mov r9, rcx
            code.resize(9, 0x90);
            code.extend([0x4c, 0x89, 0xc8]); // mov rax, r9
            code.extend([0x49, 0x83, 0xc1, 0x40]); // add r9, 0x40
            code.extend([0x0f, 0x01, 0xfc]); // clzero
            code.extend([0x49, 0x39, 0xd1]); // cmp r9, rdx
            code.extend([0x72, 0xf1]); // jb the mov
            code.extend([0x4c, 0x89, 0xc8, 0xc3]); // mov rax, r9; ret
            std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len());
            let mut block = vec![0xaau8; 64 * 8 + 64];
            let base = block.as_mut_ptr() as usize;
            let first = base.next_multiple_of(64);
            let at = first - base;
            assert_eq!(replace_at_fault(page + 16), patches_clzero15());
            if patches_clzero15() {
                assert_eq!(*((page + 16) as *const u8), 0xe9, "a jump");
                assert!(replace_at_fault(page + 16), "a second fault there finds the jump");
                let run: extern "system" fn(usize, usize) -> usize = std::mem::transmute(page);
                // Lines from the start until `a` (64 past the line zeroed) reaches the end.
                assert_eq!(run(first, first + 130), first + 192);
                assert!(block[at..at + 192].iter().all(|&b| b == 0));
                assert!(block[..at].iter().chain(&block[at + 192..]).all(|&b| b == 0xaa));
                block.fill(0xaa);
                assert_eq!(run(first, first), first + 64);
                assert!(block[at..at + 64].iter().all(|&b| b == 0) && block[at + 64] == 0xaa);
            }
            VirtualFree(page as *mut _, 0, MEM_RELEASE);
        }
    }
}
