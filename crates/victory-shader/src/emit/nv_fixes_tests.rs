//! Translator fixes for hosts of 32-lane waves (NVIDIA) ported from upstream and sibling forks:
//! the code table's start, LDS bounds, wave64 `ds_bpermute_b32` within 32-lane halves, the
//! automatic barrier mode, wave64 masks restored from split scalar copies, the thread-group
//! info SGPR and the `s_pack_*` scalar packs. Lowered, validated by `dxil.dll` when present,
//! and run on the host GPU when one takes the program.

use crate::{ir, resource};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A program's constant table starts after its instructions, unless a load at a fixed offset
/// reads below that (upstream e04c0781, desp).
#[test]
fn the_code_table_starts_after_the_instructions() {
    // s_nop 0, s_endpgm, then 1,000 dwords of constants.
    let mut words = vec![0xBF80_0000, S_ENDPGM];
    words.extend(0..1000u32);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    let at = |offset: u32, dynamic: bool| resource::Address {
        param: resource::CODE,
        offset,
        dynamic: dynamic.then_some(ir::Value(0)),
    };
    let (computed, fixed_after, fixed_inside) = (at(0, true), at(12, false), at(4, false));
    let start = super::lower::code_table_start;
    // A load at a computed offset from the code's start: the table is the constants only, not
    // the whole program.
    assert_eq!(start(&p, &[&computed], 0), 8);
    assert_eq!(start(&p, &[&computed, &fixed_after], 0), 8);
    // A load at a fixed offset inside the instructions keeps the start.
    assert_eq!(start(&p, &[&computed, &fixed_inside], 0), 0);
    // Loads past the instructions only: from the lowest of them.
    assert_eq!(start(&p, &[&fixed_after], 12), 12);
}
