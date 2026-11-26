//! The structure builder's programs (a spilled user-data table) on a host of 32-lane waves:
//! never in an approximate form ([`super::approximate_allowed`]), which other programs may
//! take, marked as such ([`super::HostShader::approximate`]).

use super::{HostShader, lower, take_form};
use crate::native::{NativeProgram, RootParam, RootType};
use crate::{ShaderError, io, ir, resource, structure};

/// [`super::emit`] of `native` as for a host of 32-lane waves.
fn emitted_on_32(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let p = crate::spill::build(native).expect("spill::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let r = super::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    r
}

/// The wave64 program of a wave OR inside a per-lane branch (`s_cbranch_scc0` past it, which
/// leaving skips out does not take away),
/// which only the approximate form (each host wave ORs its own half) takes on 32-lane waves
/// while wave-wide branches are off (`VICTORY_WAVE_WIDE`, off by default).
fn only_approximate() -> NativeProgram {
    let native = super::wave64_tests::program(0xBF84_0000);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    assert!(lower::wave_local(&p, &st).is_err(), "not the exact allocation idiom");
    native
}

#[test]
fn a_program_only_the_approximate_form_takes_is_marked_approximate() {
    let native = only_approximate();
    assert!(!crate::spill::is_spilled(&native));
    take_form();
    let shader = emitted_on_32(&native).expect("translated approximately");
    assert!(shader.approximate);
    assert!(take_form().starts_with("APPROXIMATE"));
}

#[test]
fn a_spilled_program_only_the_approximate_form_takes_is_refused_on_32_lanes() {
    let mut native = only_approximate();
    // The same program with a parameter in a spilled user-data table (index 16, past its 5
    // user SGPRs): a program of the structure builder.
    native.layout.push(RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 16,
        slot: 2,
    });
    assert!(crate::spill::is_spilled(&native));
    let r = emitted_on_32(&native);
    assert!(
        matches!(&r, Err(ShaderError::Unsupported { .. })),
        "{:?}",
        r.map(|s| s.approximate)
    );
}

#[test]
fn an_exact_translation_is_not_marked_approximate() {
    let native = super::wave64_tests::wave_number_program(9);
    take_form();
    let shader = emitted_on_32(&native).expect("translated");
    assert!(!shader.approximate);
    assert!(!take_form().starts_with("APPROXIMATE"));
}
