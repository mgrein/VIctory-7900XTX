//! Translator fixes for hosts of 32-lane waves (NVIDIA) ported from upstream and sibling forks:
//! the code table's start, LDS bounds, wave64 `ds_bpermute_b32` within 32-lane halves, the
//! automatic barrier mode, wave64 masks restored from split scalar copies, the thread-group
//! info SGPR and the `s_pack_*` scalar packs. Lowered, validated by `dxil.dll` when present,
//! and run on the host GPU when one takes the program.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A compute program of `threads` threads (the thread index in v0) with 4 root constants in
/// s0..s3, a raw UAV's descriptor in s[8:11] and `lds_bytes` of LDS, then `body`.
fn native(body: &[u32], wave64: bool, threads: u32, lds_bytes: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            // 5 user SGPRs (s0..s4), LDS in 512-byte granules
            rsrc2: 5 << 1 | lds_bytes.div_ceil(512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `buffer_store_dword v<vdata>, v16, s[8:11], 0 offen offset:<offset>` (v16: the byte
/// address).
fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x80 << 24 | 2 << 16 | u32::from(vdata) << 8 | 16]
}

/// A DS instruction: `op`, its 16-bit offset, `vdst`, `addr` and `data0`.
fn ds(op: u32, offset: u16, vdst: u8, addr: u8, data: u8) -> [u32; 2] {
    [
        0xD800_0000 | op << 18 | u32::from(offset),
        u32::from(vdst) << 24 | u32::from(data) << 8 | u32::from(addr),
    ]
}

/// `native` lowered for a host whose waves have at most `host_wave` lanes, and validated by
/// `dxil.dll` when present.
fn lowered(native: &NativeProgram, host_wave: u32) -> Result<(HostShader, lower::Lowered), ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure_wide(&p, &HashSet::new())?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil.clone(),
        user_data: lowered.user_data.clone(),
        samplers: lowered.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs.clone(),
        root_srvs: lowered.root_srvs.clone(),
        root_uavs: lowered.root_uavs.clone(),
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    Ok((shader, lowered))
}

/// Runs `shader` (one group) on the hardware device with root constants `constants` and a raw
/// UAV of `dwords` dwords, which it returns; `None` where no device takes the program.
fn run(shader: &HostShader, constants: [u32; 4], dwords: u32) -> Option<Vec<u32>> {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program: {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0) = (u.root, u.slot)
            && let Some(&k) = constants.get(u.dword as usize)
        {
            ud[u.index as usize] = k;
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0].clone())
}

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

/// LDS accesses past the program's allocation (512 bytes here) read 0 and drop writes (upstream
/// eabb6b85): dwords, bytes and returning atomics; a dropped write lands nowhere in bounds, and
/// on a 32-lane host the dwords the translation keeps after the title's LDS (`wave_base`) are
/// not the title's to write. Each thread uses its own dword and the dword 512 bytes past it.
#[test]
fn lds_out_of_bounds_reads_zero_and_drops_writes() {
    let mut body = vec![
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x4A02_00FF,
        0x0000_0100, // v_add_nc_u32 v1, 0x100, v0
        0x7E04_02FF,
        0x0000_DEAD, // v_mov_b32 v2, 0xdead
        0x7E0E_02FF,
        0x0000_0077, // v_mov_b32 v7, 0x77
    ];
    body.extend(ds(0x0D, 0, 0, 16, 1)); // ds_write_b32 [own], v1
    body.extend(ds(0x0D, 512, 0, 16, 2)); // ds_write_b32 [own + 512], v2 (dropped)
    body.extend(ds(0x1E, 513, 0, 16, 2)); // ds_write_b8 [own + 513], v2 (dropped)
    body.extend(ds(0x36, 0, 5, 16, 0)); // ds_read_b32 v5, [own]
    body.extend(ds(0x36, 512, 6, 16, 0)); // ds_read_b32 v6, [own + 512]
    body.extend(ds(0x20, 512, 7, 16, 2)); // ds_add_rtn_u32 v7, [own + 512], v2
    body.extend(ds(0x3A, 600, 8, 16, 0)); // ds_read_u8 v8, [own + 600]
    body.extend(ds(0x3A, 1, 9, 16, 0)); // ds_read_u8 v9, [own + 1]
    for (k, v) in [5u8, 6, 7, 8, 9].into_iter().enumerate() {
        body.extend(store(v, 256 * k as u32));
    }
    for (wave64, threads) in [(false, 32), (true, 64), (true, 32)] {
        let native = native(&body, wave64, threads, 512);
        let (shader, _) = lowered(&native, 32).unwrap_or_else(|e| panic!("wave64={wave64}: {e:?}"));
        lowered(&native, 64).unwrap_or_else(|e| panic!("wave64={wave64} on 64-lane hosts: {e:?}"));
        let Some(out) = run(&shader, [0; 4], 5 * 64) else { continue };
        for t in 0..threads as usize {
            let at = |k: usize| out[64 * k + t];
            assert_eq!(
                [at(0), at(1), at(2), at(3), at(4)],
                [0x100 + t as u32, 0, 0, 0, 1],
                "wave64={wave64}, {threads} threads, thread {t}"
            );
        }
        eprintln!("ran on the hardware device: wave64={wave64}, {threads} threads");
    }
}

/// The wave64 half-wave base (`lane_index & !31`) appears only in a wave64 program's
/// `ds_bpermute_b32` (upstream 0665a0e8, QB): without reading v0 nothing else asks the lane
/// index. On a 32-lane host the wave64 program now runs in waves of 32 (it used to need
/// 64-lane waves, or an exchange that refused the permute).
#[test]
fn ds_bpermute_wave64_preserves_32_lane_half() {
    let words = [
        0x7E02_02FF,
        0x0000_0104, // v_mov_b32 v1, 0x104
        0x7E04_0285, // v_mov_b32 v2, 5
        0xDACC_000C,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:12
        0x7E00_0503, // v_readfirstlane_b32 s0, v3
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    for wave64 in [false, true] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code: code.clone(),
        };
        let (_, l) = lowered(&native, 64).unwrap_or_else(|e| panic!("wave64={wave64}: {e:?}"));
        assert!(l.wave_ops);
        assert_eq!(l.uses_lane_index, wave64, "wave64={wave64}");
        assert_eq!(l.has_half_wave_mask, wave64, "wave64={wave64}");
        let (_, l) = lowered(&native, 32).unwrap_or_else(|e| panic!("wave64={wave64} on 32 lanes: {e:?}"));
        assert_eq!(l.wave_size, Some(32), "wave64={wave64} on 32 lanes");
    }
}

/// Lowers `body` (each lane's result left in v3) as one group of one console wave, runs it on
/// the hardware device and compares v3 of every lane with `want`.
fn run_permute(wave64: bool, body: &[u32], want: &[u32]) {
    let wave = if wave64 { 64 } else { 32 };
    assert_eq!(want.len(), wave as usize);
    let mut words = body.to_vec();
    words.push(0x3420_0082); // v_lshlrev_b32 v16, 2, v0
    words.extend(store(3, 0));
    let native = native(&words, wave64, wave, 0);
    lowered(&native, 64).unwrap_or_else(|e| panic!("wave64={wave64} on 64 lanes: {e:?}"));
    let (shader, l) = lowered(&native, 32).unwrap_or_else(|e| panic!("wave64={wave64}: {e:?}"));
    assert_eq!(l.wave_size, Some(32));
    if let Some(out) = run(&shader, [0; 4], wave) {
        assert_eq!(out, want, "wave64={wave64}");
        eprintln!("ran on the hardware device: wave64={wave64}");
    }
}

/// High byte-address bits (0x100) and sub-dword bits (+3) are ignored, so only bits 6:2 select
/// the source lane. Lanes 16..31 skip the permute via `s_cbranch_execz`, so reads from them give
/// 0 while they keep v3.
#[test]
fn permutes_mask_high_address_bits_and_divergent_branches() {
    let body = [
        0x4A04_00FF,
        0x0000_3000, // v_add_nc_u32 v2, 0x3000, v0
        0x4A06_00FF,
        0xBEEF_0000, // v_add_nc_u32 v3, 0xbeef0000, v0
        0x4A02_0088, // v_add_nc_u32 v1, 8, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x4A02_02FF,
        0x0000_0103, // v_add_nc_u32 v1, 0x103, v1
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0xBF88_0002, // s_cbranch_execz +2
        0xDACC_0000,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
    ];
    let want: Vec<u32> = (0..32)
        .map(|l| if l < 8 { 0x3000 + l + 8 } else if l < 16 { 0 } else { 0xBEEF_0000 + l })
        .collect();
    run_permute(false, &body, &want);
}

/// RDNA 2 ISA 10.4.4: wave64 permutes lanes 0..31 and 32..63 independently. v1 = ((lane & 31)
/// << 2) + 0x103 with offset:12 wraps within each half (29..31 -> 0..2, 61..63 -> 32..34; lane
/// 32 pulls from lane 35, not lane 3).
#[test]
fn wave64_permutes_stay_within_each_32_lane_half() {
    let body = [
        0x4A04_00FF,
        0x0000_4000, // v_add_nc_u32 v2, 0x4000, v0
        0x3602_009F, // v_and_b32 v1, 31, v0
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x4A02_02FF,
        0x0000_0103, // v_add_nc_u32 v1, 0x103, v1
        0xDACC_000C,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:12
    ];
    let want: Vec<u32> = (0..64).map(|l| 0x4000 + ((l & !31) | ((l + 3) & 31))).collect();
    run_permute(true, &body, &want);
}

/// Active lanes 0..39 of a wave64: lanes 32..39 pull from upper-half lanes 40..47 (inactive:
/// 0) though lower-half lanes 8..15 are active; inactive lanes 40..63 keep 0xcafe0000 + lane.
#[test]
fn wave64_permutes_check_exec_within_each_half() {
    let body = [
        0x4A04_00FF,
        0x0000_5000, // v_add_nc_u32 v2, 0x5000, v0
        0x4A06_00FF,
        0xCAFE_0000, // v_add_nc_u32 v3, 0xcafe0000, v0
        0x4A02_0088, // v_add_nc_u32 v1, 8, v0
        0x3602_029F, // v_and_b32 v1, 31, v1
        0x3402_0282, // v_lshlrev_b32 v1, 2, v1
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xDACC_0000,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
    ];
    let want: Vec<u32> = (0..64)
        .map(|l| if l < 32 { 0x5000 + ((l + 8) & 31) } else if l < 40 { 0 } else { 0xCAFE_0000 + l })
        .collect();
    run_permute(true, &body, &want);
}
