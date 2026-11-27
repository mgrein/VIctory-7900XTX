//! Mesh draws launched by the card: an amplification shader of ours
//! (`launch.hlsl`, its code in `launch_code.rs`) in front of every
//! translated mesh shader (`VICTORY_MESH_LAUNCH`, off by default).
//!
//! The title draws its meshes one indirect record a call, each from the
//! next slot of an argument buffer its own compute passes wrote (3,523 mesh
//! draws in a captured frame of gameplay). Each record is bounded before
//! it runs by a compute pass of ours (`crate::indirect`, in batches:
//! `crate::indirect::batch`), which waits for every earlier shader write
//! and moves its scratch buffer to indirect arguments, so the card turns
//! from drawing to compute and back around every batch.
//!
//! Here the record is read by the draw itself: the pipeline's amplification
//! stage loads its three words where the title's passes wrote them, bounds
//! them as the clamp does (a record over 65,535 groups runs empty, one
//! dimension), charges the list's budget as the clamp does (with an atomic
//! on the same word: a record that does not fit runs empty, and with
//! `VICTORY_INDIRECT_REST` on takes the rest), notes the same words for the
//! frame line, and launches the mesh groups. A mesh dispatch the CPU knows
//! (`DispatchMesh`) gives its count as a root constant and is launched the
//! same way, unbounded by the budget as before.
//!
//! What the record reads must be written before the draw runs: with
//! `VICTORY_MESH_LAUNCH=1` a launch waits for every earlier shader write
//! when the list may have recorded work that writes buffers since its last
//! such wait (`crate::host::rec::Rec::uav_work`: dispatches, draws whose
//! shaders store, copies into buffers, our clamp passes, title barriers),
//! and in the first launch of a recording; the launches between, after
//! draws that store nothing, have no wait. `VICTORY_MESH_LAUNCH=nowait`
//! leaves the waits out, relying on the title's own barriers alone (as the
//! fork this came from does).
//!
//! The draws' amplification groups may run in another order than the
//! list's (nothing orders draws now), so when the list's budget runs out,
//! which of its records run empty is the card's order: the bound on the
//! list's total work is the same.
//!
//! An indirect call the stage does not launch (a count buffer, more than
//! one record, a record not 4-byte aligned) keeps the clamp and runs on the
//! same pipeline made without the amplification stage
//! (`HostPipeline::launch`'s `plain`, made the first time one needs it): the
//! translated mesh shader declares the payload either way
//! (`victory_shader::emit::MESH_PAYLOAD_BYTES`), which the driver takes
//! with or without an amplification shader.

use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::HostList;

#[path = "launch_code.rs"]
mod launch_code;

/// The amplification shader's code (see `launch.hlsl`).
const CODE: &[u8] = launch_code::CODE;

/// The root constant that has the card read the record.
pub(crate) const DIRECT_NONE: u32 = u32::MAX;

/// Where the amplification shader binds: `b0`, `t0`, `u0` in this space.
const SPACE: u32 = victory_shader::emit::LAUNCH_SPACE;

/// Whether mesh pipelines get the amplification stage and indirect mesh
/// records are launched by it (`VICTORY_MESH_LAUNCH` set and not `0`; the
/// translator reads the same switch, as its mesh shaders then declare a
/// payload).
pub(crate) fn on() -> bool {
    victory_shader::emit::mesh_launch()
}

/// Whether a launch waits for earlier shader writes when the list may have
/// some (`VICTORY_MESH_LAUNCH=nowait`: never).
pub(crate) fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MESH_LAUNCH").map_or(true, |v| v.trim() != "nowait"))
}

/// The amplification stage's code.
pub(crate) fn code() -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE { pShaderBytecode: CODE.as_ptr().cast(), BytecodeLength: CODE.len() }
}

/// The bytes of [`code`], for the pipeline library's names.
pub(crate) fn bytes() -> &'static [u8] {
    CODE
}

/// The three root parameters the stage reads, in order: its two constants
/// (the direct group count, the flags), the record (a root SRV), the list's
/// budget (a root UAV). Six root words.
pub(crate) fn params() -> [D3D12_ROOT_PARAMETER1; 3] {
    let descriptor = |kind| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: 0,
                RegisterSpace: SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
    };
    [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: SPACE, Num32BitValues: 2 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        },
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV),
    ]
}

/// The flags constant: bit 0, a record that does not fit takes the rest of
/// the budget (`VICTORY_INDIRECT_REST`, as the clamp).
fn flags() -> u32 {
    static FLAGS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *FLAGS.get_or_init(crate::indirect::rest)
}

/// Mesh draws launched since the last line: records read by the card,
/// dispatches whose count the CPU gave, waits for earlier shader writes
/// before a launch, indirect calls run on the pipeline without the stage.
static LAUNCHED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// What one launch reads.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Launch {
    /// The record at this address on the card.
    Record(u64),
    /// A group count known on the CPU.
    Direct(u32),
}

/// Records the launch of one mesh draw, the mesh pipeline with its stage
/// bound and the stage's parameters at `roots`: the record's address on
/// the card (the card reads it), or the group count the CPU knows. False,
/// with nothing recorded, if the list has no budget for a record.
pub(crate) fn record(list: &HostList, roots: [u32; 3], what: Launch) -> bool {
    record_with(list, roots, what, list.budget())
}

/// [`record`] charging `budget` (a host address) for a record.
pub(crate) fn record_with(list: &HostList, roots: [u32; 3], what: Launch, budget: Option<u64>) -> bool {
    let Ok(l6) = windows::core::Interface::cast::<ID3D12GraphicsCommandList6>(&*list.list) else {
        return false;
    };
    let (direct, record, budget) = match what {
        Launch::Record(address) => {
            let Some(budget) = budget else {
                return false;
            };
            (DIRECT_NONE, address, budget)
        }
        // The stage reads neither the record nor the budget; live
        // addresses all the same where the list has them.
        Launch::Direct(x) => (x, budget.unwrap_or(0), budget.unwrap_or(0)),
    };
    let constants = [direct, flags()];
    // SAFETY: an open list with the mesh pipeline and its root signature
    // set; the record and the budget are live buffers the list keeps.
    unsafe {
        list.list.SetGraphicsRoot32BitConstants(roots[0], 2, constants.as_ptr().cast(), 0);
        list.list.SetGraphicsRootShaderResourceView(roots[1], record);
        list.list.SetGraphicsRootUnorderedAccessView(roots[2], budget);
        l6.DispatchMesh(1, 1, 1);
    }
    LAUNCHED[usize::from(matches!(what, Launch::Direct(_)))].fetch_add(1, Ordering::Relaxed);
    true
}

/// Before the launch of a record in `list`: a wait for every earlier
/// shader write if the list may have recorded work that writes buffers
/// since its last one, or this is its first launch since its reset
/// ([`waits`]).
pub(crate) fn wait_if_written(list: &HostList) {
    /// A key of `HostList::first_use`: the first launch since the reset.
    const LAUNCH_USE: usize = usize::MAX - 0x7b;
    if !waits() {
        return;
    }
    let first = list.first_use(LAUNCH_USE);
    if first || list.list.uav_work() {
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        // SAFETY: an open list; a barrier on every unordered access names no
        // resource. (It clears the list's note of writes.)
        unsafe { list.list.barriers(&[barrier]) };
        LAUNCHED[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// Before a draw on a pipeline's twin without the stage (same root
/// signature): the stage's parameters get values all the same, as the
/// debug layer wants every root parameter set before a draw. Nothing
/// reads them.
pub(crate) fn fill(list: &HostList, roots: [u32; 3]) {
    let address = list.budget().unwrap_or(0);
    let constants = [0u32, 0];
    // SAFETY: an open list with the twin's root signature set; the budget
    // is a live buffer the list keeps (or none).
    unsafe {
        list.list.SetGraphicsRoot32BitConstants(roots[0], 2, constants.as_ptr().cast(), 0);
        list.list.SetGraphicsRootShaderResourceView(roots[1], address);
        list.list.SetGraphicsRootUnorderedAccessView(roots[2], address);
    }
}

/// An indirect mesh call ran on a pipeline's twin without the stage.
pub(crate) fn note_plain() {
    LAUNCHED[3].fetch_add(1, Ordering::Relaxed);
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    let [records, direct, waited, plain] = [0, 1, 2, 3].map(|i| LAUNCHED[i].swap(0, Ordering::Relaxed) as f64 / frames);
    if records + direct + plain > 0.0 {
        tracing::info!(
            "mesh draws launched by the card per frame (VICTORY_MESH_LAUNCH): {records:.0} records read where the title wrote them \
             ({waited:.0} waits for earlier shader writes before them), {direct:.0} counts known on the CPU, {plain:.0} indirect \
             calls clamped and run without the launch stage"
        );
    }
}

#[cfg(test)]
mod tests {
    /// `launch_code.rs` holds `launch.hlsl` compiled (where the Windows
    /// SDK's dxc is: `VICTORY_PC_DXC`, else the newest SDK's).
    #[test]
    fn code_is_the_source_compiled() {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => std::path::PathBuf::from(p),
            None => match victory_shader::emit::validator::Validator::default_path() {
                Some(p) => p.with_file_name("dxc.exe"),
                None => return,
            },
        };
        if !dxc.exists() {
            eprintln!("no PC dxc.exe; skipped");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("launch.dxil");
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/launch.hlsl");
        let status = std::process::Command::new(&dxc)
            .args(["-nologo", "-T", "as_6_6", "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&src)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed on launch.hlsl");
        let made = std::fs::read(out).unwrap();
        assert!(
            made == super::CODE,
            "launch_code.rs is not launch.hlsl compiled by {} ({} bytes against {}): compile it again",
            dxc.display(),
            made.len(),
            super::CODE.len()
        );
    }
}
