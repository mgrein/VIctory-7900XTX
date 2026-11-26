//! VGPR values the same in every lane of a console wave that holds them, for a console wave64
//! compute program run as two host waves of 32 ([`super::lower`]'s `exchange`).
//!
//! A first-lane read (`v_readfirstlane_b32`) of such a value needs nothing from the other host
//! wave: whichever lane of the console wave is first, it holds what this host wave's first lane
//! holds. The console compiler reads wave-uniform data this way inside loops and branches that
//! only some waves of the group run, where the two host waves cannot meet (a group barrier
//! there would wait for threads that never come): the ray-tracing structure builder's
//! group-wide scans add up the totals of the waves below their own in a loop over the wave
//! number, `tid.x >> 6`.
//!
//! A divergence analysis over console waves: a value differs within a wave when it derives from
//! the lane (thread ids, lane counts, cross-lane data, memory) or when lanes of one wave may
//! bring it in over different paths (a join of a branch that differs within the wave; a loop
//! such a branch leaves at different iterations). The rest is the same in every lane of the
//! wave that holds it. Seeds beyond the scalars and constants:
//!
//! - `tid.x >> k` (k ≥ 6, also as a bit field from bit 6 or a mask clearing bits 0-5) when the
//!   group's first dimension is a multiple of 64: a console wave is 64 consecutive threads of
//!   one row (with thread ids packed in v0, the rows' bits lie above bit 9);
//! - an LDS read at an address the same in every lane, after a group barrier from which no LDS
//!   write can be reached: every write is done before the barrier lets any thread on, and none
//!   follows, so the two host waves read what the console wave reads.

use std::collections::HashSet;

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// Whether `inst` may write LDS (a store, an atomic, anything but a read or a lane swap).
fn writes_lds(inst: &Inst) -> bool {
    let Fields::Ds { gds, .. } = inst.fields else { return false };
    let n = inst.name();
    !gds && !(n.starts_with("ds_read") || n.starts_with("ds_swizzle") || n.contains("permute"))
}

/// See the module notes. `threads`: the group's size; `regions`: every per-lane branch with
/// its region (`lower::lane_regions`; `None`, with an indirect exit, gives nothing); `tid`: the
/// value of the thread's index in the group's first dimension on entry.
pub(super) fn wave_same(
    p: &Program,
    st: &Structured,
    threads: [u32; 3],
    regions: Option<&[(usize, Vec<bool>)]>,
    tid: Option<Value>,
) -> HashSet<Value> {
    let Some(regions) = regions else { return HashSet::new() };
    let n = p.blocks.len();
    if st.guards.len() != n || st.exits.len() != n {
        return HashSet::new();
    }
    let rows = threads[0] % 64 == 0;
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        index[b] = i;
    }
    let idom = crate::ir::idoms(&p.blocks, &order, &index);
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if index[b] == usize::MAX || idom[b] == usize::MAX || idom[b] == b {
            return false;
        }
        b = idom[b];
    };
    let reachable_from = |start: &[usize]| {
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = start.to_vec();
        while let Some(b) = stack.pop() {
            if !seen[b] {
                seen[b] = true;
                stack.extend(p.blocks[b].exit.successors());
            }
        }
        seen
    };
    // natural loops, by their back edges
    let mut loops: Vec<Vec<bool>> = Vec::new();
    for l in 0..n {
        for h in p.blocks[l].exit.successors() {
            if index[l] != usize::MAX && index[h] != usize::MAX && index[h] <= index[l] {
                let mut body = vec![false; n];
                body[h] = true;
                let mut stack = vec![l];
                while let Some(b) = stack.pop() {
                    if !body[b] {
                        body[b] = true;
                        stack.extend(p.blocks[b].preds.iter().copied());
                    }
                }
                loops.push(body);
            }
        }
    }
    // group barriers no LDS write can follow
    let mut settled: Vec<(usize, usize)> = Vec::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            if inst.name() != "s_barrier" {
                continue;
            }
            let after = reachable_from(&p.blocks[b].exit.successors());
            let later_here = block.insts[i + 1..].iter().any(writes_lds);
            let later = (0..n).any(|r| after[r] && p.blocks[r].insts.iter().any(writes_lds));
            if !later_here && !later {
                settled.push((b, i));
            }
        }
    }
    let quiet = |b: usize, i: usize| {
        settled.iter().any(|&(sb, si)| if sb == b { si < i } else { dominates(sb, b) })
    };

    // values that may differ between the lanes of one console wave, and lane-mask phis whose
    // bits may
    let mut d: HashSet<Value> = HashSet::new();
    let mut dphi: HashSet<Value> = HashSet::new();
    for &(r, v) in &p.inputs {
        if matches!(r, Reg::V(_)) {
            d.insert(v);
        }
    }
    loop {
        let before = (d.len(), dphi.len());
        let atom_same = |a: &Atom, d: &HashSet<Value>, dphi: &HashSet<Value>| match *a {
            Atom::Uniform(_) => true,
            Atom::Lane(v) => !d.contains(&v),
            Atom::Phi(v) => !dphi.contains(&v),
            _ => false,
        };
        let bool_same =
            |b: Bool, d: &HashSet<Value>, dphi: &HashSet<Value>| st.bools.support(b).iter().all(|a| atom_same(a, d, dphi));
        for (&id, edges) in &st.phi_masks {
            if !dphi.contains(&id) && !edges.iter().all(|&e| bool_same(e, &d, &dphi)) {
                dphi.insert(id);
            }
        }
        // branches that differ within a console wave: the phis where their lanes meet, and every
        // value of a loop holding one (lanes leave it at different iterations)
        for (q, region) in regions {
            let LaneExit::Branch { when, .. } = st.exits[*q] else { continue };
            if st.wide.contains(q) || bool_same(when, &d, &dphi) {
                continue;
            }
            // (lane-mask joins too: each edge's mask, simplified under that edge, can look alike
            // though lanes of one wave come in on different edges)
            for block in &p.blocks {
                let from = block.preds.iter().filter(|&&s| s == *q || region[s]).count();
                if from >= 2 {
                    d.extend(block.phis.iter().filter(|ph| matches!(ph.reg, Reg::V(_))).map(|ph| ph.value));
                    dphi.extend(block.phis.iter().map(|ph| ph.value));
                }
            }
            for body in loops.iter().filter(|body| body[*q]) {
                for (_, block) in p.blocks.iter().enumerate().filter(|(b, _)| body[*b]) {
                    d.extend(block.phis.iter().map(|ph| ph.value));
                    dphi.extend(block.phis.iter().map(|ph| ph.value));
                    for inst in &block.insts {
                        d.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                    }
                }
            }
        }
        for &b in &order {
            let block = &p.blocks[b];
            for ph in &block.phis {
                if matches!(ph.reg, Reg::V(_)) && ph.args.iter().any(|a| d.contains(a)) {
                    d.insert(ph.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let same = match &inst.fields {
                    Fields::Valu(v) => {
                        let name = inst.name();
                        if matches!(name, "v_readfirstlane_b32" | "v_readlane_b32") {
                            continue; // a scalar: one value per wave
                        }
                        let read = |reg: Reg| inst.read(reg);
                        // Every value the instruction reads, not only its named operands: the
                        // accumulator of v_fmac / v_mac, VCC of v_div_fmas, the registers an
                        // M0-relative move may pick, and the old value of a destination it only
                        // partly writes (16-bit halves, SDWA, v_fma_mix, v_swap). The old value
                        // of a destination written whole is left out.
                        let whole = v.sdwa.is_none()
                            && v.opsel & 8 == 0
                            && !["mac", "fma_mix", "swap", "movrel", "16"].iter().any(|k| name.contains(k));
                        let old: Vec<Value> = if whole {
                            inst.writes.iter().filter(|w| matches!(w.0, Reg::V(_))).filter_map(|w| inst.read(w.0)).collect()
                        } else {
                            Vec::new()
                        };
                        let reads_same = inst.reads.iter().all(|&(r, x)| match r {
                            Reg::V(_) => old.contains(&x) || !d.contains(&x),
                            Reg::S(c) if c == crate::ir::EXEC || c == crate::ir::EXEC + 1 => true,
                            Reg::S(_) => {
                                let low = st.highs.get(&x).copied().unwrap_or(x);
                                st.masks.get(&low).is_none_or(|&m| bool_same(m, &d, &dphi))
                            }
                            Reg::Scc => true,
                        });
                        let operand_same = |op: &Operand| match *op {
                            Operand::Vector { reg, dwords } => (0..dwords.max(1))
                                .all(|k| read(Reg::V(reg + k)).is_some_and(|x| !d.contains(&x))),
                            Operand::Scalar { code, .. } if code != NULL => match read(Reg::S(code)) {
                                // a lane mask read per lane: its bits must be alike
                                Some(x) => st.masks.get(&x).is_none_or(|&m| bool_same(m, &d, &dphi))
                                    && !st.highs.contains_key(&x),
                                None => true,
                            },
                            Operand::Attr { .. } => false,
                            _ => true,
                        };
                        let konst = |op: &Operand| match *op {
                            Operand::Const(c) => crate::resource::inline(c),
                            Operand::Literal(k) => Some(k),
                            _ => None,
                        };
                        let is_tid = |op: &Operand| {
                            matches!(*op, Operand::Vector { reg, .. } if tid.is_some() && read(Reg::V(reg)) == tid)
                        };
                        // the wave's number and what lies above it in the thread id
                        let wave_bits = rows
                            && v.dpp.is_none()
                            && v.sdwa.is_none()
                            && st.guards[b][i].is_none()
                            && match (name, &v.src[..]) {
                                ("v_lshrrev_b32", [k, x]) => is_tid(x) && konst(k).is_some_and(|k| (6..32).contains(&k)),
                                ("v_bfe_u32", [x, k, w]) => {
                                    is_tid(x) && konst(k).is_some_and(|k| (6..32).contains(&k)) && operand_same(w)
                                }
                                ("v_and_b32", [a, c]) => {
                                    let clears = |x: &Operand, k: &Operand| is_tid(x) && konst(k).is_some_and(|k| k & 63 == 0);
                                    clears(a, c) || clears(c, a)
                                }
                                _ => false,
                            };
                        if wave_bits {
                            true
                        } else if v.dpp.is_some()
                            || name.starts_with("v_mbcnt")
                            || name.contains("lane")
                            || name.starts_with("v_interp")
                        {
                            false
                        } else {
                            let guard_ok = match st.guards[b][i] {
                                None => true,
                                // lanes outside the guard keep the old value: alike too, and the
                                // guard taken by all lanes of the wave or none
                                Some(g) => {
                                    bool_same(g, &d, &dphi)
                                        && inst.writes.iter().filter(|w| matches!(w.0, Reg::V(_))).all(|w| {
                                            inst.read(w.0).is_some_and(|old| !d.contains(&old))
                                        })
                                }
                            };
                            guard_ok && reads_same && v.src.iter().all(operand_same)
                        }
                    }
                    Fields::Ds { addr, gds: false, .. } if inst.name().starts_with("ds_read") => {
                        st.guards[b][i].is_none()
                            && quiet(b, i)
                            // (a 16-bit read keeps the other half of its destination)
                            && inst.reads.iter().all(|&(r, x)| !matches!(r, Reg::V(_)) || !d.contains(&x))
                            && match *addr {
                                Some(Operand::Vector { reg, .. }) => {
                                    inst.read(Reg::V(reg)).is_some_and(|x| !d.contains(&x))
                                }
                                _ => false,
                            }
                    }
                    _ => false,
                };
                if !same {
                    // its VGPRs, and the lane masks a VALU writes (compares, carries); not EXEC's
                    // old value or SCC
                    let valu = matches!(inst.fields, Fields::Valu(_));
                    d.extend(
                        inst.writes
                            .iter()
                            .filter(|w| matches!(w.0, Reg::V(_)) || (valu && w.0 != Reg::Scc))
                            .map(|w| w.1),
                    );
                }
            }
        }
        if (d.len(), dphi.len()) == before {
            break;
        }
    }
    let mut same = HashSet::new();
    for block in &p.blocks {
        same.extend(block.phis.iter().filter(|ph| matches!(ph.reg, Reg::V(_))).map(|ph| ph.value));
        for inst in &block.insts {
            same.extend(inst.writes.iter().filter(|w| matches!(w.0, Reg::V(_))).map(|w| w.1));
        }
    }
    same.retain(|v| !d.contains(v));
    same
}
