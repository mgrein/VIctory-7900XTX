// The launch of one mesh draw by the card (`launch.rs`): an amplification
// shader of ours in front of every translated mesh shader
// (`VICTORY_MESH_LAUNCH`). It reads the title's record where the title's
// passes wrote it, bounds it as the clamp of `indirect.rs` bounds a mesh
// record, charges the list's budget the same way, and launches the mesh
// groups. No clamp pass of ours runs between the draws.
//
// Compiled with the Windows SDK's dxc (signed by its dxil.dll) and kept as
// bytes in `launch_code.rs`, which says how; a test in `pipeline::host`
// runs those bytes on the card and checks they do what this source says.

// The payload every translated mesh shader declares with the launch on and
// never reads (`victory_shader::emit::MESH_PAYLOAD_BYTES`, 4 bytes).
struct Payload {
    uint unused;
};
groupshared Payload payload;

// `direct`: the group count of a mesh dispatch the CPU knows (the title's
// DispatchMesh), or 0xffffffff for a record the card reads at `record`.
// `flags` bit 0: a record that does not fit in what is left of the budget
// takes the rest (`VICTORY_INDIRECT_REST`, as the clamp).
cbuffer Launch : register(b0, space9) {
    uint direct;
    uint flags;
};
// The record: three words, x, y, z (the console's mesh dispatches are
// one-dimensional).
ByteAddressBuffer record : register(t0, space9);
// The list's budget words, as the clamp keeps them: the work so far; the
// largest group count a record asked for; the groups asked in all; the
// records refused; the largest cost.
RWByteAddressBuffer budget : register(u0, space9);

static const uint BUDGET = 1u << 27;
static const uint DIRECT_NONE = 0xffffffffu;

[numthreads(1, 1, 1)]
void main() {
    uint3 g = uint3(0, 0, 0);
    if (direct != DIRECT_NONE) {
        // Known on the CPU, as before: no budget, the host's limit only.
        g = uint3(min(direct, 65535u), 1u, 1u);
    } else {
        g = record.Load3(0);
        // A record over the most a dispatch runs is a slot the title has
        // not filled: empty, and not counted below either.
        if (g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        uint asked = g.x;
        g = uint3(g.x, min(g.y, 1u), min(g.z, 1u));
        uint cost = g.x * g.y * g.z * 128u;
        bool rest = (flags & 1u) != 0;
        if (cost > 0) {
            uint before;
            budget.InterlockedOr(0, 0, before);
            [allow_uav_condition]
            while (before < BUDGET) {
                uint next = (BUDGET - before < cost) ? (rest ? BUDGET : before) : (before + cost);
                if (next == before) break;
                uint prev;
                budget.InterlockedCompareExchange(0, before, next, prev);
                if (prev == before) break;
                before = prev;
            }
            uint told;
            if (before >= BUDGET || BUDGET - before < cost) {
                g = uint3(0, 0, 0);
                budget.InterlockedAdd(12, 1, told);
            }
            if (cost >= (1u << 22)) {
                budget.InterlockedMax(16, cost, told);
            }
        }
        if (asked > 0) {
            uint was;
            budget.InterlockedMax(4, asked, was);
            budget.InterlockedAdd(8, min(asked, 1u << 24), was);
        }
    }
    payload.unused = 0;
    DispatchMesh(g.x, g.y, g.z, payload);
}
