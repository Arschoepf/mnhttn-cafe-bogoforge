
// Requires wgpu::Features::SHADER_INT64.
//
// u64 is used for seeds and indices. Workgroup shared memory and storage
// buffers use u32 pairs (lo/hi) to avoid alignment edge cases.

struct Params {
    base_seed_lo: u32, base_seed_hi: u32,
    lo_lo:        u32, lo_hi:        u32,
    hi_lo:        u32, hi_hi:        u32,
}
@group(0) @binding(0) var<storage, read>       params:  Params;

struct BlockResult {
    score:  i32,
    _pad:   u32,
    idx_lo: u32,
    idx_hi: u32,
    arr:    array<u32, 25>,
    _pad2:  array<u32, 3>,
}
@group(0) @binding(1) var<storage, read_write> results: array<BlockResult>;

// lds: column-major Fisher-Yates array. arr[p] for thread lid = lds[p*256+lid].
// After the main loop, lds is repurposed to hold each thread's best_arr so
// thread 0 can read the winner's array after score reduction.

var<workgroup> lds:       array<u32, 6400u>; // 25 × 256
var<workgroup> red_score: array<i32,  256u>;
var<workgroup> red_lid:   array<u32,  256u>;
var<workgroup> red_lo:    array<u32,  256u>;
var<workgroup> red_hi:    array<u32,  256u>;


fn rotl32(x: u32, k: u32) -> u32 { return (x << k) | (x >> (32u - k)); }
fn mk64(lo: u32, hi: u32) -> u64 { return u64(lo) | (u64(hi) << 32u); }

// Native u64 splitmix64 — no manual carry arithmetic needed.
fn sm64_step(z: u64) -> vec2<u64> { // returns (new_z, output)
    let GOLDEN: u64 = mk64(0x7f4a7c15u, 0x9e3779b9u);
    let MUL1:   u64 = mk64(0x1ce4e5b9u, 0xbf58476du);
    let MUL2:   u64 = mk64(0x133111ebu, 0x94d049bbu);

    let nz = z + GOLDEN;
    var v = nz;
    v = (v ^ (v >> 30u)) * MUL1;
    v = (v ^ (v >> 27u)) * MUL2;
    v = v ^ (v >> 31u);
    return vec2<u64>(nz, v);
}

struct Prng { s: array<u32, 4> }

fn xseed(seed: u64) -> Prng {
    let r1 = sm64_step(seed);
    let r2 = sm64_step(r1.x);
    var p: Prng;
    p.s[0] = u32(r1.y        & u64(0xffffffffu));
    p.s[1] = u32(r1.y >> 32u & u64(0xffffffffu));
    p.s[2] = u32(r2.y        & u64(0xffffffffu));
    p.s[3] = u32(r2.y >> 32u & u64(0xffffffffu));
    if p.s[0] == 0u && p.s[1] == 0u && p.s[2] == 0u && p.s[3] == 0u { p.s[0] = 1u; }
    return p;
}

fn xnext(p: ptr<function, Prng>) -> u32 {
    let r = rotl32((*p).s[0] + (*p).s[3], 7u) + (*p).s[0];
    let t = (*p).s[1] << 9u;
    (*p).s[2] ^= (*p).s[0]; (*p).s[3] ^= (*p).s[1];
    (*p).s[1] ^= (*p).s[2]; (*p).s[0] ^= (*p).s[3];
    (*p).s[2] ^= t;
    (*p).s[3] = rotl32((*p).s[3], 11u);
    return r;
}

fn xint(p: ptr<function, Prng>, max: u32) -> u32 {
    let thr = u32((u64(1u) << 32u) % u64(max));
    loop {
        let x = xnext(p);
        if x >= thr { return x % max; }
    }
    return 0u;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id)           wid: vec3<u32>,
    @builtin(num_workgroups)         nwg: vec3<u32>,
) {
    let GOLDEN: u64 = mk64(0x7f4a7c15u, 0x9e3779b9u);

    let base_seed = mk64(params.base_seed_lo, params.base_seed_hi);
    let lo        = mk64(params.lo_lo,        params.lo_hi);
    let hi        = mk64(params.hi_lo,        params.hi_hi);
    let stride    = u64(nwg.x) * u64(256u);

    var best_score:  i32 = -1;
    var best_idx:    u64 = lo;
    var best_arr:    array<u32, 25>;

    var iter = lo + u64(gid.x);
    while iter < hi {
        let seed_i = base_seed + iter * GOLDEN;
        var rng = xseed(seed_i);

        for (var q = 0u; q < 25u; q++) { lds[q * 256u + lid] = q + 1u; }

        for (var i = 24u; i > 0u; i--) {
            let j   = xint(&rng, i + 1u);
            let tmp = lds[i * 256u + lid];
            lds[i * 256u + lid] = lds[j * 256u + lid];
            lds[j * 256u + lid] = tmp;
        }

        var c: i32 = 0;
        for (var q = 0u; q < 25u; q++) {
            if lds[q * 256u + lid] == q + 1u { c += 1; }
        }

        if c > best_score {
            best_score = c;
            best_idx   = iter;
            for (var q = 0u; q < 25u; q++) { best_arr[q] = lds[q * 256u + lid]; }
            if c == 25 { break; }
        }

        iter += stride;
    }

    workgroupBarrier();
    for (var q = 0u; q < 25u; q++) { lds[q * 256u + lid] = best_arr[q]; }

    workgroupBarrier();
    red_score[lid] = best_score;
    red_lid[lid]   = lid;
    red_lo[lid]    = u32(best_idx        & u64(0xffffffffu));
    red_hi[lid]    = u32(best_idx >> 32u & u64(0xffffffffu));
    workgroupBarrier();

    var step = 128u;
    while step > 0u {
        if lid < step {
            let o = lid + step;
            let a_idx = mk64(red_lo[lid], red_hi[lid]);
            let b_idx = mk64(red_lo[o],   red_hi[o]);
            if red_score[o] > red_score[lid] ||
               (red_score[o] == red_score[lid] && b_idx < a_idx) {
                red_score[lid] = red_score[o];
                red_lid[lid]   = red_lid[o];
                red_lo[lid]    = red_lo[o];
                red_hi[lid]    = red_hi[o];
            }
        }
        step >>= 1u;
        workgroupBarrier();
    }

    if lid == 0u {
        let wl = red_lid[0u];
        results[wid.x].score   = red_score[0u];
        results[wid.x]._pad    = 0u;
        results[wid.x].idx_lo  = red_lo[0u];
        results[wid.x].idx_hi  = red_hi[0u];
        for (var q = 0u; q < 25u; q++) {
            results[wid.x].arr[q] = lds[q * 256u + wl];
        }
    }
}
