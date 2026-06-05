use orx_parallel::*;

use super::ComputeBackend;
use crate::net::types::RangeResult;

const IDENTITY: [u8; 25] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
];

struct LocalBest {
    score: i32,
    arr: [u8; 25],
    index: u64,
}

pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    /// `threads` = 0 means use all logical cores.
    pub fn new(threads: usize) -> Self {
        Self { num_threads: threads }
    }
}

impl ComputeBackend for CpuBackend {
    fn compute_range(&mut self, base_seed: u64, lo: u64, hi: u64) -> RangeResult {
        let threads = self.num_threads;
        let result = (lo as usize..hi as usize)
            .par()
            .num_threads(threads)
            .map(|i| {
                let i = i as u64;
                let seed_i = base_seed.wrapping_add(i.wrapping_mul(0x9e3779b97f4a7c15));
                let (score, arr) = one_shuffle(seed_i);
                LocalBest { score, arr, index: i }
            })
            .reduce(|a, b| if a.score >= b.score { a } else { b })
            .unwrap_or(LocalBest { score: -1, arr: [0u8; 25], index: lo });

        RangeResult {
            lo,
            hi,
            best_correct: result.score,
            best_arr: result.arr,
            best_index: result.index,
        }
    }
}

#[inline(always)]
fn splitmix64(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9e3779b97f4a7c15);
    let mut v = *z;
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    v ^ (v >> 31)
}

#[inline(always)]
fn xseed(seed: u64) -> [u32; 4] {
    let mut z = seed;
    let a = splitmix64(&mut z);
    let b = splitmix64(&mut z);
    let mut s = [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32];
    if s == [0, 0, 0, 0] { s[0] = 1; }
    s
}

#[inline(always)]
fn xnext(s: &mut [u32; 4]) -> u32 {
    let result = s[0].wrapping_add(s[3]).rotate_left(7).wrapping_add(s[0]);
    let t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = s[3].rotate_left(11);
    result
}

// MAX is a const generic so LLVM sees constant divisors and replaces both %
// operations with multiply+shift (~3 cycles vs ~30 for integer division).
#[inline(always)]
fn xint_const<const MAX: u32>(s: &mut [u32; 4]) -> u32 {
    let THR: u32 = (0x100000000u64 % MAX as u64) as u32;
    loop {
        let x = xnext(s);
        if x >= THR { return x % MAX; }
    }
}

macro_rules! fy_step {
    ($s:expr, $arr:expr, $i:literal) => {{
        let j = xint_const::<{ $i + 1 }>(&mut $s) as usize;
        $arr.swap($i, j);
    }};
}

#[inline(always)]
fn one_shuffle(seed: u64) -> (i32, [u8; 25]) {
    let mut s = xseed(seed);
    let mut arr = IDENTITY;

    fy_step!(s, arr, 24);
    fy_step!(s, arr, 23);
    fy_step!(s, arr, 22);
    fy_step!(s, arr, 21);
    fy_step!(s, arr, 20);
    fy_step!(s, arr, 19);
    fy_step!(s, arr, 18);
    fy_step!(s, arr, 17);
    fy_step!(s, arr, 16);
    fy_step!(s, arr, 15);
    fy_step!(s, arr, 14);
    fy_step!(s, arr, 13);
    fy_step!(s, arr, 12);
    fy_step!(s, arr, 11);
    fy_step!(s, arr, 10);
    fy_step!(s, arr,  9);
    fy_step!(s, arr,  8);
    fy_step!(s, arr,  7);
    fy_step!(s, arr,  6);
    fy_step!(s, arr,  5);
    fy_step!(s, arr,  4);
    fy_step!(s, arr,  3);
    fy_step!(s, arr,  2);
    fy_step!(s, arr,  1);

    let correct = arr.iter().zip(1u8..).map(|(&a, b)| (a == b) as i32).sum();
    (correct, arr)
}
