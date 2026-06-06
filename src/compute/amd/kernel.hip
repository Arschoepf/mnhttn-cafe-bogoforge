#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t *z) {
    *z += 0x9e3779b97f4a7c15ULL;
    uint64_t v = *z;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

__device__ __forceinline__ void xseed(uint64_t seed, uint32_t s[4]) {
    uint64_t z = seed;
    uint64_t a = splitmix64(&z);
    uint64_t b = splitmix64(&z);
    s[0] = (uint32_t)(a & 0xffffffffULL);
    s[1] = (uint32_t)(a >> 32);
    s[2] = (uint32_t)(b & 0xffffffffULL);
    s[3] = (uint32_t)(b >> 32);
    if (!s[0] && !s[1] && !s[2] && !s[3]) s[0] = 1;
}

__device__ __forceinline__ uint32_t xnext(uint32_t s[4]) {
    const uint32_t result = rotl32(s[0] + s[3], 7) + s[0];
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11);
    return result;
}

__device__ __forceinline__ uint32_t xint_c(uint32_t s[4], const uint32_t max) {
    const uint32_t threshold = (uint32_t)(0x100000000ULL % (uint64_t)max);
    uint32_t x;
    do { x = xnext(s); } while (x < threshold);
    return x % max;
}

__device__ __forceinline__ int hip_shfl_xor_i32(int v, int lane_mask) {
    return __shfl_xor(v, lane_mask, 32);
}

__device__ __forceinline__ uint32_t hip_shfl_xor_u32(uint32_t v, int lane_mask) {
    return (uint32_t)__shfl_xor((unsigned int)v, lane_mask, 32);
}

__device__ __forceinline__ unsigned long long hip_shfl_xor_u64(unsigned long long v, int lane_mask) {
    uint32_t lo = (uint32_t)(v & 0xffffffffULL);
    uint32_t hi = (uint32_t)(v >> 32);

    lo = hip_shfl_xor_u32(lo, lane_mask);
    hi = hip_shfl_xor_u32(hi, lane_mask);

    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}

__device__ __forceinline__ unsigned int hip_shfl_u32(unsigned int v, int src_lane) {
    return __shfl(v, src_lane, 32);
}


__device__ __forceinline__ void warp_reduce(
    int *score, unsigned long long *index, uint32_t *lane_out
) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        int                ps = hip_shfl_xor_i32(*score, d);
        unsigned long long pi = hip_shfl_xor_u64(*index, d);
        uint32_t           pl = hip_shfl_xor_u32(*lane_out, d);
        if (ps > *score || (ps == *score && pl < *lane_out)) {
            *score    = ps;
            *index    = pi;
            *lane_out = pl;
        }
    }
}


extern "C" __global__
__launch_bounds__(256, 2)
void bogo_range_kernel(
    uint64_t            base_seed,
    uint64_t            lo,
    uint64_t            hi,
    unsigned long long *dev_best_and_bid,
    uint8_t            *dev_all_arrays,
    uint64_t           *dev_all_indices
) {
    __shared__ unsigned int lds[25 * 256];
    unsigned int *arr = lds + threadIdx.x;  // arr[p] = lds[p * 256 + threadIdx.x]

    const uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint32_t lane   = threadIdx.x & 31;
    const uint32_t warpid = threadIdx.x >> 5;

    int      best_score = -1;
    uint8_t  best_arr[25];
    uint64_t best_idx   = lo;

    for (uint64_t iter = lo + tid; iter < hi; iter += stride) {
        uint32_t s[4];
        xseed(base_seed + iter * 0x9e3779b97f4a7c15ULL, s);

        #pragma unroll
        for (int p = 0; p < 25; p++) arr[p * 256] = (unsigned int)(p + 1);

        #pragma unroll 24
        for (int i = 24; i > 0; i--) {
            const uint32_t j = xint_c(s, (uint32_t)(i + 1));
            const unsigned int tmp = arr[i * 256];
            arr[i * 256] = arr[j * 256];
            arr[j * 256] = tmp;
        }

        int c = 0;
        #pragma unroll
        for (int p = 0; p < 25; p++) c += ((int)arr[p * 256] == p + 1);

        if (c > best_score) {
            best_score = c;
            best_idx   = iter;
            #pragma unroll
            for (int p = 0; p < 25; p++) best_arr[p] = (uint8_t)arr[p * 256];
            if (c == 25) break;
        }
    }

    __syncthreads();

    int                *smem_scores  = (int *)lds;
    unsigned long long *smem_indices = (unsigned long long *)(smem_scores  + 8);
    uint8_t            *smem_arrays  = (uint8_t *)           (smem_indices + 8);

    int                ws = best_score;
    unsigned long long wi = (unsigned long long)best_idx;
    uint32_t           wl = lane;
    warp_reduce(&ws, &wi, &wl);

    uint8_t warp_arr[25];
    #pragma unroll
    for (int p = 0; p < 25; p++) {
        unsigned v = hip_shfl_u32((unsigned)best_arr[p], wl);
        if (lane == 0) warp_arr[p] = (uint8_t)v;
    }

    if (lane == 0) {
        smem_scores [warpid] = ws;
        smem_indices[warpid] = wi;
        #pragma unroll
        for (int p = 0; p < 25; p++) smem_arrays[warpid * 25 + p] = warp_arr[p];
    }
    
    __syncthreads();

    if (warpid != 0) return;

    const uint32_t n_warps = blockDim.x >> 5;
    int                bs = (lane < n_warps) ? smem_scores [lane] : -1;
    unsigned long long bi = (lane < n_warps) ? smem_indices[lane] : 0ULL;
    uint32_t           bw = lane;
    warp_reduce(&bs, &bi, &bw);

    if (lane == 0 && bs >= 0) {
        const uint32_t blk = blockIdx.x;
        dev_all_indices[blk] = (uint64_t)bi;
        #pragma unroll
        for (int p = 0; p < 25; p++)
            dev_all_arrays[(uint64_t)blk * 25 + p] = smem_arrays[bw * 25 + p];
        atomicMax(dev_best_and_bid,
            ((unsigned long long)(uint32_t)bs << 32) | (unsigned long long)blk);
    }
}
