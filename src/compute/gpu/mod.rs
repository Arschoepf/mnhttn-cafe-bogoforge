use anyhow::{Context, Result};
use cust::prelude::*;
use log::debug;

use super::ComputeBackend;
use crate::net::types::RangeResult;

static PTX: &str = include_str!(env!("KERNEL_PTX_PATH"));

pub struct GpuBackend {
    _ctx: cust::context::Context,
    module: Module,
    stream: Stream,
    blocks: u32,
    threads_per_block: u32,
    dev_best: DeviceBuffer<u64>,
    dev_arrays: DeviceBuffer<u8>,
    dev_indices: DeviceBuffer<u64>,
    host_arrays: Vec<u8>,
    host_indices: Vec<u64>,
}

unsafe impl Send for GpuBackend {}

impl GpuBackend {
    pub fn new(blocks: u32, threads_per_block: u32) -> Result<Self> {
        anyhow::ensure!(
            threads_per_block == 256,
            "kernel.cu hardcodes 256 threads/block (column-major smem layout); \
             set cuda_threads_per_block = 256 in config.toml"
        );

        let ctx = cust::quick_init().context("failed to initialize CUDA")?;

        let module = Module::from_ptx(PTX, &[])
            .context("failed to load PTX — check CUDA_ARCH matches your GPU")?;

        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("failed to create CUDA stream")?;

        let n_blocks = blocks as usize;

        let dev_best = DeviceBuffer::zeroed(1).context("alloc dev_best")?;
        let dev_arrays = DeviceBuffer::zeroed(n_blocks * 25).context("alloc dev_arrays")?;
        let dev_indices = DeviceBuffer::zeroed(n_blocks).context("alloc dev_indices")?;

        let host_arrays = vec![0u8; n_blocks * 25];
        let host_indices = vec![0u64; n_blocks];

        Ok(Self {
            _ctx: ctx,
            module,
            stream,
            blocks,
            threads_per_block,
            dev_best,
            dev_arrays,
            dev_indices,
            host_arrays,
            host_indices,
        })
    }
}

impl ComputeBackend for GpuBackend {
    fn compute_range(&mut self, base_seed: u64, lo: u64, hi: u64) -> RangeResult {
        self.dev_best.copy_from(&[0u64]).expect("H2D reset failed");

        let kernel = self
            .module
            .get_function("bogo_range_kernel")
            .expect("bogo_range_kernel not found in PTX");

        let stream = &self.stream;

        debug!(
            "[cuda] launching kernel base_seed={} lo={} hi={} blocks={} threads={}",
            base_seed,
            lo,
            hi,
            self.blocks,
            self.threads_per_block,
        );
        unsafe {
            launch!(
                kernel<<<self.blocks, self.threads_per_block, 0, stream>>>(
                    base_seed,
                    lo,
                    hi,
                    self.dev_best.as_device_ptr(),
                    self.dev_arrays.as_device_ptr(),
                    self.dev_indices.as_device_ptr(),
                )
            )
            .expect("kernel launch failed");
        }

        self.stream.synchronize().expect("stream sync failed");
        debug!("[cuda] kernel complete, reading back results");

        let mut host_best = [0u64; 1];
        self.dev_best
            .copy_to(&mut host_best)
            .expect("D2H best failed");

        if host_best[0] == 0 {
            return RangeResult {
                lo,
                hi,
                best_correct: -1,
                best_arr: [0u8; 25],
                best_index: lo,
            };
        }

        let best_score = (host_best[0] >> 32) as i32;
        let best_bid = (host_best[0] & 0xffff_ffff) as usize;

        self.dev_arrays
            .copy_to(&mut self.host_arrays)
            .expect("D2H arrays failed");
        self.dev_indices
            .copy_to(&mut self.host_indices)
            .expect("D2H indices failed");

        let arr_start = best_bid * 25;
        let mut best_arr = [0u8; 25];
        best_arr.copy_from_slice(&self.host_arrays[arr_start..arr_start + 25]);

        RangeResult {
            lo,
            hi,
            best_correct: best_score,
            best_arr,
            best_index: self.host_indices[best_bid],
        }
    }
}
