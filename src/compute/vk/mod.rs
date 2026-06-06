use anyhow::{Context, Result};
use log::{ info, debug };
use pollster::FutureExt as _;
use wgpu::util::DeviceExt;

use crate::net::types::RangeResult;
use super::ComputeBackend;

const SHADER: &str = include_str!("kernel.wgsl");

// Per-workgroup result layout must match BlockResult in the WGSL shader (128 bytes).
// score(4) + _pad(4) + idx_lo(4) + idx_hi(4) + arr(100) + _pad2(12) = 128
const BLOCK_RESULT_BYTES: usize = 128;

pub struct VkBackend {
    device:     wgpu::Device,
    queue:      wgpu::Queue,
    pipeline:   wgpu::ComputePipeline,
    bgl:        wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    results_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    blocks:     u32,
}

impl VkBackend {
    pub fn new(blocks: u32, threads_per_block: u32) -> Result<Self> {
        anyhow::ensure!(
            threads_per_block == 256,
            "Vulkan kernel is compiled for workgroup_size(256); set gpu_threads_per_block = 256"
        );

        let (device, queue) = async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .context("no Vulkan-capable GPU found")?;

            let info = adapter.get_info();
            info!("[vk] using: {} ({:?})", info.name, info.backend);

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("bogoforge"),
                        required_features: wgpu::Features::SHADER_INT64,
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .context("failed to create wgpu device (GPU may not support SHADER_INT64)")?;

            anyhow::Ok((device, queue))
        }.block_on()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bogo_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bogo_bgl"),
            entries: &[
                // binding 0: params (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: results (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bogo_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bogo_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let n = blocks as usize;

        // params: 6 × u32 = 24 bytes
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 24,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // results: one BlockResult per workgroup
        let results_size = (n * BLOCK_RESULT_BYTES) as u64;
        let results_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("results"),
            size: results_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: results_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self { device, queue, pipeline, bgl, params_buf, results_buf, staging_buf, blocks })
    }
}

impl ComputeBackend for VkBackend {
    fn compute_range(&mut self, base_seed: u64, lo: u64, hi: u64) -> RangeResult {
        // Pack params as 6 × u32 (split u64s into lo/hi pairs).
        let params_data: [u32; 6] = [
            base_seed as u32, (base_seed >> 32) as u32,
            lo        as u32, (lo        >> 32) as u32,
            hi        as u32, (hi        >> 32) as u32,
        ];
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.results_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.blocks, 1, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.results_buf, 0,
            &self.staging_buf, 0,
            (self.blocks as usize * BLOCK_RESULT_BYTES) as u64,
        );

        debug!("[vk] dispatching workgroups blocks={} lo={} hi={}", self.blocks, lo, hi);

        self.queue.submit(Some(enc.finish()));

        // Wait for GPU and map the staging buffer.
        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map_async channel").expect("map failed");

        let data = slice.get_mapped_range();
        let result = find_winner(&data, self.blocks as usize, lo, hi);
        drop(data);
        self.staging_buf.unmap();

        result
    }
}

fn find_winner(data: &[u8], blocks: usize, lo: u64, hi: u64) -> RangeResult {
    let mut best_score: i32 = -1;
    let mut best_arr = [0u8; 25];
    let mut best_index = lo;

    for b in 0..blocks {
        let base = b * BLOCK_RESULT_BYTES;
        let score = i32::from_le_bytes(data[base..base + 4].try_into().unwrap());
        if score <= best_score { continue; }

        let idx_lo = u32::from_le_bytes(data[base + 8..base + 12].try_into().unwrap());
        let idx_hi = u32::from_le_bytes(data[base + 12..base + 16].try_into().unwrap());
        let index = (idx_lo as u64) | ((idx_hi as u64) << 32);

        // arr: 25 × u32 at offset 16
        let arr_start = base + 16;
        let mut arr = [0u8; 25];
        for i in 0..25 {
            arr[i] = u32::from_le_bytes(data[arr_start + i * 4..arr_start + i * 4 + 4].try_into().unwrap()) as u8;
        }

        best_score = score;
        best_index = index;
        best_arr   = arr;
    }

    RangeResult { lo, hi, best_correct: best_score, best_arr, best_index }
}
