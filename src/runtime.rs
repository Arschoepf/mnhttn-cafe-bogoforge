use std::sync::Arc;
use std::time::Duration;

use anyhow::bail;
use tokio::runtime::{self, Runtime};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[cfg(feature = "cuda")]
use crate::compute::gpu::GpuBackend;
#[cfg(feature = "hip")]
use crate::compute::amd::AmdBackend;
use crate::compute::{cpu::CpuBackend, run_compute_worker};
use crate::config::Config;
use crate::metrics::Metrics;
use crate::net::{
    types::{Chunk, Lease, RangeResult, Report},
    NetClient,
};
use crate::scheduler::{BackendHandle, Scheduler};
use crate::tui;

pub struct ForgeRuntime {
    runtime: Runtime,
    config: Arc<Config>,
}

impl ForgeRuntime {
    pub fn new(config: Config) -> Self {
        ForgeRuntime {
            runtime: runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Could not build threaded runtime."),
            config: Arc::new(config),
        }
    }

    pub fn startup(&self) -> anyhow::Result<()> {
        if !self.config.compute.use_cpu && !self.config.compute.use_gpu && !self.config.compute.use_amd {
            bail!("no compute backend enabled; set use_cpu, use_gpu, or use_amd to true in config.toml");
        }

        let metrics = Metrics::new();
        let cancel = CancellationToken::new();

        let tui_metrics = Arc::clone(&metrics);
        let tui_cancel = cancel.clone();
        let tui_handle = std::thread::spawn(move || {
            tui::run(tui_metrics, tui_cancel);
        });

        let result = self.runtime.block_on(async {
            let (lease_tx, lease_rx) = mpsc::channel::<Lease>(4);
            let (report_tx, report_rx) = mpsc::channel::<Report>(16);
            // All backends write tagged results into one shared done channel.
            let (done_tx, done_rx) = mpsc::channel::<(usize, RangeResult)>(16);

            let mut backends: Vec<BackendHandle> = Vec::new();

            if self.config.compute.use_cpu {
                let id = backends.len();
                let (work_tx, work_rx) = mpsc::channel::<Chunk>(2);
                let cpu_done_tx = done_tx.clone();
                let cpu_threads = self.config.compute.cpu_threads;
                tokio::task::spawn_blocking(move || {
                    run_compute_worker(id, CpuBackend::new(cpu_threads), work_rx, cpu_done_tx);
                });
                backends.push(BackendHandle::new(
                    self.config.compute.cpu_chunk_size,
                    work_tx,
                ));
            }

            #[cfg(feature = "cuda")]
            if self.config.compute.use_gpu {
                let id = backends.len();
                let (work_tx, work_rx) = mpsc::channel::<Chunk>(2);
                let gpu_done_tx = done_tx.clone();
                let blocks = self.config.compute.cuda_blocks;
                let tpb = self.config.compute.cuda_threads_per_block;
                let chunk_size = self.config.compute.gpu_chunk_size;
                let metrics_gpu = Arc::clone(&metrics);

                // Use a oneshot to surface init errors into the TUI before
                // proceeding. The worker stays on the same OS thread as new()
                // so the CUDA context remains valid.
                let (init_tx, init_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

                tokio::task::spawn_blocking(move || match GpuBackend::new(blocks, tpb) {
                    Ok(backend) => {
                        let _ = init_tx.send(Ok(()));
                        run_compute_worker(id, backend, work_rx, gpu_done_tx);
                    }
                    Err(e) => {
                        let msg = format!("gpu error: {e:#}");
                        metrics_gpu.set_status(msg.clone());
                        let _ = init_tx.send(Err(msg));
                    }
                });

                match tokio::time::timeout(Duration::from_secs(15), init_rx).await {
                    Ok(Ok(Ok(()))) => {
                        backends.push(BackendHandle::new(chunk_size, work_tx));
                    }
                    Ok(Ok(Err(e))) => {
                        // Error already written to TUI status. Drop work_tx so
                        // the scheduler never tries to dispatch to this backend.
                        eprintln!("[gpu] {e}");
                    }
                    _ => {
                        metrics.set_status("gpu init timed out");
                    }
                }
            }

            #[cfg(feature = "hip")]
            if self.config.compute.use_amd {
                let id = backends.len();
                let (work_tx, work_rx) = mpsc::channel::<Chunk>(2);
                let amd_done_tx = done_tx.clone();
                let blocks = self.config.compute.amd_blocks;
                let tpb = self.config.compute.amd_threads_per_block;
                let chunk_size = self.config.compute.amd_chunk_size;
                let metrics_amd = Arc::clone(&metrics);

                let (init_tx, init_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

                tokio::task::spawn_blocking(move || match AmdBackend::new(blocks, tpb) {
                    Ok(backend) => {
                        let _ = init_tx.send(Ok(()));
                        run_compute_worker(id, backend, work_rx, amd_done_tx);
                    }
                    Err(e) => {
                        let msg = format!("amd error: {e:#}");
                        metrics_amd.set_status(msg.clone());
                        let _ = init_tx.send(Err(msg));
                    }
                });

                match tokio::time::timeout(Duration::from_secs(15), init_rx).await {
                    Ok(Ok(Ok(()))) => {
                        backends.push(BackendHandle::new(chunk_size, work_tx));
                    }
                    Ok(Ok(Err(e))) => {
                        eprintln!("[amd] {e}");
                    }
                    _ => {
                        metrics.set_status("amd init timed out");
                    }
                }
            }

            // Drop the original sender so done_rx closes when all workers exit.
            drop(done_tx);

            let net = NetClient::new(
                Arc::clone(&self.config),
                Arc::clone(&metrics),
                lease_tx,
                report_rx,
            );
            let scheduler = Scheduler::new(
                Arc::clone(&self.config),
                Arc::clone(&metrics),
                lease_rx,
                report_tx,
                backends,
                done_rx,
            );

            let net_handle = tokio::spawn(net.run(cancel.clone()));
            let sched_handle = tokio::spawn(scheduler.run(cancel.clone()));

            tokio::select! {
                res = net_handle => {
                    if let Ok(Err(e)) = res { eprintln!("[net] exited with error: {e}"); }
                }
                res = sched_handle => {
                    if let Ok(Err(e)) = res { eprintln!("[scheduler] exited with error: {e}"); }
                }
                _ = cancel.cancelled() => {}
            }

            cancel.cancel();
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(())
        });

        let _ = tui_handle.join();
        result
    }
}
