pub mod cpu;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "hip")]
pub mod amd;

use tokio::sync::mpsc;

use crate::net::types::{Chunk, RangeResult};

pub trait ComputeBackend: Send + 'static {
    fn compute_range(&mut self, seed: u64, lo: u64, hi: u64) -> RangeResult;
}

/// Blocking loop that drives any ComputeBackend.
/// Tags every result with `id` so the scheduler can route the next chunk back
/// to the correct backend.
pub fn run_compute_worker(
    id: usize,
    mut backend: impl ComputeBackend,
    mut work_rx: mpsc::Receiver<Chunk>,
    done_tx: mpsc::Sender<(usize, RangeResult)>,
) {
    while let Some(chunk) = work_rx.blocking_recv() {
        let result = backend.compute_range(chunk.seed, chunk.lo, chunk.hi);
        if done_tx.blocking_send((id, result)).is_err() {
            break;
        }
    }
}
