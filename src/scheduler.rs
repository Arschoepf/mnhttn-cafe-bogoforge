use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::metrics::SharedMetrics;
use crate::net::types::{Chunk, Lease, RangeResult, Report};

/// Owned by the scheduler; one per active backend.
pub struct BackendHandle {
    pub chunk_size: u64,
    pub work_tx: mpsc::Sender<Chunk>,
    in_flight: bool,
}

impl BackendHandle {
    pub fn new(chunk_size: u64, work_tx: mpsc::Sender<Chunk>) -> Self {
        Self {
            chunk_size,
            work_tx,
            in_flight: false,
        }
    }
}

pub struct Scheduler {
    config: Arc<Config>,
    metrics: SharedMetrics,
    lease_rx: mpsc::Receiver<Lease>,
    report_tx: mpsc::Sender<Report>,
    backends: Vec<BackendHandle>,
    done_rx: mpsc::Receiver<(usize, RangeResult)>,
}

impl Scheduler {
    pub fn new(
        config: Arc<Config>,
        metrics: SharedMetrics,
        lease_rx: mpsc::Receiver<Lease>,
        report_tx: mpsc::Sender<Report>,
        backends: Vec<BackendHandle>,
        done_rx: mpsc::Receiver<(usize, RangeResult)>,
    ) -> Self {
        Self {
            config,
            metrics,
            lease_rx,
            report_tx,
            backends,
            done_rx,
        }
    }

    pub async fn run(mut self, cancel: CancellationToken) -> Result<()> {
        self.metrics.set_status("waiting for lease");
        loop {
            tokio::select! {
                lease = self.lease_rx.recv() => {
                    match lease {
                        Some(lease) => {
                            if let Err(_e) = self.process_lease(lease, &cancel).await {
                                if !cancel.is_cancelled() {
                                    //eprintln!("[scheduler] lease error: {_e}");
                                }
                            }
                        }
                        None => break,
                    }
                }
                _ = cancel.cancelled() => break,
            }
        }
        Ok(())
    }

    async fn process_lease(&mut self, lease: Lease, cancel: &CancellationToken) -> Result<()> {
        let interval = Duration::from_millis(self.config.reporting.report_interval);

        self.metrics.set_status("computing");
        self.metrics.set_seed(lease.seed_str.clone());

        for b in &mut self.backends {
            b.in_flight = false;
        }

        let mut next_dispatch: u64 = 0;
        let mut in_flight: usize = 0;
        let mut total_done: u64 = 0;
        let mut win_best: i32 = -1;
        let mut win_arr = [0u8; 25];
        let mut win_index: u64 = 0;
        let mut last_report = Instant::now();

        let mut rate_clock = Instant::now();

        loop {
            // Dispatch to every idle backend that still has work to receive.
            for b in self.backends.iter_mut() {
                if !b.in_flight && next_dispatch < lease.count {
                    let lo = next_dispatch;
                    let hi = (lo + b.chunk_size).min(lease.count);
                    next_dispatch = hi;
                    b.work_tx
                        .send(Chunk {
                            seed: lease.seed,
                            lo,
                            hi,
                        })
                        .await?;
                    b.in_flight = true;
                    in_flight += 1;
                }
            }

            if in_flight == 0 {
                break;
            }

            let (id, result) = tokio::select! {
                r = self.done_rx.recv() => match r {
                    Some(r) => r,
                    None => break,
                },
                _ = cancel.cancelled() => break,
            };

            self.backends[id].in_flight = false;
            in_flight -= 1;

            let count = result.hi - result.lo;
            total_done += count;
            self.metrics
                .session_shuffles
                .fetch_add(count, std::sync::atomic::Ordering::Relaxed);

            let rate_elapsed = rate_clock.elapsed().as_secs_f64();
            if rate_elapsed > 0.0 {
                let rate = count as f64 / rate_elapsed;
                let prev = self.metrics.compute_rate();
                let new_rate = if prev <= 0.0 {
                    rate
                } else {
                    0.7 * prev + 0.3 * rate
                };
                self.metrics.set_compute_rate(new_rate);
                rate_clock = Instant::now();
            }

            if result.best_correct > win_best {
                win_best = result.best_correct;
                win_arr = result.best_arr;
                win_index = result.best_index;
            }

            let lease_done = total_done >= lease.count && in_flight == 0;
            let report_due = last_report.elapsed() >= interval;

            if (report_due || lease_done) && win_best >= 0 {
                let _ = self
                    .report_tx
                    .send(Report {
                        seed_str: lease.seed_str.clone(),
                        total_done,
                        best_correct: win_best as u32,
                        best_arr: win_arr,
                        best_index: win_index,
                    })
                    .await;

                self.metrics.push_report(win_best, win_arr);

                win_best = -1;
                win_arr = [0u8; 25];
                win_index = 0;
                last_report = Instant::now();
            }
        }

        self.metrics.set_status("waiting for lease");
        Ok(())
    }
}
