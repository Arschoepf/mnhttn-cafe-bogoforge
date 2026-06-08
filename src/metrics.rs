use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

pub type SharedMetrics = Arc<Metrics>;

const RECENT_BESTS_CAP: usize = 20;

pub struct Metrics {
    // Compute
    pub session_shuffles: AtomicU64,
    compute_rate: AtomicU64,

    // Best tracking
    pub last_report_best: AtomicI32,
    pub session_best: AtomicI32,
    pub all_time_best: AtomicI32,

    // Server
    pub lifetime_shuffles: AtomicU64,

    pub recent_bests: Mutex<VecDeque<i32>>,
    pub status: Mutex<String>,

    // Latest reported permutation (for visualisation) + the seed it came from
    pub last_best_arr: Mutex<[u8; 25]>,
    pub current_seed: Mutex<String>,

    // Host CPU / RAM, sampled periodically by the TUI's background sampler.
    cpu_usage_pct: AtomicU64,   // f64 bits
    pub mem_used_bytes: AtomicU64,
    pub mem_total_bytes: AtomicU64,

    // GPU stats, sampled via `nvidia-smi` (None if it isn't available — e.g.
    // non-NVIDIA hardware or a headless/driver-less box). -1 sentinels mean
    // "unknown" for the integer gauges.
    pub gpu_name: Mutex<Option<String>>,
    pub gpu_util_pct: AtomicI32,
    pub gpu_mem_used_mb: AtomicU64,
    pub gpu_mem_total_mb: AtomicU64,
    pub gpu_temp_c: AtomicI32,

    // Session timing
    pub started_at: Instant,
}

impl Metrics {
    pub fn new() -> SharedMetrics {
        Arc::new(Self {
            session_shuffles: AtomicU64::new(0),
            compute_rate: AtomicU64::new(0f64.to_bits()),
            last_report_best: AtomicI32::new(-1),
            session_best: AtomicI32::new(-1),
            all_time_best: AtomicI32::new(-1),
            lifetime_shuffles: AtomicU64::new(0),
            recent_bests: Mutex::new(VecDeque::new()),
            status: Mutex::new("starting".into()),
            last_best_arr: Mutex::new([0u8; 25]),
            current_seed: Mutex::new(String::new()),
            cpu_usage_pct: AtomicU64::new(0f64.to_bits()),
            mem_used_bytes: AtomicU64::new(0),
            mem_total_bytes: AtomicU64::new(0),
            gpu_name: Mutex::new(None),
            gpu_util_pct: AtomicI32::new(-1),
            gpu_mem_used_mb: AtomicU64::new(0),
            gpu_mem_total_mb: AtomicU64::new(0),
            gpu_temp_c: AtomicI32::new(-1),
            started_at: Instant::now(),
        })
    }

    pub fn compute_rate(&self) -> f64 {
        f64::from_bits(self.compute_rate.load(Ordering::Relaxed))
    }

    pub fn set_compute_rate(&self, rate: f64) {
        self.compute_rate.store(rate.to_bits(), Ordering::Relaxed);
    }

    pub fn push_report_best(&self, best: i32) {
        self.push_report(best, [0u8; 25]);
    }

    /// Record a finished report: its score, the winning permutation (for the
    /// live visualisation), and roll it into the recent-history ring buffer.
    pub fn push_report(&self, best: i32, arr: [u8; 25]) {
        self.last_report_best.store(best, Ordering::Relaxed);
        self.session_best.fetch_max(best, Ordering::Relaxed);
        if best >= 0 {
            *self.last_best_arr.lock() = arr;
        }

        let mut history = self.recent_bests.lock();
        if history.len() >= RECENT_BESTS_CAP {
            history.pop_front();
        }
        history.push_back(best);
    }

    pub fn set_status(&self, s: impl Into<String>) {
        *self.status.lock() = s.into();
    }

    pub fn set_seed(&self, seed: impl Into<String>) {
        *self.current_seed.lock() = seed.into();
    }

    // ── Host CPU / RAM (written by the TUI's background sampler) ──────────────

    pub fn set_host_stats(&self, cpu_usage_pct: f64, mem_used_bytes: u64, mem_total_bytes: u64) {
        self.cpu_usage_pct.store(cpu_usage_pct.to_bits(), Ordering::Relaxed);
        self.mem_used_bytes.store(mem_used_bytes, Ordering::Relaxed);
        self.mem_total_bytes.store(mem_total_bytes, Ordering::Relaxed);
    }

    pub fn cpu_usage_pct(&self) -> f64 {
        f64::from_bits(self.cpu_usage_pct.load(Ordering::Relaxed))
    }

    // ── GPU (written by the TUI's background sampler, via `nvidia-smi`) ───────

    pub fn set_gpu_stats(
        &self,
        name: String,
        util_pct: i32,
        mem_used_mb: u64,
        mem_total_mb: u64,
        temp_c: i32,
    ) {
        *self.gpu_name.lock() = Some(name);
        self.gpu_util_pct.store(util_pct, Ordering::Relaxed);
        self.gpu_mem_used_mb.store(mem_used_mb, Ordering::Relaxed);
        self.gpu_mem_total_mb.store(mem_total_mb, Ordering::Relaxed);
        self.gpu_temp_c.store(temp_c, Ordering::Relaxed);
    }

    /// Marks the GPU as unmonitorable (e.g. `nvidia-smi` isn't on PATH).
    pub fn clear_gpu_stats(&self) {
        *self.gpu_name.lock() = None;
        self.gpu_util_pct.store(-1, Ordering::Relaxed);
        self.gpu_temp_c.store(-1, Ordering::Relaxed);
    }
}
