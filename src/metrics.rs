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
        self.last_report_best.store(best, Ordering::Relaxed);
        self.session_best.fetch_max(best, Ordering::Relaxed);

        let mut history = self.recent_bests.lock();
        if history.len() >= RECENT_BESTS_CAP {
            history.pop_front();
        }
        history.push_back(best);
    }

    pub fn set_status(&self, s: impl Into<String>) {
        *self.status.lock() = s.into();
    }
}
