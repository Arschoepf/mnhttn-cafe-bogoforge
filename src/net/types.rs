#[derive(Debug, Clone)]
pub struct Lease {
    pub seed_str: String,
    pub seed: u64,
    pub count: u64,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub seed: u64,
    pub lo: u64,
    pub hi: u64,
}

#[derive(Debug, Clone)]
pub struct RangeResult {
    pub lo: u64,
    pub hi: u64,
    /// Number of fixed points in the best shuffle found.
    /// -1 only if the range was empty (should never occur in practice).
    pub best_correct: i32,
    pub best_arr: [u8; 25],
    pub best_index: u64,
}

/// A result payload ready to be sent to the server.
#[derive(Debug, Clone)]
pub struct Report {
    /// Echoed verbatim from the job seed field.
    pub seed_str: String,
    /// Cumulative shuffles computed on this lease.
    pub total_done: u64,
    pub best_correct: u32,
    pub best_arr: [u8; 25],
    /// The global index whose shuffle produced best_arr.
    pub best_index: u64,
}
