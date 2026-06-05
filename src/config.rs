use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub identity: IdentityConfig,
    pub server: ServerConfig,
    pub compute: ComputeConfig,
    pub reporting: ReportingConfig,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IdentityConfig {
    pub uuid: String,
    pub nickname: String,
    /// Empty string on first run. The server issues a code in `welcome`;
    /// call `Config::persist_code` to write it back here.
    pub code: String,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ComputeConfig {
    pub use_cpu: bool,
    /// 0 = use all logical cores.
    #[serde(default)]
    pub cpu_threads: usize,
    pub cpu_chunk_size: u64,

    pub use_gpu: bool,
    pub gpu_chunk_size: u64,
    pub cuda_blocks: u32,
    pub cuda_threads_per_block: u32,

    #[serde(default)]
    pub use_amd: bool,
    #[serde(default = "default_amd_chunk_size")]
    pub amd_chunk_size: u64,
    #[serde(default = "default_amd_blocks")]
    pub amd_blocks: u32,

    #[serde(default = "default_amd_tpb")]
    pub amd_threads_per_block: u32,
}

fn default_amd_chunk_size() -> u64 { 536870912 }
fn default_amd_blocks()      -> u32 { 2048 }
fn default_amd_tpb()         -> u32 { 256 }

#[derive(Debug, Deserialize)]
pub struct ReportingConfig {
    pub report_interval: u64,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;

        let config: Self = toml::from_str(&raw)
            .with_context(|| format!("failed to parse config file: {}", path.display()))?;

        config.validate()?;
        Ok(config)
    }

    /// Write the server-issued recovery code back into the config file.
    /// Preserves all other fields exactly as they appear on disk.
    pub fn persist_code(path: &Path, code: &str) -> Result<()> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;

        let mut doc: toml::Value = toml::from_str(&raw)?;

        doc["identity"]["code"] = toml::Value::String(code.to_string());

        std::fs::write(path, toml::to_string_pretty(&doc)?)
            .with_context(|| format!("failed to write config file: {}", path.display()))?;

        Ok(())
    }

    pub fn total_cuda_threads(&self) -> u32 {
        self.compute.cuda_blocks * self.compute.cuda_threads_per_block
    }

    fn validate(&self) -> Result<()> {
        let uuid = &self.identity.uuid;
        let hex_only: String = uuid.chars().filter(|c| *c != '-').collect();

        if hex_only.len() < 16 || hex_only.len() > 64 {
            bail!("identity.uuid must be 16–64 hex characters (dashes allowed)");
        }

        if !hex_only.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("identity.uuid must contain only hex characters and dashes");
        }

        let nick_len = self.identity.nickname.len();
        if nick_len < 2 || nick_len > 8 {
            bail!("identity.nickname must be 2–8 characters");
        }

        if self.compute.cpu_chunk_size == 0 {
            bail!("compute.cpu_chunk_size must be > 0");
        }

        if self.compute.gpu_chunk_size == 0 {
            bail!("compute.gpu_chunk_size must be > 0");
        }

        if self.compute.cuda_blocks == 0 || self.compute.cuda_threads_per_block == 0 {
            bail!("compute.cuda_blocks and compute.cuda_threads_per_block must be > 0");
        }

        if self.reporting.report_interval < 5 {
            bail!(
                "reporting.report_interval must be >= 5 (server rejects reports faster than 5ms)"
            );
        }

        Ok(())
    }
}
