use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub identity: IdentityConfig,
    pub server: ServerConfig,
    pub compute: ComputeConfig,
    #[serde(default)]
    pub ui: UiConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    pub reporting: ReportingConfig,
}

#[derive(Debug, Deserialize, Default)]
pub struct UiConfig {
    #[serde(default)]
    pub disable_tui: bool,
}

#[derive(Debug, Deserialize, Default)]
pub struct LoggingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub kernel_activity: bool,
}

#[derive(Debug, Deserialize)]
pub struct IdentityConfig {
    pub uuid: String,
    pub nickname: String,
    pub code: String,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ComputeConfig {
    pub use_cpu: bool,
    /// 0 = all logical cores.
    #[serde(default)]
    pub cpu_threads: usize,
    pub cpu_chunk_size: u64,

    pub use_gpu: bool,
    /// Named GPU profile. See `ResolvedGpu::from_profile` for the full list.
    /// Determines backend (CUDA or HIP), blocks, threads/block, and chunk size.
    #[serde(default)]
    pub gpu_profile: String,
    /// Override the profile's chunk size.
    #[serde(default)]
    pub gpu_chunk_size: Option<u64>,
    /// Override the profile's block count.
    #[serde(default)]
    pub gpu_blocks: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ReportingConfig {
    pub report_interval: u64,
}

// ── GPU profile resolution ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind { Cuda, Hip, Vulkan }

#[derive(Debug, Clone)]
pub struct ResolvedGpu {
    pub kind:               GpuBackendKind,
    pub blocks:             u32,
    pub threads_per_block:  u32,
    pub chunk_size:         u64,
}

impl ResolvedGpu {
    pub fn from_profile(profile: &str, block_override: Option<u32>, chunk_override: Option<u64>) -> Result<Self> {
        let (kind, blocks, tpb, chunk) = match profile {
            "vulkan" => {
                let b = block_override.unwrap_or(1024);
                (GpuBackendKind::Vulkan, b, 256, chunk_override.unwrap_or(268435456))
            }
            // ── Manual / custom ────────────────────────────────────────────
            "cuda" => {
                let b = block_override.ok_or_else(|| anyhow::anyhow!(
                    "gpu_profile = \"cuda\" requires gpu_blocks in config.toml"
                ))?;
                (GpuBackendKind::Cuda, b, 256, chunk_override.unwrap_or(536870912))
            }
            "hip" => {
                let b = block_override.ok_or_else(|| anyhow::anyhow!(
                    "gpu_profile = \"hip\" requires gpu_blocks in config.toml"
                ))?;
                (GpuBackendKind::Hip, b, 256, chunk_override.unwrap_or(536870912))
            }
            other => bail!(
                "unknown gpu_profile \"{other}\"\n\
                 valid profiles: vulkan, cuda, hip\n"
            ),
        };

        Ok(Self {
            kind,
            blocks:            block_override.unwrap_or(blocks),
            threads_per_block: tpb,
            chunk_size:        chunk_override.unwrap_or(chunk),
        })
    }
}

// ── Config loading ────────────────────────────────────────────────────────────

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;
        let config: Self = toml::from_str(&raw)
            .with_context(|| format!("failed to parse config file: {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn persist_code(path: &Path, code: &str) -> Result<()> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;
        let mut doc: toml::Value = toml::from_str(&raw)?;
        doc["identity"]["code"] = toml::Value::String(code.to_string());
        std::fs::write(path, toml::to_string_pretty(&doc)?)
            .with_context(|| format!("failed to write config file: {}", path.display()))?;
        Ok(())
    }

    pub fn resolve_gpu(&self) -> Result<ResolvedGpu> {
        ResolvedGpu::from_profile(
            &self.compute.gpu_profile,
            self.compute.gpu_blocks,
            self.compute.gpu_chunk_size,
        )
    }

    fn validate(&self) -> Result<()> {
        let hex: String = self.identity.uuid.chars().filter(|c| *c != '-').collect();
        if hex.len() < 16 || hex.len() > 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("identity.uuid must be 16–64 hex characters");
        }
        let nick = self.identity.nickname.len();
        if nick < 2 || nick > 8 {
            bail!("identity.nickname must be 2–8 characters");
        }
        if self.compute.cpu_chunk_size == 0 {
            bail!("compute.cpu_chunk_size must be > 0");
        }
        if self.compute.use_gpu && self.compute.gpu_profile.is_empty() {
            bail!("compute.gpu_profile must be set when use_gpu = true");
        }
        if self.reporting.report_interval < 5 {
            bail!("reporting.report_interval must be >= 5ms");
        }
        Ok(())
    }
}
