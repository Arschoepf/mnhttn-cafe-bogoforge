use std::path::Path;

pub mod compute;
pub mod config;
pub mod metrics;
pub mod net;
pub mod runtime;
pub mod scheduler;
pub mod tui;

fn main() -> anyhow::Result<()> {
    let config = config::Config::load(Path::new("conf.toml"))?;

    runtime::ForgeRuntime::new(config).startup()
}
