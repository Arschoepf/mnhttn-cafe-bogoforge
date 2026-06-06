use std::path::Path;

use chrono::Local;
use fern::Dispatch;
use log::LevelFilter;

pub mod compute;
pub mod config;
pub mod metrics;
pub mod net;
pub mod runtime;
pub mod scheduler;
pub mod tui;

fn main() -> anyhow::Result<()> {
    let config = config::Config::load(Path::new("conf.toml"))?;
    init_logging(&config)?;

    log::info!("starting bogoforge (disable_tui={}, kernel_activity={})", config.ui.disable_tui, config.logging.kernel_activity);

    runtime::ForgeRuntime::new(config).startup()
}

fn init_logging(config: &config::Config) -> anyhow::Result<()> {
    let default_level = if config.logging.kernel_activity {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    };

    let logger = Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{} [{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                message
            ))
        })
        .level(default_level)
        .chain(std::io::stdout());

    logger.apply()?;
    Ok(())
}
