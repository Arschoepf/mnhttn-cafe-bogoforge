pub mod protocol;
pub mod types;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use http::header::HeaderValue;
use log::{debug, error, warn};
use tokio::sync::mpsc;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, Message},
};
use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::metrics::SharedMetrics;
use crate::net::types::{Lease, Report};

pub struct NetClient {
    config: Arc<Config>,
    metrics: SharedMetrics,
    lease_tx: mpsc::Sender<Lease>,
    result_rx: mpsc::Receiver<Report>,
}

impl NetClient {
    pub fn new(
        config: Arc<Config>,
        metrics: SharedMetrics,
        lease_tx: mpsc::Sender<Lease>,
        result_rx: mpsc::Receiver<Report>,
    ) -> Self {
        Self {
            config,
            metrics,
            lease_tx,
            result_rx,
        }
    }

    /// Runs the WebSocket loop, reconnecting on failure with exponential backoff.
    /// Returns when the cancel token fires or a fatal server message is received.
    pub async fn run(mut self, cancel: CancellationToken) -> Result<()> {
        let mut backoff = Duration::from_secs(2);
        loop {
            match self.connect_and_run(&cancel).await {
                Ok(()) => break,
                Err(_) if cancel.is_cancelled() => break,
                Err(e) => {
                    warn!("[net] connection error: {e}, reconnecting in {backoff:?}");
                    tokio::select! {
                        _ = tokio::time::sleep(backoff) => {}
                        _ = cancel.cancelled() => break,
                    }
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
        }
        Ok(())
    }

    async fn connect_and_run(&mut self, cancel: &CancellationToken) -> Result<()> {
        let mut request = self.config.server.url.as_str().into_client_request()?;

        {
            let headers = request.headers_mut();

            headers.insert(
                "origin",
                HeaderValue::from_static("https://bogo.swapjs.dev"),
            );
            headers.insert(
                "referer",
                HeaderValue::from_static("https://bogo.swapjs.dev/contribute"),
            );
            headers.insert(
                "user-agent",
                HeaderValue::from_static("Mozilla/5.0 (X11; Linux x86_64) BogoForge/1.0"),
            );
        }

        let (ws_stream, _) = connect_async(request).await?;
        let (mut write, mut read) = ws_stream.split();

        write
            .send(Message::Text(protocol::hello(
                &self.config.identity.uuid,
                &self.config.identity.nickname,
                &self.config.identity.code,
            )))
            .await?;

        loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            self.handle_message(&text, cancel).await?;

                            if cancel.is_cancelled() {
                                break;
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            write.send(Message::Pong(data)).await?;
                        }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => return Err(e.into()),
                        _ => {}
                    }
                }
                report = self.result_rx.recv() => {
                    match report {
                        Some(report) => {
                            write.send(Message::Text(protocol::result(&report))).await?;
                        }
                        None => break,
                    }
                }
                _ = cancel.cancelled() => {
                    let _ = write.send(Message::Text(protocol::stop())).await;
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_message(&self, text: &str, cancel: &CancellationToken) -> Result<()> {
        use protocol::ServerMessage;

        match ServerMessage::parse(text)? {
            ServerMessage::Welcome(w) => {
                self.metrics
                    .lifetime_shuffles
                    .store(w.lifetime_shuffles, std::sync::atomic::Ordering::Relaxed);
                if w.all_time_best > 0 {
                    self.metrics
                        .all_time_best
                        .store(w.all_time_best as i32, std::sync::atomic::Ordering::Relaxed);
                }
                self.metrics.set_status("waiting for lease");
                if let Some(code) = w.code {
                    warn!("[net] server issued recovery code: {code}");
                    warn!("[net] add it to identity.code in conf.toml");
                }
            }

            ServerMessage::Job(job) => {
                let seed = job
                    .seed
                    .parse::<u64>()
                    .map_err(|_| anyhow::anyhow!("job seed is not a valid u64: {}", job.seed))?;
                self.metrics.set_status("computing");
                let lease = Lease {
                    seed_str: job.seed,
                    seed,
                    count: job.count,
                };
                let _ = self.lease_tx.try_send(lease);
            }

            ServerMessage::Credited(c) => {
                self.metrics
                    .lifetime_shuffles
                    .store(c.lifetime_shuffles, std::sync::atomic::Ordering::Relaxed);

                if let Some(best) = c.all_time_best {
                    self.metrics
                        .all_time_best
                        .store(best, std::sync::atomic::Ordering::Relaxed);
                }
            }

            ServerMessage::Rejected(r) => {
                warn!("[net] rejected: {}", r.reason);
            }

            ServerMessage::ClientOutdated => {
                warn!("[net] server says client is outdated");
            }

            ServerMessage::Banned(b) => {
                error!("[net] banned: {}", b.reason);
                cancel.cancel();
            }

            ServerMessage::ContributionsClosed => {
                warn!("[net] contributions closed");
                cancel.cancel();
            }

            ServerMessage::Unknown(t) => {
                debug!("[net] unrecognised message type: {t} -- {text}");
            }
        }

        Ok(())
    }
}
