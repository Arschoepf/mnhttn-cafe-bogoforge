use anyhow::Result;
use serde::Deserialize;

use crate::net::types::Report;

#[derive(Debug, Deserialize)]
pub struct Welcome {
    pub lifetime_shuffles: u64,
    pub all_time_best: u32,
    /// Only present on first registration — must be persisted by the caller.
    pub code: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Job {
    /// The server sends seed as a JSON string (may exceed 2^53).
    pub seed: String,
    pub count: u64,
}

#[derive(Debug, Deserialize)]
pub struct Credited {
    pub credit: u64,
    pub lifetime_shuffles: u64,
    pub batch_best: Option<i32>,
    pub my_session_best: Option<i32>,
    pub all_time_best: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct Rejected {
    pub reason: String,
}

#[derive(Debug, Deserialize)]
pub struct Banned {
    pub reason: String,
}

#[derive(Debug)]
pub enum ServerMessage {
    Welcome(Welcome),
    Job(Job),
    Credited(Credited),
    Rejected(Rejected),
    ClientOutdated,
    Banned(Banned),
    ContributionsClosed,
    Unknown(String),
}

impl ServerMessage {
    pub fn parse(raw: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(raw)?;
        let msg_type = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(match msg_type.as_str() {
            "welcome" => ServerMessage::Welcome(serde_json::from_value(value)?),
            "job" => ServerMessage::Job(serde_json::from_value(value)?),
            "credited" => ServerMessage::Credited(serde_json::from_value(value)?),
            "rejected" => ServerMessage::Rejected(serde_json::from_value(value)?),
            "client_outdated" => ServerMessage::ClientOutdated,
            "banned" => ServerMessage::Banned(serde_json::from_value(value)?),
            "contributions_closed" => ServerMessage::ContributionsClosed,
            other => ServerMessage::Unknown(other.to_string()),
        })
    }
}

pub fn hello(uuid: &str, nickname: &str, code: &str) -> String {
    serde_json::json!({
        "type": "hello",
        "v": 5,
        "uuid": uuid,
        "nickname": nickname,
        "code": code,
    })
    .to_string()
}

pub fn result(report: &Report) -> String {
    let arr: Vec<u32> = report.best_arr.iter().map(|&v| v as u32).collect();
    serde_json::json!({
        "type": "result",
        "seed": report.seed_str,
        "total_done": report.total_done,
        "best_correct": report.best_correct,
        "best_arr": arr,
        "best_index": report.best_index,
    })
    .to_string()
}

pub fn stop() -> String {
    serde_json::json!({ "type": "stop" }).to_string()
}
