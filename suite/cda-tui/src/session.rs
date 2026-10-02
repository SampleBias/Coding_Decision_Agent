//! Append-only session log. Every grade the TUI or `--check` receives is one
//! JSON object per line, plus a report computed from the latest grade per case.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{json, Value};

use crate::api::{Answer, GradeResponse};
use crate::catalog::Case;
use crate::metrics::{Agreement, Report};

#[derive(Clone, Debug, Serialize)]
pub struct SessionRow {
    pub ts: String,
    pub screen: String,
    pub case_id: String,
    pub family: String,
    pub title: String,
    pub backend: String,
    pub source: String,
    pub model_id: String,
    pub device: String,
    pub latency_ms: f64,
    pub threshold: f64,
    pub blind: bool,
    pub fields: Value,
    pub state: Value,
    pub gold: std::collections::BTreeMap<String, String>,
    pub answers: std::collections::BTreeMap<String, Answer>,
    pub note: String,
    /// `laya` or `jev`. Both rows for one case share the screen and case id.
    pub bench: String,
}

pub fn now_rfc3339() -> String {
    chrono::Local::now().to_rfc3339()
}

pub fn file_stamp() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

pub fn row_from(
    screen: &str,
    case: &Case,
    grade: &GradeResponse,
    threshold: f64,
    blind: bool,
    bench: &str,
) -> SessionRow {
    SessionRow {
        ts: now_rfc3339(),
        screen: screen.to_string(),
        case_id: case.case_id.clone(),
        family: case.family.clone(),
        title: case.title.clone(),
        backend: grade.backend.clone(),
        source: grade.source.clone(),
        model_id: grade.model_id.clone(),
        device: grade.device.clone(),
        latency_ms: grade.latency_ms,
        threshold,
        blind,
        fields: case.fields.clone(),
        state: grade.state.clone(),
        gold: case.gold.clone(),
        answers: grade.answers.clone(),
        note: grade.note.clone(),
        bench: bench.to_string(),
    }
}

pub fn append(path: &Path, row: &SessionRow) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("create {}: {err}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("open {}: {err}", path.display()))?;
    let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
    writeln!(file, "{line}").map_err(|err| format!("write {}: {err}", path.display()))?;
    Ok(())
}

pub fn export(
    dir: &Path,
    rows: &[SessionRow],
    report: &Report,
    backend: &str,
    model_id: &str,
    threshold: f64,
    jev_report: Option<&Report>,
    agreement: Option<&Agreement>,
) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|err| format!("create {}: {err}", dir.display()))?;
    let stamp = file_stamp();
    let jsonl = dir.join(format!("{stamp}.jsonl"));
    let report_path = dir.join(format!("{stamp}.report.json"));
    let mut file =
        fs::File::create(&jsonl).map_err(|err| format!("create {}: {err}", jsonl.display()))?;
    for row in rows {
        let line = serde_json::to_string(row).map_err(|err| err.to_string())?;
        writeln!(file, "{line}").map_err(|err| err.to_string())?;
    }
    let mut payload = serde_json::to_value(report).map_err(|err| err.to_string())?;
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "scope".into(),
            json!("latest grade per case in this session"),
        );
        object.insert("backend".into(), json!(backend));
        object.insert("model_id".into(), json!(model_id));
        object.insert("threshold".into(), json!(threshold));
        object.insert("rows".into(), json!(rows.len()));
        if let Some(jev_report) = jev_report {
            object.insert(
                "jev".into(),
                serde_json::to_value(jev_report).unwrap_or(Value::Null),
            );
        }
        if let Some(agreement) = agreement {
            object.insert(
                "agreement".into(),
                serde_json::to_value(agreement).unwrap_or(Value::Null),
            );
        }
    }
    fs::write(
        &report_path,
        serde_json::to_string_pretty(&payload).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("write {}: {err}", report_path.display()))?;
    Ok(report_path)
}
