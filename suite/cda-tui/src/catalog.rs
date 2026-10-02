//! The contrast catalog written by `suite/cases/build_catalog.py`.
//!
//! Question order is the training order. Option order is the label index order.
//! Neither is sorted.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
pub struct Catalog {
    pub version: u32,
    #[serde(default = "default_threshold")]
    pub threshold_default: f64,
    pub families: Vec<Family>,
    pub cases: Vec<Case>,
    pub pairs: Vec<Pair>,
    pub replays: Vec<Replay>,
}

fn default_threshold() -> f64 {
    0.7
}

#[derive(Clone, Debug, Deserialize)]
pub struct Family {
    pub id: String,
    #[serde(default)]
    pub description: String,
    pub questions: Vec<Question>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Question {
    pub id: String,
    #[serde(rename = "type")]
    pub qtype: String,
    #[serde(default)]
    pub instructions: String,
    pub options: Vec<String>,
    #[serde(default)]
    pub gloss: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Case {
    pub case_id: String,
    pub family: String,
    pub title: String,
    #[serde(default)]
    pub blurb: String,
    pub fields: Value,
    pub gold: BTreeMap<String, String>,
    #[serde(default)]
    pub mock: BTreeMap<String, BTreeMap<String, f64>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Pair {
    pub pair_id: String,
    pub title: String,
    pub left: String,
    pub right: String,
    #[serde(default)]
    pub higher_on_left: Vec<String>,
    #[serde(default)]
    pub higher_on_right: Vec<String>,
    #[serde(default)]
    pub prefer_on_left: BTreeMap<String, String>,
    #[serde(default)]
    pub prefer_on_right: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Replay {
    pub replay_id: String,
    pub title: String,
    pub steps: Vec<ReplayStep>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReplayStep {
    pub case_id: String,
    pub caption: String,
}

impl Catalog {
    pub fn family(&self, id: &str) -> Option<&Family> {
        self.families.iter().find(|family| family.id == id)
    }

    pub fn case(&self, id: &str) -> Option<&Case> {
        self.cases.iter().find(|case| case.case_id == id)
    }

    pub fn question<'a>(&'a self, family: &str, qid: &str) -> Option<&'a Question> {
        self.family(family)?
            .questions
            .iter()
            .find(|question| question.id == qid)
    }
}

pub fn load(path: &Path) -> Result<Catalog, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let catalog: Catalog =
        serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))?;
    if catalog.version != 1 {
        return Err(format!("unsupported catalog version {}", catalog.version));
    }
    if catalog.cases.is_empty() {
        return Err(format!("{} has no cases", path.display()));
    }
    Ok(catalog)
}

pub fn locate(explicit: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!("cases file not found: {}", path.display()));
    }
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            let candidate = ancestor.join("suite/cases/catalog.json");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let beside = PathBuf::from("suite/cases/catalog.json");
    if beside.is_file() {
        return Ok(beside);
    }
    Err("could not find suite/cases/catalog.json (pass --cases)".into())
}

pub fn default_session_dir(catalog: &Path) -> PathBuf {
    catalog
        .parent()
        .and_then(|cases| cases.parent())
        .map(|suite| suite.join("sessions"))
        .unwrap_or_else(|| PathBuf::from("suite/sessions"))
}
