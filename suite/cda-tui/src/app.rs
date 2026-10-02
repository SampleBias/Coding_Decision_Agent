//! Keyboard state for the four screens. Network calls happen outside this type.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, TryRecvError};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::api::{CompareOutcome, GradeResponse, Health};
use crate::catalog::{Case, Catalog, Pair, Replay};
use crate::metrics::{self, Report, THRESHOLDS};
use crate::session::{self, SessionRow};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Screen {
    Grade,
    Duel,
    Replay,
    Study,
}

impl Screen {
    pub fn index(self) -> usize {
        match self {
            Screen::Grade => 0,
            Screen::Duel => 1,
            Screen::Replay => 2,
            Screen::Study => 3,
        }
    }

    pub fn from_index(index: usize) -> Self {
        match index % 4 {
            0 => Screen::Grade,
            1 => Screen::Duel,
            2 => Screen::Replay,
            _ => Screen::Study,
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Screen::Grade => "Grade",
            Screen::Duel => "Duel",
            Screen::Replay => "Replay",
            Screen::Study => "Study",
        }
    }

    pub fn next(self) -> Self {
        Self::from_index(self.index() + 1)
    }

    pub fn prev(self) -> Self {
        Self::from_index(self.index() + 3)
    }
}

pub enum Job {
    Connect,
    Grades { screen: String, cases: Vec<Case> },
}

pub enum Msg {
    Health(Result<Health, String>),
    Graded {
        screen: String,
        case_id: String,
        result: Result<CompareOutcome, String>,
    },
    Finished {
        study: bool,
    },
}

pub enum Action {
    None,
    Quit,
    Spawn(Job),
    Export,
}

pub struct App {
    pub backend_url: String,
    pub catalog: Catalog,
    pub session_dir: PathBuf,
    pub live_path: PathBuf,
    pub screen: Screen,
    pub threshold: f64,
    pub threshold_ix: usize,
    pub blind: bool,
    pub revealed: bool,
    pub show_model_state: bool,
    pub help: bool,
    pub case_ix: usize,
    pub pair_ix: usize,
    pub replay_ix: usize,
    pub step_ix: usize,
    pub study_page: usize,
    pub state_scroll: u16,
    pub verdict_scroll: u16,
    pub study_scroll: u16,
    pub status: String,
    pub health: Option<Health>,
    pub grades: BTreeMap<String, GradeResponse>,
    pub jev_grades: BTreeMap<String, GradeResponse>,
    pub jev_errors: BTreeMap<String, String>,
    pub study_ids: Vec<String>,
    pub rows: Vec<SessionRow>,
    pub rx: Option<Receiver<Msg>>,
}

impl App {
    pub fn new(catalog: Catalog, backend_url: String, session_dir: PathBuf) -> Self {
        let threshold_ix = THRESHOLDS
            .iter()
            .position(|value| (*value - catalog.threshold_default).abs() < 1e-9)
            .unwrap_or(2);
        let live_path = session_dir.join("live.jsonl");
        Self {
            backend_url,
            threshold: THRESHOLDS[threshold_ix],
            threshold_ix,
            catalog,
            session_dir,
            live_path,
            screen: Screen::Grade,
            blind: false,
            revealed: true,
            show_model_state: false,
            help: false,
            case_ix: 0,
            pair_ix: 0,
            replay_ix: 0,
            step_ix: 0,
            study_page: 0,
            state_scroll: 0,
            verdict_scroll: 0,
            study_scroll: 0,
            status: "connecting to the sidecar".into(),
            health: None,
            grades: BTreeMap::new(),
            jev_grades: BTreeMap::new(),
            jev_errors: BTreeMap::new(),
            study_ids: Vec::new(),
            rows: Vec::new(),
            rx: None,
        }
    }

    pub fn busy(&self) -> bool {
        self.rx.is_some()
    }

    pub fn hidden(&self) -> bool {
        self.blind && !self.revealed
    }

    pub fn selected_case(&self) -> Option<&Case> {
        self.catalog.cases.get(self.case_ix)
    }

    pub fn selected_pair(&self) -> Option<&Pair> {
        self.catalog.pairs.get(self.pair_ix)
    }

    pub fn selected_replay(&self) -> Option<&Replay> {
        self.catalog.replays.get(self.replay_ix)
    }

    pub fn connect_job() -> Job {
        Job::Connect
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Action {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Action::Quit;
        }
        if self.help {
            return match key.code {
                KeyCode::Char('q') => Action::Quit,
                KeyCode::Esc | KeyCode::Char('?') => {
                    self.help = false;
                    Action::None
                }
                _ => Action::None,
            };
        }
        match key.code {
            KeyCode::Char('q') => Action::Quit,
            KeyCode::Char('?') => {
                self.help = true;
                Action::None
            }
            KeyCode::Tab => {
                self.screen = self.screen.next();
                Action::None
            }
            KeyCode::BackTab => {
                self.screen = self.screen.prev();
                Action::None
            }
            KeyCode::Char(digit @ '1'..='4') => {
                self.screen = Screen::from_index((digit as u8 - b'1') as usize);
                Action::None
            }
            KeyCode::Char('b') => {
                self.blind = !self.blind;
                self.revealed = !self.blind;
                self.status = if self.blind {
                    "blind: grades stay hidden until v".into()
                } else {
                    "blind off".into()
                };
                Action::None
            }
            KeyCode::Char('v') => {
                self.revealed = true;
                self.status = "revealed".into();
                Action::None
            }
            KeyCode::Char('t') => {
                self.threshold_ix = (self.threshold_ix + 1) % THRESHOLDS.len();
                self.threshold = THRESHOLDS[self.threshold_ix];
                self.status = format!("threshold {:.2}", self.threshold);
                Action::None
            }
            KeyCode::Char('m') => {
                self.show_model_state = !self.show_model_state;
                self.status = if self.show_model_state {
                    "showing the compacted model state".into()
                } else {
                    "showing the case fields".into()
                };
                Action::None
            }
            KeyCode::Char('e') => Action::Export,
            KeyCode::Char('r') => {
                if self.busy() {
                    self.status = "already waiting on the sidecar".into();
                    Action::None
                } else {
                    self.status = "reconnecting".into();
                    Action::Spawn(Job::Connect)
                }
            }
            KeyCode::Char('n') if self.screen == Screen::Study => {
                self.study_page = 1;
                Action::None
            }
            KeyCode::Char('p') if self.screen == Screen::Study => {
                self.study_page = 0;
                Action::None
            }
            KeyCode::Char('g') | KeyCode::Enter => {
                self.grade_action(matches!(key.code, KeyCode::Enter))
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.nudge(1);
                Action::None
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.nudge(-1);
                Action::None
            }
            KeyCode::Char(']') | KeyCode::PageDown => {
                self.state_scroll = self.state_scroll.saturating_add(1);
                Action::None
            }
            KeyCode::Char('[') | KeyCode::PageUp => {
                self.state_scroll = self.state_scroll.saturating_sub(1);
                Action::None
            }
            KeyCode::Char('.') => {
                self.verdict_scroll = self.verdict_scroll.saturating_add(1);
                Action::None
            }
            KeyCode::Char(',') => {
                self.verdict_scroll = self.verdict_scroll.saturating_sub(1);
                Action::None
            }
            _ => Action::None,
        }
    }

    fn grade_action(&mut self, enter: bool) -> Action {
        if self.busy() {
            self.status = "already waiting on the sidecar".into();
            return Action::None;
        }
        let job = if self.screen == Screen::Replay && enter {
            self.selected_replay().and_then(|replay| {
                replay.steps.get(self.step_ix).and_then(|step| {
                    self.catalog
                        .case(&step.case_id)
                        .cloned()
                        .map(|case| Job::Grades {
                            screen: "replay".into(),
                            cases: vec![case],
                        })
                })
            })
        } else {
            self.batch_job()
        };
        match job {
            Some(job) => {
                if let Job::Grades { screen, .. } = &job {
                    if screen == "study" {
                        self.study_ids.clear();
                        self.study_page = 0;
                    }
                    self.status = "grading".into();
                }
                Action::Spawn(job)
            }
            None => {
                self.status = "nothing to grade on this screen".into();
                Action::None
            }
        }
    }

    fn batch_job(&self) -> Option<Job> {
        let cases = match self.screen {
            Screen::Grade => vec![self.catalog.cases.get(self.case_ix)?.clone()],
            Screen::Duel => {
                let pair = self.catalog.pairs.get(self.pair_ix)?;
                vec![
                    self.catalog.case(&pair.left)?.clone(),
                    self.catalog.case(&pair.right)?.clone(),
                ]
            }
            Screen::Replay => {
                let replay = self.catalog.replays.get(self.replay_ix)?;
                let cases: Vec<Case> = replay
                    .steps
                    .iter()
                    .filter_map(|step| self.catalog.case(&step.case_id).cloned())
                    .collect();
                if cases.is_empty() {
                    return None;
                }
                cases
            }
            Screen::Study => self.catalog.cases.clone(),
        };
        let screen = match self.screen {
            Screen::Grade => "grade",
            Screen::Duel => "duel",
            Screen::Replay => "replay",
            Screen::Study => "study",
        };
        Some(Job::Grades {
            screen: screen.into(),
            cases,
        })
    }

    fn nudge(&mut self, delta: i32) {
        if self.screen == Screen::Study {
            if delta > 0 {
                self.study_scroll = self.study_scroll.saturating_add(1);
            } else {
                self.study_scroll = self.study_scroll.saturating_sub(1);
            }
            return;
        }
        let len = self.list_len();
        if len == 0 {
            return;
        }
        let current = self.list_index() as i32;
        let next = (current + delta).rem_euclid(len as i32) as usize;
        self.set_list_index(next);
        self.state_scroll = 0;
        self.verdict_scroll = 0;
    }

    fn list_len(&self) -> usize {
        match self.screen {
            Screen::Grade => self.catalog.cases.len(),
            Screen::Duel => self.catalog.pairs.len(),
            Screen::Replay => self
                .selected_replay()
                .map(|replay| replay.steps.len())
                .unwrap_or(0),
            Screen::Study => 0,
        }
    }

    fn list_index(&self) -> usize {
        match self.screen {
            Screen::Grade => self.case_ix,
            Screen::Duel => self.pair_ix,
            Screen::Replay => self.step_ix,
            Screen::Study => 0,
        }
    }

    fn set_list_index(&mut self, index: usize) {
        match self.screen {
            Screen::Grade => self.case_ix = index,
            Screen::Duel => self.pair_ix = index,
            Screen::Replay => self.step_ix = index,
            Screen::Study => {}
        }
    }

    pub fn poll_work(&mut self) -> bool {
        let message = match self.rx.as_ref().map(|rx| rx.try_recv()) {
            None => return false,
            Some(Ok(message)) => message,
            Some(Err(TryRecvError::Empty)) => return false,
            Some(Err(TryRecvError::Disconnected)) => {
                self.rx = None;
                return false;
            }
        };
        let finished = matches!(message, Msg::Finished { .. });
        self.apply(message);
        if finished {
            self.rx = None;
        }
        true
    }

    fn apply(&mut self, message: Msg) {
        match message {
            Msg::Health(Ok(health)) => {
                let jev = if health.jev_configured {
                    format!("jev {}", health.jev_model)
                } else {
                    "jev off".into()
                };
                self.status = format!("connected  {}  {}  {jev}", health.model_id, health.device);
                self.health = Some(health);
            }
            Msg::Health(Err(err)) => {
                self.health = None;
                self.status = friendly(&err, &self.backend_url);
            }
            Msg::Graded {
                screen,
                case_id,
                result,
            } => match result {
                Ok(outcome) => {
                    if screen == "study" {
                        self.study_ids.push(case_id.clone());
                    }
                    if self.blind {
                        self.revealed = false;
                    }
                    let latency = outcome.laya.latency_ms;
                    let source = outcome.laya.source.clone();
                    let jev_note = match &outcome.jev {
                        Some(grade) => format!("jev {:.1} ms", grade.latency_ms),
                        None => {
                            let err = outcome.jev_error.clone().unwrap_or_else(|| "off".into());
                            format!("jev {}", fit_status(&err))
                        }
                    };
                    if let Some(case) = self.catalog.case(&case_id).cloned() {
                        self.log_row(&screen, &case, &outcome.laya, "laya");
                        if let Some(grade) = &outcome.jev {
                            self.log_row(&screen, &case, grade, "jev");
                        }
                    }
                    self.grades.insert(case_id.clone(), outcome.laya);
                    match outcome.jev {
                        Some(grade) => {
                            self.jev_errors.remove(&case_id);
                            self.jev_grades.insert(case_id.clone(), grade);
                        }
                        None => {
                            self.jev_grades.remove(&case_id);
                            if let Some(err) = outcome.jev_error {
                                self.jev_errors.insert(case_id.clone(), err);
                            }
                        }
                    }
                    let progress = if screen == "study" {
                        format!(
                            "study {}/{}  ",
                            self.study_ids.len(),
                            self.catalog.cases.len()
                        )
                    } else {
                        String::new()
                    };
                    self.status =
                        format!("{progress}{case_id}  {source}  {latency:.1} ms  {jev_note}");
                }
                Err(err) => self.status = friendly(&err, &self.backend_url),
            },
            Msg::Finished { study } => {
                if study {
                    self.status = format!(
                        "study finished  {}/{}  e exports the log",
                        self.study_ids.len(),
                        self.catalog.cases.len()
                    );
                }
            }
        }
    }

    fn log_row(&mut self, screen: &str, case: &Case, grade: &GradeResponse, bench: &str) {
        let row = session::row_from(screen, case, grade, self.threshold, self.blind, bench);
        if let Err(err) = session::append(&self.live_path, &row) {
            self.status = err;
        }
        self.rows.push(row);
    }

    pub fn study_grades(&self) -> BTreeMap<String, GradeResponse> {
        self.study_ids
            .iter()
            .filter_map(|id| self.grades.get(id).map(|grade| (id.clone(), grade.clone())))
            .collect()
    }

    pub fn study_jev_grades(&self) -> BTreeMap<String, GradeResponse> {
        self.study_ids
            .iter()
            .filter_map(|id| {
                self.jev_grades
                    .get(id)
                    .map(|grade| (id.clone(), grade.clone()))
            })
            .collect()
    }

    pub fn cache_report(&self) -> Report {
        metrics::evaluate(&self.catalog, &self.grades, &THRESHOLDS)
    }

    pub fn export(&self) -> Result<PathBuf, String> {
        let report = self.cache_report();
        let (backend, model) = match &self.health {
            Some(health) => (health.backend.as_str(), health.model_id.as_str()),
            None => ("unknown", "unknown"),
        };
        let jev_report = if self.jev_grades.is_empty() {
            None
        } else {
            Some(metrics::evaluate(
                &self.catalog,
                &self.jev_grades,
                &THRESHOLDS,
            ))
        };
        let agreement = if self.jev_grades.is_empty() {
            None
        } else {
            Some(metrics::agreement(
                &self.catalog,
                &self.grades,
                &self.jev_grades,
            ))
        };
        session::export(
            &self.session_dir,
            &self.rows,
            &report,
            backend,
            model,
            self.threshold,
            jev_report.as_ref(),
            agreement.as_ref(),
        )
    }
}

fn fit_status(text: &str) -> String {
    let mut out = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= 72 {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

pub fn friendly(err: &str, url: &str) -> String {
    if err.contains("connect ") || err.contains("Connection refused") || err.contains("timed out") {
        format!("sidecar not reachable at {url} — python3 suite/sidecar/server.py")
    } else {
        err.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn tiny() -> App {
        let catalog = serde_json::from_str(
            r#"{
              "version": 1,
              "threshold_default": 0.7,
              "families": [{"id": "routing", "description": "", "questions": [
                {"id": "model_tier", "type": "choice", "instructions": "", "options": ["small_fast", "mid"], "gloss": ["small_fast", "mid"]}
              ]}],
              "cases": [
                {"case_id": "a", "family": "routing", "title": "A", "blurb": "", "fields": {"request": "a"}, "gold": {"model_tier": "mid"}, "mock": {}},
                {"case_id": "b", "family": "routing", "title": "B", "blurb": "", "fields": {"request": "b"}, "gold": {"model_tier": "small_fast"}, "mock": {}}
              ],
              "pairs": [],
              "replays": []
            }"#,
        )
        .unwrap();
        App::new(catalog, "http://127.0.0.1:9".into(), std::env::temp_dir())
    }

    #[test]
    fn tab_cycles_screens_and_j_wraps_the_case_list() {
        let mut app = tiny();
        assert_eq!(app.screen, Screen::Grade);
        assert!(matches!(app.on_key(key(KeyCode::Tab)), Action::None));
        assert_eq!(app.screen, Screen::Duel);
        assert!(matches!(app.on_key(key(KeyCode::Char('1'))), Action::None));
        assert_eq!(app.screen, Screen::Grade);
        assert_eq!(app.case_ix, 0);
        app.on_key(key(KeyCode::Char('j')));
        assert_eq!(app.case_ix, 1);
        app.on_key(key(KeyCode::Char('j')));
        assert_eq!(app.case_ix, 0);
        assert!(matches!(app.on_key(key(KeyCode::Char('q'))), Action::Quit));
    }

    #[test]
    fn threshold_cycle_stays_on_the_study_grid() {
        let mut app = tiny();
        app.on_key(key(KeyCode::Char('t')));
        assert!((app.threshold - 0.8).abs() < 1e-9);
        app.on_key(key(KeyCode::Char('t')));
        app.on_key(key(KeyCode::Char('t')));
        assert!((app.threshold - 0.5).abs() < 1e-9);
    }
}
