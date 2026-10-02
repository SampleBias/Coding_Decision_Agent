//! Study metrics. The definitions match `train/evaluate.py` where that script
//! has an opinion (argmax for scores, the stated choice for choice questions,
//! one-hot gold when the suite only stores a label) and `suite/sidecar/scoring.py`
//! for coherence and contrast pairs.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::api::{Answer, GradeResponse};
use crate::catalog::{Catalog, Question};

pub const THRESHOLDS: [f64; 5] = [0.5, 0.6, 0.7, 0.8, 0.9];

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub n_cases: usize,
    pub overall: Metrics,
    pub by_family: BTreeMap<String, Metrics>,
    pub by_question: BTreeMap<String, Metrics>,
    pub pairs_passed: usize,
    pub pairs_failed: usize,
    pub pairs_pending: usize,
    pub pair_failures: Vec<String>,
    pub coherence: Vec<String>,
    pub sweep: Vec<SweepRow>,
    pub latency_ms_p50: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Metrics {
    pub n: usize,
    pub accuracy: f64,
    pub soft_accuracy: f64,
    pub brier: f64,
    pub ece: f64,
    pub score_mae: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SweepRow {
    pub threshold: f64,
    pub coverage: f64,
    pub auto: usize,
    pub gated: usize,
    pub conditional_accuracy: Option<f64>,
}

#[derive(Default)]
struct Bucket {
    n: usize,
    correct: usize,
    soft: f64,
    brier: f64,
    score_abs: f64,
    score_n: usize,
    conf: Vec<f64>,
    hit: Vec<f64>,
}

pub fn argmax<'a>(probs: &BTreeMap<String, f64>, options: &'a [String]) -> &'a str {
    let mut best_i = 0usize;
    let mut best_p = f64::NEG_INFINITY;
    for (index, key) in options.iter().enumerate() {
        let probability = probs.get(key).copied().unwrap_or(0.0);
        if probability > best_p {
            best_p = probability;
            best_i = index;
        }
    }
    options.get(best_i).map(String::as_str).unwrap_or("")
}

/// Choice questions use the model's stated choice, matching `train/evaluate.py`.
/// Score questions use the argmax of the distribution.
pub fn decision_label<'a>(answer: &'a Answer, options: &'a [String]) -> &'a str {
    if answer.qtype == "choice" {
        if let Some(choice) = answer.choice.as_deref() {
            if options.iter().any(|option| option == choice) {
                return choice;
            }
        }
        if options.iter().any(|option| option == &answer.label) {
            return answer.label.as_str();
        }
    }
    argmax(&answer.probabilities, options)
}

pub fn expected_score(probs: &BTreeMap<String, f64>, options: &[String]) -> f64 {
    let weights: Vec<f64> = options
        .iter()
        .map(|key| probs.get(key).copied().unwrap_or(0.0).max(0.0))
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    weights
        .iter()
        .enumerate()
        .map(|(index, weight)| index as f64 * weight)
        .sum::<f64>()
        / total
}

pub fn coherence_violations(family: &str, labels: &BTreeMap<String, String>) -> Vec<String> {
    let mut out = Vec::new();
    let lab = |qid: &str| labels.get(qid).map(String::as_str);
    match family {
        "code_review" => {
            if lab("likely_correct") == Some("no") && lab("merge_action") == Some("accept") {
                out.push("likely_correct=no but merge_action=accept".to_string());
            }
        }
        "tool_call" => {
            if lab("call_verdict") == Some("execute") && lab("parameter_structure") == Some("0") {
                out.push("call_verdict=execute but parameter_structure=0".to_string());
            }
            if lab("call_verdict") == Some("execute")
                && lab("should_call_tool") == Some("answer_directly")
            {
                out.push("call_verdict=execute but should_call_tool=answer_directly".to_string());
            }
            if lab("call_verdict") == Some("execute") && lab("tool_selection") == Some("0") {
                out.push("call_verdict=execute but tool_selection=0".to_string());
            }
            if lab("call_verdict") == Some("abstain")
                && lab("should_call_tool") == Some("call_tool")
            {
                out.push("call_verdict=abstain but should_call_tool=call_tool".to_string());
            }
        }
        "agent_trace" => {
            if lab("outcome") == Some("harmful") && lab("action") == Some("continue") {
                out.push("outcome=harmful but action=continue".to_string());
            }
            if lab("outcome") == Some("harmful") && lab("risk") == Some("0") {
                out.push("outcome=harmful but risk=0".to_string());
            }
            if lab("outcome") == Some("success") && lab("action") == Some("stop") {
                out.push("outcome=success but action=stop".to_string());
            }
        }
        "routing" => {
            if lab("skill") == Some("explain")
                && lab("task_difficulty") == Some("0")
                && lab("model_tier") == Some("reasoning")
            {
                out.push("trivial explain routed to reasoning".to_string());
            }
        }
        _ => {}
    }
    out
}

pub fn ece(conf: &[f64], hit: &[f64], bins: usize) -> f64 {
    if conf.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; bins];
    let mut conf_sum = vec![0.0; bins];
    let mut hit_sum = vec![0.0; bins];
    for (confidence, correct) in conf.iter().zip(hit) {
        let mut bin = (confidence.clamp(0.0, 1.0) * bins as f64) as usize;
        if bin >= bins {
            bin = bins - 1;
        }
        counts[bin] += 1;
        conf_sum[bin] += confidence;
        hit_sum[bin] += correct;
    }
    let total = conf.len() as f64;
    let mut score = 0.0;
    for index in 0..bins {
        if counts[index] == 0 {
            continue;
        }
        let accuracy = hit_sum[index] / counts[index] as f64;
        let mean_conf = conf_sum[index] / counts[index] as f64;
        score += (counts[index] as f64 / total) * (accuracy - mean_conf).abs();
    }
    score
}

/// Gate the runtime would auto-act on. Routing has no execute gate.
pub fn gate(family: &str) -> Option<(&'static str, &'static str)> {
    match family {
        "code_review" => Some(("merge_action", "accept")),
        "tool_call" => Some(("call_verdict", "execute")),
        "agent_trace" => Some(("action", "continue")),
        _ => None,
    }
}

pub fn evaluate(
    catalog: &Catalog,
    grades: &BTreeMap<String, GradeResponse>,
    thresholds: &[f64],
) -> Report {
    let mut overall = Bucket::default();
    let mut by_family: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut by_question: BTreeMap<String, Bucket> = BTreeMap::new();
    let mut coherence = Vec::new();
    let mut latencies = Vec::new();
    let mut gated: Vec<Gated> = Vec::new();

    for case in &catalog.cases {
        let Some(grade) = grades.get(&case.case_id) else {
            continue;
        };
        let Some(family) = catalog.family(&case.family) else {
            continue;
        };
        latencies.push(grade.latency_ms);
        let mut labels = BTreeMap::new();
        for question in &family.questions {
            let Some(answer) = grade.answers.get(&question.id) else {
                continue;
            };
            let predicted = decision_label(answer, &question.options).to_string();
            labels.insert(question.id.clone(), predicted.clone());
            let Some(gold) = case.gold.get(&question.id) else {
                continue;
            };
            accumulate(&mut overall, question, gold, answer, &predicted);
            accumulate(
                by_family.entry(case.family.clone()).or_default(),
                question,
                gold,
                answer,
                &predicted,
            );
            let qkey = format!("{}.{}", case.family, question.id);
            accumulate(
                by_question.entry(qkey).or_default(),
                question,
                gold,
                answer,
                &predicted,
            );
        }
        for violation in coherence_violations(&case.family, &labels) {
            coherence.push(format!("{}: {violation}", case.case_id));
        }
        if let Some((qid, pass)) = gate(&case.family) {
            if let (Some(answer), Some(question), Some(gold)) = (
                grade.answers.get(qid),
                family.questions.iter().find(|question| question.id == qid),
                case.gold.get(qid),
            ) {
                gated.push(Gated {
                    predicted: decision_label(answer, &question.options).to_string(),
                    gold: gold.clone(),
                    pass: pass.to_string(),
                    confidence: answer.answer_confidence,
                });
            }
        }
    }

    let mut pair_failures = Vec::new();
    let mut pairs_passed = 0;
    let mut pairs_pending = 0;
    for pair in &catalog.pairs {
        if !grades.contains_key(&pair.left) || !grades.contains_key(&pair.right) {
            pairs_pending += 1;
            continue;
        }
        let failures = pair_failures_for(catalog, pair, &grades[&pair.left], &grades[&pair.right]);
        if failures.is_empty() {
            pairs_passed += 1;
        } else {
            pair_failures.extend(failures);
        }
    }
    let pairs_failed = catalog.pairs.len() - pairs_passed - pairs_pending;

    let sweep = thresholds
        .iter()
        .copied()
        .map(|threshold| sweep_row(&gated, threshold))
        .collect();

    Report {
        n_cases: grades.len(),
        overall: finalize(&overall),
        by_family: by_family
            .iter()
            .map(|(key, bucket)| (key.clone(), finalize(bucket)))
            .collect(),
        by_question: by_question
            .iter()
            .map(|(key, bucket)| (key.clone(), finalize(bucket)))
            .collect(),
        pairs_passed,
        pairs_failed,
        pairs_pending,
        pair_failures,
        coherence,
        sweep,
        latency_ms_p50: percentile(&mut latencies, 0.5),
    }
}

struct Gated {
    predicted: String,
    gold: String,
    pass: String,
    confidence: f64,
}

fn sweep_row(gated: &[Gated], threshold: f64) -> SweepRow {
    let mut auto = 0usize;
    let mut correct = 0usize;
    for row in gated {
        let acts = row.predicted == row.pass && row.confidence >= threshold;
        if acts {
            auto += 1;
            if row.gold == row.pass {
                correct += 1;
            }
        }
    }
    SweepRow {
        threshold,
        coverage: if gated.is_empty() {
            0.0
        } else {
            auto as f64 / gated.len() as f64
        },
        auto,
        gated: gated.len(),
        conditional_accuracy: if auto == 0 {
            None
        } else {
            Some(correct as f64 / auto as f64)
        },
    }
}

pub fn pair_ok(
    catalog: &Catalog,
    pair_index: usize,
    grades: &BTreeMap<String, GradeResponse>,
) -> Option<bool> {
    let pair = catalog.pairs.get(pair_index)?;
    if !grades.contains_key(&pair.left) || !grades.contains_key(&pair.right) {
        return None;
    }
    Some(pair_failures_for(catalog, pair, &grades[&pair.left], &grades[&pair.right]).is_empty())
}

fn pair_failures_for(
    catalog: &Catalog,
    pair: &crate::catalog::Pair,
    left: &GradeResponse,
    right: &GradeResponse,
) -> Vec<String> {
    let mut failures = Vec::new();
    let family = catalog.family(&left.family);
    let question = |qid: &str| -> Option<&Question> {
        family?.questions.iter().find(|question| question.id == qid)
    };
    let exp = |grade: &GradeResponse, qid: &str| -> Option<f64> {
        let question = question(qid)?;
        let answer = grade.answers.get(qid)?;
        Some(expected_score(&answer.probabilities, &question.options))
    };
    let label = |grade: &GradeResponse, qid: &str| -> Option<String> {
        let question = question(qid)?;
        let answer = grade.answers.get(qid)?;
        Some(decision_label(answer, &question.options).to_string())
    };
    for qid in &pair.higher_on_left {
        if let (Some(el), Some(er)) = (exp(left, qid), exp(right, qid)) {
            if el <= er {
                failures.push(format!(
                    "{}: {qid} expected left higher ({el:.3} vs {er:.3})",
                    pair.pair_id
                ));
            }
        }
    }
    for qid in &pair.higher_on_right {
        if let (Some(el), Some(er)) = (exp(left, qid), exp(right, qid)) {
            if er <= el {
                failures.push(format!(
                    "{}: {qid} expected right higher ({el:.3} vs {er:.3})",
                    pair.pair_id
                ));
            }
        }
    }
    for (qid, want) in &pair.prefer_on_left {
        if let Some(got) = label(left, qid) {
            if &got != want {
                failures.push(format!(
                    "{}: left {qid} got {got} expected {want}",
                    pair.pair_id
                ));
            }
        }
    }
    for (qid, want) in &pair.prefer_on_right {
        if let Some(got) = label(right, qid) {
            if &got != want {
                failures.push(format!(
                    "{}: right {qid} got {got} expected {want}",
                    pair.pair_id
                ));
            }
        }
    }
    failures
}

fn accumulate(
    bucket: &mut Bucket,
    question: &Question,
    gold: &str,
    answer: &Answer,
    predicted: &str,
) {
    let pred = renorm(&answer.probabilities, &question.options);
    let mut gold_vec = vec![0.0; question.options.len()];
    if let Some(index) = question.options.iter().position(|option| option == gold) {
        gold_vec[index] = 1.0;
    }
    bucket.n += 1;
    let hit = predicted == gold;
    if hit {
        bucket.correct += 1;
    }
    let soft: f64 = gold_vec.iter().zip(&pred).map(|(g, p)| g * p).sum();
    let brier: f64 = gold_vec
        .iter()
        .zip(&pred)
        .map(|(g, p)| (p - g).powi(2))
        .sum();
    bucket.soft += soft;
    bucket.brier += brier;
    if question.qtype == "score" {
        if let (Some(got), Some(want)) = (
            question
                .options
                .iter()
                .position(|option| option == predicted),
            question.options.iter().position(|option| option == gold),
        ) {
            bucket.score_abs += (got as f64 - want as f64).abs();
            bucket.score_n += 1;
        }
    }
    bucket.conf.push(answer.answer_confidence);
    bucket.hit.push(if hit { 1.0 } else { 0.0 });
}

fn renorm(probs: &BTreeMap<String, f64>, options: &[String]) -> Vec<f64> {
    let mut values: Vec<f64> = options
        .iter()
        .map(|key| probs.get(key).copied().unwrap_or(0.0).max(0.0))
        .collect();
    let total: f64 = values.iter().sum();
    if total <= 0.0 {
        let uniform = 1.0 / options.len().max(1) as f64;
        values.fill(uniform);
    } else {
        for value in &mut values {
            *value /= total;
        }
    }
    values
}

fn finalize(bucket: &Bucket) -> Metrics {
    let n = bucket.n.max(1) as f64;
    Metrics {
        n: bucket.n,
        accuracy: if bucket.n == 0 {
            0.0
        } else {
            bucket.correct as f64 / n
        },
        soft_accuracy: if bucket.n == 0 { 0.0 } else { bucket.soft / n },
        brier: if bucket.n == 0 { 0.0 } else { bucket.brier / n },
        ece: ece(&bucket.conf, &bucket.hit, 10),
        score_mae: if bucket.score_n == 0 {
            None
        } else {
            Some(bucket.score_abs / bucket.score_n as f64)
        },
    }
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f64 * p).round() as usize;
    values[index.min(values.len() - 1)]
}

/// Build the grade the mock sidecar would return for a catalog case.
/// Used by tests and the UI render test. The live TUI does not call this.
pub fn scripted_grade(catalog: &Catalog, case: &crate::catalog::Case) -> GradeResponse {
    let family = catalog.family(&case.family).expect("family");
    let mut answers = BTreeMap::new();
    for question in &family.questions {
        let probs = case.mock.get(&question.id).cloned().unwrap_or_default();
        let label = argmax(&probs, &question.options).to_string();
        let confidence = probs.values().copied().fold(0.0_f64, f64::max);
        answers.insert(
            question.id.clone(),
            Answer {
                qtype: question.qtype.clone(),
                label: label.clone(),
                probabilities: probs.clone(),
                answer_confidence: confidence,
                choice: if question.qtype == "choice" {
                    Some(label)
                } else {
                    None
                },
                score: if question.qtype == "score" {
                    Some(expected_score(&probs, &question.options))
                } else {
                    None
                },
            },
        );
    }
    GradeResponse {
        ok: true,
        backend: "mock".into(),
        source: "scripted".into(),
        model_id: "mock-scripted".into(),
        device: "none".into(),
        family: case.family.clone(),
        case_id: Some(case.case_id.clone()),
        latency_ms: 0.0,
        state: serde_json::Value::Null,
        answers,
        missing: Vec::new(),
        note: "scripted mock, not the checkpoint".into(),
        cost_usd: None,
    }
}

pub fn sources_of(grades: &BTreeMap<String, GradeResponse>) -> BTreeSet<String> {
    grades.values().map(|grade| grade.source.clone()).collect()
}

/// How often two graders pick the same decision label on questions both answered.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Agreement {
    pub n: usize,
    pub same: usize,
    pub rate: f64,
    pub disagreements: Vec<String>,
}

pub fn agreement(
    catalog: &Catalog,
    left: &BTreeMap<String, GradeResponse>,
    right: &BTreeMap<String, GradeResponse>,
) -> Agreement {
    let mut n = 0usize;
    let mut same = 0usize;
    let mut disagreements = Vec::new();
    for case in &catalog.cases {
        let (Some(laya), Some(jev)) = (left.get(&case.case_id), right.get(&case.case_id)) else {
            continue;
        };
        let Some(family) = catalog.family(&case.family) else {
            continue;
        };
        for question in &family.questions {
            let (Some(laya_answer), Some(jev_answer)) = (
                laya.answers.get(&question.id),
                jev.answers.get(&question.id),
            ) else {
                continue;
            };
            n += 1;
            let laya_label = decision_label(laya_answer, &question.options);
            let jev_label = decision_label(jev_answer, &question.options);
            if laya_label == jev_label {
                same += 1;
            } else if disagreements.len() < 80 {
                disagreements.push(format!(
                    "{} {} laya={} jev={}",
                    case.case_id, question.id, laya_label, jev_label
                ));
            }
        }
    }
    Agreement {
        n,
        same,
        rate: if n == 0 { 0.0 } else { same as f64 / n as f64 },
        disagreements,
    }
}

/// One sentence for the Study footer. Accuracies are the 0–1 scores shown in the table.
pub fn accuracy_sentence(laya: Option<(f64, usize)>, jev: Option<(f64, usize)>) -> String {
    match (laya, jev) {
        (None, None) => "Press g to grade the suite and compare Laya-CDA with Jev.".into(),
        (Some((acc, n)), None) => {
            format!("Laya-CDA accuracy is {acc:.3} on {n} questions. Jev has no grades yet.")
        }
        (None, Some((acc, n))) => {
            format!("Jev accuracy is {acc:.3} on {n} questions. Laya-CDA has no grades yet.")
        }
        (Some((laya_acc, _)), Some((jev_acc, _))) => {
            let gap = laya_acc - jev_acc;
            if gap.abs() < 0.0005 {
                format!("Laya-CDA and Jev have the same accuracy ({laya_acc:.3}).")
            } else if gap > 0.0 {
                format!(
                    "Laya-CDA is {gap:.3} more accurate than Jev ({laya_acc:.3} vs {jev_acc:.3})."
                )
            } else {
                format!(
                    "Jev is {:.3} more accurate than Laya-CDA ({jev_acc:.3} vs {laya_acc:.3}).",
                    -gap
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn accuracy_sentence_names_the_leader() {
        assert_eq!(
            accuracy_sentence(Some((1.0, 73)), Some((0.2, 73))),
            "Laya-CDA is 0.800 more accurate than Jev (1.000 vs 0.200)."
        );
        assert_eq!(
            accuracy_sentence(Some((0.5, 10)), Some((0.8, 10))),
            "Jev is 0.300 more accurate than Laya-CDA (0.800 vs 0.500)."
        );
        assert_eq!(
            accuracy_sentence(Some((0.8, 10)), Some((0.8, 10))),
            "Laya-CDA and Jev have the same accuracy (0.800)."
        );
        assert!(accuracy_sentence(Some((1.0, 73)), None).contains("Jev has no grades"));
        assert!(accuracy_sentence(None, None).contains("Press g"));
    }

    #[test]
    fn ece_is_zero_when_confidence_matches_hits() {
        let conf = [1.0, 1.0, 0.0, 0.0];
        let hit = [1.0, 1.0, 0.0, 0.0];
        assert!(ece(&conf, &hit, 10) < 1e-9);
    }

    #[test]
    fn ece_of_always_certain_and_mostly_wrong() {
        let conf = [1.0, 1.0, 1.0, 1.0];
        let hit = [1.0, 0.0, 0.0, 0.0];
        let score = ece(&conf, &hit, 10);
        assert!((score - 0.75).abs() < 1e-9, "{score}");
    }

    #[test]
    fn choice_label_follows_the_stated_choice_not_the_mode() {
        let answer = Answer {
            qtype: "choice".into(),
            label: "accept".into(),
            probabilities: BTreeMap::from([("accept".into(), 0.2), ("reject".into(), 0.8)]),
            answer_confidence: 0.2,
            choice: Some("accept".into()),
            score: None,
        };
        let options = vec!["accept".into(), "reject".into()];
        assert_eq!(decision_label(&answer, &options), "accept");
    }

    #[test]
    fn flags_accepting_a_patch_the_model_thinks_is_wrong() {
        let labels = BTreeMap::from([
            ("likely_correct".into(), "no".into()),
            ("merge_action".into(), "accept".into()),
        ]);
        assert_eq!(
            coherence_violations("code_review", &labels),
            vec!["likely_correct=no but merge_action=accept".to_string()]
        );
    }

    #[test]
    fn scripted_catalog_passes_its_own_labels() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../cases/catalog.json");
        let catalog = crate::catalog::load(&path).expect("catalog");
        let mut grades = BTreeMap::new();
        for case in &catalog.cases {
            grades.insert(case.case_id.clone(), scripted_grade(&catalog, case));
        }
        let report = evaluate(&catalog, &grades, &THRESHOLDS);
        assert_eq!(report.n_cases, 16);
        assert_eq!(report.overall.n, 73);
        assert!((report.overall.accuracy - 1.0).abs() < 1e-9);
        assert!(
            report.pair_failures.is_empty(),
            "{:?}",
            report.pair_failures
        );
        assert_eq!(report.pairs_passed, catalog.pairs.len());
        assert_eq!(report.pairs_pending, 0);
        assert!(report.coherence.is_empty(), "{:?}", report.coherence);
        let at_070 = report
            .sweep
            .iter()
            .find(|row| (row.threshold - 0.7).abs() < 1e-9)
            .unwrap();
        assert!(at_070.gated > 0);
        assert_eq!(at_070.conditional_accuracy, Some(1.0));
        let agree = agreement(&catalog, &grades, &grades);
        assert_eq!(agree.n, 73);
        assert_eq!(agree.same, 73);
        assert!(agree.disagreements.is_empty());
    }
}
