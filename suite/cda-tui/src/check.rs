//! Grade the catalog through the sidecar and print the study report.
//!
//! Mock mode exits 1 when the scripted labels, contrast pairs, or coherence
//! rules do not hold. A real checkpoint is allowed to miss; the report is the
//! result.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::Path;

use crate::api::{self, GradeResponse};
use crate::catalog::Catalog;
use crate::metrics::{self, Agreement, Report, THRESHOLDS};
use crate::session::{self, SessionRow};

pub fn run(base: &str, catalog: &Catalog, session_dir: &Path) -> Result<i32, String> {
    let health = api::health(base)
        .map_err(|err| format!("{err}\nStart it with: python3 suite/sidecar/server.py"))?;
    let remote = api::schema(base)?;
    if !api::schemas_match(&catalog.families, &remote) {
        return Err("sidecar schema does not match this catalog. Rebuild with: python3 suite/cases/build_catalog.py".into());
    }
    let jev_note = if health.jev_configured {
        format!("jev={}", health.jev_model)
    } else {
        "jev=off".into()
    };
    println!(
        "Laya-CDA  backend={}  model={}  device={}  {}  cases={}",
        health.backend,
        health.model_id,
        health.device,
        jev_note,
        catalog.cases.len()
    );

    let mut grades: BTreeMap<String, GradeResponse> = BTreeMap::new();
    let mut rows: Vec<SessionRow> = Vec::new();
    let mut failed = false;
    for case in &catalog.cases {
        match api::grade(base, case) {
            Ok(grade) => {
                println!(
                    "  {:<24} {:<10} {:>7.1} ms  {}",
                    case.case_id, grade.source, grade.latency_ms, grade.model_id
                );
                let row = session::row_from(
                    "check",
                    case,
                    &grade,
                    catalog.threshold_default,
                    false,
                    "laya",
                );
                if let Err(err) = session::append(&session_dir.join("live.jsonl"), &row) {
                    eprintln!("log: {err}");
                }
                rows.push(row);
                grades.insert(case.case_id.clone(), grade);
            }
            Err(err) => {
                failed = true;
                println!("  {:<24} ERROR  {err}", case.case_id);
            }
        }
        let _ = io::stdout().flush();
    }
    if failed {
        println!("\none or more grades failed");
        return Ok(1);
    }

    let report = metrics::evaluate(catalog, &grades, &THRESHOLDS);
    print_report(&report);
    let (jev_report, agreement) = if health.jev_configured {
        grade_jev(base, catalog, session_dir, &grades, &mut rows)
    } else {
        println!("\nJev skipped — set OPENROUTER_API_KEY to compare on the same cases");
        (None, None)
    };
    if let Err(err) = session::export(
        session_dir,
        &rows,
        &report,
        &health.backend,
        &health.model_id,
        catalog.threshold_default,
        jev_report.as_ref(),
        agreement.as_ref(),
    ) {
        eprintln!("export: {err}");
    }

    let sources = metrics::sources_of(&grades);
    if sources.iter().all(|source| source == "scripted") {
        let ok = (report.overall.accuracy - 1.0).abs() < 1e-9
            && report.pair_failures.is_empty()
            && report.pairs_pending == 0
            && report.coherence.is_empty()
            && report.overall.n == 73;
        if ok {
            println!("self-check passed (harness agrees with the scripted labels)");
            Ok(0)
        } else {
            println!("self-check failed");
            if !report.pair_failures.is_empty() {
                println!("pairs: {}", report.pair_failures.join("; "));
            }
            if !report.coherence.is_empty() {
                println!("coherence: {}", report.coherence.join("; "));
            }
            Ok(1)
        }
    } else if sources.iter().any(|source| source == "model") {
        println!("checkpoint run — accuracy is the suite result, not a harness check");
        Ok(0)
    } else {
        println!("unexpected grade source {sources:?}; scripted cases should not fall through to the heuristic");
        Ok(1)
    }
}

fn grade_jev(
    base: &str,
    catalog: &Catalog,
    session_dir: &Path,
    laya: &BTreeMap<String, GradeResponse>,
    rows: &mut Vec<SessionRow>,
) -> (Option<Report>, Option<Agreement>) {
    println!("\nJev  OpenRouter Decisions API");
    let mut grades: BTreeMap<String, GradeResponse> = BTreeMap::new();
    for case in &catalog.cases {
        match api::grade_jev(base, case) {
            Ok(grade) => {
                let cost = grade
                    .cost_usd
                    .map(|value| format!("  ${value:.4}"))
                    .unwrap_or_default();
                println!(
                    "  {:<24} {:>7.0} ms  {}{cost}",
                    case.case_id, grade.latency_ms, grade.model_id
                );
                let row = session::row_from(
                    "check",
                    case,
                    &grade,
                    catalog.threshold_default,
                    false,
                    "jev",
                );
                if let Err(err) = session::append(&session_dir.join("live.jsonl"), &row) {
                    eprintln!("log: {err}");
                }
                rows.push(row);
                grades.insert(case.case_id.clone(), grade);
            }
            Err(err) => println!("  {:<24} ERROR  {err}", case.case_id),
        }
        let _ = io::stdout().flush();
    }
    if grades.is_empty() {
        println!("Jev returned no grades. Laya-CDA results above are unchanged.");
        return (None, None);
    }
    let report = metrics::evaluate(catalog, &grades, &THRESHOLDS);
    print_report(&report);
    let agree = metrics::agreement(catalog, laya, &grades);
    println!(
        "agree {:.3}  ({}/{})  same decision label",
        agree.rate, agree.same, agree.n
    );
    for row in agree.disagreements.iter().take(12) {
        println!("  {row}");
    }
    (Some(report), Some(agree))
}

fn print_report(report: &Report) {
    println!();
    let overall = &report.overall;
    println!(
        "questions {}   acc {:.3}   soft {:.3}   brier {:.3}   ece {:.3}   mae {}   p50 {:.1} ms",
        overall.n,
        overall.accuracy,
        overall.soft_accuracy,
        overall.brier,
        overall.ece,
        overall
            .score_mae
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "—".into()),
        report.latency_ms_p50
    );
    println!(
        "{:<16} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "family", "n", "acc", "soft", "brier", "ece", "mae"
    );
    for (family, metrics) in &report.by_family {
        println!(
            "{:<16} {:>5} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7}",
            family,
            metrics.n,
            metrics.accuracy,
            metrics.soft_accuracy,
            metrics.brier,
            metrics.ece,
            metrics
                .score_mae
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "—".into())
        );
    }
    println!(
        "pairs {}/{}  pending {}  coherence {}",
        report.pairs_passed,
        report.pairs_passed + report.pairs_failed,
        report.pairs_pending,
        report.coherence.len()
    );
    println!("{:<8} {:>8} {:>12} {:>10}", "thr", "cover", "auto", "cond");
    for row in &report.sweep {
        println!(
            "{:<8.2} {:>8.3} {:>12} {:>10}",
            row.threshold,
            row.coverage,
            format!("{}/{}", row.auto, row.gated),
            row.conditional_accuracy
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "—".into())
        );
    }
}
