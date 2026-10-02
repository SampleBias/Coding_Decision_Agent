//! Four screens: Grade, Duel, Replay, Study.
//!
//! Bars are the probability the sidecar returned. The highlighted label is the
//! decision `metrics::decision_label` scores, which is the stated choice for a
//! choice question and the argmax for a score.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Tabs, Wrap};
use ratatui::Frame;

use crate::api::GradeResponse;
use crate::app::{App, Screen};
use crate::catalog::Case;
use crate::metrics::{self, Report};

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    if area.width < 76 || area.height < 16 {
        frame.render_widget(
            Paragraph::new("Resize the terminal to at least 76×16."),
            area,
        );
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(8),
        Constraint::Length(1),
    ])
    .split(area);
    draw_header(frame, app, rows[0]);
    draw_tabs(frame, app, rows[1]);
    match app.screen {
        Screen::Grade => draw_grade(frame, app, rows[2]),
        Screen::Duel => draw_duel(frame, app, rows[2]),
        Screen::Replay => draw_replay(frame, app, rows[2]),
        Screen::Study => draw_study(frame, app, rows[2]),
    }
    draw_status(frame, app, rows[3]);
    if app.help {
        draw_help(frame, area);
    }
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let (model, device, jev_name) = match &app.health {
        Some(health) if health.jev_configured => (
            health.model_id.clone(),
            health.device.clone(),
            health
                .jev_model
                .trim_start_matches("~typesafe/")
                .to_string(),
        ),
        Some(health) => (health.model_id.clone(), health.device.clone(), "off".into()),
        None => ("offline".into(), "—".into(), "off".into()),
    };
    let blind = if app.blind { "blind" } else { "show" };
    let tail = format!("{device}  {blind}  thr {:.2}", app.threshold);
    let brand = " Laya-CDA ";
    let rest = " vs Jev ";
    let used = brand.chars().count()
        + rest.chars().count()
        + model.chars().count()
        + jev_name.chars().count()
        + tail.chars().count()
        + 8;
    let pad = (area.width as usize).saturating_sub(used);
    let line = Line::from(vec![
        Span::styled(
            brand,
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            rest,
            Style::default()
                .fg(Color::Black)
                .bg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" ".repeat(pad)),
        Span::styled("Laya ", laya_name()),
        Span::styled(format!("{model}  "), dim()),
        Span::styled("Jev ", jev_name_style()),
        Span::styled(format!("{jev_name}  {tail}"), dim()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_tabs(frame: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = [Screen::Grade, Screen::Duel, Screen::Replay, Screen::Study]
        .into_iter()
        .map(|screen| Line::from(format!(" {} ", screen.title())))
        .collect();
    let tabs = Tabs::new(titles)
        .select(app.screen.index())
        .divider(" ")
        .style(dim())
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(tabs, area);
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let hint = "1-4 screen  j/k move  g grade  v reveal  b blind  t thr  m state  e export  ? help  q quit";
    let left = format!(" {}", app.status);
    let room = (area.width as usize).saturating_sub(left.chars().count() + 1);
    let hint = fit_end(&hint, room);
    let pad = (area.width as usize).saturating_sub(left.chars().count() + hint.chars().count());
    let style = if app.health.is_none() {
        Style::default().fg(Color::Yellow)
    } else {
        dim()
    };
    let line = Line::from(vec![
        Span::styled(left, style),
        Span::raw(" ".repeat(pad)),
        Span::styled(hint, dim()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_grade(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::horizontal([
        Constraint::Length(if area.width >= 110 { 32 } else { 24 }),
        Constraint::Min(24),
        Constraint::Min(36),
    ])
    .split(area);
    draw_case_list(frame, app, cols[0]);
    if let Some(case) = app.selected_case() {
        let grade = app.grades.get(&case.case_id);
        draw_state(frame, app, cols[1], case, grade);
        draw_verdict(frame, app, cols[2], case, grade);
    }
}

fn draw_duel(frame: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([Constraint::Min(8), Constraint::Length(5)]).split(area);
    let cols = Layout::horizontal([
        Constraint::Length(if area.width >= 110 { 28 } else { 22 }),
        Constraint::Min(26),
        Constraint::Min(26),
    ])
    .split(rows[0]);
    draw_pair_list(frame, app, cols[0]);
    if let Some(pair) = app.selected_pair() {
        if let Some(left) = app.catalog.case(&pair.left) {
            draw_duel_column(frame, app, cols[1], left, "Left");
        }
        if let Some(right) = app.catalog.case(&pair.right) {
            draw_duel_column(frame, app, cols[2], right, "Right");
        }
        draw_delta(frame, app, rows[1]);
    } else {
        frame.render_widget(
            Paragraph::new(" This catalog has no contrast pairs.").block(pane("Duel")),
            rows[0],
        );
    }
}

fn draw_replay(frame: &mut Frame, app: &App, area: Rect) {
    let Some(replay) = app.selected_replay() else {
        frame.render_widget(
            Paragraph::new(" This catalog has no replay.").block(pane("Replay")),
            area,
        );
        return;
    };
    let cols = Layout::horizontal([Constraint::Length(36), Constraint::Min(30)]).split(area);
    let mut lines = vec![
        Line::from(Span::styled(
            replay.title.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(" stamp is Laya-CDA/Jev", dim())),
    ];
    for (index, step) in replay.steps.iter().enumerate() {
        let case = app.catalog.case(&step.case_id);
        let grade = app.grades.get(&step.case_id);
        let jev = app.jev_grades.get(&step.case_id);
        let stamp = match (case, grade, app.hidden()) {
            (_, _, true) if grade.is_some() => "????".to_string(),
            (Some(case), Some(grade), _) => {
                let laya = stamp_word(&case.family, grade, app.threshold).0;
                let other = jev
                    .map(|grade| stamp_word(&case.family, grade, app.threshold).0)
                    .unwrap_or_else(|| "·".into());
                format!("{laya}/{other}")
            }
            _ => "·".to_string(),
        };
        let text = format!(" {:>2}  {:<12} {}", index + 1, stamp, step.caption);
        let style = if index == app.step_ix {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        lines.push(Line::from(Span::styled(text, style)));
        if let Some(case) = case {
            lines.push(Line::from(Span::styled(
                format!("     {}", case.title),
                dim(),
            )));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(" g grades every step", dim())));
    lines.push(Line::from(Span::styled(" enter grades this step", dim())));
    frame.render_widget(Paragraph::new(lines).block(pane("Replay")), cols[0]);

    if let Some(step) = replay.steps.get(app.step_ix) {
        if let Some(case) = app.catalog.case(&step.case_id) {
            let detail = Layout::vertical([Constraint::Percentage(42), Constraint::Percentage(58)])
                .split(cols[1]);
            let grade = app.grades.get(&case.case_id);
            draw_state(frame, app, detail[0], case, grade);
            draw_verdict(frame, app, detail[1], case, grade);
        }
    }
}

fn draw_study(frame: &mut Frame, app: &App, area: Rect) {
    let title = if app.study_page == 0 {
        "Study  ·  summary"
    } else {
        "Study  ·  questions"
    };
    let block = pane(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 2 || inner.width < 8 {
        return;
    }
    let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(inner);
    if app.hidden() && !app.study_ids.is_empty() {
        frame.render_widget(
            Paragraph::new("\n Blind is on. Press v to reveal the scoreboard.\n The grades are already in the session log."),
            rows[0],
        );
        frame.render_widget(Paragraph::new(" Hidden until v."), rows[1]);
        return;
    }
    let grades = app.study_grades();
    if grades.is_empty() {
        let cached = app.grades.len();
        let body = format!(
            "\n Press g to grade all {} cases through the sidecar.\n\n {} case(s) already graded on other screens.\n Those stay out of this table until you run Study.\n\n n questions   p summary",
            app.catalog.cases.len(),
            cached
        );
        frame.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), rows[0]);
    } else {
        let report = metrics::evaluate(&app.catalog, &grades, &metrics::THRESHOLDS);
        let lines = if app.study_page == 0 {
            summary_lines(app, &report)
        } else {
            question_lines(&report)
        };
        let start = app.study_scroll as usize;
        let visible: Vec<Line> = lines.into_iter().skip(start).collect();
        frame.render_widget(Paragraph::new(visible).wrap(Wrap { trim: false }), rows[0]);
    }
    let (sentence, style) = study_takeaway(app);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(format!(" {sentence}"), style))),
        rows[1],
    );
}

fn study_takeaway(app: &App) -> (String, Style) {
    let laya = app.study_grades();
    let jev = app.study_jev_grades();
    let laya_score = scored(&app.catalog, &laya);
    let jev_score = scored(&app.catalog, &jev);
    let text = metrics::accuracy_sentence(laya_score, jev_score);
    let style = match (laya_score, jev_score) {
        (Some((laya_acc, _)), Some((jev_acc, _))) if laya_acc > jev_acc + 0.0005 => laya_name(),
        (Some((laya_acc, _)), Some((jev_acc, _))) if jev_acc > laya_acc + 0.0005 => {
            jev_name_style()
        }
        (Some(_), Some(_)) => Style::default().add_modifier(Modifier::BOLD),
        _ => dim(),
    };
    (text, style)
}

fn scored(
    catalog: &crate::catalog::Catalog,
    grades: &std::collections::BTreeMap<String, GradeResponse>,
) -> Option<(f64, usize)> {
    if grades.is_empty() {
        return None;
    }
    let report = metrics::evaluate(catalog, grades, &metrics::THRESHOLDS);
    Some((report.overall.accuracy, report.overall.n))
}

fn summary_lines(app: &App, report: &Report) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let banner = match app.health.as_ref().map(|health| health.backend.as_str()) {
        Some("mock") => "scripted mock — these numbers check the harness, not the checkpoint",
        Some("real") => "checkpoint — suite result, separate from the training split",
        _ => "waiting for the sidecar",
    };
    lines.push(Line::from(Span::styled(banner.to_string(), dim())));
    let overall = &report.overall;
    lines.push(Line::from(format!(
        "Laya-CDA  cases {}   questions {}   acc {:.3}   soft {:.3}   brier {:.3}   ece {:.3}   mae {}   p50 {:.1} ms",
        report.n_cases,
        overall.n,
        overall.accuracy,
        overall.soft_accuracy,
        overall.brier,
        overall.ece,
        report
            .overall
            .score_mae
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "—".into()),
        report.latency_ms_p50
    )));
    let pair_style = if report.pair_failures.is_empty() && report.pairs_pending == 0 {
        green()
    } else {
        yellow()
    };
    lines.push(Line::from(Span::styled(
        format!(
            "pairs {}/{}   pending {}   coherence {}",
            report.pairs_passed,
            report.pairs_passed + report.pairs_failed,
            report.pairs_pending,
            report.coherence.len()
        ),
        pair_style,
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(
            " {:<16} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "family", "n", "acc", "soft", "brier", "ece", "mae"
        ),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    for (family, metrics) in &report.by_family {
        lines.push(Line::from(format!(
            " {:<16} {:>5} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7}",
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
        )));
    }
    lines.extend(jev_summary_lines(app));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " gate coverage is code_review=accept, tool_call=execute, agent_trace=continue. routing is not gated.",
        dim(),
    )));
    lines.push(Line::from(Span::styled(
        format!(
            " {:<7} {:>8} {:>10} {:>10}",
            "thr", "cover", "auto", "cond acc"
        ),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    for row in &report.sweep {
        let mark = if (row.threshold - app.threshold).abs() < 1e-9 {
            ">"
        } else {
            " "
        };
        let auto = format!("{}/{}", row.auto, row.gated);
        let cond = row
            .conditional_accuracy
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "—".into());
        lines.push(Line::from(format!(
            "{mark}{:<7.2} {:>8.3} {auto:>10} {cond:>10}",
            row.threshold, row.coverage
        )));
    }
    if !report.pair_failures.is_empty() || !report.coherence.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" failures", red())));
        for failure in report
            .pair_failures
            .iter()
            .chain(report.coherence.iter())
            .take(8)
        {
            lines.push(Line::from(Span::styled(format!(" {failure}"), red())));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " n per-question table    p summary    j/k scroll",
        dim(),
    )));
    lines
}

fn question_lines(report: &Report) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(
        format!(
            " {:<32} {:>5} {:>7} {:>7} {:>7}",
            "question", "n", "acc", "brier", "ece"
        ),
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    for (question, metrics) in &report.by_question {
        lines.push(Line::from(format!(
            " {:<32} {:>5} {:>7.3} {:>7.3} {:>7.3}",
            question, metrics.n, metrics.accuracy, metrics.brier, metrics.ece
        )));
    }
    lines.push(Line::from(Span::styled(" p summary", dim())));
    lines
}

fn draw_case_list(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines = Vec::new();
    for (index, case) in app.catalog.cases.iter().enumerate() {
        let grade = app.grades.get(&case.case_id);
        let (mark, mark_style) = list_mark(app, case, grade);
        let title = fit(&case.title, (area.width as usize).saturating_sub(12));
        let style = if index == app.case_ix {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {mark}"), mark_style),
            Span::styled(format!(" {:<5} {title}", short_family(&case.family)), style),
        ]));
    }
    let start = window_start(
        app.catalog.cases.len(),
        app.case_ix,
        area.height.saturating_sub(2) as usize,
    );
    let visible: Vec<Line> = lines.into_iter().skip(start).collect();
    frame.render_widget(Paragraph::new(visible).block(pane("Cases")), area);
}

fn draw_pair_list(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines = Vec::new();
    for (index, pair) in app.catalog.pairs.iter().enumerate() {
        let mark = match metrics::pair_ok(&app.catalog, index, &app.grades) {
            Some(true) if !app.hidden() => ("+", green()),
            Some(false) if !app.hidden() => ("x", red()),
            Some(_) => ("?", dim()),
            None => ("·", dim()),
        };
        let style = if index == app.pair_ix {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        let title = fit(&pair.title, (area.width as usize).saturating_sub(6));
        lines.push(Line::from(vec![
            Span::styled(format!(" {}", mark.0), mark.1),
            Span::styled(format!(" {title}"), style),
        ]));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(pane("Contrasts"))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_duel_column(frame: &mut Frame, app: &App, area: Rect, case: &Case, side: &str) {
    let rows = Layout::vertical([Constraint::Length(8), Constraint::Min(4)]).split(area);
    let preview = preview_lines(case);
    frame.render_widget(
        Paragraph::new(preview)
            .block(pane(&format!("{side}  {}", case.title)))
            .wrap(Wrap { trim: false }),
        rows[0],
    );
    draw_verdict(frame, app, rows[1], case, app.grades.get(&case.case_id));
}

fn draw_delta(frame: &mut Frame, app: &App, area: Rect) {
    let Some(pair) = app.selected_pair() else {
        return;
    };
    if app.hidden() && app.grades.contains_key(&pair.left) {
        frame.render_widget(
            Paragraph::new(" hidden — press v to reveal the contrast").block(pane("Contrast")),
            area,
        );
        return;
    }
    if app.catalog.case(&pair.left).is_none() || app.catalog.case(&pair.right).is_none() {
        return;
    }
    let (left, right) = (app.grades.get(&pair.left), app.grades.get(&pair.right));
    let (Some(left), Some(right)) = (left, right) else {
        frame.render_widget(
            Paragraph::new(" press g to grade both sides").block(pane("Contrast")),
            area,
        );
        return;
    };
    let Some(family) = app.catalog.family(&left.family) else {
        return;
    };
    let mut lines = vec![Line::from(Span::styled(
        pair.title.clone(),
        Style::default().add_modifier(Modifier::BOLD),
    ))];
    for question in &family.questions {
        let Some(l) = left.answers.get(&question.id) else {
            continue;
        };
        let Some(r) = right.answers.get(&question.id) else {
            continue;
        };
        let ll = metrics::decision_label(l, &question.options);
        let rr = metrics::decision_label(r, &question.options);
        if question.qtype == "score" {
            let el = metrics::expected_score(&l.probabilities, &question.options);
            let er = metrics::expected_score(&r.probabilities, &question.options);
            let constrained_left = pair.higher_on_left.iter().any(|qid| qid == &question.id);
            let constrained_right = pair.higher_on_right.iter().any(|qid| qid == &question.id);
            let style = if constrained_left {
                if el > er {
                    green()
                } else {
                    red()
                }
            } else if constrained_right {
                if er > el {
                    green()
                } else {
                    red()
                }
            } else {
                dim()
            };
            lines.push(Line::from(Span::styled(
                format!(" {:<20} {el:.2} → {er:.2}", question.id),
                style,
            )));
        } else if ll != rr
            || pair.prefer_on_left.contains_key(&question.id)
            || pair.prefer_on_right.contains_key(&question.id)
        {
            let left_ok = pair
                .prefer_on_left
                .get(&question.id)
                .map(|want| want == ll)
                .unwrap_or(true);
            let right_ok = pair
                .prefer_on_right
                .get(&question.id)
                .map(|want| want == rr)
                .unwrap_or(true);
            let style = if pair.prefer_on_left.contains_key(&question.id)
                || pair.prefer_on_right.contains_key(&question.id)
            {
                if left_ok && right_ok {
                    green()
                } else {
                    red()
                }
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(
                format!(" {:<20} {ll} → {rr}", question.id),
                style,
            )));
        }
    }
    frame.render_widget(Paragraph::new(lines).block(pane("Contrast")), area);
}

fn draw_state(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    case: &Case,
    grade: Option<&GradeResponse>,
) {
    let model = app.show_model_state && grade.is_some();
    let title = if model {
        let chars = grade
            .map(|grade| {
                serde_json::to_string(&grade.state)
                    .map(|text| text.len())
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        format!("Model state  {chars} chars")
    } else {
        case.title.clone()
    };
    let lines = if model {
        let pretty = serde_json::to_string_pretty(&grade.unwrap().state).unwrap_or_default();
        pretty
            .lines()
            .map(|line| Line::from(line.to_string()))
            .collect()
    } else {
        field_lines(case)
    };
    frame.render_widget(
        Paragraph::new(lines)
            .block(pane(&title))
            .wrap(Wrap { trim: false })
            .scroll((app.state_scroll, 0)),
        area,
    );
}

fn draw_verdict(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    case: &Case,
    grade: Option<&GradeResponse>,
) {
    let block = pane(&format!("{}   Laya-CDA vs Jev", case.family));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 2 || inner.width < 8 {
        return;
    }
    if grade.is_some() && app.hidden() {
        frame.render_widget(Paragraph::new("\n hidden\n press v to reveal"), inner);
        return;
    }
    let Some(grade) = grade else {
        frame.render_widget(Paragraph::new("\n press g to grade both"), inner);
        return;
    };
    let jev = app.jev_grades.get(&case.case_id);
    let jev_error = app.jev_errors.get(&case.case_id).map(String::as_str);
    let stamp = compare_stamps(app, case, grade, jev, jev_error);
    let bits = Layout::vertical([Constraint::Length(3), Constraint::Min(1)]).split(inner);
    frame.render_widget(Paragraph::new(stamp), bits[0]);
    let lines = if bits[1].width >= 46 {
        bar_lines(app, case, grade, jev, bits[1].width)
    } else {
        compact_lines(app, case, grade, jev)
    };
    frame.render_widget(
        Paragraph::new(lines).scroll((app.verdict_scroll, 0)),
        bits[1],
    );
}

fn jev_summary_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from("")];
    let jev_grades = app.study_jev_grades();
    if jev_grades.is_empty() {
        let note = match app.health.as_ref() {
            Some(health) if health.jev_configured => {
                "Jev is configured. Press g to grade it on the same cases."
            }
            _ => "Jev off. Set OPENROUTER_API_KEY to compare on the same cases.",
        };
        lines.push(Line::from(Span::styled(note.to_string(), dim())));
        return lines;
    }
    let report = metrics::evaluate(&app.catalog, &jev_grades, &metrics::THRESHOLDS);
    let laya = app.study_grades();
    let agree = metrics::agreement(&app.catalog, &laya, &jev_grades);
    let cost: f64 = jev_grades.values().filter_map(|grade| grade.cost_usd).sum();
    lines.push(Line::from(Span::styled(
        format!(
            " Jev   questions {}   acc {:.3}   soft {:.3}   brier {:.3}   ece {:.3}   p50 {:.0} ms   ${cost:.4}",
            report.overall.n,
            report.overall.accuracy,
            report.overall.soft_accuracy,
            report.overall.brier,
            report.overall.ece,
            report.latency_ms_p50
        ),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    let style = if agree.n > 0 && agree.rate >= 0.8 {
        green()
    } else if agree.n > 0 {
        yellow()
    } else {
        dim()
    };
    lines.push(Line::from(Span::styled(
        format!(
            " agree {:.3}  ({}/{})  Laya-CDA and Jev chose the same label",
            agree.rate, agree.same, agree.n
        ),
        style,
    )));
    for row in agree.disagreements.iter().take(8) {
        lines.push(Line::from(Span::styled(format!("  {row}"), yellow())));
    }
    if agree.disagreements.len() > 8 {
        lines.push(Line::from(Span::styled(
            format!("  {} more disagreements", agree.disagreements.len() - 8),
            dim(),
        )));
    }
    lines
}

fn compare_stamps(
    app: &App,
    case: &Case,
    grade: &GradeResponse,
    jev: Option<&GradeResponse>,
    jev_error: Option<&str>,
) -> Vec<Line<'static>> {
    let (word, style, detail) = stamp_word(&case.family, grade, app.threshold);
    let hits = gold_hits(app, case, grade);
    let note = if grade.source != "model" {
        grade.note.clone()
    } else {
        format!("{}  {:.1} ms", grade.model_id, grade.latency_ms)
    };
    vec![
        Line::from(vec![
            Span::raw(" "),
            Span::styled("Laya-CDA", laya_name()),
            Span::styled(format!("  {word}"), style),
            Span::raw(format!("  {detail}")),
        ]),
        jev_stamp_line(app, case, jev, jev_error),
        Line::from(Span::styled(format!(" {hits}  {note}"), dim())),
    ]
}

fn jev_stamp_line(
    app: &App,
    case: &Case,
    jev: Option<&GradeResponse>,
    jev_error: Option<&str>,
) -> Line<'static> {
    let Some(grade) = jev else {
        let text = match jev_error {
            Some(err) => err.to_string(),
            None => "set OPENROUTER_API_KEY".into(),
        };
        return Line::from(vec![
            Span::raw(" "),
            Span::styled("Jev     ", jev_name_style()),
            Span::styled(format!("  {text}"), yellow()),
        ]);
    };
    let (word, style, detail) = stamp_word(&case.family, grade, app.threshold);
    let cost = grade
        .cost_usd
        .map(|value| format!("  ${value:.4}"))
        .unwrap_or_default();
    Line::from(vec![
        Span::raw(" "),
        Span::styled("Jev     ", jev_name_style()),
        Span::styled(format!("  {word}"), style),
        Span::raw(format!("  {detail}")),
        Span::styled(format!("  {:.0} ms{cost}", grade.latency_ms), dim()),
    ])
}

fn stamp_word(family: &str, grade: &GradeResponse, threshold: f64) -> (String, Style, String) {
    let gate_answer = |qid: &str| grade.answers.get(qid);
    match family {
        "code_review" => {
            let Some(answer) = gate_answer("merge_action") else {
                return missing_stamp();
            };
            let detail = format!("{}  {:.2}", answer.label, answer.answer_confidence);
            if answer.label == "accept" && answer.answer_confidence >= threshold {
                ("PASS".into(), green(), detail)
            } else if answer.label == "reject" {
                ("REJECT".into(), red(), detail)
            } else {
                ("HOLD".into(), yellow(), detail)
            }
        }
        "tool_call" => {
            let Some(answer) = gate_answer("call_verdict") else {
                return missing_stamp();
            };
            let detail = format!("{}  {:.2}", answer.label, answer.answer_confidence);
            if answer.label == "execute" && answer.answer_confidence >= threshold {
                ("RUN".into(), green(), detail)
            } else if answer.label == "fix_args" {
                ("FIX".into(), yellow(), detail)
            } else if answer.label == "execute" {
                (
                    "HOLD".into(),
                    yellow(),
                    format!("execute {detail} below {threshold:.2}"),
                )
            } else {
                ("HOLD".into(), yellow(), detail)
            }
        }
        "agent_trace" => {
            let Some(answer) = gate_answer("action") else {
                return missing_stamp();
            };
            let detail = format!("{}  {:.2}", answer.label, answer.answer_confidence);
            match answer.label.as_str() {
                "stop" => ("HALT".into(), red(), detail),
                "human_review" => ("REVIEW".into(), yellow(), detail),
                "continue" if answer.answer_confidence >= threshold => {
                    ("RUN".into(), green(), detail)
                }
                "observe" => ("WATCH".into(), cyan(), detail),
                "continue" => (
                    "HOLD".into(),
                    yellow(),
                    format!(
                        "continue {:.2} below {threshold:.2}",
                        answer.answer_confidence
                    ),
                ),
                _ => ("HOLD".into(), yellow(), detail),
            }
        }
        "routing" => {
            let Some(answer) = gate_answer("model_tier") else {
                return missing_stamp();
            };
            let word = match answer.label.as_str() {
                "small_fast" => "SMALL",
                "mid" => "MID",
                "frontier" => "FRONTIER",
                "reasoning" => "REASON",
                other => other,
            };
            let style = if answer.answer_confidence >= threshold {
                green()
            } else {
                yellow()
            };
            (
                word.into(),
                style,
                format!("{:.2}", answer.answer_confidence),
            )
        }
        _ => missing_stamp(),
    }
}

fn missing_stamp() -> (String, Style, String) {
    ("—".into(), dim(), "no answer".into())
}

fn compact_lines(
    app: &App,
    case: &Case,
    grade: &GradeResponse,
    jev: Option<&GradeResponse>,
) -> Vec<Line<'static>> {
    let Some(family) = app.catalog.family(&case.family) else {
        return Vec::new();
    };
    let mut lines = vec![Line::from(vec![
        Span::styled("Laya-CDA", laya_name()),
        Span::raw(" / "),
        Span::styled("Jev", jev_name_style()),
    ])];
    for question in &family.questions {
        let laya = grade
            .answers
            .get(&question.id)
            .map(|answer| metrics::decision_label(answer, &question.options).to_string());
        let other = jev.and_then(|grade| {
            grade
                .answers
                .get(&question.id)
                .map(|answer| metrics::decision_label(answer, &question.options).to_string())
        });
        let laya_txt = laya.as_deref().unwrap_or("—");
        let jev_txt = other.as_deref().unwrap_or("—");
        let gold = case.gold.get(&question.id).map(String::as_str);
        let style = match gold {
            Some(gold) if Some(gold) == laya.as_deref() => green(),
            Some(_) => red(),
            None => Style::default(),
        };
        lines.push(Line::from(Span::styled(
            format!("{}  {laya_txt} / {jev_txt}", question.id),
            style,
        )));
    }
    lines
}

fn bar_lines(
    app: &App,
    case: &Case,
    grade: &GradeResponse,
    jev: Option<&GradeResponse>,
    width: u16,
) -> Vec<Line<'static>> {
    let Some(family) = app.catalog.family(&case.family) else {
        return Vec::new();
    };
    let mut lines = vec![Line::from(Span::styled(
        "bars are Laya-CDA. The Jev line under each question is Jev's label.",
        dim(),
    ))];
    for question in &family.questions {
        let Some(answer) = grade.answers.get(&question.id) else {
            lines.push(Line::from(Span::styled(
                format!("{}  missing", question.id),
                dim(),
            )));
            continue;
        };
        let predicted = metrics::decision_label(answer, &question.options);
        let gold = case.gold.get(&question.id).map(String::as_str);
        let hit = gold.map(|gold| gold == predicted);
        let head_style = match hit {
            Some(true) => green(),
            Some(false) => red(),
            None => Style::default().add_modifier(Modifier::BOLD),
        };
        let extra = if question.qtype == "score" {
            format!(
                "  E {:.2}",
                metrics::expected_score(&answer.probabilities, &question.options)
            )
        } else {
            String::new()
        };
        let gold_txt = gold
            .map(|gold| format!("  gold {gold}"))
            .unwrap_or_default();
        lines.push(Line::from(vec![
            Span::styled(
                format!("{qid}  Laya-CDA {predicted}", qid = question.id),
                head_style,
            ),
            Span::styled(
                format!("  {:.2}{extra}{gold_txt}", answer.answer_confidence),
                dim(),
            ),
        ]));
        if let Some(jev_answer) = jev.and_then(|grade| grade.answers.get(&question.id)) {
            let jev_label = metrics::decision_label(jev_answer, &question.options);
            let same = jev_label == predicted;
            lines.push(Line::from(Span::styled(
                format!(
                    "  Jev       {jev_label}  {:.2}",
                    jev_answer.answer_confidence
                ),
                if same { dim() } else { yellow() },
            )));
        }
        let name_width = 16usize.min((width as usize).saturating_sub(12));
        let bar_width = (width as usize).saturating_sub(name_width + 8);
        for (index, option) in question.options.iter().enumerate() {
            let probability = answer.probabilities.get(option).copied().unwrap_or(0.0);
            let gloss = question
                .gloss
                .get(index)
                .map(String::as_str)
                .unwrap_or(option);
            let name = fit(&format!("{option} {gloss}"), name_width);
            let chosen = option == predicted;
            let style = if chosen {
                if answer.answer_confidence >= app.threshold {
                    green()
                } else {
                    yellow()
                }
            } else {
                dim()
            };
            lines.push(Line::from(Span::styled(
                format!(" {name} {probability:>4.2} {}", bar(probability, bar_width)),
                style,
            )));
        }
    }
    lines
}

fn field_lines(case: &Case) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if !case.blurb.is_empty() {
        lines.push(Line::from(Span::styled(case.blurb.clone(), dim())));
        lines.push(Line::from(""));
    }
    let fields = match case.fields.as_object() {
        Some(map) => map,
        None => return lines,
    };
    let order = [
        "task",
        "request",
        "language",
        "diff",
        "code",
        "tests",
        "notes",
        "tools",
        "proposed_calls",
        "history",
        "agent",
        "constraints",
        "trace_summary",
        "recent_steps",
        "context",
    ];
    let mut seen = std::collections::BTreeSet::new();
    for key in order {
        if let Some(value) = fields.get(key) {
            seen.insert(key);
            push_field(&mut lines, key, value);
        }
    }
    for (key, value) in fields {
        if !seen.contains(key.as_str()) {
            push_field(&mut lines, key, value);
        }
    }
    lines
}

fn push_field(lines: &mut Vec<Line<'static>>, key: &str, value: &serde_json::Value) {
    lines.push(Line::from(Span::styled(
        key.to_string(),
        Style::default().fg(Color::Cyan),
    )));
    match value {
        serde_json::Value::String(text) if text.is_empty() => {
            lines.push(Line::from(Span::styled("  (empty)".to_string(), dim())));
        }
        serde_json::Value::String(text) => {
            let diff = key == "diff" || key == "code";
            for line in text.lines() {
                let style = if diff {
                    diff_style(line)
                } else {
                    Style::default()
                };
                lines.push(Line::from(Span::styled(line.to_string(), style)));
            }
        }
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                lines.push(Line::from(Span::styled("  (none)".to_string(), dim())));
            }
            for item in items {
                lines.push(Line::from(format!("  {}", compact_json(item))));
            }
        }
        other => lines.push(Line::from(format!("  {}", compact_json(other)))),
    }
    lines.push(Line::from(""));
}

fn preview_lines(case: &Case) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from(Span::styled(case.blurb.clone(), dim()))];
    let fields = case.fields.as_object();
    if let Some(text) = fields
        .and_then(|map| map.get("task").or_else(|| map.get("request")))
        .and_then(|v| v.as_str())
    {
        lines.push(Line::from(fit(text, 80)));
    }
    if let Some(diff) = fields
        .and_then(|map| map.get("diff"))
        .and_then(|v| v.as_str())
    {
        for line in diff.lines().take(5) {
            lines.push(Line::from(Span::styled(line.to_string(), diff_style(line))));
        }
    }
    if let Some(summary) = fields
        .and_then(|map| map.get("trace_summary"))
        .and_then(|v| v.as_str())
    {
        lines.push(Line::from(fit(summary, 80)));
    }
    lines
}

fn compact_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let name = map.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let role = map.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = map.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if !role.is_empty() {
                return format!("{role}: {content}");
            }
            if !name.is_empty() {
                let args = map
                    .get("arguments")
                    .or_else(|| map.get("parameters"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                return format!(
                    "{name} {}",
                    serde_json::to_string(&args).unwrap_or_default()
                );
            }
            serde_json::to_string(value).unwrap_or_default()
        }
        serde_json::Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn list_mark(app: &App, case: &Case, grade: Option<&GradeResponse>) -> (&'static str, Style) {
    let Some(grade) = grade else {
        return ("·", dim());
    };
    if app.hidden() {
        return ("?", dim());
    }
    let Some(family) = app.catalog.family(&case.family) else {
        return ("·", dim());
    };
    let mut saw_gold = false;
    let mut all = true;
    for question in &family.questions {
        let Some(gold) = case.gold.get(&question.id) else {
            continue;
        };
        let Some(answer) = grade.answers.get(&question.id) else {
            continue;
        };
        saw_gold = true;
        if metrics::decision_label(answer, &question.options) != gold {
            all = false;
        }
    }
    if !saw_gold {
        ("·", dim())
    } else if all {
        ("+", green())
    } else {
        ("x", red())
    }
}

fn gold_hits(app: &App, case: &Case, grade: &GradeResponse) -> String {
    let Some(family) = app.catalog.family(&case.family) else {
        return String::new();
    };
    let mut hit = 0usize;
    let mut total = 0usize;
    for question in &family.questions {
        let Some(gold) = case.gold.get(&question.id) else {
            continue;
        };
        let Some(answer) = grade.answers.get(&question.id) else {
            continue;
        };
        total += 1;
        if metrics::decision_label(answer, &question.options) == gold {
            hit += 1;
        }
    }
    if total == 0 {
        grade.note.clone()
    } else {
        format!("gold {hit}/{total}")
    }
}

fn diff_style(line: &str) -> Style {
    if line.starts_with("+++")
        || line.starts_with("---")
        || line.starts_with("diff ")
        || line.starts_with("index ")
    {
        dim()
    } else if line.starts_with('+') {
        Style::default().fg(Color::Green)
    } else if line.starts_with('-') {
        Style::default().fg(Color::Red)
    } else if line.starts_with("@@") {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    }
}

fn bar(probability: f64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut cells = vec!['░'; width];
    let exact = probability.clamp(0.0, 1.0) * width as f64;
    let full = (exact.floor() as usize).min(width);
    for cell in cells.iter_mut().take(full) {
        *cell = '█';
    }
    if full < width {
        let eighth = ((exact - full as f64) * 8.0).round() as usize;
        const PART: [char; 9] = ['░', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
        cells[full] = PART[eighth.min(8)];
    }
    cells.into_iter().collect()
}

fn fit(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut out: String = text.chars().take(width).collect();
    if text.chars().count() > width && width > 1 {
        out = text.chars().take(width - 1).collect();
        out.push('…');
    }
    out
}

fn fit_end(text: &str, width: usize) -> String {
    let count = text.chars().count();
    if count <= width {
        return text.to_string();
    }
    text.chars().skip(count - width).collect()
}

fn window_start(len: usize, selected: usize, height: usize) -> usize {
    if height == 0 || len <= height {
        return 0;
    }
    if selected + 1 > height {
        selected + 1 - height
    } else {
        0
    }
}

fn short_family(id: &str) -> &'static str {
    match id {
        "code_review" => "code",
        "tool_call" => "tool",
        "agent_trace" => "trace",
        "routing" => "route",
        _ => "?",
    }
}

fn pane(title: &str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .title(format!(" {title} "))
}

fn green() -> Style {
    Style::default()
        .fg(Color::Green)
        .add_modifier(Modifier::BOLD)
}
fn yellow() -> Style {
    Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD)
}
fn red() -> Style {
    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
}
fn laya_name() -> Style {
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}

fn jev_name_style() -> Style {
    Style::default()
        .fg(Color::Magenta)
        .add_modifier(Modifier::BOLD)
}

fn cyan() -> Style {
    Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD)
}
fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let width = 68.min(area.width.saturating_sub(2));
    let height = 20.min(area.height.saturating_sub(2));
    let popup = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, popup);
    let body = "\
The TUI does not load weights. It grades through the sidecar.

  1-4 / tab     screen: Grade, Duel, Replay, Study
  j k           move the selection
  g             grade Laya-CDA and Jev together
  enter         grade the highlighted replay step with both
  v             reveal a blind grade
  b             hide grades until v
  t             cycle the gate threshold
  m             case fields / compacted model state
  [ ]           scroll the text
  , .           scroll the bars
  n p           study questions / summary
  e             write JSONL and a report under suite/sessions
  r             reconnect
  q             quit

The gate auto-acts on accept, execute, and continue.
Routing has no execute gate. Mock mode checks the harness.
g grades Laya-CDA and Jev on the same questions.
Jev is the OpenRouter Decisions API. Set OPENROUTER_API_KEY.
JEV_MODEL overrides the default ~typesafe/jev-latest.
python3 suite/sidecar/server.py --backend real   tests the model.";
    frame.render_widget(
        Paragraph::new(body)
            .block(pane("Help"))
            .wrap(Wrap { trim: false }),
        popup,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    use crate::api::Health;

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn draws_a_pass_stamp_and_the_study_table() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../cases/catalog.json");
        let catalog = crate::catalog::load(&path).expect("catalog");
        let mut app = App::new(
            catalog,
            "http://127.0.0.1:8765".into(),
            std::env::temp_dir(),
        );
        app.health = Some(Health {
            ok: true,
            backend: "mock".into(),
            model_id: "mock-scripted".into(),
            device: "none".into(),
            n_cases: app.catalog.cases.len(),
            jev_configured: false,
            jev_model: String::new(),
        });
        for case in app.catalog.cases.clone() {
            let grade = metrics::scripted_grade(&app.catalog, &case);
            app.grades.insert(case.case_id.clone(), grade);
            app.study_ids.push(case.case_id);
        }
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let view = buffer_text(&terminal);
        assert!(view.contains("Laya-CDA"), "{view}");
        assert!(view.contains("Jev"), "{view}");
        assert!(view.contains("PASS"), "{view}");
        assert!(view.contains("accept"), "{view}");

        app.screen = Screen::Duel;
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let duel = buffer_text(&terminal);
        assert!(
            duel.contains("Contrast") || duel.contains("accept"),
            "{duel}"
        );

        app.screen = Screen::Replay;
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let replay = buffer_text(&terminal);
        assert!(
            replay.contains("Read src/checkout.py") || replay.contains("RUN"),
            "{replay}"
        );

        app.screen = Screen::Study;
        terminal.draw(|frame| draw(frame, &app)).unwrap();
        let study = buffer_text(&terminal);
        assert!(study.contains("1.000"), "{study}");
        assert!(study.contains("code_review"), "{study}");
        assert!(study.contains("Jev has no grades yet"), "{study}");
    }

    #[test]
    fn a_full_bar_is_solid_and_an_empty_bar_is_not() {
        assert!(bar(1.0, 8).chars().all(|ch| ch == '█'));
        assert!(bar(0.0, 8).chars().all(|ch| ch == '░'));
        assert_eq!(bar(0.5, 4).chars().count(), 4);
    }
}
