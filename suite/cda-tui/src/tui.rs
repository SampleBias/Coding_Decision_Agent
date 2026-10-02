//! Terminal lifecycle and the thread that talks to the sidecar.

use std::io::{self, stdout};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::api;
use crate::app::{Action, App, Job, Msg};
use crate::ui;

pub fn run(mut app: App) -> io::Result<()> {
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    enable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        EnterAlternateScreen,
        crossterm::cursor::Hide
    )?;
    let _guard = Restore;
    spawn(&mut app, Job::Connect);
    loop {
        // One result per frame so a catalog run fills in instead of appearing at once.
        let _ = app.poll_work();
        terminal.draw(|frame| ui::draw(frame, &app))?;
        let timeout = if app.busy() {
            Duration::from_millis(16)
        } else {
            Duration::from_millis(100)
        };
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                match app.on_key(key) {
                    Action::Quit => break,
                    Action::None => {}
                    Action::Export => match app.export() {
                        Ok(path) => app.status = format!("wrote {}", path.display()),
                        Err(err) => app.status = err,
                    },
                    Action::Spawn(job) => spawn(&mut app, job),
                }
            }
        }
    }
    Ok(())
}

fn spawn(app: &mut App, job: Job) {
    let (tx, rx) = mpsc::channel();
    app.rx = Some(rx);
    let url = app.backend_url.clone();
    match job {
        Job::Connect => {
            thread::spawn(move || {
                let _ = tx.send(Msg::Health(api::health(&url)));
                let _ = tx.send(Msg::Finished { study: false });
            });
        }
        Job::Grades { screen, cases } => {
            thread::spawn(move || {
                for case in cases {
                    let case_id = case.case_id.clone();
                    let result = api::compare(&url, &case);
                    if tx
                        .send(Msg::Graded {
                            screen: screen.clone(),
                            case_id,
                            result,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                let _ = tx.send(Msg::Finished {
                    study: screen == "study",
                });
            });
        }
    }
}

struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen, crossterm::cursor::Show);
    }
}
