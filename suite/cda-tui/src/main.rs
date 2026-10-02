//! `cda-tui` draws the suite. `cda-tui --check` grades the catalog and exits.

use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use cda_tui::app::App;
use cda_tui::catalog;

#[derive(Parser)]
#[command(
    name = "cda-tui",
    about = "Terminal test suite for the Laya Coding Decision Agent"
)]
struct Args {
    /// Sidecar origin. The model is loaded there, not in this process.
    #[arg(long, default_value = "http://127.0.0.1:8765")]
    backend: String,
    /// Path to suite/cases/catalog.json. Searched upward from the working directory when omitted.
    #[arg(long)]
    cases: Option<PathBuf>,
    /// Directory for the JSONL log and the exported report.
    #[arg(long)]
    session_dir: Option<PathBuf>,
    /// Grade every catalog case and print metrics. No terminal UI.
    #[arg(long)]
    check: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let catalog_path = match catalog::locate(args.cases) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };
    let catalog = match catalog::load(&catalog_path) {
        Ok(catalog) => catalog,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };
    let session_dir = args
        .session_dir
        .unwrap_or_else(|| catalog::default_session_dir(&catalog_path));

    if args.check {
        return match cda_tui::check::run(&args.backend, &catalog, &session_dir) {
            Ok(code) => ExitCode::from(code as u8),
            Err(err) => {
                eprintln!("{err}");
                ExitCode::from(2)
            }
        };
    }

    if !io::stdout().is_terminal() {
        eprintln!("stdout is not a terminal. Use --check for a headless run.");
        return ExitCode::from(2);
    }

    let app = App::new(catalog, args.backend, session_dir);
    if let Err(err) = cda_tui::tui::run(app) {
        eprintln!("{err}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
