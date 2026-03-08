//! MasterSelects Native Helper
//!
//! A cross-platform download helper providing video downloads via yt-dlp
//! over WebSocket for the MasterSelects web application.
//!
//! On Windows (default): runs as a system tray app with no console window.
//! On Windows (--console): runs in a terminal like on other platforms.
//! On Linux/macOS: always runs in console mode.

mod download;
mod protocol;
mod server;
mod session;
#[cfg(windows)]
mod tray;
#[cfg(windows)]
mod updater;
mod utils;

use clap::Parser;
use tracing::{error, Level};
use tracing_subscriber::FmtSubscriber;

/// MasterSelects Native Helper - Download acceleration for masterselects.app
#[derive(Parser, Debug)]
#[command(name = "masterselects-helper")]
#[command(about = "Cross-platform download helper for MasterSelects web application")]
#[command(version)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "9876")]
    port: u16,

    /// Run in background (minimal output)
    #[arg(long)]
    background: bool,

    /// Allowed origins (comma-separated, empty = allow all localhost)
    #[arg(long)]
    allowed_origins: Option<String>,

    /// Generate and print auth token, then exit
    #[arg(long)]
    generate_token: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Run in console mode (show terminal window, no system tray).
    /// On Linux/macOS this is always the default.
    #[arg(long)]
    console: bool,
}

fn main() {
    let args = Args::parse();

    // Handle token generation (quick exit)
    if args.generate_token {
        let token = session::generate_auth_token();
        println!("{}", token);
        return;
    }

    // Initialize logging
    init_logging(&args);

    // Build server config
    let config = build_config(&args);

    // Decide: tray mode or console mode
    #[cfg(windows)]
    {
        if !args.console {
            run_with_tray(config, &args);
            return;
        }
    }

    // Console mode (all platforms, or --console on Windows)
    run_console(config, &args);
}

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

fn init_logging(args: &Args) {
    if !args.background {
        let level = match args.log_level.to_lowercase().as_str() {
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "info" => Level::INFO,
            "warn" => Level::WARN,
            "error" => Level::ERROR,
            _ => Level::INFO,
        };

        let _subscriber = FmtSubscriber::builder()
            .with_max_level(level)
            .with_target(false)
            .compact()
            .init();
    }
}

fn build_config(args: &Args) -> server::ServerConfig {
    let allowed_origins: Vec<String> = args
        .allowed_origins
        .as_ref()
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| {
            vec![
                "https://masterselects.app".to_string(),
                "https://app.masterselects.com".to_string(),
                "http://localhost:5173".to_string(),
                "http://localhost:3000".to_string(),
                "http://127.0.0.1:5173".to_string(),
                "http://127.0.0.1:3000".to_string(),
            ]
        });

    server::ServerConfig {
        port: args.port,
        allowed_origins,
    }
}

fn print_banner(config: &server::ServerConfig) {
    let ytdlp_path = download::get_ytdlp_command();
    let ytdlp_available = download::find_ytdlp().is_some();
    let deno_available = download::find_deno().is_some();

    let os_name = if cfg!(windows) {
        "Windows"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "macos") {
        "macOS"
    } else {
        "Unknown"
    };

    println!();
    println!("========================================================");
    println!(
        "  MasterSelects Native Helper v{}",
        env!("CARGO_PKG_VERSION")
    );
    println!("  Platform: {}", os_name);
    println!("========================================================");
    println!("  WebSocket: ws://127.0.0.1:{}", config.port);
    println!("  HTTP File: http://127.0.0.1:{}", config.port + 1);
    println!(
        "  yt-dlp:    {} [{}]",
        ytdlp_path,
        if ytdlp_available { "OK" } else { "NOT FOUND" }
    );
    println!(
        "  deno:      {}",
        if deno_available {
            "OK"
        } else {
            "not found (optional)"
        }
    );
    println!("  Downloads: {}", utils::get_download_dir().display());
    println!("  Projects:  {}", utils::get_project_root().display());
    println!("========================================================");
    println!();
}

// ---------------------------------------------------------------------------
// Run modes
// ---------------------------------------------------------------------------

/// Console mode: print banner, run server in a tokio runtime (blocks forever).
fn run_console(config: server::ServerConfig, args: &Args) {
    if !args.background {
        print_banner(&config);
    }

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    if let Err(e) = rt.block_on(server::run(config)) {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}

/// Windows tray mode: hide console, tray icon on main thread, server on worker thread.
#[cfg(windows)]
fn run_with_tray(config: server::ServerConfig, _args: &Args) {
    use std::sync::Arc;

    // Hide the console window
    tray::hide_console_window();

    // Prevent multiple instances
    let _lock = match tray::acquire_single_instance_lock() {
        Some(handle) => handle,
        None => {
            // Another instance is already running — exit silently
            return;
        }
    };

    let port = config.port;
    let state = Arc::new(tray::TrayState::new());
    let state_for_server = state.clone();

    // Spawn server on a worker thread (with its own tokio runtime)
    let server_thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
        if let Err(e) = rt.block_on(server::run_with_shutdown(config, state_for_server)) {
            eprintln!("Server error: {}", e);
        }
    });

    // Run tray message pump on the main thread (blocks until Quit)
    if let Err(e) = tray::run_tray(state, port) {
        eprintln!("Tray error: {}", e);
    }

    // Wait for the server thread to finish
    let _ = server_thread.join();
}
