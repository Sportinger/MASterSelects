//! MasterSelects Native Helper
//!
//! A lightweight video codec helper that provides hardware-accelerated
//! ProRes/DNxHD decoding and encoding via WebSocket for the MasterSelects
//! web application.

mod cache;
mod decoder;
mod encoder;
mod protocol;
mod server;
mod session;

use clap::Parser;
use tracing::{info, error, Level};
use tracing_subscriber::FmtSubscriber;

/// MasterSelects Native Helper - Video codec acceleration for masterselects.app
#[derive(Parser, Debug)]
#[command(name = "masterselects-helper")]
#[command(about = "Video codec helper for MasterSelects web application")]
#[command(version)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "9876")]
    port: u16,

    /// Run in background (minimal output)
    #[arg(long)]
    background: bool,

    /// Maximum cache size in MB
    #[arg(long, default_value = "2048")]
    cache_mb: usize,

    /// Maximum number of open decoder contexts
    #[arg(long, default_value = "8")]
    max_decoders: usize,

    /// Allowed origins (comma-separated, empty = allow all localhost)
    #[arg(long)]
    allowed_origins: Option<String>,

    /// Generate and print auth token, then exit
    #[arg(long)]
    generate_token: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Handle token generation
    if args.generate_token {
        let token = session::generate_auth_token();
        println!("{}", token);
        return Ok(());
    }

    // Initialize logging
    if !args.background {
        let level = match args.log_level.to_lowercase().as_str() {
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "info" => Level::INFO,
            "warn" => Level::WARN,
            "error" => Level::ERROR,
            _ => Level::INFO,
        };

        let subscriber = FmtSubscriber::builder()
            .with_max_level(level)
            .with_target(false)
            .compact()
            .init();
    }

    // Initialize FFmpeg
    ffmpeg_next::init().map_err(|e| anyhow::anyhow!("Failed to initialize FFmpeg: {}", e))?;
    info!("FFmpeg initialized");

    // Parse allowed origins
    let allowed_origins: Vec<String> = args
        .allowed_origins
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| vec![
            "https://masterselects.app".to_string(),
            "https://app.masterselects.com".to_string(),
            "http://localhost:5173".to_string(),
            "http://localhost:3000".to_string(),
            "http://127.0.0.1:5173".to_string(),
            "http://127.0.0.1:3000".to_string(),
        ]);

    // Create server config
    let config = server::ServerConfig {
        port: args.port,
        cache_mb: args.cache_mb,
        max_decoders: args.max_decoders,
        allowed_origins,
    };

    // Print startup banner
    if !args.background {
        println!();
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║         MasterSelects Native Helper v{}              ║", env!("CARGO_PKG_VERSION"));
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║  WebSocket: ws://127.0.0.1:{}                          ║", args.port);
        println!("║  Cache: {} MB                                          ║", args.cache_mb);
        println!("║  Max decoders: {}                                        ║", args.max_decoders);
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
    }

    // Start server
    if let Err(e) = server::run(config).await {
        error!("Server error: {}", e);
        return Err(e);
    }

    Ok(())
}
