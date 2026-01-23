//! MasterSelects Helper Lite - YouTube downloader only (no FFmpeg required)

use anyhow::Result;
use clap::Parser;
use futures_util::{SinkExt, StreamExt, stream::SplitSink};
use std::sync::Mutex as StdMutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command as TokioCommand;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{info, warn, error};
use warp::Filter;

#[derive(Parser, Debug)]
#[command(name = "masterselects-helper-lite")]
#[command(about = "YouTube downloader helper for MasterSelects (no FFmpeg)")]
#[command(version)]
struct Args {
    #[arg(short, long, default_value = "9876")]
    port: u16,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    Ping { id: String },
    Info { id: String },
    ListFormats { id: String, url: String },
    DownloadYoutube {
        id: String,
        url: String,
        format_id: Option<String>,
        output_dir: Option<String>,
    },
    GetFile { id: String, path: String },
}

#[derive(Debug, Serialize)]
struct Response {
    id: String,
    ok: bool,
    #[serde(flatten)]
    data: serde_json::Value,
}

impl Response {
    fn ok(id: &str, data: serde_json::Value) -> Self {
        Self { id: id.to_string(), ok: true, data }
    }
    fn error(id: &str, code: &str, message: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            ok: false,
            data: serde_json::json!({ "error": { "code": code, "message": message.into() } }),
        }
    }
    fn progress(id: &str, percent: u8) -> Self {
        Self {
            id: id.to_string(),
            ok: true,
            data: serde_json::json!({ "type": "progress", "percent": percent }),
        }
    }
}

// Type for sending WebSocket messages
type WsSender = Arc<tokio::sync::Mutex<SplitSink<tokio_tungstenite::WebSocketStream<TcpStream>, Message>>>;

/// Cross-platform download directory
fn get_download_dir() -> PathBuf {
    let base = std::env::temp_dir();
    base.join("masterselects-downloads")
}

/// Find yt-dlp executable, checking common install locations on Windows
fn find_ytdlp() -> Option<PathBuf> {
    // First check if yt-dlp is in PATH
    if let Ok(output) = std::process::Command::new("yt-dlp").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("yt-dlp"));
        }
    }

    // On Windows, check common Python user install locations
    #[cfg(windows)]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            // Check various Python versions
            let appdata_path = PathBuf::from(&appdata);
            if let Ok(entries) = std::fs::read_dir(appdata_path.join("Python")) {
                for entry in entries.flatten() {
                    let scripts = entry.path().join("Scripts").join("yt-dlp.exe");
                    if scripts.exists() {
                        return Some(scripts);
                    }
                }
            }
        }

        // Also check LocalAppData (pip --user on some configs)
        if let Ok(localappdata) = std::env::var("LOCALAPPDATA") {
            let local_path = PathBuf::from(&localappdata);
            if let Ok(entries) = std::fs::read_dir(local_path.join("Programs").join("Python")) {
                for entry in entries.flatten() {
                    let scripts = entry.path().join("Scripts").join("yt-dlp.exe");
                    if scripts.exists() {
                        return Some(scripts);
                    }
                }
            }
        }
    }

    None
}

/// Find deno executable for yt-dlp JavaScript runtime
fn find_deno() -> Option<PathBuf> {
    // Check if deno is in PATH
    if let Ok(output) = std::process::Command::new("deno").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("deno"));
        }
    }

    // On Windows, check winget install location
    #[cfg(windows)]
    {
        if let Ok(localappdata) = std::env::var("LOCALAPPDATA") {
            let winget_path = PathBuf::from(&localappdata)
                .join("Microsoft/WinGet/Packages");
            if let Ok(entries) = std::fs::read_dir(&winget_path) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_lowercase();
                    if name.contains("deno") {
                        let deno_exe = entry.path().join("deno.exe");
                        if deno_exe.exists() {
                            return Some(deno_exe);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Get yt-dlp command (returns path or "yt-dlp" if in PATH)
fn get_ytdlp_command() -> String {
    find_ytdlp()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "yt-dlp".to_string())
}

/// Cross-platform allowed paths
fn is_path_allowed(path: &std::path::Path) -> bool {
    let temp_dir = std::env::temp_dir();
    let downloads_dir = dirs::download_dir();

    let path_str = path.to_string_lossy().to_lowercase();
    let temp_str = temp_dir.to_string_lossy().to_lowercase();

    if path_str.starts_with(&temp_str) {
        return true;
    }

    if let Some(dl) = downloads_dir {
        let dl_str = dl.to_string_lossy().to_lowercase();
        if path_str.starts_with(&dl_str) {
            return true;
        }
    }

    false
}

async fn handle_command(cmd: Command) -> Response {
    match cmd {
        Command::Ping { id } => Response::ok(&id, serde_json::json!({ "pong": true })),

        Command::Info { id } => {
            // Check if yt-dlp is available
            let ytdlp_cmd = get_ytdlp_command();
            let ytdlp_available = TokioCommand::new(&ytdlp_cmd)
                .arg("--version")
                .output()
                .await
                .map(|o| o.status.success())
                .unwrap_or(false);

            Response::ok(&id, serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "lite": true,
                "ytdlp_available": ytdlp_available,
                "download_dir": get_download_dir().to_string_lossy(),
            }))
        }

        Command::ListFormats { id, url } => {
            handle_list_formats(&id, &url).await
        }

        Command::DownloadYoutube { id, url, format_id, output_dir } => {
            handle_download(&id, &url, format_id.as_deref(), output_dir.as_deref(), None).await
        }

        Command::GetFile { id, path } => {
            handle_get_file(&id, &path)
        }
    }
}

/// Build yt-dlp args with deno runtime if available
fn get_deno_args() -> Vec<String> {
    if let Some(deno_path) = find_deno() {
        vec![
            "--js-runtimes".to_string(),
            format!("deno:{}", deno_path.to_string_lossy()),
        ]
    } else {
        vec![]
    }
}

async fn handle_list_formats(id: &str, url: &str) -> Response {
    use std::process::Stdio;

    if !url.contains("youtube.com") && !url.contains("youtu.be") {
        return Response::error(id, "INVALID_URL", "Not a valid YouTube URL");
    }

    let ytdlp_cmd = get_ytdlp_command();
    let deno_args = get_deno_args();

    let mut cmd = TokioCommand::new(&ytdlp_cmd);
    for arg in &deno_args {
        cmd.arg(arg);
    }
    let result = cmd
        .args(["--dump-json", "--no-playlist", url])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;

    match result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            match serde_json::from_str::<serde_json::Value>(&stdout) {
                Ok(info) => {
                    let title = info.get("title").and_then(|v| v.as_str()).unwrap_or("Unknown");
                    let uploader = info.get("uploader").and_then(|v| v.as_str()).unwrap_or("Unknown");
                    let duration = info.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let thumbnail = info.get("thumbnail").and_then(|v| v.as_str()).unwrap_or("");

                    // Parse formats
                    let mut recommendations = Vec::new();
                    if let Some(formats) = info.get("formats").and_then(|v| v.as_array()) {
                        let mut by_height: HashMap<i64, Vec<&serde_json::Value>> = HashMap::new();

                        for fmt in formats {
                            let acodec = fmt.get("acodec").and_then(|v| v.as_str()).unwrap_or("none");
                            let vcodec = fmt.get("vcodec").and_then(|v| v.as_str()).unwrap_or("none");
                            let ext = fmt.get("ext").and_then(|v| v.as_str()).unwrap_or("");

                            // Skip audio-only and AV1
                            if vcodec == "none" || vcodec.contains("av01") {
                                continue;
                            }

                            if let Some(height) = fmt.get("height").and_then(|v| v.as_i64()) {
                                if height >= 360 {
                                    by_height.entry(height).or_default().push(fmt);
                                }
                            }
                        }

                        // Get best format per resolution, prefer H.264
                        let mut heights: Vec<_> = by_height.keys().copied().collect();
                        heights.sort_by(|a, b| b.cmp(a));

                        for height in heights.into_iter().take(6) {
                            if let Some(fmts) = by_height.get(&height) {
                                // Prefer H.264 (avc1) over VP9
                                let best = fmts.iter()
                                    .max_by_key(|f| {
                                        let vcodec = f.get("vcodec").and_then(|v| v.as_str()).unwrap_or("");
                                        let tbr = f.get("tbr").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                        let codec_score = if vcodec.contains("avc") { 1000.0 } else { 0.0 };
                                        (codec_score + tbr) as i64
                                    });

                                if let Some(fmt) = best {
                                    let format_id = fmt.get("format_id").and_then(|v| v.as_str()).unwrap_or("");
                                    let vcodec = fmt.get("vcodec").and_then(|v| v.as_str()).unwrap_or("");
                                    let fps = fmt.get("fps").and_then(|v| v.as_f64()).unwrap_or(30.0);
                                    let filesize = fmt.get("filesize").and_then(|v| v.as_i64())
                                        .or_else(|| fmt.get("filesize_approx").and_then(|v| v.as_i64()));

                                    let codec_name = if vcodec.contains("avc") { "H.264" }
                                        else if vcodec.contains("vp9") { "VP9" }
                                        else { vcodec };

                                    recommendations.push(serde_json::json!({
                                        "id": format_id,
                                        "label": format!("{}p {} ({:.0}fps)", height, codec_name, fps),
                                        "resolution": format!("{}p", height),
                                        "vcodec": codec_name,
                                        "acodec": serde_json::Value::Null,
                                        "needsMerge": true,
                                        "filesize": filesize,
                                    }));
                                }
                            }
                        }
                    }

                    Response::ok(id, serde_json::json!({
                        "title": title,
                        "uploader": uploader,
                        "duration": duration,
                        "thumbnail": thumbnail,
                        "recommendations": recommendations,
                    }))
                }
                Err(e) => Response::error(id, "PARSE_ERROR", format!("Failed to parse yt-dlp output: {}", e)),
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Response::error(id, "YTDLP_ERROR", stderr.to_string())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Response::error(id, "YTDLP_NOT_FOUND", "yt-dlp not found. Install with: pip install yt-dlp")
        }
        Err(e) => Response::error(id, "ERROR", e.to_string()),
    }
}

async fn handle_download(id: &str, url: &str, format_id: Option<&str>, output_dir: Option<&str>, ws_sender: Option<WsSender>) -> Response {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, BufReader};

    if !url.contains("youtube.com") && !url.contains("youtu.be") {
        return Response::error(id, "INVALID_URL", "Not a valid YouTube URL");
    }

    let download_dir = output_dir.map(PathBuf::from).unwrap_or_else(get_download_dir);

    if let Err(e) = std::fs::create_dir_all(&download_dir) {
        return Response::error(id, "PERMISSION_DENIED", format!("Cannot create directory: {}", e));
    }

    info!("Downloading: {} to {:?}", url, download_dir);

    let output_template = download_dir.join("%(title)s.%(ext)s").to_string_lossy().to_string();

    let format_str = if let Some(fid) = format_id {
        format!("{}+bestaudio[ext=m4a]/{}+bestaudio/best", fid, fid)
    } else {
        "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best".to_string()
    };

    let ytdlp_cmd = get_ytdlp_command();
    let deno_args = get_deno_args();

    let mut cmd = TokioCommand::new(&ytdlp_cmd);
    for arg in &deno_args {
        cmd.arg(arg);
    }

    let mut child = match cmd
        .args([
            "-f", &format_str,
            "--merge-output-format", "mp4",
            "-o", &output_template,
            "--print", "after_move:filepath",
            "--no-playlist",
            "--newline",  // Progress on separate lines
            "--progress",
            "--restrict-filenames",  // Replace special chars with ASCII
            "--windows-filenames",   // Windows-safe filenames
            url,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Response::error(id, "YTDLP_NOT_FOUND", "yt-dlp not found. Install with: pip install yt-dlp");
            }
            Err(e) => {
                return Response::error(id, "SPAWN_ERROR", e.to_string());
            }
        };

    // Stream stderr for progress
    let stderr = child.stderr.take();
    let mut last_percent: u8 = 0;
    if let Some(stderr) = stderr {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            // Parse progress percentage from yt-dlp output
            // Format: "[download]  45.2% of 100.00MiB"
            if line.contains('%') {
                if let Some(pct_str) = line.split('%').next() {
                    let pct_part = pct_str.trim().rsplit_once(' ').map(|(_, p)| p).unwrap_or(pct_str.trim());
                    if let Ok(pct) = pct_part.trim().parse::<f32>() {
                        let percent = (pct as u8).min(99);
                        // Only send if changed by at least 5%
                        if percent >= last_percent + 5 || percent == 99 {
                            last_percent = percent;
                            info!("[yt-dlp] Progress: {}%", percent);
                            // Send progress via WebSocket
                            if let Some(ref sender) = ws_sender {
                                let progress_msg = Response::progress(id, percent);
                                let json = serde_json::to_string(&progress_msg).unwrap();
                                let mut sender = sender.lock().await;
                                let _ = sender.send(Message::Text(json)).await;
                            }
                        }
                    }
                }
            } else if line.contains("Downloading") || line.contains("Merging") {
                info!("[yt-dlp] {}", line);
            } else if !line.is_empty() && !line.starts_with("[") {
                warn!("[yt-dlp] {}", line);
            }
        }
    }

    let result = child.wait_with_output().await;

    match result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let output_path = stdout.lines().last().unwrap_or("").trim();

            if output_path.is_empty() {
                return Response::error(id, "DOWNLOAD_FAILED", "yt-dlp did not return output path");
            }

            info!("Download complete: {}", output_path);
            Response::ok(id, serde_json::json!({ "path": output_path }))
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!("yt-dlp failed: {}", stderr);
            Response::error(id, "DOWNLOAD_FAILED", stderr.lines().last().unwrap_or("Unknown error"))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Response::error(id, "YTDLP_NOT_FOUND", "yt-dlp not found. Install with: pip install yt-dlp")
        }
        Err(e) => Response::error(id, "DOWNLOAD_FAILED", e.to_string()),
    }
}

fn handle_get_file(id: &str, path: &str) -> Response {
    use std::io::Read;

    let path = std::path::Path::new(path);

    if !path.is_absolute() {
        return Response::error(id, "INVALID_PATH", "Path must be absolute");
    }

    if !is_path_allowed(path) {
        return Response::error(id, "PERMISSION_DENIED", "Path not in allowed directory");
    }

    if !path.exists() {
        return Response::error(id, "FILE_NOT_FOUND", format!("File not found: {}", path.display()));
    }

    match std::fs::read(path) {
        Ok(data) => {
            use base64::Engine;
            let encoded = base64::engine::general_purpose::STANDARD.encode(&data);
            info!("Serving file: {} ({} bytes)", path.display(), data.len());
            Response::ok(id, serde_json::json!({
                "size": data.len(),
                "path": path.display().to_string(),
                "data": encoded,
            }))
        }
        Err(e) => Response::error(id, "FILE_NOT_FOUND", format!("Cannot read: {}", e)),
    }
}

async fn handle_websocket(stream: TcpStream) {
    let ws_stream = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            error!("WebSocket handshake failed: {}", e);
            return;
        }
    };

    info!("New WebSocket connection");
    let (write, mut read) = ws_stream.split();
    let write = Arc::new(tokio::sync::Mutex::new(write));

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<Command>(&text) {
                    Ok(cmd) => {
                        // Handle download specially to stream progress
                        let response = match cmd {
                            Command::DownloadYoutube { id, url, format_id, output_dir } => {
                                handle_download(&id, &url, format_id.as_deref(), output_dir.as_deref(), Some(write.clone())).await
                            }
                            other => handle_command(other).await,
                        };
                        let json = serde_json::to_string(&response).unwrap();
                        let mut w = write.lock().await;
                        if let Err(e) = w.send(Message::Text(json)).await {
                            error!("Failed to send response: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Invalid command: {}", e);
                        let resp = Response::error("unknown", "INVALID_COMMAND", e.to_string());
                        let json = serde_json::to_string(&resp).unwrap();
                        let mut w = write.lock().await;
                        let _ = w.send(Message::Text(json)).await;
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    info!("WebSocket connection closed");
}

async fn serve_file(params: HashMap<String, String>) -> impl warp::Reply {
    use warp::http::StatusCode;

    // Helper to create response with CORS
    fn cors_response(status: StatusCode, body: Vec<u8>, content_type: &str) -> impl warp::Reply {
        warp::reply::with_header(
            warp::reply::with_header(
                warp::reply::with_header(
                    warp::reply::with_status(body, status),
                    "Content-Type",
                    content_type,
                ),
                "Access-Control-Allow-Origin",
                "*",
            ),
            "Access-Control-Allow-Methods",
            "GET, OPTIONS",
        )
    }

    let path = match params.get("path") {
        Some(p) => p,
        None => {
            warn!("HTTP: No path parameter");
            return cors_response(StatusCode::BAD_REQUEST, b"Missing path parameter".to_vec(), "text/plain");
        }
    };

    let path = PathBuf::from(path);
    info!("HTTP: Request for {:?}", path);

    if !path.is_absolute() {
        warn!("HTTP: Path not absolute: {:?}", path);
        return cors_response(StatusCode::BAD_REQUEST, b"Path must be absolute".to_vec(), "text/plain");
    }

    if !is_path_allowed(&path) {
        warn!("HTTP: Path not allowed: {:?}", path);
        return cors_response(StatusCode::FORBIDDEN, b"Path not in allowed directory".to_vec(), "text/plain");
    }

    if !path.exists() {
        warn!("HTTP: File not found: {:?}", path);
        return cors_response(StatusCode::NOT_FOUND, format!("File not found: {}", path.display()).into_bytes(), "text/plain");
    }

    match tokio::fs::read(&path).await {
        Ok(data) => {
            info!("HTTP: Serving {} ({} bytes)", path.display(), data.len());
            cors_response(StatusCode::OK, data, "video/mp4")
        }
        Err(e) => {
            error!("HTTP: Failed to read file: {}", e);
            cors_response(StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read: {}", e).into_bytes(), "text/plain")
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let ws_addr = format!("127.0.0.1:{}", args.port);
    let http_port = args.port + 1;

    let ytdlp_path = get_ytdlp_command();
    let ytdlp_status = if find_ytdlp().is_some() { "OK" } else { "NOT FOUND" };
    let deno_path = find_deno();
    let deno_status = if deno_path.is_some() { "OK" } else { "NOT FOUND" };

    println!();
    println!("========================================");
    println!("  MasterSelects Helper Lite v{}", env!("CARGO_PKG_VERSION"));
    println!("  (YouTube downloader - no FFmpeg)");
    println!("========================================");
    println!("  WebSocket: ws://127.0.0.1:{}", args.port);
    println!("  HTTP:      http://127.0.0.1:{}", http_port);
    println!("  Downloads: {}", get_download_dir().display());
    println!("  yt-dlp:    {} [{}]", ytdlp_path, ytdlp_status);
    println!("  deno:      {} [{}]", deno_path.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|| "not found".to_string()), deno_status);
    println!("========================================");
    println!();

    // HTTP file server with CORS preflight support
    let file_route = warp::path("file")
        .and(warp::get())
        .and(warp::query::<HashMap<String, String>>())
        .then(serve_file);

    // Handle OPTIONS preflight requests
    let options_route = warp::path("file")
        .and(warp::options())
        .map(|| {
            warp::reply::with_header(
                warp::reply::with_header(
                    warp::reply::with_header(
                        "",
                        "Access-Control-Allow-Origin",
                        "*",
                    ),
                    "Access-Control-Allow-Methods",
                    "GET, OPTIONS",
                ),
                "Access-Control-Allow-Headers",
                "*",
            )
        });

    let routes = file_route.or(options_route);

    let http_server = warp::serve(routes)
        .run(([127, 0, 0, 1], http_port));

    tokio::spawn(http_server);
    info!("HTTP file server on port {}", http_port);

    // WebSocket server
    let listener = TcpListener::bind(&ws_addr).await?;
    info!("WebSocket server on port {}", args.port);

    loop {
        let (stream, addr) = listener.accept().await?;
        tokio::spawn(handle_websocket(stream));
    }
}
