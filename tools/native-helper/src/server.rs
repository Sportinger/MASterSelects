//! WebSocket server implementation

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_tungstenite::WebSocketStream;
use tracing::{debug, error, info, warn};
use warp::Filter;

#[cfg(windows)]
use std::sync::atomic::Ordering;

use crate::download;
use crate::protocol::{Command, Response};
use crate::session::{AppState, Session};
use crate::utils;

/// Server configuration
pub struct ServerConfig {
    pub port: u16,
    pub allowed_origins: Vec<String>,
}

/// Run the WebSocket server and HTTP file server
pub async fn run(config: ServerConfig) -> Result<()> {
    let ws_addr = format!("127.0.0.1:{}", config.port);
    let http_port = config.port + 1;

    let listener = TcpListener::bind(&ws_addr).await?;
    info!("WebSocket server listening on ws://{}", ws_addr);

    let state = Arc::new(AppState::new(None));
    let allowed_origins = Arc::new(config.allowed_origins);

    tokio::spawn(async move {
        run_http_server(http_port).await;
    });

    while let Ok((stream, addr)) = listener.accept().await {
        let state = state.clone();
        let allowed_origins = allowed_origins.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, addr, state, allowed_origins).await {
                error!("Connection error from {}: {}", addr, e);
            }
        });
    }

    Ok(())
}

/// Run the server with graceful shutdown support (Windows tray mode).
#[cfg(windows)]
pub async fn run_with_shutdown(
    config: ServerConfig,
    tray_state: Arc<crate::tray::TrayState>,
) -> Result<()> {
    let ws_addr = format!("127.0.0.1:{}", config.port);
    let http_port = config.port + 1;

    let listener = TcpListener::bind(&ws_addr).await?;
    info!("WebSocket server listening on ws://{}", ws_addr);

    let state = Arc::new(AppState::new(None));
    let allowed_origins = Arc::new(config.allowed_origins);

    tray_state.running.store(true, Ordering::Relaxed);

    tokio::spawn(async move {
        run_http_server(http_port).await;
    });

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((stream, addr)) => {
                        let state = state.clone();
                        let allowed_origins = allowed_origins.clone();
                        let ts = tray_state.clone();

                        ts.connection_count.fetch_add(1, Ordering::Relaxed);

                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, addr, state, allowed_origins).await {
                                error!("Connection error from {}: {}", addr, e);
                            }
                            ts.connection_count.fetch_sub(1, Ordering::Relaxed);
                        });
                    }
                    Err(e) => {
                        error!("Accept error: {}", e);
                    }
                }
            }
            _ = wait_for_quit(&tray_state) => {
                info!("Shutdown requested, stopping server...");
                break;
            }
        }
    }

    Ok(())
}

#[cfg(windows)]
async fn wait_for_quit(tray_state: &Arc<crate::tray::TrayState>) {
    loop {
        if tray_state.quit_requested.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

async fn run_http_server(port: u16) {
    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(vec!["GET", "POST", "OPTIONS"])
        .allow_headers(vec!["Content-Type"]);

    // GET /file?path=... — serve a file
    let file_route = warp::path("file")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(serve_file);

    // POST /upload?path=... — write binary body to file
    let upload_route = warp::path("upload")
        .and(warp::post())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and(warp::body::bytes())
        .and_then(handle_upload);

    // GET /project-root — return the default project root path
    let project_root_route = warp::path("project-root")
        .and(warp::get())
        .and_then(get_project_root);

    let routes = file_route
        .or(upload_route)
        .or(project_root_route)
        .with(cors);

    info!("HTTP file server listening on http://127.0.0.1:{}", port);
    warp::serve(routes).run(([127, 0, 0, 1], port)).await;
}

/// Guess Content-Type from file extension
fn guess_content_type(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "avi" => "video/x-msvideo",
        "mkv" => "video/x-matroska",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "aac" => "audio/aac",
        "m4a" => "audio/mp4",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "json" => "application/json",
        "xml" => "application/xml",
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "css" => "text/css",
        "js" => "application/javascript",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

async fn serve_file(
    params: std::collections::HashMap<String, String>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let path = params.get("path").ok_or_else(warp::reject::not_found)?;
    let path = PathBuf::from(path);

    if !path.is_absolute() {
        return Err(warp::reject::not_found());
    }

    if !utils::is_path_allowed(&path) {
        warn!("HTTP: Rejected file request for: {}", path.display());
        return Err(warp::reject::not_found());
    }

    if !path.exists() {
        return Err(warp::reject::not_found());
    }

    let content_type = guess_content_type(&path);

    match tokio::fs::read(&path).await {
        Ok(data) => {
            info!("HTTP: Serving file: {} ({} bytes)", path.display(), data.len());
            Ok(warp::reply::with_header(data, "Content-Type", content_type))
        }
        Err(_) => Err(warp::reject::not_found()),
    }
}

/// POST /upload?path=<absolute_path> — write binary body to disk
async fn handle_upload(
    params: std::collections::HashMap<String, String>,
    body: warp::hyper::body::Bytes,
) -> Result<impl warp::Reply, warp::Rejection> {
    let path = params.get("path").ok_or_else(warp::reject::not_found)?;
    let path = PathBuf::from(path);

    if !path.is_absolute() {
        warn!("HTTP upload: Rejected non-absolute path");
        return Err(warp::reject::not_found());
    }

    if !utils::is_path_allowed(&path) {
        warn!("HTTP upload: Rejected path: {}", path.display());
        return Err(warp::reject::not_found());
    }

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                warn!("HTTP upload: Cannot create parent dirs: {}", e);
                return Err(warp::reject::not_found());
            }
        }
    }

    let size = body.len();

    // Atomic write: .tmp then rename
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));

    match tokio::fs::write(&tmp_path, &body).await {
        Ok(()) => {
            if let Err(_) = tokio::fs::rename(&tmp_path, &path).await {
                // Rename failed — fallback to direct write
                let _ = tokio::fs::remove_file(&tmp_path).await;
                if let Err(e) = tokio::fs::write(&path, &body).await {
                    warn!("HTTP upload: Write failed: {}", e);
                    return Err(warp::reject::not_found());
                }
            }
            info!("HTTP upload: {} ({} bytes)", path.display(), size);
            Ok(warp::reply::json(&serde_json::json!({
                "ok": true,
                "written": true,
                "size": size
            })))
        }
        Err(e) => {
            warn!("HTTP upload: Write failed: {}", e);
            Err(warp::reject::not_found())
        }
    }
}

/// GET /project-root — return the default project root path
async fn get_project_root() -> Result<impl warp::Reply, warp::Rejection> {
    let root = utils::get_project_root();
    Ok(warp::reply::json(&serde_json::json!({
        "ok": true,
        "path": root.to_string_lossy()
    })))
}

async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    state: Arc<AppState>,
    allowed_origins: Arc<Vec<String>>,
) -> Result<()> {
    info!("New connection from {}", addr);

    let ws =
        tokio_tungstenite::accept_hdr_async(stream, |request: &http::Request<()>, response| {
            if let Some(origin) = request.headers().get("Origin") {
                let origin_str = origin.to_str().unwrap_or("");
                let allowed = origin_str.starts_with("http://localhost")
                    || origin_str.starts_with("http://127.0.0.1")
                    || origin_str.starts_with("https://localhost")
                    || origin_str.starts_with("https://127.0.0.1")
                    || allowed_origins.iter().any(|o| o == origin_str);
                if !allowed {
                    warn!("Rejected connection from origin: {}", origin_str);
                }
            }
            Ok(response)
        })
        .await?;

    handle_websocket(ws, addr, state).await
}

async fn handle_websocket(
    ws: WebSocketStream<TcpStream>,
    addr: SocketAddr,
    state: Arc<AppState>,
) -> Result<()> {
    let (write, mut read) = ws.split();
    let write = Arc::new(tokio::sync::Mutex::new(write));
    let mut session = Session::new(state);

    while let Some(msg) = read.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                error!("WebSocket error from {}: {}", addr, e);
                break;
            }
        };

        match msg {
            Message::Text(text) => {
                let cmd: Command = match serde_json::from_str(&text) {
                    Ok(c) => c,
                    Err(e) => {
                        let response = Response::error("", "PARSE_ERROR", e.to_string());
                        let json = serde_json::to_string(&response)?;
                        let mut w = write.lock().await;
                        w.send(Message::Text(json)).await?;
                        continue;
                    }
                };

                debug!("Received command: {:?}", cmd);

                match cmd {
                    Command::DownloadYoutube {
                        id, url, format_id, output_dir,
                    }
                    | Command::Download {
                        id, url, format_id, output_dir,
                    } => {
                        let response = download::handle_download(
                            &id, &url, format_id.as_deref(), output_dir.as_deref(),
                            Some(write.clone()),
                        ).await;
                        let json = serde_json::to_string(&response)?;
                        let mut w = write.lock().await;
                        w.send(Message::Text(json)).await?;
                    }
                    Command::ListFormats { id, url } => {
                        let response = download::handle_list_formats(&id, &url).await;
                        let json = serde_json::to_string(&response)?;
                        let mut w = write.lock().await;
                        w.send(Message::Text(json)).await?;
                    }
                    other => {
                        if let Some(response) = session.handle_command(other).await {
                            let json = serde_json::to_string(&response)?;
                            let mut w = write.lock().await;
                            w.send(Message::Text(json)).await?;
                        }
                    }
                }
            }

            Message::Ping(data) => {
                let mut w = write.lock().await;
                w.send(Message::Pong(data)).await?;
            }
            Message::Pong(_) => {}
            Message::Close(_) => {
                info!("Client {} disconnected", addr);
                break;
            }
            Message::Binary(_) => {
                warn!("Received unexpected binary data from {}", addr);
            }
            Message::Frame(_) => {}
        }
    }

    info!("Connection closed: {}", addr);
    Ok(())
}
