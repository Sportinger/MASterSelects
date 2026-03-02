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
    pub cache_mb: usize,
    pub max_decoders: usize,
    pub allowed_origins: Vec<String>,
}

/// Run the WebSocket server and HTTP file server
pub async fn run(config: ServerConfig) -> Result<()> {
    let ws_addr = format!("127.0.0.1:{}", config.port);
    let http_port = config.port + 1;

    let listener = TcpListener::bind(&ws_addr).await?;
    info!("WebSocket server listening on ws://{}", ws_addr);

    // Create shared state
    let state = Arc::new(AppState::new(
        config.cache_mb,
        config.max_decoders,
        None,
    ));

    let allowed_origins = Arc::new(config.allowed_origins);

    // Start HTTP file server in background
    tokio::spawn(async move {
        run_http_server(http_port).await;
    });

    // Accept WebSocket connections
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
/// Checks `tray_state.quit_requested` to know when to stop.
#[cfg(windows)]
pub async fn run_with_shutdown(
    config: ServerConfig,
    tray_state: Arc<crate::tray::TrayState>,
) -> Result<()> {
    let ws_addr = format!("127.0.0.1:{}", config.port);
    let http_port = config.port + 1;

    let listener = TcpListener::bind(&ws_addr).await?;
    info!("WebSocket server listening on ws://{}", ws_addr);

    let state = Arc::new(AppState::new(
        config.cache_mb,
        config.max_decoders,
        None,
    ));

    let allowed_origins = Arc::new(config.allowed_origins);

    // Signal that the server is ready
    tray_state.running.store(true, Ordering::Relaxed);

    // Start HTTP file server in background
    tokio::spawn(async move {
        run_http_server(http_port).await;
    });

    // Accept connections with shutdown awareness
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

/// Poll `quit_requested` until it becomes true.
#[cfg(windows)]
async fn wait_for_quit(tray_state: &Arc<crate::tray::TrayState>) {
    loop {
        if tray_state.quit_requested.load(Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// Run HTTP file server for fast file downloads
async fn run_http_server(port: u16) {
    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(vec!["GET", "OPTIONS"])
        .allow_headers(vec!["Content-Type"]);

    let file_route = warp::path("file")
        .and(warp::get())
        .and(warp::query::<std::collections::HashMap<String, String>>())
        .and_then(serve_file)
        .with(cors);

    info!("HTTP file server listening on http://127.0.0.1:{}", port);
    warp::serve(file_route).run(([127, 0, 0, 1], port)).await;
}

/// Serve a file from allowed directories
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

    match tokio::fs::read(&path).await {
        Ok(data) => {
            info!("HTTP: Serving file: {} ({} bytes)", path.display(), data.len());
            Ok(warp::reply::with_header(
                data,
                "Content-Type",
                "video/mp4",
            ))
        }
        Err(_) => Err(warp::reject::not_found()),
    }
}

/// Handle a single WebSocket connection
async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    state: Arc<AppState>,
    allowed_origins: Arc<Vec<String>>,
) -> Result<()> {
    info!("New connection from {}", addr);

    // Perform WebSocket handshake with origin check
    let ws = tokio_tungstenite::accept_hdr_async(stream, |request: &http::Request<()>, response| {
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

/// Handle WebSocket messages
async fn handle_websocket(
    ws: WebSocketStream<TcpStream>,
    addr: SocketAddr,
    state: Arc<AppState>,
) -> Result<()> {
    let (write, mut read) = ws.split();
    let write = Arc::new(tokio::sync::Mutex::new(write));
    let mut session = Session::new(state);

    // Currently buffered binary data for encode frames
    let mut pending_encode_id: Option<String> = None;

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
                // Parse command
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

                // Check if this is an EncodeFrame command (binary follows)
                if let Command::EncodeFrame { id, frame_num: _ } = &cmd {
                    pending_encode_id = Some(id.clone());
                    continue;
                }

                // Handle download commands specially to stream progress
                match cmd {
                    Command::DownloadYoutube { id, url, format_id, output_dir }
                    | Command::Download { id, url, format_id, output_dir } => {
                        let response = download::handle_download(
                            &id, &url,
                            format_id.as_deref(),
                            output_dir.as_deref(),
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
                        // Handle all other commands through session
                        let (response, binary) = session.handle_command(other).await;

                        let mut w = write.lock().await;

                        if let Some(resp) = response {
                            let json = serde_json::to_string(&resp)?;
                            w.send(Message::Text(json)).await?;
                        }

                        if let Some(data) = binary {
                            w.send(Message::Binary(data)).await?;
                        }
                    }
                }
            }

            Message::Binary(data) => {
                // Binary data is for encode frames
                if let Some(encode_id) = pending_encode_id.take() {
                    if let Some(response) = session.handle_encode_frame(&encode_id, &data) {
                        let json = serde_json::to_string(&response)?;
                        let mut w = write.lock().await;
                        w.send(Message::Text(json)).await?;
                    }
                } else {
                    warn!("Received unexpected binary data from {}", addr);
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

            Message::Frame(_) => {}
        }
    }

    info!("Connection closed: {}", addr);
    Ok(())
}
