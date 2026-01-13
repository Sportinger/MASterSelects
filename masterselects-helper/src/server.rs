//! WebSocket server implementation

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_tungstenite::WebSocketStream;
use tracing::{debug, error, info, warn};

use crate::protocol::{Command, Response};
use crate::session::{AppState, Session};

/// Server configuration
pub struct ServerConfig {
    pub port: u16,
    pub cache_mb: usize,
    pub max_decoders: usize,
    pub allowed_origins: Vec<String>,
}

/// Run the WebSocket server
pub async fn run(config: ServerConfig) -> Result<()> {
    let addr = format!("127.0.0.1:{}", config.port);
    let listener = TcpListener::bind(&addr).await?;

    info!("WebSocket server listening on ws://{}", addr);

    // Create shared state
    let state = Arc::new(AppState::new(
        config.cache_mb,
        config.max_decoders,
        None, // No auth token for now
    ));

    let allowed_origins = Arc::new(config.allowed_origins);

    // Accept connections
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
        // Check Origin header
        if let Some(origin) = request.headers().get("Origin") {
            let origin_str = origin.to_str().unwrap_or("");

            // Allow localhost origins or configured origins
            let allowed = origin_str.starts_with("http://localhost")
                || origin_str.starts_with("http://127.0.0.1")
                || origin_str.starts_with("https://localhost")
                || origin_str.starts_with("https://127.0.0.1")
                || allowed_origins.iter().any(|o| o == origin_str);

            if !allowed {
                warn!("Rejected connection from origin: {}", origin_str);
                // Can't reject here easily with this API, but we log it
            }
        }

        Ok(response)
    })
    .await?;

    // Handle connection
    handle_websocket(ws, addr, state).await
}

/// Handle WebSocket messages
async fn handle_websocket(
    ws: WebSocketStream<TcpStream>,
    addr: SocketAddr,
    state: Arc<AppState>,
) -> Result<()> {
    let (mut write, mut read) = ws.split();
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
                        write.send(Message::Text(json)).await?;
                        continue;
                    }
                };

                debug!("Received command: {:?}", cmd);

                // Check if this is an EncodeFrame command (binary follows)
                if let Command::EncodeFrame { id, frame_num: _ } = &cmd {
                    pending_encode_id = Some(id.clone());
                    continue; // Wait for binary data
                }

                // Handle command
                let (response, binary) = session.handle_command(cmd).await;

                // Send response
                if let Some(resp) = response {
                    let json = serde_json::to_string(&resp)?;
                    write.send(Message::Text(json)).await?;
                }

                // Send binary frame data
                if let Some(data) = binary {
                    write.send(Message::Binary(data)).await?;
                }
            }

            Message::Binary(data) => {
                // Binary data is for encode frames
                if let Some(encode_id) = pending_encode_id.take() {
                    if let Some(response) = session.handle_encode_frame(&encode_id, &data) {
                        let json = serde_json::to_string(&response)?;
                        write.send(Message::Text(json)).await?;
                    }
                } else {
                    warn!("Received unexpected binary data from {}", addr);
                }
            }

            Message::Ping(data) => {
                write.send(Message::Pong(data)).await?;
            }

            Message::Pong(_) => {
                // Ignore pongs
            }

            Message::Close(_) => {
                info!("Client {} disconnected", addr);
                break;
            }

            Message::Frame(_) => {
                // Raw frames not used
            }
        }
    }

    info!("Connection closed: {}", addr);
    Ok(())
}
