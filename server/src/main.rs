use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, RwLock};
use tokio_tungstenite::accept_async;
use tungstenite::Message;
use uuid::Uuid;

mod auth;
mod config;
mod session;

use auth::AuthManager;
use config::ServerConfig;
use session::TerminalSession;

// === Protocol Messages ===

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    #[serde(rename = "auth")]
    Auth { token: String },
    #[serde(rename = "input")]
    Input { session_id: String, data: String },
    #[serde(rename = "resize")]
    Resize { session_id: String, cols: u16, rows: u16 },
    #[serde(rename = "create_session")]
    CreateSession { shell: Option<String>, name: Option<String> },
    #[serde(rename = "close_session")]
    CloseSession { session_id: String },
    #[serde(rename = "list_sessions")]
    ListSessions,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "auth_result")]
    AuthResult { success: bool, message: String },
    #[serde(rename = "output")]
    Output { session_id: String, data: String },
    #[serde(rename = "session_created")]
    SessionCreated { session_id: String, name: String, shell: String },
    #[serde(rename = "session_closed")]
    SessionClosed { session_id: String },
    #[serde(rename = "session_list")]
    SessionList { sessions: Vec<SessionInfo> },
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SessionInfo {
    id: String,
    name: String,
    shell: String,
    created_at: String,
}

// === App State ===

struct AppState {
    sessions: RwLock<HashMap<String, TerminalSession>>,
    auth: AuthManager,
    config: ServerConfig,
}

// === Main ===

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let config = ServerConfig::load_or_create()?;
    let auth = AuthManager::new(&config.auth_token);

    info!("===========================================");
    info!("  Terminal Remote Server v0.1.0");
    info!("===========================================");
    info!("Port: {}", config.port);
    info!("Auth Token: {}", config.auth_token);
    info!("===========================================");
    info!("Use this token in your Android app to connect.");

    let state = Arc::new(AppState {
        sessions: RwLock::new(HashMap::new()),
        auth,
        config: config.clone(),
    });

    let addr: SocketAddr = format!("0.0.0.0:{}", config.port).parse()?;
    let listener = TcpListener::bind(&addr).await?;
    info!("Listening on: {}", addr);

    while let Ok((stream, peer_addr)) = listener.accept().await {
        info!("New connection from: {}", peer_addr);
        let state = state.clone();
        tokio::spawn(async move {
            match accept_async(stream).await {
                Ok(ws) => handle_connection(ws, state, peer_addr).await,
                Err(e) => error!("WebSocket handshake failed for {}: {}", peer_addr, e),
            }
        });
    }

    Ok(())
}

// === Connection Handler ===

async fn handle_connection(
    ws_stream: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    state: Arc<AppState>,
    peer_addr: SocketAddr,
) {
    let (ws_sender, mut ws_receiver) = ws_stream.split();
    let ws_sender = Arc::new(Mutex::new(ws_sender));
    let mut authenticated = false;

    while let Some(msg) = ws_receiver.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => { error!("Receive error from {}: {}", peer_addr, e); break; }
        };

        let text = match msg {
            Message::Text(t) => t,
            Message::Ping(d) => { let _ = ws_sender.lock().await.send(Message::Pong(d)).await; continue; }
            Message::Close(_) => { info!("Client {} disconnected", peer_addr); break; }
            _ => continue,
        };

        let client_msg: ClientMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                send_msg(&ws_sender, &ServerMessage::Error { message: format!("Invalid message: {}", e) }).await;
                continue;
            }
        };

        // Auth gate
        if !authenticated {
            if let ClientMessage::Auth { token } = &client_msg {
                if state.auth.verify(token) {
                    authenticated = true;
                    send_msg(&ws_sender, &ServerMessage::AuthResult { success: true, message: "OK".into() }).await;
                    info!("Client {} authenticated", peer_addr);
                } else {
                    send_msg(&ws_sender, &ServerMessage::AuthResult { success: false, message: "Invalid token".into() }).await;
                    warn!("Failed auth from {}", peer_addr);
                }
            } else {
                send_msg(&ws_sender, &ServerMessage::Error { message: "Not authenticated".into() }).await;
            }
            continue;
        }

        // Authenticated commands
        match client_msg {
            ClientMessage::CreateSession { shell, name } => {
                let shell_cmd = shell.unwrap_or_else(|| state.config.default_shell.clone());
                let session_name = name.unwrap_or_else(|| {
                    format!("Session {}", state.sessions.read().await.len() + 1)
                });
                let session_id = Uuid::new_v4().to_string();

                match TerminalSession::new(&session_id, &session_name, &shell_cmd).await {
                    Ok(session) => {
                        // Start output forwarding
                        let sender_clone = ws_sender.clone();
                        let sid = session_id.clone();
                        let mut output_rx = session.output_tx.subscribe();

                        tokio::spawn(async move {
                            while let Ok(data) = output_rx.recv().await {
                                let msg = ServerMessage::Output { session_id: sid.clone(), data };
                                if send_msg(&sender_clone, &msg).await.is_err() {
                                    break;
                                }
                            }
                        });

                        state.sessions.write().await.insert(session_id.clone(), session);
                        send_msg(&ws_sender, &ServerMessage::SessionCreated {
                            session_id, name: session_name, shell: shell_cmd,
                        }).await;
                    }
                    Err(e) => {
                        send_msg(&ws_sender, &ServerMessage::Error {
                            message: format!("Failed to create session: {}", e),
                        }).await;
                    }
                }
            }

            ClientMessage::Input { session_id, data } => {
                let sessions = state.sessions.read().await;
                if let Some(session) = sessions.get(&session_id) {
                    if let Err(e) = session.write_input(&data).await {
                        send_msg(&ws_sender, &ServerMessage::Error {
                            message: format!("Input error: {}", e),
                        }).await;
                    }
                } else {
                    send_msg(&ws_sender, &ServerMessage::Error {
                        message: format!("Session not found: {}", session_id),
                    }).await;
                }
            }

            ClientMessage::Resize { session_id, cols, rows } => {
                let sessions = state.sessions.read().await;
                if let Some(session) = sessions.get(&session_id) {
                    session.resize(cols, rows);
                }
            }

            ClientMessage::CloseSession { session_id } => {
                if let Some(session) = state.sessions.write().await.remove(&session_id) {
                    session.kill();
                    send_msg(&ws_sender, &ServerMessage::SessionClosed { session_id }).await;
                }
            }

            ClientMessage::ListSessions => {
                let sessions = state.sessions.read().await;
                let list: Vec<SessionInfo> = sessions.values().map(|s| SessionInfo {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    shell: s.shell.clone(),
                    created_at: s.created_at.clone(),
                }).collect();
                send_msg(&ws_sender, &ServerMessage::SessionList { sessions: list }).await;
            }

            ClientMessage::Auth { .. } => {
                send_msg(&ws_sender, &ServerMessage::AuthResult {
                    success: true, message: "Already authenticated".into(),
                }).await;
            }
        }
    }

    info!("Connection closed: {}", peer_addr);
}

async fn send_msg(
    sender: &Arc<Mutex<futures_util::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>, Message>>>,
    msg: &ServerMessage,
) -> Result<(), ()> {
    let json = serde_json::to_string(msg).unwrap();
    sender.lock().await.send(Message::Text(json)).await.map_err(|_| ())
}
