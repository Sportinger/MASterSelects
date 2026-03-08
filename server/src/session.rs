use log::{error, info};
use std::process::Stdio;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::broadcast;

pub struct TerminalSession {
    pub id: String,
    pub name: String,
    pub shell: String,
    pub created_at: String,
    stdin_tx: tokio::sync::mpsc::Sender<Vec<u8>>,
    pub output_tx: broadcast::Sender<String>,
    kill_tx: tokio::sync::oneshot::Sender<()>,
}

impl TerminalSession {
    pub async fn new(
        id: &str,
        name: &str,
        shell: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut child = Command::new(shell)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().expect("Failed to get stdout");
        let stderr = child.stderr.take().expect("Failed to get stderr");
        let stdin = child.stdin.take().expect("Failed to get stdin");

        let (output_tx, _) = broadcast::channel(1024);
        let (stdin_tx, mut stdin_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
        let (kill_tx, kill_rx) = tokio::sync::oneshot::channel::<()>();

        // Stdin writer
        let mut stdin_writer = stdin;
        tokio::spawn(async move {
            while let Some(data) = stdin_rx.recv().await {
                if stdin_writer.write_all(&data).await.is_err() {
                    break;
                }
                let _ = stdin_writer.flush().await;
            }
        });

        // Stdout reader
        let tx_out = output_tx.clone();
        let mut stdout_reader = stdout;
        tokio::spawn(async move {
            let mut buf = [0u8; 4096];
            loop {
                match stdout_reader.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => {
                        let data = String::from_utf8_lossy(&buf[..n]).to_string();
                        if tx_out.send(data).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Stdout read error: {}", e);
                        break;
                    }
                }
            }
        });

        // Stderr reader
        let tx_err = output_tx.clone();
        let mut stderr_reader = stderr;
        tokio::spawn(async move {
            let mut buf = [0u8; 4096];
            loop {
                match stderr_reader.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => {
                        let data = String::from_utf8_lossy(&buf[..n]).to_string();
                        if tx_err.send(data).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Stderr read error: {}", e);
                        break;
                    }
                }
            }
        });

        // Kill handler
        tokio::spawn(async move {
            let _ = kill_rx.await;
            let _ = child.kill().await;
            info!("Process killed");
        });

        info!("Created session '{}' with shell '{}'", name, shell);

        Ok(Self {
            id: id.to_string(),
            name: name.to_string(),
            shell: shell.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            stdin_tx,
            output_tx,
            kill_tx,
        })
    }

    pub async fn write_input(&self, data: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.stdin_tx
            .send(data.as_bytes().to_vec())
            .await
            .map_err(|e| format!("Failed to send input: {}", e))?;
        Ok(())
    }

    pub fn resize(&self, _cols: u16, _rows: u16) {
        // ConPTY resize would go here for full PTY support
        log::debug!("Resize requested: {}x{}", _cols, _rows);
    }

    pub fn kill(self) {
        let _ = self.kill_tx.send(());
    }
}
