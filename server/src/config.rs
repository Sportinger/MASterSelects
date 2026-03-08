use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub port: u16,
    pub auth_token: String,
    pub default_shell: String,
    pub max_sessions: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8765,
            auth_token: generate_token(),
            default_shell: "powershell.exe".to_string(),
            max_sessions: 5,
        }
    }
}

impl ServerConfig {
    pub fn config_path() -> PathBuf {
        let mut path = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push("terminal-remote");
        std::fs::create_dir_all(&path).ok();
        path.push("config.toml");
        path
    }

    pub fn load_or_create() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::config_path();

        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let config: ServerConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            let config = ServerConfig::default();
            let content = toml::to_string_pretty(&config)?;
            std::fs::write(&path, content)?;
            log::info!("Created config file at: {}", path.display());
            Ok(config)
        }
    }
}

fn generate_token() -> String {
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
    hex::encode(bytes)
}
