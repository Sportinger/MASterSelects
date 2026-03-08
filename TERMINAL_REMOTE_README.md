# TerminalRemote

Remote terminal access from your Android phone to your Windows PC.

## Architecture

```
┌─────────────────┐         WebSocket (ws://)         ┌──────────────────┐
│  Windows PC     │◄────────────────────────────────►  │  Android App     │
│  (Rust Server)  │     Port 8765 + Auth Token        │  (Kotlin)        │
│                 │                                    │                  │
│  - PowerShell   │     ◄── Input commands             │  - Terminal View │
│  - CMD          │     ──► Terminal output             │  - Special Keys  │
│  - Sessions     │     ──► Session management         │  - Themes        │
└─────────────────┘                                    └──────────────────┘
```

## Components

### Server (`server/`)
- **Language:** Rust
- **Platform:** Windows (.exe)
- **Protocol:** WebSocket on configurable port (default: 8765)
- **Auth:** Token-based authentication (auto-generated on first run)
- **Features:** Multi-session support, PowerShell/CMD

### Android App (`android/`)
- **Language:** Kotlin (Native)
- **Min SDK:** 26 (Android 8.0)
- **Features:**
  - Terminal display with monospace font
  - Special keys bar (ESC, TAB, CTRL, Ctrl+C, Arrow keys)
  - Multiple session tabs
  - Color themes (Dark, Monokai, Solarized)
  - Font size adjustment
  - Connection saving
  - Keep screen on option

## Setup

### Server (Windows)
```bash
cd server
cargo build --release
# Binary at: target/release/terminal-server.exe
# Config at: %APPDATA%/terminal-remote/config.toml
```

On first run, the server generates an auth token and prints it.
Use this token in your Android app to connect.

### Internet Access (DynDNS)
1. Set up DynDNS (e.g., noip.com, duckdns.org)
2. Forward port 8765 in your router to your PC
3. Use your DynDNS hostname in the Android app

### Android
Open `android/` in Android Studio and build normally.

## Protocol

All messages are JSON over WebSocket.

### Client → Server
```json
{"type": "auth", "token": "your-token"}
{"type": "create_session", "shell": "powershell.exe", "name": "Main"}
{"type": "input", "session_id": "uuid", "data": "ls\n"}
{"type": "resize", "session_id": "uuid", "cols": 80, "rows": 24}
{"type": "close_session", "session_id": "uuid"}
{"type": "list_sessions"}
```

### Server → Client
```json
{"type": "auth_result", "success": true, "message": "..."}
{"type": "output", "session_id": "uuid", "data": "terminal output..."}
{"type": "session_created", "session_id": "uuid", "name": "Main", "shell": "powershell.exe"}
{"type": "session_closed", "session_id": "uuid"}
{"type": "session_list", "sessions": [...]}
{"type": "error", "message": "..."}
```
