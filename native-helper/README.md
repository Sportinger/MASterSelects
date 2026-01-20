# MasterSelects Native Helper

Native WebSocket server for MasterSelects that enables YouTube video downloads.

## Requirements

- Node.js 18+
- yt-dlp (`brew install yt-dlp`)

## Installation

```bash
cd native-helper
npm install
```

## Usage

```bash
# Start the server
npm start

# Or with auto-reload during development
npm run dev
```

The server runs on `ws://127.0.0.1:9876` and downloads to `~/Movies/MasterSelects Downloads/`.

## Commands

| Command | Description |
|---------|-------------|
| `ping` | Health check |
| `info` | Server info and yt-dlp status |
| `download_youtube` | Download a YouTube video |
| `get_file` | Retrieve a downloaded file |
| `cancel_download` | Cancel an active download |

## Protocol

All messages are JSON over WebSocket.

### Download YouTube

```json
{
  "cmd": "download_youtube",
  "id": "req_1",
  "url": "https://youtube.com/watch?v=..."
}
```

Response:
```json
{
  "id": "req_1",
  "ok": true,
  "path": "/Users/.../video.mp4",
  "filename": "video.mp4"
}
```

Progress updates:
```json
{
  "id": "req_1",
  "progress": 0.45,
  "status": "downloading"
}
```
