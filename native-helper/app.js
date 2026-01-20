/**
 * MasterSelects Native Helper - macOS Menubar App
 *
 * Electron app with tray icon that runs the WebSocket server.
 */

import { app, Tray, Menu, nativeImage, shell, Notification } from 'electron';
import { spawn, execSync } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { homedir } from 'os';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer } from 'ws';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PORT = 9876;
const VERSION = '1.0.0';
const DOWNLOAD_DIR = join(homedir(), 'Movies', 'MasterSelects Downloads');

// Ensure download directory exists
if (!existsSync(DOWNLOAD_DIR)) {
  mkdirSync(DOWNLOAD_DIR, { recursive: true });
}

// Check if yt-dlp is installed
function checkYtDlp() {
  try {
    execSync('which yt-dlp', { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

function getYtDlpVersion() {
  try {
    return execSync('yt-dlp --version', { encoding: 'utf-8' }).trim();
  } catch {
    return 'not installed';
  }
}

let tray = null;
let wss = null;
let connectedClients = 0;
let activeDownloads = new Map();
let hasYtDlp = checkYtDlp();
let ytDlpVersion = getYtDlpVersion();

// Don't show in dock
app.dock?.hide();

app.whenReady().then(() => {
  createTray();
  startServer();
});

app.on('window-all-closed', (e) => {
  e.preventDefault(); // Keep running in background
});

function createTray() {
  // Create tray icon (yellow lightning bolt)
  const iconPath = join(__dirname, 'assets', 'trayIconTemplate.png');

  // If icon doesn't exist, create a simple one
  let image;
  if (existsSync(iconPath)) {
    image = nativeImage.createFromPath(iconPath);
  } else {
    // Create a simple 22x22 icon programmatically
    image = nativeImage.createEmpty();
  }

  tray = new Tray(image);
  tray.setToolTip('MasterSelects Helper');

  updateTrayMenu();
}

function updateTrayMenu() {
  const statusText = connectedClients > 0
    ? `✓ Connected (${connectedClients} client${connectedClients > 1 ? 's' : ''})`
    : '○ Waiting for connection...';

  const ytdlpText = hasYtDlp
    ? `yt-dlp: ${ytDlpVersion}`
    : 'yt-dlp: Not installed';

  const contextMenu = Menu.buildFromTemplate([
    { label: 'MasterSelects Helper', enabled: false },
    { label: `v${VERSION}`, enabled: false },
    { type: 'separator' },
    { label: statusText, enabled: false },
    { label: `Port: ${PORT}`, enabled: false },
    { label: ytdlpText, enabled: false },
    { type: 'separator' },
    {
      label: 'Open Downloads Folder',
      click: () => shell.openPath(DOWNLOAD_DIR)
    },
    {
      label: 'Open MasterSelects',
      click: () => shell.openExternal('http://localhost:5173')
    },
    { type: 'separator' },
    {
      label: 'Install yt-dlp',
      visible: !hasYtDlp,
      click: () => {
        shell.openExternal('https://github.com/yt-dlp/yt-dlp#installation');
      }
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        if (wss) wss.close();
        app.quit();
      }
    }
  ]);

  tray.setContextMenu(contextMenu);

  // Update tray title (shows next to icon on macOS)
  if (connectedClients > 0) {
    tray.setTitle(''); // Connected - no text needed, icon is enough
  } else {
    tray.setTitle('');
  }
}

function startServer() {
  wss = new WebSocketServer({ port: PORT });

  console.log(`WebSocket server listening on ws://127.0.0.1:${PORT}`);

  wss.on('connection', (ws) => {
    connectedClients++;
    console.log(`[+] Client connected (${connectedClients} total)`);
    updateTrayMenu();

    ws.on('message', async (data) => {
      try {
        const message = JSON.parse(data.toString());
        await handleMessage(ws, message);
      } catch (err) {
        console.error('[ERROR] Failed to parse message:', err);
      }
    });

    ws.on('close', () => {
      connectedClients--;
      console.log(`[-] Client disconnected (${connectedClients} remaining)`);
      updateTrayMenu();
    });

    ws.on('error', (err) => {
      console.error('[ERROR] WebSocket error:', err);
    });
  });

  wss.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      new Notification({
        title: 'MasterSelects Helper',
        body: `Port ${PORT} is already in use. Another instance may be running.`
      }).show();
    }
  });
}

async function handleMessage(ws, message) {
  console.log(`[CMD] ${message.cmd} (id: ${message.id})`);

  switch (message.cmd) {
    case 'ping':
      ws.send(JSON.stringify({ id: message.id, ok: true, pong: true }));
      break;

    case 'info':
      ws.send(JSON.stringify({
        id: message.id,
        ok: true,
        version: VERSION,
        ffmpeg_version: ytDlpVersion,
        hw_accel: hasYtDlp ? ['yt-dlp'] : [],
        cache_used_mb: 0,
        cache_max_mb: 1000,
        open_files: activeDownloads.size,
        download_dir: DOWNLOAD_DIR,
      }));
      break;

    case 'download_youtube':
      await handleYouTubeDownload(ws, message);
      break;

    case 'get_file':
      handleGetFile(ws, message);
      break;

    case 'cancel_download':
      handleCancelDownload(ws, message);
      break;

    default:
      ws.send(JSON.stringify({
        id: message.id,
        ok: false,
        error: { code: 'UNKNOWN_COMMAND', message: `Unknown command: ${message.cmd}` }
      }));
  }
}

async function handleYouTubeDownload(ws, message) {
  const { id, url } = message;

  if (!hasYtDlp) {
    ws.send(JSON.stringify({
      id,
      ok: false,
      error: { code: 'YT_DLP_NOT_FOUND', message: 'yt-dlp is not installed. Run: brew install yt-dlp' }
    }));
    return;
  }

  if (!url) {
    ws.send(JSON.stringify({
      id,
      ok: false,
      error: { code: 'INVALID_URL', message: 'No URL provided' }
    }));
    return;
  }

  console.log(`[DOWNLOAD] Starting: ${url}`);

  // Show notification
  new Notification({
    title: 'Download Started',
    body: 'Downloading video from YouTube...'
  }).show();

  const outputTemplate = join(DOWNLOAD_DIR, '%(title)s.%(ext)s');

  const args = [
    url,
    '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    '-o', outputTemplate,
    '--no-playlist',
    '--progress',
    '--newline',
    '--no-warnings',
  ];

  const process = spawn('yt-dlp', args);
  activeDownloads.set(id, process);
  updateTrayMenu();

  let outputPath = null;
  let lastProgress = 0;

  process.stdout.on('data', (data) => {
    const line = data.toString().trim();

    const progressMatch = line.match(/(\d+\.?\d*)%/);
    if (progressMatch) {
      const progress = parseFloat(progressMatch[1]);
      if (progress - lastProgress >= 5 || progress >= 100) {
        lastProgress = progress;
        ws.send(JSON.stringify({
          id,
          progress: progress / 100,
          status: 'downloading'
        }));
      }
    }

    const destMatch = line.match(/\[download\] Destination: (.+)$/);
    if (destMatch) outputPath = destMatch[1];

    const mergeMatch = line.match(/\[Merger\] Merging formats into "(.+)"$/);
    if (mergeMatch) outputPath = mergeMatch[1];

    const alreadyMatch = line.match(/\[download\] (.+) has already been downloaded/);
    if (alreadyMatch) outputPath = alreadyMatch[1];
  });

  process.stderr.on('data', (data) => {
    console.error(`[yt-dlp stderr] ${data.toString().trim()}`);
  });

  process.on('close', (code) => {
    activeDownloads.delete(id);
    updateTrayMenu();

    if (code === 0 && outputPath) {
      const filename = outputPath.split('/').pop();
      console.log(`[DOWNLOAD] Complete: ${filename}`);

      new Notification({
        title: 'Download Complete',
        body: filename
      }).show();

      ws.send(JSON.stringify({
        id,
        ok: true,
        path: outputPath,
        filename
      }));
    } else {
      console.error(`[DOWNLOAD] Failed with code ${code}`);

      new Notification({
        title: 'Download Failed',
        body: `yt-dlp exited with code ${code}`
      }).show();

      ws.send(JSON.stringify({
        id,
        ok: false,
        error: { code: 'DOWNLOAD_FAILED', message: `yt-dlp exited with code ${code}` }
      }));
    }
  });

  process.on('error', (err) => {
    activeDownloads.delete(id);
    updateTrayMenu();
    console.error(`[DOWNLOAD] Error: ${err.message}`);
    ws.send(JSON.stringify({
      id,
      ok: false,
      error: { code: 'DOWNLOAD_ERROR', message: err.message }
    }));
  });
}

function handleGetFile(ws, message) {
  const { id, path } = message;

  if (!path) {
    ws.send(JSON.stringify({
      id,
      ok: false,
      error: { code: 'INVALID_PATH', message: 'No path provided' }
    }));
    return;
  }

  if (!path.startsWith(DOWNLOAD_DIR)) {
    ws.send(JSON.stringify({
      id,
      ok: false,
      error: { code: 'PERMISSION_DENIED', message: 'Access denied' }
    }));
    return;
  }

  import('fs').then(({ existsSync, readFileSync, statSync }) => {
    if (!existsSync(path)) {
      ws.send(JSON.stringify({
        id,
        ok: false,
        error: { code: 'FILE_NOT_FOUND', message: 'File not found' }
      }));
      return;
    }

    try {
      const stats = statSync(path);
      const data = readFileSync(path);
      const base64 = data.toString('base64');

      ws.send(JSON.stringify({
        id,
        ok: true,
        data: base64,
        size: stats.size,
        filename: path.split('/').pop()
      }));
    } catch (err) {
      ws.send(JSON.stringify({
        id,
        ok: false,
        error: { code: 'READ_ERROR', message: err.message }
      }));
    }
  });
}

function handleCancelDownload(ws, message) {
  const { id, download_id } = message;
  const targetId = download_id || id;

  const process = activeDownloads.get(targetId);
  if (process) {
    process.kill('SIGTERM');
    activeDownloads.delete(targetId);
    updateTrayMenu();
    console.log(`[DOWNLOAD] Cancelled: ${targetId}`);
    ws.send(JSON.stringify({ id, ok: true, cancelled: true }));
  } else {
    ws.send(JSON.stringify({
      id,
      ok: false,
      error: { code: 'NOT_FOUND', message: 'No active download with that ID' }
    }));
  }
}
