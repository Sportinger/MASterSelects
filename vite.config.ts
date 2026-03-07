import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { APP_VERSION } from './src/version'
import crypto from 'crypto'
import fs from 'fs'
import path from 'path'

// Local File Server - serves local files for AI-driven import
function localFileServer(): Plugin {
  return {
    name: 'local-file-server',
    configureServer(server) {
      // Serve a local file by absolute path
      server.middlewares.use('/api/local-file', (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const filePath = url.searchParams.get('path');
        if (!filePath) {
          res.statusCode = 400;
          res.end('Missing path parameter');
          return;
        }

        const resolved = path.resolve(filePath);
        if (!fs.existsSync(resolved) || fs.statSync(resolved).isDirectory()) {
          res.statusCode = 404;
          res.end('File not found');
          return;
        }

        const stat = fs.statSync(resolved);
        const ext = path.extname(resolved).toLowerCase();
        const mimeTypes: Record<string, string> = {
          '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
          '.mkv': 'video/x-matroska', '.avi': 'video/x-msvideo',
          '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.aac': 'audio/aac', '.ogg': 'audio/ogg',
          '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
          '.gif': 'image/gif', '.webp': 'image/webp',
        };
        res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
        res.setHeader('Access-Control-Allow-Origin', '*');

        // Support range requests for video seeking
        const range = req.headers.range;
        if (range) {
          const parts = range.replace(/bytes=/, '').split('-');
          const start = parseInt(parts[0], 10);
          const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
          res.statusCode = 206;
          res.setHeader('Content-Range', `bytes ${start}-${end}/${stat.size}`);
          res.setHeader('Content-Length', end - start + 1);
          res.setHeader('Accept-Ranges', 'bytes');
          fs.createReadStream(resolved, { start, end }).pipe(res);
        } else {
          res.setHeader('Content-Length', stat.size);
          res.setHeader('Accept-Ranges', 'bytes');
          fs.createReadStream(resolved).pipe(res);
        }
      });

      // List media files in a directory
      server.middlewares.use('/api/local-files', (req, res) => {
        if (req.method !== 'GET') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const dirPath = url.searchParams.get('dir');
        const extFilter = url.searchParams.get('ext')?.split(',') ||
          ['.mp4', '.webm', '.mov', '.mkv', '.avi', '.mp3', '.wav', '.png', '.jpg', '.jpeg'];

        if (!dirPath) {
          res.statusCode = 400;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: 'Missing dir parameter' }));
          return;
        }

        const resolved = path.resolve(dirPath);
        if (!fs.existsSync(resolved)) {
          res.statusCode = 404;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: 'Directory not found' }));
          return;
        }

        try {
          const entries = fs.readdirSync(resolved);
          const files = entries
            .filter(f => extFilter.some(ext => f.toLowerCase().endsWith(ext)))
            .map(f => {
              const fullPath = path.join(resolved, f);
              const stat = fs.statSync(fullPath);
              return {
                name: f,
                path: fullPath.replace(/\\/g, '/'),
                size: stat.size,
                modified: stat.mtime.toISOString(),
              };
            });
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ files }));
        } catch {
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: 'Failed to list directory' }));
        }
      });
    },
  };
}

// Browser Log Bridge - allows AI agents to read browser console logs
function browserLogBridge(): Plugin {
  const logFile = path.resolve(__dirname, '.browser-logs.json');

  return {
    name: 'browser-log-bridge',
    configureServer(server) {
      // Handle log sync from browser
      server.middlewares.use('/api/logs', (req, res) => {
        if (req.method === 'POST') {
          let body = '';
          req.on('data', (chunk: Buffer) => body += chunk.toString());
          req.on('end', () => {
            try {
              fs.writeFileSync(logFile, body);
              res.statusCode = 200;
              res.end('ok');
            } catch {
              res.statusCode = 500;
              res.end('write error');
            }
          });
        } else if (req.method === 'GET') {
          // AI agent reads logs via this endpoint
          try {
            const logs = fs.existsSync(logFile)
              ? fs.readFileSync(logFile, 'utf-8')
              : '{"totalLogs":0,"errorCount":0,"warnCount":0,"recentErrors":[],"activeModules":[]}';
            res.setHeader('Content-Type', 'application/json');
            res.end(logs);
          } catch {
            res.statusCode = 500;
            res.end('{}');
          }
        } else {
          res.statusCode = 405;
          res.end('Method not allowed');
        }
      });
    }
  };
}

// AI Tools Bridge - lets external agents (Claude CLI) execute aiTools via HTTP
// Flow: POST /api/ai-tools → Vite server → HMR → browser → aiTools.execute() → HMR → HTTP response
function aiToolsBridge(): Plugin {
  const pendingRequests = new Map<string, { resolve: (value: unknown) => void; timer: ReturnType<typeof setTimeout> }>();
  let requestCounter = 0;

  return {
    name: 'ai-tools-bridge',
    configureServer(server) {
      // Listen for results coming back from the browser via HMR
      server.hot.on('ai-tools:result', (data: { requestId: string; result: unknown }) => {
        const pending = pendingRequests.get(data.requestId);
        if (pending) {
          clearTimeout(pending.timer);
          pendingRequests.delete(data.requestId);
          pending.resolve(data.result);
        }
      });

      server.middlewares.use('/api/ai-tools', (req, res) => {
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

        if (req.method === 'OPTIONS') {
          res.statusCode = 204;
          res.end();
          return;
        }

        if (req.method === 'GET') {
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ status: 'ready', pending: pendingRequests.size }));
          return;
        }

        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end('Method not allowed');
          return;
        }

        let body = '';
        req.on('data', (chunk: Buffer) => body += chunk.toString());
        req.on('end', () => {
          try {
            const { tool, args = {} } = JSON.parse(body);
            if (!tool) {
              res.statusCode = 400;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ success: false, error: 'Missing "tool" field' }));
              return;
            }

            const requestId = `r${++requestCounter}-${crypto.randomUUID().slice(0, 8)}`;

            const resultPromise = new Promise((resolve) => {
              const timer = setTimeout(() => {
                pendingRequests.delete(requestId);
                resolve({ success: false, error: 'Timeout: no browser tab responded within 30s' });
              }, 30000);

              pendingRequests.set(requestId, { resolve, timer });
              server.hot.send('ai-tools:execute', { requestId, tool, args });
            });

            resultPromise.then((result) => {
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify(result));
            });
          } catch {
            res.statusCode = 400;
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ success: false, error: 'Invalid JSON body' }));
          }
        });
      });
    },
  };
}

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => ({
  plugins: [
    react(),
    localFileServer(),
    browserLogBridge(),
    aiToolsBridge(),
    // Replace __APP_VERSION__ in index.html during build
    {
      name: 'html-version-replace',
      transformIndexHtml(html) {
        return html.replace(/__APP_VERSION__/g, APP_VERSION);
      },
    },
  ],
  define: {
    __APP_VERSION__: JSON.stringify(APP_VERSION),
    // Show changelog: always in production builds, only with --mode changelog in dev
    __SHOW_CHANGELOG__: command === 'build' || mode === 'changelog',
  },
  server: {
    headers: {
      // Required for SharedArrayBuffer (FFmpeg multi-threaded, cross-tab sync)
      // Using 'credentialless' instead of 'require-corp' to allow CDN resources
      // (FFmpeg WASM from unpkg, transformers.js from HuggingFace)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      output: {
        manualChunks: {
          // Force heavy libs into separate chunks (loaded on demand)
          'mp4box': ['mp4box'],
          'mp4-muxer': ['mp4-muxer'],
          'webm-muxer': ['webm-muxer'],
          'onnxruntime': ['onnxruntime-web'],
        },
      },
    },
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'esnext',
    },
    // Exclude transformers.js and onnxruntime from pre-bundling
    exclude: ['@huggingface/transformers', 'onnxruntime-web'],
  },
}))
