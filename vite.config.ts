import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { APP_VERSION } from './src/version'
import fs from 'fs'
import path from 'path'

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

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => ({
  plugins: [
    react(),
    browserLogBridge(),
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
