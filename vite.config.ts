import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      // Required for SharedArrayBuffer (cross-tab sync)
      // Using 'credentialless' instead of 'require-corp' to allow
      // cross-origin requests to Hugging Face CDN for transformers.js models
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
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'esnext',
    },
    // Exclude transformers.js from pre-bundling (it has dynamic imports)
    exclude: ['@xenova/transformers'],
  },
  worker: {
    format: 'es',
  },
})
