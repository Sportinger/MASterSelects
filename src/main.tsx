import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { useTimelineStore } from './stores/timeline'

// Expose store for debugging
if (import.meta.env.DEV) {
  (window as unknown as { store: typeof useTimelineStore }).store = useTimelineStore;
}

// Note: StrictMode disabled for WebGPU compatibility in development
// StrictMode causes double-mounting which breaks external texture references
createRoot(document.getElementById('root')!).render(<App />)
