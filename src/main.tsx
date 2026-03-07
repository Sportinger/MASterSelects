import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { useTimelineStore } from './stores/timeline'
import { executeAITool, AI_TOOLS, getQuickTimelineSummary } from './services/aiTools'

// Expose AI tools API for browser console, Claude skills, and external agents
(window as any).aiTools = {
  execute: executeAITool,
  list: () => AI_TOOLS,
  status: getQuickTimelineSummary,
};

// Bridge: allow external agents to call aiTools via HTTP POST /api/ai-tools
import('./services/aiTools/bridge');

// Expose store for debugging
if (import.meta.env.DEV) {
  (window as unknown as { store: typeof useTimelineStore }).store = useTimelineStore;
}

// Note: StrictMode disabled for WebGPU compatibility in development
// StrictMode causes double-mounting which breaks external texture references
createRoot(document.getElementById('root')!).render(<App />)
