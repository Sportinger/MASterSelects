// WebVJ Mixer - Main Application

import { Toolbar } from './components';
import { DockContainer } from './components/dock';
import { useGlobalHistory } from './hooks/useGlobalHistory';
import { useClipPanelSync } from './hooks/useClipPanelSync';
import './App.css';

function App() {
  // Initialize global undo/redo system
  useGlobalHistory();

  // Auto-switch panels based on clip selection
  useClipPanelSync();

  return (
    <div className="app">
      <Toolbar />
      <DockContainer />
    </div>
  );
}

export default App;
