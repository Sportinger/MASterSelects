import { useState } from 'react';
import type { ScopeViewMode } from './useScopeAnalysis';
import { ScopeModeToolbar } from './ScopeModeToolbar';
import { WaveformScope } from './WaveformScope';
import './ScopesPanel.css';

export function WaveformPanel() {
  const [viewMode, setViewMode] = useState<ScopeViewMode>('rgb');

  return (
    <div className="scope-panel">
      <ScopeModeToolbar mode={viewMode} onModeChange={setViewMode} />
      <div className="scope-panel-content">
        <WaveformScope viewMode={viewMode} />
      </div>
    </div>
  );
}
