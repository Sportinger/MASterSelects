import { useState } from 'react';
import type { ScopeViewMode } from './useScopeAnalysis';
import { ScopeModeToolbar } from './ScopeModeToolbar';
import { HistogramScope } from './HistogramScope';
import './ScopesPanel.css';

export function HistogramPanel() {
  const [viewMode, setViewMode] = useState<ScopeViewMode>('rgb');

  return (
    <div className="scope-panel">
      <ScopeModeToolbar mode={viewMode} onModeChange={setViewMode} />
      <div className="scope-panel-content">
        <HistogramScope viewMode={viewMode} />
      </div>
    </div>
  );
}
