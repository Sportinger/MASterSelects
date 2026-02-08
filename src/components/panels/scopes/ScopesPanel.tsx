import { useState } from 'react';
import { useScopeAnalysis, type ScopeTab } from './useScopeAnalysis';
import { HistogramScope } from './HistogramScope';
import { VectorscopeScope } from './VectorscopeScope';
import { WaveformScope } from './WaveformScope';
import './ScopesPanel.css';

const TABS: { id: ScopeTab; label: string }[] = [
  { id: 'histogram', label: 'Histogram' },
  { id: 'vectorscope', label: 'Vectorscope' },
  { id: 'waveform', label: 'Waveform' },
];

export function ScopesPanel() {
  const [activeTab, setActiveTab] = useState<ScopeTab>('waveform');
  const { histogramData, vectorscopeData, waveformData } = useScopeAnalysis(activeTab, true);

  return (
    <div className="scopes-panel">
      <div className="scopes-tab-bar">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`scopes-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="scopes-content">
        {activeTab === 'histogram' && <HistogramScope data={histogramData} />}
        {activeTab === 'vectorscope' && <VectorscopeScope data={vectorscopeData} />}
        {activeTab === 'waveform' && <WaveformScope data={waveformData} />}
      </div>
    </div>
  );
}
