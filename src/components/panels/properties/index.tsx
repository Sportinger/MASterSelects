// Properties Panel - Main container with lazy-loaded tabs
import { useState, useEffect, useCallback, Suspense, lazy } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { TextTab } from '../TextTab';

// Tab type
type PropertiesTab = 'transform' | 'effects' | 'masks' | 'volume' | 'transcript' | 'analysis' | 'text';

// Lazy load tab components for code splitting
const TransformTab = lazy(() => import('./TransformTab').then(m => ({ default: m.TransformTab })));
const VolumeTab = lazy(() => import('./VolumeTab').then(m => ({ default: m.VolumeTab })));
const EffectsTab = lazy(() => import('./EffectsTab').then(m => ({ default: m.EffectsTab })));
const MasksTab = lazy(() => import('./MasksTab').then(m => ({ default: m.MasksTab })));
const TranscriptTab = lazy(() => import('./TranscriptTab').then(m => ({ default: m.TranscriptTab })));
const AnalysisTab = lazy(() => import('./AnalysisTab').then(m => ({ default: m.AnalysisTab })));

// Tab loading fallback
function TabLoading() {
  return <div className="properties-tab-loading">Loading...</div>;
}

export function PropertiesPanel() {
  // Reactive data - subscribe to specific values only
  const clips = useTimelineStore(state => state.clips);
  const tracks = useTimelineStore(state => state.tracks);
  const selectedClipIds = useTimelineStore(state => state.selectedClipIds);
  const playheadPosition = useTimelineStore(state => state.playheadPosition);
  // Actions from getState() - stable, no subscription needed
  const { getInterpolatedTransform, getInterpolatedSpeed } = useTimelineStore.getState();
  const [activeTab, setActiveTab] = useState<PropertiesTab>('transform');
  const [lastClipId, setLastClipId] = useState<string | null>(null);

  const selectedClipId = selectedClipIds.size > 0 ? [...selectedClipIds][0] : null;
  const selectedClip = clips.find(c => c.id === selectedClipId);

  // Check if it's an audio clip
  const selectedTrack = selectedClip ? tracks.find(t => t.id === selectedClip.trackId) : null;
  const isAudioClip = selectedTrack?.type === 'audio';

  // Check if it's a text clip
  const isTextClip = selectedClip?.source?.type === 'text';

  // Check if it's a solid clip
  const isSolidClip = selectedClip?.source?.type === 'solid';

  // Reset tab when switching between audio/video/text/solid clips
  useEffect(() => {
    if (selectedClipId && selectedClipId !== lastClipId) {
      setLastClipId(selectedClipId);
      // Set appropriate default tab based on clip type
      if (isSolidClip) {
        setActiveTab('transform');
      } else if (isTextClip) {
        setActiveTab('text');
      } else if (isAudioClip && (activeTab === 'transform' || activeTab === 'masks' || activeTab === 'text')) {
        setActiveTab('volume');
      } else if (!isAudioClip && !isTextClip && (activeTab === 'volume' || activeTab === 'text')) {
        setActiveTab('transform');
      }
    }
  }, [selectedClipId, isAudioClip, isTextClip, isSolidClip, lastClipId, activeTab]);

  const handleSolidColorChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!selectedClipId) return;
    useTimelineStore.getState().updateSolidColor(selectedClipId, e.target.value);
  }, [selectedClipId]);

  if (!selectedClip) {
    return (
      <div className="properties-panel">
        <div className="panel-header"><h3>Properties</h3></div>
        <div className="panel-empty"><p>Select a clip to edit properties</p></div>
      </div>
    );
  }

  const clipLocalTime = playheadPosition - selectedClip.startTime;
  const transform = getInterpolatedTransform(selectedClip.id, clipLocalTime);
  const interpolatedSpeed = getInterpolatedSpeed(selectedClip.id, clipLocalTime);

  // Count non-audio effects for badge
  const visualEffects = (selectedClip.effects || []).filter(e => e.type !== 'audio-volume' && e.type !== 'audio-eq');

  return (
    <div className="properties-panel">
      {/* Solid color picker — always visible at top when a solid clip is selected */}
      {isSolidClip && (
        <div className="solid-color-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px' }}>
            <input
              type="color"
              value={selectedClip.solidColor || '#ffffff'}
              onChange={handleSolidColorChange}
              style={{ width: '28px', height: '22px', padding: '0', border: '1px solid #3a3a3a', borderRadius: '3px', cursor: 'pointer', background: 'transparent' }}
            />
            <span style={{ fontSize: '11px', color: '#aaa', fontFamily: 'monospace' }}>
              {selectedClip.solidColor || '#ffffff'}
            </span>
          </div>
        </div>
      )}

      <div className="properties-tabs">
        {isAudioClip ? (
          <>
            <button className={`tab-btn ${activeTab === 'volume' ? 'active' : ''}`} onClick={() => setActiveTab('volume')}>Volume</button>
            <button className={`tab-btn ${activeTab === 'effects' ? 'active' : ''}`} onClick={() => setActiveTab('effects')}>
              Effects {visualEffects.length > 0 && <span className="badge">{visualEffects.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>
              Transcript {selectedClip.transcript && selectedClip.transcript.length > 0 && <span className="badge">{selectedClip.transcript.length}</span>}
            </button>
          </>
        ) : isTextClip ? (
          <>
            <button className={`tab-btn ${activeTab === 'text' ? 'active' : ''}`} onClick={() => setActiveTab('text')}>Text</button>
            <button className={`tab-btn ${activeTab === 'transform' ? 'active' : ''}`} onClick={() => setActiveTab('transform')}>Transform</button>
            <button className={`tab-btn ${activeTab === 'effects' ? 'active' : ''}`} onClick={() => setActiveTab('effects')}>
              Effects {visualEffects.length > 0 && <span className="badge">{visualEffects.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'masks' ? 'active' : ''}`} onClick={() => setActiveTab('masks')}>
              Masks {selectedClip.masks && selectedClip.masks.length > 0 && <span className="badge">{selectedClip.masks.length}</span>}
            </button>
          </>
        ) : (
          <>
            <button className={`tab-btn ${activeTab === 'transform' ? 'active' : ''}`} onClick={() => setActiveTab('transform')}>Transform</button>
            {!isSolidClip && (
              <button className={`tab-btn ${activeTab === 'volume' ? 'active' : ''}`} onClick={() => setActiveTab('volume')}>Audio</button>
            )}
            <button className={`tab-btn ${activeTab === 'effects' ? 'active' : ''}`} onClick={() => setActiveTab('effects')}>
              Effects {visualEffects.length > 0 && <span className="badge">{visualEffects.length}</span>}
            </button>
            <button className={`tab-btn ${activeTab === 'masks' ? 'active' : ''}`} onClick={() => setActiveTab('masks')}>
              Masks {selectedClip.masks && selectedClip.masks.length > 0 && <span className="badge">{selectedClip.masks.length}</span>}
            </button>
            {!isSolidClip && (
              <>
                <button className={`tab-btn ${activeTab === 'transcript' ? 'active' : ''}`} onClick={() => setActiveTab('transcript')}>
                  Transcript {selectedClip.transcript && selectedClip.transcript.length > 0 && <span className="badge">{selectedClip.transcript.length}</span>}
                </button>
                <button className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>
                  Analysis {selectedClip.analysisStatus === 'ready' && <span className="badge">✓</span>}
                </button>
              </>
            )}
          </>
        )}
      </div>

      <div className="properties-content">
        <Suspense fallback={<TabLoading />}>
          {activeTab === 'text' && isTextClip && selectedClip.textProperties && (
            <TextTab clipId={selectedClip.id} textProperties={selectedClip.textProperties} />
          )}
          {activeTab === 'transform' && !isAudioClip && <TransformTab clipId={selectedClip.id} transform={transform} speed={interpolatedSpeed} />}
          {activeTab === 'volume' && <VolumeTab clipId={selectedClip.id} effects={selectedClip.effects || []} />}
          {activeTab === 'effects' && <EffectsTab clipId={selectedClip.id} effects={selectedClip.effects || []} />}
          {activeTab === 'masks' && !isAudioClip && <MasksTab clipId={selectedClip.id} masks={selectedClip.masks} />}
          {activeTab === 'transcript' && (
            <TranscriptTab
              clipId={selectedClip.id}
              transcript={selectedClip.transcript || []}
              transcriptStatus={selectedClip.transcriptStatus || 'none'}
              transcriptProgress={selectedClip.transcriptProgress || 0}
              clipStartTime={selectedClip.startTime}
              inPoint={selectedClip.inPoint}
            />
          )}
          {activeTab === 'analysis' && !isAudioClip && (
            <AnalysisTab
              clipId={selectedClip.id}
              analysis={selectedClip.analysis}
              analysisStatus={selectedClip.analysisStatus || 'none'}
              analysisProgress={selectedClip.analysisProgress || 0}
              clipStartTime={selectedClip.startTime}
              inPoint={selectedClip.inPoint}
              outPoint={selectedClip.outPoint}
            />
          )}
        </Suspense>
      </div>
    </div>
  );
}
