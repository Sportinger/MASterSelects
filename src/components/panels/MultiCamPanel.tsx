// MultiCam Panel - AI-powered multicam editing interface

import { useState, useCallback, useMemo } from 'react';
import { useMultiCamStore } from '../../stores/multicamStore';
import { useMediaStore, type MediaFile } from '../../stores/mediaStore';
import './MultiCamPanel.css';

// =============================================================================
// Sub-components
// =============================================================================

interface CameraCardProps {
  camera: {
    id: string;
    name: string;
    role: string;
    syncOffset: number;
    thumbnailUrl?: string;
  };
  isMaster: boolean;
  isSelected: boolean;
  onSelect: () => void;
  onSetMaster: () => void;
  onRemove: () => void;
  onUpdateRole: (role: string) => void;
}

function CameraCard({
  camera,
  isMaster,
  isSelected,
  onSelect,
  onSetMaster,
  onRemove,
  onUpdateRole,
}: CameraCardProps) {
  return (
    <div
      className={`multicam-camera-card ${isSelected ? 'selected' : ''} ${isMaster ? 'master' : ''}`}
      onClick={onSelect}
    >
      <div className="camera-thumbnail">
        {camera.thumbnailUrl ? (
          <img src={camera.thumbnailUrl} alt={camera.name} />
        ) : (
          <div className="camera-placeholder">CAM</div>
        )}
        {isMaster && <span className="master-badge" title="Master (audio reference)">M</span>}
      </div>
      <div className="camera-info">
        <div className="camera-name" title={camera.name}>
          {camera.name}
        </div>
        <select
          className="camera-role"
          value={camera.role}
          onChange={(e) => onUpdateRole(e.target.value)}
          onClick={(e) => e.stopPropagation()}
        >
          <option value="wide">Wide</option>
          <option value="closeup">Close-up</option>
          <option value="detail">Detail</option>
          <option value="custom">Custom</option>
        </select>
        {camera.syncOffset !== 0 && (
          <div className="sync-offset">
            {camera.syncOffset > 0 ? '+' : ''}{(camera.syncOffset / 1000).toFixed(2)}s
          </div>
        )}
      </div>
      <div className="camera-actions">
        {!isMaster && (
          <button
            className="btn-icon"
            title="Set as master (audio reference)"
            onClick={(e) => { e.stopPropagation(); onSetMaster(); }}
          >
            M
          </button>
        )}
        <button
          className="btn-icon btn-danger"
          title="Remove camera"
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
        >
          x
        </button>
      </div>
    </div>
  );
}

interface AddCameraDropdownProps {
  availableFiles: MediaFile[];
  onAdd: (file: MediaFile) => void;
}

function AddCameraDropdown({ availableFiles, onAdd }: AddCameraDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (availableFiles.length === 0) {
    return (
      <div className="add-camera-empty">
        <span>Import video files in Media panel first</span>
      </div>
    );
  }

  return (
    <div className="add-camera-dropdown">
      <button
        className="add-camera-btn"
        onClick={() => setIsOpen(!isOpen)}
      >
        + Add Camera
      </button>
      {isOpen && (
        <div className="dropdown-menu">
          {availableFiles.map((file) => (
            <button
              key={file.id}
              className="dropdown-item"
              onClick={() => {
                onAdd(file);
                setIsOpen(false);
              }}
            >
              {file.thumbnailUrl && (
                <img src={file.thumbnailUrl} alt="" className="dropdown-thumb" />
              )}
              <span>{file.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface ProgressBarProps {
  progress: number;
  status: string;
  label?: string;
}

function ProgressBar({ progress, status, label }: ProgressBarProps) {
  return (
    <div className="progress-container">
      {label && <span className="progress-label">{label}</span>}
      <div className="progress-bar">
        <div
          className={`progress-fill ${status}`}
          style={{ width: `${progress}%` }}
        />
      </div>
      <span className="progress-text">{progress}%</span>
    </div>
  );
}

interface TranscriptViewerProps {
  entries: Array<{
    id: string;
    start: number;
    end: number;
    speaker: string;
    text: string;
  }>;
  onUpdate: (id: string, updates: { speaker?: string; text?: string }) => void;
}

function TranscriptViewer({ entries, onUpdate }: TranscriptViewerProps) {
  const formatTime = (ms: number) => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (entries.length === 0) {
    return (
      <div className="transcript-empty">
        No transcript available. Click "Generate" to create one.
      </div>
    );
  }

  return (
    <div className="transcript-viewer">
      {entries.map((entry) => (
        <div key={entry.id} className="transcript-entry">
          <span className="transcript-time">{formatTime(entry.start)}</span>
          <input
            className="transcript-speaker"
            value={entry.speaker}
            onChange={(e) => onUpdate(entry.id, { speaker: e.target.value })}
            placeholder="Speaker"
          />
          <span className="transcript-text">{entry.text}</span>
        </div>
      ))}
    </div>
  );
}

interface EDLPreviewProps {
  decisions: Array<{
    id: string;
    start: number;
    end: number;
    cameraId: string;
    reason?: string;
  }>;
  cameras: Array<{ id: string; name: string }>;
  totalDuration: number;
}

function EDLPreview({ decisions, cameras, totalDuration }: EDLPreviewProps) {
  const getCameraColor = (index: number) => {
    const colors = ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b'];
    return colors[index % colors.length];
  };

  const cameraColorMap = useMemo(() => {
    const map = new Map<string, string>();
    cameras.forEach((cam, i) => map.set(cam.id, getCameraColor(i)));
    return map;
  }, [cameras]);

  const formatTime = (ms: number) => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  if (decisions.length === 0) {
    return (
      <div className="edl-empty">
        No edit decisions yet. Click "Generate Edit" to create an EDL.
      </div>
    );
  }

  return (
    <div className="edl-preview">
      <div className="edl-legend">
        {cameras.map((cam, i) => (
          <span key={cam.id} className="edl-legend-item">
            <span
              className="edl-legend-color"
              style={{ backgroundColor: getCameraColor(i) }}
            />
            {cam.name}
          </span>
        ))}
      </div>
      <div className="edl-timeline">
        {decisions.map((decision) => {
          const camera = cameras.find(c => c.id === decision.cameraId);
          const left = (decision.start / totalDuration) * 100;
          const width = ((decision.end - decision.start) / totalDuration) * 100;

          return (
            <div
              key={decision.id}
              className="edl-clip"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                backgroundColor: cameraColorMap.get(decision.cameraId) || '#666',
              }}
              title={`${camera?.name || 'Unknown'}: ${formatTime(decision.start)} - ${formatTime(decision.end)}${decision.reason ? `\n${decision.reason}` : ''}`}
            >
              <span className="edl-clip-label">{camera?.name?.substring(0, 8)}</span>
            </div>
          );
        })}
      </div>
      <div className="edl-time-markers">
        <span>0:00</span>
        <span>{formatTime(totalDuration / 2)}</span>
        <span>{formatTime(totalDuration)}</span>
      </div>
    </div>
  );
}

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiKeySet: boolean;
  editStyle: string;
  customPrompt: string;
  onSetApiKey: (key: string) => void;
  onClearApiKey: () => void;
  onSetEditStyle: (style: string) => void;
  onSetCustomPrompt: (prompt: string) => void;
}

function SettingsModal({
  isOpen,
  onClose,
  apiKeySet,
  editStyle,
  customPrompt,
  onSetApiKey,
  onClearApiKey,
  onSetEditStyle,
  onSetCustomPrompt,
}: SettingsModalProps) {
  const [newApiKey, setNewApiKey] = useState('');

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>MultiCam Settings</h3>
          <button className="btn-close" onClick={onClose}>x</button>
        </div>
        <div className="modal-body">
          <div className="settings-section">
            <label>Claude API Key</label>
            {apiKeySet ? (
              <div className="api-key-status">
                <span className="api-key-set">API key configured</span>
                <button className="btn-small btn-danger" onClick={onClearApiKey}>
                  Clear
                </button>
              </div>
            ) : (
              <div className="api-key-input">
                <input
                  type="password"
                  placeholder="sk-ant-..."
                  value={newApiKey}
                  onChange={(e) => setNewApiKey(e.target.value)}
                />
                <button
                  className="btn-small btn-primary"
                  onClick={() => {
                    if (newApiKey.trim()) {
                      onSetApiKey(newApiKey.trim());
                      setNewApiKey('');
                    }
                  }}
                  disabled={!newApiKey.trim()}
                >
                  Save
                </button>
              </div>
            )}
            <p className="settings-hint">
              Your API key is encrypted and stored locally.
            </p>
          </div>

          <div className="settings-section">
            <label>Edit Style</label>
            <select
              value={editStyle}
              onChange={(e) => onSetEditStyle(e.target.value)}
            >
              <option value="podcast">Podcast</option>
              <option value="interview">Interview</option>
              <option value="music">Music Video</option>
              <option value="documentary">Documentary</option>
              <option value="custom">Custom Only</option>
            </select>
          </div>

          <div className="settings-section">
            <label>Custom Instructions</label>
            <textarea
              placeholder="Add custom editing instructions (e.g., 'Focus on speaker reactions', 'Cut on beat drops')"
              value={customPrompt}
              onChange={(e) => onSetCustomPrompt(e.target.value)}
              rows={4}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function MultiCamPanel() {
  const [showSettings, setShowSettings] = useState(false);

  // MultiCam store
  const {
    cameras,
    masterCameraId,
    analysis,
    analysisProgress,
    analysisStatus,
    transcript,
    transcriptProgress,
    transcriptStatus,
    edl,
    edlStatus,
    edlError,
    apiKeySet,
    editStyle,
    customPrompt,
    selectedCameraId,
    addCamera,
    removeCamera,
    updateCamera,
    setMasterCamera,
    selectCamera,
    syncCameras,
    analyzeAll,
    generateTranscript,
    generateEDL,
    updateTranscriptEntry,
    applyEDLToTimeline,
    setApiKey,
    clearApiKey,
    setEditStyle,
    setCustomPrompt,
  } = useMultiCamStore();

  // Media store - get available video files
  const { files } = useMediaStore();
  const videoFiles = useMemo(
    () => files.filter(f => f.type === 'video'),
    [files]
  );

  // Files not yet added as cameras
  const availableFiles = useMemo(
    () => videoFiles.filter(f => !cameras.some(c => c.mediaFileId === f.id)),
    [videoFiles, cameras]
  );

  // Calculate total duration from cameras
  const totalDuration = useMemo(() => {
    if (cameras.length === 0) return 0;
    return Math.max(...cameras.map(c => c.duration));
  }, [cameras]);

  // Handlers
  const handleAddCamera = useCallback((file: MediaFile) => {
    addCamera(file);
  }, [addCamera]);

  const handleSync = useCallback(async () => {
    await syncCameras();
  }, [syncCameras]);

  const handleAnalyze = useCallback(async () => {
    await analyzeAll();
  }, [analyzeAll]);

  const handleGenerateTranscript = useCallback(async () => {
    await generateTranscript();
  }, [generateTranscript]);

  const handleGenerateEDL = useCallback(async () => {
    await generateEDL();
  }, [generateEDL]);

  const handleApplyToTimeline = useCallback(() => {
    applyEDLToTimeline();
  }, [applyEDLToTimeline]);

  const handleSetApiKey = useCallback(async (key: string) => {
    await setApiKey(key);
  }, [setApiKey]);

  const handleClearApiKey = useCallback(async () => {
    await clearApiKey();
  }, [clearApiKey]);

  return (
    <div className="multicam-panel">
      {/* Header */}
      <div className="multicam-header">
        <h2>Multi-Cam Editor</h2>
        <button
          className="btn-icon settings-btn"
          title="Settings"
          onClick={() => setShowSettings(true)}
        >
          Settings
        </button>
      </div>

      {/* Camera Grid */}
      <section className="multicam-section">
        <div className="section-header">
          <h3>Cameras ({cameras.length})</h3>
          {cameras.length >= 2 && (
            <button
              className="btn-small btn-primary"
              onClick={handleSync}
              disabled={analysisStatus === 'analyzing'}
            >
              Sync Audio
            </button>
          )}
        </div>
        <div className="camera-grid">
          {cameras.map((camera) => (
            <CameraCard
              key={camera.id}
              camera={camera}
              isMaster={camera.id === masterCameraId}
              isSelected={camera.id === selectedCameraId}
              onSelect={() => selectCamera(camera.id)}
              onSetMaster={() => setMasterCamera(camera.id)}
              onRemove={() => removeCamera(camera.id)}
              onUpdateRole={(role) => updateCamera(camera.id, { role: role as 'wide' | 'closeup' | 'detail' | 'custom' })}
            />
          ))}
          <AddCameraDropdown
            availableFiles={availableFiles}
            onAdd={handleAddCamera}
          />
        </div>
      </section>

      {/* Analysis Section */}
      <section className="multicam-section">
        <div className="section-header">
          <h3>Analysis</h3>
          <button
            className="btn-small btn-primary"
            onClick={handleAnalyze}
            disabled={cameras.length === 0 || analysisStatus === 'analyzing'}
          >
            {analysisStatus === 'analyzing' ? 'Analyzing...' : 'Analyze All'}
          </button>
        </div>
        {analysisStatus === 'analyzing' && (
          <ProgressBar
            progress={analysisProgress}
            status={analysisStatus}
            label="Analyzing frames..."
          />
        )}
        {analysisStatus === 'complete' && analysis && (
          <div className="analysis-summary">
            <span>{analysis.cameras.length} cameras analyzed</span>
            <span>{Math.round(totalDuration / 1000)}s duration</span>
          </div>
        )}
        {analysisStatus === 'error' && (
          <div className="error-message">Analysis failed. Please try again.</div>
        )}
      </section>

      {/* Transcript Section */}
      <section className="multicam-section">
        <div className="section-header">
          <h3>Transcript</h3>
          <button
            className="btn-small btn-primary"
            onClick={handleGenerateTranscript}
            disabled={cameras.length === 0 || transcriptStatus === 'generating' || transcriptStatus === 'loading-model'}
          >
            {transcriptStatus === 'loading-model' ? 'Loading model...' :
             transcriptStatus === 'generating' ? 'Generating...' : 'Generate'}
          </button>
        </div>
        {(transcriptStatus === 'generating' || transcriptStatus === 'loading-model') && (
          <ProgressBar
            progress={transcriptProgress}
            status={transcriptStatus}
            label={transcriptStatus === 'loading-model' ? 'Loading Whisper model...' : 'Transcribing...'}
          />
        )}
        <TranscriptViewer
          entries={transcript}
          onUpdate={(id, updates) => updateTranscriptEntry(id, updates)}
        />
      </section>

      {/* EDL Generation Section */}
      <section className="multicam-section">
        <div className="section-header">
          <h3>Edit Style</h3>
          <select
            className="style-select"
            value={editStyle}
            onChange={(e) => setEditStyle(e.target.value as typeof editStyle)}
          >
            <option value="podcast">Podcast</option>
            <option value="interview">Interview</option>
            <option value="music">Music Video</option>
            <option value="documentary">Documentary</option>
            <option value="custom">Custom</option>
          </select>
          <button
            className="btn-small btn-primary"
            onClick={handleGenerateEDL}
            disabled={cameras.length === 0 || edlStatus === 'generating' || !apiKeySet}
            title={!apiKeySet ? 'Configure API key in settings first' : ''}
          >
            {edlStatus === 'generating' ? 'Generating...' : 'Generate Edit'}
          </button>
        </div>
        {!apiKeySet && (
          <div className="warning-message">
            Configure your Claude API key in settings to generate edits.
          </div>
        )}
        {edlStatus === 'error' && edlError && (
          <div className="error-message">{edlError}</div>
        )}
      </section>

      {/* EDL Preview */}
      <section className="multicam-section edl-section">
        <div className="section-header">
          <h3>EDL Preview</h3>
          {edl.length > 0 && (
            <button
              className="btn-small btn-success"
              onClick={handleApplyToTimeline}
            >
              Apply to Timeline
            </button>
          )}
        </div>
        <EDLPreview
          decisions={edl}
          cameras={cameras}
          totalDuration={totalDuration || 60000}
        />
      </section>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        apiKeySet={apiKeySet}
        editStyle={editStyle}
        customPrompt={customPrompt}
        onSetApiKey={handleSetApiKey}
        onClearApiKey={handleClearApiKey}
        onSetEditStyle={setEditStyle}
        onSetCustomPrompt={setCustomPrompt}
      />
    </div>
  );
}
