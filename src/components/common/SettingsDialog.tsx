// Settings Dialog - After Effects style preferences with sidebar navigation

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSettingsStore, type TranscriptionProvider, type PreviewQuality, type AutosaveInterval, type GPUPowerPreference } from '../../stores/settingsStore';
import { useIsMobile } from '../../hooks/useIsMobile';

interface SettingsDialogProps {
  onClose: () => void;
}

type SettingsCategory =
  | 'general'
  | 'previews'
  | 'import'
  | 'transcription'
  | 'output'
  | 'performance'
  | 'apiKeys';

interface CategoryConfig {
  id: SettingsCategory;
  label: string;
  icon: string;
}

const categories: CategoryConfig[] = [
  { id: 'general', label: 'General', icon: '⚙' },
  { id: 'previews', label: 'Previews', icon: '▶' },
  { id: 'import', label: 'Import', icon: '📥' },
  { id: 'transcription', label: 'Transcription', icon: '🎤' },
  { id: 'output', label: 'Output', icon: '📤' },
  { id: 'performance', label: 'Performance', icon: '⚡' },
  { id: 'apiKeys', label: 'API Keys', icon: '🔑' },
];

export function SettingsDialog({ onClose }: SettingsDialogProps) {
  const [activeCategory, setActiveCategory] = useState<SettingsCategory>('general');

  // Drag state
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const dialogRef = useRef<HTMLDivElement>(null);

  // Center dialog on mount
  useEffect(() => {
    if (dialogRef.current) {
      const rect = dialogRef.current.getBoundingClientRect();
      setPosition({
        x: (window.innerWidth - rect.width) / 2,
        y: (window.innerHeight - rect.height) / 2,
      });
    }
  }, []);

  // Handle drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (dialogRef.current) {
      const rect = dialogRef.current.getBoundingClientRect();
      dragOffset.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
      setIsDragging(true);
    }
  }, []);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newX = e.clientX - dragOffset.current.x;
      const newY = e.clientY - dragOffset.current.y;

      // Keep dialog within viewport bounds
      const maxX = window.innerWidth - (dialogRef.current?.offsetWidth || 720);
      const maxY = window.innerHeight - (dialogRef.current?.offsetHeight || 560);

      setPosition({
        x: Math.max(0, Math.min(newX, maxX)),
        y: Math.max(0, Math.min(newY, maxY)),
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  const {
    apiKeys,
    transcriptionProvider,
    previewQuality,
    showTransparencyGrid,
    autosaveEnabled,
    autosaveInterval,
    turboModeEnabled,
    nativeHelperPort,
    nativeHelperConnected,
    forceDesktopMode,
    gpuPowerPreference,
    copyMediaToProject,
    outputResolution,
    fps,
    setApiKey,
    setTranscriptionProvider,
    setPreviewQuality,
    setShowTransparencyGrid,
    setAutosaveEnabled,
    setAutosaveInterval,
    setTurboModeEnabled,
    setNativeHelperPort,
    setGpuPowerPreference,
    setCopyMediaToProject,
    setForceDesktopMode,
    setResolution,
  } = useSettingsStore();

  const isMobileDevice = useIsMobile();

  // Local state for API keys (to avoid saving on every keystroke)
  const [localKeys, setLocalKeys] = useState(apiKeys);
  const [showKeys, setShowKeys] = useState({
    openai: false,
    assemblyai: false,
    deepgram: false,
    piapi: false,
    youtube: false,
  });

  const handleSave = useCallback(() => {
    Object.entries(localKeys).forEach(([provider, key]) => {
      setApiKey(provider as keyof typeof apiKeys, key);
    });
    onClose();
  }, [localKeys, setApiKey, onClose]);

  const handleKeyChange = (provider: keyof typeof apiKeys, value: string) => {
    setLocalKeys((prev) => ({ ...prev, [provider]: value }));
  };

  const toggleShowKey = (provider: keyof typeof showKeys) => {
    setShowKeys((prev) => ({ ...prev, [provider]: !prev[provider] }));
  };

  const handleSwitchToMobile = useCallback(() => {
    setForceDesktopMode(false);
    window.location.reload();
  }, [setForceDesktopMode]);

  const providers: { id: TranscriptionProvider; label: string; description: string }[] = [
    { id: 'local', label: 'Local (Whisper)', description: 'Runs in browser, no API key needed. Slower, less accurate.' },
    { id: 'openai', label: 'OpenAI Whisper API', description: 'High accuracy, $0.006/minute. Requires API key.' },
    { id: 'assemblyai', label: 'AssemblyAI', description: 'Excellent accuracy, speaker diarization. $0.015/minute.' },
    { id: 'deepgram', label: 'Deepgram', description: 'Fast, good accuracy. $0.0125/minute.' },
  ];

  const renderCategoryContent = () => {
    switch (activeCategory) {
      case 'general':
        return (
          <div className="settings-category-content">
            <h2>General</h2>

            {/* Autosave */}
            <div className="settings-group">
              <div className="settings-group-title">Autosave</div>

              <label className="settings-row">
                <span className="settings-label">Enable Autosave</span>
                <input
                  type="checkbox"
                  checked={autosaveEnabled}
                  onChange={(e) => setAutosaveEnabled(e.target.checked)}
                  className="settings-checkbox"
                />
              </label>

              <label className="settings-row">
                <span className="settings-label">Autosave Interval</span>
                <select
                  value={autosaveInterval}
                  onChange={(e) => setAutosaveInterval(Number(e.target.value) as AutosaveInterval)}
                  disabled={!autosaveEnabled}
                  className="settings-select"
                >
                  <option value={1}>1 minute</option>
                  <option value={2}>2 minutes</option>
                  <option value={5}>5 minutes</option>
                  <option value={10}>10 minutes</option>
                </select>
              </label>
            </div>

            {/* View Mode - only on mobile */}
            {isMobileDevice && forceDesktopMode && (
              <div className="settings-group">
                <div className="settings-group-title">View Mode</div>
                <p className="settings-description">
                  You're viewing the desktop interface on a mobile device.
                </p>
                <button className="settings-button" onClick={handleSwitchToMobile}>
                  Switch to Mobile View
                </button>
              </div>
            )}
          </div>
        );

      case 'previews':
        return (
          <div className="settings-category-content">
            <h2>Previews</h2>

            <div className="settings-group">
              <div className="settings-group-title">Quality</div>

              <label className="settings-row">
                <span className="settings-label">Preview Resolution</span>
                <select
                  value={previewQuality}
                  onChange={(e) => setPreviewQuality(Number(e.target.value) as PreviewQuality)}
                  className="settings-select"
                >
                  <option value={1}>Full (100%)</option>
                  <option value={0.5}>Half (50%)</option>
                  <option value={0.25}>Quarter (25%)</option>
                </select>
              </label>
              <p className="settings-hint">Lower resolution improves playback performance.</p>
            </div>

            <div className="settings-group">
              <div className="settings-group-title">Transparency</div>

              <label className="settings-row">
                <span className="settings-label">Show Transparency Grid</span>
                <input
                  type="checkbox"
                  checked={showTransparencyGrid}
                  onChange={(e) => setShowTransparencyGrid(e.target.checked)}
                  className="settings-checkbox"
                />
              </label>
              <p className="settings-hint">Display checkerboard pattern for transparent areas.</p>
            </div>
          </div>
        );

      case 'import':
        return (
          <div className="settings-category-content">
            <h2>Import</h2>

            <div className="settings-group">
              <div className="settings-group-title">Media Import</div>

              <label className="settings-row">
                <span className="settings-label">Copy media to project folder</span>
                <input
                  type="checkbox"
                  checked={copyMediaToProject}
                  onChange={(e) => setCopyMediaToProject(e.target.checked)}
                  className="settings-checkbox"
                />
              </label>
              <p className="settings-hint">
                When importing clips, copy them to the project's Raw folder for easier relinking.
              </p>
            </div>
          </div>
        );

      case 'transcription':
        return (
          <div className="settings-category-content">
            <h2>Transcription</h2>

            <div className="settings-group">
              <div className="settings-group-title">Provider</div>

              <div className="provider-list">
                {providers.map((provider) => (
                  <label
                    key={provider.id}
                    className={`provider-option ${transcriptionProvider === provider.id ? 'active' : ''}`}
                  >
                    <input
                      type="radio"
                      name="transcriptionProvider"
                      value={provider.id}
                      checked={transcriptionProvider === provider.id}
                      onChange={() => setTranscriptionProvider(provider.id)}
                    />
                    <div className="provider-info">
                      <span className="provider-label">{provider.label}</span>
                      <span className="provider-description">{provider.description}</span>
                    </div>
                    {provider.id !== 'local' && localKeys[provider.id] && (
                      <span className="provider-status">✓</span>
                    )}
                  </label>
                ))}
              </div>
              <p className="settings-hint">
                API keys for transcription providers can be configured in the API Keys section.
              </p>
            </div>
          </div>
        );

      case 'output':
        return (
          <div className="settings-category-content">
            <h2>Output</h2>

            <div className="settings-group">
              <div className="settings-group-title">Composition Settings</div>

              <label className="settings-row">
                <span className="settings-label">Width</span>
                <input
                  type="number"
                  value={outputResolution.width}
                  onChange={(e) => setResolution(Number(e.target.value), outputResolution.height)}
                  className="settings-input settings-input-number"
                  min={1}
                  max={7680}
                />
              </label>

              <label className="settings-row">
                <span className="settings-label">Height</span>
                <input
                  type="number"
                  value={outputResolution.height}
                  onChange={(e) => setResolution(outputResolution.width, Number(e.target.value))}
                  className="settings-input settings-input-number"
                  min={1}
                  max={4320}
                />
              </label>

              <div className="preset-buttons">
                <button className="preset-btn" onClick={() => setResolution(1920, 1080)}>1080p</button>
                <button className="preset-btn" onClick={() => setResolution(2560, 1440)}>1440p</button>
                <button className="preset-btn" onClick={() => setResolution(3840, 2160)}>4K</button>
                <button className="preset-btn" onClick={() => setResolution(1080, 1920)}>9:16</button>
              </div>
            </div>

            <div className="settings-group">
              <div className="settings-group-title">Frame Rate</div>
              <p className="settings-hint">
                Current: {fps} FPS (configured per composition)
              </p>
            </div>
          </div>
        );

      case 'performance':
        return (
          <div className="settings-category-content">
            <h2>Performance</h2>

            <div className="settings-group">
              <div className="settings-group-title">GPU</div>

              <label className="settings-row">
                <span className="settings-label">GPU Power Preference</span>
                <select
                  value={gpuPowerPreference}
                  onChange={(e) => setGpuPowerPreference(e.target.value as GPUPowerPreference)}
                  className="settings-select"
                >
                  <option value="high-performance">High Performance (Discrete GPU)</option>
                  <option value="low-power">Low Power (Integrated GPU)</option>
                </select>
              </label>
              <p className="settings-hint">
                Requires page reload to take effect.
              </p>
            </div>

            <div className="settings-group">
              <div className="settings-group-title">Native Helper (Turbo Mode)</div>

              <label className="settings-row">
                <span className="settings-label">Enable Turbo Mode</span>
                <input
                  type="checkbox"
                  checked={turboModeEnabled}
                  onChange={(e) => setTurboModeEnabled(e.target.checked)}
                  className="settings-checkbox"
                />
              </label>

              <label className="settings-row">
                <span className="settings-label">WebSocket Port</span>
                <input
                  type="number"
                  value={nativeHelperPort}
                  onChange={(e) => setNativeHelperPort(Number(e.target.value))}
                  className="settings-input settings-input-number"
                  min={1024}
                  max={65535}
                  disabled={!turboModeEnabled}
                />
              </label>

              <div className="settings-status">
                <span className={`status-indicator ${nativeHelperConnected ? 'connected' : 'disconnected'}`} />
                <span className="status-text">
                  {nativeHelperConnected ? 'Connected' : 'Not Connected'}
                </span>
              </div>
              <p className="settings-hint">
                Uses native FFmpeg for faster video decoding when available.
              </p>
            </div>
          </div>
        );

      case 'apiKeys':
        return (
          <div className="settings-category-content">
            <h2>API Keys</h2>
            <p className="settings-hint" style={{ marginTop: 0, marginBottom: 8 }}>
              Keys are stored locally and encrypted in your browser.
            </p>

            <div className="settings-group">
              <div className="settings-group-title">Transcription</div>

              <div className="api-key-row">
                <label>OpenAI API Key</label>
                <div className="api-key-input">
                  <input
                    type={showKeys.openai ? 'text' : 'password'}
                    value={localKeys.openai}
                    onChange={(e) => handleKeyChange('openai', e.target.value)}
                    placeholder="sk-..."
                  />
                  <button
                    className="toggle-visibility"
                    onClick={() => toggleShowKey('openai')}
                  >
                    {showKeys.openai ? '👁' : '○'}
                  </button>
                </div>
                <a className="api-key-link" href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer">
                  Get API Key
                </a>
              </div>

              <div className="api-key-row">
                <label>AssemblyAI API Key</label>
                <div className="api-key-input">
                  <input
                    type={showKeys.assemblyai ? 'text' : 'password'}
                    value={localKeys.assemblyai}
                    onChange={(e) => handleKeyChange('assemblyai', e.target.value)}
                    placeholder="Enter API key..."
                  />
                  <button
                    className="toggle-visibility"
                    onClick={() => toggleShowKey('assemblyai')}
                  >
                    {showKeys.assemblyai ? '👁' : '○'}
                  </button>
                </div>
                <a className="api-key-link" href="https://www.assemblyai.com/dashboard/signup" target="_blank" rel="noopener noreferrer">
                  Get API Key
                </a>
              </div>

              <div className="api-key-row">
                <label>Deepgram API Key</label>
                <div className="api-key-input">
                  <input
                    type={showKeys.deepgram ? 'text' : 'password'}
                    value={localKeys.deepgram}
                    onChange={(e) => handleKeyChange('deepgram', e.target.value)}
                    placeholder="Enter API key..."
                  />
                  <button
                    className="toggle-visibility"
                    onClick={() => toggleShowKey('deepgram')}
                  >
                    {showKeys.deepgram ? '👁' : '○'}
                  </button>
                </div>
                <a className="api-key-link" href="https://console.deepgram.com/signup" target="_blank" rel="noopener noreferrer">
                  Get API Key
                </a>
              </div>
            </div>

            <div className="settings-group">
              <div className="settings-group-title">AI Video Generation</div>

              <div className="api-key-row">
                <label>PiAPI API Key</label>
                <div className="api-key-input">
                  <input
                    type={showKeys.piapi ? 'text' : 'password'}
                    value={localKeys.piapi}
                    onChange={(e) => handleKeyChange('piapi', e.target.value)}
                    placeholder="Enter PiAPI key..."
                  />
                  <button
                    className="toggle-visibility"
                    onClick={() => toggleShowKey('piapi')}
                  >
                    {showKeys.piapi ? '👁' : '○'}
                  </button>
                </div>
                <a className="api-key-link" href="https://piapi.ai/workspace" target="_blank" rel="noopener noreferrer">
                  Get API Key
                </a>
              </div>
            </div>

            <div className="settings-group">
              <div className="settings-group-title">YouTube</div>

              <div className="api-key-row">
                <label>YouTube Data API v3 Key</label>
                <div className="api-key-input">
                  <input
                    type={showKeys.youtube ? 'text' : 'password'}
                    value={localKeys.youtube}
                    onChange={(e) => handleKeyChange('youtube', e.target.value)}
                    placeholder="Enter YouTube API key..."
                  />
                  <button
                    className="toggle-visibility"
                    onClick={() => toggleShowKey('youtube')}
                  >
                    {showKeys.youtube ? '👁' : '○'}
                  </button>
                </div>
                <a className="api-key-link" href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener noreferrer">
                  Get API Key
                </a>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="settings-container">
      <div
        ref={dialogRef}
        className={`settings-dialog ${isDragging ? 'dragging' : ''}`}
        style={{
          left: position.x,
          top: position.y,
        }}
      >
        {/* Header - Draggable */}
        <div
          className="settings-header"
          onMouseDown={handleMouseDown}
        >
          <h1>Preferences</h1>
          <button className="settings-close" onClick={onClose} onMouseDown={(e) => e.stopPropagation()}>×</button>
        </div>

        {/* Main content with sidebar */}
        <div className="settings-main">
          {/* Sidebar */}
          <div className="settings-sidebar">
            {categories.map((cat) => (
              <button
                key={cat.id}
                className={`sidebar-item ${activeCategory === cat.id ? 'active' : ''}`}
                onClick={() => setActiveCategory(cat.id)}
              >
                <span className="sidebar-icon">{cat.icon}</span>
                <span className="sidebar-label">{cat.label}</span>
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="settings-content">
            {renderCategoryContent()}
          </div>
        </div>

        {/* Footer */}
        <div className="settings-footer">
          <button className="btn-cancel" onClick={onClose}>Cancel</button>
          <button className="btn-save" onClick={handleSave}>OK</button>
        </div>

        <style>{`
          .settings-container {
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 10000;
          }

          .settings-dialog {
            position: fixed;
            pointer-events: auto;
            background: var(--bg-primary, #1a1a1a);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            width: 640px;
            max-width: 95vw;
            height: 480px;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
          }

          .settings-dialog.dragging {
            cursor: grabbing;
            user-select: none;
          }

          /* Header */
          .settings-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            flex-shrink: 0;
            cursor: grab;
            user-select: none;
          }

          .settings-dialog.dragging .settings-header {
            cursor: grabbing;
          }

          .settings-header h1 {
            margin: 0;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            pointer-events: none;
          }

          .settings-close {
            background: none;
            border: none;
            font-size: 16px;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0;
            line-height: 1;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 3px;
          }

          .settings-close:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
          }

          /* Main layout */
          .settings-main {
            display: flex;
            flex: 1;
            min-height: 0;
          }

          /* Sidebar */
          .settings-sidebar {
            width: 140px;
            flex-shrink: 0;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            padding: 4px 0;
            overflow-y: auto;
          }

          .sidebar-item {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            padding: 6px 12px;
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 11px;
            text-align: left;
            cursor: pointer;
            transition: all 0.15s;
          }

          .sidebar-item:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
          }

          .sidebar-item.active {
            background: var(--accent);
            color: white;
          }

          .sidebar-icon {
            font-size: 12px;
            width: 16px;
            text-align: center;
          }

          .sidebar-label {
            font-weight: 500;
          }

          /* Content area */
          .settings-content {
            flex: 1;
            padding: 10px 14px;
            overflow-y: auto;
            background: var(--bg-primary, #1a1a1a);
          }

          .settings-category-content h2 {
            margin: 0 0 10px 0;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border-color);
          }

          .settings-group {
            margin-bottom: 14px;
          }

          .settings-group:last-child {
            margin-bottom: 0;
          }

          .settings-group-title {
            font-size: 10px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
          }

          .settings-description {
            font-size: 11px;
            color: var(--text-secondary);
            margin: 0 0 8px 0;
            line-height: 1.4;
          }

          .settings-hint {
            font-size: 10px;
            color: var(--text-secondary);
            margin: 4px 0 0 0;
            opacity: 0.8;
          }

          /* Settings row */
          .settings-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          }

          .settings-row:last-child {
            border-bottom: none;
          }

          .settings-label {
            font-size: 11px;
            color: var(--text-primary);
          }

          .settings-checkbox {
            width: 14px;
            height: 14px;
            accent-color: var(--accent);
            cursor: pointer;
          }

          .settings-select {
            padding: 4px 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 3px;
            color: var(--text-primary);
            font-size: 11px;
            min-width: 160px;
            cursor: pointer;
          }

          .settings-select:focus {
            outline: none;
            border-color: var(--accent);
          }

          .settings-select:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }

          .settings-input {
            padding: 4px 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 3px;
            color: var(--text-primary);
            font-size: 11px;
          }

          .settings-input:focus {
            outline: none;
            border-color: var(--accent);
          }

          .settings-input:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }

          .settings-input-number {
            width: 80px;
            text-align: right;
          }

          .settings-button {
            padding: 4px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 3px;
            color: var(--text-primary);
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s;
          }

          .settings-button:hover {
            background: var(--bg-hover);
            border-color: var(--text-secondary);
          }

          /* Preset buttons */
          .preset-buttons {
            display: flex;
            gap: 6px;
            margin-top: 8px;
          }

          .preset-btn {
            padding: 3px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 3px;
            color: var(--text-secondary);
            font-size: 10px;
            cursor: pointer;
            transition: all 0.15s;
          }

          .preset-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--text-secondary);
          }

          /* Status indicator */
          .settings-status {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 0;
          }

          .status-indicator {
            width: 6px;
            height: 6px;
            border-radius: 50%;
          }

          .status-indicator.connected {
            background: #22c55e;
            box-shadow: 0 0 6px #22c55e;
          }

          .status-indicator.disconnected {
            background: var(--text-secondary);
          }

          .status-text {
            font-size: 11px;
            color: var(--text-secondary);
          }

          /* Provider list */
          .provider-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
          }

          .provider-option {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.15s;
          }

          .provider-option:hover {
            border-color: var(--text-secondary);
          }

          .provider-option.active {
            border-color: var(--accent);
            background: rgba(59, 130, 246, 0.1);
          }

          .provider-option input {
            margin: 0;
            accent-color: var(--accent);
          }

          .provider-info {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 2px;
          }

          .provider-label {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-primary);
          }

          .provider-description {
            font-size: 10px;
            color: var(--text-secondary);
          }

          .provider-status {
            color: #22c55e;
            font-size: 12px;
          }

          /* API Key inputs */
          .api-key-row {
            margin-bottom: 10px;
          }

          .api-key-row:last-child {
            margin-bottom: 0;
          }

          .api-key-row label {
            display: block;
            margin-bottom: 3px;
            font-size: 11px;
            color: var(--text-secondary);
          }

          .api-key-input {
            display: flex;
            gap: 3px;
          }

          .api-key-input input {
            flex: 1;
            padding: 4px 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 3px;
            color: var(--text-primary);
            font-size: 11px;
            font-family: monospace;
          }

          .api-key-input input:focus {
            outline: none;
            border-color: var(--accent);
          }

          .api-key-input input::placeholder {
            color: var(--text-secondary);
            opacity: 0.5;
          }

          .toggle-visibility {
            padding: 4px 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            color: var(--text-secondary);
            transition: all 0.15s;
          }

          .toggle-visibility:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
          }

          .api-key-link {
            display: inline-block;
            margin-top: 3px;
            font-size: 10px;
            color: var(--accent);
            text-decoration: none;
          }

          .api-key-link:hover {
            text-decoration: underline;
          }

          /* Footer */
          .settings-footer {
            display: flex;
            justify-content: flex-end;
            gap: 6px;
            padding: 6px 12px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            flex-shrink: 0;
          }

          .settings-footer button {
            padding: 4px 16px;
            border-radius: 3px;
            border: 1px solid var(--border-color);
            font-size: 11px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
          }

          .btn-cancel {
            background: var(--bg-tertiary);
            color: var(--text-primary);
          }

          .btn-cancel:hover {
            background: var(--bg-hover);
          }

          .btn-save {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
            min-width: 80px;
          }

          .btn-save:hover {
            opacity: 0.9;
          }
        `}</style>
      </div>
    </div>
  );
}
