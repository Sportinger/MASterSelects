// Settings Dialog for API keys configuration

import { useState, useCallback } from 'react';
import { useSettingsStore, type TranscriptionProvider } from '../../stores/settingsStore';

interface SettingsDialogProps {
  onClose: () => void;
}

export function SettingsDialog({ onClose }: SettingsDialogProps) {
  const {
    apiKeys,
    transcriptionProvider,
    setApiKey,
    setTranscriptionProvider,
  } = useSettingsStore();

  // Local state for editing (to avoid saving on every keystroke)
  const [localKeys, setLocalKeys] = useState(apiKeys);
  const [showKeys, setShowKeys] = useState({
    openai: false,
    assemblyai: false,
    deepgram: false,
    kling: false,
  });

  const handleSave = useCallback(() => {
    // Save all keys
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

  const providers: { id: TranscriptionProvider; label: string; description: string }[] = [
    {
      id: 'local',
      label: 'Local (Whisper)',
      description: 'Runs in browser, no API key needed. Slower, less accurate.',
    },
    {
      id: 'openai',
      label: 'OpenAI Whisper API',
      description: 'High accuracy, $0.006/minute. Requires API key.',
    },
    {
      id: 'assemblyai',
      label: 'AssemblyAI',
      description: 'Excellent accuracy, speaker diarization. $0.015/minute.',
    },
    {
      id: 'deepgram',
      label: 'Deepgram',
      description: 'Fast, good accuracy. $0.0125/minute.',
    },
  ];

  return (
    <div className="settings-dialog-overlay" onClick={onClose}>
      <div className="settings-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="settings-close" onClick={onClose}>
            ×
          </button>
        </div>

        <div className="settings-content">
          {/* Transcription Provider Selection */}
          <div className="settings-section">
            <h3>Transcription Provider</h3>
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
          </div>

          {/* API Keys */}
          <div className="settings-section">
            <h3>API Keys</h3>
            <p className="settings-hint">
              Keys are stored locally in your browser and never sent to our servers.
            </p>

            {/* OpenAI */}
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
                  title={showKeys.openai ? 'Hide' : 'Show'}
                >
                  {showKeys.openai ? '🙈' : '👁'}
                </button>
              </div>
              <a
                className="api-key-link"
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
              >
                Get API Key →
              </a>
            </div>

            {/* AssemblyAI */}
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
                  title={showKeys.assemblyai ? 'Hide' : 'Show'}
                >
                  {showKeys.assemblyai ? '🙈' : '👁'}
                </button>
              </div>
              <a
                className="api-key-link"
                href="https://www.assemblyai.com/dashboard/signup"
                target="_blank"
                rel="noopener noreferrer"
              >
                Get API Key →
              </a>
            </div>

            {/* Deepgram */}
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
                  title={showKeys.deepgram ? 'Hide' : 'Show'}
                >
                  {showKeys.deepgram ? '🙈' : '👁'}
                </button>
              </div>
              <a
                className="api-key-link"
                href="https://console.deepgram.com/signup"
                target="_blank"
                rel="noopener noreferrer"
              >
                Get API Key →
              </a>
            </div>
          </div>

          {/* AI Video Generation */}
          <div className="settings-section">
            <h3>AI Video Generation</h3>

            {/* Kling */}
            <div className="api-key-row">
              <label>Kling AI API Key</label>
              <div className="api-key-input">
                <input
                  type={showKeys.kling ? 'text' : 'password'}
                  value={localKeys.kling}
                  onChange={(e) => handleKeyChange('kling', e.target.value)}
                  placeholder="Enter API key..."
                />
                <button
                  className="toggle-visibility"
                  onClick={() => toggleShowKey('kling')}
                  title={showKeys.kling ? 'Hide' : 'Show'}
                >
                  {showKeys.kling ? '🙈' : '👁'}
                </button>
              </div>
              <a
                className="api-key-link"
                href="https://app.klingai.com/global/dev/document-api"
                target="_blank"
                rel="noopener noreferrer"
              >
                Get API Key →
              </a>
            </div>
          </div>
        </div>

        <div className="settings-actions">
          <button className="btn-cancel" onClick={onClose}>
            Cancel
          </button>
          <button className="btn-save" onClick={handleSave}>
            Save
          </button>
        </div>

        <style>{`
          .settings-dialog-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
          }

          .settings-dialog {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            width: 500px;
            max-width: 90vw;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
          }

          .settings-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
          }

          .settings-header h2 {
            margin: 0;
            font-size: 18px;
            color: var(--text-primary);
          }

          .settings-close {
            background: none;
            border: none;
            font-size: 24px;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0;
            line-height: 1;
          }

          .settings-close:hover {
            color: var(--text-primary);
          }

          .settings-content {
            padding: 20px;
            overflow-y: auto;
            flex: 1;
          }

          .settings-section {
            margin-bottom: 24px;
          }

          .settings-section:last-child {
            margin-bottom: 0;
          }

          .settings-section h3 {
            margin: 0 0 12px 0;
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
          }

          .settings-hint {
            margin: 0 0 16px 0;
            font-size: 12px;
            color: var(--text-secondary);
          }

          .provider-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
          }

          .provider-option {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
          }

          .provider-option:hover {
            border-color: var(--text-secondary);
          }

          .provider-option.active {
            border-color: var(--accent);
            background: rgba(var(--accent-rgb, 59, 130, 246), 0.1);
          }

          .provider-option input {
            margin: 0;
          }

          .provider-info {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 2px;
          }

          .provider-label {
            font-size: 14px;
            font-weight: 500;
            color: var(--text-primary);
          }

          .provider-description {
            font-size: 12px;
            color: var(--text-secondary);
          }

          .provider-status {
            color: var(--success, #22c55e);
            font-size: 14px;
          }

          .api-key-row {
            margin-bottom: 16px;
          }

          .api-key-row:last-child {
            margin-bottom: 0;
          }

          .api-key-row label {
            display: block;
            margin-bottom: 6px;
            font-size: 13px;
            color: var(--text-secondary);
          }

          .api-key-input {
            display: flex;
            gap: 8px;
          }

          .api-key-input input {
            flex: 1;
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 14px;
            font-family: monospace;
          }

          .api-key-input input:focus {
            outline: none;
            border-color: var(--accent);
          }

          .api-key-input input::placeholder {
            color: var(--text-secondary);
            opacity: 0.6;
          }

          .toggle-visibility {
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
          }

          .toggle-visibility:hover {
            background: var(--bg-hover);
          }

          .api-key-link {
            display: inline-block;
            margin-top: 6px;
            font-size: 12px;
            color: var(--accent);
            text-decoration: none;
          }

          .api-key-link:hover {
            text-decoration: underline;
          }

          .settings-actions {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            padding: 16px 20px;
            border-top: 1px solid var(--border-color);
          }

          .settings-actions button {
            padding: 10px 20px;
            border-radius: 4px;
            border: none;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
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
            color: white;
            font-weight: 600;
          }

          .btn-save:hover {
            opacity: 0.9;
          }
        `}</style>
      </div>
    </div>
  );
}
