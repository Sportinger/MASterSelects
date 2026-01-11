// WelcomeOverlay - First-time user welcome with folder picker
// Shows on first load to ask for proxy/analysis data storage folder

import { useState, useCallback } from 'react';
import { pickProxyFolder, getProxyFolderName, isFileSystemAccessSupported } from '../../services/fileSystemService';
import { useSettingsStore } from '../../stores/settingsStore';

interface WelcomeOverlayProps {
  onComplete: () => void;
}

export function WelcomeOverlay({ onComplete }: WelcomeOverlayProps) {
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(getProxyFolderName());
  const [error, setError] = useState<string | null>(null);

  const isSupported = isFileSystemAccessSupported();

  const handleSelectFolder = useCallback(async () => {
    setIsSelecting(true);
    setError(null);

    try {
      const handle = await pickProxyFolder();
      if (handle) {
        setSelectedFolder(handle.name);
      }
    } catch (e) {
      console.error('[WelcomeOverlay] Failed to select folder:', e);
      setError('Failed to select folder. Please try again.');
    } finally {
      setIsSelecting(false);
    }
  }, []);

  const handleContinue = useCallback(() => {
    // Mark first-run as complete
    useSettingsStore.getState().setHasCompletedSetup(true);
    onComplete();
  }, [onComplete]);

  const handleSkip = useCallback(() => {
    // Skip without selecting folder
    useSettingsStore.getState().setHasCompletedSetup(true);
    onComplete();
  }, [onComplete]);

  return (
    <div className="welcome-overlay-backdrop">
      <div className="welcome-overlay">
        <div className="welcome-header">
          <h1>Welcome to MASterSelects</h1>
          <p>Professional video editing powered by WebGPU</p>
        </div>

        <div className="welcome-content">
          <div className="welcome-section">
            <h2>Choose Storage Location</h2>
            <p>
              Select a folder to store proxy files and analysis data.
              This allows faster video preview and editing.
            </p>

            {!isSupported ? (
              <div className="welcome-warning">
                <p>
                  Your browser doesn't support the File System Access API.
                  Proxy files will be stored in memory (temporary).
                </p>
              </div>
            ) : (
              <>
                <button
                  className="welcome-button primary"
                  onClick={handleSelectFolder}
                  disabled={isSelecting}
                >
                  {isSelecting ? 'Selecting...' : 'Select Folder'}
                </button>

                {selectedFolder && (
                  <div className="welcome-folder-selected">
                    Selected: <strong>{selectedFolder}</strong>
                  </div>
                )}

                {error && (
                  <div className="welcome-error">{error}</div>
                )}
              </>
            )}
          </div>

          <div className="welcome-features">
            <h3>Getting Started</h3>
            <ul>
              <li>Drag media files into the Media panel to import</li>
              <li>Drag clips to the timeline to start editing</li>
              <li>Use keyboard shortcuts: Space (play), J/K/L (shuttle)</li>
              <li>Press ? for full keyboard shortcuts list</li>
            </ul>
          </div>
        </div>

        <div className="welcome-footer">
          <button
            className="welcome-button secondary"
            onClick={handleSkip}
          >
            Skip for now
          </button>
          <button
            className="welcome-button primary"
            onClick={handleContinue}
            disabled={isSupported && !selectedFolder}
          >
            {selectedFolder ? 'Continue' : 'Continue without folder'}
          </button>
        </div>
      </div>
    </div>
  );
}
