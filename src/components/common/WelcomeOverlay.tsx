// WelcomeOverlay - First-time user welcome with folder picker
// Shows on first load to ask for proxy/analysis data storage folder

import { useState, useCallback, useEffect } from 'react';
import { pickProxyFolder, getProxyFolderName, isFileSystemAccessSupported, initFileSystemService } from '../../services/fileSystemService';
import { useSettingsStore } from '../../stores/settingsStore';

interface WelcomeOverlayProps {
  onComplete: () => void;
}

export function WelcomeOverlay({ onComplete }: WelcomeOverlayProps) {
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Restore folder handle from IndexedDB on mount
  useEffect(() => {
    initFileSystemService().then(() => {
      setSelectedFolder(getProxyFolderName());
    });
  }, []);

  // Handle Enter key to continue
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && !isSelecting) {
        handleContinue();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleContinue, isSelecting]);

  const isSupported = isFileSystemAccessSupported();

  const handleSelectFolder = useCallback(async () => {
    if (isSelecting) return;
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
  }, [isSelecting]);

  const handleContinue = useCallback(() => {
    useSettingsStore.getState().setHasCompletedSetup(true);
    onComplete();
  }, [onComplete]);


  return (
    <div className="welcome-overlay-backdrop">
      <div className="welcome-overlay">
        {/* Privacy badge */}
        <div className="welcome-badge">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          <span>Private · Local · Free</span>
        </div>

        {/* Title */}
        <h1 className="welcome-title">
          <span className="welcome-title-master">Master</span>
          <span className="welcome-title-selects">Selects</span>
        </h1>

        <p className="welcome-subtitle">Video editing in your browser</p>

        {/* Folder Selection Card */}
        <div className="welcome-folder-card">
          <div className="welcome-folder-card-header">
            <span className="welcome-folder-card-label">Proxy Storage</span>
            <span className="welcome-folder-card-optional">optional</span>
          </div>

          {!isSupported ? (
            <p className="welcome-note">
              Using browser memory for proxy files.
            </p>
          ) : (
            <button
              className={`welcome-folder-btn ${selectedFolder ? 'has-folder' : ''}`}
              onClick={handleSelectFolder}
              disabled={isSelecting}
            >
              <div className="welcome-folder-btn-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                </svg>
              </div>
              <div className="welcome-folder-btn-text">
                {selectedFolder ? (
                  <>
                    <span className="welcome-folder-name">{selectedFolder}</span>
                    <span className="welcome-folder-change">Click to change</span>
                  </>
                ) : (
                  <>
                    <span className="welcome-folder-name">{isSelecting ? 'Opening...' : 'Choose folder'}</span>
                    <span className="welcome-folder-change">For faster editing</span>
                  </>
                )}
              </div>
              <svg className="welcome-folder-btn-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 18l6-6-6-6"/>
              </svg>
            </button>
          )}

          {error && <p className="welcome-error">{error}</p>}
        </div>

        {/* Enter hint */}
        <button className="welcome-enter" onClick={handleContinue}>
          <span>Start editing</span>
          <kbd>↵</kbd>
        </button>
      </div>
    </div>
  );
}
