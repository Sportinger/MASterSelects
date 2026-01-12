// WelcomeOverlay - First-time user welcome with folder picker
// Shows on first load to ask for project storage folder

import { useState, useCallback, useEffect } from 'react';
import { isFileSystemAccessSupported } from '../../services/fileSystemService';
import { projectFileService } from '../../services/projectFileService';
import { openExistingProject } from '../../services/projectSync';

interface WelcomeOverlayProps {
  onComplete: () => void;
}

export function WelcomeOverlay({ onComplete }: WelcomeOverlayProps) {
  const [isSelecting, setIsSelecting] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isSupported = isFileSystemAccessSupported();

  const handleSelectFolder = useCallback(async () => {
    if (isSelecting || isClosing) return;
    setIsSelecting(true);
    setError(null);

    try {
      // Let user pick where to store projects
      const handle = await (window as any).showDirectoryPicker({
        mode: 'readwrite',
        startIn: 'documents',
      });

      if (handle) {
        setSelectedFolder(handle.name);

        // Create "Untitled" project in the selected folder
        const success = await projectFileService.createProjectInFolder(handle, 'Untitled');

        if (success) {
          // Auto-close after project creation
          setIsClosing(true);
          setTimeout(() => {
            onComplete();
          }, 200);
        } else {
          setError('Failed to create project. Please try again.');
        }
      }
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User cancelled - not an error
        return;
      }
      console.error('[WelcomeOverlay] Failed to select folder:', e);
      setError('Failed to select folder. Please try again.');
    } finally {
      setIsSelecting(false);
    }
  }, [isSelecting, isClosing, onComplete]);

  // Open existing project from local folder
  const handleOpenExisting = useCallback(async () => {
    if (isSelecting || isClosing) return;
    setIsSelecting(true);
    setError(null);

    try {
      // Let user pick existing project folder and load it into stores
      const success = await openExistingProject();

      if (success) {
        const projectData = projectFileService.getProjectData();
        setSelectedFolder(projectData?.name || 'Project');

        // Auto-close after project opens
        setIsClosing(true);
        setTimeout(() => {
          onComplete();
        }, 200);
      } else {
        // User cancelled or folder has no project.json
        setError('No valid project found. Select a folder containing project.json');
      }
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User cancelled - not an error
        return;
      }
      console.error('[WelcomeOverlay] Failed to open project:', e);
      setError('Failed to open project. Please try again.');
    } finally {
      setIsSelecting(false);
    }
  }, [isSelecting, isClosing, onComplete]);

  const handleContinue = useCallback(() => {
    if (isClosing) return;
    setIsClosing(true);
    // Wait for exit animation to complete
    setTimeout(() => {
      onComplete();
    }, 200);
  }, [onComplete, isClosing]);

  // Check if there's already an open project
  useEffect(() => {
    if (projectFileService.isProjectOpen()) {
      const projectData = projectFileService.getProjectData();
      if (projectData) {
        setSelectedFolder(projectData.name);
      }
    }
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


  return (
    <div className={`welcome-overlay-backdrop ${isClosing ? 'closing' : ''}`}>
      <div className="welcome-overlay">
        {/* Privacy tagline */}
        <div className="welcome-tagline">
          <span className="welcome-tag-local">Local</span>
          <span className="welcome-tag-dot">·</span>
          <span className="welcome-tag-private">Private</span>
          <span className="welcome-tag-dot">·</span>
          <span className="welcome-tag-free">Free</span>
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
            <span className="welcome-folder-card-label">Project</span>
            <span className="welcome-folder-card-optional">required</span>
          </div>

          {!isSupported ? (
            <p className="welcome-note">
              Your browser does not support local file storage.
              Please use Chrome, Edge, or another Chromium-based browser.
            </p>
          ) : (
            <div className="welcome-folder-buttons">
              {/* New Project Button */}
              <button
                className={`welcome-folder-btn ${selectedFolder ? 'has-folder' : ''}`}
                onClick={handleSelectFolder}
                disabled={isSelecting}
              >
                <div className="welcome-folder-btn-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 5v14M5 12h14"/>
                  </svg>
                </div>
                <div className="welcome-folder-btn-text">
                  {selectedFolder ? (
                    <>
                      <span className="welcome-folder-name">{selectedFolder}</span>
                      <span className="welcome-folder-change">Project created</span>
                    </>
                  ) : (
                    <>
                      <span className="welcome-folder-name">{isSelecting ? 'Creating...' : 'New Project'}</span>
                      <span className="welcome-folder-change">Create in a new folder</span>
                    </>
                  )}
                </div>
                <svg className="welcome-folder-btn-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9 18l6-6-6-6"/>
                </svg>
              </button>

              {/* Open Existing Project Button */}
              <button
                className="welcome-folder-btn welcome-folder-btn-secondary"
                onClick={handleOpenExisting}
                disabled={isSelecting || !!selectedFolder}
              >
                <div className="welcome-folder-btn-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                  </svg>
                </div>
                <div className="welcome-folder-btn-text">
                  <span className="welcome-folder-name">Open Existing</span>
                  <span className="welcome-folder-change">Resume a saved project</span>
                </div>
                <svg className="welcome-folder-btn-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M9 18l6-6-6-6"/>
                </svg>
              </button>
            </div>
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
