// WelcomeOverlay - First-time user welcome with folder picker
// Shows on first load to ask for project storage folder

import { useState, useCallback, useEffect, useMemo } from 'react';
import { Logger } from '../../services/logger';

const log = Logger.create('WelcomeOverlay');
import { isFileSystemAccessSupported } from '../../services/fileSystemService';
import { projectFileService } from '../../services/projectFileService';
import { openExistingProject } from '../../services/projectSync';

// Detect browser name and if it's Chromium-based
function detectBrowser(): { name: string; isChromium: boolean } {
  const ua = navigator.userAgent;

  // Check specific browsers (order matters - more specific first)
  if (/Edg\//.test(ua)) {
    return { name: 'Microsoft Edge', isChromium: true };
  }
  if (/OPR\//.test(ua) || /Opera/.test(ua)) {
    return { name: 'Opera', isChromium: true };
  }
  if (/Chrome\//.test(ua) && !/Chromium\//.test(ua)) {
    return { name: 'Google Chrome', isChromium: true };
  }
  if (/Chromium\//.test(ua)) {
    return { name: 'Chromium', isChromium: true };
  }
  if (/Safari\//.test(ua) && !/Chrome\//.test(ua)) {
    return { name: 'Safari', isChromium: false };
  }
  if (/Firefox\//.test(ua)) {
    return { name: 'Firefox', isChromium: false };
  }

  return { name: 'Unknown Browser', isChromium: false };
}

interface WelcomeOverlayProps {
  onComplete: () => void;
  noFadeOnClose?: boolean; // Don't fade blur when another dialog follows
}

// Typewriter sequence with typo correction
const TYPEWRITER_SEQUENCE = [
  { action: 'type', text: 'Local', class: 'local' },
  { action: 'pause', duration: 400 },
  { action: 'type', text: '·', class: 'dot' },
  { action: 'type', text: 'Private', class: 'private' },
  { action: 'pause', duration: 350 },
  { action: 'type', text: '·', class: 'dot' },
  { action: 'type', text: 'Tre', class: 'free' },  // Typo!
  { action: 'pause', duration: 280 },
  { action: 'delete', count: 3 },                   // Delete "Tre"
  { action: 'pause', duration: 200 },
  { action: 'type', text: 'Free', class: 'free' }, // Correct it
  { action: 'pause', duration: 400 },
  { action: 'hideCursor' },
];

export function WelcomeOverlay({ onComplete, noFadeOnClose = false }: WelcomeOverlayProps) {
  const [isSelecting, setIsSelecting] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Typewriter state
  const [typewriterParts, setTypewriterParts] = useState<Array<{ text: string; class: string }>>([]);
  const [cursorVisible, setCursorVisible] = useState(false);
  const [cursorBlink, setCursorBlink] = useState(true);

  const isSupported = isFileSystemAccessSupported();
  const browser = useMemo(() => detectBrowser(), []);

  // Typewriter effect
  useEffect(() => {
    let step = 0;
    let charIndex = 0;
    let deleteCount = 0;
    let cancelled = false;
    const timeouts: ReturnType<typeof setTimeout>[] = [];

    const scheduleNext = (fn: () => void, delay: number) => {
      const t = setTimeout(fn, delay);
      timeouts.push(t);
      return t;
    };

    const randomDelay = (base: number, variance: number) =>
      base + Math.random() * variance - variance / 2;

    const processStep = () => {
      if (cancelled || step >= TYPEWRITER_SEQUENCE.length) return;

      const action = TYPEWRITER_SEQUENCE[step];

      if (action.action === 'type' && action.text) {
        if (charIndex < action.text.length) {
          const char = action.text[charIndex];
          setTypewriterParts(prev => {
            const newParts = [...prev];
            const lastPart = newParts[newParts.length - 1];
            if (lastPart && lastPart.class === action.class) {
              newParts[newParts.length - 1] = { ...lastPart, text: lastPart.text + char };
            } else {
              newParts.push({ text: char, class: action.class! });
            }
            return newParts;
          });
          charIndex++;
          scheduleNext(processStep, randomDelay(75, 45));
        } else {
          step++;
          charIndex = 0;
          scheduleNext(processStep, randomDelay(30, 20));
        }
      } else if (action.action === 'pause') {
        step++;
        scheduleNext(processStep, action.duration || 300);
      } else if (action.action === 'delete') {
        const toDelete = action.count || 1;
        if (deleteCount < toDelete) {
          setTypewriterParts(prev => {
            const newParts = [...prev];
            const lastPart = newParts[newParts.length - 1];
            if (lastPart && lastPart.text.length > 0) {
              newParts[newParts.length - 1] = { ...lastPart, text: lastPart.text.slice(0, -1) };
              if (newParts[newParts.length - 1].text.length === 0) {
                newParts.pop();
              }
            }
            return newParts;
          });
          deleteCount++;
          scheduleNext(processStep, randomDelay(50, 25));
        } else {
          step++;
          deleteCount = 0;
          scheduleNext(processStep, randomDelay(30, 20));
        }
      } else if (action.action === 'hideCursor') {
        setCursorBlink(false);
        setCursorVisible(false);
      }
    };

    // Reset state for fresh start (handles Strict Mode remount)
    setTypewriterParts([]);
    setCursorVisible(false);
    setCursorBlink(true);

    // Start after overlay fade-in animation (1.0s delay + 0.4s animation + buffer)
    scheduleNext(() => {
      setCursorVisible(true);
      processStep();
    }, 1600);

    return () => {
      cancelled = true;
      timeouts.forEach(t => clearTimeout(t));
    };
  }, []);

  // Cursor blink
  useEffect(() => {
    if (!cursorBlink) return;
    const interval = setInterval(() => {
      setCursorVisible(prev => !prev);
    }, 530);
    return () => clearInterval(interval);
  }, [cursorBlink]);

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
          }, noFadeOnClose ? 80 : 120);
        } else {
          setError('Failed to create project. Please try again.');
        }
      }
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User cancelled - not an error
        return;
      }
      log.error('Failed to select folder', e);
      setError('Failed to select folder. Please try again.');
    } finally {
      setIsSelecting(false);
    }
  }, [isSelecting, isClosing, onComplete, noFadeOnClose]);

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
        }, noFadeOnClose ? 80 : 120);
      } else {
        // User cancelled or folder has no project.json
        setError('No valid project found. Select a folder containing project.json');
      }
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User cancelled - not an error
        return;
      }
      log.error('Failed to open project', e);
      setError('Failed to open project. Please try again.');
    } finally {
      setIsSelecting(false);
    }
  }, [isSelecting, isClosing, onComplete, noFadeOnClose]);

  const handleContinue = useCallback(() => {
    if (isClosing) return;
    setIsClosing(true);
    // Wait for exit animation to complete
    setTimeout(() => {
      onComplete();
    }, noFadeOnClose ? 80 : 120);
  }, [onComplete, isClosing, noFadeOnClose]);

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


  const backdropClass = `welcome-overlay-backdrop ${isClosing ? 'closing' : ''} ${isClosing && noFadeOnClose ? 'no-fade' : ''}`;

  return (
    <div className={backdropClass}>
      <div className="welcome-overlay">
        {/* Privacy tagline - Typewriter effect */}
        <div className="welcome-tagline">
          {typewriterParts.map((part, i) => (
            <span key={i} className={`welcome-tag-${part.class}`}>{part.text}</span>
          ))}
          <span className={`welcome-cursor ${cursorVisible ? 'visible' : ''}`}>|</span>
        </div>

        {/* Title */}
        <h1 className="welcome-title">
          <span className="welcome-title-master">Master</span>
          <span className="welcome-title-selects">Selects</span>
        </h1>

        <p className="welcome-subtitle">Video editing in your browser</p>

        {/* Browser Warning for non-Chromium browsers */}
        {!browser.isChromium && (
          <div className="welcome-browser-warning">
            <svg className="welcome-browser-warning-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span className="welcome-browser-warning-label">Unsupported Browser</span>
            <span className="welcome-browser-warning-name">{browser.name}</span>
            <span className="welcome-browser-warning-desc">MasterSelects requires WebGPU which is currently only fully supported in Chrome.</span>
            <a className="welcome-browser-warning-btn" href="https://www.google.com/chrome/" target="_blank" rel="noopener noreferrer">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/>
                <circle cx="12" cy="12" r="4"/>
                <line x1="21.17" y1="8" x2="12" y2="8"/>
                <line x1="3.95" y1="6.06" x2="8.54" y2="14"/>
                <line x1="10.88" y1="21.94" x2="15.46" y2="14"/>
              </svg>
              Download Chrome
            </a>
          </div>
        )}

        {/* Folder Selection Card - hide if browser not supported */}
        {(isSupported || browser.isChromium) && (
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
        )}

        {/* Enter hint */}
        <button className="welcome-enter" onClick={handleContinue}>
          <span>Start editing</span>
          <kbd>↵</kbd>
        </button>
      </div>
    </div>
  );
}
