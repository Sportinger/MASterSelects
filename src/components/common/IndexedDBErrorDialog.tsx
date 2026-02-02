// IndexedDBErrorDialog - Shows when browser storage is corrupted
// Provides instructions for clearing site data to fix the issue

import { useState, useEffect, useCallback } from 'react';

interface IndexedDBErrorDialogProps {
  onClose: () => void;
}

export function IndexedDBErrorDialog({ onClose }: IndexedDBErrorDialogProps) {
  const [isClosing, setIsClosing] = useState(false);

  const handleClose = useCallback(() => {
    if (isClosing) return;
    setIsClosing(true);
    setTimeout(() => {
      onClose();
    }, 120);
  }, [onClose, isClosing]);

  const handleRefresh = useCallback(() => {
    window.location.reload();
  }, []);

  // Handle Escape key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleClose]);

  // Handle backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      handleClose();
    }
  };

  return (
    <div
      className={`whats-new-backdrop ${isClosing ? 'closing' : ''}`}
      onClick={handleBackdropClick}
    >
      <div className="welcome-overlay indexeddb-error-dialog">
        {/* Warning Icon */}
        <div className="indexeddb-error-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              stroke="#f59e0b"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        {/* Title */}
        <h2 className="indexeddb-error-title">Browser Storage Error</h2>

        {/* Description */}
        <div className="indexeddb-error-content">
          <p className="indexeddb-error-description">
            Chrome's IndexedDB storage has become corrupted. This prevents the app from
            saving your project folder location and settings between sessions.
          </p>

          <div className="indexeddb-error-note">
            <strong>Your project files are safe</strong> - they're stored in your project folder,
            not in the browser. You just need to re-select the folder after fixing this.
          </div>

          <div className="indexeddb-error-steps">
            <h3>To fix this issue:</h3>
            <ol>
              <li>Press <kbd>F12</kbd> to open DevTools</li>
              <li>Go to the <strong>Application</strong> tab</li>
              <li>In the left sidebar, click <strong>Storage</strong></li>
              <li>Click the <strong>Clear site data</strong> button</li>
              <li>Refresh the page</li>
            </ol>
          </div>

          <p className="indexeddb-error-alternative">
            Or paste this in your browser URL bar:<br />
            <code>chrome://settings/content/siteDetails?site=http://localhost:5173</code><br />
            Then click "Clear data"
          </p>
        </div>

        {/* Buttons */}
        <div className="indexeddb-error-buttons">
          <button className="indexeddb-error-btn secondary" onClick={handleClose}>
            Dismiss
          </button>
          <button className="indexeddb-error-btn primary" onClick={handleRefresh}>
            Refresh Page
          </button>
        </div>
      </div>
    </div>
  );
}
