// WhatsNewDialog - Shows changelog and known issues on every startup
// Displayed after the welcome overlay (if shown) or immediately on refresh

import { useState, useEffect, useCallback } from 'react';
import { APP_VERSION, CHANGELOG, KNOWN_ISSUES } from '../../version';

interface WhatsNewDialogProps {
  onClose: () => void;
}

export function WhatsNewDialog({ onClose }: WhatsNewDialogProps) {
  const [isClosing, setIsClosing] = useState(false);

  const handleClose = useCallback(() => {
    if (isClosing) return;
    setIsClosing(true);
    setTimeout(() => {
      onClose();
    }, 120);
  }, [onClose, isClosing]);

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

  // Get the current version's changelog entry
  const currentChangelog = CHANGELOG.find(entry => entry.version === APP_VERSION);

  return (
    <div
      className={`whats-new-backdrop ${isClosing ? 'closing' : ''}`}
      onClick={handleBackdropClick}
    >
      <div className="welcome-overlay whats-new-dialog">
        {/* Header */}
        <div className="whats-new-header">
          <h2 className="whats-new-title">What's New</h2>
          <span className="whats-new-version">v{APP_VERSION}</span>
        </div>

        {/* Last update info */}
        {currentChangelog && (
          <div className="whats-new-date">
            Updated {currentChangelog.date}
          </div>
        )}

        {/* Changes section */}
        <div className="whats-new-section">
          <h3 className="whats-new-section-title">Changes</h3>
          <ul className="whats-new-list">
            {currentChangelog?.changes.map((change, i) => (
              <li key={i} className="whats-new-item">
                <span className={`whats-new-tag whats-new-tag-${change.type}`}>
                  {change.type}
                </span>
                <span>{change.description}</span>
              </li>
            ))}
            {!currentChangelog && (
              <li className="whats-new-item">
                <span className="whats-new-tag whats-new-tag-fix">fix</span>
                <span>Bug fixes and improvements</span>
              </li>
            )}
          </ul>
        </div>

        {/* Known issues section */}
        {KNOWN_ISSUES.length > 0 && (
          <div className="whats-new-section whats-new-issues">
            <h3 className="whats-new-section-title">Known Issues</h3>
            <ul className="whats-new-list">
              {KNOWN_ISSUES.map((issue, i) => (
                <li key={i} className="whats-new-item whats-new-issue-item">
                  <span className="whats-new-issue-icon">!</span>
                  <span>{issue}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Close button */}
        <button className="welcome-enter" onClick={handleClose}>
          <span>Got it</span>
          <kbd>Esc</kbd>
        </button>
      </div>
    </div>
  );
}

