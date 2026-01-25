// WhatsNewDialog - Shows changelog grouped by time periods
// Displays changes categorized as "Today", "Last Week", "This Month", "Earlier"

import { useState, useEffect, useCallback, useMemo } from 'react';
import { APP_VERSION, getGroupedChangelog, type ChangeEntry } from '../../version';

interface WhatsNewDialogProps {
  onClose: () => void;
}

// Icon components for change types
function NewIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function FixIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M2 7l3.5 3.5L12 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ImproveIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M7 11V3M4 5l3-3 3 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ChangeItem({ change }: { change: ChangeEntry }) {
  const [expanded, setExpanded] = useState(false);
  const hasDescription = !!change.description;

  return (
    <div
      className={`changelog-item changelog-item-${change.type} ${hasDescription ? 'has-description' : ''} ${expanded ? 'expanded' : ''}`}
      onClick={() => hasDescription && setExpanded(!expanded)}
    >
      <div className="changelog-item-header">
        <span className={`changelog-icon changelog-icon-${change.type}`}>
          {change.type === 'new' && <NewIcon />}
          {change.type === 'fix' && <FixIcon />}
          {change.type === 'improve' && <ImproveIcon />}
        </span>
        <span className="changelog-title">{change.title}</span>
        {hasDescription && (
          <span className="changelog-expand">
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }}>
              <path d="M2 3.5l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </span>
        )}
      </div>
      {expanded && change.description && (
        <div className="changelog-description">{change.description}</div>
      )}
    </div>
  );
}

export function WhatsNewDialog({ onClose }: WhatsNewDialogProps) {
  const [isClosing, setIsClosing] = useState(false);
  const [activeTab, setActiveTab] = useState<'all' | 'new' | 'fix' | 'improve'>('all');

  const groupedChangelog = useMemo(() => getGroupedChangelog(), []);

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

  // Filter changes based on active tab
  const filteredGroups = useMemo(() => {
    if (activeTab === 'all') return groupedChangelog;
    return groupedChangelog
      .map(group => ({
        ...group,
        changes: group.changes.filter(c => c.type === activeTab),
      }))
      .filter(group => group.changes.length > 0);
  }, [groupedChangelog, activeTab]);

  // Count changes by type
  const counts = useMemo(() => {
    const all = groupedChangelog.flatMap(g => g.changes);
    return {
      all: all.length,
      new: all.filter(c => c.type === 'new').length,
      fix: all.filter(c => c.type === 'fix').length,
      improve: all.filter(c => c.type === 'improve').length,
    };
  }, [groupedChangelog]);

  return (
    <div
      className={`whats-new-backdrop ${isClosing ? 'closing' : ''}`}
      onClick={handleBackdropClick}
    >
      <div className="welcome-overlay whats-new-dialog changelog-dialog">
        {/* Header */}
        <div className="changelog-header">
          <div className="changelog-header-left">
            <h2 className="changelog-title">Changelog</h2>
            <span className="changelog-version">v{APP_VERSION}</span>
          </div>
          <button className="changelog-close" onClick={handleClose}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Filter tabs */}
        <div className="changelog-tabs">
          <button
            className={`changelog-tab ${activeTab === 'all' ? 'active' : ''}`}
            onClick={() => setActiveTab('all')}
          >
            All <span className="changelog-tab-count">{counts.all}</span>
          </button>
          <button
            className={`changelog-tab changelog-tab-new ${activeTab === 'new' ? 'active' : ''}`}
            onClick={() => setActiveTab('new')}
          >
            New <span className="changelog-tab-count">{counts.new}</span>
          </button>
          <button
            className={`changelog-tab changelog-tab-fix ${activeTab === 'fix' ? 'active' : ''}`}
            onClick={() => setActiveTab('fix')}
          >
            Fixes <span className="changelog-tab-count">{counts.fix}</span>
          </button>
          <button
            className={`changelog-tab changelog-tab-improve ${activeTab === 'improve' ? 'active' : ''}`}
            onClick={() => setActiveTab('improve')}
          >
            Improved <span className="changelog-tab-count">{counts.improve}</span>
          </button>
        </div>

        {/* Scrollable content */}
        <div className="changelog-content">
          {filteredGroups.map((group, groupIndex) => (
            <div key={group.label} className="changelog-group">
              <div className="changelog-group-header">
                <span className="changelog-group-label">{group.label}</span>
                <span className="changelog-group-date">{group.dateRange}</span>
                <div className="changelog-group-line" />
              </div>
              <div className="changelog-group-items">
                {group.changes.map((change, i) => (
                  <ChangeItem key={`${groupIndex}-${i}`} change={change} />
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="changelog-footer">
          <button className="welcome-enter" onClick={handleClose}>
            <span>Got it</span>
            <kbd>Esc</kbd>
          </button>
        </div>
      </div>
    </div>
  );
}
