// Mobile Options Menu - Swipe in from right

import { useCallback, useState, useEffect } from 'react';
import { undo, redo } from '../../stores/historyStore';
import { saveCurrentProject, openExistingProject, createNewProject } from '../../services/projectSync';
import { projectFileService } from '../../services/projectFileService';
import { useSettingsStore } from '../../stores/settingsStore';

interface MobileOptionsMenuProps {
  isOpen: boolean;
  onClose: () => void;
}

export function MobileOptionsMenu({ isOpen, onClose }: MobileOptionsMenuProps) {
  const setForceDesktopMode = useSettingsStore((s) => s.setForceDesktopMode);

  // Project name state
  const [projectName, setProjectName] = useState('Untitled');

  // Sync project name from service when menu opens
  useEffect(() => {
    const data = projectFileService.getProjectData();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setProjectName(data?.name || 'Untitled');
  }, [isOpen]);

  // File actions
  const handleNew = useCallback(async () => {
    const name = prompt('Project name:', 'New Project');
    if (name) {
      await createNewProject(name);
      onClose();
    }
  }, [onClose]);

  const handleOpen = useCallback(async () => {
    await openExistingProject();
    onClose();
  }, [onClose]);

  const handleSave = useCallback(async () => {
    await saveCurrentProject();
    onClose();
  }, [onClose]);

  const handleExport = useCallback(() => {
    // TODO: Open export dialog
    alert('Export coming soon...');
    onClose();
  }, [onClose]);

  const handleSwitchToDesktop = useCallback(() => {
    setForceDesktopMode(true);
    // Force reload to apply the change
    window.location.reload();
  }, [setForceDesktopMode]);

  return (
    <div
      className={`mobile-options-menu ${isOpen ? 'open' : ''}`}
      onClick={onClose}
    >
      <div className="mobile-options-content" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="mobile-panel-header">
          <h3>Options</h3>
          <button className="mobile-panel-close" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* Project info */}
        <div className="mobile-options-project">
          <span className="project-label">Project:</span>
          <span className="project-name">{projectName}</span>
        </div>

        {/* Menu items */}
        <div className="mobile-options-list">
          {/* Undo/Redo */}
          <div className="mobile-options-section">
            <div className="section-title">Edit</div>
            <button
              className="mobile-option-btn"
              onClick={undo}
            >
              <span className="option-icon">↩️</span>
              <span>Undo</span>
              <span className="option-hint">2-finger swipe ←</span>
            </button>
            <button
              className="mobile-option-btn"
              onClick={redo}
            >
              <span className="option-icon">↪️</span>
              <span>Redo</span>
              <span className="option-hint">2-finger swipe →</span>
            </button>
          </div>

          {/* File */}
          <div className="mobile-options-section">
            <div className="section-title">File</div>
            <button className="mobile-option-btn" onClick={handleNew}>
              <span className="option-icon">📄</span>
              <span>New Project</span>
            </button>
            <button className="mobile-option-btn" onClick={handleOpen}>
              <span className="option-icon">📂</span>
              <span>Open Project</span>
            </button>
            <button className="mobile-option-btn" onClick={handleSave}>
              <span className="option-icon">💾</span>
              <span>Save Project</span>
            </button>
          </div>

          {/* Export */}
          <div className="mobile-options-section">
            <div className="section-title">Export</div>
            <button className="mobile-option-btn" onClick={handleExport}>
              <span className="option-icon">🎬</span>
              <span>Export Video</span>
            </button>
          </div>

          {/* Settings */}
          <div className="mobile-options-section">
            <div className="section-title">View</div>
            <button className="mobile-option-btn" onClick={handleSwitchToDesktop}>
              <span className="option-icon">🖥️</span>
              <span>Desktop Mode</span>
              <span className="option-hint">Full UI</span>
            </button>
            <button className="mobile-option-btn" disabled>
              <span className="option-icon">⚙️</span>
              <span>Preferences</span>
              <span className="option-hint">Coming soon</span>
            </button>
          </div>
        </div>

        {/* Version */}
        <div className="mobile-options-footer">
          <span>MASterSelects Mobile</span>
        </div>
      </div>
    </div>
  );
}
