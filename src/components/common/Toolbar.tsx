// Toolbar component - After Effects style menu bar

import { useState, useEffect, useCallback, useRef } from 'react';
import { useEngine } from '../../hooks/useEngine';
import { useMixerStore } from '../../stores/mixerStore';
import { useDockStore } from '../../stores/dockStore';
import { PANEL_CONFIGS, type PanelType } from '../../types/dock';
import { useSettingsStore, type PreviewQuality } from '../../stores/settingsStore';
import { useMIDI } from '../../hooks/useMIDI';
import { SettingsDialog } from './SettingsDialog';
import { projectFileService } from '../../services/projectFileService';
import {
  createNewProject,
  openExistingProject,
  saveCurrentProject,
  loadProjectToStores,
  setupAutoSync,
} from '../../services/projectSync';

type MenuId = 'file' | 'edit' | 'view' | 'output' | 'window' | null;

export function Toolbar() {
  const { isEngineReady, createOutputWindow } = useEngine();
  const { setPlaying, outputWindows } = useMixerStore();

  // Auto-start playback when engine is ready
  useEffect(() => {
    if (isEngineReady) {
      setPlaying(true);
    }
  }, [isEngineReady, setPlaying]);
  const { resetLayout, isPanelTypeVisible, togglePanelType, saveLayoutAsDefault } = useDockStore();
  const { isSupported: midiSupported, isEnabled: midiEnabled, enableMIDI, disableMIDI, devices } = useMIDI();
  const { isSettingsOpen, openSettings, closeSettings, previewQuality, setPreviewQuality } = useSettingsStore();

  const [openMenu, setOpenMenu] = useState<MenuId>(null);
  const [isEditingName, setIsEditingName] = useState(false);
  const [projectName, setProjectName] = useState('Untitled Project');
  const [editName, setEditName] = useState(projectName);
  const [isProjectOpen, setIsProjectOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [needsPermission, setNeedsPermission] = useState(false);
  const [pendingProjectName, setPendingProjectName] = useState<string | null>(null);
  const menuBarRef = useRef<HTMLDivElement>(null);

  // Update project name from service
  useEffect(() => {
    const data = projectFileService.getProjectData();
    if (data) {
      setProjectName(data.name);
      setIsProjectOpen(true);
    } else {
      setProjectName('No Project Open');
      setIsProjectOpen(false);
    }
  }, []);

  // Try to restore last project on mount
  useEffect(() => {
    const restoreProject = async () => {
      setIsLoading(true);
      const restored = await projectFileService.restoreLastProject();
      if (restored) {
        // Load project data into stores
        await loadProjectToStores();
        const data = projectFileService.getProjectData();
        if (data) {
          setProjectName(data.name);
          setIsProjectOpen(true);
        }
      } else if (projectFileService.needsPermission()) {
        // Permission needed - show button instead of auto-popup
        setNeedsPermission(true);
        setPendingProjectName(projectFileService.getPendingProjectName());
      }
      setIsLoading(false);

      // Setup auto-sync after initialization
      setupAutoSync();
    };
    restoreProject();
  }, []);

  // Close menu when clicking outside
  useEffect(() => {
    if (!openMenu) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (menuBarRef.current && !menuBarRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [openMenu]);

  // Keyboard shortcut handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      // Ctrl+S: Save
      if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        handleSave();
      }
      // Ctrl+N: New
      if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        handleNew();
      }
      // Ctrl+O: Open
      if (e.ctrlKey && e.key === 'o') {
        e.preventDefault();
        handleOpen();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleSave = useCallback(async () => {
    if (!projectFileService.isProjectOpen()) {
      // No project open, prompt to create one
      const name = prompt('Enter project name:', 'New Project');
      if (!name) return;
      setIsLoading(true);
      const success = await createNewProject(name);
      if (success) {
        setProjectName(name);
        setIsProjectOpen(true);
      }
      setIsLoading(false);
    } else {
      // Save current project with store synchronization
      await saveCurrentProject();
    }
    setOpenMenu(null);
  }, []);

  const handleOpen = useCallback(async () => {
    if (projectFileService.hasUnsavedChanges()) {
      if (!confirm('You have unsaved changes. Open a different project?')) {
        return;
      }
    }
    setIsLoading(true);
    const success = await openExistingProject();
    if (success) {
      const data = projectFileService.getProjectData();
      if (data) {
        setProjectName(data.name);
        setIsProjectOpen(true);
      }
    }
    setIsLoading(false);
    setOpenMenu(null);
  }, []);

  const handleNameSubmit = useCallback(() => {
    if (editName.trim()) {
      const newName = editName.trim();
      setProjectName(newName);
      // Update project data in service
      const data = projectFileService.getProjectData();
      if (data) {
        projectFileService.updateProjectData({ name: newName });
      }
    }
    setIsEditingName(false);
  }, [editName]);

  const handleNew = useCallback(async () => {
    if (projectFileService.hasUnsavedChanges()) {
      if (!confirm('You have unsaved changes. Create a new project?')) {
        return;
      }
    }
    const name = prompt('Enter project name:', 'New Project');
    if (!name) return;

    setIsLoading(true);
    const success = await createNewProject(name);
    if (success) {
      setProjectName(name);
      setIsProjectOpen(true);
      setNeedsPermission(false);
    }
    setIsLoading(false);
    setOpenMenu(null);
  }, []);

  // Handle restoring permission for pending project
  const handleRestorePermission = useCallback(async () => {
    setIsLoading(true);
    const success = await projectFileService.requestPendingPermission();
    if (success) {
      await loadProjectToStores();
      const data = projectFileService.getProjectData();
      if (data) {
        setProjectName(data.name);
        setIsProjectOpen(true);
      }
      setNeedsPermission(false);
      setPendingProjectName(null);
    }
    setIsLoading(false);
  }, []);

  const handleNewOutput = useCallback(() => {
    const output = createOutputWindow(`Output ${Date.now()}`);
    if (output) {
      console.log('Created output window:', output.id);
    }
    setOpenMenu(null);
  }, [createOutputWindow]);

  const handleMenuClick = (menuId: MenuId) => {
    setOpenMenu(openMenu === menuId ? null : menuId);
  };

  const handleMenuHover = (menuId: MenuId) => {
    if (openMenu !== null) {
      setOpenMenu(menuId);
    }
  };

  const closeMenu = () => setOpenMenu(null);

  return (
    <div className="toolbar">
      {/* Project Name */}
      <div className="toolbar-project">
        {needsPermission ? (
          <button
            className="restore-permission-btn"
            onClick={handleRestorePermission}
            disabled={isLoading}
            title={`Click to restore access to ${pendingProjectName}`}
          >
            {isLoading ? 'Restoring...' : `Restore "${pendingProjectName}"`}
          </button>
        ) : isEditingName ? (
          <input
            type="text"
            className="project-name-input"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={handleNameSubmit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleNameSubmit();
              if (e.key === 'Escape') setIsEditingName(false);
            }}
            autoFocus
          />
        ) : (
          <span
            className={`project-name ${!isProjectOpen ? 'no-project' : ''}`}
            onClick={() => {
              if (isProjectOpen) {
                setEditName(projectName);
                setIsEditingName(true);
              }
            }}
            title={isProjectOpen ? 'Click to rename project' : 'No project open'}
          >
            {projectName}
            {projectFileService.hasUnsavedChanges() && ' •'}
          </span>
        )}
      </div>

      {/* Menu Bar */}
      <div className="menu-bar" ref={menuBarRef}>
        {/* File Menu */}
        <div className="menu-item">
          <button
            className={`menu-trigger ${openMenu === 'file' ? 'active' : ''}`}
            onClick={() => handleMenuClick('file')}
            onMouseEnter={() => handleMenuHover('file')}
          >
            File
          </button>
          {openMenu === 'file' && (
            <div className="menu-dropdown">
              <button className="menu-option" onClick={handleNew} disabled={isLoading}>
                <span>New Project...</span>
                <span className="shortcut">Ctrl+N</span>
              </button>
              <button className="menu-option" onClick={handleOpen} disabled={isLoading}>
                <span>Open Project...</span>
                <span className="shortcut">Ctrl+O</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-option" onClick={handleSave} disabled={isLoading || !isProjectOpen}>
                <span>Save</span>
                <span className="shortcut">Ctrl+S</span>
              </button>
              {isProjectOpen && (
                <>
                  <div className="menu-separator" />
                  <div className="menu-submenu">
                    <span className="menu-label">Project Info</span>
                    <span className="menu-info">
                      {projectFileService.hasUnsavedChanges() ? '● Unsaved changes' : '✓ All changes saved'}
                    </span>
                  </div>
                </>
              )}
              <div className="menu-separator" />
              <button
                className="menu-option"
                onClick={async () => {
                  if (confirm('This will clear ALL cached data and reload. Continue?')) {
                    // Set flag to prevent beforeunload from saving data back
                    (window as any).__CLEARING_CACHE__ = true;

                    // Clear all localStorage
                    localStorage.clear();
                    sessionStorage.clear();

                    // Delete all known IndexedDB databases
                    const dbNames = ['webvj-db', 'webvj-projects', 'webvj-apikeys', 'keyval-store', 'MASterSelectsDB', 'multicam-settings'];
                    for (const name of dbNames) {
                      indexedDB.deleteDatabase(name);
                    }

                    // Clear caches
                    if ('caches' in window) {
                      const names = await caches.keys();
                      for (const name of names) {
                        await caches.delete(name);
                      }
                    }

                    // Unregister service workers
                    if ('serviceWorker' in navigator) {
                      const registrations = await navigator.serviceWorker.getRegistrations();
                      for (const reg of registrations) {
                        await reg.unregister();
                      }
                    }

                    // Clear again after a small delay to catch any last writes
                    setTimeout(() => {
                      localStorage.clear();
                      sessionStorage.clear();
                      // Force navigation to prevent any beforeunload handlers
                      window.location.href = window.location.origin + window.location.pathname + '?cleared=' + Date.now();
                    }, 100);
                  }
                }}
              >
                <span>Clear All Cache & Reload</span>
              </button>
            </div>
          )}
        </div>

        {/* Edit Menu */}
        <div className="menu-item">
          <button
            className={`menu-trigger ${openMenu === 'edit' ? 'active' : ''}`}
            onClick={() => handleMenuClick('edit')}
            onMouseEnter={() => handleMenuHover('edit')}
          >
            Edit
          </button>
          {openMenu === 'edit' && (
            <div className="menu-dropdown">
              <button className="menu-option" onClick={() => { document.execCommand('copy'); closeMenu(); }}>
                <span>Copy</span>
                <span className="shortcut">Ctrl+C</span>
              </button>
              <button className="menu-option" onClick={() => { document.execCommand('paste'); closeMenu(); }}>
                <span>Paste</span>
                <span className="shortcut">Ctrl+V</span>
              </button>
              <div className="menu-separator" />
              <button className="menu-option" onClick={() => { openSettings(); closeMenu(); }}>
                <span>Settings...</span>
              </button>
            </div>
          )}
        </div>

        {/* View Menu */}
        <div className="menu-item">
          <button
            className={`menu-trigger ${openMenu === 'view' ? 'active' : ''}`}
            onClick={() => handleMenuClick('view')}
            onMouseEnter={() => handleMenuHover('view')}
          >
            View
          </button>
          {openMenu === 'view' && (
            <div className="menu-dropdown menu-dropdown-wide">
              <div className="menu-submenu">
                <span className="menu-label">Panels</span>
                {(Object.keys(PANEL_CONFIGS) as PanelType[])
                  .filter(type => type !== 'slots') // Hide slots panel
                  .map((type) => {
                    const config = PANEL_CONFIGS[type];
                    const isVisible = isPanelTypeVisible(type);
                    return (
                      <button
                        key={type}
                        className={`menu-option ${isVisible ? 'checked' : ''}`}
                        onClick={() => togglePanelType(type)}
                      >
                        <span>{isVisible ? '✓ ' : '   '}{config.title}</span>
                      </button>
                    );
                  })}
              </div>
              <div className="menu-separator" />
              <button className="menu-option" onClick={handleNewOutput} disabled={!isEngineReady}>
                <span>New Output Window</span>
              </button>
              <div className="menu-separator" />
              <div className="menu-submenu">
                <span className="menu-label">Preview Quality</span>
                {([
                  { value: 1 as PreviewQuality, label: 'Full (100%)', desc: '1920×1080' },
                  { value: 0.5 as PreviewQuality, label: 'Half (50%)', desc: '960×540 - 4× faster' },
                  { value: 0.25 as PreviewQuality, label: 'Quarter (25%)', desc: '480×270 - 16× faster' },
                ]).map(({ value, label, desc }) => (
                  <button
                    key={value}
                    className={`menu-option ${previewQuality === value ? 'checked' : ''}`}
                    onClick={() => { setPreviewQuality(value); closeMenu(); }}
                  >
                    <span>{previewQuality === value ? '✓ ' : '   '}{label}</span>
                    <span className="menu-hint">{desc}</span>
                  </button>
                ))}
              </div>
              <div className="menu-separator" />
              <button className="menu-option" onClick={() => { saveLayoutAsDefault(); closeMenu(); }}>
                <span>Save Layout as Default</span>
              </button>
              <button className="menu-option" onClick={() => { resetLayout(); closeMenu(); }}>
                <span>Reset Layout</span>
              </button>
            </div>
          )}
        </div>

        {/* Output Menu */}
        <div className="menu-item">
          <button
            className={`menu-trigger ${openMenu === 'output' ? 'active' : ''}`}
            onClick={() => handleMenuClick('output')}
            onMouseEnter={() => handleMenuHover('output')}
          >
            Output
          </button>
          {openMenu === 'output' && (
            <div className="menu-dropdown">
              <button className="menu-option" onClick={handleNewOutput} disabled={!isEngineReady}>
                <span>New Output Window</span>
              </button>
              {outputWindows.length > 0 && (
                <>
                  <div className="menu-separator" />
                  <div className="menu-submenu">
                    <span className="menu-label">Active Outputs</span>
                    {outputWindows.map((output) => (
                      <div key={output.id} className="menu-option">
                        <span>{output.name || `Output ${output.id}`}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Window Menu */}
        <div className="menu-item">
          <button
            className={`menu-trigger ${openMenu === 'window' ? 'active' : ''}`}
            onClick={() => handleMenuClick('window')}
            onMouseEnter={() => handleMenuHover('window')}
          >
            Window
          </button>
          {openMenu === 'window' && (
            <div className="menu-dropdown">
              {midiSupported ? (
                <button
                  className={`menu-option ${midiEnabled ? 'checked' : ''}`}
                  onClick={() => { midiEnabled ? disableMIDI() : enableMIDI(); closeMenu(); }}
                >
                  <span>{midiEnabled ? '✓ ' : '   '}MIDI Control {midiEnabled && devices.length > 0 ? `(${devices.length} devices)` : ''}</span>
                </button>
              ) : (
                <span className="menu-option disabled">MIDI not supported</span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Spacer */}
      <div className="toolbar-spacer" />

      {/* Status */}
      <div className="toolbar-section toolbar-right">
        <span className={`status ${isEngineReady ? 'ready' : 'loading'}`}>
          {isEngineReady ? '● WebGPU Ready' : '○ Loading...'}
        </span>
      </div>

      {/* Settings Dialog */}
      {isSettingsOpen && <SettingsDialog onClose={closeSettings} />}
    </div>
  );
}
