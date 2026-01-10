// Toolbar component - After Effects style menu bar

import { useState, useEffect, useCallback, useRef } from 'react';
import { useEngine } from '../../hooks/useEngine';
import { useMixerStore } from '../../stores/mixerStore';
import { useDockStore } from '../../stores/dockStore';
import { PANEL_CONFIGS, type PanelType } from '../../types/dock';
import { useMediaStore } from '../../stores/mediaStore';
import { useSettingsStore, type PreviewQuality } from '../../stores/settingsStore';
import { useMIDI } from '../../hooks/useMIDI';
import { SettingsDialog } from './SettingsDialog';
import type { StoredProject } from '../../services/projectDB';

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
  const {
    currentProjectName,
    setProjectName,
    saveProject,
    loadProject,
    newProject,
    getProjectList,
    deleteProject,
    isLoading,
  } = useMediaStore();
  const { isSupported: midiSupported, isEnabled: midiEnabled, enableMIDI, disableMIDI, devices } = useMIDI();
  const { isSettingsOpen, openSettings, closeSettings, previewQuality, setPreviewQuality } = useSettingsStore();

  const [openMenu, setOpenMenu] = useState<MenuId>(null);
  const [projects, setProjects] = useState<StoredProject[]>([]);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editName, setEditName] = useState(currentProjectName);
  const menuBarRef = useRef<HTMLDivElement>(null);

  // Load project list when file menu opens
  useEffect(() => {
    if (openMenu === 'file') {
      getProjectList().then(setProjects);
    }
  }, [openMenu, getProjectList]);

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
        saveProject();
      }
      // Ctrl+N: New
      if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        handleNew();
      }
      // Ctrl+O: Open
      if (e.ctrlKey && e.key === 'o') {
        e.preventDefault();
        setOpenMenu('file');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [saveProject]);

  const handleSave = useCallback(async () => {
    await saveProject();
    setOpenMenu(null);
  }, [saveProject]);

  const handleLoad = useCallback(async (projectId: string) => {
    await loadProject(projectId);
    setOpenMenu(null);
  }, [loadProject]);

  const handleDelete = useCallback(async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Delete this project?')) {
      await deleteProject(projectId);
      const updated = await getProjectList();
      setProjects(updated);
    }
  }, [deleteProject, getProjectList]);

  const handleNameSubmit = useCallback(() => {
    if (editName.trim()) {
      setProjectName(editName.trim());
    }
    setIsEditingName(false);
  }, [editName, setProjectName]);

  const handleNew = useCallback(() => {
    if (confirm('Create a new project? Unsaved changes will be lost.')) {
      newProject();
      setOpenMenu(null);
    }
  }, [newProject]);

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
        {isEditingName ? (
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
            className="project-name"
            onClick={() => {
              setEditName(currentProjectName);
              setIsEditingName(true);
            }}
            title="Click to rename project"
          >
            {currentProjectName}
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
                <span>New Project</span>
                <span className="shortcut">Ctrl+N</span>
              </button>
              <button className="menu-option" onClick={handleSave} disabled={isLoading}>
                <span>Save</span>
                <span className="shortcut">Ctrl+S</span>
              </button>
              <div className="menu-separator" />
              <div className="menu-submenu">
                <span className="menu-label">Open Recent</span>
                {projects.length === 0 ? (
                  <span className="menu-empty">No recent projects</span>
                ) : (
                  projects
                    .sort((a, b) => b.updatedAt - a.updatedAt)
                    .slice(0, 10)
                    .map((project) => (
                      <div
                        key={project.id}
                        className="menu-option project-item"
                        onClick={() => handleLoad(project.id)}
                      >
                        <span>{project.name}</span>
                        <button
                          className="delete-btn"
                          onClick={(e) => handleDelete(project.id, e)}
                          title="Delete"
                        >
                          ×
                        </button>
                      </div>
                    ))
                )}
              </div>
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
