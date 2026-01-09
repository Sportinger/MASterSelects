// Toolbar component

import { useState, useEffect, useCallback } from 'react';
import { useEngine } from '../../hooks/useEngine';
import { useMixerStore } from '../../stores/mixerStore';
import { useDockStore } from '../../stores/dockStore';
import { useMediaStore } from '../../stores/mediaStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { useMIDI } from '../../hooks/useMIDI';
import { SettingsDialog } from './SettingsDialog';
import type { StoredProject } from '../../services/projectDB';

export function Toolbar() {
  const { isEngineReady, createOutputWindow } = useEngine();
  const { isPlaying, setPlaying, outputResolution, setResolution } = useMixerStore();
  const { resetLayout } = useDockStore();
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
  const { isSettingsOpen, openSettings, closeSettings } = useSettingsStore();

  const [showProjectMenu, setShowProjectMenu] = useState(false);
  const [projects, setProjects] = useState<StoredProject[]>([]);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editName, setEditName] = useState(currentProjectName);

  // Load project list when menu opens
  useEffect(() => {
    if (showProjectMenu) {
      getProjectList().then(setProjects);
    }
  }, [showProjectMenu, getProjectList]);

  const handleSave = useCallback(async () => {
    await saveProject();
    setShowProjectMenu(false);
  }, [saveProject]);

  const handleLoad = useCallback(async (projectId: string) => {
    await loadProject(projectId);
    setShowProjectMenu(false);
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
      setShowProjectMenu(false);
    }
  }, [newProject]);

  const handleNewOutput = () => {
    const output = createOutputWindow(`Output ${Date.now()}`);
    if (output) {
      console.log('Created output window:', output.id);
    }
  };

  return (
    <div className="toolbar">
      <div className="toolbar-section">
        <span className="logo">MASterSelects</span>
      </div>

      {/* Project Section */}
      <div className="toolbar-section project-section">
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
        <div className="project-buttons">
          <button className="btn btn-sm" onClick={handleNew} disabled={isLoading} title="New Project">
            New
          </button>
          <button className="btn btn-sm" onClick={handleSave} disabled={isLoading} title="Save Project">
            Save
          </button>
          <button
            className="btn btn-sm"
            onClick={() => setShowProjectMenu(!showProjectMenu)}
            disabled={isLoading}
            title="Open Project"
          >
            {isLoading ? 'Loading...' : 'Open'}
          </button>
        </div>
        {showProjectMenu && (
          <div className="project-menu">
            <div className="project-menu-header">Recent Projects</div>
            {projects.length === 0 ? (
              <div className="project-menu-empty">No saved projects</div>
            ) : (
              projects
                .sort((a, b) => b.updatedAt - a.updatedAt)
                .map((project) => (
                  <div
                    key={project.id}
                    className="project-menu-item"
                    onClick={() => handleLoad(project.id)}
                  >
                    <span className="project-menu-name">{project.name}</span>
                    <span className="project-menu-date">
                      {new Date(project.updatedAt).toLocaleDateString()}
                    </span>
                    <button
                      className="project-menu-delete"
                      onClick={(e) => handleDelete(project.id, e)}
                      title="Delete project"
                    >
                      ×
                    </button>
                  </div>
                ))
            )}
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="toolbar-section">
        <button className="btn btn-sm" onClick={openSettings} title="Settings">
          ⚙ Settings
        </button>
      </div>

      <div className="toolbar-section">
        <button
          className={`btn ${isPlaying ? 'btn-active' : ''}`}
          onClick={() => setPlaying(!isPlaying)}
          disabled={!isEngineReady}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
      </div>

      <div className="toolbar-section">
        <label>Resolution:</label>
        <select
          value={`${outputResolution.width}x${outputResolution.height}`}
          onChange={(e) => {
            const [w, h] = e.target.value.split('x').map(Number);
            setResolution(w, h);
          }}
        >
          <option value="1920x1080">1920×1080 (1080p)</option>
          <option value="1280x720">1280×720 (720p)</option>
          <option value="3840x2160">3840×2160 (4K)</option>
          <option value="1920x1200">1920×1200 (16:10)</option>
          <option value="1024x768">1024×768 (4:3)</option>
        </select>
      </div>

      <div className="toolbar-section">
        <button className="btn" onClick={handleNewOutput} disabled={!isEngineReady}>
          + Output Window
        </button>
      </div>

      <div className="toolbar-section">
        <button className="btn" onClick={resetLayout} title="Reset panel layout to default">
          Reset Layout
        </button>
      </div>

      <div className="toolbar-section">
        {midiSupported ? (
          <button
            className={`btn ${midiEnabled ? 'btn-active' : ''}`}
            onClick={() => (midiEnabled ? disableMIDI() : enableMIDI())}
          >
            🎹 MIDI {midiEnabled ? `(${devices.length})` : 'Off'}
          </button>
        ) : (
          <span className="midi-unsupported">MIDI not supported</span>
        )}
      </div>

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
