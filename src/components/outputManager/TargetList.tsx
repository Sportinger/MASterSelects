// TargetList - lists all output-type render targets with controls
// Slices are shown nested under each target

import { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { useSliceStore } from '../../stores/sliceStore';
import { SourceSelector } from './SourceSelector';
import { renderScheduler } from '../../services/renderScheduler';
import { engine } from '../../engine/WebGPUEngine';
import type { RenderSource, RenderTarget } from '../../types/renderTarget';

interface TargetListProps {
  selectedTargetId: string | null;
  onSelect: (id: string) => void;
}

function isTargetClosed(target: RenderTarget): boolean {
  return target.window === null || target.window === undefined || target.window.closed;
}

/** Inline editable name — double-click to edit, Enter/blur to confirm, Escape to cancel */
function InlineEdit({ value, onCommit, className }: { value: string; onCommit: (v: string) => void; className?: string }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      setDraft(value);
      // Focus + select all after mount
      requestAnimationFrame(() => inputRef.current?.select());
    }
  }, [editing, value]);

  const commit = useCallback(() => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== value) {
      onCommit(trimmed);
    }
    setEditing(false);
  }, [draft, value, onCommit]);

  if (!editing) {
    return (
      <span
        className={className}
        onDoubleClick={(e) => { e.stopPropagation(); setEditing(true); }}
        title="Double-click to rename"
      >
        {value}
      </span>
    );
  }

  return (
    <input
      ref={inputRef}
      className="om-inline-edit"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit();
        if (e.key === 'Escape') setEditing(false);
      }}
      onClick={(e) => e.stopPropagation()}
      autoFocus
    />
  );
}

export function TargetList({ selectedTargetId, onSelect }: TargetListProps) {
  const targets = useRenderTargetStore((s) => s.targets);
  const sliceConfigs = useSliceStore((s) => s.configs);
  const addSlice = useSliceStore((s) => s.addSlice);
  const removeSlice = useSliceStore((s) => s.removeSlice);
  const selectSlice = useSliceStore((s) => s.selectSlice);
  const setSliceEnabled = useSliceStore((s) => s.setSliceEnabled);
  const resetSliceWarp = useSliceStore((s) => s.resetSliceWarp);
  const renameSlice = useSliceStore((s) => s.renameSlice);
  const updateTargetName = useRenderTargetStore((s) => s.updateTargetName);

  const outputTargets = useMemo(() => {
    const result: RenderTarget[] = [];
    for (const t of targets.values()) {
      if (t.destinationType === 'window' || t.destinationType === 'tab') {
        result.push(t);
      }
    }
    return result;
  }, [targets]);

  const handleSourceChange = (targetId: string, source: RenderSource) => {
    const store = useRenderTargetStore.getState();
    store.updateTargetSource(targetId, source);

    // If switching to/from independent source, update scheduler
    if (source.type !== 'activeComp') {
      renderScheduler.register(targetId);
      renderScheduler.updateTargetSource(targetId);
    } else {
      renderScheduler.unregister(targetId);
    }
  };

  const handleToggleEnabled = (targetId: string, enabled: boolean) => {
    useRenderTargetStore.getState().setTargetEnabled(targetId, enabled);
  };

  const handleClose = (targetId: string) => {
    engine.closeOutputWindow(targetId);
  };

  const handleRestore = (targetId: string) => {
    engine.restoreOutputWindow(targetId);
  };

  const handleRemove = (targetId: string) => {
    engine.removeOutputTarget(targetId);
  };

  const handleNewOutput = () => {
    const id = `output_${Date.now()}`;
    engine.createOutputWindow(id, `Output ${Date.now()}`);
  };

  const handleAddSlice = () => {
    if (selectedTargetId) {
      addSlice(selectedTargetId);
    }
  };

  return (
    <div className="om-target-list">
      <div className="om-target-list-header">
        <span className="om-target-list-title">Outputs</span>
        <div className="om-header-buttons">
          <button className="om-add-btn" onClick={handleNewOutput} title="Add Output Window">
            + Output
          </button>
          <button
            className="om-add-btn om-add-slice-btn"
            onClick={handleAddSlice}
            disabled={!selectedTargetId}
            title={selectedTargetId ? 'Add Slice to selected output' : 'Select an output first'}
          >
            + Slice
          </button>
        </div>
      </div>
      <div className="om-target-items">
        {outputTargets.length === 0 && (
          <div className="om-empty">No output targets. Click "+ Output" to create one.</div>
        )}
        {outputTargets.map((target) => {
          const closed = isTargetClosed(target);
          const isSelected = selectedTargetId === target.id;
          const config = sliceConfigs.get(target.id);
          const slices = config?.slices ?? [];
          const selectedSliceId = config?.selectedSliceId ?? null;

          return (
            <div key={target.id}>
              <div
                className={`om-target-item ${isSelected ? 'selected' : ''} ${closed ? 'closed' : ''}`}
                onClick={() => onSelect(target.id)}
              >
                <div className="om-target-row">
                  <span className={`om-target-status ${closed ? 'closed' : target.enabled ? 'enabled' : 'disabled'}`} />
                  <InlineEdit
                    value={target.name}
                    onCommit={(name) => updateTargetName(target.id, name)}
                    className="om-target-name"
                  />
                  <span className="om-target-type">{closed ? 'closed' : target.destinationType}</span>
                </div>
                <div className="om-target-row om-target-controls">
                  {closed ? (
                    <>
                      <span className="om-source-label-readonly">
                        {target.source.type === 'activeComp' ? 'Active Comp' :
                         target.source.type === 'composition' ? 'Composition' :
                         target.source.type}
                      </span>
                      <button
                        className="om-restore-btn"
                        onClick={(e) => { e.stopPropagation(); handleRestore(target.id); }}
                        title="Restore output window"
                      >
                        Restore
                      </button>
                      <button
                        className="om-remove-btn"
                        onClick={(e) => { e.stopPropagation(); handleRemove(target.id); }}
                        title="Remove from list"
                      >
                        Remove
                      </button>
                    </>
                  ) : (
                    <>
                      <SourceSelector
                        currentSource={target.source}
                        onChange={(source) => handleSourceChange(target.id, source)}
                      />
                      <button
                        className={`om-toggle-btn ${target.enabled ? 'active' : ''}`}
                        onClick={(e) => { e.stopPropagation(); handleToggleEnabled(target.id, !target.enabled); }}
                        title={target.enabled ? 'Disable' : 'Enable'}
                      >
                        {target.enabled ? 'ON' : 'OFF'}
                      </button>
                      <button
                        className="om-close-btn"
                        onClick={(e) => { e.stopPropagation(); handleClose(target.id); }}
                        title="Close output"
                      >
                        X
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Slices nested under this target */}
              {slices.length > 0 && (
                <div className="om-slice-items-nested">
                  {slices.map((slice) => (
                    <div
                      key={slice.id}
                      className={`om-slice-item ${selectedSliceId === slice.id ? 'selected' : ''} ${!slice.enabled ? 'disabled' : ''}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelect(target.id);
                        selectSlice(target.id, slice.id);
                      }}
                    >
                      <div className="om-slice-row">
                        <span className={`om-target-status small ${slice.enabled ? 'enabled' : 'disabled'}`} />
                        <InlineEdit
                          value={slice.name}
                          onCommit={(name) => renameSlice(target.id, slice.id, name)}
                          className="om-slice-name"
                        />
                        <span className="om-slice-mode">Corner Pin</span>
                      </div>
                      <div className="om-slice-controls">
                        <button
                          className={`om-toggle-btn ${slice.enabled ? 'active' : ''}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSliceEnabled(target.id, slice.id, !slice.enabled);
                          }}
                        >
                          {slice.enabled ? 'ON' : 'OFF'}
                        </button>
                        <button
                          className="om-close-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            resetSliceWarp(target.id, slice.id);
                          }}
                          title="Reset warp"
                        >
                          Reset
                        </button>
                        <button
                          className="om-remove-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeSlice(target.id, slice.id);
                          }}
                          title="Delete slice"
                        >
                          Del
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
