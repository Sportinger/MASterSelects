// TargetList - lists all output-type render targets with controls

import { useMemo } from 'react';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
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

export function TargetList({ selectedTargetId, onSelect }: TargetListProps) {
  const targets = useRenderTargetStore((s) => s.targets);

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

  return (
    <div className="om-target-list">
      <div className="om-target-list-header">
        <span className="om-target-list-title">Output Targets</span>
        <button className="om-add-btn" onClick={handleNewOutput} title="Add Output Window">
          + Add
        </button>
      </div>
      <div className="om-target-items">
        {outputTargets.length === 0 && (
          <div className="om-empty">No output targets. Click "+ Add" to create one.</div>
        )}
        {outputTargets.map((target) => {
          const closed = isTargetClosed(target);
          return (
            <div
              key={target.id}
              className={`om-target-item ${selectedTargetId === target.id ? 'selected' : ''} ${closed ? 'closed' : ''}`}
              onClick={() => onSelect(target.id)}
            >
              <div className="om-target-row">
                <span className={`om-target-status ${closed ? 'closed' : target.enabled ? 'enabled' : 'disabled'}`} />
                <span className="om-target-name">{target.name}</span>
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
          );
        })}
      </div>
    </div>
  );
}
