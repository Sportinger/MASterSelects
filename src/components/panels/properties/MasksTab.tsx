// Masks Tab - Create and edit clip masks
import { useState, useCallback } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import { startBatch, endBatch } from '../../../stores/historyStore';
import type { MaskMode, ClipMask } from '../../../types';
import { DraggableNumber } from './shared';

const MASK_MODES: { value: MaskMode; label: string }[] = [
  { value: 'add', label: 'Add' },
  { value: 'subtract', label: 'Subtract' },
  { value: 'intersect', label: 'Intersect' },
];

interface MaskItemProps {
  clipId: string;
  mask: ClipMask;
  isActive: boolean;
  onSelect: () => void;
}

function MaskItem({ clipId, mask, isActive, onSelect }: MaskItemProps) {
  // Use getState() for actions - they're stable and don't need subscriptions
  const { updateMask, removeMask, setActiveMask, setMaskEditMode } = useTimelineStore.getState();
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(mask.name);

  const handleBatchStart = useCallback(() => startBatch('Adjust mask'), []);
  const handleBatchEnd = useCallback(() => endBatch(), []);

  const handleNameDoubleClick = () => { setIsEditing(true); setEditName(mask.name); };
  const handleNameChange = () => { if (editName.trim()) updateMask(clipId, mask.id, { name: editName.trim() }); setIsEditing(false); };
  const handleEditMask = () => { onSelect(); setActiveMask(clipId, mask.id); setMaskEditMode('editing'); };

  return (
    <div className={`mask-item ${isActive ? 'active' : ''} ${mask.expanded ? 'expanded' : ''}`}>
      <div className="mask-item-header" onClick={onSelect}>
        <button className="mask-expand-btn" onClick={(e) => { e.stopPropagation(); updateMask(clipId, mask.id, { expanded: !mask.expanded }); }}>
          {mask.expanded ? '\u25BC' : '\u25B6'}
        </button>
        {isEditing ? (
          <input type="text" className="mask-name-input" value={editName} onChange={(e) => setEditName(e.target.value)}
            onBlur={handleNameChange} onKeyDown={(e) => { if (e.key === 'Enter') handleNameChange(); if (e.key === 'Escape') setIsEditing(false); }}
            autoFocus onClick={(e) => e.stopPropagation()} />
        ) : (
          <span className="mask-name" onDoubleClick={handleNameDoubleClick}>{mask.name}</span>
        )}
        <select className="mask-mode-select" value={mask.mode} onChange={(e) => updateMask(clipId, mask.id, { mode: e.target.value as MaskMode })}
          onClick={(e) => e.stopPropagation()}>
          {MASK_MODES.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
        <button className="mask-visible-btn" onClick={(e) => { e.stopPropagation(); updateMask(clipId, mask.id, { visible: !mask.visible }); }}
          title={mask.visible ? "Hide mask outline" : "Show mask outline"} style={{ opacity: mask.visible ? 1 : 0.5 }}>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            {mask.visible ? (<><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></>)
              : (<><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" /><line x1="1" y1="1" x2="23" y2="23" /></>)}
          </svg>
        </button>
        <button className="mask-edit-btn" onClick={(e) => { e.stopPropagation(); handleEditMask(); }} title="Edit mask path">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
          </svg>
        </button>
        <button className="mask-delete-btn" onClick={(e) => { e.stopPropagation(); removeMask(clipId, mask.id); }} title="Delete mask">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
      {mask.expanded && (
        <div className="mask-item-properties">
          <div className="control-row"><label>Opacity</label>
            <DraggableNumber value={mask.opacity * 100} onChange={(v) => updateMask(clipId, mask.id, { opacity: v / 100 })}
              defaultValue={100} sensitivity={1} decimals={0} suffix="%"
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} /></div>
          <div className="control-row"><label>Feather</label>
            <DraggableNumber value={mask.feather} onChange={(v) => updateMask(clipId, mask.id, { feather: v })}
              defaultValue={0} sensitivity={1} decimals={1} suffix="px"
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} /></div>
          <div className="control-row"><label>Quality</label>
            <DraggableNumber value={mask.featherQuality ?? 50} onChange={(v) => updateMask(clipId, mask.id, { featherQuality: Math.min(100, Math.max(1, Math.round(v))) })}
              defaultValue={50} min={1} max={100} sensitivity={1} decimals={0}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} /></div>
          <div className="control-row"><label>Position X</label>
            <DraggableNumber value={mask.position.x} onChange={(v) => updateMask(clipId, mask.id, { position: { ...mask.position, x: v } })}
              defaultValue={0} sensitivity={100} decimals={3}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} /></div>
          <div className="control-row"><label>Position Y</label>
            <DraggableNumber value={mask.position.y} onChange={(v) => updateMask(clipId, mask.id, { position: { ...mask.position, y: v } })}
              defaultValue={0} sensitivity={100} decimals={3}
              onDragStart={handleBatchStart} onDragEnd={handleBatchEnd} /></div>
          <div className="control-row"><label>Inverted</label>
            <input type="checkbox" checked={mask.inverted} onChange={(e) => updateMask(clipId, mask.id, { inverted: e.target.checked })} /></div>
          <div className="mask-info">{mask.vertices.length} vertices | {mask.closed ? 'Closed' : 'Open'}</div>
        </div>
      )}
    </div>
  );
}

interface MasksTabProps {
  clipId: string;
  masks: ClipMask[] | undefined;
}

export function MasksTab({ clipId, masks }: MasksTabProps) {
  // Reactive data - subscribe to specific values only
  const activeMaskId = useTimelineStore(state => state.activeMaskId);
  const maskEditMode = useTimelineStore(state => state.maskEditMode);
  // Actions from getState() - stable, no subscription needed
  const { addRectangleMask, addEllipseMask, setActiveMask, setMaskEditMode } = useTimelineStore.getState();
  const [showMaskMenu, setShowMaskMenu] = useState(false);

  const handleStartDrawMode = (mode: 'drawingRect' | 'drawingEllipse' | 'drawingPen') => setMaskEditMode(mode);

  return (
    <div className="properties-tab-content masks-tab">
      <div className="section-header-with-button">
        <div className="mask-add-menu-container">
          <button className="btn btn-sm btn-add" onClick={() => setShowMaskMenu(!showMaskMenu)}>+ Add</button>
          {showMaskMenu && (
            <div className="mask-add-menu">
              <button onClick={() => { addRectangleMask(clipId); setShowMaskMenu(false); }}>Rectangle</button>
              <button onClick={() => { addEllipseMask(clipId); setShowMaskMenu(false); }}>Ellipse</button>
            </div>
          )}
        </div>
      </div>

      <div className="mask-shape-tools">
        <button className={`mask-tool-btn ${maskEditMode === 'drawingRect' ? 'active' : ''}`}
          onClick={() => handleStartDrawMode('drawingRect')} title="Draw Rectangle Mask">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="1" /></svg>
        </button>
        <button className={`mask-tool-btn ${maskEditMode === 'drawingEllipse' ? 'active' : ''}`}
          onClick={() => handleStartDrawMode('drawingEllipse')} title="Draw Ellipse Mask">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2"><ellipse cx="12" cy="12" rx="9" ry="9" /></svg>
        </button>
        <button className={`mask-tool-btn ${maskEditMode === 'drawingPen' ? 'active' : ''}`}
          onClick={() => handleStartDrawMode('drawingPen')} title="Pen Tool">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 19l7-7 3 3-7 7-3-3z" /><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
            <path d="M2 2l7.586 7.586" /><circle cx="11" cy="11" r="2" />
          </svg>
        </button>
        {maskEditMode !== 'none' && maskEditMode !== 'editing' && (
          <button className="mask-tool-btn cancel" onClick={() => setMaskEditMode('none')} title="Cancel drawing (ESC)">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}
      </div>

      {maskEditMode !== 'none' && maskEditMode !== 'editing' && (
        <div className="mask-draw-hint">
          {maskEditMode === 'drawingRect' && 'Click and drag on preview to draw rectangle'}
          {maskEditMode === 'drawingEllipse' && 'Click and drag on preview to draw ellipse'}
          {maskEditMode === 'drawingPen' && 'Click to add points, click first point to close'}
        </div>
      )}

      {(!masks || masks.length === 0) ? (
        <div className="mask-empty">No masks. Use tools above or click "+ Add".</div>
      ) : (
        <div className="mask-list">
          {masks.map((mask) => (
            <MaskItem key={mask.id} clipId={clipId} mask={mask} isActive={activeMaskId === mask.id}
              onSelect={() => setActiveMask(clipId, mask.id)} />
          ))}
        </div>
      )}
    </div>
  );
}
