// Transitions Panel - Drag and drop transitions for timeline clips

import { useState, useCallback } from 'react';
import { getAllTransitions, type TransitionDefinition } from '../../transitions';
import './TransitionsPanel.css';

// MIME type for drag data
export const TRANSITION_MIME_TYPE = 'application/x-transition-type';

// Transition preview thumbnail component
function TransitionPreview({ type }: { type: string }) {
  if (type === 'crossfade') {
    return (
      <svg viewBox="0 0 80 40" className="transition-preview-svg">
        {/* Left clip (fading out) */}
        <defs>
          <linearGradient id="fadeOutGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#4a9eff" stopOpacity="1" />
            <stop offset="100%" stopColor="#4a9eff" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="fadeInGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ff6b4a" stopOpacity="0" />
            <stop offset="100%" stopColor="#ff6b4a" stopOpacity="1" />
          </linearGradient>
        </defs>
        <rect x="0" y="8" width="50" height="24" fill="url(#fadeOutGrad)" rx="2" />
        <rect x="30" y="8" width="50" height="24" fill="url(#fadeInGrad)" rx="2" />
        {/* Overlap indicator */}
        <rect x="30" y="4" width="20" height="32" fill="rgba(255,255,255,0.1)" rx="2" />
      </svg>
    );
  }
  // Default preview
  return (
    <svg viewBox="0 0 80 40" className="transition-preview-svg">
      <rect x="5" y="8" width="30" height="24" fill="#4a9eff" rx="2" />
      <rect x="45" y="8" width="30" height="24" fill="#ff6b4a" rx="2" />
      <path d="M38 20 L42 20" stroke="white" strokeWidth="2" />
    </svg>
  );
}

interface TransitionItemProps {
  transition: TransitionDefinition;
  duration: number;
}

function TransitionItem({ transition, duration }: TransitionItemProps) {
  const handleDragStart = useCallback((e: React.DragEvent) => {
    e.dataTransfer.setData(TRANSITION_MIME_TYPE, JSON.stringify({
      type: transition.id,
      duration,
    }));
    e.dataTransfer.effectAllowed = 'copy';

    // Create drag image
    const dragEl = document.createElement('div');
    dragEl.className = 'transition-drag-preview';
    dragEl.textContent = transition.name;
    dragEl.style.cssText = 'position:fixed;top:-100px;background:#3b82f6;color:white;padding:4px 12px;border-radius:4px;font-size:12px;font-weight:500;pointer-events:none;';
    document.body.appendChild(dragEl);
    e.dataTransfer.setDragImage(dragEl, 40, 15);
    setTimeout(() => dragEl.remove(), 0);
  }, [transition.id, transition.name, duration]);

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      className="transition-item"
      title={transition.description}
    >
      <div className="transition-item-preview">
        <TransitionPreview type={transition.id} />
      </div>
      <span className="transition-item-name">{transition.name}</span>
    </div>
  );
}

export function TransitionsPanel() {
  const [duration, setDuration] = useState(0.5);
  const allTransitions = getAllTransitions();

  const handleDurationChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value) && value > 0) {
      setDuration(Math.min(Math.max(value, 0.1), 5.0));
    }
  }, []);

  return (
    <div className="transitions-panel">
      {/* Header with duration */}
      <div className="transitions-panel-header">
        <span className="transitions-panel-title">Transitions</span>
        <div className="transitions-duration-control">
          <input
            type="number"
            value={duration}
            onChange={handleDurationChange}
            min={0.1}
            max={5.0}
            step={0.1}
            className="transitions-duration-input"
          />
          <span className="transitions-duration-unit">s</span>
        </div>
      </div>

      {/* Transitions list */}
      <div className="transitions-list">
        {allTransitions.map((transition) => (
          <TransitionItem
            key={transition.id}
            transition={transition}
            duration={duration}
          />
        ))}

        {allTransitions.length === 0 && (
          <div className="transitions-empty">
            No transitions available
          </div>
        )}
      </div>

      {/* Footer hint */}
      <div className="transitions-panel-footer">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4M12 8h.01" />
        </svg>
        <span>Drag onto clip junction</span>
      </div>
    </div>
  );
}
