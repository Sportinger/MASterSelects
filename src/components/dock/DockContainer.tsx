// Root dock container - wraps docked panels and renders floating panels

import { useEffect, useRef } from 'react';
import { useDockStore } from '../../stores/dockStore';
import { DockNode } from './DockNode';
import { FloatingPanel } from './FloatingPanel';
import './dock.css';

export function DockContainer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { layout, dragState, endDrag, cancelDrag, updateDrag } = useDockStore();

  // Global mouse handlers for drag operations
  useEffect(() => {
    if (!dragState.isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      // Update drag position (drop target is updated by individual panes)
      updateDrag({ x: e.clientX, y: e.clientY }, dragState.dropTarget);
    };

    const handleMouseUp = () => {
      endDrag();
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        cancelDrag();
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [dragState.isDragging, dragState.dropTarget, endDrag, cancelDrag, updateDrag]);

  return (
    <div
      ref={containerRef}
      className={`dock-container ${dragState.isDragging ? 'dragging' : ''}`}
    >
      {/* Main docked layout */}
      <div className="dock-root">
        <DockNode node={layout.root} />
      </div>

      {/* Floating panels */}
      {layout.floatingPanels.map((floating) => (
        <FloatingPanel key={floating.id} floating={floating} />
      ))}

      {/* Drag preview */}
      {dragState.isDragging && dragState.draggedPanel && (
        <div
          className="dock-drag-preview"
          style={{
            left: dragState.currentPos.x - dragState.dragOffset.x,
            top: dragState.currentPos.y - dragState.dragOffset.y,
          }}
        >
          {dragState.draggedPanel.title}
        </div>
      )}
    </div>
  );
}
