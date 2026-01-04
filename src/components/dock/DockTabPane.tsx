// Tab group container with tab bar and panel content

import { useCallback, useRef, useEffect, useState } from 'react';
import type { DockTabGroup, DockPanel, DropPosition } from '../../types/dock';
import { useDockStore } from '../../stores/dockStore';
import { DockPanelContent } from './DockPanelContent';
import { calculateDropPosition } from '../../utils/dockLayout';

const HOLD_DURATION = 500; // ms to hold before drag starts

interface DockTabPaneProps {
  group: DockTabGroup;
}

export function DockTabPane({ group }: DockTabPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const tabBarRef = useRef<HTMLDivElement>(null);
  const holdTimerRef = useRef<number | null>(null);
  const holdStartRef = useRef<{
    panel: DockPanel;
    offset: { x: number; y: number };
    mousePos: { x: number; y: number };
  } | null>(null);
  const [holdingTabId, setHoldingTabId] = useState<string | null>(null);
  const [holdProgress, setHoldProgress] = useState<'idle' | 'holding' | 'ready' | 'fading'>('idle');

  const { setActiveTab, startDrag, updateDrag, dragState, setPanelZoom, layout } = useDockStore();

  const activePanel = group.panels[group.activeIndex];
  const isDropTarget = dragState.dropTarget?.groupId === group.id;
  const dropPosition = dragState.dropTarget?.position;
  const panelZoom = activePanel ? (layout.panelZoom?.[activePanel.id] ?? 1.0) : 1.0;

  // Cancel any ongoing hold
  const cancelHold = useCallback(() => {
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    holdStartRef.current = null;

    // If we were holding, trigger fade out animation
    if (holdProgress === 'holding') {
      setHoldProgress('fading');
      // After fade animation, reset to idle
      setTimeout(() => {
        setHoldProgress('idle');
        setHoldingTabId(null);
      }, HOLD_DURATION);
    } else {
      setHoldProgress('idle');
      setHoldingTabId(null);
    }
  }, [holdProgress]);

  const handleTabClick = useCallback((index: number) => {
    setActiveTab(group.id, index);
  }, [group.id, setActiveTab]);

  const handleTabMouseDown = useCallback((e: React.MouseEvent, panel: DockPanel, index: number) => {
    if (e.button !== 0) return;

    // Set this tab as active
    setActiveTab(group.id, index);

    // Store offset and mouse position for when drag actually starts
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const offset = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    const mousePos = { x: e.clientX, y: e.clientY };

    // Start hold animation
    setHoldingTabId(panel.id);
    setHoldProgress('holding');
    holdStartRef.current = { panel, offset, mousePos };

    // After hold duration, start the actual drag
    holdTimerRef.current = window.setTimeout(() => {
      if (holdStartRef.current) {
        setHoldProgress('ready');
        const { panel: p, offset: o, mousePos: pos } = holdStartRef.current;
        // Start drag with correct initial position
        startDrag(p, group.id, o, pos);
        // Reset hold state after drag starts
        setTimeout(() => {
          setHoldProgress('idle');
          setHoldingTabId(null);
        }, 100);
      }
    }, HOLD_DURATION);
  }, [group.id, setActiveTab, startDrag]);

  const handleTabMouseUp = useCallback(() => {
    // Only cancel if we're still in holding phase (not yet dragging)
    if (holdProgress === 'holding') {
      cancelHold();
    }
  }, [holdProgress, cancelHold]);

  const handleTabMouseLeave = useCallback(() => {
    // Cancel hold if mouse leaves the tab before 500ms is reached
    if (holdProgress === 'holding') {
      cancelHold();
    }
  }, [holdProgress, cancelHold]);

  // Clean up timer on unmount and handle global mouse events during hold
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (holdProgress === 'holding') {
        cancelHold();
      }
    };

    const handleGlobalMouseMove = (e: MouseEvent) => {
      // Update stored mouse position during hold so drag starts at correct pos
      if (holdProgress === 'holding' && holdStartRef.current) {
        holdStartRef.current.mousePos = { x: e.clientX, y: e.clientY };
      }
    };

    // Add global listeners
    window.addEventListener('mouseup', handleGlobalMouseUp);
    window.addEventListener('mousemove', handleGlobalMouseMove);

    return () => {
      window.removeEventListener('mouseup', handleGlobalMouseUp);
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      // Don't clear timer here - it's managed by cancelHold and mouseDown
    };
  }, [holdProgress, cancelHold]);

  // Cleanup timer on unmount only
  useEffect(() => {
    return () => {
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
      }
    };
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragState.isDragging || !containerRef.current) return;
    if (dragState.sourceGroupId === group.id && group.panels.length === 1) return;

    const rect = containerRef.current.getBoundingClientRect();
    const position = calculateDropPosition(rect, e.clientX, e.clientY);

    updateDrag(
      { x: e.clientX, y: e.clientY },
      { groupId: group.id, position }
    );
  }, [dragState.isDragging, dragState.sourceGroupId, group.id, group.panels.length, updateDrag]);

  const handleMouseLeave = useCallback(() => {
    if (dragState.isDragging && dragState.dropTarget?.groupId === group.id) {
      updateDrag(dragState.currentPos, null);
    }
  }, [dragState, group.id, updateDrag]);

  // Handle Ctrl+wheel for panel zoom (only on tab bar)
  useEffect(() => {
    const tabBar = tabBarRef.current;
    if (!tabBar) return;

    const handleWheel = (e: WheelEvent) => {
      if (!e.ctrlKey || !activePanel) return;

      // Prevent browser zoom
      e.preventDefault();
      e.stopPropagation();

      // Calculate new zoom
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      const currentZoom = layout.panelZoom?.[activePanel.id] ?? 1.0;
      setPanelZoom(activePanel.id, currentZoom + delta);
    };

    tabBar.addEventListener('wheel', handleWheel, { passive: false });
    return () => tabBar.removeEventListener('wheel', handleWheel);
  }, [activePanel, layout.panelZoom, setPanelZoom]);

  return (
    <div
      ref={containerRef}
      className={`dock-tab-pane ${isDropTarget ? 'drop-target' : ''}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Tab bar - Ctrl+wheel here to zoom panel */}
      <div ref={tabBarRef} className="dock-tab-bar" title="Ctrl+Scroll to zoom | Hold to drag">
        {group.panels.map((panel, index) => {
          const isHolding = holdingTabId === panel.id && holdProgress === 'holding';
          const isReady = holdingTabId === panel.id && holdProgress === 'ready';
          const isFading = holdingTabId === panel.id && holdProgress === 'fading';
          const isDragging = dragState.isDragging && dragState.draggedPanel?.id === panel.id;

          return (
            <div
              key={panel.id}
              className={`dock-tab ${index === group.activeIndex ? 'active' : ''} ${
                isDragging ? 'dragging' : ''
              } ${isHolding ? 'hold-glow' : ''} ${isReady ? 'hold-ready' : ''} ${isFading ? 'hold-fade' : ''}`}
              onClick={() => handleTabClick(index)}
              onMouseDown={(e) => handleTabMouseDown(e, panel, index)}
              onMouseUp={handleTabMouseUp}
              onMouseLeave={handleTabMouseLeave}
            >
              <span className="dock-tab-title">{panel.title}</span>
            </div>
          );
        })}
      </div>

      {/* Panel content with zoom */}
      <div
        className="dock-panel-content"
        style={{ '--panel-zoom': panelZoom } as React.CSSProperties}
      >
        <div className="dock-panel-content-inner">
          {activePanel && <DockPanelContent type={activePanel.type} />}
        </div>
        {/* Zoom indicator */}
        {panelZoom !== 1.0 && (
          <div className="dock-zoom-indicator">
            {Math.round(panelZoom * 100)}%
          </div>
        )}
      </div>

      {/* Drop zone overlay */}
      {isDropTarget && dropPosition && (
        <div className={`dock-drop-overlay ${dropPosition}`} />
      )}
    </div>
  );
}
