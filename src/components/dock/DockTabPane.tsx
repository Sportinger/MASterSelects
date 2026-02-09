// Tab group container with tab bar and panel content

import { useCallback, useRef, useEffect, useState, useMemo } from 'react';
import type { DockTabGroup, DockPanel } from '../../types/dock';
import { WIP_PANEL_TYPES } from '../../types/dock';
import { useDockStore } from '../../stores/dockStore';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import { DockPanelContent } from './DockPanelContent';
import { calculateDropPosition } from '../../utils/dockLayout';

// Truncate text with ellipsis
const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 2) + '..';
};

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

  const { setActiveTab, startDrag, updateDrag, dragState, setPanelZoom, layout, activatePanelType } = useDockStore();
  const {
    getOpenCompositions,
    activeCompositionId,
    setActiveComposition,
    closeCompositionTab,
    reorderCompositionTabs
  } = useMediaStore();
  const { clips, selectedClipIds } = useTimelineStore();

  // Get selected clip name for dynamic tab titles (Properties/Audio panels)
  const selectedClipName = useMemo(() => {
    if (selectedClipIds.size === 0) return null;
    const clipId = [...selectedClipIds][0];
    const clip = clips.find(c => c.id === clipId);
    return clip?.name || null;
  }, [clips, selectedClipIds]);

  // State for dragging composition tabs
  const [draggedCompIndex, setDraggedCompIndex] = useState<number | null>(null);
  const [dropTargetIndex, setDropTargetIndex] = useState<number | null>(null);

  // Middle mouse drag scrolling for tabs
  const [isMiddleDragging, setIsMiddleDragging] = useState(false);
  const middleDragStartRef = useRef<{ x: number; scrollLeft: number } | null>(null);

  const activePanel = group.panels[group.activeIndex];
  const isDropTarget = dragState.dropTarget?.groupId === group.id;
  const dropPosition = dragState.dropTarget?.position;
  const panelZoom = activePanel ? (layout.panelZoom?.[activePanel.id] ?? 1.0) : 1.0;

  // Check if this group contains a timeline panel
  const hasTimelinePanel = group.panels.some(p => p.type === 'timeline');
  const openCompositions = hasTimelinePanel ? getOpenCompositions() : [];

  // Composition tab drag handlers (for reordering)
  const handleCompDragStart = useCallback((e: React.DragEvent, index: number) => {
    // Only start reorder drag if not holding for dock drag
    if (holdProgress !== 'idle') {
      e.preventDefault();
      return;
    }
    setDraggedCompIndex(index);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(index));
  }, [holdProgress]);

  const handleCompDragOver = useCallback((e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (draggedCompIndex !== null && draggedCompIndex !== index) {
      setDropTargetIndex(index);
    }
  }, [draggedCompIndex]);

  const handleCompDragLeave = useCallback(() => {
    setDropTargetIndex(null);
  }, []);

  const handleCompDrop = useCallback((e: React.DragEvent, toIndex: number) => {
    e.preventDefault();
    if (draggedCompIndex !== null && draggedCompIndex !== toIndex) {
      reorderCompositionTabs(draggedCompIndex, toIndex);
    }
    setDraggedCompIndex(null);
    setDropTargetIndex(null);
  }, [draggedCompIndex, reorderCompositionTabs]);

  const handleCompDragEnd = useCallback(() => {
    setDraggedCompIndex(null);
    setDropTargetIndex(null);
  }, []);

  // Hold-to-drag handler for composition tabs (to move the timeline panel)
  // Note: Currently unused but kept for potential future implementation
  const _handleCompTabMouseDown = useCallback((e: React.MouseEvent, compId: string) => {
    if (e.button !== 0) return;

    // Find the timeline panel in this group
    const timelinePanel = group.panels.find(p => p.type === 'timeline');
    if (!timelinePanel) return;

    // Set composition as active
    setActiveComposition(compId);

    // Store offset and mouse position for when dock drag actually starts
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const offset = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    const mousePos = { x: e.clientX, y: e.clientY };

    // Start hold animation
    setHoldingTabId(compId);
    setHoldProgress('holding');
    holdStartRef.current = { panel: timelinePanel, offset, mousePos };

    // After hold duration, start the actual dock panel drag
    holdTimerRef.current = window.setTimeout(() => {
      if (holdStartRef.current) {
        setHoldProgress('ready');
        const { panel: p, offset: o, mousePos: pos } = holdStartRef.current;
        startDrag(p, group.id, o, pos);
        setTimeout(() => {
          setHoldProgress('idle');
          setHoldingTabId(null);
        }, 100);
      }
    }, HOLD_DURATION);
  }, [group.panels, group.id, setActiveComposition, startDrag]);
  void _handleCompTabMouseDown; // Silence unused warning

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

  const _handleCompTabMouseUp = useCallback(() => {
    if (holdProgress === 'holding') {
      cancelHold();
    }
  }, [holdProgress, cancelHold]);
  void _handleCompTabMouseUp; // Silence unused warning

  const _handleCompTabMouseLeave = useCallback(() => {
    if (holdProgress === 'holding') {
      cancelHold();
    }
  }, [holdProgress, cancelHold]);
  void _handleCompTabMouseLeave; // Silence unused warning

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

  // Calculate tab insert index based on mouse position over tab bar
  const calculateTabInsertIndex = useCallback((mouseX: number): number => {
    if (!tabBarRef.current) return group.panels.length;

    const tabBar = tabBarRef.current;
    const tabs = tabBar.querySelectorAll('.dock-tab');
    const tabBarRect = tabBar.getBoundingClientRect();

    // Adjust for scroll position
    const relativeX = mouseX - tabBarRect.left + tabBar.scrollLeft;

    // Find the slot - check midpoint of each tab
    let insertIndex = 0;
    tabs.forEach((tab, index) => {
      const tabEl = tab as HTMLElement;
      const tabMidpoint = tabEl.offsetLeft + tabEl.offsetWidth / 2;
      if (relativeX > tabMidpoint) {
        insertIndex = index + 1;
      }
    });

    return insertIndex;
  }, [group.panels.length]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragState.isDragging || !containerRef.current) return;
    if (dragState.sourceGroupId === group.id && group.panels.length === 1) return;

    const rect = containerRef.current.getBoundingClientRect();
    const tabBarRect = tabBarRef.current?.getBoundingClientRect();
    let position = calculateDropPosition(rect, e.clientX, e.clientY);

    // Check if hovering over tab bar area for center position
    let tabInsertIndex: number | undefined;
    if (tabBarRect && e.clientY >= tabBarRect.top && e.clientY <= tabBarRect.bottom) {
      // Mouse is over tab bar - use center with specific insert index
      position = 'center';
      tabInsertIndex = calculateTabInsertIndex(e.clientX);
    } else if (position === 'center') {
      // Center but not over tab bar - insert at end
      tabInsertIndex = group.panels.length;
    }

    updateDrag(
      { x: e.clientX, y: e.clientY },
      { groupId: group.id, position, tabInsertIndex }
    );
  }, [dragState.isDragging, dragState.sourceGroupId, group.id, group.panels.length, updateDrag, calculateTabInsertIndex]);

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

  // Middle mouse drag to scroll tabs horizontally (like Blender)
  const handleTabBarMouseDown = useCallback((e: React.MouseEvent) => {
    // Only handle middle mouse button
    if (e.button !== 1) return;
    e.preventDefault();

    const tabBar = tabBarRef.current;
    if (!tabBar) return;

    setIsMiddleDragging(true);
    middleDragStartRef.current = {
      x: e.clientX,
      scrollLeft: tabBar.scrollLeft,
    };
  }, []);

  // Global mouse move/up for middle drag scrolling
  useEffect(() => {
    if (!isMiddleDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const tabBar = tabBarRef.current;
      if (!tabBar || !middleDragStartRef.current) return;

      const deltaX = e.clientX - middleDragStartRef.current.x;
      tabBar.scrollLeft = middleDragStartRef.current.scrollLeft - deltaX;
    };

    const handleMouseUp = () => {
      setIsMiddleDragging(false);
      middleDragStartRef.current = null;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isMiddleDragging]);

  return (
    <div
      ref={containerRef}
      className={`dock-tab-pane ${isDropTarget ? 'drop-target' : ''}`}
      data-group-id={group.id}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Tab bar - Ctrl+wheel here to zoom panel, middle mouse to scroll */}
      <div
        ref={tabBarRef}
        className={`dock-tab-bar ${isMiddleDragging ? 'middle-dragging' : ''}`}
        title="Ctrl+Scroll to zoom | Hold to drag | Middle-click drag to scroll"
        onMouseDown={handleTabBarMouseDown}
      >
        {/* For timeline panels, show drag handle + composition tabs */}
        {hasTimelinePanel && openCompositions.length > 0 ? (
          <>
            {/* Drag handle for repositioning the timeline panel */}
            {(() => {
              const timelinePanel = group.panels.find(p => p.type === 'timeline');
              const isHandleHolding = holdingTabId === 'timeline-handle' && holdProgress === 'holding';
              const isHandleReady = holdingTabId === 'timeline-handle' && holdProgress === 'ready';
              const isHandleFading = holdingTabId === 'timeline-handle' && holdProgress === 'fading';
              return timelinePanel ? (
                <div
                  className={`dock-tab-handle ${isHandleHolding ? 'hold-glow' : ''} ${isHandleReady ? 'hold-ready' : ''} ${isHandleFading ? 'hold-fade' : ''}`}
                  title="Hold to reposition panel"
                  onMouseDown={(e) => {
                    if (e.button !== 0) return;
                    const rect = (e.target as HTMLElement).getBoundingClientRect();
                    const offset = { x: e.clientX - rect.left, y: e.clientY - rect.top };
                    const mousePos = { x: e.clientX, y: e.clientY };
                    setHoldingTabId('timeline-handle');
                    setHoldProgress('holding');
                    holdStartRef.current = { panel: timelinePanel, offset, mousePos };
                    holdTimerRef.current = window.setTimeout(() => {
                      if (holdStartRef.current) {
                        setHoldProgress('ready');
                        const { panel: p, offset: o, mousePos: pos } = holdStartRef.current;
                        startDrag(p, group.id, o, pos);
                        setTimeout(() => {
                          setHoldProgress('idle');
                          setHoldingTabId(null);
                        }, 100);
                      }
                    }, HOLD_DURATION);
                  }}
                  onMouseUp={() => { if (holdProgress === 'holding') cancelHold(); }}
                  onMouseLeave={() => { if (holdProgress === 'holding') cancelHold(); }}
                >
                  ⋮⋮
                </div>
              ) : null;
            })()}
            {/* Composition tabs - drag to reorder only */}
            {openCompositions.map((comp, index) => (
              <div
                key={comp.id}
                className={`dock-tab ${comp.id === activeCompositionId ? 'active' : ''} ${
                  draggedCompIndex === index ? 'dragging' : ''
                } ${dropTargetIndex === index ? 'drop-target-tab' : ''}`}
                onClick={() => {
                  setActiveComposition(comp.id);
                  activatePanelType('media');
                }}
                title={comp.name}
                draggable
                onDragStart={(e) => handleCompDragStart(e, index)}
                onDragOver={(e) => handleCompDragOver(e, index)}
                onDragLeave={handleCompDragLeave}
                onDrop={(e) => handleCompDrop(e, index)}
                onDragEnd={handleCompDragEnd}
              >
                <span className="dock-tab-title">{comp.name}</span>
                <button
                  className="dock-tab-close"
                  onClick={(e) => {
                    e.stopPropagation();
                    closeCompositionTab(comp.id);
                  }}
                  title="Close"
                >
                  ×
                </button>
              </div>
            ))}
          </>
        ) : (
          /* Normal dock tabs for non-timeline panels */
          group.panels.map((panel, index) => {
            const isHolding = holdingTabId === panel.id && holdProgress === 'holding';
            const isReady = holdingTabId === panel.id && holdProgress === 'ready';
            const isFading = holdingTabId === panel.id && holdProgress === 'fading';
            const isDragging = dragState.isDragging && dragState.draggedPanel?.id === panel.id;

            // Dynamic tab title for clip-properties panel
            let tabTitle = panel.title;
            if (panel.type === 'clip-properties' && selectedClipName) {
              tabTitle = truncateText(selectedClipName, 18);
            }

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
                title={panel.type === 'clip-properties' ? selectedClipName || panel.title : panel.title}
              >
                <span className="dock-tab-title">{tabTitle}{WIP_PANEL_TYPES.includes(panel.type) && <span className="menu-wip-badge">🐛</span>}</span>
              </div>
            );
          })
        )}
      </div>

      {/* Panel content with zoom */}
      <div
        className="dock-panel-content"
        style={{ '--panel-zoom': panelZoom } as React.CSSProperties}
      >
        <div className="dock-panel-content-inner">
          {activePanel && <DockPanelContent panel={activePanel} />}
        </div>
        {/* Zoom indicator */}
        {panelZoom !== 1.0 && (
          <div className="dock-zoom-indicator">
            {Math.round(panelZoom * 100)}%
          </div>
        )}
      </div>

      {/* Drop zone overlay */}
      {isDropTarget && dropPosition && dropPosition !== 'center' && (
        <div className={`dock-drop-overlay ${dropPosition}`} />
      )}

      {/* Tab slot indicators when dragging to center/tabs */}
      {isDropTarget && dropPosition === 'center' && (
        <div className="dock-tab-slots-overlay">
          {group.panels.map((_, index) => (
            <div
              key={`slot-${index}`}
              className={`dock-tab-slot ${dragState.dropTarget?.tabInsertIndex === index ? 'active' : ''}`}
            />
          ))}
          {/* Final slot after last tab */}
          <div
            className={`dock-tab-slot ${dragState.dropTarget?.tabInsertIndex === group.panels.length ? 'active' : ''}`}
          />
        </div>
      )}
    </div>
  );
}
