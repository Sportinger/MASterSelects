// Zustand store for dock layout state management

import { create } from 'zustand';
import { subscribeWithSelector, persist } from 'zustand/middleware';
import type {
  DockLayout,
  DockNode,
  DockPanel,
  DockDragState,
  DropTarget,
  FloatingPanel,
  DockTabGroup,
  DockSplit,
} from '../types/dock';
import {
  findNodeById,
  findParentOfNode,
  removePanel,
  insertPanelAtTarget,
  collapseSingleChildSplits,
  generateId,
} from '../utils/dockLayout';

// Default layout configuration
// Large Preview on top-left, Effects/Slots tabbed on right, Timeline at bottom
const DEFAULT_LAYOUT: DockLayout = {
  root: {
    kind: 'split',
    id: 'root-split',
    direction: 'vertical',
    ratio: 0.7,
    children: [
      {
        kind: 'split',
        id: 'top-split',
        direction: 'horizontal',
        ratio: 0.8,
        children: [
          {
            kind: 'tab-group',
            id: 'preview-group',
            panels: [{ id: 'preview', type: 'preview', title: 'Preview' }],
            activeIndex: 0,
          },
          {
            kind: 'tab-group',
            id: 'right-group',
            panels: [
              { id: 'media', type: 'media', title: 'Media' },
              { id: 'clip-properties', type: 'clip-properties', title: 'Properties' },
              { id: 'effects', type: 'effects', title: 'Effects' },
              // Slots panel disabled - uses same layer system as timeline, causing conflicts
              // TODO: Create separate timelineLayers system for proper separation
            ],
            activeIndex: 0,
          },
        ],
      },
      {
        kind: 'tab-group',
        id: 'timeline-group',
        panels: [{ id: 'timeline', type: 'timeline', title: 'Timeline' }],
        activeIndex: 0,
      },
    ],
  },
  floatingPanels: [],
  panelZoom: {},
};

const DEFAULT_DRAG_STATE: DockDragState = {
  isDragging: false,
  draggedPanel: null,
  sourceGroupId: null,
  dropTarget: null,
  dragOffset: { x: 0, y: 0 },
  currentPos: { x: 0, y: 0 },
};

interface DockState {
  layout: DockLayout;
  dragState: DockDragState;
  maxZIndex: number;

  // Layout mutations
  setActiveTab: (groupId: string, index: number) => void;
  setSplitRatio: (splitId: string, ratio: number) => void;
  movePanel: (panelId: string, sourceGroupId: string, target: DropTarget) => void;
  closePanel: (panelId: string, groupId: string) => void;

  // Floating panel actions
  floatPanel: (panelId: string, groupId: string, position: { x: number; y: number }) => void;
  dockFloatingPanel: (floatingId: string, target: DropTarget) => void;
  updateFloatingPosition: (floatingId: string, position: { x: number; y: number }) => void;
  updateFloatingSize: (floatingId: string, size: { width: number; height: number }) => void;
  bringToFront: (floatingId: string) => void;

  // Drag state actions
  startDrag: (panel: DockPanel, sourceGroupId: string, offset: { x: number; y: number }, initialPos?: { x: number; y: number }) => void;
  updateDrag: (pos: { x: number; y: number }, dropTarget: DropTarget | null) => void;
  endDrag: () => void;
  cancelDrag: () => void;

  // Panel zoom
  setPanelZoom: (panelId: string, zoom: number) => void;
  getPanelZoom: (panelId: string) => number;

  // Layout management
  resetLayout: () => void;
}

export const useDockStore = create<DockState>()(
  subscribeWithSelector(
    persist(
      (set, get) => ({
        layout: DEFAULT_LAYOUT,
        dragState: DEFAULT_DRAG_STATE,
        maxZIndex: 1000,

        setActiveTab: (groupId, index) => {
          set((state) => ({
            layout: updateNodeInLayout(state.layout, groupId, (node) => {
              if (node.kind === 'tab-group') {
                return { ...node, activeIndex: Math.min(index, node.panels.length - 1) };
              }
              return node;
            }),
          }));
        },

        setSplitRatio: (splitId, ratio) => {
          set((state) => ({
            layout: updateNodeInLayout(state.layout, splitId, (node) => {
              if (node.kind === 'split') {
                return { ...node, ratio: Math.max(0.1, Math.min(0.9, ratio)) };
              }
              return node;
            }),
          }));
        },

        movePanel: (panelId, sourceGroupId, target) => {
          const { layout } = get();

          // Remove panel from source
          let newLayout = removePanel(layout, panelId, sourceGroupId);

          // Insert at target
          const panel = findPanelById(layout, panelId);
          if (panel) {
            newLayout = insertPanelAtTarget(newLayout, panel, target);
          }

          // Clean up empty groups and single-child splits
          newLayout = {
            ...newLayout,
            root: collapseSingleChildSplits(newLayout.root),
          };

          set({ layout: newLayout });
        },

        closePanel: (panelId, groupId) => {
          const { layout } = get();
          let newLayout = removePanel(layout, panelId, groupId);
          newLayout = {
            ...newLayout,
            root: collapseSingleChildSplits(newLayout.root),
          };
          set({ layout: newLayout });
        },

        floatPanel: (panelId, groupId, position) => {
          const { layout, maxZIndex } = get();
          const panel = findPanelById(layout, panelId);
          if (!panel) return;

          // Remove from dock
          let newLayout = removePanel(layout, panelId, groupId);
          newLayout = {
            ...newLayout,
            root: collapseSingleChildSplits(newLayout.root),
          };

          // Add as floating
          const floatingPanel: FloatingPanel = {
            id: `floating-${panelId}-${Date.now()}`,
            panel,
            position,
            size: { width: 400, height: 300 },
            zIndex: maxZIndex + 1,
          };

          set({
            layout: {
              ...newLayout,
              floatingPanels: [...newLayout.floatingPanels, floatingPanel],
            },
            maxZIndex: maxZIndex + 1,
          });
        },

        dockFloatingPanel: (floatingId, target) => {
          const { layout } = get();
          const floating = layout.floatingPanels.find((f) => f.id === floatingId);
          if (!floating) return;

          // Remove from floating
          const newFloating = layout.floatingPanels.filter((f) => f.id !== floatingId);

          // Insert at target
          const newLayout = insertPanelAtTarget(
            { ...layout, floatingPanels: newFloating },
            floating.panel,
            target
          );

          set({ layout: newLayout });
        },

        updateFloatingPosition: (floatingId, position) => {
          set((state) => ({
            layout: {
              ...state.layout,
              floatingPanels: state.layout.floatingPanels.map((f) =>
                f.id === floatingId ? { ...f, position } : f
              ),
            },
          }));
        },

        updateFloatingSize: (floatingId, size) => {
          set((state) => ({
            layout: {
              ...state.layout,
              floatingPanels: state.layout.floatingPanels.map((f) =>
                f.id === floatingId ? { ...f, size } : f
              ),
            },
          }));
        },

        bringToFront: (floatingId) => {
          const { maxZIndex } = get();
          set((state) => ({
            layout: {
              ...state.layout,
              floatingPanels: state.layout.floatingPanels.map((f) =>
                f.id === floatingId ? { ...f, zIndex: maxZIndex + 1 } : f
              ),
            },
            maxZIndex: maxZIndex + 1,
          }));
        },

        startDrag: (panel, sourceGroupId, offset, initialPos) => {
          set({
            dragState: {
              isDragging: true,
              draggedPanel: panel,
              sourceGroupId,
              dropTarget: null,
              dragOffset: offset,
              currentPos: initialPos || { x: 0, y: 0 },
            },
          });
        },

        updateDrag: (pos, dropTarget) => {
          set((state) => ({
            dragState: {
              ...state.dragState,
              currentPos: pos,
              dropTarget,
            },
          }));
        },

        endDrag: () => {
          const { dragState } = get();
          if (dragState.isDragging && dragState.draggedPanel && dragState.dropTarget && dragState.sourceGroupId) {
            get().movePanel(dragState.draggedPanel.id, dragState.sourceGroupId, dragState.dropTarget);
          }
          set({ dragState: DEFAULT_DRAG_STATE });
        },

        cancelDrag: () => {
          set({ dragState: DEFAULT_DRAG_STATE });
        },

        setPanelZoom: (panelId, zoom) => {
          const clampedZoom = Math.max(0.5, Math.min(2.0, zoom));
          set((state) => ({
            layout: {
              ...state.layout,
              panelZoom: {
                ...state.layout.panelZoom,
                [panelId]: clampedZoom,
              },
            },
          }));
        },

        getPanelZoom: (panelId) => {
          return get().layout.panelZoom[panelId] ?? 1.0;
        },

        resetLayout: () => {
          set({ layout: DEFAULT_LAYOUT, maxZIndex: 1000 });
        },
      }),
      {
        name: 'webvj-dock-layout',
        partialize: (state) => ({ layout: state.layout, maxZIndex: state.maxZIndex }),
      }
    )
  )
);

// Helper: Update a node in the layout tree
function updateNodeInLayout(
  layout: DockLayout,
  nodeId: string,
  updater: (node: DockNode) => DockNode
): DockLayout {
  return {
    ...layout,
    root: updateNodeRecursive(layout.root, nodeId, updater),
  };
}

function updateNodeRecursive(
  node: DockNode,
  nodeId: string,
  updater: (node: DockNode) => DockNode
): DockNode {
  if (node.id === nodeId) {
    return updater(node);
  }
  if (node.kind === 'split') {
    return {
      ...node,
      children: [
        updateNodeRecursive(node.children[0], nodeId, updater),
        updateNodeRecursive(node.children[1], nodeId, updater),
      ] as [DockNode, DockNode],
    };
  }
  return node;
}

// Helper: Find a panel by ID in the layout
function findPanelById(layout: DockLayout, panelId: string): DockPanel | null {
  // Check floating panels
  for (const floating of layout.floatingPanels) {
    if (floating.panel.id === panelId) {
      return floating.panel;
    }
  }
  // Check docked panels
  return findPanelInNode(layout.root, panelId);
}

function findPanelInNode(node: DockNode, panelId: string): DockPanel | null {
  if (node.kind === 'tab-group') {
    return node.panels.find((p) => p.id === panelId) || null;
  }
  const left = findPanelInNode(node.children[0], panelId);
  if (left) return left;
  return findPanelInNode(node.children[1], panelId);
}
