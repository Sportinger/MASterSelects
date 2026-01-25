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
  PanelType,
  DockTabGroup,
  PanelData,
  PreviewPanelData,
} from '../types/dock';
import { PANEL_CONFIGS } from '../types/dock';
import {
  removePanel,
  insertPanelAtTarget,
  collapseSingleChildSplits,
} from '../utils/dockLayout';
import { Logger } from '../services/logger';

const log = Logger.create('DockStore');

// Valid panel types (used to filter out removed panels from saved layouts)
const VALID_PANEL_TYPES = new Set(Object.keys(PANEL_CONFIGS));

// Filter out invalid panel types from a layout node
function filterInvalidPanels(node: DockNode): DockNode | null {
  if (node.kind === 'tab-group') {
    const validPanels = node.panels.filter(p => VALID_PANEL_TYPES.has(p.type));
    if (validPanels.length === 0) return null;
    return {
      ...node,
      panels: validPanels,
      activeIndex: Math.min(node.activeIndex, validPanels.length - 1),
    };
  } else {
    const [left, right] = node.children;
    const filteredLeft = filterInvalidPanels(left);
    const filteredRight = filterInvalidPanels(right);
    if (!filteredLeft && !filteredRight) return null;
    if (!filteredLeft) return filteredRight;
    if (!filteredRight) return filteredLeft;
    return { ...node, children: [filteredLeft, filteredRight] };
  }
}

// Clean up a persisted layout by removing invalid panels
function cleanupPersistedLayout(layout: DockLayout): DockLayout {
  const cleanedRoot = filterInvalidPanels(layout.root);
  return {
    ...layout,
    root: cleanedRoot || DEFAULT_LAYOUT.root,
    floatingPanels: layout.floatingPanels.filter(fp => VALID_PANEL_TYPES.has(fp.panel.type)),
  };
}

// Default layout configuration
// 3-column layout: Media/Properties/AI Chat left, Preview center, Multi-Cam/Export right
// Timeline at bottom
const DEFAULT_LAYOUT: DockLayout = {
  root: {
    kind: 'split',
    id: 'root-split',
    direction: 'vertical',
    ratio: 0.55, // Top section 55%, Timeline 45%
    children: [
      {
        kind: 'split',
        id: 'top-split',
        direction: 'horizontal',
        ratio: 0.25, // Left column 25%
        children: [
          {
            kind: 'tab-group',
            id: 'left-group',
            panels: [
              { id: 'media', type: 'media', title: 'Media' },
              { id: 'clip-properties', type: 'clip-properties', title: 'Properties' },
              { id: 'ai-chat', type: 'ai-chat', title: 'AI Chat' },
              { id: 'ai-video', type: 'ai-video', title: 'AI Video' },
              { id: 'youtube', type: 'youtube', title: 'YouTube' },
            ],
            activeIndex: 0, // Media active
          },
          {
            kind: 'split',
            id: 'center-right-split',
            direction: 'horizontal',
            ratio: 0.73, // Center 73% of remaining (≈55% total), Right 27% (≈20% total)
            children: [
              {
                kind: 'tab-group',
                id: 'preview-group',
                panels: [
                  { id: 'preview', type: 'preview', title: 'Preview' },
                ],
                activeIndex: 0,
              },
              {
                kind: 'tab-group',
                id: 'right-group',
                panels: [
                  { id: 'multicam', type: 'multicam', title: 'Multi-Cam' },
                  { id: 'export', type: 'export', title: 'Export' },
                ],
                activeIndex: 1,
              },
            ],
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
  closePanelById: (panelId: string) => void;

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

  // Panel visibility
  getVisiblePanelTypes: () => PanelType[];
  isPanelTypeVisible: (type: PanelType) => boolean;
  togglePanelType: (type: PanelType) => void;
  showPanelType: (type: PanelType) => void;
  hidePanelType: (type: PanelType) => void;
  activatePanelType: (type: PanelType) => void;

  // Multiple preview panels
  addPreviewPanel: (compositionId: string | null) => void;
  updatePanelData: (panelId: string, data: Partial<import('../types/dock').PanelData>) => void;

  // Layout management
  resetLayout: () => void;
  saveLayoutAsDefault: () => void;

  // Project persistence (for saving/loading layout from project file)
  getLayoutForProject: () => DockLayout;
  setLayoutFromProject: (layout: DockLayout) => void;
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

        closePanelById: (panelId) => {
          const { layout } = get();
          // First check floating panels
          const floating = layout.floatingPanels.find(f => f.panel.id === panelId);
          if (floating) {
            set({
              layout: {
                ...layout,
                floatingPanels: layout.floatingPanels.filter(f => f.panel.id !== panelId),
              },
            });
            return;
          }
          // Find in docked panels
          const groupId = findGroupIdByPanelId(layout.root, panelId);
          if (groupId) {
            let newLayout = removePanel(layout, panelId, groupId);
            newLayout = {
              ...newLayout,
              root: collapseSingleChildSplits(newLayout.root),
            };
            set({ layout: newLayout });
          }
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

        getVisiblePanelTypes: () => {
          const { layout } = get();
          const types: PanelType[] = [];
          collectPanelTypes(layout.root, types);
          // Also check floating panels
          layout.floatingPanels.forEach((f) => {
            if (!types.includes(f.panel.type)) {
              types.push(f.panel.type);
            }
          });
          return types;
        },

        isPanelTypeVisible: (type) => {
          return get().getVisiblePanelTypes().includes(type);
        },

        togglePanelType: (type) => {
          const { isPanelTypeVisible, showPanelType, hidePanelType } = get();
          if (isPanelTypeVisible(type)) {
            hidePanelType(type);
          } else {
            showPanelType(type);
          }
        },

        showPanelType: (type) => {
          const { layout, isPanelTypeVisible } = get();
          if (isPanelTypeVisible(type)) return; // Already visible

          const config = PANEL_CONFIGS[type];
          const newPanel: DockPanel = {
            id: type,
            type,
            title: config.title,
          };

          // Find the right-group to add to, or create a new floating panel
          const rightGroup = findTabGroupById(layout.root, 'right-group');
          if (rightGroup) {
            const newLayout = insertPanelAtTarget(layout, newPanel, {
              groupId: 'right-group',
              position: 'center',
            });
            set({ layout: newLayout });
          } else {
            // Fallback: find any tab group
            const anyGroup = findFirstTabGroup(layout.root);
            if (anyGroup) {
              const newLayout = insertPanelAtTarget(layout, newPanel, {
                groupId: anyGroup.id,
                position: 'center',
              });
              set({ layout: newLayout });
            }
          }
        },

        hidePanelType: (type) => {
          const { layout } = get();

          // Find and remove the panel from the layout
          const result = findPanelAndGroup(layout.root, type);
          if (result) {
            let newLayout = removePanel(layout, result.panel.id, result.groupId);
            newLayout = {
              ...newLayout,
              root: collapseSingleChildSplits(newLayout.root),
            };
            set({ layout: newLayout });
          }

          // Also check floating panels
          const floatingIndex = layout.floatingPanels.findIndex((f) => f.panel.type === type);
          if (floatingIndex >= 0) {
            set({
              layout: {
                ...layout,
                floatingPanels: layout.floatingPanels.filter((_, i) => i !== floatingIndex),
              },
            });
          }
        },

        activatePanelType: (type) => {
          const { layout, setActiveTab, showPanelType, isPanelTypeVisible, bringToFront } = get();

          // First make sure the panel is visible
          if (!isPanelTypeVisible(type)) {
            showPanelType(type);
          }

          // Find the panel in the layout and activate it
          const result = findPanelAndGroup(layout.root, type);
          if (result) {
            // Find the actual tab group to get the panel index
            const group = findTabGroupById(layout.root, result.groupId);
            if (group) {
              const panelIndex = group.panels.findIndex(p => p.type === type);
              if (panelIndex >= 0) {
                setActiveTab(result.groupId, panelIndex);
              }
            }
          }

          // Also check floating panels
          const floatingPanel = layout.floatingPanels.find(f => f.panel.type === type);
          if (floatingPanel) {
            bringToFront(floatingPanel.id);
          }
        },

        addPreviewPanel: (compositionId) => {
          const { layout } = get();

          // Find the preview-group to add to
          const previewGroup = findTabGroupById(layout.root, 'preview-group');
          const newPanelId = `preview-${Date.now()}`;
          const newPanel: DockPanel = {
            id: newPanelId,
            type: 'preview',
            title: 'Preview',
            data: { compositionId } as PreviewPanelData,
          };

          if (previewGroup) {
            // Insert to the RIGHT of the preview group (side-by-side)
            const newLayout = insertPanelAtTarget(layout, newPanel, {
              groupId: 'preview-group',
              position: 'right',
            });
            set({ layout: newLayout });
          } else {
            // Fallback: find any tab group or create floating
            const anyGroup = findFirstTabGroup(layout.root);
            if (anyGroup) {
              const newLayout = insertPanelAtTarget(layout, newPanel, {
                groupId: anyGroup.id,
                position: 'right',
              });
              set({ layout: newLayout });
            }
          }
        },

        updatePanelData: (panelId, data) => {
          set((state) => ({
            layout: updatePanelDataInLayout(state.layout, panelId, data),
          }));
        },

        resetLayout: () => {
          // Check if there's a saved default layout
          const savedDefault = localStorage.getItem('webvj-dock-layout-default');
          if (savedDefault) {
            try {
              const parsed = JSON.parse(savedDefault);
              set({ layout: parsed, maxZIndex: 1000 });
              return;
            } catch (e) {
              console.error('Failed to parse saved default layout:', e);
            }
          }
          set({ layout: DEFAULT_LAYOUT, maxZIndex: 1000 });
        },

        saveLayoutAsDefault: () => {
          const { layout } = get();
          localStorage.setItem('webvj-dock-layout-default', JSON.stringify(layout));
        },

        getLayoutForProject: () => {
          return get().layout;
        },

        setLayoutFromProject: (layout: DockLayout) => {
          // Clean up any invalid panel types from the loaded layout
          const cleanedLayout = cleanupPersistedLayout(layout);
          set({ layout: cleanedLayout, maxZIndex: 1000 });
        },
      }),
      {
        name: 'webvj-dock-layout',
        partialize: (state) => ({ layout: state.layout, maxZIndex: state.maxZIndex }),
        merge: (persistedState, currentState) => {
          const persisted = persistedState as Partial<DockState> | undefined;
          if (persisted?.layout) {
            // Clean up any invalid panel types from persisted layout
            const cleanedLayout = cleanupPersistedLayout(persisted.layout);
            return {
              ...currentState,
              layout: cleanedLayout,
              maxZIndex: persisted.maxZIndex ?? currentState.maxZIndex,
            };
          }
          return currentState;
        },
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

// Helper: Collect all panel types in a node
function collectPanelTypes(node: DockNode, types: PanelType[]): void {
  if (node.kind === 'tab-group') {
    node.panels.forEach((p) => {
      if (!types.includes(p.type)) {
        types.push(p.type);
      }
    });
  } else {
    collectPanelTypes(node.children[0], types);
    collectPanelTypes(node.children[1], types);
  }
}

// Helper: Find a tab group by ID
function findTabGroupById(node: DockNode, groupId: string): DockTabGroup | null {
  if (node.kind === 'tab-group') {
    return node.id === groupId ? node : null;
  }
  const left = findTabGroupById(node.children[0], groupId);
  if (left) return left;
  return findTabGroupById(node.children[1], groupId);
}

// Helper: Find the first tab group in the tree
function findFirstTabGroup(node: DockNode): DockTabGroup | null {
  if (node.kind === 'tab-group') {
    return node;
  }
  const left = findFirstTabGroup(node.children[0]);
  if (left) return left;
  return findFirstTabGroup(node.children[1]);
}

// Helper: Find a panel and its group by panel type
function findPanelAndGroup(
  node: DockNode,
  panelType: PanelType
): { panel: DockPanel; groupId: string } | null {
  if (node.kind === 'tab-group') {
    const panel = node.panels.find((p) => p.type === panelType);
    if (panel) {
      return { panel, groupId: node.id };
    }
    return null;
  }
  const left = findPanelAndGroup(node.children[0], panelType);
  if (left) return left;
  return findPanelAndGroup(node.children[1], panelType);
}

// Helper: Find a panel's group ID by panel ID
function findGroupIdByPanelId(node: DockNode, panelId: string): string | null {
  if (node.kind === 'tab-group') {
    const panel = node.panels.find((p) => p.id === panelId);
    if (panel) {
      return node.id;
    }
    return null;
  }
  const left = findGroupIdByPanelId(node.children[0], panelId);
  if (left) return left;
  return findGroupIdByPanelId(node.children[1], panelId);
}

// Helper: Update panel data in layout
function updatePanelDataInLayout(
  layout: DockLayout,
  panelId: string,
  data: Partial<PanelData>
): DockLayout {
  return {
    ...layout,
    root: updatePanelDataInNode(layout.root, panelId, data),
    floatingPanels: layout.floatingPanels.map((f) =>
      f.panel.id === panelId
        ? { ...f, panel: { ...f.panel, data: { ...f.panel.data, ...data } as PanelData } }
        : f
    ),
  };
}

function updatePanelDataInNode(
  node: DockNode,
  panelId: string,
  data: Partial<PanelData>
): DockNode {
  if (node.kind === 'tab-group') {
    const panelIndex = node.panels.findIndex((p) => p.id === panelId);
    if (panelIndex >= 0) {
      const newPanels = [...node.panels];
      newPanels[panelIndex] = {
        ...newPanels[panelIndex],
        data: { ...newPanels[panelIndex].data, ...data } as PanelData,
      };
      return { ...node, panels: newPanels };
    }
    return node;
  }
  return {
    ...node,
    children: [
      updatePanelDataInNode(node.children[0], panelId, data),
      updatePanelDataInNode(node.children[1], panelId, data),
    ] as [DockNode, DockNode],
  };
}
