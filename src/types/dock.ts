// Dock system type definitions

// Panel types that can be docked
export type PanelType = 'preview' | 'effects' | 'timeline' | 'slots' | 'clip-properties' | 'media' | 'export' | 'multicam' | 'transcript' | 'analysis' | 'audio' | 'ai-chat';

// Panel-specific data for configurable panels
export interface PreviewPanelData {
  compositionId: string | null; // null = active composition
}

export type PanelData = PreviewPanelData;

// A panel instance
export interface DockPanel {
  id: string;
  type: PanelType;
  title: string;
  data?: PanelData; // Optional panel-specific configuration
}

// A group of tabbed panels
export interface DockTabGroup {
  kind: 'tab-group';
  id: string;
  panels: DockPanel[];
  activeIndex: number;
}

// A split container with two children
export interface DockSplit {
  kind: 'split';
  id: string;
  direction: 'horizontal' | 'vertical';
  children: [DockNode, DockNode];
  ratio: number; // 0-1, position of splitter
}

// Union type for dock tree nodes
export type DockNode = DockTabGroup | DockSplit;

// Floating panel (detached from dock)
export interface FloatingPanel {
  id: string;
  panel: DockPanel;
  position: { x: number; y: number };
  size: { width: number; height: number };
  zIndex: number;
}

// Root layout state
export interface DockLayout {
  root: DockNode;
  floatingPanels: FloatingPanel[];
  panelZoom: Record<string, number>; // Panel ID -> zoom level (1.0 = 100%)
}

// Drop target for drag operations
export type DropPosition = 'center' | 'left' | 'right' | 'top' | 'bottom';

export interface DropTarget {
  groupId: string;
  position: DropPosition;
  tabInsertIndex?: number; // When position is 'center', which slot to insert at
}

// Drag state
export interface DockDragState {
  isDragging: boolean;
  draggedPanel: DockPanel | null;
  sourceGroupId: string | null;
  dropTarget: DropTarget | null;
  dragOffset: { x: number; y: number };
  currentPos: { x: number; y: number };
}

// Panel metadata for configuration
export interface PanelConfig {
  type: PanelType;
  title: string;
  icon?: string;
  minWidth?: number;
  minHeight?: number;
  closable?: boolean;
}

export const PANEL_CONFIGS: Record<PanelType, PanelConfig> = {
  preview: {
    type: 'preview',
    title: 'Preview',
    minWidth: 200,
    minHeight: 150,
    closable: false,
  },
  effects: {
    type: 'effects',
    title: 'Effects',
    minWidth: 200,
    minHeight: 100,
    closable: false,
  },
  timeline: {
    type: 'timeline',
    title: 'Timeline',
    minWidth: 300,
    minHeight: 150,
    closable: false,
  },
  slots: {
    type: 'slots',
    title: 'Slots',
    minWidth: 200,
    minHeight: 200,
    closable: false,
  },
  'clip-properties': {
    type: 'clip-properties',
    title: 'Properties',
    minWidth: 200,
    minHeight: 150,
    closable: false,
  },
  media: {
    type: 'media',
    title: 'Media',
    minWidth: 200,
    minHeight: 200,
    closable: false,
  },
  export: {
    type: 'export',
    title: 'Export',
    minWidth: 200,
    minHeight: 300,
    closable: false,
  },
  multicam: {
    type: 'multicam',
    title: 'Multi-Cam',
    minWidth: 300,
    minHeight: 400,
    closable: false,
  },
  transcript: {
    type: 'transcript',
    title: 'Transcript',
    minWidth: 280,
    minHeight: 200,
    closable: false,
  },
  analysis: {
    type: 'analysis',
    title: 'Analysis',
    minWidth: 280,
    minHeight: 200,
    closable: false,
  },
  audio: {
    type: 'audio',
    title: 'Audio',
    minWidth: 280,
    minHeight: 300,
    closable: false,
  },
  'ai-chat': {
    type: 'ai-chat',
    title: 'AI Chat',
    minWidth: 300,
    minHeight: 300,
    closable: false,
  },
};
