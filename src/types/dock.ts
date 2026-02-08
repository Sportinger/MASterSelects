// Dock system type definitions

// Panel types that can be docked
// Note: Effects, Transcript, Analysis are now integrated into Properties panel
export type PanelType = 'preview' | 'timeline' | 'clip-properties' | 'media' | 'export' | 'multicam' | 'ai-chat' | 'ai-video' | 'youtube' | 'transitions' | 'scope-waveform' | 'scope-histogram' | 'scope-vectorscope';

// Scope panel types for filtering in View menu
export const SCOPE_PANEL_TYPES: PanelType[] = ['scope-waveform', 'scope-histogram', 'scope-vectorscope'];

// WIP panel types — shown grayed out with bug icon in View menu
export const WIP_PANEL_TYPES: PanelType[] = ['multicam', 'transitions'];

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
  timeline: {
    type: 'timeline',
    title: 'Timeline',
    minWidth: 300,
    minHeight: 150,
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
  'ai-chat': {
    type: 'ai-chat',
    title: 'AI Chat',
    minWidth: 300,
    minHeight: 300,
    closable: false,
  },
  'ai-video': {
    type: 'ai-video',
    title: 'AI Video',
    minWidth: 300,
    minHeight: 400,
    closable: false,
  },
  youtube: {
    type: 'youtube',
    title: 'YouTube',
    minWidth: 300,
    minHeight: 400,
    closable: false,
  },
  transitions: {
    type: 'transitions',
    title: 'Transitions',
    icon: 'Blend',
    minWidth: 200,
    minHeight: 200,
    closable: false,
  },
  'scope-waveform': {
    type: 'scope-waveform',
    title: 'Waveform',
    minWidth: 200,
    minHeight: 200,
    closable: false,
  },
  'scope-histogram': {
    type: 'scope-histogram',
    title: 'Histogram',
    minWidth: 200,
    minHeight: 200,
    closable: false,
  },
  'scope-vectorscope': {
    type: 'scope-vectorscope',
    title: 'Vectorscope',
    minWidth: 200,
    minHeight: 200,
    closable: false,
  },
};
