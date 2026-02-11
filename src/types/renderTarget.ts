// Unified RenderTarget system types
// Every video destination is a RenderTarget with a source (what to show)
// and a destination (where to show it)

// === Source Types ===

export type RenderSourceType = 'activeComp' | 'composition' | 'layer' | 'slot' | 'program';

export interface RenderSourceActiveComp { type: 'activeComp' }
export interface RenderSourceComposition { type: 'composition'; compositionId: string }
export interface RenderSourceLayer { type: 'layer'; compositionId: string; layerIds: string[] }
export interface RenderSourceSlot { type: 'slot'; slotIndex: number }
export interface RenderSourceProgram { type: 'program' }  // main mix output

export type RenderSource =
  | RenderSourceActiveComp
  | RenderSourceComposition
  | RenderSourceLayer
  | RenderSourceSlot
  | RenderSourceProgram;

// === Destination Types ===

export type RenderDestinationType = 'canvas' | 'window' | 'tab';

// === RenderTarget ===

export interface RenderTarget {
  id: string;
  name: string;
  source: RenderSource;
  destinationType: RenderDestinationType;
  enabled: boolean;
  showTransparencyGrid: boolean;
  // Runtime state (not serialized)
  canvas: HTMLCanvasElement | null;
  context: GPUCanvasContext | null;
  window: Window | null;
  isFullscreen: boolean;
}
