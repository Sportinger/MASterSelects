// Multi-layer playback actions slice - extracted from compositionSlice
// Resolume-style layer activation/deactivation

import type { Composition, MediaSliceCreator } from '../types';

export interface MultiLayerActions {
  activateOnLayer: (compositionId: string, layerIndex: number) => void;
  deactivateLayer: (layerIndex: number) => void;
  activateColumn: (colIndex: number) => void;
  deactivateAllLayers: () => void;
  setLayerOpacity: (layerIndex: number, opacity: number) => void;
}

export const createMultiLayerSlice: MediaSliceCreator<MultiLayerActions> = (set, get) => ({
  activateOnLayer: (compositionId: string, layerIndex: number) => {
    const { activeLayerSlots } = get();
    const newSlots = { ...activeLayerSlots };
    // Remove this comp from any other layer it might be on
    for (const [key, val] of Object.entries(newSlots)) {
      if (val === compositionId) {
        delete newSlots[Number(key)];
      }
    }
    newSlots[layerIndex] = compositionId;
    set({ activeLayerSlots: newSlots });
  },

  deactivateLayer: (layerIndex: number) => {
    const { activeLayerSlots } = get();
    const newSlots = { ...activeLayerSlots };
    delete newSlots[layerIndex];
    set({ activeLayerSlots: newSlots });
  },

  activateColumn: (colIndex: number) => {
    const GRID_COLS = 12;
    const GRID_ROWS = 4;
    const TOTAL_SLOTS = GRID_COLS * GRID_ROWS;
    const { compositions, slotAssignments } = get();
    const slotMap: (Composition | null)[] = new Array(TOTAL_SLOTS).fill(null);
    for (const [compId, slotIdx] of Object.entries(slotAssignments)) {
      if (slotIdx >= 0 && slotIdx < TOTAL_SLOTS) {
        const comp = compositions.find(c => c.id === compId);
        if (comp) { slotMap[slotIdx] = comp; }
      }
    }

    const newSlots: Record<number, string | null> = {};
    for (let row = 0; row < GRID_ROWS; row++) {
      const slotIndex = row * GRID_COLS + colIndex;
      const comp = slotMap[slotIndex];
      if (comp) {
        newSlots[row] = comp.id;
      }
    }
    set({ activeLayerSlots: newSlots });
  },

  deactivateAllLayers: () => {
    set({ activeLayerSlots: {} });
  },

  setLayerOpacity: (layerIndex: number, opacity: number) => {
    const { layerOpacities } = get();
    set({ layerOpacities: { ...layerOpacities, [layerIndex]: Math.max(0, Math.min(1, opacity)) } });
  },
});
