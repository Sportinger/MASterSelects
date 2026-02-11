// Slot assignment actions slice - extracted from compositionSlice
// Manages Resolume-style slot grid assignments

import type { Composition, MediaSliceCreator } from '../types';

export interface SlotActions {
  moveSlot: (compId: string, toSlotIndex: number) => void;
  unassignSlot: (compId: string) => void;
  getSlotMap: (totalSlots: number) => (Composition | null)[];
}

export const createSlotSlice: MediaSliceCreator<SlotActions> = (set, get) => ({
  moveSlot: (compId: string, toSlotIndex: number) => {
    const { slotAssignments } = get();
    const newAssignments = { ...slotAssignments };
    // Remove any comp currently at the target slot
    for (const [id, idx] of Object.entries(newAssignments)) {
      if (idx === toSlotIndex && id !== compId) {
        // Swap: move displaced comp to the dragged comp's old slot
        const oldSlot = newAssignments[compId];
        if (oldSlot !== undefined) {
          newAssignments[id] = oldSlot;
        } else {
          delete newAssignments[id];
        }
        break;
      }
    }
    newAssignments[compId] = toSlotIndex;
    set({ slotAssignments: newAssignments });
  },

  unassignSlot: (compId: string) => {
    const { slotAssignments } = get();
    const newAssignments = { ...slotAssignments };
    delete newAssignments[compId];
    set({ slotAssignments: newAssignments });
  },

  getSlotMap: (totalSlots: number) => {
    const { compositions, slotAssignments } = get();
    const map: (Composition | null)[] = new Array(totalSlots).fill(null);

    for (const [compId, slotIdx] of Object.entries(slotAssignments)) {
      if (slotIdx >= 0 && slotIdx < totalSlots) {
        const comp = compositions.find((c: Composition) => c.id === compId);
        if (comp) {
          map[slotIdx] = comp;
        }
      }
    }

    return map;
  },
});
