// Selection actions

import type { MediaSliceCreator, LabelColor } from '../types';

export interface SelectionActions {
  moveToFolder: (itemIds: string[], folderId: string | null) => void;
  setSelection: (ids: string[]) => void;
  addToSelection: (id: string) => void;
  removeFromSelection: (id: string) => void;
  clearSelection: () => void;
  setLabelColor: (itemIds: string[], color: LabelColor) => void;
}

export const createSelectionSlice: MediaSliceCreator<SelectionActions> = (set) => ({
  moveToFolder: (itemIds: string[], folderId: string | null) => {
    set((state) => ({
      files: state.files.map((f) =>
        itemIds.includes(f.id) ? { ...f, parentId: folderId } : f
      ),
      compositions: state.compositions.map((c) =>
        itemIds.includes(c.id) ? { ...c, parentId: folderId } : c
      ),
      folders: state.folders.map((f) =>
        itemIds.includes(f.id) ? { ...f, parentId: folderId } : f
      ),
      textItems: state.textItems.map((t) =>
        itemIds.includes(t.id) ? { ...t, parentId: folderId } : t
      ),
      solidItems: state.solidItems.map((s) =>
        itemIds.includes(s.id) ? { ...s, parentId: folderId } : s
      ),
    }));
  },

  setSelection: (ids: string[]) => {
    set({ selectedIds: ids });
  },

  addToSelection: (id: string) => {
    set((state) => ({
      selectedIds: state.selectedIds.includes(id)
        ? state.selectedIds
        : [...state.selectedIds, id],
    }));
  },

  removeFromSelection: (id: string) => {
    set((state) => ({
      selectedIds: state.selectedIds.filter((sid) => sid !== id),
    }));
  },

  clearSelection: () => {
    set({ selectedIds: [] });
  },

  setLabelColor: (itemIds: string[], color: LabelColor) => {
    set((state) => ({
      files: state.files.map((f) =>
        itemIds.includes(f.id) ? { ...f, labelColor: color } : f
      ),
      compositions: state.compositions.map((c) =>
        itemIds.includes(c.id) ? { ...c, labelColor: color } : c
      ),
      folders: state.folders.map((f) =>
        itemIds.includes(f.id) ? { ...f, labelColor: color } : f
      ),
      textItems: state.textItems.map((t) =>
        itemIds.includes(t.id) ? { ...t, labelColor: color } : t
      ),
      solidItems: state.solidItems.map((s) =>
        itemIds.includes(s.id) ? { ...s, labelColor: color } : s
      ),
    }));
  },
});
