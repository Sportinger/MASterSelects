// Mask-related actions slice

import type { MaskActions, SliceCreator, ClipMask, MaskVertex, MaskEditMode } from './types';

export const createMaskSlice: SliceCreator<MaskActions> = (set, get) => ({
  setMaskEditMode: (mode: MaskEditMode) => {
    set({ maskEditMode: mode, maskDrawStart: null });
    if (mode === 'none') {
      set({ activeMaskId: null, selectedVertexIds: new Set() });
    }
  },

  setMaskDrawStart: (point) => {
    set({ maskDrawStart: point });
  },

  setActiveMask: (clipId, maskId) => {
    set({ activeMaskId: maskId, selectedVertexIds: new Set() });
    if (clipId && maskId) {
      set({ maskEditMode: 'editing' });
    }
  },

  selectVertex: (vertexId, addToSelection = false) => {
    const { selectedVertexIds } = get();
    if (addToSelection) {
      const newSet = new Set(selectedVertexIds);
      if (newSet.has(vertexId)) {
        newSet.delete(vertexId);
      } else {
        newSet.add(vertexId);
      }
      set({ selectedVertexIds: newSet });
    } else {
      set({ selectedVertexIds: new Set([vertexId]) });
    }
  },

  deselectAllVertices: () => {
    set({ selectedVertexIds: new Set() });
  },

  // Mask CRUD
  addMask: (clipId, maskData) => {
    const { clips, invalidateCache } = get();
    const maskId = `mask-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;

    const existingMasks = clips.find(c => c.id === clipId)?.masks || [];
    const maskCount = existingMasks.length + 1;

    const newMask: ClipMask = {
      id: maskId,
      name: maskData?.name || `Mask ${maskCount}`,
      vertices: maskData?.vertices || [],
      closed: maskData?.closed ?? false,
      opacity: maskData?.opacity ?? 1,
      feather: maskData?.feather ?? 0,
      featherQuality: maskData?.featherQuality ?? 50, // 1-100 (1-33=low, 34-66=medium, 67-100=high)
      inverted: maskData?.inverted ?? false,
      mode: maskData?.mode ?? 'add',
      expanded: maskData?.expanded ?? true,
      position: maskData?.position ?? { x: 0, y: 0 },
      visible: maskData?.visible ?? true,
    };

    set({
      clips: clips.map(c =>
        c.id === clipId
          ? { ...c, masks: [...(c.masks || []), newMask] }
          : c
      ),
    });

    invalidateCache();
    return maskId;
  },

  removeMask: (clipId, maskId) => {
    const { clips, activeMaskId, invalidateCache } = get();

    set({
      clips: clips.map(c =>
        c.id === clipId
          ? { ...c, masks: (c.masks || []).filter(m => m.id !== maskId) }
          : c
      ),
      activeMaskId: activeMaskId === maskId ? null : activeMaskId,
    });

    invalidateCache();
  },

  updateMask: (clipId, maskId, updates) => {
    const { clips, invalidateCache } = get();

    set({
      clips: clips.map(c =>
        c.id === clipId
          ? {
              ...c,
              masks: (c.masks || []).map(m =>
                m.id === maskId ? { ...m, ...updates } : m
              ),
            }
          : c
      ),
    });

    invalidateCache();
  },

  reorderMasks: (clipId, fromIndex, toIndex) => {
    const { clips, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip?.masks) return;

    const masks = [...clip.masks];
    const [removed] = masks.splice(fromIndex, 1);
    masks.splice(toIndex, 0, removed);

    set({
      clips: clips.map(c =>
        c.id === clipId ? { ...c, masks } : c
      ),
    });

    invalidateCache();
  },

  getClipMasks: (clipId) => {
    const { clips } = get();
    return clips.find(c => c.id === clipId)?.masks || [];
  },

  // Vertex CRUD
  addVertex: (clipId, maskId, vertexData, index) => {
    const { clips, invalidateCache } = get();
    const vertexId = `vertex-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;

    const newVertex: MaskVertex = {
      id: vertexId,
      x: vertexData.x,
      y: vertexData.y,
      handleIn: vertexData.handleIn || { x: 0, y: 0 },
      handleOut: vertexData.handleOut || { x: 0, y: 0 },
    };

    set({
      clips: clips.map(c => {
        if (c.id !== clipId) return c;
        return {
          ...c,
          masks: (c.masks || []).map(m => {
            if (m.id !== maskId) return m;
            const vertices = [...m.vertices];
            if (index !== undefined) {
              vertices.splice(index, 0, newVertex);
            } else {
              vertices.push(newVertex);
            }
            return { ...m, vertices };
          }),
        };
      }),
    });

    invalidateCache();
    return vertexId;
  },

  removeVertex: (clipId, maskId, vertexId) => {
    const { clips, selectedVertexIds, invalidateCache } = get();

    set({
      clips: clips.map(c => {
        if (c.id !== clipId) return c;
        return {
          ...c,
          masks: (c.masks || []).map(m => {
            if (m.id !== maskId) return m;
            return {
              ...m,
              vertices: m.vertices.filter(v => v.id !== vertexId),
            };
          }),
        };
      }),
      selectedVertexIds: new Set(
        Array.from(selectedVertexIds).filter(id => id !== vertexId)
      ),
    });

    invalidateCache();
  },

  updateVertex: (clipId, maskId, vertexId, updates) => {
    const { clips, invalidateCache } = get();

    set({
      clips: clips.map(c => {
        if (c.id !== clipId) return c;
        return {
          ...c,
          masks: (c.masks || []).map(m => {
            if (m.id !== maskId) return m;
            return {
              ...m,
              vertices: m.vertices.map(v =>
                v.id === vertexId ? { ...v, ...updates } : v
              ),
            };
          }),
        };
      }),
    });

    invalidateCache();
  },

  closeMask: (clipId, maskId) => {
    const { updateMask } = get();
    updateMask(clipId, maskId, { closed: true });
  },

  // Preset shapes
  addRectangleMask: (clipId) => {
    const { addMask, invalidateCache } = get();
    const maskId = addMask(clipId, { name: 'Rectangle Mask' });

    // Add rectangle vertices (normalized 0-1 coordinates)
    // Default rectangle covers 80% of the clip area, centered
    const margin = 0.1;
    const vertices: MaskVertex[] = [
      { id: `v-${Date.now()}-1`, x: margin, y: margin, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
      { id: `v-${Date.now()}-2`, x: 1 - margin, y: margin, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
      { id: `v-${Date.now()}-3`, x: 1 - margin, y: 1 - margin, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
      { id: `v-${Date.now()}-4`, x: margin, y: 1 - margin, handleIn: { x: 0, y: 0 }, handleOut: { x: 0, y: 0 } },
    ];

    const currentClips = get().clips;
    set({
      clips: currentClips.map(c => {
        if (c.id !== clipId) return c;
        return {
          ...c,
          masks: (c.masks || []).map(m =>
            m.id === maskId ? { ...m, vertices, closed: true } : m
          ),
        };
      }),
    });

    invalidateCache();
    return maskId;
  },

  addEllipseMask: (clipId) => {
    const { addMask, invalidateCache } = get();
    const maskId = addMask(clipId, { name: 'Ellipse Mask' });

    // Create ellipse using bezier curves (approximation)
    // Control point offset for circular bezier (~0.5523)
    const k = 0.5523;
    const cx = 0.5;
    const cy = 0.5;
    const rx = 0.4;
    const ry = 0.4;

    const vertices: MaskVertex[] = [
      // Top
      {
        id: `v-${Date.now()}-1`,
        x: cx,
        y: cy - ry,
        handleIn: { x: -rx * k, y: 0 },
        handleOut: { x: rx * k, y: 0 },
      },
      // Right
      {
        id: `v-${Date.now()}-2`,
        x: cx + rx,
        y: cy,
        handleIn: { x: 0, y: -ry * k },
        handleOut: { x: 0, y: ry * k },
      },
      // Bottom
      {
        id: `v-${Date.now()}-3`,
        x: cx,
        y: cy + ry,
        handleIn: { x: rx * k, y: 0 },
        handleOut: { x: -rx * k, y: 0 },
      },
      // Left
      {
        id: `v-${Date.now()}-4`,
        x: cx - rx,
        y: cy,
        handleIn: { x: 0, y: ry * k },
        handleOut: { x: 0, y: -ry * k },
      },
    ];

    const currentClips = get().clips;
    set({
      clips: currentClips.map(c => {
        if (c.id !== clipId) return c;
        return {
          ...c,
          masks: (c.masks || []).map(m =>
            m.id === maskId ? { ...m, vertices, closed: true } : m
          ),
        };
      }),
    });

    invalidateCache();
    return maskId;
  },
});
