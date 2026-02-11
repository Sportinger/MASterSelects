// Unified RenderTarget store - single source of truth for all render destinations
// Not persisted: targets are transient (canvas refs, window refs)

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { RenderTarget, RenderSource } from '../types/renderTarget';
import { useMediaStore } from './mediaStore';

interface RenderTargetState {
  targets: Map<string, RenderTarget>;
  selectedTargetId: string | null;
}

interface RenderTargetActions {
  // Target lifecycle
  registerTarget: (target: RenderTarget) => void;
  unregisterTarget: (id: string) => void;
  deactivateTarget: (id: string) => void;

  // Source routing
  updateTargetSource: (id: string, source: RenderSource) => void;
  updateTargetName: (id: string, name: string) => void;
  setTargetEnabled: (id: string, enabled: boolean) => void;

  // Canvas binding (runtime GPU context)
  setTargetCanvas: (id: string, canvas: HTMLCanvasElement, context: GPUCanvasContext) => void;
  clearTargetCanvas: (id: string) => void;

  // Window binding (for output windows)
  setTargetWindow: (id: string, win: Window) => void;
  setTargetFullscreen: (id: string, isFullscreen: boolean) => void;

  // Transparency grid
  setTargetTransparencyGrid: (id: string, show: boolean) => void;

  // UI selection
  setSelectedTarget: (id: string | null) => void;

  // Derived helpers
  getActiveCompTargets: () => RenderTarget[];
  getIndependentTargets: () => RenderTarget[];
  resolveSourceToCompId: (source: RenderSource) => string | null;
}

export const useRenderTargetStore = create<RenderTargetState & RenderTargetActions>()(
  subscribeWithSelector((set, get) => ({
    targets: new Map(),
    selectedTargetId: null,

    registerTarget: (target) => {
      set((state) => {
        const next = new Map(state.targets);
        next.set(target.id, target);
        return { targets: next };
      });
    },

    unregisterTarget: (id) => {
      set((state) => {
        const target = state.targets.get(id);
        // Close output window if applicable
        if (target?.window && !target.window.closed) {
          target.window.close();
        }
        const next = new Map(state.targets);
        next.delete(id);
        return {
          targets: next,
          selectedTargetId: state.selectedTargetId === id ? null : state.selectedTargetId,
        };
      });
    },

    deactivateTarget: (id) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, canvas: null, context: null, window: null });
        return { targets: next };
      });
    },

    updateTargetSource: (id, source) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, source });
        return { targets: next };
      });
    },

    updateTargetName: (id, name) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, name });
        return { targets: next };
      });
    },

    setTargetEnabled: (id, enabled) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, enabled });
        return { targets: next };
      });
    },

    setTargetCanvas: (id, canvas, context) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, canvas, context });
        return { targets: next };
      });
    },

    clearTargetCanvas: (id) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, canvas: null, context: null });
        return { targets: next };
      });
    },

    setTargetWindow: (id, win) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, window: win });
        return { targets: next };
      });
    },

    setTargetFullscreen: (id, isFullscreen) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, isFullscreen });
        return { targets: next };
      });
    },

    setTargetTransparencyGrid: (id, show) => {
      set((state) => {
        const target = state.targets.get(id);
        if (!target) return state;
        const next = new Map(state.targets);
        next.set(id, { ...target, showTransparencyGrid: show });
        return { targets: next };
      });
    },

    setSelectedTarget: (id) => {
      set({ selectedTargetId: id });
    },

    // Returns all enabled targets that follow the active composition
    // (source type is 'activeComp' or 'program')
    getActiveCompTargets: () => {
      const { targets } = get();
      const result: RenderTarget[] = [];
      for (const target of targets.values()) {
        if (!target.enabled || !target.context) continue;
        if (target.source.type === 'activeComp' || target.source.type === 'program') {
          result.push(target);
        }
        // Composition targets that match the active comp are also served by main loop
        if (target.source.type === 'composition') {
          const activeCompId = useMediaStore.getState().activeCompositionId;
          if (target.source.compositionId === activeCompId) {
            result.push(target);
          }
        }
      }
      return result;
    },

    // Returns all enabled targets that need independent rendering
    // (source is a specific composition different from active, or a layer/slot)
    getIndependentTargets: () => {
      const { targets } = get();
      const activeCompId = useMediaStore.getState().activeCompositionId;
      const result: RenderTarget[] = [];
      for (const target of targets.values()) {
        if (!target.enabled || !target.context) continue;
        if (target.source.type === 'composition' && target.source.compositionId !== activeCompId) {
          result.push(target);
        }
        if (target.source.type === 'layer' || target.source.type === 'slot') {
          result.push(target);
        }
      }
      return result;
    },

    // Resolves a RenderSource to a compositionId (for rendering)
    resolveSourceToCompId: (source) => {
      switch (source.type) {
        case 'activeComp':
        case 'program':
          return useMediaStore.getState().activeCompositionId;
        case 'composition':
          return source.compositionId;
        case 'layer':
          return source.compositionId;
        case 'slot': {
          // Resolve slot to composition via activeLayerSlots
          // activeLayerSlots: Record<number, string | null> where key=layerIndex, value=compositionId
          const slots = useMediaStore.getState().activeLayerSlots;
          const compId = slots[source.slotIndex];
          return compId ?? null;
        }
        default:
          return null;
      }
    },
  }))
);
