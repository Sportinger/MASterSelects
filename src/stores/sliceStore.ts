// Slice Store - manages slice configurations per output target
// Separate from renderTargetStore since slices are user-editing state

import { create } from 'zustand';
import { useRenderTargetStore } from './renderTargetStore';
import type {
  OutputSlice,
  TargetSliceConfig,
  SliceInputRect,
  SliceWarp,
  Point2D,
} from '../types/outputSlice';
import { createDefaultSlice } from '../types/outputSlice';

interface SliceState {
  configs: Map<string, TargetSliceConfig>;
  activeTab: 'input' | 'output';
}

interface SliceActions {
  setActiveTab: (tab: 'input' | 'output') => void;
  getOrCreateConfig: (targetId: string) => TargetSliceConfig;
  removeConfig: (targetId: string) => void;
  addSlice: (targetId: string, name?: string) => string;
  removeSlice: (targetId: string, sliceId: string) => void;
  selectSlice: (targetId: string, sliceId: string | null) => void;
  setSliceEnabled: (targetId: string, sliceId: string, enabled: boolean) => void;
  setSliceInputRect: (targetId: string, sliceId: string, rect: SliceInputRect) => void;
  setCornerPinCorner: (targetId: string, sliceId: string, cornerIndex: number, point: Point2D) => void;
  updateWarp: (targetId: string, sliceId: string, warp: SliceWarp) => void;
  resetSliceWarp: (targetId: string, sliceId: string) => void;
}

function updateSliceInConfig(
  config: TargetSliceConfig,
  sliceId: string,
  updater: (slice: OutputSlice) => OutputSlice
): TargetSliceConfig {
  return {
    ...config,
    slices: config.slices.map((s) => (s.id === sliceId ? updater(s) : s)),
  };
}

export const useSliceStore = create<SliceState & SliceActions>()((set, get) => ({
  configs: new Map(),
  activeTab: 'output',

  setActiveTab: (tab) => set({ activeTab: tab }),

  getOrCreateConfig: (targetId) => {
    const { configs } = get();
    const existing = configs.get(targetId);
    if (existing) return existing;

    const config: TargetSliceConfig = {
      targetId,
      slices: [],
      selectedSliceId: null,
    };
    const next = new Map(configs);
    next.set(targetId, config);
    set({ configs: next });
    return config;
  },

  removeConfig: (targetId) => {
    set((state) => {
      const next = new Map(state.configs);
      next.delete(targetId);
      return { configs: next };
    });
  },

  addSlice: (targetId, name?) => {
    const config = get().getOrCreateConfig(targetId);
    const slice = createDefaultSlice(name);
    const next = new Map(get().configs);
    next.set(targetId, {
      ...config,
      slices: [...config.slices, slice],
      selectedSliceId: slice.id,
    });
    set({ configs: next });
    return slice.id;
  },

  removeSlice: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const newSlices = config.slices.filter((s) => s.id !== sliceId);
      const next = new Map(state.configs);
      next.set(targetId, {
        ...config,
        slices: newSlices,
        selectedSliceId: config.selectedSliceId === sliceId
          ? (newSlices.length > 0 ? newSlices[0].id : null)
          : config.selectedSliceId,
      });
      return { configs: next };
    });
  },

  selectSlice: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, { ...config, selectedSliceId: sliceId });
      return { configs: next };
    });
  },

  setSliceEnabled: (targetId, sliceId, enabled) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({ ...s, enabled })));
      return { configs: next };
    });
  },

  setSliceInputRect: (targetId, sliceId, rect) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({ ...s, inputRect: rect })));
      return { configs: next };
    });
  },

  setCornerPinCorner: (targetId, sliceId, cornerIndex, point) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => {
        if (s.warp.mode !== 'cornerPin') return s;
        const corners = [...s.warp.corners] as [Point2D, Point2D, Point2D, Point2D];
        corners[cornerIndex] = point;
        return { ...s, warp: { ...s.warp, corners } };
      }));
      return { configs: next };
    });
  },

  updateWarp: (targetId, sliceId, warp) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({ ...s, warp })));
      return { configs: next };
    });
  },

  resetSliceWarp: (targetId, sliceId) => {
    set((state) => {
      const config = state.configs.get(targetId);
      if (!config) return state;
      const next = new Map(state.configs);
      next.set(targetId, updateSliceInConfig(config, sliceId, (s) => ({
        ...s,
        inputRect: { x: 0, y: 0, width: 1, height: 1 },
        warp: {
          mode: 'cornerPin' as const,
          corners: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 1, y: 1 },
            { x: 0, y: 1 },
          ] as [Point2D, Point2D, Point2D, Point2D],
        },
      })));
      return { configs: next };
    });
  },
}));

// Cleanup subscription: remove orphaned configs when targets are removed
useRenderTargetStore.subscribe(
  (state) => state.targets,
  (targets) => {
    const { configs } = useSliceStore.getState();
    for (const targetId of configs.keys()) {
      if (!targets.has(targetId)) {
        useSliceStore.getState().removeConfig(targetId);
      }
    }
  }
);
