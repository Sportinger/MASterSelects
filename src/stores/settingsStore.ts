// Settings store for API keys and app configuration
// Persisted to localStorage

import { create } from 'zustand';
import { persist, subscribeWithSelector } from 'zustand/middleware';

// Transcription provider options
export type TranscriptionProvider = 'local' | 'openai' | 'assemblyai' | 'deepgram';

// Preview quality options (multiplier on base resolution)
export type PreviewQuality = 1 | 0.5 | 0.25;

interface APIKeys {
  openai: string;
  assemblyai: string;
  deepgram: string;
  piapi: string;  // PiAPI key for AI video generation (Kling, Luma, etc.)
  // Legacy Kling keys (deprecated, use piapi instead)
  klingAccessKey: string;
  klingSecretKey: string;
}

// Autosave interval options (in minutes)
export type AutosaveInterval = 1 | 2 | 5 | 10;

interface SettingsState {
  // API Keys
  apiKeys: APIKeys;

  // Transcription settings
  transcriptionProvider: TranscriptionProvider;

  // Preview settings
  previewQuality: PreviewQuality;
  showTransparencyGrid: boolean;  // Show checkerboard pattern for transparent areas

  // Autosave settings
  autosaveEnabled: boolean;
  autosaveInterval: AutosaveInterval;  // in minutes

  // First-run state
  hasCompletedSetup: boolean;

  // UI state
  isSettingsOpen: boolean;

  // Actions
  setApiKey: (provider: keyof APIKeys, key: string) => void;
  setTranscriptionProvider: (provider: TranscriptionProvider) => void;
  setPreviewQuality: (quality: PreviewQuality) => void;
  setShowTransparencyGrid: (show: boolean) => void;
  setAutosaveEnabled: (enabled: boolean) => void;
  setAutosaveInterval: (interval: AutosaveInterval) => void;
  setHasCompletedSetup: (completed: boolean) => void;
  openSettings: () => void;
  closeSettings: () => void;
  toggleSettings: () => void;

  // Helpers
  getActiveApiKey: () => string | null;
  hasApiKey: (provider: keyof APIKeys) => boolean;
}

export const useSettingsStore = create<SettingsState>()(
  subscribeWithSelector(
    persist(
      (set, get) => ({
      // Initial state
      apiKeys: {
        openai: '',
        assemblyai: '',
        deepgram: '',
        piapi: '',
        klingAccessKey: '',
        klingSecretKey: '',
      },
      transcriptionProvider: 'local',
      previewQuality: 1, // Full quality by default
      showTransparencyGrid: false, // Don't show checkerboard by default
      autosaveEnabled: false, // Autosave disabled by default
      autosaveInterval: 5, // 5 minutes default interval
      hasCompletedSetup: false, // Show welcome overlay on first run
      isSettingsOpen: false,

      // Actions
      setApiKey: (provider, key) => {
        set((state) => ({
          apiKeys: {
            ...state.apiKeys,
            [provider]: key,
          },
        }));
      },

      setTranscriptionProvider: (provider) => {
        set({ transcriptionProvider: provider });
      },

      setPreviewQuality: (quality) => {
        set({ previewQuality: quality });
      },

      setShowTransparencyGrid: (show) => {
        set({ showTransparencyGrid: show });
      },

      setAutosaveEnabled: (enabled) => {
        set({ autosaveEnabled: enabled });
      },

      setAutosaveInterval: (interval) => {
        set({ autosaveInterval: interval });
      },

      setHasCompletedSetup: (completed) => {
        set({ hasCompletedSetup: completed });
      },

      openSettings: () => set({ isSettingsOpen: true }),
      closeSettings: () => set({ isSettingsOpen: false }),
      toggleSettings: () => set((state) => ({ isSettingsOpen: !state.isSettingsOpen })),

      // Helpers
      getActiveApiKey: () => {
        const { transcriptionProvider, apiKeys } = get();
        if (transcriptionProvider === 'local') return null;
        return apiKeys[transcriptionProvider] || null;
      },

      hasApiKey: (provider) => {
        return !!get().apiKeys[provider];
      },
    }),
      {
        name: 'masterselects-settings',
        partialize: (state) => ({
          apiKeys: state.apiKeys,
          transcriptionProvider: state.transcriptionProvider,
          previewQuality: state.previewQuality,
          showTransparencyGrid: state.showTransparencyGrid,
          autosaveEnabled: state.autosaveEnabled,
          autosaveInterval: state.autosaveInterval,
          hasCompletedSetup: state.hasCompletedSetup,
        }),
      }
    )
  )
);
