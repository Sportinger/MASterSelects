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
  kling: string;
}

interface SettingsState {
  // API Keys
  apiKeys: APIKeys;

  // Transcription settings
  transcriptionProvider: TranscriptionProvider;

  // Preview settings
  previewQuality: PreviewQuality;

  // UI state
  isSettingsOpen: boolean;

  // Actions
  setApiKey: (provider: keyof APIKeys, key: string) => void;
  setTranscriptionProvider: (provider: TranscriptionProvider) => void;
  setPreviewQuality: (quality: PreviewQuality) => void;
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
        kling: '',
      },
      transcriptionProvider: 'local',
      previewQuality: 1, // Full quality by default
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
        }),
      }
    )
  )
);
