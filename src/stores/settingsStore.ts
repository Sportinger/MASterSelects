// Settings store for API keys and app configuration
// Global settings persisted in browser localStorage
// API keys stored encrypted in IndexedDB via apiKeyManager

import { create } from 'zustand';
import { subscribeWithSelector, persist } from 'zustand/middleware';
import { apiKeyManager, type ApiKeyType } from '../services/apiKeyManager';
import { Logger } from '../services/logger';

const log = Logger.create('SettingsStore');

// Transcription provider options
export type TranscriptionProvider = 'local' | 'openai' | 'assemblyai' | 'deepgram';

// Preview quality options (multiplier on base resolution)
export type PreviewQuality = 1 | 0.5 | 0.25;

// GPU power preference options
export type GPUPowerPreference = 'high-performance' | 'low-power';

interface APIKeys {
  openai: string;
  assemblyai: string;
  deepgram: string;
  piapi: string;  // PiAPI key for AI video generation (Kling, Luma, etc.)
  youtube: string; // YouTube Data API v3 key (optional, Invidious works without)
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

  // Native Helper (Turbo Mode)
  turboModeEnabled: boolean;  // Use native helper for decoding when available
  nativeHelperPort: number;   // WebSocket port (default 9876)
  nativeHelperConnected: boolean;  // Current connection status

  // Mobile/Desktop view
  forceDesktopMode: boolean;  // Show desktop UI even on mobile devices

  // GPU preference
  gpuPowerPreference: GPUPowerPreference;  // 'high-performance' (dGPU) or 'low-power' (iGPU)

  // Media import settings
  copyMediaToProject: boolean;  // Copy imported files to project Raw/ folder

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
  setTurboModeEnabled: (enabled: boolean) => void;
  setNativeHelperPort: (port: number) => void;
  setNativeHelperConnected: (connected: boolean) => void;
  setForceDesktopMode: (force: boolean) => void;
  setGpuPowerPreference: (preference: GPUPowerPreference) => void;
  setCopyMediaToProject: (enabled: boolean) => void;
  setHasCompletedSetup: (completed: boolean) => void;
  openSettings: () => void;
  closeSettings: () => void;
  toggleSettings: () => void;

  // Helpers
  getActiveApiKey: () => string | null;
  hasApiKey: (provider: keyof APIKeys) => boolean;

  // API key persistence (encrypted in IndexedDB)
  loadApiKeys: () => Promise<void>;
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
        youtube: '',
        klingAccessKey: '',
        klingSecretKey: '',
      },
      transcriptionProvider: 'local',
      previewQuality: 1, // Full quality by default
      showTransparencyGrid: false, // Don't show checkerboard by default
      autosaveEnabled: false, // Autosave disabled by default
      autosaveInterval: 5, // 5 minutes default interval
      turboModeEnabled: true, // Try to use native helper by default
      nativeHelperPort: 9876, // Default WebSocket port
      nativeHelperConnected: false, // Not connected initially
      forceDesktopMode: false, // Use responsive detection by default
      gpuPowerPreference: 'high-performance', // Prefer dGPU by default
      copyMediaToProject: true, // Copy imported files to Raw/ folder by default
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
        // Also save to encrypted IndexedDB
        apiKeyManager.storeKeyByType(provider as ApiKeyType, key).catch((err) => {
          log.error('Failed to save API key:', err);
        });
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

      setTurboModeEnabled: (enabled) => {
        set({ turboModeEnabled: enabled });
      },

      setNativeHelperPort: (port) => {
        set({ nativeHelperPort: port });
      },

      setNativeHelperConnected: (connected) => {
        set({ nativeHelperConnected: connected });
      },

      setForceDesktopMode: (force) => {
        set({ forceDesktopMode: force });
      },

      setGpuPowerPreference: (preference) => {
        set({ gpuPowerPreference: preference });
      },

      setCopyMediaToProject: (enabled) => {
        set({ copyMediaToProject: enabled });
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

      // Load API keys from encrypted IndexedDB (call on app startup)
      loadApiKeys: async () => {
        try {
          const keys = await apiKeyManager.getAllKeys();
          set({ apiKeys: keys });
          log.info('API keys loaded from encrypted storage');
        } catch (err) {
          log.error('Failed to load API keys:', err);
        }
      },
    }),
    {
      name: 'masterselects-settings',
      // Don't persist API keys in localStorage - they go to encrypted IndexedDB
      // Don't persist transient UI state like isSettingsOpen
      partialize: (state) => ({
        transcriptionProvider: state.transcriptionProvider,
        previewQuality: state.previewQuality,
        showTransparencyGrid: state.showTransparencyGrid,
        autosaveEnabled: state.autosaveEnabled,
        autosaveInterval: state.autosaveInterval,
        turboModeEnabled: state.turboModeEnabled,
        nativeHelperPort: state.nativeHelperPort,
        forceDesktopMode: state.forceDesktopMode,
        gpuPowerPreference: state.gpuPowerPreference,
        copyMediaToProject: state.copyMediaToProject,
        hasCompletedSetup: state.hasCompletedSetup,
      }),
    }
  )
  )
);
