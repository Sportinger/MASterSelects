// Zustand store for AI Multicam Editor
// Handles multi-camera sync, CV analysis, transcription, and AI-powered edit generation

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { MediaFile } from './mediaStore';

// =============================================================================
// Types
// =============================================================================

export interface MultiCamSource {
  id: string;
  mediaFileId: string;
  name: string;
  role: 'wide' | 'closeup' | 'detail' | 'custom';
  customRole?: string;
  // Sync offset in milliseconds (relative to master camera)
  syncOffset: number;
  // Duration in milliseconds
  duration: number;
  // Thumbnail URL
  thumbnailUrl?: string;
}

export interface CameraAnalysis {
  cameraId: string;
  // Per-frame analysis data (sampled at intervals)
  frames: FrameAnalysis[];
}

export interface FrameAnalysis {
  timestamp: number; // ms
  motion: number; // 0-1
  sharpness: number; // 0-1
  faces: DetectedFace[];
  audioLevel: number; // 0-1
}

export interface DetectedFace {
  id: string;
  bbox: { x: number; y: number; width: number; height: number };
  confidence: number;
  position: 'left' | 'center' | 'right';
  size: number; // Relative size in frame (0-1)
}

export interface MultiCamAnalysis {
  projectDuration: number; // ms
  sampleInterval: number; // ms between samples
  cameras: CameraAnalysis[];
  audioLevels: { timestamp: number; level: number }[];
}

export interface TranscriptEntry {
  id: string;
  start: number; // ms
  end: number; // ms
  speaker: string;
  text: string;
}

export interface EditDecision {
  id: string;
  start: number; // ms
  end: number; // ms
  cameraId: string;
  reason?: string;
  confidence?: number; // 0-1, how confident the AI is in this decision
}

export type EditStyle = 'podcast' | 'interview' | 'music' | 'documentary' | 'custom';

export type AnalysisStatus = 'idle' | 'analyzing' | 'complete' | 'error';
export type TranscriptStatus = 'idle' | 'loading-model' | 'generating' | 'complete' | 'error';
export type EDLStatus = 'idle' | 'generating' | 'complete' | 'error';

// =============================================================================
// Store Interface
// =============================================================================

interface MultiCamState {
  // Cameras
  cameras: MultiCamSource[];
  masterCameraId: string | null;

  // Analysis
  analysis: MultiCamAnalysis | null;
  analysisProgress: number;
  analysisStatus: AnalysisStatus;
  analysisError: string | null;

  // Transcript
  transcript: TranscriptEntry[];
  transcriptProgress: number;
  transcriptStatus: TranscriptStatus;
  transcriptError: string | null;

  // Edit Decision List
  edl: EditDecision[];
  edlStatus: EDLStatus;
  edlError: string | null;

  // Settings
  apiKey: string | null;
  apiKeySet: boolean; // True if key is stored (we don't expose the actual key)
  editStyle: EditStyle;
  customPrompt: string;

  // UI State
  selectedCameraId: string | null;
  previewingEDL: boolean;
  edlPlayheadPosition: number;
}

interface MultiCamActions {
  // Camera management
  addCamera: (mediaFile: MediaFile) => void;
  removeCamera: (id: string) => void;
  updateCamera: (id: string, updates: Partial<MultiCamSource>) => void;
  setMasterCamera: (id: string) => void;
  reorderCameras: (fromIndex: number, toIndex: number) => void;

  // Sync
  syncCameras: () => Promise<void>;
  setSyncOffset: (cameraId: string, offset: number) => void;

  // Analysis
  analyzeAll: () => Promise<void>;
  cancelAnalysis: () => void;

  // Transcript
  generateTranscript: () => Promise<void>;
  importTranscript: (entries: TranscriptEntry[]) => void;
  updateTranscriptEntry: (id: string, updates: Partial<TranscriptEntry>) => void;
  clearTranscript: () => void;

  // EDL Generation
  generateEDL: () => Promise<void>;
  updateEditDecision: (id: string, updates: Partial<EditDecision>) => void;
  insertEditDecision: (decision: Omit<EditDecision, 'id'>, index?: number) => void;
  removeEditDecision: (id: string) => void;
  clearEDL: () => void;

  // Timeline Integration
  applyEDLToTimeline: () => void;

  // Settings
  setApiKey: (key: string) => Promise<void>;
  clearApiKey: () => Promise<void>;
  setEditStyle: (style: EditStyle) => void;
  setCustomPrompt: (prompt: string) => void;

  // UI
  selectCamera: (id: string | null) => void;
  setPreviewingEDL: (previewing: boolean) => void;
  setEDLPlayheadPosition: (position: number) => void;

  // Reset
  reset: () => void;
}

type MultiCamStore = MultiCamState & MultiCamActions;

// =============================================================================
// Initial State
// =============================================================================

const initialState: MultiCamState = {
  cameras: [],
  masterCameraId: null,

  analysis: null,
  analysisProgress: 0,
  analysisStatus: 'idle',
  analysisError: null,

  transcript: [],
  transcriptProgress: 0,
  transcriptStatus: 'idle',
  transcriptError: null,

  edl: [],
  edlStatus: 'idle',
  edlError: null,

  apiKey: null,
  apiKeySet: false,
  editStyle: 'podcast',
  customPrompt: '',

  selectedCameraId: null,
  previewingEDL: false,
  edlPlayheadPosition: 0,
};

// =============================================================================
// Helpers
// =============================================================================

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Track cancellation for async operations
let analysisController: { cancelled: boolean } | null = null;

// =============================================================================
// Store
// =============================================================================

export const useMultiCamStore = create<MultiCamStore>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    // =========================================================================
    // Camera Management
    // =========================================================================

    addCamera: (mediaFile: MediaFile) => {
      const { cameras } = get();

      // Check if already added
      if (cameras.some(c => c.mediaFileId === mediaFile.id)) {
        console.warn('[MultiCam] Camera already added:', mediaFile.name);
        return;
      }

      const camera: MultiCamSource = {
        id: generateId(),
        mediaFileId: mediaFile.id,
        name: mediaFile.name,
        role: 'custom',
        syncOffset: 0,
        duration: (mediaFile.duration || 0) * 1000, // Convert to ms
        thumbnailUrl: mediaFile.thumbnailUrl,
      };

      set(state => {
        const newCameras = [...state.cameras, camera];
        // Auto-set first camera as master
        const masterId = state.masterCameraId || camera.id;
        return {
          cameras: newCameras,
          masterCameraId: masterId,
        };
      });

      console.log('[MultiCam] Added camera:', camera.name);
    },

    removeCamera: (id: string) => {
      set(state => {
        const newCameras = state.cameras.filter(c => c.id !== id);
        let masterId = state.masterCameraId;

        // If removing master, set new master
        if (masterId === id) {
          masterId = newCameras.length > 0 ? newCameras[0].id : null;
        }

        return {
          cameras: newCameras,
          masterCameraId: masterId,
          selectedCameraId: state.selectedCameraId === id ? null : state.selectedCameraId,
        };
      });
    },

    updateCamera: (id: string, updates: Partial<MultiCamSource>) => {
      set(state => ({
        cameras: state.cameras.map(c =>
          c.id === id ? { ...c, ...updates } : c
        ),
      }));
    },

    setMasterCamera: (id: string) => {
      set({ masterCameraId: id });
    },

    reorderCameras: (fromIndex: number, toIndex: number) => {
      const { cameras } = get();
      if (fromIndex < 0 || fromIndex >= cameras.length) return;
      if (toIndex < 0 || toIndex >= cameras.length) return;
      if (fromIndex === toIndex) return;

      const newCameras = [...cameras];
      const [moved] = newCameras.splice(fromIndex, 1);
      newCameras.splice(toIndex, 0, moved);
      set({ cameras: newCameras });
    },

    // =========================================================================
    // Sync
    // =========================================================================

    syncCameras: async () => {
      const { cameras, masterCameraId } = get();

      if (cameras.length < 2) {
        console.warn('[MultiCam] Need at least 2 cameras to sync');
        return;
      }

      if (!masterCameraId) {
        console.warn('[MultiCam] No master camera set');
        return;
      }

      set({ analysisStatus: 'analyzing', analysisProgress: 0 });

      try {
        // Import audio sync service dynamically
        const { audioSync } = await import('../services/audioSync');

        // Get master camera
        const masterCamera = cameras.find(c => c.id === masterCameraId);
        if (!masterCamera) {
          throw new Error('Master camera not found');
        }

        // Sync each camera to master
        const updatedCameras = [...cameras];
        for (let i = 0; i < cameras.length; i++) {
          const camera = cameras[i];
          if (camera.id === masterCameraId) {
            // Master has 0 offset
            updatedCameras[i] = { ...camera, syncOffset: 0 };
            continue;
          }

          // Calculate sync offset
          const offset = await audioSync.findOffset(
            masterCamera.mediaFileId,
            camera.mediaFileId
          );

          updatedCameras[i] = { ...camera, syncOffset: offset };

          // Update progress
          set({ analysisProgress: Math.round(((i + 1) / cameras.length) * 100) });
        }

        set({
          cameras: updatedCameras,
          analysisStatus: 'complete',
          analysisProgress: 100,
        });

        console.log('[MultiCam] Sync complete');
      } catch (error) {
        console.error('[MultiCam] Sync failed:', error);
        set({
          analysisStatus: 'error',
          analysisError: error instanceof Error ? error.message : 'Sync failed',
        });
      }
    },

    setSyncOffset: (cameraId: string, offset: number) => {
      set(state => ({
        cameras: state.cameras.map(c =>
          c.id === cameraId ? { ...c, syncOffset: offset } : c
        ),
      }));
    },

    // =========================================================================
    // Analysis
    // =========================================================================

    analyzeAll: async () => {
      const { cameras } = get();

      if (cameras.length === 0) {
        console.warn('[MultiCam] No cameras to analyze');
        return;
      }

      // Set up cancellation
      analysisController = { cancelled: false };
      set({
        analysisStatus: 'analyzing',
        analysisProgress: 0,
        analysisError: null,
      });

      try {
        // Import analyzer dynamically
        const { multicamAnalyzer } = await import('../services/multicamAnalyzer');

        const analysis = await multicamAnalyzer.analyze(
          cameras,
          (progress) => {
            if (!analysisController?.cancelled) {
              set({ analysisProgress: progress });
            }
          },
          () => analysisController?.cancelled ?? false
        );

        if (analysisController?.cancelled) {
          set({ analysisStatus: 'idle', analysisProgress: 0 });
          return;
        }

        set({
          analysis,
          analysisStatus: 'complete',
          analysisProgress: 100,
        });

        console.log('[MultiCam] Analysis complete');
      } catch (error) {
        console.error('[MultiCam] Analysis failed:', error);
        set({
          analysisStatus: 'error',
          analysisError: error instanceof Error ? error.message : 'Analysis failed',
        });
      } finally {
        analysisController = null;
      }
    },

    cancelAnalysis: () => {
      if (analysisController) {
        analysisController.cancelled = true;
      }
      set({ analysisStatus: 'idle', analysisProgress: 0 });
    },

    // =========================================================================
    // Transcript
    // =========================================================================

    generateTranscript: async () => {
      const { cameras, masterCameraId } = get();

      if (cameras.length === 0) {
        console.warn('[MultiCam] No cameras for transcription');
        return;
      }

      // Use master camera's audio for transcription
      const sourceCamera = masterCameraId
        ? cameras.find(c => c.id === masterCameraId)
        : cameras[0];

      if (!sourceCamera) {
        console.warn('[MultiCam] No source camera for transcription');
        return;
      }

      set({
        transcriptStatus: 'loading-model',
        transcriptProgress: 0,
        transcriptError: null,
      });

      try {
        // Import Whisper service dynamically
        const { whisperService } = await import('../services/whisperService');

        set({ transcriptStatus: 'generating' });

        const transcript = await whisperService.transcribe(
          sourceCamera.mediaFileId,
          (progress) => set({ transcriptProgress: progress })
        );

        set({
          transcript,
          transcriptStatus: 'complete',
          transcriptProgress: 100,
        });

        console.log('[MultiCam] Transcription complete:', transcript.length, 'entries');
      } catch (error) {
        console.error('[MultiCam] Transcription failed:', error);
        set({
          transcriptStatus: 'error',
          transcriptError: error instanceof Error ? error.message : 'Transcription failed',
        });
      }
    },

    importTranscript: (entries: TranscriptEntry[]) => {
      set({
        transcript: entries,
        transcriptStatus: 'complete',
        transcriptProgress: 100,
      });
    },

    updateTranscriptEntry: (id: string, updates: Partial<TranscriptEntry>) => {
      set(state => ({
        transcript: state.transcript.map(e =>
          e.id === id ? { ...e, ...updates } : e
        ),
      }));
    },

    clearTranscript: () => {
      set({
        transcript: [],
        transcriptStatus: 'idle',
        transcriptProgress: 0,
      });
    },

    // =========================================================================
    // EDL Generation
    // =========================================================================

    generateEDL: async () => {
      const { cameras, analysis, transcript, apiKeySet, editStyle, customPrompt } = get();

      if (cameras.length === 0) {
        console.warn('[MultiCam] No cameras for EDL generation');
        return;
      }

      if (!apiKeySet) {
        set({
          edlStatus: 'error',
          edlError: 'API key not set. Please configure your Claude API key in settings.',
        });
        return;
      }

      set({
        edlStatus: 'generating',
        edlError: null,
      });

      try {
        // Import Claude service dynamically
        const { claudeService } = await import('../services/claudeService');

        const edl = await claudeService.generateEDL({
          cameras,
          analysis,
          transcript,
          editStyle,
          customPrompt: customPrompt || undefined,
        });

        set({
          edl,
          edlStatus: 'complete',
        });

        console.log('[MultiCam] EDL generated:', edl.length, 'decisions');
      } catch (error) {
        console.error('[MultiCam] EDL generation failed:', error);
        set({
          edlStatus: 'error',
          edlError: error instanceof Error ? error.message : 'EDL generation failed',
        });
      }
    },

    updateEditDecision: (id: string, updates: Partial<EditDecision>) => {
      set(state => ({
        edl: state.edl.map(d =>
          d.id === id ? { ...d, ...updates } : d
        ),
      }));
    },

    insertEditDecision: (decision: Omit<EditDecision, 'id'>, index?: number) => {
      const newDecision: EditDecision = {
        ...decision,
        id: generateId(),
      };

      set(state => {
        const newEDL = [...state.edl];
        if (index !== undefined && index >= 0 && index <= newEDL.length) {
          newEDL.splice(index, 0, newDecision);
        } else {
          newEDL.push(newDecision);
        }
        return { edl: newEDL };
      });
    },

    removeEditDecision: (id: string) => {
      set(state => ({
        edl: state.edl.filter(d => d.id !== id),
      }));
    },

    clearEDL: () => {
      set({
        edl: [],
        edlStatus: 'idle',
        edlError: null,
      });
    },

    // =========================================================================
    // Timeline Integration
    // =========================================================================

    applyEDLToTimeline: () => {
      const { edl, cameras } = get();

      if (edl.length === 0) {
        console.warn('[MultiCam] No EDL to apply');
        return;
      }

      // Import timeline store and apply EDL
      import('./timeline').then(({ useTimelineStore }) => {
        const timelineStore = useTimelineStore.getState();

        // Create a new track for multicam output
        const trackId = timelineStore.addTrack('Multicam Edit', 'video');

        // Import media store to get media files
        import('./mediaStore').then(({ useMediaStore }) => {
          const mediaStore = useMediaStore.getState();

          // Add clips for each edit decision
          for (const decision of edl) {
            const camera = cameras.find(c => c.id === decision.cameraId);
            if (!camera) continue;

            const mediaFile = mediaStore.files.find(f => f.id === camera.mediaFileId);
            if (!mediaFile || !mediaFile.file) continue;

            // Calculate in/out points considering sync offset
            const inPoint = (decision.start + camera.syncOffset) / 1000; // Convert to seconds
            const outPoint = (decision.end + camera.syncOffset) / 1000;
            const startTime = decision.start / 1000;

            timelineStore.addClipToTrack(
              trackId,
              mediaFile.file,
              startTime,
              inPoint,
              outPoint - inPoint
            );
          }

          console.log('[MultiCam] Applied EDL to timeline:', edl.length, 'clips');
        });
      });
    },

    // =========================================================================
    // Settings
    // =========================================================================

    setApiKey: async (key: string) => {
      try {
        // Import API key manager dynamically
        const { apiKeyManager } = await import('../services/apiKeyManager');
        await apiKeyManager.storeKey(key);
        set({ apiKeySet: true, apiKey: null }); // Don't store raw key in state
        console.log('[MultiCam] API key stored');
      } catch (error) {
        console.error('[MultiCam] Failed to store API key:', error);
        throw error;
      }
    },

    clearApiKey: async () => {
      try {
        const { apiKeyManager } = await import('../services/apiKeyManager');
        await apiKeyManager.clearKey();
        set({ apiKeySet: false, apiKey: null });
        console.log('[MultiCam] API key cleared');
      } catch (error) {
        console.error('[MultiCam] Failed to clear API key:', error);
      }
    },

    setEditStyle: (style: EditStyle) => {
      set({ editStyle: style });
    },

    setCustomPrompt: (prompt: string) => {
      set({ customPrompt: prompt });
    },

    // =========================================================================
    // UI
    // =========================================================================

    selectCamera: (id: string | null) => {
      set({ selectedCameraId: id });
    },

    setPreviewingEDL: (previewing: boolean) => {
      set({ previewingEDL: previewing });
    },

    setEDLPlayheadPosition: (position: number) => {
      set({ edlPlayheadPosition: position });
    },

    // =========================================================================
    // Reset
    // =========================================================================

    reset: () => {
      // Cancel any ongoing analysis
      if (analysisController) {
        analysisController.cancelled = true;
      }
      set(initialState);
    },
  }))
);

// =============================================================================
// Initialize - Check if API key exists on load
// =============================================================================

if (typeof window !== 'undefined') {
  setTimeout(async () => {
    try {
      const { apiKeyManager } = await import('../services/apiKeyManager');
      const hasKey = await apiKeyManager.hasKey();
      useMultiCamStore.setState({ apiKeySet: hasKey });
    } catch (error) {
      console.warn('[MultiCam] Failed to check API key:', error);
    }
  }, 100);
}
