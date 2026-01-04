// React hook for WebGPU engine integration - Optimized

import { useEffect, useRef, useCallback } from 'react';
import { engine } from '../engine/WebGPUEngine';
import { useMixerStore } from '../stores/mixerStore';
import { useTimelineStore } from '../stores/timelineStore';

export function useEngine() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isEngineReady = useMixerStore((state) => state.isEngineReady);
  const isPlaying = useMixerStore((state) => state.isPlaying);
  const initRef = useRef(false);

  // Initialize engine - only once
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    async function init() {
      const success = await engine.initialize();
      useMixerStore.getState().setEngineReady(success);
      if (success) {
        useMixerStore.getState().setPlaying(true);
      }
    }

    init();

    return () => {
      // Don't destroy on unmount - singleton should persist
    };
  }, []);

  // Set up canvas
  useEffect(() => {
    if (isEngineReady && canvasRef.current) {
      engine.setPreviewCanvas(canvasRef.current);
    }
  }, [isEngineReady]);

  // Render loop - optimized with layer snapshotting to prevent flickering
  useEffect(() => {
    if (!isEngineReady) return;

    // Stats update throttle
    let lastStatsUpdate = 0;

    const renderFrame = () => {
      try {
        // Always try to use cached frame first (works even during RAM preview rendering)
        const { playheadPosition, isRamPreviewing } = useTimelineStore.getState();
        if (engine.renderCachedFrame(playheadPosition)) {
          // Successfully rendered cached frame, skip live render
          return;
        }

        // CRITICAL: Skip live rendering during RAM Preview generation
        // Live rendering would seek videos and interfere with RAM Preview's seeking,
        // causing glitchy/wrong frames to be cached
        if (isRamPreviewing) {
          return;
        }

        // IMPORTANT: Snapshot layers array at start of frame to ensure consistency
        // This prevents flickering from reading partially updated state
        const layersSnapshot = useMixerStore.getState().layers;

        // Create a stable copy for this frame (shallow copy is sufficient
        // since we only read from it, never modify)
        const frameLayers = layersSnapshot.slice();

        // Render with snapshotted layers (engine handles empty layers by clearing to black)
        engine.render(frameLayers);

        // Throttle stats updates to reduce React re-renders (every 100ms for responsive display)
        const now = performance.now();
        if (now - lastStatsUpdate > 100) {
          useMixerStore.getState().setEngineStats(engine.getStats());
          lastStatsUpdate = now;
        }
      } catch (e) {
        console.error('Render error:', e);
      }
    };

    if (isPlaying) {
      engine.start(renderFrame);
    } else {
      engine.stop();
    }

    return () => {
      engine.stop();
    };
  }, [isEngineReady, isPlaying]);

  const createOutputWindow = useCallback((name: string) => {
    const id = `output_${Date.now()}`;
    return engine.createOutputWindow(id, name);
  }, []);

  const closeOutputWindow = useCallback((id: string) => {
    engine.closeOutputWindow(id);
  }, []);

  return {
    canvasRef,
    isEngineReady,
    isPlaying,
    createOutputWindow,
    closeOutputWindow,
  };
}
