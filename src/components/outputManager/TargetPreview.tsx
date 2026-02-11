// TargetPreview - live preview canvas for the selected output target
// In Input tab: renders source without slicing (default)
// In Output tab: renders sliced output when slices exist

import { useEffect, useRef } from 'react';
import { useRenderTargetStore } from '../../stores/renderTargetStore';
import { useSliceStore } from '../../stores/sliceStore';
import { renderScheduler } from '../../services/renderScheduler';
import { engine } from '../../engine/WebGPUEngine';

interface TargetPreviewProps {
  targetId: string | null;
}

const PREVIEW_ID = '__om_preview__';

export function TargetPreview({ targetId }: TargetPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const registeredRef = useRef(false);

  // Get the source from the selected target
  const selectedTarget = useRenderTargetStore((s) => targetId ? s.targets.get(targetId) ?? null : null);
  const source = selectedTarget?.source ?? null;

  // Get slice config for the selected target (for output tab rendering)
  const activeTab = useSliceStore((s) => s.activeTab);
  const sliceConfig = useSliceStore((s) => targetId ? s.configs.get(targetId) : undefined);
  const enabledSlices = sliceConfig?.slices.filter((s) => s.enabled) ?? [];
  const hasSlices = enabledSlices.length > 0 && activeTab === 'output';

  useEffect(() => {
    if (!canvasRef.current || !source) {
      // Cleanup if no source
      if (registeredRef.current) {
        renderScheduler.unregister(PREVIEW_ID);
        useRenderTargetStore.getState().unregisterTarget(PREVIEW_ID);
        engine.unregisterTargetCanvas(PREVIEW_ID);
        registeredRef.current = false;
      }
      return;
    }

    // Register canvas with engine
    const gpuContext = engine.registerTargetCanvas(PREVIEW_ID, canvasRef.current);
    if (!gpuContext) return;

    // Register as render target
    useRenderTargetStore.getState().registerTarget({
      id: PREVIEW_ID,
      name: 'Output Manager Preview',
      source,
      destinationType: 'canvas',
      enabled: true,
      canvas: canvasRef.current,
      context: gpuContext,
      window: null,
      isFullscreen: false,
    });

    // Register with scheduler for independent sources
    if (source.type !== 'activeComp') {
      renderScheduler.register(PREVIEW_ID);
    }

    registeredRef.current = true;

    return () => {
      if (source.type !== 'activeComp') {
        renderScheduler.unregister(PREVIEW_ID);
      }
      useRenderTargetStore.getState().unregisterTarget(PREVIEW_ID);
      engine.unregisterTargetCanvas(PREVIEW_ID);
      registeredRef.current = false;
    };
  }, [source]);

  // When in output tab with slices, render sliced output on each animation frame
  useEffect(() => {
    if (!hasSlices || !registeredRef.current) return;

    let rafId: number;
    const renderFrame = () => {
      if (enabledSlices.length > 0) {
        engine.renderSlicedToCanvas(PREVIEW_ID, enabledSlices);
      }
      rafId = requestAnimationFrame(renderFrame);
    };
    rafId = requestAnimationFrame(renderFrame);

    return () => cancelAnimationFrame(rafId);
  }, [hasSlices, enabledSlices]);

  if (!targetId || !source) {
    return (
      <div className="om-preview-empty">
        <span>Select a target to preview</span>
      </div>
    );
  }

  return (
    <div className="om-preview">
      <canvas
        ref={canvasRef}
        width={1920}
        height={1080}
        className="om-preview-canvas"
      />
    </div>
  );
}
