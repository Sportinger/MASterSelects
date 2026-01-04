// Preview canvas component

import { useEffect, useRef, useState } from 'react';
import { useEngine } from '../hooks/useEngine';
import { useMixerStore } from '../stores/mixerStore';

export function Preview() {
  const { canvasRef, isEngineReady } = useEngine();
  const { engineStats, outputResolution } = useMixerStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 1920, height: 1080 });

  // Calculate canvas size to fit container while maintaining aspect ratio
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;

      if (containerWidth === 0 || containerHeight === 0) return;

      const videoAspect = outputResolution.width / outputResolution.height;
      const containerAspect = containerWidth / containerHeight;

      let width: number;
      let height: number;

      if (containerAspect > videoAspect) {
        // Container is wider than video - fit to height
        height = containerHeight;
        width = height * videoAspect;
      } else {
        // Container is taller than video - fit to width
        width = containerWidth;
        height = width / videoAspect;
      }

      setCanvasSize({
        width: Math.floor(width),
        height: Math.floor(height),
      });
    };

    updateSize();

    const resizeObserver = new ResizeObserver(updateSize);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, [outputResolution.width, outputResolution.height]);

  return (
    <div className="preview-container" ref={containerRef}>
      <div className="preview-stats">
        {engineStats.fps} FPS | {outputResolution.width}x{outputResolution.height}
      </div>
      <div className="preview-canvas-wrapper">
        {!isEngineReady ? (
          <div className="loading">
            <div className="loading-spinner" />
            <p>Initializing WebGPU...</p>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={outputResolution.width}
            height={outputResolution.height}
            className="preview-canvas"
            style={{
              width: canvasSize.width,
              height: canvasSize.height,
            }}
          />
        )}
      </div>
    </div>
  );
}
