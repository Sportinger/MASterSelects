import { useRef, useEffect } from 'react';
import { useGpuScope } from './useScopeAnalysis';

export function VectorscopeScope() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useGpuScope(canvasRef, 'vectorscope', true);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      const dpr = window.devicePixelRatio || 1;
      const size = Math.min(width, height);
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
      canvas.style.width = `${size}px`;
      canvas.style.height = `${size}px`;
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="scope-canvas-container vectorscope-container">
      <canvas ref={canvasRef} />
    </div>
  );
}
