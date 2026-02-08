import { useRef, useEffect } from 'react';

interface WaveformScopeProps {
  data: ImageData | null;
}

// IRE scale labels (0-100)
const IRE_MARKS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

export function WaveformScope({ data }: WaveformScopeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Responsive resize
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, w, h);

    // Layout: left margin for labels
    const labelW = 30 * dpr;
    const padR = 4 * dpr;
    const padT = 4 * dpr;
    const padB = 4 * dpr;
    const plotW = w - labelW - padR;
    const plotH = h - padT - padB;

    // Graticule — horizontal IRE lines
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.font = `${9 * dpr}px monospace`;

    for (const ire of IRE_MARKS) {
      const y = padT + (1 - ire / 100) * plotH;

      // Line
      ctx.strokeStyle = 'rgba(255, 200, 50, 0.12)';
      ctx.lineWidth = dpr * 0.5;
      ctx.beginPath();
      ctx.moveTo(labelW, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();

      // Label
      ctx.fillStyle = 'rgba(255, 200, 50, 0.5)';
      ctx.fillText(`${ire}`, labelW - 4 * dpr, y);
    }

    // Draw waveform data
    if (data) {
      // Scale the waveform ImageData to fill the plot area
      const offscreen = new OffscreenCanvas(data.width, data.height);
      const offCtx = offscreen.getContext('2d')!;
      offCtx.putImageData(data, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, labelW, padT, plotW, plotH);
    }
  }, [data]);

  return (
    <div ref={containerRef} className="scope-canvas-container">
      <canvas ref={canvasRef} />
    </div>
  );
}
