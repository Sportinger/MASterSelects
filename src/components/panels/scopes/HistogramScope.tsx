import { useRef, useEffect } from 'react';
import type { HistogramData } from '../../../engine/analysis/ScopeAnalyzer';

interface HistogramScopeProps {
  data: HistogramData | null;
}

// 3-tap moving average for smoother curves
function smooth(arr: Uint32Array): Float32Array {
  const out = new Float32Array(256);
  out[0] = arr[0];
  out[255] = arr[255];
  for (let i = 1; i < 255; i++) {
    out[i] = (arr[i - 1] + arr[i] * 2 + arr[i + 1]) / 4;
  }
  return out;
}

export function HistogramScope({ data }: HistogramScopeProps) {
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

  // Draw histogram
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, w, h);

    const padL = 2 * dpr;
    const padR = 2 * dpr;
    const padT = 4 * dpr;
    const padB = 4 * dpr;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Graticule — vertical lines at shadows, midtones, highlights
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = dpr;
    for (const mark of [64, 128, 192]) {
      const x = padL + (mark / 255) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
    }

    // Horizontal guide lines
    for (const frac of [0.25, 0.5, 0.75]) {
      const y = h - padB - frac * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
    }

    if (data.max === 0) return;

    // Sqrt-scaled max for non-linear normalization
    const sqrtMax = Math.sqrt(data.max);

    // Smooth channels
    const smoothR = smooth(data.r);
    const smoothG = smooth(data.g);
    const smoothB = smooth(data.b);
    const smoothL = smooth(data.luma);

    // Draw order: luma (behind), then R, G, B with blending
    const channels: { arr: Float32Array; fill: string; stroke: string }[] = [
      { arr: smoothL, fill: 'rgba(255, 255, 255, 0.08)', stroke: 'rgba(255, 255, 255, 0.2)' },
      { arr: smoothR, fill: 'rgba(220, 50, 50, 0.35)', stroke: 'rgba(255, 80, 80, 0.7)' },
      { arr: smoothG, fill: 'rgba(50, 200, 50, 0.35)', stroke: 'rgba(80, 255, 80, 0.7)' },
      { arr: smoothB, fill: 'rgba(50, 80, 220, 0.35)', stroke: 'rgba(80, 120, 255, 0.7)' },
    ];

    for (const { arr, fill, stroke } of channels) {
      ctx.beginPath();
      ctx.moveTo(padL, h - padB);

      for (let i = 0; i < 256; i++) {
        const x = padL + (i / 255) * plotW;
        // Sqrt scaling: spreads out the lower values, compresses peaks
        const normalized = Math.sqrt(arr[i]) / sqrtMax;
        const y = h - padB - Math.min(normalized, 1) * plotH;
        ctx.lineTo(x, y);
      }

      ctx.lineTo(w - padR, h - padB);
      ctx.closePath();

      // Fill
      ctx.fillStyle = fill;
      ctx.fill();

      // Stroke on top for definition
      ctx.strokeStyle = stroke;
      ctx.lineWidth = dpr * 0.75;
      ctx.stroke();
    }
  }, [data]);

  return (
    <div ref={containerRef} className="scope-canvas-container">
      <canvas ref={canvasRef} />
    </div>
  );
}
