// Render waveform for audio clips using canvas for better performance
// Supports trimming: only displays the portion of waveform between inPoint and outPoint

import { memo, useRef, useEffect } from 'react';

export const ClipWaveform = memo(function ClipWaveform({
  waveform,
  width,
  height,
  inPoint,
  outPoint,
  naturalDuration,
}: {
  waveform: number[];
  width: number;
  height: number;
  inPoint: number;
  outPoint: number;
  naturalDuration: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveform || waveform.length === 0 || width <= 0 || naturalDuration <= 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate which portion of the waveform to display based on trim points
    const startRatio = inPoint / naturalDuration;
    const endRatio = outPoint / naturalDuration;
    const startSample = Math.floor(startRatio * waveform.length);
    const endSample = Math.ceil(endRatio * waveform.length);

    // Extract the visible portion of the waveform
    const visibleWaveform = waveform.slice(startSample, endSample);
    if (visibleWaveform.length === 0) return;

    // Limit canvas size to browser maximum (16384 is safe for most browsers)
    const MAX_CANVAS_WIDTH = 16384;
    const canvasWidth = Math.min(width, MAX_CANVAS_WIDTH);

    // Set canvas size (account for device pixel ratio for sharpness)
    const dpr = window.devicePixelRatio || 1;
    // Also limit by dpr to avoid exceeding canvas limits
    const effectiveDpr = Math.min(dpr, MAX_CANVAS_WIDTH / canvasWidth);

    canvas.width = canvasWidth * effectiveDpr;
    canvas.height = height * effectiveDpr;
    ctx.scale(effectiveDpr, effectiveDpr);

    // Clear
    ctx.clearRect(0, 0, canvasWidth, height);

    // Determine number of bars to draw (max 2 per pixel for detail)
    const maxBars = Math.floor(canvasWidth * 2);
    const samplesPerBar = Math.max(1, Math.floor(visibleWaveform.length / maxBars));
    const numBars = Math.ceil(visibleWaveform.length / samplesPerBar);
    const barWidth = canvasWidth / numBars;

    ctx.fillStyle = 'rgba(200, 200, 200, 0.5)';

    // Draw bars, using peak value for each segment
    for (let i = 0; i < numBars; i++) {
      const startIdx = i * samplesPerBar;
      const endIdx = Math.min(startIdx + samplesPerBar, visibleWaveform.length);

      // Get peak value for this segment
      let peak = 0;
      for (let j = startIdx; j < endIdx; j++) {
        if (visibleWaveform[j] > peak) peak = visibleWaveform[j];
      }

      const barHeight = Math.max(2, peak * (height - 4));
      const x = i * barWidth;
      const y = (height - barHeight) / 2;
      ctx.fillRect(x, y, Math.max(1, barWidth - 0.5), barHeight);
    }
  }, [waveform, width, height, inPoint, outPoint, naturalDuration]);

  if (!waveform || waveform.length === 0) return null;

  return (
    <canvas
      ref={canvasRef}
      className="waveform-canvas"
      style={{ width, height }}
    />
  );
});
