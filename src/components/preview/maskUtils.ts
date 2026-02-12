// Mask utility functions: throttle, path generation, coordinate conversion

import type { MaskVertex } from '../../types';

// Throttle helper - limits function calls to once per interval
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function throttle<T extends (...args: any[]) => void>(fn: T, interval: number): T {
  let lastCall = 0;
  let pendingArgs: Parameters<T> | null = null;
  let rafId: number | null = null;

  const throttled = (...args: Parameters<T>) => {
    const now = performance.now();
    if (now - lastCall >= interval) {
      lastCall = now;
      fn(...args);
    } else {
      pendingArgs = args;
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = null;
          if (pendingArgs) {
            lastCall = performance.now();
            fn(...pendingArgs);
            pendingArgs = null;
          }
        });
      }
    }
  };

  return throttled as T;
}

// Generate SVG path data from mask vertices using cubic bezier curves
export function generatePathData(
  vertices: MaskVertex[],
  closed: boolean,
  positionX: number = 0,
  positionY: number = 0,
  canvasWidth: number = 1920,
  canvasHeight: number = 1080
): string {
  if (vertices.length < 2) return '';

  let d = '';

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];
    const vx = (v.x + positionX) * canvasWidth;
    const vy = (v.y + positionY) * canvasHeight;

    if (i === 0) {
      d += `M ${vx} ${vy}`;
    } else {
      const prev = vertices[i - 1];
      const prevX = (prev.x + positionX) * canvasWidth;
      const prevY = (prev.y + positionY) * canvasHeight;

      const cp1x = prevX + prev.handleOut.x * canvasWidth;
      const cp1y = prevY + prev.handleOut.y * canvasHeight;
      const cp2x = vx + v.handleIn.x * canvasWidth;
      const cp2y = vy + v.handleIn.y * canvasHeight;
      d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${vx},${vy}`;
    }
  }

  if (closed && vertices.length > 2) {
    const last = vertices[vertices.length - 1];
    const first = vertices[0];
    const lastX = (last.x + positionX) * canvasWidth;
    const lastY = (last.y + positionY) * canvasHeight;
    const firstX = (first.x + positionX) * canvasWidth;
    const firstY = (first.y + positionY) * canvasHeight;

    const cp1x = lastX + last.handleOut.x * canvasWidth;
    const cp1y = lastY + last.handleOut.y * canvasHeight;
    const cp2x = firstX + first.handleIn.x * canvasWidth;
    const cp2y = firstY + first.handleIn.y * canvasHeight;
    d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${firstX},${firstY} Z`;
  }

  return d;
}

// Convert normalized (0-1) coordinates to canvas coordinates
export function normalizedToCanvas(
  x: number,
  y: number,
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number } {
  return {
    x: x * canvasWidth,
    y: y * canvasHeight,
  };
}
