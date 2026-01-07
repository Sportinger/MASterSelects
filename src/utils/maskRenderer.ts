// Mask Renderer - Generates mask textures from ClipMask data using Canvas2D

import type { ClipMask, MaskVertex } from '../types';

// Canvas for rendering masks (reused for performance)
let maskCanvas: OffscreenCanvas | null = null;
let maskCtx: OffscreenCanvasRenderingContext2D | null = null;

// Ensure canvas exists at given size
function ensureMaskCanvas(width: number, height: number): OffscreenCanvasRenderingContext2D {
  if (!maskCanvas || maskCanvas.width !== width || maskCanvas.height !== height) {
    maskCanvas = new OffscreenCanvas(width, height);
    maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (!maskCtx) {
    throw new Error('Failed to get 2D context for mask canvas');
  }
  return maskCtx;
}

// Draw a single bezier path for a mask
function drawMaskPath(
  ctx: OffscreenCanvasRenderingContext2D,
  vertices: MaskVertex[],
  closed: boolean,
  width: number,
  height: number,
  offsetX: number = 0,
  offsetY: number = 0
): void {
  if (vertices.length < 2) return;

  ctx.beginPath();

  for (let i = 0; i < vertices.length; i++) {
    const v = vertices[i];
    const x = (v.x + offsetX) * width;
    const y = (v.y + offsetY) * height;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      const prev = vertices[i - 1];
      const prevX = (prev.x + offsetX) * width;
      const prevY = (prev.y + offsetY) * height;

      // Control points for cubic bezier
      const cp1x = prevX + prev.handleOut.x * width;
      const cp1y = prevY + prev.handleOut.y * height;
      const cp2x = x + v.handleIn.x * width;
      const cp2y = y + v.handleIn.y * height;

      // Check if handles are at origin (straight line)
      const isStraight =
        prev.handleOut.x === 0 && prev.handleOut.y === 0 &&
        v.handleIn.x === 0 && v.handleIn.y === 0;

      if (isStraight) {
        ctx.lineTo(x, y);
      } else {
        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y);
      }
    }
  }

  // Close path if needed
  if (closed && vertices.length > 2) {
    const last = vertices[vertices.length - 1];
    const first = vertices[0];

    const lastX = (last.x + offsetX) * width;
    const lastY = (last.y + offsetY) * height;
    const firstX = (first.x + offsetX) * width;
    const firstY = (first.y + offsetY) * height;

    const cp1x = lastX + last.handleOut.x * width;
    const cp1y = lastY + last.handleOut.y * height;
    const cp2x = firstX + first.handleIn.x * width;
    const cp2y = firstY + first.handleIn.y * height;

    const isStraight =
      last.handleOut.x === 0 && last.handleOut.y === 0 &&
      first.handleIn.x === 0 && first.handleIn.y === 0;

    if (isStraight) {
      ctx.lineTo(firstX, firstY);
    } else {
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, firstX, firstY);
    }
    ctx.closePath();
  }
}

// Generate a mask texture from an array of ClipMask
export function generateMaskTexture(
  masks: ClipMask[],
  width: number,
  height: number
): ImageData | null {
  if (!masks || masks.length === 0) return null;

  const ctx = ensureMaskCanvas(width, height);

  // Start with black (no visibility)
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  // Process each mask
  for (const mask of masks) {
    if (mask.vertices.length < 3 || !mask.closed) continue;

    ctx.save();

    // Set composite operation based on mask mode
    switch (mask.mode) {
      case 'add':
        // Add to existing visibility
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillStyle = `rgba(255, 255, 255, ${mask.opacity})`;
        break;
      case 'subtract':
        // Remove from existing visibility (use destination-out with white alpha)
        ctx.globalCompositeOperation = 'destination-out';
        ctx.fillStyle = `rgba(255, 255, 255, ${mask.opacity})`;
        break;
      case 'intersect':
        // Keep only intersecting areas
        ctx.globalCompositeOperation = 'destination-in';
        ctx.fillStyle = `rgba(255, 255, 255, ${mask.opacity})`;
        break;
    }

    // Draw the mask path with position offset
    drawMaskPath(ctx, mask.vertices, mask.closed, width, height, mask.position.x, mask.position.y);
    ctx.fill();

    ctx.restore();
  }

  // Apply feather (blur) if any mask has feather > 0
  const maxFeather = Math.max(...masks.map(m => m.feather));
  if (maxFeather > 0) {
    // Get current content
    const imageData = ctx.getImageData(0, 0, width, height);

    // Apply gaussian blur approximation using box blur
    const blurRadius = Math.min(maxFeather, 50); // Cap at 50px for performance
    if (blurRadius > 0) {
      ctx.filter = `blur(${blurRadius}px)`;
      ctx.clearRect(0, 0, width, height);

      // Create temporary canvas for blur
      const tempCanvas = new OffscreenCanvas(width, height);
      const tempCtx = tempCanvas.getContext('2d')!;
      tempCtx.putImageData(imageData, 0, 0);

      ctx.drawImage(tempCanvas, 0, 0);
      ctx.filter = 'none';
    }
  }

  // Handle inversion for any inverted masks
  const hasInverted = masks.some(m => m.inverted);
  if (hasInverted) {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      // Invert RGB (keep alpha at 255)
      data[i] = 255 - data[i];
      data[i + 1] = 255 - data[i + 1];
      data[i + 2] = 255 - data[i + 2];
    }

    ctx.putImageData(imageData, 0, 0);
  }

  return ctx.getImageData(0, 0, width, height);
}

// Generate mask texture for a single mask (simpler API for common case)
export function generateSingleMaskTexture(
  mask: ClipMask,
  width: number,
  height: number
): ImageData | null {
  if (!mask || mask.vertices.length < 3 || !mask.closed) return null;

  const ctx = ensureMaskCanvas(width, height);

  // Start with black (mask area = white, outside = black)
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  // Draw mask as white
  ctx.fillStyle = `rgba(255, 255, 255, ${mask.opacity})`;
  drawMaskPath(ctx, mask.vertices, mask.closed, width, height);
  ctx.fill();

  // Apply feather
  if (mask.feather > 0) {
    const imageData = ctx.getImageData(0, 0, width, height);
    const blurRadius = Math.min(mask.feather, 50);

    ctx.filter = `blur(${blurRadius}px)`;
    ctx.clearRect(0, 0, width, height);

    const tempCanvas = new OffscreenCanvas(width, height);
    const tempCtx = tempCanvas.getContext('2d')!;
    tempCtx.putImageData(imageData, 0, 0);

    ctx.drawImage(tempCanvas, 0, 0);
    ctx.filter = 'none';
  }

  // Handle inversion
  if (mask.inverted) {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let i = 0; i < data.length; i += 4) {
      data[i] = 255 - data[i];
      data[i + 1] = 255 - data[i + 1];
      data[i + 2] = 255 - data[i + 2];
    }

    ctx.putImageData(imageData, 0, 0);
  }

  return ctx.getImageData(0, 0, width, height);
}

// Convert ImageData to a format suitable for WebGPU texture
export function imageDataToUint8Array(imageData: ImageData): Uint8Array {
  return new Uint8Array(imageData.data.buffer);
}
