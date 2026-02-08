// Pure analysis functions for video scopes
// No side effects — easy to move to a Web Worker later

export interface HistogramData {
  r: Uint32Array;
  g: Uint32Array;
  b: Uint32Array;
  luma: Uint32Array;
  max: number;
}

/**
 * Compute RGB + Luma histogram from raw RGBA pixel data.
 * @param pixels - Uint8ClampedArray in RGBA order
 * @param width - frame width (unused but kept for future per-row analysis)
 * @param height - frame height
 * @param step - pixel sampling stride (e.g. 4 = every 4th pixel → 16x faster)
 */
export function computeHistogram(
  pixels: Uint8ClampedArray,
  _width: number,
  height: number,
  step: number
): HistogramData {
  const r = new Uint32Array(256);
  const g = new Uint32Array(256);
  const b = new Uint32Array(256);
  const luma = new Uint32Array(256);

  const totalPixels = pixels.length / 4;
  const rowPixels = totalPixels / height;

  for (let y = 0; y < height; y += step) {
    const rowStart = y * rowPixels * 4;
    const rowEnd = rowStart + rowPixels * 4;
    for (let i = rowStart; i < rowEnd; i += step * 4) {
      const rv = pixels[i];
      const gv = pixels[i + 1];
      const bv = pixels[i + 2];
      r[rv]++;
      g[gv]++;
      b[bv]++;
      // BT.709 luma
      const l = (0.2126 * rv + 0.7152 * gv + 0.0722 * bv + 0.5) | 0;
      luma[l < 256 ? l : 255]++;
    }
  }

  // Find max bin value — skip bins 0 and 255 which often spike unrealistically
  let max = 0;
  for (let i = 1; i < 255; i++) {
    if (r[i] > max) max = r[i];
    if (g[i] > max) max = g[i];
    if (b[i] > max) max = b[i];
  }

  // Soft-cap: clamp outlier bins to 2x the interior max so spikes don't crush the rest
  const cap = max * 2;
  if (r[0] > cap) r[0] = cap;
  if (r[255] > cap) r[255] = cap;
  if (g[0] > cap) g[0] = cap;
  if (g[255] > cap) g[255] = cap;
  if (b[0] > cap) b[0] = cap;
  if (b[255] > cap) b[255] = cap;
  if (luma[0] > cap) luma[0] = cap;
  if (luma[255] > cap) luma[255] = cap;

  // Recalculate max including capped edge bins
  for (const ch of [r, g, b]) {
    if (ch[0] > max) max = ch[0];
    if (ch[255] > max) max = ch[255];
  }

  return { r, g, b, luma, max };
}

/**
 * Compute vectorscope as a pre-rendered ImageData.
 * Maps each pixel's chrominance (Cb, Cr) to a 2D plot.
 * Uses additive blending so dense clusters appear brighter.
 *
 * @param pixels - Uint8ClampedArray in RGBA order
 * @param width - frame width
 * @param height - frame height
 * @param step - pixel sampling stride
 * @returns ImageData of the vectorscope plot (size × size)
 */
export function computeVectorscope(
  pixels: Uint8ClampedArray,
  _width: number,
  height: number,
  step: number,
  size: number = 256
): ImageData {
  // Accumulator — 32-bit per channel for additive blending
  const acc = new Uint32Array(size * size * 3);

  const center = size / 2;
  const scale = center * 0.85; // leave margin for labels

  const totalPixels = pixels.length / 4;
  const rowPixels = totalPixels / height;

  for (let y = 0; y < height; y += step) {
    const rowStart = y * rowPixels * 4;
    const rowEnd = rowStart + rowPixels * 4;
    for (let i = rowStart; i < rowEnd; i += step * 4) {
      const rv = pixels[i];
      const gv = pixels[i + 1];
      const bv = pixels[i + 2];

      // BT.709: Y'CbCr
      // Cb = -0.1687*R - 0.3313*G + 0.5*B
      // Cr =  0.5*R    - 0.4187*G - 0.0813*B
      const cb = -0.1687 * rv - 0.3313 * gv + 0.5 * bv;
      const cr = 0.5 * rv - 0.4187 * gv - 0.0813 * bv;

      // Map to plot coordinates — Cb = horizontal, Cr = vertical (inverted)
      const px = (center + (cb / 128) * scale) | 0;
      const py = (center - (cr / 128) * scale) | 0;

      if (px >= 0 && px < size && py >= 0 && py < size) {
        const idx = (py * size + px) * 3;
        // Tint dots with the source color for a natural look
        acc[idx] += rv;
        acc[idx + 1] += gv;
        acc[idx + 2] += bv;
      }
    }
  }

  // Normalize accumulator to 0-255 range
  let maxVal = 1;
  for (let i = 0; i < acc.length; i++) {
    if (acc[i] > maxVal) maxVal = acc[i];
  }

  const imageData = new ImageData(size, size);
  const d = imageData.data;
  // Use a gamma curve so dim areas are still visible
  const invMax = 1 / maxVal;

  for (let i = 0; i < size * size; i++) {
    const si = i * 3;
    const di = i * 4;
    const rVal = acc[si];
    const gVal = acc[si + 1];
    const bVal = acc[si + 2];
    if (rVal | gVal | bVal) {
      const brightness = Math.max(rVal, gVal, bVal);
      const t = Math.pow(brightness * invMax, 0.4); // gamma
      const norm = t / (brightness * invMax);
      d[di] = Math.min(255, (rVal * invMax * norm * 255) | 0);
      d[di + 1] = Math.min(255, (gVal * invMax * norm * 255) | 0);
      d[di + 2] = Math.min(255, (bVal * invMax * norm * 255) | 0);
      d[di + 3] = 255;
    }
  }

  return imageData;
}
