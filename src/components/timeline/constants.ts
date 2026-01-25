// Timeline constants

import type { BlendMode } from '../../types';

// All blend modes in order for cycling with Shift+Plus/Minus or Numpad +/-
export const ALL_BLEND_MODES: BlendMode[] = [
  // Normal
  'normal', 'dissolve', 'dancing-dissolve',
  // Darken
  'darken', 'multiply', 'color-burn', 'classic-color-burn', 'linear-burn', 'darker-color',
  // Lighten
  'add', 'lighten', 'screen', 'color-dodge', 'classic-color-dodge', 'linear-dodge', 'lighter-color',
  // Contrast
  'overlay', 'soft-light', 'hard-light', 'linear-light', 'vivid-light', 'pin-light', 'hard-mix',
  // Inversion
  'difference', 'classic-difference', 'exclusion', 'subtract', 'divide',
  // Component
  'hue', 'saturation', 'color', 'luminosity',
  // Stencil
  'stencil-alpha', 'stencil-luma', 'silhouette-alpha', 'silhouette-luma', 'alpha-add',
];

// Snap threshold in pixels
export const SNAP_THRESHOLD_PX = 10;

// Default clip preview duration (when dragging external files)
export const DEFAULT_PREVIEW_DURATION = 5;

// Thumbnail filmstrip settings
export const THUMB_WIDTH = 71;

// Playhead update interval (ms) - ~15fps for UI updates
export const PLAYHEAD_UPDATE_INTERVAL = 66;

// Seek throttle interval (ms) for scrubbing
export const SEEK_THROTTLE_MS = 80;

// Auto RAM preview delay (ms)
export const RAM_PREVIEW_IDLE_DELAY = 2000;

// Auto proxy generation delay (ms)
export const PROXY_IDLE_DELAY = 3000;

// Quick duration check timeout (ms)
export const DURATION_CHECK_TIMEOUT = 500;
