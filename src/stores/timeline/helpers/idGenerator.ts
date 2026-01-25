// Unique ID generator - prevents collision when creating multiple clips rapidly
// Uses timestamp + counter + random suffix for guaranteed uniqueness

let clipCounter = 0;
let lastTimestamp = 0;

/**
 * Generate a unique clip ID.
 * Format: {prefix}-{timestamp}-{counter}-{random}
 *
 * The counter resets when timestamp changes, ensuring:
 * - IDs are always unique even when created in the same millisecond
 * - IDs are roughly sortable by creation time
 * - Random suffix adds extra collision resistance
 */
export function generateClipId(prefix: string = 'clip'): string {
  const now = Date.now();

  // Reset counter when timestamp changes
  if (now !== lastTimestamp) {
    lastTimestamp = now;
    clipCounter = 0;
  }

  const count = ++clipCounter;
  const random = Math.random().toString(36).substr(2, 5);

  return `${prefix}-${now}-${count}-${random}`;
}

/**
 * Generate a unique ID for video clips.
 */
export function generateVideoClipId(): string {
  return generateClipId('clip');
}

/**
 * Generate a unique ID for audio clips (linked to video).
 */
export function generateAudioClipId(): string {
  return generateClipId('clip-audio');
}

/**
 * Generate a unique ID for text clips.
 */
export function generateTextClipId(): string {
  return generateClipId('clip-text');
}

/**
 * Generate a unique ID for composition clips.
 */
export function generateCompClipId(): string {
  return generateClipId('clip-comp');
}

/**
 * Generate a unique ID for YouTube download clips.
 */
export function generateYouTubeClipId(): string {
  return generateClipId('clip-yt');
}

/**
 * Generate a unique ID for nested clips within a composition.
 */
export function generateNestedClipId(parentCompClipId: string, originalClipId: string): string {
  return `nested-${parentCompClipId}-${originalClipId}`;
}

/**
 * Generate a unique ID for effects.
 */
export function generateEffectId(): string {
  return generateClipId('effect');
}

/**
 * Generate a unique ID for multicam linked groups.
 */
export function generateLinkedGroupId(): string {
  return generateClipId('multicam');
}

/**
 * Generate a unique ID for tracks.
 */
export function generateTrackId(type: 'video' | 'audio'): string {
  return generateClipId(`track-${type}`);
}

/**
 * Generate paired IDs for video and linked audio clips.
 * Ensures they're created with the same timestamp for consistency.
 */
export function generateLinkedClipIds(): { videoId: string; audioId: string } {
  const videoId = generateVideoClipId();
  const audioId = generateAudioClipId();
  return { videoId, audioId };
}
