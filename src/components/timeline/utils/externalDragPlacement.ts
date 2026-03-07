import type { TimelineClip, TimelineTrack } from '../../../types';

type PlacementClip = Pick<TimelineClip, 'trackId' | 'startTime' | 'duration'>;
type PlacementTrack = Pick<TimelineTrack, 'id' | 'type'>;

function hasOverlap(
  trackId: string,
  startTime: number,
  duration: number,
  clips: PlacementClip[]
): boolean {
  const endTime = startTime + duration;
  return clips.some((clip) => {
    if (clip.trackId !== trackId) return false;
    const clipEnd = clip.startTime + clip.duration;
    return !(endTime <= clip.startTime || startTime >= clipEnd);
  });
}

export function canPlaceOnTrack(
  trackId: string,
  startTime: number,
  duration: number,
  clips: PlacementClip[]
): boolean {
  const safeStartTime = Math.max(0, startTime);
  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;

  if (safeDuration === 0) return true;
  return !hasOverlap(trackId, safeStartTime, safeDuration, clips);
}

export function findClosestNonOverlappingStartTime(
  trackId: string,
  desiredStartTime: number,
  duration: number,
  clips: PlacementClip[]
): number {
  const safeStartTime = Math.max(0, desiredStartTime);
  const safeDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;

  if (safeDuration === 0 || canPlaceOnTrack(trackId, safeStartTime, safeDuration, clips)) {
    return safeStartTime;
  }

  const trackClips = clips
    .filter((clip) => clip.trackId === trackId)
    .sort((a, b) => a.startTime - b.startTime);

  const candidates = new Set<number>([0]);
  for (const clip of trackClips) {
    candidates.add(Math.max(0, clip.startTime - safeDuration));
    candidates.add(clip.startTime + clip.duration);
  }

  let bestStartTime: number | null = null;
  let bestDistance = Infinity;

  for (const candidate of candidates) {
    if (!canPlaceOnTrack(trackId, candidate, safeDuration, clips)) continue;

    const distance = Math.abs(candidate - safeStartTime);
    if (
      distance < bestDistance ||
      (distance === bestDistance && (bestStartTime === null || candidate < bestStartTime))
    ) {
      bestStartTime = candidate;
      bestDistance = distance;
    }
  }

  if (bestStartTime !== null) {
    return bestStartTime;
  }

  const lastClip = trackClips[trackClips.length - 1];
  return lastClip ? lastClip.startTime + lastClip.duration : safeStartTime;
}

export function findFirstTrackWithoutOverlap(
  trackType: PlacementTrack['type'],
  startTime: number,
  duration: number,
  tracks: PlacementTrack[],
  clips: PlacementClip[]
): string | null {
  for (const track of tracks) {
    if (track.type !== trackType) continue;
    if (canPlaceOnTrack(track.id, startTime, duration, clips)) {
      return track.id;
    }
  }

  return null;
}
