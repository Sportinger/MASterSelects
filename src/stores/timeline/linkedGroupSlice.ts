// Linked group actions slice - extracted from clipSlice

import type { LinkedGroupActions, SliceCreator } from './types';
import { generateLinkedGroupId } from './helpers/idGenerator';
import { Logger } from '../../services/logger';

const log = Logger.create('LinkedGroupSlice');

export const createLinkedGroupSlice: SliceCreator<LinkedGroupActions> = (set, get) => ({
  createLinkedGroup: (clipIds, offsets) => {
    const { clips, invalidateCache } = get();
    const groupId = generateLinkedGroupId();
    const selectedClips = clips.filter(c => clipIds.includes(c.id));
    if (selectedClips.length === 0) return;

    let masterStartTime = selectedClips[0].startTime;
    for (const clipId of clipIds) {
      if (offsets.get(clipId) === 0) {
        const masterClip = clips.find(c => c.id === clipId);
        if (masterClip) { masterStartTime = masterClip.startTime; break; }
      }
    }

    set({
      clips: clips.map(c => {
        if (!clipIds.includes(c.id)) return c;
        const offset = offsets.get(c.id) || 0;
        return { ...c, linkedGroupId: groupId, startTime: Math.max(0, masterStartTime - offset / 1000) };
      }),
    });
    invalidateCache();
    log.debug('Created linked group', { groupId, clipCount: clipIds.length });
  },

  unlinkGroup: (clipId) => {
    const { clips, invalidateCache } = get();
    const clip = clips.find(c => c.id === clipId);
    if (!clip?.linkedGroupId) return;

    set({ clips: clips.map(c => c.linkedGroupId === clip.linkedGroupId ? { ...c, linkedGroupId: undefined } : c) });
    invalidateCache();
    log.debug('Unlinked group', { groupId: clip.linkedGroupId });
  },
});
