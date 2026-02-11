// Download clip actions slice - extracted from clipSlice
// Handles YouTube pending download clips

import type { TimelineClip } from '../../types';
import type { DownloadClipActions, SliceCreator } from './types';
import { DEFAULT_TRANSFORM } from './constants';
import { generateYouTubeClipId } from './helpers/idGenerator';
import { updateClipById } from './helpers/clipStateHelpers';
import { completeDownload as completeDownloadImpl } from './clip/completeDownload';
import { Logger } from '../../services/logger';

const log = Logger.create('DownloadClipSlice');

export const createDownloadClipSlice: SliceCreator<DownloadClipActions> = (set, get) => ({
  addPendingDownloadClip: (trackId, startTime, videoId, title, thumbnail, estimatedDuration = 30) => {
    const { clips, tracks, updateDuration, findNonOverlappingPosition } = get();
    const track = tracks.find(t => t.id === trackId);
    if (!track || track.type !== 'video') {
      log.warn('Pending download clips can only be added to video tracks');
      return '';
    }

    const clipId = generateYouTubeClipId();
    const finalStartTime = findNonOverlappingPosition(clipId, startTime, trackId, estimatedDuration);

    const pendingClip: TimelineClip = {
      id: clipId,
      trackId,
      name: title,
      file: new File([], `${title}.mp4`, { type: 'video/mp4' }),
      startTime: finalStartTime,
      duration: estimatedDuration,
      inPoint: 0,
      outPoint: estimatedDuration,
      source: null,
      transform: { ...DEFAULT_TRANSFORM },
      effects: [],
      isLoading: false,
      isPendingDownload: true,
      downloadProgress: 0,
      youtubeVideoId: videoId,
      youtubeThumbnail: thumbnail,
    };

    set({ clips: [...clips, pendingClip] });
    updateDuration();
    log.debug('Added pending download clip', { clipId });
    return clipId;
  },

  updateDownloadProgress: (clipId, progress) => {
    set({ clips: updateClipById(get().clips, clipId, { downloadProgress: progress }) });
  },

  completeDownload: async (clipId, file) => {
    await completeDownloadImpl({
      clipId,
      file,
      clips: get().clips,
      waveformsEnabled: get().waveformsEnabled,
      findAvailableAudioTrack: get().findAvailableAudioTrack,
      updateDuration: get().updateDuration,
      invalidateCache: get().invalidateCache,
      set,
      get,
    });
  },

  setDownloadError: (clipId, error) => {
    set({ clips: updateClipById(get().clips, clipId, { downloadError: error, isPendingDownload: false }) });
  },
});
