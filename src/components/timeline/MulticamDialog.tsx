// MulticamDialog - Dialog for selecting master audio track and syncing clips
// Opens when user selects multiple clips and clicks "Combine Multicam"

import { useState, useMemo, useCallback, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { TimelineClip } from '../../types';

interface MulticamDialogProps {
  open: boolean;
  onClose: () => void;
  selectedClipIds: Set<string>;
}

export function MulticamDialog({ open, onClose, selectedClipIds }: MulticamDialogProps) {
  const { clips, createLinkedGroup } = useTimelineStore();

  const [masterClipId, setMasterClipId] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Get selected clips
  const selectedClips = useMemo(() => {
    return clips.filter(c => selectedClipIds.has(c.id));
  }, [clips, selectedClipIds]);

  // Categorize clips: with audio vs without
  const { clipsWithAudio, clipsWithoutAudio } = useMemo(() => {
    const withAudio: TimelineClip[] = [];
    const withoutAudio: TimelineClip[] = [];

    for (const clip of selectedClips) {
      // Clips have audio if they are audio type or have waveform data
      const hasAudioData =
        clip.source?.type === 'audio' ||
        (clip.waveform && clip.waveform.length > 0);

      if (hasAudioData) {
        withAudio.push(clip);
      } else {
        withoutAudio.push(clip);
      }
    }

    return { clipsWithAudio: withAudio, clipsWithoutAudio: withoutAudio };
  }, [selectedClips]);

  // Auto-select first clip with audio as master
  useEffect(() => {
    if (open && clipsWithAudio.length > 0 && !masterClipId) {
      setMasterClipId(clipsWithAudio[0].id);
    }
  }, [open, clipsWithAudio, masterClipId]);

  // Reset state when dialog opens
  useEffect(() => {
    if (open) {
      setError(null);
      setProgress(0);
      setSyncing(false);
      setMasterClipId(clipsWithAudio[0]?.id || null);
    }
  }, [open, clipsWithAudio]);

  // Handle sync and link
  const handleSyncAndLink = useCallback(async () => {
    if (!masterClipId || clipsWithAudio.length < 2) {
      setError('Need at least 2 clips with audio to sync');
      return;
    }

    setSyncing(true);
    setError(null);
    setProgress(0);

    try {
      // Import audioSync service (singleton instance)
      const { audioSync } = await import('../../services/audioSync');

      // Get media file IDs for clips with audio
      const masterClip = clipsWithAudio.find(c => c.id === masterClipId);
      if (!masterClip?.source?.mediaFileId) {
        setError('Master clip has no media file reference');
        setSyncing(false);
        return;
      }

      const masterMediaFileId = masterClip.source.mediaFileId;
      const targetMediaFileIds = clipsWithAudio
        .filter(c => c.id !== masterClipId && c.source?.mediaFileId)
        .map(c => c.source!.mediaFileId!);

      if (targetMediaFileIds.length === 0) {
        setError('No other clips with media file references found');
        setSyncing(false);
        return;
      }

      // Progress callback
      const onProgress = (p: number) => {
        setProgress(p);
      };

      // Run sync - returns Map<mediaFileId, offsetInSeconds>
      const mediaFileOffsets = await audioSync.syncMultiple(
        masterMediaFileId,
        targetMediaFileIds,
        onProgress
      );

      // Convert mediaFileId offsets to clipId offsets
      const clipOffsets = new Map<string, number>();

      // Master clip has 0 offset
      clipOffsets.set(masterClipId, 0);

      // Map media file offsets to clip offsets
      for (const clip of clipsWithAudio) {
        if (clip.id === masterClipId) continue;

        const mediaFileId = clip.source?.mediaFileId;
        if (mediaFileId && mediaFileOffsets.has(mediaFileId)) {
          const offsetSeconds = mediaFileOffsets.get(mediaFileId)!;
          clipOffsets.set(clip.id, offsetSeconds);
        }
      }

      // Add clips without audio with their current relative position to master
      for (const clip of clipsWithoutAudio) {
        const currentOffset = clip.startTime - masterClip.startTime;
        clipOffsets.set(clip.id, currentOffset);
      }

      // Create linked group with all clips
      const allClipIds = [...clipsWithAudio.map(c => c.id), ...clipsWithoutAudio.map(c => c.id)];
      createLinkedGroup(allClipIds, clipOffsets);

      setProgress(100);
      setSyncing(false);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error during sync');
      setSyncing(false);
    }
  }, [masterClipId, clipsWithAudio, clipsWithoutAudio, createLinkedGroup, onClose]);

  // Handle close
  const handleClose = useCallback(() => {
    if (!syncing) {
      onClose();
    }
  }, [syncing, onClose]);

  if (!open) return null;

  return (
    <div className="multicam-dialog-overlay" onClick={handleClose}>
      <div className="multicam-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="multicam-dialog-header">
          <h2>Combine Multicam</h2>
          <button className="dialog-close-btn" onClick={handleClose} disabled={syncing}>
            ×
          </button>
        </div>

        <div className="multicam-dialog-content">
          {/* Master selection */}
          <div className="multicam-section">
            <h3>Select Master Audio Track</h3>
            <p className="section-hint">
              Other clips will be synchronized to this master track's audio.
            </p>

            {clipsWithAudio.length === 0 ? (
              <div className="multicam-warning">
                No clips with audio selected. At least 2 clips with audio are needed for sync.
              </div>
            ) : (
              <div className="multicam-clip-list">
                {clipsWithAudio.map((clip) => (
                  <label key={clip.id} className="multicam-clip-item">
                    <input
                      type="radio"
                      name="masterClip"
                      value={clip.id}
                      checked={masterClipId === clip.id}
                      onChange={() => setMasterClipId(clip.id)}
                      disabled={syncing}
                    />
                    <span className="clip-icon">
                      {clip.source?.type === 'audio' ? '\uD83D\uDD0A' : '\uD83C\uDFAC'}
                    </span>
                    <span className="clip-name" title={clip.name}>
                      {clip.name}
                    </span>
                    {masterClipId === clip.id && (
                      <span className="master-badge">Master</span>
                    )}
                  </label>
                ))}
              </div>
            )}
          </div>

          {/* Clips without audio warning */}
          {clipsWithoutAudio.length > 0 && (
            <div className="multicam-section">
              <h3>Clips Without Audio</h3>
              <p className="section-hint warning">
                These clips will be linked but not synchronized (no audio detected).
              </p>
              <div className="multicam-clip-list no-audio">
                {clipsWithoutAudio.map((clip) => (
                  <div key={clip.id} className="multicam-clip-item disabled">
                    <span className="clip-icon">{'\uD83C\uDFAC'}</span>
                    <span className="clip-name" title={clip.name}>
                      {clip.name}
                    </span>
                    <span className="no-audio-badge">No Audio</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Progress bar */}
          {syncing && (
            <div className="multicam-progress-section">
              <div className="multicam-progress">
                <div
                  className="multicam-progress-bar"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className="progress-label">Synchronizing... {progress}%</span>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="multicam-error">
              {error}
            </div>
          )}
        </div>

        <div className="multicam-dialog-footer">
          <button
            className="btn-cancel"
            onClick={handleClose}
            disabled={syncing}
          >
            Cancel
          </button>
          <button
            className="btn-sync"
            onClick={handleSyncAndLink}
            disabled={syncing || clipsWithAudio.length < 2 || !masterClipId}
          >
            {syncing ? 'Syncing...' : 'Sync & Link'}
          </button>
        </div>
      </div>
    </div>
  );
}
