// MulticamDialog - Dialog for selecting master audio track and syncing clips
// Opens when user selects multiple clips and clicks "Combine Multicam"

import { useState, useMemo, useCallback, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { TimelineClip, TranscriptWord } from '../../types';
import { Logger } from '../../services/logger';

const log = Logger.create('MulticamDialog');

type SyncMethod = 'audio' | 'transcript';

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
  const [syncMethod, setSyncMethod] = useState<SyncMethod>('audio');

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

  // Categorize clips: with transcript vs without
  const { clipsWithTranscript, clipsWithoutTranscript } = useMemo(() => {
    const withTranscript: TimelineClip[] = [];
    const withoutTranscript: TimelineClip[] = [];

    for (const clip of selectedClips) {
      if (clip.transcript && clip.transcript.length > 0) {
        withTranscript.push(clip);
      } else {
        withoutTranscript.push(clip);
      }
    }

    return { clipsWithTranscript: withTranscript, clipsWithoutTranscript: withoutTranscript };
  }, [selectedClips]);

  // Check if transcript sync is available (need at least 2 clips with transcripts)
  const transcriptSyncAvailable = clipsWithTranscript.length >= 2;

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
      // Default to transcript sync if available and no audio, otherwise audio
      setSyncMethod(transcriptSyncAvailable && clipsWithAudio.length < 2 ? 'transcript' : 'audio');
    }
  }, [open, clipsWithAudio, transcriptSyncAvailable]);

  // Update masterClipId when sync method changes
  useEffect(() => {
    if (syncMethod === 'audio') {
      // Select first audio clip if current master isn't in audio list
      if (!clipsWithAudio.find(c => c.id === masterClipId)) {
        setMasterClipId(clipsWithAudio[0]?.id || null);
      }
    } else {
      // Select first transcript clip if current master isn't in transcript list
      if (!clipsWithTranscript.find(c => c.id === masterClipId)) {
        setMasterClipId(clipsWithTranscript[0]?.id || null);
      }
    }
  }, [syncMethod, clipsWithAudio, clipsWithTranscript, masterClipId]);

  // Handle sync and link
  const handleSyncAndLink = useCallback(async () => {
    // Validate based on sync method
    if (syncMethod === 'audio') {
      if (!masterClipId || clipsWithAudio.length < 2) {
        setError('Need at least 2 clips with audio to sync');
        return;
      }
    } else {
      if (!masterClipId || clipsWithTranscript.length < 2) {
        setError('Need at least 2 clips with transcripts to sync');
        return;
      }
    }

    setSyncing(true);
    setError(null);
    setProgress(0);

    try {
      let clipOffsets: Map<string, number>;

      if (syncMethod === 'transcript') {
        // Transcript-based sync
        const { syncClipsByTranscript } = await import('../../services/transcriptSync');

        // Get master clip
        const masterClip = clipsWithTranscript.find(c => c.id === masterClipId);
        if (!masterClip?.transcript) {
          setError('Master clip has no transcript');
          setSyncing(false);
          return;
        }

        setProgress(20);

        // Build target clips with transcripts
        const targetClips = clipsWithTranscript
          .filter(c => c.id !== masterClipId && c.transcript && c.transcript.length > 0)
          .map(c => ({
            clipId: c.id,
            transcript: c.transcript as TranscriptWord[],
          }));

        if (targetClips.length === 0) {
          setError('No other clips with transcripts found');
          setSyncing(false);
          return;
        }

        setProgress(40);

        // Run transcript sync
        const syncResults = syncClipsByTranscript(
          masterClipId,
          masterClip.transcript,
          targetClips
        );

        setProgress(80);

        // Convert to simple offset map
        clipOffsets = new Map<string, number>();
        for (const [clipId, result] of syncResults) {
          clipOffsets.set(clipId, result.offsetMs);
          if (result.confidence < 0.3 && clipId !== masterClipId) {
            log.warn(`Low confidence sync for ${clipId}: ${(result.confidence * 100).toFixed(1)}%`);
          }
        }

        // Add clips without transcript with their current relative position to master
        for (const clip of clipsWithoutTranscript) {
          const currentOffsetMs = (clip.startTime - masterClip.startTime) * 1000;
          clipOffsets.set(clip.id, currentOffsetMs);
        }

      } else {
        // Audio-based sync
        const { audioSync } = await import('../../services/audioSync');

        // Get master clip info
        const masterClip = clipsWithAudio.find(c => c.id === masterClipId);
        if (!masterClip?.source?.mediaFileId) {
          setError('Master clip has no media file reference');
          setSyncing(false);
          return;
        }

        // Build clip sync info with inPoint and duration for accurate sync
        const masterSyncInfo = {
          mediaFileId: masterClip.source.mediaFileId,
          clipId: masterClip.id,
          inPoint: masterClip.inPoint,
          duration: masterClip.duration,
        };

        const targetSyncInfos = clipsWithAudio
          .filter(c => c.id !== masterClipId && c.source?.mediaFileId)
          .map(c => ({
            mediaFileId: c.source!.mediaFileId!,
            clipId: c.id,
            inPoint: c.inPoint,
            duration: c.duration,
          }));

        if (targetSyncInfos.length === 0) {
          setError('No other clips with media file references found');
          setSyncing(false);
          return;
        }

        // Progress callback
        const onProgress = (p: number) => {
          setProgress(p);
        };

        // Run sync with clip bounds - returns Map<clipId, offsetInMs>
        clipOffsets = await audioSync.syncMultipleClips(
          masterSyncInfo,
          targetSyncInfos,
          onProgress
        );

        // Add clips without audio with their current relative position to master
        for (const clip of clipsWithoutAudio) {
          const currentOffsetMs = (clip.startTime - masterClip.startTime) * 1000;
          clipOffsets.set(clip.id, currentOffsetMs);
        }
      }

      // Create linked group with all clips
      const allClipIds = selectedClips.map(c => c.id);
      createLinkedGroup(allClipIds, clipOffsets);

      setProgress(100);
      setSyncing(false);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error during sync');
      setSyncing(false);
    }
  }, [masterClipId, syncMethod, clipsWithAudio, clipsWithTranscript, clipsWithoutAudio, clipsWithoutTranscript, selectedClips, createLinkedGroup, onClose]);

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
          {/* Sync method selection */}
          <div className="multicam-section">
            <h3>Sync Method</h3>
            <div className="multicam-method-selector">
              <label className={`method-option ${syncMethod === 'audio' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="syncMethod"
                  value="audio"
                  checked={syncMethod === 'audio'}
                  onChange={() => setSyncMethod('audio')}
                  disabled={syncing || clipsWithAudio.length < 2}
                />
                <span className="method-icon">{'\uD83D\uDD0A'}</span>
                <span className="method-label">Audio Waveform</span>
                <span className="method-count">
                  {clipsWithAudio.length >= 2 ? `${clipsWithAudio.length} clips` : 'Need 2+ clips'}
                </span>
              </label>
              <label className={`method-option ${syncMethod === 'transcript' ? 'selected' : ''}`}>
                <input
                  type="radio"
                  name="syncMethod"
                  value="transcript"
                  checked={syncMethod === 'transcript'}
                  onChange={() => setSyncMethod('transcript')}
                  disabled={syncing || !transcriptSyncAvailable}
                />
                <span className="method-icon">{'\uD83D\uDCDD'}</span>
                <span className="method-label">Transcript Text</span>
                <span className="method-count">
                  {transcriptSyncAvailable ? `${clipsWithTranscript.length} clips` : 'Need 2+ transcribed'}
                </span>
              </label>
            </div>
          </div>

          {/* Master selection */}
          <div className="multicam-section">
            <h3>Select Master {syncMethod === 'audio' ? 'Audio' : 'Transcript'} Track</h3>
            <p className="section-hint">
              {syncMethod === 'audio'
                ? "Other clips will be synchronized to this master track's audio."
                : "Other clips will be aligned to this master track's transcript."}
            </p>

            {syncMethod === 'audio' ? (
              clipsWithAudio.length === 0 ? (
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
              )
            ) : (
              clipsWithTranscript.length === 0 ? (
                <div className="multicam-warning">
                  No clips with transcripts selected. Transcribe clips first using the context menu.
                </div>
              ) : (
                <div className="multicam-clip-list">
                  {clipsWithTranscript.map((clip) => (
                    <label key={clip.id} className="multicam-clip-item">
                      <input
                        type="radio"
                        name="masterClip"
                        value={clip.id}
                        checked={masterClipId === clip.id}
                        onChange={() => setMasterClipId(clip.id)}
                        disabled={syncing}
                      />
                      <span className="clip-icon">{'\uD83D\uDCDD'}</span>
                      <span className="clip-name" title={clip.name}>
                        {clip.name}
                      </span>
                      <span className="transcript-word-count">
                        {clip.transcript?.length || 0} words
                      </span>
                      {masterClipId === clip.id && (
                        <span className="master-badge">Master</span>
                      )}
                    </label>
                  ))}
                </div>
              )
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
            disabled={
              syncing ||
              !masterClipId ||
              (syncMethod === 'audio' && clipsWithAudio.length < 2) ||
              (syncMethod === 'transcript' && clipsWithTranscript.length < 2)
            }
          >
            {syncing ? 'Syncing...' : 'Sync & Link'}
          </button>
        </div>
      </div>
    </div>
  );
}
