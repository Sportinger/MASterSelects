// Transcript Tab - View and interact with clip transcription
import { useState, useCallback, useMemo, useRef } from 'react';
import { useTimelineStore } from '../../../stores/timeline';
import type { TranscriptWord } from '../../../types';

const LANGUAGES = [
  { code: 'de', name: 'Deutsch' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'fr', name: 'Français' },
  { code: 'it', name: 'Italiano' },
  { code: 'pt', name: 'Português' },
];

function formatTimeShort(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

interface TranscriptTabProps {
  clipId: string;
  transcript: TranscriptWord[];
  transcriptStatus: 'none' | 'transcribing' | 'ready' | 'error';
  transcriptProgress: number;
  clipStartTime: number;
  inPoint: number;
}

export function TranscriptTab({ clipId, transcript, transcriptStatus, transcriptProgress, clipStartTime, inPoint }: TranscriptTabProps) {
  // Reactive data - subscribe to specific value only
  const playheadPosition = useTimelineStore(state => state.playheadPosition);
  // Actions from getState() - stable, no subscription needed
  const { setPlayheadPosition } = useTimelineStore.getState();
  const [language, setLanguage] = useState(() => localStorage.getItem('transcriptLanguage') || 'de');
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate clip-local time for word matching
  const clipLocalTime = playheadPosition - clipStartTime + inPoint;

  // Find current word based on playhead position
  const currentWordId = useMemo(() => {
    if (clipLocalTime < 0 || transcript.length === 0) return null;
    for (const word of transcript) {
      if (clipLocalTime >= word.start && clipLocalTime <= word.end) {
        return word.id;
      }
    }
    return null;
  }, [transcript, clipLocalTime]);

  // Filter words by search query
  const filteredWords = useMemo(() => {
    if (!searchQuery.trim()) return transcript;
    const query = searchQuery.toLowerCase();
    return transcript.filter(w => w.text.toLowerCase().includes(query));
  }, [transcript, searchQuery]);

  const handleWordClick = useCallback((sourceTime: number) => {
    const timelinePosition = clipStartTime + (sourceTime - inPoint);
    setPlayheadPosition(Math.max(0, timelinePosition));
  }, [clipStartTime, inPoint, setPlayheadPosition]);

  const handleTranscribe = useCallback(async () => {
    const { transcribeClip } = await import('../../../services/clipTranscriber');
    await transcribeClip(clipId, language);
  }, [clipId, language]);

  const handleCancel = useCallback(async () => {
    const { cancelTranscription } = await import('../../../services/clipTranscriber');
    cancelTranscription();
  }, []);

  const handleDelete = useCallback(async () => {
    const { clearClipTranscript } = await import('../../../services/clipTranscriber');
    clearClipTranscript(clipId);
  }, [clipId]);

  const handleLanguageChange = useCallback((newLanguage: string) => {
    setLanguage(newLanguage);
    localStorage.setItem('transcriptLanguage', newLanguage);
  }, []);

  return (
    <div className="properties-tab-content transcript-tab">
      {/* Language and actions */}
      <div className="properties-section">
        <div className="control-row">
          <label>Language</label>
          <select value={language} onChange={(e) => handleLanguageChange(e.target.value)}
            disabled={transcriptStatus === 'transcribing'}>
            {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.name}</option>)}
          </select>
        </div>
        <div className="transcript-tab-actions">
          {transcriptStatus !== 'ready' && transcriptStatus !== 'transcribing' && (
            <button className="btn btn-sm" onClick={handleTranscribe}>Transcribe</button>
          )}
          {transcriptStatus === 'transcribing' && (
            <button className="btn btn-sm btn-danger" onClick={handleCancel}>Cancel</button>
          )}
          {transcriptStatus === 'ready' && (
            <>
              <button className="btn btn-sm" onClick={handleTranscribe}>Re-transcribe</button>
              <button className="btn btn-sm btn-danger" onClick={handleDelete}>Delete</button>
            </>
          )}
        </div>
      </div>

      {/* Progress */}
      {transcriptStatus === 'transcribing' && (
        <div className="properties-section">
          <div className="transcript-progress-bar">
            <div className="transcript-progress-fill" style={{ width: `${transcriptProgress}%` }} />
          </div>
          <span className="transcript-progress-text">{transcriptProgress}%</span>
        </div>
      )}

      {/* Search */}
      {transcript.length > 0 && (
        <div className="properties-section">
          <input type="text" placeholder="Search transcript..." value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)} className="transcript-search-input" />
        </div>
      )}

      {/* Transcript content */}
      <div className="transcript-content-embedded" ref={containerRef}>
        {transcript.length === 0 ? (
          <div className="transcript-empty-state">
            {transcriptStatus === 'transcribing' ? 'Transcribing...' : 'No transcript. Click "Transcribe" to generate.'}
          </div>
        ) : (
          <div className="transcript-words-flow">
            {filteredWords.map(word => (
              <span
                key={word.id}
                className={`transcript-word-inline ${word.id === currentWordId ? 'active' : ''} ${searchQuery && word.text.toLowerCase().includes(searchQuery.toLowerCase()) ? 'highlighted' : ''}`}
                onClick={() => handleWordClick(word.start)}
                title={formatTimeShort(word.start)}
              >
                {word.text}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Status */}
      {transcriptStatus === 'ready' && (
        <div className="transcript-status-bar">
          {transcript.length} words
        </div>
      )}
    </div>
  );
}
