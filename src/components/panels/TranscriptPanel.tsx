// Transcript Panel - Premiere Pro-style speech-to-text transcript viewer
// Shows individual words with real-time highlighting during playback

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useTimelineStore } from '../../stores/timeline';
import type { TranscriptWord } from '../../types';
import './TranscriptPanel.css';

// =============================================================================
// Sub-components
// =============================================================================

interface WordProps {
  word: TranscriptWord;
  isActive: boolean;
  isHighlighted: boolean;
  onClick: (time: number) => void;
}

function Word({ word, isActive, isHighlighted, onClick }: WordProps) {
  return (
    <span
      className={`transcript-word ${isActive ? 'active' : ''} ${isHighlighted ? 'highlighted' : ''}`}
      onClick={() => onClick(word.start)}
      title={`${formatTime(word.start)} - ${formatTime(word.end)}`}
    >
      {word.text}
    </span>
  );
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

interface SpeakerBlockProps {
  speaker: string;
  startTime: number;
  endTime: number;
  words: TranscriptWord[];
  currentWordId: string | null;
  searchQuery: string;
  onClick: (time: number) => void;
}

function SpeakerBlock({
  speaker,
  startTime,
  endTime,
  words,
  currentWordId,
  searchQuery,
  onClick,
}: SpeakerBlockProps) {
  const query = searchQuery.toLowerCase();

  return (
    <div className="transcript-block">
      <div className="transcript-block-header">
        <span className="transcript-speaker">{speaker}</span>
        <span className="transcript-time">
          {formatTime(startTime)} - {formatTime(endTime)}
        </span>
      </div>
      <div className="transcript-words">
        {words.map((word) => {
          const isHighlighted = query && word.text.toLowerCase().includes(query);
          return (
            <Word
              key={word.id}
              word={word}
              isActive={word.id === currentWordId}
              isHighlighted={!!isHighlighted}
              onClick={onClick}
            />
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// Language options
// =============================================================================

const LANGUAGES = [
  { code: 'de', name: 'Deutsch' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'fr', name: 'Français' },
  { code: 'it', name: 'Italiano' },
  { code: 'pt', name: 'Português' },
  { code: 'nl', name: 'Nederlands' },
  { code: 'pl', name: 'Polski' },
  { code: 'ru', name: 'Русский' },
  { code: 'ja', name: '日本語' },
  { code: 'zh', name: '中文' },
  { code: 'ko', name: '한국어' },
];

// =============================================================================
// Main Component
// =============================================================================

export function TranscriptPanel() {
  const [searchQuery, setSearchQuery] = useState('');
  const [showMarkersGlobal, setShowMarkersGlobal] = useState(true);
  const [language, setLanguage] = useState(() => {
    // Load from localStorage or default to German
    return localStorage.getItem('transcriptLanguage') || 'de';
  });
  const containerRef = useRef<HTMLDivElement>(null);
  const activeWordRef = useRef<HTMLSpanElement | null>(null);

  // Timeline store
  const {
    clips,
    selectedClipId,
    playheadPosition,
    setPlayheadPosition,
  } = useTimelineStore();

  // Get selected clip or first clip with transcript
  const selectedClip = useMemo(() => {
    if (selectedClipId) {
      return clips.find(c => c.id === selectedClipId);
    }
    // Find first clip with transcript
    return clips.find(c => c.transcript && c.transcript.length > 0);
  }, [clips, selectedClipId]);

  // Get transcript from selected clip
  const transcript = selectedClip?.transcript ?? [];
  const transcriptStatus = selectedClip?.transcriptStatus ?? 'none';
  const transcriptProgress = selectedClip?.transcriptProgress ?? 0;
  const transcriptMessage = selectedClip?.transcriptMessage;

  // Calculate clip-local time for word matching
  const clipLocalTime = useMemo(() => {
    if (!selectedClip) return -1;
    return playheadPosition - selectedClip.startTime + selectedClip.inPoint;
  }, [selectedClip, playheadPosition]);

  // Find current word based on playhead position
  const currentWordId = useMemo(() => {
    if (clipLocalTime < 0 || transcript.length === 0) return null;

    // Binary search for current word
    let left = 0;
    let right = transcript.length - 1;

    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const word = transcript[mid];

      if (clipLocalTime >= word.start && clipLocalTime <= word.end) {
        return word.id;
      } else if (clipLocalTime < word.start) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // If not exactly in a word, find nearest
    for (const word of transcript) {
      if (clipLocalTime >= word.start && clipLocalTime <= word.end) {
        return word.id;
      }
    }

    return null;
  }, [transcript, clipLocalTime]);

  // Group words into speaker blocks (by speaker and time gaps)
  const speakerBlocks = useMemo(() => {
    if (transcript.length === 0) return [];

    const blocks: Array<{
      speaker: string;
      startTime: number;
      endTime: number;
      words: TranscriptWord[];
    }> = [];

    let currentBlock: typeof blocks[0] | null = null;

    for (const word of transcript) {
      const speaker = word.speaker || 'Speaker 1';

      // Start new block if speaker changes or gap > 2 seconds
      if (
        !currentBlock ||
        currentBlock.speaker !== speaker ||
        word.start - currentBlock.endTime > 2
      ) {
        if (currentBlock) {
          blocks.push(currentBlock);
        }
        currentBlock = {
          speaker,
          startTime: word.start,
          endTime: word.end,
          words: [word],
        };
      } else {
        currentBlock.words.push(word);
        currentBlock.endTime = word.end;
      }
    }

    if (currentBlock) {
      blocks.push(currentBlock);
    }

    return blocks;
  }, [transcript]);

  // Filter blocks by search query
  const filteredBlocks = useMemo(() => {
    if (!searchQuery.trim()) return speakerBlocks;

    const query = searchQuery.toLowerCase();
    return speakerBlocks.filter(block =>
      block.words.some(w => w.text.toLowerCase().includes(query))
    );
  }, [speakerBlocks, searchQuery]);

  // Handle click on word - seek to that time
  const handleWordClick = useCallback((sourceTime: number) => {
    if (!selectedClip) return;

    // Convert source time to timeline position
    const timelinePosition = selectedClip.startTime + (sourceTime - selectedClip.inPoint);
    setPlayheadPosition(Math.max(0, timelinePosition));
  }, [selectedClip, setPlayheadPosition]);

  // Auto-scroll to active word
  useEffect(() => {
    if (currentWordId && containerRef.current) {
      const activeElement = containerRef.current.querySelector('.transcript-word.active');
      if (activeElement) {
        activeElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currentWordId]);

  // Handle language change
  const handleLanguageChange = useCallback((newLanguage: string) => {
    setLanguage(newLanguage);
    localStorage.setItem('transcriptLanguage', newLanguage);
  }, []);

  // Handle transcribe button click
  const handleTranscribe = useCallback(async () => {
    if (!selectedClipId) return;

    // Import and call transcription with selected language
    const { transcribeClip } = await import('../../services/clipTranscriber');
    await transcribeClip(selectedClipId, language);
  }, [selectedClipId, language]);

  // Handle cancel transcription
  const handleCancel = useCallback(async () => {
    const { cancelTranscription } = await import('../../services/clipTranscriber');
    cancelTranscription();
  }, []);

  // Render empty state
  if (!selectedClip) {
    return (
      <div className="transcript-panel">
        <div className="transcript-header">
          <h2>Transcript</h2>
        </div>
        <div className="transcript-empty">
          <p>Select a clip to view or generate transcript</p>
        </div>
      </div>
    );
  }

  return (
    <div className="transcript-panel">
      {/* Header */}
      <div className="transcript-header">
        <h2>Transcript</h2>
        <div className="transcript-header-actions">
          <label className="marker-toggle" title="Show word markers on timeline">
            <input
              type="checkbox"
              checked={showMarkersGlobal}
              onChange={(e) => setShowMarkersGlobal(e.target.checked)}
            />
            <span>Markers</span>
          </label>
        </div>
      </div>

      {/* Search */}
      <div className="transcript-search">
        <input
          type="text"
          placeholder="Search transcript..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <button
            className="search-clear"
            onClick={() => setSearchQuery('')}
          >
            x
          </button>
        )}
      </div>

      {/* Clip info */}
      <div className="transcript-clip-info">
        <span className="clip-name" title={selectedClip.name}>
          {selectedClip.name}
        </span>
        {transcriptStatus === 'transcribing' && (
          <span className="transcript-status transcribing" title={transcriptMessage}>
            {transcript.length > 0 ? `${transcript.length} words` : `${transcriptProgress}%`}
          </span>
        )}
        {transcriptStatus === 'ready' && (
          <span className="transcript-status ready">
            {transcript.length} words
          </span>
        )}
        {transcriptStatus === 'error' && (
          <span className="transcript-status error">
            Error
          </span>
        )}
      </div>

      {/* Language selector */}
      <div className="transcript-language">
        <label htmlFor="language-select">Sprache:</label>
        <select
          id="language-select"
          value={language}
          onChange={(e) => handleLanguageChange(e.target.value)}
          disabled={transcriptStatus === 'transcribing'}
        >
          {LANGUAGES.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name}
            </option>
          ))}
        </select>
      </div>

      {/* Actions */}
      <div className="transcript-actions">
        {transcriptStatus !== 'ready' && transcriptStatus !== 'transcribing' && (
          <button
            className="btn-transcribe"
            onClick={handleTranscribe}
          >
            Transcribe
          </button>
        )}
        {transcriptStatus === 'transcribing' && (
          <button
            className="btn-transcribe btn-cancel"
            onClick={handleCancel}
          >
            Cancel
          </button>
        )}
        {transcriptStatus === 'ready' && (
          <button
            className="btn-transcribe btn-secondary"
            onClick={handleTranscribe}
          >
            Re-transcribe
          </button>
        )}
      </div>

      {/* Progress bar */}
      {transcriptStatus === 'transcribing' && (
        <div className="transcript-progress">
          <div
            className="transcript-progress-bar"
            style={{ width: `${transcriptProgress}%` }}
          />
        </div>
      )}

      {/* Transcription status message */}
      {transcriptStatus === 'transcribing' && transcriptMessage && (
        <div className="transcript-message">
          {transcriptMessage}
        </div>
      )}

      {/* Transcript content */}
      <div className="transcript-content" ref={containerRef}>
        {filteredBlocks.length === 0 ? (
          <div className="transcript-empty">
            {transcriptStatus === 'transcribing' ? (
              <p>Transcribing... Words will appear here as they're processed.</p>
            ) : transcript.length === 0 ? (
              <p>No transcript available. Click "Transcribe" to generate.</p>
            ) : (
              <p>No results found for "{searchQuery}"</p>
            )}
          </div>
        ) : (
          filteredBlocks.map((block, index) => (
            <SpeakerBlock
              key={`${block.startTime}-${index}`}
              speaker={block.speaker}
              startTime={block.startTime}
              endTime={block.endTime}
              words={block.words}
              currentWordId={currentWordId}
              searchQuery={searchQuery}
              onClick={handleWordClick}
            />
          ))
        )}
      </div>

      {/* Footer */}
      <div className="transcript-footer">
        <span className="transcript-hint">
          Click word to seek. Right-click clip to transcribe.
        </span>
      </div>
    </div>
  );
}
