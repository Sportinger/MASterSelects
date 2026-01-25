// Transcript Sync Service
// Synchronizes clips based on their transcribed text content
// More accurate than audio waveform correlation when transcripts are available

import { Logger } from './logger';
import type { TranscriptWord } from '../types';

const log = Logger.create('TranscriptSync');

export interface TranscriptSyncResult {
  offsetMs: number;           // Time offset in milliseconds
  confidence: number;         // 0-1 confidence score
  matchedWords: number;       // Number of words that matched
  matchedSequence: string;    // The matched text sequence
}

/**
 * Normalize text for comparison (lowercase, remove punctuation)
 */
function normalizeWord(word: string): string {
  return word.toLowerCase().replace(/[^\wäöüß]/g, '');
}

/**
 * Find the longest common subsequence of words between two transcripts
 * Returns the matching words with their positions in both transcripts
 */
function findMatchingSequences(
  transcript1: TranscriptWord[],
  transcript2: TranscriptWord[],
  minSequenceLength: number = 3
): Array<{ words1: TranscriptWord[]; words2: TranscriptWord[]; length: number }> {
  const matches: Array<{ words1: TranscriptWord[]; words2: TranscriptWord[]; length: number }> = [];

  // Normalize words for comparison
  const norm1 = transcript1.map(w => normalizeWord(w.text));
  const norm2 = transcript2.map(w => normalizeWord(w.text));

  // Find all matching sequences using a sliding window approach
  for (let i = 0; i < norm1.length; i++) {
    for (let j = 0; j < norm2.length; j++) {
      if (norm1[i] === norm2[j] && norm1[i].length > 0) {
        // Found a match, extend it as far as possible
        let len = 1;
        while (
          i + len < norm1.length &&
          j + len < norm2.length &&
          norm1[i + len] === norm2[j + len] &&
          norm1[i + len].length > 0
        ) {
          len++;
        }

        if (len >= minSequenceLength) {
          matches.push({
            words1: transcript1.slice(i, i + len),
            words2: transcript2.slice(j, j + len),
            length: len,
          });
        }
      }
    }
  }

  // Sort by length (longest first) and remove overlapping matches
  matches.sort((a, b) => b.length - a.length);

  // Filter out overlapping matches (keep longest)
  const used1 = new Set<number>();
  const used2 = new Set<number>();
  const filtered: typeof matches = [];

  for (const match of matches) {
    const idx1Start = transcript1.indexOf(match.words1[0]);
    const idx2Start = transcript2.indexOf(match.words2[0]);

    let overlaps = false;
    for (let k = 0; k < match.length; k++) {
      if (used1.has(idx1Start + k) || used2.has(idx2Start + k)) {
        overlaps = true;
        break;
      }
    }

    if (!overlaps) {
      filtered.push(match);
      for (let k = 0; k < match.length; k++) {
        used1.add(idx1Start + k);
        used2.add(idx2Start + k);
      }
    }
  }

  return filtered;
}

/**
 * Calculate time offset between two transcripts based on matching word sequences
 * Returns the offset in milliseconds (positive = transcript2 is delayed)
 */
export function calculateTranscriptOffset(
  masterTranscript: TranscriptWord[],
  targetTranscript: TranscriptWord[],
  minMatchLength: number = 3
): TranscriptSyncResult | null {
  if (masterTranscript.length === 0 || targetTranscript.length === 0) {
    log.warn('Empty transcript(s)');
    return null;
  }

  // Find matching sequences
  const matches = findMatchingSequences(masterTranscript, targetTranscript, minMatchLength);

  if (matches.length === 0) {
    log.warn('No matching sequences found');
    return null;
  }

  log.debug(`Found ${matches.length} matching sequence(s)`);

  // Calculate offset for each match and use weighted average
  const offsets: { offsetMs: number; weight: number }[] = [];
  let totalMatchedWords = 0;

  for (const match of matches) {
    // Calculate average offset for this sequence
    let sumOffset = 0;
    for (let i = 0; i < match.length; i++) {
      // Offset = master time - target time
      // Positive offset means target clip should move earlier
      const masterTime = (match.words1[i].start + match.words1[i].end) / 2;
      const targetTime = (match.words2[i].start + match.words2[i].end) / 2;
      sumOffset += (masterTime - targetTime) * 1000; // Convert to ms
    }

    const avgOffset = sumOffset / match.length;
    offsets.push({ offsetMs: avgOffset, weight: match.length });
    totalMatchedWords += match.length;
  }

  // Calculate weighted average offset
  let weightedSum = 0;
  let totalWeight = 0;
  for (const { offsetMs, weight } of offsets) {
    weightedSum += offsetMs * weight;
    totalWeight += weight;
  }

  const finalOffset = weightedSum / totalWeight;

  // Calculate confidence based on how many words matched
  const totalWords = Math.min(masterTranscript.length, targetTranscript.length);
  const confidence = Math.min(1, totalMatchedWords / totalWords);

  // Build matched sequence string from longest match
  const longestMatch = matches[0];
  const matchedSequence = longestMatch.words1.map(w => w.text).join(' ');

  log.info(`Offset: ${finalOffset.toFixed(1)}ms, confidence: ${(confidence * 100).toFixed(1)}%, matched: ${totalMatchedWords} words`);
  log.debug(`Longest match: "${matchedSequence.substring(0, 100)}${matchedSequence.length > 100 ? '...' : ''}"`);

  return {
    offsetMs: finalOffset,
    confidence,
    matchedWords: totalMatchedWords,
    matchedSequence,
  };
}

/**
 * Sync multiple clips using their transcripts
 * Returns a map of clipId to offset in milliseconds
 */
export function syncClipsByTranscript(
  masterClipId: string,
  masterTranscript: TranscriptWord[],
  targetClips: Array<{ clipId: string; transcript: TranscriptWord[] }>
): Map<string, { offsetMs: number; confidence: number; matchedWords: number }> {
  const results = new Map<string, { offsetMs: number; confidence: number; matchedWords: number }>();

  // Master has zero offset
  results.set(masterClipId, { offsetMs: 0, confidence: 1, matchedWords: masterTranscript.length });

  for (const target of targetClips) {
    const result = calculateTranscriptOffset(masterTranscript, target.transcript);

    if (result) {
      results.set(target.clipId, {
        offsetMs: result.offsetMs,
        confidence: result.confidence,
        matchedWords: result.matchedWords,
      });
    } else {
      // No match found - keep current position (0 offset)
      log.warn(`No transcript match for clip ${target.clipId}`);
      results.set(target.clipId, { offsetMs: 0, confidence: 0, matchedWords: 0 });
    }
  }

  return results;
}
