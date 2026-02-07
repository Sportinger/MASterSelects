// Audio detection helpers for various video container formats
// Supports: MP4, MOV, M4V, 3GP (via MP4Box), WebM, MKV, AVI, etc.

import { Logger } from '../../../services/logger';

const log = Logger.create('AudioDetection');

// Lazy-load mp4box only when needed (saves ~200KB from initial bundle)
let _MP4Box: any = null;
async function getMP4Box() {
  if (!_MP4Box) {
    const mod = await import('mp4box');
    _MP4Box = (mod as any).default || mod;
  }
  return _MP4Box;
}

// MP4-based containers that MP4Box can parse
const MP4_CONTAINERS = ['mp4', 'm4v', 'mov', '3gp', 'mp4v'];

// WebM/Matroska containers
const WEBM_CONTAINERS = ['webm', 'mkv'];

/**
 * Detect if a video file has audio tracks.
 * Uses multiple detection methods depending on container format.
 */
export async function detectVideoAudio(file: File): Promise<boolean> {
  const ext = file.name.split('.').pop()?.toLowerCase() || '';

  log.debug('Detecting audio', { file: file.name, ext });

  // Method 1: MP4Box for MP4-based containers (most reliable for positive detection)
  if (MP4_CONTAINERS.includes(ext)) {
    const result = await detectAudioMP4Box(file);
    if (result === true) {
      log.debug('MP4Box detection result', { file: file.name, hasAudio: true });
      return true;
    }
    // Don't trust false from MP4Box - camera MOV files often have moov atom at
    // end of file (past read limit) or use PCM audio codecs MP4Box may not classify
    // as audioTracks. Fall through to VideoElement detection.
    log.debug('MP4Box returned non-positive, trying fallback', { file: file.name, result });
  }

  // Method 2: HTMLVideoElement for WebM and other formats
  const videoElementResult = await detectAudioVideoElement(file);
  if (videoElementResult !== null) {
    log.debug('VideoElement detection result', { file: file.name, hasAudio: videoElementResult });
    return videoElementResult;
  }

  // Method 3: MediaSource probe for WebM/MKV
  if (WEBM_CONTAINERS.includes(ext)) {
    const webmResult = await detectAudioWebM(file);
    if (webmResult !== null) {
      log.debug('WebM detection result', { file: file.name, hasAudio: webmResult });
      return webmResult;
    }
  }

  // Fallback: Assume audio exists (better than wrongly removing it)
  log.debug('Using fallback - assuming audio exists', { file: file.name });
  return true;
}

/**
 * Detect audio using MP4Box (for MP4/MOV/M4V/3GP containers).
 * Returns null if detection fails.
 */
async function detectAudioMP4Box(file: File): Promise<boolean | null> {
  const MP4Box = await getMP4Box();
  return new Promise((resolve) => {
    const mp4boxFile = MP4Box.createFile();
    let resolved = false;

    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        log.debug('MP4Box timeout', { file: file.name });
        resolve(null);
      }
    }, 5000);

    mp4boxFile.onReady = (info: any) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);

      const hasAudio = info.audioTracks && info.audioTracks.length > 0;
      log.debug('MP4Box parsed', {
        file: file.name,
        audioTracks: info.audioTracks?.length || 0,
        videoTracks: info.videoTracks?.length || 0
      });
      resolve(hasAudio);
    };

    mp4boxFile.onError = (error: any) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);
      log.debug('MP4Box error', { file: file.name, error });
      resolve(null);
    };

    // Read file in chunks
    const chunkSize = 1024 * 1024; // 1MB
    let offset = 0;

    const reader = new FileReader();
    reader.onload = (e) => {
      if (resolved) return;
      const buffer = e.target?.result as ArrayBuffer;
      if (buffer) {
        (buffer as any).fileStart = offset;
        mp4boxFile.appendBuffer(buffer as any);
        offset += buffer.byteLength;

        // Read up to 5MB for metadata
        if (offset < Math.min(file.size, 5 * 1024 * 1024)) {
          const nextChunk = file.slice(offset, offset + chunkSize);
          reader.readAsArrayBuffer(nextChunk);
        } else {
          mp4boxFile.flush();
        }
      }
    };

    reader.onerror = () => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeout);
      resolve(null);
    };

    const firstChunk = file.slice(0, chunkSize);
    reader.readAsArrayBuffer(firstChunk);
  });
}

/**
 * Detect audio using HTMLVideoElement properties.
 * Works for formats the browser can play natively.
 * Returns null if detection is inconclusive.
 */
async function detectAudioVideoElement(file: File): Promise<boolean | null> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(file);
    video.src = url;
    video.muted = true;
    video.preload = 'metadata';

    const timeout = setTimeout(() => {
      cleanup();
      resolve(null);
    }, 5000);

    const cleanup = () => {
      clearTimeout(timeout);
      URL.revokeObjectURL(url);
      video.src = '';
    };

    video.onloadedmetadata = () => {
      // Method A: audioTracks API (Chrome/Safari)
      if ('audioTracks' in video) {
        const audioTracks = (video as any).audioTracks;
        if (audioTracks && typeof audioTracks.length === 'number') {
          cleanup();
          resolve(audioTracks.length > 0);
          return;
        }
      }

      // Method B: webkitAudioDecodedByteCount (Chrome)
      // Need to play briefly to get this value
      video.currentTime = 0.1;
      video.play().then(() => {
        setTimeout(() => {
          video.pause();
          const audioBytes = (video as any).webkitAudioDecodedByteCount;
          if (typeof audioBytes === 'number') {
            cleanup();
            resolve(audioBytes > 0);
            return;
          }
          cleanup();
          resolve(null); // Inconclusive
        }, 300); // 300ms for large camera MOV files that need more decode time
      }).catch(() => {
        cleanup();
        resolve(null);
      });
    };

    video.onerror = () => {
      cleanup();
      resolve(null);
    };

    video.load();
  });
}

/**
 * Detect audio in WebM/MKV by parsing EBML header.
 * WebM uses EBML (Extensible Binary Meta Language).
 */
async function detectAudioWebM(file: File): Promise<boolean | null> {
  try {
    // Read first 1MB to find track info
    const maxBytes = 1024 * 1024;
    const blob = file.slice(0, Math.min(file.size, maxBytes));
    const buffer = await blob.arrayBuffer();
    const data = new Uint8Array(buffer);

    // Simple WebM/MKV parsing - look for audio track type marker
    // Track type 2 = audio in EBML
    // This is a simplified check - look for TrackType element (0x83) with value 2

    // EBML audio track markers we're looking for
    const audioMarkers = [
      [0x83, 0x02], // TrackType = 2 (audio)
      [0x86, 0x41, 0x5f, 0x4f, 0x50, 0x55, 0x53], // CodecID "A_OPUS"
      [0x86, 0x41, 0x5f, 0x56, 0x4f, 0x52, 0x42, 0x49, 0x53], // CodecID "A_VORBIS"
      [0x86, 0x41, 0x5f, 0x41, 0x41, 0x43], // CodecID "A_AAC"
    ];

    for (const marker of audioMarkers) {
      if (findSequence(data, marker)) {
        return true;
      }
    }

    // Didn't find audio markers, but this doesn't mean no audio
    // Return null to try other methods
    return null;
  } catch (e) {
    log.debug('WebM parsing failed', e);
    return null;
  }
}

/**
 * Find a byte sequence in a buffer.
 */
function findSequence(data: Uint8Array, sequence: number[]): boolean {
  outer: for (let i = 0; i <= data.length - sequence.length; i++) {
    for (let j = 0; j < sequence.length; j++) {
      if (data[i + j] !== sequence[j]) continue outer;
    }
    return true;
  }
  return false;
}
