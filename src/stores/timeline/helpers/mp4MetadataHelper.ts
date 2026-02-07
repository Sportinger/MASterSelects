// Fast MP4/MOV metadata extraction using MP4Box
// Reads from both start AND end of file to handle camera MOV files
// where the moov atom is at the end (not faststart/web-optimized).

import { Logger } from '../../../services/logger';

const log = Logger.create('MP4Metadata');

// Lazy-load mp4box
let _MP4Box: any = null;
async function getMP4Box() {
  if (!_MP4Box) {
    const mod = await import('mp4box');
    _MP4Box = (mod as any).default || mod;
  }
  return _MP4Box;
}

// MP4-based containers
const MP4_EXTENSIONS = ['mp4', 'm4v', 'mov', '3gp', 'mp4v', 'mxf'];

export interface MP4Metadata {
  duration: number;
  width?: number;
  height?: number;
  fps?: number;
  hasAudio: boolean;
  codec?: string;
}

/**
 * Extract metadata from MP4/MOV container using MP4Box.
 * Reads from both start and end of file to handle camera files
 * with moov atom at end. Much faster than waiting for HTMLVideoElement.
 *
 * Returns null if file is not MP4/MOV or parsing fails.
 */
export async function getMP4MetadataFast(file: File, timeoutMs = 5000): Promise<MP4Metadata | null> {
  const ext = file.name.split('.').pop()?.toLowerCase() || '';
  if (!MP4_EXTENSIONS.includes(ext)) return null;

  try {
    const MP4Box = await getMP4Box();

    return new Promise<MP4Metadata | null>((resolve) => {
      const mp4boxFile = MP4Box.createFile();
      let resolved = false;

      const done = (result: MP4Metadata | null) => {
        if (resolved) return;
        resolved = true;
        clearTimeout(timeout);
        resolve(result);
      };

      const timeout = setTimeout(() => {
        log.debug('MP4Box metadata timeout', { file: file.name });
        done(null);
      }, timeoutMs);

      mp4boxFile.onReady = (info: any) => {
        const videoTrack = info.videoTracks?.[0];
        const duration = info.duration && info.timescale
          ? info.duration / info.timescale
          : videoTrack?.duration && videoTrack?.timescale
            ? videoTrack.duration / videoTrack.timescale
            : null;

        if (!duration || !isFinite(duration) || duration <= 0) {
          log.debug('MP4Box: no valid duration', { file: file.name });
          done(null);
          return;
        }

        const result: MP4Metadata = {
          duration,
          hasAudio: (info.audioTracks?.length || 0) > 0,
        };

        if (videoTrack) {
          // Extract dimensions from track header
          if (videoTrack.track_width && videoTrack.track_height) {
            result.width = videoTrack.track_width;
            result.height = videoTrack.track_height;
          } else if (videoTrack.video?.width && videoTrack.video?.height) {
            result.width = videoTrack.video.width;
            result.height = videoTrack.video.height;
          }

          // Extract FPS
          if (videoTrack.nb_samples && duration > 0) {
            result.fps = Math.round(videoTrack.nb_samples / duration);
          }

          // Extract codec
          if (videoTrack.codec) {
            result.codec = videoTrack.codec;
          }
        }

        log.debug('MP4Box metadata extracted', {
          file: file.name,
          duration: result.duration.toFixed(2),
          width: result.width,
          height: result.height,
          fps: result.fps,
          hasAudio: result.hasAudio,
        });

        done(result);
      };

      mp4boxFile.onError = (error: any) => {
        log.debug('MP4Box parse error', { file: file.name, error });
        done(null);
      };

      // Strategy: Read from start AND end of file in parallel
      // Camera MOV files often have moov atom at the end
      const chunkSize = 1024 * 1024; // 1MB chunks
      const maxFromStart = 5 * 1024 * 1024; // Read up to 5MB from start
      const maxFromEnd = 5 * 1024 * 1024; // Read up to 5MB from end

      let startOffset = 0;
      let endDone = false;
      let startDone = false;

      // Read from end of file (where moov usually is for camera files)
      const readEnd = async () => {
        try {
          const endStart = Math.max(0, file.size - maxFromEnd);
          // Don't overlap with start reading
          if (endStart <= maxFromStart) {
            endDone = true;
            return;
          }

          let offset = endStart;
          while (offset < file.size && !resolved) {
            const end = Math.min(offset + chunkSize, file.size);
            const blob = file.slice(offset, end);
            const buffer = await blob.arrayBuffer();
            if (resolved) return;
            (buffer as any).fileStart = offset;
            try {
              mp4boxFile.appendBuffer(buffer as any);
            } catch {
              // MP4Box may throw if it gets confused by non-sequential data
              break;
            }
            offset = end;
          }
        } catch (e) {
          log.debug('Error reading end of file', e);
        }
        endDone = true;
        if (startDone && endDone && !resolved) {
          try { mp4boxFile.flush(); } catch { /* ignore */ }
        }
      };

      // Read from start of file
      const readStart = async () => {
        try {
          while (startOffset < Math.min(file.size, maxFromStart) && !resolved) {
            const end = Math.min(startOffset + chunkSize, file.size);
            const blob = file.slice(startOffset, end);
            const buffer = await blob.arrayBuffer();
            if (resolved) return;
            (buffer as any).fileStart = startOffset;
            try {
              mp4boxFile.appendBuffer(buffer as any);
            } catch {
              break;
            }
            startOffset = end;
          }
        } catch (e) {
          log.debug('Error reading start of file', e);
        }
        startDone = true;
        if (startDone && endDone && !resolved) {
          try { mp4boxFile.flush(); } catch { /* ignore */ }
        }
      };

      // Run both in parallel
      readStart();
      readEnd();
    });
  } catch (err) {
    log.debug('MP4Box metadata extraction failed', err);
    return null;
  }
}

/**
 * Estimate video duration from file size (very rough fallback).
 * Uses typical bitrates for common camera codecs.
 * Better than showing 5 seconds for a multi-minute video.
 */
export function estimateDurationFromFileSize(file: File): number {
  const sizeMB = file.size / (1024 * 1024);
  const ext = file.name.split('.').pop()?.toLowerCase() || '';

  // Camera MOV files are typically 100-200 Mbps
  // Conservative estimate: assume ~150 Mbps for MOV, ~50 Mbps for MP4
  let bitrateMbps: number;
  if (ext === 'mov' || ext === 'mxf') {
    bitrateMbps = 150; // ProRes/camera H.264 tend to be high bitrate
  } else if (ext === 'mp4' || ext === 'm4v') {
    bitrateMbps = 50; // Compressed H.264/H.265
  } else {
    bitrateMbps = 80; // General estimate
  }

  const durationSeconds = (sizeMB * 8) / bitrateMbps;
  // Clamp to reasonable range
  return Math.max(1, Math.min(durationSeconds, 7200)); // 1s to 2h
}
