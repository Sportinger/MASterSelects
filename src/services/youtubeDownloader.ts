// YouTube video downloader service using Native Helper + yt-dlp
// Downloads videos locally without third-party web services

import { Logger } from './logger';
import { NativeHelperClient } from './nativeHelper';

const log = Logger.create('YouTubeDownloader');
import { projectFileService } from './projectFileService';

export interface DownloadProgress {
  videoId: string;
  status: 'pending' | 'downloading' | 'processing' | 'complete' | 'error';
  progress: number; // 0-100
  error?: string;
  file?: File;
  title: string;
  thumbnail: string;
}

export type DownloadCallback = (progress: DownloadProgress) => void;

// Active downloads map
const activeDownloads = new Map<string, DownloadProgress>();
const downloadCallbacks = new Map<string, Set<DownloadCallback>>();

// Subscribe to download updates
export function subscribeToDownload(videoId: string, callback: DownloadCallback): () => void {
  if (!downloadCallbacks.has(videoId)) {
    downloadCallbacks.set(videoId, new Set());
  }
  downloadCallbacks.get(videoId)!.add(callback);

  // Immediately send current status if exists
  const current = activeDownloads.get(videoId);
  if (current) {
    callback(current);
  }

  // Return unsubscribe function
  return () => {
    downloadCallbacks.get(videoId)?.delete(callback);
  };
}

// Notify all subscribers of a download
function notifySubscribers(progress: DownloadProgress) {
  activeDownloads.set(progress.videoId, progress);
  downloadCallbacks.get(progress.videoId)?.forEach(cb => cb(progress));
}

// Get download status
export function getDownloadStatus(videoId: string): DownloadProgress | undefined {
  return activeDownloads.get(videoId);
}

// Check if Native Helper is available for downloads
export function isDownloadAvailable(): boolean {
  return NativeHelperClient.isConnected();
}

// Download video from any yt-dlp-supported URL using Native Helper
export async function downloadVideo(
  url: string,
  videoId: string,
  title: string,
  thumbnail: string,
  formatId?: string,
  onProgress?: DownloadCallback,
  platform?: string
): Promise<File> {
  // Check if already downloading
  const existing = activeDownloads.get(videoId);
  if (existing && existing.status === 'downloading') {
    throw new Error('Video is already downloading');
  }

  // Subscribe to updates if callback provided
  if (onProgress) {
    subscribeToDownload(videoId, onProgress);
  }

  const progress: DownloadProgress = {
    videoId,
    status: 'pending',
    progress: 0,
    title,
    thumbnail,
  };

  notifySubscribers(progress);

  try {
    // Check Native Helper connection
    if (!NativeHelperClient.isConnected()) {
      throw new Error('Native Helper not connected. Please start the helper application.');
    }

    progress.status = 'downloading';
    progress.progress = 5;
    notifySubscribers(progress);

    // Request download from Native Helper
    log.info(`Starting download: ${videoId} (${url})`);

    const result = await NativeHelperClient.download(url, formatId, (percent) => {
      progress.progress = 5 + (percent * 0.9); // 5% to 95%
      notifySubscribers(progress);
    });

    if (!result.success) {
      throw new Error(result.error || 'Download failed');
    }

    progress.status = 'processing';
    progress.progress = 95;
    notifySubscribers(progress);

    // Transfer file from helper via WebSocket
    log.debug(`Fetching file from helper: ${result.path}`);
    const fileResponse = await NativeHelperClient.getDownloadedFile(result.path!);
    if (!fileResponse) {
      throw new Error('Failed to read downloaded file from helper');
    }

    // Try to save to project's Downloads folder if a project is open
    let file: File;
    const blob = new Blob([fileResponse], { type: 'video/mp4' });
    if (projectFileService.isProjectOpen()) {
      const savedFile = await projectFileService.saveDownload(blob, title, platform || 'youtube');
      if (savedFile) {
        file = savedFile;
        log.info('Saved to project downloads folder');
      } else {
        // Fallback to in-memory file
        const sanitizedTitle = title.replace(/[^a-zA-Z0-9\s-]/g, '').substring(0, 100);
        file = new File([fileResponse], `${sanitizedTitle}.mp4`, { type: 'video/mp4' });
      }
    } else {
      // No project open, keep in memory
      const sanitizedTitle = title.replace(/[^a-zA-Z0-9\s-]/g, '').substring(0, 100);
      file = new File([fileResponse], `${sanitizedTitle}.mp4`, { type: 'video/mp4' });
      log.debug('No project open, file kept in memory only');
    }

    progress.status = 'complete';
    progress.progress = 100;
    progress.file = file;
    notifySubscribers(progress);

    return file;
  } catch (error) {
    progress.status = 'error';
    progress.error = (error as Error).message;
    notifySubscribers(progress);
    throw error;
  }
}

// Download video from YouTube (backward-compatible wrapper)
export async function downloadYouTubeVideo(
  videoId: string,
  title: string,
  thumbnail: string,
  formatId?: string,
  onProgress?: DownloadCallback
): Promise<File> {
  const youtubeUrl = `https://www.youtube.com/watch?v=${videoId}`;
  return downloadVideo(youtubeUrl, videoId, title, thumbnail, formatId, onProgress, 'youtube');
}

// Cancel a download
export function cancelDownload(videoId: string) {
  const progress = activeDownloads.get(videoId);
  if (progress && progress.status === 'downloading') {
    progress.status = 'error';
    progress.error = 'Download cancelled';
    notifySubscribers(progress);
    // TODO: Tell Native Helper to cancel
  }
}

// Clear completed/errored downloads from memory
export function clearDownload(videoId: string) {
  activeDownloads.delete(videoId);
  downloadCallbacks.delete(videoId);
}

// Get all active downloads
export function getActiveDownloads(): DownloadProgress[] {
  return Array.from(activeDownloads.values()).filter(
    d => d.status === 'pending' || d.status === 'downloading' || d.status === 'processing'
  );
}
