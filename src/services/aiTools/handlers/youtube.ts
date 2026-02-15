// YouTube AI Tool Handlers

import { Logger } from '../../logger';
import { NativeHelperClient } from '../../nativeHelper';
import { downloadVideo } from '../../youtubeDownloader';
import { useYouTubeStore } from '../../../stores/youtubeStore';
import { useTimelineStore } from '../../../stores/timeline';
import { useSettingsStore } from '../../../stores/settingsStore';
import type { ToolResult } from '../types';

const log = Logger.create('AITool:YouTube');

// --- Helpers (replicated from DownloadPanel) ---

function parseISO8601Duration(duration: string): number {
  const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
  if (!match) return 0;
  return parseInt(match[1] || '0') * 3600 + parseInt(match[2] || '0') * 60 + parseInt(match[3] || '0');
}

function formatDuration(seconds: number): string {
  if (!seconds) return '?:??';
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatViews(count: number): string {
  if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M views`;
  if (count >= 1000) return `${(count / 1000).toFixed(1)}K views`;
  return `${count} views`;
}

// --- Handlers ---

export async function handleSearchYouTube(args: Record<string, unknown>): Promise<ToolResult> {
  const query = args.query as string;
  const maxResults = Math.min(Math.max((args.maxResults as number) || 10, 1), 20);

  if (!query) {
    return { success: false, error: 'query is required' };
  }

  const youtubeApiKey = useSettingsStore.getState().apiKeys.youtube;
  if (!youtubeApiKey) {
    return { success: false, error: 'YouTube API key not configured. Please set it in Settings > API Keys.' };
  }

  try {
    // Search YouTube Data API v3
    const searchResponse = await fetch(
      `https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=${maxResults}&q=${encodeURIComponent(query)}&key=${youtubeApiKey}`
    );

    if (!searchResponse.ok) {
      const errorData = await searchResponse.json();
      throw new Error(errorData.error?.message || 'YouTube API error');
    }

    const searchData = await searchResponse.json();
    const videoIds = searchData.items.map((item: any) => item.id.videoId).join(',');

    // Get video details (duration, views)
    const detailsResponse = await fetch(
      `https://www.googleapis.com/youtube/v3/videos?part=contentDetails,statistics&id=${videoIds}&key=${youtubeApiKey}`
    );

    const detailsData = await detailsResponse.json();
    const detailsMap = new Map(detailsData.items?.map((item: any) => [item.id, item]) || []);

    const videos = searchData.items.map((item: any) => {
      const details = detailsMap.get(item.id.videoId) as any;
      const durationSeconds = details?.contentDetails?.duration
        ? parseISO8601Duration(details.contentDetails.duration)
        : 0;
      return {
        id: item.id.videoId,
        title: item.snippet.title,
        thumbnail: item.snippet.thumbnails.medium?.url || item.snippet.thumbnails.default?.url,
        channelTitle: item.snippet.channelTitle,
        publishedAt: item.snippet.publishedAt,
        durationSeconds,
        duration: formatDuration(durationSeconds),
        viewCount: details?.statistics?.viewCount
          ? formatViews(parseInt(details.statistics.viewCount))
          : undefined,
        url: `https://www.youtube.com/watch?v=${item.id.videoId}`,
      };
    });

    // Add results to YouTube store (appears in Downloads panel)
    useYouTubeStore.getState().addVideos(videos);
    useYouTubeStore.getState().setLastQuery(query);

    log.info(`YouTube search: "${query}" returned ${videos.length} results`);

    return {
      success: true,
      data: {
        query,
        resultCount: videos.length,
        videos,
      },
    };
  } catch (error) {
    log.error('YouTube search failed', error);
    return {
      success: false,
      error: `YouTube search failed: ${(error as Error).message}`,
    };
  }
}

export async function handleListVideoFormats(args: Record<string, unknown>): Promise<ToolResult> {
  let url = args.url as string;

  if (!url) {
    return { success: false, error: 'url is required' };
  }

  // If just a video ID, convert to YouTube URL
  if (/^[a-zA-Z0-9_-]{11}$/.test(url)) {
    url = `https://www.youtube.com/watch?v=${url}`;
  }

  if (!NativeHelperClient.isConnected()) {
    return { success: false, error: 'Native Helper not connected. Please start the helper application and enable Turbo Mode in settings.' };
  }

  try {
    const info = await NativeHelperClient.listFormats(url);

    if (!info) {
      return { success: false, error: 'Failed to get video info. The URL may be invalid or unsupported.' };
    }

    log.info(`Listed formats for: ${info.title} (${info.recommendations.length} recommendations, ${info.allFormats.length} formats)`);

    return {
      success: true,
      data: {
        title: info.title,
        duration: info.duration,
        uploader: info.uploader,
        platform: info.platform,
        thumbnail: info.thumbnail,
        recommendations: info.recommendations.map(r => ({
          id: r.id,
          label: r.label,
          resolution: r.resolution,
          videoCodec: r.vcodec,
          audioCodec: r.acodec,
          needsMerge: r.needsMerge,
        })),
        allFormats: info.allFormats.map(f => ({
          formatId: f.format_id,
          ext: f.ext,
          resolution: f.resolution,
          fps: f.fps,
          videoCodec: f.vcodec,
          audioCodec: f.acodec,
          filesize: f.filesize,
          bitrate: f.tbr,
          note: f.format_note,
          hasVideo: f.hasVideo,
          hasAudio: f.hasAudio,
        })),
      },
    };
  } catch (error) {
    log.error('List formats failed', error);
    return {
      success: false,
      error: `Failed to list formats: ${(error as Error).message}`,
    };
  }
}

export async function handleDownloadAndImportVideo(args: Record<string, unknown>): Promise<ToolResult> {
  const url = args.url as string;
  const title = args.title as string;
  const formatId = args.formatId as string | undefined;
  const thumbnail = (args.thumbnail as string) || '';

  if (!url) {
    return { success: false, error: 'url is required' };
  }
  if (!title) {
    return { success: false, error: 'title is required' };
  }

  if (!NativeHelperClient.isConnected()) {
    return { success: false, error: 'Native Helper not connected. Please start the helper application and enable Turbo Mode in settings.' };
  }

  // Extract video ID for tracking
  const videoIdMatch = url.match(/(?:v=|\/)([\w-]{11})(?:\?|&|$)/);
  const videoId = videoIdMatch ? videoIdMatch[1] : url.replace(/[^a-zA-Z0-9]/g, '').substring(0, 20);

  const timelineStore = useTimelineStore.getState();

  // Find or create a video track
  let videoTrack = timelineStore.tracks.find(t => t.type === 'video');
  if (!videoTrack) {
    timelineStore.addTrack('video');
    videoTrack = useTimelineStore.getState().tracks.find(t => t.type === 'video');
    if (!videoTrack) {
      return { success: false, error: 'Failed to create video track' };
    }
  }

  // Add pending clip at end of timeline
  const startTime = timelineStore.duration || 0;
  const clipId = timelineStore.addPendingDownloadClip(
    videoTrack.id,
    startTime,
    videoId,
    title,
    thumbnail,
    30 // estimated duration
  );

  if (!clipId) {
    return { success: false, error: 'Failed to create pending download clip' };
  }

  log.info(`Starting download: ${title} (${url}), clipId: ${clipId}`);

  try {
    // Download and wait for completion
    const file = await downloadVideo(
      url,
      videoId,
      title,
      thumbnail,
      formatId,
      (progress) => {
        // Update pending clip progress
        const store = useTimelineStore.getState();
        if (progress.status === 'downloading' || progress.status === 'processing') {
          store.updateDownloadProgress(clipId, progress.progress);
        } else if (progress.status === 'error') {
          store.setDownloadError(clipId, progress.error || 'Download failed');
        }
      }
    );

    // Complete the download — convert pending clip to real clip
    await useTimelineStore.getState().completeDownload(clipId, file);

    log.info(`Download complete: ${title}, file size: ${(file.size / 1024 / 1024).toFixed(1)}MB`);

    return {
      success: true,
      data: {
        clipId,
        title,
        fileName: file.name,
        fileSize: file.size,
        message: `Video "${title}" downloaded and imported to timeline.`,
      },
    };
  } catch (error) {
    log.error('Download failed', error);
    // Mark clip as errored
    useTimelineStore.getState().setDownloadError(clipId, (error as Error).message);
    return {
      success: false,
      error: `Download failed: ${(error as Error).message}`,
    };
  }
}

export async function handleGetYouTubeVideos(): Promise<ToolResult> {
  const { videos } = useYouTubeStore.getState();

  return {
    success: true,
    data: {
      videoCount: videos.length,
      videos: videos.map(v => ({
        id: v.id,
        title: v.title,
        channel: v.channelTitle,
        duration: v.duration,
        durationSeconds: v.durationSeconds,
        views: v.viewCount,
        platform: v.platform || 'youtube',
        url: v.sourceUrl || `https://www.youtube.com/watch?v=${v.id}`,
        thumbnail: v.thumbnail,
      })),
    },
  };
}
