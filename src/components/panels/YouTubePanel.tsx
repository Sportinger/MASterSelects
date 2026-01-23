// YouTube Search Panel
// Supports YouTube Data API (with key) and direct URL paste (no key)

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useTimelineStore } from '../../stores/timeline';
import { useYouTubeStore, type YouTubeVideo as StoreYouTubeVideo } from '../../stores/youtubeStore';
import { downloadYouTubeVideo, subscribeToDownload, isDownloadAvailable, type DownloadProgress } from '../../services/youtubeDownloader';
import { NativeHelperClient } from '../../services/nativeHelper';
import type { VideoInfo } from '../../services/nativeHelper';
import './YouTubePanel.css';

interface YouTubeVideo {
  id: string;
  title: string;
  thumbnail: string;
  channel: string;
  duration: string;
  durationSeconds: number;
  views?: string;
}

// Convert store video to panel format
function storeToPanel(v: StoreYouTubeVideo): YouTubeVideo {
  return {
    id: v.id,
    title: v.title,
    thumbnail: v.thumbnail,
    channel: v.channelTitle,
    duration: v.duration || '?:??',
    durationSeconds: v.durationSeconds || 0,
    views: v.viewCount,
  };
}

// Convert panel video to store format
function panelToStore(v: YouTubeVideo): StoreYouTubeVideo {
  return {
    id: v.id,
    title: v.title,
    thumbnail: v.thumbnail,
    channelTitle: v.channel,
    publishedAt: new Date().toISOString(),
    duration: v.duration,
    durationSeconds: v.durationSeconds,
    viewCount: v.views,
  };
}

// Extract video ID from various YouTube URL formats
function extractVideoId(input: string): string | null {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})/,
    /^([a-zA-Z0-9_-]{11})$/, // Just the ID
  ];

  for (const pattern of patterns) {
    const match = input.match(pattern);
    if (match) return match[1];
  }
  return null;
}

// Get video info via YouTube oEmbed (supports CORS!)
async function getVideoInfo(videoId: string): Promise<YouTubeVideo | null> {
  try {
    const response = await fetch(
      `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`
    );
    if (!response.ok) return null;

    const data = await response.json();
    return {
      id: videoId,
      title: data.title || 'Untitled',
      thumbnail: `https://i.ytimg.com/vi/${videoId}/mqdefault.jpg`,
      channel: data.author_name || 'Unknown',
      duration: '?:??', // oEmbed doesn't provide duration
      durationSeconds: 0,
    };
  } catch {
    return null;
  }
}

// Format selection dialog
function FormatDialog({
  videoInfo,
  onSelect,
  onCancel,
}: {
  videoInfo: VideoInfo;
  onSelect: (formatId: string) => void;
  onCancel: () => void;
}) {
  return (
    <div className="format-dialog-backdrop" onClick={onCancel}>
      <div className="format-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="format-dialog-header">
          <h3>Select Quality</h3>
          <button className="format-dialog-close" onClick={onCancel}>×</button>
        </div>

        <div className="format-dialog-video">
          <img src={videoInfo.thumbnail} alt={videoInfo.title} />
          <div className="format-dialog-info">
            <span className="format-dialog-title">{videoInfo.title}</span>
            <span className="format-dialog-uploader">{videoInfo.uploader}</span>
          </div>
        </div>

        <div className="format-dialog-options">
          {videoInfo.recommendations.map((format) => (
            <button
              key={format.id}
              className="format-option"
              onClick={() => onSelect(format.id)}
            >
              <span className="format-label">{format.label}</span>
              <span className="format-details">
                {format.vcodec && <span className="format-codec">{format.vcodec.split('.')[0]}</span>}
                {format.acodec && <span className="format-codec">{format.acodec.split('.')[0]}</span>}
              </span>
            </button>
          ))}
        </div>

        <button className="format-dialog-cancel" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  );
}

export function YouTubePanel() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draggingVideo, setDraggingVideo] = useState<string | null>(null);
  const [autoDownload, setAutoDownload] = useState(false);
  const [downloadingVideos, setDownloadingVideos] = useState<Set<string>>(new Set());
  const [helperConnected, setHelperConnected] = useState(isDownloadAvailable());

  // YouTube store - videos persist with project
  const storeVideos = useYouTubeStore(s => s.videos);
  const addVideos = useYouTubeStore(s => s.addVideos);
  const _removeVideo = useYouTubeStore(s => s.removeVideo); // For future use
  const _clearVideos = useYouTubeStore(s => s.clearVideos); // For future use
  void _removeVideo; void _clearVideos; // Suppress unused warnings

  // Convert store videos to panel format
  const results = storeVideos.map(storeToPanel);

  // Format selection state
  const [formatDialog, setFormatDialog] = useState<{
    video: YouTubeVideo;
    info: VideoInfo;
    mode: 'download' | 'timeline';
  } | null>(null);
  const [fetchingFormats, setFetchingFormats] = useState<string | null>(null);

  const { apiKeys, openSettings } = useSettingsStore();
  const youtubeApiKey = apiKeys.youtube || '';

  // Track Native Helper connection status
  useEffect(() => {
    const unsubscribe = NativeHelperClient.onStatusChange((status) => {
      setHelperConnected(status === 'connected');
    });
    return unsubscribe;
  }, []);

  // Timeline store actions
  const addPendingDownloadClip = useTimelineStore(s => s.addPendingDownloadClip);
  const updateDownloadProgress = useTimelineStore(s => s.updateDownloadProgress);
  const completeDownload = useTimelineStore(s => s.completeDownload);
  const setDownloadError = useTimelineStore(s => s.setDownloadError);
  const tracks = useTimelineStore(s => s.tracks);
  const playheadPosition = useTimelineStore(s => s.playheadPosition);

  // Track active downloads
  const activeDownloadsRef = useRef<Set<string>>(new Set());

  // Format duration from seconds
  const formatDuration = (seconds: number): string => {
    if (!seconds) return '?:??';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Parse ISO 8601 duration
  const parseISO8601Duration = (duration: string): number => {
    const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
    if (!match) return 0;
    return parseInt(match[1] || '0') * 3600 + parseInt(match[2] || '0') * 60 + parseInt(match[3] || '0');
  };

  // Format view count
  const formatViews = (count: number): string => {
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M views`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K views`;
    return `${count} views`;
  };

  // Search using YouTube Data API
  const searchYouTubeAPI = async (searchQuery: string): Promise<YouTubeVideo[]> => {
    const searchResponse = await fetch(
      `https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=20&q=${encodeURIComponent(searchQuery)}&key=${youtubeApiKey}`
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

    return searchData.items.map((item: any) => {
      const details = detailsMap.get(item.id.videoId) as any;
      const durationSeconds = details?.contentDetails?.duration
        ? parseISO8601Duration(details.contentDetails.duration)
        : 0;
      return {
        id: item.id.videoId,
        title: item.snippet.title,
        thumbnail: item.snippet.thumbnails.medium?.url || item.snippet.thumbnails.default?.url,
        channel: item.snippet.channelTitle,
        durationSeconds,
        duration: formatDuration(durationSeconds),
        views: details?.statistics?.viewCount
          ? formatViews(parseInt(details.statistics.viewCount))
          : undefined,
      };
    });
  };

  // Main search/add handler
  const handleSearch = useCallback(async (overrideQuery?: string) => {
    const input = (overrideQuery ?? query).trim();
    if (!input) return;

    setLoading(true);
    setError(null);

    try {
      // Check if it's a YouTube URL or video ID
      const videoId = extractVideoId(input);

      if (videoId) {
        // Direct video URL/ID - get info and show it
        const videoInfo = await getVideoInfo(videoId);
        if (videoInfo) {
          // Add to store (handles duplicates)
          addVideos([panelToStore(videoInfo)]);
          // Clear input for next paste
          setQuery('');
          // Auto-download if enabled
          if (autoDownload) {
            downloadVideoOnly(videoInfo);
          }
        } else {
          setError('Could not load video info');
        }
      } else if (youtubeApiKey) {
        // Search query with API key
        const videos = await searchYouTubeAPI(input);
        // Add all to store
        addVideos(videos.map(panelToStore));
      } else {
        // No API key and not a URL
        setError('Paste a YouTube URL, or add API key in settings for search');
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [query, youtubeApiKey, autoDownload]);

  // Handle Enter key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  // Handle paste - auto-add if it's a YouTube URL
  const handlePaste = (e: React.ClipboardEvent) => {
    const pastedText = e.clipboardData.getData('text').trim();
    const videoId = extractVideoId(pastedText);

    if (videoId) {
      e.preventDefault();
      setQuery(pastedText);
      // Trigger search immediately with pasted text
      handleSearch(pastedText);
    }
  };

  // Open video in new tab
  const openVideo = (videoId: string) => {
    window.open(`https://www.youtube.com/watch?v=${videoId}`, '_blank');
  };

  // Copy video URL
  const copyVideoUrl = (videoId: string) => {
    navigator.clipboard.writeText(`https://www.youtube.com/watch?v=${videoId}`);
  };

  // Drag handlers
  const handleDragStart = (e: React.DragEvent, video: YouTubeVideo) => {
    e.dataTransfer.setData('application/x-youtube-video', JSON.stringify(video));
    e.dataTransfer.effectAllowed = 'copy';
    setDraggingVideo(video.id);
  };

  const handleDragEnd = () => setDraggingVideo(null);

  // Show format selection dialog before download
  const showFormatDialog = async (video: YouTubeVideo, mode: 'download' | 'timeline') => {
    if (fetchingFormats) return;

    setFetchingFormats(video.id);
    setError(null);

    try {
      const url = `https://www.youtube.com/watch?v=${video.id}`;
      const info = await NativeHelperClient.listFormats(url);

      if (info && info.recommendations.length > 0) {
        setFormatDialog({ video, info, mode });
      } else {
        // Fallback: download with default format
        if (mode === 'download') {
          await executeDownload(video);
        } else {
          await executeAddToTimeline(video);
        }
      }
    } catch (err) {
      setError(`Failed to fetch formats: ${(err as Error).message}`);
    } finally {
      setFetchingFormats(null);
    }
  };

  // Handle format selection from dialog
  const handleFormatSelect = async (formatId: string) => {
    if (!formatDialog) return;

    const { video, mode } = formatDialog;
    setFormatDialog(null);

    if (mode === 'download') {
      await executeDownload(video, formatId);
    } else {
      await executeAddToTimeline(video, formatId);
    }
  };

  // Download video only (save to downloads folder)
  const downloadVideoOnly = async (video: YouTubeVideo) => {
    if (downloadingVideos.has(video.id)) return;
    await showFormatDialog(video, 'download');
  };

  // Execute download with optional format
  const executeDownload = async (video: YouTubeVideo, formatId?: string) => {
    if (downloadingVideos.has(video.id)) return;

    setDownloadingVideos(prev => new Set(prev).add(video.id));

    const unsubscribe = subscribeToDownload(video.id, (_progress: DownloadProgress) => {
      // Progress tracked via downloadingVideos set
    });

    try {
      const file = await downloadYouTubeVideo(video.id, video.title, video.thumbnail, formatId);
      // Trigger browser download
      const url = URL.createObjectURL(file);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${video.title.replace(/[^a-zA-Z0-9 ]/g, '')}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Download failed: ${(err as Error).message}`);
    } finally {
      setDownloadingVideos(prev => {
        const next = new Set(prev);
        next.delete(video.id);
        return next;
      });
      unsubscribe();
    }
  };

  // Add video to timeline - show format dialog first
  const addVideoToTimeline = async (video: YouTubeVideo) => {
    if (activeDownloadsRef.current.has(video.id)) return;
    await showFormatDialog(video, 'timeline');
  };

  // Execute add to timeline with optional format
  const executeAddToTimeline = async (video: YouTubeVideo, formatId?: string) => {
    if (activeDownloadsRef.current.has(video.id)) return;

    const videoTrack = tracks.find(t => t.type === 'video');
    if (!videoTrack) {
      setError('No video track available');
      return;
    }

    const clipId = addPendingDownloadClip(
      videoTrack.id,
      playheadPosition,
      video.id,
      video.title,
      video.thumbnail,
      video.durationSeconds || 60
    );

    if (!clipId) {
      setError('Failed to add clip');
      return;
    }

    activeDownloadsRef.current.add(video.id);

    const unsubscribe = subscribeToDownload(video.id, (progress: DownloadProgress) => {
      if (progress.status === 'downloading' || progress.status === 'processing') {
        updateDownloadProgress(clipId, progress.progress);
      }
    });

    try {
      const file = await downloadYouTubeVideo(video.id, video.title, video.thumbnail, formatId);
      await completeDownload(clipId, file);
    } catch (err) {
      setDownloadError(clipId, (err as Error).message);
    } finally {
      activeDownloadsRef.current.delete(video.id);
      unsubscribe();
    }
  };

  // Auto-download when video is added to results (reserved for future use)
  // const handleVideoAdded = async (video: YouTubeVideo) => {
  //   if (autoDownload && !downloadingVideos.has(video.id)) {
  //     downloadVideoOnly(video);
  //   }
  // };

  return (
    <div className="youtube-panel">
      <div className="youtube-header">
        <div className="youtube-search-row">
          <input
            type="text"
            className="youtube-search-input"
            placeholder={youtubeApiKey ? "Search or paste URL..." : "Paste YouTube URL..."}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
          />
          <button
            className="youtube-search-btn"
            onClick={() => handleSearch()}
            disabled={loading || !query.trim()}
          >
            {loading ? '...' : youtubeApiKey ? 'Search' : 'Add'}
          </button>
        </div>

        <div className="youtube-options">
          {youtubeApiKey ? (
            <span className="api-status api-active">API Key Active</span>
          ) : (
            <>
              <span className="api-status">No API Key</span>
              <button className="btn-settings-small" onClick={openSettings}>
                Add Key
              </button>
            </>
          )}
          {helperConnected ? (
            <span className="api-status api-active">yt-dlp Ready</span>
          ) : (
            <span className="api-status api-warning">No Helper</span>
          )}
          <label className="auto-download-toggle" title={helperConnected ? "Auto-download when URL pasted" : "Native Helper required"}>
            <input
              type="checkbox"
              checked={autoDownload}
              onChange={(e) => setAutoDownload(e.target.checked)}
              disabled={!helperConnected}
            />
            <span>Auto Download</span>
          </label>
        </div>

        {!helperConnected && (
          <div className="youtube-hint youtube-hint-warning">
            Native Helper required for downloads. Start helper or use yt-dlp manually.
          </div>
        )}
        {!youtubeApiKey && helperConnected && (
          <div className="youtube-hint">
            Paste YouTube URLs to add videos. Add API key for search.
          </div>
        )}
      </div>

      {error && (
        <div className="youtube-error">
          <span className="error-icon">!</span>
          {error}
        </div>
      )}

      <div className="youtube-results">
        {loading ? (
          <div className="youtube-loading">
            <div className="loading-spinner" />
            <span>Loading...</span>
          </div>
        ) : results.length > 0 ? (
          <div className="youtube-grid">
            {results.map((video) => (
              <div
                key={video.id}
                className={`youtube-video-card ${draggingVideo === video.id ? 'dragging' : ''} ${downloadingVideos.has(video.id) ? 'downloading' : ''}`}
                draggable
                onDragStart={(e) => handleDragStart(e, video)}
                onDragEnd={handleDragEnd}
                onClick={() => openVideo(video.id)}
              >
                <div className="video-thumbnail">
                  <img src={video.thumbnail} alt={video.title} loading="lazy" draggable={false} />
                  {/* Text overlay on thumbnail */}
                  <div className="video-info-overlay">
                    <h4 className="video-title">{video.title}</h4>
                    <span className="video-channel">{video.channel}</span>
                  </div>
                  <span className="video-duration">{video.duration}</span>
                  {/* Action buttons */}
                  <div className="video-actions">
                    <button
                      className={`btn-download ${!helperConnected ? 'disabled' : ''}`}
                      onClick={(e) => { e.stopPropagation(); downloadVideoOnly(video); }}
                      title={helperConnected ? "Download video" : "Native Helper required for download"}
                      disabled={downloadingVideos.has(video.id) || fetchingFormats === video.id || !helperConnected}
                    >
                      {downloadingVideos.has(video.id) || fetchingFormats === video.id ? '...' : '↓'}
                    </button>
                    <button
                      className="btn-add-timeline"
                      onClick={(e) => { e.stopPropagation(); addVideoToTimeline(video); }}
                      title="Add to timeline"
                    >
                      +
                    </button>
                    <button
                      className="btn-copy-url"
                      onClick={(e) => { e.stopPropagation(); copyVideoUrl(video.id); }}
                      title="Copy URL"
                    >
                      Copy
                    </button>
                  </div>
                  {/* Downloading indicator */}
                  {downloadingVideos.has(video.id) && (
                    <div className="download-overlay">
                      <div className="download-spinner" />
                      <span>Downloading...</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="youtube-empty">
            <span className="youtube-icon">YouTube</span>
            {youtubeApiKey ? (
              <>
                <p>Search for videos</p>
                <span>Or paste a YouTube URL</span>
              </>
            ) : (
              <>
                <p>Paste a YouTube URL</p>
                <span>e.g. youtube.com/watch?v=...</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Format selection dialog */}
      {formatDialog && (
        <FormatDialog
          videoInfo={formatDialog.info}
          onSelect={handleFormatSelect}
          onCancel={() => setFormatDialog(null)}
        />
      )}
    </div>
  );
}
