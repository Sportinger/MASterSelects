// AI Video Panel - AI video generation using PiAPI (Kling, Luma, Hailuo, etc.)
// Supports text-to-video and image-to-video generation with timeline integration

import { useState, useCallback, useRef, useEffect } from 'react';
import { Logger } from '../../services/logger';

const log = Logger.create('AIVideoPanel');
import { useSettingsStore } from '../../stores/settingsStore';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import {
  piApiService,
  getVideoProviders,
  getProvider,
  calculateCost,
  type VideoProvider,
  type VideoTask,
  type TextToVideoParams,
  type ImageToVideoParams,
  type AccountInfo,
} from '../../services/piApiService';
import { ImageCropper, exportCroppedImage, type CropData } from './ImageCropper';
import './AIVideoPanel.css';

type GenerationType = 'text-to-video' | 'image-to-video';
type PanelTab = 'generate' | 'history';

interface GenerationJob {
  id: string;
  type: GenerationType;
  provider: string;
  version: string;
  prompt: string;
  status: VideoTask['status'];
  progress?: number;
  videoUrl?: string;
  thumbnailUrl?: string;
  error?: string;
  createdAt: Date;
  completedAt?: Date;
  duration?: number;
  addedToTimeline?: boolean;
}

// Store history in localStorage
const HISTORY_KEY = 'piapi-generation-history';

function loadHistory(): GenerationJob[] {
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.map((job: GenerationJob) => ({
        ...job,
        createdAt: new Date(job.createdAt),
        completedAt: job.completedAt ? new Date(job.completedAt) : undefined,
      }));
    }
  } catch (e) {
    log.warn('Failed to load generation history', e);
  }
  return [];
}

function saveHistory(history: GenerationJob[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 50))); // Keep last 50
  } catch (e) {
    log.warn('Failed to save generation history', e);
  }
}

// Get or create AI Video folder in media panel
function getOrCreateAIVideoFolder(): string {
  const { folders, createFolder } = useMediaStore.getState();
  const existing = folders.find(f => f.name === 'AI Video');
  if (existing) return existing.id;
  const newFolder = createFolder('AI Video');
  return newFolder.id;
}

// Capture current frame from engine
async function captureCurrentFrame(): Promise<string | null> {
  try {
    // Dynamic import to avoid circular deps
    const { engine } = await import('../../engine/WebGPUEngine');
    if (!engine) return null;

    const pixels = await engine.readPixels();
    if (!pixels) return null;

    const { width, height } = engine.getOutputDimensions();

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
    ctx.putImageData(imageData, 0, 0);

    return canvas.toDataURL('image/png');
  } catch (e) {
    log.error('Failed to capture frame', e);
    return null;
  }
}

// Download video from URL and create File object
async function downloadVideoAsFile(url: string, filename: string): Promise<File | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch video');
    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type || 'video/mp4' });
  } catch (e) {
    log.error('Failed to download video', e);
    return null;
  }
}

// Get aspect ratio dimensions from string
function getAspectRatioDimensions(aspectRatio: string): { width: number; height: number } {
  const [w, h] = aspectRatio.split(':').map(Number);
  return { width: w || 16, height: h || 9 };
}

export function AIVideoPanel() {
  const { apiKeys, openSettings } = useSettingsStore();
  const { importFile } = useMediaStore();
  const { tracks, addClip, addTrack } = useTimelineStore();

  // Panel tab state
  const [activeTab, setActiveTab] = useState<PanelTab>('generate');

  // Provider and model selection
  const [providers] = useState<VideoProvider[]>(() => getVideoProviders());
  const [selectedProvider, setSelectedProvider] = useState<string>(providers[0]?.id || 'kling');
  const [selectedVersion, setSelectedVersion] = useState<string>(providers[0]?.versions[0] || '2.6');

  // Get current provider config
  const currentProvider = getProvider(selectedProvider) || providers[0];

  // Generation type (default to image-to-video)
  const [generationType, setGenerationType] = useState<GenerationType>('image-to-video');

  // Common parameters
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [duration, setDuration] = useState<number>(5);
  const [aspectRatio, setAspectRatio] = useState<string>('16:9');
  const [mode, setMode] = useState<string>('std');
  const [cfgScale, setCfgScale] = useState<number>(0.5);

  // Image-to-video specific
  const [startImagePreview, setStartImagePreview] = useState<string | null>(null);
  const [startCropData, setStartCropData] = useState<CropData>({ offsetX: 0, offsetY: 0, scale: 1 });
  const [endImagePreview, setEndImagePreview] = useState<string | null>(null);
  const [endCropData, setEndCropData] = useState<CropData>({ offsetX: 0, offsetY: 0, scale: 1 });

  // Get current aspect ratio dimensions
  const aspectDimensions = getAspectRatioDimensions(aspectRatio);

  // Timeline integration options
  const [addToTimeline, setAddToTimeline] = useState(true);

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [history, setHistory] = useState<GenerationJob[]>(() => loadHistory());
  const [error, setError] = useState<string | null>(null);

  // History playback
  const [playingVideoId, setPlayingVideoId] = useState<string | null>(null);
  const videoRefs = useRef<Map<string, HTMLVideoElement>>(new Map());

  // Account balance
  const [accountInfo, setAccountInfo] = useState<AccountInfo | null>(null);
  const [isLoadingBalance, setIsLoadingBalance] = useState(false);

  // Check if API credentials are available
  const hasApiKey = !!apiKeys.piapi;

  // Fetch account balance
  const fetchAccountBalance = useCallback(async () => {
    if (!apiKeys.piapi) return;

    setIsLoadingBalance(true);
    try {
      const info = await piApiService.getAccountInfo();
      setAccountInfo(info);
    } catch (err) {
      log.error('Failed to fetch account balance', err);
      // Don't show error to user, just log it
    } finally {
      setIsLoadingBalance(false);
    }
  }, [apiKeys.piapi]);

  // Set API key when it changes and fetch balance
  useEffect(() => {
    if (apiKeys.piapi) {
      piApiService.setApiKey(apiKeys.piapi);
      fetchAccountBalance();
    }
  }, [apiKeys.piapi, fetchAccountBalance]);

  // Update version when provider changes
  useEffect(() => {
    const provider = getProvider(selectedProvider);
    if (provider && !provider.versions.includes(selectedVersion)) {
      setSelectedVersion(provider.versions[0]);
    }
    // Reset mode if not supported
    if (provider && !provider.supportedModes.includes(mode)) {
      setMode(provider.supportedModes[0]);
    }
    // Reset duration if not supported
    if (provider && !provider.supportedDurations.includes(duration)) {
      setDuration(provider.supportedDurations[0]);
    }
  }, [selectedProvider, selectedVersion, mode, duration]);

  // Save history when it changes
  useEffect(() => {
    saveHistory(history);
  }, [history]);

  // Handle file drop for start image
  const handleStartDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => setStartImagePreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  }, []);

  // Handle file drop for end image
  const handleEndDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => setEndImagePreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  }, []);

  // Open file picker for start image
  const openStartFilePicker = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = () => setStartImagePreview(reader.result as string);
        reader.readAsDataURL(file);
      }
    };
    input.click();
  }, []);

  // Open file picker for end image
  const openEndFilePicker = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = () => setEndImagePreview(reader.result as string);
        reader.readAsDataURL(file);
      }
    };
    input.click();
  }, []);

  // Clear start image
  const clearStartImage = useCallback(() => {
    setStartImagePreview(null);
    setStartCropData({ offsetX: 0, offsetY: 0, scale: 1 });
  }, []);

  // Clear end image
  const clearEndImage = useCallback(() => {
    setEndImagePreview(null);
    setEndCropData({ offsetX: 0, offsetY: 0, scale: 1 });
  }, []);

  // Use current frame from timeline for start
  const useCurrentFrameStart = useCallback(async () => {
    const dataUrl = await captureCurrentFrame();
    if (dataUrl) {
      setStartImagePreview(dataUrl);
      setStartCropData({ offsetX: 0, offsetY: 0, scale: 1 });
    }
  }, []);

  // Use current frame from timeline for end
  const useCurrentFrameEnd = useCallback(async () => {
    const dataUrl = await captureCurrentFrame();
    if (dataUrl) {
      setEndImagePreview(dataUrl);
      setEndCropData({ offsetX: 0, offsetY: 0, scale: 1 });
    }
  }, []);

  // Export cropped image for API upload
  const getCroppedImageUrl = async (
    imagePreview: string,
    cropData: CropData
  ): Promise<string> => {
    return exportCroppedImage(imagePreview, cropData, aspectDimensions, 1280);
  };

  // Import video to media panel and optionally add to timeline
  const importVideoToProject = useCallback(async (job: GenerationJob) => {
    if (!job.videoUrl) return;

    try {
      // Download video file
      const filename = `${job.provider}_${job.id.slice(0, 8)}_${Date.now()}.mp4`;
      const file = await downloadVideoAsFile(job.videoUrl, filename);
      if (!file) {
        log.error('Failed to download video');
        return;
      }

      // Get or create AI Video folder
      const folderId = getOrCreateAIVideoFolder();

      // Import to media panel
      const mediaFile = await importFile(file);

      // Move to AI Video folder
      useMediaStore.getState().moveToFolder([mediaFile.id], folderId);

      log.info('Imported video to media panel', { name: mediaFile.name });

      // Add to timeline if option is enabled
      if (addToTimeline) {
        // Find an empty video track or create one
        const videoTracks = tracks.filter(t => t.type === 'video');
        let targetTrackId: string | null = null;

        // Try to find an empty video track
        for (const track of videoTracks) {
          const { clips } = useTimelineStore.getState();
          const trackClips = clips.filter(c => c.trackId === track.id);
          if (trackClips.length === 0) {
            targetTrackId = track.id;
            break;
          }
        }

        // If no empty track, create a new one
        if (!targetTrackId) {
          targetTrackId = addTrack('video');
        }

        // Add clip to timeline at playhead position
        const { playheadPosition } = useTimelineStore.getState();
        await addClip(targetTrackId, file, playheadPosition, job.duration, mediaFile.id);

        log.info('Added clip to timeline');

        // Update job to mark as added
        setJobs(prev => prev.map(j =>
          j.id === job.id ? { ...j, addedToTimeline: true } : j
        ));
        setHistory(prev => prev.map(h =>
          h.id === job.id ? { ...h, addedToTimeline: true } : h
        ));
      }
    } catch (err) {
      log.error('Failed to import video', err);
    }
  }, [importFile, tracks, addClip, addTrack, addToTimeline]);

  // Generate video
  const generateVideo = useCallback(async () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setError(null);

    try {
      let taskId: string;

      if (generationType === 'text-to-video') {
        const params: TextToVideoParams = {
          provider: selectedProvider,
          version: selectedVersion,
          prompt: prompt.trim(),
          negativePrompt: negativePrompt.trim() || undefined,
          duration,
          aspectRatio,
          mode,
          cfgScale,
        };

        taskId = await piApiService.createTextToVideo(params);
      } else {
        // Image-to-video - use cropped images
        const params: ImageToVideoParams = {
          provider: selectedProvider,
          version: selectedVersion,
          prompt: prompt.trim(),
          negativePrompt: negativePrompt.trim() || undefined,
          duration,
          aspectRatio,
          mode,
          cfgScale,
          startImageUrl: startImagePreview ? await getCroppedImageUrl(startImagePreview, startCropData) : undefined,
          endImageUrl: endImagePreview ? await getCroppedImageUrl(endImagePreview, endCropData) : undefined,
        };

        taskId = await piApiService.createImageToVideo(params);
      }

      // Add job to list
      const job: GenerationJob = {
        id: taskId,
        type: generationType,
        provider: selectedProvider,
        version: selectedVersion,
        prompt: prompt.trim(),
        status: 'pending',
        createdAt: new Date(),
        duration,
      };
      setJobs(prev => [job, ...prev]);

      // Poll for completion
      piApiService.pollTaskUntilComplete(taskId, (task) => {
        setJobs(prev => prev.map(j =>
          j.id === taskId
            ? {
              ...j,
              status: task.status,
              progress: task.progress,
              videoUrl: task.videoUrl,
              error: task.error,
              completedAt: task.status === 'completed' ? new Date() : undefined,
            }
            : j
        ));
      }).then(async (completedTask) => {
        if (completedTask.status === 'completed' && completedTask.videoUrl) {
          // Get the updated job
          const updatedJob = {
            ...job,
            status: completedTask.status,
            videoUrl: completedTask.videoUrl,
            completedAt: new Date(),
          };

          // Add to history
          setHistory(prev => [updatedJob, ...prev]);

          // Import to project
          await importVideoToProject(updatedJob);
        }
      }).catch(err => {
        setJobs(prev => prev.map(j =>
          j.id === taskId
            ? { ...j, status: 'failed', error: err.message }
            : j
        ));
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start generation');
    } finally {
      setIsGenerating(false);
    }
  }, [
    prompt, negativePrompt, selectedProvider, selectedVersion, duration, aspectRatio, mode, cfgScale,
    generationType, startImagePreview, startCropData, endImagePreview, endCropData, isGenerating,
    importVideoToProject, getCroppedImageUrl,
  ]);

  // Remove job from list
  const removeJob = useCallback((jobId: string) => {
    setJobs(prev => prev.filter(j => j.id !== jobId));
  }, []);

  // Remove from history
  const removeFromHistory = useCallback((jobId: string) => {
    setHistory(prev => prev.filter(h => h.id !== jobId));
  }, []);

  // Play/pause video in history
  const toggleVideoPlayback = useCallback((jobId: string) => {
    const video = videoRefs.current.get(jobId);
    if (!video) return;

    if (playingVideoId === jobId) {
      video.pause();
      setPlayingVideoId(null);
    } else {
      // Pause any currently playing
      if (playingVideoId) {
        const currentVideo = videoRefs.current.get(playingVideoId);
        currentVideo?.pause();
      }
      video.play();
      setPlayingVideoId(jobId);
    }
  }, [playingVideoId]);

  // Handle drag start for history item
  const handleHistoryDragStart = useCallback((e: React.DragEvent, job: GenerationJob) => {
    if (!job.videoUrl) return;
    e.dataTransfer.setData('text/plain', job.videoUrl);
    e.dataTransfer.setData('application/x-ai-video', JSON.stringify({
      id: job.id,
      prompt: job.prompt,
      videoUrl: job.videoUrl,
    }));
    e.dataTransfer.effectAllowed = 'copy';
  }, []);

  // Add history item to timeline
  const addHistoryToTimeline = useCallback(async (job: GenerationJob) => {
    if (!job.videoUrl) return;
    await importVideoToProject({ ...job });
  }, [importVideoToProject]);

  // Calculate current cost
  const currentCost = calculateCost(selectedProvider, mode, duration);

  return (
    <div className={`ai-video-panel ${!hasApiKey ? 'no-api-key' : ''}`}>
      {/* API Key Overlay */}
      {!hasApiKey && (
        <div className="ai-video-overlay">
          <div className="ai-video-overlay-content">
            <span className="no-key-icon">🎬</span>
            <p>PiAPI key required for AI video generation</p>
            <span className="no-key-hint">
              Access Kling, Luma, Hailuo, and more models
            </span>
            <button className="btn-settings" onClick={openSettings}>
              Open Settings
            </button>
          </div>
        </div>
      )}
      {/* Sub-tabs with provider dropdown */}
      <div className="panel-tabs-row">
        <div className="panel-tabs">
          <button
            className={`panel-tab ${activeTab === 'generate' ? 'active' : ''}`}
            onClick={() => setActiveTab('generate')}
          >
            AI Video
          </button>
          <button
            className={`panel-tab ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            History ({history.length})
          </button>
        </div>
        <select
          className="provider-select"
          value={selectedProvider}
          onChange={(e) => setSelectedProvider(e.target.value)}
          disabled={isGenerating}
        >
          {providers.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
      </div>

      {/* Content */}
      {activeTab === 'generate' ? (
        <div className="ai-video-content">
          {/* Generation Type Tabs */}
          <div className="generation-tabs">
            <button
              className={`tab ${generationType === 'text-to-video' ? 'active' : ''}`}
              onClick={() => setGenerationType('text-to-video')}
              disabled={isGenerating || !currentProvider?.supportsTextToVideo}
            >
              Text to Video
            </button>
            <button
              className={`tab ${generationType === 'image-to-video' ? 'active' : ''}`}
              onClick={() => setGenerationType('image-to-video')}
              disabled={isGenerating || !currentProvider?.supportsImageToVideo}
            >
              Image to Video
            </button>
          </div>

          {/* Image-to-Video: Aspect Ratio + Image Croppers */}
          {generationType === 'image-to-video' && (
            <>
              {/* Aspect Ratio Selection */}
              <div className="aspect-ratio-row">
                <label>Aspect Ratio</label>
                <div className="aspect-ratio-options">
                  {(currentProvider?.supportedAspectRatios || ['16:9', '9:16', '1:1']).map(ar => (
                    <button
                      key={ar}
                      className={`aspect-btn ${aspectRatio === ar ? 'active' : ''}`}
                      onClick={() => setAspectRatio(ar)}
                      disabled={isGenerating}
                    >
                      {ar}
                    </button>
                  ))}
                </div>
              </div>

              {/* Image Croppers */}
              <div className="image-inputs">
                <ImageCropper
                  label="Start Frame"
                  imageUrl={startImagePreview}
                  aspectRatio={aspectDimensions}
                  onClear={clearStartImage}
                  onCropChange={setStartCropData}
                  disabled={isGenerating}
                  onDropOrClick={openStartFilePicker}
                  onDrop={handleStartDrop}
                  onUseCurrentFrame={useCurrentFrameStart}
                />
                {selectedProvider === 'kling' && (
                  <ImageCropper
                    label="End Frame (optional)"
                    imageUrl={endImagePreview}
                    aspectRatio={aspectDimensions}
                    onClear={clearEndImage}
                    onCropChange={setEndCropData}
                    disabled={isGenerating}
                    onDropOrClick={openEndFilePicker}
                    onDrop={handleEndDrop}
                    onUseCurrentFrame={useCurrentFrameEnd}
                  />
                )}
              </div>
            </>
          )}

          {/* Prompt Input */}
          <div className="input-group">
            <label>Prompt</label>
            <textarea
              className="prompt-input"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the video you want to generate..."
              disabled={isGenerating}
              rows={3}
            />
          </div>

          {/* Negative Prompt */}
          <div className="input-group">
            <label>Negative Prompt (optional)</label>
            <textarea
              className="prompt-input negative"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="What to avoid in the generation..."
              disabled={isGenerating}
              rows={2}
            />
          </div>

          {/* Parameters Grid */}
          <div className="params-grid">
            {/* Version */}
            <div className="param-group">
              <label>Version</label>
              <select
                value={selectedVersion}
                onChange={(e) => setSelectedVersion(e.target.value)}
                disabled={isGenerating}
              >
                {currentProvider?.versions.map(v => (
                  <option key={v} value={v}>v{v}</option>
                ))}
              </select>
            </div>

            {/* Duration */}
            <div className="param-group">
              <label>Duration</label>
              <select
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                disabled={isGenerating}
              >
                {currentProvider?.supportedDurations.map(d => (
                  <option key={d} value={d}>{d} seconds</option>
                ))}
              </select>
            </div>

            {/* Aspect Ratio (only for text-to-video) */}
            {generationType === 'text-to-video' && (
              <div className="param-group">
                <label>Aspect Ratio</label>
                <select
                  value={aspectRatio}
                  onChange={(e) => setAspectRatio(e.target.value)}
                  disabled={isGenerating}
                >
                  {currentProvider?.supportedAspectRatios.map(ar => (
                    <option key={ar} value={ar}>{ar}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Mode */}
            {currentProvider?.supportedModes.length > 1 && (
              <div className="param-group">
                <label>Quality</label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  disabled={isGenerating}
                >
                  {currentProvider.supportedModes.map(m => (
                    <option key={m} value={m}>{m === 'std' ? 'Standard' : 'Professional'}</option>
                  ))}
                </select>
              </div>
            )}

            {/* CFG Scale (Kling only) */}
            {selectedProvider === 'kling' && (
              <div className="param-group cfg-slider">
                <label>CFG Scale: {cfgScale.toFixed(2)}</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={cfgScale}
                  onChange={(e) => setCfgScale(Number(e.target.value))}
                  disabled={isGenerating}
                />
              </div>
            )}
          </div>

          {/* Timeline Integration Option */}
          <label className="timeline-option">
            <input
              type="checkbox"
              checked={addToTimeline}
              onChange={(e) => setAddToTimeline(e.target.checked)}
              disabled={isGenerating}
            />
            <span>Add to timeline when complete</span>
          </label>

          {/* Credit Info */}
          <div className="credit-info">
            <div className="credit-balance">
              {accountInfo ? (
                <span className="balance-amount">
                  Balance: ${accountInfo.creditsUsd.toFixed(2)}
                </span>
              ) : (
                <span className="balance-loading">
                  {isLoadingBalance ? 'Loading...' : 'Balance: --'}
                </span>
              )}
              <button
                className="btn-refresh-balance"
                onClick={fetchAccountBalance}
                disabled={isLoadingBalance}
                title="Refresh balance"
              >
                {isLoadingBalance ? '...' : '↻'}
              </button>
            </div>
            <span className="credit-cost">
              Est. cost: ~${currentCost.toFixed(2)}
            </span>
          </div>

          {/* Generate Button */}
          <button
            className="btn-generate"
            onClick={generateVideo}
            disabled={isGenerating || !prompt.trim()}
          >
            {isGenerating ? 'Starting...' : `Generate (~$${currentCost.toFixed(2)})`}
          </button>

          {/* Error */}
          {error && (
            <div className="ai-video-error">
              <span className="error-icon">!</span>
              {error}
            </div>
          )}

          {/* Jobs List */}
          {jobs.length > 0 && (
            <div className="jobs-section">
              <h3>Generation Queue</h3>
              <div className="jobs-list">
                {jobs.map(job => (
                  <div key={job.id} className={`job-item ${job.status}`}>
                    <div className="job-header">
                      <span className="job-type">
                        {job.provider.toUpperCase()}
                      </span>
                      <span className={`job-status ${job.status}`}>
                        {job.status === 'pending' && 'Queued'}
                        {job.status === 'processing' && 'Processing...'}
                        {job.status === 'completed' && 'Done'}
                        {job.status === 'failed' && 'Failed'}
                      </span>
                      <button
                        className="btn-remove"
                        onClick={() => removeJob(job.id)}
                        title="Remove"
                      >
                        x
                      </button>
                    </div>
                    <div className="job-prompt">{job.prompt}</div>
                    {job.error && (
                      <div className="job-error">{job.error}</div>
                    )}
                    {job.videoUrl && (
                      <div className="job-result">
                        <video
                          src={job.videoUrl}
                          controls
                          preload="metadata"
                        />
                        <a
                          href={job.videoUrl}
                          download
                          className="btn-download"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Download
                        </a>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        /* History Tab */
        <div className="ai-video-history">
          {history.length === 0 ? (
            <div className="history-empty">
              <p>No generated videos yet</p>
              <span>Videos you generate will appear here</span>
            </div>
          ) : (
            <div className="history-list">
              {history.map(job => (
                <div
                  key={job.id}
                  className="history-item"
                  draggable={!!job.videoUrl}
                  onDragStart={(e) => handleHistoryDragStart(e, job)}
                >
                  <div className="history-preview">
                    {job.videoUrl ? (
                      <video
                        ref={(el) => {
                          if (el) videoRefs.current.set(job.id, el);
                          else videoRefs.current.delete(job.id);
                        }}
                        src={job.videoUrl}
                        preload="metadata"
                        muted
                        loop
                        onClick={() => toggleVideoPlayback(job.id)}
                        onEnded={() => setPlayingVideoId(null)}
                      />
                    ) : (
                      <div className="history-preview-placeholder">
                        {job.status === 'failed' ? 'Failed' : 'Processing...'}
                      </div>
                    )}
                    {job.videoUrl && (
                      <button
                        className="play-overlay"
                        onClick={() => toggleVideoPlayback(job.id)}
                      >
                        {playingVideoId === job.id ? '⏸' : '▶'}
                      </button>
                    )}
                  </div>
                  <div className="history-info">
                    <div className="history-prompt">{job.prompt}</div>
                    <div className="history-meta">
                      <span className="history-type">
                        {job.provider?.toUpperCase() || 'KLING'}
                      </span>
                      <span className="history-date">
                        {job.createdAt.toLocaleDateString()}
                      </span>
                      {job.addedToTimeline && (
                        <span className="history-added">In Timeline</span>
                      )}
                    </div>
                    <div className="history-actions">
                      {job.videoUrl && !job.addedToTimeline && (
                        <button
                          className="btn-add-timeline"
                          onClick={() => addHistoryToTimeline(job)}
                          title="Add to timeline"
                        >
                          + Timeline
                        </button>
                      )}
                      <button
                        className="btn-remove-history"
                        onClick={() => removeFromHistory(job.id)}
                        title="Remove from history"
                      >
                        x
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
