// AI Video Panel - AI video generation using Kling API
// Supports text-to-video and image-to-video generation with timeline integration

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useMediaStore } from '../../stores/mediaStore';
import { useTimelineStore } from '../../stores/timeline';
import {
  klingService,
  KLING_MODELS,
  KLING_DURATIONS,
  KLING_ASPECT_RATIOS,
  KLING_MODES,
  KLING_CAMERA_CONTROLS,
  type KlingTask,
  type TextToVideoParams,
  type ImageToVideoParams,
} from '../../services/klingService';
import './AIVideoPanel.css';

// Available AI video services
const AI_SERVICES = [
  { id: 'kling', name: 'Kling AI' },
] as const;

type GenerationType = 'text-to-video' | 'image-to-video';
type PanelTab = 'generate' | 'history';

interface GenerationJob {
  id: string;
  type: GenerationType;
  prompt: string;
  status: KlingTask['status'];
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
const HISTORY_KEY = 'kling-generation-history';

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
    console.warn('Failed to load generation history:', e);
  }
  return [];
}

function saveHistory(history: GenerationJob[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 50))); // Keep last 50
  } catch (e) {
    console.warn('Failed to save generation history:', e);
  }
}

// Get or create KlingAI folder in media panel
function getOrCreateKlingFolder(): string {
  const { folders, createFolder } = useMediaStore.getState();
  const existing = folders.find(f => f.name === 'KlingAI');
  if (existing) return existing.id;
  const newFolder = createFolder('KlingAI');
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

    const imageData = new ImageData(pixels, width, height);
    ctx.putImageData(imageData, 0, 0);

    return canvas.toDataURL('image/png');
  } catch (e) {
    console.error('Failed to capture frame:', e);
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
    console.error('Failed to download video:', e);
    return null;
  }
}

export function AIVideoPanel() {
  const { apiKeys, openSettings } = useSettingsStore();
  const { importFile, folders, files } = useMediaStore();
  const { tracks, addClip, createTrack } = useTimelineStore();

  // Panel tab state
  const [activeTab, setActiveTab] = useState<PanelTab>('generate');

  // Service and model selection
  const [service, setService] = useState<string>('kling');
  const [model, setModel] = useState<string>(KLING_MODELS[3].id); // Default to v2.0

  // Generation type
  const [generationType, setGenerationType] = useState<GenerationType>('text-to-video');

  // Common parameters
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [duration, setDuration] = useState<number>(5);
  const [aspectRatio, setAspectRatio] = useState<string>('16:9');
  const [mode, setMode] = useState<string>('std');
  const [cfgScale, setCfgScale] = useState<number>(0.5);
  const [cameraControl, setCameraControl] = useState<string>('');

  // Image-to-video specific
  const [startImage, setStartImage] = useState<File | null>(null);
  const [startImagePreview, setStartImagePreview] = useState<string | null>(null);
  const [endImage, setEndImage] = useState<File | null>(null);
  const [endImagePreview, setEndImagePreview] = useState<string | null>(null);

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

  // Refs for drag-drop
  const startImageRef = useRef<HTMLDivElement>(null);
  const endImageRef = useRef<HTMLDivElement>(null);

  // Check if API credentials are available
  const hasApiKey = !!apiKeys.klingAccessKey && !!apiKeys.klingSecretKey;

  // Set credentials when they change
  useEffect(() => {
    if (apiKeys.klingAccessKey && apiKeys.klingSecretKey) {
      klingService.setCredentials(apiKeys.klingAccessKey, apiKeys.klingSecretKey);
    }
  }, [apiKeys.klingAccessKey, apiKeys.klingSecretKey]);

  // Save history when it changes
  useEffect(() => {
    saveHistory(history);
  }, [history]);

  // Handle file drop
  const handleFileDrop = useCallback((
    e: React.DragEvent<HTMLDivElement>,
    setFile: (file: File | null) => void,
    setPreview: (url: string | null) => void
  ) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  // Handle file input change
  const handleFileChange = useCallback((
    e: React.ChangeEvent<HTMLInputElement>,
    setFile: (file: File | null) => void,
    setPreview: (url: string | null) => void
  ) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  // Clear image
  const clearImage = useCallback((
    setFile: (file: File | null) => void,
    setPreview: (url: string | null) => void
  ) => {
    setFile(null);
    setPreview(null);
  }, []);

  // Use current frame from timeline
  const useCurrentFrame = useCallback(async (
    setPreview: (url: string | null) => void
  ) => {
    const dataUrl = await captureCurrentFrame();
    if (dataUrl) {
      setPreview(dataUrl);
      // Convert data URL to File for upload
      const response = await fetch(dataUrl);
      const blob = await response.blob();
      const file = new File([blob], `frame-${Date.now()}.png`, { type: 'image/png' });
      if (setPreview === setStartImagePreview) {
        setStartImage(file);
      } else {
        setEndImage(file);
      }
    }
  }, []);

  // Upload image to get URL (returns data URL for now)
  const uploadImage = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // Import video to media panel and optionally add to timeline
  const importVideoToProject = useCallback(async (job: GenerationJob) => {
    if (!job.videoUrl) return;

    try {
      // Download video file
      const filename = `kling_${job.id.slice(0, 8)}_${Date.now()}.mp4`;
      const file = await downloadVideoAsFile(job.videoUrl, filename);
      if (!file) {
        console.error('Failed to download video');
        return;
      }

      // Get or create KlingAI folder
      const folderId = getOrCreateKlingFolder();

      // Import to media panel
      const mediaFile = await importFile(file);

      // Move to KlingAI folder
      useMediaStore.getState().moveToFolder([mediaFile.id], folderId);

      console.log('[AIVideo] Imported video to media panel:', mediaFile.name);

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
          const newTrack = createTrack('video', `Video ${videoTracks.length + 1}`);
          targetTrackId = newTrack.id;
        }

        // Add clip to timeline at playhead position
        const { playheadPosition } = useTimelineStore.getState();
        await addClip(targetTrackId, file, playheadPosition, job.duration, mediaFile.id);

        console.log('[AIVideo] Added clip to timeline');

        // Update job to mark as added
        setJobs(prev => prev.map(j =>
          j.id === job.id ? { ...j, addedToTimeline: true } : j
        ));
        setHistory(prev => prev.map(h =>
          h.id === job.id ? { ...h, addedToTimeline: true } : h
        ));
      }
    } catch (err) {
      console.error('Failed to import video:', err);
    }
  }, [importFile, tracks, addClip, createTrack, addToTimeline]);

  // Generate video
  const generateVideo = useCallback(async () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setError(null);

    try {
      let taskId: string;

      if (generationType === 'text-to-video') {
        const params: TextToVideoParams = {
          prompt: prompt.trim(),
          negativePrompt: negativePrompt.trim() || undefined,
          model,
          duration,
          aspectRatio,
          mode,
          cfgScale,
          cameraControl: cameraControl || undefined,
        };

        taskId = await klingService.createTextToVideo(params);
      } else {
        // Image-to-video
        const params: ImageToVideoParams = {
          prompt: prompt.trim(),
          negativePrompt: negativePrompt.trim() || undefined,
          model,
          duration,
          mode,
          cfgScale,
          startImageUrl: startImage ? await uploadImage(startImage) : undefined,
          endImageUrl: endImage ? await uploadImage(endImage) : undefined,
        };

        taskId = await klingService.createImageToVideo(params);
      }

      // Add job to list
      const job: GenerationJob = {
        id: taskId,
        type: generationType,
        prompt: prompt.trim(),
        status: 'pending',
        createdAt: new Date(),
        duration,
      };
      setJobs(prev => [job, ...prev]);

      // Poll for completion
      klingService.pollTaskUntilComplete(taskId, (task) => {
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
    prompt, negativePrompt, model, duration, aspectRatio, mode, cfgScale,
    cameraControl, generationType, startImage, endImage, isGenerating,
    importVideoToProject,
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
    e.dataTransfer.setData('application/x-kling-video', JSON.stringify({
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

  // Render empty state if no API credentials
  if (!hasApiKey) {
    return (
      <div className="ai-video-panel">
        <div className="ai-video-header">
          <h2>AI Video</h2>
        </div>
        <div className="ai-video-empty">
          <div className="ai-video-no-key">
            <span className="no-key-icon">🎬</span>
            <p>Kling Access Key + Secret Key required</p>
            <button className="btn-settings" onClick={openSettings}>
              Open Settings
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="ai-video-panel">
      {/* Sub-tabs with service dropdown */}
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
          className="service-select"
          value={service}
          onChange={(e) => setService(e.target.value)}
          disabled={isGenerating}
        >
          {AI_SERVICES.map(s => (
            <option key={s.id} value={s.id}>{s.name}</option>
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
              disabled={isGenerating}
            >
              Text to Video
            </button>
            <button
              className={`tab ${generationType === 'image-to-video' ? 'active' : ''}`}
              onClick={() => setGenerationType('image-to-video')}
              disabled={isGenerating}
            >
              Image to Video
            </button>
          </div>

          {/* Image Drop Zones (only for image-to-video) */}
          {generationType === 'image-to-video' && (
            <div className="image-inputs">
              <div className="image-input-group">
                <label>Start Frame</label>
                <div
                  ref={startImageRef}
                  className={`image-drop-zone ${startImagePreview ? 'has-image' : ''}`}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => handleFileDrop(e, setStartImage, setStartImagePreview)}
                  onClick={() => {
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.onchange = (e) => handleFileChange(
                      e as unknown as React.ChangeEvent<HTMLInputElement>,
                      setStartImage,
                      setStartImagePreview
                    );
                    input.click();
                  }}
                >
                  {startImagePreview ? (
                    <>
                      <img src={startImagePreview} alt="Start frame" />
                      <button
                        className="clear-image"
                        onClick={(e) => {
                          e.stopPropagation();
                          clearImage(setStartImage, setStartImagePreview);
                        }}
                      >
                        x
                      </button>
                    </>
                  ) : (
                    <span className="drop-hint">Drop or click</span>
                  )}
                </div>
                <button
                  className="btn-use-current"
                  onClick={() => useCurrentFrame(setStartImagePreview)}
                  disabled={isGenerating}
                >
                  Use Current Frame
                </button>
              </div>

              <div className="image-input-group">
                <label>End Frame (optional)</label>
                <div
                  ref={endImageRef}
                  className={`image-drop-zone ${endImagePreview ? 'has-image' : ''}`}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => handleFileDrop(e, setEndImage, setEndImagePreview)}
                  onClick={() => {
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.onchange = (e) => handleFileChange(
                      e as unknown as React.ChangeEvent<HTMLInputElement>,
                      setEndImage,
                      setEndImagePreview
                    );
                    input.click();
                  }}
                >
                  {endImagePreview ? (
                    <>
                      <img src={endImagePreview} alt="End frame" />
                      <button
                        className="clear-image"
                        onClick={(e) => {
                          e.stopPropagation();
                          clearImage(setEndImage, setEndImagePreview);
                        }}
                      >
                        x
                      </button>
                    </>
                  ) : (
                    <span className="drop-hint">Drop or click</span>
                  )}
                </div>
                <button
                  className="btn-use-current"
                  onClick={() => useCurrentFrame(setEndImagePreview)}
                  disabled={isGenerating}
                >
                  Use Current Frame
                </button>
              </div>
            </div>
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
            {/* Model */}
            <div className="param-group">
              <label>Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={isGenerating}
              >
                {KLING_MODELS.map(m => (
                  <option key={m.id} value={m.id}>{m.name}</option>
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
                {KLING_DURATIONS.map(d => (
                  <option key={d.value} value={d.value}>{d.label}</option>
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
                  {KLING_ASPECT_RATIOS.map(ar => (
                    <option key={ar.value} value={ar.value}>{ar.label}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Mode */}
            <div className="param-group">
              <label>Quality</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                disabled={isGenerating}
              >
                {KLING_MODES.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>

            {/* Camera Control (only for text-to-video) */}
            {generationType === 'text-to-video' && (
              <div className="param-group">
                <label>Camera</label>
                <select
                  value={cameraControl}
                  onChange={(e) => setCameraControl(e.target.value)}
                  disabled={isGenerating}
                >
                  {KLING_CAMERA_CONTROLS.map(c => (
                    <option key={c.value} value={c.value}>{c.label}</option>
                  ))}
                </select>
              </div>
            )}

            {/* CFG Scale */}
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

          {/* Generate Button */}
          <button
            className="btn-generate"
            onClick={generateVideo}
            disabled={isGenerating || !prompt.trim()}
          >
            {isGenerating ? 'Starting...' : 'Generate Video'}
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
                        {job.type === 'text-to-video' ? 'T2V' : 'I2V'}
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
                        {job.type === 'text-to-video' ? 'T2V' : 'I2V'}
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
