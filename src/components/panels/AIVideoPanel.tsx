// AI Video Panel - AI video generation using Kling API
// Supports text-to-video and image-to-video generation

import { useState, useCallback, useRef, useEffect } from 'react';
import { useSettingsStore } from '../../stores/settingsStore';
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

interface GenerationJob {
  id: string;
  type: GenerationType;
  prompt: string;
  status: KlingTask['status'];
  progress?: number;
  videoUrl?: string;
  error?: string;
  createdAt: Date;
}

export function AIVideoPanel() {
  const { apiKeys, openSettings } = useSettingsStore();

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

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Refs for drag-drop
  const startImageRef = useRef<HTMLDivElement>(null);
  const endImageRef = useRef<HTMLDivElement>(null);

  // Check if API key is available
  const hasApiKey = !!apiKeys.kling;

  // Set API key when it changes
  useEffect(() => {
    if (apiKeys.kling) {
      klingService.setApiKey(apiKeys.kling);
    }
  }, [apiKeys.kling]);

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

  // Upload image to get URL (placeholder - in production, upload to your server/S3)
  const uploadImage = async (file: File): Promise<string> => {
    // For now, return data URL - in production, upload to server
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

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
            }
            : j
        ));
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
  ]);

  // Remove job from list
  const removeJob = useCallback((jobId: string) => {
    setJobs(prev => prev.filter(j => j.id !== jobId));
  }, []);

  // Render empty state if no API key
  if (!hasApiKey) {
    return (
      <div className="ai-video-panel">
        <div className="ai-video-header">
          <h2>AI Video</h2>
        </div>
        <div className="ai-video-empty">
          <div className="ai-video-no-key">
            <span className="no-key-icon">🎬</span>
            <p>Kling API key required</p>
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
      {/* Header */}
      <div className="ai-video-header">
        <h2>AI Video</h2>
        <div className="ai-video-controls">
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
      </div>

      {/* Content */}
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
    </div>
  );
}
