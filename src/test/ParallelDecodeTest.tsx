/**
 * Parallel Decode Test Page
 * Access via: http://localhost:5173/test/parallel-decode
 *
 * This page tests the ParallelDecodeManager in isolation
 * to debug export issues without the full app.
 */

import { useState, useRef } from 'react';
import { ParallelDecodeManager } from '../engine/ParallelDecodeManager';

interface LogEntry {
  time: string;
  message: string;
  type: 'info' | 'success' | 'error' | 'warn';
}

interface LoadedFile {
  name: string;
  arrayBuffer: ArrayBuffer;
  video: HTMLVideoElement;
  duration: number;
  width: number;
  height: number;
}

export function ParallelDecodeTest() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [files, setFiles] = useState<LoadedFile[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const decoderRef = useRef<ParallelDecodeManager | null>(null);

  // Local test videos (served from public folder - no CORS issues)
  const TEST_VIDEOS = [
    { name: 'Kling AI Video', url: '/test-videos/test1.mp4' },
    { name: 'TV Noise 1080p', url: '/test-videos/test2.mp4' },
  ];

  const log = (message: string, type: LogEntry['type'] = 'info') => {
    const time = new Date().toISOString().substr(11, 12);
    setLogs(prev => [...prev, { time, message, type }]);
    console.log(`[${type.toUpperCase()}] ${message}`);
  };

  const loadTestVideos = async () => {
    setIsLoading(true);
    log('Loading test videos from URLs...', 'info');

    for (const testVideo of TEST_VIDEOS) {
      try {
        log(`Fetching: ${testVideo.name}...`, 'info');
        const response = await fetch(testVideo.url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const arrayBuffer = await response.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: 'video/mp4' });
        const url = URL.createObjectURL(blob);

        const video = document.createElement('video');
        video.src = url;
        video.muted = true;
        video.preload = 'metadata';

        await new Promise<void>((resolve, reject) => {
          video.onloadedmetadata = () => resolve();
          video.onerror = () => reject(new Error(`Failed to load ${testVideo.name}`));
          setTimeout(() => reject(new Error('Timeout')), 10000);
        });

        setFiles(prev => [...prev, {
          name: testVideo.name,
          arrayBuffer,
          video,
          duration: video.duration,
          width: video.videoWidth,
          height: video.videoHeight
        }]);

        log(`Loaded: ${testVideo.name} - ${video.videoWidth}x${video.videoHeight}, ${video.duration.toFixed(2)}s`, 'success');
      } catch (err) {
        log(`Failed to load ${testVideo.name}: ${err}`, 'error');
      }
    }

    setIsLoading(false);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const newFiles: LoadedFile[] = [];

    for (const file of Array.from(e.target.files)) {
      log(`Loading: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'info');

      try {
        const arrayBuffer = await file.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: file.type });
        const url = URL.createObjectURL(blob);

        const video = document.createElement('video');
        video.src = url;
        video.muted = true;
        video.preload = 'metadata';

        await new Promise<void>((resolve, reject) => {
          video.onloadedmetadata = () => resolve();
          video.onerror = () => reject(new Error(`Failed to load ${file.name}`));
        });

        newFiles.push({
          name: file.name,
          arrayBuffer,
          video,
          duration: video.duration,
          width: video.videoWidth,
          height: video.videoHeight
        });

        log(`Loaded: ${file.name} - ${video.videoWidth}x${video.videoHeight}, ${video.duration.toFixed(2)}s`, 'success');
      } catch (err) {
        log(`Failed to load ${file.name}: ${err}`, 'error');
      }
    }

    setFiles(prev => [...prev, ...newFiles]);
  };

  const runTest = async () => {
    if (files.length === 0) {
      log('Please load video files first!', 'error');
      return;
    }

    setIsRunning(true);
    setProgress(0);
    log('=== Starting Parallel Decode Test ===', 'info');

    try {
      const decoder = new ParallelDecodeManager();
      decoderRef.current = decoder;

      // Create clip infos - stagger clips
      const clipInfos = files.map((file, i) => ({
        clipId: `clip-${i}`,
        clipName: file.name,
        fileData: file.arrayBuffer,
        startTime: i * 2, // Stagger by 2 seconds
        duration: Math.min(file.duration, 10), // Max 10 seconds
        inPoint: 0,
        outPoint: Math.min(file.duration, 10),
        reversed: false,
        speed: 1
      }));

      log(`Initializing ${clipInfos.length} clips...`, 'info');
      const initStart = performance.now();

      await decoder.initialize(clipInfos, 30);

      log(`Initialized in ${(performance.now() - initStart).toFixed(0)}ms`, 'success');

      // Export simulation
      const fps = 30;
      const totalDuration = Math.max(...clipInfos.map(c => c.startTime + c.duration));
      const totalFrames = Math.ceil(totalDuration * fps);

      log(`Simulating export: ${totalFrames} frames at ${fps}fps, ${totalDuration.toFixed(2)}s`, 'info');

      const exportStart = performance.now();
      let framesDecoded = 0;
      let framesMissing = 0;
      const missingDetails: string[] = [];

      for (let frame = 0; frame < totalFrames; frame++) {
        const time = frame / fps;

        // Prefetch frames
        await decoder.prefetchFramesForTime(time);

        // Get frames for each clip at this time
        for (const clipInfo of clipInfos) {
          if (time >= clipInfo.startTime && time < clipInfo.startTime + clipInfo.duration) {
            const videoFrame = decoder.getFrameForClip(clipInfo.clipId, time);
            if (videoFrame) {
              framesDecoded++;
            } else {
              framesMissing++;
              if (missingDetails.length < 10) {
                missingDetails.push(`${clipInfo.clipName} @ ${time.toFixed(3)}s`);
              }
            }
          }
        }

        // Advance buffer
        decoder.advanceToTime(time);

        // Progress update
        const pct = ((frame + 1) / totalFrames) * 100;
        setProgress(pct);

        if (frame % 60 === 0 || frame === totalFrames - 1) {
          const elapsed = performance.now() - exportStart;
          const exportFps = (frame + 1) / (elapsed / 1000);
          log(`Frame ${frame + 1}/${totalFrames} (${pct.toFixed(0)}%) - ${exportFps.toFixed(1)} export fps`, 'info');
        }

        // Yield to UI
        if (frame % 10 === 0) {
          await new Promise(r => setTimeout(r, 0));
        }
      }

      const totalTime = performance.now() - exportStart;
      const avgFps = totalFrames / (totalTime / 1000);

      log('=== Export Test Complete ===', 'success');
      log(`Total time: ${(totalTime / 1000).toFixed(2)}s`, 'info');
      log(`Average export FPS: ${avgFps.toFixed(1)}`, avgFps > 15 ? 'success' : 'error');
      log(`Frames decoded: ${framesDecoded}`, 'success');
      log(`Frames missing: ${framesMissing}`, framesMissing === 0 ? 'success' : 'error');

      if (missingDetails.length > 0) {
        log(`Missing frame details: ${missingDetails.join(', ')}`, 'warn');
      }

      decoder.cleanup();
      decoderRef.current = null;
      log('Cleanup complete', 'success');

    } catch (e: any) {
      log(`ERROR: ${e.message}`, 'error');
      log(e.stack || '', 'error');
    }

    setIsRunning(false);
  };

  const stopTest = () => {
    if (decoderRef.current) {
      decoderRef.current.cleanup();
      decoderRef.current = null;
    }
    setIsRunning(false);
    log('Test stopped by user', 'warn');
  };

  return (
    <div style={{ padding: 20, fontFamily: 'monospace', background: '#1a1a1a', minHeight: '100vh', color: '#eee' }}>
      <h2>Parallel Decode Export Test</h2>

      <div style={{ marginBottom: 20 }}>
        <input
          type="file"
          multiple
          accept="video/*"
          onChange={handleFileSelect}
          disabled={isRunning}
        />
        <button
          onClick={loadTestVideos}
          disabled={isRunning || isLoading}
          style={{ marginLeft: 10, padding: '8px 16px' }}
        >
          {isLoading ? 'Loading...' : 'Load Test Videos'}
        </button>
        <button
          onClick={runTest}
          disabled={isRunning || files.length === 0}
          style={{ marginLeft: 10, padding: '8px 16px' }}
        >
          {isRunning ? 'Running...' : 'Run Export Test'}
        </button>
        {isRunning && (
          <button onClick={stopTest} style={{ marginLeft: 10, padding: '8px 16px' }}>
            Stop
          </button>
        )}
        <button
          onClick={() => setLogs([])}
          style={{ marginLeft: 10, padding: '8px 16px' }}
        >
          Clear Log
        </button>
      </div>

      {files.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <strong>Loaded files:</strong>
          <ul style={{ margin: '5px 0', paddingLeft: 20 }}>
            {files.map((f, i) => (
              <li key={i}>{f.name} - {f.width}x{f.height}, {f.duration.toFixed(2)}s</li>
            ))}
          </ul>
        </div>
      )}

      {isRunning && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ background: '#333', height: 20, borderRadius: 4, overflow: 'hidden' }}>
            <div
              style={{
                background: '#0f0',
                height: '100%',
                width: `${progress}%`,
                transition: 'width 0.1s'
              }}
            />
          </div>
          <div style={{ textAlign: 'center', marginTop: 5 }}>{progress.toFixed(1)}%</div>
        </div>
      )}

      <div
        style={{
          background: '#111',
          padding: 10,
          borderRadius: 4,
          maxHeight: 500,
          overflow: 'auto',
          fontSize: 12
        }}
      >
        {logs.map((entry, i) => (
          <div
            key={i}
            style={{
              color: entry.type === 'error' ? '#f55' :
                     entry.type === 'success' ? '#5f5' :
                     entry.type === 'warn' ? '#fa0' : '#0af'
            }}
          >
            [{entry.time}] {entry.message}
          </div>
        ))}
      </div>
    </div>
  );
}

export default ParallelDecodeTest;
