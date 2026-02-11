// Proxy frame cache - loads and caches WebP frames for fast playback

import { Logger } from './logger';
import { projectFileService } from './projectFileService';

const log = Logger.create('ProxyFrameCache');
import { fileSystemService } from './fileSystemService';
import { useMediaStore } from '../stores/mediaStore';

// Cache settings - tuned for fast scrubbing
const MAX_CACHE_SIZE = 900; // 30 seconds at 30fps - larger cache for scrubbing
const PRELOAD_AHEAD_FRAMES = 60; // 2 seconds ahead for playback
const PRELOAD_BEHIND_FRAMES = 30; // 1 second behind for reverse scrubbing
const PARALLEL_LOAD_COUNT = 16; // More parallel loads for faster preload
const SCRUB_PRELOAD_RANGE = 90; // 3 seconds around scrub position

// Frame cache entry
interface CachedFrame {
  mediaFileId: string;
  frameIndex: number;
  image: HTMLImageElement;
  timestamp: number; // For LRU eviction
}

class ProxyFrameCache {
  private cache: Map<string, CachedFrame> = new Map();
  private loadingPromises: Map<string, Promise<HTMLImageElement | null>> = new Map();
  private preloadQueue: string[] = [];
  private isPreloading = false;

  // Audio proxy cache
  private audioCache: Map<string, HTMLAudioElement> = new Map();
  private audioLoadingPromises: Map<string, Promise<HTMLAudioElement | null>> = new Map();

  // Audio buffer cache for instant scrubbing (Web Audio API)
  private audioBufferCache: Map<string, AudioBuffer> = new Map();
  private audioBufferFailed: Set<string> = new Set(); // Track files with no audio
  private audioContext: AudioContext | null = null;
  private scrubGain: GainNode | null = null;

  // Get cache key
  private getKey(mediaFileId: string, frameIndex: number): string {
    return `${mediaFileId}_${frameIndex}`;
  }

  // Synchronously get a frame if it's already in memory cache
  // Also triggers preloading of upcoming frames (even if current frame not cached)
  getCachedFrame(mediaFileId: string, frameIndex: number, fps: number = 30): HTMLImageElement | null {
    const key = this.getKey(mediaFileId, frameIndex);
    const cached = this.cache.get(key);

    // ALWAYS trigger preloading, even if current frame isn't cached
    // This ensures nested composition frames get preloaded when playhead enters them
    this.schedulePreload(mediaFileId, frameIndex, fps);

    if (cached) {
      cached.timestamp = Date.now(); // Update for LRU
      this.cacheHits++;
      return cached.image;
    }
    this.cacheMisses++;
    return null;
  }

  // Get nearest cached frame for scrubbing fallback
  // Returns the closest frame within maxDistance frames
  // Searches in scrub direction first for smoother scrubbing
  getNearestCachedFrame(mediaFileId: string, frameIndex: number, maxDistance: number = 30): HTMLImageElement | null {
    // Check exact frame first
    const exactKey = this.getKey(mediaFileId, frameIndex);
    const exact = this.cache.get(exactKey);
    if (exact) {
      exact.timestamp = Date.now();
      return exact.image;
    }

    // Search in scrub direction first for visual continuity
    const searchForward = this.scrubDirection >= 0;

    for (let d = 1; d <= maxDistance; d++) {
      // Search primary direction first
      const primaryOffset = searchForward ? d : -d;
      const primaryFrame = frameIndex + primaryOffset;
      if (primaryFrame >= 0) {
        const primaryKey = this.getKey(mediaFileId, primaryFrame);
        const primary = this.cache.get(primaryKey);
        if (primary) {
          primary.timestamp = Date.now();
          return primary.image;
        }
      }

      // Then search opposite direction
      const secondaryOffset = searchForward ? -d : d;
      const secondaryFrame = frameIndex + secondaryOffset;
      if (secondaryFrame >= 0) {
        const secondaryKey = this.getKey(mediaFileId, secondaryFrame);
        const secondary = this.cache.get(secondaryKey);
        if (secondary) {
          secondary.timestamp = Date.now();
          return secondary.image;
        }
      }
    }

    return null;
  }

  // Get a frame from cache or load it
  async getFrame(mediaFileId: string, time: number, fps: number = 30): Promise<HTMLImageElement | null> {
    const frameIndex = Math.floor(time * fps);
    const key = this.getKey(mediaFileId, frameIndex);

    // Check cache first
    const cached = this.cache.get(key);
    if (cached) {
      cached.timestamp = Date.now(); // Update for LRU
      return cached.image;
    }

    // Check if already loading
    const loadingPromise = this.loadingPromises.get(key);
    if (loadingPromise) {
      return loadingPromise;
    }

    // Load from IndexedDB
    const promise = this.loadFrame(mediaFileId, frameIndex);
    this.loadingPromises.set(key, promise);

    try {
      const image = await promise;
      if (image) {
        this.addToCache(mediaFileId, frameIndex, image);
        // Trigger preload of upcoming frames
        this.schedulePreload(mediaFileId, frameIndex, fps);
      }
      return image;
    } finally {
      this.loadingPromises.delete(key);
    }
  }

  // Load a single frame - ONLY from project folder (no browser cache)
  private async loadFrame(mediaFileId: string, frameIndex: number): Promise<HTMLImageElement | null> {
    try {
      let blob: Blob | null = null;

      // Get the media file to find its fileHash (used for proxy folder naming)
      const mediaFile = useMediaStore.getState().files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      // Debug logging
      if (frameIndex === 0) {
        log.debug(`Loading frame 0 for: ${mediaFile?.name}`);
        log.debug(`storageKey: ${storageKey}, projectOpen: ${projectFileService.isProjectOpen()}, proxyStatus: ${mediaFile?.proxyStatus}`);
      }

      // Load from project folder ONLY (no IndexedDB fallback)
      if (projectFileService.isProjectOpen()) {
        blob = await projectFileService.getProxyFrame(storageKey, frameIndex);
        if (frameIndex === 0) {
          log.debug(`Frame 0 blob: ${blob ? `${blob.size} bytes` : 'null'}`);
        }
      }

      if (!blob) return null;

      // Create image from blob
      const url = URL.createObjectURL(blob);
      const image = new Image();

      return new Promise((resolve) => {
        image.onload = () => {
          URL.revokeObjectURL(url);
          resolve(image);
        };
        image.onerror = () => {
          URL.revokeObjectURL(url);
          resolve(null);
        };
        image.src = url;
      });
    } catch (e) {
      log.warn('Failed to load frame', e);
      return null;
    }
  }

  // Add frame to cache
  private addToCache(mediaFileId: string, frameIndex: number, image: HTMLImageElement) {
    const key = this.getKey(mediaFileId, frameIndex);

    // Evict old frames if cache is full
    if (this.cache.size >= MAX_CACHE_SIZE) {
      this.evictOldest();
    }

    this.cache.set(key, {
      mediaFileId,
      frameIndex,
      image,
      timestamp: Date.now(),
    });
  }

  // Evict oldest frame from cache (LRU)
  private evictOldest() {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache) {
      if (entry.timestamp < oldestTime) {
        oldestTime = entry.timestamp;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  // Scrub tracking for smart preloading
  private lastScrubFrame = -1;
  private scrubDirection = 0; // -1 = backward, 0 = stopped, 1 = forward
  private isScrubbing = false;

  // Schedule preloading of frames around current position (bidirectional)
  private schedulePreload(mediaFileId: string, currentFrameIndex: number, _fps: number) {
    // Detect scrub direction
    if (this.lastScrubFrame >= 0) {
      const delta = currentFrameIndex - this.lastScrubFrame;
      if (Math.abs(delta) > 0 && Math.abs(delta) < 100) {
        this.scrubDirection = delta > 0 ? 1 : -1;
        this.isScrubbing = true;
      }
    }
    this.lastScrubFrame = currentFrameIndex;

    // Calculate preload range based on scrubbing state
    const preloadAhead = this.isScrubbing ? SCRUB_PRELOAD_RANGE : PRELOAD_AHEAD_FRAMES;
    const preloadBehind = this.isScrubbing ? SCRUB_PRELOAD_RANGE : PRELOAD_BEHIND_FRAMES;

    // Priority queue: current frame first, then direction-based loading
    const framesToPreload: number[] = [currentFrameIndex];

    // Add frames in scrub direction first (higher priority)
    if (this.scrubDirection >= 0) {
      // Forward or stopped: prioritize ahead
      for (let i = 1; i <= preloadAhead; i++) {
        framesToPreload.push(currentFrameIndex + i);
      }
      for (let i = 1; i <= preloadBehind; i++) {
        if (currentFrameIndex - i >= 0) {
          framesToPreload.push(currentFrameIndex - i);
        }
      }
    } else {
      // Backward scrubbing: prioritize behind
      for (let i = 1; i <= preloadBehind; i++) {
        if (currentFrameIndex - i >= 0) {
          framesToPreload.push(currentFrameIndex - i);
        }
      }
      for (let i = 1; i <= preloadAhead; i++) {
        framesToPreload.push(currentFrameIndex + i);
      }
    }

    // Add to preload queue
    for (let i = 0; i < framesToPreload.length; i++) {
      const frameIndex = framesToPreload[i];
      if (frameIndex < 0) continue;

      const key = this.getKey(mediaFileId, frameIndex);

      // Skip if already cached or in queue
      if (!this.cache.has(key) && !this.preloadQueue.includes(key)) {
        // Insert current frame at front of queue for priority loading
        if (i === 0) {
          this.preloadQueue.unshift(key);
        } else {
          this.preloadQueue.push(key);
        }
      }
    }

    // Start preloading if not already
    if (!this.isPreloading) {
      this.processPreloadQueue();
    }
  }

  // Call this when scrubbing stops to reset state
  resetScrubState(): void {
    this.isScrubbing = false;
    this.scrubDirection = 0;
    this.lastScrubFrame = -1;
  }

  // Process preload queue with parallel loading for speed
  private async processPreloadQueue() {
    this.isPreloading = true;

    while (this.preloadQueue.length > 0) {
      // Load multiple frames in parallel for faster preloading
      const batch: string[] = [];
      while (batch.length < PARALLEL_LOAD_COUNT && this.preloadQueue.length > 0) {
        const key = this.preloadQueue.shift();
        if (key && !this.cache.has(key)) {
          batch.push(key);
        }
      }

      if (batch.length === 0) continue;

      // Load batch in parallel
      const loadPromises = batch.map(async (key) => {
        const [mediaFileId, frameIndexStr] = key.split('_');
        const frameIndex = parseInt(frameIndexStr, 10);

        const image = await this.loadFrame(mediaFileId, frameIndex);
        if (image) {
          this.addToCache(mediaFileId, frameIndex, image);
        }
        return { key, success: !!image };
      });

      await Promise.all(loadPromises);

      // Brief yield to main thread between batches
      await new Promise((r) => setTimeout(r, 0));
    }

    this.isPreloading = false;
  }

  // ============================================
  // AUDIO PROXY METHODS
  // ============================================

  /**
   * Get cached audio proxy element, or load it if not cached
   * Returns null if no audio proxy exists
   */
  async getAudioProxy(mediaFileId: string): Promise<HTMLAudioElement | null> {
    // Check cache first
    const cached = this.audioCache.get(mediaFileId);
    if (cached) {
      return cached;
    }

    // Check if already loading
    const existingPromise = this.audioLoadingPromises.get(mediaFileId);
    if (existingPromise) {
      return existingPromise;
    }

    // Start loading
    const loadPromise = this.loadAudioProxy(mediaFileId);
    this.audioLoadingPromises.set(mediaFileId, loadPromise);

    try {
      const audio = await loadPromise;
      if (audio) {
        this.audioCache.set(mediaFileId, audio);
      }
      return audio;
    } finally {
      this.audioLoadingPromises.delete(mediaFileId);
    }
  }

  /**
   * Get cached audio proxy synchronously (returns null if not yet loaded)
   */
  getCachedAudioProxy(mediaFileId: string): HTMLAudioElement | null {
    return this.audioCache.get(mediaFileId) || null;
  }

  /**
   * Preload audio proxy for a media file
   */
  async preloadAudioProxy(mediaFileId: string): Promise<void> {
    // Just call getAudioProxy which handles caching
    await this.getAudioProxy(mediaFileId);
  }

  /**
   * Load audio proxy from project folder
   */
  private async loadAudioProxy(mediaFileId: string): Promise<HTMLAudioElement | null> {
    try {
      // Get storage key (prefer fileHash for deduplication)
      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      // Load audio file from project folder
      const audioFile = await projectFileService.getProxyAudio(storageKey);
      if (!audioFile) {
        return null;
      }

      // Create audio element with object URL
      const audio = new Audio();
      audio.src = URL.createObjectURL(audioFile);
      audio.preload = 'auto';

      // Wait for audio to be ready
      await new Promise<void>((resolve, reject) => {
        const onCanPlay = () => {
          audio.removeEventListener('canplaythrough', onCanPlay);
          audio.removeEventListener('error', onError);
          resolve();
        };
        const onError = () => {
          audio.removeEventListener('canplaythrough', onCanPlay);
          audio.removeEventListener('error', onError);
          reject(new Error('Failed to load audio proxy'));
        };
        audio.addEventListener('canplaythrough', onCanPlay);
        audio.addEventListener('error', onError);
        // Start loading
        audio.load();
      });

      log.info(`Audio proxy loaded for ${mediaFileId}`);
      return audio;
    } catch (e) {
      log.warn(`Failed to load audio proxy for ${mediaFileId}`, e);
      return null;
    }
  }

  // ============================================
  // VARISPEED AUDIO SCRUBBING (Web Audio API)
  // Like Premiere/Resolve: continuous audio that follows scrub speed
  // ============================================

  // Varispeed scrubbing state
  private scrubSource: AudioBufferSourceNode | null = null;
  private scrubSourceGain: GainNode | null = null;
  private scrubStartTime = 0; // AudioContext time when scrub started
  private scrubStartPosition = 0; // Audio position when scrub started
  private scrubCurrentMediaId: string | null = null;
  private scrubLastPosition = 0;
  private scrubLastTime = 0;
  private scrubIsActive = false;

  /**
   * Get or create AudioContext for scrubbing
   */
  private getAudioContext(): AudioContext {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
      this.scrubGain = this.audioContext.createGain();
      this.scrubGain.connect(this.audioContext.destination);
      this.scrubGain.gain.value = 0.85; // Slightly lower for scrubbing
    }
    return this.audioContext;
  }

  /**
   * Get AudioBuffer for a media file (decode on first request)
   * Works with BOTH proxy audio AND original video files
   */
  async getAudioBuffer(mediaFileId: string): Promise<AudioBuffer | null> {
    // Check cache
    const cached = this.audioBufferCache.get(mediaFileId);
    if (cached) return cached;

    // Skip files that have no audio (failed decoding = audio doesn't exist)
    if (this.audioBufferFailed.has(mediaFileId)) {
      return null;
    }

    // Check if already loading
    if (this.audioBufferLoading.has(mediaFileId)) {
      return null; // Loading in progress
    }

    // Cooldown for "source not found" - retry after 3 seconds (source may become available)
    const lastAttempt = this.audioBufferRetryTime.get(mediaFileId);
    if (lastAttempt && performance.now() - lastAttempt < 3000) {
      return null;
    }

    this.audioBufferLoading.add(mediaFileId);

    try {
      const mediaStore = useMediaStore.getState();
      const mediaFile = mediaStore.files.find(f => f.id === mediaFileId);
      const storageKey = mediaFile?.fileHash || mediaFileId;

      let arrayBuffer: ArrayBuffer | null = null;

      // Try 1: Proxy audio file (fastest, smallest)
      const audioFile = await projectFileService.getProxyAudio(storageKey);
      if (audioFile) {
        log.debug(`Loading from proxy audio: ${mediaFileId}`);
        arrayBuffer = await audioFile.arrayBuffer();
      }

      // Try 2: Original video file URL (extract audio from video)
      if (!arrayBuffer && mediaFile?.url) {
        log.debug(`Loading from video URL: ${mediaFileId}`);
        try {
          const response = await fetch(mediaFile.url);
          arrayBuffer = await response.arrayBuffer();
        } catch (e) {
          log.warn('Failed to fetch video URL', e);
        }
      }

      // Try 3: File handle (if available)
      if (!arrayBuffer) {
        const fileHandle = fileSystemService.getFileHandle(mediaFileId);
        if (fileHandle) {
          log.debug(`Loading from file handle: ${mediaFileId}`);
          try {
            const file = await fileHandle.getFile();
            arrayBuffer = await file.arrayBuffer();
          } catch (e) {
            log.warn('Failed to read file handle', e);
          }
        }
      }

      // Try 4: Direct File object from media store (e.g. YouTube downloads)
      if (!arrayBuffer && mediaFile?.file) {
        log.debug(`Loading from File object: ${mediaFileId}`);
        try {
          arrayBuffer = await mediaFile.file.arrayBuffer();
        } catch (e) {
          log.warn('Failed to read File object', e);
        }
      }

      if (!arrayBuffer) {
        log.warn(`No audio source found for ${mediaFileId}`);
        // Use cooldown instead of permanent failure - source may become available later
        this.audioBufferRetryTime.set(mediaFileId, performance.now());
        this.audioBufferLoading.delete(mediaFileId);
        return null;
      }

      // Decode to AudioBuffer
      const audioContext = this.getAudioContext();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0)); // Clone to avoid detached buffer

      // Cache it
      this.audioBufferCache.set(mediaFileId, audioBuffer);
      this.audioBufferLoading.delete(mediaFileId);
      this.audioBufferRetryTime.delete(mediaFileId);
      log.debug(`Decoded ${mediaFileId}: ${audioBuffer.duration.toFixed(1)}s, ${audioBuffer.numberOfChannels}ch`);

      return audioBuffer;
    } catch (e) {
      // Mark as failed so we don't keep retrying (video probably has no audio track)
      this.audioBufferFailed.add(mediaFileId);
      this.audioBufferLoading.delete(mediaFileId);
      log.debug(`No audio in ${mediaFileId} (or decode failed)`);
      return null;
    }
  }

  // Track loading state to prevent duplicate loads
  private audioBufferLoading = new Set<string>();
  // Cooldown for "source not found" retries (not permanent failure like audioBufferFailed)
  private audioBufferRetryTime = new Map<string, number>();


  /**
   * VARISPEED SCRUBBING - Call this continuously while scrubbing
   * Audio plays continuously and follows the scrub position/speed
   * Like Premiere Pro / DaVinci Resolve
   */
  playScrubAudio(mediaFileId: string, targetTime: number, _duration: number = 0.15): void {
    const buffer = this.audioBufferCache.get(mediaFileId);
    if (!buffer) {
      log.debug(`No AudioBuffer for ${mediaFileId} - loading...`);
      this.getAudioBuffer(mediaFileId);
      return;
    }

    // Debug: Log that varispeed is active
    if (!this.scrubIsActive) {
      log.debug(`VARISPEED starting at ${targetTime.toFixed(2)}s`);
    }

    const ctx = this.getAudioContext();
    if (ctx.state === 'suspended') {
      ctx.resume();
    }

    const now = performance.now();
    const clampedTarget = Math.max(0, Math.min(targetTime, buffer.duration - 0.1));

    // Calculate scrub velocity (how fast user is scrubbing)
    const timeDelta = (now - this.scrubLastTime) / 1000; // seconds
    const posDelta = clampedTarget - this.scrubLastPosition;
    this.scrubLastPosition = clampedTarget;
    this.scrubLastTime = now;

    // Need new source if: different media, not active, or position jumped too far
    const needNewSource =
      !this.scrubIsActive ||
      this.scrubCurrentMediaId !== mediaFileId ||
      !this.scrubSource;

    if (needNewSource) {
      // Stop existing source
      this.stopScrubAudio();

      // Create new continuous source
      this.scrubSourceGain = ctx.createGain();
      this.scrubSourceGain.connect(this.scrubGain!);
      this.scrubSourceGain.gain.value = 0.9;

      this.scrubSource = ctx.createBufferSource();
      this.scrubSource.buffer = buffer;
      this.scrubSource.connect(this.scrubSourceGain);
      this.scrubSource.playbackRate.value = 1.0;

      // Start playing from target position
      this.scrubSource.start(0, clampedTarget);
      this.scrubStartTime = ctx.currentTime;
      this.scrubStartPosition = clampedTarget;
      this.scrubCurrentMediaId = mediaFileId;
      this.scrubIsActive = true;

      // Guard: only deactivate if THIS source is still the current one
      // (onended fires asynchronously and could clobber a newer source)
      const thisSource = this.scrubSource;
      thisSource.onended = () => {
        if (this.scrubSource === thisSource) {
          this.scrubIsActive = false;
          this.scrubSource = null;
        }
      };
    } else if (this.scrubSource && timeDelta > 0.001) {
      // Calculate where audio SHOULD be vs where it IS
      const elapsedAudioTime = (ctx.currentTime - this.scrubStartTime) * this.scrubSource.playbackRate.value;
      const currentAudioPos = this.scrubStartPosition + elapsedAudioTime;
      const drift = clampedTarget - currentAudioPos;

      // If drift is too large (>300ms), restart at correct position
      if (Math.abs(drift) > 0.3) {
        this.stopScrubAudio();
        // Will restart on next call
        return;
      }

      // Calculate target playback rate based on scrub velocity
      // scrubSpeed = how many seconds of audio per second of real time
      const scrubSpeed = timeDelta > 0.01 ? Math.abs(posDelta) / timeDelta : 1;

      // Clamp to reasonable range and add drift correction
      const driftCorrection = drift * 2; // Gentle drift correction
      let targetRate = Math.max(0.25, Math.min(4.0, scrubSpeed + driftCorrection));

      // If scrubbing backwards, we can't play backwards, so just slow down a lot
      if (posDelta < -0.001) {
        targetRate = 0.25; // Minimum speed for backwards feel
      }

      // Smooth rate changes to avoid clicks
      const currentRate = this.scrubSource.playbackRate.value;
      const smoothedRate = currentRate + (targetRate - currentRate) * 0.3;
      this.scrubSource.playbackRate.value = Math.max(0.25, Math.min(4.0, smoothedRate));
    }
  }

  /**
   * Stop scrub audio - call when scrubbing ends
   */
  stopScrubAudio(): void {
    if (this.scrubSource) {
      try {
        this.scrubSource.onended = null; // Prevent async callback from clobbering new source
        this.scrubSource.stop();
        this.scrubSource.disconnect();
      } catch { /* ignore */ }
      this.scrubSource = null;
    }
    if (this.scrubSourceGain) {
      try {
        this.scrubSourceGain.disconnect();
      } catch { /* ignore */ }
      this.scrubSourceGain = null;
    }
    this.scrubIsActive = false;
    this.scrubCurrentMediaId = null;

    // Also reset frame scrub tracking state
    this.resetScrubState();
  }

  /**
   * Check if audio buffer is ready for instant scrubbing
   */
  hasAudioBuffer(mediaFileId: string): boolean {
    return this.audioBufferCache.has(mediaFileId);
  }

  // Clear cache for a specific media file
  clearForMedia(mediaFileId: string) {
    for (const [key] of this.cache) {
      if (key.startsWith(mediaFileId + '_')) {
        this.cache.delete(key);
      }
    }
    this.preloadQueue = this.preloadQueue.filter((k) => !k.startsWith(mediaFileId + '_'));

    // Also clear audio cache
    const audio = this.audioCache.get(mediaFileId);
    if (audio) {
      audio.pause();
      URL.revokeObjectURL(audio.src);
      this.audioCache.delete(mediaFileId);
    }

    // Clear audio buffer cache
    this.audioBufferCache.delete(mediaFileId);
  }

  // Clear entire cache
  clearAll() {
    this.cache.clear();
    this.preloadQueue = [];

    // Clear audio cache
    for (const [, audio] of this.audioCache) {
      audio.pause();
      URL.revokeObjectURL(audio.src);
    }
    this.audioCache.clear();

    // Clear audio buffer cache
    this.audioBufferCache.clear();
  }

  // Bulk preload frames around a position - call when scrubbing starts
  async preloadAroundPosition(mediaFileId: string, frameIndex: number, _fps: number = 30, range: number = SCRUB_PRELOAD_RANGE): Promise<void> {
    const framesToPreload: string[] = [];

    // Generate list of frames to preload (current position +/- range)
    for (let i = -range; i <= range; i++) {
      const frame = frameIndex + i;
      if (frame < 0) continue;

      const key = this.getKey(mediaFileId, frame);
      if (!this.cache.has(key) && !this.preloadQueue.includes(key)) {
        framesToPreload.push(key);
      }
    }

    // Add all to front of queue (highest priority)
    this.preloadQueue = [...framesToPreload, ...this.preloadQueue];

    // Start preloading if not already
    if (!this.isPreloading) {
      this.processPreloadQueue();
    }

    log.debug(`Bulk preload started: ${framesToPreload.length} frames around frame ${frameIndex}`);
  }

  // Preload ALL frames for a media file (for manual cache button)
  // Returns a promise that resolves when preloading is complete
  // onProgress callback receives (loadedFrames, totalFrames)
  async preloadAllFrames(
    mediaFileId: string,
    totalFrames: number,
    _fps: number,
    onProgress?: (loaded: number, total: number) => void
  ): Promise<void> {
    log.info(`Starting full preload for ${mediaFileId}: ${totalFrames} frames`);

    let loadedCount = 0;
    const batchSize = 32; // Load 32 frames at a time

    for (let startFrame = 0; startFrame < totalFrames; startFrame += batchSize) {
      const endFrame = Math.min(startFrame + batchSize, totalFrames);
      const batch: Promise<void>[] = [];

      for (let frame = startFrame; frame < endFrame; frame++) {
        const key = this.getKey(mediaFileId, frame);

        // Skip if already cached
        if (this.cache.has(key)) {
          loadedCount++;
          continue;
        }

        // Load frame
        batch.push(
          this.loadFrame(mediaFileId, frame).then(image => {
            if (image) {
              this.addToCache(mediaFileId, frame, image);
            }
            loadedCount++;
          })
        );
      }

      // Wait for batch to complete
      await Promise.all(batch);

      // Report progress
      if (onProgress) {
        onProgress(loadedCount, totalFrames);
      }

      // Yield to main thread
      await new Promise(r => setTimeout(r, 0));
    }

    log.info(`Full preload complete for ${mediaFileId}: ${loadedCount}/${totalFrames} frames cached`);
  }

  // Cancel ongoing preload (for when user clicks stop or navigates away)
  cancelPreload(): void {
    this.preloadQueue = [];
    log.debug('Preload cancelled');
  }

  // Get cache stats with more detail
  getStats() {
    return {
      cachedFrames: this.cache.size,
      maxCacheSize: MAX_CACHE_SIZE,
      preloadQueueSize: this.preloadQueue.length,
      isPreloading: this.isPreloading,
      isScrubbing: this.isScrubbing,
      scrubDirection: this.scrubDirection,
      hitRate: this.cacheHits / Math.max(1, this.cacheHits + this.cacheMisses),
    };
  }

  // Cache hit/miss tracking
  private cacheHits = 0;
  private cacheMisses = 0;

  // Log cache performance (call periodically for debugging)
  logPerformance(): void {
    const total = this.cacheHits + this.cacheMisses;
    const hitRate = total > 0 ? (this.cacheHits / total * 100).toFixed(1) : '0';
    log.debug(`Hit rate: ${hitRate}% (${this.cacheHits}/${total}), cached: ${this.cache.size}/${MAX_CACHE_SIZE}, queue: ${this.preloadQueue.length}`);
  }

  // Reset performance counters
  resetPerformanceCounters(): void {
    this.cacheHits = 0;
    this.cacheMisses = 0;
  }

  // Get cached frame ranges for a specific media file (for timeline display)
  // Returns ranges in seconds relative to media file start
  getCachedRanges(mediaFileId: string, fps: number = 30): Array<{ start: number; end: number }> {
    // Collect all cached frame indices for this media file
    const cachedFrames: number[] = [];
    for (const [, entry] of this.cache) {
      if (entry.mediaFileId === mediaFileId) {
        cachedFrames.push(entry.frameIndex);
      }
    }

    if (cachedFrames.length === 0) return [];

    // Sort frames
    cachedFrames.sort((a, b) => a - b);

    // Convert to time ranges, merging adjacent frames
    const ranges: Array<{ start: number; end: number }> = [];
    const frameInterval = 1 / fps;
    const maxGap = frameInterval * 3; // Allow gap of 3 frames before starting new range

    let rangeStart = cachedFrames[0] / fps;
    let rangeEnd = rangeStart + frameInterval;

    for (let i = 1; i < cachedFrames.length; i++) {
      const frameTime = cachedFrames[i] / fps;
      if (frameTime - rangeEnd <= maxGap) {
        // Extend current range
        rangeEnd = frameTime + frameInterval;
      } else {
        // Save current range and start new one
        ranges.push({ start: rangeStart, end: rangeEnd });
        rangeStart = frameTime;
        rangeEnd = frameTime + frameInterval;
      }
    }

    // Add final range
    ranges.push({ start: rangeStart, end: rangeEnd });

    return ranges;
  }

  // Get all cached media file IDs
  getCachedMediaIds(): string[] {
    const ids = new Set<string>();
    for (const entry of this.cache.values()) {
      ids.add(entry.mediaFileId);
    }
    return Array.from(ids);
  }
}

// Singleton instance
export const proxyFrameCache = new ProxyFrameCache();
