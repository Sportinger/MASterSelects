// WebCodecsAudioPlayer - Web Audio API based audio playback
// Replaces HTMLAudioElement for timeline audio sync
// Uses AudioBufferSourceNode for precise, frame-accurate playback

import { Logger } from '../services/logger';

const log = Logger.create('WebCodecsAudioPlayer');

export interface AudioTrackInfo {
  codec: string;
  sampleRate: number;
  channels: number;
  duration: number;
}

export class WebCodecsAudioPlayer {
  private audioContext: AudioContext;
  private audioBuffer: AudioBuffer | null = null;
  private sourceNode: AudioBufferSourceNode | null = null;
  private gainNode: GainNode;

  // Playback state
  private _isPlaying = false;
  private startedAt = 0;      // audioContext.currentTime when playback started
  private startOffset = 0;    // offset into the buffer when playback started
  private _playbackRate = 1;
  private _muted = false;
  private _volume = 1;
  private _duration = 0;

  // For scrub audio
  private scrubTimeout: ReturnType<typeof setTimeout> | null = null;

  constructor(audioContext?: AudioContext) {
    // Reuse shared AudioContext if provided, otherwise create one
    this.audioContext = audioContext || new AudioContext();
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
  }

  /**
   * Decode audio from an ArrayBuffer (same buffer used by WebCodecsPlayer)
   */
  async loadFromArrayBuffer(buffer: ArrayBuffer): Promise<void> {
    const endLoad = log.time('loadAudioBuffer');
    try {
      // decodeAudioData consumes the buffer, so we need a copy
      const copy = buffer.slice(0);
      this.audioBuffer = await this.audioContext.decodeAudioData(copy);
      this._duration = this.audioBuffer.duration;
      log.info(`Audio decoded: ${this.audioBuffer.numberOfChannels}ch, ${this.audioBuffer.sampleRate}Hz, ${this._duration.toFixed(2)}s`);
    } catch (e) {
      log.error('Failed to decode audio', e);
      throw new Error(`Audio decode failed: ${e}`);
    } finally {
      endLoad();
    }
  }

  /**
   * Load audio from a File object
   */
  async loadFromFile(file: File): Promise<void> {
    const buffer = await file.arrayBuffer();
    await this.loadFromArrayBuffer(buffer);
  }

  /**
   * Start playback from a given offset (seconds into the audio buffer)
   */
  play(offset?: number, playbackRate?: number): void {
    if (!this.audioBuffer) return;

    // Resume AudioContext if suspended (browser autoplay policy)
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume().catch(() => {});
    }

    // Stop any existing source
    this.stopSource();

    const effectiveOffset = offset ?? this.startOffset;
    const effectiveRate = playbackRate ?? this._playbackRate;

    // Clamp offset to valid range
    const clampedOffset = Math.max(0, Math.min(effectiveOffset, this._duration));

    // Create new source node
    this.sourceNode = this.audioContext.createBufferSource();
    this.sourceNode.buffer = this.audioBuffer;
    this.sourceNode.playbackRate.value = effectiveRate;
    this.sourceNode.connect(this.gainNode);

    // Apply current volume/muted state
    this.gainNode.gain.value = this._muted ? 0 : this._volume;

    // Start playback
    this.sourceNode.start(0, clampedOffset);
    this.startedAt = this.audioContext.currentTime;
    this.startOffset = clampedOffset;
    this._playbackRate = effectiveRate;
    this._isPlaying = true;

    // Handle end of buffer
    this.sourceNode.onended = () => {
      if (this._isPlaying) {
        this._isPlaying = false;
      }
    };
  }

  /**
   * Pause playback, preserving current position
   */
  pause(): void {
    if (!this._isPlaying) return;

    // Save current position before stopping
    this.startOffset = this.currentTime;
    this.stopSource();
    this._isPlaying = false;
  }

  /**
   * Seek to a specific time (seconds). If playing, restarts from new position.
   */
  seek(time: number): void {
    const wasPlaying = this._isPlaying;
    if (wasPlaying) {
      this.stopSource();
      this._isPlaying = false;
    }
    this.startOffset = Math.max(0, Math.min(time, this._duration));
    if (wasPlaying) {
      this.play(this.startOffset);
    }
  }

  /**
   * Play a short audio snippet for scrubbing feedback
   */
  playScrubSnippet(time: number, snippetDuration = 0.08): void {
    if (!this.audioBuffer) return;

    // Resume AudioContext if suspended
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume().catch(() => {});
    }

    // Stop any existing source
    this.stopSource();

    // Clear any existing scrub timeout
    if (this.scrubTimeout) {
      clearTimeout(this.scrubTimeout);
      this.scrubTimeout = null;
    }

    const clampedTime = Math.max(0, Math.min(time, this._duration));

    this.sourceNode = this.audioContext.createBufferSource();
    this.sourceNode.buffer = this.audioBuffer;
    this.sourceNode.playbackRate.value = 1;
    this.sourceNode.connect(this.gainNode);

    this.gainNode.gain.value = this._muted ? 0 : this._volume * 0.8;

    this.sourceNode.start(0, clampedTime, snippetDuration);
    this._isPlaying = true;

    // Fade out after snippet duration
    this.scrubTimeout = setTimeout(() => {
      this.fadeOutAndStop();
      this.scrubTimeout = null;
    }, snippetDuration * 1000);
  }

  /**
   * Fade out gain and stop source
   */
  private fadeOutAndStop(): void {
    if (!this.sourceNode) return;

    const now = this.audioContext.currentTime;
    this.gainNode.gain.setValueAtTime(this.gainNode.gain.value, now);
    this.gainNode.gain.linearRampToValueAtTime(0, now + 0.02);

    // Stop after fade
    setTimeout(() => {
      this.stopSource();
      this._isPlaying = false;
      // Restore gain for next playback
      this.gainNode.gain.value = this._muted ? 0 : this._volume;
    }, 25);
  }

  /**
   * Current playback position in seconds (master clock)
   */
  get currentTime(): number {
    if (!this._isPlaying) return this.startOffset;
    const elapsed = (this.audioContext.currentTime - this.startedAt) * this._playbackRate;
    return this.startOffset + elapsed;
  }

  get duration(): number {
    return this._duration;
  }

  get isPlaying(): boolean {
    return this._isPlaying;
  }

  get volume(): number {
    return this._volume;
  }

  get muted(): boolean {
    return this._muted;
  }

  get playbackRate(): number {
    return this._playbackRate;
  }

  /**
   * Get the underlying AudioBuffer (for export or analysis)
   */
  getAudioBuffer(): AudioBuffer | null {
    return this.audioBuffer;
  }

  /**
   * Get the AudioContext (for routing/effects)
   */
  getAudioContext(): AudioContext {
    return this.audioContext;
  }

  /**
   * Get the gain node (for external routing like EQ)
   */
  getGainNode(): GainNode {
    return this.gainNode;
  }

  setVolume(volume: number): void {
    this._volume = Math.max(0, Math.min(1, volume));
    if (!this._muted) {
      this.gainNode.gain.value = this._volume;
    }
  }

  setMuted(muted: boolean): void {
    this._muted = muted;
    this.gainNode.gain.value = muted ? 0 : this._volume;
  }

  setPlaybackRate(rate: number): void {
    this._playbackRate = Math.max(0.25, Math.min(4, rate));
    if (this.sourceNode && this._isPlaying) {
      this.sourceNode.playbackRate.value = this._playbackRate;
    }
  }

  /**
   * Stop the current AudioBufferSourceNode
   */
  private stopSource(): void {
    if (this.sourceNode) {
      try {
        this.sourceNode.onended = null;
        this.sourceNode.stop();
        this.sourceNode.disconnect();
      } catch {
        // Already stopped
      }
      this.sourceNode = null;
    }
  }

  /**
   * Clean up all resources
   */
  destroy(): void {
    this.stopSource();

    if (this.scrubTimeout) {
      clearTimeout(this.scrubTimeout);
      this.scrubTimeout = null;
    }

    this.gainNode.disconnect();
    this.audioBuffer = null;
    this._isPlaying = false;
    this.startOffset = 0;
    this.startedAt = 0;

    log.debug('WebCodecsAudioPlayer destroyed');
  }
}
