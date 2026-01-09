// Audio Manager - Master volume and EQ control using Web Audio API

export interface EQBand {
  frequency: number;
  gain: number; // -12 to +12 dB
}

// 10-band EQ standard frequencies
export const EQ_FREQUENCIES = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

class AudioManager {
  private audioContext: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private eqFilters: BiquadFilterNode[] = [];
  private mediaElementSources: Map<HTMLMediaElement, MediaElementAudioSourceNode> = new Map();
  private initialized = false;

  // EQ band gains (-12 to +12 dB)
  private eqGains: number[] = EQ_FREQUENCIES.map(() => 0);

  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      this.audioContext = new AudioContext();

      // Create master gain node
      this.masterGain = this.audioContext.createGain();

      // Create EQ filters (10-band parametric EQ)
      this.eqFilters = EQ_FREQUENCIES.map((freq, index) => {
        const filter = this.audioContext!.createBiquadFilter();
        filter.type = 'peaking';
        filter.frequency.value = freq;
        filter.Q.value = 1.4; // Standard Q for 10-band EQ
        filter.gain.value = this.eqGains[index];
        return filter;
      });

      // Chain: input -> EQ filters -> master gain -> destination
      // Connect EQ filters in series
      for (let i = 0; i < this.eqFilters.length - 1; i++) {
        this.eqFilters[i].connect(this.eqFilters[i + 1]);
      }

      // Connect last EQ filter to master gain
      this.eqFilters[this.eqFilters.length - 1].connect(this.masterGain);

      // Connect master gain to output
      this.masterGain.connect(this.audioContext.destination);

      this.initialized = true;
      console.log('[AudioManager] Initialized with 10-band EQ');
    } catch (error) {
      console.error('[AudioManager] Failed to initialize:', error);
    }
  }

  // Connect a media element (video/audio) to the audio chain
  connectMediaElement(element: HTMLMediaElement): void {
    if (!this.audioContext || !this.eqFilters.length) {
      console.warn('[AudioManager] Not initialized, cannot connect media element');
      return;
    }

    // Don't reconnect if already connected
    if (this.mediaElementSources.has(element)) {
      return;
    }

    try {
      // Resume context if suspended (autoplay policy)
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }

      // Create media element source
      const source = this.audioContext.createMediaElementSource(element);

      // Connect to first EQ filter
      source.connect(this.eqFilters[0]);

      // Store reference
      this.mediaElementSources.set(element, source);

      // Un-mute the element since we're handling audio through Web Audio
      element.muted = false;

      console.log('[AudioManager] Connected media element');
    } catch (error) {
      console.error('[AudioManager] Failed to connect media element:', error);
    }
  }

  // Disconnect a media element
  disconnectMediaElement(element: HTMLMediaElement): void {
    const source = this.mediaElementSources.get(element);
    if (source) {
      try {
        source.disconnect();
      } catch (e) {
        // Ignore disconnect errors
      }
      this.mediaElementSources.delete(element);
      console.log('[AudioManager] Disconnected media element');
    }
  }

  // Set master volume (0-1)
  setMasterVolume(volume: number): void {
    if (this.masterGain) {
      this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  // Set EQ band gain (-12 to +12 dB)
  setEQBand(bandIndex: number, gainDB: number): void {
    if (bandIndex >= 0 && bandIndex < this.eqFilters.length) {
      const clampedGain = Math.max(-12, Math.min(12, gainDB));
      this.eqGains[bandIndex] = clampedGain;
      this.eqFilters[bandIndex].gain.value = clampedGain;
    }
  }

  // Get all EQ band values
  getEQBands(): EQBand[] {
    return EQ_FREQUENCIES.map((freq, index) => ({
      frequency: freq,
      gain: this.eqGains[index],
    }));
  }

  // Set all EQ bands at once
  setAllEQBands(gains: number[]): void {
    gains.forEach((gain, index) => {
      this.setEQBand(index, gain);
    });
  }

  // Reset EQ to flat
  resetEQ(): void {
    this.eqFilters.forEach((filter, index) => {
      filter.gain.value = 0;
      this.eqGains[index] = 0;
    });
  }

  // Get current master volume
  getMasterVolume(): number {
    return this.masterGain?.gain.value ?? 1;
  }

  // Destroy and cleanup
  destroy(): void {
    // Disconnect all media elements
    this.mediaElementSources.forEach((source) => {
      try {
        source.disconnect();
      } catch (e) {
        // Ignore
      }
    });
    this.mediaElementSources.clear();

    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.masterGain = null;
    this.eqFilters = [];
    this.initialized = false;
  }

  isInitialized(): boolean {
    return this.initialized;
  }
}

// Singleton instance
export const audioManager = new AudioManager();
