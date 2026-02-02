/**
 * AudioRoutingManager - Routes audio through Web Audio API for live EQ and volume
 *
 * Architecture:
 * HTMLMediaElement → MediaElementSourceNode → GainNode → EQ Filters → Destination
 *
 * Efficiency:
 * - Single shared AudioContext
 * - Lazy connection (only when playing)
 * - Node caching per element (MediaElementSourceNode can only be created once)
 * - Delta updates for filter gains
 */

import { Logger } from './logger';

const log = Logger.create('AudioRouting');

// EQ frequencies (10-band)
const EQ_FREQUENCIES = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

interface AudioRoute {
  sourceNode: MediaElementAudioSourceNode;
  gainNode: GainNode;
  eqFilters: BiquadFilterNode[];
  isConnected: boolean;
  lastVolume: number;
  lastEQGains: number[];
}

class AudioRoutingManager {
  private audioContext: AudioContext | null = null;
  private routes = new Map<HTMLMediaElement, AudioRoute>();
  private contextResumePromise: Promise<void> | null = null;

  /**
   * Get or create the shared AudioContext
   */
  private async getContext(): Promise<AudioContext> {
    if (!this.audioContext || this.audioContext.state === 'closed') {
      this.audioContext = new AudioContext();
      log.info('Created new AudioContext');
    }

    // Resume if suspended (autoplay policy)
    if (this.audioContext.state === 'suspended') {
      if (!this.contextResumePromise) {
        this.contextResumePromise = this.audioContext.resume().then(() => {
          this.contextResumePromise = null;
        });
      }
      await this.contextResumePromise;
    }

    return this.audioContext;
  }

  /**
   * Get or create audio route for an element
   */
  private async getOrCreateRoute(element: HTMLMediaElement): Promise<AudioRoute | null> {
    // Check if route already exists
    let route = this.routes.get(element);
    if (route) return route;

    try {
      const ctx = await this.getContext();

      // Create source node (can only be done ONCE per element)
      const sourceNode = ctx.createMediaElementSource(element);

      // Create gain node for volume
      const gainNode = ctx.createGain();
      gainNode.gain.value = 1;

      // Create EQ filter chain
      const eqFilters: BiquadFilterNode[] = EQ_FREQUENCIES.map(freq => {
        const filter = ctx.createBiquadFilter();
        filter.type = 'peaking';
        filter.frequency.value = freq;
        filter.Q.value = 1.4; // Standard Q for 10-band EQ
        filter.gain.value = 0; // Default: no boost/cut
        return filter;
      });

      // Connect chain: source → gain → eq[0] → eq[1] → ... → eq[9] → destination
      sourceNode.connect(gainNode);
      gainNode.connect(eqFilters[0]);
      for (let i = 0; i < eqFilters.length - 1; i++) {
        eqFilters[i].connect(eqFilters[i + 1]);
      }
      eqFilters[eqFilters.length - 1].connect(ctx.destination);

      route = {
        sourceNode,
        gainNode,
        eqFilters,
        isConnected: true,
        lastVolume: 1,
        lastEQGains: new Array(10).fill(0),
      };

      this.routes.set(element, route);
      log.debug('Created audio route for element');

      return route;
    } catch (err) {
      // MediaElementSourceNode can fail if element is already connected elsewhere
      // or if there's a CORS issue with the audio source
      log.warn('Failed to create audio route:', err);
      return null;
    }
  }

  /**
   * Apply volume and EQ to an audio element
   * Call this every frame for elements that are playing
   */
  async applyEffects(
    element: HTMLMediaElement,
    volume: number,
    eqGains: number[] // Array of 10 gain values in dB (-12 to +12)
  ): Promise<boolean> {
    const route = await this.getOrCreateRoute(element);
    if (!route) {
      // Fallback: just set element volume directly (no EQ)
      element.volume = Math.max(0, Math.min(1, volume));
      return false;
    }

    // Update volume if changed (with small delta threshold)
    if (Math.abs(route.lastVolume - volume) > 0.001) {
      // Web Audio gain can go above 1, but clamp for sanity
      route.gainNode.gain.value = Math.max(0, Math.min(4, volume));
      route.lastVolume = volume;
    }

    // Update EQ gains if changed
    for (let i = 0; i < 10; i++) {
      const gain = eqGains[i] ?? 0;
      if (Math.abs(route.lastEQGains[i] - gain) > 0.01) {
        route.eqFilters[i].gain.value = gain;
        route.lastEQGains[i] = gain;
      }
    }

    // When using Web Audio routing, the element volume should be 1
    // (volume is controlled by the GainNode)
    if (element.volume !== 1) {
      element.volume = 1;
    }

    return true;
  }

  /**
   * Check if an element has an active audio route
   */
  hasRoute(element: HTMLMediaElement): boolean {
    return this.routes.has(element);
  }

  /**
   * Disconnect and remove route for an element
   * Call when element is no longer needed
   */
  removeRoute(element: HTMLMediaElement): void {
    const route = this.routes.get(element);
    if (route) {
      try {
        route.sourceNode.disconnect();
        route.gainNode.disconnect();
        route.eqFilters.forEach(f => f.disconnect());
      } catch {
        // Ignore disconnect errors
      }
      this.routes.delete(element);
      log.debug('Removed audio route');
    }
  }

  /**
   * Clean up all routes and close context
   */
  dispose(): void {
    for (const [element] of this.routes) {
      this.removeRoute(element);
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
    this.audioContext = null;
    log.info('AudioRoutingManager disposed');
  }

  /**
   * Get the number of active routes (for debugging)
   */
  get activeRouteCount(): number {
    return this.routes.size;
  }
}

// Singleton instance
export const audioRoutingManager = new AudioRoutingManager();
