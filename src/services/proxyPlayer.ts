// ProxyPlayer - WebCodecs-based proxy video player for instant scrubbing
// Parses all-keyframe MP4 via MP4Box, decodes single frames on demand via VideoDecoder
// Every frame is a keyframe → decode exactly 1 frame per seek → instant random access

import { Logger } from './logger';
import * as MP4BoxModule from 'mp4box';

const log = Logger.create('ProxyPlayer');

const MP4Box = (MP4BoxModule as any).default || MP4BoxModule;

const MAX_CACHE_SIZE = 30; // ~1 second of proxy at 30fps

interface MP4ArrayBuffer extends ArrayBuffer {
  fileStart: number;
}

interface Sample {
  number: number;
  track_id: number;
  data: ArrayBuffer;
  size: number;
  cts: number;
  dts: number;
  duration: number;
  is_sync: boolean;
  timescale: number;
}

interface MP4VideoTrack {
  id: number;
  codec: string;
  duration: number;
  timescale: number;
  nb_samples: number;
  video: { width: number; height: number };
}

export class ProxyPlayer {
  private samples: Sample[] = [];
  private codecConfig: VideoDecoderConfig | null = null;
  private videoTrack: MP4VideoTrack | null = null;
  private _currentFrame: VideoFrame | null = null;

  // LRU cache of recently decoded frames
  private frameCache = new Map<number, VideoFrame>(); // sampleIndex → VideoFrame
  private cacheOrder: number[] = []; // LRU order

  private _frameRate = 30;
  private _duration = 0;
  private _totalFrames = 0;
  private _ready = false;

  get duration(): number { return this._duration; }
  get totalFrames(): number { return this._totalFrames; }
  get frameRate(): number { return this._frameRate; }
  get ready(): boolean { return this._ready; }

  async loadFromBlob(blob: Blob): Promise<void> {
    const buffer = await blob.arrayBuffer();

    return new Promise((resolve, reject) => {
      const mp4File = MP4Box.createFile();
      let resolved = false;

      const timeout = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          reject(new Error('MP4 parsing timeout'));
        }
      }, 10000);

      mp4File.onReady = async (info: { videoTracks: MP4VideoTrack[] }) => {
        if (info.videoTracks.length === 0) {
          clearTimeout(timeout);
          reject(new Error('No video track'));
          return;
        }

        this.videoTrack = info.videoTracks[0];
        const track = this.videoTrack;

        this._duration = track.duration / track.timescale;
        this._frameRate = track.nb_samples / this._duration;
        this._totalFrames = track.nb_samples;

        // Extract codec description
        let description: ArrayBuffer | undefined;
        try {
          const trak = mp4File.getTrackById(track.id);
          if (trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]) {
            const entry = trak.mdia.minf.stbl.stsd.entries[0];
            const configBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
            if (configBox) {
              const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
              configBox.write(stream);
              description = stream.buffer.slice(8);
            }
          }
        } catch (e) {
          log.warn('Failed to extract codec description', e);
        }

        // Build codec string
        let codec = track.codec;
        if (codec.startsWith('avc1')) {
          try {
            const trak = mp4File.getTrackById(track.id);
            const avcC = trak?.mdia?.minf?.stbl?.stsd?.entries?.[0]?.avcC;
            if (avcC) {
              const profile = avcC.AVCProfileIndication.toString(16).padStart(2, '0');
              const compat = avcC.profile_compatibility.toString(16).padStart(2, '0');
              const level = avcC.AVCLevelIndication.toString(16).padStart(2, '0');
              codec = `avc1.${profile}${compat}${level}`;
            }
          } catch { /* use raw codec */ }
        }

        this.codecConfig = {
          codec,
          codedWidth: track.video.width,
          codedHeight: track.video.height,
          hardwareAcceleration: 'prefer-hardware',
          optimizeForLatency: true,
          description,
        };

        // Verify codec support
        try {
          const support = await VideoDecoder.isConfigSupported(this.codecConfig);
          if (!support.supported) {
            // Try without description
            this.codecConfig = {
              codec,
              codedWidth: track.video.width,
              codedHeight: track.video.height,
              hardwareAcceleration: 'prefer-hardware',
              optimizeForLatency: true,
            };
            const support2 = await VideoDecoder.isConfigSupported(this.codecConfig);
            if (!support2.supported) {
              clearTimeout(timeout);
              reject(new Error(`Codec ${codec} not supported`));
              return;
            }
          }
        } catch (e) {
          clearTimeout(timeout);
          reject(e);
          return;
        }

        // Set up sample extraction
        mp4File.setExtractionOptions(track.id, null, { nbSamples: Infinity });
        mp4File.start();
        mp4File.flush();
      };

      mp4File.onSamples = (_trackId: number, _ref: any, samples: Sample[]) => {
        this.samples.push(...samples);

        if (this.videoTrack && this.samples.length >= this.videoTrack.nb_samples && !resolved) {
          resolved = true;
          clearTimeout(timeout);
          this._ready = true;
          log.debug(`ProxyPlayer ready: ${this.samples.length} samples, ${this._duration.toFixed(1)}s`);
          resolve();
        }
      };

      mp4File.onError = (e: string) => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          reject(new Error(`MP4Box error: ${e}`));
        }
      };

      // Feed buffer
      const mp4Buffer = buffer as MP4ArrayBuffer;
      mp4Buffer.fileStart = 0;
      mp4File.appendBuffer(mp4Buffer);
      mp4File.flush();

      // Fallback: if samples received quickly via onSamples, we might already be done
      // If not, poll briefly
      setTimeout(() => {
        if (!resolved && this.samples.length > 0 && this.videoTrack) {
          resolved = true;
          clearTimeout(timeout);
          this._ready = true;
          log.debug(`ProxyPlayer ready (poll): ${this.samples.length} samples`);
          resolve();
        }
      }, 2000);
    });
  }

  seekToTime(timeSeconds: number): VideoFrame | null {
    if (!this._ready || this.samples.length === 0 || !this.videoTrack) return null;

    const timescale = this.videoTrack.timescale;
    const targetCts = timeSeconds * timescale;

    // Binary search for closest sample by CTS
    let bestIdx = 0;
    let bestDiff = Infinity;
    for (let i = 0; i < this.samples.length; i++) {
      const diff = Math.abs(this.samples[i].cts - targetCts);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestIdx = i;
      }
    }

    return this.decodeSampleSync(bestIdx);
  }

  seekToFrame(frameIndex: number): VideoFrame | null {
    if (!this._ready || frameIndex < 0 || frameIndex >= this.samples.length) return null;
    return this.decodeSampleSync(frameIndex);
  }

  getCurrentFrame(): VideoFrame | null {
    return this._currentFrame;
  }

  private decodeSampleSync(sampleIndex: number): VideoFrame | null {
    // Check LRU cache
    const cached = this.frameCache.get(sampleIndex);
    if (cached) {
      // Move to end of LRU order
      this.cacheOrder = this.cacheOrder.filter(i => i !== sampleIndex);
      this.cacheOrder.push(sampleIndex);
      this._currentFrame = cached;
      return cached;
    }

    // Since all frames are keyframes, we can decode a single frame synchronously
    // by creating a decoder, feeding one sample, and getting the output
    const sample = this.samples[sampleIndex];
    if (!sample || !this.codecConfig) return this._currentFrame;

    // Fire-and-forget async decode - cache result for next call
    this.decodeAndCache(sampleIndex, sample);

    // Return current frame as fallback while decoding
    return this._currentFrame;
  }

  private decodeAndCache(sampleIndex: number, sample: Sample): void {
    // Avoid duplicate decode requests
    if (this.frameCache.has(sampleIndex)) return;

    const decoder = new VideoDecoder({
      output: (frame) => {
        // Evict if cache full
        while (this.cacheOrder.length >= MAX_CACHE_SIZE) {
          const evicted = this.cacheOrder.shift()!;
          const evictedFrame = this.frameCache.get(evicted);
          if (evictedFrame && evictedFrame !== this._currentFrame) {
            evictedFrame.close();
          }
          this.frameCache.delete(evicted);
        }

        this.frameCache.set(sampleIndex, frame);
        this.cacheOrder.push(sampleIndex);
        this._currentFrame = frame;

        try { decoder.close(); } catch { /* ignore */ }
      },
      error: () => {
        try { decoder.close(); } catch { /* ignore */ }
      },
    });

    try {
      decoder.configure(this.codecConfig!);

      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? 'key' : 'delta',
        timestamp: (sample.cts * 1_000_000) / sample.timescale,
        duration: (sample.duration * 1_000_000) / sample.timescale,
        data: sample.data,
      });

      decoder.decode(chunk);
      decoder.flush();
    } catch (e) {
      log.warn(`Failed to decode sample ${sampleIndex}`, e);
      try { decoder.close(); } catch { /* ignore */ }
    }
  }

  dispose(): void {
    for (const frame of this.frameCache.values()) {
      try { frame.close(); } catch { /* ignore */ }
    }
    this.frameCache.clear();
    this.cacheOrder = [];
    this._currentFrame = null;
    this._ready = false;
    this.samples = [];
    log.debug('ProxyPlayer disposed');
  }
}

// Manager singleton
const proxyPlayers = new Map<string, ProxyPlayer>();

export function getProxyPlayer(mediaFileId: string): ProxyPlayer | null {
  return proxyPlayers.get(mediaFileId) || null;
}

export async function loadProxyPlayer(mediaFileId: string, blob: Blob): Promise<ProxyPlayer> {
  // Dispose existing player
  const existing = proxyPlayers.get(mediaFileId);
  if (existing) existing.dispose();

  const player = new ProxyPlayer();
  await player.loadFromBlob(blob);
  proxyPlayers.set(mediaFileId, player);
  log.info(`Loaded proxy player for ${mediaFileId}: ${player.totalFrames} frames, ${player.duration.toFixed(1)}s`);
  return player;
}

export function disposeProxyPlayer(mediaFileId: string): void {
  const player = proxyPlayers.get(mediaFileId);
  if (player) {
    player.dispose();
    proxyPlayers.delete(mediaFileId);
  }
}

export function disposeAllProxyPlayers(): void {
  for (const player of proxyPlayers.values()) {
    player.dispose();
  }
  proxyPlayers.clear();
}
