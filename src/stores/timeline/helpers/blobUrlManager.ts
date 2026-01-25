// Blob URL Manager - prevents memory leaks by tracking and cleaning up object URLs
// Each clip can have associated blob URLs that need to be revoked when the clip is removed

type UrlType = 'video' | 'audio' | 'image' | 'file';

interface ManagedUrl {
  url: string;
  type: UrlType;
  createdAt: number;
}

/**
 * Centralized manager for blob URLs.
 * Tracks URLs by clip ID and ensures proper cleanup.
 */
class BlobUrlManager {
  // Map of clipId -> Map of urlType -> ManagedUrl
  private urls = new Map<string, Map<UrlType, ManagedUrl>>();

  // Track total URLs for debugging
  private totalCreated = 0;
  private totalRevoked = 0;

  /**
   * Create a blob URL for a file and associate it with a clip.
   * Automatically revokes any existing URL of the same type for this clip.
   */
  create(clipId: string, file: File | Blob, type: UrlType = 'file'): string {
    // Revoke existing URL of this type if present
    this.revokeType(clipId, type);

    const url = URL.createObjectURL(file);

    let clipUrls = this.urls.get(clipId);
    if (!clipUrls) {
      clipUrls = new Map();
      this.urls.set(clipId, clipUrls);
    }

    clipUrls.set(type, {
      url,
      type,
      createdAt: Date.now(),
    });

    this.totalCreated++;
    return url;
  }

  /**
   * Get the URL for a clip and type, if it exists.
   */
  get(clipId: string, type: UrlType = 'file'): string | undefined {
    return this.urls.get(clipId)?.get(type)?.url;
  }

  /**
   * Check if a clip has a URL of a specific type.
   */
  has(clipId: string, type: UrlType = 'file'): boolean {
    return this.urls.get(clipId)?.has(type) ?? false;
  }

  /**
   * Revoke a specific URL type for a clip.
   */
  revokeType(clipId: string, type: UrlType): void {
    const clipUrls = this.urls.get(clipId);
    if (!clipUrls) return;

    const managed = clipUrls.get(type);
    if (managed) {
      URL.revokeObjectURL(managed.url);
      clipUrls.delete(type);
      this.totalRevoked++;

      // Clean up empty map
      if (clipUrls.size === 0) {
        this.urls.delete(clipId);
      }
    }
  }

  /**
   * Revoke all URLs associated with a clip.
   * Call this when removing a clip from the timeline.
   */
  revokeAll(clipId: string): void {
    const clipUrls = this.urls.get(clipId);
    if (!clipUrls) return;

    for (const managed of clipUrls.values()) {
      URL.revokeObjectURL(managed.url);
      this.totalRevoked++;
    }

    this.urls.delete(clipId);
  }

  /**
   * Revoke URLs for multiple clips.
   * Useful when removing multiple clips at once.
   */
  revokeMany(clipIds: string[]): void {
    for (const clipId of clipIds) {
      this.revokeAll(clipId);
    }
  }

  /**
   * Transfer URL ownership from one clip to another.
   * Useful when splitting clips.
   */
  transfer(fromClipId: string, toClipId: string, type: UrlType): void {
    const fromUrls = this.urls.get(fromClipId);
    if (!fromUrls) return;

    const managed = fromUrls.get(type);
    if (!managed) return;

    // Remove from source (without revoking)
    fromUrls.delete(type);
    if (fromUrls.size === 0) {
      this.urls.delete(fromClipId);
    }

    // Add to destination
    let toUrls = this.urls.get(toClipId);
    if (!toUrls) {
      toUrls = new Map();
      this.urls.set(toClipId, toUrls);
    }
    toUrls.set(type, managed);
  }

  /**
   * Clone URL reference for a new clip (e.g., when splitting).
   * The URL is shared, so only the last clip should revoke it.
   * Returns the shared URL.
   */
  share(fromClipId: string, toClipId: string, type: UrlType): string | undefined {
    const url = this.get(fromClipId, type);
    if (!url) return undefined;

    // Don't create a new URL, just track the same one for the new clip
    let toUrls = this.urls.get(toClipId);
    if (!toUrls) {
      toUrls = new Map();
      this.urls.set(toClipId, toUrls);
    }

    toUrls.set(type, {
      url,
      type,
      createdAt: Date.now(),
    });

    return url;
  }

  /**
   * Get statistics about URL usage.
   */
  getStats(): { active: number; created: number; revoked: number } {
    let active = 0;
    for (const clipUrls of this.urls.values()) {
      active += clipUrls.size;
    }

    return {
      active,
      created: this.totalCreated,
      revoked: this.totalRevoked,
    };
  }

  /**
   * Clear all URLs. Use only during cleanup/reset.
   */
  clear(): void {
    for (const clipUrls of this.urls.values()) {
      for (const managed of clipUrls.values()) {
        URL.revokeObjectURL(managed.url);
        this.totalRevoked++;
      }
    }
    this.urls.clear();
  }

  /**
   * Debug: log all tracked URLs.
   */
  debug(): void {
    console.log('[BlobUrlManager] Active URLs:');
    for (const [clipId, clipUrls] of this.urls) {
      for (const [type, managed] of clipUrls) {
        console.log(`  ${clipId} [${type}]: ${managed.url} (created ${Date.now() - managed.createdAt}ms ago)`);
      }
    }
    console.log('[BlobUrlManager] Stats:', this.getStats());
  }
}

// Singleton instance
export const blobUrlManager = new BlobUrlManager();

// Export class for testing
export { BlobUrlManager };
