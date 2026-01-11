// IndexedDB service for project persistence
// Stores media file blobs and project data

const DB_NAME = 'MASterSelectsDB';
const DB_VERSION = 5; // Upgraded for fileHash proxy deduplication

// Store names
const STORES = {
  MEDIA_FILES: 'mediaFiles',
  PROJECTS: 'projects',
  PROXY_FRAMES: 'proxyFrames', // New store for proxy frame sequences
  FS_HANDLES: 'fsHandles', // Store for FileSystemHandles (directories, files)
  ANALYSIS_CACHE: 'analysisCache', // Cache for clip analysis data
  THUMBNAILS: 'thumbnails', // Deduplicated thumbnails by file hash
} as const;

// Thumbnail stored by file hash for deduplication
export interface StoredThumbnail {
  fileHash: string; // Primary key
  blob: Blob;
  width?: number;
  height?: number;
  createdAt: number;
}

export interface StoredMediaFile {
  id: string;
  name: string;
  type: 'video' | 'audio' | 'image';
  // No longer storing blob - only metadata and file hash for deduplication
  fileHash?: string; // SHA-256 hash for proxy/thumbnail deduplication
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
  codec?: string;
  container?: string;
  fileSize?: number;
  createdAt: number;
}

// Proxy frame data - stores frames for a media file
export interface StoredProxyFrame {
  id: string; // Format: fileHash_frameIndex (e.g., "abc123_0042")
  mediaFileId: string; // Legacy: kept for backwards compatibility
  fileHash?: string; // SHA-256 hash of file content (for deduplication)
  frameIndex: number;
  blob: Blob; // WebP image blob
}

// Proxy metadata stored with media file
export interface ProxyMetadata {
  mediaFileId: string;
  frameCount: number;
  fps: number;
  width: number;
  height: number;
  createdAt: number;
}

// Cached analysis data for a media file
export interface StoredAnalysis {
  mediaFileId: string;
  // Analysis data per time range (key: "inPoint-outPoint")
  // Allows caching different trim ranges of the same file
  analyses: {
    [rangeKey: string]: {
      frames: Array<{
        timestamp: number;
        motion: number;
        globalMotion: number;
        localMotion: number;
        focus: number;
        faceCount: number;
        isSceneCut?: boolean;
      }>;
      sampleInterval: number;
      createdAt: number;
    };
  };
}

export interface StoredProject {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  // Full project state
  data: {
    compositions: unknown[];
    folders: unknown[];
    activeCompositionId: string | null;
    openCompositionIds?: string[];
    expandedFolderIds: string[];
    // Media file IDs (actual blobs stored separately)
    mediaFileIds: string[];
  };
}

class ProjectDatabase {
  private db: IDBDatabase | null = null;
  private initPromise: Promise<IDBDatabase> | null = null;

  // Initialize the database
  async init(): Promise<IDBDatabase> {
    if (this.db) return this.db;
    if (this.initPromise) return this.initPromise;

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error('Failed to open IndexedDB:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        console.log('[ProjectDB] Database opened successfully');
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create media files store
        if (!db.objectStoreNames.contains(STORES.MEDIA_FILES)) {
          const mediaStore = db.createObjectStore(STORES.MEDIA_FILES, { keyPath: 'id' });
          mediaStore.createIndex('name', 'name', { unique: false });
          mediaStore.createIndex('type', 'type', { unique: false });
        }

        // Create projects store
        if (!db.objectStoreNames.contains(STORES.PROJECTS)) {
          const projectStore = db.createObjectStore(STORES.PROJECTS, { keyPath: 'id' });
          projectStore.createIndex('name', 'name', { unique: false });
          projectStore.createIndex('updatedAt', 'updatedAt', { unique: false });
        }

        // Create proxy frames store (new in v2)
        if (!db.objectStoreNames.contains(STORES.PROXY_FRAMES)) {
          const proxyStore = db.createObjectStore(STORES.PROXY_FRAMES, { keyPath: 'id' });
          proxyStore.createIndex('mediaFileId', 'mediaFileId', { unique: false });
          proxyStore.createIndex('frameIndex', 'frameIndex', { unique: false });
          proxyStore.createIndex('fileHash', 'fileHash', { unique: false });
        } else if (event.oldVersion < 5) {
          // Add fileHash index for proxy deduplication (v5)
          const proxyStore = event.target!.transaction!.objectStore(STORES.PROXY_FRAMES);
          if (!proxyStore.indexNames.contains('fileHash')) {
            proxyStore.createIndex('fileHash', 'fileHash', { unique: false });
          }
        }

        // Create file system handles store (new in v3)
        if (!db.objectStoreNames.contains(STORES.FS_HANDLES)) {
          db.createObjectStore(STORES.FS_HANDLES, { keyPath: 'key' });
        }

        // Create analysis cache store (new in v4)
        if (!db.objectStoreNames.contains(STORES.ANALYSIS_CACHE)) {
          db.createObjectStore(STORES.ANALYSIS_CACHE, { keyPath: 'mediaFileId' });
        }

        // Create thumbnails store for deduplication (new in v5)
        if (!db.objectStoreNames.contains(STORES.THUMBNAILS)) {
          db.createObjectStore(STORES.THUMBNAILS, { keyPath: 'fileHash' });
        }

        console.log('[ProjectDB] Database schema created/upgraded');
      };
    });

    return this.initPromise;
  }

  // ============ Media Files ============

  // Store a media file blob
  async saveMediaFile(file: StoredMediaFile): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.MEDIA_FILES, 'readwrite');
      const store = transaction.objectStore(STORES.MEDIA_FILES);
      const request = store.put(file);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // Get a media file by ID
  async getMediaFile(id: string): Promise<StoredMediaFile | undefined> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.MEDIA_FILES, 'readonly');
      const store = transaction.objectStore(STORES.MEDIA_FILES);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Get all media files
  async getAllMediaFiles(): Promise<StoredMediaFile[]> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.MEDIA_FILES, 'readonly');
      const store = transaction.objectStore(STORES.MEDIA_FILES);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Delete a media file
  async deleteMediaFile(id: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.MEDIA_FILES, 'readwrite');
      const store = transaction.objectStore(STORES.MEDIA_FILES);
      const request = store.delete(id);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // ============ Projects ============

  // Save a project
  async saveProject(project: StoredProject): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROJECTS, 'readwrite');
      const store = transaction.objectStore(STORES.PROJECTS);
      const request = store.put(project);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // Get a project by ID
  async getProject(id: string): Promise<StoredProject | undefined> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROJECTS, 'readonly');
      const store = transaction.objectStore(STORES.PROJECTS);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Get all projects (metadata only, not full data)
  async getAllProjects(): Promise<StoredProject[]> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROJECTS, 'readonly');
      const store = transaction.objectStore(STORES.PROJECTS);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Delete a project
  async deleteProject(id: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROJECTS, 'readwrite');
      const store = transaction.objectStore(STORES.PROJECTS);
      const request = store.delete(id);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // ============ Utilities ============

  // Clear all data (for debugging/reset)
  async clearAll(): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORES.MEDIA_FILES, STORES.PROJECTS], 'readwrite');

      transaction.objectStore(STORES.MEDIA_FILES).clear();
      transaction.objectStore(STORES.PROJECTS).clear();

      transaction.oncomplete = () => {
        console.log('[ProjectDB] All data cleared');
        resolve();
      };
      transaction.onerror = () => reject(transaction.error);
    });
  }

  // Get database stats
  async getStats(): Promise<{ mediaFiles: number; projects: number; proxyFrames: number }> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORES.MEDIA_FILES, STORES.PROJECTS, STORES.PROXY_FRAMES], 'readonly');

      const mediaRequest = transaction.objectStore(STORES.MEDIA_FILES).count();
      const projectRequest = transaction.objectStore(STORES.PROJECTS).count();
      const proxyRequest = transaction.objectStore(STORES.PROXY_FRAMES).count();

      let mediaCount = 0;
      let projectCount = 0;
      let proxyCount = 0;

      mediaRequest.onsuccess = () => { mediaCount = mediaRequest.result; };
      projectRequest.onsuccess = () => { projectCount = projectRequest.result; };
      proxyRequest.onsuccess = () => { proxyCount = proxyRequest.result; };

      transaction.oncomplete = () => {
        resolve({ mediaFiles: mediaCount, projects: projectCount, proxyFrames: proxyCount });
      };
      transaction.onerror = () => reject(transaction.error);
    });
  }

  // ============ Proxy Frames ============

  // Save a single proxy frame
  async saveProxyFrame(frame: StoredProxyFrame): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readwrite');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const request = store.put(frame);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // Save multiple proxy frames in a batch (more efficient)
  async saveProxyFramesBatch(frames: StoredProxyFrame[]): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readwrite');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);

      for (const frame of frames) {
        store.put(frame);
      }

      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  }

  // Get a specific proxy frame
  async getProxyFrame(mediaFileId: string, frameIndex: number): Promise<StoredProxyFrame | undefined> {
    const db = await this.init();
    const id = `${mediaFileId}_${frameIndex.toString().padStart(6, '0')}`;
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readonly');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Get all proxy frames for a media file
  async getProxyFramesForMedia(mediaFileId: string): Promise<StoredProxyFrame[]> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readonly');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const index = store.index('mediaFileId');
      const request = index.getAll(mediaFileId);

      request.onsuccess = () => {
        // Sort by frame index
        const frames = request.result.sort((a, b) => a.frameIndex - b.frameIndex);
        resolve(frames);
      };
      request.onerror = () => reject(request.error);
    });
  }

  // Check if proxy exists for a media file
  async hasProxy(mediaFileId: string): Promise<boolean> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readonly');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const index = store.index('mediaFileId');
      const request = index.count(mediaFileId);

      request.onsuccess = () => resolve(request.result > 0);
      request.onerror = () => reject(request.error);
    });
  }

  // Get proxy frame count for a media file
  async getProxyFrameCount(mediaFileId: string): Promise<number> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readonly');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const index = store.index('mediaFileId');
      const request = index.count(mediaFileId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Delete all proxy frames for a media file
  async deleteProxyFrames(mediaFileId: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readwrite');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const index = store.index('mediaFileId');
      const request = index.openCursor(mediaFileId);

      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest<IDBCursorWithValue>).result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };

      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  }

  // Clear all proxy frames (for all media)
  async clearAllProxyFrames(): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readwrite');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // ============ Hash-based Proxy Deduplication ============

  // Get proxy frame count by file hash (for deduplication)
  async getProxyFrameCountByHash(fileHash: string): Promise<number> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readonly');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      try {
        const index = store.index('fileHash');
        const request = index.count(fileHash);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(0); // Fallback if index doesn't exist
      } catch {
        resolve(0); // Index doesn't exist yet
      }
    });
  }

  // Get a proxy frame by file hash
  async getProxyFrameByHash(fileHash: string, frameIndex: number): Promise<StoredProxyFrame | undefined> {
    const db = await this.init();
    const id = `${fileHash}_${frameIndex.toString().padStart(6, '0')}`;
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.PROXY_FRAMES, 'readonly');
      const store = transaction.objectStore(STORES.PROXY_FRAMES);
      const request = store.get(id);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Check if proxy exists by file hash
  async hasProxyByHash(fileHash: string): Promise<boolean> {
    const count = await this.getProxyFrameCountByHash(fileHash);
    return count > 0;
  }

  // ============ Thumbnail Deduplication ============

  // Save thumbnail by file hash
  async saveThumbnail(thumbnail: StoredThumbnail): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.THUMBNAILS, 'readwrite');
      const store = transaction.objectStore(STORES.THUMBNAILS);
      const request = store.put(thumbnail);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // Get thumbnail by file hash
  async getThumbnail(fileHash: string): Promise<StoredThumbnail | undefined> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.THUMBNAILS, 'readonly');
      const store = transaction.objectStore(STORES.THUMBNAILS);
      const request = store.get(fileHash);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Check if thumbnail exists by hash
  async hasThumbnail(fileHash: string): Promise<boolean> {
    const thumbnail = await this.getThumbnail(fileHash);
    return !!thumbnail;
  }

  // ============ File System Handles ============

  // Store a FileSystemHandle (directory or file)
  async storeHandle(key: string, handle: FileSystemHandle): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.FS_HANDLES, 'readwrite');
      const store = transaction.objectStore(STORES.FS_HANDLES);
      const request = store.put({ key, handle });

      request.onsuccess = () => {
        console.log('[ProjectDB] Stored handle:', key);
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }

  // Get a stored FileSystemHandle
  async getStoredHandle(key: string): Promise<FileSystemHandle | null> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.FS_HANDLES, 'readonly');
      const store = transaction.objectStore(STORES.FS_HANDLES);
      const request = store.get(key);

      request.onsuccess = () => {
        const result = request.result;
        resolve(result?.handle ?? null);
      };
      request.onerror = () => reject(request.error);
    });
  }

  // Delete a stored handle
  async deleteHandle(key: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.FS_HANDLES, 'readwrite');
      const store = transaction.objectStore(STORES.FS_HANDLES);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // Get all stored handles
  async getAllHandles(): Promise<Array<{ key: string; handle: FileSystemHandle }>> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.FS_HANDLES, 'readonly');
      const store = transaction.objectStore(STORES.FS_HANDLES);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });
  }

  // Check if there's a stored last project handle (for determining if welcome overlay should show)
  async hasLastProject(): Promise<boolean> {
    try {
      const handle = await this.getStoredHandle('lastProject');
      return handle !== null;
    } catch {
      return false;
    }
  }

  // ============ Analysis Cache ============

  /**
   * Generate a range key for analysis cache
   * @param inPoint Start time in seconds
   * @param outPoint End time in seconds
   */
  private getAnalysisRangeKey(inPoint: number, outPoint: number): string {
    return `${inPoint.toFixed(2)}-${outPoint.toFixed(2)}`;
  }

  /**
   * Save analysis data for a media file
   * @param mediaFileId The media file ID
   * @param inPoint Start time of analyzed range
   * @param outPoint End time of analyzed range
   * @param frames The analysis frame data
   * @param sampleInterval Sample interval in milliseconds
   */
  async saveAnalysis(
    mediaFileId: string,
    inPoint: number,
    outPoint: number,
    frames: StoredAnalysis['analyses'][string]['frames'],
    sampleInterval: number
  ): Promise<void> {
    const db = await this.init();
    const rangeKey = this.getAnalysisRangeKey(inPoint, outPoint);

    // First, get existing analysis data for this media file
    const existing = await this.getAnalysisRecord(mediaFileId);

    const record: StoredAnalysis = existing || {
      mediaFileId,
      analyses: {},
    };

    // Add or update the analysis for this range
    record.analyses[rangeKey] = {
      frames,
      sampleInterval,
      createdAt: Date.now(),
    };

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.ANALYSIS_CACHE, 'readwrite');
      const store = transaction.objectStore(STORES.ANALYSIS_CACHE);
      const request = store.put(record);

      request.onsuccess = () => {
        console.log(`[ProjectDB] Saved analysis for ${mediaFileId} (range: ${rangeKey})`);
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Get analysis record for a media file
   */
  private async getAnalysisRecord(mediaFileId: string): Promise<StoredAnalysis | undefined> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.ANALYSIS_CACHE, 'readonly');
      const store = transaction.objectStore(STORES.ANALYSIS_CACHE);
      const request = store.get(mediaFileId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Get cached analysis for a specific time range
   * @param mediaFileId The media file ID
   * @param inPoint Start time of analyzed range
   * @param outPoint End time of analyzed range
   * @returns The cached analysis or undefined if not found
   */
  async getAnalysis(
    mediaFileId: string,
    inPoint: number,
    outPoint: number
  ): Promise<StoredAnalysis['analyses'][string] | undefined> {
    const record = await this.getAnalysisRecord(mediaFileId);
    if (!record) return undefined;

    const rangeKey = this.getAnalysisRangeKey(inPoint, outPoint);
    return record.analyses[rangeKey];
  }

  /**
   * Check if analysis exists for a specific time range
   */
  async hasAnalysis(mediaFileId: string, inPoint: number, outPoint: number): Promise<boolean> {
    const analysis = await this.getAnalysis(mediaFileId, inPoint, outPoint);
    return !!analysis;
  }

  /**
   * Get all cached analysis ranges for a media file
   */
  async getAnalysisRanges(mediaFileId: string): Promise<string[]> {
    const record = await this.getAnalysisRecord(mediaFileId);
    if (!record) return [];
    return Object.keys(record.analyses);
  }

  /**
   * Delete all cached analysis for a media file
   */
  async deleteAnalysis(mediaFileId: string): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.ANALYSIS_CACHE, 'readwrite');
      const store = transaction.objectStore(STORES.ANALYSIS_CACHE);
      const request = store.delete(mediaFileId);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Clear all cached analysis data
   */
  async clearAllAnalysis(): Promise<void> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORES.ANALYSIS_CACHE, 'readwrite');
      const store = transaction.objectStore(STORES.ANALYSIS_CACHE);
      const request = store.clear();

      request.onsuccess = () => {
        console.log('[ProjectDB] All analysis cache cleared');
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }
}

// Singleton instance
export const projectDB = new ProjectDatabase();
