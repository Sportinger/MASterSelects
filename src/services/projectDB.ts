// IndexedDB service for project persistence
// Stores media file blobs and project data

const DB_NAME = 'MASterSelectsDB';
const DB_VERSION = 1;

// Store names
const STORES = {
  MEDIA_FILES: 'mediaFiles',
  PROJECTS: 'projects',
} as const;

export interface StoredMediaFile {
  id: string;
  name: string;
  type: 'video' | 'audio' | 'image';
  blob: Blob;
  thumbnailBlob?: Blob;
  duration?: number;
  width?: number;
  height?: number;
  createdAt: number;
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
  async getStats(): Promise<{ mediaFiles: number; projects: number }> {
    const db = await this.init();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORES.MEDIA_FILES, STORES.PROJECTS], 'readonly');

      const mediaRequest = transaction.objectStore(STORES.MEDIA_FILES).count();
      const projectRequest = transaction.objectStore(STORES.PROJECTS).count();

      let mediaCount = 0;
      let projectCount = 0;

      mediaRequest.onsuccess = () => { mediaCount = mediaRequest.result; };
      projectRequest.onsuccess = () => { projectCount = projectRequest.result; };

      transaction.oncomplete = () => {
        resolve({ mediaFiles: mediaCount, projects: projectCount });
      };
      transaction.onerror = () => reject(transaction.error);
    });
  }
}

// Singleton instance
export const projectDB = new ProjectDatabase();
