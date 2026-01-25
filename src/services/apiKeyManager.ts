// API Key Manager
// Securely stores and retrieves API keys using Web Crypto API encryption

import { Logger } from './logger';

const log = Logger.create('ApiKeyManager');

const DB_NAME = 'multicam-settings';
const STORE_NAME = 'api-keys';
const ENCRYPTION_KEY_ID = 'encryption-key';

// Supported API key types
export type ApiKeyType = 'openai' | 'assemblyai' | 'deepgram' | 'piapi' | 'youtube' | 'klingAccessKey' | 'klingSecretKey';

// Key IDs for each API key type (stored in IndexedDB)
const KEY_IDS: Record<ApiKeyType, string> = {
  openai: 'openai-api-key',
  assemblyai: 'assemblyai-api-key',
  deepgram: 'deepgram-api-key',
  piapi: 'piapi-api-key',
  youtube: 'youtube-api-key',
  klingAccessKey: 'kling-access-key',
  klingSecretKey: 'kling-secret-key',
};

// Legacy key ID for backwards compatibility
const LEGACY_KEY_ID = 'claude-api-key';

/**
 * Generate a random encryption key
 */
async function generateEncryptionKey(): Promise<CryptoKey> {
  return crypto.subtle.generateKey(
    { name: 'AES-GCM', length: 256 },
    true, // extractable
    ['encrypt', 'decrypt']
  );
}

/**
 * Export a CryptoKey to raw bytes
 */
async function exportKey(key: CryptoKey): Promise<ArrayBuffer> {
  return crypto.subtle.exportKey('raw', key);
}

/**
 * Import raw bytes as a CryptoKey
 */
async function importKey(rawKey: ArrayBuffer): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    rawKey,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  );
}

/**
 * Encrypt a string using AES-GCM
 */
async function encrypt(text: string, key: CryptoKey): Promise<{ iv: Uint8Array; data: ArrayBuffer }> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encoder = new TextEncoder();
  const data = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    encoder.encode(text)
  );
  return { iv, data };
}

/**
 * Decrypt data using AES-GCM
 */
async function decrypt(encryptedData: ArrayBuffer, iv: Uint8Array, key: CryptoKey): Promise<string> {
  const decrypted = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: iv as Uint8Array<ArrayBuffer> },
    key,
    encryptedData
  );
  const decoder = new TextDecoder();
  return decoder.decode(decrypted);
}

/**
 * Open the IndexedDB database
 */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
  });
}

/**
 * Get a value from IndexedDB
 */
async function dbGet<T>(id: string): Promise<T | null> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(id);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result?.value ?? null);
  });
}

/**
 * Set a value in IndexedDB
 */
async function dbSet(id: string, value: any): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.put({ id, value });

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

/**
 * Delete a value from IndexedDB
 */
async function dbDelete(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    const request = store.delete(id);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

class ApiKeyManager {
  private encryptionKey: CryptoKey | null = null;

  /**
   * Get or create the encryption key
   */
  private async getEncryptionKey(): Promise<CryptoKey> {
    if (this.encryptionKey) {
      return this.encryptionKey;
    }

    // Try to load existing key
    const storedKey = await dbGet<ArrayBuffer>(ENCRYPTION_KEY_ID);
    if (storedKey) {
      this.encryptionKey = await importKey(storedKey);
      return this.encryptionKey;
    }

    // Generate new key
    this.encryptionKey = await generateEncryptionKey();
    const rawKey = await exportKey(this.encryptionKey);
    await dbSet(ENCRYPTION_KEY_ID, rawKey);

    return this.encryptionKey;
  }

  /**
   * Store an API key securely by type
   */
  async storeKeyByType(keyType: ApiKeyType, apiKey: string): Promise<void> {
    if (!apiKey) {
      // If empty, delete the key
      await this.clearKeyByType(keyType);
      return;
    }

    const key = await this.getEncryptionKey();
    const { iv, data } = await encrypt(apiKey, key);
    const keyId = KEY_IDS[keyType];

    await dbSet(keyId, {
      iv: Array.from(iv),
      data: Array.from(new Uint8Array(data)),
    });

    log.info(`API key stored: ${keyType}`);
  }

  /**
   * Retrieve an API key by type
   */
  async getKeyByType(keyType: ApiKeyType): Promise<string | null> {
    const keyId = KEY_IDS[keyType];
    const stored = await dbGet<{ iv: number[]; data: number[] }>(keyId);
    if (!stored) {
      return null;
    }

    const key = await this.getEncryptionKey();
    const iv = new Uint8Array(stored.iv);
    const data = new Uint8Array(stored.data).buffer;

    try {
      return await decrypt(data, iv, key);
    } catch (error) {
      log.error(`Failed to decrypt API key: ${keyType}`, error);
      return null;
    }
  }

  /**
   * Check if an API key is stored by type
   */
  async hasKeyByType(keyType: ApiKeyType): Promise<boolean> {
    const keyId = KEY_IDS[keyType];
    const stored = await dbGet(keyId);
    return stored !== null;
  }

  /**
   * Clear an API key by type
   */
  async clearKeyByType(keyType: ApiKeyType): Promise<void> {
    const keyId = KEY_IDS[keyType];
    await dbDelete(keyId);
    log.info(`API key cleared: ${keyType}`);
  }

  /**
   * Get all stored API keys
   */
  async getAllKeys(): Promise<Record<ApiKeyType, string>> {
    const keys: Record<ApiKeyType, string> = {
      openai: '',
      assemblyai: '',
      deepgram: '',
      piapi: '',
      youtube: '',
      klingAccessKey: '',
      klingSecretKey: '',
    };

    for (const keyType of Object.keys(KEY_IDS) as ApiKeyType[]) {
      const value = await this.getKeyByType(keyType);
      if (value) {
        keys[keyType] = value;
      }
    }

    return keys;
  }

  /**
   * Store multiple API keys at once
   */
  async storeAllKeys(keys: Partial<Record<ApiKeyType, string>>): Promise<void> {
    for (const [keyType, value] of Object.entries(keys)) {
      if (value !== undefined) {
        await this.storeKeyByType(keyType as ApiKeyType, value);
      }
    }
  }

  // ============================================
  // Legacy methods for backwards compatibility
  // ============================================

  /**
   * Store an API key securely (legacy - uses openai key)
   * @deprecated Use storeKeyByType instead
   */
  async storeKey(apiKey: string): Promise<void> {
    // Store in legacy location for backwards compatibility
    const key = await this.getEncryptionKey();
    const { iv, data } = await encrypt(apiKey, key);

    await dbSet(LEGACY_KEY_ID, {
      iv: Array.from(iv),
      data: Array.from(new Uint8Array(data)),
    });

    log.info('API key stored (legacy)');
  }

  /**
   * Retrieve the stored API key (legacy)
   * @deprecated Use getKeyByType instead
   */
  async getKey(): Promise<string | null> {
    const stored = await dbGet<{ iv: number[]; data: number[] }>(LEGACY_KEY_ID);
    if (!stored) {
      return null;
    }

    const key = await this.getEncryptionKey();
    const iv = new Uint8Array(stored.iv);
    const data = new Uint8Array(stored.data).buffer;

    try {
      return await decrypt(data, iv, key);
    } catch (error) {
      log.error('Failed to decrypt API key', error);
      return null;
    }
  }

  /**
   * Check if an API key is stored (legacy)
   * @deprecated Use hasKeyByType instead
   */
  async hasKey(): Promise<boolean> {
    const stored = await dbGet(LEGACY_KEY_ID);
    return stored !== null;
  }

  /**
   * Clear the stored API key (legacy)
   * @deprecated Use clearKeyByType instead
   */
  async clearKey(): Promise<void> {
    await dbDelete(LEGACY_KEY_ID);
    log.info('API key cleared (legacy)');
  }
}

// Singleton instance
export const apiKeyManager = new ApiKeyManager();
