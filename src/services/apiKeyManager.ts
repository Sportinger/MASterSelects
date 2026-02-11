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

// File encryption constants — deterministic key derived via PBKDF2
// Security model: obfuscation on local disk, not secure against source code access
const FILE_KEY_SALT = 'MasterSelects-FileKeys-v1-2024';
const FILE_KEY_PASSPHRASE = 'ms-local-encryption-key-v1';
const FILE_KEY_ITERATIONS = 100000;

interface FileKeyEntry {
  iv: number[];
  data: number[];
}

interface KeysFileData {
  v: 1;
  keys: Record<string, FileKeyEntry>;
}

async function deriveFileEncryptionKey(): Promise<CryptoKey> {
  const baseKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(FILE_KEY_PASSPHRASE),
    'PBKDF2',
    false,
    ['deriveKey']
  );
  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      hash: 'SHA-256',
      salt: new TextEncoder().encode(FILE_KEY_SALT),
      iterations: FILE_KEY_ITERATIONS,
    },
    baseKey,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

class ApiKeyManager {
  private encryptionKey: CryptoKey | null = null;
  private fileKey: CryptoKey | null = null;

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
  // File-based key export/import (.keys.enc)
  // ============================================

  private async getFileKey(): Promise<CryptoKey> {
    if (this.fileKey) return this.fileKey;
    this.fileKey = await deriveFileEncryptionKey();
    return this.fileKey;
  }

  /**
   * Export all stored keys as an encrypted JSON string for file storage.
   * Returns null if no keys are stored.
   */
  async exportKeysForFile(): Promise<string | null> {
    try {
      const allKeys = await this.getAllKeys();

      // Filter out empty keys
      const nonEmpty = Object.entries(allKeys).filter(([, v]) => v !== '');
      if (nonEmpty.length === 0) return null;

      const fileKey = await this.getFileKey();
      const fileKeys: Record<string, FileKeyEntry> = {};

      for (const [keyType, value] of nonEmpty) {
        const { iv, data } = await encrypt(value, fileKey);
        fileKeys[keyType] = {
          iv: Array.from(iv),
          data: Array.from(new Uint8Array(data)),
        };
      }

      const payload: KeysFileData = { v: 1, keys: fileKeys };
      return JSON.stringify(payload, null, 2);
    } catch (error) {
      log.error('Failed to export keys for file:', error);
      return null;
    }
  }

  /**
   * Import keys from an encrypted file string and store them in IndexedDB.
   * Returns true if at least one key was restored.
   */
  async importKeysFromFile(fileContent: string): Promise<boolean> {
    try {
      const payload = JSON.parse(fileContent) as KeysFileData;
      if (payload.v !== 1 || !payload.keys) {
        log.warn('Unknown keys file format');
        return false;
      }

      const fileKey = await this.getFileKey();
      let restored = 0;

      for (const [keyType, entry] of Object.entries(payload.keys)) {
        if (!KEY_IDS[keyType as ApiKeyType]) continue;

        try {
          const iv = new Uint8Array(entry.iv);
          const data = new Uint8Array(entry.data).buffer;
          const plaintext = await decrypt(data, iv, fileKey);

          if (plaintext) {
            await this.storeKeyByType(keyType as ApiKeyType, plaintext);
            restored++;
          }
        } catch (err) {
          log.warn(`Failed to decrypt file key: ${keyType}`, err);
        }
      }

      log.info(`Restored ${restored} keys from file`);
      return restored > 0;
    } catch (error) {
      log.error('Failed to import keys from file:', error);
      return false;
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
