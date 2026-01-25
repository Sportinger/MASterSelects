// API Key Manager
// Securely stores and retrieves API keys using Web Crypto API encryption

import { Logger } from './logger';

const log = Logger.create('ApiKeyManager');

const DB_NAME = 'multicam-settings';
const STORE_NAME = 'api-keys';
const KEY_ID = 'claude-api-key';
const ENCRYPTION_KEY_ID = 'encryption-key';

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
   * Store an API key securely
   */
  async storeKey(apiKey: string): Promise<void> {
    const key = await this.getEncryptionKey();
    const { iv, data } = await encrypt(apiKey, key);

    await dbSet(KEY_ID, {
      iv: Array.from(iv),
      data: Array.from(new Uint8Array(data)),
    });

    log.info('API key stored');
  }

  /**
   * Retrieve the stored API key
   */
  async getKey(): Promise<string | null> {
    const stored = await dbGet<{ iv: number[]; data: number[] }>(KEY_ID);
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
   * Check if an API key is stored
   */
  async hasKey(): Promise<boolean> {
    const stored = await dbGet(KEY_ID);
    return stored !== null;
  }

  /**
   * Clear the stored API key
   */
  async clearKey(): Promise<void> {
    await dbDelete(KEY_ID);
    log.info('API key cleared');
  }
}

// Singleton instance
export const apiKeyManager = new ApiKeyManager();
