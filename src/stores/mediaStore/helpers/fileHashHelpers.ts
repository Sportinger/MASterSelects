// File hash calculation for proxy deduplication

import { HASH_SIZE } from '../constants';
import { Logger } from '../../../services/logger';

const log = Logger.create('FileHash');

/**
 * Calculate SHA-256 hash of file (first 2MB + file size for speed).
 * Used for proxy and thumbnail deduplication.
 */
export async function calculateFileHash(file: File): Promise<string> {
  try {
    const slice = file.slice(0, Math.min(file.size, HASH_SIZE));
    const buffer = await slice.arrayBuffer();

    // Include file size in hash
    const sizeBuffer = new ArrayBuffer(8);
    const sizeView = new DataView(sizeBuffer);
    sizeView.setBigUint64(0, BigInt(file.size), true);

    // Combine buffers
    const combined = new Uint8Array(buffer.byteLength + 8);
    combined.set(new Uint8Array(buffer), 0);
    combined.set(new Uint8Array(sizeBuffer), buffer.byteLength);

    // Calculate SHA-256
    const hashBuffer = await crypto.subtle.digest('SHA-256', combined);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  } catch (e) {
    console.warn('[FileHash] Failed to calculate:', e);
    return '';
  }
}
