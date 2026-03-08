import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

function createStorageMock(): Storage {
  const data = new Map<string, string>();
  return {
    get length() {
      return data.size;
    },
    clear() {
      data.clear();
    },
    getItem(key: string) {
      return data.has(key) ? data.get(key)! : null;
    },
    key(index: number) {
      return [...data.keys()][index] ?? null;
    },
    removeItem(key: string) {
      data.delete(key);
    },
    setItem(key: string, value: string) {
      data.set(key, String(value));
    },
  };
}

async function loadLoggerModule() {
  vi.resetModules();
  const storage = createStorageMock();
  vi.stubGlobal('localStorage', storage);
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: true } as Response)));
  const mod = await import('../../src/services/logger');
  (window as { LogSync?: { stop?: () => void } }).LogSync?.stop?.();
  return mod;
}

describe('logger hot-path buffering', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    (window as { LogSync?: { stop?: () => void } }).LogSync?.stop?.();
    vi.unstubAllGlobals();
  });

  it('does not buffer disabled debug or info logs', async () => {
    const { Logger } = await loadLoggerModule();
    Logger.disable();
    Logger.setLevel('WARN');

    const log = Logger.create('RenderHotPath');
    const before = Logger.getBuffer().length;

    log.debug('hidden debug');
    log.info('hidden info');
    log.warn('visible warn');

    const buffer = Logger.getBuffer();
    expect(buffer).toHaveLength(before + 1);
    expect(buffer.at(-1)?.message).toBe('visible warn');
  });

  it('keeps buffering when debug logging is explicitly enabled', async () => {
    const { Logger } = await loadLoggerModule();
    Logger.setLevel('DEBUG');
    Logger.enable('RenderHotPath');

    const log = Logger.create('RenderHotPath');
    const before = Logger.getBuffer().length;

    log.debug('visible debug');

    const buffer = Logger.getBuffer();
    expect(buffer).toHaveLength(before + 1);
    expect(buffer.at(-1)).toMatchObject({
      level: 'DEBUG',
      module: 'RenderHotPath',
      message: 'visible debug',
    });
  });
});
