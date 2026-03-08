import '@testing-library/jest-dom'
import { afterEach, vi } from 'vitest'

// Mock modules that have side effects requiring browser APIs (WebGPU, HMR, etc.)
// These must be vi.mock() calls at the top level so they're hoisted before imports

vi.mock('../src/engine/WebGPUEngine', () => ({
  engine: {
    start: vi.fn(),
    stop: vi.fn(),
    render: vi.fn(),
    setGeneratingRamPreview: vi.fn(),
    clearCompositeCache: vi.fn(),
    cacheCompositeFrame: vi.fn(),
    requestRender: vi.fn(),
    requestNewFrameRender: vi.fn(),
    cleanupVideo: vi.fn(),
    clearVideoCache: vi.fn(),
    getTextureManager: vi.fn().mockReturnValue({
      updateCanvasTexture: vi.fn().mockReturnValue(true),
    }),
  },
  WebGPUEngine: vi.fn(),
}))

vi.mock('../src/engine/WebCodecsPlayer', () => ({
  WebCodecsPlayer: vi.fn(),
}))

// Mock services needed by clipSlice and its sub-modules
vi.mock('../src/services/textRenderer', () => ({
  textRenderer: {
    createCanvas: vi.fn().mockReturnValue(document.createElement('canvas')),
    render: vi.fn(),
  },
}))

vi.mock('../src/services/googleFontsService', () => ({
  googleFontsService: {
    loadFont: vi.fn().mockResolvedValue(undefined),
  },
}))

vi.mock('../src/services/thumbnailRenderer', () => ({
  thumbnailRenderer: {
    generateClipThumbnails: vi.fn().mockResolvedValue([]),
  },
}))

vi.mock('../src/services/nativeHelper', () => ({
  NativeDecoder: vi.fn(),
}))

vi.mock('../src/services/nativeHelper/NativeHelperClient', () => ({
  NativeHelperClient: vi.fn(),
}))

vi.mock('../src/services/layerBuilder', () => ({
  layerBuilder: {
    invalidateCache: vi.fn(),
    buildLayers: vi.fn().mockReturnValue([]),
    buildLayersFromStore: vi.fn().mockReturnValue([]),
  },
}))

vi.mock('../src/services/proxyFrameCache', () => ({
  proxyFrameCache: {
    getCachedRanges: vi.fn().mockReturnValue([]),
    cancelPreload: vi.fn(),
  },
}))

vi.mock('../src/services/audioAnalyzer', () => ({
  audioAnalyzer: {
    generateFingerprint: vi.fn(),
  },
}))

vi.mock('../src/services/fileSystemService', () => ({
  fileSystemService: {
    getFileHandle: vi.fn(),
  },
}))

// Mock store modules that trigger heavy import chains
vi.mock('../src/stores/mediaStore', () => ({
  useMediaStore: Object.assign(vi.fn(() => ({})), {
    getState: vi.fn(() => ({
      files: [],
      compositions: [],
      folders: [],
      selectedIds: [],
      expandedFolderIds: [],
      textItems: [],
      solidItems: [],
      activeCompositionId: null,
      outputResolution: { width: 1920, height: 1080 },
      addMediaFile: vi.fn(),
      updateComposition: vi.fn(),
      getActiveComposition: vi.fn().mockReturnValue({ width: 1920, height: 1080 }),
      getOrCreateTextFolder: vi.fn().mockReturnValue('text-folder-1'),
      createTextItem: vi.fn(),
      getOrCreateSolidFolder: vi.fn().mockReturnValue('solid-folder-1'),
      createSolidItem: vi.fn(),
    })),
    setState: vi.fn(),
    subscribe: vi.fn(),
  }),
}))

vi.mock('../src/stores/settingsStore', () => ({
  useSettingsStore: Object.assign(vi.fn(() => ({})), {
    getState: vi.fn(() => ({
      outputResolution: { width: 1920, height: 1080 },
    })),
    setState: vi.fn(),
    subscribe: vi.fn(),
  }),
}))

// Mock WebGPU API
Object.defineProperty(navigator, 'gpu', {
  value: {
    requestAdapter: vi.fn().mockResolvedValue({
      requestDevice: vi.fn().mockResolvedValue({
        createBuffer: vi.fn(),
        createTexture: vi.fn(),
        createShaderModule: vi.fn(),
        createBindGroupLayout: vi.fn(),
        createPipelineLayout: vi.fn(),
        createRenderPipeline: vi.fn(),
        createComputePipeline: vi.fn(),
        createBindGroup: vi.fn(),
        createCommandEncoder: vi.fn(),
        queue: {
          submit: vi.fn(),
          writeBuffer: vi.fn(),
          copyExternalImageToTexture: vi.fn(),
        },
        destroy: vi.fn(),
      }),
      features: new Set(),
      limits: {},
    }),
    getPreferredCanvasFormat: vi.fn().mockReturnValue('bgra8unorm'),
  },
  writable: true,
  configurable: true,
})

// Mock WebCodecs
class MockVideoDecoder {
  static isConfigSupported = vi.fn().mockResolvedValue({ supported: true })
  configure = vi.fn()
  decode = vi.fn()
  flush = vi.fn().mockResolvedValue(undefined)
  close = vi.fn()
  reset = vi.fn()
  state = 'unconfigured' as const
}

class MockVideoFrame {
  readonly timestamp: number
  readonly codedWidth: number
  readonly codedHeight: number
  constructor(data: unknown, init?: { timestamp?: number; codedWidth?: number; codedHeight?: number }) {
    this.timestamp = init?.timestamp ?? 0
    this.codedWidth = init?.codedWidth ?? 1920
    this.codedHeight = init?.codedHeight ?? 1080
  }
  close = vi.fn()
  clone = vi.fn().mockReturnThis()
}

if (typeof globalThis.VideoDecoder === 'undefined') {
  (globalThis as Record<string, unknown>).VideoDecoder = MockVideoDecoder
}
if (typeof globalThis.VideoFrame === 'undefined') {
  (globalThis as Record<string, unknown>).VideoFrame = MockVideoFrame
}

// Mock AudioContext
class MockAudioContext {
  sampleRate = 44100
  state = 'running' as const
  destination = { maxChannelCount: 2 }
  createGain = vi.fn().mockReturnValue({
    gain: { value: 1, setValueAtTime: vi.fn() },
    connect: vi.fn(),
    disconnect: vi.fn(),
  })
  createBufferSource = vi.fn().mockReturnValue({
    buffer: null,
    connect: vi.fn(),
    disconnect: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
  })
  createAnalyser = vi.fn().mockReturnValue({
    fftSize: 2048,
    frequencyBinCount: 1024,
    getByteFrequencyData: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
  })
  decodeAudioData = vi.fn().mockResolvedValue({
    numberOfChannels: 2,
    sampleRate: 44100,
    length: 44100,
    duration: 1,
    getChannelData: vi.fn().mockReturnValue(new Float32Array(44100)),
  })
  close = vi.fn().mockResolvedValue(undefined)
  resume = vi.fn().mockResolvedValue(undefined)
}

if (typeof globalThis.AudioContext === 'undefined') {
  (globalThis as Record<string, unknown>).AudioContext = MockAudioContext
}

// Mock HTMLMediaElement play/pause
HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined)
HTMLMediaElement.prototype.pause = vi.fn()

// Clean up mocks after each test
afterEach(() => {
  vi.clearAllMocks()
})
