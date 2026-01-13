/**
 * Native Helper Service
 *
 * Provides hardware-accelerated video decoding and encoding through
 * a native helper application that runs locally.
 */

export { NativeHelperClient } from './NativeHelperClient';
export type { NativeHelperConfig, DecodedFrame, ConnectionStatus } from './NativeHelperClient';

export { NativeDecoder, isNativeHelperAvailable, getNativeCodecs } from './NativeDecoder';
export type { NativeDecoderOptions } from './NativeDecoder';

export type {
  FileMetadata,
  SystemInfo,
  EncodeOutput,
  AudioSettings,
  Command,
  Response,
} from './protocol';

export { ERROR_CODES } from './protocol';
