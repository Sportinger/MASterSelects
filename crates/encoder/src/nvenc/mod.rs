//! NVENC hardware video encoder module.
//!
//! Provides hardware-accelerated video encoding on NVIDIA GPUs through
//! the NVENC API. The nvEncodeAPI library is loaded dynamically at runtime,
//! so the application can gracefully degrade if no NVIDIA hardware is present.
//!
//! # Module Structure
//!
//! - [`ffi`] -- Raw FFI bindings for nvEncodeAPI (loaded via `libloading`).
//! - [`params`] -- Parameter builders mapping `EncoderConfig` to NVENC structs.
//! - [`buffer`] -- Input/output buffer pool management.
//! - [`NvEncoder`] -- High-level encoder implementing `HwEncoder` trait.
//!
//! # Architecture
//!
//! The encode pipeline works as follows:
//!
//! 1. Load the NVENC library once via [`NvencLibrary::load()`].
//! 2. Create an [`NvEncoder`] with the desired `EncoderConfig`.
//! 3. For each frame:
//!    a. Register the external CUDA device pointer.
//!    b. Map it as NVENC input.
//!    c. Encode the picture.
//!    d. Lock the output bitstream and copy to `EncodedPacket`.
//!    e. Unlock the bitstream and unmap the input.
//! 4. Flush remaining packets at end of stream.
//!
//! # Usage
//!
//! ```ignore
//! use ms_encoder::nvenc::{NvEncoder, NvencLibrary};
//! use ms_common::HwEncoder;
//!
//! let lib = Arc::new(NvencLibrary::load()?);
//! let config = EncoderConfig { /* ... */ };
//! let mut encoder = NvEncoder::new(lib, &config)?;
//!
//! // Encode frames
//! for frame in gpu_frames {
//!     let packet = encoder.encode(&frame)?;
//!     muxer.write_packet(&packet)?;
//! }
//!
//! // Flush remaining
//! let remaining = encoder.flush()?;
//! ```

pub mod buffer;
pub mod ffi;
pub mod params;

// Re-export primary public types
pub use ffi::{NvencLibrary, NvencFunctionList};
pub use buffer::{BufferPool, MappedInput, OutputBuffer, RegisteredResource};

use std::ffi::c_void;
use std::sync::Arc;

use tracing::{debug, error, info, warn};

use ms_common::config::EncoderConfig;
use ms_common::gpu_traits::{EncodedPacket, HwEncoder};
use ms_common::packet::GpuFrame;
use ms_common::types::TimeCode;
use ms_common::EncodeError;

use ffi::{
    check_nvenc_status, NvEncBufferFormat, NvEncLockBitstream,
    NvEncOpenEncodeSessionExParams, NvEncPicParams, NvEncPicType, NV_ENC_PIC_FLAG_EOS,
    NV_ENC_SUCCESS,
};

/// The number of output bitstream buffers in the pool.
const DEFAULT_OUTPUT_BUFFER_COUNT: usize = 4;

// ---------------------------------------------------------------------------
// NvEncoder
// ---------------------------------------------------------------------------

/// High-level NVENC encoder implementing the `HwEncoder` trait.
///
/// Manages the NVENC encoder session lifecycle, input resource registration,
/// and output bitstream handling. Encodes `GpuFrame`s (CUDA device pointers)
/// into H.264 or H.265 bitstream packets.
///
/// # Resource Management (RAII)
///
/// The encoder session and all buffers are cleaned up on drop. The drop order
/// is: buffer pool -> encoder session -> library reference.
pub struct NvEncoder {
    /// NVENC encoder handle.
    encoder: *mut c_void,
    /// Reference to the loaded NVENC library.
    lib: Arc<NvencLibrary>,
    /// Output buffer pool.
    buffer_pool: BufferPool,
    /// Encoder configuration.
    config: EncoderConfig,
    /// Total frames encoded.
    frames_encoded: u64,
    /// Input buffer format used for encoding.
    input_format: NvEncBufferFormat,
    /// The NvEncConfig must be kept alive for the duration of the encoder
    /// session because NvEncInitializeParams holds a raw pointer to it.
    /// We store it here as a boxed value to ensure stable address.
    _enc_config: Box<ffi::NvEncConfig>,
}

// SAFETY: NvEncoder contains raw pointers to NVENC handles. The encoder
// handle is only accessed through the NVENC API which is thread-safe when
// used from a single thread (the encode thread). NvEncoder is Send but
// NOT Sync -- it should only be used from one thread.
unsafe impl Send for NvEncoder {}

impl std::fmt::Debug for NvEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvEncoder")
            .field("codec", &self.config.codec)
            .field("resolution", &self.config.resolution)
            .field("fps", &self.config.fps)
            .field("preset", &self.config.preset)
            .field("frames_encoded", &self.frames_encoded)
            .field("buffer_pool", &self.buffer_pool)
            .finish()
    }
}

impl NvEncoder {
    /// Create a new NVENC encoder.
    ///
    /// Opens an NVENC encode session, initializes the encoder with the given
    /// configuration, and creates the output buffer pool.
    ///
    /// # Arguments
    /// * `lib` -- Loaded NVENC library (shared across encoders).
    /// * `config` -- Encoder configuration.
    ///
    /// # Errors
    /// Returns `EncodeError::HwEncoderInit` if session creation or initialization fails.
    /// Returns `EncodeError::UnsupportedCodec` for codecs that NVENC cannot encode.
    pub fn new(lib: Arc<NvencLibrary>, config: &EncoderConfig) -> Result<Self, EncodeError> {
        // Validate config first
        params::validate_config(config)?;

        // Build NVENC params from our common config
        let (mut init_params, enc_config) = params::build_init_params(config)?;

        // Box the config so it has a stable address for the pointer in init_params
        let mut boxed_config = Box::new(enc_config);
        init_params.encode_config = &mut *boxed_config as *mut ffi::NvEncConfig;

        // Open encode session
        // Device type 0 = CUDA. In a real implementation, we would pass
        // the CUcontext here. For now, null means "current CUDA context".
        let mut session_params = NvEncOpenEncodeSessionExParams {
            device_type: 0,
            device: std::ptr::null_mut(),
            ..NvEncOpenEncodeSessionExParams::default()
        };

        let mut encoder: *mut c_void = std::ptr::null_mut();

        // SAFETY: session_params is properly initialized. NVENC writes the
        // encoder handle to `encoder`. The device (CUcontext) must be valid
        // and current on this thread.
        let status = unsafe {
            (lib.api.nvEncOpenEncodeSessionEx)(&mut session_params, &mut encoder)
        };

        check_nvenc_status(status, "nvEncOpenEncodeSessionEx").map_err(|reason| {
            EncodeError::HwEncoderInit(reason)
        })?;

        // Initialize the encoder
        // SAFETY: encoder is a valid handle from nvEncOpenEncodeSessionEx.
        // init_params is fully initialized and its encode_config pointer
        // points to valid memory (boxed_config).
        let status = unsafe {
            (lib.api.nvEncInitializeEncoder)(encoder, &mut init_params)
        };

        if status != NV_ENC_SUCCESS {
            // Clean up the session on failure
            // SAFETY: encoder is a valid handle that we just opened.
            unsafe { (lib.api.nvEncDestroyEncoder)(encoder) };
            return Err(EncodeError::HwEncoderInit(
                check_nvenc_status(status, "nvEncInitializeEncoder")
                    .unwrap_err(),
            ));
        }

        // Create buffer pool
        // SAFETY: encoder is a valid NVENC handle from nvEncOpenEncodeSessionEx,
        // which was successfully initialized above.
        let buffer_pool = unsafe { BufferPool::new(encoder, lib.clone(), DEFAULT_OUTPUT_BUFFER_COUNT) }
            .map_err(|e| EncodeError::HwEncoderInit(e.to_string()))?;

        let input_format = params::pixel_format_to_nvenc(ms_common::PixelFormat::Nv12);

        info!(
            codec = config.codec.display_name(),
            width = config.resolution.width,
            height = config.resolution.height,
            fps = %config.fps,
            preset = ?config.preset,
            "NVENC encoder initialized"
        );

        Ok(Self {
            encoder,
            lib,
            buffer_pool,
            config: config.clone(),
            frames_encoded: 0,
            input_format,
            _enc_config: boxed_config,
        })
    }

    /// Get the number of frames encoded so far.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Get the input buffer format used for encoding.
    pub fn input_format(&self) -> NvEncBufferFormat {
        self.input_format
    }

    /// Get a reference to the encoder configuration.
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Encode a single GPU frame and produce an encoded packet.
    ///
    /// Registers the frame's device pointer as NVENC input, encodes it,
    /// and returns the bitstream data as an `EncodedPacket`.
    fn encode_frame(&mut self, frame: &GpuFrame) -> Result<EncodedPacket, EncodeError> {
        let input_format = params::pixel_format_to_nvenc(frame.format);

        // Register the external CUDA device pointer
        let resource = self
            .buffer_pool
            .register_input(
                frame.device_ptr,
                frame.resolution.width,
                frame.resolution.height,
                frame.pitch,
                input_format,
            )
            .map_err(|e| EncodeError::EncodeFailed {
                frame: self.frames_encoded,
                reason: e.to_string(),
            })?;

        // Map it for NVENC use
        let mapped = self
            .buffer_pool
            .map_input(&resource)
            .map_err(|e| EncodeError::EncodeFailed {
                frame: self.frames_encoded,
                reason: e.to_string(),
            })?;

        // Acquire an output buffer
        let (buf_idx, output_handle) =
            self.buffer_pool
                .acquire_output()
                .map_err(|e| EncodeError::EncodeFailed {
                    frame: self.frames_encoded,
                    reason: e.to_string(),
                })?;

        // Set up encode parameters
        let mut pic_params = NvEncPicParams {
            input_width: frame.resolution.width,
            input_height: frame.resolution.height,
            input_pitch: frame.pitch,
            input_buffer: mapped.mapped_handle,
            output_bitstream: output_handle,
            buffer_fmt: mapped.format,
            pic_struct: 1, // Frame encoding (progressive)
            input_time_stamp: (frame.pts.as_secs() * 1_000_000.0) as u64,
            frame_idx: self.frames_encoded as u32,
            ..NvEncPicParams::default()
        };

        // SAFETY: encoder is a valid handle. pic_params references valid
        // mapped input and output buffers. The input device pointer must
        // contain valid frame data.
        let status = unsafe {
            (self.lib.api.nvEncEncodePicture)(self.encoder, &mut pic_params)
        };

        // Read the output regardless of encode status (NVENC may have produced output)
        let packet = self.read_output(output_handle, frame.pts)?;

        // Release resources in reverse order
        self.buffer_pool.release_output(buf_idx);

        let _ = self.buffer_pool.unmap_input(&mapped).map_err(|e| {
            warn!(error = %e, "Failed to unmap input resource");
        });

        let _ = self.buffer_pool.unregister_input(&resource).map_err(|e| {
            warn!(error = %e, "Failed to unregister input resource");
        });

        check_nvenc_status(status, "nvEncEncodePicture").map_err(|reason| {
            EncodeError::EncodeFailed {
                frame: self.frames_encoded,
                reason,
            }
        })?;

        self.frames_encoded += 1;

        debug!(
            frame = self.frames_encoded,
            pts = frame.pts.as_secs(),
            size = packet.data.len(),
            keyframe = packet.is_keyframe,
            "Encoded frame"
        );

        Ok(packet)
    }

    /// Read encoded data from an output bitstream buffer.
    fn read_output(
        &self,
        output_handle: *mut c_void,
        pts: TimeCode,
    ) -> Result<EncodedPacket, EncodeError> {
        let mut lock_params = NvEncLockBitstream {
            output_bitstream: output_handle,
            ..NvEncLockBitstream::default()
        };

        // SAFETY: encoder is valid and output_handle is a valid bitstream
        // buffer from our pool. NVENC fills the output fields of lock_params.
        let status = unsafe {
            (self.lib.api.nvEncLockBitstream)(self.encoder, &mut lock_params)
        };

        check_nvenc_status(status, "nvEncLockBitstream").map_err(|reason| {
            EncodeError::EncodeFailed {
                frame: self.frames_encoded,
                reason,
            }
        })?;

        // Copy the bitstream data
        let data = if !lock_params.bitstream_buffer_ptr.is_null()
            && lock_params.bitstream_size_in_bytes > 0
        {
            // SAFETY: bitstream_buffer_ptr and bitstream_size_in_bytes are set
            // by nvEncLockBitstream. The pointer is valid until nvEncUnlockBitstream.
            let slice = unsafe {
                std::slice::from_raw_parts(
                    lock_params.bitstream_buffer_ptr as *const u8,
                    lock_params.bitstream_size_in_bytes as usize,
                )
            };
            slice.to_vec()
        } else {
            Vec::new()
        };

        let is_keyframe = matches!(lock_params.pic_type, NvEncPicType::Idr | NvEncPicType::I);

        let output_pts = TimeCode::from_secs(lock_params.output_time_stamp as f64 / 1_000_000.0);

        // Unlock the bitstream
        // SAFETY: encoder is valid and output_handle was successfully locked.
        let unlock_status = unsafe {
            (self.lib.api.nvEncUnlockBitstream)(self.encoder, output_handle)
        };

        if unlock_status != NV_ENC_SUCCESS {
            warn!(
                status = unlock_status,
                "Failed to unlock NVENC bitstream buffer"
            );
        }

        Ok(EncodedPacket {
            data,
            pts,
            dts: output_pts,
            is_keyframe,
        })
    }

    /// Send an end-of-stream signal and flush remaining encoded packets.
    fn flush_internal(&mut self) -> Result<Vec<EncodedPacket>, EncodeError> {
        let mut packets = Vec::new();

        // Acquire an output buffer for the EOS signal
        let (buf_idx, output_handle) =
            self.buffer_pool
                .acquire_output()
                .map_err(|e| EncodeError::EncodeFailed {
                    frame: self.frames_encoded,
                    reason: e.to_string(),
                })?;

        let mut pic_params = NvEncPicParams {
            encode_params_flags: NV_ENC_PIC_FLAG_EOS,
            output_bitstream: output_handle,
            ..NvEncPicParams::default()
        };

        // SAFETY: encoder is valid. EOS pic_params has null input buffer
        // and the EOS flag set, which is the documented flush mechanism.
        let status = unsafe {
            (self.lib.api.nvEncEncodePicture)(self.encoder, &mut pic_params)
        };

        // Even if the flush call returns an error, try to read any output
        if status == NV_ENC_SUCCESS {
            match self.read_output(output_handle, TimeCode::ZERO) {
                Ok(packet) => {
                    if !packet.data.is_empty() {
                        packets.push(packet);
                    }
                }
                Err(e) => {
                    debug!(error = %e, "No output from flush signal");
                }
            }
        }

        self.buffer_pool.release_output(buf_idx);

        info!(
            flushed_packets = packets.len(),
            total_encoded = self.frames_encoded,
            "NVENC encoder flushed"
        );

        Ok(packets)
    }
}

impl HwEncoder for NvEncoder {
    /// Encode a GPU frame.
    ///
    /// The frame's `device_ptr` must be a valid CUDA device pointer containing
    /// frame data in the format specified by `frame.format`.
    fn encode(&mut self, frame: &GpuFrame) -> Result<EncodedPacket, EncodeError> {
        self.encode_frame(frame)
    }

    /// Flush remaining packets from the encoder pipeline.
    ///
    /// Sends an end-of-stream signal and retrieves all buffered encoded packets.
    fn flush(&mut self) -> Result<Vec<EncodedPacket>, EncodeError> {
        self.flush_internal()
    }
}

impl Drop for NvEncoder {
    fn drop(&mut self) {
        // Buffer pool is dropped first (it's a field, dropped in field order).
        // But we explicitly note that it will be dropped before the encoder.

        if !self.encoder.is_null() {
            debug!("Destroying NVENC encoder session");
            // SAFETY: encoder is a valid handle from nvEncOpenEncodeSessionEx.
            let status = unsafe { (self.lib.api.nvEncDestroyEncoder)(self.encoder) };
            if status != NV_ENC_SUCCESS {
                error!(status, "Failed to destroy NVENC encoder session");
            }
            self.encoder = std::ptr::null_mut();
        }

        info!(
            codec = self.config.codec.display_name(),
            frames_encoded = self.frames_encoded,
            "NVENC encoder destroyed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ms_common::config::{EncoderBitrate, EncoderPreset, EncoderProfile};
    use ms_common::types::{Rational, Resolution};

    #[test]
    fn default_output_buffer_count() {
        assert!(DEFAULT_OUTPUT_BUFFER_COUNT >= 2);
        assert!(DEFAULT_OUTPUT_BUFFER_COUNT <= 16);
    }

    #[test]
    fn encoder_config_roundtrip() {
        // Verify we can build params from a config without panicking
        let config = EncoderConfig {
            codec: ms_common::VideoCodec::H264,
            resolution: Resolution::HD,
            fps: Rational::FPS_30,
            bitrate: EncoderBitrate::Vbr {
                target: 20_000_000,
                max: 30_000_000,
            },
            preset: EncoderPreset::Medium,
            profile: EncoderProfile::High,
        };

        let (init, enc) = params::build_init_params(&config).unwrap();
        assert_eq!(init.encode_width, 1920);
        assert_eq!(init.encode_height, 1080);
        assert_ne!(enc.version, 0);
    }

    // GPU-dependent tests are ignored since they require NVIDIA hardware
    #[test]
    #[ignore]
    fn create_encoder_on_gpu() {
        let lib = Arc::new(NvencLibrary::load().expect("NVENC library"));
        let config = EncoderConfig {
            codec: ms_common::VideoCodec::H264,
            resolution: Resolution::HD,
            fps: Rational::FPS_30,
            bitrate: EncoderBitrate::Vbr {
                target: 20_000_000,
                max: 30_000_000,
            },
            preset: EncoderPreset::Medium,
            profile: EncoderProfile::High,
        };
        let encoder = NvEncoder::new(lib, &config);
        assert!(encoder.is_ok());
    }
}
