//! NVDEC hardware video decoder module.
//!
//! Provides hardware-accelerated video decoding on NVIDIA GPUs through
//! the CUVID/NVDEC API. The nvcuvid library is loaded dynamically at
//! runtime, so the application can gracefully degrade if no NVIDIA
//! hardware is present.
//!
//! # Module Structure
//!
//! - [`ffi`] — Raw FFI bindings for nvcuvid (loaded via `libloading`).
//! - [`session`] — Safe session wrapper with RAII resource management.
//! - [`decoder`] — High-level decoder implementing `HwDecoder` trait.
//!
//! # Architecture
//!
//! The decode pipeline works as follows:
//!
//! 1. The caller loads the NVDEC library once via [`NvcuvidLibrary::load()`].
//! 2. An [`NvDecoder`] is created for each video stream to decode.
//! 3. Compressed NAL units (Annex-B format) are fed via the `HwDecoder::decode()` trait method.
//! 4. Internally, the CUVID video parser triggers callbacks:
//!    - **Sequence callback**: Creates the NVDEC hardware decoder when SPS is received.
//!    - **Decode callback**: Submits a picture for hardware decoding.
//!    - **Display callback**: Queues a decoded frame for retrieval.
//! 5. Decoded frames are mapped from GPU memory as NV12 device pointers.
//! 6. The caller copies the frame data and releases the NVDEC surface.
//!
//! # Usage
//!
//! ```ignore
//! use ms_decoder::nvdec::{NvDecoder, NvcuvidLibrary};
//! use ms_common::{HwDecoder, VideoCodec};
//! use std::sync::Arc;
//!
//! // Load the NVDEC library (once, shared across decoders)
//! let lib = Arc::new(NvcuvidLibrary::load()?);
//!
//! // Create a decoder for H.264
//! let mut decoder = NvDecoder::new(lib, VideoCodec::H264)?;
//!
//! // Feed packets from the demuxer
//! for packet in demuxer.read_video_packets() {
//!     if let Some(frame) = decoder.decode(&packet)? {
//!         // frame.device_ptr is NV12 on GPU — copy it before releasing
//!         gpu_copy(frame.device_ptr, persistent_buffer);
//!         decoder.release_frame(&frame);
//!     }
//! }
//!
//! // Flush remaining frames at end of stream
//! let remaining = decoder.flush()?;
//! for frame in &remaining {
//!     gpu_copy(frame.device_ptr, persistent_buffer);
//!     decoder.release_frame(frame);
//! }
//! ```

pub mod decoder;
pub mod ffi;
pub mod session;

// Re-export primary public types
pub use decoder::NvDecoder;
pub use ffi::{NvcuvidLibrary, NvcuvidLoadError};
pub use session::{DecodedFrameInfo, MappedFrame, NvDecSession, SessionStats};
