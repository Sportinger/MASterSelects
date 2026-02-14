//! High-level NVDEC decoder — implements the `HwDecoder` trait.
//!
//! `NvDecoder` wraps `NvDecSession` and provides the clean interface
//! defined by `ms_common::HwDecoder`. It accepts `VideoPacket`s from
//! the demuxer and returns `GpuFrame`s with NV12 data on GPU.
//!
//! ## Surface Lifecycle
//!
//! When the decoder maps a frame, NVDEC provides a temporary device pointer
//! that is only valid while the mapping is held. The `NvDecoder` keeps the
//! [`MappedFrame`] RAII guard alive in an internal collection so that the
//! returned `GpuFrame`'s device pointer remains valid.
//!
//! The caller must call [`NvDecoder::release_frame`] (or
//! [`NvDecoder::release_all_frames`]) after it has copied the frame data to
//! a persistent GPU buffer. Failing to release frames will eventually exhaust
//! the NVDEC surface pool.

use std::collections::VecDeque;
use std::sync::Arc;

use tracing::{debug, info, warn};

use ms_common::color::PixelFormat;
use ms_common::packet::{GpuFrame, VideoPacket};
use ms_common::types::{Resolution, TimeCode};
use ms_common::{DecodeError, HwDecoder, VideoCodec};

use super::ffi::NvcuvidLibrary;
use super::session::{MappedFrame, NvDecSession, SessionStats};

// ---------------------------------------------------------------------------
// Active frame tracking
// ---------------------------------------------------------------------------

/// An actively mapped frame that keeps the NVDEC surface alive.
///
/// The `GpuFrame` returned by `decode()` contains a raw device pointer that
/// is only valid while this guard exists. When the guard is dropped, the
/// NVDEC surface is unmapped and the device pointer becomes invalid.
struct ActiveFrame {
    /// The RAII mapped frame guard — dropping this unmaps the surface.
    _mapped: MappedFrame,
    /// The PTS of this frame, used to identify it for release.
    pts: TimeCode,
}

// ---------------------------------------------------------------------------
// NvDecoder
// ---------------------------------------------------------------------------

/// High-level NVDEC decoder that implements the `HwDecoder` trait.
///
/// Wraps an `NvDecSession` and converts between the common packet/frame
/// types and the NVDEC-specific types.
///
/// # Surface Management
///
/// Each call to `decode()` may produce a `GpuFrame` whose device pointer
/// is backed by a mapped NVDEC surface. The decoder keeps the mapping alive
/// internally until the caller explicitly releases it via [`release_frame`]
/// or [`release_all_frames`].
///
/// If the caller does not release frames, the NVDEC DPB will eventually fill
/// up and decoding will stall or fail. A typical workflow:
///
/// ```ignore
/// let lib = Arc::new(NvcuvidLibrary::load()?);
/// let mut decoder = NvDecoder::new(lib, VideoCodec::H264)?;
///
/// for packet in demuxer.read_video_packets() {
///     if let Some(frame) = decoder.decode(&packet)? {
///         // Copy frame.device_ptr to a persistent buffer (e.g., via CUDA memcpy)
///         gpu_copy(frame.device_ptr, persistent_buffer);
///
///         // Release the NVDEC surface so it can be reused
///         decoder.release_frame(&frame);
///     }
/// }
///
/// // Flush remaining frames
/// let remaining = decoder.flush()?;
/// for frame in &remaining {
///     gpu_copy(frame.device_ptr, persistent_buffer);
///     decoder.release_frame(frame);
/// }
/// ```
pub struct NvDecoder {
    /// The underlying NVDEC session.
    session: NvDecSession,
    /// The video codec being decoded.
    codec: VideoCodec,
    /// Output pixel format (always NV12 for 8-bit, P010 for 10-bit).
    output_format: PixelFormat,
    /// Cached output resolution (updated after first decode).
    output_resolution: Resolution,
    /// Total frames decoded (for statistics).
    frames_decoded: u64,
    /// Currently active (mapped) frames. These keep the NVDEC surfaces alive
    /// so that the GpuFrame device pointers remain valid.
    active_frames: VecDeque<ActiveFrame>,
    /// Maximum number of active frames before we start warning. This is a
    /// soft limit — we don't hard-fail, but log a warning because the DPB
    /// is likely filling up.
    max_active_frames: usize,
}

impl std::fmt::Debug for NvDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvDecoder")
            .field("codec", &self.codec)
            .field("output_format", &self.output_format)
            .field("output_resolution", &self.output_resolution)
            .field("frames_decoded", &self.frames_decoded)
            .field("active_frames", &self.active_frames.len())
            .field("decoder_ready", &self.session.is_decoder_ready())
            .finish()
    }
}

impl NvDecoder {
    /// Create a new NVDEC decoder for the given codec.
    ///
    /// # Arguments
    /// * `lib` — Loaded nvcuvid library (shared across decoders).
    /// * `codec` — The video codec to decode (H264, H265, VP9, AV1).
    ///
    /// # Errors
    /// Returns `DecodeError::UnsupportedCodec` if the codec is not supported by NVDEC.
    pub fn new(lib: Arc<NvcuvidLibrary>, codec: VideoCodec) -> Result<Self, DecodeError> {
        Self::with_config(lib, codec, 20, 4)
    }

    /// Create a new NVDEC decoder with explicit configuration.
    ///
    /// # Arguments
    /// * `lib` — Loaded nvcuvid library.
    /// * `codec` — The video codec.
    /// * `num_surfaces` — Number of DPB surfaces (8-32, higher = more memory but smoother).
    /// * `max_display_delay` — Maximum reordering delay in frames (0 = low latency).
    pub fn with_config(
        lib: Arc<NvcuvidLibrary>,
        codec: VideoCodec,
        num_surfaces: u32,
        max_display_delay: u32,
    ) -> Result<Self, DecodeError> {
        let session = NvDecSession::new(lib, codec, num_surfaces, max_display_delay)?;

        info!(
            codec = codec.display_name(),
            surfaces = num_surfaces,
            delay = max_display_delay,
            "NvDecoder created"
        );

        Ok(Self {
            session,
            codec,
            output_format: PixelFormat::Nv12,
            output_resolution: Resolution::new(0, 0),
            frames_decoded: 0,
            active_frames: VecDeque::with_capacity(8),
            // NVDEC typically supports num_output_surfaces (2) simultaneous
            // mappings, but we allow more in our tracking since the caller
            // may batch-process several frames before releasing.
            max_active_frames: num_surfaces.min(16) as usize,
        })
    }

    /// Get total number of frames decoded so far.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Check if the hardware decoder has been initialized (SPS parsed).
    pub fn is_ready(&self) -> bool {
        self.session.is_decoder_ready()
    }

    /// Get the number of frames waiting in the decode output queue.
    pub fn pending_frames(&self) -> usize {
        self.session.pending_frame_count()
    }

    /// Get the number of currently active (mapped) frames.
    ///
    /// These are frames that have been returned by `decode()` or `flush()`
    /// but not yet released via `release_frame()`.
    pub fn active_frame_count(&self) -> usize {
        self.active_frames.len()
    }

    /// Get decode session statistics.
    pub fn stats(&self) -> SessionStats {
        self.session.stats()
    }

    /// Release an active frame, unmapping its NVDEC surface.
    ///
    /// This must be called after the caller has copied the frame data to a
    /// persistent GPU buffer. The `GpuFrame`'s device pointer becomes invalid
    /// after this call.
    ///
    /// Frames are matched by their PTS. If no matching frame is found (e.g.,
    /// already released), this is a no-op.
    pub fn release_frame(&mut self, frame: &GpuFrame) {
        let target_pts = frame.pts;
        let before = self.active_frames.len();

        self.active_frames
            .retain(|af| (af.pts.as_secs() - target_pts.as_secs()).abs() > 1e-9);

        let released = before - self.active_frames.len();
        if released > 0 {
            debug!(
                pts = target_pts.as_secs(),
                remaining = self.active_frames.len(),
                "Released {released} active frame(s)"
            );
        }
    }

    /// Release all active frames, unmapping all NVDEC surfaces.
    ///
    /// Call this when you're done processing a batch of frames or before
    /// seeking.
    pub fn release_all_frames(&mut self) {
        let count = self.active_frames.len();
        self.active_frames.clear();
        if count > 0 {
            debug!(count, "Released all active frames");
        }
    }

    /// Reset the decoder for seeking.
    ///
    /// Releases all active frames and resets the parser state. After calling
    /// this, feed data starting from the nearest keyframe before the seek
    /// target.
    pub fn reset(&mut self) -> Result<(), DecodeError> {
        self.release_all_frames();
        self.session.reset()
    }

    /// Attempt to retrieve all currently decoded frames without feeding new data.
    ///
    /// Useful for draining the output queue after feeding multiple packets.
    /// Each returned frame keeps its NVDEC surface mapped until released.
    pub fn drain_decoded_frames(&mut self) -> Result<Vec<GpuFrame>, DecodeError> {
        let mut frames = Vec::new();
        while self.session.has_decoded_frames() {
            if let Some(frame) = self.map_next_gpu_frame()? {
                frames.push(frame);
            }
        }
        Ok(frames)
    }

    /// Internal: map the next decoded frame and convert to GpuFrame.
    ///
    /// The MappedFrame RAII guard is stored in `active_frames` to keep the
    /// surface alive. The caller must eventually call `release_frame()` to
    /// free the surface.
    fn map_next_gpu_frame(&mut self) -> Result<Option<GpuFrame>, DecodeError> {
        let mapped = match self.session.map_next_frame()? {
            Some(frame) => frame,
            None => return Ok(None),
        };

        // Update cached resolution from the decoded frame.
        let width = mapped.width;
        let height = mapped.height;
        self.output_resolution = Resolution::new(width, height);

        // Update format based on bit depth from video format.
        if let Some(fmt) = self.session.video_format() {
            if fmt.bit_depth_luma_minus8 > 0 {
                self.output_format = PixelFormat::P010;
            }
        }

        // Convert timestamp to TimeCode.
        // NVDEC timestamps are in the parser's clock units. We feed
        // microseconds from the PTS, so convert back to seconds.
        let pts = TimeCode::from_secs(mapped.timestamp as f64 / 1_000_000.0);

        let pitch = mapped.pitch;
        let device_ptr = mapped.device_ptr;
        let device_ptr_uv = Some(mapped.uv_device_ptr());

        self.frames_decoded += 1;

        debug!(
            frame_num = self.frames_decoded,
            width,
            height,
            pitch,
            pts = pts.as_secs(),
            active_frames = self.active_frames.len() + 1,
            "Decoded frame to GpuFrame"
        );

        // Warn if we're accumulating too many active frames.
        if self.active_frames.len() >= self.max_active_frames {
            warn!(
                active = self.active_frames.len(),
                max = self.max_active_frames,
                "Active frame count exceeds soft limit — caller should release frames"
            );
        }

        let gpu_frame = GpuFrame {
            device_ptr,
            device_ptr_uv,
            resolution: Resolution::new(width, height),
            format: self.output_format,
            pitch,
            pts,
        };

        // Store the mapped frame guard so the NVDEC surface stays mapped
        // and the device pointer remains valid.
        self.active_frames.push_back(ActiveFrame {
            _mapped: mapped,
            pts,
        });

        Ok(Some(gpu_frame))
    }
}

impl HwDecoder for NvDecoder {
    /// Decode a single video packet.
    ///
    /// Feeds the packet data to the NVDEC parser. If a decoded frame
    /// is ready, it is returned. Due to decode latency and B-frame
    /// reordering, there may not be an output frame for every input packet.
    ///
    /// The returned `GpuFrame`'s device pointer is valid until
    /// [`release_frame`](NvDecoder::release_frame) is called.
    fn decode(&mut self, packet: &VideoPacket) -> Result<Option<GpuFrame>, DecodeError> {
        // Convert PTS to microseconds for the parser timestamp.
        let timestamp = (packet.pts.as_secs() * 1_000_000.0) as i64;

        // Feed data to the parser (triggers sequence/decode/display callbacks).
        self.session.parse_data(&packet.data, timestamp)?;

        // Try to get a decoded frame.
        self.map_next_gpu_frame()
    }

    /// Flush remaining frames from the decode pipeline.
    ///
    /// Sends an end-of-stream signal and retrieves all buffered frames.
    /// Each returned frame keeps its NVDEC surface mapped until released.
    fn flush(&mut self) -> Result<Vec<GpuFrame>, DecodeError> {
        self.session.flush()?;

        let mut frames = Vec::new();
        while self.session.has_decoded_frames() {
            if let Some(frame) = self.map_next_gpu_frame()? {
                frames.push(frame);
            }
        }

        info!(
            flushed_frames = frames.len(),
            total_decoded = self.frames_decoded,
            active_frames = self.active_frames.len(),
            "NVDEC decoder flushed"
        );

        Ok(frames)
    }

    /// Get the output pixel format (NV12 for 8-bit, P010 for 10-bit).
    fn output_format(&self) -> PixelFormat {
        self.output_format
    }

    /// Get the output resolution (valid after first frame is decoded).
    fn output_resolution(&self) -> Resolution {
        self.output_resolution
    }

    /// Get the codec this decoder handles.
    fn codec(&self) -> VideoCodec {
        self.codec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_default_state() {
        // We can't create an actual decoder without the NVIDIA library,
        // but we can test the type relationships.
        let _codec = VideoCodec::H264;
        let _format = PixelFormat::Nv12;
        let _res = Resolution::new(1920, 1080);
    }

    #[test]
    fn timestamp_conversion() {
        // Verify our timestamp conversion: microseconds -> seconds
        let us: i64 = 2_000_000; // 2 seconds
        let tc = TimeCode::from_secs(us as f64 / 1_000_000.0);
        assert!((tc.as_secs() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn timestamp_roundtrip() {
        // Verify timestamp survives the encode/decode roundtrip:
        // seconds -> microseconds -> seconds
        let original_secs = 3.14159;
        let as_us = (original_secs * 1_000_000.0) as i64;
        let recovered = TimeCode::from_secs(as_us as f64 / 1_000_000.0);
        // Allow 1 microsecond of precision loss
        assert!((recovered.as_secs() - original_secs).abs() < 1e-6);
    }

    #[test]
    fn gpu_frame_uv_pointer() {
        // Verify the UV plane offset in a GpuFrame
        let frame = GpuFrame {
            device_ptr: 0x1000_0000,
            device_ptr_uv: Some(0x1000_0000 + 1080 * 2048),
            resolution: Resolution::new(1920, 1080),
            format: PixelFormat::Nv12,
            pitch: 2048,
            pts: TimeCode::ZERO,
        };
        assert_eq!(frame.device_ptr_uv, Some(frame.device_ptr + 1080 * 2048));
    }

    #[test]
    fn gpu_frame_byte_size() {
        let frame = GpuFrame {
            device_ptr: 0,
            device_ptr_uv: None,
            resolution: Resolution::HD,
            format: PixelFormat::Nv12,
            pitch: 1920,
            pts: TimeCode::ZERO,
        };
        // NV12: Y (1920*1080) + UV (1920*540) = 3110400
        assert_eq!(frame.byte_size(), 1920 * 1080 + 1920 * 540);
    }
}
