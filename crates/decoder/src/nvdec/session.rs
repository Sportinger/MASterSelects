//! Safe NVDEC decoder session wrapper with RAII resource management.
//!
//! `NvDecSession` owns the NVDEC decoder handle and the CUVID video parser,
//! providing a safe interface for submitting compressed bitstream data and
//! retrieving decoded NV12 frames from GPU memory.
//!
//! The session uses NVIDIA's callback-based architecture:
//! 1. Compressed data is fed to the video parser via `parse_data()`.
//! 2. The parser invokes callbacks: sequence (creates decoder), decode, display.
//! 3. Decoded frames are mapped from GPU surfaces and returned to the caller.
//!
//! ## Surface Lifecycle
//!
//! NVDEC manages a fixed-size pool of decode surfaces (DPB — Decoded Picture
//! Buffer). When a frame is decoded, it occupies a DPB slot. The frame must be
//! "mapped" via `cuvidMapVideoFrame64` to obtain a device pointer, and then
//! "unmapped" via `cuvidUnmapVideoFrame64` when the caller is finished with it.
//!
//! This module uses [`MappedFrame`] as an RAII guard that automatically unmaps
//! the surface on drop. Callers must copy the frame data (e.g., via a CUDA
//! memcpy or a GPU kernel) before dropping the `MappedFrame`, since the device
//! pointer becomes invalid after unmapping.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{debug, error, info, warn};

use ms_common::{DecodeError, VideoCodec};

use super::ffi::{
    check_cuvid_result, packet_flags, CuVideoFormat, CudaVideoCodec,
    CudaVideoDeinterlaceMode, CudaVideoSurfaceFormat, CuvidDecodeCreateInfo,
    CuvidParserDispInfo, CuvidParserParams, CuvidPicParams, CuvidProcParams,
    CuvidSourceDataPacket, CUvideodecoder, CUvideoparser, NvcuvidLibrary,
    CUDA_SUCCESS,
};

// ---------------------------------------------------------------------------
// Decoded frame info
// ---------------------------------------------------------------------------

/// Information about a decoded frame ready for retrieval.
#[derive(Clone, Debug)]
pub struct DecodedFrameInfo {
    /// Index of the decoded surface in the DPB.
    pub picture_index: i32,
    /// Whether the frame is progressive.
    pub progressive_frame: bool,
    /// Top field first (for interlaced content).
    pub top_field_first: bool,
    /// Presentation timestamp from the parser.
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// Mapped frame (RAII surface guard)
// ---------------------------------------------------------------------------

/// A mapped decoded frame — NV12 data on GPU device memory.
///
/// The frame is automatically unmapped on drop. The caller must consume the
/// device pointer data (via a GPU copy or kernel) before dropping this guard.
///
/// # Important
///
/// The `device_ptr` is only valid while this `MappedFrame` is alive. Storing
/// the raw pointer without keeping the `MappedFrame` alive results in a
/// use-after-free on the GPU.
pub struct MappedFrame {
    /// Device pointer to the NV12 Y plane.
    pub device_ptr: u64,
    /// Row pitch in bytes.
    pub pitch: u32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp.
    pub timestamp: i64,
    /// Reference to the decoder for unmapping.
    decoder_handle: CUvideodecoder,
    /// Reference to the library for calling unmap.
    lib: Arc<NvcuvidLibrary>,
    /// Whether the frame has been consumed (unmap still happens, but
    /// this flag is available for diagnostics).
    consumed: bool,
}

impl MappedFrame {
    /// Mark this frame as consumed. This is informational — the surface
    /// is always unmapped on drop regardless.
    pub fn mark_consumed(&mut self) {
        self.consumed = true;
    }

    /// Whether the frame data has been consumed by the caller.
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    /// Get the device pointer to the UV plane (NV12 layout: UV starts
    /// immediately after the Y plane at `height * pitch` offset).
    pub fn uv_device_ptr(&self) -> u64 {
        self.device_ptr + self.height as u64 * self.pitch as u64
    }
}

impl std::fmt::Debug for MappedFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedFrame")
            .field("device_ptr", &format_args!("0x{:x}", self.device_ptr))
            .field("pitch", &self.pitch)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("timestamp", &self.timestamp)
            .field("consumed", &self.consumed)
            .finish()
    }
}

impl Drop for MappedFrame {
    fn drop(&mut self) {
        if self.device_ptr != 0 && !self.decoder_handle.is_null() {
            if !self.consumed {
                debug!(
                    dev_ptr = format_args!("0x{:x}", self.device_ptr),
                    "Unmapping unconsumed MappedFrame — data may have been lost"
                );
            }

            // SAFETY: We own this mapped frame and the decoder handle is valid.
            // cuvidUnmapVideoFrame64 releases the GPU mapping created by
            // cuvidMapVideoFrame64. This call is idempotent — unmapping an
            // already-unmapped frame is a no-op in the NVDEC driver.
            let result = unsafe {
                (self.lib.cuvidUnmapVideoFrame64)(self.decoder_handle, self.device_ptr)
            };
            if result != CUDA_SUCCESS {
                error!(
                    error_code = result,
                    dev_ptr = format_args!("0x{:x}", self.device_ptr),
                    "Failed to unmap video frame in Drop"
                );
            }
        }
    }
}

// SAFETY: MappedFrame contains raw pointers to GPU resources. These are
// valid to send between threads because the NVDEC API is thread-safe
// when accessed through a single CUDA context (which is managed by
// the CudaBackend). The decoder_handle is an opaque pointer that is
// only passed to cuvidUnmapVideoFrame64, and the lib Arc ensures the
// library stays loaded.
unsafe impl Send for MappedFrame {}

// ---------------------------------------------------------------------------
// Shared state for parser callbacks
// ---------------------------------------------------------------------------

/// State shared between the parser callbacks and the session.
/// Protected by a Mutex because callbacks can be invoked from any thread.
struct CallbackState {
    /// The NVDEC decoder handle (created in sequence callback).
    decoder: CUvideodecoder,
    /// Reference to the loaded library.
    lib: Arc<NvcuvidLibrary>,
    /// Queue of frames ready for display (populated by display callback).
    display_queue: VecDeque<DecodedFrameInfo>,
    /// Video format from the latest sequence callback.
    format: Option<CuVideoFormat>,
    /// Codec type (stored for diagnostics; used by sequence callback
    /// to validate that the stream matches the expected codec).
    #[allow(dead_code)]
    cuda_codec: CudaVideoCodec,
    /// Decoded width (display area, may differ from coded size).
    width: u32,
    /// Decoded height (display area).
    height: u32,
    /// Number of decode surfaces requested by the user.
    num_decode_surfaces: u32,
    /// Last error from a callback (checked after parse_data returns).
    last_error: Option<String>,
    /// Total frames that passed through the decode callback.
    frames_decoded: u64,
    /// Total frames that reached the display callback.
    frames_displayed: u64,
}

// SAFETY: CallbackState contains a CUvideodecoder (raw pointer) that is
// only accessed through Mutex-protected methods, ensuring exclusive access.
unsafe impl Send for CallbackState {}

// ---------------------------------------------------------------------------
// Parser callbacks (extern "C")
// ---------------------------------------------------------------------------

/// Sequence callback — called when the parser encounters a new sequence (SPS).
/// Creates the NVDEC decoder with parameters from the video format.
///
/// # Safety
///
/// Called by the CUVID parser from within `cuvidParseVideoData`. The
/// `user_data` pointer must be a valid `*const Mutex<CallbackState>` that
/// outlives the parser. The `format` pointer is valid for the duration
/// of this callback.
unsafe extern "C" fn sequence_callback(
    user_data: *mut c_void,
    format: *mut CuVideoFormat,
) -> i32 {
    // SAFETY: user_data is a pointer to our Box<Mutex<CallbackState>> which
    // lives as long as the NvDecSession. The parser guarantees this callback
    // is only invoked while the parser exists.
    let state = &*(user_data as *const Mutex<CallbackState>);
    let mut state = state.lock();

    // SAFETY: format is a valid pointer provided by the CUVID parser,
    // guaranteed to be valid for the duration of this callback.
    let fmt = &*format;

    info!(
        codec = ?fmt.codec,
        width = fmt.coded_width,
        height = fmt.coded_height,
        chroma = ?fmt.chroma_format,
        bit_depth = fmt.bit_depth_luma_minus8 + 8,
        min_surfaces = fmt.min_num_decode_surfaces,
        "NVDEC sequence callback: new sequence detected"
    );

    // Compute display dimensions from the display area rectangle.
    // Fall back to coded dimensions if the display area is empty.
    state.width = fmt.display_area_right.saturating_sub(fmt.display_area_left);
    state.height = fmt.display_area_bottom.saturating_sub(fmt.display_area_top);
    if state.width == 0 {
        state.width = fmt.coded_width;
    }
    if state.height == 0 {
        state.height = fmt.coded_height;
    }
    state.format = Some(fmt.clone());

    // Destroy existing decoder if we're handling a resolution change
    // (mid-stream SPS with different dimensions).
    if !state.decoder.is_null() {
        debug!("Destroying existing decoder for resolution change");
        // SAFETY: decoder handle is valid, obtained from cuvidCreateDecoder.
        let result = (state.lib.cuvidDestroyDecoder)(state.decoder);
        if result != CUDA_SUCCESS {
            warn!(error_code = result, "Failed to destroy old decoder");
        }
        state.decoder = ptr::null_mut();
    }

    // Determine number of decode surfaces: parser minimum + headroom for
    // the display pipeline. The DPB must be large enough for reference
    // frames plus frames waiting in the display queue.
    let num_decode_surfaces = std::cmp::max(
        fmt.min_num_decode_surfaces as u32 + 4,
        state.num_decode_surfaces,
    );

    // Select output format based on bit depth.
    // 8-bit content -> NV12, 10-bit content -> P016 (16-bit NV12 variant).
    let output_format = if fmt.bit_depth_luma_minus8 > 0 {
        CudaVideoSurfaceFormat::P016
    } else {
        CudaVideoSurfaceFormat::Nv12
    };

    let mut create_info = CuvidDecodeCreateInfo {
        coded_width: fmt.coded_width,
        coded_height: fmt.coded_height,
        num_decode_surfaces,
        codec_type: fmt.codec,
        chroma_format: fmt.chroma_format,
        bit_depth_minus8: fmt.bit_depth_luma_minus8 as u32,
        output_format,
        deinterlace_mode: CudaVideoDeinterlaceMode::Adaptive,
        bitrate: fmt.bitrate,
        display_left: fmt.display_area_left,
        display_top: fmt.display_area_top,
        display_right: fmt.display_area_right,
        display_bottom: fmt.display_area_bottom,
        target_width: state.width,
        target_height: state.height,
        num_output_surfaces: 2,
        ..CuvidDecodeCreateInfo::default()
    };

    // SAFETY: create_info is fully initialized above. cuvidCreateDecoder
    // writes the opaque decoder handle to state.decoder.
    let result = (state.lib.cuvidCreateDecoder)(&mut state.decoder, &mut create_info);
    if result != CUDA_SUCCESS {
        error!(error_code = result, "cuvidCreateDecoder failed");
        state.last_error = Some(format!("cuvidCreateDecoder failed: error {result}"));
        return 0;
    }

    info!(
        width = state.width,
        height = state.height,
        surfaces = num_decode_surfaces,
        output_format = ?output_format,
        "NVDEC decoder created successfully"
    );

    // Return the number of decode surfaces we allocated. The parser uses
    // this to manage the DPB slot assignment.
    num_decode_surfaces as i32
}

/// Decode picture callback — called when the parser has a complete picture to decode.
///
/// # Safety
///
/// Called by the CUVID parser. `user_data` must be a valid `*const Mutex<CallbackState>`.
/// `pic_params` is a valid pointer to picture parameters filled by the parser.
unsafe extern "C" fn decode_picture_callback(
    user_data: *mut c_void,
    pic_params: *mut CuvidPicParams,
) -> i32 {
    // SAFETY: user_data points to our CallbackState Mutex, valid for parser lifetime.
    let state = &*(user_data as *const Mutex<CallbackState>);
    let mut state = state.lock();

    if state.decoder.is_null() {
        error!("Decode callback called but no decoder exists (SPS not yet parsed?)");
        state.last_error =
            Some("Decode callback invoked before decoder was created".to_string());
        return 0;
    }

    // SAFETY: decoder is a valid handle created by cuvidCreateDecoder in
    // the sequence callback, and pic_params is a valid pointer from the parser.
    let result = (state.lib.cuvidDecodePicture)(state.decoder, pic_params);
    if result != CUDA_SUCCESS {
        error!(error_code = result, "cuvidDecodePicture failed");
        state.last_error = Some(format!("cuvidDecodePicture failed: error {result}"));
        return 0;
    }

    state.frames_decoded += 1;
    1
}

/// Display picture callback — called when a decoded picture is ready for display.
///
/// # Safety
///
/// Called by the CUVID parser. `user_data` must be a valid `*const Mutex<CallbackState>`.
/// `disp_info` is NULL when the parser signals end-of-stream or flush.
unsafe extern "C" fn display_picture_callback(
    user_data: *mut c_void,
    disp_info: *mut CuvidParserDispInfo,
) -> i32 {
    // SAFETY: user_data points to our CallbackState Mutex, valid for parser lifetime.
    let state = &*(user_data as *const Mutex<CallbackState>);
    let mut state = state.lock();

    if disp_info.is_null() {
        // Null display info signals end-of-stream or flush.
        debug!("Display callback: end-of-stream / flush signal");
        return 1;
    }

    // SAFETY: disp_info is a valid pointer from the parser when non-null.
    let info = &*disp_info;

    let frame_info = DecodedFrameInfo {
        picture_index: info.picture_index,
        progressive_frame: info.progressive_frame != 0,
        top_field_first: info.top_field_first != 0,
        timestamp: info.timestamp,
    };

    debug!(
        pic_idx = info.picture_index,
        pts = info.timestamp,
        progressive = frame_info.progressive_frame,
        "Display callback: frame ready"
    );

    state.display_queue.push_back(frame_info);
    state.frames_displayed += 1;
    1
}

// ---------------------------------------------------------------------------
// NvDecSession
// ---------------------------------------------------------------------------

/// A safe NVDEC decoder session with RAII cleanup.
///
/// Manages the lifecycle of an NVDEC decoder and CUVID video parser.
/// Feed compressed NAL unit data via `parse_data()`, then retrieve
/// decoded frames via `map_next_frame()`.
///
/// # Thread Safety
///
/// The session itself is `Send` but not `Sync`. It should be used from a
/// single decode thread. The internal callback state is protected by a
/// `Mutex` because the CUVID parser may invoke callbacks from an internal
/// thread.
///
/// # Example
///
/// ```ignore
/// let lib = Arc::new(NvcuvidLibrary::load()?);
/// let mut session = NvDecSession::new(lib, VideoCodec::H264, 20, 4)?;
///
/// // Feed NAL units from demuxer
/// session.parse_data(&nal_data, pts_microseconds)?;
///
/// // Retrieve decoded frames
/// while session.has_decoded_frames() {
///     if let Some(frame) = session.map_next_frame()? {
///         // frame.device_ptr => NV12 Y plane on GPU
///         // frame.uv_device_ptr() => UV plane on GPU
///         // Copy data before dropping frame!
///     }
/// }
/// ```
pub struct NvDecSession {
    /// CUVID video parser handle.
    parser: CUvideoparser,
    /// Shared state between callbacks and the session.
    /// Boxed so the pointer remains stable for C callbacks.
    callback_state: Box<Mutex<CallbackState>>,
    /// Reference to the nvcuvid library.
    lib: Arc<NvcuvidLibrary>,
    /// Video codec.
    codec: VideoCodec,
}

// SAFETY: NvDecSession contains raw pointers (parser handle) that are
// only used through the safe wrapper methods. The Mutex protects the
// shared callback state. The parser handle is only used from the thread
// that calls parse_data/flush.
unsafe impl Send for NvDecSession {}

impl std::fmt::Debug for NvDecSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.callback_state.lock();
        f.debug_struct("NvDecSession")
            .field("codec", &self.codec)
            .field("decoder_active", &!state.decoder.is_null())
            .field("width", &state.width)
            .field("height", &state.height)
            .field("queued_frames", &state.display_queue.len())
            .field("frames_decoded", &state.frames_decoded)
            .field("frames_displayed", &state.frames_displayed)
            .finish()
    }
}

impl NvDecSession {
    /// Create a new NVDEC decoder session.
    ///
    /// The actual NVDEC decoder is created lazily when the first SPS
    /// (sequence parameter set) is parsed — this happens automatically
    /// in the sequence callback.
    ///
    /// # Arguments
    /// * `lib` — Reference to the loaded nvcuvid library.
    /// * `codec` — The video codec to decode.
    /// * `num_decode_surfaces` — Minimum number of DPB surfaces (typically 8-20).
    /// * `max_display_delay` — Maximum frame reordering delay (0 = low latency).
    ///
    /// # Errors
    /// Returns `DecodeError::UnsupportedCodec` if the codec cannot be mapped to
    /// a CUVID codec type, or `DecodeError::HwDecoderInit` if parser creation fails.
    pub fn new(
        lib: Arc<NvcuvidLibrary>,
        codec: VideoCodec,
        num_decode_surfaces: u32,
        max_display_delay: u32,
    ) -> Result<Self, DecodeError> {
        let cuda_codec = CudaVideoCodec::from_common(codec)
            .ok_or(DecodeError::UnsupportedCodec(codec))?;

        // Clamp surfaces to a sane minimum. The NVDEC driver needs at least
        // min_num_decode_surfaces (reported by the sequence callback), but we
        // use the user's hint as a lower bound.
        let clamped_surfaces = num_decode_surfaces.clamp(8, 32);

        let callback_state = Box::new(Mutex::new(CallbackState {
            decoder: ptr::null_mut(),
            lib: lib.clone(),
            display_queue: VecDeque::with_capacity(8),
            format: None,
            cuda_codec,
            width: 0,
            height: 0,
            num_decode_surfaces: clamped_surfaces,
            last_error: None,
            frames_decoded: 0,
            frames_displayed: 0,
        }));

        // Obtain a raw pointer to the callback state. This pointer is stored
        // in the parser params and passed to every callback invocation. It
        // remains valid because the Box keeps the Mutex at a stable address
        // for the lifetime of NvDecSession.
        let state_ptr: *const Mutex<CallbackState> = &*callback_state;

        let mut parser_params = CuvidParserParams {
            codec_type: cuda_codec,
            max_num_decode_surfaces: clamped_surfaces,
            clock_rate: 0, // use default (parser timestamp units = input units)
            error_threshold: 0,
            max_display_delay,
            user_data: state_ptr as *mut c_void,
            pfn_sequence_callback: Some(sequence_callback),
            pfn_decode_picture: Some(decode_picture_callback),
            pfn_display_picture: Some(display_picture_callback),
            ..CuvidParserParams::default()
        };

        let mut parser: CUvideoparser = ptr::null_mut();

        // SAFETY: parser_params is properly initialized above. The user_data
        // pointer points to our Box<Mutex<CallbackState>> which lives as long
        // as this NvDecSession (the Box is a field of the struct).
        // cuvidCreateVideoParser writes the parser handle to `parser`.
        let result = unsafe {
            (lib.cuvidCreateVideoParser)(&mut parser, &mut parser_params)
        };

        check_cuvid_result(result, "cuvidCreateVideoParser").map_err(|reason| {
            DecodeError::HwDecoderInit { codec, reason }
        })?;

        info!(
            codec = codec.display_name(),
            surfaces = clamped_surfaces,
            delay = max_display_delay,
            "NVDEC parser created"
        );

        Ok(Self {
            parser,
            callback_state,
            lib,
            codec,
        })
    }

    /// Feed compressed NAL unit data to the parser.
    ///
    /// The parser will invoke the sequence, decode, and display callbacks
    /// as appropriate. After calling this, check `has_decoded_frames()` and
    /// retrieve frames with `map_next_frame()`.
    ///
    /// # Arguments
    /// * `data` — Annex-B formatted NAL unit data (with 0x00000001 start codes).
    /// * `timestamp` — Presentation timestamp in arbitrary units (the same
    ///   units are returned in `MappedFrame::timestamp`).
    ///
    /// # Errors
    /// Returns `DecodeError::DecodeFailed` if the parser or a callback fails.
    pub fn parse_data(&mut self, data: &[u8], timestamp: i64) -> Result<(), DecodeError> {
        // Clear any stale error from a previous call.
        {
            let mut state = self.callback_state.lock();
            state.last_error = None;
        }

        let mut packet = CuvidSourceDataPacket {
            flags: packet_flags::CUVID_PKT_TIMESTAMP,
            payload_size: data.len() as u32,
            payload: data.as_ptr(),
            timestamp,
        };

        // SAFETY: parser is a valid handle from cuvidCreateVideoParser.
        // The packet.payload pointer is valid for the duration of this call
        // because `data` is borrowed and cuvidParseVideoData processes it
        // synchronously before returning.
        let result = unsafe {
            (self.lib.cuvidParseVideoData)(self.parser, &mut packet)
        };

        check_cuvid_result(result, "cuvidParseVideoData").map_err(|reason| {
            DecodeError::DecodeFailed { frame: 0, reason }
        })?;

        // Check if any callback reported an error during parsing.
        let state = self.callback_state.lock();
        if let Some(ref err) = state.last_error {
            return Err(DecodeError::DecodeFailed {
                frame: state.frames_decoded,
                reason: err.clone(),
            });
        }

        Ok(())
    }

    /// Send an end-of-stream signal to the parser, flushing remaining frames.
    ///
    /// After calling this, drain the display queue with `map_next_frame()`.
    /// The parser remains usable — new data can be fed after flushing (e.g.,
    /// for seeking).
    pub fn flush(&mut self) -> Result<(), DecodeError> {
        let mut packet = CuvidSourceDataPacket {
            flags: packet_flags::CUVID_PKT_ENDOFSTREAM,
            payload_size: 0,
            payload: ptr::null(),
            timestamp: 0,
        };

        // SAFETY: parser is valid. A null payload with the EOS flag is
        // the documented way to flush the CUVID parser pipeline.
        let result = unsafe {
            (self.lib.cuvidParseVideoData)(self.parser, &mut packet)
        };

        check_cuvid_result(result, "cuvidParseVideoData (flush)").map_err(|reason| {
            DecodeError::DecodeFailed { frame: 0, reason }
        })?;

        Ok(())
    }

    /// Reset the session for seeking.
    ///
    /// Sends a discontinuity signal to the parser, clears the display queue,
    /// and resets internal state. The decoder itself is not destroyed — it
    /// will be reused for the new position.
    ///
    /// After calling this, feed data starting from the nearest keyframe
    /// before the seek target.
    pub fn reset(&mut self) -> Result<(), DecodeError> {
        // Drain and discard any pending frames.
        {
            let mut state = self.callback_state.lock();
            let discarded = state.display_queue.len();
            state.display_queue.clear();
            state.last_error = None;
            if discarded > 0 {
                debug!(discarded, "Discarded pending frames during reset");
            }
        }

        // Send a discontinuity packet to reset the parser's internal state
        // (reference frames, timestamp tracking, etc.).
        let mut packet = CuvidSourceDataPacket {
            flags: packet_flags::CUVID_PKT_DISCONTINUITY,
            payload_size: 0,
            payload: ptr::null(),
            timestamp: 0,
        };

        // SAFETY: parser is valid. A null payload with the DISCONTINUITY flag
        // tells the parser to reset its internal bitstream state.
        let result = unsafe {
            (self.lib.cuvidParseVideoData)(self.parser, &mut packet)
        };

        check_cuvid_result(result, "cuvidParseVideoData (discontinuity)").map_err(
            |reason| DecodeError::DecodeFailed { frame: 0, reason },
        )?;

        info!("NVDEC session reset for seek");
        Ok(())
    }

    /// Check if there are decoded frames ready for retrieval.
    pub fn has_decoded_frames(&self) -> bool {
        let state = self.callback_state.lock();
        !state.display_queue.is_empty()
    }

    /// Get the number of decoded frames waiting in the display queue.
    pub fn pending_frame_count(&self) -> usize {
        let state = self.callback_state.lock();
        state.display_queue.len()
    }

    /// Map the next decoded frame from GPU memory.
    ///
    /// Returns `None` if no frames are ready. The returned `MappedFrame`
    /// will automatically unmap the GPU surface when dropped.
    ///
    /// # Important
    ///
    /// The caller should copy the frame data to a persistent GPU buffer
    /// before dropping the `MappedFrame`. The device pointer becomes
    /// invalid after the frame is unmapped. NVDEC has a limited number
    /// of simultaneously mappable surfaces (typically 1-2 on most hardware).
    ///
    /// # Errors
    /// Returns `DecodeError::InvalidSession` if the decoder has not been
    /// created yet (no SPS parsed), or `DecodeError::DecodeFailed` if the
    /// mapping operation fails.
    pub fn map_next_frame(&self) -> Result<Option<MappedFrame>, DecodeError> {
        // Pop the next frame from the display queue.
        let frame_info = {
            let mut state = self.callback_state.lock();
            match state.display_queue.pop_front() {
                Some(info) => info,
                None => return Ok(None),
            }
        };

        // Lock again to read decoder state. We release the lock between
        // the pop and the map to avoid holding it during the GPU call.
        let (decoder_handle, width, height) = {
            let state = self.callback_state.lock();
            if state.decoder.is_null() {
                return Err(DecodeError::InvalidSession);
            }
            (state.decoder, state.width, state.height)
        };

        let mut dev_ptr: u64 = 0;
        let mut pitch: u32 = 0;

        let mut proc_params = CuvidProcParams {
            progressive_frame: i32::from(frame_info.progressive_frame),
            top_field_first: i32::from(frame_info.top_field_first),
            ..CuvidProcParams::default()
        };

        // SAFETY: decoder_handle is a valid handle from cuvidCreateDecoder.
        // frame_info.picture_index is a valid DPB slot index from the
        // display callback. dev_ptr and pitch are out-params written by
        // cuvidMapVideoFrame64. proc_params controls deinterlacing behavior.
        let result = unsafe {
            (self.lib.cuvidMapVideoFrame64)(
                decoder_handle,
                frame_info.picture_index,
                &mut dev_ptr,
                &mut pitch,
                &mut proc_params,
            )
        };

        check_cuvid_result(result, "cuvidMapVideoFrame64").map_err(|reason| {
            DecodeError::DecodeFailed {
                frame: frame_info.timestamp as u64,
                reason,
            }
        })?;

        debug!(
            dev_ptr = format_args!("0x{:x}", dev_ptr),
            pitch,
            width,
            height,
            pic_idx = frame_info.picture_index,
            pts = frame_info.timestamp,
            "Mapped decoded frame"
        );

        Ok(Some(MappedFrame {
            device_ptr: dev_ptr,
            pitch,
            width,
            height,
            timestamp: frame_info.timestamp,
            decoder_handle,
            lib: self.lib.clone(),
            consumed: false,
        }))
    }

    /// Map all pending decoded frames at once.
    ///
    /// Convenience method that drains the display queue and maps every frame.
    /// Returns an empty Vec if no frames are ready.
    ///
    /// # Warning
    ///
    /// NVDEC typically only supports 1-2 simultaneously mapped frames. If
    /// there are more pending frames than the hardware supports, this may
    /// fail. Prefer calling `map_next_frame()` one at a time and consuming
    /// each frame before mapping the next.
    pub fn map_all_frames(&self) -> Result<Vec<MappedFrame>, DecodeError> {
        let count = self.pending_frame_count();
        let mut frames = Vec::with_capacity(count);
        for _ in 0..count {
            match self.map_next_frame()? {
                Some(frame) => frames.push(frame),
                None => break,
            }
        }
        Ok(frames)
    }

    /// Get the current decoded video resolution (0x0 if no sequence received yet).
    pub fn resolution(&self) -> (u32, u32) {
        let state = self.callback_state.lock();
        (state.width, state.height)
    }

    /// Get the video format info from the most recent sequence callback.
    pub fn video_format(&self) -> Option<CuVideoFormat> {
        let state = self.callback_state.lock();
        state.format.clone()
    }

    /// Check if the decoder has been created (i.e., SPS has been parsed).
    pub fn is_decoder_ready(&self) -> bool {
        let state = self.callback_state.lock();
        !state.decoder.is_null()
    }

    /// Get the codec this session handles.
    pub fn codec(&self) -> VideoCodec {
        self.codec
    }

    /// Get decode statistics.
    pub fn stats(&self) -> SessionStats {
        let state = self.callback_state.lock();
        SessionStats {
            frames_decoded: state.frames_decoded,
            frames_displayed: state.frames_displayed,
            pending_frames: state.display_queue.len() as u64,
            decoder_ready: !state.decoder.is_null(),
            width: state.width,
            height: state.height,
        }
    }
}

/// Statistics from an NVDEC decode session.
#[derive(Clone, Debug)]
pub struct SessionStats {
    /// Total frames passed through the decode callback.
    pub frames_decoded: u64,
    /// Total frames that reached the display callback.
    pub frames_displayed: u64,
    /// Frames currently waiting in the display queue.
    pub pending_frames: u64,
    /// Whether the hardware decoder has been initialized.
    pub decoder_ready: bool,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

impl Drop for NvDecSession {
    fn drop(&mut self) {
        // Destroy parser first — it may trigger final callbacks that reference
        // the decoder, so the decoder must still be alive.
        if !self.parser.is_null() {
            debug!("Destroying CUVID video parser");
            // SAFETY: parser is a valid handle from cuvidCreateVideoParser.
            // After this call, no more callbacks will be invoked.
            let result = unsafe { (self.lib.cuvidDestroyVideoParser)(self.parser) };
            if result != CUDA_SUCCESS {
                error!(error_code = result, "Failed to destroy video parser");
            }
            self.parser = ptr::null_mut();
        }

        // Destroy decoder after the parser so that any final callbacks can
        // complete. Clear the display queue first since those frames reference
        // decoder surfaces.
        let mut state = self.callback_state.lock();
        state.display_queue.clear();

        if !state.decoder.is_null() {
            debug!("Destroying NVDEC decoder");
            // SAFETY: decoder is a valid handle from cuvidCreateDecoder.
            let result = unsafe { (self.lib.cuvidDestroyDecoder)(state.decoder) };
            if result != CUDA_SUCCESS {
                error!(error_code = result, "Failed to destroy NVDEC decoder");
            }
            state.decoder = ptr::null_mut();
        }

        info!(
            codec = self.codec.display_name(),
            frames_decoded = state.frames_decoded,
            frames_displayed = state.frames_displayed,
            "NVDEC session destroyed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_frame_info_fields() {
        let info = DecodedFrameInfo {
            picture_index: 3,
            progressive_frame: true,
            top_field_first: false,
            timestamp: 12345,
        };
        assert_eq!(info.picture_index, 3);
        assert!(info.progressive_frame);
        assert!(!info.top_field_first);
        assert_eq!(info.timestamp, 12345);
    }

    #[test]
    fn session_stats_default() {
        let stats = SessionStats {
            frames_decoded: 0,
            frames_displayed: 0,
            pending_frames: 0,
            decoder_ready: false,
            width: 0,
            height: 0,
        };
        assert!(!stats.decoder_ready);
        assert_eq!(stats.frames_decoded, 0);
    }

    #[test]
    fn mapped_frame_uv_offset() {
        // Verify UV plane offset calculation for NV12.
        // For a 1920x1080 frame with pitch=2048:
        //   UV plane starts at height * pitch = 1080 * 2048 = 2211840
        let frame_base = 0x1000_0000u64;
        let width = 1920u32;
        let height = 1080u32;
        let pitch = 2048u32;

        // We can't create a real MappedFrame without GPU hardware,
        // but we can verify the math.
        let expected_uv = frame_base + height as u64 * pitch as u64;
        assert_eq!(expected_uv, 0x1000_0000 + 1080 * 2048);

        // Verify dimensions are reasonable
        assert!(width <= pitch);
        assert_eq!(width * height, 1920 * 1080);
    }
}
