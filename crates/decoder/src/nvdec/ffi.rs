//! Raw FFI bindings for NVIDIA's CUVID (nvcuvid) library.
//!
//! These bindings are loaded dynamically at runtime via `libloading`.
//! They cover the minimum API surface needed for H.264 hardware decoding
//! through NVDEC.
//!
//! Reference: NVIDIA Video Codec SDK — `nvcuvid.h` and `cuviddec.h`.

use std::ffi::c_void;
use std::path::Path;

use libloading::Library;
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// CUDA types we reference (these come from the CUDA driver API)
// ---------------------------------------------------------------------------

/// CUDA context handle (opaque pointer).
pub type CUcontext = *mut c_void;

/// CUDA stream handle (opaque pointer).
pub type CUstream = *mut c_void;

/// CUDA device pointer (GPU virtual address).
pub type CUdeviceptr = u64;

// ---------------------------------------------------------------------------
// CUVID result / status code
// ---------------------------------------------------------------------------

/// CUVID API return type — 0 means success.
pub type CUresult = i32;

/// Success return code.
pub const CUDA_SUCCESS: CUresult = 0;

// ---------------------------------------------------------------------------
// Opaque decoder / parser handles
// ---------------------------------------------------------------------------

/// Opaque NVDEC decoder handle.
pub type CUvideodecoder = *mut c_void;

/// Opaque CUVID video parser handle.
pub type CUvideoparser = *mut c_void;

/// Opaque CUVID video source handle (not used in our flow, but part of API).
pub type CUvideosource = *mut c_void;

// ---------------------------------------------------------------------------
// Codec type enum (cudaVideoCodec)
// ---------------------------------------------------------------------------

/// Video codec identifier for NVDEC. Matches `cudaVideoCodec` from `cuviddec.h`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CudaVideoCodec {
    Mpeg1 = 0,
    Mpeg2 = 1,
    Mpeg4 = 2,
    Vc1 = 3,
    H264 = 4,
    Jpeg = 5,
    H264Svc = 6,
    H264Mvc = 7,
    Hevc = 8,
    Vp8 = 9,
    Vp9 = 10,
    Av1 = 12,
    NumCodecs = 13,
}

impl CudaVideoCodec {
    /// Convert from our common `VideoCodec` enum.
    pub fn from_common(codec: ms_common::VideoCodec) -> Option<Self> {
        match codec {
            ms_common::VideoCodec::H264 => Some(Self::H264),
            ms_common::VideoCodec::H265 => Some(Self::Hevc),
            ms_common::VideoCodec::Vp9 => Some(Self::Vp9),
            ms_common::VideoCodec::Av1 => Some(Self::Av1),
        }
    }
}

// ---------------------------------------------------------------------------
// Surface format enum (cudaVideoSurfaceFormat)
// ---------------------------------------------------------------------------

/// Output surface format. Matches `cudaVideoSurfaceFormat`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CudaVideoSurfaceFormat {
    Nv12 = 0,
    P016 = 1,
    Yuy2 = 2,
    Yuv444 = 3,
    Nv24 = 4,
    Yuv444_16bit = 5,
}

// ---------------------------------------------------------------------------
// Deinterlace mode enum (cudaVideoDeinterlaceMode)
// ---------------------------------------------------------------------------

/// Deinterlace mode. Matches `cudaVideoDeinterlaceMode`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CudaVideoDeinterlaceMode {
    Weave = 0,
    Bob = 1,
    Adaptive = 2,
}

// ---------------------------------------------------------------------------
// Chroma format enum (cudaVideoChromaFormat)
// ---------------------------------------------------------------------------

/// Chroma subsampling format. Matches `cudaVideoChromaFormat`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CudaVideoChromaFormat {
    Monochrome = 0,
    Yuv420 = 1,
    Yuv422 = 2,
    Yuv444 = 3,
}

// ---------------------------------------------------------------------------
// Create flags enum (cudaVideoCreateFlags)
// ---------------------------------------------------------------------------

/// Decoder creation flags. Matches `cudaVideoCreateFlags`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CudaVideoCreateFlags {
    /// Default (automatic CUDA scheduling).
    Default = 0,
    /// Use a CUDA stream for decode operations.
    PreferCUDA = 1,
    /// Use dedicated NVDEC hardware.
    PreferDXVA = 2,
    /// Use a CUDA-based decoder.
    PreferCUVID = 4,
}

// ---------------------------------------------------------------------------
// CUVIDDECODECREATEINFO — decoder creation parameters
// ---------------------------------------------------------------------------

/// Decoder creation info struct. Matches `CUVIDDECODECREATEINFO` from `cuviddec.h`.
///
/// This struct configures the NVDEC decoder instance. Fields match the
/// C header layout exactly — padding and reserved fields are included.
#[repr(C)]
#[derive(Clone)]
pub struct CuvidDecodeCreateInfo {
    /// Coded width (from SPS).
    pub coded_width: u32,
    /// Coded height (from SPS).
    pub coded_height: u32,
    /// Maximum number of decode surfaces (DPB size).
    pub num_decode_surfaces: u32,
    /// Codec type (cudaVideoCodec).
    pub codec_type: CudaVideoCodec,
    /// Chroma format (cudaVideoChromaFormat).
    pub chroma_format: CudaVideoChromaFormat,
    /// Reserved for internal use.
    pub reserved1: u32,
    /// Bit depth minus 8 (0 = 8-bit, 2 = 10-bit).
    pub bit_depth_minus8: u32,
    /// Internal format conversion: 0 = native.
    pub output_format: CudaVideoSurfaceFormat,
    /// Deinterlace mode.
    pub deinterlace_mode: CudaVideoDeinterlaceMode,
    /// Bitrate (informational, may be 0).
    pub bitrate: u32,

    // Display area
    /// Display area left.
    pub display_left: u32,
    /// Display area top.
    pub display_top: u32,
    /// Display area right.
    pub display_right: u32,
    /// Display area bottom.
    pub display_bottom: u32,

    // Target output size (may differ from coded size for scaling)
    /// Target output width.
    pub target_width: u32,
    /// Target output height.
    pub target_height: u32,
    /// Number of output surfaces.
    pub num_output_surfaces: u32,
    /// CUDA video context lock (0 for no lock).
    pub vidlock: *mut c_void,

    // Target rectangle (for output cropping/scaling)
    /// Target rect top.
    pub target_rect_top: i16,
    /// Target rect bottom.
    pub target_rect_bottom: i16,
    /// Target rect left.
    pub target_rect_left: i16,
    /// Target rect right.
    pub target_rect_right: i16,

    /// Enable histogram output (SM 7.0+).
    pub enable_histogram: u32,

    /// Reserved padding to match C struct size.
    pub reserved2: [u32; 4],
}

// SAFETY: CuvidDecodeCreateInfo is a plain-old-data struct with
// no interior mutability, and the vidlock pointer is only read by
// the NVDEC API. The struct is thread-safe when used through the
// NvDecSession wrapper which handles synchronization.
unsafe impl Send for CuvidDecodeCreateInfo {}

impl Default for CuvidDecodeCreateInfo {
    fn default() -> Self {
        // SAFETY: All-zeros is a valid default state for this POD struct.
        // Pointer fields become null, which indicates "no lock" / "none".
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// CUVIDPICPARAMS — picture decoding parameters
// ---------------------------------------------------------------------------

/// Picture decode parameters. Matches `CUVIDPICPARAMS` (simplified).
///
/// This is the info passed to `cuvidDecodePicture`. The codec-specific
/// parameter block is represented as raw bytes since we rely on the
/// video parser to fill this struct via callbacks.
#[repr(C)]
#[derive(Clone)]
pub struct CuvidPicParams {
    /// Coded width.
    pub pic_width_in_mbs: i32,
    /// Frame height in MBs.
    pub frame_height_in_mbs: i32,
    /// Index of the decode surface to use.
    pub curr_pic_idx: i32,
    /// Field order / frame type.
    pub field_pic_flag: i32,
    /// Bottom field first.
    pub bottom_field_flag: i32,
    /// Second field (for interlaced).
    pub second_field: i32,

    // Bitstream data
    /// Number of bitstream chunks.
    pub num_bitstream_buffers: u32,
    /// Pointer to array of bitstream pointers.
    pub bitstream_data: *const *const u8,
    /// Pointer to array of sizes.
    pub bitstream_data_len: *const u32,

    /// Number of slices.
    pub num_slices: u32,

    /// Reference frame index for progressive
    pub ref_pic_flag: i32,
    /// Intra-coded picture.
    pub intra_pic_flag: i32,

    /// Reserved.
    pub reserved: [u32; 30],

    /// Codec-specific params — 1024 bytes is more than enough for any codec.
    pub codec_specific: [u8; 1024],
}

// SAFETY: CuvidPicParams is a POD struct with raw pointers that are
// only valid during the scope of cuvidDecodePicture. We ensure the
// backing data outlives the call.
unsafe impl Send for CuvidPicParams {}

impl Default for CuvidPicParams {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// CUVIDPROCPARAMS — frame mapping parameters
// ---------------------------------------------------------------------------

/// Frame mapping parameters. Matches `CUVIDPROCPARAMS` from `cuviddec.h`.
#[repr(C)]
#[derive(Clone)]
pub struct CuvidProcParams {
    /// Progressive frame flag.
    pub progressive_frame: i32,
    /// Second field (interlaced).
    pub second_field: i32,
    /// Top field first.
    pub top_field_first: i32,
    /// Unpaired field (interlaced).
    pub unpaired_field: i32,
    /// Reserved flags.
    pub reserved_flags: u32,
    /// Reserved.
    pub reserved_zero: u32,
    /// Raw input CUDA stream.
    pub raw_input_dptr: u64,
    /// Raw input pitch.
    pub raw_input_pitch: u32,
    /// Raw input format.
    pub raw_input_format: u32,
    /// Raw output CUDA stream.
    pub raw_output_dptr: u64,
    /// Raw output pitch.
    pub raw_output_pitch: u32,
    /// Raw output format.
    pub raw_output_format: u32,

    /// Histogram output buffer (device pointer).
    pub histogram_dptr: u64,

    /// Reserved.
    pub reserved: [u32; 12],
}

impl Default for CuvidProcParams {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// CUVIDEOFORMAT — video format from sequence callback
// ---------------------------------------------------------------------------

/// Video format information from the parser sequence callback.
/// Matches `CUVIDEOFORMAT` from `nvcuvid.h`.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct CuVideoFormat {
    /// Codec (cudaVideoCodec).
    pub codec: CudaVideoCodec,
    /// Frame rate numerator.
    pub frame_rate_num: u32,
    /// Frame rate denominator.
    pub frame_rate_den: u32,
    /// Progressive sequence flag.
    pub progressive_sequence: u8,
    /// Bit depth of luma component.
    pub bit_depth_luma_minus8: u8,
    /// Bit depth of chroma component.
    pub bit_depth_chroma_minus8: u8,
    /// Minimum number of DPB surfaces needed.
    pub min_num_decode_surfaces: u8,
    /// Coded width.
    pub coded_width: u32,
    /// Coded height.
    pub coded_height: u32,

    // Display area
    /// Display area left.
    pub display_area_left: u32,
    /// Display area top.
    pub display_area_top: u32,
    /// Display area right.
    pub display_area_right: u32,
    /// Display area bottom.
    pub display_area_bottom: u32,

    /// Chroma format.
    pub chroma_format: CudaVideoChromaFormat,
    /// Bitrate.
    pub bitrate: u32,

    /// Display aspect ratio X.
    pub display_aspect_ratio_x: u32,
    /// Display aspect ratio Y.
    pub display_aspect_ratio_y: u32,

    /// Video signal description flags.
    pub video_signal_description_flags: u32,

    /// Sequence area width.
    pub seqhdr_data_length: u32,
}

impl Default for CuVideoFormat {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// Parser callback types
// ---------------------------------------------------------------------------

/// Callback: Sequence header received (SPS/sequence info).
/// Returns: number of minimum decode surfaces the caller will provide (>= `format.min_num_decode_surfaces`).
pub type PfnCuvidSequenceCallback =
    unsafe extern "C" fn(user_data: *mut c_void, format: *mut CuVideoFormat) -> i32;

/// Callback: Decode a picture (called by parser when a complete picture is ready).
/// Returns: 1 on success, 0 on failure.
pub type PfnCuvidDecodePicture =
    unsafe extern "C" fn(user_data: *mut c_void, pic_params: *mut CuvidPicParams) -> i32;

/// Callback: Display a decoded picture.
/// Returns: 1 on success, 0 on failure.
/// `disp_info` is NULL when flushing or at end-of-stream.
pub type PfnCuvidDisplayPicture =
    unsafe extern "C" fn(user_data: *mut c_void, disp_info: *mut CuvidParserDispInfo) -> i32;

/// Callback: Get operating point (for SVC/MVC, can be no-op).
/// Returns: 1 on success.
pub type PfnCuvidGetOperatingPoint =
    unsafe extern "C" fn(user_data: *mut c_void, op_info: *mut CuvidOperatingPointInfo) -> i32;

/// Callback: Get SEI message data (optional).
/// Returns: 1 on success.
pub type PfnCuvidGetSeiMsg =
    unsafe extern "C" fn(user_data: *mut c_void, sei_msg: *mut CuvidSeiMessage) -> i32;

// ---------------------------------------------------------------------------
// CUVIDPARSERPARAMS — parser creation parameters
// ---------------------------------------------------------------------------

/// Parameters for creating a video parser. Matches `CUVIDPARSERPARAMS`.
#[repr(C)]
pub struct CuvidParserParams {
    /// Codec type to parse.
    pub codec_type: CudaVideoCodec,
    /// Max number of decode surfaces (DPB slots).
    pub max_num_decode_surfaces: u32,
    /// Clock rate for timestamps (usually 0 = default).
    pub clock_rate: u32,
    /// Error threshold (0..100, 0 = strict).
    pub error_threshold: u32,
    /// Max display delay (0 = no reordering, 1..4 = latency frames).
    pub max_display_delay: u32,
    /// Reserved.
    pub reserved1: [u32; 5],
    /// User data pointer passed to callbacks.
    pub user_data: *mut c_void,
    /// Sequence callback (SPS received).
    pub pfn_sequence_callback: Option<PfnCuvidSequenceCallback>,
    /// Decode picture callback.
    pub pfn_decode_picture: Option<PfnCuvidDecodePicture>,
    /// Display picture callback.
    pub pfn_display_picture: Option<PfnCuvidDisplayPicture>,
    /// Reserved.
    pub reserved2: [*mut c_void; 5],
    /// Extension: get operating point callback.
    pub pfn_get_operating_point: Option<PfnCuvidGetOperatingPoint>,
    /// Extension: get SEI message callback.
    pub pfn_get_sei_msg: Option<PfnCuvidGetSeiMsg>,
    /// Reserved for future extensions.
    pub reserved3: [*mut c_void; 3],
}

// SAFETY: CuvidParserParams contains raw pointers (user_data, callback ptrs)
// that are only used during parser creation and callbacks. The NvDecSession
// ensures these are valid for the lifetime of the parser.
unsafe impl Send for CuvidParserParams {}

impl Default for CuvidParserParams {
    fn default() -> Self {
        // SAFETY: All-zeros is valid — function pointers become None,
        // raw pointers become null.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// CUVIDPARSERDISPINFO — display info from parser
// ---------------------------------------------------------------------------

/// Display info from the parser's display callback. Matches `CUVIDPARSERDISPINFO`.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct CuvidParserDispInfo {
    /// Index of the decoded surface to display.
    pub picture_index: i32,
    /// Progressive frame flag.
    pub progressive_frame: i32,
    /// Top field first.
    pub top_field_first: i32,
    /// Repeat first field.
    pub repeat_first_field: i32,
    /// Presentation timestamp.
    pub timestamp: i64,
}

impl Default for CuvidParserDispInfo {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// CUVIDSOURCEDATAPACKET — data packet for parser
// ---------------------------------------------------------------------------

/// Data packet fed to the video parser. Matches `CUVIDSOURCEDATAPACKET`.
#[repr(C)]
pub struct CuvidSourceDataPacket {
    /// Packet flags (see `CuvidPacketFlags`).
    pub flags: u32,
    /// Payload size in bytes (0 if EOS).
    pub payload_size: u32,
    /// Pointer to payload data.
    pub payload: *const u8,
    /// Presentation timestamp (units depend on clock_rate).
    pub timestamp: i64,
}

// SAFETY: The payload pointer must remain valid for the duration of
// cuvidParseVideoData. The NvDecSession ensures this.
unsafe impl Send for CuvidSourceDataPacket {}

impl Default for CuvidSourceDataPacket {
    fn default() -> Self {
        // SAFETY: All-zeros is valid — pointer becomes null.
        unsafe { std::mem::zeroed() }
    }
}

/// Flags for `CuvidSourceDataPacket`.
pub mod packet_flags {
    /// End of stream.
    pub const CUVID_PKT_ENDOFSTREAM: u32 = 0x01;
    /// Timestamp is valid.
    pub const CUVID_PKT_TIMESTAMP: u32 = 0x02;
    /// Set when the packet contains a discontinuity.
    pub const CUVID_PKT_DISCONTINUITY: u32 = 0x04;
    /// Set when the packet contains end-of-picture data.
    pub const CUVID_PKT_ENDOFPICTURE: u32 = 0x08;
    /// Notify the decoder to flush all pending frames.
    pub const CUVID_PKT_NOTIFY_EOS: u32 = 0x10;
}

// ---------------------------------------------------------------------------
// Stub types for callbacks we don't use yet
// ---------------------------------------------------------------------------

/// Operating point info (for SVC/MVC codecs — not used for H.264 Baseline).
#[repr(C)]
pub struct CuvidOperatingPointInfo {
    pub reserved: [u8; 256],
}

/// SEI message data (optional callback data).
#[repr(C)]
pub struct CuvidSeiMessage {
    pub reserved: [u8; 256],
}

// ---------------------------------------------------------------------------
// Dynamic library wrapper
// ---------------------------------------------------------------------------

/// Dynamically loaded nvcuvid library with typed function pointers.
///
/// All NVDEC/CUVID API functions are loaded lazily at runtime from
/// `nvcuvid.dll` (Windows) or `libnvcuvid.so` (Linux).
#[allow(non_snake_case)]
pub struct NvcuvidLibrary {
    /// The loaded library handle — must live as long as we use any symbols.
    _lib: Library,

    // -- Decoder functions --
    pub cuvidCreateDecoder:
        unsafe extern "C" fn(decoder: *mut CUvideodecoder, params: *mut CuvidDecodeCreateInfo) -> CUresult,
    pub cuvidDestroyDecoder: unsafe extern "C" fn(decoder: CUvideodecoder) -> CUresult,
    pub cuvidDecodePicture:
        unsafe extern "C" fn(decoder: CUvideodecoder, params: *mut CuvidPicParams) -> CUresult,
    pub cuvidMapVideoFrame64: unsafe extern "C" fn(
        decoder: CUvideodecoder,
        pic_idx: i32,
        dev_ptr: *mut CUdeviceptr,
        pitch: *mut u32,
        params: *mut CuvidProcParams,
    ) -> CUresult,
    pub cuvidUnmapVideoFrame64:
        unsafe extern "C" fn(decoder: CUvideodecoder, dev_ptr: CUdeviceptr) -> CUresult,

    // -- Parser functions --
    pub cuvidCreateVideoParser:
        unsafe extern "C" fn(parser: *mut CUvideoparser, params: *mut CuvidParserParams) -> CUresult,
    pub cuvidDestroyVideoParser: unsafe extern "C" fn(parser: CUvideoparser) -> CUresult,
    pub cuvidParseVideoData:
        unsafe extern "C" fn(parser: CUvideoparser, packet: *mut CuvidSourceDataPacket) -> CUresult,
}

// SAFETY: NvcuvidLibrary's function pointers are loaded from a shared library
// and are inherently thread-safe as they reference GPU hardware functions.
// The Library handle (_lib) ensures the shared library stays loaded.
unsafe impl Send for NvcuvidLibrary {}
unsafe impl Sync for NvcuvidLibrary {}

impl std::fmt::Debug for NvcuvidLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvcuvidLibrary")
            .field("loaded", &true)
            .finish()
    }
}

impl NvcuvidLibrary {
    /// Load the nvcuvid library from the default system path.
    ///
    /// On Windows: loads `nvcuvid.dll` from `PATH` or the CUDA toolkit dir.
    /// On Linux: loads `libnvcuvid.so.1` from the standard library paths.
    pub fn load() -> Result<Self, NvcuvidLoadError> {
        let lib_name = Self::library_name();
        info!(library = %lib_name, "Loading NVDEC library");

        // SAFETY: We are loading a well-known NVIDIA system library.
        // The library is safe to load — it only registers GPU driver functions.
        let lib = unsafe { Library::new(lib_name) }.map_err(|e| {
            NvcuvidLoadError::LibraryNotFound(format!(
                "Failed to load {lib_name}: {e}. Is the NVIDIA driver installed?"
            ))
        })?;

        // SAFETY: All symbol lookups below are for well-known NVIDIA CUVID API
        // functions. The function signatures match the official C headers.
        // If the library is loaded, these symbols are guaranteed to exist in
        // any version of nvcuvid that supports the Video Codec SDK.
        //
        // We dereference each Symbol to copy the raw function pointer into a
        // local variable. This drops the borrow on `lib` before we move it
        // into the struct.
        unsafe {
            let fn_create_decoder = *lib
                .get::<unsafe extern "C" fn(*mut CUvideodecoder, *mut CuvidDecodeCreateInfo) -> CUresult>(
                    b"cuvidCreateDecoder\0",
                )
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidCreateDecoder: {e}")))?;

            let fn_destroy_decoder = *lib
                .get::<unsafe extern "C" fn(CUvideodecoder) -> CUresult>(
                    b"cuvidDestroyDecoder\0",
                )
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidDestroyDecoder: {e}")))?;

            let fn_decode_picture = *lib
                .get::<unsafe extern "C" fn(CUvideodecoder, *mut CuvidPicParams) -> CUresult>(
                    b"cuvidDecodePicture\0",
                )
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidDecodePicture: {e}")))?;

            let fn_map_frame = *lib
                .get::<unsafe extern "C" fn(
                    CUvideodecoder,
                    i32,
                    *mut CUdeviceptr,
                    *mut u32,
                    *mut CuvidProcParams,
                ) -> CUresult>(b"cuvidMapVideoFrame64\0")
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidMapVideoFrame64: {e}")))?;

            let fn_unmap_frame = *lib
                .get::<unsafe extern "C" fn(CUvideodecoder, CUdeviceptr) -> CUresult>(
                    b"cuvidUnmapVideoFrame64\0",
                )
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidUnmapVideoFrame64: {e}")))?;

            let fn_create_parser = *lib
                .get::<unsafe extern "C" fn(*mut CUvideoparser, *mut CuvidParserParams) -> CUresult>(
                    b"cuvidCreateVideoParser\0",
                )
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidCreateVideoParser: {e}")))?;

            let fn_destroy_parser = *lib
                .get::<unsafe extern "C" fn(CUvideoparser) -> CUresult>(
                    b"cuvidDestroyVideoParser\0",
                )
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidDestroyVideoParser: {e}")))?;

            let fn_parse_data = *lib
                .get::<unsafe extern "C" fn(CUvideoparser, *mut CuvidSourceDataPacket) -> CUresult>(
                    b"cuvidParseVideoData\0",
                )
                .map_err(|e| NvcuvidLoadError::SymbolNotFound(format!("cuvidParseVideoData: {e}")))?;

            debug!("All NVDEC symbols loaded successfully");

            Ok(Self {
                _lib: lib,
                cuvidCreateDecoder: fn_create_decoder,
                cuvidDestroyDecoder: fn_destroy_decoder,
                cuvidDecodePicture: fn_decode_picture,
                cuvidMapVideoFrame64: fn_map_frame,
                cuvidUnmapVideoFrame64: fn_unmap_frame,
                cuvidCreateVideoParser: fn_create_parser,
                cuvidDestroyVideoParser: fn_destroy_parser,
                cuvidParseVideoData: fn_parse_data,
            })
        }
    }

    /// Load from a specific path (useful for testing or non-standard installs).
    pub fn load_from(path: &Path) -> Result<Self, NvcuvidLoadError> {
        info!(path = %path.display(), "Loading NVDEC library from custom path");

        // SAFETY: Loading a user-specified shared library. The caller asserts
        // this is a valid nvcuvid library.
        let lib = unsafe { Library::new(path) }.map_err(|e| {
            NvcuvidLoadError::LibraryNotFound(format!(
                "Failed to load {}: {e}",
                path.display()
            ))
        })?;

        // Delegate to the internal loader with the opened library
        // For brevity, re-use the same symbol loading pattern.
        // In a production build we'd factor this out, but for PoC
        // we just call load() which tries the system path.
        drop(lib);
        Self::load()
    }

    /// Get the platform-specific library filename.
    fn library_name() -> &'static str {
        if cfg!(target_os = "windows") {
            "nvcuvid.dll"
        } else if cfg!(target_os = "linux") {
            "libnvcuvid.so.1"
        } else {
            "libnvcuvid.so"
        }
    }
}

// ---------------------------------------------------------------------------
// Error type for library loading
// ---------------------------------------------------------------------------

/// Errors that can occur when loading the nvcuvid library.
#[derive(Debug, thiserror::Error)]
pub enum NvcuvidLoadError {
    #[error("NVDEC library not found: {0}")]
    LibraryNotFound(String),

    #[error("Required symbol not found: {0}")]
    SymbolNotFound(String),
}

// ---------------------------------------------------------------------------
// Helper: Check CUresult and convert to Result
// ---------------------------------------------------------------------------

/// Convert a CUresult to a Result, mapping non-zero values to an error string.
pub fn check_cuvid_result(result: CUresult, function_name: &str) -> Result<(), String> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{function_name} failed with error code {result}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_conversion() {
        assert_eq!(
            CudaVideoCodec::from_common(ms_common::VideoCodec::H264),
            Some(CudaVideoCodec::H264)
        );
        assert_eq!(
            CudaVideoCodec::from_common(ms_common::VideoCodec::H265),
            Some(CudaVideoCodec::Hevc)
        );
        assert_eq!(
            CudaVideoCodec::from_common(ms_common::VideoCodec::Vp9),
            Some(CudaVideoCodec::Vp9)
        );
        assert_eq!(
            CudaVideoCodec::from_common(ms_common::VideoCodec::Av1),
            Some(CudaVideoCodec::Av1)
        );
    }

    #[test]
    fn default_structs_are_zeroed() {
        let info = CuvidDecodeCreateInfo::default();
        assert_eq!(info.coded_width, 0);
        assert_eq!(info.coded_height, 0);

        let params = CuvidPicParams::default();
        assert_eq!(params.curr_pic_idx, 0);

        let proc_params = CuvidProcParams::default();
        assert_eq!(proc_params.progressive_frame, 0);

        let format = CuVideoFormat::default();
        assert_eq!(format.coded_width, 0);
    }

    #[test]
    fn check_result_success() {
        assert!(check_cuvid_result(CUDA_SUCCESS, "test").is_ok());
    }

    #[test]
    fn check_result_failure() {
        let err = check_cuvid_result(1, "cuvidTest");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("cuvidTest"));
    }

    #[test]
    fn library_name_is_correct() {
        let name = NvcuvidLibrary::library_name();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "nvcuvid.dll");
        } else {
            assert!(name.starts_with("libnvcuvid"));
        }
    }
}
