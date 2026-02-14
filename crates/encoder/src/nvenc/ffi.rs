//! Raw FFI bindings for NVIDIA's NVENC (nvEncodeAPI) library.
//!
//! These bindings are loaded dynamically at runtime via `libloading`.
//! They cover the minimum API surface needed for H.264 and H.265 hardware
//! encoding through NVENC.
//!
//! Reference: NVIDIA Video Codec SDK -- `nvEncodeAPI.h`.

use std::ffi::c_void;
use std::path::Path;

use libloading::Library;
use tracing::{debug, info};

use crate::error::NvencLoadError;

// ---------------------------------------------------------------------------
// CUDA types we reference (from the CUDA driver API)
// ---------------------------------------------------------------------------

/// CUDA context handle (opaque pointer).
pub type CUcontext = *mut c_void;

/// CUDA device pointer (GPU virtual address).
pub type CUdeviceptr = u64;

// ---------------------------------------------------------------------------
// NVENC status codes
// ---------------------------------------------------------------------------

/// NVENC API return type.
pub type NvencStatus = i32;

/// Success return code.
pub const NV_ENC_SUCCESS: NvencStatus = 0;

/// Error: invalid parameter.
pub const NV_ENC_ERR_INVALID_PARAM: NvencStatus = 8;

/// Error: out of memory.
pub const NV_ENC_ERR_OUT_OF_MEMORY: NvencStatus = 4;

/// Error: encoder not initialized.
pub const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: NvencStatus = 7;

/// Error: need more input (async mode).
pub const NV_ENC_ERR_NEED_MORE_INPUT: NvencStatus = 11;

/// Error: device not found.
pub const NV_ENC_ERR_NO_ENCODE_DEVICE: NvencStatus = 1;

/// Error: unsupported.
pub const NV_ENC_ERR_UNSUPPORTED_DEVICE: NvencStatus = 2;

// ---------------------------------------------------------------------------
// NVENC API version
// ---------------------------------------------------------------------------

/// NVENC API major version we target (Video Codec SDK 12.x).
pub const NVENCAPI_MAJOR_VERSION: u32 = 12;

/// NVENC API minor version.
pub const NVENCAPI_MINOR_VERSION: u32 = 2;

/// Packed API version for struct versioning.
pub const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24);

/// Macro-equivalent for struct versioning: `(struct_ver) | (NVENCAPI_VERSION << 16)`.
/// This is how NVENC identifies which version of a struct is being passed.
pub const fn nvenc_struct_version(struct_ver: u32) -> u32 {
    struct_ver | (NVENCAPI_VERSION << 16)
}

// ---------------------------------------------------------------------------
// GUIDs
// ---------------------------------------------------------------------------

/// GUID structure matching NVENC's `GUID` type (Windows-compatible layout).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NvGuid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

/// Codec GUID: H.264.
pub const NV_ENC_CODEC_H264_GUID: NvGuid = NvGuid {
    data1: 0x6BC8_2762,
    data2: 0x4E63,
    data3: 0x4CA4,
    data4: [0xAA, 0x85, 0x1A, 0x4D, 0x14, 0x15, 0x26, 0xD2],
};

/// Codec GUID: H.265/HEVC.
pub const NV_ENC_CODEC_HEVC_GUID: NvGuid = NvGuid {
    data1: 0x790C_DC88,
    data2: 0x4522,
    data3: 0x4D7B,
    data4: [0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03],
};

// -- Preset GUIDs (NVENC SDK 12.x: new preset API with tuning info) --

/// Preset GUID: P1 (fastest).
pub const NV_ENC_PRESET_P1_GUID: NvGuid = NvGuid {
    data1: 0xFC0E_8692,
    data2: 0x8FF1,
    data3: 0x4C3D,
    data4: [0xBA, 0xD8, 0xF5, 0x64, 0xC0, 0x1D, 0x2A, 0xB1],
};

/// Preset GUID: P2.
pub const NV_ENC_PRESET_P2_GUID: NvGuid = NvGuid {
    data1: 0xF581_CFB8,
    data2: 0x88D6,
    data3: 0x4381,
    data4: [0x93, 0xF0, 0xDF, 0x13, 0xF9, 0xC2, 0x7D, 0xAB],
};

/// Preset GUID: P3.
pub const NV_ENC_PRESET_P3_GUID: NvGuid = NvGuid {
    data1: 0x3685_0110,
    data2: 0x3A07,
    data3: 0x441F,
    data4: [0x94, 0xD5, 0x34, 0x70, 0x63, 0x1F, 0x91, 0xF6],
};

/// Preset GUID: P4 (medium).
pub const NV_ENC_PRESET_P4_GUID: NvGuid = NvGuid {
    data1: 0x90A7_B826,
    data2: 0xDF06,
    data3: 0x4862,
    data4: [0xB9, 0xD2, 0xCD, 0x6D, 0x73, 0xA0, 0x8A, 0x81],
};

/// Preset GUID: P5.
pub const NV_ENC_PRESET_P5_GUID: NvGuid = NvGuid {
    data1: 0x21C6_E6B4,
    data2: 0x297A,
    data3: 0x4CBA,
    data4: [0x99, 0x8F, 0xB6, 0xCB, 0xDE, 0x72, 0xAD, 0xE3],
};

/// Preset GUID: P6.
pub const NV_ENC_PRESET_P6_GUID: NvGuid = NvGuid {
    data1: 0x8E75_C279,
    data2: 0x6299,
    data3: 0x4AB6,
    data4: [0x83, 0x02, 0x0F, 0x72, 0x6F, 0x97, 0x2F, 0x90],
};

/// Preset GUID: P7 (slowest/best quality).
pub const NV_ENC_PRESET_P7_GUID: NvGuid = NvGuid {
    data1: 0x8484_8C12,
    data2: 0x6F71,
    data3: 0x4C13,
    data4: [0x93, 0x1B, 0x53, 0xE5, 0xD9, 0x03, 0xF6, 0x03],
};

// -- Profile GUIDs --

/// Profile GUID: H.264 Baseline.
pub const NV_ENC_H264_PROFILE_BASELINE_GUID: NvGuid = NvGuid {
    data1: 0x0727_BCAA,
    data2: 0x78C4,
    data3: 0x4C83,
    data4: [0x8C, 0x2F, 0xEF, 0x3D, 0xFF, 0x26, 0x7C, 0x6A],
};

/// Profile GUID: H.264 Main.
pub const NV_ENC_H264_PROFILE_MAIN_GUID: NvGuid = NvGuid {
    data1: 0x6085_1BF2,
    data2: 0x8F35,
    data3: 0x4F4D,
    data4: [0x86, 0x88, 0x70, 0x92, 0x6D, 0xD6, 0x3B, 0xE1],
};

/// Profile GUID: H.264 High.
pub const NV_ENC_H264_PROFILE_HIGH_GUID: NvGuid = NvGuid {
    data1: 0xE7CB_C309,
    data2: 0x4F7A,
    data3: 0x4B89,
    data4: [0xAF, 0x2A, 0xD5, 0x37, 0xC9, 0x2B, 0xE3, 0x10],
};

/// Profile GUID: HEVC Main.
pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: NvGuid = NvGuid {
    data1: 0xB514_C39A,
    data2: 0xB55B,
    data3: 0x40FA,
    data4: [0x87, 0x8F, 0xF1, 0x25, 0x3B, 0x4D, 0xFD, 0xEC],
};

/// Profile GUID: HEVC Main10 (10-bit).
pub const NV_ENC_HEVC_PROFILE_MAIN10_GUID: NvGuid = NvGuid {
    data1: 0xFA4D_2B6C,
    data2: 0x3A5B,
    data3: 0x411A,
    data4: [0x80, 0x18, 0x0A, 0x3F, 0x5E, 0x3C, 0x9B, 0xE5],
};

// -- Tuning info --

/// Tuning info for high quality.
pub const NV_ENC_TUNING_INFO_HIGH_QUALITY: u32 = 1;
/// Tuning info for low latency.
pub const NV_ENC_TUNING_INFO_LOW_LATENCY: u32 = 2;
/// Tuning info for ultra low latency.
pub const NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY: u32 = 3;
/// Tuning info for lossless.
pub const NV_ENC_TUNING_INFO_LOSSLESS: u32 = 4;

// ---------------------------------------------------------------------------
// Buffer format enum
// ---------------------------------------------------------------------------

/// Input buffer format. Matches `NV_ENC_BUFFER_FORMAT`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NvEncBufferFormat {
    /// Undefined format.
    Undefined = 0x0000_0000,
    /// Semi-planar YUV 4:2:0, 8-bit (NV12).
    Nv12 = 0x0000_0001,
    /// Planar YUV 4:2:0, 8-bit (YV12).
    Yv12 = 0x0000_0010,
    /// Planar YUV 4:2:0, 8-bit (IYUV).
    Iyuv = 0x0000_0100,
    /// Planar YUV 4:4:4, 8-bit.
    Yuv444 = 0x0000_1000,
    /// Planar YUV 4:2:0, 10-bit.
    Yuv420_10bit = 0x0001_0000,
    /// Planar YUV 4:4:4, 10-bit.
    Yuv444_10bit = 0x0010_0000,
    /// Interleaved ARGB, 8-bit.
    Argb = 0x0100_0000,
    /// Interleaved ARGB, 10-bit.
    Argb10 = 0x0200_0000,
    /// Interleaved AYUV, 8-bit (packed).
    Ayuv = 0x0400_0000,
    /// Interleaved ABGR, 8-bit.
    Abgr = 0x1000_0000,
    /// Interleaved ABGR, 10-bit.
    Abgr10 = 0x2000_0000,
}

// ---------------------------------------------------------------------------
// Picture type enum
// ---------------------------------------------------------------------------

/// Encode picture type. Matches `NV_ENC_PIC_TYPE`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NvEncPicType {
    /// P-frame.
    P = 0,
    /// B-frame.
    B = 1,
    /// I-frame.
    I = 2,
    /// IDR frame.
    Idr = 3,
    /// Bidirectional frame.
    Bi = 4,
    /// Skip frame.
    Skipped = 8,
    /// Intra-refresh.
    IntraRefresh = 9,
    /// Non-reference P-frame.
    NonrefP = 10,
    /// Unknown.
    Unknown = 0xFF,
}

// ---------------------------------------------------------------------------
// Input resource type enum
// ---------------------------------------------------------------------------

/// Resource type for registering external resources. Matches `NV_ENC_INPUT_RESOURCE_TYPE`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NvEncInputResourceType {
    /// DirectX 9 resource.
    Directx9 = 0,
    /// CUDA device pointer.
    CudaDeviceptr = 1,
    /// CUDA array.
    CudaArray = 2,
    /// DirectX 11 resource.
    Directx11 = 3,
    /// OpenGL texture.
    Opengl = 4,
}

// ---------------------------------------------------------------------------
// Rate control mode enum
// ---------------------------------------------------------------------------

/// Rate control mode. Matches `NV_ENC_PARAMS_RC_MODE`.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NvEncRcMode {
    /// Constant QP.
    ConstQp = 0x0,
    /// Variable bitrate.
    Vbr = 0x1,
    /// Constant bitrate.
    Cbr = 0x2,
    /// VBR with minimum QP.
    VbrMinQp = 0x4,
    /// Two-pass quality.
    TwoPassQuality = 0x8,
    /// Two-pass frame size cap.
    TwoPassFrameSizeCap = 0x10,
    /// Two-pass VBR.
    TwoPassVbr = 0x20,
    /// CBR with low delay high quality.
    CbrLowDelayHq = 0x40,
    /// CBR high quality.
    CbrHq = 0x80,
    /// VBR high quality.
    VbrHq = 0x100,
}

// ---------------------------------------------------------------------------
// Core NVENC structs
// ---------------------------------------------------------------------------

/// Open encode session parameters. Matches `NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS`.
#[repr(C)]
pub struct NvEncOpenEncodeSessionExParams {
    /// Struct version (use `nvenc_struct_version`).
    pub version: u32,
    /// Device type: 0 = CUDA.
    pub device_type: u32,
    /// Device handle (CUcontext for CUDA).
    pub device: *mut c_void,
    /// Reserved.
    pub reserved: *mut c_void,
    /// API version the client is compiled against.
    pub api_version: u32,
    /// Reserved.
    pub reserved1: [u32; 253],
    /// Reserved.
    pub reserved2: [*mut c_void; 64],
}

// SAFETY: NvEncOpenEncodeSessionExParams is a POD struct with raw pointers
// that are only read by the NVENC API during session creation. The device
// pointer (CUcontext) is valid for the thread calling the API.
unsafe impl Send for NvEncOpenEncodeSessionExParams {}

impl Default for NvEncOpenEncodeSessionExParams {
    fn default() -> Self {
        // SAFETY: All-zeros is a valid default state for this POD struct.
        // Pointer fields become null. We set version and api_version after.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s.api_version = NVENCAPI_VERSION;
        s
    }
}

/// Encode initialization parameters. Matches `NV_ENC_INITIALIZE_PARAMS`.
#[repr(C)]
pub struct NvEncInitializeParams {
    /// Struct version.
    pub version: u32,
    /// Codec GUID (H264 or HEVC).
    pub encode_guid: NvGuid,
    /// Preset GUID.
    pub preset_guid: NvGuid,
    /// Encode width.
    pub encode_width: u32,
    /// Encode height.
    pub encode_height: u32,
    /// Display aspect ratio X.
    pub dar_width: u32,
    /// Display aspect ratio Y.
    pub dar_height: u32,
    /// Frame rate numerator.
    pub frame_rate_num: u32,
    /// Frame rate denominator.
    pub frame_rate_den: u32,
    /// Enable asynchronous encode mode.
    pub enable_encode_async: u32,
    /// Enable Picture Type Decision.
    pub enable_ptd: u32,
    /// Report slice offsets.
    pub report_slice_offsets: u32,
    /// Enable subframe readback.
    pub enable_sub_frame_write: u32,
    /// Enable external ME hints (motion estimation).
    pub enable_external_me_hints: u32,
    /// Enable ME only mode.
    pub enable_me_only_mode: u32,
    /// Enable weighted prediction.
    pub enable_weighted_prediction: u32,
    /// Enable output in video memory.
    pub enable_output_in_video_mem: u32,
    /// Reserved.
    pub reserved1: [u32; 233],
    /// Pointer to codec-specific config (NV_ENC_CONFIG).
    pub encode_config: *mut NvEncConfig,
    /// Maximum width for dynamic resolution change.
    pub max_encode_width: u32,
    /// Maximum height for dynamic resolution change.
    pub max_encode_height: u32,
    /// ME hints per block count (for external ME).
    pub max_me_hint_count_per_block: [u32; 2],
    /// Tuning info.
    pub tuning_info: u32,
    /// Reserved.
    pub reserved2: [*mut c_void; 62],
}

// SAFETY: NvEncInitializeParams is a POD struct. The encode_config pointer
// must point to valid memory during the NvEncInitializeEncoder call.
unsafe impl Send for NvEncInitializeParams {}

impl Default for NvEncInitializeParams {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s.enable_ptd = 1; // Let NVENC decide picture types
        s
    }
}

/// Encoder configuration. Matches `NV_ENC_CONFIG`.
#[repr(C)]
pub struct NvEncConfig {
    /// Struct version.
    pub version: u32,
    /// Profile GUID.
    pub profile_guid: NvGuid,
    /// GOP length (number of frames between keyframes, 0 = auto).
    pub gop_length: u32,
    /// Frame interval for P-frames.
    pub frame_interval_p: i32,
    /// Number of B-frames.
    pub num_b_frames: u32,
    /// Rate control parameters.
    pub rc_params: NvEncRcParams,
    /// Codec-specific config (union in C, we use raw bytes).
    pub encode_codec_config: [u8; 2048],
    /// Reserved.
    pub reserved: [u32; 278],
    /// Reserved.
    pub reserved2: [*mut c_void; 64],
}

// SAFETY: NvEncConfig is a POD struct.
unsafe impl Send for NvEncConfig {}

impl Default for NvEncConfig {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s
    }
}

/// Rate control parameters. Matches `NV_ENC_RC_PARAMS`.
#[repr(C)]
#[derive(Clone)]
pub struct NvEncRcParams {
    /// Rate control mode.
    pub rate_control_mode: NvEncRcMode,
    /// Constant QP for I-frames (QP mode only).
    pub const_qp_i: u32,
    /// Constant QP for P-frames.
    pub const_qp_p: u32,
    /// Constant QP for B-frames.
    pub const_qp_b: u32,
    /// Average bitrate (bits/sec).
    pub average_bitrate: u32,
    /// Maximum bitrate (bits/sec, for VBR).
    pub max_bitrate: u32,
    /// VBV buffer size.
    pub vbv_buffer_size: u32,
    /// VBV initial delay.
    pub vbv_initial_delay: u32,
    /// Enable min QP.
    pub enable_min_qp: u32,
    /// Enable max QP.
    pub enable_max_qp: u32,
    /// Enable initial RC QP.
    pub enable_initial_rc_qp: u32,
    /// Enable look-ahead.
    pub enable_aq: u32,
    /// Reserved / additional params we don't use yet.
    pub reserved: [u32; 256],
}

impl Default for NvEncRcParams {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        unsafe { std::mem::zeroed() }
    }
}

// ---------------------------------------------------------------------------
// Per-frame encode params
// ---------------------------------------------------------------------------

/// Picture encode parameters. Matches `NV_ENC_PIC_PARAMS`.
#[repr(C)]
pub struct NvEncPicParams {
    /// Struct version.
    pub version: u32,
    /// Input width.
    pub input_width: u32,
    /// Input height.
    pub input_height: u32,
    /// Input pitch.
    pub input_pitch: u32,
    /// Encode params flags.
    pub encode_params_flags: u32,
    /// Frame index.
    pub frame_idx: u32,
    /// Input timestamp.
    pub input_time_stamp: u64,
    /// Input duration.
    pub input_duration: u64,
    /// Registered input buffer handle.
    pub input_buffer: *mut c_void,
    /// Output bitstream buffer handle.
    pub output_bitstream: *mut c_void,
    /// Completion event (async mode).
    pub completion_event: *mut c_void,
    /// Buffer format.
    pub buffer_fmt: NvEncBufferFormat,
    /// Picture struct (frame / field).
    pub pic_struct: u32,
    /// Picture type.
    pub pic_type: NvEncPicType,
    /// Codec-specific per-pic params.
    pub codec_pic_params: [u8; 256],
    /// ME hint counts per block.
    pub me_hint_count_per_block: [u32; 2],
    /// ME hints.
    pub me_external_hints: *mut c_void,
    /// Reserved.
    pub reserved1: [u32; 6],
    /// Reserved.
    pub reserved_internal: [*mut c_void; 2],
    /// Reserved.
    pub reserved2: [u32; 284],
    /// Reserved.
    pub reserved3: [*mut c_void; 60],
}

// SAFETY: NvEncPicParams is a POD struct with raw pointers that are only
// valid during NvEncEncodePicture. We ensure backing resources outlive the call.
unsafe impl Send for NvEncPicParams {}

impl Default for NvEncPicParams {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s
    }
}

// ---------------------------------------------------------------------------
// Bitstream lock params
// ---------------------------------------------------------------------------

/// Lock bitstream parameters. Matches `NV_ENC_LOCK_BITSTREAM`.
#[repr(C)]
pub struct NvEncLockBitstream {
    /// Struct version.
    pub version: u32,
    /// Lock flags.
    pub do_not_wait: u32,
    /// Lock flags.
    pub lkey: u32,
    /// Output bitstream buffer handle to lock.
    pub output_bitstream: *mut c_void,
    /// [out] Slice offsets array.
    pub slice_offsets: *mut u32,
    /// [out] Frame index.
    pub frame_idx: u32,
    /// [out] HW encode status.
    pub hw_encode_status: u32,
    /// [out] Number of slices.
    pub num_slices: u32,
    /// [out] Bitstream size in bytes.
    pub bitstream_size_in_bytes: u32,
    /// [out] Output timestamp.
    pub output_time_stamp: u64,
    /// [out] Output duration.
    pub output_duration: u64,
    /// [out] Pointer to bitstream data.
    pub bitstream_buffer_ptr: *mut c_void,
    /// [out] Picture type.
    pub pic_type: NvEncPicType,
    /// [out] Picture struct.
    pub pic_struct: u32,
    /// [out] Frame average QP.
    pub frame_avg_qp: u32,
    /// [out] Frame satd cost.
    pub frame_satd: u32,
    /// [out] Luma SSIM.
    pub ltr_frame_idx: u32,
    /// [out] LTR frame flag.
    pub ltr_frame_flag: u32,
    /// Reserved.
    pub reserved: [u32; 236],
    /// Reserved.
    pub reserved2: [*mut c_void; 64],
}

// SAFETY: NvEncLockBitstream is a POD struct.
unsafe impl Send for NvEncLockBitstream {}

impl Default for NvEncLockBitstream {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s
    }
}

// ---------------------------------------------------------------------------
// Input buffer creation
// ---------------------------------------------------------------------------

/// Create input buffer params. Matches `NV_ENC_CREATE_INPUT_BUFFER`.
#[repr(C)]
pub struct NvEncCreateInputBuffer {
    /// Struct version.
    pub version: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Memory heap (0 = default).
    pub memory_heap: u32,
    /// Buffer format.
    pub buffer_fmt: NvEncBufferFormat,
    /// Reserved.
    pub reserved: u32,
    /// [out] Input buffer handle.
    pub input_buffer: *mut c_void,
    /// Private data (NVENC internal).
    pub reserved1: [u32; 57],
    /// Reserved.
    pub reserved2: [*mut c_void; 63],
}

// SAFETY: NvEncCreateInputBuffer is a POD struct.
unsafe impl Send for NvEncCreateInputBuffer {}

impl Default for NvEncCreateInputBuffer {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s
    }
}

// ---------------------------------------------------------------------------
// Output bitstream buffer creation
// ---------------------------------------------------------------------------

/// Create bitstream buffer params. Matches `NV_ENC_CREATE_BITSTREAM_BUFFER`.
#[repr(C)]
pub struct NvEncCreateBitstreamBuffer {
    /// Struct version.
    pub version: u32,
    /// Reserved.
    pub reserved: u32,
    /// Memory heap.
    pub memory_heap: u32,
    /// Reserved.
    pub reserved1: u32,
    /// [out] Bitstream buffer handle.
    pub bitstream_buffer: *mut c_void,
    /// [out] Bitstream buffer pointer (host-visible).
    pub bitstream_buffer_ptr: *mut c_void,
    /// Reserved.
    pub reserved2: [u32; 58],
    /// Reserved.
    pub reserved3: [*mut c_void; 64],
}

// SAFETY: NvEncCreateBitstreamBuffer is a POD struct.
unsafe impl Send for NvEncCreateBitstreamBuffer {}

impl Default for NvEncCreateBitstreamBuffer {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s
    }
}

// ---------------------------------------------------------------------------
// Register resource (for external CUDA device pointers)
// ---------------------------------------------------------------------------

/// Register external resource. Matches `NV_ENC_REGISTER_RESOURCE`.
#[repr(C)]
pub struct NvEncRegisterResource {
    /// Struct version.
    pub version: u32,
    /// Resource type.
    pub resource_type: NvEncInputResourceType,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Pitch.
    pub pitch: u32,
    /// Sub-resource index (DX only).
    pub sub_resource_index: u32,
    /// Pointer to the external resource.
    pub resource_to_register: *mut c_void,
    /// [out] Registered resource handle.
    pub registered_resource: *mut c_void,
    /// Buffer format.
    pub buffer_format: NvEncBufferFormat,
    /// Buffer usage: 0 = input.
    pub buffer_usage: u32,
    /// Reserved.
    pub reserved1: [u32; 248],
    /// Reserved.
    pub reserved2: [*mut c_void; 62],
}

// SAFETY: NvEncRegisterResource is a POD struct.
unsafe impl Send for NvEncRegisterResource {}

impl Default for NvEncRegisterResource {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s
    }
}

// ---------------------------------------------------------------------------
// Map input resource
// ---------------------------------------------------------------------------

/// Map input resource. Matches `NV_ENC_MAP_INPUT_RESOURCE`.
#[repr(C)]
pub struct NvEncMapInputResource {
    /// Struct version.
    pub version: u32,
    /// Sub-resource index.
    pub sub_resource_index: u32,
    /// Input resource (reserved).
    pub input_resource: *mut c_void,
    /// Registered resource handle.
    pub registered_resource: *mut c_void,
    /// [out] Mapped resource handle (used as input_buffer in PicParams).
    pub mapped_resource: *mut c_void,
    /// [out] Mapped buffer format.
    pub mapped_buffer_fmt: NvEncBufferFormat,
    /// Reserved.
    pub reserved1: [u32; 251],
    /// Reserved.
    pub reserved2: [*mut c_void; 63],
}

// SAFETY: NvEncMapInputResource is a POD struct.
unsafe impl Send for NvEncMapInputResource {}

impl Default for NvEncMapInputResource {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s
    }
}

// ---------------------------------------------------------------------------
// Preset config (for querying default config for a preset)
// ---------------------------------------------------------------------------

/// Preset config query. Matches `NV_ENC_PRESET_CONFIG`.
#[repr(C)]
pub struct NvEncPresetConfig {
    /// Struct version.
    pub version: u32,
    /// [out] Preset config.
    pub preset_cfg: NvEncConfig,
    /// Reserved.
    pub reserved1: [u32; 255],
    /// Reserved.
    pub reserved2: [*mut c_void; 64],
}

// SAFETY: NvEncPresetConfig is a POD struct.
unsafe impl Send for NvEncPresetConfig {}

impl Default for NvEncPresetConfig {
    fn default() -> Self {
        // SAFETY: All-zeros is valid for this POD struct.
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.version = nvenc_struct_version(1);
        s.preset_cfg.version = nvenc_struct_version(1);
        s
    }
}

// ---------------------------------------------------------------------------
// Encode-end-of-stream flag
// ---------------------------------------------------------------------------

/// Flag indicating end-of-stream (flush) in NvEncPicParams::encode_params_flags.
pub const NV_ENC_PIC_FLAG_EOS: u32 = 0x01;

// ---------------------------------------------------------------------------
// Function pointer table
// ---------------------------------------------------------------------------

/// NVENC API function pointer table.
///
/// Contains all NVENC API functions loaded dynamically from the nvEncodeAPI
/// shared library. Function signatures match `NV_ENCODE_API_FUNCTION_LIST`
/// from `nvEncodeAPI.h`.
#[allow(non_snake_case)]
pub struct NvencFunctionList {
    /// Open an encode session.
    pub nvEncOpenEncodeSessionEx:
        unsafe extern "C" fn(params: *mut NvEncOpenEncodeSessionExParams, encoder: *mut *mut c_void) -> NvencStatus,

    /// Initialize the encoder.
    pub nvEncInitializeEncoder:
        unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncInitializeParams) -> NvencStatus,

    /// Encode a picture.
    pub nvEncEncodePicture:
        unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncPicParams) -> NvencStatus,

    /// Lock the output bitstream.
    pub nvEncLockBitstream:
        unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncLockBitstream) -> NvencStatus,

    /// Unlock the output bitstream.
    pub nvEncUnlockBitstream:
        unsafe extern "C" fn(encoder: *mut c_void, output_bitstream: *mut c_void) -> NvencStatus,

    /// Create an input buffer.
    pub nvEncCreateInputBuffer:
        unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncCreateInputBuffer) -> NvencStatus,

    /// Destroy an input buffer.
    pub nvEncDestroyInputBuffer:
        unsafe extern "C" fn(encoder: *mut c_void, input_buffer: *mut c_void) -> NvencStatus,

    /// Create a bitstream output buffer.
    pub nvEncCreateBitstreamBuffer:
        unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncCreateBitstreamBuffer) -> NvencStatus,

    /// Destroy a bitstream output buffer.
    pub nvEncDestroyBitstreamBuffer:
        unsafe extern "C" fn(encoder: *mut c_void, bitstream_buffer: *mut c_void) -> NvencStatus,

    /// Register an external resource (CUDA device ptr, DX texture, etc.).
    pub nvEncRegisterResource:
        unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncRegisterResource) -> NvencStatus,

    /// Unregister an external resource.
    pub nvEncUnregisterResource:
        unsafe extern "C" fn(encoder: *mut c_void, registered_resource: *mut c_void) -> NvencStatus,

    /// Map a registered resource for use as encoder input.
    pub nvEncMapInputResource:
        unsafe extern "C" fn(encoder: *mut c_void, params: *mut NvEncMapInputResource) -> NvencStatus,

    /// Unmap a mapped input resource.
    pub nvEncUnmapInputResource:
        unsafe extern "C" fn(encoder: *mut c_void, mapped_resource: *mut c_void) -> NvencStatus,

    /// Destroy the encoder session.
    pub nvEncDestroyEncoder:
        unsafe extern "C" fn(encoder: *mut c_void) -> NvencStatus,

    /// Get the preset configuration for a given codec/preset/tuning.
    pub nvEncGetEncodePresetConfigEx:
        unsafe extern "C" fn(
            encoder: *mut c_void,
            encode_guid: NvGuid,
            preset_guid: NvGuid,
            tuning_info: u32,
            preset_config: *mut NvEncPresetConfig,
        ) -> NvencStatus,
}

// SAFETY: NvencFunctionList's function pointers are loaded from the NVENC
// shared library and are inherently thread-safe as they reference GPU driver
// functions. The Library handle is stored separately.
unsafe impl Send for NvencFunctionList {}
unsafe impl Sync for NvencFunctionList {}

impl std::fmt::Debug for NvencFunctionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvencFunctionList")
            .field("loaded", &true)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Dynamic library wrapper
// ---------------------------------------------------------------------------

/// Dynamically loaded NVENC library with the API function table.
///
/// All NVENC API functions are loaded lazily at runtime from
/// `nvEncodeAPI64.dll` (Windows) or `libnvidia-encode.so.1` (Linux).
pub struct NvencLibrary {
    /// The loaded library handle -- must live as long as we use any symbols.
    _lib: Library,
    /// NVENC API function pointers.
    pub api: NvencFunctionList,
}

// SAFETY: NvencLibrary contains an opaque Library handle and function
// pointers. The Library ensures the shared library stays loaded, and
// the function pointers are thread-safe GPU driver functions.
unsafe impl Send for NvencLibrary {}
unsafe impl Sync for NvencLibrary {}

impl std::fmt::Debug for NvencLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvencLibrary")
            .field("loaded", &true)
            .finish()
    }
}

/// Type of the `NvEncodeAPICreateInstance` entry point.
///
/// This is the single entry point exported by the NVENC library. It fills
/// a function list struct with all API function pointers.
///
/// Our approach: since the function list in the C SDK is a struct with
/// a version field followed by function pointers, we load the entry point
/// and call it to populate our `NvencFunctionList`. However, because the
/// C struct layout is complex, we load individual functions by name instead
/// for reliability. This is equivalent to what the SDK does internally.
type NvEncodeApiCreateInstanceFn = unsafe extern "C" fn(function_list: *mut c_void) -> NvencStatus;

impl NvencLibrary {
    /// Load the NVENC library from the default system path.
    ///
    /// On Windows: loads `nvEncodeAPI64.dll` from `PATH` or the CUDA toolkit dir.
    /// On Linux: loads `libnvidia-encode.so.1` from the standard library paths.
    pub fn load() -> Result<Self, NvencLoadError> {
        let lib_name = Self::library_name();
        info!(library = %lib_name, "Loading NVENC library");

        // SAFETY: We are loading a well-known NVIDIA system library.
        // The library is safe to load -- it only registers GPU driver functions.
        let lib = unsafe { Library::new(lib_name) }.map_err(|e| {
            NvencLoadError::LibraryNotFound(format!(
                "Failed to load {lib_name}: {e}. Is the NVIDIA driver installed?"
            ))
        })?;

        Self::load_functions(lib)
    }

    /// Load from a specific path (useful for testing or non-standard installs).
    pub fn load_from(path: &Path) -> Result<Self, NvencLoadError> {
        info!(path = %path.display(), "Loading NVENC library from custom path");

        // SAFETY: Loading a user-specified shared library. The caller asserts
        // this is a valid nvEncodeAPI library.
        let lib = unsafe { Library::new(path) }.map_err(|e| {
            NvencLoadError::LibraryNotFound(format!("Failed to load {}: {e}", path.display()))
        })?;

        Self::load_functions(lib)
    }

    /// Load all NVENC function pointers from the opened library.
    fn load_functions(lib: Library) -> Result<Self, NvencLoadError> {
        // We verify that the entry point exists, but load individual functions
        // by name for maximum compatibility across SDK versions.
        // SAFETY: Checking for the well-known NVENC entry point.
        let _entry: libloading::Symbol<'_, NvEncodeApiCreateInstanceFn> =
            unsafe { lib.get(b"NvEncodeAPICreateInstance\0") }.map_err(|e| {
                NvencLoadError::SymbolNotFound(format!("NvEncodeAPICreateInstance: {e}"))
            })?;

        // SAFETY: All symbol lookups below are for well-known NVIDIA NVENC API
        // functions. The function signatures match the official C headers.
        // We dereference each Symbol to copy the raw function pointer.
        unsafe {
            let fn_open_session = *lib
                .get::<unsafe extern "C" fn(*mut NvEncOpenEncodeSessionExParams, *mut *mut c_void) -> NvencStatus>(
                    b"NvEncOpenEncodeSessionEx\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncOpenEncodeSessionEx: {e}")))?;

            let fn_init = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut NvEncInitializeParams) -> NvencStatus>(
                    b"NvEncInitializeEncoder\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncInitializeEncoder: {e}")))?;

            let fn_encode = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut NvEncPicParams) -> NvencStatus>(
                    b"NvEncEncodePicture\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncEncodePicture: {e}")))?;

            let fn_lock = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut NvEncLockBitstream) -> NvencStatus>(
                    b"NvEncLockBitstream\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncLockBitstream: {e}")))?;

            let fn_unlock = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>(
                    b"NvEncUnlockBitstream\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncUnlockBitstream: {e}")))?;

            let fn_create_input = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut NvEncCreateInputBuffer) -> NvencStatus>(
                    b"NvEncCreateInputBuffer\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncCreateInputBuffer: {e}")))?;

            let fn_destroy_input = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>(
                    b"NvEncDestroyInputBuffer\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncDestroyInputBuffer: {e}")))?;

            let fn_create_bitstream = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut NvEncCreateBitstreamBuffer) -> NvencStatus>(
                    b"NvEncCreateBitstreamBuffer\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncCreateBitstreamBuffer: {e}")))?;

            let fn_destroy_bitstream = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>(
                    b"NvEncDestroyBitstreamBuffer\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncDestroyBitstreamBuffer: {e}")))?;

            let fn_register = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut NvEncRegisterResource) -> NvencStatus>(
                    b"NvEncRegisterResource\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncRegisterResource: {e}")))?;

            let fn_unregister = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>(
                    b"NvEncUnregisterResource\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncUnregisterResource: {e}")))?;

            let fn_map = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut NvEncMapInputResource) -> NvencStatus>(
                    b"NvEncMapInputResource\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncMapInputResource: {e}")))?;

            let fn_unmap = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvencStatus>(
                    b"NvEncUnmapInputResource\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncUnmapInputResource: {e}")))?;

            let fn_destroy = *lib
                .get::<unsafe extern "C" fn(*mut c_void) -> NvencStatus>(
                    b"NvEncDestroyEncoder\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncDestroyEncoder: {e}")))?;

            let fn_get_preset = *lib
                .get::<unsafe extern "C" fn(*mut c_void, NvGuid, NvGuid, u32, *mut NvEncPresetConfig) -> NvencStatus>(
                    b"NvEncGetEncodePresetConfigEx\0",
                )
                .map_err(|e| NvencLoadError::SymbolNotFound(format!("NvEncGetEncodePresetConfigEx: {e}")))?;

            debug!("All NVENC symbols loaded successfully");

            Ok(Self {
                _lib: lib,
                api: NvencFunctionList {
                    nvEncOpenEncodeSessionEx: fn_open_session,
                    nvEncInitializeEncoder: fn_init,
                    nvEncEncodePicture: fn_encode,
                    nvEncLockBitstream: fn_lock,
                    nvEncUnlockBitstream: fn_unlock,
                    nvEncCreateInputBuffer: fn_create_input,
                    nvEncDestroyInputBuffer: fn_destroy_input,
                    nvEncCreateBitstreamBuffer: fn_create_bitstream,
                    nvEncDestroyBitstreamBuffer: fn_destroy_bitstream,
                    nvEncRegisterResource: fn_register,
                    nvEncUnregisterResource: fn_unregister,
                    nvEncMapInputResource: fn_map,
                    nvEncUnmapInputResource: fn_unmap,
                    nvEncDestroyEncoder: fn_destroy,
                    nvEncGetEncodePresetConfigEx: fn_get_preset,
                },
            })
        }
    }

    /// Get the platform-specific library filename.
    fn library_name() -> &'static str {
        if cfg!(target_os = "windows") {
            "nvEncodeAPI64.dll"
        } else if cfg!(target_os = "linux") {
            "libnvidia-encode.so.1"
        } else {
            "libnvidia-encode.so"
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: Check NvencStatus and convert to Result
// ---------------------------------------------------------------------------

/// Convert an `NvencStatus` to a Result, mapping non-zero values to an error string.
pub fn check_nvenc_status(status: NvencStatus, function_name: &str) -> Result<(), String> {
    if status == NV_ENC_SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "{function_name} failed with NVENC status code {status} ({})",
            nvenc_status_name(status)
        ))
    }
}

/// Get a human-readable name for an NVENC status code.
pub fn nvenc_status_name(status: NvencStatus) -> &'static str {
    match status {
        0 => "NV_ENC_SUCCESS",
        1 => "NV_ENC_ERR_NO_ENCODE_DEVICE",
        2 => "NV_ENC_ERR_UNSUPPORTED_DEVICE",
        3 => "NV_ENC_ERR_INVALID_ENCODERDEVICE",
        4 => "NV_ENC_ERR_OUT_OF_MEMORY",
        5 => "NV_ENC_ERR_INVALID_DEVICE",
        6 => "NV_ENC_ERR_INVALID_DEVICE",
        7 => "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
        8 => "NV_ENC_ERR_INVALID_PARAM",
        9 => "NV_ENC_ERR_OPERATION_NOT_ALLOWED",
        10 => "NV_ENC_ERR_ENCODER_BUSY",
        11 => "NV_ENC_ERR_NEED_MORE_INPUT",
        12 => "NV_ENC_ERR_ENCODER_RECONFIG_FAILED",
        15 => "NV_ENC_ERR_RESOURCE_NOT_REGISTERED",
        16 => "NV_ENC_ERR_RESOURCE_NOT_MAPPED",
        20 => "NV_ENC_ERR_GENERIC",
        _ => "NV_ENC_ERR_UNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_versions_are_nonzero() {
        let session_params = NvEncOpenEncodeSessionExParams::default();
        assert_ne!(session_params.version, 0);

        let init_params = NvEncInitializeParams::default();
        assert_ne!(init_params.version, 0);
        assert_eq!(init_params.enable_ptd, 1);

        let config = NvEncConfig::default();
        assert_ne!(config.version, 0);

        let pic_params = NvEncPicParams::default();
        assert_ne!(pic_params.version, 0);

        let lock = NvEncLockBitstream::default();
        assert_ne!(lock.version, 0);
    }

    #[test]
    fn default_structs_are_zeroed_except_version() {
        let params = NvEncInitializeParams::default();
        assert_eq!(params.encode_width, 0);
        assert_eq!(params.encode_height, 0);
        assert_eq!(params.frame_rate_num, 0);
        assert_eq!(params.frame_rate_den, 0);

        let config = NvEncConfig::default();
        assert_eq!(config.gop_length, 0);

        let buf = NvEncCreateInputBuffer::default();
        assert_eq!(buf.width, 0);
        assert_eq!(buf.height, 0);
    }

    #[test]
    fn nvenc_struct_version_packing() {
        let ver = nvenc_struct_version(1);
        // Version should be non-zero and contain both the struct version and API version
        assert_ne!(ver, 0);
        assert_ne!(ver, 1);
        // The lower bits should contain the struct version
        assert_eq!(ver & 0xFFFF, 1);
    }

    #[test]
    fn codec_guids_are_different() {
        assert_ne!(NV_ENC_CODEC_H264_GUID, NV_ENC_CODEC_HEVC_GUID);
    }

    #[test]
    fn preset_guids_are_unique() {
        let presets = [
            NV_ENC_PRESET_P1_GUID,
            NV_ENC_PRESET_P2_GUID,
            NV_ENC_PRESET_P3_GUID,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_PRESET_P5_GUID,
            NV_ENC_PRESET_P6_GUID,
            NV_ENC_PRESET_P7_GUID,
        ];
        for i in 0..presets.len() {
            for j in (i + 1)..presets.len() {
                assert_ne!(presets[i], presets[j], "Preset {i} and {j} have same GUID");
            }
        }
    }

    #[test]
    fn profile_guids_are_unique() {
        let profiles = [
            NV_ENC_H264_PROFILE_BASELINE_GUID,
            NV_ENC_H264_PROFILE_MAIN_GUID,
            NV_ENC_H264_PROFILE_HIGH_GUID,
        ];
        for i in 0..profiles.len() {
            for j in (i + 1)..profiles.len() {
                assert_ne!(profiles[i], profiles[j], "Profile {i} and {j} have same GUID");
            }
        }
    }

    #[test]
    fn check_status_success() {
        assert!(check_nvenc_status(NV_ENC_SUCCESS, "test").is_ok());
    }

    #[test]
    fn check_status_failure() {
        let err = check_nvenc_status(NV_ENC_ERR_INVALID_PARAM, "nvEncTest");
        assert!(err.is_err());
        let msg = err.unwrap_err();
        assert!(msg.contains("nvEncTest"));
        assert!(msg.contains("INVALID_PARAM"));
    }

    #[test]
    fn library_name_is_correct() {
        let name = NvencLibrary::library_name();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "nvEncodeAPI64.dll");
        } else {
            assert!(name.starts_with("libnvidia-encode"));
        }
    }

    #[test]
    fn nvenc_status_names() {
        assert_eq!(nvenc_status_name(0), "NV_ENC_SUCCESS");
        assert_eq!(nvenc_status_name(8), "NV_ENC_ERR_INVALID_PARAM");
        assert_eq!(nvenc_status_name(999), "NV_ENC_ERR_UNKNOWN");
    }

    #[test]
    fn buffer_format_values() {
        assert_eq!(NvEncBufferFormat::Nv12 as u32, 0x0000_0001);
        assert_eq!(NvEncBufferFormat::Argb as u32, 0x0100_0000);
        assert_eq!(NvEncBufferFormat::Abgr as u32, 0x1000_0000);
    }

    #[test]
    fn rc_mode_values() {
        assert_eq!(NvEncRcMode::ConstQp as u32, 0);
        assert_eq!(NvEncRcMode::Vbr as u32, 1);
        assert_eq!(NvEncRcMode::Cbr as u32, 2);
    }

    #[test]
    fn eos_flag() {
        assert_eq!(NV_ENC_PIC_FLAG_EOS, 1);
    }
}
