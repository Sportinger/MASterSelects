//! GPU decode pipeline -- attempts CUDA/NVDEC hardware decode, falls back gracefully.
//!
//! This module provides a self-contained GPU decode pipeline that:
//! 1. Tries to initialize CUDA via `ms-gpu-hal`
//! 2. Tries to create an NVDEC hardware decoder through the `GpuBackend` trait
//! 3. Falls back gracefully if no GPU is available (Phase 0 on non-NVIDIA systems)
//!
//! The pipeline never panics on initialization failure -- it simply marks GPU
//! decode as unavailable and returns `None` from `decode_packet()`.

use ms_common::config::DecoderConfig;
use ms_common::gpu_traits::{GpuBackend, HwDecoder};
use ms_common::{VideoCodec, VideoPacket};

/// Statistics about the GPU decode pipeline.
#[derive(Clone, Debug)]
pub struct PipelineStats {
    /// Total number of frames decoded (GPU or software).
    pub frames_decoded: u64,
    /// Whether GPU hardware decode is currently active.
    pub gpu_decode_active: bool,
    /// Human-readable name of the active backend.
    pub backend_name: String,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            frames_decoded: 0,
            gpu_decode_active: false,
            backend_name: "None".to_string(),
        }
    }
}

/// Handle wrapping a live CUDA backend and optional NVDEC decoder.
struct CudaBackendHandle {
    /// The CUDA backend instance (owns the GPU context).
    #[cfg(feature = "cuda")]
    _backend: ms_gpu_hal::cuda::CudaBackend,
    /// Human-readable device name, cached at init time.
    device_name: String,
}

/// Self-contained GPU decode pipeline.
pub struct GpuDecodePipeline {
    /// GPU backend handle (None if GPU initialization failed).
    cuda_handle: Option<CudaBackendHandle>,
    /// Hardware decoder (None if NVDEC is not available or init failed).
    decoder: Option<Box<dyn HwDecoder>>,
    /// Frame width expected from the decoder.
    _width: u32,
    /// Frame height expected from the decoder.
    _height: u32,
    /// Codec this pipeline was configured for.
    _codec: VideoCodec,
    /// Total frames decoded so far.
    frames_decoded: u64,
    /// Whether the GPU path is fully operational.
    gpu_decode_active: bool,
}

impl GpuDecodePipeline {
    /// Try to initialize a GPU decode pipeline.
    ///
    /// This constructor **never panics**. If any GPU component is unavailable
    /// (no NVIDIA GPU, missing driver, NVDEC not implemented yet, etc.), the
    /// pipeline is created in fallback mode where `is_gpu_available()` returns
    /// `false` and `decode_packet()` returns `None`.
    pub fn new(width: u32, height: u32, codec: VideoCodec) -> Self {
        let (cuda_handle, decoder, gpu_active) = Self::try_init_gpu(width, height, codec);

        if gpu_active {
            if let Some(ref handle) = cuda_handle {
                tracing::info!(
                    device = %handle.device_name,
                    "GPU decode pipeline initialized: CUDA + NVDEC active"
                );
            }
        } else if cuda_handle.is_some() {
            tracing::info!(
                "GPU decode pipeline: CUDA available but NVDEC not ready, decode will be synthetic"
            );
        } else {
            tracing::info!(
                "GPU decode pipeline: no GPU available, decode will be synthetic"
            );
        }

        Self {
            cuda_handle,
            decoder,
            _width: width,
            _height: height,
            _codec: codec,
            frames_decoded: 0,
            gpu_decode_active: gpu_active,
        }
    }

    /// Returns `true` if the full GPU decode path is operational (CUDA + NVDEC).
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_decode_active
    }

    /// Returns the GPU device name, or "None" if no GPU is available.
    pub fn gpu_device_name(&self) -> &str {
        match &self.cuda_handle {
            Some(handle) => &handle.device_name,
            None => "None",
        }
    }

    /// Attempt to decode a video packet through the GPU pipeline.
    ///
    /// Returns `Some(rgba_data)` if GPU decode succeeds, `None` otherwise.
    /// When `None` is returned, the caller should fall back to synthetic
    /// frame generation.
    pub fn decode_packet(&mut self, packet: &VideoPacket) -> Option<Vec<u8>> {
        if !self.gpu_decode_active {
            return None;
        }

        // We have a working decoder -- feed the packet
        let decoder = self.decoder.as_mut()?;

        match decoder.decode(packet) {
            Ok(Some(gpu_frame)) => {
                let _frame = gpu_frame;
                self.frames_decoded += 1;
                // TODO: Run NV12->RGBA kernel via backend.dispatch_kernel()
                // TODO: Copy result to staging via backend.copy_to_staging()
                // TODO: Read staging buffer into Vec<u8>
                tracing::debug!(
                    frame = self.frames_decoded,
                    "GPU frame decoded (NV12), RGBA conversion not yet wired"
                );
                None
            }
            Ok(None) => {
                // Decoder consumed packet but no frame produced yet (buffering B-frames).
                None
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "GPU decode failed, disabling GPU path"
                );
                self.gpu_decode_active = false;
                None
            }
        }
    }

    /// Get current pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            frames_decoded: self.frames_decoded,
            gpu_decode_active: self.gpu_decode_active,
            backend_name: match &self.cuda_handle {
                Some(handle) => format!("CUDA ({})", handle.device_name),
                None => "None".to_string(),
            },
        }
    }

    /// Reset the decoder state (e.g., after a seek).
    pub fn reset(&mut self) {
        if let Some(ref mut decoder) = self.decoder {
            match decoder.flush() {
                Ok(flushed) => {
                    tracing::debug!(
                        flushed_frames = flushed.len(),
                        "GPU decoder reset (flushed buffered frames)"
                    );
                }
                Err(e) => {
                    tracing::warn!(error = %e, "GPU decoder flush failed during reset");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal GPU initialization
    // -----------------------------------------------------------------------

    /// Try to set up the full GPU pipeline: CUDA backend + NVDEC decoder.
    fn try_init_gpu(
        width: u32,
        height: u32,
        codec: VideoCodec,
    ) -> (Option<CudaBackendHandle>, Option<Box<dyn HwDecoder>>, bool) {
        // Step 1: Try to create the CUDA backend
        let cuda_handle = Self::try_init_cuda();
        let cuda_handle = match cuda_handle {
            Some(h) => h,
            None => return (None, None, false),
        };

        // Step 2: Try to create an NVDEC decoder via the GpuBackend trait
        let decoder_config = DecoderConfig::new(codec, ms_common::Resolution::new(width, height));

        #[cfg(feature = "cuda")]
        let decoder = match cuda_handle._backend.create_decoder(&decoder_config) {
            Ok(dec) => {
                tracing::info!(
                    codec = ?codec,
                    resolution = %format!("{}x{}", width, height),
                    "NVDEC hardware decoder created successfully"
                );
                Some(dec)
            }
            Err(e) => {
                tracing::info!(
                    error = %e,
                    "NVDEC decoder not available (expected in Phase 0)"
                );
                None
            }
        };

        #[cfg(not(feature = "cuda"))]
        let decoder: Option<Box<dyn HwDecoder>> = None;

        let gpu_active = decoder.is_some();
        (Some(cuda_handle), decoder, gpu_active)
    }

    /// Try to initialize the CUDA backend on device 0.
    fn try_init_cuda() -> Option<CudaBackendHandle> {
        #[cfg(feature = "cuda")]
        {
            match ms_gpu_hal::cuda::CudaBackend::new_default() {
                Ok(backend) => {
                    let device_name = backend.device_name().to_string();
                    tracing::info!(
                        device = %device_name,
                        vram_mb = backend.vram_total() / (1024 * 1024),
                        "CUDA backend initialized for decode pipeline"
                    );
                    Some(CudaBackendHandle {
                        _backend: backend,
                        device_name,
                    })
                }
                Err(e) => {
                    tracing::info!(
                        error = %e,
                        "CUDA backend not available (no NVIDIA GPU or driver not installed)"
                    );
                    None
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            tracing::info!("CUDA feature not enabled at compile time, GPU decode unavailable");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_new_never_panics() {
        let pipeline = GpuDecodePipeline::new(1920, 1080, VideoCodec::H264);
        let _ = pipeline.is_gpu_available();
    }

    #[test]
    fn pipeline_stats_initial_values() {
        let pipeline = GpuDecodePipeline::new(1920, 1080, VideoCodec::H264);
        let stats = pipeline.stats();
        assert_eq!(stats.frames_decoded, 0);
        let _ = stats.gpu_decode_active;
        assert!(!stats.backend_name.is_empty());
    }

    #[test]
    fn decode_packet_returns_none_without_nvdec() {
        let mut pipeline = GpuDecodePipeline::new(1920, 1080, VideoCodec::H264);
        let packet = VideoPacket {
            data: vec![0x00, 0x00, 0x00, 0x01, 0x67],
            pts: ms_common::TimeCode(0.0),
            dts: ms_common::TimeCode(0.0),
            is_keyframe: true,
            codec: VideoCodec::H264,
        };
        let result = pipeline.decode_packet(&packet);
        assert!(result.is_none());
    }

    #[test]
    fn reset_does_not_panic() {
        let mut pipeline = GpuDecodePipeline::new(640, 480, VideoCodec::H264);
        pipeline.reset();
        pipeline.reset();
    }

    #[test]
    fn pipeline_stats_backend_name_is_meaningful() {
        let pipeline = GpuDecodePipeline::new(1920, 1080, VideoCodec::H265);
        let stats = pipeline.stats();
        assert!(
            stats.backend_name == "None" || stats.backend_name.starts_with("CUDA"),
            "backend_name should be None or start with CUDA, got: {}",
            stats.backend_name
        );
    }

    #[test]
    fn gpu_device_name_returns_valid_string() {
        let pipeline = GpuDecodePipeline::new(1920, 1080, VideoCodec::H264);
        let name = pipeline.gpu_device_name();
        assert!(!name.is_empty());
    }

    #[test]
    fn pipeline_different_codecs() {
        let _h264 = GpuDecodePipeline::new(1920, 1080, VideoCodec::H264);
        let _h265 = GpuDecodePipeline::new(3840, 2160, VideoCodec::H265);
        let _av1 = GpuDecodePipeline::new(1280, 720, VideoCodec::Av1);
    }

    #[test]
    fn pipeline_small_resolution() {
        let pipeline = GpuDecodePipeline::new(16, 16, VideoCodec::H264);
        assert_eq!(pipeline.stats().frames_decoded, 0);
    }
}
