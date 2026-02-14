//! Decoder pool manager — creates and manages NVDEC decoder instances.
//!
//! The `DecoderManager` handles:
//! - Loading the NVDEC library (once, shared across all decoders)
//! - Creating decoder instances per source file / codec
//! - Tracking active decoders and their resource usage
//! - Graceful fallback when NVDEC is not available

use std::collections::HashMap;
use std::sync::Arc;

use tracing::{info, warn};

use ms_common::{DecodeError, VideoCodec};

use crate::nvdec::{NvDecoder, NvcuvidLibrary};

/// Unique identifier for a decoder instance.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DecoderId(pub String);

impl DecoderId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Configuration for creating a decoder.
#[derive(Clone, Debug)]
pub struct DecoderCreateConfig {
    /// Video codec.
    pub codec: VideoCodec,
    /// Number of decode surfaces (DPB size).
    pub num_surfaces: u32,
    /// Maximum display delay (frame reordering depth).
    pub max_display_delay: u32,
}

impl Default for DecoderCreateConfig {
    fn default() -> Self {
        Self {
            codec: VideoCodec::H264,
            num_surfaces: 20,
            max_display_delay: 4,
        }
    }
}

/// Manages a pool of NVDEC decoder instances.
///
/// The manager loads the NVDEC library once and shares it across all
/// decoder instances. It provides a simple interface for creating,
/// tracking, and destroying decoders.
pub struct DecoderManager {
    /// Shared NVDEC library handle (None if loading failed).
    nvdec_lib: Option<Arc<NvcuvidLibrary>>,
    /// Active decoder instances.
    decoders: HashMap<DecoderId, NvDecoder>,
    /// Whether NVDEC is available on this system.
    nvdec_available: bool,
    /// Error message if NVDEC failed to load.
    nvdec_load_error: Option<String>,
}

impl std::fmt::Debug for DecoderManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderManager")
            .field("nvdec_available", &self.nvdec_available)
            .field("active_decoders", &self.decoders.len())
            .field("nvdec_load_error", &self.nvdec_load_error)
            .finish()
    }
}

impl DecoderManager {
    /// Create a new decoder manager.
    ///
    /// Attempts to load the NVDEC library. If loading fails (e.g., no NVIDIA
    /// GPU or driver not installed), the manager is still created but
    /// `is_nvdec_available()` will return false.
    pub fn new() -> Self {
        let (nvdec_lib, nvdec_available, nvdec_load_error) = match NvcuvidLibrary::load() {
            Ok(lib) => {
                info!("NVDEC library loaded successfully");
                (Some(Arc::new(lib)), true, None)
            }
            Err(e) => {
                warn!(error = %e, "NVDEC library not available — hardware decode disabled");
                (None, false, Some(e.to_string()))
            }
        };

        Self {
            nvdec_lib,
            decoders: HashMap::new(),
            nvdec_available,
            nvdec_load_error,
        }
    }

    /// Check if NVDEC hardware decoding is available.
    pub fn is_nvdec_available(&self) -> bool {
        self.nvdec_available
    }

    /// Get the error message if NVDEC failed to load.
    pub fn nvdec_error(&self) -> Option<&str> {
        self.nvdec_load_error.as_deref()
    }

    /// Create a new NVDEC decoder instance.
    ///
    /// # Arguments
    /// * `id` — Unique identifier for this decoder (e.g., source file path).
    /// * `config` — Decoder configuration (codec, surfaces, etc.).
    ///
    /// # Errors
    /// Returns an error if NVDEC is not available or decoder creation fails.
    pub fn create_decoder(
        &mut self,
        id: DecoderId,
        config: &DecoderCreateConfig,
    ) -> Result<&mut NvDecoder, DecodeError> {
        let lib = self
            .nvdec_lib
            .as_ref()
            .ok_or_else(|| DecodeError::HwDecoderInit {
                codec: config.codec,
                reason: self
                    .nvdec_load_error
                    .clone()
                    .unwrap_or_else(|| "NVDEC not available".to_string()),
            })?;

        let decoder = NvDecoder::with_config(
            lib.clone(),
            config.codec,
            config.num_surfaces,
            config.max_display_delay,
        )?;

        info!(
            id = %id.0,
            codec = config.codec.display_name(),
            "Created NVDEC decoder"
        );

        self.decoders.insert(id.clone(), decoder);
        Ok(self.decoders.get_mut(&id).expect("just inserted"))
    }

    /// Get a mutable reference to an existing decoder.
    pub fn get_decoder(&mut self, id: &DecoderId) -> Option<&mut NvDecoder> {
        self.decoders.get_mut(id)
    }

    /// Get an immutable reference to an existing decoder.
    pub fn get_decoder_ref(&self, id: &DecoderId) -> Option<&NvDecoder> {
        self.decoders.get(id)
    }

    /// Remove and destroy a decoder instance.
    pub fn destroy_decoder(&mut self, id: &DecoderId) -> bool {
        if self.decoders.remove(id).is_some() {
            info!(id = %id.0, "Destroyed NVDEC decoder");
            true
        } else {
            false
        }
    }

    /// Get the number of active decoder instances.
    pub fn active_decoder_count(&self) -> usize {
        self.decoders.len()
    }

    /// Destroy all active decoder instances.
    pub fn destroy_all(&mut self) {
        let count = self.decoders.len();
        self.decoders.clear();
        if count > 0 {
            info!(count, "Destroyed all NVDEC decoders");
        }
    }

    /// Get a list of supported codecs (all codecs that NVDEC can handle).
    pub fn supported_codecs() -> &'static [VideoCodec] {
        &[
            VideoCodec::H264,
            VideoCodec::H265,
            VideoCodec::Vp9,
            VideoCodec::Av1,
        ]
    }
}

impl Default for DecoderManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_id_equality() {
        let id1 = DecoderId::new("video.mp4");
        let id2 = DecoderId::new("video.mp4");
        let id3 = DecoderId::new("other.mp4");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn default_config() {
        let config = DecoderCreateConfig::default();
        assert_eq!(config.codec, VideoCodec::H264);
        assert_eq!(config.num_surfaces, 20);
        assert_eq!(config.max_display_delay, 4);
    }

    #[test]
    fn supported_codecs_not_empty() {
        let codecs = DecoderManager::supported_codecs();
        assert!(!codecs.is_empty());
        assert!(codecs.contains(&VideoCodec::H264));
    }

    #[test]
    fn manager_creation_without_nvidia() {
        // On CI or machines without NVIDIA GPUs, the manager should still
        // be created — just with nvdec_available = false.
        let manager = DecoderManager::new();
        // Don't assert nvdec_available since it depends on hardware.
        assert_eq!(manager.active_decoder_count(), 0);
    }
}
