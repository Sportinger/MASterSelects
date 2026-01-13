//! Decoder context pool management

use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

use super::VideoDecoder;
use crate::protocol::{FileMetadata, PixelFormat};

/// Open decoder with metadata
struct OpenDecoder {
    decoder: VideoDecoder,
    path: PathBuf,
    last_used: Instant,
    file_id: String,
}

/// Thread-safe decoder pool
pub struct DecoderPool {
    decoders: Mutex<HashMap<String, OpenDecoder>>,
    max_decoders: usize,
    file_id_counter: Mutex<u64>,
}

impl DecoderPool {
    /// Create a new decoder pool
    pub fn new(max_decoders: usize) -> Self {
        Self {
            decoders: Mutex::new(HashMap::new()),
            max_decoders,
            file_id_counter: Mutex::new(0),
        }
    }

    /// Generate a new file ID
    fn generate_file_id(&self) -> String {
        let mut counter = self.file_id_counter.lock();
        *counter += 1;
        format!("file_{:08x}", *counter)
    }

    /// Open a file and return its metadata
    pub fn open(&self, path: impl Into<PathBuf>) -> Result<FileMetadata> {
        let path = path.into();
        let file_id = self.generate_file_id();

        // Check if we need to evict
        {
            let mut decoders = self.decoders.lock();
            if decoders.len() >= self.max_decoders {
                // Find oldest decoder
                let oldest = decoders
                    .iter()
                    .min_by_key(|(_, d)| d.last_used)
                    .map(|(k, _)| k.clone());

                if let Some(key) = oldest {
                    debug!("Evicting decoder for {}", key);
                    decoders.remove(&key);
                }
            }
        }

        // Open decoder
        let decoder = VideoDecoder::open(&path)?;
        let metadata = decoder.metadata(&file_id);

        let open = OpenDecoder {
            decoder,
            path,
            last_used: Instant::now(),
            file_id: file_id.clone(),
        };

        self.decoders.lock().insert(file_id.clone(), open);

        info!("Opened file: {} -> {}", metadata.codec, file_id);

        Ok(metadata)
    }

    /// Close a file
    pub fn close(&self, file_id: &str) -> bool {
        self.decoders.lock().remove(file_id).is_some()
    }

    /// Decode a frame from an open file
    pub fn decode_frame(
        &self,
        file_id: &str,
        frame: u32,
        format: PixelFormat,
        scale: f32,
    ) -> Result<super::ffmpeg::DecodedFrameData> {
        let mut decoders = self.decoders.lock();

        let open = decoders
            .get_mut(file_id)
            .ok_or_else(|| anyhow::anyhow!("File not open: {}", file_id))?;

        open.last_used = Instant::now();
        open.decoder.set_output(format, scale)?;

        open.decoder.decode_frame(frame)
    }

    /// Get metadata for an open file
    pub fn get_metadata(&self, file_id: &str) -> Option<FileMetadata> {
        let decoders = self.decoders.lock();
        decoders.get(file_id).map(|d| d.decoder.metadata(file_id))
    }

    /// Get file path for an open file
    pub fn get_path(&self, file_id: &str) -> Option<PathBuf> {
        let decoders = self.decoders.lock();
        decoders.get(file_id).map(|d| d.path.clone())
    }

    /// Check if file is open
    pub fn is_open(&self, file_id: &str) -> bool {
        self.decoders.lock().contains_key(file_id)
    }

    /// Get number of open decoders
    pub fn open_count(&self) -> usize {
        self.decoders.lock().len()
    }

    /// Get all open file IDs
    pub fn open_files(&self) -> Vec<String> {
        self.decoders.lock().keys().cloned().collect()
    }

    /// Get frame count for a file
    pub fn frame_count(&self, file_id: &str) -> Option<u64> {
        let decoders = self.decoders.lock();
        decoders.get(file_id).map(|d| d.decoder.frame_count())
    }

    /// Touch a file (update last used time)
    pub fn touch(&self, file_id: &str) {
        if let Some(open) = self.decoders.lock().get_mut(file_id) {
            open.last_used = Instant::now();
        }
    }
}

/// Thread-safe wrapper for shared decoder pool
pub type SharedDecoderPool = Arc<DecoderPool>;
