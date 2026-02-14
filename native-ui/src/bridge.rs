//! Preview Bridge — transfers GPU-rendered frames to egui's texture system.
//!
//! Pipeline: GPU Frame -> Staging -> CPU RGBA bytes -> egui ColorImage -> TextureHandle
//!
//! For Phase 0 we take the simple path: copy frame data to a CPU buffer, then
//! upload as an `egui::ColorImage` texture. This avoids complex wgpu interop
//! and works for both CUDA and Vulkan backends.

use egui::{ColorImage, TextureHandle, TextureOptions};
use std::time::Instant;

/// Stats about the preview display pipeline.
#[derive(Clone, Debug)]
pub struct PreviewStats {
    pub width: u32,
    pub height: u32,
    pub last_frame_time_ms: f64,
    pub frames_displayed: u64,
    pub fps: f64,
}

/// Bridges GPU frame output to egui's texture system.
pub struct PreviewBridge {
    /// CPU-side RGBA buffer for the current frame.
    frame_buffer: Vec<u8>,
    /// egui texture handle (lazily created on first frame).
    texture_handle: Option<TextureHandle>,
    /// Frame width in pixels.
    width: u32,
    /// Frame height in pixels.
    height: u32,
    /// Timestamp of the last frame upload.
    last_frame_instant: Option<Instant>,
    /// Duration of the last frame upload in milliseconds.
    last_frame_time_ms: f64,
    /// Total number of frames displayed.
    frames_displayed: u64,
    /// Rolling FPS estimate.
    fps: f64,
    /// Timestamps for FPS averaging window.
    frame_times: Vec<Instant>,
}

impl PreviewBridge {
    /// Create a new preview bridge with the given initial dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let buffer_size = (width as usize) * (height as usize) * 4;
        Self {
            frame_buffer: vec![0u8; buffer_size],
            texture_handle: None,
            width,
            height,
            last_frame_instant: None,
            last_frame_time_ms: 0.0,
            frames_displayed: 0,
            fps: 0.0,
            frame_times: Vec::with_capacity(64),
        }
    }

    /// Upload RGBA pixel data as an egui texture.
    ///
    /// `data` must be exactly `width * height * 4` bytes (RGBA8).
    /// Creates a new texture handle on first call or when dimensions change,
    /// otherwise updates the existing texture in place.
    pub fn update_from_rgba_bytes(
        &mut self,
        ctx: &egui::Context,
        data: &[u8],
        width: u32,
        height: u32,
    ) {
        let start = Instant::now();

        let expected_size = (width as usize) * (height as usize) * 4;
        if data.len() != expected_size {
            tracing::warn!(
                "PreviewBridge: data length {} does not match expected {} ({}x{}x4)",
                data.len(),
                expected_size,
                width,
                height,
            );
            return;
        }

        // Handle resolution change
        if width != self.width || height != self.height {
            self.resize(width, height);
        }

        // Copy into our frame buffer
        self.frame_buffer.copy_from_slice(data);

        // Build egui::ColorImage from raw RGBA bytes
        let color_image = ColorImage::from_rgba_unmultiplied(
            [width as usize, height as usize],
            &self.frame_buffer,
        );

        // Create or update the texture handle
        match &mut self.texture_handle {
            Some(handle) => {
                handle.set(color_image, TextureOptions::LINEAR);
            }
            None => {
                let handle = ctx.load_texture(
                    "preview_frame",
                    color_image,
                    TextureOptions::LINEAR,
                );
                self.texture_handle = Some(handle);
            }
        }

        // Update timing stats
        let elapsed = start.elapsed();
        self.last_frame_time_ms = elapsed.as_secs_f64() * 1000.0;
        self.frames_displayed += 1;

        // FPS calculation using a rolling window
        let now = Instant::now();
        self.frame_times.push(now);
        // Keep only the last 60 timestamps
        if self.frame_times.len() > 60 {
            self.frame_times.drain(0..self.frame_times.len() - 60);
        }
        if self.frame_times.len() >= 2 {
            let window_duration = now
                .duration_since(self.frame_times[0])
                .as_secs_f64();
            if window_duration > 0.0 {
                self.fps = (self.frame_times.len() - 1) as f64 / window_duration;
            }
        }

        self.last_frame_instant = Some(now);
    }

    /// Get the current texture ID for rendering in egui, if a frame has been uploaded.
    pub fn texture_id(&self) -> Option<egui::TextureId> {
        self.texture_handle.as_ref().map(|h| h.id())
    }

    /// Get the texture size for display layout.
    pub fn texture_size(&self) -> [u32; 2] {
        [self.width, self.height]
    }

    /// Handle resolution changes by reallocating the frame buffer
    /// and dropping the old texture handle so a new one is created.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        tracing::info!(
            "PreviewBridge: resize {}x{} -> {}x{}",
            self.width,
            self.height,
            width,
            height,
        );
        self.width = width;
        self.height = height;
        let buffer_size = (width as usize) * (height as usize) * 4;
        self.frame_buffer = vec![0u8; buffer_size];
        // Drop old texture handle so a fresh one is created on next upload
        self.texture_handle = None;
    }

    /// Return current preview pipeline statistics.
    pub fn stats(&self) -> PreviewStats {
        PreviewStats {
            width: self.width,
            height: self.height,
            last_frame_time_ms: self.last_frame_time_ms,
            frames_displayed: self.frames_displayed,
            fps: self.fps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_correct_buffer_size() {
        let bridge = PreviewBridge::new(1920, 1080);
        assert_eq!(bridge.frame_buffer.len(), 1920 * 1080 * 4);
        assert_eq!(bridge.width, 1920);
        assert_eq!(bridge.height, 1080);
        assert!(bridge.texture_handle.is_none());
        assert_eq!(bridge.frames_displayed, 0);
    }

    #[test]
    fn resize_reallocates_buffer() {
        let mut bridge = PreviewBridge::new(1920, 1080);
        bridge.resize(1280, 720);
        assert_eq!(bridge.frame_buffer.len(), 1280 * 720 * 4);
        assert_eq!(bridge.width, 1280);
        assert_eq!(bridge.height, 720);
    }

    #[test]
    fn resize_no_op_same_size() {
        let mut bridge = PreviewBridge::new(1920, 1080);
        bridge.resize(1920, 1080);
        assert_eq!(bridge.frame_buffer.len(), 1920 * 1080 * 4);
    }

    #[test]
    fn stats_initial_values() {
        let bridge = PreviewBridge::new(1920, 1080);
        let stats = bridge.stats();
        assert_eq!(stats.width, 1920);
        assert_eq!(stats.height, 1080);
        assert_eq!(stats.frames_displayed, 0);
        assert_eq!(stats.last_frame_time_ms, 0.0);
    }

    #[test]
    fn texture_id_none_before_upload() {
        let bridge = PreviewBridge::new(640, 480);
        assert!(bridge.texture_id().is_none());
    }
}
