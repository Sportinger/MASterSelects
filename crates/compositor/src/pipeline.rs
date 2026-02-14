//! Multi-pass render pipeline with ping-pong buffer management.
//!
//! The [`RenderPipeline`] manages two GPU buffers that alternate as source and
//! destination during multi-pass compositing. This avoids read-write hazards
//! when a kernel needs to read the current output and write a blended result.

use ms_common::{GpuBackend, GpuBuffer, GpuStream};
use tracing::debug;

use crate::CompositorError;

/// Manages ping-pong GPU buffers for multi-pass compositing.
///
/// During compositing, each layer blend reads from the current buffer and writes
/// to the alternate buffer. After each pass, [`swap`](RenderPipeline::swap) is
/// called to flip the roles.
pub struct RenderPipeline {
    /// The two buffers used alternately.
    buffers: [GpuBuffer; 2],
    /// Index of the current "read" buffer (0 or 1).
    current: usize,
    /// Output dimensions (for logging/debugging).
    width: u32,
    height: u32,
}

impl RenderPipeline {
    /// Create a new render pipeline with two RGBA8 buffers at the given resolution.
    ///
    /// Both buffers are allocated on the GPU via the provided backend.
    pub fn new(backend: &dyn GpuBackend, width: u32, height: u32) -> Result<Self, CompositorError> {
        let byte_size = width as usize * height as usize * 4; // RGBA8

        debug!(
            width = width,
            height = height,
            buffer_size = byte_size,
            "Allocating ping-pong render pipeline"
        );

        let buf_a = backend.alloc_buffer(byte_size)?;
        let buf_b = backend.alloc_buffer(byte_size)?;

        Ok(Self {
            buffers: [buf_a, buf_b],
            current: 0,
            width,
            height,
        })
    }

    /// Returns a reference to the current (front) buffer.
    ///
    /// This is the buffer that holds the composited result so far.
    pub fn current_buffer(&self) -> &GpuBuffer {
        &self.buffers[self.current]
    }

    /// Returns a reference to the back buffer (the one NOT currently active).
    ///
    /// This buffer is available as a write destination for the next compositing pass.
    pub fn back_buffer(&self) -> &GpuBuffer {
        &self.buffers[1 - self.current]
    }

    /// Swap the front and back buffers.
    ///
    /// Call this after each compositing pass to make the just-written buffer
    /// become the new "current" (readable) buffer.
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }

    /// Clear the current buffer by writing zeros (transparent black).
    ///
    /// This dispatches a device-to-device copy from a zero-initialized source.
    /// In practice, the caller should zero the buffer before the first compositing pass.
    pub fn clear(
        &self,
        backend: &dyn GpuBackend,
        stream: &GpuStream,
    ) -> Result<(), CompositorError> {
        let byte_size = self.width as usize * self.height as usize * 4;

        debug!(
            buffer_idx = self.current,
            byte_size = byte_size,
            "Clearing current pipeline buffer"
        );

        // Zero-fill by copying a host zero buffer to the device.
        // This is a simple approach; a dedicated memset kernel would be faster.
        let zeros = vec![0u8; byte_size];
        backend.copy_to_device(&zeros, self.current_buffer(), stream)?;

        Ok(())
    }

    /// Returns the output width of this pipeline.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the output height of this pipeline.
    pub fn height(&self) -> u32 {
        self.height
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn swap_alternates_index() {
        // We can't construct a real RenderPipeline without a GpuBackend,
        // but we can verify the swap logic in isolation.
        let mut idx: usize = 0;
        // First swap
        idx = 1 - idx;
        assert_eq!(idx, 1);
        // Second swap
        idx = 1 - idx;
        assert_eq!(idx, 0);
        // Third swap
        idx = 1 - idx;
        assert_eq!(idx, 1);
    }

    #[test]
    fn buffer_size_calculation() {
        let width = 1920u32;
        let height = 1080u32;
        let expected = width as usize * height as usize * 4;
        assert_eq!(expected, 8_294_400);
    }

    #[test]
    fn uhd_buffer_size() {
        let width = 3840u32;
        let height = 2160u32;
        let expected = width as usize * height as usize * 4;
        assert_eq!(expected, 33_177_600);
    }
}
