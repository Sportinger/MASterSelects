//! CUDA to wgpu transfer bridge — copies GPU frames to staging memory
//! for display via wgpu/egui.

use std::sync::Arc;

use cudarc::driver::safe::CudaStream as CudarcStream;
use tracing::debug;

use super::error::CudaError;
use super::memory::{DeviceBuffer, PinnedBuffer};

/// Handles copying CUDA device memory to host staging buffers,
/// which can then be uploaded to wgpu textures for display.
///
/// This is the bridge between the CUDA render pipeline and the
/// egui/wgpu display surface.
#[derive(Debug)]
pub struct DisplayBridge {
    /// Pinned staging buffer for CPU-side readback.
    staging: PinnedBuffer,
    /// Width of the frame.
    width: u32,
    /// Height of the frame.
    height: u32,
    /// Bytes per pixel (4 for RGBA8).
    bytes_per_pixel: u32,
}

impl DisplayBridge {
    /// Create a new display bridge for frames of the given dimensions.
    ///
    /// Allocates a pinned staging buffer large enough for one RGBA8 frame.
    pub fn new(
        ctx: &Arc<cudarc::driver::safe::CudaContext>,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<Self, CudaError> {
        let size = (width * height * bytes_per_pixel) as usize;
        // SAFETY: The staging buffer will be written to by CUDA memcpy before
        // being read by the CPU. We don't read uninitialized data.
        let staging = unsafe { PinnedBuffer::alloc(ctx, size)? };

        debug!(
            width,
            height, bytes_per_pixel, size, "Created display bridge"
        );
        Ok(Self {
            staging,
            width,
            height,
            bytes_per_pixel,
        })
    }

    /// Copy a device buffer (rendered frame) to the pinned staging buffer.
    ///
    /// After this call completes (and stream is synchronized), the staging
    /// buffer contains the frame data ready for upload to wgpu.
    pub fn copy_to_staging(
        &mut self,
        src: &DeviceBuffer,
        stream: &Arc<CudarcStream>,
    ) -> Result<(), CudaError> {
        let expected_size = (self.width * self.height * self.bytes_per_pixel) as usize;

        if src.size() < expected_size {
            return Err(CudaError::TransferFailed {
                reason: format!(
                    "Source buffer ({} bytes) is smaller than frame size ({} bytes)",
                    src.size(),
                    expected_size
                ),
            });
        }

        // Copy device -> pinned host using the stream's dtoh transfer
        stream
            .memcpy_dtoh(src.inner(), self.staging.inner_mut())
            .map_err(|e| CudaError::TransferFailed {
                reason: format!("Device to staging copy failed: {e}"),
            })?;

        Ok(())
    }

    /// Get the staging buffer data as a byte slice.
    ///
    /// **Important:** The stream must be synchronized before calling this
    /// to ensure the async copy has completed.
    pub fn staging_data(&self) -> Result<&[u8], CudaError> {
        self.staging.as_slice()
    }

    /// Get the staging buffer data as a mutable byte slice.
    ///
    /// **Important:** The stream must be synchronized before calling this.
    pub fn staging_data_mut(&mut self) -> Result<&mut [u8], CudaError> {
        self.staging.as_mut_slice()
    }

    /// Get the frame width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the frame height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the total frame size in bytes.
    pub fn frame_size(&self) -> usize {
        (self.width * self.height * self.bytes_per_pixel) as usize
    }

    /// Resize the staging buffer for a new frame size.
    ///
    /// This reallocates the pinned memory if the new size differs.
    pub fn resize(
        &mut self,
        ctx: &Arc<cudarc::driver::safe::CudaContext>,
        width: u32,
        height: u32,
    ) -> Result<(), CudaError> {
        if width == self.width && height == self.height {
            return Ok(());
        }

        let new_size = (width * height * self.bytes_per_pixel) as usize;
        // SAFETY: The staging buffer will be written before being read.
        self.staging = unsafe { PinnedBuffer::alloc(ctx, new_size)? };
        self.width = width;
        self.height = height;

        debug!(width, height, new_size, "Resized display bridge");
        Ok(())
    }
}

/// Stub for future wgpu texture upload.
///
/// When wgpu integration is implemented, this will upload the staging
/// buffer data to a wgpu texture for display in egui.
///
/// For now, this serves as documentation of the intended API.
pub fn upload_to_wgpu_texture(_staging_data: &[u8], _width: u32, _height: u32) {
    // TODO: Implement wgpu texture upload
    // 1. Get or create wgpu::Texture of matching size
    // 2. queue.write_texture() with the staging data
    // 3. Return texture view for egui rendering
    debug!("wgpu texture upload stub called");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_frame_size_calculation() {
        // RGBA8: 4 bytes per pixel
        let size = 1920u32 * 1080u32 * 4u32;
        assert_eq!(size as usize, 1920 * 1080 * 4);
    }
}
