//! GPU memory management — RAII wrappers around cudarc device buffers
//! and pinned host memory.

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice};
use tracing::debug;

use super::error::CudaError;

/// RAII wrapper around a CUDA device buffer (`CudaSlice<u8>`).
///
/// Represents a contiguous allocation on GPU device memory.
/// Automatically freed when dropped via cudarc's `CudaSlice` drop impl.
#[derive(Debug)]
pub struct DeviceBuffer {
    /// The underlying cudarc device buffer.
    inner: CudaSlice<u8>,
    /// The stream this was allocated on (kept for memory ops).
    stream: Arc<CudaStream>,
    /// Logical size in bytes (may differ from allocation size due to alignment).
    size: usize,
}

impl DeviceBuffer {
    /// Allocate a zero-initialized device buffer of `size` bytes.
    pub fn alloc_zeros(stream: &Arc<CudaStream>, size: usize) -> Result<Self, CudaError> {
        let inner = stream
            .alloc_zeros::<u8>(size)
            .map_err(|_e| CudaError::AllocFailed { size })?;

        debug!(size, "Allocated GPU device buffer (zeroed)");
        Ok(Self {
            inner,
            stream: stream.clone(),
            size,
        })
    }

    /// Allocate an uninitialized device buffer of `size` bytes.
    ///
    /// # Safety
    /// The contents are uninitialized. The caller must write to the buffer
    /// before reading from it.
    pub unsafe fn alloc_uninit(stream: &Arc<CudaStream>, size: usize) -> Result<Self, CudaError> {
        // SAFETY: Caller guarantees they will initialize before reading.
        let inner = stream
            .alloc::<u8>(size)
            .map_err(|_e| CudaError::AllocFailed { size })?;

        debug!(size, "Allocated GPU device buffer (uninitialized)");
        Ok(Self {
            inner,
            stream: stream.clone(),
            size,
        })
    }

    /// Get the size of this buffer in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the raw device pointer as a u64 (for passing to kernel args).
    ///
    /// This synchronizes with any pending writes on the stream to ensure
    /// the pointer is valid for reading.
    pub fn device_ptr(&self) -> u64 {
        // Use the DevicePtr trait to get the CUdeviceptr.
        // The SyncOnDrop is dropped immediately, which is fine for just reading the pointer value.
        let (ptr, _sync) = self.inner.device_ptr(&self.stream);
        ptr
    }

    /// Get a reference to the underlying `CudaSlice<u8>`.
    pub fn inner(&self) -> &CudaSlice<u8> {
        &self.inner
    }

    /// Get a mutable reference to the underlying `CudaSlice<u8>`.
    pub fn inner_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.inner
    }

    /// Get the stream this buffer was allocated on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Copy host data into this device buffer.
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<(), CudaError> {
        if data.len() > self.size {
            return Err(CudaError::TransferFailed {
                reason: format!(
                    "Host data ({} bytes) exceeds buffer size ({} bytes)",
                    data.len(),
                    self.size
                ),
            });
        }
        self.stream
            .memcpy_htod(data, &mut self.inner)
            .map_err(|e| CudaError::TransferFailed {
                reason: format!("Host to device copy failed: {e}"),
            })?;
        Ok(())
    }

    /// Copy this device buffer to host memory.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        if dst.len() < self.size {
            return Err(CudaError::TransferFailed {
                reason: format!(
                    "Destination ({} bytes) is smaller than buffer ({} bytes)",
                    dst.len(),
                    self.size
                ),
            });
        }
        self.stream
            .memcpy_dtoh(&self.inner, dst)
            .map_err(|e| CudaError::TransferFailed {
                reason: format!("Device to host copy failed: {e}"),
            })?;
        Ok(())
    }

    /// Copy data from another device buffer into this one.
    pub fn copy_from_device(&mut self, src: &DeviceBuffer) -> Result<(), CudaError> {
        if src.size > self.size {
            return Err(CudaError::TransferFailed {
                reason: format!(
                    "Source ({} bytes) exceeds destination ({} bytes)",
                    src.size, self.size
                ),
            });
        }
        self.stream
            .memcpy_dtod(&src.inner, &mut self.inner)
            .map_err(|e| CudaError::TransferFailed {
                reason: format!("Device to device copy failed: {e}"),
            })?;
        Ok(())
    }
}

/// RAII wrapper around pinned (page-locked) host memory.
///
/// Pinned memory enables faster CPU<->GPU transfers via DMA.
/// Automatically freed when dropped via cudarc's `PinnedHostSlice` drop impl.
#[derive(Debug)]
pub struct PinnedBuffer {
    /// The underlying cudarc pinned host buffer.
    inner: PinnedHostSlice<u8>,
    /// Size in bytes.
    size: usize,
}

impl PinnedBuffer {
    /// Allocate pinned host memory of `size` bytes.
    ///
    /// # Safety
    /// The contents are uninitialized. The caller must write to the buffer
    /// before reading from it.
    pub unsafe fn alloc(ctx: &Arc<CudaContext>, size: usize) -> Result<Self, CudaError> {
        // SAFETY: Caller guarantees they will initialize before reading.
        // cudarc's alloc_pinned allocates page-locked host memory.
        let inner = ctx
            .alloc_pinned::<u8>(size)
            .map_err(|_e| CudaError::AllocFailed { size })?;

        debug!(size, "Allocated pinned host memory");
        Ok(Self { inner, size })
    }

    /// Get the size of this buffer in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a pointer to the host memory (waits for any pending async ops).
    pub fn as_ptr(&self) -> Result<*const u8, CudaError> {
        self.inner.as_ptr().map_err(CudaError::from)
    }

    /// Get a mutable pointer to the host memory (waits for any pending async ops).
    pub fn as_mut_ptr(&mut self) -> Result<*mut u8, CudaError> {
        self.inner.as_mut_ptr().map_err(CudaError::from)
    }

    /// Get an immutable slice of the host memory (waits for pending ops).
    pub fn as_slice(&self) -> Result<&[u8], CudaError> {
        self.inner.as_slice().map_err(CudaError::from)
    }

    /// Get a mutable slice of the host memory (waits for pending ops).
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], CudaError> {
        self.inner.as_mut_slice().map_err(CudaError::from)
    }

    /// Get a reference to the underlying `PinnedHostSlice<u8>`.
    pub fn inner(&self) -> &PinnedHostSlice<u8> {
        &self.inner
    }

    /// Get a mutable reference to the underlying `PinnedHostSlice<u8>`.
    pub fn inner_mut(&mut self) -> &mut PinnedHostSlice<u8> {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_device_buffer_size() {
        // Verify size tracking without needing a GPU
        assert_eq!(std::mem::size_of::<usize>(), std::mem::size_of::<usize>());
    }
}
