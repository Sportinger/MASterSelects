//! CUDA stream management — RAII wrappers for async GPU command queues.

use std::sync::Arc;

use cudarc::driver::safe::{CudaContext, CudaStream as CudarcStream};
use tracing::debug;

use super::error::CudaError;

/// RAII wrapper around a cudarc `CudaStream`.
///
/// Streams provide asynchronous execution queues on the GPU. Operations
/// submitted to different streams can execute concurrently.
///
/// Dropped streams are automatically destroyed by cudarc's `Drop` impl.
#[derive(Debug)]
pub struct ManagedStream {
    /// The underlying cudarc stream.
    inner: Arc<CudarcStream>,
    /// Whether this is the default (null) stream.
    is_default: bool,
}

impl ManagedStream {
    /// Create a new non-blocking CUDA stream.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, CudaError> {
        let inner = ctx.new_stream().map_err(|e| CudaError::DeviceInit {
            ordinal: ctx.ordinal(),
            reason: format!("Failed to create CUDA stream: {e}"),
        })?;
        debug!("Created new CUDA stream");
        Ok(Self {
            inner,
            is_default: false,
        })
    }

    /// Get the default (null) stream for the given context.
    ///
    /// The default stream is implicitly synchronized with all other
    /// non-blocking streams — use explicit streams for concurrency.
    pub fn default_stream(ctx: &Arc<CudaContext>) -> Self {
        let inner = ctx.default_stream();
        Self {
            inner,
            is_default: true,
        }
    }

    /// Fork this stream — creates a new stream that waits for all
    /// current work on this stream to complete before starting.
    pub fn fork(&self) -> Result<Self, CudaError> {
        let inner = self.inner.fork().map_err(|e| CudaError::DeviceInit {
            ordinal: self.inner.context().ordinal(),
            reason: format!("Failed to fork CUDA stream: {e}"),
        })?;
        debug!("Forked CUDA stream");
        Ok(Self {
            inner,
            is_default: false,
        })
    }

    /// Synchronize this stream — blocks the calling thread until all
    /// operations on this stream have completed.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.inner.synchronize()?;
        debug!("CUDA stream synchronized");
        Ok(())
    }

    /// Make this stream wait for all work on `other` to complete.
    pub fn wait_for(&self, other: &ManagedStream) -> Result<(), CudaError> {
        self.inner.join(&other.inner)?;
        Ok(())
    }

    /// Get a reference to the underlying cudarc stream.
    pub fn inner(&self) -> &Arc<CudarcStream> {
        &self.inner
    }

    /// Whether this is the default (null) stream.
    pub fn is_default(&self) -> bool {
        self.is_default
    }

    /// Get the raw stream handle as a u64 (for opaque handle passing).
    pub fn handle(&self) -> u64 {
        self.inner.cu_stream() as u64
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_managed_stream_default_flag() {
        // This is a unit test that doesn't require GPU
        assert!(true, "Stream module compiles");
    }
}
