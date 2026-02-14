//! NVENC input/output buffer pool management.
//!
//! Manages the lifecycle of NVENC buffers:
//! - **Input buffers**: External CUDA device pointers registered with NVENC.
//! - **Output buffers**: Bitstream buffers for reading encoded data.
//!
//! The buffer pool uses a ring-buffer approach: a fixed number of input/output
//! buffer pairs are created at initialization, and frames cycle through them.
//! This avoids creating/destroying buffers per-frame.
//!
//! # Buffer Lifecycle
//!
//! ```text
//! 1. Register external CUDA device ptr as NVENC input resource
//! 2. Map the registered resource -> get mapped handle
//! 3. Encode: pass mapped handle + output bitstream buffer to NvEncEncodePicture
//! 4. Lock output bitstream -> read encoded data
//! 5. Unlock output bitstream
//! 6. Unmap input resource
//! 7. (Repeat for next frame)
//! ```

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Arc;

use tracing::{debug, warn};

use crate::error::BufferError;

use super::ffi::{
    check_nvenc_status, NvEncBufferFormat, NvEncCreateBitstreamBuffer, NvencLibrary,
    NvEncMapInputResource, NvEncRegisterResource, NvEncInputResourceType,
    NV_ENC_SUCCESS,
};

// ---------------------------------------------------------------------------
// Output buffer (bitstream)
// ---------------------------------------------------------------------------

/// An NVENC output bitstream buffer.
///
/// Created once during pool initialization and reused across frames.
#[derive(Debug)]
pub struct OutputBuffer {
    /// NVENC bitstream buffer handle.
    pub handle: *mut c_void,
    /// Whether this buffer is currently in use (encode submitted but not read).
    pub in_use: bool,
}

// SAFETY: OutputBuffer contains an opaque NVENC handle that is only accessed
// through the NVENC API. The handle is valid for the encoder session lifetime.
unsafe impl Send for OutputBuffer {}

// ---------------------------------------------------------------------------
// Registered input resource
// ---------------------------------------------------------------------------

/// A registered external resource (CUDA device pointer registered with NVENC).
#[derive(Debug)]
pub struct RegisteredResource {
    /// NVENC registered resource handle (from NvEncRegisterResource).
    pub registered_handle: *mut c_void,
    /// The original device pointer that was registered.
    pub device_ptr: u64,
    /// Width of the registered resource.
    pub width: u32,
    /// Height of the registered resource.
    pub height: u32,
    /// Pitch of the registered resource.
    pub pitch: u32,
    /// Buffer format.
    pub format: NvEncBufferFormat,
}

// SAFETY: RegisteredResource contains opaque NVENC handles and a GPU device
// pointer. These are only accessed through the NVENC API functions.
unsafe impl Send for RegisteredResource {}

// ---------------------------------------------------------------------------
// Mapped input resource
// ---------------------------------------------------------------------------

/// A mapped input resource ready to be used as encoder input.
///
/// The mapping must be released (unmapped) after the encode completes.
#[derive(Debug)]
pub struct MappedInput {
    /// Mapped resource handle (used as input_buffer in PicParams).
    pub mapped_handle: *mut c_void,
    /// The registered resource this mapping refers to.
    pub registered_handle: *mut c_void,
    /// Buffer format of the mapped resource.
    pub format: NvEncBufferFormat,
}

// SAFETY: MappedInput contains opaque NVENC handles.
unsafe impl Send for MappedInput {}

// ---------------------------------------------------------------------------
// Buffer pool
// ---------------------------------------------------------------------------

/// Pool of NVENC output bitstream buffers.
///
/// Manages a fixed number of output buffers in a ring-buffer fashion.
/// Each frame uses one output buffer; when all buffers are in use, the
/// caller must wait for a previous encode to complete.
pub struct BufferPool {
    /// Output bitstream buffers.
    output_buffers: Vec<OutputBuffer>,
    /// Indices of available (not in-use) output buffers.
    available: VecDeque<usize>,
    /// NVENC encoder handle (for API calls).
    encoder: *mut c_void,
    /// Reference to the loaded NVENC library.
    lib: Arc<NvencLibrary>,
}

// SAFETY: BufferPool contains raw pointers to NVENC handles. The encoder
// handle is valid for the encoder session lifetime. The BufferPool is
// only used from the encode thread.
unsafe impl Send for BufferPool {}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("total_buffers", &self.output_buffers.len())
            .field("available", &self.available.len())
            .finish()
    }
}

impl BufferPool {
    /// Create a new buffer pool with the specified number of output buffers.
    ///
    /// # Arguments
    /// * `encoder` -- Valid NVENC encoder handle.
    /// * `lib` -- Loaded NVENC library.
    /// * `count` -- Number of output bitstream buffers to create.
    ///
    /// # Errors
    /// Returns `BufferError::OutputCreationFailed` if buffer creation fails.
    ///
    /// # Safety
    /// `encoder` must be a valid NVENC encoder handle obtained from
    /// `nvEncOpenEncodeSessionEx`. The handle must remain valid for the
    /// lifetime of this `BufferPool`.
    pub unsafe fn new(
        encoder: *mut c_void,
        lib: Arc<NvencLibrary>,
        count: usize,
    ) -> Result<Self, BufferError> {
        let mut output_buffers = Vec::with_capacity(count);
        let mut available = VecDeque::with_capacity(count);

        for i in 0..count {
            let mut params = NvEncCreateBitstreamBuffer::default();

            // SAFETY: encoder is a valid NVENC handle (guaranteed by caller).
            // params is properly initialized with the correct version. NVENC
            // writes the buffer handle to params.bitstream_buffer.
            let status = (lib.api.nvEncCreateBitstreamBuffer)(encoder, &mut params);

            check_nvenc_status(status, "nvEncCreateBitstreamBuffer")
                .map_err(BufferError::OutputCreationFailed)?;

            output_buffers.push(OutputBuffer {
                handle: params.bitstream_buffer,
                in_use: false,
            });
            available.push_back(i);

            debug!(index = i, "Created NVENC output bitstream buffer");
        }

        Ok(Self {
            output_buffers,
            available,
            encoder,
            lib,
        })
    }

    /// Acquire an available output buffer for encoding.
    ///
    /// Returns the buffer index and a pointer to the buffer handle.
    /// Returns `BufferError::PoolExhausted` if all buffers are in use.
    pub fn acquire_output(&mut self) -> Result<(usize, *mut c_void), BufferError> {
        let idx = self.available.pop_front().ok_or(BufferError::PoolExhausted {
            count: self.output_buffers.len(),
        })?;

        self.output_buffers[idx].in_use = true;
        Ok((idx, self.output_buffers[idx].handle))
    }

    /// Release an output buffer back to the pool after reading the bitstream.
    pub fn release_output(&mut self, index: usize) {
        if index < self.output_buffers.len() {
            self.output_buffers[index].in_use = false;
            self.available.push_back(index);
        }
    }

    /// Get the total number of output buffers in the pool.
    pub fn total_count(&self) -> usize {
        self.output_buffers.len()
    }

    /// Get the number of currently available output buffers.
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Check if all output buffers are currently in use.
    pub fn is_full(&self) -> bool {
        self.available.is_empty()
    }

    /// Register an external CUDA device pointer as an NVENC input resource.
    ///
    /// # Arguments
    /// * `device_ptr` -- CUDA device pointer to register.
    /// * `width` -- Width of the frame.
    /// * `height` -- Height of the frame.
    /// * `pitch` -- Row pitch in bytes.
    /// * `format` -- Buffer format.
    ///
    /// # Errors
    /// Returns `BufferError::RegisterFailed` if registration fails.
    pub fn register_input(
        &self,
        device_ptr: u64,
        width: u32,
        height: u32,
        pitch: u32,
        format: NvEncBufferFormat,
    ) -> Result<RegisteredResource, BufferError> {
        let mut params = NvEncRegisterResource {
            resource_type: NvEncInputResourceType::CudaDeviceptr,
            width,
            height,
            pitch,
            resource_to_register: device_ptr as *mut c_void,
            buffer_format: format,
            ..NvEncRegisterResource::default()
        };

        // SAFETY: encoder is a valid NVENC handle. params is properly initialized.
        // device_ptr must be a valid CUDA device pointer. NVENC writes the
        // registered resource handle to params.registered_resource.
        let status = unsafe { (self.lib.api.nvEncRegisterResource)(self.encoder, &mut params) };

        check_nvenc_status(status, "nvEncRegisterResource")
            .map_err(BufferError::RegisterFailed)?;

        Ok(RegisteredResource {
            registered_handle: params.registered_resource,
            device_ptr,
            width,
            height,
            pitch,
            format,
        })
    }

    /// Unregister a previously registered input resource.
    ///
    /// # Safety
    /// The resource must not be currently mapped.
    pub fn unregister_input(&self, resource: &RegisteredResource) -> Result<(), BufferError> {
        // SAFETY: encoder is valid and the registered_handle was obtained from
        // a successful nvEncRegisterResource call. The resource must not be
        // currently mapped.
        let status = unsafe {
            (self.lib.api.nvEncUnregisterResource)(self.encoder, resource.registered_handle)
        };

        check_nvenc_status(status, "nvEncUnregisterResource")
            .map_err(BufferError::RegisterFailed)?;

        Ok(())
    }

    /// Map a registered resource for use as encoder input.
    ///
    /// The mapping must be released via `unmap_input` after the encode completes.
    pub fn map_input(
        &self,
        resource: &RegisteredResource,
    ) -> Result<MappedInput, BufferError> {
        let mut params = NvEncMapInputResource {
            registered_resource: resource.registered_handle,
            ..NvEncMapInputResource::default()
        };

        // SAFETY: encoder is valid and registered_resource is a valid handle
        // from nvEncRegisterResource. NVENC writes the mapped resource handle
        // and format to the output fields.
        let status = unsafe { (self.lib.api.nvEncMapInputResource)(self.encoder, &mut params) };

        check_nvenc_status(status, "nvEncMapInputResource")
            .map_err(BufferError::MapFailed)?;

        Ok(MappedInput {
            mapped_handle: params.mapped_resource,
            registered_handle: resource.registered_handle,
            format: params.mapped_buffer_fmt,
        })
    }

    /// Unmap a previously mapped input resource.
    pub fn unmap_input(&self, mapped: &MappedInput) -> Result<(), BufferError> {
        // SAFETY: encoder is valid and mapped_handle was obtained from
        // a successful nvEncMapInputResource call.
        let status = unsafe {
            (self.lib.api.nvEncUnmapInputResource)(self.encoder, mapped.mapped_handle)
        };

        check_nvenc_status(status, "nvEncUnmapInputResource")
            .map_err(BufferError::UnmapFailed)?;

        Ok(())
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        // Destroy all output bitstream buffers.
        for (i, buf) in self.output_buffers.iter().enumerate() {
            if !buf.handle.is_null() {
                if buf.in_use {
                    warn!(
                        index = i,
                        "Destroying in-use output buffer -- encode may not have completed"
                    );
                }

                // SAFETY: encoder and buffer handles are valid. Destroying a
                // buffer that is in-use is allowed by the NVENC API but may
                // produce undefined output.
                let status = unsafe {
                    (self.lib.api.nvEncDestroyBitstreamBuffer)(self.encoder, buf.handle)
                };
                if status != NV_ENC_SUCCESS {
                    warn!(
                        index = i,
                        status, "Failed to destroy NVENC output buffer"
                    );
                }
            }
        }
        debug!(
            count = self.output_buffers.len(),
            "NVENC buffer pool destroyed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_buffer_default_state() {
        let buf = OutputBuffer {
            handle: std::ptr::null_mut(),
            in_use: false,
        };
        assert!(!buf.in_use);
        assert!(buf.handle.is_null());
    }

    #[test]
    fn registered_resource_fields() {
        let res = RegisteredResource {
            registered_handle: std::ptr::null_mut(),
            device_ptr: 0x1000_0000,
            width: 1920,
            height: 1080,
            pitch: 2048,
            format: NvEncBufferFormat::Nv12,
        };
        assert_eq!(res.device_ptr, 0x1000_0000);
        assert_eq!(res.width, 1920);
        assert_eq!(res.height, 1080);
        assert_eq!(res.pitch, 2048);
        assert_eq!(res.format, NvEncBufferFormat::Nv12);
    }

    #[test]
    fn mapped_input_fields() {
        let mapped = MappedInput {
            mapped_handle: std::ptr::null_mut(),
            registered_handle: std::ptr::null_mut(),
            format: NvEncBufferFormat::Nv12,
        };
        assert!(mapped.mapped_handle.is_null());
        assert_eq!(mapped.format, NvEncBufferFormat::Nv12);
    }
}
