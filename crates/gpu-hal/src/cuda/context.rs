//! CUDA context management — wraps cudarc `CudaContext` with device selection
//! and info queries.

use std::sync::Arc;

use cudarc::driver::safe::CudaContext;
use cudarc::driver::sys::CUdevice_attribute_enum;
use tracing::{debug, info};

use super::error::CudaError;

/// Information about a CUDA device discovered at init time.
#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    /// Device ordinal (0-based index).
    pub ordinal: usize,
    /// Human-readable device name (e.g. "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Total VRAM in bytes.
    pub vram_total: u64,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// CUDA driver version (major * 1000 + minor * 10).
    pub driver_version: i32,
}

/// Managed CUDA context with device info cached at init time.
#[derive(Debug)]
pub struct CudaContextWrapper {
    /// The underlying cudarc context.
    ctx: Arc<CudaContext>,
    /// Cached device info.
    info: CudaDeviceInfo,
}

impl CudaContextWrapper {
    /// Create a new CUDA context on the given device ordinal.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let device_count = CudaContext::device_count().map_err(|e| CudaError::DeviceInit {
            ordinal,
            reason: format!("Failed to get device count: {e}"),
        })?;

        if device_count == 0 {
            return Err(CudaError::NoDevices);
        }

        if ordinal >= device_count as usize {
            return Err(CudaError::InvalidOrdinal {
                ordinal,
                count: device_count,
            });
        }

        info!(ordinal, "Initializing CUDA context");

        let ctx = CudaContext::new(ordinal).map_err(|e| CudaError::DeviceInit {
            ordinal,
            reason: format!("{e}"),
        })?;

        let name = ctx.name().map_err(|e| CudaError::DeviceInit {
            ordinal,
            reason: format!("Failed to get device name: {e}"),
        })?;

        let vram_total = {
            // SAFETY: cu_device was obtained from CudaContext::new which validates the device.
            let total = unsafe { cudarc::driver::result::device::total_mem(ctx.cu_device()) }
                .map_err(|e| CudaError::DeviceInit {
                    ordinal,
                    reason: format!("Failed to get VRAM: {e}"),
                })?;
            total as u64
        };

        let cc_major = ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| CudaError::DeviceInit {
                ordinal,
                reason: format!("Failed to get compute capability major: {e}"),
            })?;

        let cc_minor = ctx
            .attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| CudaError::DeviceInit {
                ordinal,
                reason: format!("Failed to get compute capability minor: {e}"),
            })?;

        let driver_version = Self::query_driver_version()?;

        let info = CudaDeviceInfo {
            ordinal,
            name: name.clone(),
            vram_total,
            compute_capability: (cc_major as u32, cc_minor as u32),
            driver_version,
        };

        info!(
            device = %name,
            ordinal,
            vram_mb = vram_total / (1024 * 1024),
            cc = format!("{}.{}", cc_major, cc_minor),
            "CUDA device initialized"
        );

        Ok(Self { ctx, info })
    }

    /// Get the underlying cudarc context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Get cached device info.
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.info
    }

    /// Get the device name.
    pub fn device_name(&self) -> &str {
        &self.info.name
    }

    /// Get total VRAM in bytes.
    pub fn vram_total(&self) -> u64 {
        self.info.vram_total
    }

    /// Get the compute capability as (major, minor).
    pub fn compute_capability(&self) -> (u32, u32) {
        self.info.compute_capability
    }

    /// Query current free and total VRAM. Requires context to be bound.
    pub fn vram_info(&self) -> Result<(u64, u64), CudaError> {
        self.ctx.bind_to_thread()?;
        let (free, total) =
            cudarc::driver::result::mem_get_info().map_err(|e| CudaError::DeviceInit {
                ordinal: self.info.ordinal,
                reason: format!("Failed to query VRAM info: {e}"),
            })?;
        Ok((free as u64, total as u64))
    }

    /// Synchronize the entire context (waits for all pending work).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.synchronize()?;
        debug!(device = %self.info.name, "CUDA context synchronized");
        Ok(())
    }

    /// Query the CUDA driver version. Returns the raw version integer
    /// (e.g. 12080 for CUDA 12.8).
    fn query_driver_version() -> Result<i32, CudaError> {
        let mut version: i32 = 0;
        // SAFETY: cuDriverGetVersion writes to a valid pointer and is safe to call
        // after cuInit (which CudaContext::new already called).
        let result = unsafe {
            cudarc::driver::sys::cuDriverGetVersion(&mut version as *mut std::ffi::c_int)
        };
        result.result()?;
        Ok(version)
    }

    /// Format the driver version as a human-readable string (e.g. "12.8").
    pub fn driver_version_string(&self) -> String {
        let major = self.info.driver_version / 1000;
        let minor = (self.info.driver_version % 1000) / 10;
        format!("{major}.{minor}")
    }
}

/// Enumerate all available CUDA devices and return their info.
pub fn enumerate_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
    let count = CudaContext::device_count().map_err(|e| CudaError::DeviceInit {
        ordinal: 0,
        reason: format!("Failed to get device count: {e}"),
    })?;

    if count == 0 {
        return Err(CudaError::NoDevices);
    }

    let mut devices = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let wrapper = CudaContextWrapper::new(i)?;
        devices.push(wrapper.device_info().clone());
    }

    info!(count = devices.len(), "Enumerated CUDA devices");
    Ok(devices)
}

/// Convert our device info to the common `GpuDeviceInfo` type.
impl From<&CudaDeviceInfo> for ms_common::GpuDeviceInfo {
    fn from(info: &CudaDeviceInfo) -> Self {
        let major = info.driver_version / 1000;
        let minor = (info.driver_version % 1000) / 10;
        ms_common::GpuDeviceInfo {
            name: info.name.clone(),
            vendor: ms_common::GpuVendor::Nvidia,
            vram_total: info.vram_total,
            compute_capability: Some(info.compute_capability),
            api_version: format!("CUDA {major}.{minor}"),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_driver_version_format() {
        // Simulate formatting for a known version
        let major = 12080 / 1000;
        let minor = (12080 % 1000) / 10;
        assert_eq!(format!("{major}.{minor}"), "12.8");
    }
}
