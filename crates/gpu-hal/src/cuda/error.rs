//! CUDA-specific error types wrapping cudarc driver errors.

use thiserror::Error;

/// CUDA backend error type.
#[derive(Error, Debug)]
pub enum CudaError {
    /// CUDA driver error from cudarc.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// Device initialization failed.
    #[error("CUDA device init failed (ordinal {ordinal}): {reason}")]
    DeviceInit { ordinal: usize, reason: String },

    /// No CUDA devices found.
    #[error("No CUDA devices found")]
    NoDevices,

    /// Invalid device ordinal.
    #[error("Invalid CUDA device ordinal {ordinal} (found {count} devices)")]
    InvalidOrdinal { ordinal: usize, count: i32 },

    /// PTX module load failed.
    #[error("Failed to load PTX module '{name}': {reason}")]
    ModuleLoadFailed { name: String, reason: String },

    /// Kernel function not found in module.
    #[error("Kernel function '{func}' not found in module '{module}'")]
    KernelNotFound { module: String, func: String },

    /// Memory allocation failed.
    #[error("CUDA memory allocation failed: {size} bytes")]
    AllocFailed { size: usize },

    /// Memory transfer failed.
    #[error("CUDA memory transfer failed: {reason}")]
    TransferFailed { reason: String },

    /// Kernel launch failed.
    #[error("CUDA kernel launch failed '{kernel}': {reason}")]
    KernelLaunchFailed { kernel: String, reason: String },

    /// Invalid kernel arguments.
    #[error("Invalid kernel arguments for '{kernel}': {reason}")]
    InvalidKernelArgs { kernel: String, reason: String },

    /// PTX file not found on disk.
    #[error("PTX file not found: {path}")]
    PtxFileNotFound { path: String },
}

impl From<CudaError> for ms_common::GpuError {
    fn from(err: CudaError) -> Self {
        match err {
            CudaError::NoDevices => ms_common::GpuError::NoBackend,
            CudaError::DeviceInit { reason, .. } => ms_common::GpuError::DeviceInit(reason),
            CudaError::InvalidOrdinal { ordinal, count } => ms_common::GpuError::DeviceInit(
                format!("Invalid ordinal {ordinal}, only {count} devices available"),
            ),
            CudaError::AllocFailed { size } => ms_common::GpuError::AllocFailed { size },
            CudaError::TransferFailed { reason } => ms_common::GpuError::TransferFailed(reason),
            CudaError::ModuleLoadFailed { name, reason } => ms_common::GpuError::KernelFailed {
                kernel: name,
                reason,
            },
            CudaError::KernelNotFound { module, func } => ms_common::GpuError::KernelFailed {
                kernel: func,
                reason: format!("not found in module '{module}'"),
            },
            CudaError::Driver(e) => {
                ms_common::GpuError::DeviceInit(format!("CUDA driver error: {e}"))
            }
            CudaError::KernelLaunchFailed { kernel, reason } => {
                ms_common::GpuError::KernelFailed { kernel, reason }
            }
            CudaError::InvalidKernelArgs { kernel, reason } => {
                ms_common::GpuError::KernelFailed { kernel, reason }
            }
            CudaError::PtxFileNotFound { path } => ms_common::GpuError::KernelFailed {
                kernel: path,
                reason: "PTX file not found".to_string(),
            },
        }
    }
}
