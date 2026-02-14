//! PTX kernel management — loading, caching, dispatching, and argument marshaling.
//!
//! This module provides the complete CUDA kernel lifecycle:
//! 1. **Loading**: PTX modules from source strings, raw bytes, or disk files.
//! 2. **Caching**: Module and function handle caching for efficient reuse.
//! 3. **Argument Marshaling**: Type-safe conversion from `KernelArgs` to raw
//!    pointers suitable for `cuLaunchKernel`.
//! 4. **Dispatch**: Launching kernels with grid/block dimensions on a CUDA stream.
//!
//! # Architecture
//!
//! ```text
//! KernelId (from ms-common)
//!   |
//!   v
//! KernelManager::ensure_kernel_loaded()   -- loads PTX if not cached
//!   |
//!   v
//! KernelManager::get_kernel_function()    -- resolves CUfunction handle
//!   |
//!   v
//! KernelManager::launch()                 -- marshals KernelArgs, dispatches
//!   |
//!   v
//! cuLaunchKernel (raw CUDA driver API)    -- actual GPU dispatch
//! ```
//!
//! # Raw CUDA API Usage
//!
//! This module uses the raw CUDA driver API (`cudarc::driver::sys`) directly
//! for module loading and kernel launching. This is necessary because cudarc's
//! safe wrappers keep `CUfunction` and `CUmodule` handles as `pub(crate)`,
//! preventing external crates from building custom argument arrays for
//! `cuLaunchKernel`. We still use cudarc's `CudaContext` for device init
//! and thread binding.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::safe::CudaContext;
use cudarc::driver::sys;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use ms_common::kernel::{KernelArg, KernelArgs, KernelId};

use super::error::CudaError;

// ---------------------------------------------------------------------------
// Raw module handle (RAII)
// ---------------------------------------------------------------------------

/// RAII wrapper around a raw `CUmodule` handle.
///
/// Automatically calls `cuModuleUnload` on drop. The CUDA context must still
/// be valid when this is dropped.
#[derive(Debug)]
struct RawModule {
    cu_module: sys::CUmodule,
    ctx: Arc<CudaContext>,
}

impl Drop for RawModule {
    fn drop(&mut self) {
        // Best-effort unload; ignore errors during drop.
        let _ = self.ctx.bind_to_thread();
        // SAFETY: cu_module was obtained from cuModuleLoadData and has not been
        // unloaded yet. The context is still valid because we hold an Arc to it.
        let _ = unsafe { sys::cuModuleUnload(self.cu_module) };
    }
}

// SAFETY: CUmodule is a pointer that can be used from any thread as long as
// the owning CUDA context is bound. We ensure context binding before use.
unsafe impl Send for RawModule {}
unsafe impl Sync for RawModule {}

/// Newtype wrapper around `CUfunction` to implement `Send` and `Sync`.
///
/// CUDA function handles are valid across threads as long as the owning
/// module and context are alive. We ensure this through the `KernelManager`
/// which holds `Arc<RawModule>` (and thus `Arc<CudaContext>`).
#[derive(Debug, Clone, Copy)]
struct SendCUfunction(sys::CUfunction);

// SAFETY: CUfunction is a pointer into the CUDA driver's internal state.
// It is safe to send/share across threads as long as:
// 1. The owning CUmodule is not unloaded (guaranteed by Arc<RawModule>).
// 2. The CUDA context is bound before use (guaranteed by bind_to_thread calls).
unsafe impl Send for SendCUfunction {}
unsafe impl Sync for SendCUfunction {}

// ---------------------------------------------------------------------------
// KernelManager
// ---------------------------------------------------------------------------

/// Manages PTX kernel modules, cached function handles, and kernel dispatch.
///
/// Provides loading of PTX from embedded bytes, source strings, or disk files,
/// caches compiled modules and their function entry points for efficient reuse,
/// and marshals `KernelArgs` into the format required by `cuLaunchKernel`.
///
/// # Thread Safety
///
/// `KernelManager` is `Send + Sync`. Internal caches use `parking_lot::RwLock`
/// for concurrent read access with exclusive write access during loading.
///
/// # Raw API
///
/// This manager uses the raw CUDA driver API (`cuModuleLoadData`,
/// `cuModuleGetFunction`, `cuLaunchKernel`) directly rather than cudarc's
/// safe wrappers. This is because cudarc 0.16 keeps `CUfunction` and
/// `CUmodule` handles private (`pub(crate)`), preventing external crates
/// from building custom kernel argument arrays.
#[derive(Debug)]
pub struct KernelManager {
    /// The CUDA context used for loading modules.
    ctx: Arc<CudaContext>,
    /// Loaded raw modules, keyed by module name.
    modules: RwLock<HashMap<String, Arc<RawModule>>>,
    /// Cached raw function handles, keyed by (module_name, function_name).
    functions: RwLock<HashMap<(String, String), SendCUfunction>>,
    /// Optional base directory for loading PTX files from disk.
    ptx_search_dir: Option<PathBuf>,
}

impl KernelManager {
    /// Create a new kernel manager for the given CUDA context.
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            modules: RwLock::new(HashMap::new()),
            functions: RwLock::new(HashMap::new()),
            ptx_search_dir: None,
        }
    }

    /// Create a kernel manager with a base directory for PTX file lookup.
    ///
    /// When `ensure_kernel_loaded` is called and the module is not yet loaded,
    /// the manager will look for `<ptx_dir>/<module_name>` on disk.
    pub fn with_ptx_dir(ctx: Arc<CudaContext>, ptx_dir: impl Into<PathBuf>) -> Self {
        Self {
            ctx,
            modules: RwLock::new(HashMap::new()),
            functions: RwLock::new(HashMap::new()),
            ptx_search_dir: Some(ptx_dir.into()),
        }
    }

    /// Set the PTX search directory after construction.
    pub fn set_ptx_dir(&mut self, dir: impl Into<PathBuf>) {
        self.ptx_search_dir = Some(dir.into());
    }

    /// Get the current PTX search directory, if set.
    pub fn ptx_dir(&self) -> Option<&Path> {
        self.ptx_search_dir.as_deref()
    }

    // -- PTX Loading --

    /// Load a PTX module from a source string.
    ///
    /// The module is cached under the given `name` for later function lookups.
    pub fn load_ptx_source(&self, name: &str, ptx_source: &str) -> Result<(), CudaError> {
        self.load_ptx_inner(name, ptx_source.as_bytes())
    }

    /// Load a PTX module from raw bytes (e.g. embedded via `include_str!` or `include_bytes!`).
    ///
    /// The bytes should contain valid PTX text (null-terminated or not).
    pub fn load_ptx_bytes(&self, name: &str, ptx_bytes: &[u8]) -> Result<(), CudaError> {
        // Validate UTF-8 (PTX is text)
        let _ptx_str = std::str::from_utf8(ptx_bytes).map_err(|e| CudaError::ModuleLoadFailed {
            name: name.to_string(),
            reason: format!("Invalid UTF-8 in PTX data: {e}"),
        })?;
        self.load_ptx_inner(name, ptx_bytes)
    }

    /// Load a PTX module from a file on disk.
    ///
    /// The file path should point to a valid `.ptx` file produced by `nvcc --ptx`.
    pub fn load_ptx_file(&self, name: &str, path: &Path) -> Result<(), CudaError> {
        let ptx_source = std::fs::read_to_string(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CudaError::PtxFileNotFound {
                    path: path.display().to_string(),
                }
            } else {
                CudaError::ModuleLoadFailed {
                    name: name.to_string(),
                    reason: format!("Failed to read PTX file '{}': {e}", path.display()),
                }
            }
        })?;

        info!(
            module = name,
            path = %path.display(),
            "Loading PTX module from disk"
        );
        self.load_ptx_inner(name, ptx_source.as_bytes())
    }

    /// Internal method to load a PTX module via the raw CUDA driver API.
    fn load_ptx_inner(&self, name: &str, ptx_data: &[u8]) -> Result<(), CudaError> {
        // Ensure null-terminated for cuModuleLoadData
        let mut null_terminated = Vec::with_capacity(ptx_data.len() + 1);
        null_terminated.extend_from_slice(ptx_data);
        if !ptx_data.ends_with(&[0]) {
            null_terminated.push(0);
        }

        // Bind CUDA context to current thread
        self.ctx
            .bind_to_thread()
            .map_err(|e| CudaError::ModuleLoadFailed {
                name: name.to_string(),
                reason: format!("Failed to bind CUDA context: {e}"),
            })?;

        // SAFETY:
        // 1. null_terminated contains valid, null-terminated PTX text.
        // 2. The CUDA context is bound to this thread.
        // 3. cu_module is initialized by cuModuleLoadData on success.
        let cu_module = unsafe {
            let mut module = MaybeUninit::uninit();
            let result = sys::cuModuleLoadData(
                module.as_mut_ptr(),
                null_terminated.as_ptr() as *const c_void,
            );
            result.result().map_err(|e| CudaError::ModuleLoadFailed {
                name: name.to_string(),
                reason: format!("cuModuleLoadData failed: {e}"),
            })?;
            module.assume_init()
        };

        let raw_module = Arc::new(RawModule {
            cu_module,
            ctx: self.ctx.clone(),
        });

        info!(module = name, "Loaded CUDA PTX module (raw API)");
        self.modules.write().insert(name.to_string(), raw_module);
        Ok(())
    }

    // -- Function Lookup --

    /// Get a raw function handle from a loaded module.
    ///
    /// Results are cached for subsequent calls with the same module/function name.
    pub fn get_function(
        &self,
        module_name: &str,
        function_name: &str,
    ) -> Result<sys::CUfunction, CudaError> {
        let cache_key = (module_name.to_string(), function_name.to_string());

        // Check cache first
        if let Some(func) = self.functions.read().get(&cache_key) {
            return Ok(func.0);
        }

        // Load from module
        let modules = self.modules.read();
        let raw_module = modules
            .get(module_name)
            .ok_or_else(|| CudaError::ModuleLoadFailed {
                name: module_name.to_string(),
                reason: "Module not loaded".to_string(),
            })?;

        let fn_name_c = CString::new(function_name).map_err(|_| CudaError::KernelNotFound {
            module: module_name.to_string(),
            func: function_name.to_string(),
        })?;

        // Bind context for the raw API call
        self.ctx
            .bind_to_thread()
            .map_err(|e| CudaError::KernelNotFound {
                module: module_name.to_string(),
                func: format!("{function_name} (context bind failed: {e})"),
            })?;

        // SAFETY:
        // 1. raw_module.cu_module is a valid module handle from cuModuleLoadData.
        // 2. fn_name_c is a valid null-terminated C string.
        // 3. The CUDA context is bound to this thread.
        let func = unsafe {
            let mut cu_func = MaybeUninit::uninit();
            let result = sys::cuModuleGetFunction(
                cu_func.as_mut_ptr(),
                raw_module.cu_module,
                fn_name_c.as_ptr(),
            );
            result.result().map_err(|_e| CudaError::KernelNotFound {
                module: module_name.to_string(),
                func: function_name.to_string(),
            })?;
            cu_func.assume_init()
        };

        debug!(
            module = module_name,
            function = function_name,
            "Cached CUDA kernel function (raw handle)"
        );
        self.functions
            .write()
            .insert(cache_key, SendCUfunction(func));
        Ok(func)
    }

    /// Get a function handle using a `KernelId`.
    ///
    /// Uses the kernel's module name and entry point from the common types.
    pub fn get_kernel_function(&self, kernel_id: &KernelId) -> Result<sys::CUfunction, CudaError> {
        let module_name = kernel_id.cuda_module_name();
        let entry_point = kernel_id.entry_point();
        self.get_function(&module_name, entry_point)
    }

    // -- Auto-Loading --

    /// Ensure the PTX module for a given `KernelId` is loaded.
    ///
    /// If the module is already loaded, this is a no-op. Otherwise, attempts
    /// to load the PTX file from the configured `ptx_search_dir`.
    ///
    /// Returns an error if no `ptx_search_dir` is set and the module is not
    /// already loaded, or if the PTX file cannot be found or loaded.
    pub fn ensure_kernel_loaded(&self, kernel_id: &KernelId) -> Result<(), CudaError> {
        let module_name = kernel_id.cuda_module_name();

        if self.is_module_loaded(&module_name) {
            return Ok(());
        }

        let ptx_dir = self
            .ptx_search_dir
            .as_ref()
            .ok_or_else(|| CudaError::ModuleLoadFailed {
                name: module_name.clone(),
                reason: "Module not loaded and no PTX search directory configured".to_string(),
            })?;

        let ptx_path = ptx_dir.join(&module_name);
        self.load_ptx_file(&module_name, &ptx_path)
    }

    /// Load all built-in kernel PTX modules from the configured search directory.
    ///
    /// Attempts to load PTX for all known `KernelId` variants (excluding
    /// `Effect` variants which are loaded on demand). Logs warnings for any
    /// modules that fail to load but does not treat individual failures as fatal.
    ///
    /// Returns the number of successfully loaded modules.
    pub fn preload_builtin_kernels(&self) -> usize {
        let builtins = [
            KernelId::Nv12ToRgba,
            KernelId::AlphaBlend,
            KernelId::Transform,
            KernelId::Mask,
            KernelId::Transition,
        ];

        let mut loaded = 0;
        for kernel_id in &builtins {
            match self.ensure_kernel_loaded(kernel_id) {
                Ok(()) => loaded += 1,
                Err(e) => {
                    warn!(
                        kernel = kernel_id.entry_point(),
                        error = %e,
                        "Failed to preload built-in kernel (may not be compiled yet)"
                    );
                }
            }
        }

        info!(loaded, total = builtins.len(), "Preloaded built-in kernels");
        loaded
    }

    // -- Module Management --

    /// Check if a module with the given name is loaded.
    pub fn is_module_loaded(&self, name: &str) -> bool {
        self.modules.read().contains_key(name)
    }

    /// Unload a module (and remove its cached functions).
    pub fn unload_module(&self, name: &str) {
        self.modules.write().remove(name);
        self.functions
            .write()
            .retain(|(mod_name, _), _| mod_name != name);
        debug!(module = name, "Unloaded CUDA PTX module");
    }

    /// Get the number of loaded modules.
    pub fn module_count(&self) -> usize {
        self.modules.read().len()
    }

    /// Get the number of cached functions.
    pub fn function_count(&self) -> usize {
        self.functions.read().len()
    }

    /// Get the CUDA context this manager is bound to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    // -- Kernel Dispatch --

    /// Dispatch a kernel identified by `KernelId` with the given arguments.
    ///
    /// This is the primary dispatch method. It:
    /// 1. Ensures the kernel's PTX module is loaded (auto-loading from disk if configured).
    /// 2. Looks up the kernel function handle (cached after first use).
    /// 3. Marshals `KernelArgs` into the format required by `cuLaunchKernel`.
    /// 4. Launches the kernel asynchronously on the given stream.
    ///
    /// # Arguments
    ///
    /// * `kernel_id` - Which kernel to dispatch.
    /// * `grid` - Grid dimensions `[x, y, z]` (number of blocks).
    /// * `block` - Block dimensions `[x, y, z]` (threads per block).
    /// * `args` - Kernel arguments in order matching the kernel signature.
    /// * `stream` - The CUDA stream to launch on (raw `CUstream` handle).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `stream` is a valid `CUstream` handle (or null for the default stream).
    /// - `args` match the kernel's parameter list in number, order, and type.
    /// - Device pointers in `args` point to valid, allocated GPU memory.
    /// - GPU memory referenced by `args` will not be freed before the kernel completes.
    pub unsafe fn launch(
        &self,
        kernel_id: &KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        args: &KernelArgs,
        stream: sys::CUstream,
    ) -> Result<(), CudaError> {
        // SAFETY: Caller guarantees stream validity and argument correctness.
        self.launch_with_shared_mem(kernel_id, grid, block, 0, args, stream)
    }

    /// Dispatch a kernel with explicit shared memory allocation.
    ///
    /// Like [`launch`](Self::launch), but allows specifying the amount of
    /// dynamic shared memory per thread block.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `stream` is a valid `CUstream` handle (or null for the default stream).
    /// - `args` match the kernel's parameter list in number, order, and type.
    /// - Device pointers in `args` point to valid, allocated GPU memory.
    /// - GPU memory referenced by `args` will not be freed before the kernel completes.
    pub unsafe fn launch_with_shared_mem(
        &self,
        kernel_id: &KernelId,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        args: &KernelArgs,
        stream: sys::CUstream,
    ) -> Result<(), CudaError> {
        // Step 1: Ensure the module is loaded
        self.ensure_kernel_loaded(kernel_id)?;

        // Step 2: Get the raw function handle
        let func = self.get_kernel_function(kernel_id)?;

        let entry_point = kernel_id.entry_point();

        debug!(
            kernel = entry_point,
            grid = ?grid,
            block = ?block,
            shared_mem = shared_mem_bytes,
            num_args = args.len(),
            "Dispatching CUDA kernel"
        );

        // Step 3: Marshal arguments and launch
        let mut arg_storage = ArgStorage::from_kernel_args(args);
        let mut arg_ptrs = arg_storage.as_void_ptrs();

        // Step 4: Bind context and launch via raw CUDA driver API.
        self.ctx
            .bind_to_thread()
            .map_err(|e| CudaError::KernelLaunchFailed {
                kernel: entry_point.to_string(),
                reason: format!("Failed to bind context: {e}"),
            })?;

        // SAFETY:
        // - `func` was obtained from cuModuleGetFunction on a valid module.
        // - `arg_ptrs` contains pointers to valid, correctly-typed argument values
        //   whose lifetimes extend past this call (owned by `arg_storage`).
        // - The caller guarantees argument count/types match the kernel signature.
        // - The caller guarantees device pointers are valid and will outlive the launch.
        // - `stream` is a valid CUstream handle (or null for the default stream).
        unsafe {
            let result = sys::cuLaunchKernel(
                func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_mem_bytes,
                stream,
                arg_ptrs.as_mut_ptr(),
                std::ptr::null_mut(), // extra (unused)
            );
            result.result().map_err(|e| CudaError::KernelLaunchFailed {
                kernel: entry_point.to_string(),
                reason: format!("{e}"),
            })?;
        }

        debug!(kernel = entry_point, "CUDA kernel launched successfully");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Argument Marshaling
// ---------------------------------------------------------------------------

/// Holds kernel argument values in a flat, owned buffer.
///
/// Each `KernelArg` variant is stored as its concrete type so we can take
/// pointers to the stored values. The pointers are then collected into the
/// `void**` array that `cuLaunchKernel` expects.
///
/// # Memory Layout
///
/// CUDA's `cuLaunchKernel` takes `void** kernelParams` where each entry
/// points to the parameter value. For example, if a kernel takes
/// `(const uint8_t* ptr, int width)`, the params array contains:
/// - `params[0]` = pointer to a `CUdeviceptr` value (the device pointer)
/// - `params[1]` = pointer to an `i32` value (the width)
#[derive(Debug)]
struct ArgStorage {
    /// Device pointer arguments stored as u64 (CUdeviceptr).
    ptrs: Vec<u64>,
    /// u32 arguments.
    u32s: Vec<u32>,
    /// i32 arguments.
    i32s: Vec<i32>,
    /// f32 arguments.
    f32s: Vec<f32>,
    /// Vec2 arguments stored as [f32; 2].
    vec2s: Vec<[f32; 2]>,
    /// Vec4 arguments stored as [f32; 4].
    vec4s: Vec<[f32; 4]>,
    /// Ordered list of (type_tag, index_into_typed_vec) for building the
    /// void** array in the correct argument order.
    order: Vec<ArgSlot>,
}

/// Identifies which typed vector and index an argument lives at.
#[derive(Debug, Clone, Copy)]
enum ArgSlot {
    Ptr(usize),
    U32(usize),
    I32(usize),
    F32(usize),
    Vec2(usize),
    Vec4(usize),
}

impl ArgStorage {
    /// Convert `KernelArgs` into owned storage.
    fn from_kernel_args(args: &KernelArgs) -> Self {
        let mut storage = Self {
            ptrs: Vec::new(),
            u32s: Vec::new(),
            i32s: Vec::new(),
            f32s: Vec::new(),
            vec2s: Vec::new(),
            vec4s: Vec::new(),
            order: Vec::with_capacity(args.len()),
        };

        for entry in args.entries() {
            match entry {
                KernelArg::DevicePtr(ptr) => {
                    let idx = storage.ptrs.len();
                    storage.ptrs.push(*ptr);
                    storage.order.push(ArgSlot::Ptr(idx));
                }
                KernelArg::U32(val) => {
                    let idx = storage.u32s.len();
                    storage.u32s.push(*val);
                    storage.order.push(ArgSlot::U32(idx));
                }
                KernelArg::I32(val) => {
                    let idx = storage.i32s.len();
                    storage.i32s.push(*val);
                    storage.order.push(ArgSlot::I32(idx));
                }
                KernelArg::F32(val) => {
                    let idx = storage.f32s.len();
                    storage.f32s.push(*val);
                    storage.order.push(ArgSlot::F32(idx));
                }
                KernelArg::Vec2(val) => {
                    let idx = storage.vec2s.len();
                    storage.vec2s.push(*val);
                    storage.order.push(ArgSlot::Vec2(idx));
                }
                KernelArg::Vec4(val) => {
                    let idx = storage.vec4s.len();
                    storage.vec4s.push(*val);
                    storage.order.push(ArgSlot::Vec4(idx));
                }
            }
        }

        storage
    }

    /// Build the `void**` argument pointer array in parameter order.
    ///
    /// Each entry points to the stored value for that parameter position.
    /// The returned `Vec` borrows from `self` — `self` must outlive its use.
    fn as_void_ptrs(&mut self) -> Vec<*mut c_void> {
        self.order
            .iter()
            .map(|slot| match slot {
                ArgSlot::Ptr(i) => (&mut self.ptrs[*i]) as *mut u64 as *mut c_void,
                ArgSlot::U32(i) => (&mut self.u32s[*i]) as *mut u32 as *mut c_void,
                ArgSlot::I32(i) => (&mut self.i32s[*i]) as *mut i32 as *mut c_void,
                ArgSlot::F32(i) => (&mut self.f32s[*i]) as *mut f32 as *mut c_void,
                ArgSlot::Vec2(i) => (&mut self.vec2s[*i]) as *mut [f32; 2] as *mut c_void,
                ArgSlot::Vec4(i) => (&mut self.vec4s[*i]) as *mut [f32; 4] as *mut c_void,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Convenience: NV12 -> RGBA dispatch
// ---------------------------------------------------------------------------

/// Parameters for the `nv12_to_rgba` kernel.
///
/// Maps directly to the kernel signature in `kernels/cuda/nv12_to_rgba.cu`:
/// ```c
/// extern "C" __global__ void nv12_to_rgba(
///     const uint8_t* y_plane,
///     const uint8_t* uv_plane,
///     uint8_t*       rgba_out,
///     int width,
///     int height,
///     int y_pitch,
///     int uv_pitch,
///     int out_pitch)
/// ```
#[derive(Clone, Debug)]
pub struct Nv12ToRgbaParams {
    /// Device pointer to the Y (luma) plane.
    pub y_plane: u64,
    /// Device pointer to the UV (chroma) plane (interleaved Cb, Cr).
    pub uv_plane: u64,
    /// Device pointer to the output RGBA buffer (4 bytes per pixel).
    pub rgba_out: u64,
    /// Frame width in pixels.
    pub width: i32,
    /// Frame height in pixels.
    pub height: i32,
    /// Byte stride of one row in the Y plane.
    pub y_pitch: i32,
    /// Byte stride of one row in the UV plane.
    pub uv_pitch: i32,
    /// Byte stride of one row in the output RGBA buffer.
    pub out_pitch: i32,
}

impl Nv12ToRgbaParams {
    /// Build `KernelArgs` matching the nv12_to_rgba kernel signature.
    pub fn to_kernel_args(&self) -> KernelArgs {
        KernelArgs::new()
            .push_ptr(self.y_plane)
            .push_ptr(self.uv_plane)
            .push_ptr(self.rgba_out)
            .push_i32(self.width)
            .push_i32(self.height)
            .push_i32(self.y_pitch)
            .push_i32(self.uv_pitch)
            .push_i32(self.out_pitch)
    }

    /// Calculate the 2D grid/block dimensions for this frame size.
    ///
    /// Uses 16x16 thread blocks (matching the kernel's design).
    /// Returns `(grid, block)` as `[u32; 3]` arrays.
    pub fn launch_dims(&self) -> ([u32; 3], [u32; 3]) {
        let (grid, block) = compute_launch_config_2d(self.width as u32, self.height as u32);
        ([grid.0, grid.1, grid.2], [block.0, block.1, block.2])
    }
}

/// Dispatch the `nv12_to_rgba` color conversion kernel.
///
/// This is a convenience function that constructs the proper `KernelArgs`,
/// computes the launch grid, and dispatches the kernel.
///
/// # Arguments
///
/// * `km` - The kernel manager (must have the nv12_to_rgba PTX loaded).
/// * `params` - NV12-to-RGBA conversion parameters.
/// * `stream` - The raw CUDA stream handle to launch on.
///
/// # Safety
///
/// The caller must ensure that:
/// - `stream` is a valid `CUstream` handle (or null for the default stream).
/// - Device pointers in `params` point to valid, allocated GPU memory.
/// - GPU memory referenced by `params` will not be freed before the kernel completes.
pub unsafe fn dispatch_nv12_to_rgba(
    km: &KernelManager,
    params: &Nv12ToRgbaParams,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    if params.width <= 0 || params.height <= 0 {
        return Err(CudaError::InvalidKernelArgs {
            kernel: "nv12_to_rgba".to_string(),
            reason: format!("Invalid dimensions: {}x{}", params.width, params.height),
        });
    }

    let args = params.to_kernel_args();
    let (grid, block) = params.launch_dims();
    km.launch(&KernelId::Nv12ToRgba, grid, block, &args, stream)
}

// ---------------------------------------------------------------------------
// Convenience: Alpha Blend (composite) dispatch
// ---------------------------------------------------------------------------

/// Parameters for the `alpha_blend` (composite) kernel.
///
/// Maps directly to the kernel signature in `kernels/cuda/composite.cu`:
/// ```c
/// extern "C" __global__ void alpha_blend(
///     const uint8_t* fg,
///     const uint8_t* bg,
///     uint8_t*       output,
///     int width,
///     int height,
///     int fg_pitch,
///     int bg_pitch,
///     int out_pitch,
///     float opacity)
/// ```
#[derive(Clone, Debug)]
pub struct AlphaBlendParams {
    /// Device pointer to the foreground RGBA buffer.
    pub fg: u64,
    /// Device pointer to the background RGBA buffer.
    pub bg: u64,
    /// Device pointer to the output RGBA buffer (may alias `bg`).
    pub output: u64,
    /// Frame width in pixels.
    pub width: i32,
    /// Frame height in pixels.
    pub height: i32,
    /// Byte stride of one row in the foreground buffer.
    pub fg_pitch: i32,
    /// Byte stride of one row in the background buffer.
    pub bg_pitch: i32,
    /// Byte stride of one row in the output buffer.
    pub out_pitch: i32,
    /// Global foreground opacity in [0.0, 1.0].
    pub opacity: f32,
}

impl AlphaBlendParams {
    /// Build `KernelArgs` matching the alpha_blend kernel signature.
    pub fn to_kernel_args(&self) -> KernelArgs {
        KernelArgs::new()
            .push_ptr(self.fg)
            .push_ptr(self.bg)
            .push_ptr(self.output)
            .push_i32(self.width)
            .push_i32(self.height)
            .push_i32(self.fg_pitch)
            .push_i32(self.bg_pitch)
            .push_i32(self.out_pitch)
            .push_f32(self.opacity)
    }

    /// Calculate the 2D grid/block dimensions for this frame size.
    pub fn launch_dims(&self) -> ([u32; 3], [u32; 3]) {
        let (grid, block) = compute_launch_config_2d(self.width as u32, self.height as u32);
        ([grid.0, grid.1, grid.2], [block.0, block.1, block.2])
    }
}

/// Dispatch the `alpha_blend` compositing kernel.
///
/// # Arguments
///
/// * `km` - The kernel manager (must have the alpha_blend PTX loaded).
/// * `params` - Alpha blend parameters.
/// * `stream` - The raw CUDA stream handle to launch on.
///
/// # Safety
///
/// The caller must ensure that:
/// - `stream` is a valid `CUstream` handle (or null for the default stream).
/// - Device pointers in `params` point to valid, allocated GPU memory.
/// - GPU memory referenced by `params` will not be freed before the kernel completes.
pub unsafe fn dispatch_alpha_blend(
    km: &KernelManager,
    params: &AlphaBlendParams,
    stream: sys::CUstream,
) -> Result<(), CudaError> {
    if params.width <= 0 || params.height <= 0 {
        return Err(CudaError::InvalidKernelArgs {
            kernel: "alpha_blend".to_string(),
            reason: format!("Invalid dimensions: {}x{}", params.width, params.height),
        });
    }

    if !(0.0..=1.0).contains(&params.opacity) {
        return Err(CudaError::InvalidKernelArgs {
            kernel: "alpha_blend".to_string(),
            reason: format!("Opacity {} out of range [0.0, 1.0]", params.opacity),
        });
    }

    let args = params.to_kernel_args();
    let (grid, block) = params.launch_dims();
    km.launch(&KernelId::AlphaBlend, grid, block, &args, stream)
}

// ---------------------------------------------------------------------------
// Grid / Block Helpers
// ---------------------------------------------------------------------------

/// Calculate optimal grid and block dimensions for a 1D kernel launch.
///
/// Returns `(grid_dim, block_dim)` as `(u32, u32, u32)` tuples.
pub fn compute_launch_config_1d(num_elements: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    const BLOCK_SIZE: u32 = 256;
    let grid_x = num_elements.div_ceil(BLOCK_SIZE);
    ((grid_x, 1, 1), (BLOCK_SIZE, 1, 1))
}

/// Calculate optimal grid and block dimensions for a 2D kernel launch
/// (e.g. image processing).
///
/// Returns `(grid_dim, block_dim)` as `(u32, u32, u32)` tuples.
/// Uses 16x16 blocks which is optimal for 2D image operations.
pub fn compute_launch_config_2d(width: u32, height: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    const BLOCK_X: u32 = 16;
    const BLOCK_Y: u32 = 16;
    let grid_x = width.div_ceil(BLOCK_X);
    let grid_y = height.div_ceil(BLOCK_Y);
    ((grid_x, grid_y, 1), (BLOCK_X, BLOCK_Y, 1))
}

/// Convert `[u32; 3]` grid/block arrays (from `ms-common` trait) to cudarc
/// `(u32, u32, u32)` tuples used by `LaunchConfig`.
pub fn to_launch_dims(dims: [u32; 3]) -> (u32, u32, u32) {
    (dims[0], dims[1], dims[2])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config_1d() {
        let (grid, block) = compute_launch_config_1d(1024);
        assert_eq!(grid, (4, 1, 1));
        assert_eq!(block, (256, 1, 1));
    }

    #[test]
    fn test_launch_config_1d_not_multiple() {
        let (grid, block) = compute_launch_config_1d(1000);
        assert_eq!(grid, (4, 1, 1)); // ceil(1000/256) = 4
        assert_eq!(block, (256, 1, 1));
    }

    #[test]
    fn test_launch_config_2d() {
        let (grid, block) = compute_launch_config_2d(1920, 1080);
        assert_eq!(grid, (120, 68, 1)); // ceil(1920/16)=120, ceil(1080/16)=68
        assert_eq!(block, (16, 16, 1));
    }

    #[test]
    fn test_to_launch_dims() {
        assert_eq!(to_launch_dims([10, 20, 1]), (10, 20, 1));
    }

    #[test]
    fn test_kernel_id_module_name() {
        assert_eq!(KernelId::Nv12ToRgba.cuda_module_name(), "nv12_to_rgba.ptx");
        assert_eq!(KernelId::AlphaBlend.cuda_module_name(), "alpha_blend.ptx");
    }

    #[test]
    fn test_nv12_to_rgba_params() {
        let params = Nv12ToRgbaParams {
            y_plane: 0x1000,
            uv_plane: 0x2000,
            rgba_out: 0x3000,
            width: 1920,
            height: 1080,
            y_pitch: 1920,
            uv_pitch: 1920,
            out_pitch: 1920 * 4,
        };

        let args = params.to_kernel_args();
        assert_eq!(args.len(), 8);

        let (grid, block) = params.launch_dims();
        assert_eq!(grid, [120, 68, 1]); // ceil(1920/16), ceil(1080/16)
        assert_eq!(block, [16, 16, 1]);
    }

    #[test]
    fn test_alpha_blend_params() {
        let params = AlphaBlendParams {
            fg: 0x1000,
            bg: 0x2000,
            output: 0x3000,
            width: 1920,
            height: 1080,
            fg_pitch: 1920 * 4,
            bg_pitch: 1920 * 4,
            out_pitch: 1920 * 4,
            opacity: 0.75,
        };

        let args = params.to_kernel_args();
        assert_eq!(args.len(), 9); // 3 ptrs + 5 ints + 1 float

        let (grid, block) = params.launch_dims();
        assert_eq!(grid, [120, 68, 1]);
        assert_eq!(block, [16, 16, 1]);
    }

    #[test]
    fn test_arg_storage_ordering() {
        // Verify that argument marshaling preserves order across mixed types
        let args = KernelArgs::new()
            .push_ptr(0xAAAA)
            .push_i32(42)
            .push_ptr(0xBBBB)
            .push_f32(3.14)
            .push_u32(100);

        let mut storage = ArgStorage::from_kernel_args(&args);
        let ptrs = storage.as_void_ptrs();

        assert_eq!(ptrs.len(), 5);

        // Verify we can read back the values through the pointers
        // SAFETY: Pointers are valid, owned by `storage`, and correctly typed.
        unsafe {
            let ptr0 = *(ptrs[0] as *const u64);
            assert_eq!(ptr0, 0xAAAA);

            let ptr1 = *(ptrs[1] as *const i32);
            assert_eq!(ptr1, 42);

            let ptr2 = *(ptrs[2] as *const u64);
            assert_eq!(ptr2, 0xBBBB);

            let ptr3 = *(ptrs[3] as *const f32);
            assert!((ptr3 - 3.14).abs() < 0.001);

            let ptr4 = *(ptrs[4] as *const u32);
            assert_eq!(ptr4, 100);
        }
    }

    #[test]
    fn test_arg_storage_empty() {
        let args = KernelArgs::new();
        let mut storage = ArgStorage::from_kernel_args(&args);
        let ptrs = storage.as_void_ptrs();
        assert!(ptrs.is_empty());
    }

    #[test]
    fn test_arg_storage_vec2_vec4() {
        let args = KernelArgs::new()
            .push_vec2([1.0, 2.0])
            .push_vec4([3.0, 4.0, 5.0, 6.0]);

        let mut storage = ArgStorage::from_kernel_args(&args);
        let ptrs = storage.as_void_ptrs();
        assert_eq!(ptrs.len(), 2);

        // SAFETY: Pointers are valid, owned by `storage`, and correctly typed.
        unsafe {
            let v2 = *(ptrs[0] as *const [f32; 2]);
            assert_eq!(v2, [1.0, 2.0]);

            let v4 = *(ptrs[1] as *const [f32; 4]);
            assert_eq!(v4, [3.0, 4.0, 5.0, 6.0]);
        }
    }

    #[test]
    fn test_nv12_to_rgba_full_arg_roundtrip() {
        // Verify the exact argument layout matches the CUDA kernel signature:
        // (y_plane, uv_plane, rgba_out, width, height, y_pitch, uv_pitch, out_pitch)
        let params = Nv12ToRgbaParams {
            y_plane: 0xDEAD_0000,
            uv_plane: 0xBEEF_0000,
            rgba_out: 0xCAFE_0000,
            width: 3840,
            height: 2160,
            y_pitch: 3840,
            uv_pitch: 3840,
            out_pitch: 3840 * 4,
        };

        let args = params.to_kernel_args();
        let mut storage = ArgStorage::from_kernel_args(&args);
        let ptrs = storage.as_void_ptrs();

        assert_eq!(ptrs.len(), 8);

        // SAFETY: Pointers are valid, owned by `storage`, and correctly typed.
        unsafe {
            // Arg 0: y_plane (device ptr)
            assert_eq!(*(ptrs[0] as *const u64), 0xDEAD_0000);
            // Arg 1: uv_plane (device ptr)
            assert_eq!(*(ptrs[1] as *const u64), 0xBEEF_0000);
            // Arg 2: rgba_out (device ptr)
            assert_eq!(*(ptrs[2] as *const u64), 0xCAFE_0000);
            // Arg 3: width (i32)
            assert_eq!(*(ptrs[3] as *const i32), 3840);
            // Arg 4: height (i32)
            assert_eq!(*(ptrs[4] as *const i32), 2160);
            // Arg 5: y_pitch (i32)
            assert_eq!(*(ptrs[5] as *const i32), 3840);
            // Arg 6: uv_pitch (i32)
            assert_eq!(*(ptrs[6] as *const i32), 3840);
            // Arg 7: out_pitch (i32)
            assert_eq!(*(ptrs[7] as *const i32), 3840 * 4);
        }

        // Verify launch dimensions for 4K
        let (grid, block) = params.launch_dims();
        assert_eq!(grid, [240, 135, 1]); // ceil(3840/16)=240, ceil(2160/16)=135
        assert_eq!(block, [16, 16, 1]);
    }
}
