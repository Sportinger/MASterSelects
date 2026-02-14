//! Build script for `masterselects-native`.
//!
//! Currently a placeholder. GPU kernel compilation is handled by
//! `crates/gpu-hal/build.rs`. When `native-ui` gains a direct dependency
//! on `ms-gpu-hal`, the compiled kernels will be available automatically
//! through that crate's embedded bytecode.
//!
//! This build script sets up `cargo:rerun-if-changed` directives so that
//! changes to GPU kernel sources trigger a rebuild of the top-level binary.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let workspace_root = manifest_dir
        .parent()
        .expect("Cannot determine workspace root");

    let cuda_kernel_dir = workspace_root.join("kernels").join("cuda");
    let vulkan_kernel_dir = workspace_root.join("kernels").join("vulkan");

    // Rerun if kernel source directories change (new files added/removed).
    if cuda_kernel_dir.exists() {
        println!("cargo:rerun-if-changed={}", cuda_kernel_dir.display());
    }
    if vulkan_kernel_dir.exists() {
        println!("cargo:rerun-if-changed={}", vulkan_kernel_dir.display());
    }
}
