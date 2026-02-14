//! GPU kernel/shader identification and argument passing.

/// Identifies a GPU kernel by name (maps to .cu / .comp files).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelId {
    /// NV12 to RGBA color space conversion.
    Nv12ToRgba,
    /// Alpha blend (compositor).
    AlphaBlend,
    /// Transform (position, scale, rotation).
    Transform,
    /// Mask application (rect, ellipse, path).
    Mask,
    /// Transition effect.
    Transition,
    /// Named effect kernel.
    Effect(String),
}

impl KernelId {
    /// Returns the kernel function name used in CUDA PTX / Vulkan entry point.
    pub fn entry_point(&self) -> &str {
        match self {
            Self::Nv12ToRgba => "nv12_to_rgba",
            Self::AlphaBlend => "alpha_blend",
            Self::Transform => "transform_2d",
            Self::Mask => "apply_mask",
            Self::Transition => "transition",
            Self::Effect(name) => name.as_str(),
        }
    }

    /// Returns the PTX module filename (CUDA).
    pub fn cuda_module_name(&self) -> String {
        format!("{}.ptx", self.entry_point())
    }

    /// Returns the SPIR-V module filename (Vulkan).
    pub fn vulkan_module_name(&self) -> String {
        format!("{}.spv", self.entry_point())
    }
}

/// Arguments passed to a GPU kernel dispatch.
#[derive(Clone, Debug)]
pub struct KernelArgs {
    entries: Vec<KernelArg>,
}

/// A single kernel argument.
#[derive(Clone, Debug)]
pub enum KernelArg {
    /// Device buffer pointer (opaque handle).
    DevicePtr(u64),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit signed integer.
    I32(i32),
    /// 32-bit float.
    F32(f32),
    /// 2-component float vector.
    Vec2([f32; 2]),
    /// 4-component float vector.
    Vec4([f32; 4]),
}

impl KernelArgs {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn push_ptr(mut self, ptr: u64) -> Self {
        self.entries.push(KernelArg::DevicePtr(ptr));
        self
    }

    pub fn push_u32(mut self, val: u32) -> Self {
        self.entries.push(KernelArg::U32(val));
        self
    }

    pub fn push_i32(mut self, val: i32) -> Self {
        self.entries.push(KernelArg::I32(val));
        self
    }

    pub fn push_f32(mut self, val: f32) -> Self {
        self.entries.push(KernelArg::F32(val));
        self
    }

    pub fn push_vec2(mut self, val: [f32; 2]) -> Self {
        self.entries.push(KernelArg::Vec2(val));
        self
    }

    pub fn push_vec4(mut self, val: [f32; 4]) -> Self {
        self.entries.push(KernelArg::Vec4(val));
        self
    }

    pub fn entries(&self) -> &[KernelArg] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for KernelArgs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_id_entry_points() {
        assert_eq!(KernelId::Nv12ToRgba.entry_point(), "nv12_to_rgba");
        assert_eq!(
            KernelId::Effect("gaussian_blur".into()).entry_point(),
            "gaussian_blur"
        );
    }

    #[test]
    fn kernel_args_builder() {
        let args = KernelArgs::new()
            .push_ptr(0x1000)
            .push_ptr(0x2000)
            .push_u32(1920)
            .push_u32(1080);
        assert_eq!(args.len(), 4);
    }
}
