//! Build script for cross-platform FFmpeg discovery and Windows resource embedding
//!
//! On Windows: Uses FFMPEG_DIR env var or bundled ffmpeg/win64/ directory,
//!             and embeds icon + metadata in the exe via winresource.
//! On Linux:   Uses pkg-config (system FFmpeg)
//! On macOS:   Uses pkg-config or Homebrew FFmpeg

fn main() {
    // On Windows, help ffmpeg-sys-next find the libraries
    #[cfg(target_os = "windows")]
    {
        // --- FFmpeg discovery ---

        // Check if FFMPEG_DIR is already set
        if std::env::var("FFMPEG_DIR").is_ok() {
            println!("cargo:warning=Using FFMPEG_DIR from environment");
        } else {
            // Try bundled FFmpeg in project directory
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let bundled = std::path::PathBuf::from(&manifest_dir)
                .join("ffmpeg")
                .join("win64");

            if bundled.join("lib").exists() && bundled.join("include").exists() {
                println!(
                    "cargo:warning=Using bundled FFmpeg from {}",
                    bundled.display()
                );
                println!("cargo:rustc-env=FFMPEG_DIR={}", bundled.display());
                // Also set for ffmpeg-sys-next build script
                std::env::set_var("FFMPEG_DIR", &bundled);
            } else {
                println!("cargo:warning=No FFmpeg found! Set FFMPEG_DIR or place pre-built FFmpeg in ffmpeg/win64/");
                println!("cargo:warning=Download from: https://github.com/BtbN/FFmpeg-Builds/releases");
                println!("cargo:warning=Expected structure: ffmpeg/win64/{{bin,include,lib}}/");
            }
        }

        // --- Embed icon and metadata in exe ---

        let mut res = winresource::WindowsResource::new();

        // Embed icon if the file exists
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let icon_path = std::path::PathBuf::from(&manifest_dir).join("assets").join("icon.ico");
        if icon_path.exists() {
            res.set_icon(icon_path.to_str().unwrap());
        } else {
            println!("cargo:warning=No icon found at assets/icon.ico — exe will use default icon");
        }

        res.set("ProductName", "MasterSelects Helper");
        res.set("FileDescription", "MasterSelects Native Helper — video codec acceleration");
        res.set("CompanyName", "MasterSelects");
        res.set("LegalCopyright", "MIT License");

        if let Err(e) = res.compile() {
            println!(
                "cargo:warning=winresource failed (icon/metadata may not be embedded): {}",
                e
            );
        }
    }
}
