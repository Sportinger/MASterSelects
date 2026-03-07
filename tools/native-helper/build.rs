//! Build script for Windows resource embedding (icon + metadata)

fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();

        // Embed icon if the file exists
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let icon_path = std::path::PathBuf::from(&manifest_dir)
            .join("assets")
            .join("icon.ico");
        if icon_path.exists() {
            res.set_icon(icon_path.to_str().unwrap());
        } else {
            println!("cargo:warning=No icon found at assets/icon.ico — exe will use default icon");
        }

        res.set("ProductName", "MasterSelects Helper");
        res.set(
            "FileDescription",
            "MasterSelects Native Helper — download acceleration",
        );
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
