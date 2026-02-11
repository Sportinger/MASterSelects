//! Windows system tray integration
//!
//! Provides a system tray icon with context menu for the native helper.
//! Only compiled on Windows.

#![cfg(windows)]

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIconBuilder};
use windows_sys::Win32::Foundation::GetLastError;
use windows_sys::Win32::System::Console::GetConsoleWindow;
use windows_sys::Win32::System::Threading::CreateMutexW;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const APP_NAME: &str = "MasterSelects Helper";
const REGISTRY_KEY_NAME: &str = "MasterSelects Helper";
const MUTEX_NAME: &str = "Global\\MasterSelectsNativeHelper";
const FIRST_RUN_REGISTRY_VALUE: &str = "MasterSelectsHelperFirstRunDone";

/// Shared state between tray UI thread and server thread (lock-free)
pub struct TrayState {
    pub running: AtomicBool,
    pub quit_requested: AtomicBool,
    pub connection_count: AtomicU32,
}

impl TrayState {
    pub fn new() -> Self {
        Self {
            running: AtomicBool::new(false),
            quit_requested: AtomicBool::new(false),
            connection_count: AtomicU32::new(0),
        }
    }
}

/// Run the system tray icon and Win32 message pump.
/// Blocks the calling thread until Quit is selected.
pub fn run_tray(state: Arc<TrayState>, port: u16) -> Result<()> {
    let version = env!("CARGO_PKG_VERSION");

    // Create icon from RGBA data
    let (rgba, width, height) = generate_icon_rgba();
    let icon = Icon::from_rgba(rgba, width, height)
        .map_err(|e| anyhow::anyhow!("Failed to create tray icon: {}", e))?;

    // Build context menu
    let menu = Menu::new();

    let title_item = MenuItem::new(
        format!("{} v{}", APP_NAME, version),
        false, // disabled — just a label
        None,
    );

    let status_item = MenuItem::new(
        format!("Status: Starting... (port {})", port),
        false,
        None,
    );

    let autostart_item = CheckMenuItem::new(
        "Start with Windows",
        true,
        is_autostart_enabled(),
        None,
    );

    let open_downloads = MenuItem::new("Open Downloads Folder", true, None);

    let quit_item = MenuItem::new("Quit", true, None);

    menu.append(&title_item)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&status_item)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&autostart_item)?;
    menu.append(&open_downloads)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&quit_item)?;

    // Build tray icon
    let tray = TrayIconBuilder::new()
        .with_icon(icon)
        .with_tooltip(format!("{} v{}", APP_NAME, version))
        .with_menu(Box::new(menu))
        .build()?;

    // Capture menu item IDs for event matching
    let autostart_id = autostart_item.id().clone();
    let open_downloads_id = open_downloads.id().clone();
    let quit_id = quit_item.id().clone();

    // Show welcome dialog on first launch
    show_first_run_dialog();

    // Message pump
    let menu_receiver = MenuEvent::receiver();
    let mut last_tooltip_update = Instant::now();

    loop {
        // Pump Win32 messages (keeps tray icon responsive)
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                if msg.message == WM_QUIT {
                    state.quit_requested.store(true, Ordering::Relaxed);
                    return Ok(());
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // Handle menu events
        if let Ok(event) = menu_receiver.try_recv() {
            if event.id == quit_id {
                state.quit_requested.store(true, Ordering::Relaxed);
                break;
            } else if event.id == autostart_id {
                let current = is_autostart_enabled();
                let desired = !current;
                match set_autostart(desired) {
                    Ok(()) => autostart_item.set_checked(desired),
                    Err(e) => {
                        eprintln!("Failed to set autostart: {}", e);
                        autostart_item.set_checked(current);
                    }
                }
            } else if event.id == open_downloads_id {
                let dir = crate::utils::get_download_dir();
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::process::Command::new("explorer").arg(&dir).spawn();
            }
        }

        // Update tooltip and status every second
        if last_tooltip_update.elapsed() >= Duration::from_secs(1) {
            let conns = state.connection_count.load(Ordering::Relaxed);
            let running = state.running.load(Ordering::Relaxed);

            let status_str = if running {
                if conns > 0 {
                    format!(
                        "Running (port {}) \u{2014} {} client{}",
                        port,
                        conns,
                        if conns == 1 { "" } else { "s" }
                    )
                } else {
                    format!("Running (port {})", port)
                }
            } else {
                "Starting...".to_string()
            };

            let tooltip = format!("{} \u{2014} {}", APP_NAME, status_str);
            tray.set_tooltip(Some(&tooltip)).ok();
            status_item.set_text(format!("Status: {}", status_str));

            last_tooltip_update = Instant::now();
        }

        std::thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Icon generation
// ---------------------------------------------------------------------------

/// Generate a 32x32 tray icon as RGBA bytes.
/// Renders a gradient circle (teal → blue) — recognizable even at 16x16.
fn generate_icon_rgba() -> (Vec<u8>, u32, u32) {
    let size: u32 = 32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let center = size as f32 / 2.0;
    let radius = center - 2.0;

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center + 0.5;
            let dy = y as f32 - center + 0.5;
            let dist = (dx * dx + dy * dy).sqrt();
            let idx = ((y * size + x) * 4) as usize;

            if dist <= radius {
                // Anti-aliased edge
                let alpha = if dist > radius - 1.0 {
                    ((radius - dist) * 255.0).max(0.0) as u8
                } else {
                    255
                };

                // Gradient: top #00D4AA (teal) → bottom #0066CC (blue)
                let t = y as f32 / size as f32;
                rgba[idx] = 0; // R
                rgba[idx + 1] = (212.0 * (1.0 - t) + 102.0 * t) as u8; // G
                rgba[idx + 2] = (170.0 * (1.0 - t) + 204.0 * t) as u8; // B
                rgba[idx + 3] = alpha;
            }
        }
    }

    // Draw a white "M" letter
    draw_m(&mut rgba, size);

    (rgba, size, size)
}

/// Draw a simple "M" onto the icon buffer
fn draw_m(rgba: &mut [u8], size: u32) {
    let s = size as f32;
    let left = (s * 0.22) as i32;
    let right = (s * 0.78) as i32;
    let top = (s * 0.22) as i32;
    let bottom = (s * 0.78) as i32;
    let mid_x = (s * 0.5) as i32;
    let mid_y = (s * 0.48) as i32;
    let stroke = (s * 0.09).max(2.0) as i32;

    for y in top..=bottom {
        for x in left..=right {
            let draw =
                // Left vertical
                (x >= left && x < left + stroke)
                // Right vertical
                || (x > right - stroke && x <= right)
                // Left diagonal (top-left → mid-center)
                || {
                    if y <= mid_y && (mid_y - top) > 0 {
                        let frac = (y - top) as f32 / (mid_y - top) as f32;
                        let ex = left as f32 + frac * (mid_x - left) as f32;
                        (x as f32 - ex).abs() < stroke as f32
                    } else {
                        false
                    }
                }
                // Right diagonal (top-right → mid-center)
                || {
                    if y <= mid_y && (mid_y - top) > 0 {
                        let frac = (y - top) as f32 / (mid_y - top) as f32;
                        let ex = right as f32 - frac * (right - mid_x) as f32;
                        (x as f32 - ex).abs() < stroke as f32
                    } else {
                        false
                    }
                };

            if draw {
                let idx = ((y as u32 * size + x as u32) * 4) as usize;
                if idx + 3 < rgba.len() && rgba[idx + 3] > 0 {
                    rgba[idx] = 255;
                    rgba[idx + 1] = 255;
                    rgba[idx + 2] = 255;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Console window helpers
// ---------------------------------------------------------------------------

/// Hide the console window (used in tray mode)
pub fn hide_console_window() {
    unsafe {
        let console = GetConsoleWindow();
        if !console.is_null() {
            ShowWindow(console, SW_HIDE);
        }
    }
}

// ---------------------------------------------------------------------------
// Single-instance mutex
// ---------------------------------------------------------------------------

/// Opaque wrapper for the Win32 mutex handle. Prevents the handle from being
/// dropped while the program runs (which would release the mutex).
#[allow(dead_code)]
pub struct MutexLock(*mut std::ffi::c_void);
unsafe impl Send for MutexLock {}

/// Acquire a system-wide named mutex to prevent duplicate instances.
/// Returns a `MutexLock` on success, `None` if another instance already holds it.
pub fn acquire_single_instance_lock() -> Option<MutexLock> {
    let wide: Vec<u16> = MUTEX_NAME.encode_utf16().chain(std::iter::once(0)).collect();

    unsafe {
        let handle = CreateMutexW(std::ptr::null(), 1, wide.as_ptr());
        if handle.is_null() {
            return None;
        }
        // ERROR_ALREADY_EXISTS = 183
        if GetLastError() == 183 {
            return None;
        }
        Some(MutexLock(handle))
    }
}

// ---------------------------------------------------------------------------
// Auto-start (registry)
// ---------------------------------------------------------------------------

/// Check if auto-start is enabled in HKCU\...\Run
pub fn is_autostart_enabled() -> bool {
    use winreg::enums::*;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    match hkcu.open_subkey("Software\\Microsoft\\Windows\\CurrentVersion\\Run") {
        Ok(key) => key.get_value::<String, _>(REGISTRY_KEY_NAME).is_ok(),
        Err(_) => false,
    }
}

/// Enable or disable auto-start via the registry
pub fn set_autostart(enabled: bool) -> Result<()> {
    use winreg::enums::*;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let (run_key, _) =
        hkcu.create_subkey("Software\\Microsoft\\Windows\\CurrentVersion\\Run")?;

    if enabled {
        let exe = std::env::current_exe()?;
        run_key.set_value(REGISTRY_KEY_NAME, &exe.to_string_lossy().to_string())?;
    } else {
        let _ = run_key.delete_value(REGISTRY_KEY_NAME);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Welcome dialog (first-run)
// ---------------------------------------------------------------------------

/// Show a one-time welcome dialog explaining the tray icon.
fn show_first_run_dialog() {
    if has_seen_welcome() {
        return;
    }

    let title: Vec<u16> = format!("{}\0", APP_NAME).encode_utf16().collect();
    let message: Vec<u16> = concat!(
        "MasterSelects Helper is now running in the background!\n\n",
        "You can find it in the system tray (notification area) ",
        "at the bottom-right of your taskbar.\n\n",
        "Right-click the tray icon for options like:\n",
        "  \u{2022} View connection status\n",
        "  \u{2022} Start with Windows\n",
        "  \u{2022} Open downloads folder\n",
        "  \u{2022} Quit the helper\n\n",
        "Tip: If the icon is hidden, click the \u{25B2} arrow in the taskbar to reveal it.",
        "\0"
    )
    .encode_utf16()
    .collect();

    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            message.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONINFORMATION,
        );
    }

    mark_welcome_seen();
}

/// Check registry for first-run flag
fn has_seen_welcome() -> bool {
    use winreg::enums::*;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    match hkcu.open_subkey("Software\\MasterSelects") {
        Ok(key) => key.get_value::<u32, _>(FIRST_RUN_REGISTRY_VALUE).unwrap_or(0) == 1,
        Err(_) => false,
    }
}

/// Set first-run flag in registry
fn mark_welcome_seen() {
    use winreg::enums::*;
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    if let Ok((key, _)) = hkcu.create_subkey("Software\\MasterSelects") {
        let _ = key.set_value(FIRST_RUN_REGISTRY_VALUE, &1u32);
    }
}
