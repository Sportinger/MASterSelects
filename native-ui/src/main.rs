mod app;
mod bridge;
#[allow(dead_code)]
mod engine;
#[allow(dead_code)]
mod export;
#[allow(dead_code)]
mod media_panel;
mod preview_panel;
mod properties_panel;
mod theme;
#[allow(dead_code)]
mod timeline;
mod toolbar;

use app::MasterSelectsApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("MasterSelects")
            .with_inner_size([1600.0, 900.0])
            .with_min_inner_size([1024.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "MasterSelects",
        options,
        Box::new(|cc| {
            let app = MasterSelectsApp::new();
            theme::apply_theme(&cc.egui_ctx);
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
}
