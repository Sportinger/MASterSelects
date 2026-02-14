use eframe::egui;

use crate::bridge::PreviewBridge;
use crate::engine::EngineOrchestrator;
use crate::media_panel::MediaPanelState;
use crate::preview_panel::PreviewPanelState;
use crate::properties_panel::PropertiesPanelState;
use crate::timeline::TimelineState;
use crate::toolbar::ToolbarState;

pub struct MasterSelectsApp {
    pub toolbar: ToolbarState,
    pub media_panel: MediaPanelState,
    pub preview: PreviewPanelState,
    pub properties: PropertiesPanelState,
    pub timeline: TimelineState,
    pub left_panel_width: f32,
    pub right_panel_width: f32,
    pub bridge: PreviewBridge,
    pub engine: EngineOrchestrator,
}

impl MasterSelectsApp {
    pub fn new() -> Self {
        Self {
            toolbar: ToolbarState::default(),
            media_panel: MediaPanelState::default(),
            preview: PreviewPanelState::default(),
            properties: PropertiesPanelState::default(),
            timeline: TimelineState::default(),
            left_panel_width: 260.0,
            right_panel_width: 340.0,
            bridge: PreviewBridge::new(1920, 1080),
            engine: EngineOrchestrator::new(),
        }
    }
}

impl eframe::App for MasterSelectsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Pump one frame from the engine into the preview bridge
        self.engine.update(ctx, &mut self.bridge);

        // Toolbar at top
        crate::toolbar::show_toolbar(ctx, &mut self.toolbar);

        // Timeline at bottom
        egui::TopBottomPanel::bottom("timeline_panel")
            .min_height(150.0)
            .default_height(300.0)
            .resizable(true)
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(0x0f, 0x0f, 0x0f)))
            .show(ctx, |ui| {
                crate::timeline::show_timeline(ui, &mut self.timeline);
            });

        // Left panel (Media)
        egui::SidePanel::left("media_panel")
            .default_width(self.left_panel_width)
            .min_width(200.0)
            .max_width(500.0)
            .resizable(true)
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(0x16, 0x16, 0x16))
                    .inner_margin(egui::Margin::same(0)),
            )
            .show(ctx, |ui| {
                crate::media_panel::show_media_panel(ui, &mut self.media_panel);
            });

        // Right panel (Properties)
        egui::SidePanel::right("properties_panel")
            .default_width(self.right_panel_width)
            .min_width(250.0)
            .max_width(600.0)
            .resizable(true)
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(0x16, 0x16, 0x16))
                    .inner_margin(egui::Margin::same(0)),
            )
            .show(ctx, |ui| {
                crate::properties_panel::show_properties_panel(ui, &mut self.properties);
            });

        // Center panel (Preview) - fills remaining space
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(0x0a, 0x0a, 0x0a))
                    .inner_margin(egui::Margin::same(0)),
            )
            .show(ctx, |ui| {
                crate::preview_panel::show_preview_panel(ui, &mut self.preview, &self.bridge);
            });
    }
}
