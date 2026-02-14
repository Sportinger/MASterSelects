use egui::{Color32, RichText, TopBottomPanel};

// ---------------------------------------------------------------------------
// Action enum — polled by app.rs each frame
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum ToolbarAction {
    NewProject,
    OpenProject,
    SaveProject,
    SaveProjectAs,
    ImportMedia,
    Undo,
    Redo,
    ExportStart,
    // Playback
    Play,
    Pause,
    Stop,
}

// ---------------------------------------------------------------------------
// Engine state shown on the right side of the toolbar
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
pub enum EngineState {
    #[default]
    Idle,
    Playing,
    Paused,
}

impl std::fmt::Display for EngineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Playing => write!(f, "Playing"),
            Self::Paused => write!(f, "Paused"),
        }
    }
}

// ---------------------------------------------------------------------------
// Toolbar state
// ---------------------------------------------------------------------------

pub struct ToolbarState {
    // Existing
    pub project_name: String,

    // NEW: menu action signal (polled by app.rs each frame)
    pub action: Option<ToolbarAction>,

    // NEW: engine info displayed on right side
    pub gpu_name: String,
    pub engine_state: EngineState,
    pub fps: f64,
}

impl Default for ToolbarState {
    fn default() -> Self {
        Self {
            project_name: "Untitled".to_string(),
            action: None,
            gpu_name: "GPU: detecting...".to_string(),
            engine_state: EngineState::default(),
            fps: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

const BG_COLOR: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x1a);
const MENU_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const PROJECT_NAME_COLOR: Color32 = Color32::from_rgb(0xe0, 0xe0, 0xe0);
const WIP_COLOR: Color32 = Color32::from_rgb(0x4e, 0xcd, 0xc4);
const WIP_DIM_COLOR: Color32 = Color32::from_rgb(0x3a, 0x9a, 0x94);
const GPU_DOT_COLOR: Color32 = Color32::from_rgb(0x2e, 0xcc, 0x71);
const GPU_TEXT_COLOR: Color32 = Color32::from_rgb(0x99, 0x99, 0x99);
const SHORTCUT_COLOR: Color32 = Color32::from_rgb(0x77, 0x77, 0x77);
const FPS_COLOR: Color32 = Color32::from_rgb(0xaa, 0xaa, 0xaa);
const STATE_IDLE_COLOR: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
const STATE_PLAYING_COLOR: Color32 = Color32::from_rgb(0x2e, 0xcc, 0x71);
const STATE_PAUSED_COLOR: Color32 = Color32::from_rgb(0xe0, 0xc0, 0x40);

// ---------------------------------------------------------------------------
// Helper: menu item with right-aligned shortcut hint
// ---------------------------------------------------------------------------

fn menu_item(ui: &mut egui::Ui, label: &str, shortcut: &str) -> bool {
    let response = ui.horizontal(|ui| {
        let label_resp = ui.label(RichText::new(label).size(12.0));
        // Push shortcut to the right
        let remaining = (180.0 - label_resp.rect.width()).max(0.0);
        ui.add_space(remaining);
        ui.label(RichText::new(shortcut).color(SHORTCUT_COLOR).size(11.0));
    });
    // Make the whole row clickable
    let rect = response.response.rect;
    let interact = ui.interact(rect, ui.id().with(label), egui::Sense::click());
    interact.clicked()
}

// ---------------------------------------------------------------------------
// Main toolbar render
// ---------------------------------------------------------------------------

pub fn show_toolbar(ctx: &egui::Context, state: &mut ToolbarState) {
    // Clear previous frame's action
    state.action = None;

    TopBottomPanel::top("toolbar")
        .exact_height(28.0)
        .frame(
            egui::Frame::NONE
                .fill(BG_COLOR)
                .inner_margin(egui::Margin::symmetric(8, 2)),
        )
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                // 1. Project name (editable) on far left with dropdown arrow
                let text_edit = egui::TextEdit::singleline(&mut state.project_name)
                    .desired_width(120.0)
                    .text_color(PROJECT_NAME_COLOR)
                    .frame(false)
                    .margin(egui::Margin::symmetric(4, 0));
                ui.add(text_edit);
                ui.label(
                    RichText::new("\u{25be}")
                        .color(PROJECT_NAME_COLOR)
                        .size(10.0),
                );

                ui.add_space(12.0);

                // 2. Menu bar with real actions
                egui::menu::bar(ui, |ui| {
                    ui.visuals_mut().override_text_color = Some(MENU_TEXT);

                    // -------------------------------------------------------
                    // File menu
                    // -------------------------------------------------------
                    ui.menu_button(RichText::new("File").color(MENU_TEXT).size(12.0), |ui| {
                        ui.set_min_width(220.0);

                        if menu_item(ui, "New Project", "Ctrl+N") {
                            state.action = Some(ToolbarAction::NewProject);
                            ui.close_menu();
                        }
                        if menu_item(ui, "Open Project", "Ctrl+O") {
                            state.action = Some(ToolbarAction::OpenProject);
                            ui.close_menu();
                        }

                        ui.separator();

                        if menu_item(ui, "Save", "Ctrl+S") {
                            state.action = Some(ToolbarAction::SaveProject);
                            ui.close_menu();
                        }
                        if menu_item(ui, "Save As", "Ctrl+Shift+S") {
                            state.action = Some(ToolbarAction::SaveProjectAs);
                            ui.close_menu();
                        }

                        ui.separator();

                        if menu_item(ui, "Import Media", "Ctrl+I") {
                            state.action = Some(ToolbarAction::ImportMedia);
                            ui.close_menu();
                        }

                        ui.separator();

                        if menu_item(ui, "Export", "Ctrl+E") {
                            state.action = Some(ToolbarAction::ExportStart);
                            ui.close_menu();
                        }
                    });

                    // -------------------------------------------------------
                    // Edit menu
                    // -------------------------------------------------------
                    ui.menu_button(RichText::new("Edit").color(MENU_TEXT).size(12.0), |ui| {
                        ui.set_min_width(220.0);

                        if menu_item(ui, "Undo", "Ctrl+Z") {
                            state.action = Some(ToolbarAction::Undo);
                            ui.close_menu();
                        }
                        if menu_item(ui, "Redo", "Ctrl+Shift+Z") {
                            state.action = Some(ToolbarAction::Redo);
                            ui.close_menu();
                        }
                    });

                    // -------------------------------------------------------
                    // View menu (kept from original, no action wiring yet)
                    // -------------------------------------------------------
                    ui.menu_button(RichText::new("View").color(MENU_TEXT).size(12.0), |ui| {
                        if ui.button("Zoom In").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Zoom Out").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Fit to Window").clicked() {
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Toggle Fullscreen").clicked() {
                            ui.close_menu();
                        }
                    });

                    // -------------------------------------------------------
                    // Output menu (kept from original)
                    // -------------------------------------------------------
                    ui.menu_button(RichText::new("Output").color(MENU_TEXT).size(12.0), |ui| {
                        if ui.button("Export Video").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Export Frame").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Export Audio").clicked() {
                            ui.close_menu();
                        }
                    });

                    // -------------------------------------------------------
                    // Window menu (kept from original)
                    // -------------------------------------------------------
                    ui.menu_button(RichText::new("Window").color(MENU_TEXT).size(12.0), |ui| {
                        if ui.button("Reset Layout").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Toggle Media Panel").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Toggle Properties").clicked() {
                            ui.close_menu();
                        }
                    });

                    // -------------------------------------------------------
                    // Info menu (kept from original)
                    // -------------------------------------------------------
                    ui.menu_button(RichText::new("Info").color(MENU_TEXT).size(12.0), |ui| {
                        if ui.button("About").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Keyboard Shortcuts").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Check for Updates").clicked() {
                            ui.close_menu();
                        }
                    });
                });

                // 3. Center area — WIP text and version
                let available = ui.available_width();
                let wip_text = "Work in Progress - Expect Bugs!";
                let version_text = " v1.2.6";
                let text_width = 280.0;
                let right_section_width = 280.0; // wider to fit new info
                let center_offset = (available - text_width - right_section_width) / 2.0;

                if center_offset > 0.0 {
                    ui.add_space(center_offset);
                }

                ui.label(RichText::new(wip_text).color(WIP_COLOR).size(11.5));
                ui.label(RichText::new(version_text).color(WIP_DIM_COLOR).size(11.0));

                // 4. Right side — engine state info
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;

                    // FPS counter
                    let fps_text = format!("{:.0} fps", state.fps);
                    ui.label(RichText::new(fps_text).color(FPS_COLOR).size(11.0));

                    ui.label(
                        RichText::new("|")
                            .color(Color32::from_rgb(0x44, 0x44, 0x44))
                            .size(11.0),
                    );

                    // Engine state
                    let state_color = match state.engine_state {
                        EngineState::Idle => STATE_IDLE_COLOR,
                        EngineState::Playing => STATE_PLAYING_COLOR,
                        EngineState::Paused => STATE_PAUSED_COLOR,
                    };
                    ui.label(
                        RichText::new(state.engine_state.to_string())
                            .color(state_color)
                            .size(11.0),
                    );

                    ui.label(
                        RichText::new("|")
                            .color(Color32::from_rgb(0x44, 0x44, 0x44))
                            .size(11.0),
                    );

                    // GPU name with green dot
                    ui.label(
                        RichText::new(&state.gpu_name)
                            .color(GPU_TEXT_COLOR)
                            .size(11.0),
                    );
                    ui.label(RichText::new("\u{25cf}").color(GPU_DOT_COLOR).size(11.0));
                });
            });
        });
}
