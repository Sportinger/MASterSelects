use egui::{Color32, RichText, TopBottomPanel};

pub struct ToolbarState {
    pub project_name: String,
}

impl Default for ToolbarState {
    fn default() -> Self {
        Self {
            project_name: "Untitled".to_string(),
        }
    }
}

// Colors
const BG_COLOR: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x1a);
const MENU_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const PROJECT_NAME_COLOR: Color32 = Color32::from_rgb(0xe0, 0xe0, 0xe0);
const WIP_COLOR: Color32 = Color32::from_rgb(0x4e, 0xcd, 0xc4);
const WIP_DIM_COLOR: Color32 = Color32::from_rgb(0x3a, 0x9a, 0x94);
const GPU_DOT_COLOR: Color32 = Color32::from_rgb(0x2e, 0xcc, 0x71);
const GPU_TEXT_COLOR: Color32 = Color32::from_rgb(0x99, 0x99, 0x99);

pub fn show_toolbar(ctx: &egui::Context, state: &mut ToolbarState) {
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

                // 2. Menu items as plain text buttons with dropdown menus
                egui::menu::bar(ui, |ui| {
                    ui.visuals_mut().override_text_color = Some(MENU_TEXT);

                    // File menu
                    ui.menu_button(RichText::new("File").color(MENU_TEXT).size(12.0), |ui| {
                        if ui.button("New Project").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Open").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Save").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Save As").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Import Media").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Export").clicked() {
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Exit").clicked() {
                            ui.close_menu();
                        }
                    });

                    // Edit menu
                    ui.menu_button(RichText::new("Edit").color(MENU_TEXT).size(12.0), |ui| {
                        if ui.button("Undo").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Redo").clicked() {
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Cut").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Copy").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Paste").clicked() {
                            ui.close_menu();
                        }
                        if ui.button("Delete").clicked() {
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Select All").clicked() {
                            ui.close_menu();
                        }
                    });

                    // View menu
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

                    // Output menu
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

                    // Window menu
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

                    // Info menu
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
                // Use remaining space to center the text
                let available = ui.available_width();
                let wip_text = "Work in Progress - Expect Bugs!";
                let version_text = " v1.2.6";
                let text_width = 280.0; // approximate width of center text
                let right_section_width = 160.0; // approximate width of right section
                let center_offset = (available - text_width - right_section_width) / 2.0;

                if center_offset > 0.0 {
                    ui.add_space(center_offset);
                }

                ui.label(RichText::new(wip_text).color(WIP_COLOR).size(11.5));
                ui.label(RichText::new(version_text).color(WIP_DIM_COLOR).size(11.0));

                // 4. Right side — GPU status
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new("WebGPU (amd)")
                            .color(GPU_TEXT_COLOR)
                            .size(11.5),
                    );
                    ui.label(RichText::new("\u{25cf}").color(GPU_DOT_COLOR).size(11.0));
                });
            });
        });
}
