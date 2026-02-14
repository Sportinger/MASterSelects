use egui::{self, Color32, Rect, RichText, Sense, Vec2};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// A real imported media file entry.
#[derive(Clone, Debug)]
pub struct MediaFile {
    pub name: String,
    pub path: String,
    pub kind: MediaKind,
    pub duration: String,
    pub resolution: String,
    pub fps: String,
}

pub struct MediaPanelState {
    /// Currently loaded media files.
    pub files: Vec<MediaFile>,
    /// Whether an import was requested this frame.
    pub import_requested: bool,
}

impl Default for MediaPanelState {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            import_requested: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

const COLOR_BG: Color32 = Color32::from_rgb(0x16, 0x16, 0x16);
const COLOR_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COLOR_DIM: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
const COLOR_VERY_DIM: Color32 = Color32::from_rgb(0x66, 0x66, 0x66);
const COLOR_HOVER: Color32 = Color32::from_rgb(0x25, 0x25, 0x25);
const COLOR_ACCENT: Color32 = Color32::from_rgb(0x2D, 0x8C, 0xEB);
const COLOR_SEP: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);

// Item type colors
const COLOR_VIDEO: Color32 = Color32::from_rgb(0x4a, 0x9e, 0xff);
const COLOR_AUDIO: Color32 = Color32::from_rgb(0x2e, 0xcc, 0x71);

// ---------------------------------------------------------------------------
// Media kind
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum MediaKind {
    Video,
    Audio,
}

impl MediaKind {
    fn rect_color(&self) -> Color32 {
        match self {
            MediaKind::Video => COLOR_VIDEO,
            MediaKind::Audio => COLOR_AUDIO,
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn show_media_panel(ui: &mut egui::Ui, state: &mut MediaPanelState) {
    let panel_rect = ui.available_rect_before_wrap();
    ui.painter().rect_filled(panel_rect, 0.0, COLOR_BG);

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);

        // -- Header: "Media" title -----------------------------------------------
        ui.add_space(8.0);
        show_header_row(ui, state);
        ui.add_space(6.0);

        // Separator
        draw_separator(ui);

        if state.files.is_empty() {
            // Empty state
            show_empty_state(ui, state);
        } else {
            // Column headers
            show_column_headers(ui);
            draw_separator(ui);
            ui.add_space(1.0);

            // File list
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
                    for file in &state.files {
                        show_media_row(ui, file);
                    }
                    ui.add_space(4.0);
                });
        }
    });
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

fn show_empty_state(ui: &mut egui::Ui, state: &mut MediaPanelState) {
    let available = ui.available_size();
    ui.add_space(available.y * 0.3);

    ui.vertical_centered(|ui| {
        ui.label(
            RichText::new("No media files")
                .color(COLOR_DIM)
                .size(14.0),
        );
        ui.add_space(8.0);
        ui.label(
            RichText::new("Import a video or audio file to get started")
                .color(COLOR_VERY_DIM)
                .size(11.0),
        );
        ui.add_space(16.0);

        if ui
            .add(
                egui::Button::new(
                    RichText::new("Import File")
                        .color(Color32::WHITE)
                        .strong()
                        .size(12.0),
                )
                .fill(COLOR_ACCENT)
                .corner_radius(egui::CornerRadius::same(4))
                .min_size(Vec2::new(120.0, 32.0)),
            )
            .clicked()
        {
            state.import_requested = true;
        }

        ui.add_space(8.0);
        ui.label(
            RichText::new("Ctrl+I")
                .color(COLOR_VERY_DIM)
                .size(10.0),
        );
    });
}

// ---------------------------------------------------------------------------
// Separator helper
// ---------------------------------------------------------------------------

fn draw_separator(ui: &mut egui::Ui) {
    let avail = ui.available_rect_before_wrap();
    let rect = Rect::from_min_size(avail.min, Vec2::new(avail.width(), 1.0));
    ui.painter().rect_filled(rect, 0.0, COLOR_SEP);
    ui.advance_cursor_after_rect(rect);
}

// ---------------------------------------------------------------------------
// Header row
// ---------------------------------------------------------------------------

fn show_header_row(ui: &mut egui::Ui, state: &mut MediaPanelState) {
    let row_height = 24.0;
    let avail_width = ui.available_width();

    let (rect, _response) =
        ui.allocate_exact_size(Vec2::new(avail_width, row_height), Sense::hover());

    let y = rect.center().y;

    // Left side: "Media" + item count
    let project_font = egui::FontId::proportional(13.0);
    let items_font = egui::FontId::proportional(11.0);

    let project_galley =
        ui.fonts(|f| f.layout_no_wrap("Media".to_string(), project_font.clone(), Color32::WHITE));
    let items_text = format!("{} files", state.files.len());
    let items_galley = ui.fonts(|f| f.layout_no_wrap(items_text, items_font.clone(), COLOR_DIM));

    let project_x = rect.min.x + 10.0;
    ui.painter().galley(
        egui::pos2(project_x, y - project_galley.size().y / 2.0),
        project_galley.clone(),
        Color32::WHITE,
    );

    let items_x = project_x + project_galley.size().x + 8.0;
    ui.painter().galley(
        egui::pos2(items_x, y - items_galley.size().y / 2.0),
        items_galley,
        COLOR_DIM,
    );

    // Right side: "Import" button
    let import_font = egui::FontId::proportional(11.0);
    let import_galley =
        ui.fonts(|f| f.layout_no_wrap("Import".to_string(), import_font.clone(), COLOR_ACCENT));
    let import_x = rect.max.x - 10.0 - import_galley.size().x;

    // Make it clickable
    let import_rect = Rect::from_min_size(
        egui::pos2(import_x - 4.0, y - import_galley.size().y / 2.0 - 2.0),
        Vec2::new(import_galley.size().x + 8.0, import_galley.size().y + 4.0),
    );
    let import_resp = ui.interact(import_rect, ui.id().with("import_btn"), Sense::click());

    if import_resp.hovered() {
        ui.painter()
            .rect_filled(import_rect, 3.0, COLOR_HOVER);
    }

    ui.painter().galley(
        egui::pos2(import_x, y - import_galley.size().y / 2.0),
        import_galley,
        COLOR_ACCENT,
    );

    if import_resp.clicked() {
        state.import_requested = true;
    }
}

// ---------------------------------------------------------------------------
// Column headers
// ---------------------------------------------------------------------------

fn show_column_headers(ui: &mut egui::Ui) {
    let row_height = 20.0;
    let avail_width = ui.available_width();

    let (rect, _) = ui.allocate_exact_size(Vec2::new(avail_width, row_height), Sense::hover());

    let y = rect.center().y;
    let font = egui::FontId::proportional(10.0);

    let fps_x = rect.max.x - 16.0;
    let res_x = fps_x - 56.0;
    let dur_x = res_x - 60.0;
    let name_x = rect.min.x + 10.0;

    ui.painter().text(
        egui::pos2(name_x, y),
        egui::Align2::LEFT_CENTER,
        "NAME",
        font.clone(),
        COLOR_VERY_DIM,
    );

    ui.painter().text(
        egui::pos2(dur_x, y),
        egui::Align2::RIGHT_CENTER,
        "DURATION",
        font.clone(),
        COLOR_VERY_DIM,
    );

    ui.painter().text(
        egui::pos2(res_x, y),
        egui::Align2::RIGHT_CENTER,
        "RESOLUTION",
        font.clone(),
        COLOR_VERY_DIM,
    );

    ui.painter().text(
        egui::pos2(fps_x, y),
        egui::Align2::RIGHT_CENTER,
        "FPS",
        font,
        COLOR_VERY_DIM,
    );
}

// ---------------------------------------------------------------------------
// Single media row
// ---------------------------------------------------------------------------

fn show_media_row(ui: &mut egui::Ui, file: &MediaFile) {
    let row_height = 22.0;
    let avail_width = ui.available_width();

    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(avail_width, row_height), Sense::click());

    if response.hovered() {
        ui.painter().rect_filled(rect, 0.0, COLOR_HOVER);
    }

    let y = rect.center().y;
    let name_font = egui::FontId::proportional(12.0);
    let small_font = egui::FontId::proportional(11.0);

    // Colored icon
    let indent = 8.0;
    let rect_size = 10.0;
    let icon_rect = Rect::from_min_size(
        egui::pos2(rect.min.x + indent, y - rect_size / 2.0),
        Vec2::new(rect_size, rect_size),
    );
    ui.painter()
        .rect_filled(icon_rect, 2.0, file.kind.rect_color());

    // File name
    let name_x = rect.min.x + indent + 16.0;
    let fps_col_x = rect.max.x - 16.0;
    let res_col_x = fps_col_x - 56.0;
    let dur_col_x = res_col_x - 60.0;
    let max_name_width = dur_col_x - name_x - 8.0;

    let name_galley = ui.fonts(|f| {
        f.layout(
            file.name.clone(),
            name_font.clone(),
            COLOR_TEXT,
            max_name_width.max(10.0),
        )
    });
    ui.painter().galley(
        egui::pos2(name_x, y - name_galley.size().y / 2.0),
        name_galley,
        COLOR_TEXT,
    );

    // Duration, Resolution, FPS columns
    ui.painter().text(
        egui::pos2(fps_col_x, y),
        egui::Align2::RIGHT_CENTER,
        &file.fps,
        small_font.clone(),
        COLOR_DIM,
    );

    ui.painter().text(
        egui::pos2(res_col_x, y),
        egui::Align2::RIGHT_CENTER,
        &file.resolution,
        small_font.clone(),
        COLOR_DIM,
    );

    ui.painter().text(
        egui::pos2(dur_col_x, y),
        egui::Align2::RIGHT_CENTER,
        &file.duration,
        small_font,
        COLOR_DIM,
    );
}
