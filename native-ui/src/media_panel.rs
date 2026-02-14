use egui::{self, Color32, Rect, RichText, Sense, Vec2};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub struct MediaPanelState {
    pub active_tab: MediaTab,
    pub expanded_folders: HashSet<String>,
}

impl Default for MediaPanelState {
    fn default() -> Self {
        let mut expanded = HashSet::new();
        expanded.insert("YouTube".to_string());
        Self {
            active_tab: MediaTab::Media,
            expanded_folders: expanded,
        }
    }
}

#[derive(PartialEq, Clone)]
pub enum MediaTab {
    Media,
    AiChat,
    AiVideo,
    Downloads,
}

impl MediaTab {
    fn label(&self) -> &'static str {
        match self {
            MediaTab::Media => "Media",
            MediaTab::AiChat => "AI Chat",
            MediaTab::AiVideo => "AI Video",
            MediaTab::Downloads => "Downloads",
        }
    }

    const ALL: [MediaTab; 4] = [
        MediaTab::Media,
        MediaTab::AiChat,
        MediaTab::AiVideo,
        MediaTab::Downloads,
    ];
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
const COLOR_TAB_INACTIVE: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);

// Item type colors (small colored rectangles)
const COLOR_VIDEO: Color32 = Color32::from_rgb(0x4a, 0x9e, 0xff);
const COLOR_COMP: Color32 = Color32::from_rgb(0x6a, 0x5a, 0xcd);
const COLOR_AUDIO: Color32 = Color32::from_rgb(0x2e, 0xcc, 0x71);
const COLOR_IMAGE: Color32 = Color32::from_rgb(0xe6, 0x7e, 0x22);

// ---------------------------------------------------------------------------
// Media item helpers
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
enum MediaKind {
    Folder,
    Video,
    Composition,
    Audio,
    Image,
}

impl MediaKind {
    fn rect_color(&self) -> Option<Color32> {
        match self {
            MediaKind::Folder => None,
            MediaKind::Video => Some(COLOR_VIDEO),
            MediaKind::Composition => Some(COLOR_COMP),
            MediaKind::Audio => Some(COLOR_AUDIO),
            MediaKind::Image => Some(COLOR_IMAGE),
        }
    }
}

struct MediaItem {
    kind: MediaKind,
    name: &'static str,
    duration: &'static str,
    resolution: &'static str,
    fps: &'static str,
    depth: u8,
}

fn build_items(state: &MediaPanelState) -> Vec<MediaItem> {
    let mut items: Vec<MediaItem> = Vec::new();

    // -- YouTube folder -------------------------------------------------------
    items.push(MediaItem {
        kind: MediaKind::Folder,
        name: "YouTube",
        duration: "",
        resolution: "",
        fps: "",
        depth: 0,
    });

    if state.expanded_folders.contains("YouTube") {
        items.push(MediaItem {
            kind: MediaKind::Video,
            name: "Anthropic Found Why AIs Go Insan...",
            duration: "9:31",
            resolution: "1920\u{00D7}1080",
            fps: "60",
            depth: 1,
        });
        items.push(MediaItem {
            kind: MediaKind::Video,
            name: "OCEAN LOOP.mp4",
            duration: "0:15",
            resolution: "1920\u{00D7}1080",
            fps: "29.97",
            depth: 1,
        });
    }

    // -- Top-level items ------------------------------------------------------
    items.push(MediaItem {
        kind: MediaKind::Composition,
        name: "Comp 1",
        duration: "1:00",
        resolution: "1920\u{00D7}1080",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Composition,
        name: "Comp 2",
        duration: "1:00",
        resolution: "1920\u{00D7}1080",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "2026-02-09 13-42-01",
        duration: "0:12",
        resolution: "1920\u{00D7}1080",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "DK7A5405",
        duration: "0:10",
        resolution: "1920\u{00D7}1080",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "Nehmt es mit Humor albania fyp frdic...",
        duration: "0:05",
        resolution: "1080\u{00D7}1920",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "NEW FlutterFlow Designer - Complete ...",
        duration: "31:52",
        resolution: "3840\u{00D7}2160",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "rotating_rectangle",
        duration: "0:20",
        resolution: "1920\u{00D7}1080",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Composition,
        name: "Comp 8",
        duration: "1:00",
        resolution: "1920\u{00D7}1080",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "2026-02-09 13-42-01.mp4",
        duration: "0:12",
        resolution: "1920\u{00D7}1080",
        fps: "--",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "DK7A5405.MOV",
        duration: "0:10",
        resolution: "1920\u{00D7}1080",
        fps: "29.97",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "Nehmt es mit Humor albania fyp frdic...",
        duration: "0:05",
        resolution: "1080\u{00D7}1920",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "NEW FlutterFlow Designer - Complete ...",
        duration: "31:52",
        resolution: "3840\u{00D7}2160",
        fps: "30",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "rotating_rectangle.mp4",
        duration: "0:20",
        resolution: "1920\u{00D7}1080",
        fps: "29.95",
        depth: 0,
    });
    items.push(MediaItem {
        kind: MediaKind::Video,
        name: "Volonaut Airbike Test Riding a Flying S...",
        duration: "2:31",
        resolution: "1920\u{00D7}1080",
        fps: "60",
        depth: 0,
    });

    items
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn show_media_panel(ui: &mut egui::Ui, state: &mut MediaPanelState) {
    let panel_rect = ui.available_rect_before_wrap();
    ui.painter().rect_filled(panel_rect, 0.0, COLOR_BG);

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);

        // -- Tab bar ---------------------------------------------------------
        show_tab_bar(ui, state);

        // Thin separator under tab bar
        draw_separator(ui);

        ui.add_space(6.0);

        // -- Tab content -----------------------------------------------------
        match state.active_tab {
            MediaTab::Media => show_media_content(ui, state),
            _ => show_placeholder_tab(ui, state),
        }
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
// Tab bar — plain text with underline on active tab
// ---------------------------------------------------------------------------

fn show_tab_bar(ui: &mut egui::Ui, state: &mut MediaPanelState) {
    let tab_height = 30.0;
    let underline_thickness = 2.0;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
        ui.add_space(8.0);

        for tab in MediaTab::ALL.iter() {
            let is_active = state.active_tab == *tab;
            let text_color = if is_active {
                Color32::WHITE
            } else {
                COLOR_TAB_INACTIVE
            };

            // Allocate space for the tab text
            let galley = ui.fonts(|f| {
                f.layout_no_wrap(tab.label().to_string(), egui::FontId::proportional(12.0), text_color)
            });
            let text_width = galley.size().x;
            let tab_width = text_width + 20.0; // padding on each side

            let (rect, response) = ui.allocate_exact_size(
                Vec2::new(tab_width, tab_height),
                Sense::click(),
            );

            if response.clicked() {
                state.active_tab = tab.clone();
            }

            // Draw tab text centered
            let text_pos = egui::pos2(
                rect.center().x - text_width / 2.0,
                rect.center().y - galley.size().y / 2.0,
            );
            ui.painter().galley(text_pos, galley, text_color);

            // Draw blue underline for active tab
            if is_active {
                let underline_rect = Rect::from_min_size(
                    egui::pos2(rect.min.x + 10.0, rect.max.y - underline_thickness),
                    Vec2::new(rect.width() - 20.0, underline_thickness),
                );
                ui.painter().rect_filled(underline_rect, 0.0, COLOR_ACCENT);
            }

            // Hover effect for inactive tabs: slightly brighter text
            if response.hovered() && !is_active {
                let hover_galley = ui.fonts(|f| {
                    f.layout_no_wrap(
                        tab.label().to_string(),
                        egui::FontId::proportional(12.0),
                        Color32::from_rgb(0xaa, 0xaa, 0xaa),
                    )
                });
                let hover_pos = egui::pos2(
                    rect.center().x - hover_galley.size().x / 2.0,
                    rect.center().y - hover_galley.size().y / 2.0,
                );
                // Overdraw with brighter text on hover
                ui.painter().rect_filled(rect, 0.0, COLOR_BG);
                ui.painter().galley(hover_pos, hover_galley, Color32::from_rgb(0xaa, 0xaa, 0xaa));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Media tab content
// ---------------------------------------------------------------------------

fn show_media_content(ui: &mut egui::Ui, state: &mut MediaPanelState) {
    let items = build_items(state);
    let total_items = 16;

    // -- Header row: "Project" + "16 items" on left, "Import" + "+ Add ▾" on right
    show_header_row(ui, total_items);

    ui.add_space(6.0);

    // -- Column headers -------------------------------------------------------
    show_column_headers(ui);

    // -- Separator under column headers ---------------------------------------
    draw_separator(ui);

    ui.add_space(1.0);

    // -- File list (scrollable) -----------------------------------------------
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::new(0.0, 0.0);
            for item in &items {
                show_media_row(ui, item, state);
            }
            ui.add_space(4.0);
        });
}

// ---------------------------------------------------------------------------
// Header row
// ---------------------------------------------------------------------------

fn show_header_row(ui: &mut egui::Ui, total_items: usize) {
    let row_height = 24.0;
    let avail_width = ui.available_width();

    let (rect, _response) = ui.allocate_exact_size(
        Vec2::new(avail_width, row_height),
        Sense::hover(),
    );

    let y = rect.center().y;

    // Left side: "Project" (bold, white) + "16 items" (gray)
    let project_font = egui::FontId::proportional(13.0);
    let items_font = egui::FontId::proportional(11.0);

    let project_galley = ui.fonts(|f| {
        f.layout_no_wrap("Project".to_string(), project_font.clone(), Color32::WHITE)
    });
    let items_text = format!("{} items", total_items);
    let items_galley = ui.fonts(|f| {
        f.layout_no_wrap(items_text, items_font.clone(), COLOR_DIM)
    });

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

    // Right side: "Import" (plain gray text) + "+ Add ▾" (slightly styled)
    let add_text = "+ Add \u{25BE}";
    let import_font = egui::FontId::proportional(11.0);

    let add_galley = ui.fonts(|f| {
        f.layout_no_wrap(add_text.to_string(), import_font.clone(), COLOR_DIM)
    });
    let import_galley = ui.fonts(|f| {
        f.layout_no_wrap("Import".to_string(), import_font.clone(), COLOR_DIM)
    });

    let add_x = rect.max.x - 10.0 - add_galley.size().x;
    ui.painter().galley(
        egui::pos2(add_x, y - add_galley.size().y / 2.0),
        add_galley.clone(),
        COLOR_DIM,
    );

    let import_x = add_x - 16.0 - import_galley.size().x;
    ui.painter().galley(
        egui::pos2(import_x, y - import_galley.size().y / 2.0),
        import_galley,
        COLOR_DIM,
    );
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

    // Column positions — right-aligned columns from the right edge
    let fps_x = rect.max.x - 16.0;
    let res_x = fps_x - 56.0;
    let dur_x = res_x - 60.0;
    let name_x = rect.min.x + 10.0;

    // NAME with sort dot
    ui.painter().text(
        egui::pos2(name_x, y),
        egui::Align2::LEFT_CENTER,
        "NAME \u{25CF}",
        font.clone(),
        COLOR_VERY_DIM,
    );

    // DURA... (truncated)
    ui.painter().text(
        egui::pos2(dur_x, y),
        egui::Align2::RIGHT_CENTER,
        "DURA...",
        font.clone(),
        COLOR_VERY_DIM,
    );

    // RESOLUTI... (truncated)
    ui.painter().text(
        egui::pos2(res_x, y),
        egui::Align2::RIGHT_CENTER,
        "RESOLUTI...",
        font.clone(),
        COLOR_VERY_DIM,
    );

    // FPS
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

fn show_media_row(ui: &mut egui::Ui, item: &MediaItem, state: &mut MediaPanelState) {
    let row_height = 22.0;
    let base_indent = 8.0;
    let indent_per_level = 18.0;
    let indent = base_indent + indent_per_level * item.depth as f32;
    let avail_width = ui.available_width();
    let is_folder = item.kind == MediaKind::Folder;

    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(avail_width, row_height),
        Sense::click(),
    );

    // Hover highlight
    if response.hovered() {
        ui.painter().rect_filled(rect, 0.0, COLOR_HOVER);
    }

    // Folder expand/collapse toggle
    if is_folder && response.clicked() {
        let name = item.name.to_string();
        if state.expanded_folders.contains(&name) {
            state.expanded_folders.remove(&name);
        } else {
            state.expanded_folders.insert(name);
        }
    }

    let y = rect.center().y;
    let name_font = egui::FontId::proportional(12.0);
    let small_font = egui::FontId::proportional(11.0);
    let arrow_font = egui::FontId::proportional(9.0);

    if is_folder {
        // Draw expand/collapse arrow
        let arrow = if state.expanded_folders.contains(item.name) {
            "\u{25BC}" // ▼
        } else {
            "\u{25B6}" // ▶
        };
        ui.painter().text(
            egui::pos2(rect.min.x + indent, y),
            egui::Align2::LEFT_CENTER,
            arrow,
            arrow_font,
            COLOR_DIM,
        );

        // Folder name (after arrow)
        ui.painter().text(
            egui::pos2(rect.min.x + indent + 14.0, y),
            egui::Align2::LEFT_CENTER,
            item.name,
            name_font,
            COLOR_TEXT,
        );
    } else {
        // Draw small colored rectangle icon (10x10)
        if let Some(color) = item.kind.rect_color() {
            let rect_size = 10.0;
            let icon_rect = Rect::from_min_size(
                egui::pos2(rect.min.x + indent, y - rect_size / 2.0),
                Vec2::new(rect_size, rect_size),
            );
            ui.painter().rect_filled(icon_rect, 2.0, color);
        }

        // File name
        let name_x = rect.min.x + indent + 16.0;

        // Calculate the max width for the name so it doesn't overflow into columns
        let fps_col_x = rect.max.x - 16.0;
        let res_col_x = fps_col_x - 56.0;
        let dur_col_x = res_col_x - 60.0;
        let max_name_width = dur_col_x - name_x - 8.0;

        // Use job to truncate name if needed
        let name_galley = ui.fonts(|f| {
            f.layout(
                item.name.to_string(),
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

        // Duration, Resolution, FPS columns (right-aligned)
        if !item.duration.is_empty() {
            // FPS
            ui.painter().text(
                egui::pos2(fps_col_x, y),
                egui::Align2::RIGHT_CENTER,
                item.fps,
                small_font.clone(),
                COLOR_DIM,
            );

            // Resolution
            ui.painter().text(
                egui::pos2(res_col_x, y),
                egui::Align2::RIGHT_CENTER,
                item.resolution,
                small_font.clone(),
                COLOR_DIM,
            );

            // Duration
            ui.painter().text(
                egui::pos2(dur_col_x, y),
                egui::Align2::RIGHT_CENTER,
                item.duration,
                small_font,
                COLOR_DIM,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Placeholder for non-Media tabs
// ---------------------------------------------------------------------------

fn show_placeholder_tab(ui: &mut egui::Ui, state: &MediaPanelState) {
    let label = state.active_tab.label();

    ui.centered_and_justified(|ui| {
        ui.label(
            RichText::new(label)
                .size(14.0)
                .color(COLOR_DIM),
        );
    });
}
