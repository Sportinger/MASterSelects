use egui::{self, Color32, Rect, CornerRadius, Stroke, Vec2, Pos2, FontId, Align2, ScrollArea};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

pub struct TimelineState {
    pub playing: bool,
    pub looping: bool,
    pub current_time: f32,
    pub total_duration: f32,
    pub zoom: f32,
    pub tracks: Vec<Track>,
    pub playhead_pos: f32,
    pub comp_tabs: Vec<String>,
    pub active_comp: usize,
    pub scroll_offset: f32,
    pub ram_preview: bool,
    pub warmup: bool,
}

pub struct Track {
    pub name: String,
    pub track_type: TrackType,
    pub visible: bool,
    pub muted: bool,
    pub solo: bool,
    pub expanded: bool,
    pub clips: Vec<Clip>,
}

#[derive(PartialEq, Clone)]
pub enum TrackType {
    Video,
    Audio,
}

pub struct Clip {
    pub name: String,
    pub start: f32,
    pub duration: f32,
    pub color: Color32,
}

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            playing: false,
            looping: false,
            current_time: 16.09,
            total_duration: 599.31,
            zoom: 1.0,
            tracks: vec![
                Track {
                    name: "Video 2".into(),
                    track_type: TrackType::Video,
                    visible: true,
                    muted: false,
                    solo: false,
                    expanded: true,
                    clips: {
                        let mut clips = Vec::new();
                        // Multiple short clips from 0s to ~17s, each ~1s
                        for i in 0..17 {
                            clips.push(Clip {
                                name: "O...".into(),
                                start: i as f32 * 1.0,
                                duration: 0.7,
                                color: Color32::from_rgb(0x3a, 0x7b, 0xd5),
                            });
                        }
                        clips
                    },
                },
                Track {
                    name: "Video 1".into(),
                    track_type: TrackType::Video,
                    visible: true,
                    muted: false,
                    solo: false,
                    expanded: true,
                    clips: vec![
                        Clip {
                            name: "Anthropic Found Why AIs Go Insane.mp4".into(),
                            start: 19.0,
                            duration: 10.0,
                            color: Color32::from_rgb(0x2a, 0x5a, 0x9e),
                        },
                    ],
                },
                Track {
                    name: "Audio".into(),
                    track_type: TrackType::Audio,
                    visible: true,
                    muted: false,
                    solo: false,
                    expanded: true,
                    clips: vec![
                        Clip {
                            name: "clip1".into(),
                            start: 0.0,
                            duration: 1.5,
                            color: Color32::from_rgb(0x2e, 0xcc, 0x71),
                        },
                        Clip {
                            name: "clip2".into(),
                            start: 1.8,
                            duration: 1.2,
                            color: Color32::from_rgb(0x2e, 0xcc, 0x71),
                        },
                        Clip {
                            name: "Anthropic Found Why AIs Go Insane.mp4 (Audio)".into(),
                            start: 19.0,
                            duration: 10.0,
                            color: Color32::from_rgb(0x27, 0xae, 0x60),
                        },
                    ],
                },
            ],
            playhead_pos: 0.0,
            comp_tabs: vec![
                "Comp 1".into(),
                "Comp 2".into(),
                "2026-02-09 13-42-01".into(),
                "DK7A5405".into(),
                "Nehmt es mit Humor albania fyp frdic foreyou".into(),
                "NEW FlutterFlow Designer - Complete Overview Tutorial BETTER Than I Expected".into(),
                "rotating_rectangle".into(),
                "Comp 8".into(),
            ],
            active_comp: 7,
            scroll_offset: 0.0,
            ram_preview: false,
            warmup: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COMP_TABS_HEIGHT: f32 = 24.0;
const TRANSPORT_HEIGHT: f32 = 28.0;
const TRACK_HEADER_WIDTH: f32 = 150.0;
const RULER_HEIGHT: f32 = 20.0;
const TRACK_HEIGHT: f32 = 40.0;
const FPS: f32 = 30.0;

const BG_COLOR: Color32 = Color32::from_rgb(0x0f, 0x0f, 0x0f);
const HEADER_BG: Color32 = Color32::from_rgb(0x16, 0x16, 0x16);
const RULER_BG: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x1a);
const RULER_TEXT: Color32 = Color32::from_rgb(0x66, 0x66, 0x66);
const TRACK_SEP: Color32 = Color32::from_rgb(0x0a, 0x0a, 0x0a);
const PLAYHEAD_COLOR: Color32 = Color32::from_rgb(0x2d, 0x8c, 0xeb);
const TRANSPORT_BG: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x1a);
const COMP_TABS_BG: Color32 = Color32::from_rgb(0x1a, 0x1a, 0x1a);
const ACTIVE_COMP_BG: Color32 = Color32::from_rgb(0x25, 0x25, 0x25);
const BTN_HOVER: Color32 = Color32::from_rgb(0x25, 0x25, 0x25);
const ACTIVE_BTN: Color32 = Color32::from_rgb(0x2d, 0x8c, 0xeb);
const ADD_TRACK_TEXT: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format seconds into MM:SS.FF timecode display.
fn format_timecode(seconds: f32) -> String {
    let total_secs = seconds.floor() as u32;
    let frac = seconds - seconds.floor();
    let frames = (frac * FPS).round() as u32;
    let mm = total_secs / 60;
    let ss = total_secs % 60;
    format!("{:02}:{:02}.{:02}", mm, ss, frames)
}

/// Format seconds into ruler label like "00:00.00", "00:05.00".
fn format_ruler_label(seconds: f32) -> String {
    let total_secs = seconds.round() as u32;
    let mm = total_secs / 60;
    let ss = total_secs % 60;
    format!("{:02}:{:02}.00", mm, ss)
}

/// Draw a small transport-style button and return true when clicked.
fn transport_button(ui: &mut egui::Ui, label: &str, active: bool) -> bool {
    let desired = Vec2::new(24.0, 20.0);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
    let painter = ui.painter();

    let bg = if active {
        ACTIVE_BTN
    } else if response.hovered() {
        BTN_HOVER
    } else {
        TRANSPORT_BG
    };

    painter.rect_filled(rect, CornerRadius::same(3), bg);

    let text_color = if active {
        Color32::WHITE
    } else {
        Color32::from_rgb(0xcc, 0xcc, 0xcc)
    };

    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(12.0),
        text_color,
    );

    response.clicked()
}

/// Draw a text-only button (for add track buttons etc.) and return true when clicked.
fn text_button(ui: &mut egui::Ui, label: &str, color: Color32) -> bool {
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        FontId::proportional(11.0),
        color,
    );
    let desired = Vec2::new(galley.size().x + 12.0, 20.0);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
    let painter = ui.painter();

    if response.hovered() {
        painter.rect_filled(rect, CornerRadius::same(3), BTN_HOVER);
    }

    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(11.0),
        color,
    );

    response.clicked()
}

/// Draw a small labeled button (like "Fit", "I", "O") and return true when clicked.
fn small_button(ui: &mut egui::Ui, label: &str) -> bool {
    let desired = Vec2::new(22.0, 18.0);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
    let painter = ui.painter();

    let bg = if response.hovered() {
        BTN_HOVER
    } else {
        Color32::TRANSPARENT
    };

    painter.rect_filled(rect, CornerRadius::same(2), bg);

    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(10.0),
        Color32::from_rgb(0x99, 0x99, 0x99),
    );

    response.clicked()
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn show_timeline(ui: &mut egui::Ui, state: &mut TimelineState) {
    let panel_rect = ui.available_rect_before_wrap();

    // Fill the whole panel background
    ui.painter()
        .rect_filled(panel_rect, CornerRadius::ZERO, BG_COLOR);

    ui.vertical(|ui| {
        ui.set_min_width(panel_rect.width());
        ui.spacing_mut().item_spacing = Vec2::ZERO;

        // 1. Composition tabs bar
        show_comp_tabs(ui, state);

        // 2. Transport controls bar
        show_transport(ui, state);

        // Thin separator
        let sep_rect = ui.allocate_space(Vec2::new(ui.available_width(), 1.0)).1;
        ui.painter().rect_filled(sep_rect, CornerRadius::ZERO, TRACK_SEP);

        // 3. Main timeline area (ruler + track headers + lanes + playhead)
        show_main_area(ui, state);
    });

    // Keep playhead_pos in sync
    if state.total_duration > 0.0 {
        state.playhead_pos = state.current_time / state.total_duration;
    }
}

// ---------------------------------------------------------------------------
// Section 1: Composition tabs bar
// ---------------------------------------------------------------------------

fn show_comp_tabs(ui: &mut egui::Ui, state: &mut TimelineState) {
    let avail_w = ui.available_width();
    let tabs_rect = ui.allocate_space(Vec2::new(avail_w, COMP_TABS_HEIGHT)).1;

    ui.painter()
        .rect_filled(tabs_rect, CornerRadius::ZERO, COMP_TABS_BG);

    // Bottom border
    ui.painter().line_segment(
        [
            Pos2::new(tabs_rect.min.x, tabs_rect.max.y),
            Pos2::new(tabs_rect.max.x, tabs_rect.max.y),
        ],
        Stroke::new(1.0, TRACK_SEP),
    );

    let inner_rect = Rect::from_min_size(
        Pos2::new(tabs_rect.min.x + 2.0, tabs_rect.min.y),
        Vec2::new(avail_w - 4.0, COMP_TABS_HEIGHT),
    );

    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(inner_rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );

    // Use a horizontal scroll area for the tabs
    ScrollArea::horizontal()
        .max_width(inner_rect.width())
        .auto_shrink([false, false])
        .show(&mut child, |scroll_ui| {
            scroll_ui.horizontal(|ui| {
                let mut tab_to_close: Option<usize> = None;

                for (i, tab_name) in state.comp_tabs.iter().enumerate() {
                    let is_active = i == state.active_comp;

                    // Truncate long names for display
                    let display_name: String = if tab_name.len() > 18 {
                        format!("{}...", &tab_name[..15])
                    } else {
                        tab_name.clone()
                    };
                    let label = format!("{} \u{00D7}", display_name);

                    let galley = ui.painter().layout_no_wrap(
                        label.clone(),
                        FontId::proportional(10.0),
                        Color32::WHITE,
                    );
                    let tab_w = galley.size().x + 16.0;
                    let tab_h = COMP_TABS_HEIGHT - 4.0;

                    let (tab_rect, response) =
                        ui.allocate_exact_size(Vec2::new(tab_w, tab_h), egui::Sense::click());

                    let bg = if is_active {
                        ACTIVE_COMP_BG
                    } else if response.hovered() {
                        Color32::from_rgb(0x20, 0x20, 0x20)
                    } else {
                        Color32::TRANSPARENT
                    };

                    ui.painter().rect_filled(
                        tab_rect,
                        CornerRadius::same(3),
                        bg,
                    );

                    let text_color = if is_active {
                        Color32::from_rgb(0xdd, 0xdd, 0xdd)
                    } else {
                        Color32::from_rgb(0x88, 0x88, 0x88)
                    };

                    // Draw tab name
                    let name_pos = Pos2::new(tab_rect.min.x + 6.0, tab_rect.center().y);
                    ui.painter().text(
                        name_pos,
                        Align2::LEFT_CENTER,
                        &display_name,
                        FontId::proportional(10.0),
                        text_color,
                    );

                    // Draw close button (the x)
                    let close_x = tab_rect.max.x - 8.0;
                    let close_center = Pos2::new(close_x, tab_rect.center().y);
                    ui.painter().text(
                        close_center,
                        Align2::CENTER_CENTER,
                        "\u{00D7}",
                        FontId::proportional(10.0),
                        Color32::from_rgb(0x66, 0x66, 0x66),
                    );

                    if response.clicked() {
                        // Check if click was on the close button area
                        if let Some(pos) = response.interact_pointer_pos() {
                            if pos.x > tab_rect.max.x - 16.0 {
                                tab_to_close = Some(i);
                            } else {
                                state.active_comp = i;
                            }
                        }
                    }

                    ui.add_space(1.0);
                }

                // Handle tab close
                if let Some(idx) = tab_to_close {
                    if state.comp_tabs.len() > 1 {
                        state.comp_tabs.remove(idx);
                        if state.active_comp >= state.comp_tabs.len() {
                            state.active_comp = state.comp_tabs.len() - 1;
                        }
                    }
                }
            });
        });
}

// ---------------------------------------------------------------------------
// Section 2: Transport controls bar
// ---------------------------------------------------------------------------

fn show_transport(ui: &mut egui::Ui, state: &mut TimelineState) {
    let avail_w = ui.available_width();
    let transport_rect = ui.allocate_space(Vec2::new(avail_w, TRANSPORT_HEIGHT)).1;

    ui.painter()
        .rect_filled(transport_rect, CornerRadius::ZERO, TRANSPORT_BG);

    // ── Left side controls ──────────────────────────────────────────────────
    let left_rect = Rect::from_min_size(
        Pos2::new(transport_rect.min.x + 4.0, transport_rect.min.y + 2.0),
        Vec2::new(avail_w * 0.7, TRANSPORT_HEIGHT - 4.0),
    );

    let mut left = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(left_rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );

    left.add_space(4.0);

    // Stop button
    if transport_button(&mut left, "\u{23F9}", false) {
        state.playing = false;
        state.current_time = 0.0;
    }

    left.add_space(2.0);

    // Play button (blue when playing)
    let play_label = if state.playing { "\u{23F8}" } else { "\u{25B6}" };
    if transport_button(&mut left, play_label, state.playing) {
        state.playing = !state.playing;
    }

    left.add_space(2.0);

    // Forward button
    if transport_button(&mut left, "\u{23E9}", false) {
        state.current_time = (state.current_time + 5.0).min(state.total_duration);
    }

    left.add_space(8.0);

    // Timecode display
    let tc_current = format_timecode(state.current_time);
    let tc_total = format_timecode(state.total_duration);
    let tc_text = format!("{} / {}", tc_current, tc_total);
    left.label(
        egui::RichText::new(tc_text)
            .font(FontId::monospace(11.0))
            .color(Color32::from_rgb(0xcc, 0xcc, 0xcc)),
    );

    left.add_space(8.0);

    // Zoom + button
    if small_button(&mut left, "+") {
        state.zoom = (state.zoom + 0.1).min(4.0);
    }

    // Zoom - button
    if small_button(&mut left, "\u{2212}") {
        state.zoom = (state.zoom - 0.1).max(0.1);
    }

    left.add_space(4.0);

    // Fit button
    let _ = small_button(&mut left, "Fit");

    left.add_space(4.0);

    // In point
    let _ = small_button(&mut left, "I");

    // Out point
    let _ = small_button(&mut left, "O");

    left.add_space(8.0);

    // RAM OFF indicator
    {
        let indicator_color = if state.ram_preview {
            Color32::from_rgb(0x2e, 0xcc, 0x71)
        } else {
            Color32::from_rgb(0xcc, 0x66, 0x33)
        };
        let (dot_rect, _) = left.allocate_exact_size(Vec2::new(8.0, 8.0), egui::Sense::hover());
        left.painter().circle_filled(dot_rect.center(), 3.0, indicator_color);

        let ram_label = if state.ram_preview { "RAM ON" } else { "RAM OFF" };
        left.label(
            egui::RichText::new(ram_label)
                .font(FontId::proportional(9.0))
                .color(Color32::from_rgb(0x88, 0x88, 0x88)),
        );
    }

    left.add_space(6.0);

    // Warmup indicator
    {
        let warmup_color = if state.warmup {
            Color32::from_rgb(0x2e, 0xcc, 0x71)
        } else {
            Color32::from_rgb(0x66, 0x66, 0x66)
        };
        let (dot_rect, _) = left.allocate_exact_size(Vec2::new(8.0, 8.0), egui::Sense::hover());
        left.painter().circle_filled(dot_rect.center(), 3.0, warmup_color);

        left.label(
            egui::RichText::new("Warmup")
                .font(FontId::proportional(9.0))
                .color(Color32::from_rgb(0x88, 0x88, 0x88)),
        );
    }

    left.add_space(6.0);

    // View dropdown
    let _ = text_button(&mut left, "View \u{25BE}", Color32::from_rgb(0x88, 0x88, 0x88));

    // ── Right side: add track buttons ───────────────────────────────────────
    let right_rect = Rect::from_min_size(
        Pos2::new(transport_rect.max.x - 250.0, transport_rect.min.y + 2.0),
        Vec2::new(246.0, TRANSPORT_HEIGHT - 4.0),
    );

    let mut right = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(right_rect)
            .layout(egui::Layout::right_to_left(egui::Align::Center)),
    );

    right.add_space(4.0);

    // + Text
    if text_button(&mut right, "+ Text", ADD_TRACK_TEXT) {
        // Would add a text layer
    }

    right.add_space(4.0);

    // + Audio Track
    if text_button(&mut right, "+ Audio Track", ADD_TRACK_TEXT) {
        state.tracks.push(Track {
            name: format!("Audio {}", state.tracks.iter().filter(|t| t.track_type == TrackType::Audio).count() + 1),
            track_type: TrackType::Audio,
            visible: true,
            muted: false,
            solo: false,
            expanded: true,
            clips: vec![],
        });
    }

    right.add_space(4.0);

    // + Video Track
    if text_button(&mut right, "+ Video Track", ADD_TRACK_TEXT) {
        state.tracks.insert(0, Track {
            name: format!("Video {}", state.tracks.iter().filter(|t| t.track_type == TrackType::Video).count() + 1),
            track_type: TrackType::Video,
            visible: true,
            muted: false,
            solo: false,
            expanded: true,
            clips: vec![],
        });
    }
}

// ---------------------------------------------------------------------------
// Section 3 + 4: Main area (ruler + track headers + lanes + playhead)
// ---------------------------------------------------------------------------

fn show_main_area(ui: &mut egui::Ui, state: &mut TimelineState) {
    let avail = ui.available_rect_before_wrap();
    let lanes_width = (avail.width() - TRACK_HEADER_WIDTH).max(100.0);
    let track_count = state.tracks.len();
    let tracks_total_height = track_count as f32 * TRACK_HEIGHT;
    let content_height = RULER_HEIGHT + tracks_total_height;

    // pixels per second based on zoom
    let base_pps = lanes_width / 30.0;
    let pps = base_pps * state.zoom;

    // Total content width in pixels (for scrolling)
    let total_content_w = state.total_duration * pps;

    // Allocate the full area
    let area_rect =
        ui.allocate_space(Vec2::new(avail.width(), content_height.max(avail.height()))).1;

    // ── Ruler left column (Time label + F/M buttons) ────────────────────
    let ruler_header_rect = Rect::from_min_size(
        area_rect.min,
        Vec2::new(TRACK_HEADER_WIDTH, RULER_HEIGHT),
    );

    ui.painter()
        .rect_filled(ruler_header_rect, CornerRadius::ZERO, RULER_BG);

    // "Time" label
    ui.painter().text(
        Pos2::new(ruler_header_rect.min.x + 8.0, ruler_header_rect.center().y),
        Align2::LEFT_CENTER,
        "Time",
        FontId::proportional(10.0),
        RULER_TEXT,
    );

    // F and M toggle buttons
    let f_rect = Rect::from_min_size(
        Pos2::new(ruler_header_rect.max.x - 40.0, ruler_header_rect.min.y + 3.0),
        Vec2::new(16.0, 14.0),
    );
    let m_rect = Rect::from_min_size(
        Pos2::new(ruler_header_rect.max.x - 20.0, ruler_header_rect.min.y + 3.0),
        Vec2::new(16.0, 14.0),
    );

    let f_resp = ui.interact(f_rect, ui.id().with("ruler_f_btn"), egui::Sense::click());
    let m_resp = ui.interact(m_rect, ui.id().with("ruler_m_btn"), egui::Sense::click());

    let f_bg = if f_resp.hovered() { BTN_HOVER } else { Color32::TRANSPARENT };
    let m_bg = if m_resp.hovered() { BTN_HOVER } else { Color32::TRANSPARENT };

    ui.painter().rect_filled(f_rect, CornerRadius::same(2), f_bg);
    ui.painter().text(
        f_rect.center(),
        Align2::CENTER_CENTER,
        "F",
        FontId::proportional(9.0),
        Color32::from_rgb(0x66, 0x66, 0x66),
    );

    ui.painter().rect_filled(m_rect, CornerRadius::same(2), m_bg);
    ui.painter().text(
        m_rect.center(),
        Align2::CENTER_CENTER,
        "M",
        FontId::proportional(9.0),
        Color32::from_rgb(0x66, 0x66, 0x66),
    );

    // Ruler bottom border
    ui.painter().line_segment(
        [
            Pos2::new(ruler_header_rect.min.x, ruler_header_rect.max.y),
            Pos2::new(ruler_header_rect.max.x, ruler_header_rect.max.y),
        ],
        Stroke::new(1.0, TRACK_SEP),
    );

    // ── Track headers (left column) ────────────────────────────────────
    let headers_rect = Rect::from_min_size(
        Pos2::new(area_rect.min.x, area_rect.min.y + RULER_HEIGHT),
        Vec2::new(TRACK_HEADER_WIDTH, tracks_total_height),
    );

    draw_track_headers(ui, state, headers_rect);

    // ── Separator line between headers and lanes ───────────────────────
    let sep_x = area_rect.min.x + TRACK_HEADER_WIDTH;
    ui.painter().line_segment(
        [
            Pos2::new(sep_x, area_rect.min.y),
            Pos2::new(sep_x, area_rect.min.y + content_height),
        ],
        Stroke::new(1.0, TRACK_SEP),
    );

    // ── Lanes area (ruler + clips + playhead) via ScrollArea ───────────
    let lanes_origin = Pos2::new(sep_x + 1.0, area_rect.min.y);
    let lanes_rect = Rect::from_min_size(lanes_origin, Vec2::new(lanes_width, content_height));

    let mut lanes_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(lanes_rect)
            .layout(egui::Layout::left_to_right(egui::Align::Min)),
    );

    ScrollArea::horizontal()
        .max_width(lanes_width)
        .auto_shrink([false, false])
        .show(&mut lanes_ui, |scroll_ui| {
            // Reserve the full scrollable width
            let (_resp_id, scroll_rect) = scroll_ui.allocate_space(Vec2::new(
                total_content_w.max(lanes_width),
                content_height,
            ));

            let painter = scroll_ui.painter();
            let origin = scroll_rect.min;

            // ── Ruler ──────────────────────────────────────────────
            let ruler_rect =
                Rect::from_min_size(origin, Vec2::new(scroll_rect.width(), RULER_HEIGHT));
            painter.rect_filled(ruler_rect, CornerRadius::ZERO, RULER_BG);

            draw_ruler(painter, origin, pps, state.total_duration, scroll_rect.width());

            // ── Track lanes background ─────────────────────────────
            let lanes_top = origin.y + RULER_HEIGHT;
            for i in 0..track_count {
                let y = lanes_top + i as f32 * TRACK_HEIGHT;
                let lane_rect = Rect::from_min_size(
                    Pos2::new(origin.x, y),
                    Vec2::new(scroll_rect.width(), TRACK_HEIGHT),
                );

                // Alternate very slightly for readability
                let bg = if i % 2 == 0 {
                    BG_COLOR
                } else {
                    Color32::from_rgb(0x11, 0x11, 0x11)
                };
                painter.rect_filled(lane_rect, CornerRadius::ZERO, bg);

                // Separator line at bottom of each track
                painter.line_segment(
                    [
                        Pos2::new(origin.x, y + TRACK_HEIGHT),
                        Pos2::new(origin.x + scroll_rect.width(), y + TRACK_HEIGHT),
                    ],
                    Stroke::new(1.0, TRACK_SEP),
                );
            }

            // ── Clips ──────────────────────────────────────────────
            for (i, track) in state.tracks.iter().enumerate() {
                if !track.visible || !track.expanded {
                    continue;
                }
                let lane_y = lanes_top + i as f32 * TRACK_HEIGHT;

                for clip in &track.clips {
                    let cx = origin.x + clip.start * pps;
                    let cw = clip.duration * pps;
                    let clip_rect = Rect::from_min_size(
                        Pos2::new(cx, lane_y + 3.0),
                        Vec2::new(cw, TRACK_HEIGHT - 6.0),
                    );

                    painter.rect_filled(clip_rect, CornerRadius::same(4), clip.color);

                    // Slight border for definition
                    painter.rect_stroke(
                        clip_rect,
                        CornerRadius::same(4),
                        Stroke::new(1.0, Color32::from_rgba_premultiplied(255, 255, 255, 18)),
                        egui::StrokeKind::Outside,
                    );

                    // Draw simple waveform for audio clips
                    if track.track_type == TrackType::Audio && cw > 10.0 {
                        draw_waveform(painter, clip_rect);
                    }

                    // Clip name and timecode (clipped to rect)
                    let text_rect = clip_rect.shrink2(Vec2::new(4.0, 0.0));
                    if text_rect.width() > 20.0 {
                        // Clip name
                        let display_name: String = if clip.name.len() > 12 && cw < 100.0 {
                            format!("{}...", &clip.name[..clip.name.len().min(9)])
                        } else {
                            clip.name.clone()
                        };
                        painter.with_clip_rect(text_rect).text(
                            Pos2::new(text_rect.min.x + 2.0, text_rect.min.y + 8.0),
                            Align2::LEFT_CENTER,
                            &display_name,
                            FontId::proportional(9.0),
                            Color32::from_rgba_premultiplied(255, 255, 255, 200),
                        );

                        // Duration timecode
                        let dur_tc = format_timecode(clip.duration);
                        painter.with_clip_rect(text_rect).text(
                            Pos2::new(text_rect.min.x + 2.0, text_rect.max.y - 8.0),
                            Align2::LEFT_CENTER,
                            &dur_tc,
                            FontId::monospace(8.0),
                            Color32::from_rgba_premultiplied(255, 255, 255, 120),
                        );
                    }
                }
            }

            // ── Playhead ───────────────────────────────────────────
            let ph_x = origin.x + state.current_time * pps;
            if ph_x >= origin.x && ph_x <= origin.x + scroll_rect.width() {
                // Vertical line spanning ruler + tracks
                painter.line_segment(
                    [
                        Pos2::new(ph_x, origin.y),
                        Pos2::new(ph_x, lanes_top + tracks_total_height),
                    ],
                    Stroke::new(2.0, PLAYHEAD_COLOR),
                );

                // Small triangle at top of ruler
                let tri_size = 6.0;
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        Pos2::new(ph_x, origin.y + tri_size + 2.0),
                        Pos2::new(ph_x - tri_size, origin.y),
                        Pos2::new(ph_x + tri_size, origin.y),
                    ],
                    PLAYHEAD_COLOR,
                    Stroke::NONE,
                ));

                // Playhead time label at the top
                let ph_tc = format_timecode(state.current_time);
                painter.text(
                    Pos2::new(ph_x + 4.0, origin.y + 10.0),
                    Align2::LEFT_CENTER,
                    &ph_tc,
                    FontId::monospace(8.0),
                    PLAYHEAD_COLOR,
                );
            }

            // ── Click to seek ──────────────────────────────────────
            let interact_rect = Rect::from_min_size(
                origin,
                Vec2::new(scroll_rect.width(), RULER_HEIGHT),
            );
            let ruler_response = scroll_ui.interact(
                interact_rect,
                scroll_ui.id().with("ruler_seek"),
                egui::Sense::click(),
            );
            if ruler_response.clicked() {
                if let Some(pos) = ruler_response.interact_pointer_pos() {
                    let new_time = ((pos.x - origin.x) / pps).clamp(0.0, state.total_duration);
                    state.current_time = new_time;
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Track headers
// ---------------------------------------------------------------------------

fn draw_track_headers(ui: &mut egui::Ui, state: &mut TimelineState, rect: Rect) {
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::ZERO, HEADER_BG);

    let track_count = state.tracks.len();
    for i in 0..track_count {
        let y = rect.min.y + i as f32 * TRACK_HEIGHT;
        let header_rect = Rect::from_min_size(
            Pos2::new(rect.min.x, y),
            Vec2::new(TRACK_HEADER_WIDTH, TRACK_HEIGHT),
        );

        // Separator at bottom
        painter.line_segment(
            [
                Pos2::new(rect.min.x, y + TRACK_HEIGHT),
                Pos2::new(rect.min.x + TRACK_HEADER_WIDTH, y + TRACK_HEIGHT),
            ],
            Stroke::new(1.0, TRACK_SEP),
        );

        // Expand/collapse triangle
        let triangle = if state.tracks[i].expanded {
            "\u{25BC}" // expanded
        } else {
            "\u{25B6}" // collapsed
        };

        let tri_rect = Rect::from_min_size(
            Pos2::new(header_rect.min.x + 4.0, header_rect.center().y - 7.0),
            Vec2::new(14.0, 14.0),
        );

        let tri_resp = ui.interact(tri_rect, ui.id().with(("tri", i)), egui::Sense::click());
        if tri_resp.clicked() {
            state.tracks[i].expanded = !state.tracks[i].expanded;
        }

        painter.text(
            tri_rect.center(),
            Align2::CENTER_CENTER,
            triangle,
            FontId::proportional(8.0),
            Color32::from_rgb(0x88, 0x88, 0x88),
        );

        // Track name
        painter.text(
            Pos2::new(header_rect.min.x + 20.0, header_rect.center().y),
            Align2::LEFT_CENTER,
            &state.tracks[i].name,
            FontId::proportional(11.0),
            Color32::from_rgb(0xcc, 0xcc, 0xcc),
        );

        // Icon buttons on the right side of the header
        let btn_size = 18.0;
        let btn_y = header_rect.center().y - btn_size * 0.5;
        let mut btn_x = header_rect.max.x - 6.0;

        // Visibility eye / Mute speaker (rightmost button)
        btn_x -= btn_size + 2.0;
        if state.tracks[i].track_type == TrackType::Audio {
            // Mute/speaker button for audio tracks
            let mute_rect =
                Rect::from_min_size(Pos2::new(btn_x, btn_y), Vec2::splat(btn_size));
            let mute_resp =
                ui.interact(mute_rect, ui.id().with(("mute", i)), egui::Sense::click());
            if mute_resp.clicked() {
                state.tracks[i].muted = !state.tracks[i].muted;
            }
            let mute_bg = if state.tracks[i].muted {
                Color32::from_rgb(0xcc, 0x33, 0x33)
            } else if mute_resp.hovered() {
                BTN_HOVER
            } else {
                Color32::TRANSPARENT
            };
            painter.rect_filled(mute_rect, CornerRadius::same(3), mute_bg);
            let mute_icon = if state.tracks[i].muted {
                "\u{1F507}"
            } else {
                "\u{1F50A}"
            };
            painter.text(
                mute_rect.center(),
                Align2::CENTER_CENTER,
                mute_icon,
                FontId::proportional(10.0),
                if state.tracks[i].muted {
                    Color32::WHITE
                } else {
                    Color32::from_rgb(0x88, 0x88, 0x88)
                },
            );
        } else {
            // Eye/visibility for video tracks
            let eye_rect =
                Rect::from_min_size(Pos2::new(btn_x, btn_y), Vec2::splat(btn_size));
            let eye_resp =
                ui.interact(eye_rect, ui.id().with(("eye", i)), egui::Sense::click());
            if eye_resp.clicked() {
                state.tracks[i].visible = !state.tracks[i].visible;
            }
            let eye_bg = if eye_resp.hovered() {
                BTN_HOVER
            } else {
                Color32::TRANSPARENT
            };
            painter.rect_filled(eye_rect, CornerRadius::same(3), eye_bg);
            let eye_icon = if state.tracks[i].visible {
                "\u{1F441}"
            } else {
                "\u{2014}"
            };
            painter.text(
                eye_rect.center(),
                Align2::CENTER_CENTER,
                eye_icon,
                FontId::proportional(10.0),
                if state.tracks[i].visible {
                    Color32::from_rgb(0xcc, 0xcc, 0xcc)
                } else {
                    Color32::from_rgb(0x44, 0x44, 0x44)
                },
            );
        }

        // Solo button
        btn_x -= btn_size + 2.0;
        let solo_rect =
            Rect::from_min_size(Pos2::new(btn_x, btn_y), Vec2::splat(btn_size));
        let solo_resp =
            ui.interact(solo_rect, ui.id().with(("solo", i)), egui::Sense::click());
        if solo_resp.clicked() {
            state.tracks[i].solo = !state.tracks[i].solo;
        }
        let solo_bg = if state.tracks[i].solo {
            ACTIVE_BTN
        } else if solo_resp.hovered() {
            BTN_HOVER
        } else {
            Color32::TRANSPARENT
        };
        painter.rect_filled(solo_rect, CornerRadius::same(3), solo_bg);
        painter.text(
            solo_rect.center(),
            Align2::CENTER_CENTER,
            "S",
            FontId::proportional(10.0),
            if state.tracks[i].solo {
                Color32::WHITE
            } else {
                Color32::from_rgb(0x88, 0x88, 0x88)
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Waveform drawing (simple sine-like pattern for audio clips)
// ---------------------------------------------------------------------------

fn draw_waveform(painter: &egui::Painter, clip_rect: Rect) {
    let center_y = clip_rect.center().y;
    let amplitude = (clip_rect.height() * 0.3).min(10.0);
    let step = 3.0_f32;
    let mut x = clip_rect.min.x + 2.0;

    let waveform_color = Color32::from_rgba_premultiplied(255, 255, 255, 40);

    while x < clip_rect.max.x - 2.0 {
        let phase = (x - clip_rect.min.x) * 0.15;
        let h = (phase.sin().abs() * amplitude).max(1.0);

        painter.line_segment(
            [
                Pos2::new(x, center_y - h),
                Pos2::new(x, center_y + h),
            ],
            Stroke::new(1.0, waveform_color),
        );

        x += step;
    }
}

// ---------------------------------------------------------------------------
// Ruler
// ---------------------------------------------------------------------------

fn draw_ruler(painter: &egui::Painter, origin: Pos2, pps: f32, duration: f32, width: f32) {
    // Decide tick interval based on pps so labels do not overlap.
    let major_interval = if pps > 40.0 {
        5.0
    } else if pps > 15.0 {
        10.0
    } else {
        30.0
    };

    let minor_per_major = 5u32;
    let minor_interval = major_interval / minor_per_major as f32;

    let max_time = duration.max(width / pps);

    // Minor ticks
    let mut t = 0.0f32;
    while t <= max_time {
        let x = origin.x + t * pps;
        if x > origin.x + width + 1.0 {
            break;
        }

        let is_major = (t / major_interval).fract().abs() < 0.01
            || (t / major_interval).fract().abs() > 0.99;

        if is_major {
            // Major tick + label
            painter.line_segment(
                [
                    Pos2::new(x, origin.y + RULER_HEIGHT - 8.0),
                    Pos2::new(x, origin.y + RULER_HEIGHT),
                ],
                Stroke::new(1.0, RULER_TEXT),
            );
            painter.text(
                Pos2::new(x + 3.0, origin.y + RULER_HEIGHT * 0.35),
                Align2::LEFT_CENTER,
                format_ruler_label(t),
                FontId::proportional(9.0),
                RULER_TEXT,
            );
        } else {
            // Minor tick
            painter.line_segment(
                [
                    Pos2::new(x, origin.y + RULER_HEIGHT - 4.0),
                    Pos2::new(x, origin.y + RULER_HEIGHT),
                ],
                Stroke::new(1.0, Color32::from_rgb(0x33, 0x33, 0x33)),
            );
        }

        t += minor_interval;
    }

    // Bottom line of ruler
    painter.line_segment(
        [
            Pos2::new(origin.x, origin.y + RULER_HEIGHT),
            Pos2::new(origin.x + width, origin.y + RULER_HEIGHT),
        ],
        Stroke::new(1.0, TRACK_SEP),
    );
}
