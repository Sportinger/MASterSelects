use egui::{Color32, Stroke, Style, Visuals};

// Color palette
const BG_PRIMARY: Color32 = Color32::from_rgb(15, 15, 15);
const BG_SECONDARY: Color32 = Color32::from_rgb(22, 22, 22);
const BG_TERTIARY: Color32 = Color32::from_rgb(30, 30, 30);
const TEXT_PRIMARY: Color32 = Color32::from_rgb(212, 212, 212);
const TEXT_SECONDARY: Color32 = Color32::from_rgb(153, 153, 153);
const ACCENT: Color32 = Color32::from_rgb(45, 140, 235);
const ACCENT_HOVER: Color32 = Color32::from_rgb(58, 154, 245);
const SHADOW: Color32 = Color32::from_rgb(10, 10, 10);
const SEPARATOR: Color32 = Color32::from_rgb(51, 51, 51);
const ERROR_RED: Color32 = Color32::from_rgb(231, 76, 60);

/// Apply the custom dark theme to the egui context.
pub fn apply_theme(ctx: &egui::Context) {
    let mut style = Style::default();
    let mut visuals = Visuals::dark();

    // ── Background fills ───────────────────────────────────────────────
    visuals.window_fill = BG_PRIMARY;
    visuals.panel_fill = BG_SECONDARY;
    visuals.extreme_bg_color = Color32::from_rgb(8, 8, 8);
    visuals.faint_bg_color = BG_TERTIARY;

    // ── Window shadow & stroke ─────────────────────────────────────────
    visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 2],
        blur: 8,
        spread: 0,
        color: Color32::from_rgba_premultiplied(0, 0, 0, 96),
    };
    visuals.window_stroke = Stroke::new(1.0, SHADOW);
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_rgba_premultiplied(0, 0, 0, 120),
    };

    // ── Widget styles ──────────────────────────────────────────────────

    // Noninteractive (labels, static text)
    visuals.widgets.noninteractive.bg_fill = BG_SECONDARY;
    visuals.widgets.noninteractive.weak_bg_fill = BG_SECONDARY;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.0, Color32::TRANSPARENT);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_SECONDARY);
    visuals.widgets.noninteractive.expansion = 0.0;

    // Inactive (interactive widget at rest)
    visuals.widgets.inactive.bg_fill = BG_TERTIARY;
    visuals.widgets.inactive.weak_bg_fill = BG_TERTIARY;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, SEPARATOR);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.inactive.expansion = 0.0;

    // Hovered
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(40, 40, 40);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(40, 40, 40);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::from_rgb(230, 230, 230));
    visuals.widgets.hovered.expansion = 1.0;

    // Active (being clicked / dragged)
    visuals.widgets.active.bg_fill = ACCENT;
    visuals.widgets.active.weak_bg_fill = ACCENT;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT_HOVER);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.expansion = 1.0;

    // Open (e.g. combo box open)
    visuals.widgets.open.bg_fill = Color32::from_rgb(35, 35, 35);
    visuals.widgets.open.weak_bg_fill = Color32::from_rgb(35, 35, 35);
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, ACCENT);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.open.expansion = 0.0;

    // ── Selection ──────────────────────────────────────────────────────
    visuals.selection.bg_fill = Color32::from_rgba_premultiplied(45, 140, 235, 80);
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);

    // ── Hyperlink ──────────────────────────────────────────────────────
    visuals.hyperlink_color = ACCENT_HOVER;

    // ── Separator / borders ────────────────────────────────────────────
    visuals.window_stroke = Stroke::new(1.0, SHADOW);

    // ── Warn / Error colors ────────────────────────────────────────────
    visuals.warn_fg_color = Color32::from_rgb(243, 156, 18);
    visuals.error_fg_color = ERROR_RED;

    // ── Text cursor ────────────────────────────────────────────────────
    visuals.text_cursor.stroke = Stroke::new(2.0, TEXT_PRIMARY);

    // ── Scroll bar (thin, dark) ────────────────────────────────────────
    visuals.interact_cursor = None;
    visuals.resize_corner_size = 8.0;
    visuals.clip_rect_margin = 3.0;

    // ── Striped rows ───────────────────────────────────────────────────
    visuals.striped = true;

    // ── Apply visuals to style ─────────────────────────────────────────
    style.visuals = visuals;

    // ── Spacing ────────────────────────────────────────────────────────
    style.spacing.item_spacing = egui::Vec2::new(8.0, 4.0);
    style.spacing.button_padding = egui::Vec2::new(8.0, 4.0);
    style.spacing.indent = 18.0;
    style.spacing.interact_size = egui::Vec2::new(40.0, 20.0);
    style.spacing.slider_width = 120.0;
    style.spacing.combo_width = 120.0;
    style.spacing.text_edit_width = 200.0;
    style.spacing.icon_width = 14.0;
    style.spacing.icon_width_inner = 10.0;
    style.spacing.icon_spacing = 4.0;
    style.spacing.menu_margin = egui::Margin::same(4);
    style.spacing.scroll = egui::style::ScrollStyle {
        bar_width: 6.0,
        handle_min_length: 20.0,
        bar_inner_margin: 2.0,
        bar_outer_margin: 2.0,
        floating: true,
        foreground_color: false,
        ..Default::default()
    };

    // ── Animation ──────────────────────────────────────────────────────
    style.animation_time = 0.1;

    // ── Apply to context ───────────────────────────────────────────────
    ctx.set_style(style);

    // ── Default font size ──────────────────────────────────────────────
    let fonts = egui::FontDefinitions::default();
    // Ensure default proportional and monospace fonts are at size 12.0
    // We use the text styles to control sizes more reliably.
    ctx.set_fonts(fonts);

    // Override text styles with our desired sizes
    use egui::{FontFamily, FontId, TextStyle};
    let mut text_styles = std::collections::BTreeMap::new();
    text_styles.insert(TextStyle::Small, FontId::new(10.0, FontFamily::Proportional));
    text_styles.insert(TextStyle::Body, FontId::new(12.0, FontFamily::Proportional));
    text_styles.insert(TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace));
    text_styles.insert(TextStyle::Button, FontId::new(12.0, FontFamily::Proportional));
    text_styles.insert(TextStyle::Heading, FontId::new(16.0, FontFamily::Proportional));
    ctx.set_style(egui::Style {
        text_styles,
        ..ctx.style().as_ref().clone()
    });
}