use egui::{self, Color32, CornerRadius, Pos2, Rect, Stroke, Vec2};

use crate::bridge::PreviewBridge;

pub struct PreviewPanelState {
    pub quality: QualityLevel,
    pub edit_mode: bool,
}

#[derive(PartialEq, Clone)]
pub enum QualityLevel {
    Low,
    Medium,
    High,
    Ultra,
}

impl QualityLevel {
    fn label(&self) -> &'static str {
        match self {
            QualityLevel::Low => "Quarter",
            QualityLevel::Medium => "Half",
            QualityLevel::High => "Full",
            QualityLevel::Ultra => "Ultra",
        }
    }

    fn all() -> &'static [QualityLevel] {
        &[
            QualityLevel::Low,
            QualityLevel::Medium,
            QualityLevel::High,
            QualityLevel::Ultra,
        ]
    }
}

impl Default for PreviewPanelState {
    fn default() -> Self {
        Self {
            quality: QualityLevel::High,
            edit_mode: true,
        }
    }
}

pub fn show_preview_panel(
    ui: &mut egui::Ui,
    state: &mut PreviewPanelState,
    bridge: &PreviewBridge,
) {
    let panel_bg = Color32::from_rgb(0x0a, 0x0a, 0x0a);
    let canvas_bg = Color32::from_rgb(0x00, 0x00, 0x00);
    let canvas_border = Color32::from_rgb(0x22, 0x22, 0x22);
    let button_bg = Color32::from_rgb(0x2a, 0x2a, 0x2a);
    let accent_color = Color32::from_rgb(0x2D, 0x8C, 0xEB);
    let stats_green = Color32::from_rgb(0x2e, 0xcc, 0x71);
    let stats_gray = Color32::from_rgb(0x88, 0x88, 0x88);
    let text_color = Color32::from_rgb(0xcc, 0xcc, 0xcc);

    let panel_rect = ui.available_rect_before_wrap();
    ui.painter()
        .rect_filled(panel_rect, CornerRadius::ZERO, panel_bg);

    let full_rect = ui.max_rect();

    // =====================================================================
    // TOP BAR
    // =====================================================================
    let top_bar_height = 32.0;
    let top_bar_rect =
        Rect::from_min_size(full_rect.min, Vec2::new(full_rect.width(), top_bar_height));

    // -- Top-left controls --
    {
        let controls_rect = Rect::from_min_size(
            top_bar_rect.min + Vec2::new(6.0, 4.0),
            Vec2::new(300.0, top_bar_height - 8.0),
        );
        let mut controls_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(controls_rect)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        controls_ui.spacing_mut().item_spacing = Vec2::new(3.0, 0.0);

        // "Edit" toggle button
        let edit_btn_fill = if state.edit_mode {
            accent_color
        } else {
            button_bg
        };
        let edit_btn_text_color = if state.edit_mode {
            Color32::WHITE
        } else {
            text_color
        };
        let edit_btn = egui::Button::new(
            egui::RichText::new("Edit")
                .color(edit_btn_text_color)
                .size(11.0),
        )
        .fill(edit_btn_fill)
        .corner_radius(CornerRadius::same(4));
        if controls_ui
            .add_sized(Vec2::new(42.0, 22.0), edit_btn)
            .clicked()
        {
            state.edit_mode = !state.edit_mode;
        }

        // "Active" dropdown button
        let active_btn = egui::Button::new(
            egui::RichText::new("Active \u{25BE}")
                .color(text_color)
                .size(11.0),
        )
        .fill(button_bg)
        .corner_radius(CornerRadius::same(4));
        controls_ui.add_sized(Vec2::new(60.0, 22.0), active_btn);

        controls_ui.add_space(6.0);

        // "+" zoom button
        let zoom_in_btn = egui::Button::new(egui::RichText::new("+").color(text_color).size(12.0))
            .fill(button_bg)
            .corner_radius(CornerRadius::same(3));
        controls_ui.add_sized(Vec2::new(22.0, 22.0), zoom_in_btn);

        // "-" zoom button
        let zoom_out_btn =
            egui::Button::new(egui::RichText::new("\u{2212}").color(text_color).size(12.0))
                .fill(button_bg)
                .corner_radius(CornerRadius::same(3));
        controls_ui.add_sized(Vec2::new(22.0, 22.0), zoom_out_btn);
    }

    // -- Top-right stats overlay (live data from PreviewBridge) --
    {
        let painter = ui.painter();
        let stats_font = egui::FontId::proportional(11.0);
        let stats_y = top_bar_rect.min.y + (top_bar_height - 14.0) / 2.0;
        let right_edge = full_rect.max.x - 10.0;

        let preview_stats = bridge.stats();
        let fps_text = format!("{:.0} FPS", preview_stats.fps);
        let time_text = format!("{:.1}ms", preview_stats.last_frame_time_ms);
        let res_text = format!("[{}\u{00d7}{}]", preview_stats.width, preview_stats.height,);
        let frame_text = format!("F{}", preview_stats.frames_displayed);

        // Build segments right-to-left so we can position them
        let segments: Vec<(String, Color32)> = vec![
            (fps_text, stats_green),
            ("  ".to_string(), stats_gray),
            (time_text, stats_green),
            ("  ".to_string(), stats_gray),
            (frame_text, stats_gray),
            ("  ".to_string(), stats_gray),
            (res_text, stats_gray),
        ];

        // Measure total width
        let total_width: f32 = segments
            .iter()
            .map(|(text, _)| {
                let galley =
                    painter.layout_no_wrap(text.clone(), stats_font.clone(), Color32::WHITE);
                galley.size().x
            })
            .sum();

        // Draw left-to-right starting from calculated position
        let mut x = right_edge - total_width;
        for (text, color) in &segments {
            let galley = painter.layout_no_wrap(text.clone(), stats_font.clone(), *color);
            let w = galley.size().x;
            painter.galley(Pos2::new(x, stats_y), galley, *color);
            x += w;
        }
    }

    // =====================================================================
    // BOTTOM BAR
    // =====================================================================
    let bottom_bar_height = 32.0;
    let bottom_bar_rect = Rect::from_min_size(
        Pos2::new(full_rect.min.x, full_rect.max.y - bottom_bar_height),
        Vec2::new(full_rect.width(), bottom_bar_height),
    );

    // -- Bottom-left: gear icon --
    {
        let gear_rect = Rect::from_min_size(
            bottom_bar_rect.min + Vec2::new(8.0, 4.0),
            Vec2::new(28.0, bottom_bar_height - 8.0),
        );
        let mut gear_ui = ui.new_child(egui::UiBuilder::new().max_rect(gear_rect).layout(
            egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
        ));
        let gear_btn =
            egui::Button::new(egui::RichText::new("\u{2699}").color(text_color).size(14.0))
                .fill(Color32::TRANSPARENT)
                .corner_radius(CornerRadius::same(3));
        gear_ui.add(gear_btn);
    }

    // -- Bottom-center: quality selector with close button --
    {
        let combo_width = 80.0;
        let close_btn_width = 22.0;
        let group_width = combo_width + close_btn_width + 4.0;
        let group_x = full_rect.center().x - group_width / 2.0;
        let combo_rect = Rect::from_min_size(
            Pos2::new(group_x, bottom_bar_rect.min.y + 4.0),
            Vec2::new(group_width, bottom_bar_height - 8.0),
        );

        let mut combo_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(combo_rect)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        combo_ui.spacing_mut().item_spacing = Vec2::new(2.0, 0.0);

        // Style the combo box
        combo_ui.visuals_mut().widgets.inactive.weak_bg_fill = button_bg;
        combo_ui.visuals_mut().widgets.hovered.weak_bg_fill = Color32::from_rgb(0x35, 0x35, 0x35);
        combo_ui.visuals_mut().widgets.active.weak_bg_fill = button_bg;
        combo_ui.visuals_mut().widgets.inactive.fg_stroke = Stroke::new(1.0, text_color);
        combo_ui.visuals_mut().widgets.hovered.fg_stroke = Stroke::new(1.0, text_color);
        combo_ui.visuals_mut().widgets.inactive.corner_radius = CornerRadius::same(4);
        combo_ui.visuals_mut().widgets.hovered.corner_radius = CornerRadius::same(4);

        egui::ComboBox::from_id_salt("quality_selector")
            .selected_text(
                egui::RichText::new(state.quality.label())
                    .color(text_color)
                    .size(11.0),
            )
            .width(50.0)
            .show_ui(&mut combo_ui, |ui| {
                for q in QualityLevel::all() {
                    ui.selectable_value(&mut state.quality, q.clone(), q.label());
                }
            });

        // Close "x" button
        let close_btn =
            egui::Button::new(egui::RichText::new("\u{00d7}").color(stats_gray).size(12.0))
                .fill(button_bg)
                .corner_radius(CornerRadius::same(3));
        combo_ui.add_sized(Vec2::new(close_btn_width, 22.0), close_btn);
    }

    // =====================================================================
    // CANVAS AREA (center, aspect-ratio-preserving)
    // =====================================================================
    {
        let canvas_padding_x = 12.0;
        let canvas_padding_top = top_bar_height + 4.0;
        let canvas_padding_bottom = bottom_bar_height + 4.0;

        let canvas_available = Rect::from_min_max(
            Pos2::new(
                full_rect.min.x + canvas_padding_x,
                full_rect.min.y + canvas_padding_top,
            ),
            Pos2::new(
                full_rect.max.x - canvas_padding_x,
                full_rect.max.y - canvas_padding_bottom,
            ),
        );

        // Use the bridge texture dimensions for aspect ratio, fallback to 16:9
        let tex_size = bridge.texture_size();
        let aspect_ratio = if tex_size[1] > 0 {
            tex_size[0] as f32 / tex_size[1] as f32
        } else {
            16.0 / 9.0
        };
        let available_width = canvas_available.width();
        let available_height = canvas_available.height();

        let (canvas_w, canvas_h) = if available_width / available_height > aspect_ratio {
            let h = available_height;
            let w = h * aspect_ratio;
            (w, h)
        } else {
            let w = available_width;
            let h = w / aspect_ratio;
            (w, h)
        };

        let canvas_center = canvas_available.center();
        let canvas_rect = Rect::from_center_size(canvas_center, Vec2::new(canvas_w, canvas_h));

        let painter = ui.painter();

        // Black canvas fill (visible as letterbox behind the texture)
        painter.rect_filled(canvas_rect, CornerRadius::ZERO, canvas_bg);

        // Draw the bridge texture if available, otherwise show placeholder
        if let Some(texture_id) = bridge.texture_id() {
            painter.image(
                texture_id,
                canvas_rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        // Subtle border
        painter.rect_stroke(
            canvas_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, canvas_border),
            egui::StrokeKind::Outside,
        );
    }

    // Allocate full space so layout works
    ui.allocate_rect(full_rect, egui::Sense::hover());
}
