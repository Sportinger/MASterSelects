use egui::{self, Color32, CornerRadius, RichText, Stroke, Vec2};

// ---------------------------------------------------------------------------
// Action enum -- polled by app.rs each frame
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum PropertiesAction {
    // Drag batch signals
    DragStart(String),
    DragEnd,
    // Transform
    SetOpacity(f32),
    SetBlendMode(String),
    SetPosition(f32, f32, f32),
    SetScale(f32, f32),
    SetRotation(f32),
    // Effects
    AddEffect(String),
    RemoveEffect(usize),
    ToggleEffect(usize, bool),
    SetEffectParam(usize, usize, f32),
    // Masks
    AddMask,
    RemoveMask(usize),
    ToggleMask(usize, bool),
    SetMaskOpacity(usize, f32),
    SetMaskFeather(usize, f32),
    ToggleMaskInvert(usize),
    // Export
    StartExport,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone)]
pub enum PropertiesTab {
    Export,
    Properties,
    Waveform,
    Histogram,
}

pub struct EffectEntry {
    pub name: String,
    pub enabled: bool,
    pub expanded: bool,
    pub params: Vec<(String, f32, f32, f32)>, // (name, value, min, max)
}

pub struct MaskEntry {
    pub name: String,
    pub enabled: bool,
    pub opacity: f32,
    pub feather: f32,
    pub inverted: bool,
}

// ---------------------------------------------------------------------------
// Panel State
// ---------------------------------------------------------------------------

pub struct PropertiesPanelState {
    pub active_tab: PropertiesTab,
    pub clip_selected: bool,
    // Transform
    pub blend_mode: String,
    pub opacity: f32,
    pub position: [f32; 3],
    pub scale: [f32; 2],
    pub rotation: f32,
    // Effects
    pub effects: Vec<EffectEntry>,
    // Masks
    pub masks: Vec<MaskEntry>,
    // Export
    pub export_container: String,
    pub export_codec: String,
    pub export_width: u32,
    pub export_height: u32,
    pub export_fps: f32,
    pub export_bitrate: f32,
    pub export_progress: f32,
    // Export – audio
    pub export_sample_rate: u32,
    pub export_audio_bitrate: u32,
    // Action queue — polled by app.rs each frame
    pub action_queue: Vec<PropertiesAction>,
}

impl PropertiesPanelState {
    /// Queue an action for app.rs to process this frame.
    pub fn emit(&mut self, action: PropertiesAction) {
        self.action_queue.push(action);
    }

    /// Drain all queued actions (called by app.rs each frame).
    pub fn drain_actions(&mut self) -> Vec<PropertiesAction> {
        std::mem::take(&mut self.action_queue)
    }
}

impl Default for PropertiesPanelState {
    fn default() -> Self {
        Self {
            active_tab: PropertiesTab::Properties,
            clip_selected: false,
            blend_mode: "Normal".to_string(),
            opacity: 100.0,
            position: [960.0, 540.0, 0.0],
            scale: [100.0, 100.0],
            rotation: 0.0,
            effects: vec![],
            masks: vec![],
            export_container: "MP4".to_string(),
            export_codec: "H.264".to_string(),
            export_width: 1920,
            export_height: 1080,
            export_fps: 30.0,
            export_bitrate: 20.0,
            export_progress: 0.0,
            export_sample_rate: 48000,
            export_audio_bitrate: 256,
            action_queue: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

const COL_BG: Color32 = Color32::from_rgb(0x16, 0x16, 0x16);
const COL_SECTION: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
const COL_LABEL: Color32 = Color32::from_rgb(0x88, 0x88, 0x88);
const COL_ACCENT: Color32 = Color32::from_rgb(0x2D, 0x8C, 0xEB);
const COL_WIDGET_BG: Color32 = Color32::from_rgb(0x1e, 0x1e, 0x1e);
const COL_KF_ACTIVE: Color32 = Color32::from_rgb(0xf1, 0xc4, 0x0f);
const COL_SEPARATOR: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
const COL_TAB_INACTIVE: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Small keyframe diamond button. Returns true if clicked.
fn keyframe_button(ui: &mut egui::Ui, id_salt: &str, active: bool) -> bool {
    let color = if active { COL_KF_ACTIVE } else { COL_LABEL };
    let resp = ui.add(
        egui::Button::new(RichText::new("\u{25C6}").color(color).size(12.0))
            .frame(false)
            .min_size(Vec2::new(18.0, 18.0)),
    );
    let _ = id_salt;
    resp.clicked()
}

fn section_label(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).color(COL_SECTION).strong().size(13.0));
    ui.add_space(2.0);
}

fn prop_label(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).color(COL_LABEL).size(12.0));
}

fn separator(ui: &mut egui::Ui) {
    ui.add_space(4.0);
    let rect = ui.available_rect_before_wrap();
    let y = rect.top();
    ui.painter().line_segment(
        [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
        Stroke::new(1.0, COL_SEPARATOR),
    );
    ui.add_space(6.0);
}

// ---------------------------------------------------------------------------
// Main entry
// ---------------------------------------------------------------------------

pub fn show_properties_panel(ui: &mut egui::Ui, state: &mut PropertiesPanelState) {
    let frame = egui::Frame::NONE
        .fill(COL_BG)
        .inner_margin(egui::Margin::same(8));

    frame.show(ui, |ui| {
        ui.set_min_width(ui.available_width());

        // ── Tab bar ──────────────────────────────────────────────────────
        show_tab_bar(ui, state);
        ui.add_space(8.0);

        // ── Tab content ──────────────────────────────────────────────────
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| match state.active_tab {
                PropertiesTab::Export => show_export_tab(ui, state),
                PropertiesTab::Properties => show_properties_tab(ui, state),
                PropertiesTab::Waveform => show_waveform_tab(ui),
                PropertiesTab::Histogram => show_histogram_tab(ui),
            });
    });
}

// ---------------------------------------------------------------------------
// Tab bar
// ---------------------------------------------------------------------------

fn show_tab_bar(ui: &mut egui::Ui, state: &mut PropertiesPanelState) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let tabs = [
            ("Export", PropertiesTab::Export),
            ("Properties", PropertiesTab::Properties),
            ("Waveform", PropertiesTab::Waveform),
            ("Histogram", PropertiesTab::Histogram),
        ];
        for (label, tab) in &tabs {
            let active = state.active_tab == *tab;
            let text = if active {
                RichText::new(*label)
                    .color(Color32::WHITE)
                    .strong()
                    .size(12.0)
            } else {
                RichText::new(*label).color(COL_LABEL).size(12.0)
            };
            let btn = egui::Button::new(text)
                .fill(if active { COL_ACCENT } else { COL_TAB_INACTIVE })
                .corner_radius(CornerRadius::same(12))
                .min_size(Vec2::new(0.0, 26.0));
            if ui.add(btn).clicked() {
                state.active_tab = tab.clone();
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Properties tab (Transform + Effects + Masks)
// ---------------------------------------------------------------------------

fn show_properties_tab(ui: &mut egui::Ui, state: &mut PropertiesPanelState) {
    if !state.clip_selected {
        // Centered placeholder when no clip is selected
        let available = ui.available_size();
        ui.allocate_space(Vec2::new(0.0, available.y * 0.4));
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new("Select a clip to edit properties")
                    .color(COL_LABEL)
                    .size(13.0),
            );
        });
        return;
    }

    // ── Transform section ────────────────────────────────────────────────
    show_transform_section(ui, state);

    ui.add_space(6.0);
    separator(ui);

    // ── Effects section (collapsible) ────────────────────────────────────
    show_effects_section(ui, state);

    ui.add_space(6.0);
    separator(ui);

    // ── Masks section (collapsible) ──────────────────────────────────────
    show_masks_section(ui, state);
}

// ---------------------------------------------------------------------------
// Transform section
// ---------------------------------------------------------------------------

fn show_transform_section(ui: &mut egui::Ui, state: &mut PropertiesPanelState) {
    section_label(ui, "Transform");
    ui.add_space(4.0);

    // ── Blend Mode ───────────────────────────────────────────────────────
    let prev_blend = state.blend_mode.clone();
    ui.horizontal(|ui| {
        prop_label(ui, "Blend Mode");
        let blend_modes = ["Normal", "Multiply", "Screen", "Overlay", "Add", "Subtract"];
        egui::ComboBox::from_id_salt("blend_mode")
            .selected_text(RichText::new(&state.blend_mode).color(Color32::WHITE))
            .width(ui.available_width() - 8.0)
            .show_ui(ui, |ui| {
                for mode in &blend_modes {
                    ui.selectable_value(
                        &mut state.blend_mode,
                        mode.to_string(),
                        RichText::new(*mode).color(Color32::WHITE),
                    );
                }
            });
    });
    if state.blend_mode != prev_blend {
        state.emit(PropertiesAction::SetBlendMode(state.blend_mode.clone()));
    }

    ui.add_space(4.0);

    // ── Opacity ──────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        prop_label(ui, "Opacity");
        let resp = ui.add(
            egui::Slider::new(&mut state.opacity, 0.0..=100.0)
                .suffix("%")
                .text_color(Color32::WHITE),
        );
        if resp.drag_started() {
            state.emit(PropertiesAction::DragStart("Opacity".to_string()));
        }
        if resp.changed() {
            state.emit(PropertiesAction::SetOpacity(state.opacity));
        }
        if resp.drag_stopped() {
            state.emit(PropertiesAction::DragEnd);
        }
        keyframe_button(ui, "kf_opacity", false);
    });

    ui.add_space(6.0);

    // ── Position ─────────────────────────────────────────────────────────
    egui::Grid::new("pos_grid")
        .num_columns(3)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            for (i, axis) in ["Position X", "Position Y", "Position Z"]
                .iter()
                .enumerate()
            {
                prop_label(ui, axis);
                let resp = ui.add(
                    egui::DragValue::new(&mut state.position[i])
                        .speed(1.0)
                        .range(-10000.0..=10000.0),
                );
                if resp.drag_started() {
                    state.emit(PropertiesAction::DragStart(format!("{axis}")));
                }
                if resp.changed() {
                    state.emit(PropertiesAction::SetPosition(
                        state.position[0],
                        state.position[1],
                        state.position[2],
                    ));
                }
                if resp.drag_stopped() {
                    state.emit(PropertiesAction::DragEnd);
                }
                keyframe_button(ui, &format!("kf_pos_{i}"), false);
                ui.end_row();
            }
        });

    ui.add_space(6.0);

    // ── Scale ────────────────────────────────────────────────────────────
    egui::Grid::new("scale_grid")
        .num_columns(3)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            for (i, axis) in ["Scale X", "Scale Y"].iter().enumerate() {
                prop_label(ui, axis);
                let resp = ui.add(
                    egui::Slider::new(&mut state.scale[i], 0.0..=200.0)
                        .suffix("%")
                        .text_color(Color32::WHITE),
                );
                if resp.drag_started() {
                    state.emit(PropertiesAction::DragStart(format!("{axis}")));
                }
                if resp.changed() {
                    state.emit(PropertiesAction::SetScale(state.scale[0], state.scale[1]));
                }
                if resp.drag_stopped() {
                    state.emit(PropertiesAction::DragEnd);
                }
                keyframe_button(ui, &format!("kf_scale_{i}"), false);
                ui.end_row();
            }
        });

    ui.add_space(6.0);

    // ── Rotation ─────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        prop_label(ui, "Rotation");
        let resp = ui.add(
            egui::Slider::new(&mut state.rotation, -360.0..=360.0)
                .suffix("\u{00B0}")
                .text_color(Color32::WHITE),
        );
        if resp.drag_started() {
            state.emit(PropertiesAction::DragStart("Rotation".to_string()));
        }
        if resp.changed() {
            state.emit(PropertiesAction::SetRotation(state.rotation));
        }
        if resp.drag_stopped() {
            state.emit(PropertiesAction::DragEnd);
        }
        keyframe_button(ui, "kf_rotation", false);
    });
}

// ---------------------------------------------------------------------------
// Effects section (collapsible)
// ---------------------------------------------------------------------------

fn show_effects_section(ui: &mut egui::Ui, state: &mut PropertiesPanelState) {
    egui::CollapsingHeader::new(
        RichText::new("Effects")
            .color(COL_SECTION)
            .strong()
            .size(13.0),
    )
    .default_open(true)
    .show(ui, |ui| {
        // Add Effect button
        if ui
            .add(
                egui::Button::new(RichText::new("+ Add Effect").color(Color32::WHITE).strong())
                    .fill(COL_ACCENT)
                    .corner_radius(CornerRadius::same(4))
                    .min_size(Vec2::new(ui.available_width(), 28.0)),
            )
            .clicked()
        {
            let effect_name = format!("New Effect {}", state.effects.len() + 1);
            state.effects.push(EffectEntry {
                name: effect_name.clone(),
                enabled: true,
                expanded: true,
                params: vec![("Amount".to_string(), 50.0, 0.0, 100.0)],
            });
            state.emit(PropertiesAction::AddEffect(effect_name));
        }

        ui.add_space(6.0);

        let mut delete_index: Option<usize> = None;
        let mut deferred_fx: Vec<PropertiesAction> = Vec::new();

        for (idx, effect) in state.effects.iter_mut().enumerate() {
            ui.push_id(format!("effect_{idx}"), |ui| {
                // Header
                let header_frame = egui::Frame::NONE
                    .fill(COL_WIDGET_BG)
                    .corner_radius(CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(6, 4));

                header_frame.show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Expand arrow
                        let arrow = if effect.expanded {
                            "\u{25BC}"
                        } else {
                            "\u{25B6}"
                        };
                        if ui
                            .add(
                                egui::Button::new(RichText::new(arrow).color(COL_LABEL).size(10.0))
                                    .frame(false),
                            )
                            .clicked()
                        {
                            effect.expanded = !effect.expanded;
                        }

                        // Enable checkbox
                        let prev_enabled = effect.enabled;
                        ui.checkbox(&mut effect.enabled, "");
                        if effect.enabled != prev_enabled {
                            deferred_fx
                                .push(PropertiesAction::ToggleEffect(idx, effect.enabled));
                        }

                        // Name
                        ui.label(RichText::new(&effect.name).color(Color32::WHITE).size(12.0));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("\u{2715}").color(COL_LABEL).size(12.0),
                                    )
                                    .frame(false),
                                )
                                .clicked()
                            {
                                delete_index = Some(idx);
                            }
                        });
                    });
                });

                // Params (when expanded)
                if effect.expanded {
                    let params_frame = egui::Frame::NONE
                        .fill(Color32::from_rgb(0x1a, 0x1a, 0x1a))
                        .inner_margin(egui::Margin::symmetric(12, 6));

                    params_frame.show(ui, |ui| {
                        for (param_idx, (name, value, min, max)) in
                            effect.params.iter_mut().enumerate()
                        {
                            ui.horizontal(|ui| {
                                prop_label(ui, name);
                                let resp = ui.add(
                                    egui::Slider::new(value, *min..=*max)
                                        .text_color(Color32::WHITE),
                                );
                                if resp.drag_started() {
                                    deferred_fx.push(PropertiesAction::DragStart(format!(
                                        "Effect {} {}",
                                        idx, name
                                    )));
                                }
                                if resp.changed() {
                                    deferred_fx.push(PropertiesAction::SetEffectParam(
                                        idx, param_idx, *value,
                                    ));
                                }
                                if resp.drag_stopped() {
                                    deferred_fx.push(PropertiesAction::DragEnd);
                                }
                            });
                        }
                    });
                }

                ui.add_space(4.0);
            });
        }

        for action in deferred_fx {
            state.emit(action);
        }

        if let Some(idx) = delete_index {
            state.effects.remove(idx);
            state.emit(PropertiesAction::RemoveEffect(idx));
        }
    });
}

// ---------------------------------------------------------------------------
// Masks section (collapsible)
// ---------------------------------------------------------------------------

fn show_masks_section(ui: &mut egui::Ui, state: &mut PropertiesPanelState) {
    egui::CollapsingHeader::new(
        RichText::new("Masks")
            .color(COL_SECTION)
            .strong()
            .size(13.0),
    )
    .default_open(true)
    .show(ui, |ui| {
        // Add Mask button
        if ui
            .add(
                egui::Button::new(RichText::new("+ Add Mask").color(Color32::WHITE).strong())
                    .fill(COL_ACCENT)
                    .corner_radius(CornerRadius::same(4))
                    .min_size(Vec2::new(ui.available_width(), 28.0)),
            )
            .clicked()
        {
            state.masks.push(MaskEntry {
                name: format!("Mask {}", state.masks.len() + 1),
                enabled: true,
                opacity: 100.0,
                feather: 0.0,
                inverted: false,
            });
            state.emit(PropertiesAction::AddMask);
        }

        ui.add_space(6.0);

        let mut delete_index: Option<usize> = None;
        let mut deferred_mask: Vec<PropertiesAction> = Vec::new();

        for (idx, mask) in state.masks.iter_mut().enumerate() {
            ui.push_id(format!("mask_{idx}"), |ui| {
                let mask_frame = egui::Frame::NONE
                    .fill(COL_WIDGET_BG)
                    .corner_radius(CornerRadius::same(4))
                    .inner_margin(egui::Margin::symmetric(8, 6));

                mask_frame.show(ui, |ui| {
                    // Header
                    ui.horizontal(|ui| {
                        let prev_enabled = mask.enabled;
                        ui.checkbox(&mut mask.enabled, "");
                        if mask.enabled != prev_enabled {
                            deferred_mask
                                .push(PropertiesAction::ToggleMask(idx, mask.enabled));
                        }
                        ui.label(
                            RichText::new(&mask.name)
                                .color(Color32::WHITE)
                                .strong()
                                .size(12.0),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("\u{2715}").color(COL_LABEL).size(12.0),
                                    )
                                    .frame(false),
                                )
                                .clicked()
                            {
                                delete_index = Some(idx);
                            }
                        });
                    });

                    ui.add_space(4.0);

                    // Opacity
                    ui.horizontal(|ui| {
                        prop_label(ui, "Opacity");
                        let resp = ui.add(
                            egui::Slider::new(&mut mask.opacity, 0.0..=100.0)
                                .suffix("%")
                                .text_color(Color32::WHITE),
                        );
                        if resp.drag_started() {
                            deferred_mask.push(PropertiesAction::DragStart(format!(
                                "Mask {} Opacity",
                                idx
                            )));
                        }
                        if resp.changed() {
                            deferred_mask
                                .push(PropertiesAction::SetMaskOpacity(idx, mask.opacity));
                        }
                        if resp.drag_stopped() {
                            deferred_mask.push(PropertiesAction::DragEnd);
                        }
                    });

                    // Feather
                    ui.horizontal(|ui| {
                        prop_label(ui, "Feather");
                        let resp = ui.add(
                            egui::Slider::new(&mut mask.feather, 0.0..=50.0)
                                .text_color(Color32::WHITE),
                        );
                        if resp.drag_started() {
                            deferred_mask.push(PropertiesAction::DragStart(format!(
                                "Mask {} Feather",
                                idx
                            )));
                        }
                        if resp.changed() {
                            deferred_mask
                                .push(PropertiesAction::SetMaskFeather(idx, mask.feather));
                        }
                        if resp.drag_stopped() {
                            deferred_mask.push(PropertiesAction::DragEnd);
                        }
                    });

                    // Inverted
                    ui.horizontal(|ui| {
                        prop_label(ui, "Inverted");
                        let prev_inverted = mask.inverted;
                        ui.checkbox(&mut mask.inverted, "");
                        if mask.inverted != prev_inverted {
                            deferred_mask.push(PropertiesAction::ToggleMaskInvert(idx));
                        }
                    });
                });

                ui.add_space(4.0);
            });
        }

        for action in deferred_mask {
            state.emit(action);
        }

        if let Some(idx) = delete_index {
            state.masks.remove(idx);
            state.emit(PropertiesAction::RemoveMask(idx));
        }
    });
}

// ---------------------------------------------------------------------------
// Export tab
// ---------------------------------------------------------------------------

fn show_export_tab(ui: &mut egui::Ui, state: &mut PropertiesPanelState) {
    // ── Video section ────────────────────────────────────────────────────
    section_label(ui, "Video");
    ui.add_space(4.0);

    // Container
    ui.horizontal(|ui| {
        prop_label(ui, "Container");
        let containers = ["MP4", "WebM", "MOV/ProRes"];
        egui::ComboBox::from_id_salt("export_container")
            .selected_text(RichText::new(&state.export_container).color(Color32::WHITE))
            .show_ui(ui, |ui| {
                for c in &containers {
                    ui.selectable_value(
                        &mut state.export_container,
                        c.to_string(),
                        RichText::new(*c).color(Color32::WHITE),
                    );
                }
            });
    });

    // Codec
    ui.horizontal(|ui| {
        prop_label(ui, "Codec");
        let codecs = ["H.264", "H.265", "VP9", "AV1"];
        egui::ComboBox::from_id_salt("export_codec")
            .selected_text(RichText::new(&state.export_codec).color(Color32::WHITE))
            .show_ui(ui, |ui| {
                for c in &codecs {
                    ui.selectable_value(
                        &mut state.export_codec,
                        c.to_string(),
                        RichText::new(*c).color(Color32::WHITE),
                    );
                }
            });
    });

    ui.add_space(4.0);

    // Resolution
    ui.horizontal(|ui| {
        prop_label(ui, "Resolution");
        let mut w = state.export_width as i32;
        let mut h = state.export_height as i32;
        ui.add(egui::DragValue::new(&mut w).speed(1).range(1..=7680));
        ui.label(RichText::new("\u{D7}").color(COL_LABEL)); // multiplication sign
        ui.add(egui::DragValue::new(&mut h).speed(1).range(1..=4320));
        state.export_width = w.max(1) as u32;
        state.export_height = h.max(1) as u32;
    });

    // Framerate
    ui.horizontal(|ui| {
        prop_label(ui, "Framerate");
        let fps_options = [24.0_f32, 25.0, 30.0, 50.0, 60.0];
        egui::ComboBox::from_id_salt("export_fps")
            .selected_text(
                RichText::new(format!("{}", state.export_fps as u32)).color(Color32::WHITE),
            )
            .show_ui(ui, |ui| {
                for f in &fps_options {
                    ui.selectable_value(
                        &mut state.export_fps,
                        *f,
                        RichText::new(format!("{}", *f as u32)).color(Color32::WHITE),
                    );
                }
            });
    });

    // Video Bitrate
    ui.horizontal(|ui| {
        prop_label(ui, "Video Bitrate");
        ui.add(
            egui::Slider::new(&mut state.export_bitrate, 1.0..=100.0)
                .suffix(" Mbps")
                .text_color(Color32::WHITE),
        );
    });

    ui.add_space(10.0);
    separator(ui);

    // ── Audio section ────────────────────────────────────────────────────
    section_label(ui, "Audio");
    ui.add_space(4.0);

    // Sample Rate
    ui.horizontal(|ui| {
        prop_label(ui, "Sample Rate");
        let rates: [u32; 2] = [44100, 48000];
        egui::ComboBox::from_id_salt("export_sr")
            .selected_text(
                RichText::new(format!("{} Hz", state.export_sample_rate)).color(Color32::WHITE),
            )
            .show_ui(ui, |ui| {
                for r in &rates {
                    ui.selectable_value(
                        &mut state.export_sample_rate,
                        *r,
                        RichText::new(format!("{} Hz", r)).color(Color32::WHITE),
                    );
                }
            });
    });

    // Audio Bitrate
    ui.horizontal(|ui| {
        prop_label(ui, "Audio Bitrate");
        let bitrates: [u32; 4] = [128, 192, 256, 320];
        egui::ComboBox::from_id_salt("export_abr")
            .selected_text(
                RichText::new(format!("{} kbps", state.export_audio_bitrate)).color(Color32::WHITE),
            )
            .show_ui(ui, |ui| {
                for b in &bitrates {
                    ui.selectable_value(
                        &mut state.export_audio_bitrate,
                        *b,
                        RichText::new(format!("{} kbps", b)).color(Color32::WHITE),
                    );
                }
            });
    });

    ui.add_space(10.0);
    separator(ui);

    // ── Export button ────────────────────────────────────────────────────
    if ui
        .add(
            egui::Button::new(
                RichText::new("Export")
                    .color(Color32::WHITE)
                    .strong()
                    .size(14.0),
            )
            .fill(COL_ACCENT)
            .corner_radius(CornerRadius::same(4))
            .min_size(Vec2::new(ui.available_width(), 36.0)),
        )
        .clicked()
    {
        state.emit(PropertiesAction::StartExport);
    }

    ui.add_space(6.0);

    // Progress bar
    let progress_bar = egui::ProgressBar::new(state.export_progress / 100.0)
        .text(
            RichText::new(format!("{:.0}%", state.export_progress))
                .color(Color32::WHITE)
                .size(11.0),
        )
        .fill(COL_ACCENT);
    ui.add(progress_bar);
}

// ---------------------------------------------------------------------------
// Waveform tab
// ---------------------------------------------------------------------------

fn show_waveform_tab(ui: &mut egui::Ui) {
    ui.add_space(4.0);

    let available = ui.available_size();
    let size = Vec2::new(available.x, available.x.min(250.0));
    let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let rect = response.rect;

    // Dark background for scope
    painter.rect_filled(
        rect,
        CornerRadius::same(4),
        Color32::from_rgb(0x0a, 0x0a, 0x0a),
    );
    painter.rect_stroke(
        rect,
        CornerRadius::same(4),
        Stroke::new(1.0, COL_SEPARATOR),
        egui::StrokeKind::Outside,
    );

    draw_waveform(&painter, rect);
}

// ---------------------------------------------------------------------------
// Histogram tab
// ---------------------------------------------------------------------------

fn show_histogram_tab(ui: &mut egui::Ui) {
    ui.add_space(4.0);

    let available = ui.available_size();
    let size = Vec2::new(available.x, available.x.min(250.0));
    let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let rect = response.rect;

    // Dark background for scope
    painter.rect_filled(
        rect,
        CornerRadius::same(4),
        Color32::from_rgb(0x0a, 0x0a, 0x0a),
    );
    painter.rect_stroke(
        rect,
        CornerRadius::same(4),
        Stroke::new(1.0, COL_SEPARATOR),
        egui::StrokeKind::Outside,
    );

    draw_histogram(&painter, rect);
}

// ---------------------------------------------------------------------------
// Drawing functions
// ---------------------------------------------------------------------------

fn draw_waveform(painter: &egui::Painter, rect: egui::Rect) {
    let green = Color32::from_rgba_premultiplied(0x00, 0xDD, 0x44, 180);
    let n = 50;
    let step = rect.width() / n as f32;

    for i in 0..n {
        let x = rect.left() + i as f32 * step + step * 0.5;
        // Deterministic pseudo-random-ish height using simple hash
        let hash = ((i * 7 + 13) * 31) % 100;
        let h = rect.height() * 0.1 + rect.height() * 0.7 * (hash as f32 / 100.0);
        let y_top = rect.bottom() - h;
        painter.line_segment(
            [egui::pos2(x, rect.bottom()), egui::pos2(x, y_top)],
            Stroke::new(2.0, green),
        );
    }
}

fn draw_histogram(painter: &egui::Painter, rect: egui::Rect) {
    let channels: [(Color32, u32); 3] = [
        (Color32::from_rgba_premultiplied(0xFF, 0x00, 0x00, 60), 17),
        (Color32::from_rgba_premultiplied(0x00, 0xFF, 0x00, 60), 23),
        (Color32::from_rgba_premultiplied(0x00, 0x80, 0xFF, 60), 31),
    ];

    let bins = 64;
    let bin_w = rect.width() / bins as f32;

    for (color, seed) in &channels {
        for b in 0..bins {
            let hash = ((b * seed + 7) * 53) % 100;
            // Bell-curve-ish shape
            let center_dist = (b as f32 - bins as f32 / 2.0).abs() / (bins as f32 / 2.0);
            let height_factor = (1.0 - center_dist * center_dist) * 0.8 + 0.05;
            let h = rect.height() * height_factor * (0.5 + 0.5 * hash as f32 / 100.0);
            let x = rect.left() + b as f32 * bin_w;
            let bar_rect =
                egui::Rect::from_min_size(egui::pos2(x, rect.bottom() - h), Vec2::new(bin_w, h));
            painter.rect_filled(bar_rect, CornerRadius::ZERO, *color);
        }
    }
}
