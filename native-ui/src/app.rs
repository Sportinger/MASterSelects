use eframe::egui;

use crate::bridge::PreviewBridge;
use crate::engine::EngineOrchestrator;
use crate::media_panel::{MediaFile, MediaKind, MediaPanelState};
use crate::preview_panel::PreviewPanelState;
use crate::properties_panel::{PropertiesAction, PropertiesPanelState};
use crate::timeline::TimelineState;
use crate::toolbar::ToolbarState;

use ms_app_state::{AppSnapshot, AppState, ClipEffect, ClipMask, HistoryManager};
use ms_effects::EffectRegistry;
use ms_project::{AutoSaver, ProjectFile, ProjectSettings, RecentProjects};

use crate::timeline::{TimelineAction, Track, TrackType};
use crate::toolbar::ToolbarAction;

use std::path::PathBuf;
use std::time::Instant;

pub struct MasterSelectsApp {
    // Existing fields (keep them — panels still use them)
    pub toolbar: ToolbarState,
    pub media_panel: MediaPanelState,
    pub preview: PreviewPanelState,
    pub properties: PropertiesPanelState,
    pub timeline: TimelineState,
    pub left_panel_width: f32,
    pub right_panel_width: f32,
    pub bridge: PreviewBridge,
    pub engine: EngineOrchestrator,

    // Real application state
    pub app_state: AppState,
    pub history: HistoryManager,
    pub project: Option<ProjectFile>,
    pub project_path: Option<PathBuf>,
    pub auto_saver: AutoSaver,
    pub recent_projects: RecentProjects,
    pub effect_registry: EffectRegistry,

    // UI state
    pub status_message: Option<(String, Instant)>,
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

            // Initialize real application state
            app_state: AppState::default(),
            history: HistoryManager::new(50),
            project: Some(ProjectFile::new("Untitled", ProjectSettings::default())),
            project_path: None,
            auto_saver: AutoSaver::new(60),
            recent_projects: RecentProjects::load(),
            effect_registry: EffectRegistry::with_builtins(),

            // UI state
            status_message: None,
        }
    }

    // -----------------------------------------------------------------------
    // File operations
    // -----------------------------------------------------------------------

    /// Open a file dialog and load a media file into the project.
    pub fn open_media_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v"])
            .add_filter("Audio", &["mp3", "wav", "flac", "aac", "ogg"])
            .add_filter("All", &["*"])
            .pick_file()
        {
            self.capture_snapshot("Import media");
            self.engine.open_file(path.clone()).ok();

            // Add file to the media panel
            let file_name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();

            let kind = match ext.as_str() {
                "mp3" | "wav" | "flac" | "aac" | "ogg" => MediaKind::Audio,
                _ => MediaKind::Video,
            };

            // Get metadata from engine if available
            let (duration, resolution, fps) = if let Some(info) = self.engine.file_info() {
                let dur_secs = info.duration_secs;
                let mm = dur_secs as u32 / 60;
                let ss = dur_secs as u32 % 60;
                (
                    format!("{}:{:02}", mm, ss),
                    format!("{}\u{00D7}{}", info.resolution.width, info.resolution.height),
                    format!("{:.2}", info.fps.as_f64()),
                )
            } else {
                ("--".to_string(), "--".to_string(), "--".to_string())
            };

            // Avoid duplicates
            let already_added = self
                .media_panel
                .files
                .iter()
                .any(|f| f.path == path.display().to_string());

            if !already_added {
                self.media_panel.files.push(MediaFile {
                    name: file_name,
                    path: path.display().to_string(),
                    kind,
                    duration,
                    resolution,
                    fps,
                });
            }

            self.set_status(format!("Opened: {}", path.display()));
        }
    }

    /// Create a new empty project.
    pub fn new_project(&mut self) {
        self.app_state = AppState::default();
        self.history = HistoryManager::new(50);
        self.project = Some(ProjectFile::new(
            "Untitled",
            ProjectSettings::default(),
        ));
        self.project_path = None;
        self.auto_saver.mark_saved();
        self.toolbar.project_name = "Untitled".to_string();
        self.set_status("New project created".to_string());
    }

    /// Save project to current path (or save-as if no path).
    pub fn save_project(&mut self) {
        if let Some(path) = self.project_path.clone() {
            if let Some(ref project) = self.project {
                match ms_project::save_project(project, &path) {
                    Ok(()) => {
                        self.app_state.mark_clean();
                        self.auto_saver.mark_saved();
                        self.set_status(format!("Saved: {}", path.display()));
                    }
                    Err(e) => self.set_status(format!("Save failed: {}", e)),
                }
            }
        } else {
            self.save_project_as();
        }
    }

    /// Save project with file dialog.
    pub fn save_project_as(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("MasterSelects Project", &["msp"])
            .save_file()
        {
            self.project_path = Some(path.clone());
            if let Some(ref project) = self.project {
                match ms_project::save_project(project, &path) {
                    Ok(()) => {
                        let name = project.name.clone();
                        self.recent_projects.add(&path, &name);
                        let _ = self.recent_projects.save();
                        self.app_state.mark_clean();
                        self.auto_saver.mark_saved();
                        self.set_status(format!("Saved: {}", path.display()));
                    }
                    Err(e) => self.set_status(format!("Save failed: {}", e)),
                }
            }
        }
    }

    /// Open a project file.
    pub fn open_project(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("MasterSelects Project", &["msp"])
            .pick_file()
        {
            match ms_project::load_project(&path) {
                Ok(project) => {
                    let name = project.name.clone();
                    self.toolbar.project_name = name.clone();
                    self.project = Some(project);
                    self.project_path = Some(path.clone());
                    self.recent_projects.add(&path, &name);
                    let _ = self.recent_projects.save();
                    self.history.clear();
                    self.app_state.mark_clean();
                    self.auto_saver.mark_saved();
                    self.set_status("Project loaded".to_string());
                }
                Err(e) => self.set_status(format!("Load failed: {}", e)),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Engine ↔ Timeline sync
    // -----------------------------------------------------------------------

    /// Sync timeline transport state and toolbar from the engine each frame.
    fn sync_timeline_from_engine(&mut self) {
        // Sync playback state
        self.timeline.playing = matches!(
            self.engine.state(),
            crate::engine::EngineState::Playing
        );

        // Sync current time and duration from engine
        self.timeline.current_time = self.engine.current_time_secs() as f32;
        self.timeline.total_duration = self.engine.duration_secs() as f32;

        // Sync toolbar engine info
        self.toolbar.gpu_name = self.engine.gpu_name().to_string();
        self.toolbar.engine_state = match self.engine.state() {
            crate::engine::EngineState::Playing => crate::toolbar::EngineState::Playing,
            crate::engine::EngineState::Paused => crate::toolbar::EngineState::Paused,
            _ => crate::toolbar::EngineState::Idle,
        };

        // Add a clip to the timeline when a file is loaded
        if let Some(info) = self.engine.file_info() {
            // Check if we already have this file as a clip in Video 1
            let has_clip = self
                .timeline
                .tracks
                .iter()
                .any(|t| t.clips.iter().any(|c| c.name == info.file_name));

            if !has_clip {
                // Add clip to the first video track
                if let Some(video_track) = self
                    .timeline
                    .tracks
                    .iter_mut()
                    .find(|t| t.track_type == crate::timeline::TrackType::Video)
                {
                    video_track.clips.push(crate::timeline::Clip {
                        name: info.file_name.clone(),
                        start: 0.0,
                        duration: info.duration_secs as f32,
                        color: egui::Color32::from_rgb(0x2a, 0x5a, 0x9e),
                    });
                }

                // Update timeline duration
                if info.duration_secs as f32 > self.timeline.total_duration {
                    self.timeline.total_duration = info.duration_secs as f32;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Undo / Redo
    // -----------------------------------------------------------------------

    /// Undo the last action.
    pub fn undo(&mut self) {
        if let Some(snapshot) = self.history.undo() {
            snapshot.restore(&mut self.app_state);
            self.sync_properties_from_state();
            self.set_status("Undo".to_string());
        }
    }

    /// Redo the last undone action.
    pub fn redo(&mut self) {
        if let Some(snapshot) = self.history.redo() {
            snapshot.restore(&mut self.app_state);
            self.sync_properties_from_state();
            self.set_status("Redo".to_string());
        }
    }

    // -----------------------------------------------------------------------
    // Status bar helper
    // -----------------------------------------------------------------------

    fn set_status(&mut self, msg: String) {
        self.status_message = Some((msg, Instant::now()));
    }

    /// Return the project display name for the status bar.
    fn project_display_name(&self) -> String {
        if let Some(ref project) = self.project {
            let dirty_marker = if self.app_state.is_dirty { " *" } else { "" };
            format!("{}{}", project.name, dirty_marker)
        } else {
            "No Project".to_string()
        }
    }

    // -----------------------------------------------------------------------
    // Keyboard shortcut processing
    // -----------------------------------------------------------------------

    fn process_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        // Only handle shortcuts when no text input is focused
        if ctx.wants_keyboard_input() {
            return;
        }

        let modifiers = ctx.input(|i| i.modifiers);

        // Ctrl+N - New project
        if modifiers.command && ctx.input(|i| i.key_pressed(egui::Key::N)) {
            self.new_project();
        }

        // Ctrl+O - Open project
        if modifiers.command
            && !modifiers.shift
            && ctx.input(|i| i.key_pressed(egui::Key::O))
        {
            self.open_project();
        }

        // Ctrl+S - Save project
        if modifiers.command
            && !modifiers.shift
            && ctx.input(|i| i.key_pressed(egui::Key::S))
        {
            self.save_project();
        }

        // Ctrl+Shift+S - Save as
        if modifiers.command
            && modifiers.shift
            && ctx.input(|i| i.key_pressed(egui::Key::S))
        {
            self.save_project_as();
        }

        // Ctrl+Z - Undo
        if modifiers.command
            && !modifiers.shift
            && ctx.input(|i| i.key_pressed(egui::Key::Z))
        {
            self.undo();
        }

        // Ctrl+Shift+Z or Ctrl+Y - Redo
        if modifiers.command
            && modifiers.shift
            && ctx.input(|i| i.key_pressed(egui::Key::Z))
        {
            self.redo();
        }
        if modifiers.command && ctx.input(|i| i.key_pressed(egui::Key::Y)) {
            self.redo();
        }

        // Space - Play/Pause toggle
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.engine.toggle_play_pause();
        }

        // Ctrl+I - Import media file
        if modifiers.command && ctx.input(|i| i.key_pressed(egui::Key::I)) {
            self.open_media_file();
        }
    }

    // -----------------------------------------------------------------------
    // Auto-save check
    // -----------------------------------------------------------------------

    fn check_auto_save(&mut self) {
        // Sync dirty state from app_state to auto_saver
        if self.app_state.is_dirty {
            self.auto_saver.mark_dirty();
        }

        if self.auto_saver.should_save() {
            if let Some(ref path) = self.project_path.clone() {
                if let Some(ref project) = self.project {
                    if ms_project::save_project(project, path).is_ok() {
                        self.auto_saver.mark_saved();
                        self.set_status("Auto-saved".to_string());
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Snapshot helper
    // -----------------------------------------------------------------------

    /// Capture a snapshot of the current app state and push it to history.
    /// Call this BEFORE the mutation so undo restores the pre-mutation state.
    fn capture_snapshot(&mut self, label: &str) {
        let snapshot = AppSnapshot::capture(&self.app_state);
        self.history.push(label, snapshot);
        self.app_state.mark_dirty();
    }

    // -----------------------------------------------------------------------
    // Selected-clip mutation helpers
    // -----------------------------------------------------------------------

    /// Apply a mutation to the first selected clip in AppState.
    /// Captures a history snapshot before the mutation and marks the project dirty.
    fn apply_to_selected_clip(
        &mut self,
        label: &str,
        f: impl FnOnce(&mut ms_app_state::ClipState),
    ) {
        let selected = self.app_state.selection.selected_clips().to_vec();
        if let Some(clip_id) = selected.first() {
            // Capture snapshot before the change for undo
            let snapshot = AppSnapshot::capture(&self.app_state);
            self.history.push(label, snapshot);

            if let Some(clip) = self.app_state.find_clip_mut(clip_id) {
                f(clip);
            }
            self.app_state.mark_dirty();
        }
    }

    /// Apply a mutation to the first selected clip without pushing a history entry.
    /// Used during drag operations where batch grouping handles history.
    fn apply_to_selected_clip_no_history(
        &mut self,
        f: impl FnOnce(&mut ms_app_state::ClipState),
    ) {
        let selected = self.app_state.selection.selected_clips().to_vec();
        if let Some(clip_id) = selected.first() {
            if let Some(clip) = self.app_state.find_clip_mut(clip_id) {
                f(clip);
            }
            self.app_state.mark_dirty();
        }
    }

    // -----------------------------------------------------------------------
    // Sync properties panel from AppState
    // -----------------------------------------------------------------------

    /// Sync the properties panel's local UI state from the selected clip in AppState.
    /// Called after undo/redo or when selection changes.
    fn sync_properties_from_state(&mut self) {
        let selected = self.app_state.selection.selected_clips().to_vec();
        if let Some(clip_id) = selected.first() {
            if let Some((_, clip)) = self.app_state.find_clip(clip_id) {
                self.properties.clip_selected = true;
                self.properties.opacity = clip.opacity * 100.0; // AppState 0-1, UI 0-100
                self.properties.blend_mode = clip.blend_mode.clone();
                self.properties.position = [clip.position[0], clip.position[1], 0.0];
                self.properties.scale = clip.scale;
                self.properties.rotation = clip.rotation;

                // Sync effects
                self.properties.effects = clip
                    .effects
                    .iter()
                    .map(|e| crate::properties_panel::EffectEntry {
                        name: e.name.clone(),
                        enabled: e.enabled,
                        expanded: true,
                        params: e.params.clone(),
                    })
                    .collect();

                // Sync masks
                self.properties.masks = clip
                    .masks
                    .iter()
                    .map(|m| crate::properties_panel::MaskEntry {
                        name: m.name.clone(),
                        enabled: m.enabled,
                        opacity: m.opacity,
                        feather: m.feather,
                        inverted: m.inverted,
                    })
                    .collect();
            } else {
                self.properties.clip_selected = false;
            }
        } else {
            self.properties.clip_selected = false;
        }
    }

    // -----------------------------------------------------------------------
    // Action processing (polled each frame after panels render)
    // -----------------------------------------------------------------------

    fn process_toolbar_actions(&mut self) {
        let action = self.toolbar.action.take();
        if let Some(action) = action {
            match action {
                ToolbarAction::NewProject => self.new_project(),
                ToolbarAction::OpenProject => self.open_project(),
                ToolbarAction::SaveProject => self.save_project(),
                ToolbarAction::SaveProjectAs => self.save_project_as(),
                ToolbarAction::ImportMedia => self.open_media_file(),
                ToolbarAction::Undo => self.undo(),
                ToolbarAction::Redo => self.redo(),
                ToolbarAction::ExportStart => {
                    self.set_status("Export started".to_string());
                }
                ToolbarAction::Play => self.engine.toggle_play_pause(),
                ToolbarAction::Pause => self.engine.toggle_play_pause(),
                ToolbarAction::Stop => {
                    self.set_status("Stopped".to_string());
                }
            }
        }
    }

    fn process_timeline_actions(&mut self) {
        let action = self.timeline.action.take();
        if let Some(action) = action {
            match action {
                TimelineAction::AddVideoTrack => {
                    self.capture_snapshot("Add video track");
                    let count = self
                        .timeline
                        .tracks
                        .iter()
                        .filter(|t| t.track_type == TrackType::Video)
                        .count();
                    self.timeline.tracks.insert(
                        0,
                        Track {
                            name: format!("Video {}", count + 1),
                            track_type: TrackType::Video,
                            visible: true,
                            muted: false,
                            solo: false,
                            expanded: true,
                            clips: vec![],
                        },
                    );
                    self.set_status("Added video track".to_string());
                }
                TimelineAction::AddAudioTrack => {
                    self.capture_snapshot("Add audio track");
                    let count = self
                        .timeline
                        .tracks
                        .iter()
                        .filter(|t| t.track_type == TrackType::Audio)
                        .count();
                    self.timeline.tracks.push(Track {
                        name: format!("Audio {}", count + 1),
                        track_type: TrackType::Audio,
                        visible: true,
                        muted: false,
                        solo: false,
                        expanded: true,
                        clips: vec![],
                    });
                    self.set_status("Added audio track".to_string());
                }
                TimelineAction::PlayPause => {
                    self.engine.toggle_play_pause();
                }
                TimelineAction::Stop => {
                    self.engine.stop();
                    self.set_status("Stopped".to_string());
                }
                TimelineAction::Seek(time) => {
                    self.engine.seek(time as f64);
                }
            }
        }
    }

    fn process_properties_actions(&mut self) {
        let actions = self.properties.drain_actions();
        for action in actions {
            match action {
                PropertiesAction::DragStart(label) => {
                    let snapshot = AppSnapshot::capture(&self.app_state);
                    self.history.start_batch(&label, snapshot);
                }
                PropertiesAction::DragEnd => {
                    self.history.end_batch();
                    self.app_state.mark_dirty();
                }
                PropertiesAction::SetOpacity(val) => {
                    if self.history.is_batching() {
                        self.apply_to_selected_clip_no_history(|clip| {
                            clip.opacity = val / 100.0; // UI 0-100, AppState 0-1
                        });
                    } else {
                        self.apply_to_selected_clip("Set Opacity", |clip| {
                            clip.opacity = val / 100.0;
                        });
                    }
                }
                PropertiesAction::SetBlendMode(mode) => {
                    self.apply_to_selected_clip("Set Blend Mode", |clip| {
                        clip.blend_mode = mode;
                    });
                }
                PropertiesAction::SetPosition(x, y, _z) => {
                    if self.history.is_batching() {
                        self.apply_to_selected_clip_no_history(|clip| {
                            clip.position = [x, y];
                        });
                    } else {
                        self.apply_to_selected_clip("Set Position", |clip| {
                            clip.position = [x, y];
                        });
                    }
                }
                PropertiesAction::SetScale(sx, sy) => {
                    if self.history.is_batching() {
                        self.apply_to_selected_clip_no_history(|clip| {
                            clip.scale = [sx, sy];
                        });
                    } else {
                        self.apply_to_selected_clip("Set Scale", |clip| {
                            clip.scale = [sx, sy];
                        });
                    }
                }
                PropertiesAction::SetRotation(deg) => {
                    if self.history.is_batching() {
                        self.apply_to_selected_clip_no_history(|clip| {
                            clip.rotation = deg;
                        });
                    } else {
                        self.apply_to_selected_clip("Set Rotation", |clip| {
                            clip.rotation = deg;
                        });
                    }
                }
                PropertiesAction::AddEffect(name) => {
                    self.apply_to_selected_clip("Add Effect", |clip| {
                        clip.effects.push(ClipEffect {
                            name,
                            enabled: true,
                            params: vec![("Amount".to_string(), 50.0, 0.0, 100.0)],
                        });
                    });
                }
                PropertiesAction::RemoveEffect(idx) => {
                    self.apply_to_selected_clip("Remove Effect", |clip| {
                        if idx < clip.effects.len() {
                            clip.effects.remove(idx);
                        }
                    });
                }
                PropertiesAction::ToggleEffect(idx, enabled) => {
                    self.apply_to_selected_clip("Toggle Effect", |clip| {
                        if let Some(effect) = clip.effects.get_mut(idx) {
                            effect.enabled = enabled;
                        }
                    });
                }
                PropertiesAction::SetEffectParam(effect_idx, param_idx, value) => {
                    self.apply_to_selected_clip("Set Effect Param", |clip| {
                        if let Some(effect) = clip.effects.get_mut(effect_idx) {
                            if let Some(param) = effect.params.get_mut(param_idx) {
                                param.1 = value;
                            }
                        }
                    });
                }
                PropertiesAction::AddMask => {
                    self.apply_to_selected_clip("Add Mask", |clip| {
                        let idx = clip.masks.len() + 1;
                        clip.masks.push(ClipMask {
                            name: format!("Mask {idx}"),
                            enabled: true,
                            opacity: 100.0,
                            feather: 0.0,
                            inverted: false,
                        });
                    });
                }
                PropertiesAction::RemoveMask(idx) => {
                    self.apply_to_selected_clip("Remove Mask", |clip| {
                        if idx < clip.masks.len() {
                            clip.masks.remove(idx);
                        }
                    });
                }
                PropertiesAction::ToggleMask(idx, enabled) => {
                    self.apply_to_selected_clip("Toggle Mask", |clip| {
                        if let Some(mask) = clip.masks.get_mut(idx) {
                            mask.enabled = enabled;
                        }
                    });
                }
                PropertiesAction::SetMaskOpacity(idx, val) => {
                    self.apply_to_selected_clip("Set Mask Opacity", |clip| {
                        if let Some(mask) = clip.masks.get_mut(idx) {
                            mask.opacity = val;
                        }
                    });
                }
                PropertiesAction::SetMaskFeather(idx, val) => {
                    self.apply_to_selected_clip("Set Mask Feather", |clip| {
                        if let Some(mask) = clip.masks.get_mut(idx) {
                            mask.feather = val;
                        }
                    });
                }
                PropertiesAction::ToggleMaskInvert(idx) => {
                    self.apply_to_selected_clip("Toggle Mask Invert", |clip| {
                        if let Some(mask) = clip.masks.get_mut(idx) {
                            mask.inverted = !mask.inverted;
                        }
                    });
                }
                PropertiesAction::StartExport => {
                    self.set_status("Export started".to_string());
                    // TODO: wire to actual export pipeline
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Status bar colors
// ---------------------------------------------------------------------------

const STATUS_BG: egui::Color32 = egui::Color32::from_rgb(0x12, 0x12, 0x12);
const STATUS_TEXT: egui::Color32 = egui::Color32::from_rgb(0x88, 0x88, 0x88);
const STATUS_MSG_COLOR: egui::Color32 = egui::Color32::from_rgb(0x4e, 0xcd, 0xc4);
const STATUS_GPU_COLOR: egui::Color32 = egui::Color32::from_rgb(0x2e, 0xcc, 0x71);

impl eframe::App for MasterSelectsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1. Process keyboard shortcuts before anything else
        self.process_keyboard_shortcuts(ctx);

        // 2. Check auto-save
        self.check_auto_save();

        // 3. Pump engine frames
        self.engine.update(ctx, &mut self.bridge);

        // 4. Toolbar at top
        crate::toolbar::show_toolbar(ctx, &mut self.toolbar);

        // 5. Status bar at the very bottom
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .frame(
                egui::Frame::NONE
                    .fill(STATUS_BG)
                    .inner_margin(egui::Margin::symmetric(8, 2)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.spacing_mut().item_spacing.x = 16.0;

                    // Status message (fades after 3 seconds)
                    if let Some((ref msg, when)) = self.status_message {
                        let elapsed = when.elapsed().as_secs_f32();
                        if elapsed < 3.0 {
                            let alpha = if elapsed > 2.0 {
                                ((3.0 - elapsed) * 255.0) as u8
                            } else {
                                255
                            };
                            let color = egui::Color32::from_rgba_unmultiplied(
                                STATUS_MSG_COLOR.r(),
                                STATUS_MSG_COLOR.g(),
                                STATUS_MSG_COLOR.b(),
                                alpha,
                            );
                            ui.label(
                                egui::RichText::new(msg).color(color).size(11.0),
                            );
                            // Request repaint while message is fading
                            ctx.request_repaint();
                        }
                    }

                    // Right-aligned section
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            // GPU / Engine status
                            let engine_label = self.engine.state().label();
                            ui.label(
                                egui::RichText::new(engine_label)
                                    .color(STATUS_GPU_COLOR)
                                    .size(11.0),
                            );

                            ui.label(
                                egui::RichText::new("|")
                                    .color(STATUS_TEXT)
                                    .size(11.0),
                            );

                            // Project name with dirty indicator
                            let project_name = self.project_display_name();
                            ui.label(
                                egui::RichText::new(project_name)
                                    .color(STATUS_TEXT)
                                    .size(11.0),
                            );
                        },
                    );
                });
            });

        // 6. Timeline at bottom (above status bar)
        egui::TopBottomPanel::bottom("timeline_panel")
            .min_height(150.0)
            .default_height(300.0)
            .resizable(true)
            .frame(
                egui::Frame::NONE.fill(egui::Color32::from_rgb(0x0f, 0x0f, 0x0f)),
            )
            .show(ctx, |ui| {
                crate::timeline::show_timeline(ui, &mut self.timeline);
            });

        // 7. Left panel (Media)
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

        // 8. Right panel (Properties)
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
                crate::properties_panel::show_properties_panel(
                    ui,
                    &mut self.properties,
                );
            });

        // 9. Center panel (Preview) - fills remaining space
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(0x0a, 0x0a, 0x0a))
                    .inner_margin(egui::Margin::same(0)),
            )
            .show(ctx, |ui| {
                crate::preview_panel::show_preview_panel(
                    ui,
                    &mut self.preview,
                    &self.bridge,
                );
            });

        // 10. Process panel actions (snapshot captures happen here)
        self.process_toolbar_actions();
        self.process_timeline_actions();
        self.process_properties_actions();

        // 11. Handle media panel import requests
        if self.media_panel.import_requested {
            self.media_panel.import_requested = false;
            self.open_media_file();
        }

        // 12. Sync timeline transport state with engine
        self.sync_timeline_from_engine();
    }
}
