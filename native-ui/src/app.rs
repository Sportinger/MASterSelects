use eframe::egui;

use crate::bridge::PreviewBridge;
use crate::engine::EngineOrchestrator;
use crate::media_panel::MediaPanelState;
use crate::preview_panel::PreviewPanelState;
use crate::properties_panel::PropertiesPanelState;
use crate::timeline::TimelineState;
use crate::toolbar::ToolbarState;

use ms_app_state::{AppState, HistoryManager};
use ms_effects::EffectRegistry;
use ms_project::{AutoSaver, ProjectFile, ProjectSettings, RecentProjects};

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
            self.engine.open_file(path.clone()).ok();
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
    // Undo / Redo
    // -----------------------------------------------------------------------

    /// Undo the last action.
    pub fn undo(&mut self) {
        if let Some(snapshot) = self.history.undo() {
            snapshot.restore(&mut self.app_state);
            self.set_status("Undo".to_string());
        }
    }

    /// Redo the last undone action.
    pub fn redo(&mut self) {
        if let Some(snapshot) = self.history.redo() {
            snapshot.restore(&mut self.app_state);
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
    }
}
