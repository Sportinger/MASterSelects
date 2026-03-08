//! yt-dlp download implementation with progress streaming
//!
//! Supports all yt-dlp-compatible platforms: YouTube, TikTok, Instagram, Twitter, etc.
//! Includes deno runtime detection for JavaScript-based extractors.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use futures_util::SinkExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::Command as TokioCommand;
use tokio_tungstenite::tungstenite::protocol::Message;
use tracing::{info, warn};

use crate::protocol::{error_codes, Response};
use crate::utils;

/// Type for sending WebSocket messages (for progress streaming)
pub type WsSender = Arc<tokio::sync::Mutex<
    futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<TcpStream>,
        Message,
    >,
>>;

/// Find yt-dlp executable, checking common install locations
pub fn find_ytdlp() -> Option<PathBuf> {
    // First check if yt-dlp is in PATH
    if let Ok(output) = std::process::Command::new("yt-dlp").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("yt-dlp"));
        }
    }

    // On Windows, check common Python user install locations
    #[cfg(windows)]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            let appdata_path = PathBuf::from(&appdata);
            if let Ok(entries) = std::fs::read_dir(appdata_path.join("Python")) {
                for entry in entries.flatten() {
                    let scripts = entry.path().join("Scripts").join("yt-dlp.exe");
                    if scripts.exists() {
                        return Some(scripts);
                    }
                }
            }
        }

        // Also check LocalAppData (pip --user on some configs)
        if let Ok(localappdata) = std::env::var("LOCALAPPDATA") {
            let local_path = PathBuf::from(&localappdata);
            if let Ok(entries) = std::fs::read_dir(local_path.join("Programs").join("Python")) {
                for entry in entries.flatten() {
                    let scripts = entry.path().join("Scripts").join("yt-dlp.exe");
                    if scripts.exists() {
                        return Some(scripts);
                    }
                }
            }
        }
    }

    None
}

/// Find deno executable for yt-dlp JavaScript runtime
pub fn find_deno() -> Option<PathBuf> {
    // Check if deno is in PATH
    if let Ok(output) = std::process::Command::new("deno").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("deno"));
        }
    }

    // On Windows, check winget install location
    #[cfg(windows)]
    {
        if let Ok(localappdata) = std::env::var("LOCALAPPDATA") {
            let winget_path = PathBuf::from(&localappdata)
                .join("Microsoft")
                .join("WinGet")
                .join("Packages");
            if let Ok(entries) = std::fs::read_dir(&winget_path) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_lowercase();
                    if name.contains("deno") {
                        let deno_exe = entry.path().join("deno.exe");
                        if deno_exe.exists() {
                            return Some(deno_exe);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Get yt-dlp command path
pub fn get_ytdlp_command() -> String {
    find_ytdlp()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "yt-dlp".to_string())
}

/// Build yt-dlp args with deno runtime if available
pub fn get_deno_args() -> Vec<String> {
    if let Some(deno_path) = find_deno() {
        vec![
            "--js-runtimes".to_string(),
            format!("deno:{}", deno_path.to_string_lossy()),
        ]
    } else {
        vec![]
    }
}

/// List available formats for a video URL (supports all yt-dlp platforms)
pub async fn handle_list_formats(id: &str, url: &str) -> Response {
    use std::process::Stdio;

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Response::error(id, error_codes::INVALID_URL, "URL must start with http:// or https://");
    }

    let ytdlp_cmd = get_ytdlp_command();
    let deno_args = get_deno_args();

    let mut cmd = TokioCommand::new(&ytdlp_cmd);
    for arg in &deno_args {
        cmd.arg(arg);
    }
    let result = cmd
        .args(["--dump-json", "--no-playlist", url])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;

    match result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            match serde_json::from_str::<serde_json::Value>(&stdout) {
                Ok(info) => {
                    let title = info.get("title").and_then(|v| v.as_str()).unwrap_or("Unknown");
                    let uploader = info.get("uploader").and_then(|v| v.as_str()).unwrap_or("Unknown");
                    let duration = info.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let thumbnail = info.get("thumbnail").and_then(|v| v.as_str()).unwrap_or("");
                    let platform = info.get("extractor_key").and_then(|v| v.as_str()).unwrap_or("generic");

                    // Parse formats — collect video-only formats by resolution
                    let mut recommendations = Vec::new();
                    if let Some(formats) = info.get("formats").and_then(|v| v.as_array()) {
                        let mut by_height: HashMap<i64, Vec<&serde_json::Value>> = HashMap::new();

                        for fmt in formats {
                            let vcodec = fmt.get("vcodec").and_then(|v| v.as_str()).unwrap_or("none");

                            // Skip audio-only and AV1
                            if vcodec == "none" || vcodec.contains("av01") {
                                continue;
                            }

                            if let Some(height) = fmt.get("height").and_then(|v| v.as_i64()) {
                                if height >= 360 {
                                    by_height.entry(height).or_default().push(fmt);
                                }
                            }
                        }

                        // Get best format per resolution, prefer H.264
                        let mut heights: Vec<_> = by_height.keys().copied().collect();
                        heights.sort_by(|a, b| b.cmp(a));

                        for height in heights.into_iter().take(6) {
                            if let Some(fmts) = by_height.get(&height) {
                                let best = fmts.iter()
                                    .max_by_key(|f| {
                                        let vcodec = f.get("vcodec").and_then(|v| v.as_str()).unwrap_or("");
                                        let tbr = f.get("tbr").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                        let codec_score = if vcodec.contains("avc") { 1000.0 } else { 0.0 };
                                        (codec_score + tbr) as i64
                                    });

                                if let Some(fmt) = best {
                                    let format_id = fmt.get("format_id").and_then(|v| v.as_str()).unwrap_or("");
                                    let vcodec = fmt.get("vcodec").and_then(|v| v.as_str()).unwrap_or("");
                                    let fps = fmt.get("fps").and_then(|v| v.as_f64()).unwrap_or(30.0);
                                    let filesize = fmt.get("filesize").and_then(|v| v.as_i64())
                                        .or_else(|| fmt.get("filesize_approx").and_then(|v| v.as_i64()));

                                    let codec_name = if vcodec.contains("avc") { "H.264" }
                                        else if vcodec.contains("vp9") { "VP9" }
                                        else { vcodec };

                                    recommendations.push(serde_json::json!({
                                        "id": format_id,
                                        "label": format!("{}p {} ({:.0}fps)", height, codec_name, fps),
                                        "resolution": format!("{}p", height),
                                        "vcodec": codec_name,
                                        "acodec": serde_json::Value::Null,
                                        "needsMerge": true,
                                        "filesize": filesize,
                                    }));
                                }
                            }
                        }

                        // Fallback: if no separate video-only formats found (common for TikTok, Instagram),
                        // add a "Best available" option using the best combined format
                        if recommendations.is_empty() {
                            let mut best_combined: Option<&serde_json::Value> = None;
                            let mut best_score: i64 = 0;

                            for fmt in formats {
                                let vcodec = fmt.get("vcodec").and_then(|v| v.as_str()).unwrap_or("none");
                                if vcodec == "none" { continue; }

                                let height = fmt.get("height").and_then(|v| v.as_i64()).unwrap_or(0);
                                let tbr = fmt.get("tbr").and_then(|v| v.as_f64()).unwrap_or(0.0);
                                let score = height * 1000 + tbr as i64;

                                if score > best_score {
                                    best_score = score;
                                    best_combined = Some(fmt);
                                }
                            }

                            if let Some(fmt) = best_combined {
                                let format_id = fmt.get("format_id").and_then(|v| v.as_str()).unwrap_or("best");
                                let height = fmt.get("height").and_then(|v| v.as_i64()).unwrap_or(0);
                                let fps = fmt.get("fps").and_then(|v| v.as_f64()).unwrap_or(30.0);
                                let filesize = fmt.get("filesize").and_then(|v| v.as_i64())
                                    .or_else(|| fmt.get("filesize_approx").and_then(|v| v.as_i64()));

                                let label = if height > 0 {
                                    format!("Best available ({}p, {:.0}fps)", height, fps)
                                } else {
                                    "Best available".to_string()
                                };

                                recommendations.push(serde_json::json!({
                                    "id": format_id,
                                    "label": label,
                                    "resolution": if height > 0 { format!("{}p", height) } else { "?".to_string() },
                                    "vcodec": serde_json::Value::Null,
                                    "acodec": serde_json::Value::Null,
                                    "needsMerge": false,
                                    "filesize": filesize,
                                }));
                            }
                        }
                    }

                    Response::ok(id, serde_json::json!({
                        "title": title,
                        "uploader": uploader,
                        "duration": duration,
                        "thumbnail": thumbnail,
                        "platform": platform,
                        "recommendations": recommendations,
                    }))
                }
                Err(e) => Response::error(id, error_codes::DOWNLOAD_FAILED, format!("Failed to parse yt-dlp output: {}", e)),
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Response::error(id, error_codes::DOWNLOAD_FAILED, stderr.to_string())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Response::error(id, error_codes::YTDLP_NOT_FOUND, "yt-dlp not found. Install with: pip install yt-dlp")
        }
        Err(e) => Response::error(id, error_codes::DOWNLOAD_FAILED, e.to_string()),
    }
}

/// Download a video with progress streaming via WebSocket
pub async fn handle_download(
    id: &str,
    url: &str,
    format_id: Option<&str>,
    output_dir: Option<&str>,
    ws_sender: Option<WsSender>,
) -> Response {
    use std::process::Stdio;

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Response::error(id, error_codes::INVALID_URL, "URL must start with http:// or https://");
    }

    let download_dir = output_dir
        .map(PathBuf::from)
        .unwrap_or_else(utils::get_download_dir);

    if let Err(e) = std::fs::create_dir_all(&download_dir) {
        return Response::error(id, error_codes::PERMISSION_DENIED, format!("Cannot create directory: {}", e));
    }

    info!("Downloading: {} to {:?}", url, download_dir);

    let output_template = download_dir.join("%(title)s.%(ext)s").to_string_lossy().to_string();

    let format_str = if let Some(fid) = format_id {
        format!("{}+bestaudio[ext=m4a]/{}+bestaudio/{}/best", fid, fid, fid)
    } else {
        "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best".to_string()
    };

    let ytdlp_cmd = get_ytdlp_command();
    let deno_args = get_deno_args();

    let mut cmd = TokioCommand::new(&ytdlp_cmd);
    for arg in &deno_args {
        cmd.arg(arg);
    }

    let mut child = match cmd
        .args([
            "-f", &format_str,
            "--merge-output-format", "mp4",
            "-o", &output_template,
            "--print", "after_move:filepath",
            "--no-playlist",
            "--newline",           // Progress on separate lines
            "--progress",
            "--concurrent-fragments", "5", // Download up to 5 fragments in parallel
            "--restrict-filenames", // Replace special chars with ASCII
            "--windows-filenames",  // Windows-safe filenames
            url,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Response::error(id, error_codes::YTDLP_NOT_FOUND, "yt-dlp not found. Install with: pip install yt-dlp");
            }
            Err(e) => {
                return Response::error(id, error_codes::DOWNLOAD_FAILED, e.to_string());
            }
        };

    // Stream stdout for progress (yt-dlp writes [download] progress to stdout with --newline)
    // yt-dlp downloads video+audio separately when merging, so we track phases:
    //   Phase 0 = video (0-80%), Phase 1 = audio (80-95%), Merge = 95-99%
    let stdout = child.stdout.take();
    let mut last_sent_percent: u8 = 0;
    let mut download_phase: u8 = 0; // 0 = first stream (video), 1 = second stream (audio)
    let mut final_filepath: Option<String> = None;
    if let Some(stdout) = stdout {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            // Detect new download phase starting (yt-dlp prints "[download] Destination: ..." for each stream)
            if line.contains("[download] Destination:") || line.contains("[download] Downloading") {
                if download_phase == 0 && last_sent_percent > 50 {
                    // Second stream starting (audio) — switch to phase 1
                    download_phase = 1;
                }
            }

            // Detect merging phase
            if line.contains("[Merger]") || line.contains("Merging") {
                let merge_percent: u8 = 96;
                if merge_percent > last_sent_percent {
                    last_sent_percent = merge_percent;
                    info!("[yt-dlp] Merging streams...");
                    if let Some(ref sender) = ws_sender {
                        let progress_msg = Response::download_progress(id, merge_percent, None, None);
                        let json = serde_json::to_string(&progress_msg).unwrap();
                        let mut sender = sender.lock().await;
                        let _ = sender.send(Message::Text(json)).await;
                    }
                }
                continue;
            }

            // Parse progress from yt-dlp output
            // Format: "[download]  45.2% of 100.00MiB at  5.23MiB/s ETA 00:10"
            // Fragment format: "[download]  45.2% of ~100.00MiB at  5.23MiB/s ETA 00:10 (frag 5/10)"
            if line.contains('%') {
                // Extract percentage
                let mut percent_val: Option<f32> = None;
                if let Some(pct_str) = line.split('%').next() {
                    let pct_part = pct_str.trim().rsplit_once(' ').map(|(_, p)| p).unwrap_or(pct_str.trim());
                    if let Ok(pct) = pct_part.trim().parse::<f32>() {
                        percent_val = Some(pct.min(100.0));
                    }
                }

                if let Some(raw_percent) = percent_val {
                    // Extract speed (e.g. "5.23MiB/s" or "~5.23MiB/s")
                    let speed: Option<String> = if let Some(at_idx) = line.find(" at ") {
                        let after_at = &line[at_idx + 4..];
                        let speed_str = after_at.trim().split_whitespace().next().unwrap_or("");
                        let cleaned = speed_str.trim_start_matches('~');
                        if cleaned.contains("/s") { Some(cleaned.to_string()) } else { None }
                    } else {
                        None
                    };

                    // Extract ETA (e.g. "00:10")
                    let eta: Option<String> = if let Some(eta_idx) = line.find("ETA ") {
                        let after_eta = &line[eta_idx + 4..];
                        let eta_str = after_eta.trim().split_whitespace().next().unwrap_or("");
                        if !eta_str.is_empty() && eta_str != "Unknown" { Some(eta_str.to_string()) } else { None }
                    } else {
                        None
                    };

                    // Map raw percent to overall progress based on phase
                    // Phase 0 (video): 0-80%, Phase 1 (audio): 80-95%
                    let overall = match download_phase {
                        0 => (raw_percent * 0.80) as u8,    // 0% → 0%, 100% → 80%
                        _ => 80 + (raw_percent * 0.15) as u8, // 0% → 80%, 100% → 95%
                    };
                    let overall = overall.min(99);

                    // Send if changed by at least 1%
                    if overall > last_sent_percent {
                        last_sent_percent = overall;
                        info!("[yt-dlp] Phase {} raw={:.1}% overall={}% speed={:?} eta={:?}", download_phase, raw_percent, overall, speed, eta);
                        if let Some(ref sender) = ws_sender {
                            let progress_msg = Response::download_progress(
                                id, overall,
                                speed.as_deref(), eta.as_deref(),
                            );
                            let json = serde_json::to_string(&progress_msg).unwrap();
                            let mut sender = sender.lock().await;
                            let _ = sender.send(Message::Text(json)).await;
                        }
                    }
                }
            } else if line.contains("Downloading") || line.contains("Merging") {
                info!("[yt-dlp] {}", line);
            } else if !line.starts_with('[') && !line.is_empty() {
                // Non-bracketed line = likely the filepath from --print after_move:filepath
                info!("[yt-dlp] Captured output path: {}", line);
                final_filepath = Some(line.trim().to_string());
            }
        }
    }

    // stdout was already consumed above, so just wait for exit status
    let status = child.wait().await;

    match status {
        Ok(s) if s.success() => {
            let output_path = final_filepath.unwrap_or_default();

            if output_path.is_empty() {
                return Response::error(id, error_codes::DOWNLOAD_FAILED, "yt-dlp did not return output path");
            }

            info!("Download complete: {}", output_path);
            Response::ok(id, serde_json::json!({ "path": output_path }))
        }
        Ok(_) => {
            Response::error(id, error_codes::DOWNLOAD_FAILED, "yt-dlp exited with error")
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Response::error(id, error_codes::YTDLP_NOT_FOUND, "yt-dlp not found. Install with: pip install yt-dlp")
        }
        Err(e) => Response::error(id, error_codes::DOWNLOAD_FAILED, e.to_string()),
    }
}
