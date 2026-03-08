"""
Qwen3-VL Video Description Server
Native video understanding using Qwen3-VL-8B via transformers (no Ollama).
Reads video with PyAV, processes frames with temporal position encoding.
Runs on http://localhost:5555
"""

import sys
import os
import time
import re
import itertools
import logging
import gc
import torch
import numpy as np
import av
from PIL import Image
from flask import Flask, request, jsonify
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'server.log'), mode='w'),
    ],
)
log = logging.getLogger('qwen3vl')
log.info("=== Server script starting ===")

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Global model state
model = None
processor = None
model_lock = Lock()

MODEL_DIR = None
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def find_model_dir() -> str:
    """Find the cached model directory."""
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                            "models--Qwen--Qwen3-VL-8B-Instruct", "snapshots")
    if os.path.isdir(hf_cache):
        snapshots = os.listdir(hf_cache)
        if snapshots:
            return os.path.join(hf_cache, snapshots[0])
    return MODEL_ID


def load_model():
    """Load model: CPU first (no accelerate), then move to CUDA."""
    global model, processor

    if model is not None:
        return

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_path = MODEL_DIR or find_model_dir()
    log.info(f"Loading {model_path}...")
    start = time.time()

    # device_map=None avoids accelerate dispatch (segfaults on Windows)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        local_files_only=os.path.isdir(model_path),
    )

    log.info("Moving to CUDA...")
    model = model.cuda()
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=os.path.isdir(model_path))

    elapsed = time.time() - start
    vram = torch.cuda.memory_allocated() / 1024**3
    log.info(f"Model loaded in {elapsed:.1f}s, VRAM: {vram:.1f} GB")


def read_video_frames(video_path: str, target_fps: float = 1.0, max_size: int = 480,
                      max_frames: int = 48) -> dict:
    """Read video frames using PyAV at specified FPS.

    Returns dict with: frames (numpy), video_fps, duration, total_frames, frame_indices
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    video_fps = float(stream.average_rate or stream.rate or 25)
    total_frames = stream.frames or 0
    duration = float(stream.duration * stream.time_base) if stream.duration else 0

    if total_frames == 0 and duration > 0:
        total_frames = int(duration * video_fps)
    elif duration == 0 and total_frames > 0:
        duration = total_frames / video_fps

    frame_interval = max(1, int(video_fps / target_fps))

    w, h = stream.width, stream.height
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w = int(w * scale) & ~1
        h = int(h * scale) & ~1

    frames = []
    frame_indices = []
    idx = 0
    for frame in container.decode(stream):
        if idx % frame_interval == 0:
            img = frame.to_image()
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            frames.append(np.array(img))
            frame_indices.append(idx)
            if len(frames) >= max_frames:
                break
        idx += 1

    container.close()
    actual_duration = idx / video_fps if idx > 0 else duration

    log.info(f"Video: {actual_duration:.1f}s, {len(frames)} frames ({w}x{h})")
    return {
        "frames": frames,
        "video_fps": video_fps,
        "duration": actual_duration,
        "total_frames": total_frames or idx,
        "frame_indices": frame_indices,
    }


def fix_video_grid_thw(inputs: dict) -> dict:
    """Fix video_grid_thw for transformers 5.3.0 Qwen3-VL processor bug.

    The processor creates temporal chunks with vision_start/end markers between them,
    resulting in N separate video token groups in mm_token_type_ids. But video_grid_thw
    only has 1 entry [T, H, W]. get_rope_index iterates once per group and expects
    one grid entry per group, causing StopIteration.

    Fix: split the single [T, H, W] into N entries of [T/N, H, W].
    """
    if 'video_grid_thw' not in inputs or 'mm_token_type_ids' not in inputs:
        return inputs

    types = inputs['mm_token_type_ids'][0].tolist()
    n_video_groups = sum(1 for k, _ in itertools.groupby(types) if k == 2)
    grid = inputs['video_grid_thw']

    if grid.shape[0] == 1 and n_video_groups > 1:
        t, h, w = grid[0].tolist()
        t_per = t // n_video_groups
        remainder = t % n_video_groups
        new_grids = []
        for i in range(n_video_groups):
            chunk_t = t_per + (1 if i < remainder else 0)
            new_grids.append([chunk_t, h, w])
        inputs['video_grid_thw'] = torch.tensor(new_grids, dtype=grid.dtype, device=grid.device)
        log.info(f"Split video_grid_thw: {grid.tolist()} -> {new_grids}")

    return inputs


def parse_segments(raw_text: str, duration: float) -> list[dict]:
    """Parse model output into timestamped segments."""
    segments = []

    # Clean thinking tags
    if '</think>' in raw_text:
        raw_text = raw_text.split('</think>')[-1].strip()

    # Pattern 1: [00:00-00:05] or [00:00:00-00:00:05] Description
    pattern = re.compile(
        r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\s*[-–]\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\]\s*(.+?)(?=\n\[\d|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(raw_text):
        start = int(m.group(1)) * 60 + int(m.group(2))
        if m.group(3):
            start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        end = int(m.group(4)) * 60 + int(m.group(5))
        if m.group(6):
            end = int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
        text = m.group(7).strip()
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})

    if segments:
        return segments

    # Pattern 2: single timestamps
    for line in raw_text.strip().split('\n'):
        line = line.strip()
        m = re.match(r'\[?(\d{1,2}):(\d{2})\]?\s*[-–:]?\s*(.+)', line)
        if m:
            t = int(m.group(1)) * 60 + int(m.group(2))
            text = re.sub(r'\s+', ' ', m.group(3)).strip()
            if text and len(text) > 5:
                segments.append({"time": t, "text": text})

    if segments:
        result = []
        for i, s in enumerate(segments):
            end = segments[i + 1]["time"] if i + 1 < len(segments) else duration
            result.append({"start": s["time"], "end": end, "text": s["text"]})
        return result

    # Fallback
    text = raw_text.strip()
    if text:
        return [{"start": 0, "end": duration, "text": text}]
    return []


@app.route('/api/status', methods=['GET'])
def status():
    """Check server and model status."""
    vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    return jsonify({
        "available": True,
        "model_loaded": model is not None,
        "model_name": MODEL_ID,
        "backend": "transformers",
        "gpu": gpu_name,
        "vram_used_gb": round(vram_used, 1),
        "vram_total_gb": round(vram_total, 1),
    })


@app.route('/api/describe', methods=['POST'])
def describe():
    """Describe a video file with timestamped scene descriptions.

    JSON body:
    {
        "video_path": "C:/path/to/video.mp4",
        "duration": 30.0,
        "fps": 1.0,
        "max_frames": 48,
        "prompt": "optional custom prompt"
    }
    """
    data = request.get_json()
    if not data or 'video_path' not in data:
        return jsonify({"error": "video_path required"}), 400

    video_path = data['video_path']
    if not os.path.isfile(video_path):
        return jsonify({"error": f"File not found: {video_path}"}), 404

    target_fps = data.get('fps', 1.0)
    max_frames = data.get('max_frames', 48)
    custom_prompt = data.get('prompt')

    with model_lock:
        try:
            load_model()
        except Exception as e:
            log.error(f"Model load failed: {e}", exc_info=True)
            return jsonify({"error": f"Model load failed: {str(e)}"}), 500

        try:
            start_time = time.time()

            # Read video frames with PyAV
            video_data = read_video_frames(
                video_path, target_fps=target_fps, max_frames=max_frames)
            frames = video_data["frames"]
            duration = data.get('duration', video_data["duration"])

            if not frames:
                return jsonify({"error": "No frames extracted from video"}), 400

            # Build prompt
            if custom_prompt:
                prompt_text = custom_prompt
            else:
                prompt_text = (
                    f"This video is {duration:.0f} seconds long. "
                    "Describe what happens scene by scene. "
                    "For each scene, output a line in this exact format:\n"
                    "[MM:SS-MM:SS] Description\n\n"
                    "Be specific about subjects, actions, camera movements. "
                    "Cover the full video. 1-2 sentences per scene."
                )

            # Build chat template with video placeholder
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt_text},
                ],
            }]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            # Pass pre-sampled numpy frames directly to processor
            inputs = processor(
                text=[text_input],
                videos=[frames],
                do_sample_frames=False,
                video_metadata=[{
                    "fps": video_data["video_fps"],
                    "total_num_frames": video_data["total_frames"],
                    "frames_indices": video_data["frame_indices"],
                }],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Fix video_grid_thw splitting bug
            inputs = fix_video_grid_thw(inputs)

            prep_time = time.time() - start_time
            log.info(f"Prepared in {prep_time:.1f}s, generating...")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                )

            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            raw_text = processor.decode(generated_ids, skip_special_tokens=True)

            elapsed = time.time() - start_time
            log.info(f"Done in {elapsed:.1f}s\nOutput:\n{raw_text}")

            segments = parse_segments(raw_text, duration)
            for i, seg in enumerate(segments):
                seg["id"] = f"scene-{i}"

            return jsonify({
                "segments": segments,
                "raw_text": raw_text,
                "elapsed_seconds": round(elapsed, 1),
                "frames_sampled": len(frames),
            })

        except Exception as e:
            log.error(f"Describe failed: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload():
    """Accept a video file upload and save to temp directory.
    Returns the temp file path for use with /api/describe.
    """
    import tempfile
    if 'video' not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp dir with original extension
    ext = os.path.splitext(video_file.filename)[1] or '.mp4'
    fd, temp_path = tempfile.mkstemp(suffix=ext, prefix='qwen3vl_')
    os.close(fd)
    video_file.save(temp_path)
    log.info(f"Uploaded video saved to {temp_path} ({os.path.getsize(temp_path)/1024/1024:.1f}MB)")

    return jsonify({"path": temp_path})


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Delete a temp file created by /api/upload."""
    import tempfile
    data = request.get_json()
    path = data.get('path', '') if data else ''
    temp_dir = tempfile.gettempdir()
    # Only delete files in temp dir that we created
    if path and os.path.isfile(path) and os.path.dirname(os.path.abspath(path)) == os.path.abspath(temp_dir):
        if os.path.basename(path).startswith('qwen3vl_'):
            os.remove(path)
            log.info(f"Cleaned up temp file: {path}")
            return jsonify({"status": "deleted"})
    return jsonify({"status": "skipped"})


@app.route('/api/unload', methods=['POST'])
def unload():
    """Unload model from GPU."""
    global model, processor
    with model_lock:
        if model is not None:
            del model
            del processor
            model = None
            processor = None
            gc.collect()
            torch.cuda.empty_cache()
            log.info("Model unloaded")
            return jsonify({"status": "unloaded"})
        return jsonify({"status": "already_unloaded"})


if __name__ == '__main__':
    if '--preload' in sys.argv:
        load_model()

    port = int(os.environ.get('PORT', 5555))
    log.info(f"Starting Qwen3-VL server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True)
