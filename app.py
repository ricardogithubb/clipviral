import os
import io
import re
import gc
import uuid
import glob
import time
import json
import math
import queue
import shutil
import threading
import subprocess
from datetime import datetime
from typing import List, Dict, Any
import yt_dlp

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

import cv2
import numpy as np

# ------------------ CONFIG ------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
TMP_FOLDER = os.path.join(BASE_DIR, "tmp")

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(TMP_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    PROCESSED_FOLDER=PROCESSED_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS,
)

# ------------------ PROGRESS / SESSIONS ------------------
# progress_store[session_id] = {
#   "status": "idle|running|done|error",
#   "message": "...",
#   "clips": {clip_id: {"percent": float, "filename": str|None}},
#   "total": float (0-100)
# }
progress_store: Dict[str, Dict[str, Any]] = {}
progress_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------ UTILS ------------------
def ffprobe_json(video_path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed")
    return json.loads(result.stdout)


def downmix_audio_pcm16(video_path: str, out_wav: str, sr: int = 22050):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-acodec",
        "pcm_s16le",
        out_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def read_wav_as_np_int16(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    idx = raw.find(b"data")
    if idx == -1:
        return np.array([], dtype=np.int16)
    size = int.from_bytes(raw[idx + 4: idx + 8], "little", signed=False)
    start = idx + 8
    payload = raw[start: start + size]
    arr = np.frombuffer(payload, dtype=np.int16)
    return arr


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) == 0 or win <= 1:
        return np.array([], dtype=np.float32)
    x = x.astype(np.float32)
    cumsum = np.cumsum(np.insert(x * x, 0, 0))
    window_sum = cumsum[win:] - cumsum[:-win]
    rms = np.sqrt(window_sum / win)
    return rms


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_piecewise_linear_expr(points: List[Dict[str, float]], axis: str, cw_expr: str, ch_expr: str) -> str:
    if not points:
        if axis == "x":
            base = f"(iw-({cw_expr}))/2"
        else:
            base = f"(ih-({ch_expr}))/2"
        return base

    pts = sorted(points, key=lambda p: p["t"])
    filtered = []
    last_t = None
    for p in pts:
        t = float(p["t"])
        if last_t is None or abs(t - last_t) > 1e-6:
            filtered.append(p)
            last_t = t
    pts = filtered

    span_expr = f"(iw-({cw_expr}))" if axis == "x" else f"(ih-({ch_expr}))"

    if len(pts) == 1:
        v = clamp(float(pts[0][axis]), 0.0, 1.0)
        return f"({span_expr})*{v}"

    pieces = []
    for i in range(len(pts) - 1):
        t0 = float(pts[i]["t"])
        t1 = float(pts[i + 1]["t"])
        v0 = clamp(float(pts[i][axis]), 0.0, 1.0)
        v1 = clamp(float(pts[i + 1][axis]), 0.0, 1.0)
        expr_v = f"({v0}+((t-{t0})*({v1}-{v0})/({t1}-{t0})))"
        pieces.append(f"between(t,{t0},{t1})*(({span_expr})*{expr_v})")

    v_first = clamp(float(pts[0][axis]), 0.0, 1.0)
    v_last = clamp(float(pts[-1][axis]), 0.0, 1.0)
    expr_first = f"(t<{pts[0]['t']})*(({span_expr})*{v_first})"
    expr_last = f"(t>{pts[-1]['t']})*(({span_expr})*{v_last})"

    full_expr = "+".join([expr_first] + pieces + [expr_last])
    if axis == "x":
        return f"clip({full_expr},0,iw-({cw_expr}))"
    else:
        return f"clip({full_expr},0,ih-({ch_expr}))"


def remove_metadata(input_path: str, output_path: str) -> bool:
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-map_metadata",
            "-1",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def analyze_video_fast_simple(video_path: str, update_cb=None) -> List[Dict[str, Any]]:
    meta = ffprobe_json(video_path)
    duration = float(meta["format"]["duration"])
    all_keyframes = []

    if duration <= 0:
        return []

    wav_tmp = os.path.join(TMP_FOLDER, f"{uuid.uuid4().hex}.wav")
    try:
        downmix_audio_pcm16(video_path, wav_tmp, sr=22050)
        samples = read_wav_as_np_int16(wav_tmp)
    finally:
        try: os.remove(wav_tmp)
        except Exception: pass

    sr = 22050
    win = max(1, int(0.2 * sr))
    rms = moving_rms(samples, win)
    rms_t = np.arange(len(rms)) * (1.0 / sr) + (win / (2 * sr))
    rms_norm = rms / rms.max() if rms.max() > 0 else rms

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or (duration * fps))
    frame_interval = int(fps)
    face_times, face_scores = [], []

    for i in range(0, total_frames, max(1, frame_interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok: continue

        h, w = frame.shape[:2]
        new_w = 320
        new_h = int(h * (new_w / w))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        timestamp = i / fps
        score_face = 0

        if len(faces) > 0:
            # pega a maior face
            (fx, fy, fw, fh) = max(faces, key=lambda f: f[2]*f[3])
            cx = (fx + fw/2) / new_w
            cy = (fy + fh/2) / new_h

            # pega região da boca (parte inferior do rosto)
            mouth_region = gray[int(fy+fh*0.6):fy+fh, fx:fx+fw]
            mouth_activity = mouth_region.var() if mouth_region.size > 0 else 0

            # score = nº de faces + movimento da boca
            score_face = min(5, len(faces) + mouth_activity/5000.0)

            all_keyframes.append({"t": timestamp, "x": cx, "y": cy})

        face_times.append(timestamp)
        face_scores.append(score_face)

        if update_cb:
            pct = (i / total_frames) * 80
            update_cb(pct, f"Processando vídeo... {pct:.1f}%")

    cap.release()

    # normaliza sinais
    faces_arr = np.array(face_scores, dtype=np.float32)
    face_t = np.array(face_times, dtype=np.float32)
    faces_interp = np.interp(rms_t, face_t, faces_arr) if len(faces_arr) > 1 else np.zeros_like(rms_t)
    faces_norm = faces_interp / faces_interp.max() if faces_interp.max() > 0 else faces_interp
    score = 0.7 * rms_norm + 0.3 * faces_norm

    return _generate_clip_proposals(duration, rms_t, score, all_keyframes)


def _generate_clip_proposals(duration, rms_t, score, all_keyframes):
    clip_len = 60.0
    max_clips = max(1, min(3, int(duration // 15)))
    peaks_idx = score.argsort()[::-1]

    used_intervals, proposals = [], []

    def overlaps(a, b): return not (a[1] <= b[0] or b[1] <= a[0])

    for idx in peaks_idx:
        t = float(rms_t[idx])
        start = clamp(t - clip_len / 2, 0.0, max(0.0, duration - clip_len))
        end = min(start + clip_len, duration)
        candidate = (start, end)
        if any(overlaps(candidate, u) for u in used_intervals): continue

        proposals.append({
            "id": str(uuid.uuid4()),
            "start": float(start),
            "end": float(end),
            "score": float(score[idx]),
            "keyframes": [kf for kf in all_keyframes if start <= kf["t"] <= end]
        })
        used_intervals.append(candidate)
        if len(proposals) >= max_clips: break

    if not proposals:
        n = int(math.ceil(duration / clip_len))
        for i in range(min(10, n)):
            s = i * clip_len
            e = min(s + clip_len, duration)
            proposals.append({"id": str(uuid.uuid4()), "start": float(s), "end": float(e), "score": 0.0})

    return proposals



def build_crop_filter_916_with_keyframes(start: float, end: float, keyframes: List[Dict[str, float]]) -> str:
    rel_kf = []
    for k in (keyframes or []):
        t = float(k.get("time", 0.0))
        rel_t = t - start if t > end else t
        rel_kf.append({"t": clamp(rel_t, 0.0, end - start), "x": float(k.get("x", 0.5)), "y": float(k.get("y", 0.5))})

    cw = "ih*9/16"
    ch = "ih"
    x_expr = build_piecewise_linear_expr(rel_kf, "x", cw, ch)
    y_expr = build_piecewise_linear_expr(rel_kf, "y", cw, ch)

    crop = f"crop={cw}:{ch}:{x_expr}:{y_expr}"
    scale = "scale=1080:1920"
    return f"{crop},{scale}"


def ffmpeg_render_clip(input_path: str, out_path: str, start: float, end: float, keyframes: List[Dict[str, float]], update_cb=None):
    vf = build_crop_filter_916_with_keyframes(start, end, keyframes)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ss",
        str(start),
        "-to",
        str(end),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        "-progress",
        "-",
        out_path,
    ]

    cut_dur = max(0.001, float(end - start))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

    percent = 0.0
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                out_ms = float(line.split("=", 1)[1])
                tcur = out_ms / 1_000_000.0
                percent = max(0.0, min(100.0, (tcur / cut_dur) * 100.0))
                if update_cb:
                    update_cb(percent)
            elif line.startswith("progress=") and line.endswith("end"):
                percent = 100.0
                if update_cb:
                    update_cb(percent)
    finally:
        proc.wait()


def render_session(session_id: str, filepath: str, clips: List[Dict[str, Any]]):
    with progress_lock:
        progress_store[session_id] = {
            "status": "running",
            "message": "Iniciando renderização...",
            "clips": {clip["id"]: {"percent": 0.0, "filename": None} for clip in clips},
            "total": 0.0,
        }

    out_dir = os.path.join(PROCESSED_FOLDER, session_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        for idx, clip in enumerate(clips, 1):
            clip_id = clip["id"]
            start = float(clip["start"])
            end = float(clip["end"])
            keyframes = clip.get("keyframes", [])

            out_path_tmp = os.path.join(out_dir, f"{clip_id}_tmp.mp4")
            out_path = os.path.join(out_dir, f"{clip_id}.mp4")

            def on_progress(pct):
                with progress_lock:
                    progress_store[session_id]["clips"][clip_id]["percent"] = round(float(pct), 2)
                    percs = [v["percent"] for v in progress_store[session_id]["clips"].values()]
                    progress_store[session_id]["total"] = round(sum(percs) / max(1, len(percs)), 2)
                    progress_store[session_id]["message"] = f"Renderizando {idx}/{len(clips)}..."

            ffmpeg_render_clip(filepath, out_path_tmp, start, end, keyframes, on_progress)

            if not remove_metadata(out_path_tmp, out_path):
                shutil.move(out_path_tmp, out_path)
            else:
                try:
                    os.remove(out_path_tmp)
                except Exception:
                    pass

            with progress_lock:
                progress_store[session_id]["clips"][clip_id]["filename"] = f"{clip_id}.mp4"

        zip_path = os.path.join(PROCESSED_FOLDER, f"{session_id}.zip")
        import zipfile

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for clip in clips:
                fname = f"{clip['id']}.mp4"
                zipf.write(os.path.join(out_dir, fname), fname)

        with progress_lock:
            progress_store[session_id]["status"] = "done"
            progress_store[session_id]["message"] = "Renderização concluída."

    except Exception as e:
        with progress_lock:
            progress_store[session_id]["status"] = "error"
            progress_store[session_id]["message"] = f"Erro: {e}"

    finally:
        gc.collect()


# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    clear_uploads()
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(f.filename)
    prefix = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    filename_on_disk = f"{prefix}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename_on_disk)
    f.save(filepath)

    return jsonify({"filename": filename_on_disk, "filepath": filepath})

@app.route("/upload_youtube", methods=["POST"])
def upload_youtube():
    clear_uploads()
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "Nenhum link fornecido"}), 400

    session_id = str(uuid.uuid4())
    with progress_lock:
        progress_store[session_id] = {
            "status": "idle",
            "message": "Iniciando download...",
            "total": 0.0,
            "filename": None,
            "filepath": None,
        }

    def download_task():
        try:
            def hook(d):
                with progress_lock:
                    if d["status"] == "downloading":
                        percent = d.get("_percent_str", "0%")
                        progress_store[session_id]["status"] = "running"
                        progress_store[session_id]["message"] = f"Baixando... {percent}"
                        progress_store[session_id]["total"] = float(d.get("_percent", 0.0)) * 100
                    elif d["status"] == "finished":
                        # 🔧 Corrigido: garantir que aponta para o arquivo final .mp4
                        vid_id = d.get("info_dict", {}).get("id")
                        final_path = os.path.join(UPLOAD_FOLDER, f"{vid_id}.mp4")
                        progress_store[session_id]["status"] = "done"
                        progress_store[session_id]["message"] = "Download concluído."
                        progress_store[session_id]["filename"] = os.path.basename(final_path)
                        progress_store[session_id]["filepath"] = final_path

            ydl_opts = {
                "outtmpl": os.path.join(UPLOAD_FOLDER, "%(id)s.%(ext)s"),
                # 🔧 Forçar H.264 (avc1) em MP4, evita AV1
                "format": "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "retries": 10,
                "fragment_retries": 10,
                "concurrent_fragment_downloads": 3,
                "progress_hooks": [hook],
                "no_color": True,
                "prefer_ffmpeg": True,
                "ffmpeg_location": "/usr/bin/ffmpeg",
                # 🔑 Arquivo de cookies exportado do navegador
                # "cookiefile": "/home/ubuntu/cookies.txt",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print("Erro no yt-dlp:", e)  # log debug
            with progress_lock:
                progress_store[session_id]["status"] = "error"
                progress_store[session_id]["message"] = str(e)

    t = threading.Thread(target=download_task, daemon=True)
    t.start()

    return jsonify({
        "session_id": session_id,
        "filepath": progress_store[session_id]["filepath"]
    })




@app.route("/youtube_status/<session_id>")
def youtube_status(session_id: str):
    def stream():
        while True:
            with progress_lock:
                state = progress_store.get(session_id)
                if not state:
                    payload = {"status": "unknown", "total": 0}
                else:
                    payload = state.copy()
            yield f"data: {json.dumps(payload)}\n\n"

            if state and state.get("status") in ("done", "error"):
                break
            time.sleep(0.5)

    return Response(stream(), mimetype="text/event-stream")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    filepath = data.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    session_id = str(uuid.uuid4())
    with progress_lock:
        progress_store[session_id] = {
            "status": "running",
            "message": "Iniciando análise...",
            "total": 0,
            "proposals": []
        }

    def task():
        try:
            def on_update(pct, msg):
                with progress_lock:
                    progress_store[session_id]["total"] = round(pct, 2)
                    progress_store[session_id]["message"] = msg

            proposals = analyze_video_fast_simple(filepath, update_cb=on_update)

            with progress_lock:
                progress_store[session_id]["status"] = "done"
                progress_store[session_id]["message"] = "Análise concluída."
                progress_store[session_id]["total"] = 100
                progress_store[session_id]["proposals"] = proposals

        except Exception as e:
            with progress_lock:
                progress_store[session_id]["status"] = "error"
                progress_store[session_id]["message"] = str(e)


    threading.Thread(target=task, daemon=True).start()

    # retorna só o id, o frontend vai abrir SSE em /analyze_status/<session_id>
    return jsonify({"session_id": session_id})


@app.route("/analyze_status/<session_id>")
def analyze_status(session_id: str):
    def stream():
        while True:
            with progress_lock:
                state = progress_store.get(session_id)
                if not state:
                    payload = {"status": "unknown", "total": 0}
                else:
                    payload = state.copy()
            yield f"data: {json.dumps(payload)}\n\n"

            if state and state.get("status") in ("done", "error"):
                break
            time.sleep(0.5)

    return Response(stream(), mimetype="text/event-stream")

        


@app.route("/render", methods=["POST"])
def render():
    data = request.get_json(silent=True) or {}
    filepath = data.get("filepath")
    clips = data.get("clips")

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    if not isinstance(clips, list) or len(clips) == 0:
        return jsonify({"error": "Invalid clips data"}), 400

    session_id = str(uuid.uuid4())
    with progress_lock:
        progress_store[session_id] = {"status": "idle", "message": "Aguardando...", "clips": {}, "total": 0.0}

    t = threading.Thread(target=render_session, args=(session_id, filepath, clips), daemon=True)
    t.start()

    return jsonify({"session_id": session_id})


@app.route("/render_single", methods=["POST"])
def render_single():
    """
    Renderiza APENAS um corte (mais responsivo para ajustes de edição).
    Reutiliza o mesmo mecanismo de progresso por SSE.
    """
    data = request.get_json(silent=True) or {}
    filepath = data.get("filepath")
    clip = data.get("clip")

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    if not isinstance(clip, dict):
        return jsonify({"error": "Invalid clip data"}), 400

    session_id = str(uuid.uuid4())
    with progress_lock:
        progress_store[session_id] = {"status": "idle", "message": "Aguardando...", "clips": {}, "total": 0.0}

    t = threading.Thread(target=render_session, args=(session_id, filepath, [clip]), daemon=True)
    t.start()

    return jsonify({"session_id": session_id, "clip_id": clip.get("id")})


@app.route("/render_status/<session_id>")
def render_status(session_id: str):
    def stream():
        while True:
            with progress_lock:
                state = progress_store.get(session_id)
                if not state:
                    payload = {"status": "unknown", "total": 0}
                else:
                    payload = state.copy()
            yield f"data: {json.dumps(payload)}\n\n"

            if state and state.get("status") in ("done", "error"):
                break
            time.sleep(0.5)

    return Response(stream(), mimetype="text/event-stream")


@app.route("/download/<session_id>")
def download_zip(session_id: str):
    zip_path = os.path.join(PROCESSED_FOLDER, f"{session_id}.zip")
    if os.path.exists(zip_path):
        return send_from_directory(PROCESSED_FOLDER, f"{session_id}.zip", as_attachment=True)
    return jsonify({"error": "File not found"}), 404


@app.route("/download_clip/<session_id>/<clip_id>")
def download_clip(session_id, clip_id):
    clip_path = os.path.join(PROCESSED_FOLDER, session_id, f"{clip_id}.mp4")
    if os.path.exists(clip_path):
        return send_from_directory(os.path.join(PROCESSED_FOLDER, session_id), f"{clip_id}.mp4", as_attachment=True)
    return jsonify({"error": "File not found"}), 404

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# --------- LIMPEZA AUTOMÁTICA ---------
def clean_older_than(folder: str, hours: int = 6):
    now = time.time()
    cutoff = now - hours * 3600
    for p in glob.glob(os.path.join(folder, "**"), recursive=True):
        try:
            if os.path.isdir(p):
                if os.stat(p).st_mtime < cutoff and not os.listdir(p):
                    shutil.rmtree(p, ignore_errors=True)
            else:
                if os.stat(p).st_mtime < cutoff:
                    os.remove(p)
        except Exception:
            pass

def clear_uploads():
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception:
            pass


def cleanup_worker():
    while True:
        try:
            clean_older_than(UPLOAD_FOLDER, hours=12)
            clean_older_than(PROCESSED_FOLDER, hours=24)
            clean_older_than(TMP_FOLDER, hours=6)
        except Exception:
            pass
        time.sleep(3600)


threading.Thread(target=cleanup_worker, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
