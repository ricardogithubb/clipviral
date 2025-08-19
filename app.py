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
# Estrutura:
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
    # Rápido e leve: mono, 16-bit, sample rate menor
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
    # Leitura simples sem libs pesadas: usa numpy.frombuffer + wave header via ffmpeg (já garantido como pcm_s16le)
    with open(path, "rb") as f:
        raw = f.read()
    # ignorar header WAV: procure "data" chunk e leia o payload
    # WAV simples: não vamos implementar um parser completo; como é gerado pelo ffmpeg, o chunk 'data' aparece uma única vez
    # Encontra índice do b"data"
    idx = raw.find(b"data")
    if idx == -1:
        return np.array([], dtype=np.int16)
    # payload size são 4 bytes após 'data'
    size = int.from_bytes(raw[idx + 4: idx + 8], "little", signed=False)
    start = idx + 8
    payload = raw[start: start + size]
    arr = np.frombuffer(payload, dtype=np.int16)
    return arr


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) == 0 or win <= 1:
        return np.array([], dtype=np.float32)
    x = x.astype(np.float32)
    # RMS janela deslizante (raiz da média dos quadrados)
    cumsum = np.cumsum(np.insert(x * x, 0, 0))
    window_sum = cumsum[win:] - cumsum[:-win]
    rms = np.sqrt(window_sum / win)
    return rms


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_piecewise_linear_expr(points: List[Dict[str, float]], axis: str, cw_expr: str, ch_expr: str) -> str:
    """
    Constrói expressão FFmpeg para pan (x ou y) com interpolação linear entre keyframes.
    points: [{"t":sec_rel, "x":0..1, "y":0..1}, ...] - tempos relativos ao início do corte!
    axis: "x" ou "y"
    cw_expr: expr do crop width (ex.: "ih*9/16")
    ch_expr: expr do crop height (ex.: "ih")
    Retorna expressão que calcula posição em pixels.
    """
    if not points:
        # centralizado por padrão
        if axis == "x":
            base = f"(iw-({cw_expr}))/2"
        else:
            base = f"(ih-({ch_expr}))/2"
        return base

    # Ordena e remove duplicados de tempo
    pts = sorted(points, key=lambda p: p["t"])
    filtered = []
    last_t = None
    for p in pts:
        t = float(p["t"])
        if last_t is None or abs(t - last_t) > 1e-6:
            filtered.append(p)
            last_t = t
    pts = filtered

    # Converte 0..1 para pixels depois multiplicando por (iw-cw) ou (ih-ch)
    span_expr = f"(iw-({cw_expr}))" if axis == "x" else f"(ih-({ch_expr}))"

    # Se só 1 ponto: manter fixo
    if len(pts) == 1:
        v = clamp(float(pts[0][axis]), 0.0, 1.0)
        return f"({span_expr})*{v}"

    # Monta expressão piecewise com if() e between()
    pieces = []
    for i in range(len(pts) - 1):
        t0 = float(pts[i]["t"])
        t1 = float(pts[i + 1]["t"])
        v0 = clamp(float(pts[i][axis]), 0.0, 1.0)
        v1 = clamp(float(pts[i + 1][axis]), 0.0, 1.0)
        # lerp: v0 + (t - t0) * (v1 - v0) / (t1 - t0)
        expr_v = f"({v0}+((t-{t0})*({v1}-{v0})/({t1}-{t0})))"
        pieces.append(f"between(t,{t0},{t1})*(({span_expr})*{expr_v})")

    # Antes do primeiro e depois do último, mantém o valor extremo
    v_first = clamp(float(pts[0][axis]), 0.0, 1.0)
    v_last = clamp(float(pts[-1][axis]), 0.0, 1.0)
    expr_first = f"(t<{pts[0]['t']})*(({span_expr})*{v_first})"
    expr_last = f"(t>{pts[-1]['t']})*(({span_expr})*{v_last})"

    # Soma das regiões (as between() são mutuamente exclusivas nas bordas)
    full_expr = "+".join([expr_first] + pieces + [expr_last])
    # Clampeia (segurança)
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


def analyze_video_fast(video_path: str) -> List[Dict[str, Any]]:
    """
    Estratégia leve:
      - Duração via ffprobe
      - Áudio downmix PCM16 22.05kHz -> RMS móvel para achar picos (refrões/pontos altos)
      - Vídeo 1 FPS reduzido para 320px largura -> contagem de faces (dá peso para 'protagonista')
    Gera propostas de até ~10 cortes de 60s (sem sobreposição), ordenados por score áudio*faces.
    """
    meta = ffprobe_json(video_path)
    duration = float(meta["format"]["duration"])
    if duration <= 0:
        return []

    # ---- ÁUDIO ----
    wav_tmp = os.path.join(TMP_FOLDER, f"{uuid.uuid4().hex}.wav")
    try:
        downmix_audio_pcm16(video_path, wav_tmp, sr=22050)
        samples = read_wav_as_np_int16(wav_tmp)
    finally:
        try:
            os.remove(wav_tmp)
        except Exception:
            pass

    # RMS móvel em janelas de ~0.2s
    sr = 22050
    win = max(1, int(0.2 * sr))
    rms = moving_rms(samples, win)
    if rms.size == 0:
        rms = np.zeros(1, dtype=np.float32)
    # timestamps dos rms (em segundos) alinhados com o centro da janela
    rms_t = np.arange(len(rms)) * (1.0 / sr) + (win / (2 * sr))

    # Normaliza RMS
    if rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        rms_norm = rms

    # ---- VÍDEO / ROSTOS (1 FPS, 320px largura) ----
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or (duration * fps))
    frame_interval = int(fps)  # 1 fps
    face_times = []
    face_scores = []

    for i in range(0, total_frames, max(1, frame_interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        # redimensiona mantendo proporção para largura 320
        h, w = frame.shape[:2]
        new_w = 320
        new_h = int(h * (new_w / w))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        timestamp = i / fps
        face_times.append(timestamp)
        face_scores.append(min(5, len(faces)))  # clamp de segurança

    cap.release()
    faces_arr = np.array(face_scores, dtype=np.float32)
    face_t = np.array(face_times, dtype=np.float32)

    # Interpola faces para o tempo do RMS
    if len(faces_arr) > 1:
        faces_interp = np.interp(rms_t, face_t, faces_arr)
    else:
        faces_interp = np.zeros_like(rms_t)

    # Score combinado (peso 0.7 áudio, 0.3 faces)
    if faces_interp.max() > 0:
        faces_norm = faces_interp / faces_interp.max()
    else:
        faces_norm = faces_interp

    score = 0.7 * rms_norm + 0.3 * faces_norm

    # Busca picos locais e gera janelas de 60s em torno deles
    clip_len = 60.0
    max_clips = max(1, min(10, int(duration // 15)))  # adaptativo
    peaks_idx = score.argsort()[::-1]  # maiores primeiro

    used_intervals = []
    proposals = []

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for idx in peaks_idx:
        t = float(rms_t[idx])
        start = clamp(t - clip_len / 2, 0.0, max(0.0, duration - clip_len))
        end = min(start + clip_len, duration)
        # evita sobreposição grande com janelas já escolhidas
        candidate = (start, end)
        if any(overlaps(candidate, u) for u in used_intervals):
            continue

        proposals.append(
            {
                "id": str(uuid.uuid4()),
                "start": float(start),
                "end": float(end),
                "score": float(score[idx]),
                # sem keyframes inicialmente
            }
        )
        used_intervals.append(candidate)
        if len(proposals) >= max_clips:
            break

    # fallback se não encontrou nada
    if not proposals:
        n = int(math.ceil(duration / clip_len))
        for i in range(min(10, n)):
            s = i * clip_len
            e = min(s + clip_len, duration)
            proposals.append({"id": str(uuid.uuid4()), "start": float(s), "end": float(e), "score": 0.0})

    return proposals


def build_crop_filter_916_with_keyframes(start: float, end: float, keyframes: List[Dict[str, float]]) -> str:
    """
    Gera filtro:
      - Primeiro recorta para 9:16 com 'crop=cw:ch:x:y'
      - Aplica pan (x,y) com base nos keyframes normalizados (0..1) e timestamps (relativos ao corte!)
      - Depois escala para 1080x1920
    Observação importante: no grafo, 't' começa em 0 no início do corte. Portanto,
    os keyframes recebidos devem estar relativos ao 'start'. Se vierem absolutos, compensamos.
    """
    # Normaliza keyframes (se tempos são absolutos, tornam-se relativos)
    rel_kf = []
    for k in (keyframes or []):
        t = float(k.get("time", 0.0))
        # se algum kf está depois de 'end', assumimos que são absolutos e subtraímos 'start'
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
    """
    Render robusto e preciso:
      - '-ss' e '-to' APÓS o '-i' para maior precisão de corte
      - filtro de crop+pan+scale conforme keyframes
      - progresso real com '-progress -' (stdout)
    """
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
        "-",  # envia progresso em stdout
        out_path,
    ]

    # duração do corte
    cut_dur = max(0.001, float(end - start))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

    percent = 0.0
    try:
        for line in proc.stdout:
            line = line.strip()
            # linhas de interesse: out_time_ms=, progress=
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
    """
    Roda no background, processando os clipes em sequência (evita sobrecarga de CPU/GPU)
    Atualiza progress_store[session_id] continuamente e cria o ZIP ao final.
    """
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
                    # média simples do total
                    percs = [v["percent"] for v in progress_store[session_id]["clips"].values()]
                    progress_store[session_id]["total"] = round(sum(percs) / max(1, len(percs)), 2)
                    progress_store[session_id]["message"] = f"Renderizando {idx}/{len(clips)}..."

            ffmpeg_render_clip(filepath, out_path_tmp, start, end, keyframes, on_progress)

            # remove metadados
            if not remove_metadata(out_path_tmp, out_path):
                shutil.move(out_path_tmp, out_path)
            else:
                try:
                    os.remove(out_path_tmp)
                except Exception:
                    pass

            with progress_lock:
                progress_store[session_id]["clips"][clip_id]["filename"] = f"{clip_id}.mp4"

        # cria ZIP
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
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(f.filename)
    # evitar colisão
    prefix = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    filename_on_disk = f"{prefix}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename_on_disk)
    f.save(filepath)

    return jsonify({"filename": filename_on_disk, "filepath": filepath})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    filepath = data.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        proposals = analyze_video_fast(filepath)
        return jsonify({"proposals": proposals})
    except Exception as e:
        return jsonify({"proposals": [], "error": str(e)}), 200


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

    # roda em background para não bloquear a requisição
    t = threading.Thread(target=render_session, args=(session_id, filepath, clips), daemon=True)
    t.start()

    return jsonify({"session_id": session_id})


@app.route("/render_status/<session_id>")
def render_status(session_id: str):
    # SSE: envia JSON incremental com progresso
    def stream():
        while True:
            with progress_lock:
                state = progress_store.get(session_id)
                if not state:
                    payload = {"status": "unknown", "total": 0}
                else:
                    payload = state.copy()
            yield f"data: {json.dumps(payload)}\n\n"

            # termina quando done ou error
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


# --------- LIMPEZA AUTOMÁTICA ---------
def clean_older_than(folder: str, hours: int = 6):
    now = time.time()
    cutoff = now - hours * 3600
    for p in glob.glob(os.path.join(folder, "**"), recursive=True):
        try:
            if os.path.isdir(p):
                # se pasta vazia e antiga, remove
                if os.stat(p).st_mtime < cutoff and not os.listdir(p):
                    shutil.rmtree(p, ignore_errors=True)
            else:
                if os.stat(p).st_mtime < cutoff:
                    os.remove(p)
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
        time.sleep(3600)  # a cada 1h


threading.Thread(target=cleanup_worker, daemon=True).start()


if __name__ == "__main__":
    # prod: app.run(host="0.0.0.0", port=8000)
    app.run(host="0.0.0.0", debug=True)
