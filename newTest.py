import cv2
import torch
import os
import time
import threading
import queue
import json
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, Response
from typing import Dict, List
from functools import lru_cache

# ============================================================
#  YOLOv8 (Ultralytics)
# ============================================================
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ============================================================
#  Flask app
# ============================================================
app = Flask(__name__)

# ============================================================
#  Konfigurasi umum
# ============================================================
N_SLOTS = 5   # HANYA 5 slot parkir yang dipakai

TARGET_FPS = 20
DETECTION_SIZE = (512, 512)  # ukuran input untuk YOLO
CONFIDENCE_THRESHOLD = 0.15  # sedikit longgar supaya deteksi sering muncul
IOU_THRESHOLD = 0.5
DETECTION_INTERVAL = 0.2     # deteksi tiap 0.2 detik
DATABASE_UPDATE_INTERVAL = 3
DATABASE_FILE = "parking_database.sqlite"
USE_GPU = torch.cuda.is_available()

# Video rekaman
VIDEO_SOURCE = "easy1.mp4"

# ============================================================
#  Variabel global
# ============================================================
model = None
parking_slots: Dict[str, bool] = {f"slot_{i}": False for i in range(1, N_SLOTS+1)}
slots_lock = threading.Lock()

SMOOTH_FRAMES = 1
slot_counters = {f"slot_{i}": 0 for i in range(1, N_SLOTS+1)}
slot_states   = {f"slot_{i}": False for i in range(1, N_SLOTS+1)}

SLOT_THRESHOLDS = {f"slot_{i}": 0.4 for i in range(1, N_SLOTS+1)}

SLOTS_FILE = "slots.json"
SLOT_POLYGONS_NORM: Dict[str, List[List[float]]] = {}

last_parking_state = parking_slots.copy()

frame_width = 0
frame_height = 0

encoded_frame = None
frame_lock = threading.Lock()
database_lock = threading.Lock()

# share deteksi antara thread
last_detections = None
detection_lock = threading.Lock()

detection_queue = queue.Queue(maxsize=1)

# ============================================================
#  Database
# ============================================================
def initialize_database():
    """Init SQLite DB."""
    try:
        with database_lock:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parking_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    status_json TEXT,
                    occupied_count INTEGER,
                    available_count INTEGER
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')

            cursor.execute("SELECT COUNT(*) FROM system_stats WHERE key='system_start_time'")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO system_stats VALUES (?, ?)",
                              ('system_start_time', datetime.now().isoformat()))
                cursor.execute("INSERT INTO system_stats VALUES (?, ?)",
                              ('total_detections', '0'))
                cursor.execute("INSERT INTO system_stats VALUES (?, ?)",
                              ('last_detection_time', None))

            conn.commit()
            conn.close()
            print(f"✅ SQLite database initialized: {DATABASE_FILE}")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")


@lru_cache(maxsize=1)
def get_system_stats():
    try:
        with database_lock:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM system_stats")
            stats = dict(cursor.fetchall())
            conn.close()
        return stats
    except Exception as e:
        print(f"❌ Error loading system stats: {e}")
        return {}


def update_parking_database():
    """Simpan perubahan status slot ke DB."""
    global last_parking_state

    if parking_slots == last_parking_state:
        return

    try:
        ts = datetime.now().isoformat()
        occupied = [i for i, occ in [(int(k.split('_')[1]), v) for k, v in parking_slots.items()] if occ]
        available = [i for i in range(1, N_SLOTS+1) if i not in occupied]
        occ_count = len(occupied)

        status_json = json.dumps({
            "slots": parking_slots,
            "available_slots": available,
            "occupied_count": occ_count,
            "available_count": len(available)
        })

        with database_lock:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO parking_status (timestamp, status_json, occupied_count, available_count) VALUES (?, ?, ?, ?)",
                (ts, status_json, occ_count, len(available))
            )

            cursor.execute("SELECT value FROM system_stats WHERE key='total_detections'")
            total_detections = int(cursor.fetchone()[0]) + 1
            cursor.execute("UPDATE system_stats SET value=? WHERE key='total_detections'",
                          (str(total_detections),))
            cursor.execute("UPDATE system_stats SET value=? WHERE key='last_detection_time'",
                          (ts,))

            cursor.execute("DELETE FROM parking_status WHERE id NOT IN (SELECT id FROM parking_status ORDER BY id DESC LIMIT 100)")

            conn.commit()
            conn.close()

        last_parking_state = parking_slots.copy()
        print(f"📊 DB updated: Occ={occ_count}, Avail={len(available)}")

        get_system_stats.cache_clear()
        get_latest_parking_status.cache_clear()
        get_parking_history.cache_clear()

    except Exception as e:
        print(f"❌ Error updating database: {e}")


@lru_cache(maxsize=1)
def get_latest_parking_status():
    try:
        with database_lock:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT status_json FROM parking_status ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()

        if result:
            return json.loads(result[0])
        return None
    except Exception as e:
        print(f"❌ Error getting latest status: {e}")
        return None


@lru_cache(maxsize=1)
def get_parking_history(limit=50):
    try:
        with database_lock:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, status_json, occupied_count, available_count FROM parking_status ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            conn.close()

        history = []
        for ts, status_json, occ_count, avail_count in rows:
            status = json.loads(status_json)
            history.append({
                "timestamp": ts,
                "available_slots": status.get("available_slots", []),
                "occupied_slots": [i for i in range(1, N_SLOTS+1) if i not in status.get("available_slots", [])],
                "occupancy_rate": round((occ_count/float(N_SLOTS))*100, 1),
                "available_count": avail_count,
                "occupied_count": occ_count
            })
        return history
    except Exception as e:
        print(f"❌ Error getting parking history: {e}")
        return []


def database_update_thread():
    while True:
        update_parking_database()
        time.sleep(DATABASE_UPDATE_INTERVAL)

# ============================================================
#  Slot polygon (slots.json, normalized 0–1)
# ============================================================
def load_slots_from_file():
    """Load normalized polygons (0–1) dari slots.json, atau buat default."""
    global SLOT_POLYGONS_NORM

    if os.path.exists(SLOTS_FILE):
        try:
            with open(SLOTS_FILE, "r") as f:
                data = json.load(f)
            SLOT_POLYGONS_NORM = {}
            for i in range(1, N_SLOTS+1):
                key = f"slot_{i}"
                if key in data:
                    SLOT_POLYGONS_NORM[key] = data[key]
            print(f"✅ Loaded slot polygons from {SLOTS_FILE}: {list(SLOT_POLYGONS_NORM.keys())}")
            return
        except Exception as e:
            print(f"⚠️ Error reading {SLOTS_FILE}: {e}")

    print("⚠️ slots.json tidak ditemukan / gagal dibaca. Menggunakan polygon default.")
    default_polys = {}
    for i in range(N_SLOTS):
        x1 = i / float(N_SLOTS)
        x2 = (i + 1) / float(N_SLOTS)
        y1 = 0.3
        y2 = 0.7
        default_polys[f"slot_{i+1}"] = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]
    SLOT_POLYGONS_NORM = default_polys


def get_slot_polygons_abs():
    """Konversi polygon normalized ke piksel."""
    global frame_width, frame_height
    polys = {}
    if frame_width == 0 or frame_height == 0:
        return polys
    for key, pts in SLOT_POLYGONS_NORM.items():
        polys[key] = [(int(x * frame_width), int(y * frame_height)) for (x, y) in pts]
    return polys


def point_in_polygon(x, y, poly):
    """Ray-casting: cek titik dalam polygon."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-6) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def get_last_detections():
    global last_detections
    with detection_lock:
        if last_detections is None:
            return None
        return np.array(last_detections)


def filter_detections_inside_slots(detections: List):
    """Ambil deteksi yang titik tengahnya berada di dalam salah satu slot."""
    global frame_width, frame_height
    if frame_width == 0 or frame_height == 0:
        return detections

    slot_polys = get_slot_polygons_abs()
    if not slot_polys:
        return detections

    scale_x = frame_width / DETECTION_SIZE[0]
    scale_y = frame_height / DETECTION_SIZE[1]

    filtered = []
    for x1, y1, x2, y2, conf, cls in (detections or []):
        cx_det = (float(x1) + float(x2)) / 2.0
        cy_det = (float(y1) + float(y2)) / 2.0
        cx = cx_det * scale_x
        cy = cy_det * scale_y

        inside_any = False
        for i in range(1, N_SLOTS+1):
            key = f"slot_{i}"
            poly = slot_polys.get(key)
            if not poly:
                continue
            if point_in_polygon(cx, cy, poly):
                inside_any = True
                break

        if inside_any:
            filtered.append([x1, y1, x2, y2, conf, cls])

    print(f"🔍 Filtered detections inside slots: {len(filtered)}/{len(detections)}")
    return filtered

# ============================================================
#  SafeCapture (loop video)
# ============================================================
class SafeCapture:
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        self.is_file = isinstance(source, str) and os.path.exists(source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

        self.queue = queue.Queue(maxsize=1)
        self.running = True
        self.last_frame = None
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.1)
                    continue

            with self.lock:
                self.last_frame = frame
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put(frame)

    def read(self):
        try:
            with self.lock:
                if self.queue.empty() and self.last_frame is not None:
                    return True, self.last_frame.copy()
                return True, self.queue.get(timeout=0.1)
        except (queue.Empty, Exception):
            return False, None

    def stop(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

# ============================================================
#  YOLOv8 model & deteksi
# ============================================================
def load_yolov8_model(model_path: str = "best.pt") -> bool:
    global model

    if YOLO is None:
        print("❌ ultralytics belum ter-install. pip install ultralytics")
        return False

    if not os.path.exists(model_path):
        print(f"⚠️ Model {model_path} tidak ditemukan. fallback ke 'yolov8n.pt'.")
        model_path = "yolov8n.pt"

    try:
        device = 0 if USE_GPU else 'cpu'
        model = YOLO(model_path)

        print(f"✅ YOLOv8 model loaded: {model_path}")
        print(f"📋 Model classes: {model.names}")
        print(f"💻 Running on: {'GPU' if USE_GPU else 'CPU'}")

        try:
            model.fuse()
        except Exception as e:
            print(f"⚠️ Could not fuse model: {e}")

        return True
    except Exception as e:
        print(f"❌ Error loading YOLOv8 model: {e}")
        return False


def class_name(cls_id: int) -> str:
    if model is None or not hasattr(model, "names"):
        return "obj"
    names = model.names
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), "obj"))
        else:
            idx = int(cls_id)
            return str(names[idx]) if 0 <= idx < len(names) else "obj"
    except Exception:
        return "obj"


def detect_objects(frame) -> List:
    if model is None:
        return []

    try:
        img = cv2.resize(frame, DETECTION_SIZE, interpolation=cv2.INTER_AREA)

        results = model.predict(
            img,
            imgsz=DETECTION_SIZE[0],
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0 if USE_GPU else 'cpu',
            verbose=False,
            stream=False
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.data is None:
            return []

        dets = boxes.data.detach().cpu().numpy()  # [N, 6]
        print(f"📦 Detected {len(dets)} objects")
        return dets
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return []


def update_parking_states(detections: List):
    """Update OCC/FREE berdasarkan center bbox yang jatuh ke polygon slot."""
    global frame_width, frame_height, parking_slots, slot_counters, slot_states

    if frame_width == 0 or frame_height == 0:
        return

    slot_polys = get_slot_polygons_abs()
    if not slot_polys:
        return

    scale_x = frame_width / DETECTION_SIZE[0]
    scale_y = frame_height / DETECTION_SIZE[1]

    candidates = {f"slot_{i}": False for i in range(1, N_SLOTS+1)}
    conf_per_slot = {f"slot_{i}": 0.0 for i in range(1, N_SLOTS+1)}

    for x1, y1, x2, y2, conf, cls in (detections or []):
        cx_det = (float(x1) + float(x2)) / 2.0
        cy_det = (float(y1) + float(y2)) / 2.0

        cx = cx_det * scale_x
        cy = cy_det * scale_y

        for i in range(1, N_SLOTS+1):
            key = f"slot_{i}"
            poly = slot_polys.get(key)
            if not poly:
                continue

            if point_in_polygon(cx, cy, poly):
                if conf > conf_per_slot[key]:
                    conf_per_slot[key] = float(conf)

    for i in range(1, N_SLOTS+1):
        key = f"slot_{i}"
        th = SLOT_THRESHOLDS[key]
        candidates[key] = conf_per_slot[key] >= th

    with slots_lock:
        for i in range(1, N_SLOTS+1):
            key = f"slot_{i}"
            if candidates[key]:
                slot_counters[key] = min(slot_counters[key] + 1, SMOOTH_FRAMES)
            else:
                slot_counters[key] = max(slot_counters[key] - 1, -SMOOTH_FRAMES)

            if slot_counters[key] >= SMOOTH_FRAMES:
                slot_states[key] = True
            elif slot_counters[key] <= -SMOOTH_FRAMES:
                slot_states[key] = False

        for k in parking_slots.keys():
            parking_slots[k] = slot_states[k]

        print(f"Updated parking slots: {parking_slots}")


def visualize_slots(frame, detections=None):
    """Gambar polygon slot + status + bbox + titik tengah."""
    global frame_width, frame_height
    h, w, _ = frame.shape
    frame_width = w
    frame_height = h

    slot_polys = get_slot_polygons_abs()

    green  = (0, 255, 0)
    red    = (0, 0, 255)
    white  = (255, 255, 255)
    blue   = (255, 0, 0)
    yellow = (0, 255, 255)
    cyan   = (255, 255, 0)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # Polygon slot
    for i in range(1, N_SLOTS+1):
        key = f"slot_{i}"
        poly = slot_polys.get(key)
        if not poly:
            continue

        occ = parking_slots.get(key, False)
        color = red if occ else green

        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        cx = int(sum(p[0] for p in poly) / len(poly))
        cy = int(sum(p[1] for p in poly) / len(poly))
        status = "OCC" if occ else "FREE"
        cv2.putText(frame, f"S{i}:{status}", (cx - 25, cy),
                    font, 0.5, white, 1)

    # BBOX + titik tengah
    if detections is not None:
        scale_x = frame_width / DETECTION_SIZE[0]
        scale_y = frame_height / DETECTION_SIZE[1]

        for x1, y1, x2, y2, conf, cls in detections:
            x1s = int(x1 * scale_x)
            y1s = int(y1 * scale_y)
            x2s = int(x2 * scale_x)
            y2s = int(y2 * scale_y)

            cx = int((x1s + x2s) / 2)
            cy = int((y1s + y2s) / 2)

            # bbox
            label = class_name(int(cls))
            cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), blue, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1s, y1s - 5),
                        font, 0.5, blue, 1)

            # cek apakah center di salah satu slot
            inside_any = False
            hit_slot = None
            for i in range(1, N_SLOTS+1):
                key = f"slot_{i}"
                poly = slot_polys.get(key)
                if not poly:
                    continue
                if point_in_polygon(cx, cy, poly):
                    inside_any = True
                    hit_slot = i
                    break

            color_center = yellow if inside_any else cyan
            cv2.circle(frame, (cx, cy), 4, color_center, -1)
            if hit_slot is not None:
                cv2.putText(frame, f"S{hit_slot}",
                            (cx + 5, cy + 5),
                            font, 0.4, color_center, 1)

# ============================================================
#  Threads
# ============================================================
def detection_loop():
    """Thread khusus YOLO."""
    global last_detections

    last_detection_time = 0
    while True:
        now = time.time()
        if now - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.1)
            continue

        try:
            frame = detection_queue.get(timeout=0.5)
            t0 = time.time()

            dets = detect_objects(frame)

            # filter confidence
            filtered = []
            for d in dets:
                conf = float(d[4])
                if conf >= CONFIDENCE_THRESHOLD:
                    filtered.append(d)

            # filter hanya yang di dalam slot
            cars_in_slots = filter_detections_inside_slots(filtered)

            with detection_lock:
                last_detections = np.array(cars_in_slots) if len(cars_in_slots) > 0 else None

            update_parking_states(cars_in_slots)

            last_detection_time = now
            print(f"🕗 Detection Time: {time.time() - t0:.3f}s")

            if time.time() - t0 < 1.0:
                update_parking_database()

        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Detection loop error: {e}")
            time.sleep(1)


def camera_feed_thread():
    """Thread pembaca video + encoder MJPEG."""
    global frame_width, frame_height, encoded_frame

    print(f"🎞 Using video file: {VIDEO_SOURCE}")
    cap = SafeCapture(VIDEO_SOURCE).start()

    last_det_enqueue = time.time()
    last_frame_time = time.time()
    frame_interval = 1.0 / TARGET_FPS

    while True:
        now = time.time()
        if now - last_frame_time < frame_interval:
            time.sleep(0.01)
            continue

        last_frame_time = now

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame_height, frame_width = frame.shape[:2]

        # kirim ke thread deteksi
        if now - last_det_enqueue >= DETECTION_INTERVAL:
            try:
                detection_queue.put(frame.copy(), block=False)
                last_det_enqueue = now
            except queue.Full:
                pass

        dets_to_draw = get_last_detections()
        visualize_slots(frame, dets_to_draw)

        try:
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            with frame_lock:
                encoded_frame = buf.tobytes()

        except Exception as e:
            print(f"❌ Encoding error: {e}")
            time.sleep(0.1)

        time.sleep(0.001)

    if cap:
        cap.stop()

# ============================================================
#  Flask routes
# ============================================================
@app.route('/')
def index_route():
    # kalau belum ada template, bisa buat sederhana, mis:
    # <img src="/video_feed">
    return """
    <html>
    <head><title>Parking Monitor</title></head>
    <body style="margin:0;background:#000;">
      <img src="/video_feed" style="width:100vw;height:100vh;object-fit:contain;">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/parking_status')
def parking_status_route():
    occ = sum(parking_slots.values())
    avail = N_SLOTS - occ

    result = {
        "status": parking_slots,
        "summary": {
            "total_slots": N_SLOTS,
            "occupied": occ,
            "available": avail,
            "occupancy_rate": round((occ/float(N_SLOTS))*100, 1),
            "available_slots": [i for i, v in enumerate(parking_slots.values(), 1) if not v]
        }
    }

    db_status = get_latest_parking_status()
    if db_status:
        result['database_status'] = db_status

    return jsonify(result)

@app.route('/api/system_status')
def system_status_route():
    stats = get_system_stats()
    status = {
        "model_loaded": model is not None,
        "camera_connected": encoded_frame is not None,
        "fps": TARGET_FPS,
        "db_interval": DATABASE_UPDATE_INTERVAL,
        "detection_interval": DETECTION_INTERVAL,
        "using_gpu": USE_GPU
    }
    if stats:
        status['statistics'] = stats
    return jsonify(status)

@app.route('/api/parking_history')
def parking_history_route():
    logs = get_parking_history(50)
    return jsonify({"success": True, "logs": logs, "total": len(logs)})

@app.route('/api/available_slots')
def available_slots_route():
    db_status = get_latest_parking_status()
    if not db_status:
        occ = sum(parking_slots.values())
        avail = N_SLOTS - occ
        return jsonify({
            "success": True,
            "available_slots": [i for i, v in enumerate(parking_slots.values(), 1) if not v],
            "available_count": avail,
            "occupied_count": occ,
            "occupancy_rate": round((occ/float(N_SLOTS))*100, 1)
        })

    return jsonify({
        "success": True,
        "available_slots": db_status.get("available_slots", []),
        "available_count": db_status.get("available_count", 0),
        "occupied_count": db_status.get("occupied_count", 0),
        "occupancy_rate": round((db_status.get("occupied_count", 0)/float(N_SLOTS))*100, 1)
    })

@app.route('/api/refresh_detection')
def refresh_detection_route():
    update_parking_database()
    return jsonify({"success": True, "message": "Refreshed"})

# MJPEG generator
def generate_frames():
    frame_interval = 1.0 / TARGET_FPS
    empty_frame_count = 0

    while True:
        try:
            last_frame_time = time.time()

            with frame_lock:
                if encoded_frame is None:
                    empty_frame_count += 1
                    if empty_frame_count > 100:
                        empty = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(empty, "No video signal", (120, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        _, buf = cv2.imencode('.jpg', empty)
                        frame_data = buf.tobytes()
                    else:
                        time.sleep(0.01)
                        continue
                else:
                    empty_frame_count = 0
                    frame_data = encoded_frame

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

            sleep_time = frame_interval - (time.time() - last_frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)

# ============================================================
#  Main
# ============================================================
if __name__ == '__main__':
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'

    print("🚀 Starting Parking Detection System (YOLOv8) with VIDEO FILE & 5 polygon slots...")

    load_slots_from_file()
    initialize_database()

    if not load_yolov8_model():
        print("❌ Model load failed. Exiting.")
        exit(1)

    threading.Thread(target=database_update_thread, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=camera_feed_thread, daemon=True).start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
