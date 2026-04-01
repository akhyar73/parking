import cv2
import torch
import os
import time
import threading
import queue
import json
import sqlite3
from datetime import datetime
from flask import Flask, render_template, jsonify, Response
from typing import Dict, List
from functools import lru_cache

# NEW: YOLOv8 (Ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# App setup
app = Flask(__name__)

# === System Constants (Optimized) ===
TARGET_FPS = 20  # Reduced from 10 to 5 - sufficient for parking monitoring
DETECTION_SIZE = (512, 512)  # Reduced from 320x320 for faster processing
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
DETECTION_INTERVAL = 0.2 # seconds between detections (increased from 1s)
DATABASE_UPDATE_INTERVAL = 3  # seconds (increased from 2s)
DATABASE_FILE = "parking_database.sqlite"  # Changed to SQLite
USE_GPU = torch.cuda.is_available()  # Auto-detect GPU

# === Global Variables ===
model = None  # YOLOv8 model instance
parking_slots: Dict[str, bool] = {f"slot_{i}": False for i in range(1, 7)}
slots_lock = threading.Lock()

# Smoothing: butuh N frame berturut-turut untuk berubah status
SMOOTH_FRAMES = 1
slot_counters = {f"slot_{i}": 0 for i in range(1, 7)}
slot_states   = {f"slot_{i}": False for i in range(1, 7)}

# Threshold overlap/keputusan per-slot (slot ke-5 dibuat lebih ketat)
SLOT_THRESHOLDS = {f"slot_{i}": 0.4 for i in range(1, 7)}
# SLOT_THRESHOLDS["slot_5"] = 0.8

last_parking_state = parking_slots.copy()  # Track changes
frame_width = 0  # Updated each frame

# MJPEG Stream
encoded_frame = None
frame_lock = threading.Lock()
database_lock = threading.Lock()

# Queue for decoupling capture from detection
detection_queue = queue.Queue(maxsize=1)  # Reduced size to minimize latency

# === Database Functions ===
def initialize_database():
    """Initialize SQLite database for better performance."""
    try:
        with database_lock:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()

            # Create tables
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

            # Initialize system stats if needed
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
    """Get system statistics with caching to reduce DB reads."""
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


def update_system_stats(key, value):
    """Update a single system stat."""
    try:
        with database_lock:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("UPDATE system_stats SET value=? WHERE key=?", (value, key))
            conn.commit()
            conn.close()
        # Clear cache
        get_system_stats.cache_clear()
    except Exception as e:
        print(f"❌ Error updating system stat {key}: {e}")


def update_parking_database():
    """Update database only when parking status changes."""
    global last_parking_state

    # Skip if no change in parking status
    if parking_slots == last_parking_state:
        return

    try:
        ts = datetime.now().isoformat()
        occupied = [i for i, occ in [(int(k.split('_')[1]), v) for k, v in parking_slots.items()] if occ]
        available = [i for i in range(1, 7) if i not in occupied]
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

            # Insert new status
            cursor.execute(
                "INSERT INTO parking_status (timestamp, status_json, occupied_count, available_count) VALUES (?, ?, ?, ?)",
                (ts, status_json, occ_count, len(available))
            )

            # Update stats
            cursor.execute("SELECT value FROM system_stats WHERE key='total_detections'")
            total_detections = int(cursor.fetchone()[0]) + 1
            cursor.execute("UPDATE system_stats SET value=? WHERE key='total_detections'",
                          (str(total_detections),))
            cursor.execute("UPDATE system_stats SET value=? WHERE key='last_detection_time'",
                          (ts,))

            # Limit history size - keep only last 100 entries
            cursor.execute("DELETE FROM parking_status WHERE id NOT IN (SELECT id FROM parking_status ORDER BY id DESC LIMIT 100)")

            conn.commit()
            conn.close()

        # Update last state after successful DB update
        last_parking_state = parking_slots.copy()
        print(f"📊 DB updated: Occ={occ_count}, Avail={len(available)}")

        # Clear cache
        get_system_stats.cache_clear()
        get_latest_parking_status.cache_clear()
        get_parking_history.cache_clear()

    except Exception as e:
        print(f"❌ Error updating database: {e}")


@lru_cache(maxsize=1)
def get_latest_parking_status():
    """Get the latest parking status with caching."""
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
    """Get parking history with caching."""
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
                "occupied_slots": [i for i in range(1, 7) if i not in status.get("available_slots", [])],
                "occupancy_rate": round((occ_count/6)*100, 1),
                "available_count": avail_count,
                "occupied_count": occ_count
            })
        return history
    except Exception as e:
        print(f"❌ Error getting parking history: {e}")
        return []


def database_update_thread():
    """Background thread to update database at fixed interval."""
    while True:
        update_parking_database()
        time.sleep(DATABASE_UPDATE_INTERVAL)

# === SafeCapture Class ===
class SafeCapture:
    """Threaded frame capture with buffering."""
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer size
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.queue = queue.Queue(maxsize=1)  # Minimal queue size
        self.running = True
        self.last_frame = None
        self.frame_count = 0
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.last_frame = frame
                    # Replace queue content instead of adding
                    try:
                        self.queue.get_nowait()  # Clear queue
                    except queue.Empty:
                        pass
                    self.queue.put(frame)
            else:
                time.sleep(0.1)  # Longer sleep when no frame

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


# === Model & Detection (YOLOv8) ===
def load_yolov8_model(model_path: str = "best.pt") -> bool:
    """Load YOLOv8 model with optimizations, mirroring prior behavior."""
    global model

    if YOLO is None:
        print("❌ ultralytics not installed. pip install ultralytics")
        return False

    # If a custom path is provided, we trust the user to place it correctly
    if not os.path.exists(model_path):
        # Allow fallback to pretrained small model if custom path missing
        print(f"⚠️ Model not found at {model_path}. Falling back to pretrained 'best.pt' if available.")
        model_path = "yolov8n.pt"  # Gunakan model pretrained jika custom model tidak ditemukan

    try:
        device = 0 if USE_GPU else 'cpu'
        model = YOLO(model_path)

        # Print model info untuk debugging
        print(f"✅ YOLOv8 model loaded: {model_path}")
        print(f"📋 Model classes: {model.names}")
        print(f"💻 Running on: {'GPU' if USE_GPU else 'CPU'}")
        
        # Optional fusing for slightly faster inference on some backends
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
        return "car"  # Default ke "car" jika tidak ada names
    names = model.names
    try:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), "car"))
        else:
            # list/tuple
            idx = int(cls_id)
            return str(names[idx]) if 0 <= idx < len(names) else "car"
    except Exception:
        return "car"  # Default fallback



def detect_objects(frame) -> List:
    """Run YOLOv8 inference and return all detections [x1,y1,x2,y2,conf,cls]."""
    if model is None:
        return []

    try:
        img = cv2.resize(frame, DETECTION_SIZE, interpolation=cv2.INTER_AREA)
        
        # Debug: simpan gambar input untuk deteksi
        try:
            cv2.imwrite("/home/user/debug_input.jpg", img)
            print("💾 Saved detection input image")
        except Exception as e:
            print(f"❌ Error saving debug image: {e}")
        
        results = model.predict(
            img,
            imgsz=DETECTION_SIZE[0],
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0 if USE_GPU else 'cpu',
            verbose=True,  # Ubah ke True untuk melihat output verbose
            stream=False
        )
        
        # Debug: print hasil deteksi mentah
        print(f"📊 YOLOv8 results: {len(results)} results")
        
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.data is None:
            print("⚠️ No boxes detected")
            return []

        dets = boxes.data.detach().cpu().numpy()  # [N, 6]
        print(f"📦 Detected {len(dets)} objects")
        return dets
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return []



def update_parking_states(detections: List):
    global frame_width, parking_slots, slot_counters, slot_states

    if frame_width == 0:
        return

    slot_w = frame_width // 6
    scale = frame_width / DETECTION_SIZE[0]

    # Hitung “candidate occupied” per-slot di frame ini
    candidates = {f"slot_{i}": False for i in range(1, 7)}
    conf_per_slot = {f"slot_{i}": 0.0 for i in range(1, 7)}

    for x1, y1, x2, y2, conf, cls in (detections or []):
        cx = int(((x1 + x2) / 2) * scale)
        idx = min(max(cx // slot_w, 0), 5)
        key = f"slot_{idx+1}"
        # simpan confidence tertinggi per slot
        if conf > conf_per_slot[key]:
            conf_per_slot[key] = float(conf)

    # Terapkan threshold per-slot → kandidat isi
    for i in range(1, 7):
        key = f"slot_{i}"
        th = SLOT_THRESHOLDS[key]
        candidates[key] = conf_per_slot[key] >= th

    # Smoothing: butuh N frame berturut-turut untuk toggling
    with slots_lock:
        for i in range(1, 7):
            key = f"slot_{i}"
            if candidates[key]:
                slot_counters[key] = min(slot_counters[key] + 1, SMOOTH_FRAMES)
            else:
                slot_counters[key] = max(slot_counters[key] - 1, -SMOOTH_FRAMES)

            if slot_counters[key] >= SMOOTH_FRAMES:
                slot_states[key] = True
            elif slot_counters[key] <= -SMOOTH_FRAMES:
                slot_states[key] = False

        # commit ke parking_slots (satu-satunya sumber kebenaran)
        for k in parking_slots.keys():
            parking_slots[k] = slot_states[k]
            
        # Tambahkan log untuk debugging
        print(f"Updated parking slots: {parking_slots}")

def visualize_slots(frame, detections=None):
    """Overlay slot rectangles, status text, and YOLO detections."""
    h, w, _ = frame.shape
    slot_w = w // 6
    start_y = (h - slot_w) // 2
    gap = 5

    # Pre-define colors and fonts
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    blue = (255, 0, 0)   # untuk kotak YOLO
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Gambarkan slot ---
    for i in range(6):
        x1 = i * slot_w + gap//2
        x2 = (i+1) * slot_w - gap//2
        key = f"slot_{i+1}"
        occ = parking_slots[key]

        color = red if occ else green
        cv2.rectangle(frame, (x1, start_y), (x2, start_y+slot_w), color, 2)
        status = "OCC" if occ else "FREE"
        cv2.putText(frame, f"S{i+1}:{status}", (x1+5, start_y+20), font, 0.5, white, 1)

    # --- Gambarkan bounding box YOLO ---
    if detections is not None:
        for x1, y1, x2, y2, conf, cls in detections:
            label = class_name(int(cls))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), blue, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        font, 0.5, blue, 1)

# === Threads ===
def detection_loop():
    """Continuously process frames for detection without blocking capture."""
    last_detection_time = 0
    while True:
        current_time = time.time()
        if current_time - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.1)
            continue

        try:
            frame = detection_queue.get(timeout=0.5)
            t0 = time.time()

            # Deteksi objek
            dets = detect_objects(frame)

            # Print semua deteksi untuk debugging
            if len(dets) > 0:
                print("📦 Semua objek terdeteksi:")
                for i, det in enumerate(dets):
                    conf = float(det[4])
                    cls_id = int(det[5])
                    print(f"  - Deteksi #{i+1}: Class ID={cls_id}, Confidence={conf:.2f}")

            # Karena hanya ada satu kelas (car), kita tidak perlu filter berdasarkan nama kelas
            # Cukup filter berdasarkan confidence jika diperlukan
            cars_only = []
            for d in dets:
                conf = float(d[4])
                # Opsional: tambahkan filter confidence tambahan jika diperlukan
                if conf >= CONFIDENCE_THRESHOLD:
                    cars_only.append(d)

            print(f"Found {len(cars_only)} vehicles for parking detection")
            update_parking_states(cars_only)

            last_detection_time = current_time
            detection_time = time.time() - t0
            print(f"🕗 Detection Time: {detection_time:.3f}s")

            if detection_time < 1.0:
                update_parking_database()

        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Detection error: {e}")
            time.sleep(1)   


def camera_feed_thread():
    """Grab frames, enqueue for detection, render and encode MJPEG."""
    global frame_width, encoded_frame
    rtsp_urls = [
        "rtsp://admin:LabEmbedded24-@192.168.1.64:554/stream/",
        "rtsp://admin:LabEmbedded24-@192.168.1.64:554/Streaming/Channels/101",
        "rtsp://admin:LabEmbedded24-@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:LabEmbedded24-@192.168.1.64/live"
    ]

    # Connection retry logic
    cap = None
    for url in rtsp_urls:
        print(f"🔗 Trying {url}")
        try:
            cap = SafeCapture(url).start()
            time.sleep(2)
            ok, _ = cap.read()
            if ok:
                print(f"✅ Connected to camera via {url}")
                break
            cap.stop()
            cap = None
        except Exception as e:
            print(f"❌ Connection error: {e}")
            if cap:
                cap.stop()
                cap = None

    if not cap:
        print("❌ Unable to connect to any camera.")
        # Fallback to webcam
        try:
            print("⚠️ Trying webcam fallback...")
            cap = SafeCapture(0).start()
        except:
            return

    last_det_enqueue = time.time()
    last_frame_time = time.time()
    frame_interval = 1.0 / TARGET_FPS

    while True:
        current_time = time.time()
        
        # Maintain frame rate - wait if needed
        if current_time - last_frame_time < frame_interval:
            time.sleep(0.01)  # Short sleep to prevent CPU hogging
            continue
            
        last_frame_time = current_time

        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)  # Longer sleep when no frame
            continue

        frame_width = frame.shape[1]

        # Schedule detection at interval - ONLY enqueue, don't process here
        if current_time - last_det_enqueue >= DETECTION_INTERVAL:
            try:
                # Make a copy to avoid race conditions
                detection_queue.put(frame.copy(), block=False)
                last_det_enqueue = current_time
            except queue.Full:
                pass  # Skip if queue is full

        # Visualize slots (without running detection here)
        visualize_slots(frame)

        # Encode frame efficiently - use hardware acceleration if available
        try:
            # Reduce frame size for faster encoding
            if frame.shape[1] > 640:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                
            # Use lower quality for faster encoding
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            
            with frame_lock:
                encoded_frame = buf.tobytes()
                
        except Exception as e:
            print(f"❌ Encoding error: {e}")
            time.sleep(0.1)
            
        # Add a small sleep to prevent tight loop
        time.sleep(0.001)

    if cap:
        cap.stop()


# === Flask Routes ===
@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/user')
def user_route():
    return render_template('user.html')

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/parking_status')
def parking_status_route():
    occ = sum(parking_slots.values())
    avail = 6 - occ

    result = {
        "status": parking_slots,
        "summary": {
            "total_slots": 6,
            "occupied": occ,
            "available": avail,
            "occupancy_rate": round((occ/6)*100, 1),
            "available_slots": [i for i, v in enumerate(parking_slots.values(), 1) if not v]
        }
    }

    # Add database status if available
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
    return jsonify({
        "success": True,
        "logs": logs,
        "total": len(logs)
    })

@app.route('/api/available_slots')
def available_slots_route():
    db_status = get_latest_parking_status()

    if not db_status:
        # Fallback to current memory state
        occ = sum(parking_slots.values())
        avail = 6 - occ
        return jsonify({
            "success": True,
            "available_slots": [i for i, v in enumerate(parking_slots.values(), 1) if not v],
            "available_count": avail,
            "occupied_count": occ,
            "occupancy_rate": round((occ/6)*100, 1)
        })

    return jsonify({
        "success": True,
        "available_slots": db_status.get("available_slots", []),
        "available_count": db_status.get("available_count", 0),
        "occupied_count": db_status.get("occupied_count", 0),
        "occupancy_rate": round((db_status.get("occupied_count", 0)/6)*100, 1)
    })

@app.route('/api/refresh_detection')
def refresh_detection_route():
    update_parking_database()
    return jsonify({"success": True, "message": "Refreshed"})

# Frame generator for MJPEG
def generate_frames():
    frame_interval = 1.0 / TARGET_FPS
    empty_frame_count = 0
    
    while True:
        try:
            last_frame_time = time.time()
            
            with frame_lock:
                if encoded_frame is None:
                    empty_frame_count += 1
                    if empty_frame_count > 100:  # Timeout after ~1 second
                        # Generate empty frame
                        empty = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(empty, "No video signal", (180, 240), 
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
                   
            # Control frame rate
            sleep_time = frame_interval - (time.time() - last_frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)


# === Main Execution ===
if __name__ == '__main__':
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    print("🚀 Starting Parking Detection System (YOLOv8)...")

    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

    # Initialize database
    initialize_database()

    # Load YOLOv8 model (kept same behavior as original: exit if fail)
    if not load_yolov8_model():
        print("❌ Model load failed. Exiting.")
        exit(1)

    # Start background threads
    threading.Thread(target=database_update_thread, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=camera_feed_thread, daemon=True).start()

    # Start Flask with optimized settings
    app.run(host='0.0.0.0', port=5000, threaded=True)