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
from typing import Dict, List, Optional
from functools import lru_cache

# App setup
app = Flask(__name__)

# === System Constants (Optimized) ===
TARGET_FPS = 60  # Reduced from 10 to 5 - sufficient for parking monitoring
DETECTION_SIZE = (256, 256)  # Reduced from 320x320 for faster processing
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.4
DETECTION_INTERVAL = 3.0  # seconds between detections (increased from 1s)
DATABASE_UPDATE_INTERVAL = 5  # seconds (increased from 2s)
DATABASE_FILE = "parking_database.sqlite"  # Changed to SQLite
USE_GPU = torch.cuda.is_available()  # Auto-detect GPU

# === Global Variables ===
model = None  # YOLOv5 model instance
parking_slots: Dict[str, bool] = {f"slot_{i}": False for i in range(1, 7)}
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # reduced buffer size
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.queue = queue.Queue(maxsize=2)  # reduced queue size
        self.running = True
        self.last_frame = None
        self.frame_count = 0

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_count += 1
                # Skip frames to maintain target FPS
                if self.frame_count % 2 == 0:  # Process every other frame
                    if not self.queue.full():
                        self.last_frame = frame
                        self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        try:
            return True, self.queue.get(timeout=0.1)
        except queue.Empty:
            # Return last known frame if queue is empty
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None

    def stop(self):
        self.running = False
        self.cap.release()

# === Model & Detection ===
def load_yolov5_model(model_path="best.pt") -> bool:
    """Load custom YOLOv5 model with optimizations."""
    global model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    try:
        # Load model with appropriate device
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.conf = CONFIDENCE_THRESHOLD
        model.iou = IOU_THRESHOLD

        # Move to GPU if available
        if USE_GPU:
            model.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("✅ Model loaded on CPU")

        # Set model to evaluation mode
        model.eval()

        # Optimize model for inference
        if hasattr(torch, 'inference_mode'):
            # For newer PyTorch versions
            model = torch.inference_mode()(model)
        else:
            # For older PyTorch versions
            model = torch.no_grad()(model)

        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def detect_cars(frame) -> List:
    """Run inference with optimizations."""
    if model is None:
        return []

    try:
        # Resize frame efficiently
        img = cv2.resize(frame, DETECTION_SIZE, interpolation=cv2.INTER_AREA)

        # Run inference with no_grad to save memory
        with torch.no_grad():
            results = model(img)

        # Process results efficiently
        dets = results.xyxy[0].cpu().numpy()
        # Filter for cars (class 2) with confidence above threshold
        return [d for d in dets if int(d[5]) == 0 and d[4] > CONFIDENCE_THRESHOLD]
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return []


def update_parking_states(detections: List):
    """Map detections to slot occupancy."""
    global parking_slots, frame_width

    # Reset slots
    for k in parking_slots:
        parking_slots[k] = False

    if not detections or frame_width == 0:
        return

    # Calculate slot width
    slot_w = frame_width // 6
    scale = frame_width / DETECTION_SIZE[0]

    # Update occupied slots
    for x1, y1, x2, y2, conf, cls in detections:
        cx = int(((x1 + x2) / 2) * scale)
        idx = min(max(cx // slot_w, 0), 5)
        parking_slots[f"slot_{idx+1}"] = True


def visualize_slots(frame):
    """Overlay slot rectangles and status text with optimizations."""
    h, w, _ = frame.shape
    slot_w = w // 6
    start_y = (h - slot_w) // 2
    gap = 5

    # Pre-define colors and fonts for better performance
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(6):
        x1 = i * slot_w + gap//2
        x2 = (i+1) * slot_w - gap//2
        key = f"slot_{i+1}"
        occ = parking_slots[key]
        color = red if occ else green

        # Draw rectangle
        cv2.rectangle(frame, (x1, start_y), (x2, start_y+slot_w), color, 2)  # Reduced thickness

        # Add text - simplified for performance
        status = "OCC" if occ else "FREE"  # Shorter text
        cv2.putText(frame, f"S{i+1}:{status}", (x1+5, start_y+20), font, 0.5, white, 1)  # Reduced text

# === Threads ===
def detection_loop():
    """Continuously process frames for detection without blocking capture."""
    last_detection_time = 0
    while True:
        # Only process if enough time has passed since last detection
        current_time = time.time()
        if current_time - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.1)
            continue

        try:
            frame = detection_queue.get(timeout=0.5)
            t0 = time.time()

            # Run detection
            dets = detect_cars(frame)
            update_parking_states(dets)

            # Update timestamp
            last_detection_time = current_time

            # Log performance
            detection_time = time.time() - t0
            print(f"🕗 Detection: {detection_time:.3f}s")

            # Force update database if detection time is reasonable
            if detection_time < 1.0:
                update_parking_database()

        except queue.Empty:
            # No frame available, wait a bit
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Detection error: {e}")
            time.sleep(1)  # Wait longer on error


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

        # Maintain frame rate
        if current_time - last_frame_time < frame_interval:
            time.sleep(0.005)  # Short sleep to reduce CPU usage
            continue

        last_frame_time = current_time
        t_loop = current_time

        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_width = frame.shape[1]

        # Schedule detection at interval
        if current_time - last_det_enqueue >= DETECTION_INTERVAL:
            try:
                # Use smaller frame for detection
                small_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                detection_queue.put(small_frame, block=False)
                last_det_enqueue = current_time
            except queue.Full:
                pass  # Skip if queue is full

        # Visualize slots
        visualize_slots(frame)

        # Encode frame efficiently - use hardware acceleration if available
        try:
            if hasattr(cv2, 'imencode_gpu'):
                # Use GPU encoding if available (custom OpenCV build)
                _, buf = cv2.imencode_gpu('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            else:
                # Reduce frame size for faster encoding
                if frame.shape[1] > 640:
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            with frame_lock:
                encoded_frame = buf.tobytes()

        except Exception as e:
            print(f"❌ Encoding error: {e}")

        # Log performance occasionally
        if current_time % 10 < 0.1:  # Log every ~10 seconds
            print(f"🕒 Loop: {time.time()-t_loop:.3f}s")

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
    last_frame_time = 0
    frame_interval = 1.0 / TARGET_FPS

    while True:
        current_time = time.time()

        # Maintain frame rate
        if current_time - last_frame_time < frame_interval:
            time.sleep(0.01)
            continue

        last_frame_time = current_time

        with frame_lock:
            if encoded_frame is None:
                time.sleep(0.01)
                continue
            frame_data = encoded_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

# === Main Execution ===
if __name__ == '__main__':
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    print("🚀 Starting Parking Detection System...")

    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

    # Initialize database
    initialize_database()

    # Load model
    if not load_yolov5_model():
        print("❌ Model load failed. Exiting.")
        exit(1)

    # Start background threads
    threading.Thread(target=database_update_thread, daemon=True).start()
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=camera_feed_thread, daemon=True).start()

    # Start Flask with optimized settings
    app.run(host='0.0.0.0', port=5000, threaded=True)
