import cv2
import torch
import json
import time
import base64
import threading
import gc
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import numpy as np
import mysql.connector
import ast
from concurrent.futures import ThreadPoolExecutor
import queue

app = Flask(__name__)

# ==================== OPTIMIZED CONFIGURATION ====================
# Performance settings
DETECTION_INTERVAL = 3          # Deteksi setiap 3 frame (lebih cepat)
TARGET_FPS = 15                 # Target FPS yang realistis
DETECTION_SIZE = (416, 416)     # Ukuran input untuk deteksi (lebih kecil)
DISPLAY_SIZE = (640, 480)       # Ukuran display
CACHE_DURATION = 30             # Cache database selama 30 detik
MAX_QUEUE_SIZE = 2              # Maksimal queue size
CONFIDENCE_THRESHOLD = 0.4      # Lower confidence untuk speed
IOU_THRESHOLD = 0.5             # IoU threshold
OVERLAP_THRESHOLD = 0.25        # Threshold overlap untuk parking detection

# ==================== GLOBAL VARIABLES ====================
model = None
current_frame_b64 = None
frame_lock = threading.Lock()
camera_connected = False
current_source = None
recent_activities = []

# Performance optimization variables
frame_skip_counter = 0
parking_slots_cache = []
last_cache_update = 0
pending_updates = {}
frame_count = 0

# Threading queues
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
detection_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

# ==================== DATABASE FUNCTIONS ====================
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='parking_db',
            autocommit=True,  # Auto commit untuk performance
            use_unicode=True,
            charset='utf8'
        )
        return connection
    except mysql.connector.Error as err:
        print(f"❌ Error connecting to database: {err}")
        return None

def get_parking_slots_cached():
    """Cache parking slots untuk mengurangi database queries"""
    global parking_slots_cache, last_cache_update
    
    current_time = time.time()
    if current_time - last_cache_update > CACHE_DURATION or not parking_slots_cache:
        try:
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM parking_slots")
                parking_slots_cache = cursor.fetchall()
                cursor.close()
                connection.close()
                last_cache_update = current_time
                print(f"📊 Cache updated: {len(parking_slots_cache)} slots loaded")
        except Exception as e:
            print(f"❌ Cache update error: {e}")
    
    return parking_slots_cache

def batch_update_parking_status():
    """Batch update database untuk mengurangi I/O operations"""
    global pending_updates
    
    if not pending_updates:
        return
    
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            
            update_count = 0
            for slot_name, is_occupied in pending_updates.items():
                cursor.execute(
                    "UPDATE parking_slots SET occupied = %s, last_update = NOW() WHERE slot_name = %s", 
                    (is_occupied, slot_name)
                )
                update_count += 1
            
            connection.commit()
            cursor.close()
            connection.close()
            
            if update_count > 0:
                print(f"📝 Batch updated {update_count} parking slots")
            
            pending_updates.clear()
            
            # Invalidate cache setelah update
            global last_cache_update
            last_cache_update = 0
            
    except Exception as e:
        print(f"❌ Batch update error: {e}")

# ==================== MODEL OPTIMIZATION ====================
def load_optimized_model():
    """Load model dengan optimasi untuk speed"""
    global model
    try:
        print("🤖 Loading optimized YOLOv5 model...")
        
        # Gunakan YOLOv5n (nano) untuk speed terbaik
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        
        # Set confidence dan IoU threshold
        model.conf = CONFIDENCE_THRESHOLD
        model.iou = IOU_THRESHOLD
        
        # Set device (GPU jika tersedia)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Warm up model dengan dummy input
        print(f"🔥 Warming up model on {device}...")
        dummy_img = torch.zeros((1, 3, *DETECTION_SIZE)).to(device)
        with torch.no_grad():
            _ = model(dummy_img)
        
        print(f"✅ Optimized YOLOv5n loaded successfully on {device}")
        print(f"📊 Model settings: conf={CONFIDENCE_THRESHOLD}, iou={IOU_THRESHOLD}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading optimized model: {e}")
        return False

# ==================== FRAME PROCESSING OPTIMIZATION ====================
def frame_to_base64_optimized(frame):
    """Optimized frame encoding dengan compression"""
    try:
        # Encode dengan quality yang lebih rendah untuk speed
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"❌ Frame encoding error: {e}")
        return None

def resize_frame_smart(frame, target_size):
    """Smart resize yang maintain aspect ratio"""
    height, width = frame.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scale
    scale = min(target_width / width, target_height / height)
    
    if scale < 1.0:  # Only resize if frame is larger
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return frame, scale

def create_optimized_placeholder():
    """Create placeholder dengan informasi system status"""
    frame = np.zeros((*DISPLAY_SIZE[::-1], 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(DISPLAY_SIZE[1]):
        intensity = int(20 + (i / DISPLAY_SIZE[1]) * 30)
        frame[i, :] = [intensity, intensity, intensity]
    
    # Status text
    cv2.putText(frame, "PARKING MONITORING SYSTEM", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, "Searching for camera feed...", (80, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, f"Time: {timestamp}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # System info
    device = "GPU" if torch.cuda.is_available() else "CPU"
    cv2.putText(frame, f"Device: {device}", (10, DISPLAY_SIZE[1] - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Target FPS: {TARGET_FPS}", (10, DISPLAY_SIZE[1] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Detection: Every {DETECTION_INTERVAL} frames", (10, DISPLAY_SIZE[1] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# ==================== OPTIMIZED DETECTION ====================
def detect_vehicles_optimized(frame):
    """Optimized vehicle detection dengan frame skipping"""
    global frame_skip_counter, model
    
    if model is None:
        return []
    
    # Frame skipping untuk performance
    frame_skip_counter += 1
    if frame_skip_counter < DETECTION_INTERVAL:
        return []
    
    frame_skip_counter = 0
    
    try:
        # Resize frame untuk deteksi (smaller = faster)
        detection_frame, scale = resize_frame_smart(frame, DETECTION_SIZE)
        
        # Run detection dengan no_grad untuk memory efficiency
        with torch.no_grad():
            results = model(detection_frame)
            detections = results.xyxy[0].cpu().numpy()
        
        # Scale detections back ke original frame size
        if len(detections) > 0 and scale < 1.0:
            detections[:, [0, 2]] /= scale  # Scale x coordinates
            detections[:, [1, 3]] /= scale  # Scale y coordinates
        
        # Filter hanya vehicle classes (car, motorcycle, bus, truck)
        vehicle_classes = [2, 3, 5, 7]
        vehicle_detections = []
        
        for detection in detections:
            if len(detection) > 5 and int(detection[5]) in vehicle_classes:
                vehicle_detections.append(detection)
        
        return vehicle_detections
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return []

def update_parking_status_optimized(detections):
    """Update parking status dengan optimized algorithm"""
    global pending_updates
    
    parking_slots = get_parking_slots_cached()
    
    for slot in parking_slots:
        slot_name = slot['slot_name']
        was_occupied = slot['occupied']
        is_occupied = False
        
        # Parse coordinates
        try:
            coords = ast.literal_eval(slot['coords']) if isinstance(slot['coords'], str) else slot['coords']
            x1, y1, x2, y2 = coords
        except:
            continue
        
        # Check overlap dengan vehicle detections
        for detection in detections:
            det_x1, det_y1, det_x2, det_y2 = detection[:4]
            
            # Calculate overlap area
            overlap_x1 = max(x1, det_x1)
            overlap_y1 = max(y1, det_y1)
            overlap_x2 = min(x2, det_x2)
            overlap_y2 = min(y2, det_y2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                detection_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                
                if detection_area > 0:
                    overlap_ratio = overlap_area / detection_area
                    if overlap_ratio > OVERLAP_THRESHOLD:
                        is_occupied = True
                        break
        
        # Queue update jika ada perubahan
        if was_occupied != is_occupied:
            pending_updates[slot_name] = is_occupied
            print(f"🚗 Slot {slot_name}: {'OCCUPIED' if is_occupied else 'AVAILABLE'}")

def draw_parking_areas_optimized(frame):
    """Optimized drawing dengan cached data"""
    parking_slots = get_parking_slots_cached()
    
    for slot in parking_slots:
        try:
            slot_name = slot['slot_name']
            coords = ast.literal_eval(slot['coords']) if isinstance(slot['coords'], str) else slot['coords']
            x1, y1, x2, y2 = map(int, coords)
            
            # Color coding
            color = (0, 0, 255) if slot['occupied'] else (0, 255, 0)
            status_text = 'OCCUPIED' if slot['occupied'] else 'AVAILABLE'
            
            # Draw rectangle dengan thickness yang optimal
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Optimized text rendering
            label = f"{slot_name}: {status_text}"
            font_scale = 0.5
            thickness = 1
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Background rectangle untuk text
            bg_y1 = max(0, y1 - text_height - 10)
            bg_y2 = y1
            cv2.rectangle(frame, (x1, bg_y1), (x1 + text_width + 10, bg_y2), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
        except Exception as e:
            print(f"❌ Drawing error for slot {slot.get('slot_name', 'unknown')}: {e}")
            continue
    
    return frame

# ==================== FIXED CAMERA HANDLING ====================
def test_camera_connection_fast():
    """Fixed camera connection testing tanpa CAP_PROP_TIMEOUT"""
    global current_source, camera_connected
    
    rtsp_urls = [
        "rtsp://admin:BengkelIT@192.168.1.64:554/Streaming/Channels/1",
        "rtsp://admin:BengkelIT@192.168.1.64:554/Streaming/Channels/101", 
        "rtsp://admin:BengkelIT@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0",
        "rtsp://admin:BengkelIT@192.168.1.64/live"
    ]
    
    print("🔍 Fast camera connection test...")
    
    for i, url in enumerate(rtsp_urls):
        print(f"📡 Testing RTSP {i+1}/{len(rtsp_urls)}")
        cap = cv2.VideoCapture(url)
        
        # Set basic properties (tanpa TIMEOUT yang bermasalah)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test dengan timeout manual
        start_time = time.time()
        timeout_seconds = 5
        
        if cap.isOpened():
            # Try to read frame dengan manual timeout
            while time.time() - start_time < timeout_seconds:
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ RTSP {i+1} SUCCESS!")
                    current_source = url
                    camera_connected = True
                    cap.release()
                    return url
                time.sleep(0.1)
        
        cap.release()
        print(f"❌ RTSP {i+1} failed")
        time.sleep(0.5)
    
    # Webcam fallback
    print("📹 Testing webcam...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        start_time = time.time()
        while time.time() - start_time < 3:  # 3 second timeout
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Webcam SUCCESS!")
                current_source = 0
                camera_connected = True
                cap.release()
                return 0
            time.sleep(0.1)
    cap.release()
    
    print("❌ No camera available!")
    camera_connected = False
    return None

def optimized_camera_thread():
    """Optimized camera capture dengan better performance"""
    global current_frame_b64, camera_connected, current_source, frame_count
    
    print("📹 Starting optimized camera thread...")
    
    while True:
        if current_source is None:
            current_source = test_camera_connection_fast()
            if current_source is None:
                placeholder = create_optimized_placeholder()
                with frame_lock:
                    current_frame_b64 = frame_to_base64_optimized(placeholder)
                time.sleep(5)
                continue
        
        cap = cv2.VideoCapture(current_source)
        
        # Optimized camera settings (tanpa yang bermasalah)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
            
            # Try to set FOURCC, tapi skip jika error
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except:
                pass  # Skip jika tidak didukung
                
        except Exception as e:
            print(f"⚠️ Warning setting camera properties: {e}")
        
        consecutive_failures = 0
        last_detection_time = 0
        
        print(f"🎥 Camera connected: {current_source}")
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 15:  # Increased tolerance
                    print("❌ Too many consecutive failures, reconnecting...")
                    break
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            # Resize frame jika perlu
            if frame.shape[:2] != DISPLAY_SIZE[::-1]:
                frame = cv2.resize(frame, DISPLAY_SIZE)
            
            # Run detection (dengan frame skipping)
            current_time = time.time()
            detections = detect_vehicles_optimized(frame.copy())
            
            if detections:
                update_parking_status_optimized(detections)
                last_detection_time = current_time
            
            # Draw parking areas
            processed_frame = draw_parking_areas_optimized(frame.copy())
            
            # Add performance info
            fps = 1.0 / (time.time() - start_time) if time.time() - start_time > 0 else 0
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            cv2.putText(processed_frame, f"Time: {timestamp}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            detection_status = "DETECTING" if frame_count % DETECTION_INTERVAL == 0 else "SKIPPING"
            cv2.putText(processed_frame, detection_status, (10, DISPLAY_SIZE[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Update frame
            frame_b64 = frame_to_base64_optimized(processed_frame)
            if frame_b64:
                with frame_lock:
                    current_frame_b64 = frame_b64
                    camera_connected = True
            
            # Memory cleanup setiap 100 frames
            if frame_count % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Frame rate control
            elapsed = time.time() - start_time
            target_time = 1.0 / TARGET_FPS
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
        
        cap.release()
        camera_connected = False
        current_source = None
        print("🔄 Reconnecting camera...")
        time.sleep(2)

# ==================== BACKGROUND TASKS ====================
def start_background_tasks():
    """Start background tasks untuk database updates"""
    def batch_updater():
        while True:
            batch_update_parking_status()
            time.sleep(2)  # Update setiap 2 detik
    
    def cache_refresher():
        while True:
            time.sleep(CACHE_DURATION)
            get_parking_slots_cached()  # Refresh cache
    
    # Start background threads
    batch_thread = threading.Thread(target=batch_updater, daemon=True)
    cache_thread = threading.Thread(target=cache_refresher, daemon=True)
    
    batch_thread.start()
    cache_thread.start()
    
    print("🔄 Background tasks started")

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_frame')
def get_frame():
    with frame_lock:
        if current_frame_b64:
            return jsonify({
                'success': True,
                'frame': current_frame_b64,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'connected': camera_connected,
                'frame_count': frame_count,
                'detection_interval': DETECTION_INTERVAL
            })
        else:
            placeholder = create_optimized_placeholder()
            placeholder_b64 = frame_to_base64_optimized(placeholder)
            return jsonify({
                'success': False,
                'frame': placeholder_b64,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'connected': False
            })

@app.route('/api/parking_status')
def get_parking_status():
    parking_slots = get_parking_slots_cached()
    
    total_slots = len(parking_slots)
    occupied_slots = sum(1 for slot in parking_slots if slot['occupied'])
    available_slots = total_slots - occupied_slots

    status_data = {
        'slots': {},
        'summary': {
            'total': total_slots,
            'occupied': occupied_slots,
            'available': available_slots
        },
        'performance': {
            'target_fps': TARGET_FPS,
            'detection_interval': DETECTION_INTERVAL,
            'cache_duration': CACHE_DURATION,
            'pending_updates': len(pending_updates)
        },
        'recent_activities': recent_activities[:5],
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'camera_connected': camera_connected,
        'camera_source': str(current_source) if current_source else 'No camera'
    }

    for slot in parking_slots:
        slot_name = slot['slot_name']
        status_data['slots'][slot_name.lower()] = {
            'name': slot_name,
            'occupied': slot['occupied'],
            'status': 'Occupied' if slot['occupied'] else 'Available',
            'last_update': slot['last_update'].strftime('%H:%M:%S') if slot['last_update'] else None
        }

    return jsonify(status_data)

@app.route('/api/toggle_slot/<slot_name>', methods=['POST'])
def toggle_slot(slot_name):
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'success': False, 'message': 'Database connection failed'})
        
        cursor = connection.cursor()
        cursor.execute("SELECT occupied FROM parking_slots WHERE slot_name = %s", (slot_name,))
        result = cursor.fetchone()

        if result:
            occupied = result[0]
            new_status = not occupied

            cursor.execute("UPDATE parking_slots SET occupied = %s, last_update = NOW() WHERE slot_name = %s", 
                         (new_status, slot_name))
            connection.commit()
            
            # Invalidate cache
            global last_cache_update
            last_cache_update = 0

            cursor.close()
            connection.close()

            return jsonify({'success': True, 'slot': slot_name, 'occupied': new_status})

        cursor.close()
        connection.close()
        return jsonify({'success': False, 'message': 'Slot not found'})
        
    except Exception as e:
        print(f"❌ Toggle slot error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/system_stats')
def get_system_stats():
    """Endpoint untuk monitoring system performance"""
    return jsonify({
        'performance': {
            'frame_count': frame_count,
            'target_fps': TARGET_FPS,
            'detection_interval': DETECTION_INTERVAL,
            'cache_duration': CACHE_DURATION,
            'pending_updates': len(pending_updates),
            'cached_slots': len(parking_slots_cache)
        },
        'camera': {
            'connected': camera_connected,
            'source': str(current_source) if current_source else None
        },
        'model': {
            'loaded': model is not None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'confidence': CONFIDENCE_THRESHOLD,
            'iou': IOU_THRESHOLD
        }
    })

# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    print("🚀 Starting OPTIMIZED Parking Monitoring System...")
    print("=" * 60)
    print(f"🎯 Target FPS: {TARGET_FPS}")
    print(f"🔍 Detection interval: Every {DETECTION_INTERVAL} frames")
    print(f"📏 Detection size: {DETECTION_SIZE}")
    print(f"📺 Display size: {DISPLAY_SIZE}")
    print(f"💾 Cache duration: {CACHE_DURATION}s")
    print("=" * 60)
    
    print("🤖 Loading optimized AI model...")
    if load_optimized_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Model loading failed!")
    
    print("🔄 Starting background tasks...")
    start_background_tasks()
    
    print("📹 Starting optimized camera thread...")
    camera_thread = threading.Thread(target=optimized_camera_thread, daemon=True)
    camera_thread.start()
    
    print("🌐 Starting Flask web server...")
    print("🎉 System ready! Access at http://localhost:5000")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)