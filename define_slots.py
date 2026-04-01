import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
from flask import Flask, jsonify, render_template, request

# ======================================
# KONFIGURASI
# ======================================
MODEL_PATH = "yolo11s.pt"
VIDEO_PATH = "masuk1.mp4"

CONF_THRES = 0.20
IMGSZ = 640

TARGET_FPS = 10
SKIP_DET_EVERY = 3  # YOLO tiap N frame

# gate bottom point
USE_BOTTOM_POINT_GATE = True
BOTTOM_POINT_FRAC = 0.85  # titik uji dekat roda (0.85)

# COCO IDs: 2=car, 7=truck (sesuaikan)
FILTER_CLASSES = [2, 7]

SLOTS_FILE = "slots.json"

# harus terlihat selama ini agar jadi FULL (waktu nyata)
OCCUPIED_SECONDS = 1.0

# toleransi hilang sesaat (anti flicker bayangan)
MISSING_TOLERANCE = 1.2  # detik (0.5-1.2 biasanya aman)

# filter noise bbox
MIN_BBOX_AREA = 1500

# OPSIONAL: crop ROI untuk FPS lebih tinggi
ROI = None  # contoh: ROI = (x1, y1, x2, y2)

# ======================================
# FITUR SAVE VIDEO (REKAM OUTPUT)
# ======================================
ENABLE_RECORDING_FEATURE = True
RECORD_ONLY_WHEN_DETEKSI = True
RECORD_DIR = "recordings"
RECORD_FOURCC = "mp4v"
RECORD_FPS = TARGET_FPS

# ======================================
# GLOBAL SLOT
# ======================================
slots = []
current_polygon = []
define_mode = True
slot_counter = 1

# HANYA 2 STATUS: empty / full
slot_state = {}      # empty / full
slot_timer = {}      # 0..OCCUPIED_SECONDS
slot_last_seen = {}  # kapan terakhir present=True (anti flicker)

# mask slot
slot_masks = {}
slot_area = {}
masks_dirty = True
mask_hw = None

# status instant terakhir (dipakai untuk update timer setiap frame)
last_instant_status = {}
initial_state_applied = False

# ======================================
# API
# ======================================
app = Flask(__name__)
latest_status = {"updated_at": None, "slots": {}}

metrics = {
    "fps": 0.0,
    "api_calls": 0,
    "api_total_ms": 0.0,
    "api_last_ms": 0.0,
    "tp": 0, "tn": 0, "fp": 0, "fn": 0,
}

def get_metrics_snapshot():
    total_api = metrics["api_calls"]
    avg_api_ms = metrics["api_total_ms"] / total_api if total_api > 0 else 0.0

    total_cases = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
    accuracy = (metrics["tp"] + metrics["tn"]) / total_cases if total_cases > 0 else 0.0
    precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0.0
    recall = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0.0

    return {
        "fps": metrics["fps"],
        "api_last_ms": metrics["api_last_ms"],
        "api_avg_ms": avg_api_ms,
        "api_calls": metrics["api_calls"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/parking-status", methods=["GET"])
def api_parking_status():
    start = time.time()
    resp = {
        "request_time": datetime.utcnow().isoformat() + "Z",  # ← INI BARU
        "updated_at": latest_status.get("updated_at"),
        "slots": latest_status.get("slots", {}),
        "metrics": get_metrics_snapshot()
    }
    duration_ms = (time.time() - start) * 1000.0
    metrics["api_calls"] += 1
    metrics["api_total_ms"] += duration_ms
    metrics["api_last_ms"] = duration_ms
    return jsonify(resp)

@app.route("/api/groundtruth", methods=["POST"])
def api_groundtruth():
    data = request.get_json(silent=True) or {}
    gt_slots = data.get("slots", {})

    per_slot_result = {}
    tp = tn = fp = fn = 0

    for s in slots:
        name = s["name"]
        gt = bool(gt_slots.get(name, False))
        pred = (slot_state.get(name, "empty") == "full")

        if gt and pred:
            tp += 1; res = "TP"
        elif (not gt) and (not pred):
            tn += 1; res = "TN"
        elif (not gt) and pred:
            fp += 1; res = "FP"
        else:
            fn += 1; res = "FN"

        per_slot_result[name] = {"gt_occupied": gt, "pred_occupied": pred, "result": res}

    metrics["tp"] += tp
    metrics["tn"] += tn
    metrics["fp"] += fp
    metrics["fn"] += fn

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return jsonify({
        "batch_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "batch_metrics": {"accuracy": accuracy, "precision": precision, "recall": recall},
        "overall_confusion": {"tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"]},
        "overall_metrics": get_metrics_snapshot(),
        "per_slot": per_slot_result
    })

def run_api_server():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# ======================================
# LOAD / SAVE SLOT
# ======================================
def _init_slot_states(now=None):
    """Inisialisasi state, timer, last_seen, dan last_instant_status."""
    global slot_state, slot_timer, slot_last_seen, last_instant_status
    if now is None:
        now = time.time()
    slot_state = {s["name"]: "empty" for s in slots}
    slot_timer = {s["name"]: 0.0 for s in slots}
    slot_last_seen = {s["name"]: 0.0 for s in slots}
    last_instant_status = {s["name"]: False for s in slots}

def load_slots():
    global slots, slot_counter, masks_dirty

    if not os.path.exists(SLOTS_FILE):
        slots = []
        slot_counter = 1
        masks_dirty = True
        print("slots.json belum ada, mulai dari kosong.")
        _init_slot_states()
        return

    with open(SLOTS_FILE, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        print("Format slots.json tidak sesuai. Diabaikan.")
        slots = []
        slot_counter = 1
        masks_dirty = True
        _init_slot_states()
        return

    slots = []
    for d in data:
        if not isinstance(d, dict):
            continue
        poly = [tuple(p) for p in d.get("points", [])]
        if len(poly) >= 3:
            slots.append({"name": d.get("name", f"Slot {len(slots)+1}"), "poly": poly})

    slot_counter = len(slots) + 1
    masks_dirty = True
    _init_slot_states()
    print(f"Loaded {len(slots)} slots dari {SLOTS_FILE}")

def save_slots():
    data = []
    for s in slots:
        data.append({"name": s["name"], "points": [[int(x), int(y)] for (x, y) in s["poly"]]})
    with open(SLOTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Slots disimpan ke {SLOTS_FILE} (total {len(slots)} slot)")

# ======================================
# BUILD MASK SLOT
# ======================================
def build_slot_masks(frame):
    global slot_masks, slot_area, masks_dirty, mask_hw

    h, w = frame.shape[:2]
    if (not masks_dirty) and (mask_hw == (h, w)):
        return

    slot_masks = {}
    slot_area = {}
    mask_hw = (h, w)

    for s in slots:
        name = s["name"]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(s["poly"], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        slot_masks[name] = mask
        slot_area[name] = max(int(np.count_nonzero(mask)), 1)

    masks_dirty = False

# ======================================
# MOUSE CALLBACK
# ======================================
def mouse_callback(event, x, y, flags, param):
    global current_polygon, define_mode
    if not define_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))
        print(f"Titik ditambah: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        finish_current_polygon()

def finish_current_polygon():
    global current_polygon, slots, slot_counter, masks_dirty

    if len(current_polygon) >= 3:
        slot_name = f"Slot {slot_counter}"
        slot_counter += 1
        slots.append({"name": slot_name, "poly": current_polygon.copy()})
        current_polygon = []
        masks_dirty = True
        save_slots()
        _init_slot_states()
        print(f"Slot baru dibuat: {slot_name}")
    else:
        print("Minimal 3 titik untuk membuat 1 slot.")

# ======================================
# DRAW BBOX (DEBUG)
# ======================================
def draw_bboxes(frame, detections, class_names=None):
    if not detections:
        return
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2, conf, cls_int) in detections:
        x1i = int(max(0, min(w - 1, x1)))
        y1i = int(max(0, min(h - 1, y1)))
        x2i = int(max(0, min(w - 1, x2)))
        y2i = int(max(0, min(h - 1, y2)))

        bw = x2i - x1i
        bh = y2i - y1i
        if bw <= 1 or bh <= 1:
            continue
        if (bw * bh) < MIN_BBOX_AREA:
            continue

        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (255, 0, 255), 2)

        if class_names is not None and int(cls_int) in class_names:
            cname = class_names[int(cls_int)]
        else:
            cname = str(int(cls_int))
        label = f"{cname} {conf:.2f}"
        ty = max(15, y1i - 7)
        cv2.putText(frame, label, (x1i, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if USE_BOTTOM_POINT_GATE:
            cx = int((x1i + x2i) / 2)
            py = int(y1i + bh * BOTTOM_POINT_FRAC)
            cx = max(0, min(w - 1, cx))
            py = max(0, min(h - 1, py))
            cv2.circle(frame, (cx, py), 4, (255, 0, 255), -1)

# ======================================
# OCCUPANCY (BEST OVERLAP + GATES)
# ======================================
def compute_slot_occupancy_instant(detections):
    status = {s["name"]: False for s in slots}
    if not slot_masks or mask_hw is None:
        return status

    h, w = mask_hw

    for (x1, y1, x2, y2, conf, cls_int) in detections:
        x1i = int(max(0, min(w - 1, x1)))
        y1i = int(max(0, min(h - 1, y1)))
        x2i = int(max(0, min(w, x2)))
        y2i = int(max(0, min(h, y2)))

        bw = x2i - x1i
        bh = y2i - y1i
        if bw <= 1 or bh <= 1:
            continue

        bbox_area = bw * bh
        if bbox_area < MIN_BBOX_AREA:
            continue

        best_name = None
        best_overlap = 0

        for s in slots:
            name = s["name"]
            roi = slot_masks[name][y1i:y2i, x1i:x2i]
            overlap = int(np.count_nonzero(roi))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name

        if best_name is None or best_overlap <= 0:
            continue

        # gate bottom-point
        if USE_BOTTOM_POINT_GATE:
            cx = int((x1i + x2i) / 2)
            py = int(y1i + bh * BOTTOM_POINT_FRAC)
            cx = max(0, min(w - 1, cx))
            py = max(0, min(h - 1, py))
            if slot_masks[best_name][py, cx] == 0:
                continue

        status[best_name] = True

    return status

# ======================================
# STATE UPDATE (ANTI FLICKER) - 2 STATUS
# ======================================
def update_slot_states(instant_status, dt, now):
    global slot_state, slot_timer, slot_last_seen

    dt = max(0.0, min(dt, 0.2))

    for s in slots:
        name = s["name"]
        present = bool(instant_status.get(name, False))
        timer = float(slot_timer.get(name, 0.0))

        if present:
            # mobil terdeteksi → hitung verifikasi masuk
            slot_last_seen[name] = now
            timer = min(timer + dt, OCCUPIED_SECONDS)
        else:
            # mobil keluar → langsung kosong
            timer = 0.0
            slot_last_seen[name] = 0.0

        slot_timer[name] = timer
        slot_state[name] = "full" if timer >= OCCUPIED_SECONDS else "empty"


# ======================================
# DRAW SLOT
# ======================================
def draw_slots(frame, show_state=False):
    for s in slots:
        poly = s["poly"]
        if len(poly) < 2:
            continue
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        name = s["name"]

        if not show_state:
            color = (255, 0, 0)
            label = name
        else:
            state = slot_state.get(name, "empty")
            if state == "full":
                color = (0, 0, 255)  # merah
            else:
                color = (0, 255, 0)  # hijau
            label = f"{name}: {state.upper()} ({slot_timer.get(name, 0.0):.1f}s)"

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        x0, y0 = poly[0]
        cv2.putText(frame, label, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if len(current_polygon) > 0:
        for i, (x, y) in enumerate(current_polygon):
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
            if i > 0:
                cv2.line(frame, current_polygon[i - 1], (x, y), (0, 255, 255), 1)

# ======================================
# VIDEO WRITER
# ======================================
def make_video_writer(frame_w, frame_h):
    os.makedirs(RECORD_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RECORD_DIR, f"output_{ts}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*RECORD_FOURCC)
    writer = cv2.VideoWriter(out_path, fourcc, float(RECORD_FPS), (int(frame_w), int(frame_h)))

    if not writer.isOpened():
        print("GAGAL membuat VideoWriter. Coba ganti FOURCC ke 'avc1' atau install codec.")
        return None, None

    print(f"[REC] Rekam dimulai: {out_path}")
    return writer, out_path

# ======================================
# MAIN LOOP
# ======================================
def main_detector():
    global define_mode, latest_status, masks_dirty, last_instant_status, initial_state_applied

    print("Loading model YOLO...")
    model = YOLO(MODEL_PATH)
    print("Model loaded:", MODEL_PATH)
    print("Daftar kelas:", model.names)

    load_slots()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Tidak bisa membuka video:", VIDEO_PATH)
        return

    window_name = "Parking Slots - YOLO"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    fps_smooth = 0.0
    t_last_loop = time.time()
    frame_idx = 0

    last_detections = []
    recording = False
    writer = None
    writer_path = None

    print("Kontrol:")
    print("  D : ganti mode EDIT / DETEKSI")
    print("  S : simpan slot ke slots.json")
    print("  N : selesaikan 1 polygon slot (klik kanan)")
    print("  R : start/stop rekam video output")
    print("  Q / ESC : keluar")

    while True:
        now = time.time()
        loop_time = now - t_last_loop
        t_last_loop = now

        ret, frame = cap.read()
        if not ret:
            print("Video selesai, ulang dari awal...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            initial_state_applied = False

            last_detections = []
            for name in list(slot_state.keys()):
                slot_state[name] = "empty"
                slot_timer[name] = 0.0
                slot_last_seen[name] = 0.0
                last_instant_status[name] = False
            continue

        build_slot_masks(frame)
        frame_idx += 1

        detections = last_detections

        if not define_mode:
            run_infer = (frame_idx % SKIP_DET_EVERY == 0)

            if run_infer:
                if ROI is not None:
                    x1r, y1r, x2r, y2r = ROI
                    x1r = max(0, x1r); y1r = max(0, y1r)
                    x2r = min(frame.shape[1], x2r); y2r = min(frame.shape[0], y2r)
                    frame_infer = frame[y1r:y2r, x1r:x2r]
                    offset = (x1r, y1r)
                else:
                    frame_infer = frame
                    offset = (0, 0)

                results = model(
                    frame_infer,
                    conf=CONF_THRES,
                    imgsz=IMGSZ,
                    classes=FILTER_CLASSES,
                    verbose=False
                )

                detections = []
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes
                    if boxes is not None and boxes.data is not None and boxes.data.numel() > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        ox, oy = offset

                        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                            cls_int = int(cls)
                            if cls_int not in FILTER_CLASSES:
                                continue
                            detections.append([
                                float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy),
                                float(conf), cls_int
                            ])

                last_detections = detections
                last_instant_status = compute_slot_occupancy_instant(last_detections)
                # =====================================
            # 🔥 INISIALISASI KONDISI AWAL SLOT
            # =====================================
            if not initial_state_applied:
                for name, present in last_instant_status.items():
                    if present:
                        slot_timer[name] = OCCUPIED_SECONDS
                        slot_state[name] = "full"
                        slot_last_seen[name] = now
                initial_state_applied = True


            # update state tiap frame (realtime)
            update_slot_states(last_instant_status, loop_time, now)

            # Update API
            latest_status["updated_at"] = datetime.utcnow().isoformat() + "Z"
            latest_status["slots"] = {}
            for name in slot_state.keys():
                state = slot_state.get(name, "empty")
                timer = float(slot_timer.get(name, 0.0))
                occupied = (state == "full")  # ONLY FULL = True
                latest_status["slots"][name] = {
                    "occupied": occupied,
                    "state": state,
                    "timer": timer
                }

            draw_slots(frame, show_state=True)
            draw_bboxes(frame, last_detections, model.names)

        else:
            draw_slots(frame, show_state=False)

        # overlay mode + fps
        mode_text = "MODE: EDIT SLOTS" if define_mode else "MODE: DETEKSI"
        mode_color = (0, 255, 0) if define_mode else (0, 128, 255)
        cv2.putText(frame, mode_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        fps = 1.0 / loop_time if loop_time > 0 else 0.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps
        metrics["fps"] = float(fps_smooth)
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if recording:
            cv2.putText(frame, "REC", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow(window_name, frame)

        # rekam output
        if ENABLE_RECORDING_FEATURE and recording:
            if (not RECORD_ONLY_WHEN_DETEKSI) or (define_mode is False):
                if writer is None:
                    h, w = frame.shape[:2]
                    writer, writer_path = make_video_writer(w, h)
                    if writer is None:
                        recording = False
                if writer is not None:
                    writer.write(frame)

        # throttle
        elapsed = time.time() - now
        target_interval = 1.0 / TARGET_FPS
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key in (ord('d'), ord('D')):
            define_mode = not define_mode
            print("Mode sekarang:", "EDIT" if define_mode else "DETEKSI")
            last_detections = []
            for name in slot_state.keys():
                slot_state[name] = "empty"
                slot_timer[name] = 0.0
                slot_last_seen[name] = 0.0
                last_instant_status[name] = False
        elif key in (ord('s'), ord('S')):
            save_slots()
            masks_dirty = True
        elif key in (ord('n'), ord('N')):
            if define_mode:
                finish_current_polygon()
        elif key in (ord('r'), ord('R')):
            if not ENABLE_RECORDING_FEATURE:
                continue
            recording = not recording
            if recording:
                print("[REC] ON (tekan R lagi untuk stop)")
            else:
                print("[REC] OFF")
                if writer is not None:
                    writer.release()
                    print(f"[REC] File tersimpan: {writer_path}")
                writer = None
                writer_path = None

    if writer is not None:
        writer.release()
        print(f"[REC] File tersimpan: {writer_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai deteksi.")

# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    api_thread = Thread(target=run_api_server, daemon=True)
    api_thread.start()
    main_detector()
