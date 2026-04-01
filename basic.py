from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolo11s.pt")

# Path gambar
image_path = "car.jpg"

# Baca gambar
img = cv2.imread(image_path)

# Run detection
results = model.predict(img, conf=0.5, verbose=False)

for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])              # Class index
        conf = float(box.conf[0])          # Confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box

        # Print ke terminal
        print(f"Class: {model.names[cls]} | Conf: {conf:.2f} | BBox: [{x1}, {y1}, {x2}, {y2}]")

        # Gambar bbox di image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Simpan hasil ke file
output_path = "output.jpg"
cv2.imwrite(output_path, img)
print(f"Hasil deteksi disimpan ke {output_path}")
