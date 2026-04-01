# Install dulu jika belum ada
# pip install ultralytics opencv-python matplotlib

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load model YOLO (default atau custom)
# Ganti dengan model hasil trainingmu jika ada, contoh: "runs/detect/train/weights/best.pt"
model = YOLO("best.pt")

# Baca gambar input
image_path = "your_image.jpg"   # ganti dengan path gambar yang mau diuji
img = cv2.imread(image_path)

# Deteksi objek
results = model(img)

# Ambil hasil deteksi dengan bounding box
annotated_img = results[0].plot()

# Tampilkan output dengan matplotlib (biar warnanya benar, karena cv2 pakai BGR)
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Simpan hasil ke file
cv2.imwrite("output_mobil.jpg", annotated_img)
print("Hasil deteksi disimpan sebagai output_mobil.jpg")
