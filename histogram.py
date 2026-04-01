import cv2
import matplotlib.pyplot as plt

# load frame (satu gambar dulu)
img = cv2.imread("keluar1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# hitung histogram
hist = cv2.calcHist([gray],[0],None,[256],[0,256])

# plot
plt.plot(hist)
plt.title("Histogram Intensitas - Kondisi Terik")
plt.xlabel("Nilai Intensitas")
plt.ylabel("Jumlah Piksel")
plt.show()
