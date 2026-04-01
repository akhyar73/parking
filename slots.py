import cv2
import json
import os
import numpy as np
from datetime import datetime

class ParkingSlotAnnotatorPolygon:
    def __init__(self, video_path, output_json="parking_slots_polygon.json"):
        self.video_path = video_path
        self.output_json = output_json
        self.slots = []
        self.current_polygon = []
        self.frame = None
        self.display_frame = None
        self.video_width = 0
        self.video_height = 0
        self.mode = "polygon"  # polygon or rectangle
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events untuk menggambar polygon"""
        if self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                # Tambah titik polygon
                self.current_polygon.append([x, y])
                print(f"📍 Titik {len(self.current_polygon)}: ({x}, {y})")
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Selesai polygon (klik kanan)
                if len(self.current_polygon) >= 3:
                    slot_id = len(self.slots) + 1
                    slot = {
                        "id": slot_id,
                        "type": "polygon",
                        "points": self.current_polygon.copy()
                    }
                    self.slots.append(slot)
                    print(f"✅ Slot #{slot_id} ditambahkan (polygon dengan {len(self.current_polygon)} titik)")
                    self.current_polygon = []
                else:
                    print("⚠️  Minimal 3 titik untuk polygon!")
                    
        elif self.mode == "rectangle":
            # Mode rectangle (4 titik)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_polygon.append([x, y])
                print(f"📍 Titik {len(self.current_polygon)}: ({x}, {y})")
                
                # Jika sudah 4 titik, otomatis selesai
                if len(self.current_polygon) == 4:
                    slot_id = len(self.slots) + 1
                    slot = {
                        "id": slot_id,
                        "type": "rectangle",
                        "points": self.current_polygon.copy()
                    }
                    self.slots.append(slot)
                    print(f"✅ Slot #{slot_id} ditambahkan (rectangle 4 titik)")
                    self.current_polygon = []
    
    def draw_slots(self, frame):
        """Gambar semua slot yang sudah dibuat"""
        display = frame.copy()
        
        # Gambar slot yang sudah ada (hijau)
        for slot in self.slots:
            points = np.array(slot["points"], dtype=np.int32)
            
            # Gambar polygon
            cv2.polylines(display, [points], True, (0, 255, 0), 2)
            
            # Fill semi-transparan
            overlay = display.copy()
            cv2.fillPoly(overlay, [points], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
            
            # Gambar titik-titik
            for i, point in enumerate(points):
                cv2.circle(display, tuple(point), 5, (0, 255, 0), -1)
            
            # Tampilkan ID slot di tengah
            M = cv2.moments(points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display, f"#{slot['id']}", (cx - 20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Gambar polygon yang sedang dibuat (biru)
        if len(self.current_polygon) > 0:
            points = np.array(self.current_polygon, dtype=np.int32)
            
            # Gambar garis antar titik
            if len(points) > 1:
                cv2.polylines(display, [points], False, (255, 0, 0), 2)
            
            # Gambar titik-titik
            for i, point in enumerate(points):
                cv2.circle(display, tuple(point), 5, (255, 0, 0), -1)
                # Nomor titik
                cv2.putText(display, str(i+1), (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Tampilkan info di pojok kiri atas
        mode_text = "POLYGON (Bebas)" if self.mode == "polygon" else "RECTANGLE (4 Titik)"
        info_text = [
            f"Mode: {mode_text}",
            f"Total Slots: {len(self.slots)}",
            f"Titik saat ini: {len(self.current_polygon)}",
            f"Resolusi: {self.video_width}x{self.video_height}",
            "",
            "KONTROL:",
            "Klik Kiri = Tambah titik",
            "Klik Kanan = Selesai polygon" if self.mode == "polygon" else "4 titik = Auto selesai",
            "M = Ganti mode",
            "U = Undo slot terakhir",
            "ESC = Reset polygon",
            "C = Clear semua slot",
            "S = Save JSON",
            "Q = Keluar"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            color = (255, 255, 255)
            if "Mode:" in text:
                color = (0, 255, 255)
            elif "KONTROL:" in text:
                color = (0, 255, 255)
            elif text == "":
                continue
                
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y_offset += 25
        
        return display
    
    def delete_last_slot(self):
        """Hapus slot terakhir (Undo)"""
        if self.slots:
            deleted = self.slots.pop()
            print(f"🗑️  Slot #{deleted['id']} dihapus")
        else:
            print("⚠️  Tidak ada slot untuk dihapus")
    
    def clear_all_slots(self):
        """Hapus semua slot"""
        if self.slots:
            count = len(self.slots)
            self.slots = []
            print(f"🗑️  {count} slot dihapus")
        else:
            print("⚠️  Tidak ada slot untuk dihapus")
    
    def reset_current_polygon(self):
        """Reset polygon yang sedang digambar"""
        if self.current_polygon:
            print(f"🔄 Reset {len(self.current_polygon)} titik")
            self.current_polygon = []
        else:
            print("⚠️  Tidak ada polygon untuk direset")
    
    def toggle_mode(self):
        """Toggle antara polygon dan rectangle mode"""
        self.current_polygon = []
        if self.mode == "polygon":
            self.mode = "rectangle"
            print("🔄 Mode: RECTANGLE (4 titik)")
        else:
            self.mode = "polygon"
            print("🔄 Mode: POLYGON (bebas)")
    
    def save_json(self):
        """Save koordinat slot ke JSON file"""
        data = {
            "video_metadata": {
                "video_path": self.video_path,
                "width": self.video_width,
                "height": self.video_height
            },
            "slots": self.slots,
            "total_slots": len(self.slots),
            "created_at": datetime.now().isoformat()
        }
        
        with open(self.output_json, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Data disimpan ke: {self.output_json}")
        print(f"📊 Total slot: {len(self.slots)}")
        
        # Tampilkan statistik per tipe
        polygon_count = sum(1 for s in self.slots if s.get("type") == "polygon")
        rect_count = sum(1 for s in self.slots if s.get("type") == "rectangle")
        print(f"   - Polygon: {polygon_count}")
        print(f"   - Rectangle: {rect_count}")
    
    def run(self):
        """Jalankan annotator"""
        # Buka video
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Tidak bisa membuka video: {self.video_path}")
            return
        
        # Ambil info video
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print("\n" + "="*70)
        print("🅿️  PARKING SLOT ANNOTATOR - POLYGON VERSION")
        print("="*70)
        print(f"📹 Video: {os.path.basename(self.video_path)}")
        print(f"📐 Resolusi: {self.video_width}x{self.video_height}")
        print(f"🎬 FPS: {fps}")
        print("="*70)
        print("\n📝 INSTRUKSI:")
        print("\n🔷 MODE POLYGON (Default):")
        print("   1. Klik kiri untuk menambah titik")
        print("   2. Klik kanan untuk selesai (minimal 3 titik)")
        print("   3. Ulangi untuk slot berikutnya")
        print("\n🔶 MODE RECTANGLE:")
        print("   1. Tekan M untuk ganti ke mode rectangle")
        print("   2. Klik 4 titik sudut slot parkir (otomatis selesai)")
        print("   3. Titik 1-2-3-4 akan membentuk rectangle")
        print("\n⌨️  KEYBOARD:")
        print("   M = Ganti mode (Polygon ↔ Rectangle)")
        print("   U = Undo (hapus slot terakhir)")
        print("   ESC = Reset polygon yang sedang digambar")
        print("   C = Clear semua slot")
        print("   S = Save ke JSON")
        print("   Q = Keluar")
        print("="*70 + "\n")
        
        # Ambil frame pertama
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Error: Tidak bisa membaca frame dari video")
            return
        
        # Setup window dan mouse callback
        window_name = "Parking Slot Annotator (POLYGON) - Klik untuk menandai"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"🎯 Mode: {self.mode.upper()}")
        print("💡 Mulai menandai slot parkir...\n")
        
        # Main loop
        while True:
            # Update display
            self.display_frame = self.draw_slots(self.frame)
            cv2.imshow(window_name, self.display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Q - Keluar
                print("\n👋 Keluar dari annotator...")
                break
                
            elif key == ord('u'):  # U - Undo
                self.delete_last_slot()
                
            elif key == ord('c'):  # C - Clear all
                self.clear_all_slots()
                
            elif key == ord('s'):  # S - Save
                self.save_json()
                
            elif key == 27:  # ESC - Reset current polygon
                self.reset_current_polygon()
                
            elif key == ord('m'):  # M - Toggle mode
                self.toggle_mode()
        
        cv2.destroyAllWindows()
        
        # Tanya apakah mau save sebelum keluar
        if self.slots:
            print(f"\n💡 Anda memiliki {len(self.slots)} slot yang belum disimpan")
            response = input("Save sekarang? (y/n): ").lower()
            if response == 'y':
                self.save_json()
        
        print("\n✅ Selesai!\n")


# ==================== CARA MENGGUNAKAN ====================

if __name__ == "__main__":
    # Ganti dengan path video Anda
    VIDEO_PATH = "F:\parkiran/easy1.mp4"
    OUTPUT_JSON = "parking_slots_polygon.json"
    
    # Cek apakah file video ada
    if not os.path.exists(VIDEO_PATH):
        print("\n❌ ERROR: File video tidak ditemukan!")
        print(f"Path: {VIDEO_PATH}")
        print("\n📝 Cara menggunakan:")
        print("1. Edit bagian VIDEO_PATH dengan path video Anda")
        print("   Contoh Windows: VIDEO_PATH = r'F:\\parkiran\\video_parkir.mp4'")
        print("   Contoh Linux: VIDEO_PATH = '/home/user/parking.mp4'")
        print("2. Jalankan script lagi\n")
    else:
        # Jalankan annotator
        annotator = ParkingSlotAnnotatorPolygon(VIDEO_PATH, OUTPUT_JSON)
        annotator.run()