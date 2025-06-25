"""
Motion Detection Sederhana
Deteksi gerakan menggunakan Background Subtraction
Dibuat untuk pembelajaran pemula
Rekayasa Fitur - Computer Vision
"""

import cv2
import numpy as np
import time

class SimpleMotionDetector:
    def __init__(self):
        """Inisialisasi motion detector sederhana"""
        # Background subtractor untuk mendeteksi objek bergerak
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_threshold = 1000  # Area minimum untuk dianggap motion
        
    def detect_motion(self, camera_id=0):
        """Deteksi motion dari webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Kesalahan: Tidak bisa buka kamera {camera_id}")
            print("💡 Tips:")
            print("   - Coba pilih kamera lain (1, 2, atau 3)")
            print("   - Pastikan tidak ada aplikasi lain yang menggunakan kamera")
            print("   - Restart aplikasi dan coba lagi")
            return
            
        print("Motion Detection dimulai!")
        print("Tekan 'q' untuk keluar")
        print("Tunggu beberapa detik untuk kalibrasi background...")
        
        frame_count = 0
        motion_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize frame agar tidak terlalu besar
            frame = cv2.resize(frame, (640, 480))
            
            # Terapkan background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Hilangkan noise dengan morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Cari kontur objek bergerak
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            total_area = 0
            
            # Gambar kotak di sekitar objek bergerak
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.motion_threshold:
                    motion_detected = True
                    motion_count += 1
                    total_area += area
                    
                    # Gambar bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'MOTION', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Tampilkan informasi
            status_text = "GERAKAN TERDETEKSI!" if motion_detected else "Tidak ada gerakan"
            color = (0, 0, 255) if motion_detected else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Hitung persentase motion
            motion_percent = (motion_count / frame_count * 100) if frame_count > 0 else 0
            cv2.putText(frame, f'Motion: {motion_percent:.1f}%', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.putText(frame, f'Area: {int(total_area)}', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Tampilkan hasil
            cv2.imshow('Motion Detection - Tekan Q untuk keluar', frame)
            cv2.imshow('Motion Mask', fg_mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nStatistik:")
        print(f"Total frame: {frame_count}")
        print(f"Frame dengan motion: {motion_count}")
        print(f"Persentase motion: {motion_percent:.1f}%")

def main():
    print("=== Motion Detection Sederhana ===")
    detector = SimpleMotionDetector()
    
    # Pilihan kamera langsung
    print("\nPilih kamera:")
    print("1. Kamera laptop (ID 0)")
    print("2. Kamera eksternal 1 (ID 1)")
    print("3. Kamera eksternal 2 (ID 2)")
    
    try:
        choice = input("\nPilih kamera (1-3, default=1): ").strip()
        
        if choice == '2':
            camera_id = 1
            print("Menggunakan kamera eksternal 1 (ID: 1)")
        elif choice == '3':
            camera_id = 2
            print("Menggunakan kamera eksternal 2 (ID: 2)")
        else:
            camera_id = 0
            print("Menggunakan kamera laptop (ID: 0)")
            
    except:
        camera_id = 0
        print("Menggunakan kamera default (ID: 0)")
    
    detector.detect_motion(camera_id)

if __name__ == "__main__":
    main()
