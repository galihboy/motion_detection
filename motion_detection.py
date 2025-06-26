"""
Demo Deteksi Gerakan (Motion Detection)
====================================

Program ini menggunakan berbagai metode untuk mendeteksi gerakan dari webcam.
Dibuat untuk pembelajaran mahasiswa UNIKOM pada mata kuliah Rekayasa Fitur.

🛠️ Metode Deteksi Gerakan:
1. Background Subtraction - Membandingkan frame dengan model background
   * Cocok untuk kamera statis/tetap
   * Menggunakan algoritma MOG2 (Mixture of Gaussians)

2. Frame Difference - Membandingkan frame saat ini dengan frame sebelumnya
   * Sederhana dan cepat
   * Cocok untuk deteksi gerakan cepat

3. Optical Flow - Melacak pergerakan titik-titik tertentu
   * Mendetail dan dapat melacak arah gerakan
   * Menggunakan metode Lucas-Kanade (sparse optical flow)

4. Dense Optical Flow - Menghitung pergerakan setiap piksel
   * Visualisasi aliran gerakan seluruh frame
   * Menggunakan metode Farneback
   * Informasi arah dan magnitude gerakan

5. Motion History Image (MHI) - Membuat jejak temporal gerakan
   * Menampilkan riwayat gerakan dalam satu gambar
   * Piksel yang baru bergerak tampak lebih terang
   * Berguna untuk analisis pola gerakan

🎮 Kontrol Program:
- 'q' - Keluar dari program
- 'r' - Mulai/Stop recording video
- 's' - Ambil screenshot
- '1' - Beralih ke metode Background Subtraction
- '2' - Beralih ke metode Frame Difference
- '3' - Beralih ke metode Optical Flow
- '+'/'-' - Ubah sensitivity (threshold)

📊 Fitur-fitur:
- Deteksi gerakan real-time
- Multi-backend kamera (DirectShow, MSMF, Default)
- Tampilan statistik (FPS, persentase gerakan)
- Recording video otomatis
- Screenshot
- Visualisasi mask deteksi gerakan

Dibuat untuk: Mata Kuliah Rekayasa Fitur - Computer Vision
Program Studi Teknik Informatika - UNIKOM
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime

class MotionDetector:
    def __init__(self):
        """
        Inisialisasi motion detector dengan berbagai metode
        
        Atribut yang diinisialisasi:
        - background_subtractor: Model MOG2 untuk metode background subtraction
        - previous_frame: Menyimpan frame sebelumnya untuk metode perbedaan frame
        - motion_threshold: Ambang batas minimal area gerakan (1000 pixel default)
        - recording: Status apakah sedang merekam
        - video_writer: Objek VideoWriter untuk recording
        - track_points: Points untuk tracking pada metode optical flow
        """
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,  # Deteksi bayangan
            varThreshold=50,     # Sensitivitas deteksi
            history=500          # Jumlah frame untuk model
        )
        self.previous_frame = None
        self.motion_threshold = 1000  # Ambang batas area motion
        self.recording = False
        self.video_writer = None
        self.track_points = None  # Untuk optical flow tracking
        
    def detect_available_cameras(self):
        """
        Deteksi kamera yang tersedia dengan berbagai backend
        
        Mencoba beberapa ID kamera (0-4) dengan 3 backend:
        - DirectShow (CAP_DSHOW): Backend Windows yang umum
        - Microsoft Media Foundation (CAP_MSMF): Backend Windows modern
        - Default (CAP_ANY): Backend default OpenCV
        
        Hanya mengembalikan kamera yang berhasil dibuka DAN dapat membaca frame
        
        Returns:
            list: Daftar kamera yang tersedia berisi dict dengan 'id', 'backend',
                 'backend_name', dan 'resolution'
        """
        available_cameras = []
        print("🔍 Testing kamera dengan berbagai backend...")
        
        # Backend yang akan dicoba (urutan prioritas)
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Microsoft Media Foundation"),
            (cv2.CAP_ANY, "Default")
        ]
        
        for i in range(5):
            print(f"\n📹 Testing kamera {i}:")
            camera_working = False
            
            for backend_id, backend_name in backends:
                print(f"  {backend_name}...", end=" ")
                try:
                    cap = cv2.VideoCapture(i, backend_id)
                    if cap.isOpened():
                        # Test baca frame untuk memastikan kamera benar-benar berfungsi
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            print(f"✅ OK ({width}x{height})")
                            if not camera_working:
                                camera_working = True
                                # Simpan kamera dengan backend terbaik
                                available_cameras.append({
                                    'id': i, 
                                    'backend': backend_id, 
                                    'backend_name': backend_name,
                                    'resolution': f"{width}x{height}"
                                })
                            cap.release()
                            break  # Gunakan backend pertama yang berhasil
                        else:
                            print("❌ Tidak bisa membaca frame")
                    else:
                        print("❌ Tidak bisa dibuka")
                    cap.release()
                except Exception as e:
                    print(f"❌ Kesalahan: {str(e)}")
            
            if not camera_working:
                print(f"  ❌ Tidak bisa digunakan")
                
        return available_cameras
    
    def preprocess_frame(self, frame):
        """Preprocessing frame untuk motion detection"""
        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian blur untuk mengurangi noise
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        return blur
    
    def method_background_subtraction(self, frame):
        """
        Metode 1: Background Subtraction
        
        Menggunakan MOG2 (Mixture of Gaussians) untuk memisahkan background dan foreground.
        Setiap pixel diklasifikasikan sebagai background atau foreground (objek bergerak).
        Cocok untuk kamera statis dengan background yang relatif tetap.
        
        Args:
            frame (numpy.ndarray): Frame yang akan dianalisis
            
        Returns:
            tuple: (frame dengan anotasi, mask gerakan, status gerakan, total area gerakan)
        """
        # Terapkan background subtractor
        fg_mask = self.background_subtractor.apply(frame)
        
        # Morfologi untuk menghilangkan noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Temukan kontur
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Gambar bounding box untuk objek bergerak
        motion_detected = False
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:  # Filter kontur kecil
                motion_detected = True
                total_area += area
                
                # Gambar bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Area: {int(area)}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, fg_mask, motion_detected, total_area
    
    def method_frame_difference(self, frame):
        """
        Metode 2: Frame Difference
        
        Membandingkan frame saat ini dengan frame sebelumnya untuk menemukan perubahan.
        Metode lebih sederhana dan responsive terhadap pergerakan cepat.
        Dapat bekerja dengan kamera bergerak, tetapi sensitif terhadap noise.
        
        Args:
            frame (numpy.ndarray): Frame yang akan dianalisis
            
        Returns:
            tuple: (frame dengan anotasi, mask perbedaan, status gerakan, total area gerakan)
        """
        processed_frame = self.preprocess_frame(frame)
        
        if self.previous_frame is None:
            self.previous_frame = processed_frame
            return frame, np.zeros_like(processed_frame), False, 0
        
        # Hitung perbedaan antara frame sekarang dan sebelumnya
        frame_diff = cv2.absdiff(self.previous_frame, processed_frame)
        
        # Threshold untuk mendapatkan binary image
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Dilasi untuk mengisi celah
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Temukan kontur
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.motion_threshold:
                motion_detected = True
                total_area += area
                
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Motion: {int(area)}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Update previous frame
        self.previous_frame = processed_frame
        
        return frame, thresh, motion_detected, total_area
    
    def method_optical_flow(self, frame):
        """
        Metode 3: Optical Flow (Lucas-Kanade)
        
        Melacak pergerakan titik-titik fitur (feature points) antar frame.
        Dapat mendeteksi arah dan magnitude pergerakan dengan presisi tinggi.
        Cocok untuk pelacakan objek dan analisis gerakan yang lebih detail.
        
        Algoritma:
        1. Deteksi good feature points (corner detection)
        2. Track points tersebut ke frame berikutnya
        3. Hitung vektor pergerakan untuk setiap point
        
        Args:
            frame (numpy.ndarray): Frame yang akan dianalisis
            
        Returns:
            tuple: (frame dengan anotasi, visualisasi optical flow, 
                   status gerakan, total magnitude gerakan)
        """
        processed_frame = self.preprocess_frame(frame)
        
        # Inisialisasi default values
        motion_detected = False
        total_motion = 0
        
        if self.previous_frame is None:
            # Setup untuk optical flow
            # Deteksi corner points untuk tracking
            self.track_points = cv2.goodFeaturesToTrack(
                processed_frame, maxCorners=100, qualityLevel=0.3, 
                minDistance=7, blockSize=7
            )
            self.previous_frame = processed_frame
            return frame, np.zeros_like(processed_frame), motion_detected, total_motion
        
        if self.track_points is not None and len(self.track_points) > 0:
            # Hitung optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.previous_frame, processed_frame, self.track_points, None
            )
            
            # Pilih points yang bagus
            good_new = new_points[status == 1]
            good_old = self.track_points[status == 1]
            
            # Gambar tracking points dan garis pergerakan
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                
                # Hitung magnitude pergerakan
                motion_magnitude = np.sqrt((a-c)**2 + (b-d)**2)
                total_motion += motion_magnitude
                
                if motion_magnitude > 5:  # Threshold pergerakan
                    motion_detected = True
                    # Gambar garis pergerakan
                    cv2.line(frame, (a, b), (c, d), (0, 255, 255), 2)
                    cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
            
            # Update points untuk frame berikutnya
            self.track_points = good_new.reshape(-1, 1, 2)
            
            # Jika points terlalu sedikit, deteksi ulang
            if len(self.track_points) < 10:
                self.track_points = cv2.goodFeaturesToTrack(
                    processed_frame, maxCorners=100, qualityLevel=0.3, 
                    minDistance=7, blockSize=7
                )
        
        self.previous_frame = processed_frame
        
        flow_visualization = np.zeros_like(frame)
        if self.track_points is not None:
            for point in self.track_points:
                x, y = point.ravel().astype(int)
                cv2.circle(flow_visualization, (x, y), 3, (0, 255, 0), -1)
        
        return frame, flow_visualization, motion_detected, total_motion
    
    def method_dense_optical_flow(self, frame):
        """
        Metode 4: Dense Optical Flow (Farneback)
        
        Menghitung optical flow untuk setiap pixel dalam frame, tidak hanya titik tertentu.
        Memberikan informasi pergerakan yang lebih menyeluruh dibandingkan sparse optical flow.
        Komputasi lebih berat tetapi hasil lebih detail.
        
        Algoritma:
        1. Konversi frame ke grayscale
        2. Hitung vektor pergerakan setiap piksel dengan algoritma Farneback
        3. Konversi vektor ke representasi warna HSV (Hue=arah, Saturation=1, Value=magnitude)
        
        Args:
            frame (numpy.ndarray): Frame yang akan dianalisis
            
        Returns:
            tuple: (frame dengan anotasi, visualisasi dense flow, 
                   status gerakan, total magnitude gerakan)
        """
        processed_frame = self.preprocess_frame(frame)
        
        # Inisialisasi default values
        motion_detected = False
        total_motion = 0
        
        # Untuk visualisasi
        flow_visualization = np.zeros_like(frame)
        
        if self.previous_frame is None:
            self.previous_frame = processed_frame
            return frame, flow_visualization, motion_detected, total_motion
        
        # Hitung dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_frame, processed_frame,
            None,                   # Flow yang dihitung sebelumnya (None untuk inisialisasi)
            0.5,                    # Pyramid scale
            3,                      # Levels
            15,                     # Window size
            3,                      # Iterasi
            5,                      # Poly_n
            1.2,                    # Poly_sigma
            0                       # Flags
        )
        
        # Konversi flow ke magnitude dan angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Hitung total magnitude untuk deteksi gerakan
        mean_magnitude = np.mean(magnitude)
        total_motion = np.sum(magnitude)
        
        # Deteksi gerakan berdasarkan threshold magnitude
        if mean_magnitude > 1.0:  # Threshold bisa disesuaikan
            motion_detected = True
        
        # Buat visualisasi flow dengan representasi warna HSV
        # Hue berdasarkan angle, Value berdasarkan magnitude
        flow_hsv = np.zeros((processed_frame.shape[0], processed_frame.shape[1], 3), dtype=np.uint8)
        flow_hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (angle 0-360)
        flow_hsv[..., 1] = 255                      # Saturation (max)
        flow_hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) # Value = magnitude
        
        # Konversi ke BGR untuk visualisasi
        flow_visualization = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2BGR)
        
        # Tambahkan teks status
        if motion_detected:
            cv2.putText(frame, f"Flow Avg: {mean_magnitude:.2f}", (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update previous frame
        self.previous_frame = processed_frame
        
        return frame, flow_visualization, motion_detected, total_motion
    
    def method_motion_history_image(self, frame):
        """
        Metode 5: Motion History Image (MHI)
        
        Membuat representasi temporal dari gerakan dengan menyimpan "jejak" gerakan
        dari beberapa frame terakhir. Piksel yang baru bergerak memiliki intensitas
        lebih tinggi daripada piksel yang bergerak sebelumnya.
        
        Algoritma:
        1. Hitung perbedaan antar frame menggunakan threshold
        2. Update Motion History Image: tingkatkan nilai di area bergerak, turunkan di area lain
        3. Normalisasi MHI untuk visualisasi
        
        Implementasi ini tidak menggunakan cv2.motempl karena tidak tersedia di semua
        versi OpenCV. Sebagai gantinya, kita mengimplementasikan MHI secara manual dengan
        parameter decay_rate yang dapat disesuaikan menggunakan tombol 'a' (kurang) dan 'd' (tambah).
        
        Args:
            frame (numpy.ndarray): Frame yang akan dianalisis
            
        Returns:
            tuple: (frame dengan anotasi, MHI, status gerakan, jumlah gerakan)
        """
        processed_frame = self.preprocess_frame(frame)
        
        # Inisialisasi MHI jika belum ada
        if not hasattr(self, 'motion_history'):
            h, w = processed_frame.shape
            self.motion_history = np.zeros((h, w), dtype=np.float32)
            self.mhi_duration = 30  # Durasi history dalam frame
            self.timestamp = 0
            self.decay_rate = 1.0  # Rate pengurangan nilai MHI per frame
            
        if self.previous_frame is None:
            self.previous_frame = processed_frame
            return frame, np.zeros_like(frame), False, 0
        
        # Hitung perbedaan frame
        frame_diff = cv2.absdiff(self.previous_frame, processed_frame)
        _, motion_mask = cv2.threshold(frame_diff, 30, 1, cv2.THRESH_BINARY)
        
        # Update timestamp
        self.timestamp += 1
        
        # Update MHI secara manual tanpa menggunakan cv2.motempl:
        # 1. Di area yang terdapat gerakan, isi dengan nilai timestamp saat ini
        mask_idx = (motion_mask > 0)
        self.motion_history[mask_idx] = self.timestamp
        
        # 2. Kurangi nilai MHI sesuai dengan decay rate di area tanpa gerakan
        # dan pastikan tidak ada nilai negatif
        no_motion_idx = ~mask_idx
        self.motion_history[no_motion_idx] = np.maximum(0, self.motion_history[no_motion_idx] - self.decay_rate)
        
        # Normalisasi MHI untuk visualisasi (0-255)
        mhi_vis = np.clip(
            self.motion_history * (255.0 / self.mhi_duration), 
            0, 255
        ).astype(np.uint8)
        
        # Buat visualisasi berwarna
        mhi_color = cv2.applyColorMap(mhi_vis, cv2.COLORMAP_JET)
        
        # Deteksi gerakan
        motion_detected = False
        total_motion = np.sum(motion_mask)  # Jumlah piksel bergerak
        
        if total_motion > self.motion_threshold / 10:  # Sesuaikan threshold
            motion_detected = True
            
            # Temukan kontur gerakan dari MHI
            mhi_binary = cv2.threshold(mhi_vis, 50, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(mhi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Gambar kontur gerakan pada frame
            cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)
            
            # Tambahkan teks info
            cv2.putText(frame, f"MHI Motion: {total_motion:.0f}", (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Update previous frame
        self.previous_frame = processed_frame
        
        return frame, mhi_color, motion_detected, total_motion
    
    def start_recording(self, frame_width, frame_height, fps=20.0):
        """Mulai recording video"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"motion_detected_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            self.recording = True
            print(f"🔴 Mulai recording: {filename}")
            return filename
        return None
    
    def stop_recording(self):
        """Hentikan recording video"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("⏹️ Recording dihentikan")
    
    def reset_detection_state(self):
        """
        Reset state untuk semua metode detection
        
        Fungsi ini berguna saat beralih antar metode deteksi untuk memastikan
        tidak ada state dari metode sebelumnya yang memengaruhi metode baru.
        Direset:
        - previous_frame: Frame sebelumnya
        - track_points: Points untuk optical flow
        - background_subtractor: Model background
        - motion_history: Motion History Image
        """
        self.previous_frame = None
        self.track_points = None
        # Reset background subtractor jika perlu fresh start
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        # Reset motion history
        if hasattr(self, 'motion_history'):
            h, w = self.motion_history.shape
            self.motion_history = np.zeros((h, w), dtype=np.float32)
            self.timestamp = 0
            self.decay_rate = 1.0  # Reset decay rate ke nilai default
    
    def detect_motion_webcam(self, camera_info, method='background', window_width=800, window_height=600):
        """Main function untuk deteksi motion dari webcam"""
        # Extract camera info
        if isinstance(camera_info, dict):
            camera_id = camera_info['id']
            backend = camera_info['backend']
            backend_name = camera_info['backend_name']
            print(f"🎥 Menggunakan kamera {camera_id} dengan backend {backend_name}")
            cap = cv2.VideoCapture(camera_id, backend)
        else:
            # Fallback untuk backward compatibility
            camera_id = camera_info
            print(f"🎥 Menggunakan kamera {camera_id} dengan backend default")
            cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Tidak dapat mengakses kamera {camera_id}")
            print("💡 Tips troubleshooting:")
            print("   - Pastikan tidak ada aplikasi lain yang menggunakan kamera")
            print("   - Coba restart aplikasi atau komputer")
            print("   - Periksa permission kamera di Windows Settings")
            return
        
        # Setup kamera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        
        print(f"🎥 Motion Detection dimulai - Method: {method}")
        print("Kontrol:")
        print("  'q' - Keluar")
        print("  'r' - Mulai/Stop recording")
        print("  's' - Screenshot")
        print("  '1' - Background Subtraction")
        print("  '2' - Frame Difference") 
        print("  '3' - Optical Flow")
        print("  '4' - Dense Optical Flow")
        print("  '5' - Motion History Image (MHI)")
        print("  '+'/'-' - Ubah sensitivity threshold")
        print("  'a'/'d' - Kurangi/tambah MHI decay rate (hanya mode MHI)")
        
        cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Motion Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Motion Detection", window_width, window_height)
        cv2.resizeWindow("Motion Mask", window_width//2, window_height//2)
        
        # Variabel untuk statistik
        frame_count = 0
        motion_count = 0
        start_time = time.time()
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            original_frame = frame.copy()
            
            # Pilih metode detection
            if method == 'background':
                result_frame, mask, motion_detected, motion_value = self.method_background_subtraction(frame)
            elif method == 'difference':
                result_frame, mask, motion_detected, motion_value = self.method_frame_difference(frame)
            elif method == 'optical':
                result_frame, mask, motion_detected, motion_value = self.method_optical_flow(frame)
            elif method == 'dense_flow':
                result_frame, mask, motion_detected, motion_value = self.method_dense_optical_flow(frame)
            elif method == 'mhi':
                result_frame, mask, motion_detected, motion_value = self.method_motion_history_image(frame)
            else:
                result_frame, mask, motion_detected, motion_value = self.method_background_subtraction(frame)
            
            if motion_detected:
                motion_count += 1
            
            # Resize untuk tampilan
            display_frame = cv2.resize(result_frame, (window_width, window_height))
            mask_resized = cv2.resize(mask, (window_width//2, window_height//2))
            
            # Tambahkan informasi di layar
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            motion_percentage = (motion_count / frame_count * 100) if frame_count > 0 else 0
            
            # Info text
            cv2.putText(display_frame, f'Method: {method.title()}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f'FPS: {current_fps:.1f}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Motion: {motion_percentage:.1f}%', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f'Value: {motion_value:.0f}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(display_frame, f'Threshold: {self.motion_threshold}', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Status motion
            status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
            status_text = "GERAKAN TERDETEKSI!" if motion_detected else "Tidak Ada Gerakan"
            cv2.putText(display_frame, status_text, (window_width-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Recording indicator
            if self.recording:
                cv2.circle(display_frame, (window_width-30, 60), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC", (window_width-60, 67), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Tampilkan hasil
            cv2.imshow("Motion Detection", display_frame)
            cv2.imshow("Motion Mask", mask_resized)
            
            # Recording
            if self.recording and self.video_writer:
                self.video_writer.write(result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording(result_frame.shape[1], result_frame.shape[0], fps)
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"motion_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"📸 Screenshot disimpan: {filename}")
            elif key == ord("1"):
                method = 'background'
                self.reset_detection_state()
                print("🔄 Beralih ke Background Subtraction")
            elif key == ord("2"):
                method = 'difference' 
                self.reset_detection_state()
                print("🔄 Beralih ke Frame Difference")
            elif key == ord("3"):
                method = 'optical'
                self.reset_detection_state()
                print("🔄 Beralih ke Optical Flow")
            elif key == ord("4"):
                method = 'dense_flow'
                self.reset_detection_state()
                print("🔄 Beralih ke Dense Optical Flow")
            elif key == ord("5"):
                method = 'mhi'
                self.reset_detection_state()
                print("🔄 Beralih ke Motion History Image")
            elif key == ord("+") or key == ord("="):
                self.motion_threshold += 500
                print(f"📈 Threshold: {self.motion_threshold}")
            elif key == ord("-"):
                self.motion_threshold = max(100, self.motion_threshold - 500)
                print(f"📉 Threshold: {self.motion_threshold}")
            elif key == ord("d") and method == 'mhi':
                # Ubah decay rate untuk MHI
                if hasattr(self, 'decay_rate'):
                    self.decay_rate = min(5.0, self.decay_rate + 0.5)
                    print(f"📈 MHI Decay Rate: {self.decay_rate}")
            elif key == ord("a") and method == 'mhi':
                # Ubah decay rate untuk MHI
                if hasattr(self, 'decay_rate'):
                    self.decay_rate = max(0.1, self.decay_rate - 0.5)
                    print(f"📉 MHI Decay Rate: {self.decay_rate}")
        
        # Cleanup
        if self.recording:
            self.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
          # Statistik akhir
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        motion_percentage = (motion_count / frame_count * 100) if frame_count > 0 else 0
        
        print(f"\n📊 Statistik Motion Detection:")
        print(f"Total frames: {frame_count}")
        print(f"Gerakan terdeteksi: {motion_count} frames ({motion_percentage:.1f}%)")
        print(f"Rata-rata FPS: {current_fps:.1f}")

def main():
    """
    Fungsi utama program Motion Detection
    
    Alur program:
    1. Pengguna memilih metode deteksi (Background/Difference/Optical)
    2. Program mendeteksi kamera yang tersedia dengan multi-backend
    3. Pengguna memilih kamera yang akan digunakan
    4. Pengguna memilih ukuran window
    5. Motion detection dijalankan dengan parameter yang dipilih
    
    Selama program berjalan, pengguna dapat:
    - Beralih antar metode dengan tombol 1/2/3
    - Ambil screenshot dengan tombol 's'
    - Rekam video dengan tombol 'r'
    - Atur sensitivitas dengan tombol '+'/'-'
    """
    detector = MotionDetector()
    
    print("\n=== Demo Motion Detection ===")
    print("Pilih metode deteksi gerakan:")
    print("1. Background Subtraction (Rekomendasi untuk kamera statis)")
    print("2. Frame Difference (Cocok untuk gerakan cepat)")
    print("3. Optical Flow (Tracking pergerakan detail)")
    print("4. Dense Optical Flow (Visualisasi aliran gerakan)")
    print("5. Motion History Image (Jejak temporal gerakan)")
    
    method_choice = input("\nPilih metode (1-5, default=1): ").strip()
    
    methods = {'1': 'background', '2': 'difference', '3': 'optical', 
               '4': 'dense_flow', '5': 'mhi'}
    selected_method = methods.get(method_choice, 'background')
    
    # Deteksi kamera
    print("\nMendeteksi kamera yang tersedia...")
    available_cameras = detector.detect_available_cameras()
    
    if not available_cameras:
        print("❌ Tidak ada kamera yang terdeteksi!")
        return
    print(f"📷 Kamera terdeteksi: {len(available_cameras)}")
    
    # Tampilkan detail kamera
    for i, cam_info in enumerate(available_cameras):
        cam_id = cam_info['id']
        backend_name = cam_info['backend_name']
        resolution = cam_info['resolution']
        cam_desc = "Kamera bawaan laptop" if cam_id == 0 else f"Kamera eksternal {cam_id}"
        print(f"  {i}. {cam_desc} - {backend_name} ({resolution})")
    
    # Pilih kamera
    if len(available_cameras) == 1:
        selected_camera = available_cameras[0]
        print(f"\nMenggunakan: Kamera {selected_camera['id']} ({selected_camera['backend_name']})")
    else:
        print(f"\nPilih kamera (0-{len(available_cameras)-1}):")
        
        try:
            choice = int(input(f"Pilih (default=0): ").strip() or "0")
            if 0 <= choice < len(available_cameras):
                selected_camera = available_cameras[choice]
            else:
                print("⚠️ Pilihan tidak valid, menggunakan kamera pertama")
                selected_camera = available_cameras[0]
        except ValueError:
            selected_camera = available_cameras[0]
    
    # Pilihan ukuran window
    print("\n=== Pilih Ukuran Window ===")
    print("1. Kecil (640x480)")
    print("2. Sedang (800x600) - Default") 
    print("3. Besar (1024x768)")
    print("4. Full HD (1280x720)")
    
    size_choice = input("\nPilih ukuran (1-4, default=2): ").strip()
    
    sizes = {
        '1': (640, 480),
        '2': (800, 600), 
        '3': (1024, 768),
        '4': (1280, 720)
    }
    window_width, window_height = sizes.get(size_choice, (800, 600))
    print(f"\n🚀 Memulai Motion Detection...")
    print(f"Method: {selected_method}")
    print(f"Camera: {selected_camera['id']} ({selected_camera['backend_name']})")
    print(f"Window: {window_width}x{window_height}")
    
    # Mulai deteksi
    detector.detect_motion_webcam(selected_camera, selected_method, window_width, window_height)

if __name__ == "__main__":
    try:
        print("""
╔═══════════════════════════════════════╗
║    MOTION DETECTION - UNIKOM          ║
║    Computer Vision - Rekayasa Fitur   ║
║                                       ║
║    Cara penggunaan:                   ║
║    - Pilih metode deteksi             ║
║    - Pilih kamera yang tersedia       ║
║    - Pilih ukuran window              ║
║                                       ║
║    Kontrol saat berjalan:             ║
║    q - Keluar                         ║
║    r - Mulai/Stop recording           ║
║    s - Screenshot                     ║
║    1/2/3/4/5 - Ganti metode deteksi   ║
║    +/- - Atur sensitivitas threshold  ║
║    a/d - Atur MHI decay rate          ║
╚═══════════════════════════════════════╝
        """)
        main()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh pengguna")
    except Exception as e:
        print(f"\nKesalahan: {e}")
        print("Pastikan kamera tersedia dan tidak digunakan aplikasi lain")
