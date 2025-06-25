# Motion Detection - Deteksi Gerakan

Aplikasi deteksi gerakan menggunakan OpenCV dengan tiga metode berbeda untuk mendeteksi gerakan. Program ini cocok untuk pembelajaran Computer Vision dalam mata kuliah Rekayasa Fitur.

## 🚀 Fitur Utama

- **Multi-metode deteksi gerakan**:
  - Background Subtraction (MOG2)
  - Frame Difference
  - Optical Flow (Lucas-Kanade)

- **Fitur lengkap**:
  - Multi-backend kamera (DirectShow, MSMF, Default) 
  - Auto-deteksi kamera yang tersedia
  - Visualisasi mask deteksi gerakan
  - Tampilan real-time statistik (FPS, persentase gerakan)
  - Recording video otomatis saat deteksi
  - Screenshot
  - Pilihan resolusi window

## 📋 Persyaratan

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- Kamera terhubung (webcam laptop atau kamera eksternal)

## 🔧 Cara Penggunaan

### 1. Menjalankan Program

```bash
python motion_detection.py
```

### 2. Pilih Metode Deteksi

Program akan menampilkan menu untuk memilih metode deteksi:

```
=== Demo Motion Detection ===
Pilih metode deteksi gerakan:
1. Background Subtraction (Rekomendasi untuk kamera statis)
2. Frame Difference (Cocok untuk gerakan cepat)
3. Optical Flow (Tracking pergerakan detail)

Pilih metode (1-3, default=1): 
```

### 3. Auto-Deteksi Kamera

Program akan otomatis mendeteksi kamera yang tersedia dengan berbagai backend:

```
Mendeteksi kamera yang tersedia...
📷 Kamera terdeteksi: 2
  0. Kamera bawaan laptop - DirectShow (640x480)
  1. Kamera eksternal 1 - Microsoft Media Foundation (1280x720)

Pilih kamera (0-1):
```

### 4. Pilih Ukuran Window

```
=== Pilih Ukuran Window ===
1. Kecil (640x480)
2. Sedang (800x600) - Default
3. Besar (1024x768)
4. Full HD (1280x720)

Pilih ukuran (1-4, default=2):
```

### 5. Kontrol Saat Program Berjalan

- `q` - Keluar dari program
- `r` - Mulai/Stop recording video
- `s` - Ambil screenshot
- `1` - Beralih ke metode Background Subtraction
- `2` - Beralih ke metode Frame Difference
- `3` - Beralih ke metode Optical Flow
- `+/-` - Menambah/mengurangi sensitivity (threshold)

## 📊 Penjelasan Metode

### 1. Background Subtraction

Metode ini membandingkan setiap frame dengan model background yang dibuat dari frame-frame sebelumnya. Perubahan signifikan akan dianggap sebagai objek bergerak.

**Kelebihan**:
- Baik untuk kamera diam/statis
- Dapat mendeteksi objek bergerak kecil
- Kuat terhadap perubahan pencahayaan gradual

**Cocok untuk**: Keamanan, monitoring ruangan, deteksi intrusi

### 2. Frame Difference

Metode sederhana yang membandingkan frame saat ini dengan frame sebelumnya. Setiap perubahan akan terdeteksi sebagai gerakan.

**Kelebihan**:
- Sederhana dan ringan
- Responsif terhadap gerakan cepat
- Dapat bekerja dengan kamera bergerak

**Cocok untuk**: Deteksi gerakan cepat, analisis real-time

### 3. Optical Flow

Metode ini melacak pergerakan titik-titik tertentu dari satu frame ke frame berikutnya, menghitung vektor pergerakan.

**Kelebihan**:
- Detail pergerakan (arah dan kecepatan)
- Tracking objek bergerak
- Informasi lebih lengkap

**Cocok untuk**: Analisis gerakan detail, tracking objek

## 🔍 Tips Troubleshooting

- **Jika kamera tidak terdeteksi**:
  - Pastikan kamera tidak digunakan aplikasi lain
  - Coba restart aplikasi
  - Cek permission kamera di sistem operasi

- **Performance kurang optimal**:
  - Gunakan ukuran window lebih kecil
  - Optimalkan threshold dengan tombol `+/-`

- **Deteksi terlalu sensitif/tidak sensitif**:
  - Atur threshold dengan tombol `+/-`
  - Ganti metode yang lebih sesuai dengan kondisi

## 🎓 Konsep Teknis

- **Background Subtraction**: Menggunakan MOG2 (Mixture of Gaussians) untuk model background
- **Frame Difference**: Menggunakan absolute difference dan threshold
- **Optical Flow**: Menggunakan Lucas-Kanade optical flow untuk tracking points

## 📂 Output

Program akan menyimpan file dalam direktori saat ini:
- **Video recording**: `motion_detected_YYYYMMDD_HHMMSS.mp4`
- **Screenshots**: `motion_screenshot_YYYYMMDD_HHMMSS.jpg`

---

Dibuat untuk mata kuliah Rekayasa Fitur - Program Studi Teknik Informatika UNIKOM
