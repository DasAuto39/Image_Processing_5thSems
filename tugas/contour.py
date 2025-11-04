import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    exit()

# --- Spesifikasi Fitur 2: Definisikan Rentang Warna ---
# Rentang untuk warna HIJAU [cite: 75]
lower_green = np.array([35, 100, 50])
upper_green = np.array([85, 255, 255])

# --- Spesifikasi Fitur 3: Kernel Morfologi ---
# Buat kernel untuk operasi morfologi [cite: 77]
kernel = np.ones((5, 5), np.uint8)

print("Webcam terbuka. Menekan 'q' untuk keluar...")
print("Arahkan objek berwarna hijau ke kamera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Fitur 1: Konversi ke HSV ---
    # Konversi frame BGR ke HSV 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Fitur 2: Thresholding Warna ---
    # Buat mask biner menggunakan cv2.inRange() [cite: 76]
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # --- Fitur 3: Pembersihan Mask (Morfologi) ---
    # Gunakan Opening untuk menghapus bintik-bintik kecil (false positives) [cite: 78]
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Gunakan Closing untuk menutup lubang-lubang kecil (false negatives) [cite: 79]
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # --- Fitur 4: Pemicu Aksi (Temukan Kontur) ---
    # Temukan kontur pada mask yang sudah bersih [cite: 80]
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    action_triggered = False

    if contours:
        # Temukan kontur terbesar berdasarkan area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Jika kontur yang cukup besar terdeteksi, picu aksi [cite: 81]
        if area > 500: # Ambang batas area
            action_triggered = True
            
            # Dapatkan kotak pembatas (bounding box)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # AKSI: Gambar kotak pada frame asli
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # AKSI: Tampilkan teks di layar jika terdeteksi [cite: 81]
    if action_triggered:
        cv2.putText(frame, "Objek Hijau Terdeteksi!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # --- Tampilkan Hasil ---
    # Tampilkan frame asli yang sudah ada aksi (kotak dan teks)
    cv2.imshow("Hasil Akhir (Tugas 2 Lengkap)", frame)
    
    # Tampilkan mask bersih (untuk debugging)
    cv2.imshow("Mask Bersih", mask_cleaned)


    # Tombol untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()