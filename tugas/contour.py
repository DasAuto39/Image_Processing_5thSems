import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    exit()

# Rentang untuk warna HIJAU [cite: 75]
lower_green = np.array([35, 100, 50])
upper_green = np.array([85, 255, 255])

# Buat kernel untuk operasi morfologi 
kernel = np.ones((5, 5), np.uint8)

print("Webcam terbuka. Menekan 'q' untuk keluar...")
print("Arahkan objek berwarna hijau ke kamera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mengkonversi frame BGR ke HSV 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Membuat mask biner menggunakan cv2.inRange() 
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Menggunakan Morph Opening untuk menghapus bintik-bintik kecil (false positives)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Menggunakan Morph Closing untuk menutup lubang-lubang kecil (false negatives)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # Mencari kontur pada mask yang sudah bersih 
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    action_triggered = False

    if contours:
        # Menemukan kontur terbesar berdasarkan area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Jika kontur yang cukup besar terdeteksi, picu aksi
        if area > 500: # Ambang batas area
            action_triggered = True
            
            # Dapatkan kotak pembatas (bounding box)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # AKSI: Gambar kotak pada frame asli
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menampilkan teks di layar jika terdeteksi
    if action_triggered:
        cv2.putText(frame, "Objek Hijau Terdeteksi!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Menampilkan frame asli yang sudah ada aksi (kotak dan teks)
    cv2.imshow("Hasil Akhir (Tugas 2 Lengkap)", frame)
    
    # Menampilkan mask bersih (untuk debugging)
    cv2.imshow("Mask Bersih", mask_cleaned)


    # Keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()