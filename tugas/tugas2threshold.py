import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    exit()

# --- LANGKAH 2 UNTUK WARNA HITAM ---
# Definisikan rentang bawah (lower) dan atas (upper) untuk warna HITAM
# Formatnya adalah [H, S, V]

# H (Hue) bisa apa saja: 0-179
# S (Saturation) bisa apa saja: 0-255
# V (Value) harus RENDAH: 0 - 50 (Anda bisa ubah '50' ini)
lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 50]) # <--- UBAH '50' INI JIKA PERLU

# --- SELESAI ---


print("Webcam terbuka. Menekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Konversi BGR ke HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. (Opsional) Pisahkan channel untuk visualisasi
    h, s, v = cv2.split(hsv_frame)

    # 3. Buat Mask Biner untuk warna hitam
    mask = cv2.inRange(hsv_frame, lower_black, upper_black)


    # --- Tampilkan Semua Jendela ---
    cv2.imshow("Webcam Asli (BGR)", frame)
    
    # Jendela-jendela untuk analisis
    cv2.imshow("Channel H (Hue)", h)
    cv2.imshow("Channel S (Saturation)", s)
    cv2.imshow("Channel V (Value)", v) # <--- Perhatikan channel ini

    # JENDELA BARU: Hasil deteksi hitam
    cv2.imshow("Mask Biner (Deteksi Hitam)", mask)


    # Tombol untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()