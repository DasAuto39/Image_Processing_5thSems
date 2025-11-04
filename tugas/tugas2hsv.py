import cv2
import numpy as np

# Inisialisasi webcam
# (Ganti '0' ke '1' atau '2' jika webcam tidak muncul)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    print("Pastikan tidak digunakan oleh program lain.")
    exit()

print("Webcam terbuka. Menekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak bisa membaca frame.")
        break

    # 1. Konversi BGR ke HSV 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Pisahkan channel H, S, dan V
    # h, s, dan v sekarang adalah gambar grayscale (2D array)
    h, s, v = cv2.split(hsv_frame)

    # 3. Tampilkan semua jendela
    
    # Tampilkan frame asli sebagai referensi
    cv2.imshow("Webcam Asli (BGR)", frame)
    
    # Tampilkan Channel H (Hue / Warna)
    # Ini adalah channel terpenting untuk mendeteksi warna [cite: 60, 63]
    cv2.imshow("Channel H (Hue)", h)
    
    # Tampilkan Channel S (Saturation / Kepekatan)
    # Menunjukkan seberapa murni warnanya (pudar vs pekat) 
    cv2.imshow("Channel S (Saturation)", s)
    
    # Tampilkan Channel V (Value / Kecerahan)
    # Menunjukkan seberapa terang/gelap warnanya 
    cv2.imshow("Channel V (Value)", v)

    # Tombol untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()