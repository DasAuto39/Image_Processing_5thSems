import cv2
import numpy as np

# Pilih warna yang ingin dideteksi (bisa 'biru' atau 'hijau')
warna_target = 'biru'  # ubah ke 'hijau' jika ingin mendeteksi hijau

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Konversi dari BGR ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2️⃣ Definisikan rentang HSV untuk warna yang ingin dideteksi
    if warna_target == 'biru':
        lower = np.array([100, 150, 50])   # batas bawah biru
        upper = np.array([140, 255, 255])  # batas atas biru
    elif warna_target == 'hijau':
        lower = np.array([40, 70, 70])     # batas bawah hijau
        upper = np.array([80, 255, 255])   # batas atas hijau

    # Buat mask biner
    mask = cv2.inRange(hsv, lower, upper)

    # 3️⃣ Pembersihan Mask (Morfologi)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # hapus noise kecil
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)  # tutup lubang kecil

    # 4️⃣ Temukan kontur pada mask yang sudah bersih
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop semua kontur yang ditemukan
    detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # abaikan objek kecil
            detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{warna_target.capitalize()} Terdeteksi!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow('Frame Asli', frame)
    cv2.imshow('Mask', mask_clean)

    # Keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
