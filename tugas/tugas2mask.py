import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    exit()

# Rentang untuk warna HITAM (dari kode sebelumnya)
lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 50])

# Mendefiniskan 'kernel' untuk operasi morfologi.
# matriks kecil (misal 5x5) yang digunakan untuk  memproses piksel.
kernel = np.ones((5, 5), np.uint8)

print("Webcam terbuka. Menekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Mengkonversi BGR ke HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Membuat Mask Biner (Thresholding)
    mask_original = cv2.inRange(hsv_frame, lower_black, upper_black)

    # 3. Membersihan Mask (Morfologi) 
    # Gunakan Opening (MORPH_OPEN) untuk menghapus bintik-bintik kecil (false positives)
    mask_opened = cv2.morphologyEx(mask_original, cv2.MORPH_OPEN, kernel)
    
    # Menggunakan Closing (MORPH_CLOSE) untuk menutup lubang-lubang kecil (false negatives) pada hasil yang sudah di-opening
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)


    # Webcam asli
    cv2.imshow("Webcam Asli (BGR)", frame)
    
    # HSV Mask asli (sebelum morfologi)
    cv2.imshow("Mask Asli (Sebelum Morfologi)", mask_original)

    # Mask bersih (setelah morfologi)
    cv2.imshow("Mask Bersih (Setelah Morfologi)", mask_cleaned)


    # Keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()