import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    exit()

# Mendefinisikan rentang bawah (lower) dan atas (upper) untuk warna HITAM
# Formatnya adalah [H, S, V]

lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 50]) 


print("Webcam terbuka. Menekan 'q' untuk keluar...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mengkonversi BGR ke HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Memisahkan setiap channel untuk visualisasi
    h, s, v = cv2.split(hsv_frame)

    # Membuat Mask Biner untuk warna hitam
    mask = cv2.inRange(hsv_frame, lower_black, upper_black)



    cv2.imshow("Webcam Asli (BGR)", frame)
    
    # Membuka channel H, S, dan V
    cv2.imshow("Channel H (Hue)", h)
    cv2.imshow("Channel S (Saturation)", s)
    cv2.imshow("Channel V (Value)", v) 

    # Hasil deteksi hitam
    cv2.imshow("Mask Biner (Deteksi Hitam)", mask)


    # Keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()