import cv2
import numpy as np

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam tidak dapat dibuka.")
    exit()

# Rentang untuk warna HITAM 
lower_black = np.array([35, 100, 50])
upper_black = np.array([85, 255, 255])

# Kernel untuk morfologi
kernel = np.ones((5, 5), np.uint8)

print("Webcam terbuka. Menekan 'q' untuk keluar...")
print("Arahkan objek berwarna hitam ke kamera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Mengkonversi BGR ke HSV 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Membuat Mask Biner (Thresholding) 
    mask = cv2.inRange(hsv_frame, lower_black, upper_black)

    # 3. Membersihan Mask (Morfologi) 
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # 4. Temukan Kontur,mencari kontur pada mask yang sudah bersih
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    action_triggered = False

    # Cek apakah ada kontur yang ditemukan
    if contours:
        # mengambil kontur terbesar berdasarkan areanya
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Cek apakah areanya lebih besar dari ambang batas (misal 500 piksel) 
        if area > 500:
            action_triggered = True
            
            # Dapatkan kotak pembatas (bounding box) dari kontur terbesar
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Gambar kotak di frame ASLI
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan teks jika aksi terpicu 
    if action_triggered:
        cv2.putText(frame, "Objek Hitam Terdeteksi!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    # Menampilkan frame asli dengan kotak dan teks
    cv2.imshow("Hasil", frame)
    
    # Menampilkan mask bersih (untuk debugging)
    cv2.imshow("Mask Bersih (Input untuk Kontur)", mask_cleaned)


    # Keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()