import cv2
import numpy as np


cap = cv2.VideoCapture(0) 


filter_mode = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # cek input keyboard
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('0'):
        filter_mode = 0
    elif key == ord('1'):
        filter_mode = 1
    elif key == ord('2'):
        filter_mode = 2
    elif key == ord('3'):
        filter_mode = 3
    elif key == ord('q'): 
        break

    # menerapkan filter berdasarkan mode
    if filter_mode == 1:
        # menerapkan blur dengan 5x5
        # memakai 5x5
        cv2.putText(frame, "Average Blur", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output = cv2.blur(frame, (5, 5)) # 
    elif filter_mode == 2:
        # menerapkan Gaussian Blurring
        # memakai filter2D 
        kernel_size = 9
        # membuat kernel Gaussian 1D (X dan Y)
        kernel_x = cv2.getGaussianKernel(kernel_size, 0)
        kernel_y = cv2.getGaussianKernel(kernel_size, 0)
        # menggabungkan menjadi kernel 2D
        kernel_2d = kernel_x * kernel_y.T
        cv2.putText(frame, "Gaussian Blur", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        output = cv2.filter2D(frame, -1, kernel_2d) # 
        
    elif filter_mode == 3:
        # menerapkan sharpening 
        kernel_sharpen = np.array([[ 0, -1,  0],
                                   [-1,  5, -1],
                                   [ 0, -1,  0]])
        cv2.putText(frame, "Sharpen", (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output = cv2.filter2D(frame, -1, kernel_sharpen)

    else:
        # mode 0: Tampilkan frame normal
        cv2.putText(frame, "Normal", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        output = frame

    
    cv2.imshow("Tugas 1", output)

cap.release()

cv2.destroyAllWindows()